<!--en:619d9ce6-->
# Numerische Stabilität und Initialisierung

<!--en:6b4ccaff-->
In :numref:`sec_mlp-implementation` haben wir die Gewichte unseres mehrschichtigen Perzeptrons (engl. multilayer perceptron, MLP) mit einer einzigen versteckten Schicht aus $\mathcal{N}(0, 0.01^2)$ gezogen. Das Training kam in der ersten Epoche nur langsam voran, doch das Netz erreichte schließlich eine Genauigkeit von $87\%$. Was passiert, wenn wir *zehn* versteckte Schichten der Breite $256$ mit derselben Initialisierung aus $\mathcal{N}(0, 0.01^2)$ stapeln? Nach zehn Epochen liegt der Verlust immer noch bei $\log 10 \approx 2.30$. Die Genauigkeit entspricht mit $10\%$ der Trefferquote beim zufälligen Raten. Ziehen wir die Gewichte dagegen aus $\mathcal{N}(0, 1)$, werden Aktivierungen und Gradienten astronomisch groß. Ein einziger Schritt des stochastischen Gradientenabstiegs (engl. stochastic gradient descent, SGD) ändert die Gewichte so stark, dass beim nächsten Vorwärtsdurchlauf ein Überlauf auftritt und `inf` und `NaN` entstehen.

<!--en:638f0ea2-->
Wie wir in :numref:`sec_backprop` gesehen haben, durchlaufen Aktivierungen in Vorwärtsrichtung und Gradienten in Rückwärtsrichtung ein Produkt aus $K$ Faktoren, die Gewichte und Steigungen enthalten, wenn das Netz $K$ Schichten hat. Damit die Varianz des Signals erhalten bleibt, muss der Faktor jeder Schicht nahe bei $1$ liegen. Andernfalls *verschwinden* die Signale exponentiell gegen null oder *explodieren* exponentiell gegen unendlich. Wir berechnen, wie sich die Varianz von Schicht zu Schicht ändert. Daraus leiten wir die Initialisierungen nach *Xavier* und *He* ab und überprüfen sie an einem MLP mit 10 Schichten auf Fashion-MNIST. Schließlich zeigen wir, warum wir die Anfangsgewichte *zufällig* wählen müssen, um die Permutationssymmetrie der versteckten Einheiten zu brechen.

<!--en:599549e0-->
## Verschwindende und explodierende Gradienten

<!--en:c29aeb4a-->
### Ein Produkt über die Schichten

<!--en:8a30052a-->
Betrachten wir ein MLP mit $K$ Schichten. Seine versteckten Schichten sind durch $\mathbf{Z}^{(k)} = \mathbf{H}^{(k-1)}\mathbf{W}^{(k)} + \mathbf{b}^{(k)}$ und $\mathbf{H}^{(k)} = \phi(\mathbf{Z}^{(k)})$ gegeben und haben die Breite $h$. Mit der Rückwärtsrekursion aus :numref:`sec_backprop` berechnen wir aus dem Gradienten nach den Aktivierungen den entsprechenden Gradienten eine Schicht tiefer:

<!--en:b878e592-->
Wenden wir :eqref:`eq_grad-step` von der obersten versteckten Schicht $K - 1$ bis hinunter zu Schicht $k$ wiederholt an, so erhalten wir den Gradienten an Schicht $k$ als Produkt des obersten Gradienten mit $K - 1 - k$ Faktoren aus Gewichtsmatrizen und Steigungen:

<!--en:cdfd896b-->
Für ein einzelnes Beispiel sei $\mathbf{g}'$ der ankommende und $\mathbf{g}$ der weitergegebene Gradient, jeweils als Zeilenvektor. Die Schicht habe Breite $h$. Ihre Gewichte $w_{ij}$ ziehen wir unabhängig aus derselben Verteilung mit Mittelwert $0$ und Varianz $\tau^2$. Bei der Initialisierung behandeln wir Gewichte, Steigungen und ankommende Gradienten als näherungsweise unabhängig. Für jede Komponente $g_i = \sum_{j=1}^h w_{ij}\,\phi'(z_j)\,g'_j$ ergibt sich dann als Erwartungswert ihres Quadrats

<!--en:d2d46673-->
Über $t$ Schichten ändert sich das mittlere Quadrat (engl. mean square) der Gradientenkomponenten um den Faktor $\big(h\,\tau^2\,\mathbb{E}[\phi'(z)^2]\big)^t$. Bei symmetrisch verteilten Präaktivierungen und ReLU nimmt die Ableitung $\phi'(z) \in \{0, 1\}$ in der Hälfte der Fälle den Wert $1$ an. Damit gilt $\mathbb{E}[\phi'(z)^2] = 1/2$, und der Faktor pro Schicht beträgt $\frac{1}{2}h\tau^2$. Für die Breite $h = 256$ ergibt sich Folgendes:

<!--en:f73d5832-->
- Bei $\tau = 0.01$ beträgt der Faktor $256 \times 10^{-4} / 2 = 0.0128$. Nach $9$ Übergängen ist das mittlere Quadrat der Gradientenkomponenten nur noch ein Bruchteil des Ausgangswerts, nämlich $0.0128^9 \approx 10^{-17}$ (**verschwindende Gradienten**, engl. vanishing gradients).

<!--en:3e2701a1-->
- Bei $\tau = 1$ beträgt der Faktor $256 \times 1 / 2 = 128$. Nach $9$ Übergängen ist das mittlere Quadrat der Gradientenkomponenten um den Faktor $128^9 \approx 10^{19}$ gewachsen (**explodierende Gradienten**, engl. exploding gradients).

<!--en:e8d8a418-->
- Für die Sigmoidfunktion gilt $\sigma'(z) \leq 1/4$ und damit $\mathbb{E}[\sigma'(z)^2] \leq 1/16$. Selbst bei $h\tau^2 = 1$ verkleinert jede Schicht das mittlere Quadrat der Gradientenkomponenten mindestens um den Faktor $16$ (:cite:`bengio1994learning`).

<!--en:ca9168f5-->
Prüfen wir diese drei Vorhersagen mit `moments` direkt bei der Initialisierung eines Netzes mit 10 Schichten der Breite 256:

<!--en:137fee9b-->
Vergleichen wir drei Fälle: ReLU mit $\tau = 0.01$ (wie in :numref:`sec_mlp-implementation`), ReLU mit $\tau = 1$ und die Sigmoidfunktion mit $\tau^2 = 1/256$. Im letzten Fall gilt $h\tau^2 = 1$, sodass wir allein die Abschwächung durch die Aktivierungsfunktion beobachten.

<!--en:f8e3de8f-->
Bei logarithmischer Skalierung der vertikalen Achse sind alle drei Kurven Geraden. Ihre Steigungen entsprechen dem berechneten Faktor aus :eqref:`eq_grad-factor` (in Schicht 1 ergeben sich $10^{-17}$, $10^{19}$ und $5 \times 10^{-12}$).

<!--en:0e079cbe-->
## Initialisierung der Parameter

<!--en:5c833bd7-->
### Xavier-Initialisierung

<!--en:77cbf842-->
Wie wählen wir die anfängliche Varianz $\tau^2$ der Gewichte, damit der Faktor in jeder Schicht gleich $1$ ist? Betrachten wir eine lineare Schicht $o_j = \sum_{i=1}^{n_{\textrm{in}}} x_i w_{ij}$ mit $n_{\textrm{in}}$ Eingängen (*Fan-in*) und $n_{\textrm{out}}$ Ausgängen (*Fan-out*). Die Gewichte $w_{ij}$ sind unabhängig und identisch verteilt mit Mittelwert $0$ und Varianz $\tau^2$. Die Eingaben $x_i$ sind unkorreliert und haben Mittelwert $0$ und Varianz $\gamma^2$. Im Vorwärtsdurchlauf gilt $\mathbb{E}[o_j] = 0$ und

<!--en:e5af2726-->
Damit die Schicht im Vorwärtsdurchlauf die Varianz erhält, muss $n_{\textrm{in}}\tau^2 = 1$ gelten. Nach :eqref:`eq_grad-factor` muss bei einer Steigung von eins die Bedingung $n_{\textrm{out}}\tau^2 = 1$ gelten, damit im Rückwärtsdurchlauf die Gradientenvarianz erhalten bleibt. Bei $n_{\textrm{in}} \neq n_{\textrm{out}}$ verwendet die *Xavier-Initialisierung* (auch *Glorot-Initialisierung*, :cite:`Glorot.Bengio.2010`) den Mittelwert aus Fan-in und Fan-out:

<!--en:ef2affa7-->
Wir ziehen die Gewichte entweder aus $\mathcal{N}\!\left(0, \frac{2}{n_{\textrm{in}} + n_{\textrm{out}}}\right)$ oder aus der Gleichverteilung $U\!\left(-\sqrt{\frac{6}{n_{\textrm{in}} + n_{\textrm{out}}}},\, \sqrt{\frac{6}{n_{\textrm{in}} + n_{\textrm{out}}}}\right)$ mit derselben Varianz. Die Xavier-Initialisierung ist die übliche Wahl für lineare Aktivierungen und für nullzentrierte Aktivierungsfunktionen wie $\tanh$ mit Steigung eins im Ursprung.

<!--en:f236de38-->
### He-Initialisierung

<!--en:14da1bfc-->
ReLU setzt die negative Hälfte einer symmetrisch verteilten Präaktivierung $z$ mit Mittelwert null auf null. Dadurch halbiert sich ihr mittleres Quadrat: $\mathbb{E}[\mathrm{ReLU}(z)^2] = \frac{1}{2}\mathbb{E}[z^2]$. Für unabhängige Gewichte mit Mittelwert null gilt $\mathbb{E}[o_j^2] = n_{\textrm{in}}\tau^2\,\mathbb{E}[x_i^2]$ auch dann, wenn die Eingaben $x_i = \mathrm{ReLU}(z_i)$ einen positiven Mittelwert haben. Das mittlere Quadrat der Aktivierungen $v^{(k)} = \mathbb{E}[(h_i^{(k)})^2]$ ändert sich daher in einer ReLU-Schicht gemäß

<!--en:bd4df883-->
Damit $v^{(k)} = v^{(k-1)}$ über alle Schichten hinweg gilt, müssen wir also die Varianz verdoppeln:

<!--en:edd302ce-->
Dies ist die *He-Initialisierung* (auch *Kaiming-Initialisierung*, :cite:`He.Zhang.Ren.ea.2015`), die übliche Wahl für ReLU-Netze. Die Standardschichten vieler Frameworks verwenden allerdings $1/(3n_{\textrm{in}})$, $1/n_{\textrm{in}}$ oder die Xavier-Initialisierung, wenn wir die He-Initialisierung nicht ausdrücklich auswählen.

<!--en:982ea333-->
### Aktivierungen, Gradienten und Training in tiefen Netzen

<!--en:9c31edf9-->
Vergleichen wir zunächst $\mathcal{N}(0, 0.01^2)$, die Xavier-Initialisierung ($\tau^2 = 1/256$) und die He-Initialisierung ($\tau^2 = 2/256$) in unserem synthetischen Experiment mit 10 ReLU-Schichten und `moments(*relu, ...)`:

<!--en:17810156-->
Bei $\mathcal{N}(0, 0.01^2)$ sinkt das mittlere Quadrat der Aktivierungen in Schicht 10 auf $\sim 10^{-19}$. Die Xavier-Initialisierung halbiert es in jeder ReLU-Schicht (auf $\sim 2^{-10} \approx 10^{-3}$ in Schicht 10). Mit der He-Initialisierung bleiben die mittleren Quadrate der Aktivierungen und der Gradienten dagegen in der Größenordnung $O(1)$. Wie wirkt sich diese Entwicklung der Varianz auf das Training aus? Dazu definieren wir einen ReLU-Klassifikator `DeepMLP` mit 10 versteckten Schichten. Als Parameter erhält er eine Funktion `std(n_in, n_out)`, die für jede Schicht die Standardabweichung vorgibt:

<!--en:679f4878-->
Alle drei Trainingsläufe verwenden dieselbe Architektur und dieselben Hyperparameter. Nur `std(n_in, n_out)` unterscheidet sich. Vergleichen wir die Trainingskurven und die am Ende erreichte Validierungsgenauigkeit.

<!--en:afdfdcfb-->
Die Trainingskurven über 10 Epochen folgen den Vorhersagen unserer Varianzanalyse sehr genau:

<!--en:2eb8e19b-->
- Bei $\mathcal{N}(0, 0.01^2)$ verschwinden die Aktivierungen in Schicht 10 ($\sim 10^{-19}$) und die Gradienten in Schicht 1 ($\sim 10^{-17}$) praktisch vollständig. Der Trainingsverlust verharrt bei $\log 10 \approx 2.30$ und die Validierungsgenauigkeit bei $10\%$.

<!--en:d6e96f7a-->
- Bei der Xavier-Initialisierung nimmt das Signal in jeder ReLU-Schicht um den Faktor $2$ ab. Insgesamt bleibt nur das $2^{-10} \approx 10^{-3}$-Fache übrig. Deshalb kommt das Training in der ersten Epoche langsam voran, erreicht dann aber eine Genauigkeit von über $80\%$.

<!--en:77545992-->
- Bei der He-Initialisierung bleiben Aktivierungen und Gradienten über alle 10 Schichten in der Größenordnung $1$. Der Trainingsverlust sinkt von der ersten Epoche an gleichmäßig.

<!--en:54c68d86-->
Die Initialisierung allein löst jedoch nicht alle Probleme tiefer Netze. Unsere Varianzanalyse beschreibt das Netz nur im Schritt $0$. Sobald SGD die Gewichte verändert, kann der Faktor jeder Schicht aus :eqref:`eq_grad-factor` von $1$ abweichen. Netze mit Dutzenden oder Hunderten von Schichten nutzen deshalb Mechanismen, die während des gesamten Trainings wirken. Normalisierungsschichten (:numref:`sec_batch_norm`) skalieren die Aktivierungen in jedem Schritt neu. Residualverbindungen (:numref:`sec_resnet`) geben Signalen und Gradienten einen Weg an der jeweiligen Schicht vorbei. Gradienten-Clipping (:numref:`sec_rnn`) begrenzt gelegentlich auftretende explodierende Gradienten. Eine gute Initialisierung bringt das Training in Gang. Danach halten diese Mechanismen die Signale in der passenden Größenordnung.

<!--en:e65a6e87-->
## Symmetrie zwischen versteckten Einheiten

<!--en:44ae8e55-->
Warum müssen die Anfangsgewichte *zufällig* sein? Könnten wir nicht jeden Eintrag von $\mathbf{W}^{(1)}$ auf die Konstante $c = \sqrt{2 / n_{\textrm{in}}}$ setzen? Der Grund ist die Permutationssymmetrie der versteckten Einheiten einer Schicht. Sind anfangs alle Spalten von $\mathbf{W}^{(1)}$ identisch und ebenso alle Zeilen von $\mathbf{W}^{(2)}$, so berechnet jede versteckte Einheit dieselbe Aktivierung im Vorwärtsdurchlauf. Sie erhält denselben Gradienten im Rückwärtsdurchlauf und macht bei jedem SGD-Schritt dieselbe Aktualisierung mit. So wirkt eine Schicht mit 256 Einheiten wie ein einziges Neuron. Prüfen wir das an einer Schicht mit 16 Einheiten: Konstante, nach He skalierte Gewichte ergeben identische Gradientenspalten (Differenz `0.00` zwischen den Einheiten). Bei zufälligen, nach He skalierten Gewichten erhält jede Einheit eine eigene Gradientenrichtung.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:a3dd97c5-->
In einem MLP mit $K$ Schichten der Breite $h$ und anfänglicher Gewichtsvarianz $\tau^2$ multipliziert jede Schicht die mittleren Quadrate der Aktivierungen und Gradienten näherungsweise mit $h\tau^2\,\mathbb{E}[\phi'(z)^2]$. Bleibt dieser Faktor nicht nahe bei $1$, verschwinden oder explodieren die Signale exponentiell mit der Tiefe. Die Xavier-Initialisierung wählt $\tau^2 = 2/(n_{\textrm{in}} + n_{\textrm{out}})$, um die Varianz in Vorwärts- und Rückwärtsrichtung für lineare Schichten und $\tanh$-Schichten auszugleichen. Die He-Initialisierung verdoppelt die Varianz auf $\tau^2 = 2/n_{\textrm{in}}$. Das gleicht aus, dass ReLU auf der negativen Halbachse null ist. So lassen sich tiefe ReLU-Netze von der ersten Epoche an gut trainieren. Wegen der Permutationssymmetrie der versteckten Einheiten einer Schicht müssen wir die Gewichte zudem unabhängig voneinander zufällig ziehen. Bei konstanten Gewichten erhält sonst jede Einheit während des gesamten Trainings dieselben Gradienten und Aktualisierungen. Tiefere Architekturen ergänzen die varianzerhaltende Initialisierung durch Normalisierungsschichten und Residualverbindungen. So behalten die Signale auch nach Schritt $0$ eine geeignete Größenordnung, während sich die Gewichte ändern.

<!--en:2294243a-->
## Übungen

<!--en:48c5c491-->
1. **Schranken mit der Spektralnorm.** Betrachten Sie ein einzelnes Beispiel in Form eines Spaltenvektors. Der Schritt :eqref:`eq_grad-step` multipliziert den Gradienten $\mathbf{g}'$ oberhalb der Schicht $k$ mit der Matrix $\mathbf{M}^{(k)} = \mathbf{W}^{(k)} \operatorname{diag}\bigl(\phi'(\mathbf{z}^{(k)})\bigr)$. Dabei setzt $\operatorname{diag}(\cdot)$ die Einträge eines Vektors auf die Diagonale einer ansonsten mit Nullen gefüllten Matrix. Der Gradient in :eqref:`eq_grad-product` ergibt sich also aus dem Produkt von $K - 1 - k$ solchen Matrizen mit dem obersten Gradienten. Die Spektralnorm (der größte Singulärwert) ist submultiplikativ: $\|\mathbf{A}\mathbf{B}\|_2 \le \|\mathbf{A}\|_2 \|\mathbf{B}\|_2$.
    1. Beweisen Sie die Submultiplikativität anhand der Definition $\|\mathbf{A}\|_2 = \max_{\mathbf{x} \neq 0} \|\mathbf{A}\mathbf{x}\| / \|\mathbf{x}\|$.
    1. Leiten Sie daraus eine Schranke für die Gradientennorm her, die von den Normen $\|\mathbf{M}^{(k)}\|_2$ der Faktoren abhängt. Welche Bedingung an diese Normen verhindert, dass die Schranke mit der Tiefe geometrisch wächst?
    1. Diese Abschätzung liefert nur eine obere Schranke. Erklären Sie, warum die Bedingung aus der vorigen Teilaufgabe explodierende Gradienten ausschließt, verschwindende Gradienten aber zulässt. Geben Sie zwei $2 \times 2$-Matrizen an, die die Bedingung mit Gleichheit erfüllen und deren Produkt eine Spektralnorm kleiner als eins hat.

<!--en:4e3099cf-->
1. **Weitere Symmetrien.** Neben der Permutation der versteckten Einheiten gibt es weitere Parameteränderungen, die die Funktion eines MLP unverändert lassen. Betrachten Sie ein MLP mit einer versteckten Schicht. Die versteckte Einheit $j$ habe eingehende Gewichte $\mathbf{w}_j$, den Bias $b_j$ und ausgehende Gewichte $\mathbf{v}_j$.
    1. Beweisen Sie, dass $\textrm{ReLU}(\alpha z) = \alpha\, \textrm{ReLU}(z)$ für $\alpha > 0$ gilt. Zeigen Sie mit dieser Identität, dass sich die Funktion eines ReLU-Netzes nicht ändert, wenn Sie $(\mathbf{w}_j, b_j, \mathbf{v}_j)$ durch $(\alpha \mathbf{w}_j, \alpha b_j, \mathbf{v}_j / \alpha)$ ersetzen.
    1. Finden Sie eine Transformation der Parameter einer einzelnen versteckten Einheit, die die Funktion eines $\tanh$-Netzes unverändert lässt.
    1. Identische versteckte Einheiten bilden einen Fixpunkt der Permutation. Beginnt der Gradientenabstieg an diesem Fixpunkt, bleiben die Einheiten identisch. Finden Sie Parameter einer Einheit, die unter der Transformation aus der vorigen Teilaufgabe unverändert bleiben. Zeigen Sie anschließend: Sind alle Gewichte und Bias-Terme eines $\tanh$-Netzes anfangs null, so ändert der Gradientenabstieg bei jedem Schritt nur den Bias der Ausgabeschicht.

    *Nach Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), Aufgaben 3.5, 4.3 und 7.15.*

<!--en:baee39dd-->
1. **Fan-in und Mittelwert aus Fan-in und Fan-out.** Betrachten Sie wie in :citet:`He.Zhang.Ren.ea.2015` ein lineares Netz mit $K$ Schichten. Schicht $k$ bildet $n_{k-1}$ Eingaben auf $n_k$ Ausgaben ab. Ihre Gewichte werden mit Varianz $\tau_k^2$ gezogen. Nach der Rechnung in :numref:`subsec_xavier` multipliziert Schicht $k$ die Varianz des Signals im Vorwärtsdurchlauf mit $n_{k-1}\tau_k^2$ und im Rückwärtsdurchlauf mit $n_k\tau_k^2$.
    1. Setzen Sie in jeder Schicht $\tau_k^2 = 1/n_{k-1}$. Zeigen Sie, dass die Varianz im Vorwärtsdurchlauf erhalten bleibt und sich im Rückwärtsdurchlauf bei beliebiger Tiefe um den Gesamtfaktor $n_K / n_0$ ändert.
    1. Wählen Sie abwechselnde Schichtbreiten $100, 400, 100, 400, \ldots$ und verwenden Sie in jeder Schicht die Xavier-Initialisierung. Berechnen Sie den Faktor, mit dem ein Schichtpaar $100 \to 400 \to 100$ die Varianz im Vorwärtsdurchlauf multipliziert. Wie groß ist der Faktor nach 50 solchen Paaren?
    1. Bei welcher der beiden Regeln bleiben die Abweichungen beider Varianzen von ihren Anfangswerten für jede Folge von Schichtbreiten auf einen tiefenunabhängigen Faktor beschränkt?

<!--en:6086de9e-->
1. **Varianz bei ReLU.** ● Die Herleitung in :numref:`subsec_he_init` verfolgt das mittlere Quadrat $\mathbb{E}[\textrm{ReLU}(z)^2]$ anstelle der Varianz. Für die Präaktivierung $o = \sum_i x_i w_i$ der nächsten Schicht mit Eingaben $x_i = \textrm{ReLU}(z_i)$ und unabhängigen Gewichten mit Mittelwert null gilt nämlich $\textrm{Var}[o] = n_\textrm{in}\tau^2 \mathbb{E}[x_i^2]$, unabhängig vom Mittelwert von $x_i$.
    1. Berechnen Sie für $z \sim \mathcal{N}(0, \nu^2)$ die Größen $\mathbb{E}[\textrm{ReLU}(z)]$ und $\textrm{Var}[\textrm{ReLU}(z)]$. Zeigen Sie, dass die Varianz $\left(\tfrac{1}{2} - \tfrac{1}{2\pi}\right)\nu^2$ beträgt und nicht $\tfrac{1}{2}\nu^2$.
    1. Eine naheliegende Abkürzung in der Xavier-Rechnung wäre, $\textrm{Var}[x_i]$ anstelle von $\mathbb{E}[x_i^2]$ einzusetzen. Zeigen Sie, dass sich dadurch $\tau^2 = 2 / \bigl(n_\textrm{in}(1 - 1/\pi)\bigr)$ anstelle von $2/n_\textrm{in}$ ergibt. Berechnen Sie den Faktor, mit dem jede so initialisierte ReLU-Schicht das mittlere Quadrat tatsächlich multipliziert, sowie den Gesamtfaktor über fünfzig Schichten.

<!--en:88cff96b-->
1. [code] **Tanh-Netze.** ● Wiederholen Sie die Messung aus :numref:`subsec_variance_propagation` mit $\tanh$ anstelle von ReLU. Verwenden Sie die Xavier-Initialisierung und fünfzig Schichten. Eine Schicht bildet dann das mittlere Quadrat $m$ ihrer Eingabe auf $m' = \mathbb{E}[\tanh(z)^2]$ ab. Dabei folgt $z$ näherungsweise der Verteilung $\mathcal{N}(0, m)$.
    1. Verwenden Sie die Näherung $\tanh(z)^2 \approx z^2 - \tfrac{2}{3}z^4$ für kleine $z$, um $m' \approx m - 2m^2$ zu zeigen. Bleibt das mittlere Quadrat über viele Schichten konstant, fällt es geometrisch ab oder nimmt es langsamer ab? Zeigen Sie, dass das mittlere Quadrat in Schicht $k$ etwa $1/(2k)$ beträgt, wenn $k$ groß ist.
    1. Prüfen Sie die Vorhersage, indem Sie das gemessene mittlere Quadrat gegen die Schichtnummer auftragen. Skalieren Sie beide Achsen logarithmisch. Was geschieht mit dem Gradienten?

<!--en:8491e425-->
[Diskussion](https://d2l.discourse.group/t/103)

<!--en:8a1e9085-->
[Diskussion](https://d2l.discourse.group/t/104)

<!--en:343f3bc1-->
[Diskussion](https://d2l.discourse.group/t/235)

<!--en:cf9d931e-->
[Diskussion](https://d2l.discourse.group/t/17986)

<!--en:1e42f171-->
Numerische Stabilität und varianzerhaltende Initialisierung<br>**Verschwindende und explodierende Produkte · Varianzskalierung je Schicht · Xavier- und He-Initialisierung · Symmetriebrechung**

<!--en:36794125-->
::: {.slide title="Warum ein MLP mit 10 Schichten bei N(0, 0.01²) oder N(0, 1) scheitert"}

<!--en:c58ff381-->
[Die Tiefe als Flaschenhals]{.kicker}

<!--en:34852d63-->
- **Zu klein ($\mathcal{N}(0, 0.01^2)$):** Über $9$ Übergänge zwischen versteckten Schichten der Breite $h = 256$ sinkt das mittlere Quadrat der Gradientenkomponenten auf das $10^{-17}$-Fache. Der Verlust verharrt bei $\log 10 \approx 2.30$ ($10\%$ Genauigkeit wie beim zufälligen Raten).

<!--en:7f9873eb-->
- **Zu groß ($\mathcal{N}(0, 1)$):** Über $9$ Übergänge wächst das mittlere Quadrat der Gradientenkomponenten auf das $10^{19}$-Fache. Tiefere Netze überschreiten den Wertebereich von `float32` bei $3.4 \times 10^{38}$. Der Überlauf führt zu `inf`/`NaN` im Schritt $1$.

<!--en:6afdc5ed-->
- **Prinzip der Varianzerhaltung:** Die Varianz der Aktivierungen im Vorwärtsdurchlauf und der Gradienten im Rückwärtsdurchlauf muss in jeder Schicht in der Größenordnung $O(1)$ bleiben, und zwar ab Schritt $0$.

<!--en:63fbf695-->
::: {.slide title="Der Gradient im Rückwärtsdurchlauf als Produkt von Schichtmatrizen"}

<!--en:afa11a40-->
[Verschwindende und explodierende Produkte]{.kicker}

<!--en:28b55daf-->
- **Matrixprodukt über die Schichten:** Von Schicht $K-1$ bis hinunter zu Schicht $k$ ergibt die wiederholte Anwendung der Rekursion ein Produkt aus $t = K - 1 - k$ Operatoren $\odot\,\phi'(\mathbf{Z}^{(l)})\mathbf{W}^{(l)\top}$ für Steigungen und Gewichte.

<!--en:94c763b1-->
- **Exponentielle Empfindlichkeit:** Schon eine Drift von $10\%$ je Schicht ergibt über $50$ Schichten den Faktor $0.9^{50} \approx 0.005$ (Verschwinden) oder $1.1^{50} \approx 117$ (Explodieren).

<!--en:0a6509db-->
::: {.slide title="Faktor für die Gradientenvarianz je Schicht: hτ²𝔼[φ′(z)²]"}

<!--en:e56c9270-->
[Berechnung der Varianzskalierung]{.kicker}

<!--en:7fd2d082-->
- **ReLU ($\mathbb{E}[\phi'(z)^2] = 1/2$):** Bei $h = 256$ skaliert $\tau = 0.01$ die Varianz je Schicht mit $0.0128\times$ ($10^{-17}$ nach $9$ Übergängen), bei $\tau = 1$ beträgt der Faktor $128\times$ ($10^{19}$ nach $9$ Übergängen).

<!--en:220b0a96-->
- **Sigmoidfunktion ($\mathbb{E}[\sigma'(z)^2] \le 1/16$):** Selbst bei $h\tau^2 = 1$ schrumpft die Varianz pro Schicht mindestens um den Faktor $16\times$ ($\le 1.5 \times 10^{-11}$ nach $9$ Übergängen).

<!--en:c5fffff4-->
::: {.slide title="Empirische Überprüfung an 10 Schichten"}

<!--en:94c7300a-->
[Messung verschwindender und explodierender Gradienten]{.kicker}

<!--en:af5e246f-->
- **Versuchsaufbau:** Mittlere Quadrate der Aktivierungen und Gradienten bei der Initialisierung eines Netzes mit 10 Schichten der Breite 256 in den drei untersuchten Fällen

<!--en:4a136dca-->
- **Exponentieller Verlauf:** Jede Schicht multipliziert das mittlere Quadrat mit einem konstanten Faktor. Im halblogarithmischen Diagramm entstehen daher Geraden.

<!--en:f3871358-->
- **Vorhersage und Messung in Schicht 1:**
  - ReLU, $\tau = 0.01$: $\sim 10^{-17}$ (verschwunden)
  - ReLU, $\tau = 1$: $\sim 10^{19}$ (explodiert)
  - Sigmoidfunktion, $\tau = 1/16$: $\sim 5 \times 10^{-12}$

<!--en:edaf5129-->
::: {.slide title="Warum Gradienten bei der Sigmoidfunktion selbst mit skalierten Gewichten verschwinden"}

<!--en:9c46fa1f-->
[Ursachen für verschwindende und explodierende Gradienten]{.kicker}

<!--en:81af88af-->
- **Das Dilemma der Sigmoidfunktion:** Wegen $\sigma'(z) \le 1/4$ teilt bei $h\tau^2 = 1$ jede Schicht die Gradientenvarianz durch einen Faktor $\ge 16$. Größere Gewichte ($h\tau^2 \ge 16$) treiben die Präaktivierungen $z$ in die Sättigungsbereiche ($|z| \gg 3$), wo die Ableitung noch schneller gegen null geht ($\sigma'(z) \to 0$).

<!--en:5ba738db-->
- **Explodierende Gradienten erschweren die Wahl der Schrittweite:** Sind die Gradienten in Schicht 1 $10^9\times$ so groß wie in Schicht 10, lässt sich mit keiner einzelnen Lernrate $\eta$ in beiden Schichten gleichzeitig trainieren.

<!--en:0cc72979-->
::: {.slide title="Xavier-Initialisierung zum Ausgleich der Varianz in beiden Richtungen"}

<!--en:76c01ec8-->
[Xavier-Initialisierung (Glorot-Initialisierung)]{.kicker}

<!--en:2aecfc5e-->
- **Zwei Bedingungen für die Varianzerhaltung:** Im Vorwärtsdurchlauf muss $n_{\textrm{in}}\tau^2 = 1$ gelten, für die Gradienten im Rückwärtsdurchlauf $n_{\textrm{out}}\tau^2 = 1$.

<!--en:2cbfd82c-->
- **Der Kompromiss nach Xavier (Glorot und Bengio, 2010):** Fan-in und Fan-out mitteln: $\frac{1}{2}(n_{\textrm{in}} + n_{\textrm{out}})\tau^2 = 1 \implies \tau^2 = \frac{2}{n_{\textrm{in}} + n_{\textrm{out}}}$

<!--en:ef375c98-->
::: {.slide title="He-Initialisierung (Kaiming-Initialisierung) für ReLU-Netze"}

<!--en:c732d6f8-->
[He-Initialisierung]{.kicker}

<!--en:75161159-->
- **ReLU halbiert das zweite Moment:** Für eine symmetrisch verteilte Präaktivierung $z$ mit Mittelwert null ergibt sich nach dem Nullsetzen der negativen Werte $\mathbb{E}[\mathrm{ReLU}(z)^2] = \frac{1}{2}\mathbb{E}[z^2]$.

<!--en:c6f37b02-->
- **He- oder Kaiming-Regel (He et al., 2015):** Die Varianz der Gewichte auf $\tau^2 = 2 / n_{\textrm{in}}$ verdoppeln, damit $\frac{1}{2}n_{\textrm{in}}\tau^2 = 1$ auch in beliebig tiefen ReLU-Netzen gilt.

<!--en:822722b9-->
::: {.slide title="Gemessene mittlere Quadrate der Aktivierungen über 10 ReLU-Schichten"}

<!--en:df02c2d8-->
[Xavier und He bei zunehmender Tiefe]{.kicker}

<!--en:607b82f1-->
- **Versuchsaufbau:** 10 ReLU-Schichten der Breite $h = 256$ mit drei Initialisierungen. Xavier und He unterscheiden sich je Schicht um den Faktor 2.

<!--en:178ca3f1-->
- **$\mathcal{N}(0, 0.01^2)$:** Rückgang auf $\sim 10^{-19}$ bis Schicht 10

<!--en:3c42d7f9-->
- **Xavier ($\tau^2 = 1/h$):** Jede ReLU-Schicht skaliert das mittlere Quadrat mit dem Faktor $1/2$. Nach 10 Schichten bleibt das $2^{-10} \approx 10^{-3}$-Fache.

<!--en:998fed5d-->
- **He ($\tau^2 = 2/h$):** Die mittleren Quadrate der Aktivierungen und der Gradienten im Rückwärtsdurchlauf bleiben in allen 10 Schichten nahe bei $1$, mit Abweichungen um höchstens etwa den Faktor $2$.

<!--en:b2b0a80f-->
::: {.slide title="Aufbau eines parametrisierbaren DeepMLP mit 10 Schichten"}

<!--en:58b892f1-->
[Aufbau eines tiefen Netzes]{.kicker}

<!--en:396523d9-->
Wie bestimmt die anfängliche Größenordnung der Gewichte die Trainierbarkeit? Für diesen Versuch stapelt `DeepMLP(std, lr, depth=10, width=256)` insgesamt $10$ versteckte ReLU-Schichten ($784 \to 256 \times 10 \to 10$) und initialisiert die Gewichte jeder Schicht mit der Standardabweichung `std(n_in, n_out)`:

<!--en:0c33c1bf-->
::: {.slide title="Training eines MLP mit 10 Schichten auf Fashion-MNIST"}

<!--en:e1f62e58-->
[Auswirkungen auf das Training]{.kicker}

<!--en:ff060616-->
- **Versuchsaufbau:** ReLU-Netz `DeepMLP` mit 10 versteckten Schichten ($784 \to 256 \times 10 \to 10$), 10 Epochen auf Fashion-MNIST bei $\eta = 0.05$

<!--en:a37fd9a7-->
- **$\mathcal{N}(0, 0.01^2)$:** Kein Lernfortschritt. Der Verlust bleibt bei $\log 10 \approx 2.30$ ($10\%$ Genauigkeit).

<!--en:e075bf74-->
- **Xavier:** Langsamer Start in Epoche 1 durch die Signalabschwächung auf das $2^{-10}$-Fache, danach Anstieg der Genauigkeit auf über $80\%$

<!--en:da1becf4-->
- **He-Initialisierung:** Der Trainingsverlust fällt ab Epoche 1 steil ab. Die Validierungsgenauigkeit erreicht zuverlässig $85$–$87\%$.

<!--en:9cec53a9-->
::: {.slide title="He-Initialisierung als seltene Voreinstellung in Frameworks" only="pytorch"}

<!--en:81d0cf5f-->
[Fallstrick bei Framework-Voreinstellungen]{.kicker}

<!--en:0b5ea8d0-->
Die linearen Schichten der Frameworks verwenden standardmäßig *keine* He-Initialisierung. Mit den Voreinstellungen schwächt sich das Signal in einer tiefen Folge einfacher ReLU-Schichten zunehmend ab:

<!--en:70fe5b9b-->
| Schicht im Framework | Voreingestellte Verteilung | Varianz der Gewichte $\tau^2$ | ReLU-Faktor $\frac{1}{2}h\tau^2$ ($h = 256$) |

<!--en:d41ca2ae-->
| **PyTorch** `nn.Linear` | $U(-1/\sqrt{n_{\textrm{in}}}, 1/\sqrt{n_{\textrm{in}}})$ | $\frac{1}{3 n_{\textrm{in}}}$ | $\frac{1}{6} \approx 0.167$ je Schicht |

<!--en:a176ba22-->
| **Explizite He-Initialisierung mit Normalverteilung** | $\mathcal{N}(0, 2/n_{\textrm{in}})$ | $\frac{2}{n_{\textrm{in}}}$ | **$1.000$ je Schicht** |

<!--en:af974185-->
::: {.slide title="He-Initialisierung als seltene Voreinstellung in Frameworks" only="tensorflow"}

<!--en:81d0cf5f-2-->
[Fallstrick bei Framework-Voreinstellungen]{.kicker}

<!--en:0b5ea8d0-2-->
Die linearen Schichten der Frameworks verwenden standardmäßig *keine* He-Initialisierung. Mit den Voreinstellungen schwächt sich das Signal in einer tiefen Folge einfacher ReLU-Schichten zunehmend ab:

<!--en:70fe5b9b-2-->
| Schicht im Framework | Voreingestellte Verteilung | Varianz der Gewichte $\tau^2$ | ReLU-Faktor $\frac{1}{2}h\tau^2$ ($h = 256$) |

<!--en:99c8d8d8-->
| **TensorFlow** `Dense` | Glorot-Gleichverteilung ($n_{\textrm{in}} = n_{\textrm{out}} = h$) | $\frac{1}{n_{\textrm{in}}}$ | $\frac{1}{2} = 0.500$ je Schicht |

<!--en:a176ba22-2-->
| **Explizite He-Initialisierung mit Normalverteilung** | $\mathcal{N}(0, 2/n_{\textrm{in}})$ | $\frac{2}{n_{\textrm{in}}}$ | **$1.000$ je Schicht** |

<!--en:d45ea19f-->
::: {.slide title="He-Initialisierung als seltene Voreinstellung in Frameworks" only="jax"}

<!--en:81d0cf5f-3-->
[Fallstrick bei Framework-Voreinstellungen]{.kicker}

<!--en:0b5ea8d0-3-->
Die linearen Schichten der Frameworks verwenden standardmäßig *keine* He-Initialisierung. Mit den Voreinstellungen schwächt sich das Signal in einer tiefen Folge einfacher ReLU-Schichten zunehmend ab:

<!--en:70fe5b9b-3-->
| Schicht im Framework | Voreingestellte Verteilung | Varianz der Gewichte $\tau^2$ | ReLU-Faktor $\frac{1}{2}h\tau^2$ ($h = 256$) |

<!--en:395606e2-->
| **JAX / Flax** `nnx.Linear` | LeCun-Normalverteilung $\mathcal{N}(0, 1/n_{\textrm{in}})$ | $\frac{1}{n_{\textrm{in}}}$ | $\frac{1}{2} = 0.500$ je Schicht |

<!--en:a176ba22-3-->
| **Explizite He-Initialisierung mit Normalverteilung** | $\mathcal{N}(0, 2/n_{\textrm{in}})$ | $\frac{2}{n_{\textrm{in}}}$ | **$1.000$ je Schicht** |

<!--en:7188d439-->
::: {.slide title="He-Initialisierung als seltene Voreinstellung in Frameworks" only="mxnet"}

<!--en:81d0cf5f-4-->
[Fallstrick bei Framework-Voreinstellungen]{.kicker}

<!--en:0b5ea8d0-4-->
Die linearen Schichten der Frameworks verwenden standardmäßig *keine* He-Initialisierung. Mit den Voreinstellungen schwächt sich das Signal in einer tiefen Folge einfacher ReLU-Schichten zunehmend ab:

<!--en:70fe5b9b-4-->
| Schicht im Framework | Voreingestellte Verteilung | Varianz der Gewichte $\tau^2$ | ReLU-Faktor $\frac{1}{2}h\tau^2$ ($h = 256$) |

<!--en:5f412ff8-->
| **MXNet** `nn.Dense` | Gleichverteilung $U(-0.07, 0.07)$ | $\frac{0.07^2}{3} \approx 0.00163$ | $\approx 0.209$ je Schicht |

<!--en:a176ba22-4-->
| **Explizite He-Initialisierung mit Normalverteilung** | $\mathcal{N}(0, 2/n_{\textrm{in}})$ | $\frac{2}{n_{\textrm{in}}}$ | **$1.000$ je Schicht** |

<!--en:51a32fc7-->
::: {.slide title="Symmetriebrechung: Warum konstante Anfangsgewichte scheitern"}

<!--en:2eb43f08-->
[Permutationssymmetrie]{.kicker}

<!--en:68a045b5-->
Warum nicht jeden Eintrag von $\mathbf{W}^{(1)}$ deterministisch auf die Konstante $c = \sqrt{2/n_{\textrm{in}}}$ setzen? Sie liefert genau das mittlere Quadrat $c^2 = 2/n_{\textrm{in}}$ der He-Initialisierung.

<!--en:05f80feb-->
- **Permutationsinvarianz:** Die Ausgabe eines MLP bleibt unverändert, wenn die Spalten von $\mathbf{W}^{(1)}$ und die entsprechenden Zeilen von $\mathbf{W}^{(2)}$ auf dieselbe Weise permutiert werden.

<!--en:cdd4e784-->
- **Deterministischer Fixpunkt der SGD-Dynamik:** Sind anfangs alle Spalten von $\mathbf{W}^{(1)}$ identisch ($c$) und alle Zeilen von $\mathbf{W}^{(2)}$ identisch ($c_{\textrm{out}}$), erhält jede versteckte Einheit denselben Gradienten (Differenz `0.00` im Beispiel oben). Die Einheiten bleiben identisch. Von der Breite $h$ bleibt effektiv $1$ Einheit.

<!--en:e0e6ac62-->
::: {.slide title="Grenzen der Initialisierung während des Trainings"}

<!--en:62517a61-->
[Grenzen der Initialisierung]{.kicker}

<!--en:afea9464-->
- **Änderung der Korrelationen nach dem ersten Schritt:** $\mathrm{Var}[o_j] = n_{\textrm{in}}\tau^2\gamma^2$ setzt unabhängige Gewichte und Eingaben in Schritt $0$ voraus. Sobald SGD die Gewichte $\mathbf{W}^{(k)}$ aktualisiert, können die Verstärkungsfaktoren der Schichten von $1$ abweichen.

<!--en:5864a9b8-->
- **Skalierungssymmetrie von ReLU:** Wegen $\mathrm{ReLU}(\alpha z) = \alpha\,\mathrm{ReLU}(z)$ für $\alpha > 0$ bleibt die Funktion unverändert, wenn die eingehenden Gewichte mit $\alpha$ und die ausgehenden Gewichte mit $1/\alpha$ multipliziert werden. Dabei ändern sich die Gradientennormen.

<!--en:720b02ec-->
- **Stabilisierung tiefer Netze ($K \ge 50$):** He- oder Xavier-Initialisierung mit **BatchNorm oder LayerNorm** und **Residualverbindungen** kombinieren.

<!--en:d3f14a32-->
::: {.slide title="Verständnisfragen zu Skalierung und Symmetrie bei der Initialisierung"}

<!--en:ffcb46a8-->
[Verständnisfragen]{.kicker}

<!--en:603c9650-->
Wissen über die Ausbreitung der Varianz und die Brechung der Symmetrie prüfen:

<!--en:8f97d295-->
1. **Skalierung nach He berechnen:** Die erste Schicht des ReLU-MLP auf Fashion-MNIST hat die Abmessungen $n_{\textrm{in}} = 784 \to n_{\textrm{out}} = 256$. Welche Standardabweichung $\tau$ der Gewichte schreiben Xavier und He vor?

<!--en:0979bfea-->
2. **Initialisierung bei Leaky ReLU:** Eine versteckte Schicht verwende Leaky ReLU $\phi(z) = \max(z, \alpha z)$ mit $\alpha = 0.1$. Welchen Wert hat bei symmetrisch verteiltem $z$ mit Mittelwert null der Erwartungswert $\mathbb{E}[\phi'(z)^2]$? Welche Gewichtsvarianz $\tau^2$ erhält das mittlere Quadrat?

<!--en:b0dab5ba-->
3. **Symmetriebrechung durch den Bias:** Alle Gewichte seien anfangs null ($\mathbf{W}^{(1)} = \mathbf{0}, \mathbf{W}^{(2)} = \mathbf{0}$). Bricht die Initialisierung von $\mathbf{b}^{(1)}$ mit verschiedenen Zufallszahlen im ersten Schritt die Symmetrie der versteckten Einheiten?

<!--en:557b98b6-->
**Lösungen:**

<!--en:9e4a8204-->
1. **$\tau$ in der ersten Schicht:** Xavier ergibt $\tau = \sqrt{2/(784 + 256)} = \sqrt{2/1040} \approx 0.0439$ und He ergibt $\tau = \sqrt{2/784} = 1/19.8 \approx 0.0505$ ($5\times$ so groß wie $0.01$).

<!--en:5ea0a7f4-->
2. **Leaky ReLU:** Es gilt $\phi'(z)^2 = 1$ für $z > 0$ (Wahrscheinlichkeit $1/2$) und der Wert beträgt $\alpha^2$ für $z < 0$ (Wahrscheinlichkeit $1/2$). Daraus folgen $\mathbb{E}[\phi'(z)^2] = \frac{1 + \alpha^2}{2}$ und $\tau^2 = \frac{2}{(1 + \alpha^2)n_{\textrm{in}}}$.

<!--en:3e6d2b8c-->
3. **Gewichte mit Anfangswert null:** Ist $\mathbf{W}^{(2)} = \mathbf{0}$, so gilt $\partial L / \partial \mathbf{H} = (\partial L / \partial \mathbf{O})\mathbf{W}^{(2)\top} = \mathbf{0}$ und daher $\partial L / \partial \mathbf{W}^{(1)} = \mathbf{0}$ für *alle* versteckten Einheiten im ersten Schritt, unabhängig von $\mathbf{b}^{(1)}$. Zufällige Gewichte sind entscheidend.

<!--en:236a66ff-->
::: {.slide title="Zusammenfassung: Numerische Stabilität und Initialisierung"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:07434fe5-->
| Initialisierung oder Aktivierung | Varianz der Gewichte $\tau^2$ | Faktor je Schicht ($\text{ReLU}$) | Signalskalierung über 10 Schichten | Rolle und Verhalten im Deep Learning |

<!--en:13dee1ec-->
| **Kleine feste Streuung $\mathcal{N}(0, 0.01^2)$** | $10^{-4}$ | $\frac{1}{2}h(10^{-4}) = 0.0128$ | $\sim 10^{-17}$ (verschwindet) | Frühe Schichten lernen nicht. Der Verlust des MLP mit 10 Schichten bleibt bei $\log 10$. |

<!--en:ecd409c2-->
| **Große feste Streuung $\mathcal{N}(0, 1)$** | $1$ | $\frac{1}{2}h(1) = 128$ | $\sim 10^{19}$ (explodiert) | Überlauf bei `float32` mit `inf`/`NaN` oder instabile SGD-Schritte |

<!--en:a5df54e8-->
| **Sigmoidfunktion ($\sigma(z)$)** | $1/n_{\textrm{in}}$ | $\mathbb{E}[\sigma'(z)^2] \le 1/16$ | $\le 16^{-9} \approx 10^{-11}$ | Steigung $\le 1/4$ und Sättigung in den Randbereichen führen zu verschwindenden Gradienten |

<!--en:334271e3-->
| **Xavier / Glorot** | $\frac{2}{n_{\textrm{in}} + n_{\textrm{out}}}$ | $\frac{1}{2}$ (bei ReLU), $1$ (bei $\tanh$) | $2^{-10} \approx 10^{-3}$ (bei ReLU) | Standard für lineare Schichten, $\tanh$-Schichten und Projektionsschichten in Attention-Modellen |

<!--en:f10688e3-->
| **He / Kaiming** | $\frac{2}{n_{\textrm{in}}}$ | $\frac{1}{2}n_{\textrm{in}}\bigl(\frac{2}{n_{\textrm{in}}}\bigr) = 1$ | $\approx 1.0$ über alle Schichten | Standard für ReLU- und GELU-Netze. Auch tiefe Netze lernen von Beginn an. |

<!--en:9e2c6d60-->
| **Konstante Gewichte ($w_{ij} = c$)** | $c^2 = 2/n_{\textrm{in}}$ | Norm erhalten, Rang $1$ | Alle Spalten identisch | Permutationssymmetrie bleibt erhalten. $h$ Einheiten wirken wie $1$ Einheit. |
