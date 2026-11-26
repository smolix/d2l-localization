<!--en:bf55a7a0-->
# Lineare Regression von Grund auf

<!--en:6c096796-->
Wir kennen nun die Theorie der linearen Regression (:numref:`sec_linear_regression`) und haben einen synthetischen Datensatz, dessen wahre Parameter wir kennen (:numref:`sec_synthetic-regression-data`). Damit können wir die Theorie in Code umsetzen und an den Daten erproben. In diesem Abschnitt schreiben wir alle Bestandteile des Trainings selbst: die Initialisierung der Parameter, den linearen Vorwärtsdurchlauf (engl. forward pass), den quadratischen Verlust (engl. squared loss), die Ableitungsberechnung, den Minibatch-Iterator und die Schleife für den stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD). Dazu genügen elementare Tensoroperationen und automatisches Differenzieren (engl. automatic differentiation). Da wir jeden Schritt selbst schreiben, sehen wir, wie die Gradienten fließen und wie Hyperparameter wie die Lernrate (engl. learning rate) die Konvergenz beeinflussen. Sobald wir die einzelnen Schritte verstehen, fassen wir sie in :numref:`sec_oo-design` zu wiederverwendbaren Klassen zusammen (`Module`, `DataModule` und `Trainer`). In :numref:`sec_linear_concise` genügen für dasselbe Modell wenige Zeilen Code, weil wir Schichten, eingebaute Verlustfunktionen (engl. loss functions) und Optimierer des Frameworks nutzen.

<!--en:1396cf9a-->
## Modell, Verlust und Gradient

<!--en:ddf0cb95-->
Erzeugen wir zunächst $2{,}000$ synthetische Beispiele ($1{,}000$ für das Training und $1{,}000$ für die Validierung) mit den wahren Parametern $\mathbf{w}^* = (2, -3.4)^\top$ und $b^* = 4.2$ sowie Rauschen mit der Standardabweichung $\sigma = 0.01$:

<!--en:4eeafb8a-->
### Die Parameter

<!--en:0c91ca15-->
Unser Modell hat einen Gewichtsvektor $\mathbf{w} \in \mathbb{R}^2$ und einen skalaren Bias $b \in \mathbb{R}$. Wir initialisieren die Gewichte mit kleinen Zufallswerten aus $\mathcal{N}(0, 0.01^2)$ und setzen den Bias auf $0$. Bei der linearen Regression können wir ebenso gut alle Parameter mit null initialisieren (siehe Übung :exref:`ex_linear-regression-scratch-initialization`). Sobald wir jedoch in :numref:`chap_perceptrons` mehrere versteckte Schichten aufeinanderstapeln, brauchen wir zufällige Gewichte, um Symmetrien zu brechen.

<!--en:18f3ed98-->
Für beide Tensoren setzen wir `requires_grad=True`, damit das Autograd-System von PyTorch jede Operation mit diesen Tensoren aufzeichnet.

<!--en:b9d0e3cc-->
Für beide Parameter rufen wir `attach_grad()` auf. Damit reservieren wir Speicher für die Gradienten und weisen MXNet an, die Operationen innerhalb von `autograd.record()` zu verfolgen.

<!--en:e4c67612-->
Wir verpacken beide Tensoren in `tf.Variable`. Solche Variablen überwacht das `GradientTape` von TensorFlow automatisch.

<!--en:4e829160-->
Arrays in JAX sind unveränderlich. `jax.grad` differenziert reine Funktionen direkt nach ihren Argumenten, ohne an den Arrays Markierungen zu ändern.

<!--en:9a7f0043-->
### Der Vorwärtsdurchlauf

<!--en:ac1c3252-->
Gegeben seien die Entwurfsmatrix $\mathbf{X} \in \mathbb{R}^{|\mathcal{B}| \times d}$ eines Minibatchs, die Parameter $\mathbf{w} \in \mathbb{R}^{d \times 1}$ und der skalare Bias $b$. Für die lineare Vorhersage $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$ genügen ein einziges Matrix-Vektor-Produkt und eine Addition mit Broadcasting:

<!--en:762f27d1-->
### Der Verlust

<!--en:348530b5-->
Als Nächstes implementieren wir den halben mittleren quadratischen Fehler $L(\mathbf{w}, b) = \frac{1}{2|\mathcal{B}|}\sum_{i \in \mathcal{B}} (\hat{y}^{(i)} - y^{(i)})^2$. Wir bringen `y` in die Form eines Spaltenvektors `(-1, 1)`, passend zu `y_hat`. Hätte ein Tensor die Form `(n, 1)` und der andere die Form `(n,)`, entstünde bei der Subtraktion durch Broadcasting unbemerkt eine Matrix der Form `(n, n)` mit allen paarweisen Differenzen. Das Tückische an diesem Fehler ist, dass das Training weiterläuft. Der Mittelwert aller $n^2$ quadrierten Differenzen ist am kleinsten, wenn jede Vorhersage dem Mittelwert der Labels entspricht. SGD treibt daher $\mathbf{w}$ gegen $\mathbf{0}$ und $b$ gegen $\bar{y}$. Der Verlust sinkt gleichmäßig auf die halbe Varianz der Labels, bei unseren Daten also auf $\frac{1}{2}(2^2 + 3.4^2) \approx 7.8$. Eine fallende Verlustkurve sagt deshalb wenig darüber aus, ob der Code korrekt ist. Später in diesem Abschnitt vergleichen wir den Verlust mit dem Wert, den er erreichen sollte, und die Parameter mit ihren wahren Werten.

<!--en:699278a4-->
### Der Gradient

<!--en:7ced92d2-->
Bevor wir uns auf automatisches Differenzieren verlassen, prüfen wir an einem einzelnen Minibatch mit 32 Beispielen, ob es dieselben Gradienten liefert wie die von Hand hergeleiteten Formeln aus :numref:`subsec_linear-regression-minibatch-sgd`:

<!--en:a0c1cb57-->
Wählen wir per Slice die ersten 32 Trainingsbeispiele `(Xb, yb)` aus. Vergleichen wir die analytischen Gradienten mit den Ergebnissen, die automatisches Differenzieren liefert:

<!--en:be03f928-->
Die größte Abweichung zwischen entsprechenden Einträgen ist null (oder höchstens $\sim 10^{-7}$, die Rundungsgenauigkeit von `float32`). Damit ist bestätigt, dass Autograd den erwarteten Minibatch-Gradienten berechnet. Von nun an lassen wir das Framework unseren Vorwärtsdurchlauf und unseren Verlust automatisch differenzieren.

<!--en:d5d9a066-->
## Training von Grund auf

<!--en:0a21023b-->
### Das Parameterupdate

<!--en:03d148cf-->
In jedem Schritt aktualisieren wir die Parameter entgegen ihren Gradienten gemäß $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\,(\nabla_{\mathbf{w}} L, \nabla_b L)$. Dabei achten wir darauf, dass das Parameterupdate selbst nicht im Berechnungsgraphen aufgezeichnet wird.

<!--en:5573628b-->
### Minibatches

<!--en:6b6d15f0-->
Wir wollen den Datensatz in zufälligen Minibatches der Größe $|\mathcal{B}|$ durchlaufen. Dazu zieht unser Generator `minibatches` zu Beginn jeder Epoche eine neue zufällige Permutation der Indizes $0, \ldots, n - 1$. Er liefert dann nacheinander Slices der Größe `batch_size` (Übung :exref:`ex_linear-regression-scratch-why-reshuffle`):

<!--en:d3378927-->
Bei $1{,}000$ Trainingsbeispielen und `batch_size=32` erhalten wir in einer Epoche $\lceil 1000 / 32 \rceil = 32$ Minibatches: $31$ volle Batches mit je $32$ Beispielen und einen letzten Batch mit den übrigen $8$ Beispielen. Betrachten wir einen Durchlauf des Generators. Was geschieht, wenn wir versehentlich ein zweites Mal über *denselben* Generator iterieren?

<!--en:99101c97-->
Bei jedem vollen Minibatch haben die Tensoren des ausgegebenen Paars `(Xb, yb)` die Formen `(32, 2)` und `(32, 1)`. Der erste Durchlauf (eine *Epoche*) liefert wie erwartet alle $32$ Minibatches, der zweite keinen einzigen. Ein Python-Generator lässt sich nur einmal durchlaufen. Danach liefert er ohne Fehlermeldung keine weiteren Werte. Deshalb ruft unsere Trainingsschleife zu Beginn jeder Epoche `minibatches(...)` erneut auf.

<!--en:b53114de-->
### Die Trainingsschleife

<!--en:002ca26c-->
In `train(lr, num_epochs, batch_size)` setzen wir nun die vollständige Trainingsschleife zusammen. In jeder Epoche iterieren wir über `minibatches(X, y, batch_size)`, berechnen den Minibatch-Verlust, differenzieren ihn nach `w` und `b` und führen den SGD-Schritt aus:

<!--en:ce25bb1f-->
Am Ende jedes Schleifendurchlaufs setzen wir alle Gradienten auf null. PyTorch addiert nämlich jeden neuen Gradienten zum gespeicherten Gradienten, statt ihn zu ersetzen (:numref:`sec_autograd`).

<!--en:23eb57f1-->
Der Kern der Trainingsschleife umfasst nur ein Dutzend Codezeilen. Mit denselben vier Schritten trainieren wir im weiteren Verlauf des Buches jedes tiefe Netz: Vorwärtsdurchlauf $\rightarrow$ Verlust $\rightarrow$ Rückwärtsdurchlauf $\rightarrow$ Parameterupdate (siehe :numref:`fig_linreg_sgd_cycle`).

<!--en:eea94d0a-->
![Ein Schritt des Minibatch-SGD. Der Vorwärtsdurchlauf berechnet die Vorhersagen für einen Minibatch. Die Verlustfunktion fasst sie zu einer einzigen Zahl zusammen, die der Rückwärtsdurchlauf nach den Parametern differenziert. Das Parameterupdate verschiebt die Parameter entgegen dem Gradienten. Mit dem nächsten Minibatch beginnt der Zyklus von vorn.](../img/mdl-linreg-sgd-cycle.svg)

<!--en:c9217201-->
## Prüfung der Implementierung

<!--en:c974d439-->
### Der erwartete Verlust

<!--en:8504ecd2-->
Wie klein sollte der Verlust bei erfolgreichem Training werden? Selbst bei den wahren Parametern $(\mathbf{w}^*, b^*)$ entsprechen die Residuen dem gaußschen Rauschen $\hat{y}^{(i)} - y^{(i)} = -\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$. Der erwartete Verlust erreicht hier also die *durch das Rauschen gesetzte Untergrenze* (engl. irreducible noise floor):

<!--en:bac96f95-->
Bei der Initialisierung ($\mathbf{w} \approx \mathbf{0}, b = 0$) ist das Residuum dagegen näherungsweise $-y = -(\mathbf{x}^\top\mathbf{w}^* + b^* + \epsilon)$. Unter dieser Näherung hat es den Mittelwert $-b^* = -4.2$ und die Varianz $\|\mathbf{w}^*\|_2^2 + \sigma^2 = 2^2 + (-3.4)^2 + 0.01^2 = 15.5601$. Damit beträgt der erwartete Anfangsverlust

<!--en:151e1dda-->
Im Laufe des Trainings sollte der Verlust demnach um fünf Größenordnungen fallen, von rund $16.6$ bis zur rauschbedingten Untergrenze $0.00005$.

<!--en:6b0922c6-->
Mit derselben Rechnung lässt sich auch vorhersagen, wie *schnell* der Verlust sinken sollte. Bei beliebigen Parameterwerten beträgt das Residuum $r = \mathbf{x}^\top(\mathbf{w} - \mathbf{w}^*) + (b - b^*) - \epsilon$. Die beiden Merkmale sind unabhängige standardnormalverteilte Zufallsvariablen, und das Rauschen ist von beiden unabhängig. Deshalb hat jeder gemischte Term den Erwartungswert null, und es gilt

<!--en:7de38882-->
Der Gradient dieses erwarteten Verlusts nach $(\mathbf{w}, b)$ ist genau der *Parameterfehler* $(\mathbf{w} - \mathbf{w}^*, b - b^*)$. Bei einem Gradientenschritt der Schrittweite $\eta$ bleibt vom Parameterfehler nur der Anteil $1 - \eta$ und vom *Überschussverlust* (engl. excess loss) oberhalb der Untergrenze nur der Anteil $(1 - \eta)^2$. Minibatch-Gradienten stimmen nur im Mittel mit dem erwarteten Gradienten überein. Die Rechnung beschreibt die Trajektorie daher nur näherungsweise. Bei $\eta = 0.03$ sollte eine Epoche mit $32$ Minibatch-Updates den Überschussverlust ungefähr um den Faktor $1 / (1 - 0.03)^{64} \approx 7$ verringern.

<!--en:cd477beb-->
### Der Trainingslauf

<!--en:b042a525-->
Führen wir `train` über 10 Epochen mit der Lernrate $\eta = 0.03$ und der Minibatch-Größe $|\mathcal{B}| = 32$ aus. Nach jeder Epoche tragen wir den Validierungsverlust (engl. validation loss) auf einer logarithmischen Skala auf:

<!--en:4c17ff8c-->
Tatsächlich fällt der Validierungsverlust entlang einer Geraden. Der Wert ist nach jeder Epoche rund $7\times$ kleiner als zuvor. Um Epoche 7 oder 8 flacht die Kurve nahe der vorhergesagten Untergrenze von $5 \times 10^{-5}$ ab. Vergleichen wir für jede der ersten sieben Epochen den gemessenen Überschussverlust mit der Vorhersage aus unserer geschlossenen Formel.

<!--en:079d514b-->
In den ersten Epochen stimmen gemessener und vorhergesagter Überschussverlust gut überein. Allmählich gehen sie auseinander, bis Epoche 7 höchstens um einen Faktor von 3 oder 4. Woher kommt diese leichte Abweichung? Unsere idealisierte Vorhersage setzt voraus, dass die Stichprobenkovarianzmatrix $\mathbf{H} = \frac{1}{n}\sum_i \tilde{\mathbf{x}}^{(i)}(\tilde{\mathbf{x}}^{(i)})^\top$ (siehe Übung :exref:`ex_linear-regression-scratch-stability`) exakt der Einheitsmatrix entspricht. Bei einer endlichen Stichprobe von $1{,}000$ Beispielen weichen ihre Eigenwerte um bis zu etwa $10\%$ von $1$ ab. Über mehrere Epochen summieren sich diese kleinen Abweichungen. Der tatsächliche Trainingslauf eilt der idealisierten Vorhersage daher etwas voraus oder hinkt ihr etwas hinterher. Sowohl die Abklingrate als auch die rauschbedingte Untergrenze stimmen mit der Theorie überein.

<!--en:4312a4f3-->
### Die geschätzten Parameter

<!--en:f9d55cac-->
Da unser Datensatz synthetisch ist, können wir die gelernten Parameter $(\mathbf{w}, b)$ auch direkt mit den wahren Werten $\mathbf{w}^* = (2, -3.4)^\top$ und $b^* = 4.2$ vergleichen:

<!--en:6ec9bfe6-->
Sowohl die Gewichte als auch der Bias stimmen bis auf ein Tausendstel mit den wahren Werten überein. Das entspricht der statistisch bedingten Grenze der Schätzgenauigkeit $\sigma / \sqrt{n} \approx 3 \times 10^{-4}$, die wir in :numref:`sec_synthetic-regression-data` mit der Methode der kleinsten Quadrate gefunden haben.

<!--en:f319e23f-->
### Die Lernrate

<!--en:0236cbb5-->
Was geschieht, wenn wir die Lernrate ungünstig wählen? Führen wir dieselbe Schleife über 10 Epochen mit vier Lernraten aus, die drei Größenordnungen abdecken: $\eta \in \{0.003, 0.03, 0.3, 3\}$. Für jeden Lauf geben wir den endgültigen Validierungsverlust aus (Übung :exref:`ex_linear-regression-scratch-stability`):

<!--en:1ad5d5bb-->
Die Ergebnisse stimmen bei allen Lernraten mit unserer Theorie überein. Bei $\eta = 0.003$ sinkt der Fehler pro Schritt nur auf das $0.997$-Fache. Nach zehn Epochen liegt der Verlust daher noch zwischen etwa $2$ und $3$, nahe am vorhergesagten Wert $16.6 \times 0.997^{640} \approx 2.4$. Bei $\eta = 0.3$ erreicht das Modell die rauschbedingte Untergrenze schon in der ersten Epoche. Bei $\eta = 3$ hingegen verdoppelt sich mit jedem Update ungefähr der Betrag des Fehlers und das Vorzeichen wechselt. Das führt rasch zu einem Überlauf und zum Wert `nan`. Die Stabilitätsgrenze von $2$ ergibt sich aus der *Skalierung unserer Daten*, weil unsere Merkmale die Varianz eins haben. Sie ist keine universelle Konstante von SGD. Skalieren wir ein Merkmal mit $s > 1$, wächst die Krümmung des Verlusts in Richtung des zugehörigen Gewichts um den Faktor $s^2$. Die größte stabile Lernrate sinkt dadurch auf etwa $2 / s^2$. Das ist ein wichtiger praktischer Grund, Eingabemerkmale vor dem Training zu standardisieren (:numref:`sec_pandas`).

<!--en:d1b55dc8-->
In :numref:`sec_oo-design` fassen wir als Nächstes die soeben programmierten Trainingsschritte in wiederverwendbaren objektorientierten Klassen zusammen (`Module`, `DataModule` und `Trainer`). In :numref:`sec_linear_concise` sehen wir dann, wie wir unsere eigenen Tensoren, unsere Verlustfunktion und unser Parameterupdate durch fertige Komponenten des Frameworks ersetzen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:b48c5d36-->
Für eine vollständige Trainingsschleife genügen vier Bausteine: ein parametrisierter Vorwärtsdurchlauf $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$, ein skalarer Verlust $\frac{1}{2|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$, automatisches Differenzieren im Rückwärtsmodus und ein Minibatch-SGD-Schritt außerhalb der Gradientenaufzeichnung. Eine fallende Verlustkurve allein beweist noch nicht, dass der Code fehlerfrei ist. Selbst wenn die Tensorformen unbemerkt voneinander abweichen, kann der Verlust bis auf die halbe Varianz der Labels sinken. Deshalb prüfen wir unsere Implementierung quantitativ an synthetischen Daten. Der Validierungsverlust fällt exponentiell mit dem vorhergesagten Faktor $(1 - \eta)^{64}$ pro Epoche bis zur rauschbedingten Untergrenze $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$. Die gelernten Parameter weichen von $(\mathbf{w}^*, b^*)$ nur um einen Betrag der Größenordnung $O(\sigma/\sqrt{n})$ ab. Der Vergleich der Lernraten zeigt: Bei kleinen Schrittweiten kommt das Training nur langsam voran. Bei mittleren Schrittweiten erreicht es die rauschbedingte Untergrenze rasch. Bei Schrittweiten über $2/\mu_{\max}$ schießen die Iterierten über das Minimum der quadratischen Mulde hinaus und divergieren.

<!--en:2294243a-->
## Übungen

<!--en:b8f90efc-->
1. [code] **Extreme Anfangswerte.** `init_params` zieht die Gewichte mit der Standardabweichung `sigma=0.01`. Sagen Sie für zwei Fälle voraus, ob das Training die rauschbedingte Untergrenze erreicht: wenn alle Gewichte anfangs genau null sind und wenn `sigma=1000` gilt. Schätzen Sie mithilfe von :eqref:`eq_linreg-scratch-expected-loss`, wie viele Epochen der zweite Fall bei einer Lernrate von 0.03 zusätzlich benötigt. Überprüfen Sie dann Ihre Schätzung.
   :exlabel:`ex_linear-regression-scratch-initialization`

<!--en:96f68e6e-->
2. [code] **Zweite Ableitungen.** Berechnen Sie von Hand die $3 \times 3$-Hesse-Matrix des Minibatch-Verlusts bezüglich $\boldsymbol{\theta} = (\mathbf{w}, b)$. Geben Sie an, von welchen Größen sie abhängt. Berechnen Sie die Matrix anschließend durch automatisches Differenzieren mit `torch.func.hessian` (:numref:`sec_autograd`). Verwenden Sie einen festen Minibatch der synthetischen Daten und berechnen Sie die Matrix einmal bei der Initialisierung und einmal nach dem Training. Vergleichen Sie beide Ergebnisse mit Ihrer Formel und mit der Krümmung von :eqref:`eq_linreg-scratch-expected-loss`.

<!--en:dc815248-->
2. [code] **Zweite Ableitungen.** Berechnen Sie von Hand die $3 \times 3$-Hesse-Matrix des Minibatch-Verlusts bezüglich $\boldsymbol{\theta} = (\mathbf{w}, b)$. Geben Sie an, von welchen Größen sie abhängt. Berechnen Sie die Matrix anschließend durch automatisches Differenzieren mit `jax.hessian` (oder `jax.jacrev(jax.grad(...))`, siehe :numref:`sec_autograd`). Verwenden Sie einen festen Minibatch der synthetischen Daten und berechnen Sie die Matrix einmal bei der Initialisierung und einmal nach dem Training. Vergleichen Sie beide Ergebnisse mit Ihrer Formel und mit der Krümmung von :eqref:`eq_linreg-scratch-expected-loss`.

<!--en:aeb9463d-->
2. [code] **Zweite Ableitungen.** Berechnen Sie von Hand die $3 \times 3$-Hesse-Matrix des Minibatch-Verlusts bezüglich $\boldsymbol{\theta} = (\mathbf{w}, b)$. Geben Sie an, von welchen Größen sie abhängt. Berechnen Sie die Matrix anschließend durch automatisches Differenzieren. Zeichnen Sie dazu den Gradienten innerhalb eines äußeren `tf.GradientTape` auf und rufen Sie `jacobian` auf (:numref:`sec_autograd`). Verwenden Sie einen festen Minibatch der synthetischen Daten und berechnen Sie die Matrix einmal bei der Initialisierung und einmal nach dem Training. Vergleichen Sie beide Ergebnisse mit Ihrer Formel und mit der Krümmung von :eqref:`eq_linreg-scratch-expected-loss`.

<!--en:fefd91d7-->
2. [code] **Zweite Ableitungen.** Berechnen Sie von Hand die $3 \times 3$-Hesse-Matrix des Minibatch-Verlusts bezüglich $\boldsymbol{\theta} = (\mathbf{w}, b)$. Geben Sie an, von welchen Größen sie abhängt. Da `autograd` in MXNet das Differenzieren erster Ordnung im Rückwärtsmodus unterstützt (:numref:`sec_autograd`), bestimmen Sie die drei Spalten der Hesse-Matrix durch zentrale Differenzen des von Autograd berechneten Gradienten gemäß $(\nabla L(\boldsymbol{\theta} + h\mathbf{e}_j) - \nabla L(\boldsymbol{\theta} - h\mathbf{e}_j)) / (2h)$. Verwenden Sie einen festen Minibatch der synthetischen Daten und berechnen Sie die Matrix einmal bei der Initialisierung und einmal nach dem Training. Vergleichen Sie beide Ergebnisse mit Ihrer Formel und mit der Krümmung von :eqref:`eq_linreg-scratch-expected-loss`.

<!--en:14ab5122-->
1. [code] **Stabilität und Minibatch-Größe.** ● Aus dem erwarteten Verlust folgt, dass Schritte entlang seines negativen Gradienten für $\eta > 2$ divergieren. Sei $\tilde{\mathbf{x}} = (\mathbf{x}, 1)$ eine Eingabe, die wir für den Bias um eine $1$ erweitern.
    1. Zeigen Sie für den Gradientenabstieg auf dem gesamten Trainingsdatensatz, dass jeder Schritt die Differenz zwischen den Parametern und der Lösung der kleinsten Quadrate mit der Matrix $\mathbf{I} - \eta \mathbf{H}$ multipliziert. Dabei ist $\mathbf{H}$ der Mittelwert von $\tilde{\mathbf{x}}\tilde{\mathbf{x}}^\top$ über den Trainingsdatensatz. Berechnen Sie die Eigenwerte von $\mathbf{H}$ und sagen Sie die größte stabile Lernrate voraus.
    1. Trainieren Sie mit Minibatches der Größen 1000, 32 und 1. Verwenden Sie in jedem Lauf dieselbe Anzahl von Updates, etwa 320. Wählen Sie ein Raster von Lernraten. Bestimmen Sie darin für jede Minibatch-Größe die größte Lernrate, bei der der endgültige Trainingsverlust unter $10^{-3}$ liegt.
    1. Zeigen Sie, dass ein Schritt mit einem einzigen Beispiel die Komponente des Parameterfehlers in Richtung $\tilde{\mathbf{x}}$ mit $1 - \eta \|\tilde{\mathbf{x}}\|^2$ multipliziert. Erklären Sie anhand von $\mathbb{E}\|\tilde{\mathbf{x}}\|^2 = 3$ die Grenze, die Sie für Minibatches mit nur einem Beispiel gefunden haben.
   :exlabel:`ex_linear-regression-scratch-stability`

<!--en:74705400-->
1. **Warum neu mischen?** `minibatches` zieht in jeder Epoche eine neue zufällige Reihenfolge. Erklären Sie, welche Eigenschaft des Minibatch-Gradienten dadurch erhalten bleibt. Konstruieren Sie dann eine feste Reihenfolge des Trainingsdatensatzes, etwa durch Sortierung nach dem Label, bei der sich Minibatch-SGD ohne erneutes Mischen anders verhält. Beschreiben Sie die Trajektorie von $(\mathbf{w}, b)$ innerhalb einer Epoche und über mehrere Epochen hinweg.
   :exlabel:`ex_linear-regression-scratch-why-reshuffle`

<!--en:7ff3e9e9-->
1. [code] **Robuste Verlustfunktionen.** ● Ersetzen Sie `squared_loss` durch den mittleren absoluten Fehler `d2l.reduce_mean(d2l.abs(y_hat - d2l.reshape(y, (-1, 1))))`.
    1. Trainieren Sie mit den synthetischen Daten. Vergleichen Sie die geschätzten Parameter und die Verlustkurve mit den Ergebnissen für den quadratischen Verlust.
    1. Setzen Sie ein Trainingslabel auf $y^{(5)} = 10{,}000$, wie bei der Verfälschung in :numref:`subsec_linear-regression-loss-function`, und trainieren Sie mit beiden Verlustfunktionen. Zeichnen Sie $b$ nach jedem Minibatch-Update auf und tragen Sie den Wert für beide Läufe gegen die Iterationsnummer auf. Erklären Sie den Kurvenverlauf beim quadratischen Verlust und geben Sie an, was den Endwert bestimmt.
    1. Verwenden Sie für die verfälschten Daten als Verlust den Mittelwert des Huber-Verlusts über den Minibatch:

        $$l_\delta(y, y') = \begin{cases}\frac{1}{2}(y-y')^2 & \textrm{ if } |y-y'| \leq \delta \\ \delta \left(|y-y'| - \frac{\delta}{2}\right) & \textrm{ otherwise,}\end{cases}$$

        Sein quadratischer Teil stimmt mit dem quadratischen Verlust aus diesem Abschnitt überein. Trainieren Sie für $\delta \in \{1, 10, 10^2, 10^3, 10^4\}$. Tragen Sie den Abstand $|\hat{b} - b^*|$ des gelernten Bias vom wahren Bias $b^* = 4.2$ gegen $\delta$ auf. Skalieren Sie beide Achsen logarithmisch. Erklären Sie die Kurvenform anhand der Ableitung von $l_\delta$ nach $y'$ und setzen Sie das Ergebnis in Beziehung zu den Kurven der Strafterme in :numref:`fig_linreg-loss-menu`.
   :exlabel:`ex_linear-regression-scratch-robust-losses`

<!--en:b47a84f9-->
[Diskussion](https://d2l.discourse.group/t/42)

<!--en:a5e79d23-->
[Diskussion](https://d2l.discourse.group/t/43)

<!--en:e66727b4-->
[Diskussion](https://d2l.discourse.group/t/201)

<!--en:fa843179-->
[Diskussion](https://d2l.discourse.group/t/17976)

<!--en:87b5f1cc-->
Lineare Regression von Grund auf<br>
**Parameter als Tensoren · Schutz vor Broadcasting-Fehlern · Prüfung von Autograd · Trainingsschleife mit Minibatch-SGD · Verlustuntergrenze bei Konvergenz**

<!--en:147014b3-->
::: {.slide title="Warum eine fallende Verlustkurve die Korrektheit nie beweist"}

<!--en:49269ae8-->
[Eine Falle bei der Fehlersuche]{.kicker}

<!--en:a3e84709-->
- **Unbemerkter Broadcasting-Fehler:** Beim Subtrahieren des eindimensionalen Labeltensors `y` der Form `(32,)` von `y_hat` der Form `(32, 1)` entsteht durch Broadcasting eine Matrix der Form `(32, 32)` mit allen $1{,}024$ paarweisen Differenzen.

<!--en:f92390c6-->
- **Trügerische Verlustkurve:** Die Minimierung des Mittelwerts über $32 \times 32$ Einträge zieht die Vorhersagen zu $\bar{y}$. Der Verlust sinkt gleichmäßig, doch es gilt $\mathbf{w} \to \mathbf{0}$, obwohl die wahren Gewichte $\mathbf{w}^* = (2, -3.4)^\top$ sind.

<!--en:53c48109-->
- **Zweifache Prüfung:** Immer die Untergrenze des Validierungsverlusts ($\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$) und die Genauigkeit der geschätzten Parameter ($\|\hat{\mathbf{w}} - \mathbf{w}^*\|_\infty \lesssim 10^{-3}$) prüfen.

<!--en:8eb89377-->
::: {.slide title="Parameterinitialisierung mit Gradientenverfolgung"}

<!--en:019b723c-->
[Initialisierung von Grund auf]{.kicker}

<!--en:5ace2010-->
- **Gradientenverfolgung:** Die Parameter so anlegen, dass das Framework den Verlust nach ihnen differenzieren kann.

<!--en:d824ed01-->
::: {.slide title="Vorwärtsdurchlauf und Formanpassung beim quadratischen Verlust"}

<!--en:ec3910ed-->
[Grundoperationen für Modell und Verlust]{.kicker}

<!--en:9af2bf17-->
- **Passende Formen sicherstellen:** `squared_loss` ruft vor der Subtraktion `d2l.reshape(y, (-1, 1))` auf. So hat der Vektor der elementweisen Residuen garantiert die Form `(32, 1)`.

<!--en:05ff6b5e-->
::: {.slide title="Prüfung von Autograd anhand des analytischen Gradienten"}

<!--en:f07244df-->
[Gradientenprüfung]{.kicker}

<!--en:7754519d-->
- **Exakte Übereinstimmung:** Die Ergebnisse von `autograd` im Rückwärtsmodus stimmen mit $\nabla_{\mathbf{w}} L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\mathbf{X}^\top\mathbf{r}$ und $\partial_b L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\mathbf{1}^\top\mathbf{r}$ bis auf das Maschinenepsilon von `float32` überein ($\sim 10^{-7}$).

<!--en:4ab5cc17-->
::: {.slide title="Warum SGD in jeder Epoche neu mischen muss"}

<!--en:695c58aa-->
[Erwartungstreue bei zufälliger Auswahl]{.kicker}

<!--en:35b439d3-->
- **Erwartungstreue erfordert zufällige Auswahl:** Für einen gleichverteilt gezogenen Minibatch gilt $\mathbb{E}[\nabla L_{\mathcal{B}}] = \nabla L$. Mit dem Batchmittelwert $\bar{y}_{\mathcal{B}} \approx 4.2 \pm 0.7$ steigt $b$ Schritt für Schritt in Richtung $b^* = 4.2$.

<!--en:4a84ee4c-->
- **Problem bei sortierten Daten:** Nach Sortierung nach dem Label enthält Batch 1 die 32 kleinsten Labels ($\bar{y}_{\mathcal{B}} \approx -4.6$). Dadurch sinkt $b$, bewegt sich also *weg* von $4.2$, und die Parameter schwanken in jeder Epoche zyklisch und stark.

<!--en:a158ab12-->
::: {.slide title="Der Minibatch-Generator ist nach einem Durchlauf erschöpft"}

<!--en:85923e2b-->
[Eine Falle beim Durchlaufen der Daten]{.kicker}

<!--en:5f2da984-->
- **Generator nur einmal verwendbar:** Nach einem Durchlauf ist ein Python-Generator erschöpft und liefert $0$ Batches in einer zweiten `for`-Schleife. In jeder Epoche `minibatches(...)` erneut aufrufen.

<!--en:ef71ecd5-->
::: {.slide title="Vier Schritte in einer Trainingsschleife von zwölf Zeilen"}

<!--en:d811697c-->
[Die vollständige selbst geschriebene Schleife]{.kicker}

<!--en:e2ea1656-->
- **Gradientenaufzeichnung:** Das Parameterupdate läuft außerhalb der Aufzeichnung. Jeder Schritt verwendet nur den Gradienten des aktuellen Minibatchs.

<!--en:55e6850d-->
::: {.slide title="Vorhersage der Konvergenzrate vor dem Programmlauf"}

<!--en:2139f508-->
[Quantitative Beschreibung der Verlustkurve]{.kicker}

<!--en:6764b800-->
- **Anfänglicher Überschussverlust:** Bei einem Start nahe $(\mathbf{0}, 0)$ beträgt der Anfangsverlust $\frac{1}{2}(2^2 + (-3.4)^2 + 4.2^2) \approx 16.6$. Er liegt mehr als $5$ Größenordnungen über der Untergrenze $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$.

<!--en:ea4970a5-->
- **Abnahme pro Epoche ($32$ Schritte bei $\eta = 0.03$):** Jeder Schritt multipliziert den Parameterfehler mit $(1 - \eta) = 0.97$ und den Überschussverlust mit $0.97^2$. Nach einer Epoche bleibt vom Überschussverlust der Anteil $0.97^{64} \approx 1/7$ (**$7\times$ kleiner pro Epoche**).

<!--en:a7de162c-->
::: {.slide title="Trainingslauf: exponentielle Abnahme bis zur rauschbedingten Untergrenze"}

<!--en:63718f6d-->
[Empirische Prüfung]{.kicker}

<!--en:b6bfb9ed-->
Ein Lauf von `train(lr=0.03)` über 10 Epochen bestätigt die exponentielle Abnahme ($7\times$ kleiner pro Epoche) und das Plateau an der theoretischen, rauschbedingten Untergrenze:

<!--en:8e013550-->
Auf logarithmischer Skala erscheint die exponentielle Konvergenz $(1 - \eta)^{64k}$ in den ersten 7 Epochen als fallende Gerade, bis der Überschussverlust unter die rauschbedingte Untergrenze $\sigma^2/2 = 5 \times 10^{-5}$ fällt.

<!--en:2846cf2d-->
::: {.slide title="Vergleich der geschätzten Parameter mit (2, −3.4) und 4.2"}

<!--en:086ee197-->
[Genauigkeit der Parameterschätzung]{.kicker}

<!--en:84b2d4bf-->
- **Statistischer und numerischer Fehler:** Die gelernten Parameter stimmen mit $(\mathbf{w}^*, b^*)$ bis auf $\sim \sigma / \sqrt{n} \approx 3.2 \times 10^{-4}$ überein.

<!--en:b6701c45-->
- **Warum eine Abweichung von $10^{-3}$ am Verlust kaum erkennbar ist:** Eine Parameterabweichung von $5 \times 10^{-4}$ erhöht den Validierungsverlust nur um $\sim 4 \times 10^{-7}$ ($<1\%$ der rauschbedingten Untergrenze von $5 \times 10^{-5}$).

<!--en:828fab7c-->
::: {.slide title="Stabilität der Lernrate: von langsamer Konvergenz bis zur Divergenz"}

<!--en:3881afee-->
[Grenzen der Schrittweite]{.kicker}

<!--en:b2122e15-->
- **Stabilitätsgrenze ($0 < \eta < 2/\mu_{\max}$):** Bei $\eta = 0.003$ sinkt der Verlust nur langsam (auf $\approx 2$–$3$). Bei $\eta \in \{0.03, 0.3\}$ erreicht er die Untergrenze $5 \times 10^{-5}$. Bei $\eta = 3 > 2$ multipliziert jeder Schritt den Fehler mit $1 - 3 = -2$, bis ein Überlauf zu `nan` führt.

<!--en:bcbe2478-->
::: {.slide title="Verständnisfragen zu Gradienten, Verlustskalierung und Stabilität" only="pytorch"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:29e819e3-->
Wie funktioniert die Trainingsschleife in diesen drei Fällen?

<!--en:5e4cecdf-->
1. **`zero_grad()` vergessen:** Wenn in der Trainingsschleife `param.grad.zero_()` fehlt, welchen Vektor zieht Schritt $t$ von `param` ab? Wie verhalten sich die Iterierten in einer quadratischen Mulde?

<!--en:eed58c33-->
2. **Summe oder Mittelwert:** Eine Verlustfunktion berechnet die *Summe* $\sum_{i \in \mathcal{B}} (\hat{y}^{(i)} - y^{(i)})^2$ über einen Minibatch der Größe $|\mathcal{B}| = 32$, ohne durch $2|\mathcal{B}|$ zu teilen. Welche Lernrate $\eta$ ergibt denselben Schritt wie die eigene Implementierung mit $\eta_{\textrm{scratch}} = 0.03$?

<!--en:ce98df1a-->
3. **Stabilität bei skalierter Eingabe:** Wird das Merkmal $x_1$ mit $10$ multipliziert, beträgt seine Varianz $100$ statt $1$. Wie ändert sich die größte stabile Lernrate $\eta_{\max}$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:9f78f5fc-->
1. Ohne Zurücksetzen zieht Schritt $t$ den Vektor $\eta \sum_{s=1}^t \mathbf{g}_s$ ab, also die mit der Lernrate gewichtete Summe aller bisherigen Gradienten. Das wirkt wie Momentum ohne Abklingen: Die Iterierten schießen über das Minimum hinaus und oszillieren, ohne zu konvergieren.

<!--en:7e0d5430-->
2. Der summierte Verlust ist $2|\mathcal{B}| = 64$-mal so groß wie der Verlust der eigenen Implementierung. Sein Gradient ist daher $64\mathbf{g}$. Division der Lernrate durch $64$ ergibt $\eta = 0.03 / 64 \approx 4.69 \times 10^{-4}$.

<!--en:0e672b88-->
3. Die Skalierung von $x_1$ mit $10$ erhöht den größten Eigenwert der Hesse-Matrix $\mathbf{H} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ um den Faktor $10^2 = 100$. Die Stabilitätsgrenze $\eta_{\max} = 2 / \mu_{\max}$ sinkt somit um den Faktor $100\times$ (von $\approx 2$ auf $\approx 0.02$).

<!--en:c669f7b8-->
::: {.slide title="Verständnisfragen zu Gradienten, Verlustskalierung und Stabilität" except="pytorch"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:96ed735b-->
Wie funktioniert die Trainingsschleife in diesen zwei Fällen?

<!--en:7dea0389-->
1. **Summe oder Mittelwert:** Eine Verlustfunktion berechnet die *Summe* $\sum_{i \in \mathcal{B}} (\hat{y}^{(i)} - y^{(i)})^2$ über einen Minibatch der Größe $|\mathcal{B}| = 32$, ohne durch $2|\mathcal{B}|$ zu teilen. Welche Lernrate $\eta$ ergibt denselben Schritt wie die eigene Implementierung mit $\eta_{\textrm{scratch}} = 0.03$?

<!--en:db60e24c-->
2. **Stabilität bei skalierter Eingabe:** Wird das Merkmal $x_1$ mit $10$ multipliziert, beträgt seine Varianz $100$ statt $1$. Wie ändert sich die größte stabile Lernrate $\eta_{\max}$?

<!--en:557b98b6-2-->
**Lösungen:**

<!--en:d54fb870-->
1. Der summierte Verlust ist $2|\mathcal{B}| = 64$-mal so groß wie der Verlust der eigenen Implementierung. Sein Gradient ist daher $64\mathbf{g}$. Division der Lernrate durch $64$ ergibt $\eta = 0.03 / 64 \approx 4.69 \times 10^{-4}$.

<!--en:0154e750-->
2. Die Skalierung von $x_1$ mit $10$ erhöht den größten Eigenwert der Hesse-Matrix $\mathbf{H} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ um den Faktor $10^2 = 100$. Die Stabilitätsgrenze $\eta_{\max} = 2 / \mu_{\max}$ sinkt somit um den Faktor $100\times$ (von $\approx 2$ auf $\approx 0.02$).

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und zentrale Ergebnisse"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:737a3372-->
| Trainingsschritt | Mathematische Operation | Worauf es bei der Implementierung ankommt | Quantitative Prüfung ($\sigma = 0.01, n = 1000$) |

<!--en:215f3976-->
| **1. Vorwärtsdurchlauf und Verlust** | $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$, $L_{\mathcal{B}} = \frac{1}{2|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ | `y` in die Form `(-1, 1)` bringen, um Broadcasting nach dem Muster `(B,) - (B, 1) -> (B, B)` zu verhindern | Anfangsverlust $\mathbb{E}[L(\mathbf{0}, 0)] \approx 16.6$ |

<!--en:eda190b3-->
| **2. Rückwärtsdurchlauf** | $\nabla_{\mathbf{w}} L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y})$ | `autograd` für 1 Batch mit der von Hand hergeleiteten Formel vergleichen | Maximale Abweichung $\lesssim 10^{-7}$ (Maschinenepsilon von `float32`) |

<!--en:617925ae-->
| **3. SGD-Update** | $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \nabla L_{\mathcal{B}}$ | Parameter außerhalb der Gradientenaufzeichnung aktualisieren, nur den aktuellen Gradienten verwenden | Stabil für $0 < \eta < 2/\mu_{\max}$. Bei $\eta=0.03$ ist der Verlust nach jeder Epoche $7\times$ kleiner |

<!--en:72b424e3-->
| **4. Neu mischen je Epoche** | Gleichverteilte Permutation von $\{0, \ldots, n-1\}$ | Generator in jeder Epoche neu erzeugen, da er nach Epoche 1 erschöpft ist | Verlust erreicht ein Plateau bei $\sigma^2/2 = 5 \times 10^{-5}$. Es gilt $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_\infty \lesssim 10^{-3}$ |

<!--en:c7efc539-->
**Ausblick (objektorientierter Entwurf und kompakte Implementierung):** Zuerst die Trainingsschleife in den wiederverwendbaren Klassen `Module`, `DataModule` und `Trainer` kapseln (`LinearRegressionScratch`), anschließend Schichten, MSE-Verlustfunktionen und Optimierer aus der höheren Programmierschnittstelle des Frameworks einsetzen (`LinearRegression`).
