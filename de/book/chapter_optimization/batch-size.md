<!--en:cadd2e6d-->
# Batchgröße

<!--en:2bacf06e-->
In der Praxis wählen wir die Minibatchgröße (engl. minibatch size) $b$ oft so, dass der Speicher des Beschleunigers voll ausgelastet ist, und behalten sie dann bei. Sobald wir das Training auf mehrere GPUs verteilen oder die Laufzeit gegen den gesamten Rechenaufwand in FLOPs abwägen, stellt sich eine grundlegende Frage. Wenn wir die Minibatchgröße $b$ verdoppeln und damit pro Schritt doppelt so viel parallel berechnen, erreichen wir den angestrebten Verlust mit halb so vielen Schritten?

<!--en:169b74b1-->
In :numref:`sec_sgd` haben wir gesehen, dass ein Minibatch aus $b$ Beispielen die Stichprobenvarianz des Gradienten auf $\operatorname{tr}(\boldsymbol{\Sigma})/b$ senkt. Daraus ergeben sich zwei unterschiedliche Regime:

<!--en:62d29a2a-->
1. **Kleine Batches (rauschdominiertes Regime):** Ist das Stichprobenrauschen größer als der wahre Gradient, so halbiert eine Verdopplung von $b$ die Rauschvarianz. Das erlaubt eine höhere Lernrate (engl. learning rate), mit der sich die benötigte Schrittzahl $S(b)$ halbiert. Die Gesamtzahl der verarbeiteten Beispiele $E(b) = b\,S(b)$ bleibt dabei konstant (*perfekte lineare Skalierung*).

<!--en:1c4d282c-->
2. **Große Batches (krümmungsdominiertes Regime):** Ist $b$ so groß, dass der Minibatch-Gradient den wahren Gradienten auf dem gesamten Datensatz gut annähert, bringt eine weitere Erhöhung von $b$ immer weniger. Die Schrittzahl $S(b)$ sinkt nur noch wenig, während der Daten- und Rechenaufwand $E(b)$ linear wächst.

<!--en:754e990d-->
:citet:`McCandlish.Kaplan.Amodei.ea.2018` zeigten, dass der Übergang zwischen den beiden Regimen bei der **kritischen Batchgröße** (engl. critical batch size) $b_{\text{crit}}$ liegt. Sie lässt sich anhand der **Skala des Gradientenrauschens** (engl. gradient-noise scale) $b_{\text{noise}} = \operatorname{tr}(\boldsymbol{\Sigma}) / \|\nabla f\|^2$ vorhersagen. Wir entwickeln einen Schätzer für $b_{\text{noise}}$, der zwei Batchgrößen verwendet. Anschließend leiten wir die Grenzkurve für Schrittzahl und Datenbedarf sowie die Regeln zur Skalierung der Lernrate für SGD ($\eta \propto b$) und Adam ($\eta \propto \sqrt{b}$) her. Beide Vorhersagen prüfen wir an `TinyLM` und `FashionCNN`.

<!--en:0bfe3da5-->
## Die Skala des Gradientenrauschens

<!--en:dd283bf6-->
### Signal, Rauschen und ein Schätzer mit zwei Batchgrößen

<!--en:a7bb4e29-->
Sei $\nabla f(\mathbf{x})$ der exakte Gradient auf dem gesamten Datensatz und $\hat{\mathbf{g}}_b$ der über $b$ unabhängige Beispiele gemittelte Minibatch-Gradient. Die Gradienten der einzelnen Beispiele haben die Kovarianz $\boldsymbol{\Sigma}$. Es gilt $\mathbb{E}[\hat{\mathbf{g}}_b] = \nabla f$ und $\mathbb{E}[\|\hat{\mathbf{g}}_b - \nabla f\|^2] = \operatorname{tr}(\boldsymbol{\Sigma})/b$. Damit ist der Erwartungswert der quadrierten Norm des Minibatch-Gradienten die Summe aus Signalleistung $\|\nabla f\|^2$ und Rauschleistung $\operatorname{tr}(\boldsymbol{\Sigma})/b$:

<!--en:d8fc180b-->
Signal- und Rauschleistung sind gleich, wenn die Batchgröße der **einfachen Skala des Gradientenrauschens** entspricht :cite:`McCandlish.Kaplan.Amodei.ea.2018`:

<!--en:c165652c-->
Wir können sowohl $\operatorname{tr}\boldsymbol{\Sigma}$ als auch $\|\nabla f\|^2$ ohne vollständigen Durchlauf durch den Datensatz schätzen. Dazu messen wir die mittlere quadrierte Gradientennorm $N_b = \mathbb{E}[\|\hat{\mathbf{g}}_b\|^2]$ bei zwei Batchgrößen $b_{\text{s}} < b_{\text{l}}$ und lösen das lineare Gleichungssystem :eqref:`eq_sqnorm-bias` mit zwei Gleichungen und zwei Unbekannten:

<!--en:829be3bc-->
Zunächst laden wir WikiText-103 und Fashion-MNIST in den Speicher des Beschleunigers. Wir definieren zwei Funktionen, die zufällige Minibatches ziehen, `sample_fashion(b)` und `sample_lm(b)`:

<!--en:395800a5-->
Mit diesen Funktionen für das Ziehen der Minibatches definieren wir die Hilfsfunktion `ce_loss` für die Kreuzentropie, `grad_sq_norm` für die quadrierte Gradientennorm über alle Parameter und `noise_scale` für die Schätzung mit zwei Batchgrößen. Wir speichern sie in `d2l`:

<!--en:4249b65b-->
Wir werten `noise_scale` für `TinyLM` ($b_{\text{s}} = 8, b_{\text{l}} = 256$) bei der Initialisierung und erneut nach $500$ Schritten mit Adam ($\eta = 3 \times 10^{-3}$) aus:

<!--en:1fd356fa-->
Wächst die Rauschskala bei einem konvolutionären Netz für Bilder auf dieselbe Weise? Wir wiederholen die Messung an `FashionCNN` mit Batches von $b_{\text{s}} = 16$ und $b_{\text{l}} = 2{,}048$ Bildern, den voreingestellten Größen. Wir messen bei der Initialisierung und nach $200$ Schritten mit Adam ($\eta = 0.004$, die Lernrate in der folgenden Versuchsreihe bei $b = 256$):

<!--en:9c4ab989-->
Bei `TinyLM` erfordert nach der Initialisierung jedes Fenster dieselbe grobe Anpassung der Unigramm-Häufigkeiten. Daher ist das Signal $\|\nabla f\|^2$ groß, und es gilt $b_{\text{noise}} \approx 2$ Sequenzen. Sobald das Modell die gemeinsamen statistischen Eigenschaften der Bytes gelernt hat (bis Schritt $500$), sinkt $\|\nabla f\|^2$. Die Varianz zwischen den Sequenzen $\operatorname{tr}\boldsymbol{\Sigma}$ bleibt bestehen. Dadurch wächst $b_{\text{noise}}$ auf mehr als das Zehnfache und erreicht $\sim 50\text{--}150$ Sequenzen.

<!--en:47ce57dd-->
`FashionCNN` verhält sich anders. Seine zehn Klassen sind gleich häufig vertreten. Das untrainierte Netz findet daher keine Anpassung, die allen Bildern gemeinsam ist. Die Rauschskala liegt bereits beim ersten Schritt in der Größenordnung von $100$ Bildern. Über die $200$ Schritte ändert sich der Schätzwert um weniger als den Faktor zwei, je nach Lauf nach oben oder unten. Er bleibt in der Größenordnung von $100$ Bildern.

<!--en:5183c8d4-->
## Batchgröße, Fortschritt pro Schritt und Skalierung der Lernrate

<!--en:33ca62ec-->
### Optimale Schrittweite und Effizienzgrenze

<!--en:c4201e54-->
Für eine $L$-glatte Funktion gilt die Abstiegsschranke $f(\mathbf{x} - \eta\hat{\mathbf{g}}_b) - f(\mathbf{x}) \le -\eta\nabla f^\top \hat{\mathbf{g}}_b + \frac{L}{2}\eta^2\|\hat{\mathbf{g}}_b\|^2$. Bilden wir auf beiden Seiten den Erwartungswert und setzen :eqref:`eq_sqnorm-bias` ein, erhalten wir

<!--en:9dfd9547-->
Minimieren wir die rechte Seite bezüglich $\eta$, erhalten wir die optimale Schrittweite $\eta_{\text{opt}}(b)$ und die garantierte Abnahme pro Schritt:

<!--en:2485581c-->
*(Ersetzen wir die skalare Glattheitsschranke $L\mathbf{I}$ durch die lokale Hesse-Matrix $\mathbf{H}$, erhalten wir dieselben Ausdrücke mit der krümmungsgewichteten Rauschskala $\operatorname{tr}(\mathbf{H}\boldsymbol{\Sigma}) / (\nabla f^\top\mathbf{H}\nabla f)$ :cite:`McCandlish.Kaplan.Amodei.ea.2018`. Als Näherung dafür dient $b_{\text{noise}}$ aus :eqref:`eq_noise-scale`, die ohne Hesse-Matrix auskommt.)*

<!--en:b16a653f-->
Jeder Schritt bei Batchgröße $b$ erzielt den Bruchteil $(1 + b_{\text{noise}}/b)^{-1}$ des Fortschritts eines exakten Schritts auf dem gesamten Datensatz. Erfordert ein angestrebter Verlust $S_{\min}$ Schritte auf dem gesamten Datensatz, so brauchen wir bei dieser Batchgröße $S(b)$ Schritte und insgesamt $E(b) = b\,S(b)$ Trainingsbeispiele:

<!--en:9f953a45-->
Dabei ist $E_{\min} = b_{\text{noise}} S_{\min}$ das kleinste Datenbudget. Wir erreichen es im Grenzfall kleiner Batches $b \ll b_{\text{noise}}$. Bei der **kritischen Batchgröße** $b_{\text{crit}} = b_{\text{noise}}$ liegen sowohl die Schrittzahl als auch die Zahl der Beispiele beim Doppelten ihres theoretischen Minimums ($S = 2S_{\min}, E = 2E_{\min}$). Dieser Punkt bildet das Knie der Pareto-Front $((E/E_{\min} - 1)(S/S_{\min} - 1) = 1$).

<!--en:f2fe8d67-->
### Lineare Skalierung und Wurzelskalierung der Lernrate

<!--en:dcf4be1b-->
Wir erhöhen die Batchgröße im rauschdominierten Regime von $b_0$ auf $b$ ($b_0, b \ll b_{\text{noise}}$). Wie muss sich dabei die Lernrate $\eta$ ändern?

<!--en:2d22cced-->
* **SGD und Momentum (lineare Skalierung):** Aus :eqref:`eq_best-step` folgt für $b \ll b_{\text{noise}}$ die Beziehung $\eta_{\text{opt}}(b) \approx \frac{b}{L b_{\text{noise}}} \propto b$. Entsprechend wirken $k$ Schritte der Weite $\eta_0$ bei Batchgröße $b_0$ wie ein Schritt der Weite $k\eta_0$ bei Batchgröße $k b_0$. Die stationäre Rauschgrenze $\propto \eta / b$ bleibt dabei erhalten :cite:`Goyal.Dollar.Girshick.ea.2017,Smith.Kindermans.Ying.ea.2018`.

<!--en:720c7ef6-->
* **Adam und AdamW (Wurzelskalierung):** Im rauschdominierten Regime habe eine Komponente des Gradienten den Mittelwert $\bar{g}$ und die Standardabweichung $\sigma / \sqrt{b}$. Adam dividiert den Gradienten durch seinen quadratischen Mittelwert $\sqrt{\hat{v}_t} \approx \sigma / \sqrt{b}$. Das ergibt pro Schritt eine Drift von $-\eta(\sqrt{b}/\sigma)\bar{g}$ und einen Rauschanteil von $-\eta z$ ($z \sim \mathcal{N}(0, 1)$). Der Nenner $\sigma/\sqrt{b}$ liefert bereits den Faktor $\sqrt{b/b_0}$. Soll ein einziger großer Schritt dieselbe Drift und kumulierte Varianz wie $k = b/b_0$ kleine Schritte haben, müssen wir daher $\eta$ mit $\sqrt{b/b_0}$ multiplizieren :cite:`Malladi.Lyu.Panigrahi.ea.2022`:

<!--en:8470d842-->
Sobald $b$ die kritische Batchgröße $b_{\text{crit}}$ überschreitet, dominiert das Gradientenrauschen den Nenner nicht mehr. Die optimale Lernrate wächst dann nicht weiter mit $b$, sondern nähert sich der deterministischen Krümmungsgrenze :cite:`Shallue.Lee.Antognini.ea.2019`.

<!--en:62c64e47-->
## Empirische Überprüfung der Schrittzahl bis zum Zielverlust und der Skalierungsregeln

<!--en:9e42aae0-->
Wir wollen messen, wie viele Schritte $S(b)$ nötig sind, um auf zurückgehaltenen Daten einen vorgegebenen Verlust zu erreichen. Dazu definieren wir zunächst `eval_loss`. Die Funktion wertet `ce_loss` auf den Validierungstensoren in Blöcken von je $4{,}096$ Beispielen aus:

<!--en:85cfdc94-->
Die Funktion `train_to_target` prüft den Wert von `eval_loss` alle `eval_every` Schritte innerhalb von `d2l.train_steps`. Sie gibt die Schrittzahl zurück, bei der der Zielwert `target` erreicht ist, oder `inf`, wenn das Budget von `max_steps` Schritten erschöpft ist:

<!--en:e0019e25-->
Zuerst führen wir `train_to_target` für `TinyLM` aus. Der angestrebte Validierungsverlust beträgt $1.9$ Nat pro Byte. Wir verwenden Batches von $b \in \{4, 16, 64, 256\}$ Sequenzen und starten stets mit derselben Initialisierung. Die Lernrate für Adam skalieren wir nach $\eta = 0.003\sqrt{b / 64}$:

<!--en:6937fccb-->
Für `FashionCNN` wiederholen wir die Messung der Schrittzahl bis zum Zielverlust $0.35$ mit Batches von $b \in \{8, 32, 128, 512, 2048\}$ Bildern. Wir skalieren die Lernrate nach $\eta = 0.004\sqrt{\min(b, 512)/256}$ bis zur Batchgröße $b = 512$:

<!--en:ab6caecc-->
Da $S(b) = S_{\min}(1 + b_{\text{crit}}/b)$ linear von $1/b$ abhängt, kann `fit_steps` für beide Modelle die asymptotische Mindestschrittzahl $S_{\min}$ und die empirische kritische Batchgröße $b_{\text{crit}}$ mit der Methode der gewichteten kleinsten Quadrate bestimmen:

<!--en:e9818edb-->
Für jeden Lauf normieren wir die Zahl der Beispiele $E(b) = b\,S(b)$ mit $E_{\min} = b_{\text{crit}} S_{\min}$ und die Schrittzahl mit $S_{\min}$. So können wir beide Modelle mit der universellen Pareto-Front :eqref:`eq_steps-examples` vergleichen:

<!--en:f7017527-->
Die Messpunkte beider Architekturen liegen nahe an der universellen Kurve $(E/E_{\min} - 1)(S/S_{\min} - 1) = 1$. Nahe $E = E_{\min}$ verläuft die Kurve so steil, dass sich ein Punkt bei einem kleinen Messfehler in der Schrittzahl weit nach oben oder unten verschiebt. Die Anpassung ergibt kritische Batchgrößen von $b_{\text{crit}} \sim 100$ Sequenzen für `TinyLM` und $b_{\text{crit}} \sim 200\text{--}400$ Bildern für `FashionCNN`. Bei `TinyLM` stimmt das Ergebnis mit der obigen Schätzung von $b_{\text{noise}}$ aus zwei Batchgrößen überein. Sie lag bei etwas weniger als $100$ Sequenzen. Bei `FashionCNN` ist das Ergebnis zwei- bis viermal so groß wie der Schätzwert von etwa $100$ Bildern. Die einfache Rauschskala lässt die Krümmung außer Acht und liefert daher nur die Größenordnung von $b_{\text{crit}}$.

<!--en:dbd79c7e-->
Zum Schluss überprüfen wir die Regeln für die lineare Skalierung ($\eta \propto b$) und die Wurzelskalierung ($\eta \propto \sqrt{b}$) der Lernrate aus :eqref:`eq_lr-rules` direkt an `FashionCNN`. Das Datenbudget beträgt stets $12{,}800$ Bilder ($1/5$ Epoche). Die Hilfsfunktion `final_loss_at` führt `d2l.lr_sweep` über `num_examples // b` Schritte bei Batchgröße `b` aus. Als Endverluste gibt sie die Mittelwerte über das letzte Viertel jedes Laufs zurück:

<!--en:898e4ff2-->
Bei SGD mit Momentum ($\beta = 0.9$) erhöhen wir die Batchgröße um den Faktor $8\times$ von $b = 8$ auf $b = 64$. Bei der Lernratensuche über `sgd_lrs` verschiebt sich die optimale Lernrate um den Faktor $4\text{--}8\times$ nach rechts (bei $b = 64$ ändert sich der Endverlust zwischen $\eta = 0.05$ und $0.1$ kaum):

<!--en:cf749ca5-->
Bei Adam erhöhen wir die Batchgröße um den Faktor $64\times$ von $b = 2$ auf $b = 128$. In der Lernratensuche über `adam_lrs` verschiebt sich die optimale Lernrate nur um den Faktor $4\text{--}8\times$ ($2$ bis $3$ Rasterschritte mit dem Faktor $2\times$). Die Verschiebung entspricht damit dem Faktor $\sqrt{64} = 8\times$ statt $64\times$:

<!--en:439f8d23-->
### Wahl der Batchgröße in der Praxis

<!--en:de13f5a4-->
1. **Zuerst die Recheneinheiten auslasten:** Erhöhen Sie $b$ mindestens so weit, bis die Laufzeit pro Schritt mit $b$ linear zu wachsen beginnt (:numref:`sec_sgd`).

<!--en:23ea1577-->
2. **$b_{\text{noise}}$ nach der Aufwärmphase schätzen:** Messen Sie `noise_scale`, nachdem der Verlust zu Beginn gesunken ist (z. B. nach einigen Hundert Trainingsschritten). Bei der Initialisierung kann $b_{\text{noise}}$ sehr klein sein und im Training um $1\text{--}2$ Größenordnungen wachsen, wie bei `TinyLM`. Deshalb steigt die Batchgröße in der Aufwärmphase großer Pre-Training-Läufe wie dem von GPT-3 schrittweise an :cite:`brown2020language`.

<!--en:e9f0228a-->
3. **$b$ im Verhältnis zu $b_{\text{crit}} \approx b_{\text{noise}}$ wählen:** Sind Rechenaufwand oder Datenmenge der Engpass, wählen Sie $b \lesssim 0.3\,b_{\text{crit}}$. Ist die Laufzeit auf parallelen Beschleunigern der Engpass, wählen Sie $b \approx b_{\text{crit}}$ (dann gelten $S = 2S_{\min}$ und $E = 2E_{\min}$). Vermeiden Sie $b \gg b_{\text{crit}}$.

<!--en:01eccfbd-->
4. **$\eta$ nach :eqref:`eq_lr-rules` skalieren:** Skalieren Sie die Lernrate für SGD mit oder ohne Momentum gemäß $\eta \propto b$ und für AdamW gemäß $\eta \propto \sqrt{b}$, jeweils bis $b_{\text{crit}}$. Kombinieren Sie dies mit einer linearen Aufwärmphase (:numref:`subsec_scheduler-warmup`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3bc7148b-->
Die **Skala des Gradientenrauschens** $b_{\text{noise}} = \operatorname{tr}(\boldsymbol{\Sigma}) / \|\nabla f\|^2$ bezeichnet die Batchgröße, bei der die Leistung des Stichprobenrauschens gleich der Signalleistung ist. Sie lässt sich im laufenden Training aus den quadrierten Gradientennormen bei zwei Batchgrößen $b_{\text{s}} < b_{\text{l}}$ schätzen, ohne den gesamten Datensatz zu durchlaufen (:eqref:`eq_sqnorm-bias`). Bei abgestimmten Lernraten folgen die bis zum Zielverlust benötigte Schrittzahl $S(b)$ und die Zahl der verarbeiteten Beispiele $E(b) = b\,S(b)$ den hyperbolischen Beziehungen $S(b)/S_{\min} = 1 + b_{\text{crit}}/b$ und $E(b)/E_{\min} = 1 + b/b_{\text{crit}}$. Das Knie der Kurve bei $b_{\text{crit}} \approx b_{\text{noise}}$ trennt das Regime kleiner Batches mit geringem Datenbedarf ($b \ll b_{\text{crit}}$) vom Regime großer Batches, in dem die Schrittzahl kaum noch sinkt ($b \gg b_{\text{crit}}$). Unterhalb von $b_{\text{crit}}$ wächst die optimale Lernrate für SGD mit und ohne Momentum linear ($\eta \propto b$), für Adam und AdamW mit der Quadratwurzel der Batchgröße ($\eta \propto \sqrt{b}$). Sobald $b$ die kritische Batchgröße $b_{\text{crit}}$ überschreitet, flacht der Anstieg ab.

<!--en:2294243a-->
## Übungen

<!--en:788c2e66-->
1. **Größere Batchgröße bei unveränderter Lernrate.** Zeigen Sie mithilfe von :eqref:`eq_noisy-step` und :eqref:`eq_best-step`: Bleibt $\eta$ bei $\eta_0 = \eta_{\text{opt}}(b_0)$ und steigt die Batchgröße auf $b > b_0$, so wächst die garantierte Abnahme pro Schritt um den Faktor $2 - (1 + b_{\text{noise}}/b)/(1 + b_{\text{noise}}/b_0) < 2$. Selbst für $b \to \infty$ steigt sie höchstens auf das Doppelte.
   :exlabel:`ex_batch-size-fixed-rate`

<!--en:6bb1aa02-->
1. **Zeitoptimale und rechenoptimale Batchgröße.** Ein Schritt bei Batchgröße $b$ habe die Laufzeit $t(b) = t_0(1 + b/b_{\text{sat}})$. Leiten Sie mit :eqref:`eq_steps-examples` die Batchgröße $b^\star$ her, welche die Gesamtlaufzeit $T(b) = t(b)S(b)$ minimiert. Berechnen Sie den zugehörigen relativen Datenaufwand $E(b^\star)/E_{\min}$.
   :exlabel:`ex_batch-size-time-optimal-compute-optimal`

<!--en:6b6b083c-->
1. [code] **Entwicklung von Signal und Rauschen im Training.** Ändern Sie `noise_scale` so, dass die Funktion $\operatorname{tr}\boldsymbol{\Sigma}$, $\|\nabla f\|^2$ und ihr Verhältnis $b_{\text{noise}}$ einzeln zurückgibt. Erfassen Sie alle drei Größen für `TinyLM` bei den Schritten $0, 25, 50, 100, 200, 500, 1000, 2000, 3000$. Überprüfen Sie, dass der Anstieg um den Faktor $100\times$ bei $b_{\text{noise}}$ hauptsächlich auf die Abnahme des Signals $\|\nabla f\|^2$ und weniger auf eine Zunahme von $\operatorname{tr}\boldsymbol{\Sigma}$ zurückgeht.
   :exlabel:`ex_batch-size-noise-scale-over-training`

<!--en:718b1d9a-->
1. [code] **Skalierung oberhalb der kritischen Batchgröße.** Erweitern Sie die Versuchsreihe zur Schrittzahl bis zum Zielverlust für `TinyLM` aus :numref:`subsec_batch-size-steps` auf $b = 1{,}024$. Vergleichen Sie die Wurzelregel $\eta \propto \sqrt{b}$ mit der Variante, bei der $\eta$ auf den Wert für $b = 256$ begrenzt bleibt. Erklären Sie, warum sich die Konvergenz verschlechtert, wenn Sie $\eta$ oberhalb von $b_{\text{crit}}$ weiter erhöhen.
   :exlabel:`ex_batch-size-past-elbow`

<!--en:27eaf5d4-->
[Diskussion](https://d2l.discourse.group/t/36765)

<!--en:d8dc56bb-->
Batchgröße, Skala des Gradientenrauschens und kritische Batchgröße<br>**Signal- und Rauschleistung · Schätzer mit zwei Batchgrößen · Pareto-Front für Schritte und Beispiele · kritische Batchgröße · lineare Regel und √b-Regel für die Lernrate**

<!--en:b719757b-->
::: {.slide title="Wann halbiert eine Verdopplung der Batchgröße die Schrittzahl?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:ad38c601-->
- **Kleiner Batch ($b: 4 \to 16$, $4\times$):** Das Rauschen dominiert. Die Mittelung senkt die Stichprobenvarianz $\operatorname{tr}(\boldsymbol{\Sigma})/b$. Die **Schrittzahl sinkt um den Faktor $\approx 3\times$**, der Datenbedarf bleibt fast konstant ($\approx 1.1\text{--}1.3\times E(4)$).

<!--en:43fbae7a-->
- **Großer Batch ($b: 64 \to 256$, $4\times$):** Die Krümmung dominiert. Die Minibatch-Gradienten stimmen bereits weitgehend mit $\nabla f$ überein. Die **Schrittzahl sinkt nur um den Faktor $\approx 2\times$**, Datenbedarf und FLOPs verdoppeln sich.

<!--en:96eb71f5-->
- **Knie der Kurve ($b_{\text{crit}} \approx b_{\text{noise}}$):** Seine Lage bestimmt die Skala des Gradientenrauschens $b_{\text{noise}} = \operatorname{tr}(\boldsymbol{\Sigma}) / \|\nabla f\|^2$ (McCandlish et al., 2018).

<!--en:263794ad-->
::: {.slide title="Signalleistung, Rauschleistung und Skala des Gradientenrauschens"}

<!--en:797916de-->
[Signal-Rausch-Verhältnis eines Minibatchs]{.kicker}

<!--en:4195f4ab-->
- **Zerlegung des Erwartungswerts der quadrierten Norm:**
  $$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \underbrace{\|\nabla f\|^2}_{\text{Signal power}} + \underbrace{\frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}}_{\text{Noise power}}$$

<!--en:fe05393a-->
- **Einfache Skala des Gradientenrauschens:** Bei dieser Batchgröße sind Rauschleistung und Signalleistung gleich groß:
  $$b_{\text{noise}} = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{\|\nabla f\|^2}$$

<!--en:743d8572-->
- Für $b \ll b_{\text{noise}}$ besteht $\hat{\mathbf{g}}_b$ vorwiegend aus Stichprobenrauschen. Für $b \gg b_{\text{noise}}$ gilt $\hat{\mathbf{g}}_b \approx \nabla f$.

<!--en:07aa677e-->
::: {.slide title="Schätzung von b_noise aus zwei Batchgrößen ohne vollständigen Datendurchlauf"}

<!--en:027e1295-->
[Schätzung der Rauschskala im laufenden Training]{.kicker}

<!--en:19b86129-->
- Um $\|\nabla f\|^2$ direkt zu berechnen, müssten wir den gesamten Datensatz durchlaufen.

<!--en:754f274c-->
- Wir messen die mittlere quadrierte Gradientennorm $N_b = \mathbb{E}[\|\hat{\mathbf{g}}_b\|^2]$ bei zwei Minibatchgrößen $b_{\text{s}} < b_{\text{l}}$ und lösen das $2 \times 2$-Gleichungssystem $N_b = \|\nabla f\|^2 + \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}$:
  $$\operatorname{tr}\boldsymbol{\Sigma} \approx \frac{N_{b_{\text{s}}} - N_{b_{\text{l}}}}{1/b_{\text{s}} - 1/b_{\text{l}}}$$

<!--en:add7615f-->
- Für die quadrierte Norm des Signals folgt $\|\nabla f\|^2 \approx \frac{b_{\text{l}} N_{b_{\text{l}}} - b_{\text{s}} N_{b_{\text{s}}}}{b_{\text{l}} - b_{\text{s}}}$. Erhalten wir etwa $N_{16} = 0.50$ und $N_{2048} = 0.103$, so gilt $\operatorname{tr}\boldsymbol{\Sigma} \approx 6.4$, $\|\nabla f\|^2 \approx 0.10$ und $b_{\text{noise}} \approx 64$.

<!--en:713ef73f-->
::: {.slide title="Wie b_noise beim Training von TinyLM wächst"}

<!--en:1494161b-->
[Entwicklung der Rauschskala]{.kicker}

<!--en:cb84686c-->
- Wir messen `noise_scale` an `TinyLM` ($b_{\text{s}} = 8, b_{\text{l}} = 256$ Sequenzen) vor und nach $500$ Schritten mit Adam ($\eta = 3 \times 10^{-3}$):
  - **Bei der Initialisierung (Schritt 0):** $b_{\text{noise}} \approx 2$ Sequenzen. Für jedes Fenster muss das untrainierte Modell die globalen Byte-Häufigkeiten auf dieselbe Weise anpassen. Daher ist das gemeinsame Signal $\|\nabla f\|^2$ sehr stark.
  - **Nach 500 Schritten:** $b_{\text{noise}}$ wächst **um den Faktor $>30\times$ auf $\sim 50\text{--}150$ Sequenzen**. Dabei sinkt $\|\nabla f\|^2$, während die Unterschiede zwischen den Sequenzen $\operatorname{tr}\boldsymbol{\Sigma}$ bestehen bleiben.

<!--en:aa9de17e-->
::: {.slide title="Optimale Schrittweite und Fortschritt pro Schritt bei Rauschen"}

<!--en:8d7a0212-->
[Abstiegsschranke bei Rauschen]{.kicker}

<!--en:0088a06a-->
- **Abstiegsschranke bei Rauschen und $L$-Glattheit:**
  $$\mathbb{E}[f(\mathbf{x} - \eta\hat{\mathbf{g}}_b)] - f(\mathbf{x}) \le -\eta\|\nabla f\|^2 + \frac{L}{2}\eta^2\|\nabla f\|^2\Big(1 + \frac{b_{\text{noise}}}{b}\Big)$$

<!--en:622e330d-->
- **Optimale Schrittweite und Abnahme pro Schritt:**
  $$\eta_{\text{opt}}(b) = \frac{1}{L\,(1 + b_{\text{noise}}/b)}, \quad \Delta f_{\text{opt}}(b) = \frac{\|\nabla f\|^2}{2L\,(1 + b_{\text{noise}}/b)}$$

<!--en:f02ba678-->
- Bei $b = b_{\text{noise}}$ erreichen $\eta_{\text{opt}}$ und $\Delta f_{\text{opt}}$ jeweils **genau die Hälfte** ihres Werts für den gesamten Datensatz ($b \to \infty$).

<!--en:f3a41488-->
::: {.slide title="Pareto-Front für Schrittzahl und Datenbedarf und die kritische Batchgröße"}

<!--en:7bcf4b44-->
[Universeller Zusammenhang zwischen Schritten und Datenbedarf]{.kicker}

<!--en:b548adb0-->
- **Schrittzahl $S(b)$ und Zahl der Beispiele $E(b) = b\,S(b)$ ($E_{\min} = b_{\text{crit}}S_{\min}$):**
  $$\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\text{crit}}}{b}, \qquad \frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\text{crit}}}$$

<!--en:b8c8314d-->
- **Kleiner Batch ($b = 0.1\,b_{\text{crit}}$):** $11\,S_{\min}$ Schritte bei einem Datenbedarf von nur $1.10\,E_{\min}$.

<!--en:4d16c020-->
- **Kritische Batchgröße am Knie ($b = b_{\text{crit}}$):** Ausgewogenes Verhältnis mit **$2\,S_{\min}$ Schritten und $2\,E_{\min}$ Beispielen**.

<!--en:13532b42-->
- **Zu großer Batch ($b = 10\,b_{\text{crit}}$):** Spart $<50\%$ der Schritte gegenüber $b_{\text{crit}}$, benötigt aber $5.5\times$ so viele Daten ($11\,E_{\min}$).

<!--en:cfd6e915-->
::: {.slide title="Warum SGD unterhalb von b_crit die Lernrate linear skaliert (η ∝ b)"}

<!--en:7a9e2359-->
[Lineare Skalierung für Optimierer ohne Normierung]{.kicker}

<!--en:59405d5a-->
- **Grenzfall kleiner Batches ($b \ll b_{\text{noise}}$):**
  $$\eta_{\text{opt}}(b) = \frac{1}{L\,(1 + b_{\text{noise}}/b)} \approx \frac{b}{L\,b_{\text{noise}}} \implies \eta(b) = \frac{b}{b_0}\eta_0$$

<!--en:31a689fa-->
- **Gleiche Trajektorie und Rauschgrenze (Goyal et al., 2017):** $k = b/b_0$ kleine Schritte bei $(\eta_0, b_0)$ entsprechen einem Schritt bei $(k\eta_0, kb_0)$. Dabei bleiben $\eta/b$ und der Überschussverlust $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}}{4b}$ konstant.

<!--en:f36d7d5d-->
- **Sättigung oberhalb von $b_{\text{crit}}$:** Für $b \gg b_{\text{crit}}$ gilt $\eta_{\text{opt}} \to 1/L$. Eine lineare Skalierung über $b_{\text{crit}}$ hinaus verletzt $\eta < 2/L$.

<!--en:75643f1f-->
::: {.slide title="Warum Adam unterhalb von b_crit die Wurzelregel verwendet (η ∝ √b)"}

<!--en:03b6e839-->
[Wurzelregel für adaptive Optimierer]{.kicker}

<!--en:d2a0edf1-->
Warum brauchen **Adam und AdamW** im rauschdominierten Regime eine **Wurzelskalierung** $\eta(b) = \sqrt{b/b_0}\,\eta_0$ anstelle einer linearen Skalierung (Malladi et al., 2022)?

<!--en:253ba539-->
- Betrachten wir eine Koordinate, deren Minibatch-Gradient den wahren Mittelwert $\bar{g}$ und das Stichprobenrauschen $z\,\frac{\sigma}{\sqrt{b}}$ ($z \sim \mathcal{N}(0, 1)$) mit $\frac{\sigma}{\sqrt{b}} \gg |\bar{g}|$ hat.

<!--en:d16b6c14-->
- **Der Nenner von Adam liefert bei der Division bereits den Faktor $\sqrt{b}$:** Wegen $\sqrt{\hat{v}_t} \approx \sqrt{\bar{g}^2 + \sigma^2/b} \approx \frac{\sigma}{\sqrt{b}}$ verstärkt die Division durch $\sqrt{\hat{v}_t}$ die Drift des Signals um den Faktor $\sqrt{b}$:
  $$\Delta x_t \approx -\eta\,\frac{\bar{g} + z\,\sigma/\sqrt{b}}{\sigma/\sqrt{b}} = \underbrace{-\eta\,\frac{\sqrt{b}}{\sigma}\,\bar{g}}_{\text{Deterministic drift}} \;-\; \underbrace{\eta\,z}_{\text{Unit-variance step noise}}$$

<!--en:d180120e-->
- **$k = b/b_0$ kleine Schritte durch $1$ großen Schritt ersetzen:** Über $k$ kleine Schritte bei $(\eta_0, b_0)$ beträgt die kumulierte Drift $k \cdot \eta_0\frac{\sqrt{b_0}}{\sigma}\bar{g}$ und die kumulierte Rauschvarianz $k\,\eta_0^2$. Ein einzelner Schritt bei Batchgröße $b = k b_0$ erreicht **beide** kumulierten Größen, wenn gilt:
  $$\eta(b) = \sqrt{k}\,\eta_0 = \sqrt{\frac{b}{b_0}}\;\eta_0$$

<!--en:5ffde008-->
::: {.slide title="Wie die Schrittzahl bis zum Zielverlust bei TinyLM skaliert"}

<!--en:ea2b44d4-->
[Versuchsreihe mit dem Sprachmodell]{.kicker}

<!--en:5814ab3c-->
- Wir trainieren `TinyLM` mit Adam ($\eta = 0.003\sqrt{b/64}$) bis zu einem Validierungsverlust von $1.9$ Nat bei Batchgrößen von $b \in \{4, 16, 64, 256\}$ Sequenzen:
  - **$b = 4 \to 16$ ($4\times$):** Die Schrittzahl sinkt etwa um den Faktor **$3\times$** ($\approx 4{,}000 \to 1{,}200$). Sie folgt nahezu der gestrichelten Linie für perfekte Skalierung ($S \propto 1/b$).
  - **$b = 64 \to 256$ ($4\times$):** Die Schrittzahl sinkt nur etwa um den Faktor **$2\times$** ($\approx 430 \to 215$). Die Gesamtzahl der verarbeiteten Sequenzen verdoppelt sich ($\approx 27{,}000 \to 55{,}000$), wenn $b$ die kritische Größe $b_{\text{crit}} \sim 100$ überschreitet.

<!--en:0946aceb-->
::: {.slide title="Wie die Schrittzahl bis zum Zielverlust bei FashionCNN skaliert"}

<!--en:e864ea36-->
[Versuchsreihe mit dem Bildklassifikator]{.kicker}

<!--en:2a3e27f0-->
- Wir trainieren `FashionCNN` mit Adam bis zum Verlust $0.35$ bei Batchgrößen von $b \in \{8, 32, 128, 512, 2048\}$ Bildern:
  - **$b = 8 \to 32 \to 128$:** Die Schrittzahl sinkt nahezu im umgekehrten Verhältnis zur Batchgröße (von $2{,}300\text{--}3{,}000$ Schritten auf etwa $700$ und dann auf etwa $240$). Die Zahl der verarbeiteten Beispiele steigt nur langsam ($\sim 19{,}000\text{--}32{,}000$ Bilder).
  - **$b = 512 \to 2{,}048$ ($4\times$):** Die Schrittzahl sinkt nur um etwa ein Drittel ($\approx 120 \to 80$), während sich der Datenbedarf auf $163{,}840$ Bilder fast verdreifacht.

<!--en:252d2a8d-->
::: {.slide title="S_min und b_crit mit der Methode der kleinsten Quadrate in 1/b bestimmen"}

<!--en:cafa72dc-->
[Bestimmung der kritischen Batchgröße]{.kicker}

<!--en:353e1abf-->
- Da $S(b) = S_{\min}\!\left(1 + \frac{b_{\text{crit}}}{b}\right) = S_{\min} + (S_{\min}b_{\text{crit}})\frac{1}{b}$ linear von $1/b$ abhängt, liefert eine gewichtete lineare Regression beide asymptotischen Kenngrößen direkt aus den Schrittzahlen:
  - **Achsenabschnitt:** Mindestschrittzahl $S_{\min}$ für Schritte auf dem gesamten Datensatz.
  - **Steigung / Achsenabschnitt:** Empirische kritische Batchgröße $b_{\text{crit}}$.

<!--en:af54f005-->
- Die angepassten Werte ($b_{\text{crit}} \sim 80\text{--}110$ Sequenzen bei `TinyLM` und $b_{\text{crit}} \sim 200\text{--}350$ Bilder bei `FashionCNN`) haben dieselbe Größenordnung wie unsere Schätzung der Skala des Gradientenrauschens $b_{\text{noise}}$ aus zwei Batchgrößen.

<!--en:8f8f847c-->
::: {.slide title="Nach der Normierung liegen beide Modelle auf einer universellen Pareto-Kurve"}

<!--en:0bdf4442-->
[Grenzkurve für Schrittzahl und Zahl der Beispiele]{.kicker}

<!--en:6714c7de-->
- Eliminieren wir $b / b_{\text{crit}}$ aus $\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\text{crit}}}{b}$ und $\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\text{crit}}}$ (mit $E_{\min} = b_{\text{crit}}S_{\min}$), erhalten wir eine von der Architektur unabhängige Hyperbel:
  $$\left(\frac{E}{E_{\min}} - 1\right)\left(\frac{S}{S_{\min}} - 1\right) = 1$$

<!--en:242f807a-->
- Normieren wir die Messwerte von `TinyLM` und `FashionCNN` mit den jeweils angepassten Werten $(S_{\min}, E_{\min})$, liegen beide Modelle auf derselben Kurve:
  - **Ast links oben ($b \ll b_{\text{crit}}$):** Geringer Datenbedarf ($E \approx E_{\min}$), viele aufeinanderfolgende Schritte.
  - **Knie ($b = b_{\text{crit}}$):** $(2E_{\min}, 2S_{\min})$.
  - **Ast rechts unten ($b \gg b_{\text{crit}}$):** Die Schrittzahl nähert sich ihrer Untergrenze ($S \approx S_{\min}$), der Datenbedarf steigt stark an.

<!--en:2f9104b8-->
::: {.slide title="Überprüfung der linearen Skalierungsregel (η ∝ b) für SGD"}

<!--en:e4d53194-->
[SGD mit Momentum bei FashionCNN]{.kicker}

<!--en:dd7a785c-->
- Bei einem festen Datenbudget von $E = 12{,}800$ Bildern ($1/5$ Epoche) testen wir SGD mit Momentum ($\beta = 0.9$). Benachbarte Lernraten unterscheiden sich um den Faktor $2$. Wir vergleichen $b = 8$ ($1{,}600$ Schritte) und $b = 64$ ($200$ Schritte):
  - Erhöhen wir die Batchgröße um den Faktor **$8\times$ ($8 \to 64$)**, verschieben sich die U-förmige Verlustkurve und die optimale Lernrate um den Faktor **$4\text{--}8\times$** nach rechts ($0.0125 \to 0.05\text{--}0.10$).
  - Bleibt $\eta$ unverändert bei $0.0125$, ist beim Übergang zu $b = 64$ das Lernratenbudget für die $200$ Schritte viel zu klein.

<!--en:4739626c-->
::: {.slide title="Überprüfung der Wurzelregel (η ∝ √b) für Adam"}

<!--en:83198386-->
[Adam bei FashionCNN mit 64-facher Batchgröße]{.kicker}

<!--en:362b9bed-->
- Bei demselben Budget von $12{,}800$ Bildern erhöhen wir die Batchgröße für Adam um den Faktor **$64\times$ ($b = 2 \to 128$)**. Wir testen $13$ Lernraten, die sich von einer zur nächsten um den Faktor $2$ unterscheiden:
  - **Ohne Skalierung ($1\times$):** Die Lernrate $\eta$ ist um den Faktor $4\text{--}8\times$ zu klein für $b = 128$. Der Verlust liegt um rund $0.1$ Nat höher.
  - **Wurzelregel ($\sqrt{64} = 8\times = 2^3$, $+3$ Rasterschritte):** Die Lernrate trifft das empirische Optimum ($\eta = 0.0064$) oder liegt höchstens einen Rasterschritt daneben.
  - **Lineare Regel ($64\times = 2^6$, $+6$ Rasterschritte):** Die Lernrate liegt zu weit rechts im instabilen Bereich. Der Verlust ist der höchste der drei Varianten. Bei dieser oder der nächsthöheren Lernrate im Raster bleibt er bei $\log 10 \approx 2.3$. Das Netz lernt also nichts.

<!--en:c8d6d134-->
::: {.slide title="Wahl der Batchgröße und schrittweise Erhöhung im Training"}

<!--en:f3151170-->
[Vorgehen in der Praxis]{.kicker}

<!--en:edf89931-->
- **Warum $b$ während des Trainings erhöhen?** Von Schritt 0 bis zur Mitte des Trainings kann $b_{\text{noise}}$ um den Faktor $10\text{--}100\times$ wachsen. Beim Training führender Modelle steigt deshalb auch $b$ schrittweise (z. B. bei Llama 3 $4\text{M} \to 16\text{M}$ Token).

<!--en:0ab6a809-->
- **Wahl der Batchgröße in vier Schritten:**
  1. Wählen Sie $b \ge b_{\text{sat}}$ so, dass die Tensor-Rechenkerne der GPU ausgelastet sind.
  2. Messen Sie nach der Aufwärmphase $b_{\text{noise}}$ bei Schritt $300\text{--}500$.
  3. Wählen Sie $b \lesssim 0.3\,b_{\text{crit}}$ bei rechenbegrenztem Training oder $b \approx b_{\text{crit}}$ bei begrenzter Laufzeit.
  4. Skalieren Sie die Lernrate gemäß $\eta \propto b$ (SGD) oder $\eta \propto \sqrt{b}$ (AdamW) mit linearer Aufwärmphase.

<!--en:7427cffb-->
::: {.slide title="Verständnisfragen zum Schätzer mit zwei Batchgrößen und zu den Skalierungsregeln"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:c6382487-->
Leiten Sie für jedes der folgenden Szenarien zur Batchgröße die quantitative Vorhersage her:

<!--en:644e8189-->
1. **Schätzer mit zwei Batchgrößen:** Bei Schritt $500$ messen wir die mittleren quadrierten Gradientennormen für $b_{\text{s}} = 10$ und $b_{\text{l}} = 100$ und erhalten $N_{10} = 1.0$ und $N_{100} = 0.19$. Wie groß sind $\operatorname{tr}\boldsymbol{\Sigma}$, $\|\nabla f\|^2$ und $b_{\text{noise}}$?

<!--en:ce4492b4-->
2. **Zusätzliche GPUs nahe $b_{\text{crit}}$:** Wir trainieren `TinyLM` mit AdamW bei $b = 64$ und $\eta = 3 \times 10^{-3}$. Dabei gilt $b_{\text{crit}} = 100$. Sie verdoppeln die Batchgröße auf $b = 128$, verteilt auf zwei GPUs. Welche Lernrate ergibt die Wurzelregel? Um welchen Faktor sinkt die Schrittzahl $S(b)$, und um welchen Faktor steigt der Datenbedarf $E(b)$?

<!--en:3e4d835d-->
3. **Warum ist bei großem $b$ eine Aufwärmphase nötig?** In der Mitte des Trainings gelte $b_{\text{crit}} = 256$. Warum können wir die um den Faktor $\sqrt{256/16} = 4\times$ erhöhte Lernrate nicht schon bei Schritt $t = 1$ ohne Aufwärmphase einsetzen?

<!--en:557b98b6-->
**Antworten:**

<!--en:1fcd5d43-->
1. $\operatorname{tr}\boldsymbol{\Sigma} = \frac{1.0 - 0.19}{1/10 - 1/100} = \frac{0.81}{0.09} =$ **$9.0$**, $\|\nabla f\|^2 = 0.19 - \frac{9.0}{100} =$ **$0.10$** und **$b_{\text{noise}} = \frac{9.0}{0.10} = 90$**.

<!--en:1f81038b-->
2. Die Wurzelskalierung ergibt **$\eta = \sqrt{2} \times 3 \times 10^{-3} \approx 4.24 \times 10^{-3}$**. Die Schrittzahl sinkt um den Faktor $\frac{1 + 100/64}{1 + 100/128} = \frac{2.5625}{1.78125} \approx$ **$1.44\times$** (auf 2 GPUs beträgt die Laufzeit noch $69\%$ der ursprünglichen Laufzeit). Die verarbeitete Datenmenge steigt um den Faktor $\frac{1 + 128/100}{1 + 64/100} = \frac{2.28}{1.64} \approx$ **$1.39\times$ ($+39\%$ mehr Daten)**.

<!--en:b41602a6-->
3. Bei Schritt $t = 0$ gilt $b_{\text{noise}}(0) \approx 2 \ll 256$. Damit liegt $b = 256$ anfangs weit im krümmungsdominierten Regime ($b \gg b_{\text{noise}}(0)$). Eine um den Faktor $4\times$ erhöhte Lernrate überschreitet dort die anfängliche Stabilitätsgrenze.

<!--en:c660a564-->
::: {.slide title="Zusammenfassung zur Skala des Gradientenrauschens und zur kritischen Batchgröße"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:70ad3065-->
| Größe oder Gesetz | Formel | Regime und Skalierungsverhalten | Praktische Bedeutung |

<!--en:0802a335-->
| **Quadrierte Norm des Minibatch-Gradienten** | $\mathbb{E}\|\hat{\mathbf{g}}_b\|^2 = \|\nabla f\|^2 + \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}$ | Signalleistung $+\; 1/b$ der Rauschleistung pro Beispiel | Aus zwei Batchgrößen lassen sich $\operatorname{tr}\boldsymbol{\Sigma}$ und $\|\nabla f\|^2$ schätzen, wenn $b_{\text{s}} < b_{\text{l}}$ gilt |

<!--en:69e5acbb-->
| **Skala des Gradientenrauschens** | $b_{\text{noise}} = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{\|\nabla f\|^2}$ | Kann im Training um den Faktor $10\text{--}100\times$ wachsen, wenn $\|\nabla f\|^2$ sinkt | Sagt die kritische Batchgröße $b_{\text{crit}}$ ohne Hesse-Matrix voraus |

<!--en:0110c915-->
| **Pareto-Kurve für Schrittzahl und Datenbedarf** | $\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\text{crit}}}{b},\; \frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\text{crit}}}$ | Universelle Hyperbel $(E/E_{\min}-1)(S/S_{\min}-1) = 1$ | Bei $b = b_{\text{crit}}$ gelten $S = 2S_{\min}$ und $E = 2E_{\min}$. Oberhalb von $b_{\text{crit}}$ sinkt die Schrittzahl nur noch wenig |

<!--en:ed60f0a8-->
| **Skalierung der Lernrate bei SGD** | $\eta(b) = \frac{b}{b_0}\,\eta_0$ ($b \ll b_{\text{crit}}$) | Linearer Anstieg bis $b_{\text{crit}}$, dann Sättigung bei $1/L$ | $k$ kleine Schritte wirken wie $1$ großer Schritt bei konstanter Rauschgrenze $\eta/b$ |

<!--en:9dab49bd-->
| **Skalierung der Lernrate bei AdamW** | $\eta(b) = \sqrt{\frac{b}{b_0}}\,\eta_0$ ($b \ll b_{\text{crit}}$) | Wachstum mit der Quadratwurzel, da die Division durch $\sqrt{\hat{v}_t} \approx \sigma/\sqrt{b}$ den Faktor $\sqrt{b}$ liefert | Bei kleinem Anfangswert $b_{\text{noise}}(0)$ mit linearer Aufwärmphase kombinieren oder die Batchgröße schrittweise erhöhen |
