<!--en:a8637903-->
# Adam und AdamW

<!--en:ef5fe0e6-->
In :numref:`sec_test_models` benötigte der einfache stochastische Gradientenabstieg (engl. stochastic gradient descent, SGD) mit Minibatches eine Lernrate von $\eta = 0.5$, um auf `TinyLM` einen Verlust von etwa $2.1$ Nat pro Byte zu erreichen. Mit Adam und $\eta = 3 \times 10^{-3}$ ist die Lernrate um mehr als den Faktor hundert kleiner. Dennoch erreichen wir $1.59$ Nat pro Byte in denselben $2{,}000$ Schritten, wie wir weiter unten messen. Warum trainiert ein Optimierer mit einer nominell hundertmal kleineren Lernrate schneller, und warum ist er bei Transformern die Standardwahl?

<!--en:894e3ede-->
Momentum (:numref:`sec_momentum`) mittelt die Gradienten über die Zeit, verwendet aber für alle Parameter dieselbe skalare Schrittweite. In tiefen Netzen, insbesondere in Transformern, unterscheiden sich die Gradientenbeträge von Einbettungen, Attention-Projektionen, MLP-Schichten und Normalisierungsparametern jedoch um Größenordnungen. **Adam** (*Adaptive Moment Estimation*, :citet:`Kingma.Ba.2014`) kombiniert Momentum mit einem Vorkonditionierer, der für jede Koordinate das quadratische Mittel (engl. root mean square, RMS) der Gradienten nutzt. Seine Entwicklung führte über **AdaGrad** :cite:`Duchi.Hazan.Singer.2011` und **RMSProp** :cite:`Tieleman.Hinton.2012`. Teilen wir das Update jeder Koordinate durch die für sie bisher beobachtete Gradientenskala, gibt die Lernrate $\eta$ unmittelbar eine Schrittlänge im Parameterraum an. Sie dient also nicht mehr als Multiplikator für die Rohgradienten.

<!--en:8315c05e-->
Genau diese koordinatenweise Division verändert jedoch die Wirkung der üblichen $\ell_2$-Regularisierung. Addieren wir $\frac{\lambda}{2}\|\mathbf{x}\|^2$ zum Verlust, so skaliert der Kehrwert der jeweiligen Gradientenskala die Zerfallsrate jedes Gewichts. :citet:`Loshchilov.Hutter.2019` lösten dieses Problem, indem sie den Gewichtszerfall (engl. weight decay) vom adaptiven Vorkonditionierer entkoppelten. So entstand **AdamW**. Wir verfolgen zunächst die Entwicklung von AdaGrad und RMSProp bis zu Adam. Dann messen wir die Gradientenskalen der einzelnen Koordinaten und vergleichen Adam mit SGD mit Momentum auf `TinyLM`. Schließlich leiten wir AdamW, die Zeitskala $\tau = 1/(\eta\lambda)$ des Gewichtszerfalls und die Gleichgewichtsnorm (engl. equilibrium norm) her. Dabei untersuchen wir, welche Parametergruppen wir vom Gewichtszerfall ausnehmen müssen.

<!--en:8056206f-->
## Adaptive Lernraten für einzelne Koordinaten

<!--en:a42d48be-->
### AdaGrad und RMSProp

<!--en:1ba631ca-->
Bei Modellen mit dünn besetzten oder heterogenen Merkmalen eignet sich eine einzige globale Lernrate nicht für alle Koordinaten. Ein Beispiel sind Tokeneinbettungen: Häufige Token kommen in jedem Minibatch vor, seltene nur gelegentlich. **AdaGrad** :cite:`Duchi.Hazan.Singer.2011` bildet elementweise die laufende Summe der quadrierten Gradienten $\mathbf{s}_t$ und teilt den Schritt jeder Koordinate durch die entsprechende Komponente von $\sqrt{\mathbf{s}_t + \epsilon}$:

<!--en:f0633bad-->
Dabei gilt $\mathbf{s}_0 = \mathbf{0}$, und $\epsilon > 0$ verhindert eine Division durch null. In einem achsenparallelen Tal gleicht die Skalierung jeder Koordinate anhand ihres Gradientenbetrags die unterschiedlichen Krümmungen in den Achsenrichtungen automatisch aus. So nähern wir den diagonalen Vorkonditionierer aus :numref:`subsec_gd-preconditioning` allein mithilfe der Gradienten an (:numref:`subsec_mdl-per-coordinate`).

<!--en:0a169e47-->
Da $\mathbf{s}_t$ die quadrierten Gradienten aufsummiert, wächst die Summe bei anhaltenden stochastischen Gradienten monoton und unbeschränkt. Die effektive Schrittweite von AdaGrad fällt daher wie $O(t^{-1/2})$. Bei tiefen Netzen mit nichtkonvexer Zielfunktion kommt das Training schließlich zum Stillstand. **RMSProp** :cite:`Tieleman.Hinton.2012` ersetzt die kumulative Summe durch einen exponentiellen gleitenden Mittelwert (engl. exponential moving average, EMA) $\mathbf{v}_t$ der quadrierten Gradienten über ein Fenster von $\sim 1/(1-\beta_2)$ Schritten:

<!--en:7ba73e1a-->
*(Hinweis: Wie in der Optimierung üblich, bezeichnet $\mathbf{v}_t$ hier den Mittelwert der quadrierten Gradienten, also das zweite Moment. In :numref:`sec_momentum` stand das Symbol für den Geschwindigkeitsvektor.)* Da RMSProp weit zurückliegende Gradienten vergisst, bleibt die Schrittweite selbst bei einer moderaten Lernrate von $\eta = 0.4$ von null verschieden:

<!--en:7913f864-->
Beachten wir zwei Eigenschaften von :eqref:`eq_rmsprop`:

<!--en:246c077e-->
1. **Verstärkung des ersten Schritts ohne Bias-Korrektur:** Aus $\mathbf{v}_0 = \mathbf{0}$ folgt im ersten Schritt $\mathbf{v}_1 = (1 - \beta_2)\mathbf{g}_1^2$. Dadurch entsteht ein übergroßer Anfangssprung mit dem Betrag $\eta/\sqrt{1-\beta_2} \approx 3.16\,\eta$ bei $\beta_2 = 0.9$ (und $31.6\,\eta$ bei $\beta_2 = 0.999$).

<!--en:a4103cec-->
2. **Normierte Schrittlänge:** Bleibt der Gradientenbetrag einer Koordinate annähernd konstant, gilt $\sqrt{v_t} \approx |g_t|$. Das Update hat dann den Betrag $\approx \eta$, unabhängig davon, wie klein $|g_t|$ wird. Nahe einem Minimum oszilliert ein adaptives Verfahren mit konstanter Lernrate daher in einer $O(\eta)$-Umgebung, bis ein Lernratenplan $\eta$ senkt (:numref:`sec_scheduler`).

<!--en:7d762b3b-->
### Adam und Bias-Korrektur

<!--en:d91eeacc-->
**Adam** :cite:`Kingma.Ba.2014` kombiniert einen EMA für das erste Moment (Momentum, $\mathbf{m}_t$) mit einem EMA für das unzentrierte zweite Moment ($\mathbf{v}_t$). Für beide Schätzer verwendet Adam eine exakte **Bias-Korrektur**. Schreiben wir die Rekursion als $\mathbf{v}_t = (1-\beta_2)\sum_{s=1}^t \beta_2^{t-s}\mathbf{g}_s^2$ mit dem Startwert $\mathbf{v}_0 = \mathbf{0}$ aus, sehen wir, dass sich die Gewichte zu $1 - \beta_2^t$ statt zu $1$ summieren. Durch Division durch $1 - \beta_1^t$ beziehungsweise $1 - \beta_2^t$ erhalten wir die erwartungstreuen Momentenschätzer $\hat{\mathbf{m}}_t$ und $\hat{\mathbf{v}}_t$:

<!--en:7a1252af-->
Für das Parameterupdate teilen wir das erste Moment nach der Bias-Korrektur koordinatenweise durch die RMS-Skala $\sqrt{\hat{\mathbf{v}}_t}$:

<!--en:5c2b5f66-->
Übliche Standardwerte sind $\beta_1 = 0.9$, $\beta_2 = 0.999$ (oder $\beta_2 = 0.95$ beim Training großer Sprachmodelle :cite:`brown2020language`) und $\epsilon = 10^{-8}$. Für $t = 1$ gilt $\hat{\mathbf{m}}_1 = \mathbf{g}_1$ und $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$. Bei vernachlässigbarem Epsilon ergibt sich als erster Schritt $-\eta\,\operatorname{sign}(\mathbf{g}_1)$.

<!--en:e2a17231-->
Allgemeiner gilt: Skalieren wir eine beliebige Koordinate der Zielfunktion mit einer positiven Konstante $c > 0$, so multiplizieren wir sowohl $\hat{m}_{t,i}$ als auch $\sqrt{\hat{v}_{t,i}}$ mit $c$. Ihr Verhältnis bleibt damit invariant (für $\epsilon \to 0$). **Bei Adam gibt $\eta$ also unmittelbar die Schrittlänge einer Koordinate im Parameterraum an**. Bleibt das Vorzeichen des Gradienten gleich, bewegt sich die Koordinate pro Schritt um $\approx \eta$. Überwiegt dagegen Rauschen mit Mittelwert null, beträgt die Schrittlänge $\approx \eta\sqrt{(1-\beta_1)/(1+\beta_1)} \approx 0.23\,\eta$.

<!--en:bf6d195e-->
Wenn wir Adam selbst implementieren, benötigen wir für jeden Parametertensor einen Schrittzähler $t$ und zwei Puffer $(\mathbf{m}, \mathbf{v})$:

<!--en:bf97666a-->
## Adam für ein Sprachmodell

<!--en:0b5e2a05-->
### Gradientenskalen und Vergleich mit SGD auf `TinyLM`

<!--en:55d22636-->
Warum ist die koordinatenweise Skalierung bei Transformern so wichtig? Um die Gradientenskalen in `TinyLM` (:numref:`subsec_tinylm`) zu messen, laden wir zunächst den Datensatz `WikiText103` ($b = 64$, Fensterlänge $64$):

<!--en:cf723051-->
Als Nächstes definieren wir `gradients(model, data)`. Die Funktion berechnet für jeden Parametertensor das quadratische Mittel (RMS) der Gradienten über $10$ Minibatches. Mit `snapshot(model)` erstellen wir Kopien der Parameterwerte. Außerdem legen wir eine erste Instanz von `TinyLM` mit Adam an ($\eta = 3 \times 10^{-3}$):

<!--en:a9ff1c09-->
Wir ordnen die Tensoren den $8$ funktionalen Parametergruppen zu. Dazu verwenden wir `rms_by_kind`. Für jede Gruppe vergleichen wir die RMS-Werte der Gradienten bei Schritt $0$ und Schritt $500$ sowie die normierte Schrittlänge $\|\Delta\mathbf{x}\|_{\text{RMS}} / \eta$ von Adam:

<!--en:4bbed372-->
Bei der Initialisierung unterscheiden sich die Beträge der Rohgradienten um einen **Faktor von $40\times$** (von $\sim 2 \times 10^{-4}$ bei Tokeneinbettungen bis $\sim 7 \times 10^{-3}$ bei Bias-Termen). Der **Faktor beträgt noch $15\times$** bei Schritt $500$. Bei SGD mit einer einzigen skalaren Lernrate ändern sich die Gruppen mit den größten Gradienten (Bias-Terme, LayerNorm, Ausgabekopf) $15\text{--}40\times$ so schnell wie die Einbettungen. Bei Adam liegt die normierte Schrittweite aller acht Parametergruppen in einem engen Bereich mit dem Faktor $2\times$ ($0.1\eta\text{--}0.3\eta$, nahe am Verhältnis $\sqrt{(1-\beta_1)/(1+\beta_1)} \approx 0.23$ für reines Rauschen).

<!--en:9fc21fb7-->
Die Gradientenskalen in `TinyLM` liegen ungefähr bei $10^{-4}\text{--}10^{-2}$, und der EMA von Adam enthält den Faktor $(1-\beta_1) = 0.1$. Deshalb sind die Parameterschritte von Adam bei $\eta \sim 3 \times 10^{-3}$ ähnlich groß wie die von SGD mit Momentum $0.9$ bei $\eta \sim 0.3$. Wir vergleichen beide Optimierer mit denselben $3$ Zufallsstartwerten. Dazu definieren wir zunächst die Funktionen `make_lm`, `make_sgd` und `make_adam`, die Modell und Optimierer mit den jeweils gebundenen Einstellungen erzeugen:

<!--en:b105798f-->
Dann testen wir SGD mit Momentum für $\eta \in \{0.1, 0.3, 0.5, 1.0\}$ und Adam für $\eta \in \{3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}, 10^{-2}, 3 \times 10^{-2}\}$. Für jeden Zufallsstartwert beginnen beide Verfahren mit denselben Gewichten:

<!--en:60a250e9-->
Für jeden Optimierer bestimmen wir die beste Lernrate (`best_rate`). Wir vergleichen den nutzbaren Lernratenbereich (höchstens $0.15$ Nat über dem Optimum), die Schrittzahl bis zum Erreichen von $1.8$ Nat, den größten Verlustanstieg in einem einzelnen Schritt und die geglätteten Verlustverläufe:

<!--en:7102bfae-->
Auf `TinyLM` erreicht Adam den Zielverlust von $1.8$ Nat in etwa halb so vielen Schritten wie SGD mit Momentum ($650\text{--}800$ gegenüber $1{,}100\text{--}1{,}500$) und erzielt einen niedrigeren Endverlust. Der nutzbare Lernratenbereich umfasst einen Faktor von $10\times$ ($10^{-3}$ bis $10^{-2}$). Selbst bei $\eta = 3 \times 10^{-2}$ bleibt der Verlust endlich, da $\eta$ die normierte Schrittlänge im Versuch begrenzt. Die beste Lernrate von SGD, $\eta = 0.3$, liegt nur einen Gitterpunkt unter der Divergenzschwelle. Die stärksten Verlustanstiege in einem Schritt sind bei SGD um ein Mehrfaches größer als bei Adam. Bei $\eta = 0.5$ divergieren die meisten Läufe gegen `inf`. Bei größeren Transformern mit Subwortvokabular wächst der Abstand weiter. Gründe dafür sind Unterschiede zwischen den Blöcken der Hesse-Matrix :cite:`Zhang.Chen.Ding.ea.2024` und der schwere Rand der Häufigkeitsverteilung der Token :cite:`Kunstner.Chen.Lavington.ea.2023,Kunstner.Yadav.Milligan.ea.2024`.

<!--en:333a8557-->
Bei einem der Zufallsstartwerte wirft ein einzelner Verlustanstieg SGD in diesem Versuch so weit zurück, dass der Verlust den Wert $1.8$ nie erreicht.

<!--en:f11b62d5-->
### Die Rolle von $\epsilon$ und die Grenzen der Schätzung des zweiten Moments

<!--en:6dbe2a5b-->
In :eqref:`eq_adam-update` begrenzt $\epsilon$ die effektive Lernrate jeder Koordinate auf höchstens $\eta / \epsilon$. Im Grenzfall $\epsilon \gg \sqrt{\hat{v}_{t,i}}$ geht Adam in SGD mit Momentum und der Schrittweite $(1-\beta_1)\eta/\epsilon$ über. Ist umgekehrt $\beta_2$ zu klein, berücksichtigt die Schätzung des zweiten Moments nur eine kurze Vorgeschichte. Häufige kleine Gradienten können dann stärker ins Gewicht fallen als gelegentliche große Gradienten in Gegenrichtung. :citet:`Reddi.Kale.Kumar.2019` konstruierten mit diesem Effekt ein künstliches periodisches konvexes Gegenbeispiel ($+4, -1, -1$), bei dem Adam für kleine Werte von $\beta_2$ nicht konvergiert (:numref:`subsec_mdl-adam-counterexample`, Übung :exref:`ex_adam-two-repairs-variance-estimate`). In der Praxis vermeidet die Wahl $\beta_2 \in [0.95, 0.999]$ dieses Problem, ohne dass wir wie bei AMSGrad das bisherige Maximum mitführen müssen.

<!--en:faf73fbb-->
## Entkoppelter Gewichtszerfall (AdamW)

<!--en:7b574ad8-->
### Gekoppelte $\ell_2$-Regularisierung und entkoppelter Gewichtszerfall

<!--en:6529a6bf-->
Beim einfachen SGD (:numref:`sec_weight_decay`) addieren wir mit dem $\ell_2$-Strafterm $\frac{\lambda}{2}\|\mathbf{x}\|^2$ in der Zielfunktion den Term $\lambda\mathbf{x}_t$ zum Gradienten. Mathematisch ist das gleichbedeutend damit, die Gewichte vor jedem Schritt mit dem Faktor $(1 - \eta\lambda)$ zu multiplizieren:

<!--en:52fe8f24-->
Übergeben wir Adam jedoch den um den Strafterm ergänzten Gradienten $\mathbf{g}_t + \lambda\mathbf{x}_t$, teilt Adam auch den Strafterm durch den Vorkonditionierer der jeweiligen Koordinate $\sqrt{\hat{v}_{t,i}} + \epsilon$:

<!--en:35729dec-->
Bei gekoppelter $\ell_2$-Regularisierung schrumpfen Gewichte mit kleinen Verlustgradienten bis zu $10^3\times$ so stark wie Gewichte mit großen Gradienten. Dominiert $\lambda|x_{t,i}|$ den Verlustgradienten, normiert Adam den Strafterm durch seinen eigenen Betrag. Das Gewicht bewegt sich dann mit der Geschwindigkeit $\eta$ pro Schritt auf null zu, bis $\lambda|x| \sim |g_{\text{loss}}|$ gilt.

<!--en:0d7dd9bb-->
:citet:`Loshchilov.Hutter.2019` führten **AdamW** ein. Das Verfahren entkoppelt den Gewichtszerfall vom adaptiven Gradientenschritt:

<!--en:3744beaf-->
Dabei berechnen wir $\hat{\mathbf{m}}_t$ und $\hat{\mathbf{v}}_t$ ausschließlich aus dem Gradienten $\mathbf{g}_t$ des Datenverlusts.

<!--en:1d103323-->
Wir vergleichen drei Varianten: ohne Gewichtszerfall, mit gekoppeltem Gewichtszerfall (`Adam + l2`) und mit entkoppeltem Gewichtszerfall (`AdamW`). Für diesen Vergleich auf `TinyLM` definieren wir zunächst `matrix_rms`. Die Funktion berechnet das quadratische Mittel über die Einträge aller 2D-Gewichtsmatrizen mit Ausnahme der Einbettungsmatrizen. Mit `train_tracked` zeichnen wir den Verlust jedes Schritts und alle $250$ Schritte den RMS-Wert der Matrizen auf:

<!--en:7b802673-->
Wir trainieren alle drei Varianten für $1{,}000$ Schritte mit $\eta = 3 \times 10^{-3}$ und $\lambda = 0.1$ ausgehend vom selben Zufallsstartwert. Dabei zeigt sich, wie unterschiedlich der gekoppelte $\ell_2$-Strafterm und der entkoppelte Gewichtszerfall wirken:

<!--en:8f23c06c-->
Bei der gekoppelten Variante `Adam + l2` mit $\lambda = 0.1$ fällt der RMS-Wert der Gewichtsmatrizen in den versteckten Schichten vom Anfangswert $0.045$ auf $< 0.003$. Dadurch gehen die internen Repräsentationen des Netzes verloren. Der Verlust stagniert bei der Entropie der Unigrammverteilung der Bytes ($\sim 3.2$ Nat pro Byte). Mit der entkoppelten Variante `AdamW` verläuft das Training ähnlich gut wie beim unregularisierten Referenzmodell, während das Wachstum der Gewichte begrenzt bleibt.

<!--en:c41f0208-->
### Zeitskala des Gewichtszerfalls und Gleichgewichtsnorm

<!--en:3d808424-->
Schreiben wir $\mathbf{u}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ für die normierte Updaterichtung von Adam und setzen wir :eqref:`eq_adamw` über $t$ Schritte rekursiv ein, erhalten wir

<!--en:b2eb197f-->
AdamW stellt die Parameter $\mathbf{x}_t$ somit als exponentiell gewichteten gleitenden Mittelwert früherer Updates dar. Der Abklingfaktor beträgt $1 - \eta\lambda \approx e^{-\eta\lambda}$. Daraus ergibt sich für den Gewichtszerfall die charakteristische **Zeitskala** von

<!--en:b9b66ece-->
Schritten :cite:`Wang.Aitchison.2025`. Für $\eta = 3 \times 10^{-3}$ und $\lambda = 0.1$ beträgt die Zeitskala des Gewichtszerfalls $\tau = 1 / 0.0003 \approx 3{,}333$ Schritte. In einem Lauf mit $T$ Schritten wirkt der Gewichtszerfall also über $T / \tau = T\eta\lambda$ solcher Zeitspannen.

<!--en:6c1dcbd9-->
Dauert ein Lauf länger als $\tau = 1/(\eta\lambda)$, klingt der Einfluss der Anfangsgewichte ab. Jede Koordinate $x_{t,i}$ erreicht eine stochastische **Gleichgewichtsnorm**. Dabei gleicht die Schrumpfung $\eta\lambda x_{t,i}$ pro Schritt die Wirkung der normierten Updates $-\eta n_{t,i}$ aus (das mittlere Quadrat von $n_{t,i} = g_{t,i}/\sqrt{\hat{v}_{t,i}}$ beträgt $\sigma^2 \approx 1$). Es gilt dieselbe Varianzrekursion wie für die Kugel um das Minimum (engl. noise ball) bei SGD (:numref:`sec_sgd`), wobei $\lambda_{\text{decay}}$ die Krümmung ersetzt:

<!--en:fc419ecd-->
Stabilisiert sich das quadratische Mittel der Gewichte bei $\text{RMS}(\mathbf{x}_\infty) \propto \sqrt{\eta / (2\lambda)}$, so stabilisiert sich auch die **relative Winkeländerung pro Schritt** der normierten Gewichtsmatrizen bei $\eta / \text{RMS}(\mathbf{x}_\infty) \propto \sqrt{2\eta\lambda}$ (*Rotationsgleichgewicht* (engl. rotational equilibrium), :citet:`Kosson.Messmer.Jaggi.2024`). Ohne Regularisierung würden die Gewichte durch eine Irrfahrt unkontrolliert wie $\sqrt{t}$ wachsen. Bei langem Pre-Training würde die effektive Winkelschrittweite dadurch wie $1/\sqrt{t}$ abnehmen :cite:`DAngelo.Andriushchenko.Varre.ea.2024`. Variieren wir $\eta$, können wir die Schrumpfung pro Schritt $s = \eta\lambda$ und damit die Zeitskala $\tau = 1/s$ konstant halten. So entkoppeln wir die Lernratensuche von der Zeitskala des Gewichtszerfalls :cite:`Loshchilov.Hutter.2019,Wang.Aitchison.2025,Bergsma.Dey.Gosal.ea.2025b`.

<!--en:04ee3719-->
### Ausnahmen vom Gewichtszerfall und Speicherbedarf des Optimiererzustands

<!--en:6842756a-->
Beim Training von Transformern ist es üblich, den entkoppelten Gewichtszerfall auf die Gewichtsmatrizen anzuwenden und Bias-Terme sowie Normalisierungsparameter davon auszunehmen. Wie bei OLMo 2 nehmen wir hier zusätzlich die Einbettungstabellen aus. Der Gewichtszerfall wirkt dann **nur auf interne 2D-Projektionsmatrizen** ($90\%$ der Parameter von `TinyLM`). Zwei Parametergruppen bleiben ausgenommen ($\lambda = 0$):

<!--en:834db49d-->
1. **1D-Bias-Terme sowie Verstärkungs- und Verschiebungsvektoren von LayerNorm und RMSNorm:** Die Skalierungsfaktoren der Normalisierung beginnen nahe $1$ und steuern unmittelbar die Varianz der Merkmale. Schrumpfen sie gegen null, verzerrt dies die Aktivierungen der Schicht, anstatt die Winkeländerung pro Schritt zu regulieren.

<!--en:9fba5b40-->
2. **Einbettungstabellen für Token und Positionen (die Ausnahme bei OLMo 2):** Eine Zeile der Tokeneinbettung erhält nur dann einen Gradienten, wenn ihr Token im Minibatch vorkommt. Der entkoppelte Gewichtszerfall lässt dagegen bei jedem Schritt jede Zeile schrumpfen. Dadurch brechen die Repräsentationen seltener Token zusammen, und die Gradienten nachgelagerter LayerNorm-Schichten werden größer :cite:`OLMo.2025`.

<!--en:ef05da27-->
### Speicherbedarf des Optimiererzustands

<!--en:22924577-->
Die beiden Puffer $(\mathbf{m}_t, \mathbf{v}_t)$ pro Parameter verdoppeln den statischen Speicherbedarf für Gewichte und Gradienten. Bei 32-Bit-Gleitkommazahlen ($4$ Byte pro Skalar) benötigt jeder trainierbare Parameter **$16$ Byte** Gerätespeicher. Davon entfallen $4$ Byte auf den Parameter $\mathbf{x}$, $4$ Byte auf den Gradienten $\mathbf{g}$ und $8$ Byte auf den AdamW-Zustand $(\mathbf{m}, \mathbf{v})$ (:numref:`sec_memory_precision`). Bei einem Modell mit $10^9$ Parametern belegt allein der Optimiererzustand $8\text{ GB}$, noch bevor wir Aktivierungen speichern. Frieren wir Parameter ein, indem wir sie aus dem Optimierer ausnehmen, entfallen diese Puffer für die betreffenden Parameter. Durch Quantisierung oder Faktorisierung von $(\mathbf{m}, \mathbf{v})$ lassen sich die zusätzlichen $8$ Byte pro Parameter reduzieren.

<!--en:d7363093-->
AdamW normiert das Update jedes einzelnen Gewichtseintrags unabhängig von den anderen. Eine 2D-Gewichtsmatrix wirkt jedoch als linearer Operator auf Aktivierungsvektoren. In :numref:`sec_muon` untersuchen wir, ob wir noch bessere Ergebnisse erzielen, wenn wir das Update jeder 2D-Gewichtsmatrix als ganzen Operator normieren.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:45a7a112-->
**AdaGrad** teilt den Schritt jeder Koordinate durch die Wurzel aus der Summe ihrer bisherigen Gradientenquadrate. **RMSProp** ersetzt diese unbeschränkte Summe durch einen exponentiellen gleitenden Mittelwert $\mathbf{v}_t$, in dem ältere Gradienten an Gewicht verlieren. **Adam** ergänzt das Momentum des ersten Moments $\mathbf{m}_t$ sowie die Bias-Korrekturen $1 - \beta_1^t$ und $1 - \beta_2^t$. Die Division von $\hat{\mathbf{m}}_t$ durch $\sqrt{\hat{\mathbf{v}}_t}$ macht Adam invariant gegenüber diagonaler Skalierung des Gradienten. Damit gibt $\eta$ unmittelbar die Schrittlänge einer Koordinate im Parameterraum an. So gleicht Adam die Updatebeträge der Parametergruppen eines Transformers an, deren Rohgradienten sich um den Faktor $15\text{--}40\times$ unterscheiden. **AdamW** entkoppelt den Gewichtszerfall in $\mathbf{x}_{t+1} = (1 - \eta\lambda)\mathbf{x}_t - \eta\hat{\mathbf{m}}_t/(\sqrt{\hat{\mathbf{v}}_t}+\epsilon)$ vom adaptiven Vorkonditionierer. Daraus ergeben sich eine einheitliche Zeitskala des Gewichtszerfalls $\tau = 1/(\eta\lambda)$, ein RMS-Wert der Gewichte im Gleichgewicht proportional zu $\sqrt{\eta/(2\lambda)}$ und eine konstante relative Winkelschrittweite proportional zu $\sqrt{2\eta\lambda}$. Üblicherweise unterliegen nur 2D-Gewichtsmatrizen dem Gewichtszerfall. Ausgenommen sind 1D-Bias-Terme und Normalisierungsparameter sowie nach dem Vorbild von OLMo 2 die Einbettungstabellen. Die Puffer $(\mathbf{m}_t, \mathbf{v}_t)$ belegen $8$ der $16$ Byte pro trainierbarem Parameter in `float32`.

<!--en:2294243a-->
## Übungen

<!--en:8a807910-->
1. [code] **Koordinatendrehung und diagonale Skaleninvarianz.**
    1. Überprüfen Sie mit `adam_2d` auf `d2l.f_valley`, dass Adam bei Skalierung der Zielfunktion mit $c > 0$ (für $c \gg \epsilon$) und bei Änderung von $\lambda_2$ von $4$ auf $400$ invariant bleibt.
    1. Drehen Sie das Tal um $45^\circ$, indem Sie $f(\mathbf{x}) = 0.05(x_1 + x_2)^2 + (x_1 - x_2)^2$ verwenden. Führen Sie `adagrad_2d`, `rmsprop_2d` und `adam_2d` bei $\eta = 0.4$ für $2{,}000$ Schritte auf dem ungedrehten und dem gedrehten Tal aus (auch mit $\lambda_2 = 400$). Erklären Sie, warum bei diagonalen Vorkonditionierern die schlechte Konditionierung wieder auftritt, sobald die Eigenbasis gegenüber den Koordinatenachsen gedreht ist.
   :exlabel:`ex_adam-rotation`

<!--en:5dab189c-->
1. [code] **Grenzzyklen bei konstanter Lernrate und AMSGrad.** ●
    1. Zeigen Sie: Oszilliert eine einzelne Koordinate bei Adam gemäß $x_t = (-1)^t a$ auf $f(x) = \frac{\lambda}{2}x^2$, so beträgt ihre Amplitude $a = \frac{1-\beta_1}{1+\beta_1}\frac{\eta}{2}$ und ist unabhängig von $\lambda$.
    1. Führen Sie `adam_2d` auf `f_valley` für $20{,}000$ Schritte aus. Erklären Sie, warum sich die Iterierte dem Ursprung bis auf $10^{-40}$ nähert, solange $\hat{v}_t$ die anfänglichen Gradienten berücksichtigt. Warum springt sie wieder vom Ursprung weg, sobald $\hat{v}_t$ unter die Stabilitätsschwelle für das Momentum fällt? Überprüfen Sie, dass sich dieser Sprung verhindern lässt, indem Sie $\hat{v}_t$ durch $\max_{s \le t} \hat{v}_s$ ersetzen (AMSGrad).
   :exlabel:`ex_adam-constant-rate`

<!--en:661a48c7-->
1. [extended] **Häufige und seltene Token und gruppenweise Lernraten.**
    1. Teilen Sie wie bei :citet:`Kunstner.Yadav.Milligan.ea.2024` die $256$ Bytewerte von `WikiText103` in häufige ($\ge 0.1\%$) und seltene ($< 0.1\%$) Token ein. Vergleichen Sie den Validierungsverlust in jeder Gruppe für SGD mit abgestimmter Lernrate und Momentum, für Adam sowie für den Vorzeichenabstieg mit Momentum ($\eta\,\operatorname{sign}(\mathbf{m}_t)$).
       :exlabel:`ex_adam-frequent-and-rare-tokens`
    1. Weisen Sie beim SGD mit Momentum den drei Parametergruppen von `TinyLM` (Einbettungen, 2D-Matrizen, 1D-Vektoren) jeweils eine eigene Lernrate zu. Messen Sie, wie weit die gruppenweise und die koordinatenweise Skalierung jeweils den Abstand zwischen SGD und Adam verringern.
       :exlabel:`ex_adam-per-group-learning-rates`

<!--en:0765ea34-->
1. [code] **Adversarielle periodische Gradienten und Korrekturen der Schätzung des zweiten Moments.** Betrachten Sie die Folge mit Periode 3, gegeben durch $f_t(x) = 4x$ für $t \equiv 1 \pmod 3$ und $f_t(x) = -x$ in den übrigen Schritten, auf dem Intervall $x \in [-1, 1]$ :cite:`Reddi.Kale.Kumar.2019`. Führen Sie Adam mit $\beta_1 = 0, \beta_2 = 1/17, \eta_t = 0.5/\sqrt{t}$ aus und vergleichen Sie das Ergebnis mit AMSGrad und Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018`.
   :exlabel:`ex_adam-two-repairs-variance-estimate`

<!--en:dc804bc8-->
1. **Zwei Grenzfälle von AdamW und das Rotationsgleichgewicht.**
   :exlabel:`ex_adamw-two-limits`
    1. Leiten Sie für $x_{t+1} = (1 - \eta\lambda)x_t - \eta n_t$ drei Gleichgewichte her: das deterministische Gleichgewicht $|x_\infty| = 1/\lambda$ für konstantes $n_t = -1$, das Gleichgewicht $\mathbb{E}[x_\infty^2] \approx \eta/(2\lambda)$ bei Rauschen mit Mittelwert null, wenn die Rauschgrößen $n_t$ unabhängig und identisch verteilt sind und $\mathbb{E}[n_t]=0, \mathbb{E}[n_t^2]=1$ gilt, sowie das gemischte Gleichgewicht $\mathbb{E}[x_\infty^2] \approx \bar{n}^2/\lambda^2 + \eta/(2\lambda)$ für $n_t = \bar{n} + \xi_t$.
    1. Simulieren Sie $\mathbf{x}_{t+1} = (1 - \eta\lambda)\mathbf{x}_t - \eta\mathbf{u}_t$ in $\mathbb{R}^{1000}$ mit $\mathbf{u}_t \perp \mathbf{x}_t$ :cite:`Kosson.Messmer.Jaggi.2024`. Überprüfen Sie, dass der RMS-Wert im Gleichgewicht mit $\sqrt{\eta/(2\lambda)}$ und der Drehwinkel pro Schritt mit $\sqrt{2\eta\lambda}$ übereinstimmt.

<!--en:6ca9574e-->
1. [code] **Gemeinsame Suche über $(\eta, \lambda)$ und Ausnahmen für Parametergruppen.** ●
    1. Trainieren Sie `TinyLM` auf $8{,}000$ überlappenden Fenstern von `d2l.TimeMachine` für jeweils $800$ Schritte auf einem Gitter von Werten für $\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}\}$ und $\lambda$. Vergleichen Sie die Parametrisierung durch $\lambda$ mit der Parametrisierung durch die Schrumpfung pro Schritt $s = \eta\lambda$.
       :exlabel:`ex_adamw-joint-search`
    1. Vergleichen Sie die Validierungsperplexität auf `WikiText103` für $\lambda \in \{0.5, 1.0\}$, wenn alle Parameter oder nur die 2D-Gewichtsmatrizen dem Gewichtszerfall unterliegen (:numref:`subsec_adamw-what-not-to-decay`). Bestimmen Sie, welche Ausnahme den größeren Teil der Verbesserung erklärt: die Einbettungen oder die 1D-Vektoren.
       :exlabel:`ex_adamw-exemptions`

<!--en:0b71c54f-->
[Diskussion](https://d2l.discourse.group/t/1078)

<!--en:d5ba59f0-->
[Diskussion](https://d2l.discourse.group/t/1079)

<!--en:2ecbc1b4-->
Verfahren mit adaptiver Lernrate: AdaGrad, RMSProp, Adam und AdamW<br>**koordinatenweise RMS-Skalierung · Bias-Korrektur · Invarianz der Schrittlänge · entkoppelter Gewichtszerfall · Zeitskala und Gleichgewichtsnorm**

<!--en:97a0d94f-->
::: {.slide title="Warum Transformer koordinatenweise Lernraten brauchen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:08656d77-->
- **40× Unterschied zwischen Schichtgradienten:** Zwischen Tokeneinbettungen, Attention-Projektionen, MLP-Schichten und LayerNorm-Parametern unterscheiden sich die Rohgradienten im Transformer um den Faktor $15\text{--}40\times$. Bei jeder skalaren SGD-Schrittweite fallen die Updates für Einbettungen zu klein aus.

<!--en:93b5e1ac-->
- **Adaptive RMS-Vorkonditionierung:** Die Division jedes Koordinatenupdates durch die bisherige RMS-Skala seines Gradienten macht $\eta$ unmittelbar zur **Schrittlänge einer Koordinate im Parameterraum** ($\eta^\star \approx 3 \times 10^{-3}$ gegenüber $0.5$ bei SGD). Dafür muss der **Gewichtszerfall entkoppelt** werden (AdamW).

<!--en:3b357688-->
::: {.slide title="AdaGrad: Vorkonditionierung mit aufsummierten Gradientenquadraten"}

<!--en:a292733c-->
[Koordinatenweise Skalierung]{.kicker}

<!--en:b6dc4e9f-->
**AdaGrad** (Duchi et al., 2011) bildet elementweise die laufende Summe der Gradientenquadrate $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$ mit dem Startwert $\mathbf{s}_0 = \mathbf{0}$ und teilt den Schritt jeder Koordinate durch die entsprechende Komponente von $\sqrt{\mathbf{s}_t + \epsilon}$:

<!--en:765af9e7-->
- **Seltene und häufige Merkmale:** Bei selten aktualisierten Koordinaten (z. B. Einbettungen seltener Token) bleibt der Nenner $\sqrt{s_{t,i}}$ klein. Sobald das Merkmal auftritt, macht die Koordinate einen vollen Schritt.

<!--en:78de294a-->
- **Automatische achsenparallele Vorkonditionierung:** Da $|g_i| = \lambda_i |x_i|$ proportional zur Krümmung ist, gleicht die Division durch $\sqrt{s_{t,i}}$ den Krümmungsunterschied um den Faktor $\kappa = 20$ zwischen $x_1$ und $x_2$ im Tal aus.

<!--en:876d2768-->
::: {.slide title="RMSProp: Gleitende Mittelwerte mit abklingendem Gewicht für ältere Gradientenquadrate"}

<!--en:dd0f2e3d-->
[Weit zurückliegende Gradienten vergessen]{.kicker}

<!--en:0e4e33d1-->
- **Warum AdaGrad bei tiefen Netzen zum Stillstand kommt:** Da $\mathbf{s}_t = \sum_{s=1}^t \mathbf{g}_s^2$ monoton und unbeschränkt wächst ($\mathbf{s}_t \propto t$), fällt die effektive Schrittweite von AdaGrad wie $O(t^{-1/2})$. Das Lernen kommt vorzeitig zum Stillstand.

<!--en:147142ef-->
- **RMSProp** (Tieleman & Hinton, 2012) ersetzt die kumulative Summe durch einen exponentiellen gleitenden Mittelwert (EMA) mit einem Fenster von $\tau_2 = \frac{1}{1-\beta_2}$ Schritten:
  $$\mathbf{v}_t = \beta_2\,\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2$$

<!--en:b6acf52a-->
- RMSProp vergisst weit zurückliegende Gradienten ($\beta_2 = 0.9 \implies 10$ Schritte im Fenster). Dadurch bleibt die Schrittweite von null verschieden, und das Verfahren konvergiert im Tal bei $\eta = 0.4$ schnell.

<!--en:06425692-->
::: {.slide title="Zwei Folgen der RMS-Normierung ohne Korrektur"}

<!--en:e6a92478-->
[Große Anfangssprünge und späte Oszillation]{.kicker}

<!--en:88ac056d-->
- **1. Schrittverstärkung beim Kaltstart:** Aus $\mathbf{v}_0 = \mathbf{0}$ ergibt sich im ersten Schritt $\mathbf{v}_1 = (1-\beta_2)\mathbf{g}_1^2$. Die Folge ist ein übergroßer Sprung $|\Delta x_{1,i}| = \frac{\eta}{\sqrt{1-\beta_2}}$ ($3.16\,\eta$ bei $\beta_2=0.9$ und **$31.6\,\eta$** bei $\beta_2=0.999$).

<!--en:cca5b714-->
- **2. Auch spät noch nichtverschwindende Schrittlängen:** Nahe einem Minimum mit lokal annähernd konstantem $|g_t|$ gilt $\sqrt{v_t} \approx |g_t|$. Das Update $-\eta\,\operatorname{sign}(g_t)$ hat die feste Länge $\eta$. Die Koordinate springt im Bereich $\pm\eta/2$ hin und her, bis $\eta_t \to 0$ (§{sec_scheduler}).

<!--en:24d3e5ec-->
::: {.slide title="Adam: Erstes und zweites Moment mit Bias-Korrektur"}

<!--en:f96f640d-->
[Adaptive Momentenschätzung]{.kicker}

<!--en:c1f6dae8-->
- **Zwei gleitende Mittelwerte ($\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$):**
  $$\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t, \quad \mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$$

<!--en:4943e33f-->
- **Update mit Bias-Korrektur:**
  $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}, \quad \mathbf{x}_{t+1} = \mathbf{x}_t - \eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$$

<!--en:229f5793-->
- Die Division durch $1 - \beta^t$ normiert die geometrisch abfallenden Gewichte in jedem Schritt $t \ge 1$ (Standardwerte: $\beta_1 = 0.9, \beta_2 \in \{0.95, 0.999\}, \epsilon = 10^{-8}$).

<!--en:d42b4cf0-->
::: {.slide title="Skaleninvarianz: η gibt die Schrittlänge jeder Koordinate direkt an"}

<!--en:ec0cc5fe-->
[Geometrie von Adam]{.kicker}

<!--en:5d1ad436-->
- **Betrag des ersten Schritts im Grenzfall:** Bei $t = 1$ gilt $\hat{\mathbf{m}}_1 = \mathbf{g}_1$ und $\hat{\mathbf{v}}_1 = \mathbf{g}_1^2$. Für $\sqrt{\hat{v}_{1,i}} \gg \epsilon$ folgt daher näherungsweise:
  $$\Delta x_{1,i} = -\eta\,\frac{g_{1,i}}{|g_{1,i}|} = -\eta\,\operatorname{sign}(g_{1,i})$$
  Jede Koordinate bewegt sich im ersten Schritt um **$\pm\eta$** (im Tal entlang der $45^\circ$-Diagonalen).

<!--en:e58d319f-->
- **Diagonale Skaleninvarianz:** Bei Skalierung des Verlusts oder der Koordinate $i$ mit $c > 0$ gilt $g_{t,i} \to c\,g_{t,i}$, $\hat{m}_{t,i} \to c\,\hat{m}_{t,i}$ und $\sqrt{\hat{v}_{t,i}} \to c\sqrt{\hat{v}_{t,i}}$. Der Faktor $c$ kürzt sich im Quotienten heraus.

<!--en:1dadfca4-->
- **Grenzen der Schrittlänge:** Bei anhaltend gleichgerichteten Gradienten beträgt die Schrittlänge $\approx \eta$, bei Rauschen mit Mittelwert null $\approx \eta\sqrt{\frac{1-\beta_1}{1+\beta_1}} \approx 0.23\,\eta$.

<!--en:f8055ce6-->
::: {.slide title="Eigene Implementierung von Adam"}

<!--en:a4fbb6b9-->
[Umsetzung im Code]{.kicker}

<!--en:ff77a51e-->
Jeder trainierbare Parametertensor benötigt einen Schrittzähler $t$ und zwei Puffer $(\mathbf{m}, \mathbf{v})$ mit derselben Form wie der Tensor:

<!--en:645d6589-->
::: {.slide title="40× Unterschied bei Gradienten, 2× bei Adam-Schritten"}

<!--en:b4ea92bb-->
[Gradientenskalen in TinyLM]{.kicker}

<!--en:d0ff0f1d-->
- Die RMS-Werte der Gradienten über $10$ Minibatches für alle $8$ funktionalen Parametergruppen von `TinyLM` zeigen das Problem von SGD mit skalarer Lernrate:
  - **Bei Schritt 0:** Die RMS-Werte der Rohgradienten unterscheiden sich um den **Faktor $40\times$** (von $2 \times 10^{-4}$ bei `token_emb` bis $7 \times 10^{-3}$ bei `biases`).
  - **Bei Schritt 500:** Die RMS-Werte der Gradienten unterscheiden sich noch um den **Faktor $15\times$**.

<!--en:7d8bb88a-->
- **Ausgleich durch Adam:** Die Division durch $\sqrt{\hat{\mathbf{v}}_t}$ begrenzt die tatsächlichen RMS-Parameterupdates $\|\Delta\mathbf{x}\|_{\text{RMS}}/\eta$ aller $8$ Gruppen auf einen engen **Bereich mit dem Faktor $2\times$ ($0.12\text{--}0.28\,\eta$)**.

<!--en:500c43fe-->
::: {.slide title="Lernratensuche: Adam und SGD mit Momentum" layout="tall-output"}

<!--en:529d752a-->
[Gittervergleich mit 3 Zufallsstartwerten]{.kicker}

<!--en:4c50be20-->
- Die Lernratensuche umfasst für SGD mit Momentum ($\beta = 0.9$) die Werte $\eta \in \{0.1, 0.3, 0.5, 1.0\}$ und für Adam die Werte $\eta \in \{3{\times}10^{-4}, 10^{-3}, 3{\times}10^{-3}, 10^{-2}, 3{\times}10^{-2}\}$. Jeder Versuch nutzt $3$ Zufallsstartwerte ($2{,}000$ Schritte auf `TinyLM`):
  - **Faktor 100× bei der nominellen Lernrate:** Die beste Lernrate beträgt für SGD $\eta = 0.3$ und für Adam $\eta = 3 \times 10^{-3}$ ($\approx 0.1 \times \frac{0.3}{\text{scale}}$).
  - **Stabilität bei hohen Lernraten:** SGD divergiert gegen `inf` bei $\eta = 0.5$ (Faktor unter $2\times$ gegenüber dem Optimum). Bei Adam bleibt der Verlust selbst bei $\eta = 3 \times 10^{-2}$ endlich (Faktor $10\times$ über dem Optimum), da $\eta$ die Schrittlänge im Versuch begrenzt.

<!--en:4c7a976b-->
::: {.slide title="Adam erreicht den Zielverlust in halb so vielen Schritten wie SGD"}

<!--en:511a36d4-->
[Geschwindigkeit, Lernratenbereich und Stabilität]{.kicker}

<!--en:578b1421-->
- Der Vergleich der $3$ Läufe bei der jeweils besten Lernrate zeigt drei entscheidende Vorteile von Adam auf `TinyLM`:
  1. **Konvergenzgeschwindigkeit:** Adam erreicht $1.8$ Nat in **$650\text{--}775$ Schritten**, SGD mit Momentum benötigt $1{,}125\text{--}1{,}550+$ Schritte.
  2. **Nutzbarer Lernratenbereich (bis $+0.15$ Nat über dem Optimum):** **Faktor $10\times$** ($10^{-3}$ bis $10^{-2}$) gegenüber nur einem empfindlichen Gitterpunkt ($0.3$) für SGD.
  3. **Robustheit gegen Verlustspitzen:** Der größte Verlustanstieg in einem Schritt beträgt $<0.20$ Nat bei Adam gegenüber $>0.4\text{--}1.0$ Nat bei SGD.

<!--en:9003682e-->
::: {.slide title="Vorzeichenabstieg, die Rolle von ε und Probleme bei kleinem β₂"}

<!--en:6b3e6a18-->
[Funktionsweise und Grenzfälle]{.kicker}

<!--en:2bef4448-->
- **Adam als geglätteter Vorzeichenabstieg:** Mit $\beta_1 = \beta_2 = 0, \epsilon = 0$ ergibt sich $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\operatorname{sign}(\mathbf{g}_t)$. Mit Momentum erreicht der Vorzeichenabstieg auf Transformern denselben Vorsprung vor SGD wie Adam (Kunstner et al., 2023).

<!--en:2a04100f-->
- **$\epsilon$ begrenzt die Lernrate:** Es gilt $\frac{\eta}{\sqrt{\hat{v}_{t,i}} + \epsilon} \le \frac{\eta}{\epsilon}$. Für $\epsilon \gg \sqrt{\hat{v}_{t,i}}$ geht Adam in SGD mit Momentum und der Lernrate $(1-\beta_1)\eta/\epsilon$ über.

<!--en:99be6189-->
- **Warum $\beta_2$ nicht zu klein sein darf (Reddi et al., 2018):** Bei kleinem $\beta_2$ wirken Gradienten nur kurz nach. Seltene große Gradienten treiben $\sqrt{v_t}$ in die Höhe, während kleine Gradienten in Gegenrichtung größere Schritte bewirken. Die Wahl $\beta_2 \in [0.95, 0.999]$ verhindert das.

<!--en:7907a72e-->
::: {.slide title="Warum die übliche ℓ₂-Regularisierung bei Adam versagt"}

<!--en:03107fc6-->
[Gekoppelter und entkoppelter Gewichtszerfall]{.kicker}

<!--en:c893ee44-->
- **Gekoppelter $\ell_2$-Strafterm in Adam ($\mathbf{g}_t \leftarrow \mathbf{g}_t + \lambda\mathbf{x}_t$):** Adam teilt den Strafterm für den Gewichtszerfall durch den Vorkonditionierer der jeweiligen Koordinate:
  $$\Delta x_{t,i}^{\text{decay}} = -\frac{\eta\,\lambda}{\sqrt{\hat{v}_{t,i}} + \epsilon}\; x_{t,i}$$

<!--en:fa580a0d-->
- **Problem:** Gewichte mit kleinen Verlustgradienten schrumpfen bis zu $10^3\times$ so stark wie aktive Gewichte. Dominiert der Strafterm, normiert Adam ihn durch seinen eigenen Betrag. Dann gilt $x_{t,i} \to 0$ mit einer Geschwindigkeit von $\eta$ pro Schritt.

<!--en:6271520a-->
::: {.slide title="AdamW: Entkopplung des Gewichtszerfalls vom Vorkonditionierer"}

<!--en:f660ab7a-->
[Loshchilov & Hutter (2019)]{.kicker}

<!--en:9cc0fb60-->
**AdamW** berechnet $\hat{\mathbf{m}}_t$ und $\hat{\mathbf{v}}_t$ ausschließlich aus dem Gradienten $\mathbf{g}_t$ des Datenverlusts. Die Gewichte schrumpfen direkt durch Multiplikation mit dem einheitlichen Faktor $(1 - \eta\lambda)$:

<!--en:cc222010-->
::: {.slide title="Gekoppelte ℓ₂-Regularisierung lässt Gewichtsmatrizen kollabieren, AdamW lernt weiter"}

<!--en:009ce442-->
[Vergleich über 1000 Schritte bei λ = 0.1]{.kicker}

<!--en:7dc766a7-->
- Training von `TinyLM` über $1{,}000$ Schritte bei $\eta = 3 \times 10^{-3}$ und $\lambda = 0.1$ mit identischen Anfangsgewichten (anfänglicher RMS-Wert der Matrizen $= 0.0447$):
  - **Ohne Gewichtszerfall ($\lambda = 0$):** Endverlust **$\approx 1.74$ Nat**. Der RMS-Wert der Matrizen steigt auf $0.108\text{--}0.110$.
  - **Entkoppeltes `AdamW` ($\lambda = 0.1$):** Endverlust **$1.72\text{--}1.74$ Nat**. Der RMS-Wert der Matrizen bleibt auf $0.097\text{--}0.100$ begrenzt.
  - **Gekoppeltes `Adam + ℓ2` ($\lambda = 0.1$):** Der RMS-Wert der Matrizen sinkt um den Faktor $16\times$ auf **$0.0027\text{--}0.0028$**. Die internen Repräsentationen gehen verloren. Der Verlust stagniert bei **$3.15\text{--}3.16$ Nat** (Entropie der Unigrammverteilung der Bytes: $3.18$).

<!--en:0c147f33-->
::: {.slide title="AdamW in Summenform: Die Zeitskala des Gewichtszerfalls τ = 1/(ηλ)"}

<!--en:fee52234-->
[Gewichte als EMA der Updates]{.kicker}

<!--en:868cda0b-->
- **EMA der normierten Updates $\mathbf{u}_s = \frac{\hat{\mathbf{m}}_s}{\sqrt{\hat{\mathbf{v}}_s}+\epsilon}$ in Summenform (Wang & Aitchison, 2025):**
  $$\mathbf{x}_t = (1 - \eta\lambda)^t\,\mathbf{x}_0 - \eta\sum_{s=0}^{t-1} (1 - \eta\lambda)^{t-1-s}\,\mathbf{u}_s$$

<!--en:6e56b88b-->
- **Zeitskala des Gewichtszerfalls:** $\tau = \frac{1}{\eta\lambda}$ Schritte ($3{,}333$ Schritte bei $\eta = 3 \times 10^{-3}, \lambda = 0.1$).

<!--en:57162f92-->
- **Entkopplung der Hyperparameter:** Bei der Suche über $\eta$ die Schrumpfung pro Schritt $s = \eta\lambda = 1/\tau$ konstant halten ($\lambda = s/\eta$).

<!--en:97489e78-->
::: {.slide title="Gleichgewichtsnorm der Gewichte und Rotationsgleichgewicht"}

<!--en:0c89e46d-->
[Dynamik der Gewichte im stationären Zustand]{.kicker}

<!--en:05498d7d-->
- **Stochastisches Gleichgewicht der Norm ($t \gtrsim 2\tau$):** Aus dem Gleichgewicht zwischen Gewichtszerfall $(1-\eta\lambda)x_t$ und Schritten $-\eta n_t$ mit normiertem Rauschen (RMS-Wert eins) folgt:
  $$\text{RMS}(\mathbf{x}_\infty) \approx \sqrt{\frac{\eta}{2\lambda}}$$

<!--en:6f0e8879-->
- **Warum $\lambda = 0$ die Richtungsanpassung verlangsamt:** Ohne Gewichtszerfall wächst $\text{RMS}(\mathbf{x}_t) \propto \eta\sqrt{t}$ durch eine Irrfahrt. In skaleninvarianten LayerNorm-Teilschichten nimmt die relative Winkelschrittweite dadurch wie $1/\sqrt{t}$ ab.

<!--en:1bac190d-->
- **Rotationsgleichgewicht (Kosson et al., 2024):** Bei AdamW bleibt $\frac{\eta}{\text{RMS}(\mathbf{x}_\infty)} \approx \sqrt{2\eta\lambda}$ über alle Schichten hinweg konstant.

<!--en:b124f4ce-->
::: {.slide title="Ausnahmen vom Gewichtszerfall und Speicherbedarf des Optimiererzustands"}

<!--en:e0039170-->
[Parametergruppen und 16 Byte Speicherbedarf]{.kicker}

<!--en:a1eb5ac6-->
- **Gewichtszerfall nur für interne 2D-Projektionsmatrizen ($90\%$ der Parameter von `TinyLM`):** Die erste Ausnahme ist üblich, die zweite folgt OLMo 2.
  - **1D-Bias-Terme sowie Skalierungsfaktoren und Verschiebungen von LayerNorm ausnehmen ($\lambda = 0$):** Die Skalierungsfaktoren liegen anfangs nahe $1.0$ und steuern direkt die Merkmalsvarianz. Gewichtszerfall verzerrt hier die Aktivierungen.
  - **Token- und Positionseinbettungen ausnehmen ($\lambda = 0$):** Die Zeilen seltener Token erhalten nur dann Gradienten, wenn diese Token im Minibatch vorkommen. Gewichtszerfall bei jedem Schritt lässt seltene Einbettungen kollabieren und vergrößert nachgelagerte LayerNorm-Gradienten (OLMo 2, 2025).

<!--en:9a600dad-->
- **16 Byte pro Parameter in `float32`:** Gewicht ($4\text{ B}$) $+$ Gradient ($4\text{ B}$) $+$ AdamW-Zustand $\mathbf{m}, \mathbf{v}$ ($8\text{ B}$) $= 16\text{ B}$ ($8\text{ GB}$ allein für den Optimiererzustand bei einem Modell mit $1\text{B}$ Parametern).

<!--en:c011b1f1-->
::: {.slide title="Verständnisfragen zu Bias-Korrektur, Zeitskalen und VRAM"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:4ef5ca81-->
Drei Rechenaufgaben zum Verständnis:

<!--en:bf140907-->
1. **Fehlende Bias-Korrektur:** Adam läuft mit $\beta_1 = 0.9, \beta_2 = 0.999, \eta = 10^{-3}$ *ohne* Bias-Korrektur. Wie groß ist der Betrag des Parameterupdates $|\Delta x_{1,i}|$ im Schritt $t = 1$ im Vergleich zu Adam mit Korrektur?

<!--en:090372a6-->
2. **Invarianz von Zeitskala und relativer Schrittweite:** Ein Trainingslauf verwendet AdamW mit $\eta = 3 \times 10^{-3}$ und $\lambda = 0.5$. Die Lernrate wird auf $\eta = 1.5 \times 10^{-3}$ halbiert, der Gewichtszerfall auf $\lambda = 1.0$ verdoppelt. Wie ändern sich die Zeitskala des Gewichtszerfalls $\tau$, der RMS-Wert der Gewichte im Gleichgewicht und die relative Winkelschrittweite $\eta / \text{RMS}$?

<!--en:14df45e0-->
3. **VRAM-Bedarf:** Beim Training eines Transformers mit $7$ Milliarden Parametern speichert AdamW die Hauptgewichte, die Gradienten und $(\mathbf{m}, \mathbf{v})$ in `float32`. Wie viele Gigabyte belegen diese statischen Puffer, bevor Aktivierungen hinzukommen?

<!--en:557b98b6-->
**Lösungen:**

<!--en:977d5435-->
1. Ohne Bias-Korrektur gilt $|\Delta x_{1,i}| = \eta\frac{1 - \beta_1}{\sqrt{1 - \beta_2}} = \eta\frac{0.1}{\sqrt{0.001}} \approx$ **$3.16\,\eta = 3.16 \times 10^{-3}$**. Der Schritt ist damit um den Faktor $3.16\times$ größer als die Schrittlänge $\eta = 10^{-3}$ bei Adam mit Korrektur.

<!--en:692bef24-->
2. Da $\eta\lambda = 1.5 \times 10^{-3}$ konstant bleibt, bleiben sowohl die **Zeitskala von $\tau = \frac{1}{\eta\lambda} \approx 667$ Schritten** als auch die **relative Winkelschrittweite $\sqrt{2\eta\lambda} \approx 0.055$** **unverändert**. Die Gleichgewichtsnorm $\sqrt{\eta/(2\lambda)}$ **halbiert sich** (das hat auf skaleninvariante LayerNorm-Teilschichten keinen Einfluss).

<!--en:a16c6e5f-->
3. Bei $16\text{ bytes/parameter}$ beträgt der statische Speicherbedarf $7 \times 10^9 \times 16\text{ B} =$ **$112\text{ GB}$**. Davon entfallen **$56\text{ GB}$ ($50\%$)** auf den AdamW-Zustand $(\mathbf{m}, \mathbf{v})$.

<!--en:08a2fded-->
::: {.slide title="Zusammenfassung: Koordinatenweise adaptive Verfahren und AdamW"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:4d246ed8-->
| Optimierer oder Konzept | Updateregel oder Invariante | Speicherbedarf und Zeitskala | Rolle im Deep Learning |

<!--en:4cda9446-->
| **AdaGrad** | $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2,\; \Delta\mathbf{x}_t = -\frac{\eta}{\sqrt{\mathbf{s}_t+\epsilon}}\odot\mathbf{g}_t$ | $1$ Puffer ($4\text{ B}$ pro Parameter), Schrittweite fällt wie $O(t^{-1/2})$ | Dünn besetzte Merkmale bei konvexen Problemen, Stillstand bei tiefen Netzen mit nichtkonvexer Zielfunktion |

<!--en:94c79ca8-->
| **RMSProp** | $\mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$ | $1$ Puffer, Fenster für die Skalenschätzung $\tau_2 = \frac{1}{1-\beta_2}$ | Adaptive Schritte mit nichtverschwindender Länge, ohne Momentum und Bias-Korrektur |

<!--en:4d3c4bd8-->
| **Adam** | $\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t},\; \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t},\; \Delta\mathbf{x}_t = -\eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$ | $2$ Puffer ($8\text{ B}$ pro Parameter), $\tau_1 = 10, \tau_2 = 1{,}000$ | Invariant gegenüber diagonaler Skalierung, gleicht Gradientenunterschiede um den Faktor $40\times$ zwischen Schichten aus |

<!--en:38b6f7c4-->
| **AdamW** | $\mathbf{x}_{t+1} = (1 - \eta\lambda)\mathbf{x}_t - \eta\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}$ | Zeitskala des Gewichtszerfalls $\tau = \frac{1}{\eta\lambda}$ | Umgeht die Skalierung mit $\sqrt{\hat{v}_t}$, Gewichtszerfall nur für 2D-Gewichtsmatrizen |

<!--en:b1d54e68-->
| **Gleichgewichtsnorm** | $\text{RMS}(\mathbf{x}_\infty) \approx \sqrt{\frac{\eta}{2\lambda}},\quad \frac{\eta}{\text{RMS}(\mathbf{x}_\infty)} \approx \sqrt{2\eta\lambda}$ | Gleichgewicht nach $\sim 2\tau = \frac{2}{\eta\lambda}$ Schritten | Verhindert das Wachstum der Gewichte wie $\sqrt{t}$ und die Abnahme der effektiven Lernrate wie $1/\sqrt{t}$ |
