<!--en:5710f2e9-->
# Lernratenpläne

<!--en:4abce262-->
In den vorangegangenen Abschnitten haben wir die Lernrate $\eta$ während des gesamten Trainings konstant gehalten. Zweimal haben wir gesehen, warum eine feste Schrittweite ungünstig ist. Beim stochastischen Gradientenabstieg mit Minibatches (:numref:`sec_sgd`) schwankt der Iterationspunkt bei konstanter Lernrate in einer stationären Kugel um das Minimum (engl. noise ball) mit Varianz $O(\eta\sigma^2 / \lambda)$. Halbieren wir $\eta$, halbiert sich die Rauschgrenze, doch bis dorthin sind doppelt so viele Schritte nötig. Bei normierten Optimierern wie AdamW (:numref:`sec_adam`) und Muon (:numref:`sec_muon`) bleibt die Schrittweite von der Größenordnung $O(\eta)$, selbst wenn der Gradient gegen null geht. Erst wenn wir $\eta$ senken, kann der Iterationspunkt in einem Minimum zur Ruhe kommen.

<!--en:d8330be0-->
Moderne Trainingsrezepte lösen diesen Zielkonflikt mit einem **Lernratenplan** (engl. learning-rate schedule) $\eta_t$, der die Lernrate über die Schritte $t = 0, \ldots, T-1$ hinweg variiert. Fast alle heutigen Pläne verbinden zwei Phasen. In der **Aufwärmphase** (engl. warmup) steigt die Lernrate von null auf die maximale Lernrate $\eta_0$. Danach folgt der **Lernratenabfall** (engl. decay) in Richtung null. Wir untersuchen den Lernratenabfall zunächst an einem quadratischen Tal mit Rauschen. Dazu betrachten wir zwei Größen, das **kumulative Lernratenbudget** (engl. cumulative learning-rate budget) $B_T = \sum_{t=0}^{T-1}\eta_t$ und die **Endlernrate** (engl. terminal learning rate) $\eta_T$. Anschließend vergleichen wir an `TinyLM` einen konstanten Plan, einen Kosinusplan und einen Plan mit Aufwärmphase, Plateau und Abfall (engl. warmup–stable–decay, WSD). Wir zeigen, warum eine lineare Aufwärmphase die ersten Schritte auch bei hoher maximaler Lernrate stabilisiert. Schließlich zeigen wir, wie sich von einem Plateau mit konstanter Lernrate kurze Abfallphasen abzweigen lassen. So entstehen für unterschiedliche Trainingsdauern konkurrenzfähige Checkpoints.

<!--en:3164e2fa-->
## Lernratenabfall im quadratischen Tal mit Rauschen

<!--en:a753909b-->
Betrachten wir eine einzelne Koordinate im quadratischen Tal mit Krümmung $\lambda > 0$. Das Gradientenrauschen $\xi_t$ hat den Mittelwert null und die Varianz $\sigma^2$. Bei zeitlich veränderlichen Schrittweiten $\eta_t$ gilt:

<!--en:189ec525-->
Über $t$ Schritte schrumpft der deterministische Mittelwert um den Faktor $\prod_{s < t}(1 - \eta_s\lambda) \approx e^{-\lambda B_t}$. Dabei ist

<!--en:479420d8-->
das **kumulative Lernratenbudget**, also die Fläche unter der Lernratenkurve bis zum Schritt $t$. Ändert sich $\eta_t$ während der lokalen Relaxationszeit $1/(\eta_t\lambda)$ nur langsam, folgt die Varianz dem momentanen Wert $\eta_t\sigma^2 / (2\lambda)$ für die Kugel um das Minimum. Damit ergibt sich für den mittleren quadratischen Fehler die Zerlegung

<!--en:43f5eb09-->
Gleichung :eqref:`eq_scheduler-bias-noise` zeigt zwei Anforderungen an jeden Lernratenplan mit einer endlichen Zahl von $T$ Schritten:

<!--en:486172f7-->
1. **Ausreichendes kumulatives Budget $B_T$:** Das Gesamtbudget $B_T = \sum_{t=0}^{T-1}\eta_t$ muss so groß sein, dass die anfängliche Verzerrung $e^{-2\lambda_{\min} B_T} x_0^2$ auch in der flachsten Richtung verschwindet.

<!--en:37db9dbb-->
2. **Kleine Endlernrate $\eta_T$ bei ausreichendem Restbudget:** Auf ihren Endwert $\eta_T$ muss die Lernrate langsam genug absinken, damit das in Zwischenschritten $s$ eingetragene Rauschen durch das Restbudget $R_s = \sum_{u=s}^{T-1}\eta_u \gtrsim 1/\lambda_{\min}$ gedämpft wird.

<!--en:239590c3-->
Drei klassische Familien parametrischer Lernratenpläne verwenden einen stufenweisen (stückweise konstanten), einen exponentiellen oder einen polynomialen Abfall:

<!--en:e01b7652-->
Dabei bezeichnet $\gamma \in (0, 1)$ den Abfallfaktor, $\tau > 0$ die Zeitskala und $\alpha > 0$ den Exponenten. Wir simulieren $1{,}000$ unabhängige SGD-Läufe im Tal $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ über $T = 3{,}000$ Schritte. Daran sehen wir, wie das kumulative Budget $B_T$ und die Endlernrate $\eta_T$ die Konvergenz bestimmen und warum ein zu schneller exponentieller Abfall das Rauschen einfriert:

<!--en:4f317f74-->
Bei konstanter Lernrate ($\eta = 0.1$) verharrt der mittlere quadratische Fehler auf dem Wert für die stationäre Kugel um das Minimum ($0.25 + 0.016 \approx 0.27$). Beim polynomialen Abfall mit $\alpha = 1/2$ ($\eta_T \approx 0.036$) und $\alpha = 1$ ($\eta_T \approx 0.0033$) folgt der Fehler der Vorhersage aus :eqref:`eq_scheduler-bias-noise` bis auf $0.09\text{--}0.10$ bzw. $0.011\text{--}0.012$. Beim exponentiellen Abfall ($\gamma = 0.98$) beträgt die Endlernrate dagegen nur $10^{-27}$, dennoch verharrt der mittlere quadratische Fehler bei $0.12\text{--}0.13$. Nach Schritt $s$ verbleibt nur noch ein Budget von $R_s \approx \eta_s / (1-\gamma) = 50\eta_s$. Sobald $\eta_s \lesssim 0.1$ gilt ($R_s \lambda_1 \lesssim 1$), reicht es nicht mehr aus, um neu hinzukommendes Rauschen abzubauen. So bleibt eine Varianz von $(1-\gamma)\sigma^2 / (4\lambda_1^2) = 0.125$ bestehen (Aufgabe :exref:`ex_scheduler-frozen-noise`).

<!--en:e3170ea4-->
## Verlaufsformen des Lernratenabfalls bei einem Sprachmodell

<!--en:6259c3db-->
Bei einer festen Zahl von $T$ Schritten überwiegen im modernen Deep Learning zwei Verlaufsformen:

<!--en:45f3cabb-->
* **Kosinus-Abfall** (engl. cosine decay) :cite:`Loshchilov.Hutter.2016`. Die Lernrate folgt einer halben Kosinuswelle, von der maximalen Lernrate $\eta_0$ bei $t = 0$ bis zur Endlernrate $\eta_T$ bei $t = T$:
  $$
  \eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2}\left(1 + \cos\frac{\pi t}{T}\right).
  $$
  :eqlabel:`eq_scheduler-cosine`
  Für $\eta_T = 0$ ergibt sich beim Kosinus-Abfall das kumulative Budget $B_T = \frac{1}{2}\eta_0 T$, also die Hälfte des Budgets bei konstanter Lernrate $\eta_0$.

<!--en:0c893ccf-->
* **Plateau mit anschließendem Abfall (später linearer Abfall)**. Die Lernrate bleibt auf ihrem Höchstwert $\eta_0$ bis zum Schritt $t_{\text{decay}}$ (hier $0.8\,T = 1{,}600$) und fällt dann linear auf $0$ ab, verteilt über die letzten $20\%$ des Trainings. So bleiben $90\%$ des Budgets bei konstanter Lernrate erhalten ($B_T = 0.9\,\eta_0 T$), und der Plan endet dennoch bei $\eta_T = 0$.

<!--en:5d67e08d-->
Wir vergleichen den konstanten Plan, den Kosinusplan und das Plateau mit anschließendem Abfall ($\eta_0 = 3 \times 10^{-3}, T = 2{,}000$) mit AdamW ($\lambda = 0$) an `TinyLM`. Zunächst laden wir `WikiText103` und definieren den Modellkonstruktor `tiny_lm(seed)` sowie die Funktion `adamw(model, schedule)`, die den Optimierer mit dem Lernratenplan verbindet:

<!--en:a5249acd-->
Als Nächstes definieren wir die Funktionen `constant`, `cosine` und `stable_decay` für die drei Lernratenpläne. Die letzte hält $\eta_0$ bis `decay_start=1600` konstant. Danach fällt die Lernrate linear ab und erreicht bei $T = 2{,}000$ den Wert null. Die Funktion `warmup` ergänzt die Pläne um einen linearen Anstieg:

<!--en:ad966520-->
Nun trainieren wir `TinyLM` mit jedem der drei Pläne und je zwei Startwerten für den Zufallszahlengenerator. Wir erfassen das kumulative Budget $B_T$ und den Validierungsverlust bei Schritt $1{,}500$ (vor Beginn des späten Abfalls) und bei Schritt $2{,}000$:

<!--en:1cc804ed-->
Bei Schritt $1{,}500$ führt der Kosinus-Abfall. Seine Lernrate ist bereits auf $15\%$ des Höchstwerts gesunken, was hochfrequente Schwingungen dämpft. Zwischen Schritt $1{,}600$ und Schritt $2{,}000$ senkt `stable_decay` jedoch die Lernrate linear ($\eta_t \to 0$). Dabei sinkt der Verlust deutlich um $\sim 0.10$ Nat. Am Ende liegt er $0.05\text{--}0.06$ Nat unter dem Verlust beim konstanten Plan und $\sim 0.05$ Nat unter dem beim Kosinus-Abfall. Diesem stand nur etwa die Hälfte des kumulativen Budgets für die Bewegung im Tal zur Verfügung ($B_T = 3.0$ gegenüber $5.4$). Wer Checkpoints vor dem Ende des Lernratenabfalls vergleicht, kann also die umgekehrte Rangfolge wie am Trainingsende erhalten :cite:`Wen.Hall.Ma.ea.2025`.

<!--en:efe37dc7-->
## Aufwärmphase der Lernrate

<!--en:856a410f-->
Der Lernratenabfall bringt den Iterationspunkt am Trainingsende zur Ruhe. Die **Aufwärmphase der Lernrate** stabilisiert die ersten Schritte. Dazu steigt $\eta_t$ linear von $0$ auf $\eta_0$, verteilt über $W$ Schritte (typischerweise $1\%\text{--}5\%$ von $T$).

<!--en:97a923be-->
Warum ist eine Aufwärmphase nötig? Erstens verschiebt Adam (:numref:`sec_adam`) in Schritt $1$ jede Koordinate um $\pm\eta_0$, allein auf Grundlage eines einzigen Minibatches. Zu diesem Zeitpunkt enthalten weder der gleitende Mittelwert $\mathbf{m}_t$ für das Momentum noch der Mittelwert $\mathbf{v}_t$ für das zweite Moment Beiträge aus mehreren Batches. Bei `TinyLM` gilt für die anfänglichen Gewichtsmatrizen der linearen Schichten $\text{RMS}(\mathbf{W}_0) \approx 0.045$. Bei einem Kaltstart mit $\eta_0 = 0.03$ ändert sich jedes Gewicht um $\sim 67\%$ seines Anfangsbetrags, und zwar schon in Schritt $1$. Das beeinträchtigt die Initialisierung so stark, dass der Verlust bei etwa $2.8$ Nat stagniert. Zweitens sinkt, wie in :numref:`subsec_gd-edge-of-stability` gezeigt, mit dem schrittweisen Anstieg von $\eta_t$ allmählich die Obergrenze für die Schärfe. Beim Gradientenabstieg beträgt diese Grenze $2/\eta_t$, bei Adam etwa $38/\eta_t$ für die vorkonditionierte Schärfe. So gelangt das Netz in einen flachen Bereich, der die maximale Lernrate verträgt :cite:`Goyal.Dollar.Girshick.ea.2017,Kalra.Barkeshli.2024`.

<!--en:5d7ef394-->
Ein linearer Anstieg über $20$ Schritte (`warmup(peak, 20)`, nur $1\%$ der insgesamt $2{,}000$ Schritte) begrenzt die anfängliche relative Gewichtsänderung auf etwa $10\%$. Damit gelingt das Training bei $\eta_0 = 0.03$, und der Validierungsverlust am Ende sinkt um etwa ein ganzes Nat. Um die Gewichtsänderung von Schritt zu Schritt zu messen, extrahiert `linear_weights` die zweidimensionalen Gewichtsmatrizen von `TinyLM` mit Ausnahme der Einbettungsmatrizen:

<!--en:e5bf3f66-->
Mit `relative_change` berechnen wir $\|\Delta\mathbf{W}\|_{\text{RMS}} / \|\mathbf{W}\|_{\text{RMS}}$ bei den Schritten $1, 5, 20,$ und $100$. So vergleichen wir bei $\eta_0 = 0.03$ einen Kaltstart und eine lineare Aufwärmphase von $20$ Schritten über eine Trainingsdauer von $2{,}000$ Schritten:

<!--en:8b0d4a42-->
## Warmup–Stable–Decay (WSD) und Verzweigung an Checkpoints

<!--en:d22546de-->
Beim Kosinus-Abfall :eqref:`eq_scheduler-cosine` muss die Gesamtzahl $T$ der Schritte schon vor Schritt $1$ feststehen. An den Checkpoints während des Laufs ist die Lernrate noch erhöht. Für ein Training über $T$ hinaus muss man den Lernratenplan neu starten oder einen weiteren Plan anschließen. Eine kurze Aufwärmphase, gefolgt von einem Plateau bei maximaler Lernrate und einem kurzen Abfall am Ende, ergibt den **Warmup–Stable–Decay-Plan (WSD)** :cite:`Hu.Tu.Han.ea.2024,Hagele.Bakouch.Kosson.ea.2024`.

<!--en:44746d67-->
Zur geometrischen Beschreibung der WSD-Trajektorie verwenden :citet:`Wen.Li.Wang.ea.2024` eine Verlustlandschaft in Form eines **Flusstals** (:numref:`fig_opt_river_valley`). Auf dem Plateau mit hoher Lernrate schwingt der Iterationspunkt zwischen den steilen Talflanken hin und her. Zugleich kommt er im flachen Flussbett rasch talwärts voran. Der kurze Abfall am Ende dämpft die Querschwingungen und führt den Iterationspunkt direkt auf den Talboden. Da die Krümmung $\lambda_{\text{steep}}$ in Querrichtung groß ist, genügt dafür ein kleines Restbudget $R \gtrsim 1/\lambda_{\text{steep}}$.

<!--en:a21ee9d1-->
![Die Flusstalgeometrie von Warmup–Stable–Decay (WSD). Bei hoher konstanter Lernrate bewegt sich der Iterationspunkt rasch flussabwärts und schwingt dabei zwischen den steilen Talflanken hin und her. Auf einem kurzen Zweig mit fallender Lernrate kommt er rasch auf dem Talboden zur Ruhe.](../img/mdl-opt-river-valley.svg)

<!--en:95837b38-->
### Verzweigung auf dem Plateau

<!--en:925a34f3-->
Die Plateauphase hängt nicht von der angestrebten Schrittzahl $T$ ab. Wir können daher einen einzigen Lauf auf dem Plateau fortsetzen und an jedem Checkpoint eine kurze lineare Abkühlphase ($250$ Schritte) **abzweigen**. Dazu kopiert `branch(model, optimizer, schedule)` sowohl die Modellgewichte als auch die Momentenpuffer $(\mathbf{m}_t, \mathbf{v}_t)$ von AdamW. Die Kopie trainieren wir mit dem Lernratenplan `schedule` für die Abkühlphase weiter:

<!--en:f004487e-->
Für das Plateau trainieren wir über $2{,}000$ Schritte ein `TinyLM` mit konstanter Lernrate. Wir zweigen lineare Abfallphasen von je $250$ Schritten bei Schritt $500, 1{,}000,$ und $1{,}500$ ab. Den Zweig über die Schritte $1{,}000 \to 1{,}250$ vergleichen wir mit einem separaten Lauf, den wir von Anfang an auf $1{,}250$ Schritte ausgelegt haben:

<!--en:ae937b7c-->
Jede abgezweigte Abkühlphase senkt den Validierungsverlust deutlich unter den des fortgesetzten Plateaus (um $0.05\text{--}0.06$ Nat bei Schritt $1{,}250$ und Schritt $1{,}750$). Der Zweig über die Schritte $1{,}000 \to 1{,}250$ erzielt dasselbe Ergebnis wie ein von Anfang an auf $1{,}250$ Schritte ausgelegter Einzellauf. So erhalten wir fertige Modelle für drei Trainingsdauern mit insgesamt $2{,}250$ statt $3{,}750$ Schritten.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:96185357-->
Zwei Größen bestimmen einen Lernratenplan $\eta_t$. Sein **kumulatives Lernratenbudget** $B_T = \sum_{t=0}^{T-1}\eta_t$ bestimmt, wie weit der Iterationspunkt in flachen Richtungen vorankommt. Seine **Endlernrate** $\eta_T$ bestimmt die Rauschgrenze am Ende, sofern das Restbudget $R_s \gtrsim 1/\lambda$ langsam genug abnimmt, um früher eingetragenes Rauschen zu dämpfen. Senken wir bei `TinyLM` die Lernrate am Trainingsende auf null, verschwinden die Schwankungen der Größenordnung $O(\eta)$, die bei AdamW mit konstanter Lernrate von Schritt zu Schritt auftreten. Der Validierungsverlust sinkt um $0.05\text{--}0.06$ Nat. Eine **lineare Aufwärmphase** über die ersten $1\%\text{--}5\%$ der Schritte verhindert zu große Anfangsschritte bei Adam, bevor sich $(\mathbf{m}_t, \mathbf{v}_t)$ stabilisiert haben. Sie hält die Schärfe entlang der Trajektorie unter ihrer Obergrenze. Beim **Warmup–Stable–Decay-Plan (WSD)** bleibt $\eta_0$ nach der Aufwärmphase konstant. Erst während der letzten $10\%\text{--}20\%$ der Schritte fällt die Lernrate. So bleiben rund $90\%$ des Budgets für die Bewegung im Tal erhalten, und an jedem Checkpoint auf dem Plateau lässt sich eine kurze Abkühlphase abzweigen.

<!--en:2294243a-->
## Übungen

<!--en:e4d528b8-->
1. [code] **Eingefrorenes Rauschen bei exponentiellem Abfall.** Für $\eta_s = \eta_0\gamma^s$ beträgt das Restbudget $R_s \approx \eta_s/(1-\gamma)$. Nähern Sie die Summe $\sum_{s} \eta_s^2 \sigma^2 e^{-2\lambda R_s}$ durch ein Integral an. Zeigen Sie damit, dass die asymptotisch verbleibende Varianz in einer Richtung mit Krümmung $\lambda$ den Wert $(1-\gamma)\sigma^2 / (4\lambda^2)$ annimmt, wenn $2\lambda\eta_0 \gg 1-\gamma$ gilt. Überprüfen Sie diese Vorhersage mit `sgd_valley` für $\gamma \in \{0.98, 0.99, 0.995\}$.
   :exlabel:`ex_scheduler-frozen-noise`

<!--en:b48ecab0-->
1. [code] **Stufenweiser Abfall der Lernrate.** Entwerfen Sie anhand von :eqref:`eq_scheduler-bias-noise` einen $3$-stufigen, stückweise konstanten Lernratenplan über $3{,}000$ Schritte für `sgd_valley`. Vergleichen Sie den mittleren quadratischen Abstand am Ende mit dem beim polynomialen Abfall mit $\alpha = 1$ (etwa $0.01$).
   :exlabel:`ex_scheduler-step-decay`

<!--en:e52cc944-->
1. [code] **Lernratenpläne mit gleichem kumulativem Budget $B_T$ und gleicher Endlernrate $\eta_T$.** ● Vergleichen Sie einen über die gesamte Trainingsdauer linearen Abfall $\eta_t = \eta_0(1 - t/T)$ mit einem Kosinus-Abfall :eqref:`eq_scheduler-cosine` mit $\eta_T = 0$. Für beide gelten $B_T = \eta_0 T / 2$ und $\eta_T = 0$. Verwenden Sie dazu `sgd_valley` ($\eta_0 = 0.4, T = 3{,}000$). Erklären Sie anhand der Restbudgets $R_t$ nahe $t \approx T$, warum der Kosinus-Abfall am Ende einen geringeren Abstand erreicht.
   :exlabel:`ex_scheduler-same-two-numbers`

<!--en:f989084b-->
1. [code] **Dauer der Aufwärmphase und höchste verträgliche Lernrate.** ● Variieren Sie bei `TinyLM` die Dauer der linearen Aufwärmphase über $W \in \{2, 5, 10, 20, 50\}$ Schritte bei $\eta_0 = 0.03$. Vergleichen Sie die kürzeste Aufwärmphase, die stabiles Training erlaubt, mit Adams Zeitfenster $1/(1-\beta_1) = 10$ für das erste Moment. Bestimmen Sie, um wie viel eine Aufwärmphase von $20$ Schritten die höchste beim Training stabile Lernrate $\eta_0$ erhöht.
   :exlabel:`ex_scheduler-warmup-length`

<!--en:f1e21689-->
[Diskussion](https://d2l.discourse.group/t/1080)

<!--en:88321651-->
[Diskussion](https://d2l.discourse.group/t/1081)

<!--en:f7595d76-->
Lernratenpläne mit Aufwärmphase, Kosinus-Abfall und WSD<br>**Kumulatives Budget B_T und Endlernrate η_T · eingefrorenes Rauschen · lineare Aufwärmphase · Kosinus-Abfall · WSD und Verzweigung**

<!--en:af70dffc-->
::: {.slide title="Warum die Lernrate anfangs steigen und am Ende fallen muss"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:5b00265b-->
- **Risiko zu Beginn ($t = 1..W$):** Beim Kaltstart verschiebt AdamW in Schritt 1 jede Koordinate um $\pm\eta_0$, gestützt auf einen einzigen verrauschten Minibatch. Die Momentenschätzer $(\mathbf{m}_t, \mathbf{v}_t)$ enthalten dann noch keine Werte aus weiteren Schritten, und der Iterationspunkt hat noch keine flache Senke erreicht ($\lambda_{\max} \le 2/\eta_0$).

<!--en:2ee9820c-->
- **Risiko am Ende ($t \to T$):** Bei konstantem $\eta_0$ bleibt SGD in einer Kugel um das Minimum mit Varianz $\frac{\eta\sigma^2}{2\lambda b}$ gefangen. AdamW und Muon machen weiterhin Schritte der Größenordnung $O(\eta_0)$.

<!--en:3738bb95-->
- **Abhilfe durch den Lernratenplan:** Linearer Anstieg $0 \to \eta_0$, hohes Plateau für ein großes Budget $B_T$, dann Abfall $\eta_T \to 0$

<!--en:315c4ec0-->
::: {.slide title="Zerlegung in Verzerrung und Varianz beim Lernratenplan η_t"}

<!--en:61faab46-->
[Analytisches Modell für das Tal mit Rauschen]{.kicker}

<!--en:2c7a5d93-->
- **Zeitabhängige Rekursion im quadratischen Tal mit Rauschen:**
  $$\mathbb{E}\big[x_t^2\big] \approx \underbrace{e^{-2\lambda B_t}\,x_0^2}_{\text{Travel term}} + \underbrace{\frac{\eta_t\,\sigma^2}{2\lambda}}_{\text{Noise ball}}, \quad B_t = \sum_{s=0}^{t-1} \eta_s$$

<!--en:929499f2-->
- **Kumulatives Budget $B_T = \sum_{t<T} \eta_t$:** Mit wachsender Fläche unter der Lernratenkurve klingt die anfängliche Verzerrung exponentiell gemäß $e^{-2\lambda B_T}$ ab.

<!--en:a39bd6bc-->
- **Endlernrate $\eta_T$:** Sie bestimmt die Rauschgrenze $\frac{\eta_T\sigma^2}{2\lambda}$, sofern $\eta_t$ langsam genug fällt ($R_s \gtrsim 1/\lambda_{\min}$).

<!--en:254fb9ec-->
::: {.slide title="Klassische parametrische Abfallformen und ihre Budgets"}

<!--en:55a6f810-->
[Vier Lernratenpläne im Tal]{.kicker}

<!--en:569ed923-->
- Vier Lernratenpläne über $T = 3{,}000$ Schritte im Tal mit Rauschen ($\lambda_1 = 0.2, \lambda_2 = 4.0, \sigma^2 = 1$):
  - **Konstant ($\eta_t = 0.1$):** Großes Budget ($B = 300$), Stagnation auf dem Niveau der stationären Kugel um das Minimum ($\approx 0.27\text{--}0.29$).
  - **Polynomial mit $\alpha = 1/2$ und $\alpha = 1$:** Mit den Budgets $200.2$ und $48.2$ folgt der Fehler der momentanen Varianz in der Kugel um das Minimum bis auf $\approx 0.09\text{--}0.10$ bzw. $\approx 0.011\text{--}0.012$.
  - **Exponentiell ($0.4 \times 0.98^t$):** Das endliche Budget $B_\infty = 20$ tilgt den Einfluss von $x_0$, doch die **Varianz verharrt bei $0.125$**, obwohl $\eta_T \approx 10^{-27}$ gilt.

<!--en:473862e5-->
::: {.slide title="1000 Läufe im Tal: Warum ein exponentieller Abfall das Rauschen einfriert"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:3cf3bcee-->
- Mittelwerte über $1{,}000$ unabhängige Trajektorien im Tal mit Rauschen, jeweils über $T = 3{,}000$ Schritte:
  - **Konstant ($\eta = 0.1$):** Der Fehler stagniert auf dem Niveau der stationären Kugel um das Minimum ($\approx 0.27\text{--}0.29$).
  - **Polynomial mit $\alpha = 1/2$ und $\alpha = 1$:** Der Fehler folgt der momentanen Varianz in der Kugel um das Minimum bis auf $0.09\text{--}0.10$ bzw. $0.011\text{--}0.012$.
  - **Exponentiell ($\gamma = 0.98$):** Trotz der Endlernrate $\eta_T \approx 10^{-27}$ **verharrt der mittlere quadratische Abstand bei $0.12\text{--}0.13$**, und zwar ab Schritt $250$.

<!--en:1a7e2b85-->
::: {.slide title="Nur das verbleibende Budget (R_s) baut Rauschen ab"}

<!--en:6ddd2ffc-->
[Warum Rauschen einfriert]{.kicker}

<!--en:6ac2ad2e-->
- **Restbudget nach Schritt $s$ ($R_s = \sum_{u > s} \eta_u$):**
  $$\mathbb{E}\big[x_T^2\big] \approx e^{-2\lambda B_T}x_0^2 + \sigma^2 \sum_{s=0}^{T-1} \eta_s^2\,e^{-2\lambda R_s}$$

<!--en:4438107c-->
- **Zu kleines Restbudget bei geometrischem Abfall:** Für $\eta_s = \eta_0\gamma^s$ gilt $R_s \approx \frac{\eta_s}{1-\gamma}$. Sobald $\eta_s < \frac{1-\gamma}{2\lambda_1} = 0.05$ gilt, ist $2\lambda_1 R_s < 1$. Danach reicht die Kontraktion nicht mehr aus, um das Rauschen abzubauen.

<!--en:bf3910cf-->
- **Exakter Wert der eingefrorenen Varianz:** $\mathbb{E}[x_\infty^2]_{\text{exp}} \approx \frac{(1-\gamma)\sigma^2}{4\lambda^2} = 0.125$.

<!--en:9fd7eee3-->
::: {.slide title="Moderne Lernratenpläne mit festem Trainingsende"}

<!--en:7461345e-->
[Konstante Lernrate, Kosinus-Abfall und später Abfall]{.kicker}

<!--en:8fa51742-->
Bei einem festen Budget von $T = 2{,}000$ Schritten und einer maximalen Lernrate von $\eta_0 = 3 \times 10^{-3}$ stehen drei Lernratenpläne zum Vergleich:

<!--en:08ea7d1d-->
- **1. Konstant ($\eta_t = \eta_0$):** Rechtecksfläche $B_T = \eta_0 T = 6.0$, Endlernrate $\eta_T = 3 \times 10^{-3}$

<!--en:2892d3d6-->
- **2. Kosinus-Abfall (Loshchilov und Hutter, 2016):**
  $$\eta_t = \frac{\eta_0}{2}\left(1 + \cos\frac{\pi t}{T}\right)$$
  Symmetrische Halbwelle mit **halbem Budget** ($B_T = 0.5\,\eta_0 T = 3.0$) und $\eta_T = 0$.

<!--en:64c730b8-->
- **3. Plateau mit anschließendem Abfall (später linearer Abfall):** Die Lernrate bleibt bei $\eta_0$ bis $t_{\text{decay}} = 0.8\,T = 1{,}600$ und fällt dann linear auf $0$ (**$90\%$ des Budgets, $B_T = 5.4$**, $\eta_T = 0$).

<!--en:80ba28d4-->
::: {.slide title="Vergleich der Abfallformen an TinyLM"}

<!--en:d23f395d-->
[Schritt 1500 und Schritt 2000 im Vergleich]{.kicker}

<!--en:ae322feb-->
- Training von `TinyLM` mit AdamW ($\lambda = 0$) und zwei Zufallsstartwerten. Messung des Validierungsverlusts bei Schritt $1{,}500$ und Schritt $2{,}000$:
  - **Bei Schritt $1{,}500$ (vor Beginn des späten Abfalls):** Der Kosinus-Abfall führt ($1.58\text{--}1.59$ gegenüber $1.61$), weil $\eta_{1500}$ bereits auf $15\%$ von $\eta_0$ gesunken ist. Das dämpft das Rauschen der Schritte.
  - **Bei Schritt $2{,}000$ (nach Ende des späten Abfalls):** Zwischen Schritt $1{,}600$ und Schritt $2{,}000$ fällt der Verlust bei `stable_decay` deutlich um **$\sim 0.10$ Nat** auf **$1.51$ Nat**. Er liegt damit $0.05\text{--}0.06$ Nat unter dem Verlust beim konstanten Plan und $\sim 0.05$ Nat unter dem beim Kosinus-Abfall.

<!--en:a16c1b99-->
::: {.slide title="Warum der späte Abfall gewinnt und ein Zwischenvergleich täuscht"}

<!--en:d6f81460-->
[Budget und Rauschgrenze]{.kicker}

<!--en:d56e7c9d-->
- **Vorteil des späten Abfalls bei begrenztem Rechenbudget:** Bleibt $\eta_0$ über $80\%$ der Schritte konstant, stehen **$90\%$ des Budgets für die Bewegung im Tal** zur Verfügung ($B_T = 5.4$ gegenüber $3.0$ beim Kosinus-Abfall). Dennoch endet der Plan bei $\eta_T = 0$ ($1.51$ gegenüber $1.56$ Nat).

<!--en:774d3b15-->
- **Rangfolge erst nach der Abkühlphase bestimmen:** Bei Schritt $1{,}500$ schwanken die Gewichte auf dem Plateau von Schritt zu Schritt um Beträge der Größenordnung $O(\eta_0)$. Die Schwankungen verschwinden, sobald die lineare Abkühlphase über $400$ Schritte die Lernrate senkt ($\eta_t \to 0$) (Wen et al., 2025).

<!--en:e9c3974e-->
::: {.slide title="Warum hohe Lernraten beim Kaltstart scheitern: Adams erster Schritt"}

<!--en:926a4d22-->
[Warum die Lernrate in der Aufwärmphase steigt]{.kicker}

<!--en:8887e1df-->
- **Sprung nach dem Vorzeichen ohne Mittelung in Schritt 1:** Bei $t = 1$ verschiebt Adam jedes Gewicht anhand eines einzigen Batches um $\pm\eta_0$. Bei `TinyLM` ($\text{RMS}(\mathbf{W}_0) \approx 0.045$) bewirkt ein Kaltstart mit $\eta_0 = 0.03$ im ersten Schritt einen Sprung um **$67\%$ des anfänglichen Gewichtsbetrags**.

<!--en:3952423e-->
- **Linearer Anstieg über 20 Schritte ($\eta_t = \eta_0\min(1, \frac{t+1}{W})$):** Beginn bei $\eta_0/20 = 0.0015$ (relative Änderung $3.3\%$). So sammeln $\mathbf{m}_t, \mathbf{v}_t$ über $\tau_1 = 10$ Schritte Werte, während die Schärfe unter $2/\eta_t$ bleibt.

<!--en:179a0eb8-->
::: {.slide title="20 Aufwärmschritte ermöglichen das Training bei η₀ = 0.03"}

<!--en:12b6631c-->
[Kaltstart und linearer Anstieg]{.kicker}

<!--en:9690755e-->
- Vergleich eines Kaltstarts mit einer linearen Aufwärmphase über $20$ Schritte ($1\%$ von $2{,}000$ Schritten) bei $\eta_0 = 0.03$ und identischen Anfangsgewichten:
  - **Kaltstart:** Die relative Gewichtsänderung $\|\Delta\mathbf{W}\|_{\text{RMS}}/\|\mathbf{W}\|_{\text{RMS}}$ beträgt im ersten Schritt **$0.67$ ($67\%$)**. Der Verlust stagniert bei etwa $2.8\text{--}2.9$ Nat.
  - **Linearer Anstieg über 20 Schritte:** Die relative Gewichtsänderung beträgt höchstens **$0.03$** im ersten Schritt und $\le 0.10$ in den Schritten 5–20. Der Validierungsverlust am Ende sinkt um **mehr als ein ganzes Nat** auf $\approx 1.77$ Nat.

<!--en:fa30c425-->
::: {.slide title="Faustregeln für Aufwärmdauer und Wahl der maximalen Lernrate"}

<!--en:0a2dcd86-->
[Wahl des anfänglichen Anstiegs]{.kicker}

<!--en:a5089bb1-->
Die Aufwärmphase ermöglicht das Training bei der hohen maximalen Lernrate $\eta_0 = 0.03$, bei der ein Kaltstart scheitert. Dennoch liegt der Verlust am Ende bei $\eta_0 = 0.03$ mit Aufwärmphase ($\approx 1.77$ Nat) höher als bei $\eta_0 = 0.003$ ($1.56$ Nat):

<!--en:b9362f4f-->
| Parameter | Faustregel | Quantitative Begründung |

<!--en:1cd838aa-->
| **Dauer der Aufwärmphase $W$** | **$1\%\text{--}5\%$ aller $T$ Schritte** (stets $W \ge \frac{2}{1-\beta_1} = 20$ Schritte) | Der Puffer $\mathbf{m}_t$ für Adams erstes Moment ($\tau_1 = 10$) und die anfängliche Schätzung $\mathbf{v}_t$ des zweiten Moments haben Zeit, Statistiken aus mehreren Batches zu sammeln |

<!--en:d5740d9c-->
| **Budgetverlust durch die Aufwärmphase** | $\Delta B = \frac{W}{2}\,\eta_0$ (**$<0.5\%\text{--}2.5\%$ von $B_T$**) | Für $W = 20, T = 2{,}000, \eta_0 = 0.03$ fehlen durch den dreieckigen Flächenverlust $9.5\eta_0 = 0.285$ von insgesamt $B_T = 60$ ($0.48\%$) |

<!--en:25736e5f-->
| **Maximale Lernrate $\eta_0$** | $\eta_0$ **mit eingeschalteter Aufwärmphase** in Schritten mit Faktor $2\text{--}3\times$ variieren | Die optimale Rate $\eta_0^\star$ liegt voraussichtlich um den Faktor $3\text{--}10\times$ unter der höchsten Rate, die mit Aufwärmphase stabil bleibt |

<!--en:543e5d39-->
::: {.slide title="Kosinus-Abfall und Warmup–Stable–Decay (WSD) im Vergleich"}

<!--en:703321c6-->
[Festes Trainingsende oder jederzeitiger Abschluss]{.kicker}

<!--en:9ea1a362-->
- **Beim Kosinus-Abfall muss $T$ schon in Schritt 0 feststehen:** Er nutzt $50\%$ des Budgets bei maximaler Lernrate ($B_T = 0.5\eta_0 T$). Eine Verlängerung erfordert einen erneuten Anstieg der Lernrate.

<!--en:2b870512-->
- **Warmup–Stable–Decay (WSD, Hu et al., 2024):**
  - Konstantes Plateau bei $\eta_0$ (**unabhängig vom Trainingsende**, $85\text{--}90\%$ des Budgets für die Bewegung im Tal).
  - Von **jedem** Checkpoint auf dem Plateau lässt sich eine kurze Abkühlphase über $10\text{--}20\%$ der Schritte abzweigen. Ein Lauf liefert damit fertige Modelle für unterschiedliche Tokenbudgets.

<!--en:4a12d68c-->
::: {.slide title="Die Flusstalgeometrie von Warmup–Stable–Decay"}

<!--en:6c8fe828-->
[Deutung der Verlustlandschaft]{.kicker}

<!--en:81b7439a-->
- Wen et al. (2024) erklären die Effizienz von WSD anhand einer **Verlustlandschaft in Form eines Flusstals**:
  - **Plateau mit hoher Lernrate ($\eta_t = \eta_0$):** Der Iterationspunkt schwingt schnell zwischen den steilen Talflanken hin und her ($\lambda_{\text{steep}}$). Zugleich bewegt er sich im flachen Flussbett rasch talwärts ($\lambda_{\text{flat}} \ll \lambda_{\text{steep}}$).
  - **Kurze abgezweigte Abkühlphase ($\eta_t \to 0$):** Die Querschwingungen verlaufen in Richtungen mit hoher Krümmung ($\lambda_{\text{steep}} \gg 1$). Daher genügt ein kleines Restbudget $R \gtrsim 1/\lambda_{\text{steep}}$ ($10\text{--}20\%$ von $T$), damit die Schwingungen abklingen und der Iterationspunkt den Talboden erreicht.

<!--en:d707a0b9-->
::: {.slide title="Verzweigung an Checkpoints durch Kopieren von Adams Zustand"}

<!--en:d1bca0b8-->
[Erhalt der Momente des Optimierers]{.kicker}

<!--en:3a6abe98-->
Zweigt bei Schritt $s$ eine kurze Abkühlphase von einem Checkpoint auf dem Plateau ab, ist eine Kopie **sowohl der Modellparameter als auch des Optimiererzustands $(\mathbf{m}_s, \mathbf{v}_s, s)$** erforderlich:

<!--en:855e2efc-->
**Häufiger Fehler bei Checkpoints:** Startet ein Zweig die Abkühlphase mit einem frisch initialisierten `AdamW` ($\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$), bestimmt allein das Vorzeichen den ersten Schritt. Die Sprünge um $\pm\eta_s$ verfälschen die Gewichte vom Plateau.

<!--en:a48e070c-->
::: {.slide title="Modelle für mehrere Trainingsdauern aus einem einzigen Plateaulauf"}

<!--en:1291916d-->
[Abkühlphasen zu verschiedenen Zeitpunkten im Versuch]{.kicker}

<!--en:848cff4a-->
- Ein einziger Lauf über $2{,}000$ Schritte mit konstanter Lernrate ($\eta_0 = 3 \times 10^{-3}$) auf `TinyLM`. Lineare Abkühlphasen von je $250$ Schritten zweigen bei Schritt $500, 1{,}000,$ und $1{,}500$ ab:
  - Jede Abkühlphase von $250$ Schritten senkt den Validierungsverlust deutlich unter den des fortgesetzten Plateaus (um $0.05\text{--}0.06$ Nat bei Schritt $1{,}250$ und Schritt $1{,}750$).
  - Der Zweig über die Schritte $1{,}000 \to 1{,}250$ ($1.58\text{--}1.60$ Nat) erreicht den Verlust eines ab Schritt 0 auf $1{,}250$ Schritte ausgelegten Laufs bis auf eine Differenz von $<0.01$ Nat. So entstehen $3$ fertige Modelle in $2{,}250$ statt $3{,}750$ Schritten.

<!--en:49a2aadc-->
::: {.slide title="Verständnisfragen zu eingefrorenem Rauschen, Aufwärmphase und WSD-Verzweigung"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:e11b553d-->
Die folgenden Fragen zum Entwurf von Lernratenplänen rechnerisch beantworten:

<!--en:7dabf4dd-->
1. **Eingefrorenes Rauschen bei exponentiellem Abfall:** Bei einer eindimensionalen quadratischen Funktion mit Krümmung $\lambda = 0.2$ und Rauschvarianz $\sigma^2 = 1$ blieb beim exponentiellen Abfall $\eta_t = 0.4 \times \gamma^t$ mit $\gamma = 0.98$ eine Varianz von $\frac{(1-\gamma)\sigma^2}{4\lambda^2} = 0.125$ bestehen. Wie ändert sich die verbleibende Varianz bei einem langsameren Abfall mit $\gamma = 0.995$?

<!--en:cc099a97-->
2. **Kosinus-Abfall und später Abfall bei schneller Konvergenz im quadratischen Tal:** Bei `TinyLM` ($2{,}000$ Schritte) schnitt der späte lineare Abfall besser ab als der Kosinus-Abfall, weil `TinyLM` das zusätzliche Budget für die Bewegung im Tal benötigte ($B_T = 5.4$ gegenüber $3.0$). Welcher Plan erzielt bei einer eindimensionalen quadratischen Funktion mit Rauschen den kleineren Abstand am Ende, wenn der Iterationspunkt die Kugel um das Minimum schon nach $20$ Schritten erreicht?

<!--en:94c739e9-->
3. **Rechenersparnis durch WSD-Verzweigung:** Gesucht sind ausgewertete Checkpoints nach $T_1 = 1{,}000$, $T_2 = 2{,}000$ und $T_3 = 3{,}000$ Schritten, jeweils mit einer abschließenden linearen Abkühlphase von $300$ Schritten. Wie viele Trainingsschritte benötigen drei separate Kosinusläufe insgesamt, wie viele ein WSD-Plateau mit drei Zweigen?

<!--en:a6be1d16-->
::: {.slide title="Verständnisfragen zu eingefrorenem Rauschen, Aufwärmphase und WSD-Verzweigung (Lösungen)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:e10b28bf-->
1. Bei $\gamma = 0.995$ gilt $1 - \gamma = 0.005$ (um den Faktor $4\times$ kleiner als $0.02$). Damit sinkt die asymptotisch verbleibende Varianz **um den Faktor $4\times$ auf $\frac{0.005}{4 \times 0.04} = 0.03125$**.

<!--en:ffb88dd0-->
2. **Kosinus-Abfall.** Sobald die anfängliche Verzerrung $e^{-2\lambda B_t}x_0^2$ verschwunden ist (nach $20$ Schritten), zählt nur noch die Dämpfung des Rauschens gegen Trainingsende. Beim Kosinus-Abfall entfallen gegen Ende deutlich mehr Schritte auf kleine Lernraten.

<!--en:c4cfe229-->
3. Drei separate Kosinusläufe kosten $1{,}000 + 2{,}000 + 3{,}000 =$ **$6{,}000$ Schritte**. Ein WSD-Plateau bis Schritt $2{,}700$ mit drei Abkühlphasen von je $300$ Schritten, die bei Schritt $700, 1{,}700, 2{,}700$ abzweigen, kostet $2{,}700 + 3 \times 300 =$ **$3{,}600$ Schritte ($40\%$ weniger Rechenaufwand)**.

<!--en:e8a96859-->
::: {.slide title="Zusammenfassung: Lernratenpläne, Aufwärmphase und WSD"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:074a75b9-->
| Konzept | Mathematische Form oder Invariante | Budget und Verhalten am Trainingsende | Faustregel |

<!--en:7201e3a0-->
| **Zerlegung in Verzerrung und Rauschen** | $\mathbb{E}[x_t^2] \approx e^{-2\lambda B_t}x_0^2 + \frac{\eta_t\sigma^2}{2\lambda}$ | Das Budget $B_t = \sum_{s<t}\eta_s$ tilgt den Einfluss des Startwerts. $\eta_T$ bestimmt die Rauschgrenze | Erfordert ein großes $B_T$ und langsamen Abfall ($R_s \gtrsim 1/\lambda$) |

<!--en:6d793563-->
| **Risiko bei exponentiellem Abfall** | $\eta_t = \eta_0\gamma^t,\quad R_s \approx \frac{\eta_s}{1-\gamma}$ | Rauschen friert bei $\frac{(1-\gamma)\sigma^2}{4\lambda^2}$ ein, sobald $\eta_s \lesssim \frac{1-\gamma}{2\lambda}$ gilt | Schnellen geometrischen Abfall ohne Untergrenze oder polynomial auslaufende Endphase vermeiden |

<!--en:6c3f84c5-->
| **Lineare Aufwärmphase** | $\eta_t = \eta_0\min\!\left(1, \frac{t+1}{W}\right)$ | Verhindert bei AdamW den Gewichtssprung um $67\%$ im ersten Schritt | Lernrate über $W \sim 1\%\text{--}5\%$ der Schritte erhöhen (stets $\ge \frac{2}{1-\beta_1} = 20$ Schritte) |

<!--en:03f95063-->
| **Kosinus-Abfall** | $\eta_t = \eta_T + \frac{\eta_0-\eta_T}{2}\left(1 + \cos\frac{\pi t}{T}\right)$ | Budget $B_T = 0.5\,\eta_0 T$ (für $\eta_T = 0$) | Standard bei festem Trainingsende, wenn $T$ im Voraus bekannt ist |

<!--en:5d43ef22-->
| **Warmup–Stable–Decay (WSD)** | Konstantes Plateau bei $\eta_0$, dann lineare oder kosinusförmige Abkühlphase | Budget $B_T \approx 0.9\,\eta_0 T$, Verzweigung jederzeit möglich | Abkühlphasen über $10\text{--}20\%$ der Schritte abzweigen (Gewichte und $\mathbf{m}, \mathbf{v}$ kopieren) |
