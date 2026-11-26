<!--en:765ae55e-->
# Policy-Gradient

<!--en:18a2f980-->
In :numref:`sec_imitation` lernte eine Strategie (engl. policy) von einem Experten. Für jede vorgeführte Aktion machte sie einen Aufstiegsschritt auf deren Log-Wahrscheinlichkeit, jede Aktion mit Gewicht eins. In :numref:`sec_optimal_control` differenzierte ein Planer bekannte Übergangsgleichungen $f(\mathbf{x}_t, \mathbf{u}_t)$, um rückwärts die Riccati-Verstärkungen zu berechnen. Angenommen, wir haben weder einen Experten noch ein bekanntes differenzierbares Modell der Umgebung. Wir haben nur die Versuche des Agenten und die jeweilige Belohnung. Es liegt nahe, die Aktualisierungsregel des Verhaltensklonens (engl. behavior cloning) beizubehalten und die Gewichte zu ändern. Wir erhöhen die Log-Wahrscheinlichkeit jeder ausgeführten Aktion, gewichtet mit dem Ertrag (engl. return) ihrer Episode. So werden Aktionen aus lohnenden Episoden wahrscheinlicher. Lässt sich dieser Ansatz begründen? Wie ändert sich der erwartete Ertrag, wenn sich die Parameter $\theta$ der Strategie ändern?

<!--en:9ac3789b-->
Ronald Williams beantwortete diese Frage 1992 :cite:`Williams.1992`. Im Mittel ergibt die vorgeschlagene Aktualisierung den Gradienten des erwarteten Ertrags. Es handelt sich also um stochastischen Gradientenaufstieg (engl. stochastic gradient ascent). Williams nannte den Algorithmus REINFORCE. Der Ertrag einer Episode hängt von den Übergangswahrscheinlichkeiten der Umgebung ab, die der Agent nicht kennt. Für die Schätzung genügen dennoch die gezogenen Episoden und die Log-Wahrscheinlichkeiten der Strategie. Schreiben wir die Strategie als Funktion $\pi_\theta(a \mid s)$, entfällt auch eine Einschränkung der Wertetabellen aus :numref:`sec_qlearning`. Wer anhand einer Tabelle gierig handelt, muss das Maximum über eine endliche Liste von Aktionen bestimmen. Eine parametrisierte Strategie kann dagegen ein Drehmoment oder eine Verteilung über ein Vokabular von Token ausgeben. Wenden wir dieselbe Identität auf die Verteilung des nächsten Tokens eines Sprachmodells an, können wir Reasoning-Modelle mit einer einzigen Belohnung für die fertige Antwort trainieren.

<!--en:5fee8714-->
Wir leiten den Schätzer am kleinsten Problem mit einem Policy-Gradienten her. Dann erproben wir ihn an der Aufgabe mit Prüfprogramm aus :numref:`sec_mdp`, bei der sich jeder Gradient exakt berechnen lässt, und auf dem See. Im Mittel stimmt die Schätzung, doch nach jeder Aktualisierung brauchen wir neue Episoden. Bei kleinen Batches rauscht die Schätzung stark.

<!--en:1e5b5377-->
## Parametrisierung der Strategie

<!--en:3d1e860b-->
Der Gradientenaufstieg braucht eine Strategie, die sich mit ihren Parametern glatt ändert. Eine deterministische Strategie wählt das Argmax und ist damit stückweise konstant. Eine kleine Parameteränderung lässt die Aktion entweder unverändert oder führt zu einem Sprung auf eine andere Aktion. Die Wahrscheinlichkeiten einer stochastischen Strategie können sich dagegen um beliebig kleine Beträge ändern.

<!--en:a84cd6be-->
### Softmax-Präferenzen

<!--en:ad4dd53f-->
`Policy.tabular` aus :numref:`sec_imitation` speichert für jedes Zustand-Aktion-Paar einen Parameter $\theta_{s,a}$, die *Präferenz* (engl. preference). Die Softmax-Funktion über die Aktionen wandelt die Präferenzen im Zustand $s$ in Wahrscheinlichkeiten um:

<!--en:62e0dc01-->
Das ist die Softmax-Regression aus :numref:`sec_softmax` mit einem Zustandsindex als Eingabe, also dieselbe Strategie, die wir beim Verhaltensklonen angepasst haben. Eine Tabelle aus lauter Nullen ergibt die Gleichverteilungsstrategie. Anders als $Q(s,a)$ schätzt eine Präferenz keinen Ertrag. Addieren wir im selben Zustand zu allen Präferenzen dieselbe Konstante, bleibt die Strategie unverändert.

<!--en:32c59112-->
### Die Score-Funktion

<!--en:a2491da0-->
Die Softmax-Funktion lässt sich leicht differenzieren. Es gilt $\log \pi_\theta(a \mid s) = \theta_{s,a} - \log \sum_{a''} e^{\theta_{s,a''}}$. Für die Ableitung nach der Präferenz $\theta_{s,a'}$ einer beliebigen Aktion $a'$ im selben Zustand ergibt sich

<!--en:ba65204a-->
und die Ableitung nach den Präferenzen in jedem anderen Zustand ist null. Der Vektor $\nabla_\theta\log\pi_\theta(a\mid s)$ heißt *Score-Funktion* (engl. score function). Um die Log-Wahrscheinlichkeit von $a$ zu erhöhen, heben wir die Präferenz dieser Aktion an und senken zugleich jede Präferenz im Zustand $s$ proportional zur Wahrscheinlichkeit der jeweiligen Aktion. Bezeichnet $e_a$ den Indikatorvektor der Aktion $a$, dann lautet die Zeile des Scores für $s$ genau $e_a - \pi_\theta(\cdot \mid s)$. Ihre Komponenten summieren sich zu null. Der Score hat also keine Komponente in der einen Richtung, in der die Strategie unverändert bleibt, wenn wir zu jeder Präferenz bei $s$ dieselbe Konstante addieren. Wir prüfen die Formel durch automatisches Differenzieren von `Policy.log_prob` an einer zufällig gewählten Tabelle:

<!--en:868aa8cb-->
JAX differenziert eine Funktion, kein Modul an Ort und Stelle. Für die Prüfung übergeben wir daher das Strategienetz wieder an `log_prob` und differenzieren mit `nnx.grad` nach dem Netz, wie in `policy_step`.

<!--en:dd27cf6c-->
Der Gradient der Bibliotheksfunktion `log_prob` stimmt für Zustand 6 und Aktion 2 mit :eqref:`eq_softmax_score` überein. In allen anderen Tabellenzeilen ist er null.

<!--en:d6236e11-->
## Der Policy-Gradient

<!--en:dcef83f6-->
Wie lautet der Gradient des erwarteten Ertrags, und lässt er sich aus den Episoden des Agenten schätzen? Am einfachsten ist die Antwort für einen einzigen Zustand. Für Trajektorien behält sie dieselbe Form.

<!--en:c392d439-->
### Der einfachste Fall

<!--en:74e9b017-->
Betrachten wir einen Zustand und zwei Aktionen mit den Präferenzen $\theta_1$ und $\theta_2$. Aktion 1 bringt die Belohnung $1$, Aktion 2 die Belohnung $0$. Setzen wir $p = \pi_\theta(1)$. Die erwartete Belohnung ist $J(\theta) = p$. Nach :eqref:`eq_softmax_score` beträgt ihre Ableitung nach $\theta_1$ genau $p(1 - p)$. Angenommen, wir könnten $p$ nicht berechnen und nur Aktionen ausprobieren. Die Ableitung lässt sich dennoch als Mittelwert über die Aktionen der Strategie schreiben. Dazu multiplizieren und dividieren wir jeden Term mit bzw. durch $\pi_\theta(a)$:

<!--en:e1f13b10-->
Schon eine einzige gezogene Aktion liefert also eine Schätzung der Ableitung, wenn wir ihre Belohnung mit ihrem Score multiplizieren. Mit Wahrscheinlichkeit $p$ ziehen wir Aktion 1 mit Belohnung $1$ und Score $1 - p$. Mit Wahrscheinlichkeit $1 - p$ ist die Belohnung null und damit auch die Schätzung. Der Mittelwert ist $p(1 - p)$, die Ableitung selbst. Das zweite Moment beträgt $p(1-p)^2$. Daraus ergibt sich die Standardabweichung $(1-p)\sqrt{p(1-p)}$. Ihr Verhältnis zum Mittelwert ist also

<!--en:778347fc-->
und wächst, je seltener die lohnende Aktion ist. Ist sie selten, stimmt die Schätzung im Mittel, ist aber bei fast jeder einzelnen Ziehung unbrauchbar. Meist ist sie null und gelegentlich um etwa den Faktor $1/p$ zu groß. Mitteln wir über $n$ unabhängige Ziehungen, sinkt das Verhältnis um den Faktor $\sqrt n$.

<!--en:0522dfdd-->
Auf dem See finden wir zu Beginn des Lernens dieselbe Struktur. Die Gleichverteilungsstrategie erreicht das Ziel in etwa $1.4\%$ der Episoden (:numref:`sec_valueiter`), auf griffigem wie auf rutschigem Eis. Denn ein gleichverteilt gezogener Befehl führt zu einer gleichverteilten Bewegung. Eine Episode, die das Ziel verfehlt, bringt nichts ein. Bei $p = 0.014$ beträgt das Verhältnis $8.4$. Ein Batch aus 16 Episoden enthält mit Wahrscheinlichkeit $(1 - 0.014)^{16} \approx 0.80$ keinen einzigen Erfolg. Vier von fünf Batches liefern dann die Schätzung null. Im Rest dieses Abschnitts führen wir dieselbe Rechnung mit Trajektorien anstelle der beiden Aktionen durch.

<!--en:1fe0f284-->
### Ein Optimierungsproblem über Trajektorien

<!--en:3dbf5358-->
Auf dem See ist eine Episode eine Trajektorie $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$, die im ersten erreichten Endzustand $s_T$ endet. Ihre Länge $T$ ist also zufällig. Der Agent wählt jede Aktion, die Umgebung jeden Folgezustand. Die Wahrscheinlichkeit einer Trajektorie ist daher ein Produkt aus Faktoren der Strategie, Faktoren des Übergangskerns und der Verteilung der Startzustände $\mu_0$ aus :numref:`sec_mdp`:

<!--en:ace8c561-->
Nur die Faktoren der Strategie hängen von $\theta$ ab. Die Zielfunktion ist der erwartete Ertrag über diese Trajektorien mit $R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t$:

<!--en:e285f867-->
Auf dem See ist der Startzustand fest. Daher gilt $J(\theta) = V^{\pi_\theta}(s_0)$ in der Schreibweise aus :numref:`sec_valueiter`. Bei der Aufgabe mit Prüfprogramm ist $\mu_0$ die Gleichverteilung über die vier Prompts.

<!--en:32c1454d-->
### Der Trick mit der logarithmischen Ableitung

<!--en:ae8d4541-->
Der Rechenschritt aus dem einfachsten Fall lässt sich unverändert übernehmen. Der Ertrag $R(\tau)$ einer festen Trajektorie hängt nicht von $\theta$ ab, nur ihre Wahrscheinlichkeit. Ziehen wir den Gradienten in die Summe und multiplizieren und dividieren wir mit bzw. durch $P(\tau;\theta)$, erhalten wir

<!--en:5647e274-->
also

<!--en:8b503aba-->
Die Identität $\nabla \log P = \nabla P / P$ heißt *Trick mit der logarithmischen Ableitung* (engl. log-derivative trick). Sie schreibt den Gradienten wieder als Mittelwert über Trajektorien, die die aktuelle Strategie erzeugt und die wir daher ziehen können. Die Summe über Trajektorien ist unendlich, denn eine Episode kann beliebig viele Schritte dauern. Trotzdem dürfen wir den Gradienten in die Summe ziehen. Unter einer Softmax-Strategie besteht von jedem Zustand aus eine positive Wahrscheinlichkeit, in wenigen Schritten einen Endzustand zu erreichen. Lange Episoden sind daher exponentiell selten, und jeder Schritt trägt einen Score von beschränkter Länge bei.

<!--en:168894b0-->
### Wegfall der Übergangswahrscheinlichkeiten

<!--en:aaab4077-->
Der Gradient enthält noch $\log P(\tau;\theta)$, und $P(\tau;\theta)$ enthält den Übergangskern, den der Agent nicht kennt. Der Logarithmus des Produkts aus :eqref:`eq_traj_prob` ist eine Summe. Beim Differenzieren bleiben nur die Terme der Strategie übrig:

<!--en:44af7ec5-->
:numref:`fig_rl_trajectory_chain` zeigt die Kette der Faktoren. Die Verteilung der Startzustände und der Übergangskern hängen nicht von $\theta$ ab. Ihre Logarithmen haben daher den Gradienten null. Der Übergangskern bestimmt weiterhin, welche Trajektorien wir ziehen, doch der Schätzer wertet ihn nie aus. In diesem Sinne ist das Verfahren *modellfrei*. Zugleich sehen wir, welche Annahme die Herleitung braucht. Die Dynamik darf nicht von den Parametern der Strategie abhängen (Aufgabe :exref:`ex_policy-gradient-parameter-dependent-dynamics` lässt diese Annahme fallen).

<!--en:4a3306a9-->
![Die Wahrscheinlichkeit einer Trajektorie als Kette von Faktoren, $P(\tau;\theta) = \mu_0(s_0)\prod_t \pi_\theta(a_t \mid s_t)\,P(s_{t+1} \mid s_t, a_t)$. Nur die Faktoren der Strategie (blau, oberhalb der Kette) hängen von $\theta$ ab. Die Verteilung der Startzustände und der Übergangskern (grau, unten) sind davon unabhängig. Ihre Logarithmen haben daher den Gradienten null. Der Score einer Trajektorie ist somit die Summe der Scores der Strategie und lässt sich ohne Kenntnis von $P$ berechnen.](../img/mdl-rl-trajectory-chain.svg)

<!--en:a0ec3c86-->
### Der REINFORCE-Schätzer

<!--en:3095db29-->
Setzen wir die Summe der Scores in :eqref:`eq_pg_gradient` ein und ersetzen wir den Erwartungswert durch den Mittelwert über $n$ Trajektorien $\tau_1, \ldots, \tau_n$ aus der aktuellen Strategie, ergibt sich

<!--en:707b56f1-->
Jede Trajektorie erhöht die Log-Wahrscheinlichkeit jeder darin ausgeführten Aktion um einen Betrag proportional zu ihrem Ertrag. Das ist die eingangs vermutete Regel. Jeder Term hat den Erwartungswert $\nabla_\theta J(\theta)$, also ist $\hat u$ eine erwartungstreue Schätzung des Gradienten. REINFORCE :cite:`Williams.1992` macht den Aufstiegsschritt $\theta \leftarrow \theta + \eta\, \hat u$ mit der Schrittweite $\eta$ und zieht dann einen neuen Batch aus der aktualisierten Strategie. Auf dem See ist jeder Ertrag nichtnegativ. Somit wirkt jede gezogene Aktion in Richtung einer höheren eigenen Wahrscheinlichkeit. Für nicht gezogene Aktionen sinkt die Wahrscheinlichkeit durch den Anteil $-\pi_\theta(\cdot\mid s)$ des Scores. Eine Episode mit Ertrag null ändert nichts.

<!--en:25cd73a9-->
:numref:`fig_rl_score_ascent` zeigt einen Schritt für ein einstufiges Problem mit kontinuierlicher Aktion. Mit einer Softmax-Funktion über vier Aktionen ließe sich das nicht als Kurve darstellen. Die Strategie ist eine Gauß-Verteilung $\mathcal{N}(a; \mu, 1)$. Ihr Score bezüglich des Mittelwerts lautet $a - \mu$. Eine Episode besteht aus einer einzigen Aktion $a$. Ihr Ertrag $R(a)$ ist daher die Belohnung dieser Aktion. Der Schritt lautet $\mu \leftarrow \mu + \eta\, \frac{1}{n}\sum_i R(a_i)(a_i - \mu)$. Der Mittelwert bewegt sich auf die gezogenen Aktionen mit den höchsten Belohnungen zu. Zwölf Ziehungen verschieben ihn auf $0.32$, während der exakte Gradient $0.52$ ergäbe.

<!--en:0668dbbe-->
![Ein REINFORCE-Schritt bei einem einstufigen Problem. Die Strategie ist $\pi_\theta(a) = \mathcal{N}(a; \mu, 1)$ mit $\mu = 0$, die Belohnung beträgt $R(a) = 0.4 + 2e^{-(a-2)^2/2}$. Die erwartete Belohnung ist daher $J = 0.92$ und der exakte Gradient $\mathrm{d}J/\mathrm{d}\mu = 0.52$. (a) Jede der zwölf gezogenen Aktionen erhöht ihre eigene Log-Wahrscheinlichkeit, gewichtet mit ihrer Belohnung. Alle Belohnungen sind positiv, daher zeigen alle Pfeile nach oben. (b) Der Schritt $\mu \leftarrow \mu + \eta\,\frac{1}{12}\sum_i R(a_i)(a_i - \mu)$ mit $\eta = 1$ verschiebt den Mittelwert von $0$ auf $0.32$, in Richtung der Aktionen mit höherer Belohnung. Der exakte Gradient hätte ihn auf $0.52$ verschoben.](../img/mdl-rl-score-ascent.svg)

<!--en:7130dbbc-->
## REINFORCE an den durchgängigen Beispielen

<!--en:27b4451a-->
Bei der Aufgabe mit Prüfprogramm hat jede Episode einen Schritt und keine Übergänge. Daher lässt sich jede Größe exakt berechnen. Auf dem See dauern Episoden mehrere Schritte, und es gibt einen Übergangskern, den die Aktualisierungsregel nie ausliest.

<!--en:b5194d2b-->
### Die Aufgabe mit Prüfprogramm

<!--en:42eabb80-->
Die Aufgabe mit Prüfprogramm aus :numref:`sec_mdp` umfasst vier arithmetische Prompts $x$ mit denselben neun möglichen Antworten $y$. Die Strategie ist eine Tabelle $\theta$ mit einer Zeile von Präferenzen pro Prompt. Dabei ist $\pi_\theta(y \mid x)$ die Softmax-Funktion angewandt auf Zeile $x$. Das Prüfprogramm vergibt $r(x, y) = 1$ für die richtige Antwort und sonst $0$. Eine Trajektorie besteht aus einer einzigen Antwort auf einen Prompt, mit Wahrscheinlichkeit $\mu_0(x)\,\pi_\theta(y \mid x)$ und ganz ohne Übergangskern. Zeile $x$ von $\theta$ beeinflusst nur die erwartete Punktzahl für Prompt $x$, also $J_x(\theta) = \sum_y \pi_\theta(y \mid x)\, r(x, y)$. Nach :eqref:`eq_softmax_score` mit dem Prompt anstelle des Zustands ist ihr exakter Gradient eine Summe aus neun Termen:

<!--en:47445218-->
REINFORCE ersetzt diese Summe durch einen Mittelwert über $K$ gezogene Antworten. Für jeden Prompt zieht `verifier_step` insgesamt $K$ Antworten, wandelt deren Belohnungen mit der Funktion `weighting` in Gewichte um und verschiebt die Zeile von $\theta$ um `lr` $= \eta$ mal den gewichteten Mittelwert der $K$ Scores. Mit `weighting = lambda r: r` entsprechen die Gewichte den Belohnungen, und der Schritt folgt :eqref:`eq_reinforce`. In :numref:`sec_baselines` übergeben wir andere Gewichtungen. Das Prüfprogramm übergeben wir als Argument `reward`. So können wir mit demselben Schritt auch anhand eines anderen Prüfprogramms trainieren.

<!--en:4237cb5f-->
Das Prüfprogramm verlangt exakte Übereinstimmung und belohnt nur die richtige Antwort $y^*$. Die Schätzung für Prompt $x$ lautet daher $\frac{c}{K}\big(e_{y^*} - \pi_\theta(\cdot \mid x)\big)$, wobei $c$ die richtigen Antworten unter den $K$ Antworten zählt. Die Anzahl $c$ ist binomialverteilt mit Erfolgswahrscheinlichkeit $p = \pi_\theta(y^* \mid x)$. Die Schätzung hat also den Mittelwert $p\,(e_{y^*} - \pi_\theta(\cdot\mid x))$, den exakten Gradienten, und die relative Standardabweichung $\sqrt{(1-p)/(Kp)}$. Das ist das Verhältnis aus dem einfachsten Fall geteilt durch $\sqrt K$. An der Referenztabelle gilt $p \approx 1/8$ für die beiden lösbaren Prompts. Eine Gruppe mit $K = 16$ Antworten sollte also um etwa $0.66$ der Gradientenlänge vom exakten Wert abweichen.

<!--en:22a2b4d7-->
Die nächste Codezelle prüft diese Vorhersage. Sie zieht 10 000 Schätzungen an der Referenztabelle (bei `lr=1.0` ergibt der aktualisierte Parametervektor minus $\theta$ die Schätzung) und macht einen Schritt mit $\eta = 2$:

<!--en:80aadcfb-->
Eine einzelne Schätzung weicht wie vorhergesagt um $0.66$ der Gradientenlänge ab. Der Fehler des Mittelwerts aus $M$ Schätzungen sinkt mit wachsendem $M$ weiter, im Mittel wie $0.66/\sqrt M$, ohne erkennbare Untergrenze. Der Schätzer ist erwartungstreu. Jede Schätzung beruht auf einer binomialverteilten Anzahl pro lösbarem Prompt. Daher kann der Fehler eines einzelnen Mittelwerts deutlich unter diesem Richtwert liegen oder bis zu etwa doppelt so groß sein. Bei jedem lösbaren Prompt enthalten etwa $12\%$ der Gruppen keine richtige Antwort ($(7/8)^{16} \approx 0.12$) und tragen nichts bei. Bei `8*8` und `9*9` akzeptiert das Prüfprogramm keine der möglichen Antworten. Jede Gruppe bleibt ohne Treffer. Selbst beliebig viele Ziehungen liefern keinen Gradienten für einen Prompt, den keine Antwort löst. Ein exakter Schritt mit $\eta = 2$ erhöht die Erfolgswahrscheinlichkeit von $0.062$ auf $0.077$. Ein Schritt mit gezogenen Antworten schneidet im Median etwa ebenso gut ab. Er senkt die Erfolgswahrscheinlichkeit bei dieser Aufgabe nie, denn nur der Score der richtigen Antwort erhält ein Gewicht ungleich null.

<!--en:453924d2-->
### Der See

<!--en:4e0afc88-->
Auf dem See dauert eine Episode mehrere Schritte, und der Übergangskern kommt ins Spiel. Wir verwenden griffiges Eis, auf dem jeder Befehl den Agenten wie beabsichtigt bewegt. In den vorigen Abschnitten war das Eis rutschig. Der Grund für die Wahl ist, wie viel eine Episode über eine Aktion aussagt. Starten wir mit der Gleichverteilungsstrategie, ergeben beide Karten dieselbe Zufallsbewegung und dieselbe Erfolgswahrscheinlichkeit von etwa $1.4\%$. Auf rutschigem Eis bewegt ein Befehl den Agenten jedoch mit Wahrscheinlichkeit $1/3$ in jede der drei möglichen Richtungen, die befohlene oder eine der beiden dazu senkrechten. Der Befehl schließt also nur die entgegengesetzte Richtung aus. Bei der Gleichverteilungsstrategie ist der exakte Gradient auf rutschigem Eis dadurch nur ein Drittel so lang wie auf griffigem Eis. Das Rauschen einer Schätzung aus einer Episode bleibt etwa gleich. Für dieselbe Präzision muss ein Batch daher neunmal so groß sein (Aufgabe :exref:`ex_policy-gradient-theorem-table` leitet den Faktor her).

<!--en:f8eaf0dd-->
Mit besserer Strategie wächst der Abstand. Erfolgreiche Episoden dauern auf rutschigem Eis mehrere Dutzend Schritte statt sechs. Jeder Erfolg verteilt sich daher auf viel mehr Aktionen und wird mit $\gamma^{T-1}$ diskontiert. REINFORCE lernt auch auf rutschigem Eis, wenn auch langsam (Aufgabe :exref:`ex_policy-gradient-softmax-plateaus`).

<!--en:f45789b7-->
Wir entfernen die von Gymnasium vorgegebene Grenze von 100 Schritten für diese Karte. Die Zielfunktion :eqref:`eq_pg_objective` ist für vollständige Episoden definiert. Die Summe der Belohnungen einer abgebrochenen Episode ist nicht ihr Ertrag (:numref:`sec_mdp`). Eine Softmax-Strategie weist jeder Aktion eine positive Wahrscheinlichkeit zu. Auf dieser Karte endet daher jede Episode mit Wahrscheinlichkeit eins. Die Grenze von 10 000 Schritten dient nur zur Absicherung und liegt weit über der Länge aller Episoden in diesen Läufen:

<!--en:acee10d6-->
`Batch` speichert die Übergänge mehrerer Episoden in flachen Arrays und merkt sich für jede Episode den Index unmittelbar nach ihrem letzten Schritt. `episode_returns` berechnet $R(\tau)$ für jede Episode. `per_step` wiederholt einen Wert pro Episode für jeden ihrer Schritte. So wird der Ertrag zum Gewicht jeder zugehörigen Aktion. Der Container speichert auch den Folgezustand und ein Kennzeichen `term`. Es zeigt eine *Terminierung* an, niemals einen *Abbruch*. Nach der Regel aus :numref:`sec_mdp` hat eine abgebrochene Episode keinen Endzustand erreicht. Hier terminieren alle Episoden auf dem See. Das Kennzeichen spielt also erst eine Rolle, wenn Episoden abgebrochen werden.

<!--en:5ac47bc4-->
`rollout` füllt einen Batch. Die Funktion spielt vollständige Episoden mit jeder Strategie, die als Funktion `policy(obs, rng) -> action` vorliegt. Das ist die Schnittstelle zur Aktionswahl von `evaluate` aus :numref:`sec_valueiter`. Für jeden Schritt hängt die Funktion Zustand, Aktion, Belohnung, Folgezustand und Terminierungskennzeichen an. Alle Zufallsentscheidungen des Agenten stammen vom NumPy-Generator `rng`. Der Zufallsstartwert legt somit jede gezogene Episode fest.

<!--en:90526c54-->
Die Trainingsschleife verwendet wieder `policy_step` aus :numref:`sec_imitation`. Die Funktion macht einen Optimierungsschritt für $\frac{1}{N}\sum_t w_t \log \pi_\theta(a_t \mid s_t)$ über die $N$ Schritte eines Batchs. Beim Verhaltensklonen übergaben wir $w_t = 1$. REINFORCE übergibt den Ertrag der Episode, zu der Schritt $t$ gehört. Intern minimiert `policy_step` das Negative dieser Summe, einen *Ersatzverlust* (engl. surrogate loss). Sein Wert sagt nichts über die Güte der Strategie aus. Nur sein Gradient am aktuellen $\theta$ zählt, denn er liefert die Schätzung.

<!--en:7d7ef581-->
Zwei Einzelheiten weichen von :eqref:`eq_reinforce` ab. Erstens dividiert `policy_step` durch die Schrittzahl $N$ statt durch die Episodenzahl $n$. Das teilt die Schätzung durch die mittlere Episodenlänge im Batch, einen Faktor, der von Batch zu Batch schwankt. Zweitens verwenden wir Adam mit der Lernrate $0.1$ von `Policy.tabular` anstelle des einfachen Schritts $\theta \leftarrow \theta + \eta\hat u$. Einfacher Gradientenaufstieg kommt bei Aktionen mit geringer Wahrscheinlichkeit nur langsam voran, denn nach :eqref:`eq_softmax_score` enthält der Gradient der jeweiligen Präferenz diese Wahrscheinlichkeit als Faktor. Adam dagegen verschiebt jede Koordinate ungefähr um seine Lernrate (Aufgabe :exref:`ex_policy-gradient-softmax-plateaus` vergleicht beide Verfahren). Die Schleife sammelt einen neuen Batch, gewichtet jeden Schritt mit dem Ertrag seiner Episode, macht einen Optimierungsschritt und liefert den mittleren Ertrag und die Schrittzahl des Batchs:

<!--en:41092905-->
Wir trainieren zwölf Agenten ausgehend von der Gleichverteilungsstrategie, jeden mit eigenem Zufallsstartwert. Wir zeichnen den mittleren Ertrag jedes Trainingsbatchs als Schätzung von $J(\theta)$ und zum Vergleich die Obergrenze $\gamma^5 \approx 0.774$ eines Wegs mit sechs Schritten:

<!--en:10013f4a-->
Die Grafik zeigt den Median der zwölf Läufe und ein Band vom langsamsten bis zum schnellsten Lauf. Anfangs verläuft der Median flach. Wie im einfachsten Fall vorhergesagt, enthalten die meisten frühen Batches keinen Erfolg und liefern die Schätzung null. Sobald ein Batch einen Erfolg enthält, werden die zugehörigen Aktionen wahrscheinlicher. Erfolge häufen sich, und der mittlere Ertrag steigt rasch. Jeder Lauf nähert sich innerhalb einiger Dutzend Aktualisierungen der gestrichelten Obergrenze und bleibt etwas darunter, denn die Strategie ist weiterhin stochastisch und verlässt gelegentlich den kürzesten Weg. Die nächste Codezelle zeigt die erlernte Strategie des ersten Agenten mit der wahrscheinlichsten Aktion und ihrer Wahrscheinlichkeit auf jedem Feld:

<!--en:d00085a8-->
Die Strategie konzentriert sich auf einen der drei kürzesten Wege zum Ziel. Welchen sie wählt, hängt davon ab, welche Episoden zufällig zuerst Erfolg hatten. An den Löchern und am Ziel bleibt die Strategie gleichverteilt, weil der Agent dort nie eine Aktion ausführt. Auf Feldern abseits des gewählten Wegs bleiben die Präferenzen aus frühen erfolgreichen Episoden, die über diese Felder führten, weitgehend erhalten. Sie ändern sich nur, wenn eine Episode einen Umweg über diese Felder macht. Das wird selten, sobald sich der Weg festigt. Zwischen Besuchen führt das Momentum von Adam die letzte Änderung noch einige Dutzend Aktualisierungen lang fort, dann bleiben die Präferenzen konstant. An keiner Stelle hat die Schleife den Übergangskern ausgewertet. `rollout` zeichnete die vom Simulator gezogenen Folgezustände auf. Die Aktualisierung verwendete nur Zustände, Aktionen und Erträge. Hier sehen wir den Wegfall des Übergangskerns aus :numref:`subsec_pg-cancellation`, den die Aufgabe mit Prüfprogramm mangels Übergangskern nicht zeigen konnte.

<!--en:33fb9e4b-->
## Das Policy-Gradient-Theorem

<!--en:c978f12f-->
REINFORCE gewichtet den Score jeder Aktion mit dem Ertrag der gesamten Trajektorie, einschließlich der Belohnungen vor dieser Aktion. Eine Aktion kann bereits erhaltene Belohnungen nicht beeinflussen. Diese sollten daher im Mittel nichts beitragen. Das Policy-Gradient-Theorem :cite:`Sutton.McAllester.Singh.ea.2000` präzisiert diese Aussage und stellt den Gradienten als Mittelwert über die von der Strategie besuchten Zustände statt über ganze Trajektorien dar. Für die Formulierung lassen wir die Kette nach Ende der Episode ohne weitere Belohnung in ihrem Endzustand verharren. So ist $s_t$ für jedes $t$ definiert. Wir schreiben $d^{\pi_\theta}_t(s) = \Pr(s_t = s)$ für die Verteilung von $s_t$ unter $\pi_\theta$ und

<!--en:22311afe-->
für ihren diskontierten Mittelwert über die Zeit, die *diskontierte Zustandsbelegungsverteilung* (engl. discounted state-occupancy distribution). Durch den Faktor $1-\gamma$ summieren sich die Gewichte $\gamma^t$ zu eins. Somit ist $d^{\pi_\theta}_\gamma$ eine Wahrscheinlichkeitsverteilung über Zustände.

<!--en:fa871d2e-->
### Aussage und Beweis

<!--en:25143585-->
**Satz (Policy-Gradient-Theorem).** *Für die Zielfunktion $J(\theta) = \mathbb{E}_{s_0 \sim \mu_0} [V^{\pi_\theta}(s_0)]$ gilt*

<!--en:8d37cafc-->
**Beweis.** Aus :eqref:`eq_pg_gradient` und dem Wegfall des Übergangskerns folgt $\nabla_\theta J = \mathbb{E}\big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R(\tau)\big]$. Für jedes $t$ zerlegen wir den Ertrag in die Belohnungen vor $t$ und jene ab $t$. Die Belohnungen vor $t$ stehen fest, sobald der Agent in $s_t$ ist, während die Ziehung von $a_t$ noch aussteht. Der Score von $a_t$ hat bedingt auf $s_t$ den Mittelwert null, denn $\sum_a \pi_\theta(a \mid s)\nabla_\theta \log\pi_\theta(a\mid s) = \sum_a \nabla_\theta \pi_\theta(a\mid s) = \nabla_\theta 1 = 0$. Die früheren Belohnungen tragen also im Mittel nichts bei. Für die Belohnungen ab $t$ bedingen wir auf $(s_t, a_t)$. Aufgrund der Markov-Eigenschaft beträgt ihre erwartete diskontierte Summe $\gamma^t Q^{\pi_\theta}(s_t, a_t)$. Damit erhalten wir die erste Form. Für die zweite gruppieren wir die Summe über $t$ nach dem besuchten Zustand. Für jede Funktion $f$ gilt $\mathbb{E}\big[\sum_t \gamma^t f(s_t, a_t)\big] = \sum_s \sum_t \gamma^t d^{\pi_\theta}_t(s)\ \mathbb{E}_{a \sim \pi_\theta(\cdot\mid s)} f(s, a)$. Nun setzen wir die Definition von $d^{\pi_\theta}_\gamma$ ein. In Endzuständen gilt $Q^{\pi_\theta} = 0$, sodass die Schritte nach Ende einer Episode nichts beitragen. $\blacksquare$

<!--en:a089e776-->
### Folgerungen

<!--en:ab46ed47-->
Der Satz hat drei Konsequenzen. Erstens brauchen wir für den Score zum Zeitpunkt $t$ nur die Belohnungen ab $t$. Im Beweis entfielen die früheren Belohnungen als Rauschen mit Mittelwert null. Ein Schätzer, der sie weglässt, hat daher denselben Mittelwert und rauscht meist weniger. Zweitens ist der Gradient ein Mittelwert über die besuchten Zustände. Ihr Gewicht hängt davon ab, wie oft und wie früh die Strategie sie besucht. Ein selten erreichter Zustand erhält daher nur einen kleinen Gradienten. Besucht ihn ein Batch gar nicht, ist die Schätzung für seine Zeile exakt null. Deshalb ändern sich die Felder abseits des zuvor erlernten Wegs kaum. Drittens lassen viele Implementierungen in der ersten Form den Faktor $\gamma^t$ weg. Damit treten an die Stelle von $d^{\pi_\theta}_\gamma$ die undiskontierten Besuchshäufigkeiten der Zustände. Das ändert das relative Gewicht spät besuchter Zustände und liefert im Allgemeinen eine Aktualisierung, die kein Gradient einer Zielfunktion ist :cite:`Nota.Thomas.2020`. Aufgabe :exref:`ex_policy-gradient-discount-that-implementations-drop` untersucht den Unterschied.

<!--en:7ccd180d-->
## Aufwand und Grenzen

<!--en:b0b86ca3-->
### Datenbedarf von On-Policy-Verfahren

<!--en:ab017358-->
Der Schätzer :eqref:`eq_reinforce` mittelt über Trajektorien aus der *aktuellen* Strategie. Nach einer Aktualisierung haben sich die Parameter geändert, und die bereits gesammelten Trajektorien folgen nicht länger der Verteilung $P(\tau;\theta)$. Verfahren mit dieser Anforderung heißen *On-Policy-Verfahren*. Sie sammeln für jede Aktualisierung einen neuen Batch. Q-Learning ist ein *Off-Policy-Verfahren*. Seine Aktualisierungsregel lernt über die gierige Strategie anhand von Übergängen, die eine beliebige Verhaltensstrategie erzeugt hat (:numref:`sec_qlearning`). Wir messen den Datenbedarf von REINFORCE in Umgebungsschritten:

<!--en:c49c7657-->
Der Mittelwert eines Batchs erreicht erstmals $0.7$, etwa $90\%$ der Obergrenze, nach 25 bis 50 Aktualisierungen. Das entspricht 3000 bis 7000 Umgebungsschritten. Der gesamte Lauf mit 256 Aktualisierungen braucht etwa 27 000 Schritte, denn noch lange nachdem sich die Strategie nicht mehr verbessert, sammelt jede Aktualisierung einen neuen Batch. Auf derselben Karte fand tabellarisches Q-Learning mit $\alpha = 0.9$ in fast jedem Lauf eine Tabelle, mit der die gierige Strategie das Ziel erreicht. Der gesamte Lauf dauerte etwa 3500 Schritte (:numref:`sec_qlearning`). Es liegt nahe, einen Batch für mehrere Aktualisierungen zu verwenden. Ohne Korrektur ist das jedoch nicht gerechtfertigt (Aufgabe :exref:`ex_policy-gradient-breaking-policy-rule-purpose`).

<!--en:f19bfe1b-->
### Erwartungstreue und Varianz

<!--en:47ab1d45-->
Wie bei der Aufgabe mit Prüfprogramm können wir die Schätzung auf dem See mit dem exakten Gradienten vergleichen, denn das Modell ist bekannt. In :numref:`sec_valueiter` war der Wert einer Strategie die Lösung des linearen Gleichungssystems $V^\pi = r_\pi + \gamma P_\pi V^\pi$ mit $P_\pi(s' \mid s) = \sum_a \pi(a \mid s) P(s' \mid s, a)$ und $r_\pi(s) = \sum_a \pi(a \mid s)\, r(s, a)$. Die Zielfunktion lautet daher

<!--en:996939ee-->
Jeder Eintrag von $P_{\pi_\theta}$ und $r_{\pi_\theta}$ ist eine glatte Funktion von $\theta$. Automatisches Differenzieren durch die Lösung des linearen Gleichungssystems liefert daher den exakten Gradienten $\nabla_\theta J$. Löcher und Ziel sind in `TabularMDP.from_gym` absorbierende Zustände mit Belohnung null. Die Lösung berücksichtigt also die Terminierung. Nach 16 Aktualisierungen im Lauf mit Zufallsstartwert 3 halten wir die Strategie `probe` fest. Zu diesem Zeitpunkt steigt der Ertrag bereits:

<!--en:4c00875a-->
`policy_objective` berechnet $J(\theta)$ nach :eqref:`eq_pg_exact_objective` aus einer Präferenztabelle, einem Übergangskern `P[s, a, s']` und den Belohnungen `r[s, a]`. Die Funktion `grad` des Frameworks differenziert dabei durch die Lösung des Gleichungssystems. Für die festgehaltene Strategie berechnen wir $J$ und den exakten Gradienten:

<!--en:05409cb8-->
Der Ertrag dieser Strategie liegt zwischen dem der Gleichverteilungsstrategie und der Obergrenze. Halten wir nun $\theta$ fest und ziehen wir viele Schätzungen. `tabular_score` summiert die gewichteten Scores $w_t\,(e_{a_t} - \pi_\theta(\cdot \mid s_t))$ eines Batchs nach :eqref:`eq_softmax_score` jeweils in der Zeile des zugehörigen Zustands. Damit berechnet `reinforce_estimate` eine Ziehung von :eqref:`eq_reinforce`, ähnlich wie `verifier_step`. Jeder Schritt erhält den Ertrag seiner Episode als Gewicht, die Summe teilen wir durch $n$. Für jede Batchgröße $n$ ziehen wir fünfzig Schätzungen und geben drei Zahlen relativ zu $\|\nabla_\theta J\|$ aus. Die erste ist die Wurzel aus dem mittleren quadratischen Fehler einer einzelnen Ziehung, die zweite der Fehler des Mittelwerts der fünfzig Ziehungen. Die dritte gibt an, welcher Fehler für diesen Mittelwert bei einem erwartungstreuen Schätzer zu erwarten wäre, also die erste Zahl geteilt durch $\sqrt{50}$:

<!--en:df10ab1e-->
Bei $n = 4$ weicht eine einzelne Ziehung um mehr als das Dreifache der Länge des zu schätzenden Gradienten ab. Manche Ziehungen können null ergeben, weil keine Episode des Batchs das Ziel erreicht hat. Der Fehler halbiert sich bei jeder Vervierfachung von $n$, wie es das $1/\sqrt n$-Gesetz aus dem einfachsten Fall vorhersagt. Bei jedem $n$ weicht der Mittelwert aus fünfzig Ziehungen etwa so stark ab wie bei einem erwartungstreuen Schätzer. Der Test findet also keine Verzerrung. Aufgabe :exref:`ex_policy-gradient-testing-unbiasedness` erweitert ihn auf wesentlich mehr Ziehungen, bei denen auch eine kleine Verzerrung sichtbar würde.

<!--en:f82d046d-->
### Plateaus und Konvergenz

<!--en:6430a96e-->
Die Zielfunktion ist in $\theta$ nicht konkav. Schon im einfachsten Fall ist $J = p$ eine logistische Funktion von $\theta_1 - \theta_2$, die zu beiden Seiten abflacht. Bei einer nichtkonkaven Funktion kann der Gradientenaufstieg im Allgemeinen an einem ungünstigen stationären Punkt zum Stillstand kommen. Bei einer tabellarischen Softmax-Strategie ist das ausgeschlossen. Es gibt keinen suboptimalen stationären Punkt. Weist die Verteilung der Startzustände jedem Zustand eine positive Wahrscheinlichkeit zu, konvergiert der exakte Gradientenaufstieg mit hinreichend kleiner Schrittweite zu einer optimalen Strategie :cite:`Agarwal.Kakade.Lee.ea.2021,Mei.Xiao.Szepesvari.ea.2020`. Die Konvergenz kann dennoch langsam sein. Hat die bessere Aktion in einem Zustand eine geringe Wahrscheinlichkeit, enthält der Gradient ihrer Präferenz diese Wahrscheinlichkeit als Faktor. Der Aufstieg kommt auf einem solchen Plateau nur langsam voran. Es gibt Markov-Entscheidungsprozesse, bei denen die Zahl der Schritte für den exakten Softmax-Gradientenaufstieg exponentiell mit der Anzahl der Zustände wächst :cite:`Li.Wei.Chi.ea.2021`.

<!--en:ad72a7f9-->
Aktualisierungen aus Stichproben mit fester Schrittweite bergen ein zweites Risiko. Einige zufällig erfolgreiche Batches können die Strategie in einem Zustand nahezu deterministisch auf eine schlechtere Aktion festlegen. Danach ist der Gradient, der diese Wahl rückgängig machen würde, winzig. Die Strategie bleibt lange bei dieser Wahl :cite:`Mei.Dai.Xiao.ea.2021`. Auf griffigem Eis endeten alle zwölf Läufe nur wenige Hundertstel unter der Obergrenze. Auf rutschigem Eis pendelt sich dasselbe Verfahren je nach Zufallsstartwert auf unterschiedlichen Niveaus ein, teils weit unter dem Optimum. Aufgabe :exref:`ex_policy-gradient-softmax-plateaus` untersucht die Ursache.

<!--en:d8e2142c-->
Wir können eine Strategie nun allein anhand ihrer eigenen Episoden und deren Belohnungen verbessern, ohne Modell, Experten oder Wertetabelle. Bei der Aufgabe mit Prüfprogramm hat dieselbe Aktualisierung bereits die Form des Post-Trainings eines Sprachmodells. Eine Antwort ist eine Trajektorie und erhält am Ende ihre Belohnung. Die Schätzung stimmt im Mittel. Bei der festgehaltenen Strategie und vier Episoden pro Batch weicht eine einzelne Ziehung jedoch um mehr als das Dreifache der Gradientenlänge ab, und jede erfolglose Episode trägt nichts bei. In :numref:`sec_baselines` untersuchen wir, wie viel von diesem Rauschen sich entfernen lässt, ohne den Mittelwert zu verschieben.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:9ebc28cf-->
Policy-Gradient-Verfahren verbessern eine parametrisierte Strategie $\pi_\theta$ direkt durch Gradientenaufstieg auf ihrem erwarteten Ertrag $J(\theta)$. Mit dem Trick mit der logarithmischen Ableitung schreiben wir $\nabla_\theta J$ als Mittelwert über die Trajektorien der Strategie. Dabei multiplizieren wir den Ertrag jeder Trajektorie mit ihrem Score. Der Übergangskern fällt aus dem Score heraus. So lässt sich der Gradient ohne Modell aus gezogenen Episoden schätzen. REINFORCE gewichtet den Score jeder Aktion mit dem Ertrag ihrer Episode. Das entspricht der Aktualisierung beim Verhaltensklonen mit Erträgen anstelle der Einheitsgewichte. Bei der Aufgabe mit Prüfprogramm und auf griffigem Eis stimmt die Schätzung im Mittel mit dem exakten Gradienten überein. Auf dem See weicht ein kleiner Batch jedoch um ein Mehrfaches der Gradientenlänge ab, und der Fehler sinkt nur wie $1/\sqrt n$. Nach dem Policy-Gradient-Theorem tragen Belohnungen aus der Zeit vor einer Aktion nur Rauschen bei. Jede Aktualisierung erfordert einen neuen Batch aus der aktuellen Strategie.

<!--en:2294243a-->
## Übungen

<!--en:87bbf6a5-->
1. **Parameterabhängige Dynamik.** Die Herleitung in :numref:`subsec_pg-cancellation` setzt voraus, dass die Übergangswahrscheinlichkeiten nicht von $\theta$ abhängen. Ein Empfehlungssystem verletzt diese Annahme, wenn Nutzer einen Artikel umso seltener anklicken, je öfter sie ihn sehen.
    1. Die Dynamik sei $P_\theta(s_{t+1} \mid s_t, a_t)$. Leiten Sie $\nabla_\theta \log P(\tau; \theta)$ erneut her und geben Sie den Term an, der nicht mehr verschwindet. Was müsste der Agent wissen, um ihn zu berechnen?
    1. In einem einstufigen Modell zeigt das Empfehlungssystem mit Wahrscheinlichkeit $\pi_\theta(1) = \sigma(\theta)$ Artikel 1 und sonst Artikel 0. Dabei ist $\sigma$ die logistische Funktion. Ein angezeigter Artikel 1 wird mit Wahrscheinlichkeit $1 - \sigma(\theta)$ angeklickt, Artikel 0 nie. Ein Klick bringt die Belohnung $1$. Berechnen Sie $J(\theta)$ und $\mathrm{d}J/\mathrm{d}\theta$. Vergleichen Sie die Ableitung mit dem Erwartungswert der REINFORCE-Schätzung :eqref:`eq_reinforce`, die den zusätzlichen Term weglässt. Wohin konvergiert der einfache Gradientenaufstieg in den beiden Fällen?
   :exlabel:`ex_policy-gradient-parameter-dependent-dynamics`

<!--en:a1353787-->
1. [code] **Seltene Belohnungen.** Im einfachsten Fall aus dem Text gibt es einen Zustand und zwei Aktionen, die Belohnung $1$ für Aktion 1 und $0$ für Aktion 2, und es gilt $p = \pi_\theta(1)$. Bei einer Schätzung aus einer einzigen Ziehung beträgt die Standardabweichung das $\sqrt{(1-p)/p}$-Fache des Mittelwerts. Auf dem See ist die entsprechende Größe die Wahrscheinlichkeit $p_0 \approx 0.014$, mit der eine Episode der Gleichverteilungsstrategie das Ziel erreicht. In :numref:`subsec_vi_slippery_ice` berechnen wir sie durch Strategiebewertung bei $\gamma = 1$.
    1. Ein Batch ohne Erfolg liefert die Schätzung null, und `policy_step` lässt die Tabelle aus lauter Nullen dann unverändert. Bis zum ersten Erfolg bleibt die Strategie also gleichverteilt. Welcher Verteilung folgt die Anzahl der Aktualisierungen bis einschließlich der ersten Schätzung ungleich null? Wie groß ist ihr Mittelwert für Batches aus 16 Episoden? Zeichnen Sie diese Anzahl für jeden der zwölf Trainingsläufe auf und vergleichen Sie sie mit der Vorhersage und dem flachen Anfang der Trainingskurve.
    1. Zeigen Sie, dass der Mittelwert aus $n$ unabhängigen Einzelschätzungen eine Standardabweichung hat, die dem $\sqrt{(1-p)/(np)}$-Fachen seines Erwartungswerts entspricht. Bei wie vielen Episoden pro Batch wären Standardabweichung und Mittelwert für $p = p_0$ gleich? Wie ändert sich die Antwort, wenn das Training die Erfolgswahrscheinlichkeit erhöht?
   :exlabel:`ex_policy-gradient-sparse-rewards`

<!--en:294f3897-->
1. [code] **Das Theorem in einer Tabelle.** ● Bezeichnen Sie für die tabellarische Softmax-Strategie mit $\rho_\gamma(s) = \sum_{t \geq 0} \gamma^t d^{\pi_\theta}_t(s)$ die diskontierte Anzahl der Besuche in $s$, sodass $d^{\pi_\theta}_\gamma = (1 - \gamma)\rho_\gamma$ in :eqref:`eq_pg_theorem` gilt.
    1. Kombinieren Sie die Darstellung über die Zustandsbelegung aus :eqref:`eq_pg_theorem` mit :eqref:`eq_softmax_score` und zeigen Sie, dass
       $$\frac{\partial J}{\partial \theta_{s,a'}} = \rho_\gamma(s)\, \pi_\theta(a' \mid s) \big( Q^{\pi_\theta}(s,a') - V^{\pi_\theta}(s) \big).$$
    1. Nach Aufgabe :exref:`ex_value-iter-linear-system`b aus :numref:`sec_valueiter` ist $\rho_\gamma^\top$ diejenige Zeile von $(I - \gamma P_{\pi_\theta})^{-1}$, die zu $s_0$ gehört. Berechnen Sie $\rho_\gamma$, $V^{\pi_\theta}$ und $Q^{\pi_\theta}$ an den Parametern von `probe`, indem Sie lineare Gleichungssysteme mit `mdp.P` und `mdp.r` lösen. Setzen Sie die Größen in die Formel ein und vergleichen Sie das Ergebnis mit `g_exact`.
    1. Auf rutschigem Eis bewegt ein Befehl $a$ den Agenten in Richtung $a$ oder in eine der beiden dazu senkrechten Richtungen, jeweils mit Wahrscheinlichkeit $1/3$. Zeigen Sie, dass beide Karten bei der Gleichverteilungsstrategie dieselbe Kette induzieren und somit dasselbe $\rho_\gamma$ und dasselbe $V$ ergeben. Zeigen Sie außerdem, dass für den Aktionswert $Q(s, a)$ auf rutschigem Eis die Beziehung $Q_\textrm{slip}(s,a) - V(s) =
       -\tfrac{1}{3}\big(Q_\textrm{calm}(s, \bar a) - V(s)\big)$ gilt. Dabei ist $\bar a$ der zu $a$ entgegengesetzte Befehl. Folgern Sie, dass der exakte Gradient bei $\theta = 0$ auf rutschigem Eis nur ein Drittel so lang ist. Prüfen Sie das mit `policy_objective` für beide Übergangstabellen.
   :exlabel:`ex_policy-gradient-theorem-table`

<!--en:695b3000-->
1. [code] **Wiederverwendung eines Batchs.** ● Ändern Sie `train_reinforce_tabular` so, dass es `policy_step` für jeden Batch ohne Korrektur $k$-mal aufruft, mit $k \in \{1, 2, 5, 20\}$. Bezeichnen Sie mit $\theta_\textrm{old}$ die Parameter, mit denen der Batch gesammelt wurde.
    1. Protokollieren Sie auf griffigem Eis den mittleren Ertrag des Batchs und nach dem $k$-ten Schritt den Mittelwert und das Maximum des Verhältnisses $\pi_\theta(a_t \mid s_t) / \pi_{\theta_\textrm{old}}(a_t \mid s_t)$ über den Batch. Mit `log_prob_np` lassen sich beide Strategien auswerten. Wie viele Umgebungsschritte braucht jedes $k$, bis der mittlere Ertrag des Batchs $0.7$ erreicht? Wie groß wird das Verhältnis?
    1. Mit wachsendem $k$ nähern sich die Schritte einem Maximierer der Ersatzzielfunktion $\sum_t w_t \log \pi_\theta(a_t \mid s_t)$ auf dem festen Batch. Dabei ist $w_t$ der Ertrag der Episode, die Schritt $t$ enthält. Bestimmen Sie die maximierende tabellarische Strategie in geschlossener Form und erklären Sie damit das Ergebnis aus Teil a.
    1. Vergleichen Sie $k = 1$ mit $k = 20$ auf der Karte mit rutschigem Eis (`is_slippery=True`) anhand des Endwerts von $J(\theta)$. Berechnen Sie ihn mit :eqref:`eq_pg_exact_objective` aus der Übergangstabelle für rutschiges Eis. Jeder Lauf dauert nur Sekunden. Erklären Sie den Unterschied zum griffigen Eis mit der geschlossenen Form aus Teil b.
   :exlabel:`ex_policy-gradient-breaking-policy-rule-purpose`

<!--en:1c0a1669-->
1. [code] **Prüfung auf Erwartungstreue.** ● Der Test mit fünfzig Ziehungen aus dem Text kann keine Verzerrung nachweisen, die kleiner ist als das Rauschen dieser fünfzig Ziehungen. Ein empfindlicherer Test mittelt über $M$ Schätzungen. Bei einem erwartungstreuen Schätzer sinkt der Fehler des Mittelwerts mit wachsendem $M$ weiter. Bei einem verzerrten Schätzer nähert er sich der Verzerrung an.
    1. Ziehen Sie an den Parametern von `probe` insgesamt $M = 25{,}600$ Schätzungen mit `reinforce_estimate` bei $n = 1$. Tragen Sie den Fehler des fortlaufenden Mittelwerts relativ zu $\|$`g_exact`$\|$ gegen $M$ auf. Verwenden Sie logarithmische Achsen. Erklären Sie die Steigung der Kurve.
    1. Wiederholen Sie die Messung mit einer neu erzeugten Umgebung, in der Sie `max_episode_steps` auf $20$ bzw. auf $10$ senken. Übergeben Sie diese Umgebung an `reinforce_estimate`, sodass `rollout` einige Episoden vorzeitig abbricht. Auf welchem Niveau pendelt sich jede Kurve ein? Für welche Zielfunktion liefern die abgebrochenen Episoden erwartungstreue Schätzungen des Gradienten?
   :exlabel:`ex_policy-gradient-testing-unbiasedness`

<!--en:3ea06591-->
1. **Softmax-Plateaus.** ● Betrachten Sie einen einzigen Zustand mit zwei Aktionen, den Belohnungen $1$ und $0$ und den anfänglichen Präferenzen $\theta = (0, L)$ mit $L > 0$. Die bessere Aktion hat also anfangs die Wahrscheinlichkeit $1/(1 + e^L)$. Das ist die einfachste Form des Plateaus aus :numref:`subsec_pg-nonconcavity`.
    1. Zeigen Sie, dass $J$ von $\theta$ nur über $x = \theta_1 - \theta_2$ abhängt, denn es gilt $J = \sigma(x)$ mit der logistischen Funktion $\sigma$. Folgern Sie, dass $J$ nicht konkav ist.
    1. Zeigen Sie, dass ein Schritt des einfachen Gradientenaufstiegs mit Schrittweite $\eta$ die Differenz $x$ um $2\eta\, \sigma(x)(1 - \sigma(x))$ erhöht. Fassen Sie dies als Differentialgleichung auf und zeigen Sie, dass das Erreichen von $\pi_\theta(1) = 1/2$ ausgehend von $x = -L$ etwa $e^L / (2\eta)$ Schritte erfordert, wenn $L$ groß ist.
    1. Mit denselben exakten Gradienten verschiebt Adam jede Koordinate pro Schritt ungefähr um seine Lernrate $\eta$, solange das Vorzeichen des jeweiligen Gradienten gleich bleibt. Schätzen Sie die benötigte Schrittzahl und vergleichen Sie sie bei $L = 10$ und $\eta = 0.1$ mit der Antwort aus Teil b. Was legt der Vergleich über den Einsatz von Adam in `Policy.tabular` nahe?
    1. [code] Führen Sie `train_reinforce_tabular` mit mehreren Zufallsstartwerten für jeweils 1024 Aktualisierungen auf der Karte mit rutschigem Eis aus und berechnen Sie den Endwert von $J(\theta)$ mit :eqref:`eq_pg_exact_objective`. Betrachten Sie einen Lauf, der deutlich unter $V^*(s_0)$ endet. In welchen Zuständen wählt die Strategie am Ende nahezu deterministisch welche Aktionen? Berechnen Sie $\|\nabla_\theta J\|$ dort und zu Beginn bei der Gleichverteilungsstrategie. Ist der Endwert von $\theta$ ein stationärer Punkt? Führen Sie ausgehend von diesem Punkt und von $\theta = 0$ einen exakten Gradientenaufstieg durch. Setzen Sie Ihre Beobachtungen zu den Ergebnissen aus Teil b und c in Beziehung.
   :exlabel:`ex_policy-gradient-softmax-plateaus`

<!--en:04ec64f2-->
1. [code] **Der weggelassene Diskontierungsfaktor.** ●● Viele Implementierungen lassen den Faktor $\gamma^t$ in der Trajektorienform von :eqref:`eq_pg_theorem` weg. Dadurch ergibt sich das Feld der Aktualisierungsrichtungen

   $$F(\theta) = \mathbb{E}_{\tau \sim P(\cdot;\, \theta)} \Big[ \sum_{t \geq 0} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\ Q^{\pi_\theta}(s_t, a_t) \Big].$$

   :citet:`Nota.Thomas.2020` beweisen, dass $F$ nicht der Gradient einer Funktion sein muss. Sie konstruieren einen MDP, in dem exakte Aktualisierungen entlang von $F$ die Strategie in Richtung der schlechtestmöglichen Strategie treiben. Prüfen Sie in den folgenden Teilaufgaben die erste Aussage auf FrozenLake. Konstruieren Sie dann einen kleinen MDP, in dem ein Schritt entlang von $F$ den Wert von $J$ senkt. Ein solcher Schritt ist der Ausgangspunkt der zweiten Aussage.
    1. Zeigen Sie analog zu Aufgabe :exref:`ex_policy-gradient-theorem-table`a, dass

       $$F_{s,a'}(\theta) = \rho_1(s)\, \pi_\theta(a' \mid s) \big( Q^{\pi_\theta}(s,a') - V^{\pi_\theta}(s) \big),$$

       wobei $\rho_1(s) = \sum_{t \geq 0} \Pr(s_t = s,\ t < T)$ die erwartete Anzahl der Besuche in $s$ vor dem Ende der Episode ist. Berechnen Sie $\rho_1$ durch Lösen eines linearen Gleichungssystems. Setzen Sie dazu in der Übergangsmatrix der Strategie die Zeilen der Endzustände auf null.
    1. Die Jacobi-Matrix eines stetig differenzierbaren Gradientenfelds ist eine Hesse-Matrix und damit symmetrisch. Berechnen Sie die Jacobi-Matrix $DF$ an der Nulltabelle und an den Parametern von `probe` durch automatisches Differenzieren. Geben Sie $\|DF - DF^\top\| / \|DF\|$ in der Frobeniusnorm an. Prüfen Sie zur Kontrolle auf dieselbe Weise $\nabla_\theta J$.
    1. Messen Sie den Kosinus des Winkels zwischen $F$ und $\nabla_\theta J$ an denselben beiden Punkten. Nennen Sie einen praktischen Grund, warum Implementierungen dennoch $F$ verwenden.
    1. Zeigen Sie, dass $F(\theta)^\top \nabla_\theta J(\theta) \geq 0$ für die tabellarische Softmax-Strategie bei jedem $\theta$ gilt. Ein kleiner Schritt entlang von $F$ kann also auf FrozenLake den Wert von $J$ nie senken. Konstruieren Sie nun einen kleinen episodischen MDP, in dem ein einziger Parameter die Strategie in zwei nacheinander besuchten Zuständen steuert. Geben Sie außerdem ein $\theta$ an, für das $F(\theta)\, \mathrm{d}J/\mathrm{d}\theta < 0$ gilt.
   :exlabel:`ex_policy-gradient-discount-that-implementations-drop`

<!--en:d3d8dcd9-->
[Diskussionen](https://d2l.discourse.group/t/36798)

<!--en:64a6a941-->
Policy-Gradient und REINFORCE<br>**Softmax-Präferenzen und Score-Funktion · Trick mit der logarithmischen Ableitung · Wegfall des Übergangskerns · REINFORCE · Policy-Gradient-Theorem · On-Policy-Varianz**

<!--en:2e41d95c-->
::: {.slide title="Eine Strategie direkt aus Episodenerträgen lernen"}

<!--en:b0445376-->
[Motivation und Grundidee]{.kicker}

<!--en:423fc5cc-->
- **Von Gewichten beim Klonen zu Ertragsgewichten:** Beim Verhaltensklonen erhöhten wir $\frac{1}{N}\sum_t w_t \log \pi_\theta(a_t \mid s_t)$ mit $w_t = 1$ anhand von Expertendaten. REINFORCE erzeugt selbst Episoden mit $\pi_\theta$ und setzt $w_t = R(\tau)$.

<!--en:74d22bcc-->
- **Die Einsicht von Williams (1992):** Gewichten wir den Score $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ jeder gezogenen Aktion mit $R(\tau)$, erhalten wir einen erwartungstreuen stochastischen Gradienten von $J(\theta) = \mathbb{E}[R(\tau)]$.

<!--en:81b6237c-->
::: {.slide title="Softmax-Präferenzen und Score-Funktion"}

<!--en:8507060c-->
[Differenzieren einer stochastischen Strategie]{.kicker}

<!--en:e2aebfbd-->
Eine deterministische $\mathrm{argmax}$-Strategie hat fast überall den Gradienten null und ändert sich sprunghaft. Policy-Gradient-Verfahren parametrisieren daher eine glatte stochastische Strategie $\pi_\theta(a \mid s)$:

<!--en:99fa7700-->
- **Der Score-Vektor $\nabla_\theta \log \pi_\theta(a \mid s)$:** Im Zustand $s$ lautet der Gradient $\mathbf{e}_a - \pi_\theta(\cdot \mid s)$. Er hebt die Präferenz der gewählten Aktion um $1 - \pi_\theta(a \mid s)$ an und senkt die Präferenz jeder anderen Aktion proportional zu ihrer aktuellen Wahrscheinlichkeit.

<!--en:5be8ce5e-->
- **Zeilensumme null:** Wegen $\sum_{a'} (\mathbf{1}(a' = a) - \pi_\theta(a' \mid s)) = 0$ ist der Score orthogonal zu konstanten Verschiebungen einer Zeile.

<!--en:fd7a3d0e-->
::: {.slide title="Der einfachste Fall: Seltene Belohnungen führen zu hoher Varianz"}

<!--en:627804ee-->
[Verzerrung und Varianz eines Schätzers für einen Zustand]{.kicker}

<!--en:d061d047-->
- **Erwartungstreu, mit seltenen Ausschlägen:** Mit Wahrscheinlichkeit $p$ liefert Aktion $1$ den Schätzwert $1 - p$. Mit Wahrscheinlichkeit $1 - p$ liefert Aktion $2$ den Wert $0$.

<!--en:944a00b9-->
- **Relative Standardabweichung $\sqrt{(1 - p)/p}$:** Bei $p \approx 0.014$ auf FrozenLake ist das Rauschen einer einzelnen Episode $8.4\times$ so groß wie der wahre Gradient.

<!--en:b8602f4a-->
::: {.slide title="Trajektorienwahrscheinlichkeiten und der Trick mit der logarithmischen Ableitung"}

<!--en:deb34594-->
[Differenzieren eines Erwartungswerts]{.kicker}

<!--en:385fdd82-->
- **Der Trick mit der logarithmischen Ableitung ($\nabla_\theta P = P\,\nabla_\theta \log P$):** Multiplizieren und Dividieren mit bzw. durch $P(\tau; \theta)$ macht aus $\nabla_\theta \sum_\tau P(\tau; \theta)R(\tau)$ wieder einen Erwartungswert $\mathbb{E}_{\tau \sim P}[R(\tau)\,\nabla_\theta \log P(\tau; \theta)]$, den wir durch Ziehungen schätzen können.

<!--en:fe8778b4-->
::: {.slide title="Warum der unbekannte Übergangskern wegfällt"}

<!--en:e351748a-->
[Modellfreie Schätzung]{.kicker}

<!--en:b242f332-->
Der Logarithmus verwandelt das Produkt $P(\tau; \theta)$ für die Trajektorie in eine Summe, in der nur die Terme der Strategie von $\theta$ abhängen:

<!--en:de9d12e3-->
- **Modellfreie Gradientenschätzung:** Die Dynamik der Umgebung $P(s_{t+1} \mid s_t, a_t)$ bestimmt, welche Trajektorien wir ziehen. Ihre Ableitungen nach $\theta$ sind jedoch identisch null.

<!--en:2024699c-->
::: {.slide title="Der REINFORCE-Schätzer und seine geometrische Deutung"}

<!--en:d8cf329d-->
[Monte-Carlo-Schätzung des Policy-Gradienten]{.kicker}

<!--en:c44ab3eb-->
Mitteln wir $R(\tau)\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ über $n$ gezogene Trajektorien $\tau_1, \ldots, \tau_n$, erhalten wir den **REINFORCE**-Schätzer von Williams:

<!--en:f69a7b68-->
- **Einstufiges Beispiel mit Gauß-Verteilung:** Für $\pi_\theta(a) = \mathcal{N}(a; \mu, 1)$ lautet der Score $\partial_\mu \log \pi_\theta(a) = a - \mu$.

<!--en:088f8b7c-->
- **Zugkraft proportional zur Belohnung:** Jede gezogene Aktion $a_i$ zieht $\mu$ in Richtung $a_i$, proportional zu $R(a_i)$. Mit 12 Ziehungen verschiebt sich $\mu$ von $0$ auf $0.32$ (exakter Gradient $0.52$).

<!--en:15c2166b-->
::: {.slide title="REINFORCE bei der Aufgabe mit Prüfprogramm"}

<!--en:140a50c6-->
[Einstufiges Post-Training eines Sprachmodells]{.kicker}

<!--en:e34f39a5-->
Bei `VerifierTask` (4 Prompts, 9 Antworten) besteht eine Trajektorie aus einer einzigen Antwort $y \sim \pi_\theta(\cdot \mid x)$. Dabei mittelt `verifier_step` pro Prompt über $K$ mit der Belohnung gewichtete Scores $\mathbf{e}_y - \pi_\theta(\cdot \mid x)$:

<!--en:5c9eac0b-->
- **Austauschbare Gewichtung:** Mit `weighting = lambda r: r` erhalten wir gewöhnliches REINFORCE. In §{sec_baselines} machen innerhalb der Gruppe standardisierte Belohnungen aus genau derselben Funktion GRPO.

<!--en:7c97a577-->
::: {.slide title="Prüfung der Erwartungstreue an der Aufgabe mit Prüfprogramm"}

<!--en:b8503742-->
[Schätzungen aus Ziehungen und exakter Gradient aus neun Termen]{.kicker}

<!--en:d781f7f7-->
Der Vergleich von $10{,}000$ unabhängigen Gruppenschätzungen mit $K = 16$ und dem exakten Gradienten aus 9 Termen $\nabla_{\theta_x} J_x = \sum_y \pi_\theta(y \mid x)r(x, y)(\mathbf{e}_y - \pi_\theta(\cdot \mid x))$ bestätigt unsere theoretische Varianzformel:

<!--en:9ac66aa2-->
- **Exakte Übereinstimmung mit $\sqrt{(1-p)/(Kp)}$:** Bei $p \approx 1/8$ und $K = 16$ hat eine Gruppenschätzung den relativen RMS-Fehler $\sqrt{7 / 16} = 0.66$. Mitteln wir über $M$ Gruppen, sinkt der Fehler genau mit der Rate $0.66/\sqrt{M}$, ohne dass eine Verzerrung zurückbleibt.

<!--en:a534dee4-->
- **Gradient null für unlösbare Prompts:** Bei `8*8` und `9*9` ($r = 0$ für alle möglichen Antworten) liefern $100\%$ der Gruppen $\hat{u} = \mathbf{0}$.

<!--en:44ba988c-->
::: {.slide title="Implementierung von episodischem REINFORCE auf FrozenLake"}

<!--en:deaa8662-->
[Episoden in Batches und Optimierung des Ersatzverlusts]{.kicker}

<!--en:a86d8671-->
`train_reinforce_tabular` sammelt $n = 16$ vollständige Episoden in einem flachen `Batch`, wiederholt den Ertrag $R(\tau_i)$ jeder Episode mit `batch.per_step(R)` für alle ihre Schritte und ruft `d2l.policy_step` auf:

<!--en:4ba31ea3-->
- **Warum zuerst griffiges Eis?** Bei der Gleichverteilungsstrategie verringert rutschiges Eis den Einfluss jeder Aktion auf die Richtung auf den Faktor $1/3$. Dadurch schrumpft $\|\nabla_\theta J\|$ auf ein Drittel. Für dasselbe Signal-Rausch-Verhältnis brauchen wir $9\times$ so große Batches.

<!--en:127fa4ff-->
::: {.slide title="Trainingsverläufe und die erlernte Strategie auf dem Gitter" layout="tight"}

<!--en:b47186a7-->
[Von gleichverteilter Exploration zum deterministischen kürzesten Weg]{.kicker}

<!--en:4408ef86-->
Trainieren wir `Policy.tabular(16, 4)` mit 12 Zufallsstartwerten über 256 Aktualisierungen auf griffigem Eis in FrozenLake, zeigt sich die typische S-Kurve des Lernens mit Policy-Gradienten:

<!--en:bab91dd8-->
- **Das anfängliche Plateau verlassen:** Frühe Batches bleiben meist ohne Erfolg ($\hat{u} = \mathbf{0}$). Sobald ein Batch das Ziel erreicht, verstärkt Adam die 6 zugehörigen Aktionen und legt sich auf einen der kürzesten Wege mit 6 Zügen fest ($\gamma^5 \approx 0.774$).

<!--en:148f264f-->
::: {.slide title="Das Policy-Gradient-Theorem"}

<!--en:854a8a69-->
[Kausalität und diskontierte Zustandsbelegung]{.kicker}

<!--en:e6c17e94-->
- **Wegfall aufgrund der Kausalität:** Frühere Belohnungen $r_{<t}$ stehen bei $s_t$ bereits fest, während $\mathbb{E}_{a_t}[\nabla_\theta \log \pi_\theta(a_t \mid s_t)] = \mathbf{0}$ gilt.

<!--en:343ca8de-->
- **Gewichtung nach Zustandsbelegung:** Sie skaliert Aktualisierungen bei $s$ mit $d_\gamma^{\pi_\theta}(s) = (1-\gamma)\sum_{t \geq 0} \gamma^t \Pr(s_t = s)$.

<!--en:c1e56a51-->
::: {.slide title="Datenbedarf der On-Policy-Datenerhebung"}

<!--en:ffe7c45f-->
[Warum REINFORCE bei jedem Schritt neue Episoden braucht]{.kicker}

<!--en:c64ff3a7-->
Der Erwartungswert $\mathbb{E}_{\tau \sim P(\cdot; \theta)}$ bezieht sich auf die *aktuellen* Parameter $\theta$. Bereits nach einer einzigen Aktualisierung sind frühere Trajektorien daher veraltet:

<!--en:61eb1b37-->
- **Mehrbedarf an On-Policy-Daten:** Bis zu einem mittleren Batch-Ertrag von $0.70$ dauert es im Median $32$ Aktualisierungen ($4{,}313$ Schritte). Für 256 Aktualisierungen brauchen wir $\approx 27{,}000$ Schritte, gegenüber $\approx 3{,}500$ Schritten beim Off-Policy-Q-Learning auf demselben griffigen Eis.

<!--en:6476c0b5-->
::: {.slide title="Exakter Policy-Gradient durch differenzierbares Lösen eines linearen Gleichungssystems"}

<!--en:69cbf500-->
[Exakter Referenzwert auf FrozenLake]{.kicker}

<!--en:581a899e-->
Da $J(\theta) = V^{\pi_\theta}(s_0) = [(I - \gamma P_{\pi_\theta})^{-1}r_{\pi_\theta}]_{s_0}$ eine glatte rationale Funktion von $\theta$ ist, können wir durch `linalg.solve` differenzieren und erhalten den exakten Gradienten $\nabla_\theta J(\theta)$:

<!--en:0a770f6d-->
- **Festgehaltene Strategie:** Nach 16 Aktualisierungen mit Zufallsstartwert 3 erreicht die Strategie $J(\theta) = 0.313$, etwa auf halbem Weg zwischen dem Ertrag $0.008$ der Gleichverteilungsstrategie und der Obergrenze $0.774$. Sie dient uns als Bezugspunkt für die Messung des Schätzrauschens.

<!--en:d8a526b8-->
::: {.slide title="Varianz von REINFORCE im Vergleich zum exakten Gradienten von J"}

<!--en:0d4983c6-->
[Das Skalierungsgesetz 1/sqrt(n)]{.kicker}

<!--en:b436f8f8-->
Für die festgehaltene Strategie ziehen wir bei jeder Batchgröße $n \in \{4, 16, 64\}$ jeweils 50 unabhängige REINFORCE-Schätzungen. So messen wir, wie stark die direkte Gewichtung mit Erträgen rauscht:

<!--en:da96fc11-->
- **Fast vier Gradientenlängen Rauschen bei $n = 4$:** Eine Ziehung aus 4 Episoden verfehlt $\nabla_\theta J$ um **$3.85\times$ die Norm des Gradienten** (bei $3/50$ Batches ist die Schätzung identisch null). Vervierfachen wir $n$, halbiert sich der Fehler ungefähr ($3.85 \to 1.78 \to 0.96$). Der Fehler des Mittels aus 50 Ziehungen stimmt mit $\textrm{one}/\sqrt{50}$ überein, was das Fehlen einer Verzerrung bestätigt.

<!--en:98fdb748-->
::: {.slide title="Fehlende Konkavität, Softmax-Plateaus und vorzeitige Festlegung"}

<!--en:7507dbc1-->
[Optimierungslandschaft bei Policy-Gradienten]{.kicker}

<!--en:6671bdc4-->
Die Zielfunktion $J(\theta)$ ist in $\theta$ nicht konkav. Schon bei 1 Zustand und 2 Aktionen flacht die Sigmoidfunktion $J(\theta) = \sigma(\theta_1 - \theta_2)$ zu beiden Seiten ab. Dennoch hat der tabellarische Softmax-Policy-Gradient günstige globale Eigenschaften, mit Einschränkungen in der Praxis:

<!--en:52c16c49-->
- **Keine suboptimalen stationären Punkte (Agarwal et al., 2021; Mei et al., 2020):** Gilt überall $\mu_0(s) > 0$, konvergiert der exakte tabellarische Softmax-Gradientenaufstieg global gegen $\pi^*$.

<!--en:c9dd684f-->
- **Verschwindender Gradient bei seltenen optimalen Aktionen (Li et al., 2021):** Es gilt $\partial_{\theta_{s,a}} J \propto \pi_\theta(a \mid s)$. Hat die optimale Aktion $a^*$ zurzeit die winzige Wahrscheinlichkeit $\pi_\theta(a^* \mid s) \ll 1$, braucht einfacher SGD exponentiell viele Schritte über das Plateau. Die Normierung mit dem zweiten Moment bei Adam mildert das Problem.

<!--en:4f000344-->
- **Vorzeitige Festlegung durch Rauschen (Mei et al., 2021):** Große, verrauschte Schritte können $\pi_\theta(\cdot \mid s)$ nahezu deterministisch auf eine suboptimale Aktion festlegen, bevor genügend Beobachtungen vorliegen, um die Wahl umzukehren.

<!--en:120a0d51-->
::: {.slide title="Verständnisfragen: Policy-Gradienten und REINFORCE"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:bd583f31-->
Prüfen Sie Ihr Verständnis der Score-Funktion, des Wegfalls der Übergangsterme in der logarithmischen Ableitung und der Skalierung der Varianz:

<!--en:8e494dc7-->
1. **Eine glückliche Trajektorie:** Bei der Gleichverteilungsstrategie ($\pi_\theta(a \mid s) = 1/4$) und Batchgröße $n = 16$ enden 15 Episoden in Löchern ($R = 0$), und 1 Episode erreicht in 6 Schritten das Ziel ($R = \gamma^5 \approx 0.774$). Wie lautet der REINFORCE-Aktualisierungsvektor $\hat{u}$ in jedem der 6 besuchten Zustände?

<!--en:021e2b8c-->
2. **Modellfreiheit:** Wo gehen die Übergangswahrscheinlichkeiten $P(s_{t+1} \mid s_t, a_t)$ in $\nabla_\theta J(\theta)$ ein, wenn sie aus $\nabla_\theta \log P(\tau; \theta)$ verschwinden?

<!--en:2741072b-->
3. **Skalierung mit der Batchgröße:** Bei der festgehaltenen Strategie hat ein Batch aus $n = 4$ Episoden den relativen RMS-Fehler $3.85$. Wie viele Episoden pro Batch braucht man ungefähr, um den relativen Fehler eines einzelnen Batchs unter $0.50$ zu senken?

<!--en:557b98b6-->
**Antworten:**

<!--en:d09d7faa-->
1. Nur die erfolgreiche Episode trägt etwas bei. In jedem ihrer 6 besuchten Zustände gilt $\hat{u}_s = \frac{\gamma^5}{16}(\mathbf{e}_{a_t} - \frac{1}{4}\mathbf{1})$. Das ergibt $+0.0363$ für die ausgeführte Aktion und $-0.0121$ für jede der drei anderen Aktionen.

<!--en:e14650cc-->
2. $P(s_{t+1} \mid s_t, a_t)$ bestimmt die Verteilung $P(\tau; \theta)$ der gezogenen Trajektorien, auf die sich der Erwartungswert $\mathbb{E}_{\tau \sim P(\cdot; \theta)}[\cdots]$ bezieht. Der Simulator der Umgebung übernimmt die Ziehungen.

<!--en:e3fd949f-->
3. Da der RMS-Fehler wie $1/\sqrt{n}$ skaliert, brauchen wir $n \approx 4 \times (3.85 / 0.50)^2 \approx 4 \times 59.3 \approx 237$ Episoden pro Batch. Das motiviert die Varianzreduktion mit Baselines in §{sec_baselines}.

<!--en:42735f67-->
::: {.slide title="Zusammenfassung: Grundlagen des Policy-Gradienten"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:207d291d-->
| Begriff | Formel | Bedeutung für die Praxis |

<!--en:79184098-->
| **Softmax-Score-Funktion** | $\nabla_{\theta_s} \log \pi_\theta(a \mid s) = \mathbf{e}_a - \pi_\theta(\cdot \mid s)$ | Vektor mit Komponentensumme null, der die gewählte Aktion $a$ verstärkt und die anderen abschwächt |

<!--en:966f2914-->
| **Trick mit der logarithmischen Ableitung** | $\nabla_\theta P(\tau; \theta) = P(\tau; \theta)\,\nabla_\theta \log P(\tau; \theta)$ | Schreibt den Gradienten eines Erwartungswerts als Erwartungswert eines Scores |

<!--en:dabb4fd1-->
| **Wegfall des Übergangskerns** | $\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | Die unbekannte Dynamik $P(s' \mid s, a)$ fällt aus dem Gradientenschätzer heraus |

<!--en:c2aae91d-->
| **REINFORCE-Schätzer** | $\hat{u} = \frac{1}{n}\sum_{i=1}^n R(\tau_i)\sum_{t=0}^{T_i-1} \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i)$ | Erwartungstreue On-Policy-Monte-Carlo-Schätzung des Gradienten mit hoher Varianz (Fehlerskalierung $O(1/\sqrt{n})$) |

<!--en:1222f89c-->
| **Policy-Gradient-Theorem** | $\nabla_\theta J = \mathbb{E}_\tau[\sum_{t \geq 0} \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^{\pi_\theta}(s_t, a_t)]$ | Das Produkt früherer Belohnungen $r_{<t}$ mit dem Score in Schritt $t$ hat den Erwartungswert null |
