<!--en:ce329911-->
# Trust-Regionen und Proximal Policy Optimization

<!--en:25d8cdf3-->
Die Agenten aus :numref:`sec_deeprl` und :numref:`sec_actorcritic` sammeln acht Episoden, machen einen Gradientenschritt und verwerfen den Batch. Sobald der Agent den Stab gut im Gleichgewicht hält, liefern diese acht Episoden viertausend Schritte Erfahrung. All diese Erfahrung fließt in ein einziges Update ein. Zudem ist der Schritt unbeschränkt. Ein großes Update kann die Strategie (engl. policy) beliebig stark verändern. Den nächsten Batch sammelt der Agent dann mit der so entstandenen Strategie. Können wir mit einem Batch mehrere Schritte machen, und wie weit dürfen sie gehen?

<!--en:c14d7554-->
Die Antwort beginnt mit einer Eigenschaft der Parameter. Je nach Ausgangspunkt lässt dieselbe Parameteränderung das Verhalten einer Strategie fast unverändert oder verändert es vollständig. Die Schrittweite im Parameterraum kontrolliert also die entscheidende Größe nicht, nämlich die Verhaltensänderung. John Schulman und seine Kollegen setzten 2015 direkt an dieser Änderung an. Trust Region Policy Optimization (TRPO) begrenzt die Divergenz zwischen der alten und der neuen Aktionsverteilung in den besuchten Zuständen. Grundlage ist ein Satz darüber, welche Verbesserung ein solcher Schritt garantiert :cite:`Schulman.Levine.Abbeel.ea.2015`. Zwei Jahre später ersetzte Proximal Policy Optimization (PPO) die Nebenbedingung durch eine Zielfunktion mit Clipping, die sich mit jedem gradientenbasierten Optimierer optimieren lässt :cite:`Schulman.Wolski.Dhariwal.ea.2017`. PPO wurde zum Standardverfahren für Strategiegradienten, von simulierten Robotern bis zum Fine-Tuning von Sprachmodellen anhand menschlicher Rückmeldungen :cite:`ouyang2022training`.

<!--en:4bd21aa9-->
In diesem Abschnitt leiten wir beide Verfahren her. Eine Strategie mit zwei Aktionen zeigt, weshalb der Abstand im Parameterraum die falsche Größe ist. Durch einen Maßwechsel können wir eine neue Strategie anhand eines Batchs der alten Strategie exakt bewerten. Allerdings wächst die Varianz mit der Episodenlänge. Behalten wir pro Schritt nur ein Verhältnis bei, wird die Wiederverwendung praktikabel. Ein kurzes Lemma zeigt, was diese Näherung auslässt. TRPO begrenzt den Fehler, PPO setzt dafür Clipping ein. An CartPole messen wir dann, wie viele Umgebungsschritte die Wiederverwendung spart, was ohne Clipping schiefgeht und welche Kennzahlen beim Training einer Strategie aufschlussreich sind.

<!--en:3cf86ce2-->
`torch.set_num_threads(1)` verhindert, dass PyTorch die kleinen Matrixprodukte dieser Netze auf viele Kerne verteilt. Das würde die Berechnung nur verlangsamen.

<!--en:f0b46d65-->
## Parameterraum und Strategieraum

<!--en:68a69362-->
Da wir die Parameter der Strategie aktualisieren, stellt sich die Frage, wie ein Parameterupdate die Strategie selbst verändert. Eine Familie von Strategien mit einem einzigen Parameter $\theta$ und zwei Aktionen gibt darauf eine Antwort. Das Beispiel stammt aus Joshua Achiams Vorlesungen zur Strategieoptimierung :cite:`Achiam.2017`. Sei

<!--en:06fce596-->
Betrachten wir zwei gleich große Parameterupdates mit $\Delta\theta = 2$ von zwei verschiedenen Ausgangspunkten aus:

<!--en:41f12ca0-->
Vor dem ersten Update wählt der Agent beide Aktionen gleich häufig, danach bevorzugt er eine davon deutlich. Das zweite Update verändert das Verhalten für einen Beobachter nicht erkennbar. Die Steigung der Abbildung vom Parameter auf die Strategie erklärt den Unterschied. Sie beträgt nach $\sigma'(\theta)=\sigma(\theta)(1-\sigma(\theta))$ genau $0.25$ bei $\theta=0$ und etwa $0.0025$ bei $\theta=6$. :numref:`fig_rl-policy-vs-parameter` zeigt beide Updates.

<!--en:439d7b8b-->
![Zwei gleich große Parameterupdates mit $\Delta\theta = 2$. Links die Abbildung $\pi_\theta(a{=}1) = \sigma(\theta)$ mit beiden Updates und den Steigungen an ihren Ausgangspunkten. Rechts die beiden Aktionsverteilungen vor und nach dem jeweiligen Update.](../img/mdl-rl-policy-vs-parameter.svg)

<!--en:69ba2222-->
Dieser Unterschied macht das On-Policy-Training störanfällig. Ein großes, verrauschtes Update nahe $\theta=0$ kann die Strategie in die Sättigung treiben. Dort ist der Score der häufig gewählten Aktion fast null ($\nabla_\theta \log \pi_\theta(a{=}1) = 1 - \sigma(\theta) \approx 0.0025$ bei $\theta = 6$). Die andere Aktion hat zwar einen Score nahe $-1$, kommt aber nur einmal in vierhundert Schritten vor. Spätere On-Policy-Batches enthalten dann zu wenig Information, um das Update rückgängig zu machen. Eine Schranke für die euklidische Länge des Parameterschritts verhindert das nicht. Dieselbe Länge ist bei $\theta = 6$ harmlos und hat bei $\theta = 0$ drastische Folgen.

<!--en:228f8a52-->
Ein Update sollte daher den Abstand zwischen den Strategien begrenzen statt den zwischen den Parametern. Innerhalb dieser Grenze darf der Optimierer dort einen langen Parameterschritt machen, wo sich die Strategie wenig ändert. Wie wir noch sehen, darf er auch mehrere Schritte mit demselben Batch machen.

<!--en:75835b54-->
## Importance-Sampling

<!--en:e8f55151-->
### Der Maßwechsel

<!--en:cf9be377-->
Alle Schätzer seit :numref:`sec_policygradient` verwenden Trajektorien, die der Agent mit seiner aktuellen Strategie erzeugt. Deshalb war ein Batch nach einem Gradientenschritt unbrauchbar. Angenommen, der Batch stammt von einer früheren Strategie $\pi_{\theta_{\textrm{old}}}$. Können wir daran noch ablesen, wie gut eine aktualisierte Strategie $\pi_\theta$ abschneidet?

<!--en:3430e6e4-->
Importance-Sampling gibt eine exakte Antwort. Für jede Funktion $f$ von Trajektorien gilt

<!--en:c2e5ddd8-->
sofern jede Trajektorie, die die neue Strategie erzeugen kann, unter der alten eine positive Wahrscheinlichkeit hat. Softmax-Strategien erfüllen diese Bedingung, weil sie jeder Aktion eine Wahrscheinlichkeit größer als null zuweisen. Wählen wir $f = R$, also den Ertrag (engl. return), so lässt sich die Zielfunktion als Erwartungswert unter der *alten* Strategie schreiben:

<!--en:76377c36-->
Mit den Daten der alten Strategie können wir also den Ertrag der neuen ohne Verzerrung schätzen, indem wir jede Trajektorie mit ihrem Likelihood-Verhältnis gewichten. Wie in :numref:`sec_policygradient` ist die Wahrscheinlichkeit einer Trajektorie das Produkt aus der Wahrscheinlichkeit des Anfangszustands, den Aktionswahrscheinlichkeiten der Strategie und den Übergangswahrscheinlichkeiten der Umgebung. Im Verhältnis kürzen sich alle Faktoren bis auf die der Strategie heraus. Für das Gewicht brauchen wir daher kein Modell der Umgebung:

<!--en:f9631363-->
### Das explodierende Produkt

<!--en:1d382c4e-->
Der Schätzer ist erwartungstreu, seine Varianz wächst jedoch mit der Episodenlänge. Betrachten wir eine Strategie, die den Zustand ignoriert. Wir sammeln Daten mit $\pi_{\theta_{\textrm{old}}} = (0.5, 0.5)$ und bewerten die Strategie bei $\pi_\theta = (0.6, 0.4)$. Jeder Faktor in :eqref:`eq_traj_ratio` nimmt mit gleicher Wahrscheinlichkeit den Wert $1.2$ oder $0.8$ an. Sein Mittelwert ist also $1$, der Mittelwert seines Quadrats ist $(1.44 + 0.64)/2 = 1.04$. Die Faktoren verschiedener Schritte sind unabhängig. Daher hat das Trajektoriengewicht $W_T$ einer Episode mit $T$ Schritten den Mittelwert $1$ und die Varianz

<!--en:678133ff-->
Diese beträgt etwa $0.5$ für $T = 10$, etwa $50$ für $T = 100$ und etwa $3 \times 10^8$ für die $500$ Schritte von CartPole. Weicht eine Strategie bei jeder Entscheidung um zehn Prozentpunkte von der Strategie ab, mit der wir die Daten gesammelt haben, lässt sie sich anhand ganzer CartPole-Episoden also nicht bewerten. Eine Handvoll Trajektorien mit großen Gewichten bestimmt die Schätzung.

<!--en:d612b3c6-->
### Die Ersatzzielfunktion pro Schritt

<!--en:c1d6140a-->
Die Belohnung bei Schritt $t$ hängt nur von den Aktionen bis einschließlich Schritt $t$ ab. Für die exakte Korrektur dieser Belohnung brauchen wir daher nur die Verhältnisse $\rho_0 \cdots \rho_t$, wobei gilt:

<!--en:f159741f-->
Das Produkt $\rho_0 \cdots \rho_{t-1}$ der früheren Verhältnisse korrigiert die Zustandsverteilung. Es überführt die Verteilung von $s_t$ unter der alten Strategie in diejenige unter der neuen. Lassen wir das Produkt weg und behalten nur $\rho_t$ für die Aktion in $s_t$ bei, so erhalten wir ein Gewicht, dessen Varianz nicht mehr mit $t$ wächst. Mit dieser einen Näherung bewerten wir die Aktionen der neuen Strategie in den Zuständen der alten. Mit einer Vorteilsschätzung $\hat{A}_t^i$ für Schritt $t$ der Trajektorie $i$ ergibt sich eine *Ersatzzielfunktion* (engl. surrogate objective), die über alle $N = \sum_i T_i$ Schritte eines Batchs aus $n$ Trajektorien mittelt:

<!--en:501820c4-->
Der Code mittelt über Schritte statt über Trajektorien. Dadurch bezieht sich die folgende Nebenbedingung auf die mittlere Divergenz über die Zustände. Bei $\theta = \theta_{\textrm{old}}$ ist jedes Verhältnis gleich eins und es gilt $\nabla_\theta \rho_t = \nabla_\theta \log \pi_\theta(a_t \mid s_t)$. Der Gradient von $\hat{L}$ zeigt dort also in dieselbe Richtung wie der Schätzer aus :numref:`sec_baselines`. Jeder Score geht mit seinem Vorteil als Gewicht ein. Den Faktor $\gamma^t$ lassen wir wie in allen Implementierungen seither weg. Standardisieren wir die Vorteile auf Mittelwert null, gilt exakt $\hat{L}(\theta_{\textrm{old}}) = 0$. Mit wachsendem Abstand von $\theta_{\textrm{old}}$ wird die Ersatzzielfunktion ungenauer. Wie weit können wir ihr vertrauen?

<!--en:7422edf9-->
## Trust-Regionen

<!--en:4d13198d-->
### Das Lemma zur Ertragsdifferenz

<!--en:34b34a38-->
Die Ersatzzielfunktion bewertet die Aktionen der neuen Strategie in den Zuständen der alten Strategie. Um zu sehen, was sich dadurch ändert, vergleichen wir die beiden Erträge exakt. Mit $A^{\textrm{old}}(s,a)=Q^{\pi_{\theta_{\textrm{old}}}}(s,a)-V^{\pi_{\theta_{\textrm{old}}}}(s)$ bezeichnen wir den Vorteil unter der alten Strategie. Er gibt an, um wie viel die Aktion $a$ im Zustand $s$ besser ist als der Durchschnitt der alten Strategie, wenn der Agent anschließend der alten Strategie folgt. Das folgende Lemma gilt für Episoden, die in einem Endzustand enden, und für fortlaufende Aufgaben mit $\gamma < 1$. Wie in :numref:`sec_deeprl` verwenden unsere CartPole-Läufe an der Zeitgrenze von 500 Schritten den geschätzten Folgewert. Sie optimieren also die Zielfunktion für die fortlaufende Aufgabe.

<!--en:5b04c238-->
**Satz** (Lemma zur Ertragsdifferenz, :citet:`Kakade.Langford.2002`).

<!--en:bd822cb5-->
**Beweis.** Schreiben wir $V$ für $V^{\pi_{\theta_{\textrm{old}}}}$. Entlang jeder Trajektorie ist $\sum_t \gamma^t \big( r_t + \gamma V(s_{t+1}) - V(s_t) \big)$ eine Teleskopsumme. Die inneren $V$-Terme heben sich paarweise auf. Der letzte ist in einem Endzustand null oder geht für $t \to \infty$ gegen null, sofern $\gamma < 1$ gilt. Es bleibt $R(\tau) - V(s_0)$. Bilden wir nun die Erwartungswerte unter $\tau \sim P(\cdot;\, \theta)$. Bei gegebenem $(s_t, a_t)$ ist der Erwartungswert von $r_t + \gamma V(s_{t+1})$ über den Folgezustand nach der Bellman-Gleichung der alten Strategie gleich $Q^{\pi_{\theta_{\textrm{old}}}}(s_t, a_t)$. Der Term für Schritt $t$ wird also zu $\gamma^t\, \mathbb{E}\big[ A^{\textrm{old}}(s_t, a_t) \big]$. Auf der anderen Seite gilt $\mathbb{E}[R(\tau)] = J(\theta)$. Außerdem gilt $\mathbb{E}[V(s_0)] = J(\theta_{\textrm{old}})$, weil beide Strategien $s_0$ aus derselben Anfangsverteilung ziehen. $\blacksquare$

<!--en:f63c44e4-->
Nach dem Lemma ist die Verbesserung der neuen Strategie gleich der Summe der diskontierten Vorteile der alten Strategie entlang der Trajektorien der neuen. Gruppieren wir die Terme nach Zustand, können wir die Verbesserung exakt mit der Ersatzzielfunktion vergleichen. Sei $d^{\pi}_\gamma(s) = (1 - \gamma) \sum_{t \geq 0} \gamma^t \Pr(s_t = s)$ die diskontierte Verteilung der Zustandsbelegung aus :numref:`sec_policygradient`. Sie beschreibt die Verteilung der Zustände, die eine Strategie besucht, und gewichtet Schritt $t$ mit $\gamma^t$. Dann gilt

<!--en:48b5d6d8-->
Die Ersatzzielfunktion berechnet diese Summe mit der Zustandsbelegung $d^{\pi_{\theta_{\textrm{old}}}}_\gamma$ der alten Strategie anstelle der neuen. Dieser Austausch ist ihre einzige Fehlerquelle. Der Fehler hängt davon ab, wie stark sich die Zustandsverteilung verschiebt. Dafür ist entscheidend, wie stark sich die Strategie ändert, unabhängig vom Abstand der Parameter.

<!--en:ca147fd1-->
### Die Schranke für monotone Verbesserung

<!--en:2baeefc1-->
Ersetzen wir die Zustandsbelegung im letzten Ausdruck, erhalten wir die Ersatzzielfunktion für die Grundgesamtheit,

<!--en:38aa694e-->
deren Schätzung anhand der Stichprobe $\hat{L}$ ist, wenn wir den Faktor $\gamma^t$ weglassen. Da der Vorteil unter der alten Strategie in jedem Zustand im Mittel null ist, gilt $\bar{L}(\theta_{\textrm{old}}) = 0$. Wie weit darf sich $\theta$ ändern, bevor $\bar{L}$ die Verbesserung nicht mehr zuverlässig vorhersagt? Die Antwort gibt eine Schranke, die von der größten Änderung der Aktionsverteilung über alle Zustände abhängt.

<!--en:a20fa4d2-->
**Satz** (:citet:`Kakade.Langford.2002,Schulman.Levine.Abbeel.ea.2015`). *Für $\gamma < 1$ und $A_{\max} = \max_{s, a} \lvert A^{\textrm{old}}(s, a) \rvert$ gilt*

<!--en:0fc7cfe6-->
Die Ersatzzielfunktion stimmt bis auf einen Strafterm, der proportional zur größten Kullback-Leibler-Divergenz zwischen alter und neuer Aktionsverteilung über alle Zustände ist. In :exref:`ex_ppo-monotonic-bound` leiten wir die Schranke her, indem wir beide Strategien mit gekoppelten Aktionen parallel ausführen. Beide Seiten der Ungleichung haben den Wert $J(\theta_{\textrm{old}})$ bei $\theta = \theta_{\textrm{old}}$. Jeder Schritt, der die rechte Seite erhöht, erhöht also auch den tatsächlichen Ertrag. Maximieren wir die untere Schranke, verbessert sich der Ertrag monoton. Für die direkte Anwendung ist die Konstante zu groß. Bei $\gamma = 0.99$ beträgt sie das $4 \cdot 0.99 / 0.01^2 \approx 40{,}000$-Fache von $A_{\max}$. Deshalb garantiert die Schranke nur für winzige Schritte eine Verbesserung.

<!--en:0d569316-->
TRPO behält die Form der Schranke bei und ersetzt ihre Bestandteile durch Größen, die sich aus einem Batch schätzen lassen. Es maximiert die geschätzte Ersatzzielfunktion unter einer Nebenbedingung, die ein Budget $\Delta_{\textrm{KL}}$ für die mittlere Divergenz über die Zustände des Batchs vorgibt:

<!--en:eb80cf46-->
Dabei tritt an die Stelle des Maximums über alle Zustände der Mittelwert und an die Stelle des konservativen Koeffizienten ein gewähltes Budget, meist $\Delta_{\textrm{KL}} = 0.01$. TRPO löst dieses Problem mit einem Verfahren zweiter Ordnung, das Sie in :exref:`ex_ppo-trpo` aus den Bausteinen dieses Abschnitts zusammensetzen. :numref:`fig_rl_trust_region` zeigt, warum die Nebenbedingung im Strategieraum formuliert ist.

<!--en:31207e00-->
![Begrenzung eines Strategieupdates. Links berührt die Ersatzzielfunktion $L$ die tatsächliche Zielfunktion $J$ bei $\theta_{\textrm{old}}$ tangential, wird aber nach einer großen Strategieänderung ungenau. Der unbeschränkte Maximierer von $L$ senkt $J$ von $0.82$ auf $-0.38$. Der beste Punkt in der schattierten Trust-Region $D_{\textrm{KL}} \leq \Delta_{\textrm{KL}}$ (bei einem Parameter ein Intervall $\theta_{\textrm{old}} \pm \delta$) erhöht den Wert dagegen auf $1.52$. Rechts bildet die Näherung zweiter Ordnung $\frac{1}{2}\Delta\theta^\top F \Delta\theta \leq 0.02$ für die Divergenzbeschränkung einer Softmax-Strategie mit drei Aktionen eine Ellipse im Parameterraum. Dabei ist $F$ die Hesse-Matrix der Divergenz bei $\theta_{\textrm{old}}$. Zwei Parameterschritte mit gleicher euklidischer Länge ändern die Strategie mit Divergenzen von $0.009$ und $0.062$. Bei gleichem Parameterabstand ändert sich die Strategie in der einen Richtung also etwa siebenmal so stark wie in der anderen.](../img/mdl-rl-trust-region.svg)

<!--en:95644032-->
### Die Zielfunktion mit Clipping

<!--en:878c8786-->
PPO ersetzt das Problem zweiter Ordnung mit Nebenbedingung durch eine Zielfunktion mit eingebauter Begrenzung für ein Verfahren erster Ordnung. Die Verhältnisse selbst bleiben unbeschränkt. Sobald das Verhältnis einer Beobachtung ein Band mit Halbbreite $\epsilon_{\textrm{clip}}$ um eins in günstiger Richtung verlässt, steigt ihr Beitrag nicht weiter:

<!--en:53fcda75-->
Betrachten wir eine einzelne Beobachtung. Für $\hat{A}_t > 0$ wächst ihr Beitrag mit $\rho_t$ nur bis $\rho_t = 1+\epsilon_{\textrm{clip}}$. Darüber ist der begrenzte Term der kleinere, und sein Gradient ist null. Für $\hat{A}_t < 0$ liegt die Schwelle bei $1-\epsilon_{\textrm{clip}}$. Nehmen wir den üblichen Wert $\epsilon_{\textrm{clip}} = 0.2$ und eine Aktion, welche die alte Strategie mit Wahrscheinlichkeit $0.5$ wählte. Bei positivem Vorteil lohnt sich das Update für diese Beobachtung nur so lange, bis die neue Strategie der Aktion die Wahrscheinlichkeit $0.6$ zuweist. Bei negativem Vorteil endet der Anreiz, sobald die Wahrscheinlichkeit auf $0.4$ fällt. Durch das Minimum wirkt die Grenze einseitig. Eine Änderung, die den Zielfunktionswert senkt, zählt immer, während eine weitere Änderung in günstiger Richtung nichts mehr beiträgt. :numref:`fig_rl_ppo_clip` zeigt den Beitrag einer Beobachtung für beide Vorzeichen des Vorteils.

<!--en:743066ec-->
![Beitrag einer Beobachtung zur Zielfunktion mit Clipping :eqref:`eq_ppo_clip` in Abhängigkeit vom Verhältnis $\rho_t$. Bei positivem Vorteil wächst der Beitrag oberhalb von $1+\epsilon_{\textrm{clip}}$ nicht weiter, bei negativem Vorteil unterhalb von $1-\epsilon_{\textrm{clip}}$ nicht weiter. Auf der jeweils anderen Seite von eins wählt das Minimum den unbegrenzten Term. Eine Änderung, die den Zielfunktionswert senkt, bleibt also immer sichtbar.](../img/mdl-rl-ppo-clip.svg)

<!--en:e2ff6f49-->
Da das Verhältnis selbst unbeschränkt bleibt, kann es das Band aus zwei Gründen verlassen. Alle Beobachtungen teilen sich die Parameter des Netzes. Ein Schritt für eine Beobachtung verändert also auch die Verhältnisse der anderen. Außerdem bewegt das Momentum von Adam die Parameter noch einige Schritte weiter, nachdem der Gradient einer Beobachtung null geworden ist. Die Messungen von :citet:`Engstrom.Ilyas.Santurkar.ea.2020` zeigen, dass die Verhältnisse bei PPO das Band tatsächlich verlassen und die Divergenz pro Update kein festes Budget einhält. Das Clipping setzt also einen Anreiz und ist keine Nebenbedingung. Anders als :eqref:`eq_trpo_bound` garantiert es keine Verbesserung. Es erlaubt jedoch mehrere Optimierungsdurchläufe mit einem Batch, ohne immer größere Änderungen einzelner Aktionswahrscheinlichkeiten zu belohnen.

<!--en:726a1fb3-->
### Der Entropiebonus

<!--en:a43e13bd-->
Implementierungen addieren häufig einen kleinen *Entropiebonus* (engl. entropy bonus). Dazu multiplizieren sie ein Gewicht `entropy_coef` mit der mittleren Entropie der Aktionsverteilungen in den Zuständen des Batchs. Der Bonus hält die Wahrscheinlichkeiten von null und eins fern. In der Nähe dieser Werte hat die häufig gewählte Aktion wie im Beispiel mit zwei Aktionen einen kleinen Score, und die Strategie erholt sich nur langsam. Die Entropie einer Verteilung über $K$ Aktionen ist gleich $\log K$ abzüglich ihrer Divergenz von der Gleichverteilung. Der Bonus bestraft also den Abstand zur Gleichverteilungsstrategie.

<!--en:aecbe0d2-->
## PPO in der Praxis

<!--en:36893fe2-->
### Die Vorteilsschätzung

<!--en:ccf059a6-->
Für $\hat{A}$ in :eqref:`eq_ppo_clip` können wir jede Vorteilsschätzung einsetzen. Wir verwenden die verallgemeinerte Vorteilsschätzung (engl. generalized advantage estimation, GAE) aus :numref:`sec_actorcritic`. Dazu summiert `Batch.gae` die TD-Fehler $\delta_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$ des Kritikers rückwärts entlang jeder Episode mit den Gewichten $(\gamma\lambda)^k$. Dabei bestimmt $\lambda$, wie weit jeder Vorteil in die Zukunft blickt. In jenem Abschnitt war der Schätzfehler im Bereich von $\lambda = 0.9$ bis $0.95$ am kleinsten. Auch die PPO-Arbeit verwendet $0.95$. Wir berechnen die Vorteile mit dem Kritiker, bevor wir ihn an den Batch anpassen. Das entspricht der Reihenfolge in der GAE-Arbeit :cite:`Schulman.Moritz.Levine.ea.2016`. Die Funktion `td_weight` aus :numref:`sec_actorcritic` verwendete die umgekehrte Reihenfolge, weil das Gewicht des Akteurs dort der TD-Fehler für einen Schritt war und damit ganz vom Kritiker abhing. Bei $\lambda = 0.95$ stammt der Vorteil überwiegend aus den beobachteten Belohnungen des Batchs und hängt nur wenig vom Kritiker ab. Die Reihenfolge ändert daher keines der Ergebnisse in diesem Abschnitt, und wir folgen der Originalarbeit.

<!--en:fc2afe82-->
### Implementierung

<!--en:0502771f-->
Wir behalten den CartPole-Aufbau aus :numref:`sec_actorcritic` bei. Als Agent dient `ActorCritic.mlp(4, 2)` aus :numref:`sec_deeprl`. Er besteht aus einem Strategienetz und einem Wertnetz mit jeweils einer versteckten Schicht aus 64 tanh-Einheiten und einem eigenen Adam-Optimierer mit Lernrate $10^{-2}$. Für jedes Update sammeln wir einen Batch von acht Episoden. Der Kritiker macht darauf `critic_steps = 20` Regressionsdurchläufe. Drei Hyperparameter sind neu: die Zahl `num_epochs = 20` der Durchläufe des Akteurs pro Batch, die Halbbreite `epsilon_clip = 0.2` des Bands und das Entropiegewicht `entropy_coef = 0.01`. Auch der RL Baselines3 Zoo verwendet für CartPole zwanzig Durchläufe :cite:`Raffin.Hill.Gleave.ea.2021`. Seine Lernrate beträgt jedoch $10^{-3}$ und ist damit zehnmal kleiner als unsere. Bei unserer Lernrate verändert jeder Durchlauf die Strategie stark. Dadurch geht das Training schnell, und das Clipping fällt ins Gewicht. Die folgende Ablationsstudie zeigt, was ohne Clipping geschieht.

<!--en:7d59fd5a-->
Wir übergeben die Zielfunktion für eine einzelne Beobachtung als Funktion an die Update-Routine. Dabei implementiert `clipped_surrogate` die Funktion aus :eqref:`eq_ppo_clip`. Für die folgende Ablationsstudie dient `plain_surrogate`, das nur den Term $\rho_t \hat{A}_t$ enthält. Beide erhalten die Halbbreite $\epsilon_{\textrm{clip}}$ des Bands, die nur die erste Funktion verwendet.

<!--en:2f629dc2-->
`ppo_epochs` erhält die Zustände und Aktionen eines Batchs sowie die beiden Größen, die wir vor seiner Wiederverwendung festhalten: die Vorteile und die Log-Wahrscheinlichkeiten `logp_old` der Strategie, mit der wir den Batch gesammelt haben. Die Funktion macht `num_epochs` Schritte mit dem Optimierer `opt_pi` des Akteurs. Dieser minimiert den negativen Mittelwert der Ersatzzielfunktion abzüglich des Entropiebonus. Für die Entropie brauchen wir in jedem Zustand die gesamte Verteilung. Deshalb berechnet die Funktion einmal die Log-Softmax-Funktion über alle Aktionen und liest daraus $\log\pi_\theta(a_t \mid s_t)$ ab. Vor jedem Durchlauf protokolliert sie drei Kennzahlen in einer Zeile: den Anteil der Verhältnisse außerhalb des Bands, den Stichprobenmittelwert von $\log\pi_{\theta_{\textrm{old}}}-\log\pi_\theta$ und die mittlere Entropie. Die zweite Kennzahl heißt *geschätzte KL-Divergenz* (engl. approximate KL). Ihr Erwartungswert unter der alten Strategie ist die Divergenz $D_{\textrm{KL}}(\pi_{\theta_{\textrm{old}}} \Vert \pi_\theta)$ in den besuchten Zuständen, obwohl die Schätzung bei einem endlichen Batch negativ ausfallen kann. Da die Funktion die Zeile vor dem Durchlauf protokolliert, enthält die erste Zeile stets null, null und die Entropie der alten Strategie.

<!--en:cbb82714-->
`ppo_epochs` füllt den Batch mit `d2l.padded` (:numref:`sec_utils`) auf die Länge seiner Gruppe auf. Die Hilfsfunktion liefert zugleich eine Maske für die echten Zeilen. `_ppo_step` lässt die aufgefüllten Zeilen bei jedem Mittelwert aus. So lässt sich derselbe kompilierte Schritt für Batches verschiedener Länge verwenden. Die Ersatzzielfunktion ist ein statisches Argument dieses Schritts. Daher wird er für jede der beiden Ersatzzielfunktionen einmal kompiliert. Die Kennzahlen bleiben bis zum Ende aller Durchläufe auf dem Gerät.

<!--en:7545b5cc-->
Wir verwenden die Akteur-Kritiker-Trainingsschleife aus :numref:`sec_actorcritic`. Den einzelnen Akteurschritt ersetzen wir durch `ppo_epochs` und berechnen die Vorteile vor den Durchläufen des Kritikers. Die Funktion `rollout` spielt acht Episoden und speichert sie in einem `Batch`. Dann berechnet die Schleife mit `Batch.gae` und dem aktuellen Kritiker die Vorteile und standardisiert sie mit `standardize` über den Batch. Außerdem ermittelt sie `logp_old` mit `log_prob_np`. Diese Funktion liefert $\log \pi(a \mid s)$ für die gegebenen Zustände und Aktionen als NumPy-Array. Als Nächstes macht der Kritiker seine Regressionsdurchläufe mit dem $\lambda$-Ertrag als Zielwert. Dazu verwendet er `Batch.lambda_return` aus :numref:`sec_actorcritic` und `fit_value` aus :numref:`sec_deeprl`. Zuletzt folgen die Durchläufe des Akteurs. Jedes Update liefert die Zahl der bisherigen Umgebungsschritte und den mittleren Ertrag des Batchs. Die Kennzahlen jedes Batchs kommen in eine vom Aufrufer übergebene Liste `diag`. Ihre drei Spalten heißen `CLIPPED`, `KL` und `ENTROPY`.

<!--en:41160d5f-->
### Wie viele Umgebungsschritte spart die Wiederverwendung?

<!--en:6d439b61-->
Wie viele Umgebungsschritte sparen wir durch die Wiederverwendung? Wir trainieren PPO mit Clipping mit einem, vier und zwanzig Durchläufen pro Batch und jeweils acht Zufallsstartwerten. Jeden Lauf beenden wir, sobald er 50 000 Umgebungsschritte verbraucht hat. Wir tragen den mittleren Ertrag jedes Batchs gegen die bisherige Schrittzahl auf. Bei einem einzigen Durchlauf ist jedes Verhältnis eins und das Clipping greift nie. Diese Variante ähnelt also der Versuchsreihe zu $\lambda$ aus :numref:`sec_actorcritic`. Hier verwenden wir einen Entropiebonus, begrenzen den Gradienten nicht und berechnen die Vorteile vor den Durchläufen des Kritikers. Wie dort bringt `d2l.on_grid` jeden Lauf auf ein gemeinsames Raster von Schrittzahlen. Für diese Läufe benötigen wir keine Kennzahlen und übergeben daher jeweils eine leere Liste.

<!--en:9fc59f93-->
Tragen wir die interpolierten Ertragskurven aus `reuse` gegen `grid` auf, sehen wir, wie zusätzliche Gradientendurchläufe mit demselben Batch Interaktionen mit der Umgebung einsparen:

<!--en:6af2adfb-->
Mit vier Epochen pro Batch erreichen wir den Höchstwert in etwa halb so vielen Umgebungsschritten wie mit einer Epoche, nach rund 25 000 statt 50 000 oder mehr Schritten. Auch die Zahl der Updates halbiert sich ungefähr. Innerhalb des Budgets von 50 000 Schritten bleibt der Median bei nur einer Epoche knapp unter dem Höchstwert. Der Wechsel auf zwanzig Epochen bringt vergleichsweise wenig, und kein Lauf mit Clipping bricht ein. Der Gewinn kommt aus dem Batch. Ein Batch mit einigen tausend Schritten enthält mehr Information, als ein Gradientenschritt nutzen kann. Mit Clipping kann der Optimierer mehrere Schritte machen, ohne einzelne Aktionswahrscheinlichkeiten zu weit zu verschieben.

<!--en:3dd76ef9-->
### Ablationsstudie zum Clipping

<!--en:fdb1b83a-->
Vier Durchläufe reichen für den größten Teil des Gewinns. Bei unserer Lernrate verändern zwanzig Durchläufe die Strategie innerhalb eines Batchs erheblich. Spielt das Clipping dabei eine Rolle? Wir trainieren PPO mit zwanzig Epochen und aktiviertem Clipping mit denselben acht Zufallsstartwerten. Die trainierten Agenten behalten wir für spätere Prüfungen, die vollständigen Kennzahlen für den nächsten Unterabschnitt:

<!--en:41a32367-->
Nun folgt der Kontrollversuch. Der Code unterscheidet sich nur darin, dass die Ersatzzielfunktion kein Clipping verwendet. Anschließend bringen wir die Kennzahlen beider Varianten in die Form (Zufallsstartwert, Update, Epoche, Spalte):

<!--en:c32d54b3-->
Zwei Kriterien kennzeichnen jeden Lauf. Ein Lauf *hält* das Niveau, wenn sein Batchertrag nach dem ersten Erreichen von 475 nie mehr unter 250 fällt, die Hälfte des Höchstwerts. Ein Lauf endet *gescheitert*, wenn der mittlere Ertrag seiner letzten zehn Updates unter 100 liegt:

<!--en:be8536fd-->
Die Batchertragskurven `r[:, :, 1]` über alle Updates zeigen das stabile Plateau von `'clipped (PPO)'` und den Einbruch von `'no clip'`:

<!--en:32c47d5b-->
Bei zwanzig Epochen und einer Lernrate von $10^{-2}$ erreicht jeder Lauf mit Clipping den Höchstwert und hält das Niveau. Ohne Clipping hält kaum ein Lauf das Niveau, und einige scheitern am Ende. In Versuchen mit 64 Zufallsstartwerten außerhalb des Notebooks hält etwa einer von zwanzig Läufen das Niveau, etwa die Hälfte scheitert. Die meisten gescheiterten Läufe enden bei einem Ertrag von ungefähr 9. So lange dauert eine Episode, wenn der Agent den Wagen bei jedem Schritt in dieselbe Richtung schiebt. Acht Zufallsstartwerte sind zu wenig, um diese Häufigkeiten zuverlässig wiederzugeben. Die oben ausgegebene Zahl gescheiterter Läufe kann deutlich unter der Hälfte liegen. Wiederholte Durchläufe ohne Clipping treiben die Strategie in die Sättigung. Wie im Beispiel mit zwei Aktionen ist dann der Gradient der Log-Wahrscheinlichkeit der gewählten Aktion klein. Die gesättigte Strategie sammelt die nächsten Batches, in denen die zur Umkehr nötigen Aktionen selten vorkommen. Die Ablationsstudie der PPO-Arbeit ergab dasselbe: Die Variante ohne Clipping und ohne Strafterm schnitt unter allen getesteten Varianten am schlechtesten ab :cite:`Schulman.Wolski.Dhariwal.ea.2017`. Erst die große Lernrate macht das Clipping nötig. Bei der zehnmal kleineren Lernrate $10^{-3}$ für den Akteur bricht PPO ohne Clipping bei keinem von zehn Zufallsstartwerten mehr ein und erreicht den Höchstwert ebenso schnell wie PPO mit Clipping. Bei $3 \times 10^{-4}$ greift das Clipping fast nie. Bei beiden Lernraten dauert das Training jedoch länger als bei unserer: etwa 35 000 bis 40 000 beziehungsweise 85 000 bis 90 000 Schritte bis zum Höchstwert gegenüber etwa 24 000 hier. Die große Lernrate macht das Training schnell, das Clipping macht sie sicher.

<!--en:c01734a1-->
Wie oft greift das Clipping? Bei jedem Update prüfen wir jede Beobachtung einmal pro Epoche. Die erste Epoche beginnt für jede Beobachtung mit $\rho_t=1$. Wir geben an, bei welchem Anteil aller Prüfungen über die Epochen das Verhältnis außerhalb des Bands liegt, und bestimmen denselben Anteil für die letzte Epoche jedes Batchs:

<!--en:6abcccf9-->
Bei Läufen mit Clipping liegt in beiden Fällen etwa eine von zwanzig Prüfungen außerhalb des Bands. Im Kontrollversuch ohne Clipping verlassen die Verhältnisse das Band mehrmals so häufig. In der letzten Epoche eines Batchs liegen sie häufiger außerhalb als im Durchschnitt, weil sie sich bei jedem Durchlauf ungehindert weiter entfernen können.

<!--en:b26fa1a6-->
### Kennzahlen für das Training

<!--en:3da49be1-->
In :numref:`sec_deeprl` haben wir gesehen, dass der Wert des Strategiegradienten-Verlusts nichts über den Fortschritt aussagt. Er bleibt nahe null, während der Ertrag von 20 auf 500 steigt. Aufschlussreich sind Messgrößen für das Update selbst. In `diag` liegen sie bereits für jeden Versuch, jeden Zufallsstartwert, jedes Update und jede Epoche vor. Zunächst mitteln wir über die Updates und betrachten die zwanzig Epochen innerhalb eines Batchs:

<!--en:dbdf75bd-->
In der ersten Epoche eines Batchs ist jedes Verhältnis eins. Der Anteil außerhalb des Bands ist daher genau null. Er steigt bis etwa zur fünften oder sechsten Epoche auf seinen Höchstwert und sinkt bis zur zwanzigsten wieder auf etwa die Hälfte. Der Rückgang ergibt sich aus der einseitigen Wirkung des Minimums. Verlässt das Verhältnis einer Beobachtung das Band auf der ungünstigen Seite, bleibt der Gradient ihres unbegrenzten Terms erhalten und zieht das Verhältnis zurück. Jenseits der günstigen Grenze ist der Gradient der Beobachtung hingegen null, und ihr Verhältnis bleibt, wo es ist. Die geschätzte KL-Divergenz verläuft ähnlich und ist etwa zehnmal kleiner. Die Codezelle gibt ihren Höchstwert aus. Ohne Clipping wächst die Divergenz mit jedem Durchlauf weiter. In der letzten Epoche eines Batchs ist sie typischerweise mehrere Dutzend Mal so groß wie im Lauf mit Clipping.

<!--en:262361be-->
Über die Updates hinweg zeigt die Entropie der Strategien mit Clipping, wie entschieden sie ihre Aktionen wählen:

<!--en:a69a026a-->
Die Entropie der Läufe mit Clipping fällt von etwa $0.65$ Nat auf etwa $0.25$ Nat. Die Strategie entscheidet sich beim Lernen immer eindeutiger. Der Entropiebonus verlangsamt diese Entwicklung, hält sie aber nicht auf. Ohne Clipping fällt die Entropie viel schneller. Diese Sättigung führt bei den gescheiterten Läufen zum Einbruch.

<!--en:6b2ba619-->
Die zusammengefassten Anteile zeigen nicht, wie sich die Verhältnisse verteilen. Wir nehmen zwei identische, untrainierte Agenten, denn zu Beginn des Trainings ist die Drift innerhalb eines Batchs am größten. Beide erhalten denselben Batch mit denselben festgehaltenen Vorteilen und Log-Wahrscheinlichkeiten. Jeder macht zwanzig Durchläufe, einer mit und der andere ohne Clipping. Nach jedem Durchlauf lesen wir alle Verhältnisse aus. Ein dritter identischer Agent, `collector`, sammelt den Batch und bleibt unverändert. Er liefert die alte Strategie für jedes Verhältnis:

<!--en:df3e4797-->
`d2l.plot_hist` (:numref:`sec_utils`) legt die Histogramme der Verhältnisse beider Varianten nach dem letzten Durchlauf übereinander und markiert die Bandgrenzen mit gestrichelten Linien:

<!--en:922a4952-->
Nach zwanzig Durchläufen liegen die Verhältnisse des Agenten mit Clipping weiterhin nahe eins, nur eine Minderheit liegt jenseits der gestrichelten Grenzen. Ohne Clipping häufen sich die Verhältnisse nahe null. Die meisten übrigen liegen weit oberhalb des Bands und reichen bis etwa zwei oder drei. Ein Verhältnis von zwei oder drei entspricht $1/\pi_{\theta_{\textrm{old}}}(a)$ bei einer Aktion, welche die untrainierte Strategie mit Wahrscheinlichkeit ein Halb oder ein Drittel und die neue Strategie fast immer wählt. Ein Verhältnis nahe null gehört zur anderen Aktion. Wie viel ist ein so umgewichteter Batch noch wert? Das beantwortet die *effektive Stichprobengröße* (engl. effective sample size) der $n$ Wichtigkeitsgewichte: $N_{\textrm{eff}} = 1 / \sum_j \bar{w}_j^2$. Dabei ist $\bar{w}_j$ das Verhältnis für Schritt $j$ geteilt durch die Summe aller Verhältnisse des Batchs. Bei gleichen Gewichten beträgt die effektive Stichprobengröße $n$, bei einem einzigen dominierenden Gewicht eins.

<!--en:9aa572a9-->
Mit Clipping bleiben die Gewichte bis zur zwanzigsten Epoche annähernd gleich. Ohne Clipping fällt die effektive Stichprobengröße deutlich unter die der Variante mit Clipping, auf etwa die Hälfte des Batchs. Der Wert erklärt sich aus der Sättigung. Angenommen, die neue Strategie wählt mit Sicherheit eine Aktion $a^*(s)$. Eine Beobachtung im Zustand $s$ hat dann das Gewicht $1/\pi_{\theta_{\textrm{old}}}(a^* \mid s)$, wenn ihre Aktion $a^*$ war. Das geschieht mit Wahrscheinlichkeit $\pi_{\theta_{\textrm{old}}}(a^* \mid s)$. Andernfalls ist ihr Gewicht null. Das mittlere Gewicht ist eins, das mittlere quadrierte Gewicht $1/\pi_{\theta_{\textrm{old}}}(a^* \mid s)$. Mit wachsendem Batch nähert sich daher $N_{\textrm{eff}}/n$ dem harmonischen Mittel von $\pi_{\theta_{\textrm{old}}}(a^* \mid s)$ über die besuchten Zustände. Für eine untrainierte Strategie, die beide Aktionen fast gleich häufig wählt, ist das etwa die Hälfte. Zur Prüfung brauchen wir die Wahrscheinlichkeiten aller Aktionen, auch der nicht gezogenen. Die Funktion `action_probs` liefert sie als NumPy-Array. Der untrainierte Agent `collector` stellt die Strategie bereit, die den Batch gesammelt hat.

<!--en:454a9610-->
`action_probs` wertet die Strategie mit `d2l.forward_np` (:numref:`sec_utils`) aus. Das ist der kompilierte Vorwärtsdurchlauf, den auch `act` verwendet.

<!--en:baf909fd-->
Die Strategie ohne Clipping ist beinahe deterministisch. Das harmonische Mittel weicht um höchstens etwa ein Zehntel von der relativen effektiven Stichprobengröße der Variante ohne Clipping ab. Es liegt etwas darunter, weil die Strategie noch nicht ganz deterministisch ist. Die effektive Stichprobengröße misst, wie gleichmäßig sich die Gewichte verteilen. Sie berücksichtigt weder die Vorteile noch die Zustände, die die neue Strategie besuchen würde. Deshalb ergänzt sie die Divergenz und den Anteil außerhalb des Bands, ohne sie zu ersetzen.

<!--en:45a39582-->
Wählt ein mit Clipping trainierter Agent schließlich stets die wahrscheinlichste Aktion, entfällt das Rauschen durch das Sampling. Er sollte den Stab bis zur Zeitgrenze im Gleichgewicht halten:

<!--en:dd9691d3-->
### Vektorisierte Datenerhebung und Minibatches

<!--en:1f63cb11-->
Unsere Schleife spielt ganze Episoden in einer Umgebung und nutzt bei jedem Durchlauf den gesamten Batch. Produktivimplementierungen wie `cleanrl/ppo.py` :cite:`Huang.Dossa.Ye.ea.2022` führen $N$ Kopien der Umgebung parallel über einen festen Horizont $T$ aus. So entsteht ein Rechteck aus $N\times T$ Übergängen. Am Rand des Rechtecks schneiden sie jede noch laufende Episode ab. An dieser Schnittstelle liefert der Kritiker den Wert der nicht aufgezeichneten Zukunft, wie beim Bootstrap-Verfahren an der CartPole-Zeitgrenze in :numref:`sec_deeprl`. Wir sammeln mit dem trainierten Agenten ein solches Rechteck aus acht Umgebungen und 32 Schritten:

<!--en:59b6352b-->
Der trainierte Agent hält den Stab alle 32 Schritte im Gleichgewicht. Alle acht Trajektorien enden daher an der Schnittstelle. Ihre Werte liegen nahe $1/(1-\gamma)=100$, dem diskontierten Wert für unbegrenzt langes Balancieren. Ohne den geschätzten Folgewert würde GAE jede Schnittstelle wie einen Sturz behandeln und den letzten Aktionen davor große negative Vorteile zuweisen. Die Vektorumgebungen von Gymnasium starten nach einer abgeschlossenen Episode selbstständig neu. Der Schritt nach dem Ende einer Episode ignoriert die Aktion und liefert die erste Beobachtung einer neuen Episode mit Belohnung null. Implementierungen schließen diesen Schritt daher mit einer Maske aus der Berechnung der Vorteile und des Verlusts aus. Schließlich mischt der Produktivcode die Übergänge des Rechtecks und teilt sie in Minibatches auf, über die er einige Epochen durchläuft, typischerweise vier. Jedes Verhältnis wird dann viermal statt zwanzigmal geprüft. Die kleineren Schritte benötigen weniger Speicher und bringen zusätzliches Gradientenrauschen mit sich. In :exref:`ex_ppo-standard-recipe` bauen Sie diese Schleife auf.

<!--en:51d036ab-->
Die vektorisierte Datenerhebung ändert auch die Kosten einer Beobachtung. In diesem Kapitel zählen wir Umgebungsschritte, denn bei einem Roboterarm ist jeder Schritt real. Ein Physiksimulator auf der GPU bewegt Tausende Kopien eines Roboters gleichzeitig um einen Schritt :cite:`Makoviychuk.Wawrzyniak.Guo.ea.2021`. Damit liegt ein Rechteck mit Tausenden Umgebungen in einem Sekundenbruchteil vor. Für die Kosten zählt dann die tatsächlich verstrichene Zeit. PPO eignet sich dafür, weil ein sehr großer Batch jedes der wenigen Updates genau macht. Um die Strategie auf die reale Maschine übertragen zu können, variieren wir die Parameter des Simulators nach dem Zufallsprinzip. Mit dieser *Domänenrandomisierung* (engl. domain randomization) :cite:`Tobin.Fong.Ray.ea.2017` lernen Roboter mit Beinen in wenigen Minuten Training das Gehen :cite:`Rudin.Hoeller.Reist.ea.2022`. Deshalb ist PPO dort weiterhin das Standardverfahren, obwohl die Verfahren der folgenden Abschnitte weit weniger Schritte benötigen. Diese haben den Vorteil, wenn die Schritte real sind.

<!--en:d1d23f16-->
Mit PPO kann ein Agent für jeden Batch mehrere Schritte machen. In CartPole erreichen vier Epochen den Höchstwert in halb so vielen Umgebungsschritten wie eine Epoche. Das Clipping begrenzt, wie weit sich die Aktionswahrscheinlichkeiten in jedem Schritt verschieben. Es beeinflusst damit den Weg, während das Ziel dasselbe bleibt. Die Entropie der Läufe mit Clipping fällt weiterhin von $0.65$ auf $0.25$ Nat. Die Zielfunktion ist nach wie vor der Ertrag mit einem kleinen Entropiebonus, und ihr Optimum ist nahezu deterministisch. In :numref:`sec_regularized` ändern wir hingegen die Zielfunktion. Sie hält die Strategie in der Nähe einer festen Referenz und verschiebt damit auch das Optimum.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:b1def804-->
Ein On-Policy-Batch mit Tausenden von Umgebungsschritten sollte mehr als einen Gradientenschritt ermöglichen. Gleich große Schritte im Parameterraum können eine Strategie jedoch sehr unterschiedlich stark verändern. Importance-Sampling bewertet eine neue Strategie anhand alter Daten exakt, doch die Varianz wächst exponentiell mit der Episodenlänge. Behalten wir pro Schritt nur ein Verhältnis bei, entfällt dieses Wachstum. Dafür bewerten wir die Aktionen der neuen Strategie in den Zuständen der alten. Nach dem Lemma zur Ertragsdifferenz ist dieser Austausch die einzige Fehlerquelle der Ersatzzielfunktion. TRPO begrenzt den Fehler durch eine Nebenbedingung für die Divergenz zwischen alter und neuer Aktionsverteilung. PPO begrenzt stattdessen den Beitrag jeder Beobachtung zur Zielfunktion, sobald ihr Verhältnis ein Band um eins in günstiger Richtung verlässt. So lässt sich jeder Batch über mehrere Epochen nutzen. Auf CartPole erreichen vier Epochen den Höchstwert in etwa halb so vielen Umgebungsschritten wie eine Epoche. Das Clipping verhindert, dass die Strategie bei großer Lernrate kollabiert. Der Anteil außerhalb des Bands, die geschätzte KL-Divergenz, die Entropie und die effektive Stichprobengröße der Verhältnisse zeigen, wie stark jedes Update die Strategie verändert hat.

<!--en:2294243a-->
## Übungen

<!--en:8b65e204-->
1. **Schrittgrenzen für ein einzelnes Logit.** ● Betrachten Sie die Strategie $\pi_\theta(a{=}1) = \sigma(\theta)$ mit zwei Aktionen aus :numref:`fig_rl-policy-vs-parameter` und drei Möglichkeiten, ein Update $\theta \to \theta + \Delta$ zu begrenzen: eine Trust-Region für die Divergenz, das Clipping von PPO und eine obere Schranke für $|\Delta|$.
    1. Entwickeln Sie $D_{\textrm{KL}}\big(\pi_\theta \Vert \pi_{\theta+\Delta}\big)$ bis zur zweiten Ordnung in $\Delta$ und zeigen Sie, dass sich $\frac{1}{2}\sigma'(\theta)\,\Delta^2$ ergibt. Berechnen Sie die exakte Divergenz der beiden in der Abbildung dargestellten Updates mit $\Delta = 2$ von $\theta = 0$ und von $\theta = 6$ aus. Welches bleibt innerhalb einer Trust-Region mit $\Delta_{\textrm{KL}} = 0.02$?
    1. Bestimmen Sie mit der Näherung zweiter Ordnung das größte $|\Delta|$, das diese Trust-Region bei $\theta = 0$ und bei $\theta = 6$ zulässt. Berechnen Sie bei $\theta = 6$ die exakte Divergenz für einen Schritt dieser Größe in jede der beiden Richtungen. Warum ist die Näherung zweiter Ordnung dort wesentlich ungenauer als bei $\theta = 0$?
    1. Nun begrenze das Clipping in :eqref:`eq_ppo_clip` mit $\epsilon_{\textrm{clip}} = 0.2$ den Schritt für eine Beobachtung mit positivem Vorteil. Das Update erhöht die Wahrscheinlichkeit der gezogenen Aktion. Somit gilt $\Delta > 0$ für $a = 1$ und $\Delta < 0$ für $a = 2$. Bestimmen Sie das größte $|\Delta|$, bis zu dem die Zielfunktion mit Clipping noch wächst, für beide Aktionen bei $\theta = 0$ und bei $\theta = 6$.
    1. Welche der drei Begrenzungen lassen große Parameterschritte zu, wo die Strategie unempfindlich auf $\theta$ reagiert, und nur kleine, wo sie empfindlich reagiert? Für welche gezogene Aktion ist das Clipping bei $\theta = 6$ weniger restriktiv als die Trust-Region, für welche restriktiver?
   :exlabel:`ex_ppo-step-limits`

<!--en:36fddde2-->
1. **Explodierendes Produkt von Verhältnissen.** Eine zustandsunabhängige Strategie sammelt Episoden mit $\pi_{\theta_{\textrm{old}}}$. Anhand dieser Daten bewerten wir $\pi_\theta$. Bezeichnen Sie mit $W_T$ das Trajektorienverhältnis :eqref:`eq_traj_ratio` einer Episode mit $T$ Schritten.
    1. Zeigen Sie, dass unter der alten Strategie $\mathbb{E}[W_T] = 1$ und $\textrm{Var}[W_T] = (1 + \chi^2)^T - 1$ gelten, wobei $\chi^2 = \sum_a \big(\pi_\theta(a) - \pi_{\theta_{\textrm{old}}}(a)\big)^2 / \pi_{\theta_{\textrm{old}}}(a)$ ist. Prüfen Sie den Wert $1.04^T - 1$ aus dem Text für $\pi_{\theta_{\textrm{old}}} = (0.5, 0.5)$ und $\pi_\theta = (0.6, 0.4)$ nach.
    1. Mit wachsender Zahl $n$ von Episoden nähert sich die effektive Stichprobengröße der Gewichte $W_T$ dem Wert $n / \mathbb{E}[W_T^2]$. Für welches $n$ ist dieser Grenzwert bei $T = 200$ für die beiden Strategien aus (a) gleich eins? Simulieren Sie einen Batch mit so vielen Episoden (die Zahl der Schritte mit $a = 1$ in einer Episode ist binomialverteilt) und berechnen Sie seine effektive Stichprobengröße. Warum ergibt sich für den Batch eine wesentlich größere effektive Stichprobengröße als der Grenzwert? Welche der beiden Größen beschreibt die Varianz der Schätzung mit Wichtigkeitsgewichten?
    1. Die Ersatzzielfunktion pro Schritt :eqref:`eq_surrogate` gewichtet jeden Schritt mit einem einzigen Verhältnis $\rho_t$. Welche Varianz hat dieses Gewicht hier, und durch welche Näherung in :eqref:`eq_surrogate` wird sie unabhängig von $T$?
   :exlabel:`ex_ppo-exploding-product`

<!--en:2acf163f-->
1. **Was die Kennzahlen schätzen.** ● Wählen Sie einen festen Zustand $s$, ziehen Sie $a \sim \pi_{\theta_{\textrm{old}}}(\cdot \mid s)$ und setzen Sie $\rho = \pi_\theta(a \mid s) / \pi_{\theta_{\textrm{old}}}(a \mid s)$.
    1. Für die geschätzte KL-Divergenz mittelt `ppo_epochs` die Größe $k_1 = -\log \rho$ über den Batch. Zeigen Sie, dass $\mathbb{E}[k_1] = D_{\textrm{KL}}\big(\pi_{\theta_{\textrm{old}}}(\cdot \mid s) \Vert \pi_\theta(\cdot \mid s)\big)$ gilt. Zeigen Sie dann, dass $k_3 = (\rho - 1) - \log \rho$ denselben Erwartungswert hat und für jede Beobachtung nichtnegativ ist. Entwickeln Sie beide Schätzer bis zur zweiten Ordnung in $\rho - 1$. Warum hat $k_3$ eine wesentlich kleinere Varianz als $k_1$, wenn sich die Strategie nur wenig ändert? `cleanrl/ppo.py` protokolliert beide.
    1. In :exref:`ex_ppo-exploding-product`b verwenden wir den Grenzwert $n / \mathbb{E}[w^2]$ der effektiven Stichprobengröße für $n$ unabhängige Gewichte $w$ mit $\mathbb{E}[w] = 1$. Leiten Sie diesen Grenzwert mit dem Gesetz der großen Zahlen her. Zeigen Sie dann, dass $\mathbb{E}[\rho^2] = 1 + \chi^2$ gilt. Damit nähert sich die effektive Stichprobengröße der Gewichte $\rho$ geteilt durch $n$ dem Wert $1 / (1 + \chi^2)$, wobei

       $$\chi^2 = \sum_a \frac{\big(\pi_\theta(a \mid s) - \pi_{\theta_{\textrm{old}}}(a \mid s)\big)^2}{\pi_{\theta_{\textrm{old}}}(a \mid s)}.$$

       Zeigen Sie, dass bis zur zweiten Ordnung in der Strategieänderung $\chi^2 \approx 2 D_{\textrm{KL}}$ gilt.
    1. Bei Zuständen aus einem Batch tritt an die Stelle von $\chi^2$ der Mittelwert über die besuchten Zustände. Welche Werte von $\chi^2$ ergeben sich aus den beiden Anteilen `ess[name][-1]` in :numref:`subsec_ppo_diagnostics`? Welche geschätzte KL-Divergenz sagt die Beziehung zweiter Ordnung aus (b) jeweils voraus? Vergleichen Sie jede Vorhersage mit der $k_1$-Schätzung aus denselben Verhältnissen, also `(-np.log(rhos[arm][-1])).mean()` für die jeweilige Variante `arm`. Für welchen Agenten versagt die Beziehung zweiter Ordnung? Welche Annahme aus (b) verletzen die Verhältnisse neben der Größe der Strategieänderung, wenn wir sie auf genau dem Batch messen, auf dem die Strategie trainiert wurde?

   *Nach John Schulman, [Näherung der KL-Divergenz](http://joschu.net/blog/kl-approx.html), Schätzer $k_1$ und $k_3$.*
   :exlabel:`ex_ppo-diagnostics-estimate`

<!--en:19bf5cb7-->
1. [code] **Early Stopping anhand der KL-Divergenz.** ● Neben dem Clipping kann eine Implementierung die Arbeit mit einem Batch beenden, sobald sich die Strategie weit genug verändert hat. Die PPO-Implementierung von Spinning Up berechnet die geschätzte KL-Divergenz vor jedem Durchlauf und beendet die Arbeit mit diesem Batch, sobald die Schätzung das $1.5$-Fache von `target_kl` übersteigt. Der Standardwert dieses Parameters ist $0.01$.
    1. Ersetzen Sie den Aufruf von `ppo_epochs` in `train_ppo` durch eine Schleife aus einzelnen Durchläufen mit `ppo_epochs(..., 1, ...)`, wie in der Schleife, die `rhos` in :numref:`subsec_ppo_diagnostics` sammelt. Berechnen Sie vor jedem Durchlauf die geschätzte KL-Divergenz aus `log_prob_np` und wenden Sie die Abbruchregel an. Protokollieren Sie die Zahl der Durchläufe pro Batch.
    1. Führen Sie die Variante ohne Clipping mit Early Stopping für die acht Zufallsstartwerte aus der Ablationsstudie aus. Verhindert die Abbruchregel allein den Einbruch? Wie verändert sich die Zahl der Durchläufe pro Batch im Trainingsverlauf?
    1. Führen Sie auch die Variante mit Clipping und Early Stopping aus. Welcher der beiden Mechanismen begrenzt in diesem Lauf die Strategieänderung pro Batch? Was bedeutet das für die Beschreibung des Clippings als Trust-Region?

   *Nach Joshua Achiam, [Spinning Up in Deep RL](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py), PPO-Implementierung mit Early Stopping anhand von `target_kl`.*

<!--en:9216e3f5-->
1. [code] **Das Standardverfahren.** ●● Implementieren Sie die Schleife aus :numref:`subsec_ppo_vectorized`. Führen Sie acht Umgebungen in einer Instanz von `gym.vector.SyncVectorEnv` für 32 Schritte pro Update aus. Berechnen Sie GAE über das $32 \times 8$-Rechteck mit dem Wert des Kritikers an jeder Schnittstelle und maskieren Sie den Schritt mit automatischem Neustart aus. Trainieren Sie dann vier Epochen auf den gemischten Daten des Rechtecks in Minibatches der Größe 64. Machen Sie pro Minibatch einen Durchlauf von `ppo_epochs` mit den innerhalb dieses Minibatchs standardisierten Vorteilen und einen Schritt von `fit_value` in Richtung des vor den Epochen berechneten $\lambda$-Ertrags. Verwenden Sie für beide Netze die Lernrate $10^{-3}$.
    1. Trainieren Sie mit drei Zufallsstartwerten und erfassen Sie die Zahl der Umgebungsschritte, bis die letzten acht abgeschlossenen Episoden einen mittleren Ertrag von mindestens 475 erreichen. Eine korrekte Implementierung braucht etwa 45 000 Schritte, bei den meisten Zufallsstartwerten zwischen 35 000 und 55 000. Das sind mehr als die etwa 25 000 Schritte der Variante mit vier Epochen im Versuch zur Wiederverwendung in :numref:`subsec_ppo_reuse`.
    1. Erhöhen Sie die Lernrate des Kritikers auf $10^{-2}$, den Wert unserer Schleife, und belassen Sie die des Akteurs bei $10^{-3}$. Eine korrekte Implementierung erreicht 475 nun in etwa 15 000 bis 20 000 Schritten, weniger als jede Variante des Versuchs zur Wiederverwendung. Welches Netz begrenzte den Lauf in (a)? Unsere Schleife braucht bei denselben beiden Lernraten ohne Vektorisierung und ohne Minibatches etwa 35 000 bis 40 000 Schritte. Wie verändern die kurzen Rechtecke die Häufigkeit der Strategieupdates pro Umgebungsschritt?
    1. Entfernen Sie die Maske für den Schritt mit automatischem Neustart. Lernt der Agent weiterhin? Welche Übergänge kommen ohne Maske zu den Zielwerten des Kritikers hinzu, und warum fallen sie bei CartPole kaum ins Gewicht?
   :exlabel:`ex_ppo-standard-recipe`

<!--en:b65e1265-->
1. **Die Schranke für monotone Verbesserung.** ●● Leiten Sie :eqref:`eq_trpo_bound` mit dem Kopplungsargument von :citet:`Schulman.Levine.Abbeel.ea.2015` her. Betrachten Sie eine Aufgabe mit unendlichem Horizont und $\gamma < 1$. Schreiben Sie $\pi = \pi_{\theta_{\textrm{old}}}$, $\tilde{\pi} = \pi_\theta$, $A = A^{\textrm{old}}$ und $\bar{A}(s) = \sum_a \tilde{\pi}(a \mid s)\, A(s, a)$. Sei $\Delta_{\textrm{TV}} = \max_s D_{\textrm{TV}}\big(\pi(\cdot \mid s), \tilde{\pi}(\cdot \mid s)\big)$ mit dem Totalvariationsabstand $D_{\textrm{TV}}(p, q) = \frac{1}{2} \sum_a |p(a) - q(a)|$.
    1. Zeigen Sie anhand von :eqref:`eq_perf_diff`, dass $J(\theta) - J(\theta_{\textrm{old}}) = \sum_t \gamma^t \mathbb{E}_{s_t \sim \tilde{\pi}}\big[\bar{A}(s_t)\big]$ gilt und dass die Ersatzzielfunktion für die Grundgesamtheit aus :eqref:`eq_pop_surrogate` die Form $\bar{L}(\theta) = \sum_t \gamma^t \mathbb{E}_{s_t \sim \pi}\big[\bar{A}(s_t)\big]$ hat. Dabei bezeichnet $s_t \sim \pi$ die Zustandsverteilung zum Zeitpunkt $t$ unter $\pi$.
    1. Zwei Verteilungen mit Totalvariationsabstand $\Delta_{\textrm{TV}}$ lassen sich koppeln. Es gibt eine gemeinsame Verteilung von $(a, \tilde{a})$ mit Randverteilungen $\pi(\cdot \mid s)$ und $\tilde{\pi}(\cdot \mid s)$, unter der in jedem Zustand $P(a \neq \tilde{a}) \leq \Delta_{\textrm{TV}}$ gilt. Zeigen Sie mithilfe von $\mathbb{E}_{a \sim \pi}[A(s, a)] = 0$, dass $|\bar{A}(s)| \leq 2\Delta_{\textrm{TV}} A_{\max}$ gilt.
    1. Führen Sie beide Strategien vom selben Anfangszustand aus und koppeln Sie ihre Aktionen in jedem Schritt. Dabei zähle $m_t$ die Schritte vor $t$, in denen sich die beiden Aktionen unterscheiden. Tritt das Ereignis $m_t = 0$ ein, stimmen die beiden Trajektorien bis zum Zeitpunkt $t$ überein. Zeigen Sie, dass

       $$\big|\mathbb{E}_{s_t \sim \tilde{\pi}}[\bar{A}(s_t)] - \mathbb{E}_{s_t \sim \pi}[\bar{A}(s_t)]\big| \leq 4\Delta_{\textrm{TV}} \big(1 - (1 - \Delta_{\textrm{TV}})^t\big) A_{\max}.$$
    1. Summieren Sie über $t$ und erhalten Sie

       $$\big|J(\theta) - J(\theta_{\textrm{old}}) - \bar{L}(\theta)\big| \leq \frac{4 \Delta_{\textrm{TV}}^2 \gamma A_{\max}}{(1-\gamma)^2},$$

       Schließen Sie den Beweis mit $D_{\textrm{TV}}^2 \leq D_{\textrm{KL}}$ ab.
    1. Berechnen Sie den Koeffizienten $4\gamma / (1-\gamma)^2$ bei $\gamma = 0.99$. Zeigen Sie mithilfe von (b), dass $\bar{L}(\theta) \leq 2\Delta_{\textrm{TV}} A_{\max} / (1-\gamma)$ gilt. Zeigen Sie mit der Pinsker-Ungleichung $D_{\textrm{KL}} \geq 2 D_{\textrm{TV}}^2$, dass die Schranke nur dann eine Verbesserung garantiert, wenn $\Delta_{\textrm{TV}} < (1-\gamma) / (4\gamma)$ gilt. Wie groß kann $\bar{L}(\theta)$ für einen Schritt mit garantierter Verbesserung dann in Einheiten von $A_{\max}$ sein? Vergleichen Sie diese Grenze für $\Delta_{\textrm{TV}}$ mit der totalen Variation $\sqrt{\Delta_{\textrm{KL}} / 2}$, die die Pinsker-Ungleichung für einen Zustand zulässt, dessen Divergenz das Budget $\Delta_{\textrm{KL}} = 0.01$ ausschöpft. Dieses Budget verwenden :citet:`Schulman.Levine.Abbeel.ea.2015` in allen ihren Experimenten.
   :exlabel:`ex_ppo-monotonic-bound`

<!--en:0d6f9602-->
1. [extended] **TRPO aus seinen Bausteinen.** ●● Ersetzen Sie den Aufruf von `ppo_epochs` in `train_ppo` durch ein TRPO-Update :cite:`Schulman.Levine.Abbeel.ea.2015` für das Problem mit Nebenbedingung :eqref:`eq_trpo_constraint` und $\Delta_{\textrm{KL}} = 0.01$. Behalten Sie den Kritiker und die Vorteile bei. Fassen Sie die Parameter von `ac.policy` als einen einzigen flachen Vektor auf. Halten Sie die Aktionswahrscheinlichkeiten der Strategie, mit der Sie die Daten gesammelt haben, `action_probs(ac, b.obs)`, zusammen mit `logp_old` fest.
    1. Berechnen Sie den Gradienten $\mathbf{g}$ der Ersatzzielfunktion :eqref:`eq_surrogate` bei $\theta_{\textrm{old}}$. Schreiben Sie eine Funktion, die das Fisher-Vektor-Produkt $F\mathbf{v}$ zurückgibt. Dazu wenden Sie die Hesse-Matrix der mittleren Divergenz über die Zustände des Batchs bei $\theta_{\textrm{old}}$ auf $\mathbf{v}$ an, indem Sie den Gradienten dieser Divergenz ein weiteres Mal differenzieren. Prüfen Sie das Ergebnis anhand einer endlichen Differenz des Gradienten.
    1. Lösen Sie $F\mathbf{x} = \mathbf{g}$ näherungsweise mit zehn Iterationen des Verfahrens der konjugierten Gradienten. Skalieren Sie den Schritt auf $\xi\mathbf{x}$ mit $\xi = \sqrt{2\Delta_{\textrm{KL}} / \mathbf{x}^\top F \mathbf{x}}$. Halbieren Sie $\xi$ so oft, bis die gemessene mittlere Divergenz höchstens $\Delta_{\textrm{KL}}$ beträgt und der Wert der Ersatzzielfunktion gestiegen ist.
    1. Trainieren Sie mit drei Zufallsstartwerten und vergleichen Sie die Ergebnisse mit den Läufen mit Clipping aus :numref:`subsec_ppo_ablation`. Vergleichen Sie die Lernkurven, die gemessene Divergenz jedes Updates und die Zahl der Halbierungen. Wie viele Gradienten und Fisher-Vektor-Produkte berechnet jedes Verfahren pro Batch?
    1. :citet:`Engstrom.Ilyas.Santurkar.ea.2020` stellten fest, dass PPO und TRPO ähnlich gut abschneiden, wenn sie mit denselben Implementierungsentscheidungen arbeiten. Hier stimmen beide Verfahren bis auf das Strategieupdate in allen Entscheidungen überein. Kommen Sie zum selben Ergebnis?
   :exlabel:`ex_ppo-trpo`

<!--en:fe4caa17-->
[Diskussionen](https://d2l.discourse.group/t/36802)

<!--en:345d6ff1-->
Trust-Regionen und Proximal Policy Optimization<br>**Parameterabstand und Strategieabstand · Importance-Sampling · Lemma zur Ertragsdifferenz · TRPO · PPO-Clipping · Batchdiagnostik**

<!--en:ec0e327f-->
::: {.slide title="Warum ein Gradientenschritt einen Batch von 4000 Schritten kaum ausschöpft"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:71c5dd2b-->
- **Ungenutzte Information:** Tausende Übergänge liefern weit mehr Information, als ein einziger Schritt erster Ordnung verwerten kann.

<!--en:e95315d2-->
- **Warum mehrere Epochen ohne Begrenzung scheitern:** Schon nach dem ersten Update von $\theta$ stammt der Batch von einer älteren Strategie $\pi_{\theta_{\textrm{old}}}$. Weitere unbeschränkte Durchläufe lassen die Aktionswahrscheinlichkeiten unumkehrbar kollabieren.

<!--en:e789a251-->
::: {.slide title="Der Parameterabstand begrenzt den Strategieabstand nicht"}

<!--en:b45158bc-->
[Geometrie des Strategieraums]{.kicker}

<!--en:a48e41d5-->
- Betrachten wir eine Bernoulli-Strategie $\pi_\theta(a{=}1) = \sigma(\theta) = (1 + e^{-\theta})^{-1}$ mit einem Parameter und zwei gleich großen Parameterschritten $\Delta \theta = 2$ (Achiam, 2017):
  - Bei $\theta = 0$ (Steigung $\sigma'(0) = 0.25$) macht $\pi(a{=}1)$ einen Sprung: **$0.50 \to 0.88$**.
  - Bei $\theta = 6$ (Steigung $\sigma'(6) \approx 0.0025$) ändert sich $\pi(a{=}1)$ kaum: **$0.9975 \to 0.9997$**.

<!--en:d03d0cca-->
- **Sättigungsfalle:** Bei $\theta = 6$ verschwindet der Score $\nabla_\theta \log \pi_\theta(a{=}1) = 1 - \sigma(6) \approx 0.0025$ nahezu. Zugleich kommt $a{=}2$ nur einmal in $400$ Versuchen vor. On-Policy-Verfahren kommen aus dieser Falle kaum heraus.

<!--en:f1f35ab3-->
::: {.slide title="Maßwechsel zur Bewertung neuer Strategien mit alten Daten"}

<!--en:bf736773-->
[Importance-Sampling]{.kicker}

<!--en:0f6c722b-->
- **Die Dynamik kürzt sich heraus:** Anfangsverteilung $\mu_0(s_0)$ und Übergangsdynamik $P(s_{t+1} \mid s_t, a_t)$ kürzen sich in $P(\tau; \theta) / P(\tau; \theta_{\textrm{old}})$ vollständig heraus. Es bleibt $\prod_{t=0}^{T-1} \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\textrm{old}}}(a_t \mid s_t)}$.

<!--en:ba7282ef-->
::: {.slide title="Warum Wichtigkeitsgewichte ganzer Trajektorien explodieren"}

<!--en:e2fdd1e5-->
[Der Fluch des langen Horizonts]{.kicker}

<!--en:6108d254-->
- **Exponentielles Wachstum der Varianz:** Selbst wenn jedes Schrittverhältnis $\rho_t \in \{0.8, 1.2\}$ nur die kleine Varianz $0.04$ hat, hat das Produkt $W_T = \prod_{t=0}^{T-1} \rho_t$ die Varianz $\textrm{Var}(W_T) = 1.04^T - 1$.

<!--en:078fd999-->
- **Kollaps bei 500 CartPole-Schritten:** Für $T = 500$ gilt $\textrm{Var}(W_T) = 3.3 \times 10^8$. Eine einzige Trajektorie dominiert dann den Batch. PPO behält nur das Verhältnis $\rho_t(\theta)$ für den einzelnen Schritt bei.

<!--en:4b090456-->
::: {.slide title="Die Ersatzzielfunktion pro Schritt"}

<!--en:1c1ddd02-->
[Lokale Näherung]{.kicker}

<!--en:303a02e5-->
Um das exponentielle Wachstum mit dem Horizont zu vermeiden, lassen wir das Produkt $\rho_0 \cdots \rho_{t-1}$ der früheren Schritte weg (es korrigiert die Zustandsverteilung zum Zeitpunkt $t$). Wir behalten nur das **Wahrscheinlichkeitsverhältnis des einzelnen Schritts** $\rho_t(\theta)$ bei:

<!--en:c7c17a58-->
- **Übereinstimmung erster Ordnung:** Bei $\theta = \theta_{\textrm{old}}$ gilt für jedes Verhältnis $\rho_t(\theta_{\textrm{old}}) = 1$ und außerdem $\nabla_\theta \rho_t(\theta_{\textrm{old}}) = \nabla_\theta \log \pi_\theta(a_t \mid s_t)$. Also ist $\nabla_\theta \hat{L}(\theta_{\textrm{old}})$ gleich dem üblichen Strategiegradienten.

<!--en:fde2f17d-->
- **Was entfällt:** Die Ersatzzielfunktion $\hat{L}(\theta)$ bewertet die Aktionswahrscheinlichkeiten der *neuen* Strategie anhand der Zustandsbelegung der *alten* Strategie.

<!--en:927b6f28-->
::: {.slide title="Das Lemma zur Ertragsdifferenz"}

<!--en:55053c6f-->
[Exakte Zerlegung des Ertrags]{.kicker}

<!--en:f6625ee6-->
- **Beweis durch Teleskopsumme (Kakade & Langford, 2002):** Bei der Summation von $\gamma^t(r_t + \gamma V^{\textrm{old}}(s_{t+1}) - V^{\textrm{old}}(s_t))$ heben sich Terme paarweise auf. Es bleibt $R(\tau) - V^{\textrm{old}}(s_0)$.

<!--en:a3f8a8c0-->
- **Fehler der Ersatzzielfunktion:** In der Ersatzzielfunktion $\bar{L}(\theta)$ für die Grundgesamtheit ersetzt die alte Zustandsbelegung die neue: An die Stelle von $d^{\pi_\theta}_\gamma(s)$ tritt $d^{\pi_{\theta_{\textrm{old}}}}_\gamma(s)$.

<!--en:6b9d9925-->
::: {.slide title="Monotone Verbesserung und Trust-Region-Schranken (TRPO)"}

<!--en:414a73ea-->
[Von der Theorie zu TRPO]{.kicker}

<!--en:2b62a928-->
- **Schranke nach Kakade und Langford sowie Schulman:** Die größte KL-Divergenz über alle Zustände begrenzt den Abstand zwischen der tatsächlichen Verbesserung $J(\theta) - J(\theta_{\textrm{old}})$ und $\bar{L}(\theta)$:
  $$J(\theta) \ge J(\theta_{\textrm{old}}) + \bar{L}(\theta) - \frac{4\gamma A_{\max}}{(1-\gamma)^2} \max_s D_{\textrm{KL}}(\pi_{\theta_{\textrm{old}}} \Vert \pi_\theta)$$

<!--en:1db3b538-->
- **TRPO (Schulman et al., 2015):** Da der Faktor $4\gamma/(1-\gamma)^2 \approx 40{,}000$ bei $\gamma=0.99$ eine zu konservative Schranke ergibt, maximiert TRPO $\hat{L}(\theta)$ mit konjugierten Gradienten unter einer Nebenbedingung für die mittlere KL-Divergenz $\bar{D}_{\textrm{KL}} \le \Delta_{\textrm{KL}} = 0.01$.

<!--en:85da90ea-->
::: {.slide title="PPO mit Clipping erster Ordnung für Wahrscheinlichkeitsverhältnisse"}

<!--en:8437012b-->
[Proximal Policy Optimization]{.kicker}

<!--en:7a48b182-->
- **PPO (Schulman et al., 2017)** ersetzt die TRPO-Nebenbedingung zweiter Ordnung, die Fisher-Vektor-Produkte erfordert, durch eine Ersatzzielfunktion mit Clipping für ein Verfahren erster Ordnung ($\epsilon_{\textrm{clip}} = 0.2$):
  $$L^{\textrm{CLIP}}(\theta) = \frac{1}{N} \sum_{i,t} \min\Big(\rho_t^i \hat{A}_t^i,\ \textrm{clip}(\rho_t^i, 1-\epsilon_{\textrm{clip}}, 1+\epsilon_{\textrm{clip}})\hat{A}_t^i\Big)$$

<!--en:33e2322e-->
- **Pessimistische untere Schranke:** Bei $\hat{A}_t > 0$ verläuft die Zielfunktion oberhalb von $\rho_t = 1 + \epsilon_{\textrm{clip}}$ flach (Gradient null), bei $\hat{A}_t < 0$ unterhalb von $1 - \epsilon_{\textrm{clip}}$. Damit entfällt der Anreiz, Verhältnisse über das Intervall $[0.8, 1.2]$ hinaus zu verschieben.

<!--en:7676fd1e-->
::: {.slide title="Warum Verhältnisse das Band dennoch verlassen und wie der Entropiebonus wirkt"}

<!--en:9a67a6d8-->
[Feinheiten der Optimierung]{.kicker}

<!--en:0f5dba15-->
- **Kopplung durch gemeinsame Gewichte und Momentum:** Übergänge ohne Clipping verändern die gemeinsamen Gewichte $\theta$ auch für Übergänge, bei denen das Clipping schon greift. Zudem wirkt bei Adam der Akkumulator $m_t$ für das erste Moment nach.

<!--en:9592a239-->
- **Steigung auf der ungünstigen Seite und Entropiebonus (`0.01`):** Jenseits der ungünstigen Grenze ($\rho < 0.8$ für $\hat{A} > 0$) wirkt die Steigung weiterhin zurück in Richtung des Bands. Zugleich hält $+\beta_H H(\pi_\theta)$ die Wahrscheinlichkeiten von $0$ und $1$ fern.

<!--en:ff52c79f-->
::: {.slide title="Implementierung der Ersatzzielfunktionen mit und ohne Clipping"}

<!--en:8f069e89-->
[Codebausteine]{.kicker}

<!--en:3c0edbb7-->
Wir implementieren die PPO-Zielfunktion mit Clipping und die Zielfunktion mit Wichtigkeitsgewichtung ohne Clipping ($\rho_t \hat{A}_t$). So können wir beide in der Ablationsstudie unter identischen Bedingungen vergleichen:

<!--en:512336f8-->
In unserem CartPole-Aufbau (`ActorCritic.mlp(4, 2)` mit `tanh`-Schichten aus je 64 Einheiten und Adam-Lernrate $10^{-2}$) verwenden wir GAE mit $\gamma = 0.99$, $\lambda = 0.95$, `critic_steps = 20` und `num_epochs = 20` Akteurdurchläufen pro Batch aus 8 Episoden.

<!--en:9c4b97ae-->
::: {.slide title="Akteurupdates über mehrere Epochen und laufende Diagnose" layout="tight" only="pytorch"}

<!--en:44740089-->
[Innere Epochenschleife]{.kicker}

<!--en:b1cd7fb1-->
Vor jedem der `num_epochs` Durchläufe über den festgehaltenen Batch `(obs, act, adv, logp_old)` protokolliert `ppo_epochs` drei skalare Kennzahlen: den **Anteil außerhalb des Bands**, die **geschätzte KL-Divergenz** $\frac{1}{N}\sum (\log \pi_{\theta_{\textrm{old}}} - \log \pi_\theta)$ und die **Entropie der Strategie**:

<!--en:a60b7773-->
::: {.slide title="Akteurupdates über mehrere Epochen und laufende Diagnose (Fortsetzung)" layout="tight" only="pytorch"}

<!--en:44740089-2-->
[Innere Epochenschleife]{.kicker}

<!--en:1c76b3f2-->
Jeder Durchlauf aktualisiert den Akteur anhand der mittleren Ersatzzielfunktion mit Clipping und eines zusätzlichen Entropiebonus und protokolliert die drei Kennzahlen:

<!--en:4553b26a-->
::: {.slide title="Akteurupdates über mehrere Epochen und laufende Diagnose" layout="tight" only="jax"}

<!--en:44740089-3-->
[Innere Epochenschleife]{.kicker}

<!--en:b1cd7fb1-2-->
Vor jedem der `num_epochs` Durchläufe über den festgehaltenen Batch `(obs, act, adv, logp_old)` protokolliert `ppo_epochs` drei skalare Kennzahlen: den **Anteil außerhalb des Bands**, die **geschätzte KL-Divergenz** $\frac{1}{N}\sum (\log \pi_{\theta_{\textrm{old}}} - \log \pi_\theta)$ und die **Entropie der Strategie**:

<!--en:e0b13df8-->
::: {.slide title="Akteurupdates über mehrere Epochen und laufende Diagnose (Fortsetzung)" layout="tight" only="jax"}

<!--en:44740089-4-->
[Innere Epochenschleife]{.kicker}

<!--en:9474d9b5-->
Ein kompilierter Schritt liefert die vor dem Update gemessenen Kennzahlen. `ppo_epochs` führt `num_epochs` solche Schritte auf dem aufgefüllten Batch aus:

<!--en:297fc889-->
::: {.slide title="Die vollständige PPO-Trainingsschleife"}

<!--en:36914a72-->
[Äußere Schleife]{.kicker}

<!--en:1b2e0418-->
Jede äußere Iteration sammelt 8 Episoden und hält die standardisierten GAE-Vorteile (berechnet mit dem noch nicht angepassten Kritiker) und die alten Log-Wahrscheinlichkeiten fest. Dann passt sie den Kritiker in 20 Schritten an den $\lambda$-Ertrag an und führt `ppo_epochs` aus:

<!--en:a43b5020-->
::: {.slide title="Die vollständige PPO-Trainingsschleife (Fortsetzung)"}

<!--en:36914a72-2-->
[Äußere Schleife]{.kicker}

<!--en:d03c6635-->
In jeder Iteration sammeln wir Trajektorien, halten Vorteile und alte Log-Wahrscheinlichkeiten fest, passen den Kritiker neu an und führen dann die Epochen des Akteurs aus:

<!--en:d603e5e5-->
::: {.slide title="Die Wiederverwendung des Batchs halbiert die nötigen Umgebungsschritte"}

<!--en:a7187495-->
[Dateneffizienz]{.kicker}

<!--en:5e025b69-->
- Bei 8 Zufallsstartwerten und einem Budget von je $50{,}000$ Umgebungsschritten erreichen **4 Epochen** pro Batch den maximalen Ertrag ($475+$) in **$\sim 25{,}000$ Schritten**. Das sind halb so viele Schritte und Trajektorien wie bei 1 Epoche ($\ge 50{,}000$ Schritte).

<!--en:106c4043-->
- Bei 1 Epoche pro Batch gilt für jedes Verhältnis $\rho_t = 1$. Das Clipping greift nie, und PPO reduziert sich auf das Akteur-Kritiker-Verfahren mit GAE.

<!--en:09d7116d-->
- Der Wechsel von 4 zu 20 Epochen beschleunigt das Training nur wenig. Mit Clipping bleibt es dabei durchweg stabil.

<!--en:d87320f1-->
::: {.slide title="Ablationsstudie: Was geschieht bei 20 Epochen ohne Clipping?"}

<!--en:a4b31f09-->
[Ablationsstudie zum Clipping]{.kicker}

<!--en:68c49a63-->
- Bei Lernrate $10^{-2}$ und 20 Epochen pro Batch erreicht **jeder PPO-Lauf mit Clipping (8/8)** den Wert 500 und hält den Ertrag über 250.

<!--en:220b0d08-->
- **Ohne Clipping** treiben wiederholte unbeschränkte Durchläufe die Strategie in die Sättigung. Bei 64 Zufallsstartwerten hält nur $\sim 1$ von $20$ Läufen das Niveau, **die Hälfte scheitert** bei Ertrag $\sim 9$ (der Wagen fährt in dieselbe Richtung, bis der Stab fällt).

<!--en:55b34704-->
- Senken wir die Lernrate des Akteurs um den Faktor $10\times$ auf $10^{-3}$, bleibt der Einbruch ohne Clipping aus. Die Zahl der Schritte bis zur Konvergenz steigt jedoch von $24\text{k}$ auf $\sim 40\text{k}$.

<!--en:8c2d6e88-->
::: {.slide title="PPO überwachen: Anteil außerhalb des Bands und geschätzte KL-Divergenz"}

<!--en:c43d1777-->
[Kennzahlen für das Training]{.kicker}

<!--en:0979ecb9-->
- Da der Strategiegradienten-Verlust unabhängig vom Ertrag nahe null bleibt, verfolgen wir den **Anteil außerhalb des Bands** und die **geschätzte KL-Divergenz** über die 20 Epochen eines Batchs.

<!--en:9c2d058d-->
- In Epoche 1 gilt $\rho_t = 1$, also sind beide Kennzahlen $0$. Mit Clipping erreicht der Anteil außerhalb des Bands um Epoche 5 bis 6 seinen Höchstwert ($\sim 7\%$) und **sinkt bis Epoche 20 wieder auf die Hälfte**.

<!--en:d4b5f38f-->
- **Warum sinkt er?** Bei Beobachtungen jenseits der *ungünstigen* Bandgrenze ($\rho < 0.8$ für $\hat{A} > 0$) bleibt der Gradient ohne Clipping erhalten und zieht die Verhältnisse zurück ins Band.

<!--en:e23b8b0d-->
::: {.slide title="Abnahme der Strategieentropie mit und ohne Clipping"}

<!--en:0c7bed25-->
[Diagnose der Sättigung]{.kicker}

<!--en:f5886b19-->
- **Zunehmend eindeutige Entscheidungen:** Während PPO mit Clipping CartPole in 60 Updates zu beherrschen lernt, sinkt die Strategieentropie allmählich von $0.65$ auf $0.25\text{ nats}$. Der Bonus mit `entropy_coef = 0.01` bremst den Rückgang.

<!--en:18353dab-->
- **Sättigung ohne Clipping:** Die Strategieentropie fällt rasch gegen $0\text{ nats}$, weil wiederholte Durchläufe die Aktionswahrscheinlichkeiten zu $0$ und $1$ treiben.

<!--en:6c5dc029-->
::: {.slide title="Kollaps der Verhältnisse und effektive Stichprobengröße" layout="tight"}

<!--en:57b01190-->
[Diagnose der Wichtigkeitsgewichte]{.kicker}

<!--en:8d819336-->
- Nach 20 Durchläufen über denselben anfänglichen Batch liegen die Verhältnisse mit Clipping weiterhin dicht bei $1.0$. Ohne Clipping häufen sie sich nahe $0$ und $2 \approx 1/\pi_{\theta_{\textrm{old}}}(a^*)$.

<!--en:80d07de8-->
::: {.slide title="Kollaps der Verhältnisse und effektive Stichprobengröße (Fortsetzung)" layout="tight"}

<!--en:57b01190-2-->
[Diagnose der Wichtigkeitsgewichte]{.kicker}

<!--en:8470dad8-->
- **Effektive Stichprobengröße ($N_{\textrm{eff}} = 1 / \sum_j \bar{w}_j^2$):** Mit Clipping bleiben in Epoche 20 effektiv **$N_{\textrm{eff}}/N \approx 96\%$–$97\%$** des Batchs erhalten. Ohne Clipping senkt die Sättigung $N_{\textrm{eff}}/N$ auf etwa die Hälfte ($\approx 47\%$–$55\%$, nahe dem harmonischen Mittel von $\pi_{\theta_{\textrm{old}}}(a^* \mid s)$).

<!--en:134ff6d0-->
::: {.slide title="PPO im Produktiveinsatz: vektorisierte Trajektorien und Bootstrap an der Schnittstelle"}

<!--en:5012f40d-->
[Systemarchitektur]{.kicker}

<!--en:361bdcc6-->
Produktivimplementierungen (`cleanrl/ppo.py`) führen $N = 8$ parallele Umgebungen über einen festen Horizont $T = 32$ aus. So entsteht ein $32 \times 8$-Rechteck von Übergängen. Am Rand setzen sie den geschätzten Folgewert ein und trainieren mit zufällig gemischten Minibatches:

<!--en:58ff8598-->
**Zwei Fehlerquellen im Produktiveinsatz:** (1) Bei Schritt $T$ noch laufende Episoden benötigen den geschätzten Folgewert $\hat{V}(s_T) \approx 100$ anstelle von null. (2) In Gymnasium setzt `SyncVectorEnv` die Umgebung im Schritt *nach* dem Episodenende automatisch zurück. GAE muss diesen Übergang mit einer Maske ausschließen.

<!--en:f1135f78-->
::: {.slide title="Verständnisfragen zu Clipping-Schwellen und abgeschnittenen Trajektorien"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:7c8d0c29-->
Prüfen Sie Ihre Intuition anhand von drei Rechnungen:

<!--en:ac454583-->
1. **Identität beim ersten Durchlauf:** Die Vorteile $\hat{A}_t$ eines neuen Batchs sind auf Mittelwert $0$ und Varianz eins standardisiert. Welche Werte haben $\rho_t(\theta_{\textrm{old}})$, $L^{\textrm{CLIP}}(\theta_{\textrm{old}})$ und $\nabla_\theta L^{\textrm{CLIP}}(\theta_{\textrm{old}})$ in Epoche 1?

<!--en:9c0f1fc1-->
2. **Relative Bandbreite:** Für $\epsilon_{\textrm{clip}} = 0.2$ und $\hat{A}_t > 0$: Bis zu welcher neuen Wahrscheinlichkeit $\pi_\theta(a_t \mid s_t)$ belohnt die Zielfunktion eine Erhöhung, wenn die Aktion zuvor die Wahrscheinlichkeit $\pi_{\theta_{\textrm{old}}} = 0.01$ beziehungsweise $\pi_{\theta_{\textrm{old}}} = 0.60$ hatte?

<!--en:5e9dcf88-->
3. **Fehlender Folgewert an der Schnittstelle:** Eine Trajektorie endet nach 32 Schritten bei $\hat{V}(s_T) = 100$, $\gamma = 0.99$ und $\lambda = 0.95$. Um wie viel sinkt der GAE-Vorteil, wenn eine Maske die Schnittstelle irrtümlich als Endzustand behandelt ($\hat{V}(s_T) = 0$), und zwar bei Schritt $T-1$ und bei Schritt $T-11$?

<!--en:557b98b6-->
**Antworten:**

<!--en:e21ea8be-->
1. Für jedes Verhältnis gilt $\rho_t = 1$, also $L^{\textrm{CLIP}}(\theta_{\textrm{old}}) = \frac{1}{N}\sum \hat{A}_t = 0$ und $\nabla_\theta L^{\textrm{CLIP}}(\theta_{\textrm{old}}) = \frac{1}{N}\sum \hat{A}_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ (der exakte Strategiegradient für einen Schritt).

<!--en:f86e7d3e-->
2. Bis $1.2 \times 0.01 = \mathbf{0.012}$ (absoluter Zuwachs $+0.002$) für die seltene Aktion, bis $1.2 \times 0.60 = \mathbf{0.72}$ (Zuwachs $+0.12$) für die häufige Aktion.

<!--en:0ae2a3f8-->
3. Bei Schritt $T-1$ sinkt er um $\gamma \hat{V}(s_T) = \mathbf{99}$, zehn Schritte früher ($T-11$) um $(\gamma\lambda)^{10} \cdot 99 \approx \mathbf{54}$. Dadurch werden gute Aktionen vor der Schnittstelle fälschlich bestraft.

<!--en:96fd4c8a-->
::: {.slide title="Strategieoptimierung von REINFORCE bis PPO im Überblick"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:a33fcd14-->
| Verfahren | Datennutzung pro Batch | Begrenzung des Schritts | Optimierungsaufwand | Verhalten auf CartPole (Budget $50\text{k}$) |

<!--en:be83de85-->
| **Akteur-Kritiker + GAE** | $1$ Durchlauf ($\rho_t = 1$) | Optionale Begrenzung der Gradientennorm | $1$ Schritt erster Ordnung | Erreicht $\sim 450$ nach etwa $50{,}000$ Schritten |

<!--en:c3da831d-->
| **Wahrscheinlichkeitsverhältnis ohne Clipping** | $K$ Durchläufe mit $\rho_t \hat{A}_t$ | Keine (unbeschränktes Verhältnis) | $K$ Schritte erster Ordnung | Ertrag bricht auf $\sim 9$ ein bei $K=20, \eta=10^{-2}$ |

<!--en:fd3944ad-->
| **TRPO** | $1$ Schritt mit Liniensuche | Feste Schranke für mittlere KL-Divergenz $\bar{D}_{\textrm{KL}} \le \Delta_{\textrm{KL}}$ | Konjugierter Gradient ($F\mathbf{x}=\mathbf{g}$) | Monotoner Fortschritt, höherer Rechenaufwand pro Update |

<!--en:7a4ebd2a-->
| **PPO mit Clipping** | $4$–$20$ Epochen mit $L^{\textrm{CLIP}}$ | Einseitiges Clipping des Verhältnisses bei $1 \pm \epsilon_{\textrm{clip}}$ | $K$ Adam-Schritte erster Ordnung | Löst CartPole ($475+$) in $\sim 25{,}000$ Schritten |
