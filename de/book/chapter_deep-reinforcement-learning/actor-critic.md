<!--en:0984934b-->
# Akteur-Kritiker-Verfahren und Mehrschritterträge

<!--en:e10976f8-->
Der Agent aus :numref:`sec_deeprl` lernt, den Stab zu balancieren, doch er lernt mit Verzögerung. Vor jedem Update wartet er acht vollständige Episoden ab, in denen er den Stab bis zu 4000 Schritte lang balanciert. Erst dann prüft er für jede Aktion, ob der Rest der Episode besser verlief als erwartet. Der Q-Learning-Agent aus :numref:`sec_qlearning` brauchte weniger Geduld. Nach jedem Übergang änderte er einen Tabelleneintrag. Dabei ersetzte er den Rest der Episode durch seine eigene Schätzung des weiteren Verlaufs. Kann ein Agent mit Policy-Gradienten genauso lernen?

<!--en:4943fe0b-->
Ja, und er hat bereits alles, was er dazu braucht. Die gelernte Baseline $\hat{V}$ aus :numref:`sec_deeprl` sagt den künftigen Ertrag (engl. reward-to-go) voraus, also die diskontierte Summe der Belohnungen, die auf einen Zustand folgen. Wenn sie den Rest der Episode vorhersagt, braucht der Agent ihn nicht abzuwarten. Eine Belohnung plus die Vorhersage für den nächsten Zustand können den beobachteten Rest der Episode ersetzen. Die Strategie (engl. policy) übernimmt damit die Rolle des *Akteurs* (engl. actor), die Wertschätzung die des *Kritikers* (engl. critic). :citet:`Barto.Sutton.Anderson.1983` entwickelten das erste derartige Paar für genau dieses Wagen-Stab-Problem. Ihr „adaptive critic element“ gab dem Kritiker seinen Namen.

<!--en:5053891f-->
Mit dieser Ersetzung tritt an die Stelle der zufälligen künftigen Belohnungen der Fehler des Kritikers. Wir untersuchen, was ein einzelner Übergang dem Akteur verrät und wie viel vom beobachteten Rest der Episode wir beibehalten sollten. Auf CartPole zeigt sich, dass der Fehler des Kritikers stärker ins Gewicht fällt als die eingesparte Varianz. Die meisten Verfahren im weiteren Verlauf dieses Kapitels verwenden eine Form des Kritikers. Der Schätzer, den wir hier entwickeln, mischt Einschritt- und Mehrschrittziele mit geometrisch abnehmenden Gewichten. Auch PPO verwendet ihn.

<!--en:1561e3e6-->
Wir behalten die Aufgabe und den Agenten aus :numref:`sec_deeprl` bei. Vier Zahlen beschreiben den Zustand in CartPole, nämlich Position und Geschwindigkeit des Wagens sowie Winkel und Winkelgeschwindigkeit des Stabs. Mit den beiden Aktionen schiebt der Agent den Wagen nach links oder rechts. Jeder Schritt mit aufrechtem Stab bringt die Belohnung $1$. Nach 500 Schritten bricht die Episode ab, der höchstmögliche Ertrag beträgt also 500. Der Diskontierungsfaktor ist $\gamma = 0.99$. Der Agent `ActorCritic.mlp` besteht aus einem Strategienetz und einem Wertnetz, jeweils mit einer versteckten Schicht aus 64 Einheiten und einem eigenen Adam-Optimierer. Die Hauptversuche umfassen 150 Updates mit je acht Episoden, also bis zu 600 000 Umgebungsschritte. Der neue Hyperparameter `critic_steps` bestimmt, wie viele Regressionsdurchläufe der Kritiker pro Batch macht. Mit `grad_clip` begrenzen wir die Norm des Gradienten des Akteurs. Beide Parameter erläutern wir anhand der Trainingsschleife.

<!--en:43e6495a-->
## Vorteilsschätzung mit zeitlichen Differenzen

<!--en:747d6f24-->
### Bootstrapping für den künftigen Ertrag

<!--en:ab8eec65-->
Um zu sehen, was der Kritiker ersetzen kann, schreiben wir den ersten Schritt des künftigen Ertrags explizit hin:

<!--en:019d352c-->
Wir passen die gelernte Baseline durch Regression an den künftigen Ertrag an. Damit sagt $\hat{V}(s_{t+1})$ genau die Größe voraus, für die $\hat{G}_{t+1}$ einen Stichprobenwert liefert. Statt diesen Wert abzuwarten, setzen wir die Vorhersage ein. Es gilt also $\hat{G}_t \approx r_t + \gamma \hat{V}(s_{t+1})$. Das Gewicht des Scores in Schritt $t$ war $\hat{G}_t - \hat{V}(s_t)$. Durch die Ersetzung ergibt sich

<!--en:9d170512-->
mit $\hat{V}(s_{t+1}) = 0$, wenn $s_{t+1}$ ein Endzustand ist. Dieser *Fehler der zeitlichen Differenz* (engl. temporal-difference error, TD-Fehler) vergleicht zwei Vorhersagen desselben Ertrags. Die erste stammt aus Zustand $s_t$, die zweite aus dem darauffolgenden Schritt, für den bereits eine tatsächliche Belohnung vorliegt.

<!--en:2901c000-->
Der TD-Fehler aus :numref:`sec_qlearning` hatte dieselbe Form, mit einem Unterschied. Q-Learning verwendet für das Bootstrapping die beste Fortsetzung $\max_{a'} \hat{Q}(s', a')$, weil es die optimalen Werte schätzt. Hier verwenden wir wie beim SARSA-Ziel im selben Abschnitt den Fortsetzungswert der aktuellen Strategie. Die Maske für Endzustände richtet sich nach `terminated`, nicht nach `truncated`. Ein Zeitlimit beendet die Beobachtung, doch die Aufgabe läuft weiter. Der Zustand am Abbruchpunkt hat daher einen von null verschiedenen Wert (:numref:`sec_mdp`). Die Funktion `td_target` berechnet das Ziel $r_t + \gamma \hat{V}(s_{t+1})$ für jeden Übergang eines `Batch`. Diese flache Datenstruktur enthält die Übergänge und Episodengrenzen, die `d2l.rollout` aufzeichnet (:numref:`sec_policygradient`). Zur Berechnung genügt eine beliebige Funktion, die Zuständen Werte zuordnet. Die Rückgabe ist ein NumPy-Array, sodass der Gradient des Kritikers nicht durch das Ziel fließt.

<!--en:d864a1b0-->
Wie beim Schätzer für den künftigen Ertrag aus :numref:`sec_baselines` entfällt beim Update des Akteurs der Faktor $\gamma^t$, der späte Schritte einer Episode gegenüber frühen diskontieren würde.

<!--en:8cc3e9a3-->
### Ein Übergang von Hand

<!--en:4f71f245-->
Drei Zahlen zeigen, was das neue Gewicht aussagt. Angenommen, der Kritiker sagt für jeden Zustand $100$ voraus. Das ist der Wert $1/(1-\gamma)$ für unbegrenzt langes Balancieren bei $\gamma = 0.99$. Bleibt der Stab bei einem Schritt aufrecht, gilt $\delta_t = 1 + 0.99 \cdot 100 - 100 = 0$. Der Schritt verlief wie vorhergesagt, die gewählte Aktion wird weder begünstigt noch benachteiligt. Fällt der Stab, endet die Episode. Die Maske setzt den Bootstrap-Wert auf null, und es gilt $\delta_t = 1 - 100 = -99$. Die Wahrscheinlichkeit der Aktion, die die Episode beendete, sinkt dadurch stark.

<!--en:0fd3af25-->
Ein frisch initialisierter Kritiker sagt überall Werte nahe $0$ voraus. Dann ergibt jeder Schritt $\delta_t \approx 1$, unabhängig davon, ob der Stab bei der Aktion aufrecht blieb oder fiel. Jede gewählte Aktion wird gleichermaßen begünstigt. Der Akteur lernt wenig, bis der Kritiker die beiden Fälle unterscheiden kann.

<!--en:9f08753e-->
### Ein exakter Kritiker liefert den Vorteil

<!--en:27fec17d-->
Mit einem perfekten Kritiker schätzt der TD-Fehler dieselbe Größe wie das Monte-Carlo-Gewicht. Nehmen wir kurz an, es gelte $\hat{V} = V^\pi$. Der Aktionswert erfüllt $Q^\pi(s, a) = r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')$. Das ist die Bellman-Gleichung aus :numref:`sec_valueiter`, ausgedrückt durch $V^\pi$. Mitteln wir :eqref:`eq_td_error_v` über den nächsten Zustand, erhalten wir daher

<!--en:ec8ef3cc-->
Die rechte Seite ist der Vorteil $A^\pi(s_t, a_t)$. Er gibt an, um wie viel die gewählte Aktion besser ist als eine durchschnittliche Aktion der Strategie in diesem Zustand. Mit der gelernten Baseline aus :numref:`sec_baselines` schätzten wir ihn durch $\hat{G}_t - \hat{V}(s_t)$. Ein Übergang mit einer Belohnung und zwei Auswertungen des Kritikers liefert eine erwartungstreue Schätzung des Vorteils. Das Monte-Carlo-Gewicht brauchte dafür den gesamten Rest der Episode.

<!--en:c7d73386-->
Während des Trainings weicht $\hat{V}$ von $V^\pi$ ab. Dann ist $\delta_t$ ein verzerrter Schätzer des Vorteils. Nach der Identität zum Erwartungswert null aus :numref:`sec_baselines` dürfen wir vom Gewicht jede Größe abziehen, die vor der Aktion feststeht. Der Erwartungswert des Gradienten bleibt dabei gleich. Das gilt für $\hat{V}(s_t)$, weil dieser Wert nur vom aktuellen Zustand abhängt. Für $\hat{V}(s_{t+1})$ gilt es nicht, denn der nächste Zustand hängt von der Aktion ab. Der Fehler des Kritikers in den Zuständen, zu denen die Aktionen führen, verzerrt also das Update (Aufgabe :exref:`ex_actor-critic-why-baseline-argument-does`). Diese Verzerrung kommt zu der hinzu, die bereits durch das Weglassen von $\gamma^t$ entsteht (:numref:`sec_baselines`).

<!--en:96072dab-->
Das Update des Kritikers ist eine stichprobenbasierte Strategiebewertung. Bei der Strategiebewertung in :numref:`sec_valueiter` verwendeten wir das Bellman-Backup mit Erwartungswert und Fixpunkt $V^\pi$. Passen wir $\hat{V}(s_t)$ durch Regression an $r_t + \gamma \hat{V}(s_{t+1})$ an, ersetzen wir diesen Erwartungswert durch einen beobachteten Übergang. Analog dazu ersetzte Q-Learning das Optimalitäts-Backup durch einen Stichprobenwert. Der Kritiker schätzt den Wert der aktuellen Strategie, und der Akteur verbessert die Strategie anhand dieser Schätzung. Das Akteur-Kritiker-Verfahren ist also eine Näherung an die verallgemeinerte Strategieiteration aus :numref:`sec_valueiter`. Weil das Regressionsziel selbst $\hat{V}$ enthält, berechnen wir den Gradienten nur über die Vorhersage $\hat{V}(s_t)$ und halten das Ziel dabei fest. Ein solches Update heißt *Semigradientenschritt*. Sein Erwartungswert ist für keine feste Zielfunktion deren Gradient.

<!--en:99f26d2f-->
### Eine Kette als Beispiel

<!--en:31745466-->
Wie viel Varianz spart das Bootstrapping ein, und wie viel Verzerrung führt es ein? Für eine kurze Kette lassen sich beide Größen in geschlossener Form angeben. Betrachten wir zehn Zustände $0, 1, \ldots, 9$. Vom Zustand $s$ wechselt der Agent immer nach $s+1$. Der Übergang aus Zustand $9$ beendet die Episode und bringt die Belohnung $1$. Jede Belohnung enthält unabhängiges gaußsches Rauschen mit Standardabweichung $\sigma = 0.15$, und es gilt $\gamma = 0.97$. Der Fehler des Kritikers beträgt $e(s) = 0.5\,(1 - s/9)^2$. Er ist am Anfang groß und am Ende null. Beim TD-Lernen mit Initialisierung bei null ergeben sich häufig Fehler dieser Form, weil die Belohnung am Ende zuerst die späten Zustände erreicht.

<!--en:92c5f040-->
Ein Ziel, das $n$ Belohnungen beobachtet und dann den Wert des Kritikers im Zustand $n$ einsetzt, übernimmt dessen dortigen Fehler in diskontierter Form. Hinzu kommt das Rauschen von $n$ Belohnungen. Für Verzerrung und Varianz gilt

<!--en:ab8a113b-->
für $n \leq 9$. Das Monte-Carlo-Ziel bei $n = 10$ ist erwartungstreu. Bei $n = 1$ beträgt die Verzerrung $0.38$ und die Varianz $0.02$. Bei $n = 10$ ist die Verzerrung null und die Varianz beträgt $0.17$. Der mittlere quadratische Fehler ist die Summe aus quadrierter Verzerrung und Varianz. Er sinkt von $0.17$ bei $n = 1$ auf $0.10$ bei $n = 4$ und steigt dann wieder auf $0.17$ bei $n = 10$.

<!--en:122b1dfc-->
Die beste Tiefe liegt im Inneren des Bereichs, weil tiefere Ziele für das Bootstrapping Zustände nutzen, deren Werte der Kritiker besser kennt. Hätte der Kritiker in jedem Zustand denselben Fehler $0.5$, verliefe die Fehlerkurve über alle Tiefen mit Bootstrapping monoton (Aufgabe :exref:`ex_actor-critic-optimal-bootstrap-depth`). :numref:`fig_rl_td_mc_spectrum` zeigt die Familie der Ziele und die gemessenen Kurven.

<!--en:604231de-->
![Ziele mit Bootstrapping. Links sind die Backup-Diagramme für die Tiefen eins, zwei und $n$ sowie bis zur Terminierung dargestellt. Der grüne Knoten markiert den Zustand, an dem das Sampling endet und der Kritiker den Rest schätzt. Die Gewichte $(1-\lambda)\lambda^{n-1}$ stammen aus dem $\lambda$-Ertrag dieses Abschnitts für $\lambda = 0.9$. Rechts sind Verzerrung, Varianz und mittlerer quadratischer Fehler des Ziels der Tiefe $n$ für die Kette aus zehn Zuständen dargestellt. Die Messung über $20{,}000$ Rollouts ergibt den kleinsten Fehler bei $n = 4$.](../img/mdl-rl-td-mc-spectrum.svg)

<!--en:20898e9e-->
## Das Akteur-Kritiker-Update

<!--en:ddede46d-->
### Zwei Updates aus einer Zahl

<!--en:cc5eb809-->
Beide Netze lernen nun aus demselben Skalar. Wir schreiben $w$ für die Parameter des Kritikers und $\hat{V}_w$ für den Kritiker selbst. Nach dem Übergang $(s_t, a_t, r_t, s_{t+1})$ lauten die beiden Updates

<!--en:ff5bbda5-->
mit der Schrittweite $\alpha$ für den Kritiker wie beim TD-Lernen in :numref:`sec_qlearning` und der Schrittweite $\eta$ für den Akteur wie beim Policy-Gradienten in :numref:`sec_policygradient`. Der Kritiker macht einen Semigradientenschritt in Richtung des Einschrittziels. Der Akteur macht den Policy-Gradienten-Schritt aus :numref:`sec_baselines` mit $\delta_t$ als Gewicht. Eine Zahl erfüllt beide Aufgaben. Der Wert $\delta_t$ zeigt dem Kritiker, wie weit seine Vorhersage danebenlag. Dem Akteur zeigt er, ob die gewählte Aktion besser oder schlechter ausging als vom Kritiker erwartet (:numref:`fig_rl_actor_critic_loop`). Weil $\delta_t$ nur einen Übergang benötigt, können wir :eqref:`eq_actor_critic` bei jedem Schritt anwenden, während der Agent handelt. So aktualisierte auch Q-Learning seine Tabelle. Das Lernen kann schon vor der ersten Terminierung beginnen.

<!--en:f4b5f9c3-->
![Die Akteur-Kritiker-Schleife. Ein Übergang liefert einen TD-Fehler. Er korrigiert die Vorhersage des Kritikers und gewichtet den Schritt des Akteurs.](../img/mdl-rl-actor-critic-loop.svg)

<!--en:e8e0fe5e-->
Das Ziel des Kritikers ändert sich mit beiden Netzen. Ein Schritt des Akteurs verändert die Werte, die der Kritiker vorhersagen soll. Ein Schritt des Kritikers verändert seine eigenen Ziele. Ändert sich der Akteur schneller, als der Kritiker folgen kann, gewichten die Fehler eines veralteten Kritikers den Schritt des Akteurs. :citet:`Konda.Tsitsiklis.2000` bewiesen die Konvergenz des Akteur-Kritiker-Verfahrens für lineare Kritiker und abklingende Schrittweiten unter einer Bedingung. Die Schrittweiten des Kritikers müssen langsamer abnehmen als die des Akteurs, damit der Kritiker mit der aktuellen Strategie Schritt hält. In der folgenden Schleife erzielen wir denselben Effekt mit mehreren Regressionsdurchläufen des Kritikers pro Batch.

<!--en:7c44dbb7-->
### Die Trainingsschleife mit Batches

<!--en:b092719b-->
Der Code wendet :eqref:`eq_actor_critic` nicht nach jedem Übergang an. Wir behalten die Batches aus je acht Episoden aus :numref:`sec_deeprl` bei und ändern die Verarbeitung jedes Batchs. Zunächst passt sich der Kritiker in `critic_steps` Regressionsdurchläufen an die Einschrittziele des Batchs an. Vor jedem Durchlauf berechnen wir die Ziele mit dem aktuellen Kritiker neu. Eine Regression auf Ziele, in die die Vorhersagen des Regressionsmodells selbst eingehen, heißt *Fitted TD*. Durch die Neuberechnung ändern sich die Ziele von Durchlauf zu Durchlauf. Anschließend macht der Akteur einen Schritt, gewichtet mit den TD-Fehlern des neu angepassten Kritikers. Wir bezeichnen diese Schleife als Einschritt-Akteur-Kritiker-Verfahren mit Batches. Der bekannteste Verwandte ist Advantage Actor-Critic von :citet:`Mnih.Badia.Mirza.ea.2016`. Dieses Verfahren gewichtet denselben Score mit $n$-Schritt-Vorteilen und addiert einen Entropiebonus. In der ursprünglichen Version A3C arbeiteten viele Prozesse asynchron. Die heute übliche synchrone Variante *A2C* sammelt parallel Daten aus vielen Umgebungen und macht pro Rollout einen gemeinsamen Gradientenschritt.

<!--en:1f4f8c02-->
Als Gewicht des Akteurs verwenden wir den rohen TD-Fehler. Anders als das Monte-Carlo-Gewicht in :numref:`sec_deeprl` normalisieren wir ihn innerhalb des Batchs nicht. Der Grund liegt darin, wie Adam seine Schritte skaliert. Adam dividiert jede Gradientenkoordinate durch die Quadratwurzel eines gleitenden Mittels ihrer bisherigen Quadrate. Bei der voreingestellten Abklingrate erfasst dieses Mittel ungefähr tausend Schritte, also mehr als unsere 150 Updates. Daher sind die ersten Schritte für jede Gewichtsskala etwa gleich groß. Spätere Schritte werden erst kleiner, wenn die Gradienten unter ihre anfängliche Größe fallen. Mit zunehmender Genauigkeit des Kritikers sinkt die Standardabweichung der TD-Fehler von etwa $3$ zu Beginn des Trainings auf $0.1$ oder weniger nahe der Ertragsobergrenze. Dort geht sie vor allem auf das Rauschen des Kritikers selbst zurück. Auch die Schritte von Adam werden im Laufe eines Trainingslaufs kleiner, etwa um den Faktor acht. Normalisierte TD-Fehler würden den Gradienten nahe seiner anfänglichen Größe halten. Adam würde weiterhin große Schritte in Richtungen machen, die das Rauschen vorgibt (Aufgabe :exref:`ex_actor-critic-gradient-norms`).

<!--en:6a7b0bff-->
Zu Beginn des Trainings sind die rohen Gewichte und damit der Gradient des Akteurs größer als bei normalisierten Gewichten. Wir begrenzen die Gradientennorm auf den üblichen Wert $0.5$. Bei Adam begrenzt das nicht die Schrittlänge, sondern den Einfluss der anfänglich großen Gradienten auf die gleitenden Mittelwerte. Die Funktion `policy_step_clip` erweitert `policy_step` aus :numref:`sec_imitation` um diese Begrenzung. Sie macht einen Gradientenschritt auf den gewichteten Log-Wahrscheinlichkeiten der Aktionen des Batchs und gibt die Norm vor der Begrenzung zurück.

<!--en:1830f367-->
Den Verlust allein berechnet `actor_loss`. Diese Funktion verwenden wir bei der Messung am Ende des Abschnitts erneut.

<!--en:66a1198b-->
Die Funktion `policy_step_clip` füllt ihren Batch mit `d2l.padded` auf die Länge der jeweiligen Größenklasse auf, wie es auch `d2l.policy_step` tut. So lässt sich die kompilierte Funktion `_actor_pass` für Batches unterschiedlicher Länge wiederverwenden. Die Maske schließt die aufgefüllten Zeilen vom Verlust aus. Den Gradienten allein berechnet `actor_grads`. Diese Funktion verwenden wir bei der Messung am Ende des Abschnitts erneut.

<!--en:57ed6861-->
Die Trainingsschleife steht in `train`. Bei jedem Update sammelt sie mit `d2l.rollout` einen Batch. Von der Funktion `weight(ac, b)` erhält sie für jeden Schritt des Batchs `b` das Gewicht des Akteurs. Mit diesen Gewichten macht der Akteur einen Schritt mit begrenzter Gradientennorm. Die Schleife liefert die bisherige Zahl der Umgebungsschritte, den mittleren Ertrag des Batchs und die Gradientennorm des Akteurs vor der Begrenzung. Die Gewichtsfunktion passt auch den Kritiker an. So bleibt die Schleife für alle Schätzer dieses Abschnitts gleich, nur das Gewicht ändert sich. Das Gewicht im Akteur-Kritiker-Verfahren berechnet `td_weight`. Dabei macht `d2l.fit_value` (:numref:`sec_deeprl`) einen Adam-Schritt auf dem quadratischen Fehler zwischen den Vorhersagen des Kritikers und einem Zielvektor. Die Funktion `ac.value_np` wertet den Kritiker für einen Batch von Zuständen aus. Nach den Durchläufen des Kritikers dient sein neuer roher TD-Fehler als Gewicht.

<!--en:d2c40009-->
Wir haben uns dafür entschieden, den Kritiker anzupassen, bevor wir die Gewichte des Akteurs berechnen. A2C und die meisten GAE-Implementierungen berechnen die TD-Fehler mit dem Stand des Kritikers zum Zeitpunkt der Datenerhebung. Erst danach aktualisieren sie beide Netze. Bei einem Kritikerschritt pro Rollout wie in A2C spielt die Reihenfolge kaum eine Rolle. Bei zwanzig Durchläufen pro Batch kann sie wichtig sein. Für die Zustände des Batchs liegt der neu angepasste Kritiker viel näher an $V^\pi$. Dadurch geht weniger von seinem Fehler in die Gewichte des Akteurs ein, die hier ganz vom Kritiker abhängen (Aufgabe :exref:`ex_actor-critic-critic-passes` vergleicht beide Reihenfolgen).

<!--en:efed8168-->
Neue Batches helfen dem Kritiker noch auf andere Weise. Jeder Batch stammt aus der aktuellen Strategie und enthält die Zustände, die sie besucht. Ist ein Fehler des Kritikers für das aktuelle Verhalten von Bedeutung, enthält der nächste Batch daher wahrscheinlich einschlägige Übergänge. Diese Rückkopplung garantiert keine Konvergenz. TD-Lernen mit Semigradienten und einem nichtlinearen Kritiker kann selbst auf Daten der eigenen Strategie divergieren :cite:`Tsitsiklis.VanRoy.1997`. Trainieren wir mit gespeicherten Übergängen älterer Strategien, vereint dasselbe Update Funktionsapproximation, Bootstrapping und Off-Policy-Daten. Das ist die *tödliche Triade* von :citet:`Sutton.Barto.2018`, bei der Divergenz häufig auftritt. Diesen Fall behandelt :numref:`sec_dqn`.

<!--en:1fcf03a6-->
### Bootstrap- und Monte-Carlo-Gewichte auf CartPole

<!--en:0761e890-->
Lernt der Agent mit dem Bootstrap-Gewicht schneller als mit dem Monte-Carlo-Gewicht, das es ersetzt? Als Vergleich dient REINFORCE mit gelernter Baseline, also die Schleife aus :numref:`sec_deeprl`. Wir schreiben sie als zweite Gewichtsfunktion `mc_weight`, sodass beide Varianten ansonsten denselben Code nutzen. Zwei weitere Details sorgen dafür, dass sich nur das Gewicht unterscheidet. Der Akteur macht seinen Schritt mit `policy_step_clip`, und der Kritiker erhält dieselben `critic_steps` Regressionsdurchläufe pro Batch, hier mit den festen Monte-Carlo-Zielen $\hat{G}_t$. Wie in :numref:`sec_deeprl` addiert `truncation_bootstrap` beim Abbruch am Zeitlimit den diskontierten Wert des Kritikers zum künftigen Ertrag. Die Funktion `d2l.standardize` standardisiert die Gewichte innerhalb des Batchs. Wir berechnen die Gewichte vor den Durchläufen des Kritikers. Da Akteur und Kritiker getrennte Netze sind, ändert diese Reihenfolge den Schritt des Akteurs nicht.

<!--en:980df5ed-->
Wir trainieren jede Variante mit drei Zufallsstartwerten und tragen den mittleren Ertrag des Batchs gegen die Zahl der Umgebungsschritte auf. Diese Zahl dient im weiteren Kapitel als Maß für den Aufwand. Die Funktion `d2l.plot_curves` (:numref:`sec_utils`) zeichnet den Median über die Zufallsstartwerte und ein Band vom kleinsten bis zum größten Ergebnis. Je nach Zufallsstartwert variiert die Zahl der Schritte pro Update. Daher interpoliert `d2l.on_grid` zunächst jeden Lauf auf ein gemeinsames Raster von Schrittzahlen.

<!--en:3b5e8dd5-->
Nun trainieren wir die Variante `'actor-critic'` mit `td_weight` für dieselben drei Zufallsstartwerte. Wir interpolieren beide Varianten auf das gemeinsame Schrittraster `grid` mithilfe von `d2l.on_grid` und zeichnen den Median des Batchertrags und das Band zwischen Minimum und Maximum über die Zufallsstartwerte:

<!--en:f16fd4b4-->
Für jeden Zufallsstartwert zählen wir, wie viele Schritte der Lauf braucht, bis der mittlere Ertrag seiner letzten zehn Batches erstmals 450 erreicht. Außerdem zählen wir, wie oft die Begrenzung des Akteursgradienten greift. Die Funktion `steps_to` findet den ersten solchen Schritt in den Datenzeilen eines Laufs. In :numref:`sec_sac` verwenden wir sie erneut.

<!--en:0312f550-->
Wenden wir `steps_to(run, 450, 10)` auf jeden Zufallsstartwert in `runs` an, sehen wir für beide Varianten, wie viele Tausend Umgebungsschritte sie für einen mittleren Ertrag von 450 über zehn Batches brauchen. Dazu kommt der Anteil der Updates, bei denen die Gradientennorm des Akteurs über `grad_clip` lag:

<!--en:b9e5d5a6-->
Für mindestens zwei der drei gezeigten Zufallsstartwerte erreicht das Monte-Carlo-Gewicht zuerst die Obergrenze. Drei Zufallsstartwerte sind eine kleine Stichprobe. Bei 40 Zufallsstartwerten und denselben zwanzig Kritikerdurchläufen pro Batch erreichte REINFORCE in mindestens 35 Fällen zuerst einen Ertrag von 450. Typischerweise brauchte es dafür etwa ein Sechstel bis ein Drittel weniger Umgebungsschritte. Keiner der REINFORCE-Läufe endete weit unter der Obergrenze, beim Akteur-Kritiker-Verfahren waren es zwei. Die Gradientenbegrenzung greift bei einem bis zwei Fünfteln der Akteur-Kritiker-Updates. Bei REINFORCE greift sie nie, da die normalisierten Gewichte den Gradienten klein halten.

<!--en:a9273f04-->
Auf CartPole macht das Bootstrapping die Läufe nicht einmal gleichmäßiger. Die Belohnung pro Schritt beträgt deterministisch $+1$. Außerdem ersetzt das Bootstrapping am Zeitlimit den langen Rest jeder abgebrochenen Episode bereits durch den Wert des Kritikers. Das Monte-Carlo-Gewicht rauscht daher viel weniger als im ungünstigsten Fall. Was bleibt, ist die Verzerrung durch den Kritiker. Sie ist am größten, solange der Kritiker noch wenig trainiert ist. Genau in dieser Phase fällt das Akteur-Kritiker-Verfahren zurück. Der Extremfall ist der überall gleiche Wert $\delta_t \approx 1$ eines untrainierten Kritikers.

<!--en:d1790810-->
### Beide Gewichte auf denselben Batches

<!--en:f72fef94-->
Was ändert das Bootstrapping, wenn es hier das Lernen nicht beschleunigt? Wir messen beide Gewichte auf denselben Batches. Die folgende Zelle wiederholt das Akteur-Kritiker-Verfahren für einen Zufallsstartwert mit einer Gewichtsfunktion, die `td_weight` aufruft. Bei jedem Update zeichnet sie zusätzlich die Standardabweichungen des TD-Fehlers und des Monte-Carlo-Vorteils `A_mc` auf. Letzterer ist das Gewicht aus `mc_weight` vor der Standardisierung.

<!--en:72302ea0-->
Anhand von `trace` vergleichen wir den Median der Standardabweichung des TD-Fehlers mit dem des Monte-Carlo-Vorteils, jeweils für die ersten 10 und die letzten 50 Updates. Wir berechnen den Median des Varianzverhältnisses und tragen beide Varianzen pro Batch auf einer logarithmischen Skala auf:

<!--en:ba2de90c-->
Für dieselben Übergänge ist die Standardabweichung des Monte-Carlo-Vorteils bei den ersten Updates zwei- bis dreimal so groß wie die des TD-Fehlers. Die Varianz ist fünf- bis zehnmal so groß, über den gesamten Lauf sogar einige Dutzend Mal so groß. Die Grafik hat eine logarithmische Skala. Der TD-Fehler hängt von einer Belohnung und einem Übergang ab, der Monte-Carlo-Vorteil von jeder verbleibenden Belohnung und jedem verbleibenden Übergang der Episode. Beide Varianzen sinken stark, sobald fast jede Episode die Grenze von 500 Schritten erreicht. Beim TD-Fehler liegt das daran, dass der Kritiker inzwischen genau schätzt. Beim Monte-Carlo-Vorteil ersetzt das Bootstrapping den Rest jeder abgebrochenen Episode durch den Wert des Kritikers.

<!--en:f0bcf2cd-->
Das Bootstrapping senkt die Varianz tatsächlich. Auf CartPole begrenzt diese jedoch nicht das Lernen. REINFORCE mittelt bei jedem Update über acht Episoden und normalisiert seine Gewichte. Mit einer gut angepassten Baseline genügt das. Die Verzerrung durch den Kritiker hingegen bremst jedes Update, bis der Kritiker aufgeholt hat.

<!--en:ab85078d-->
Der bleibende Vorteil des Bootstrappings liegt im Aufbau des Updates. Für den TD-Fehler genügt ein Übergang, sodass wir schon vor dem Episodenende aktualisieren können. Bei einer fortlaufenden Aufgabe liegt der vollständige Ertrag nie vor. Dauern die Episoden Tausende von Schritten, brauchen acht Episoden pro Update Zehntausende von Schritten. Hier bleibt nur ein Kritiker (in Aufgabe :exref:`ex_actor-critic-fully-online` entwickeln Sie die vollständig online arbeitende Variante). Zwischen den Extremen lässt sich die Zahl der Schritte einstellen. Setzen wir den Bootstrap-Wert erst nach mehreren Schritten ein, bleibt ein Großteil der Varianzreduktion erhalten, während die Verzerrung durch den Kritiker geringer ausfällt.

<!--en:edef40c8-->
## Mehrschritterträge und GAE

<!--en:46000c1f-->
### n-Schritt-Erträge

<!--en:aa091060-->
Das Einschrittziel und der Monte-Carlo-Ertrag bilden die Endpunkte einer Familie von Schätzern. Wir berücksichtigen $n$ beobachtete Belohnungen und setzen danach den Wert des Kritikers ein:

<!--en:bae22576-->
Als Schätzung des Vorteils verwenden wir $\hat{G}^{(n)}_t-\hat{V}(s_t)$. Für $n=1$ ergibt sich $\delta_t$. Reicht $n$ bis zum Ende der Episode, erhalten wir den Monte-Carlo-Vorteil. Jeder zusätzliche Schritt diskontiert den Fehler des Kritikers mit einem weiteren Faktor $\gamma$ und wertet ihn in einem späteren Zustand aus. Zugleich kommen eine beobachtete Belohnung und ein beobachteter Übergang hinzu. Diesen Zielkonflikt kennen wir aus dem Kettenbeispiel, wo $n = 4$ den besten Kompromiss ergab.

<!--en:b3c92ca0-->
### Der $\lambda$-Ertrag und die Teleskopidentität

<!--en:16327494-->
Statt eine Tiefe zu wählen, können wir über alle Tiefen mitteln. Mit geometrischen Gewichten lässt sich dieser Mittelwert effizient berechnen:

<!--en:c7187455-->
Die Gewichte $(1-\lambda)\lambda^{n-1}$ summieren sich zu eins. Bei $\lambda = 0$ liegt das gesamte Gewicht auf der Einschrittschätzung. In einer Episode, die terminiert, liefert jede Tiefe jenseits des Endes den Monte-Carlo-Vorteil. Für $\lambda \to 1$ strebt die Mischung daher gegen den Monte-Carlo-Vorteil. In dieser Darstellung scheint es, als müssten wir jede Tiefe gesondert berechnen. Das erübrigt sich, weil sich die Mischung vereinfachen lässt.

<!--en:4aa5f413-->
**Proposition.** Die Mischung ist eine diskontierte Summe von TD-Fehlern:

<!--en:42612293-->
**Beweis.** Für jede Tiefe ergibt sich eine Teleskopsumme von TD-Fehlern. Es gilt $\sum_{l=0}^{n-1} \gamma^l \delta_{t+l} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n \hat{V}(s_{t+n}) - \hat{V}(s_t) = \hat{G}^{(n)}_t - \hat{V}(s_t)$, da sich die inneren Terme mit $\hat{V}$ paarweise aufheben. Setzen wir dies in :eqref:`eq_lambda_return` ein und vertauschen die beiden Summen. Das Paar $(n, l)$ mit $l < n$ liefert den Beitrag $(1-\lambda)\lambda^{n-1} \gamma^l \delta_{t+l}$, also gilt $\hat{A}^{\textrm{GAE}}_t = \sum_{l \ge 0} \gamma^l \delta_{t+l}\, (1-\lambda) \sum_{n \ge l+1} \lambda^{n-1}$. Die innere geometrische Summe ergibt $\lambda^l$. $\blacksquare$

<!--en:8f706c1a-->
Die Identität macht aus dem Mittelwert über alle Tiefen einen einzigen Rückwärtsdurchlauf durch die Episode. Der Aufwand wächst linear mit ihrer Länge. So berechnen Implementierungen den Schätzer :cite:`Schulman.Moritz.Levine.ea.2016`. Bei einer episodischen Aufgabe enden die Summen mit der Terminierung. Dort gilt $\hat{V} = 0$, und es folgen keine Belohnungen mehr. Somit ist jeder TD-Fehler $\delta$ nach dem Ende null, und die unendliche Summe stimmt mit der endlichen überein.

<!--en:46601a7f-->
### GAE als Rückwärtsdurchlauf

<!--en:17bbaeda-->
Die Summe :eqref:`eq_gae_deltas` liefert einen Algorithmus für die *verallgemeinerte Vorteilsschätzung* (engl. generalized advantage estimation, GAE). Wir berechnen die TD-Fehler und summieren sie rückwärts mit dem Faktor $\gamma\lambda$. An jeder Episodengrenze beginnen wir erneut. Die Methode `Batch.backward_scan` aus :numref:`sec_baselines` verwendet diese Rekursion bereits, um mit $\hat{G}_t = r_t + \gamma \hat{G}_{t+1}$ den künftigen Ertrag zu berechnen. Nur die Eingabe und der Faktor ändern sich.

<!--en:8660bcf5-->
An den beiden Endpunkten ergeben sich Identitäten, die wir an einem Batch prüfen können. Bei $\lambda = 0$ liefert der Rückwärtsdurchlauf die TD-Fehler selbst. Bei $\lambda = 1$ reicht die Teleskopsumme in Episoden, die terminieren, bis zur Terminierung. Sie liefert den künftigen Ertrag abzüglich der Baseline, also den Monte-Carlo-Vorteil aus :numref:`sec_baselines`.

<!--en:2544e4b6-->
Die Identität für $\lambda = 1$ setzt voraus, dass die Episoden terminieren. Bricht eine Episode am Zeitlimit ab, berücksichtigt `gae` die fehlende Zukunft durch Bootstrapping im letzten TD-Fehler. Dagegen setzt `reward_to_go` den künftigen Beitrag zunächst auf null, bis `truncation_bootstrap` ihn wieder addiert. Die untrainierte Strategie oben lässt den Stab lange vor dem Zeitlimit fallen, daher gilt die geprüfte Gleichheit. Dieselbe Rückwärtssumme hat eine Online-Variante, TD($\lambda$). Sie führt für jeden Parameter des Kritikers eine abklingende *Berechtigungsspur* (engl. eligibility trace). Das eignet sich für Agenten, die aus einem Datenstrom lernen, ohne Episoden zu speichern :cite:`Sutton.1988,Elsayed.Vasan.Mahmood.2024`.

<!--en:91dc9a94-->
## Verzerrung und Varianz des $\lambda$-Ertrags

<!--en:82cd75cc-->
### Versuchsreihe mit verschiedenen Werten für $\lambda$

<!--en:486bd1cc-->
Welches $\lambda$ liefert das beste Trainingsergebnis? Um diesen Parameter zu variieren, brauchen wir ein neues Gewicht und ein neues Ziel für den Kritiker. Wir gewichten den Akteur mit dem GAE-Vorteil und passen den Kritiker an den $\lambda$-Ertrag an. Dieser Ertrag ist der Vorteil plus der Wert, auf den er sich bezieht. Die Funktion `lambda_return` berechnet ihn. Bei $\lambda = 0$ entspricht er dem Einschrittziel von `td_weight`. In :numref:`sec_ppo` passen wir den Kritiker auf dieselbe Weise an.

<!--en:043131ee-->
Die Funktion `gae_weight(lam)` liefert die Gewichtsfunktion für einen Parameterwert. Mit `gae_run` trainieren wir damit einen Lauf für einen Zufallsstartwert. Anders als bei `td_weight` standardisieren wir das Gewicht. Nahe $\lambda = 1$ hat es dieselbe Größenordnung wie der Monte-Carlo-Vorteil, mit einer Varianz von mehreren Dutzend. Rohe Gewichte würden die Gradientenbegrenzung bei großem $\lambda$ auslösen, bei kleinem $\lambda$ dagegen nicht. Das Budget ist mit 50 Updates zu je vier Episoden knapp. Der Vergleich zeigt deshalb, welcher Schätzer einem Agenten hilft, dessen Kritiker noch lernt.

<!--en:0b1ad7ca-->
Für jedes $\lambda$ in `sweep` geben wir pro Zufallsstartwert den besten gleitenden Mittelwert des Ertrags über 20 Updates sowie den Median dieser Werte an. Außerdem zeichnen wir für jedes der 50 Updates den Median des Batchertrags über die Zufallsstartwerte:

<!--en:e8a07d4c-->
Im Median über die Zufallsstartwerte erreichen $\lambda = 0$ und $\lambda = 0.5$ im besten Fenster aus 20 Updates nur etwa 150 bis 250, weit unter der Obergrenze. Bei $\lambda \geq 0.9$ sind die Erträge ungefähr doppelt so hoch. Die Ergebnisse einzelner Zufallsstartwerte überlappen zwischen diesen Gruppen, daher vergleichen wir die Mediane. Bei kurzem Training schätzt der Kritiker die Werte die meiste Zeit ungenau. Am schnellsten lernen dann die Schätzer, die sich am wenigsten auf ihn stützen. So zeigt sich der Befund des vorigen CartPole-Vergleichs aus der anderen Richtung. Die Verzerrung eines noch ungenauen Kritikers wiegt auf CartPole schwerer als die Varianz, die er einspart.

<!--en:86166ce3-->
### Messung bei fester Strategie

<!--en:3c5c641a-->
In Trainingskurven überlagern sich die Qualität des Schätzers, die Optimierung und Änderungen in den Daten. Um den Schätzer für sich zu untersuchen, halten wir die Parameter eines teilweise trainierten Agenten fest und ziehen damit viele Batches. Für jedes $\lambda$ vergleichen wir die Schätzung des Policy-Gradienten mit einer Referenz. Als Referenz dient der Mittelwert der Schätzungen für $\lambda = 1$ über alle Ziehungen. Terminieren die Episoden vor dem Zeitlimit, nähert diese Referenz den erwartungstreuen Monte-Carlo-Gradienten ohne Umgebungsmodell an (Aufgabe :exref:`ex_actor-critic-lambda-return-recursion` behandelt abgebrochene Episoden). Die Funktion `grad_vec` gibt den Gradienten der gewichteten Log-Wahrscheinlichkeiten eines Batchs als flachen Vektor zurück.

<!--en:5e036e65-->
Als Nächstes trainiert `freeze` einen neuen Agenten mit `td_weight` für `warmup=40` Akteur-Kritiker-Updates (etwa ein Viertel eines vollständigen Laufs). So können wir seine Schätzungen des Policy-Gradienten untersuchen, solange sein Kritiker noch ungenau ist:

<!--en:7eb39b9b-->
Die Funktion `probe` zieht bei festgehaltenem Agenten 200 Batches zu je vier Episoden. Für jedes $\lambda$ liefert sie die Verzerrung der mittleren Schätzung und die Varianz einer einzelnen Schätzung. Wir beziehen die Verzerrung auf die Länge des Referenzvektors und die Varianz auf das Quadrat dieser Länge. Die Summe aus quadrierter Verzerrung und Varianz ist der erwartete quadratische Fehler einer einzelnen Ziehung.

<!--en:23827e43-->
Ein festgehaltener Agent ist nur eine Realisierung unter vielen möglichen. Welches $\lambda$ für ihn am besten ist, hängt vom Fehler seines Kritikers ab. Wir halten deshalb acht Agenten fest und geben für jedes $\lambda$ drei Mediane über die Agenten an. Sie betreffen die Verzerrung, die Varianz und den Fehler, letzterer geteilt durch den Fehler desselben Agenten bei $\lambda = 1$:

<!--en:638662b0-->
Über die acht festgehaltenen Agenten ist der Median des Fehlers bei $\lambda = 0.9$ oder $0.95$ am kleinsten. Er beträgt etwa die Hälfte bis drei Fünftel des Fehlers bei $\lambda = 1$. Er liegt auch unter dem Fehler bei $\lambda = 0$, wobei der Abstand von Lauf zu Lauf stark schwankt. Von $\lambda = 0$ bis $\lambda = 1$ sinkt die Verzerrung auf null, während die Varianz auf mehr als das Zehnfache steigt. Der größte Anstieg liegt im letzten Schritt, weil das Gewicht $(\gamma\lambda)^l$ weit entfernter TD-Fehler dann nur noch langsam abklingt. Die Agenten unterscheiden sich. Die meisten erreichen ihren kleinsten Fehler bei $0.9$ oder $0.95$. Bei den übrigen ist ein kleineres $\lambda$ günstiger, wie der folgende Proposition für einen bereits genauen Kritiker erwarten lässt. Die Zeile für $\lambda = 1$ hat konstruktionsbedingt die Verzerrung null, weil ihr eigener Mittelwert als Referenz dient. PPO-Implementierungen verwenden üblicherweise $\lambda = 0.95$ als Standardwert :cite:`Schulman.Wolski.Dhariwal.ea.2017`.

<!--en:34b39483-->
### Warum das Optimum im Inneren liegt

<!--en:32ec377b-->
Warum sollte das beste $\lambda$ strikt zwischen den Endpunkten liegen? Der Vergleich mit einem perfekten Kritiker gibt die Antwort.

<!--en:6bb1fa9a-->
**Proposition.** Der Kritiker sei exakt, es gelte also $\hat{V} = V^\pi$, und die Episoden hätten kein Zeitlimit. Dann liefert jedes $\lambda$ eine erwartungstreue Schätzung von $A^\pi(s_t, a_t)$, und es gilt

<!--en:f8357cbe-->
Der Fehler der Schätzung ist somit bei $\lambda = 0$ am kleinsten.

<!--en:e7d3026d-->
**Beweis.** Bei einem exakten Kritiker hat der TD-Fehler in jedem Zustand bedingt auf diesen Zustand den Erwartungswert null. Es gilt $\mathbb{E}[\delta_{t+l} \mid s_{t+l}] = \sum_a \pi(a \mid s_{t+l})\, Q^\pi(s_{t+l}, a) - V^\pi(s_{t+l}) = 0$, wie die über die Aktion gemittelte Identität aus dem vorigen Abschnitt zeigt. Aufgrund der Markov-Eigenschaft gilt dies auch bedingt auf die gesamte Vorgeschichte bis $s_{t+l}$. Für $l \geq 1$ umfasst sie $s_t$, $a_t$ und alle früheren TD-Fehler. Daher hat jeder Term mit $l \geq 1$ in :eqref:`eq_gae_deltas` bedingt auf $(s_t, a_t)$ den Erwartungswert null. Der Erwartungswert der Schätzung ist also $\mathbb{E}[\delta_t \mid s_t, a_t] = A^\pi(s_t, a_t)$. Für $k < l$ verschwindet der Kreuzterm $\mathbb{E}[\delta_{t+k}\, \delta_{t+l} \mid s_t, a_t]$ aus demselben Grund. Es bleibt die Summe der Quadrate. Jeder quadrierte Term ist nichtnegativ, und sein Gewicht $(\gamma\lambda)^{2l}$ wächst mit $\lambda$, während der Erwartungswert gleich bleibt. Damit wächst der mittlere quadratische Fehler mit $\lambda$. $\blacksquare$

<!--en:c426439a-->
Bei einem exakten Kritiker bringt der beobachtete Rest der Episode nur zusätzliches Rauschen. Der Einschritt-TD-Fehler ist dann am besten. Ein Optimum im Inneren gibt daher Aufschluss über den Fehler des Kritikers. Es entsteht, weil der Kritiker ungenau ist, und sollte sich mit zunehmender Genauigkeit in Richtung $\lambda = 0$ verschieben. Das prüfen Sie in Aufgabe :exref:`ex_actor-critic-bias-of-gae` bei fester Strategie. In diesem Sinne drückt der Standardwert $\lambda = 0.95$ aus, wie genau Kritiker während des Trainings typischerweise sind.

<!--en:dda65763-->
### Kritiker oder Gruppen-Baseline

<!--en:76c9f2ba-->
Ein gelernter Kritiker ist eine Möglichkeit, den Vorteil zu schätzen. In :numref:`sec_baselines` haben wir eine zweite kennengelernt. Die Gruppen-Baseline vergleicht mehrere Stichprobenwerte aus demselben Startzustand miteinander. Die Abwägung aus diesem Abschnitt hilft bei der Wahl. Der Kritiker ist der bessere Schätzer, wenn es in jedem Schritt Belohnungen gibt, die Episoden lang sind und sich Zustände mit wenig Aufwand bewerten lassen. Sein Fehler sinkt beim Lernen, und Updates sind schon vor dem Ende der Episode möglich. Die Gruppen-Baseline ist der bessere Schätzer, wenn jede Episode nur eine Belohnung am Ende liefert und die Datenerhebung teuer ist. Das ist etwa der Fall, wenn ein Sprachmodell auf einen Prompt antwortet und ein Prüfprogramm die Antwort bewertet. Ein Kritiker müsste aus jedem Präfix jeder Antwort die Endbewertung vorhersagen, selbst ein schwieriges Regressionsproblem. Die anderen Antworten auf denselben Prompt liefern dagegen eine Baseline ohne zweites Netz. Diese Wahl trifft GRPO, dessen Gewichte wir in :numref:`sec_baselines` hergeleitet haben.

<!--en:0177a1e8-->
Auch in der Praxis sammeln Akteur-Kritiker-Implementierungen keine vollständigen Episoden. Sie führen parallel in vielen Umgebungen eine feste Anzahl von Schritten aus. Dann schneiden sie jedes Segment ab und aktualisieren die Netze, unabhängig davon, ob die Episode beendet ist. Sie behandeln den Schnitt wie ein Zeitlimit. Bei einer noch laufenden Episode nutzt der letzte TD-Fehler des Segments den Wert des Kritikers am Schnitt als Bootstrap-Wert. Genau das tut `td_target` bei einer abgebrochenen Episode. Der Rückwärtsdurchlauf von `gae` beginnt an jedem Schnitt erneut. Bei fester Segmentlänge ist der Abstand zwischen Updates unabhängig von der Episodenlänge. Batches aus ganzen Episoden lassen diesen strukturellen Vorteil des Bootstrappings ungenutzt.

<!--en:cd5cc470-->
Wir können eine Strategie nun anhand eines einzigen Übergangs aktualisieren und bestimmen, wie viel vom beobachteten Rest der Episode wir beibehalten. Für einen Schritt genügt der TD-Fehler. Der $\lambda$-Ertrag mischt alle Tiefen mit geometrischen Gewichten, und GAE berechnet diese Mischung in einem Rückwärtsdurchlauf. Auf CartPole ist es am besten, den Rest der Episode größtenteils beizubehalten, also $\lambda$ nahe $0.95$ zu wählen. Der Kritiker ist während eines großen Teils des Trainings noch ungenau. Zwei Einschränkungen aus :numref:`sec_deeprl` bleiben bestehen. Wir verwenden jeden Batch für genau einen Akteurschritt und verwerfen ihn danach. Außerdem gibt es keine Grenze dafür, wie stark dieser Schritt die Strategie verändert. Beide Punkte behandelt :numref:`sec_ppo`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:e63e70cc-->
Ein Akteur-Kritiker-Verfahren gewichtet den Policy-Gradienten mit dem TD-Fehler eines gelernten Kritikers anstelle des beobachteten Ertrags. Bei einem exakten Kritiker liefert der TD-Fehler aus einem einzigen Übergang eine erwartungstreue Schätzung des Vorteils. Bei einem gelernten Kritiker tritt an die Stelle der Varianz des beobachteten Restertrags die Verzerrung durch den Kritiker. Der Kritiker selbst macht einen Semigradientenschritt zur stichprobenbasierten Strategiebewertung. Auf CartPole erreicht das Monte-Carlo-Gewicht bei gleichem Rechenbudget für den Kritiker die Obergrenze in weniger Umgebungsschritten. Die Verzerrung des noch ungenauen Kritikers schadet mehr als die eingesparte Varianz. Der bleibende Vorteil des Bootstrappings ist, dass ein Update keine abgeschlossene Episode braucht. Die $n$-Schritt-Erträge und der $\lambda$-Ertrag interpolieren zwischen beiden Gewichten. GAE berechnet den $\lambda$-Ertrag durch einen Rückwärtsdurchlauf über die TD-Fehler. Bei festgehaltener Strategie ist der Fehler im Bereich von $\lambda = 0.9$ bis $0.95$ am kleinsten. Mit einem exakten Kritiker wäre er bei $\lambda = 0$ am kleinsten. Das Optimum im Inneren gibt somit Aufschluss über den Fehler des Kritikers.

<!--en:2294243a-->
## Übungen

<!--en:010374b1-->
1. **Verzerrung des Bootstrap-Gewichts.** Nach der Identität zum bedingten Erwartungswert null aus :numref:`sec_baselines` dürfen wir jede Funktion $b(s_t)$ des aktuellen Zustands vom Gewicht des Scores $\boldsymbol{\psi}_t = \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ abziehen, ohne den Gradienten zu verzerren. Bezeichnen Sie den Fehler des Kritikers mit $e(s) = \hat{V}(s) - V^{\pi}(s)$. In Endzuständen gilt $e = 0$, weil dort beide Werte null sind.
    1. Zeigen Sie, dass sich der TD-Fehler aus :eqref:`eq_td_error_v` zerlegen lässt als $\delta_t = \delta^{\pi}_t + \gamma\, e(s_{t+1}) - e(s_t)$. Dabei ist $\delta^{\pi}_t$ der TD-Fehler des exakten Kritikers $V^{\pi}$. Die Identität lässt einen der beiden Fehlerterme verschwinden. Welcher Beweisschritt versagt beim anderen Term?
    1. Zeigen Sie, dass der Term für Schritt $t$, also $\mathbb{E}[\delta_t\, \boldsymbol{\psi}_t]$, gegenüber $\mathbb{E}[A^{\pi}(s_t, a_t)\, \boldsymbol{\psi}_t]$ die Verzerrung $\gamma\, \mathbb{E}\big[\bar{e}(s_t, a_t)\, \boldsymbol{\psi}_t\big]$ mit $\bar{e}(s, a) = \mathbb{E}[e(s_{t+1}) \mid s_t = s, a_t = a]$ aufweist. Unter welcher Bedingung an $e$ verschwindet die Verzerrung? Ist $\hat{V} = V^{\pi}$ dazu notwendig?
    1. Der Kritiker überschätze jeden nichtterminalen Zustand um dieselbe Konstante $c > 0$. In einem Zustand beendet eine Aktion die Episode, die andere führt sie fort. Welche Aktion begünstigt die Verzerrung? Was ändert sich für $c < 0$? Auf CartPole mit der Belohnung $1$ pro Schritt sagt ein frisch initialisierter Kritiker Werte nahe null voraus, während $V^{\pi}$ positiv ist. In welche Richtung drängt die Verzerrung nach diesem Modell die Strategie zu Beginn des Trainings?
   :exlabel:`ex_actor-critic-why-baseline-argument-does`

<!--en:708dffcc-->
1. [code] **Optimale Bootstrap-Tiefe.** Betrachten Sie die Kette aus diesem Abschnitt mit $\sigma = 0.15$, $\gamma = 0.97$ und dem Fehler $e(s) = 0.5\,(1 - s/9)^2$ des Kritikers.
    1. Leiten Sie die Verzerrung $\gamma^n e(n)$ und die Varianz $\sigma^2 (1 - \gamma^{2n}) / (1 - \gamma^2)$ des Ziels der Tiefe $n$ vom Zustand $0$ aus her. Simulieren Sie dann $20{,}000$ Rollouts je Tiefe. Vergleichen Sie die gemessenen mittleren quadratischen Fehler mit den Formeln und bestätigen Sie das Optimum $n = 4$.
    1. Ersetzen Sie den Fehler des Kritikers in jedem nichtterminalen Zustand durch die Konstante $e(s) = 0.5$. Zeigen Sie, dass der mittlere quadratische Fehler für $n \leq 9$ folgende Form hat:

       $$\frac{\sigma^2}{1 - \gamma^2} + \gamma^{2n} \Big( 0.25 - \frac{\sigma^2}{1 - \gamma^2} \Big),$$

       Damit ist er für jedes $\sigma$ über alle Tiefen mit Bootstrapping monoton. Bleibt er monoton, wenn Sie das Monte-Carlo-Ziel $n = 10$ einbeziehen? Welche Tiefen können für einen solchen Kritiker optimal sein? Welche ist bei $\sigma = 0.15$ optimal?
    1. Kehren Sie zum abnehmenden Fehler zurück und bestimmen Sie die optimale Tiefe für $\sigma = 0.04$ und für $\sigma = 0.3$. Erklären Sie, in welche Richtung sich das Optimum verschiebt.
   :exlabel:`ex_actor-critic-optimal-bootstrap-depth`

<!--en:fa7c97df-->
1. [code] **Durchläufe des Kritikers und Reihenfolge der Updates.** Die Funktion `td_weight` gibt dem Kritiker `critic_steps` Regressionsdurchläufe pro Batch und berechnet `td_target` vor jedem Durchlauf neu.
    1. Wiederholen Sie das Akteur-Kritiker-Experiment mit `critic_steps` aus $\{1, 5, 20\}$ und je zwei Zufallsstartwerten. Zeichnen Sie die Kurven. Wie verändert die Zahl der Durchläufe die Lerngeschwindigkeit und das Verhalten nahe der Obergrenze von $500$?
    1. Berechnen Sie das Ziel nun einmal pro Batch und passen Sie den Kritiker mit `d2l.fit_value(ac, b.obs, target, num_steps=critic_steps)` daran an. Trainieren Sie mit denselben beiden Zufallsstartwerten und $20$ Durchläufen. Vernachlässigen Sie die Generalisierung zwischen Zuständen. Wie viele Schritte kann sich der Einfluss einer Belohnung auf $\hat{V}$ innerhalb eines Batchs mit festem Ziel entlang der Episode rückwärts ausbreiten? Wie viele Schritte sind es höchstens, wenn Sie das Ziel vor jedem Durchlauf neu berechnen? Erklären Sie anhand Ihrer Antworten den Unterschied zwischen den Kurven für feste und neu berechnete Ziele bei $20$ Durchläufen.
    1. Verschieben Sie die Zeile, die `delta` berechnet, vor die Durchläufe des Kritikers. So gewichtet der Akteur seinen Schritt mit den TD-Fehlern des Kritikers, der bei der Erhebung des Batchs aktuell war. Trainieren Sie für jede Reihenfolge mit zehn Zufallsstartwerten. Zählen Sie die Läufe, deren letzte zehn Batches im Mittel unter 300 liegen. Warum spielt die Reihenfolge bei zwanzig Durchläufen eine größere Rolle als bei einem?
   :exlabel:`ex_actor-critic-critic-passes`

<!--en:74a551db-->
1. **Rekursion für den $\lambda$-Ertrag.** Bezeichnen Sie mit $G^{\lambda}_t = \hat{A}^{\textrm{GAE}}_t + \hat{V}(s_t)$ das Ziel, an das `gae_weight` den Kritiker anpasst.
    1. Zeigen Sie anhand von :eqref:`eq_gae_deltas`, dass innerhalb einer Episode $\hat{A}^{\textrm{GAE}}_t = \delta_t + \gamma \lambda\, \hat{A}^{\textrm{GAE}}_{t+1}$ gilt. Folgern Sie daraus für einen nichtterminalen Schritt, der nicht der letzte seiner Episode ist:

       $$G^{\lambda}_t = r_t + \gamma \big( (1 - \lambda)\, \hat{V}(s_{t+1}) + \lambda\, G^{\lambda}_{t+1} \big).$$

       Wie lautet die Rekursion für den letzten Schritt einer Episode, die terminiert? Wie lautet sie für den letzten Schritt einer Episode, die am Zeitlimit abbricht?
    1. Eine Episode breche nach $T$ Schritten ab, und $s_T$ sei ihr letztes `next_obs`. Zeigen Sie, dass im Schritt $t$ dieser Episode der Wert von `b.gae(value_fn, gamma, 1.0)` den Wert von `b.reward_to_go(gamma) - value_fn(b.obs)` um genau $\gamma^{T-t}\, \hat{V}(s_T)$ übersteigt. Berechnen Sie für allgemeines $\lambda$ die Ableitung von $G^{\lambda}_t$ nach $\hat{V}(s_T)$. Welche Schritte einer abgebrochenen Episode beeinflusst der Wert des Kritikers am Abbruchpunkt, wenn $\lambda < 1$ gilt?
    1. Das Gewicht von $\delta_{t+l}$ in :eqref:`eq_gae_deltas` beträgt $(\gamma \lambda)^l$. Leiten Sie die Halbwertszeit $\tau$ her, also die Anzahl der Schritte, nach denen das Gewicht auf die Hälfte seines Werts bei $l = 0$ gefallen ist. Berechnen Sie die Halbwertszeit bei $\gamma = 0.99$ für $\lambda \in \{0.5, 0.9, 0.95, 1\}$. Die bei fester Strategie gemessene Varianz wächst zwischen $\lambda = 0.95$ und $\lambda = 1$ am stärksten. Erklären Sie den Zusammenhang mit den Halbwertszeiten.

   *Nach Richard S. Sutton und Andrew G. Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf), Aufgaben 12.1 und 12.2.*
   :exlabel:`ex_actor-critic-lambda-return-recursion`

<!--en:b2e3d594-->
1. [extended] **Vollständig online lernen.** ● Das Update :eqref:`eq_actor_critic` lässt sich nach jedem Übergang ganz ohne Batch anwenden. Der Kritiker macht einen Schritt in Richtung des einzelnen Ziels $r_t + \gamma\, \hat{V}(s_{t+1})$ mit Maske für die Terminierung. Der Akteur macht einen Schritt mit $\delta_t$ selbst als Gewicht, wie in `td_weight`.
    1. Implementieren Sie diese Variante, indem Sie `d2l.fit_value` und `policy_step_clip` auf Arrays der Länge eins anwenden. Zählen Sie die Umgebungsschritte eines Akteur-Kritiker-Laufs von `train` und geben Sie der Online-Variante dieselbe Anzahl. Trainieren Sie mit drei Zufallsstartwerten bei der Standardlernrate von `ActorCritic.mlp` und bei einem Zehntel dieser Lernrate. Übergeben Sie die Lernrate als `lr` in `cartpole_agent`. Vergleichen Sie die Kurven mit denen der Schleife mit Batches.
    1. Aufeinanderfolgende Updates stammen nun aus aufeinanderfolgenden Zuständen einer Episode. Welche Probleme entstehen? Ein Replay-Puffer speichert vergangene Übergänge, aus denen er gleichverteilt Minibatches für das Training zieht. Welche Probleme würde er lösen? Warum kann der Akteur sein Update mit der Score-Funktion nicht direkt auf gespeicherten Übergängen trainieren?
    1. Parallele Umgebungen dekorrelieren die Daten ohne Replay-Puffer :cite:`Mnih.Badia.Mirza.ea.2016`. Führen Sie acht CartPole-Umgebungen im Gleichschritt aus. Aktualisieren Sie die Netze nach jedem Schritt anhand von acht Übergängen, einem aus jeder Umgebung. Welche Probleme löst das, welche bleiben bestehen?
   :exlabel:`ex_actor-critic-fully-online`

<!--en:0fd97653-->
1. [code] **Gradientennormen und ihre Begrenzung.** ● Die von `policy_step_clip` ausgegebene Norm ist die Norm von $\hat{\mathbf{g}} = \frac{1}{N} \sum_t \hat{A}_t\, \boldsymbol{\psi}_t$. Dabei ist $\hat{A}_t$ das Gewicht des Akteurs, also der normalisierte Monte-Carlo-Vorteil aus `mc_weight` oder der rohe TD-Fehler aus `td_weight`. Außerdem gilt $\boldsymbol{\psi}_t = \nabla_\theta \log \pi_\theta(a_t \mid s_t)$. Die letzte Spalte von `runs` hält die Norm für jedes Update fest.
    1. Vergleichen Sie die Mediane der Norm für die beiden Varianten in `runs`. Nehmen Sie an, dass die $N$ Terme unabhängig und identisch verteilt sind. Zeigen Sie, dass gilt:

       $$\mathbb{E}\big[\|\hat{\mathbf{g}}\|^2\big] = \big\|\mathbb{E}[\hat{A}_t\, \boldsymbol{\psi}_t]\big\|^2 + \frac{1}{N} \Big( \mathbb{E}\big[\|\hat{A}_t\, \boldsymbol{\psi}_t\|^2\big] - \big\|\mathbb{E}[\hat{A}_t\, \boldsymbol{\psi}_t]\big\|^2 \Big).$$

       Zerlegen Sie das Gewicht in sein Signal $\mathbb{E}[\hat{A}_t \mid s_t, a_t] - \mathbb{E}[\hat{A}_t \mid s_t]$ und den Rest. Zeigen Sie, dass nur das Signal zu $\mathbb{E}[\hat{A}_t\, \boldsymbol{\psi}_t]$ beiträgt. Wie wirkt sich ein großer Rest bei einem Gewicht mit Varianz eins auf die Norm aus?
    1. Ziehen Sie für den Agenten `freeze(0)` mit `d2l.rollout` 50 Batches zu je acht Episoden. Berechnen Sie für jeden Batch die Norm von `grad_vec(ac, b, d2l.standardize(w))` für den TD-Fehler und für den Monte-Carlo-Vorteil `b.reward_to_go(gamma) - ac.value_np(b.obs)`. Wiederholen Sie die Rechnung, nachdem Sie jeden Gewichtsvektor innerhalb des Batchs zufällig permutiert haben. Die Normalisierung beider Gewichte beseitigt den Skalenunterschied, den der rohe TD-Fehler aus `td_weight` mit sich bringt. Was beseitigt die Permutation? Wie unterscheidet sich die Norm jedes Schätzers von der Norm seiner permutierten Variante?
    1. Ändern Sie die Akteur-Kritiker-Variante so, dass sie für jeden Batch die Standardabweichung von `delta` (in `td_weight`) und die Norm der Parameteränderung des Akteurs durch den Schritt (in `train`) aufzeichnet. Trainieren Sie mit zehn Zufallsstartwerten unter drei Bedingungen: wie angegeben, mit `clip=float('inf')` und mit `d2l.standardize(delta)` als Gewicht. Vergleichen Sie für jede Bedingung die Parameteränderungen während der ersten zehn und der letzten fünfzig Updates. Zählen Sie die Läufe, deren gleitender Mittelwert nach Erreichen von 475 unter 300 fällt. Erklären Sie anhand der aufgezeichneten Standardabweichungen und der koordinatenweisen Skalierung von Adam, warum die Normalisierung eines Gewichts gegen Ende des Trainings schaden kann, wenn es bis auf das Rauschen des Kritikers schrumpft. Was ändert die Gradientenbegrenzung bei Adam?
   :exlabel:`ex_actor-critic-gradient-norms`

<!--en:405ad163-->
1. [code] **Verzerrung von GAE.** ● In Aufgabe :exref:`ex_actor-critic-why-baseline-argument-does` haben Sie die Verzerrung des Gewichts $\delta_t$ als Funktion des Fehlers $e = \hat{V} - V^{\pi}$ des Kritikers bestimmt. Erweitern Sie das Ergebnis auf jedes $\lambda$. Betrachten Sie Episoden, die terminieren und für die bei und nach der Terminierung $e = 0$ gilt.
    1. Zeigen Sie, dass gilt:

       $$\hat{A}^{\textrm{GAE}}_t = \sum_{l \geq 0} (\gamma \lambda)^l\, \delta^{\pi}_{t+l} - e(s_t) + (1 - \lambda) \sum_{k \geq 1} \gamma^k \lambda^{k-1}\, e(s_{t+k}),$$

       Dabei ist $\delta^{\pi}$ der TD-Fehler von $V^{\pi}$. Folgern Sie mit der Proposition über das Optimum im Inneren, dass der Beitrag von Schritt $t$ zur Gradientenschätzung die Verzerrung $\mathbf{b}(\lambda) = (1 - \lambda) \sum_{k \geq 1} \gamma^k \lambda^{k-1}\, \mathbf{c}_k$ mit $\mathbf{c}_k = \mathbb{E}[e(s_{t+k})\, \boldsymbol{\psi}_t]$ aufweist.
    1. Angenommen, es gilt $\mathbf{c}_k = q^{k-1} \mathbf{c}_1$ für einen festen Vektor $\mathbf{c}_1$ und ein $0 \leq q \leq 1$. Der Einfluss der Aktion auf den Fehler des Kritikers in späteren Zuständen klingt also geometrisch ab. Zeigen Sie, dass $\|\mathbf{b}(\lambda)\| / \|\mathbf{b}(0)\| = (1 - \lambda) / (1 - \gamma \lambda q)$ gilt. Berechnen Sie dieses Verhältnis für die Werte in `lams` bei $\gamma = 0.99$ mit $q = 1$ und $q = 0.9$. Dividieren Sie für jeden der acht Agenten aus der Messung bei fester Strategie die Verzerrung bei jedem $\lambda$ durch die Verzerrung bei $\lambda = 0$. Vergleichen Sie die Ergebnisse mit den vorhergesagten Verhältnissen. Welches $q$ passt? Was sagt ein Verhältnis größer als $1$ über die Vektoren $\mathbf{c}_k$ aus?
    1. Halten Sie die Strategie von `freeze(0)` fest und passen Sie nur den Kritiker weiter an. Führen Sie seine Durchläufe für 50 weitere Batches ohne den Akteurschritt aus und wiederholen Sie die Messung. Wie verändern sich die relative Verzerrung und der Wert von $\lambda$ mit dem kleinsten Fehler einer einzelnen Ziehung, wenn Sie den Kritiker bei fester Strategie länger anpassen? Verschiebt sich das gemessene Optimum in Richtung $\lambda = 0$, wie es die Proposition für einen exakten Kritiker vorhersagt?
   :exlabel:`ex_actor-critic-bias-of-gae`

<!--en:741da015-->
[Diskussion](https://d2l.discourse.group/t/36801)

<!--en:f3cb366d-->
Akteur-Kritiker-Verfahren und verallgemeinerte Vorteilsschätzung<br>**TD-Fehler als Vorteil · Kritiker mit Semigradienten · Zielkonflikt zwischen Verzerrung und Varianz · n-Schritt-Erträge · GAE im Rückwärtsdurchlauf**

<!--en:8e5f789f-->
::: {.slide title="Warum mit dem Update bis zum Ende der Episode warten?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:2f0fa2b9-->
- **Engpass bei Monte Carlo:** Die Gewichtung mit $\hat{G}_t - \hat{V}(s_t)$ verlangt, dass wir die Terminierung abwarten. Über die verbleibenden Schritte summiert sich die Varianz.

<!--en:e6f4c2ec-->
- **Akteur und Kritiker:** Da $\hat{V}(s_{t+1})$ bereits $\hat{G}_{t+1}$ vorhersagt, ersetzt die Methode den beobachteten Rest der Episode durch die Vorhersage des Kritikers. Dafür tritt an die Stelle der Varianz des Restertrags die Verzerrung des noch ungenauen Kritikers.

<!--en:3ad198c4-->
::: {.slide title="Bootstrapping für den künftigen Ertrag mit einem Kritiker"}

<!--en:fea45a4a-->
[Der TD-Vorteil]{.kicker}

<!--en:888d2dc6-->
Entwickelt man die Rekursion für den künftigen Ertrag $\hat{G}_t = r_t + \gamma \hat{G}_{t+1}$ um einen Schritt und setzt $\hat{V}(s_{t+1})$ für $\hat{G}_{t+1}$ ein, so wird aus dem Gewicht nach Abzug der Baseline $\hat{G}_t - \hat{V}(s_t)$ der **Fehler der zeitlichen Differenz (TD-Fehler)**:

<!--en:3b4151fc-->
**Terminierung und Abbruch:** Nur wenn `terminated` wahr ist, ist der Bootstrap-Wert null ($\hat{V}(s_{\textrm{term}}) = 0$). Beim Abbruch nach 500 Schritten durch `truncated` ist der Stab noch aufrecht und der künftige Wert ungleich null.

<!--en:b86ae5e7-->
::: {.slide title="Ein Übergang in CartPole von Hand"}

<!--en:21490583-->
[Anschauliches Beispiel]{.kicker}

<!--en:afbbe55f-->
- **Kalibrierter Kritiker ($\hat{V}(s) = 100$):** Bleibt der Stab aufrecht, gilt $\delta_t = 1 + 0.99(100) - 100 = 0$. Fällt er (`terminated`), gilt $\delta_t = 1 + 0 - 100 = -99$.

<!--en:bfd3b994-->
- **Frisch initialisierter Kritiker ($\hat{V}(s) \approx 0$):** Ob der Stab aufrecht bleibt oder fällt, ergibt denselben positiven TD-Fehler $\delta_t \approx +1$. Der Akteur lernt daraus nichts, bis der Kritiker genauer wird.

<!--en:07d7b39a-->
::: {.slide title="Ein exakter Kritiker liefert eine erwartungstreue Schätzung des Vorteils"}

<!--en:014e75ac-->
[Theoretische Grundlage]{.kicker}

<!--en:4c2babd6-->
- **Der Fehler im aktuellen Zustand fällt weg:** Der Fehler $e(s_t) = \hat{V}(s_t) - V^\pi(s_t)$ hängt nur von $s_t$ ab. Sein Beitrag zum Policy-Gradienten ist im Mittel null.

<!--en:eb24e44e-->
- **Abhängigkeit vom Folgezustand:** Der Fehler $\gamma e(s_{t+1})$ hängt von der Aktion $a_t$ über $s_{t+1} \sim P(\cdot \mid s_t, a_t)$ ab und verzerrt das Update des Akteurs.

<!--en:22cb3ce7-->
::: {.slide title="Strategiebewertung mit Semigradienten im Kritiker"}

<!--en:ae66fdcb-->
[Funktionsweise des Kritikers]{.kicker}

<!--en:cb73393c-->
- **Stichprobenbasiertes Bellman-Backup:** Eine Regression passt $\hat{V}_w(s_t)$ an das Einschrittziel $y_t = r_t + \gamma \hat{V}_w(s_{t+1})$ an.

<!--en:4722814d-->
- **Warum Semigradienten?** `td_target` unterbricht den Gradientenfluss durch $y_t$. Die Ableitung berücksichtigt nur die Vorhersage $\hat{V}_w(s_t)$.

<!--en:7cab5db0-->
::: {.slide title="Verzerrung und Varianz auf einer Kette mit zehn Zuständen"}

<!--en:31238753-->
[Analyse in geschlossener Form]{.kicker}

<!--en:fe1ab023-->
- Gegeben sei eine deterministische Kette mit 10 Zuständen ($s \to s+1$, Belohnungsrauschen mit Standardabweichung $\sigma = 0.15$, $\gamma = 0.97$). Der Fehler $e(s) = 0.5(1 - s/9)^2$ des noch ungenauen Kritikers nimmt zum Ende hin ab.

<!--en:698f5425-->
- Ein $n$-Schritt-Ziel verwendet $n$ beobachtete Belohnungen und den Bootstrap-Wert im Zustand $n$:
  $$\textrm{Bias}(n) = \gamma^n e(n), \quad \textrm{Var}(n) = \sigma^2 \frac{1 - \gamma^{2n}}{1 - \gamma^2}$$

<!--en:6de3f706-->
- **Optimum im Inneren ($n = 4$):** $\textrm{MSE} = \textrm{Bias}^2 + \textrm{Var}$ fällt von $0.17$ bei $n=1$ auf $0.10$ bei $n=4$ und steigt dann wieder auf $0.17$ bei $n=10$ (Monte Carlo).

<!--en:0b94c5ac-->
::: {.slide title="Ein Skalar steuert zwei Parameterupdates"}

<!--en:1f34cfe9-->
[Die Akteur-Kritiker-Schleife]{.kicker}

<!--en:3c971ad6-->
- Jeder Übergang $(s_t, a_t, r_t, s_{t+1})$ liefert einen skalaren TD-Fehler $\delta_t = r_t + \gamma \hat{V}_w(s_{t+1}) - \hat{V}_w(s_t)$, der beide Netze steuert:
  $$w \leftarrow w + \alpha\, \delta_t\, \nabla_w \hat{V}_w(s_t)$$
  $$\theta \leftarrow \theta + \eta\, \delta_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

<!--en:d4471ad1-->
- **Zwei Zeitskalen (Konda & Tsitsiklis, 2000):** Ändert sich der Akteur schneller als der Kritiker, folgt er veralteten Wertschätzungen. Der Kritiker braucht schnellere Updates (`critic_steps = 20` Durchläufe pro Batch), um mit $V^{\pi_\theta}$ Schritt zu halten.

<!--en:506d00e0-->
::: {.slide title="Begrenzung des Akteursgradienten bei Adam" only="pytorch"}

<!--en:f3cf294d-->
[Optimierung]{.kicker}

<!--en:eca6a08f-->
Der Akteur macht einen Schritt mit rohen (unnormalisierten) TD-Fehlern. Dabei begrenzt der Algorithmus die Gradientennorm mit `grad_clip = 0.5`:

<!--en:7876bdac-->
**Warum den rohen Fehler $\delta_t$ mit Adam verwenden?** Mit der Konvergenz des Kritikers sinkt $\textrm{std}(\delta_t)$ von $\sim 3.0$ auf $\le 0.1$ (vor allem Rauschen des Kritikers). Adam skaliert anhand eines langfristigen gleitenden Mittels quadrierter Gradienten ($\beta_2 = 0.999$). Dadurch verringern die schrumpfenden rohen TD-Fehler die effektive Schrittweite nahe der Obergrenze um den Faktor $\sim 8\times$. Eine Standardisierung pro Batch würde das späte Rauschen des Kritikers verstärken.

<!--en:54ba8c42-->
::: {.slide title="Begrenzung des Akteursgradienten bei Adam" only="jax"}

<!--en:f3cf294d-2-->
[Optimierung]{.kicker}

<!--en:eca6a08f-2-->
Der Akteur macht einen Schritt mit rohen (unnormalisierten) TD-Fehlern. Dabei begrenzt der Algorithmus die Gradientennorm mit `grad_clip = 0.5`:

<!--en:54bb5e53-->
::: {.slide title="Begrenzung des Akteursgradienten bei Adam (Fortsetzung)" only="jax"}

<!--en:f3cf294d-3-->
[Optimierung]{.kicker}

<!--en:a6e0e4cb-->
Die Funktion `policy_step_clip` füllt den Batch auf und führt den kompilierten Schritt mit Gradientenbegrenzung aus:

<!--en:7876bdac-2-->
**Warum den rohen Fehler $\delta_t$ mit Adam verwenden?** Mit der Konvergenz des Kritikers sinkt $\textrm{std}(\delta_t)$ von $\sim 3.0$ auf $\le 0.1$ (vor allem Rauschen des Kritikers). Adam skaliert anhand eines langfristigen gleitenden Mittels quadrierter Gradienten ($\beta_2 = 0.999$). Dadurch verringern die schrumpfenden rohen TD-Fehler die effektive Schrittweite nahe der Obergrenze um den Faktor $\sim 8\times$. Eine Standardisierung pro Batch würde das späte Rauschen des Kritikers verstärken.

<!--en:e804c3f4-->
::: {.slide title="Die Akteur-Kritiker-Schleife mit Batches und Fitted TD"}

<!--en:ce0f4436-->
[Implementierung]{.kicker}

<!--en:a2775fc9-->
Für jedes Update sammelt die Schleife einen Rollout mit 8 Episoden. Sie passt den Kritiker in `critic_steps = 20` Durchläufen an jeweils neu berechnete Einschrittziele an (*Fitted TD*). Anschließend macht der Akteur einen Schritt mit begrenztem Gradienten:

<!--en:54532a0e-->
::: {.slide title="Drei Entscheidungen beim Akteur-Kritiker-Verfahren mit Batches"}

<!--en:17e72255-->
[Entwurfsentscheidungen]{.kicker}

<!--en:d0b6fb39-->
- **1. Kritiker *vor* der Gewichtung des Akteurs anpassen:** Die 20 Durchläufe von Fitted TD auf `b` gehen der Berechnung von $\delta_t$ voraus. So geht weniger vom Fehler des Kritikers in die Gewichte des Akteurs ein.

<!--en:2cccc504-->
- **2. Ziele vor jedem Durchlauf neu berechnen:** Der Aufruf von `b.td_target` innerhalb der Schleife mit 20 Schritten gibt die Belohnung pro Batch bis zu 20 Schritte rückwärts weiter.

<!--en:56e1b092-->
- **3. Roher TD-Fehler $\delta_t$ und Gradientengrenze $0.5$:** So bleibt das späte Rauschen des Kritikers unverstärkt und die großen anfänglichen TD-Gradienten bleiben bei Adam begrenzt.

<!--en:3af76e05-->
::: {.slide title="Bootstrap- und Monte-Carlo-Gewichte auf CartPole"}

<!--en:5b1e3b8b-->
[Experimenteller Vergleich]{.kicker}

<!--en:267a7970-->
- Um den Einfluss des Gewichts zu isolieren, gibt `mc_weight` REINFORCE dieselbe Zahl von Regressionsdurchläufen (`critic_steps = 20`) und dieselbe Grenze für die Gradientennorm (`0.5`).

<!--en:0bd727e8-->
- Bei mindestens 35 von 40 Zufallsstartwerten erreicht **REINFORCE mit Baseline** zuerst einen Ertrag von 450 und benötigt dabei $15\%$ bis $35\%$ weniger Umgebungsschritte als das Einschritt-Akteur-Kritiker-Verfahren.

<!--en:e7bfbeaf-->
- Die Gradientenbegrenzung greift bei $20\%$ bis $40\%$ der Akteur-Kritiker-Updates (wegen der anfänglich unnormalisierten $\delta_t$) und bei $0\%$ der REINFORCE-Updates.

<!--en:5f53ab89-->
::: {.slide title="Warum die Verzerrung des Kritikers das frühe Lernen auf CartPole bremst"}

<!--en:858fcfb2-->
[Fehleranalyse]{.kicker}

<!--en:6fbef916-->
- Die Messung beider Gewichte auf denselben Trajektorien bestätigt, dass die Varianz des Monte-Carlo-Vorteils **$25\times$ bis $60\times$ so groß** ist wie die des TD-Fehlers.

<!--en:8ffee643-->
- **Warum entscheidet die geringere Varianz auf CartPole nicht den Vergleich?** Jeder Schritt liefert deterministisch $+1$. Das Bootstrapping beim Abbruch nach 500 Schritten entfernt bereits die Varianz der unendlich langen Fortsetzung.

<!--en:0ab0bc99-->
- Bei 8 Episoden pro Batch ist die Monte-Carlo-Varianz bereits hinreichend klein. Die **Verzerrung** des noch ungenauen Kritikers lenkt die frühen Updates des Akteurs in die falsche Richtung.

<!--en:2585ca13-->
::: {.slide title="Zwischen Horizonten interpolieren mit n-Schritt-Erträgen"}

<!--en:5780512f-->
[Mehrschritterträge]{.kicker}

<!--en:c92004dd-->
- **Endpunkt $\lambda = 0$:** Das gesamte Gewicht liegt auf $n = 1$. Das ergibt den Einschritt-TD-Fehler $\delta_t$.

<!--en:b373039a-->
- **Endpunkt $\lambda \to 1$:** Das gesamte Gewicht liegt auf dem Ertrag der vollständigen Episode. Das ergibt $\hat{G}_t - \hat{V}(s_t)$.

<!--en:9642aee8-->
::: {.slide title="Der Lambda-Ertrag als Teleskopsumme von TD-Fehlern"}

<!--en:de057a66-->
[Mathematische Herleitung]{.kicker}

<!--en:51505bcf-->
- **Teleskopsumme:** Jeder $n$-Schritt-Vorteil $\hat{G}^{(n)}_t - \hat{V}(s_t)$ vereinfacht sich zu $\sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$, weil sich die inneren Terme $\pm \gamma^l \hat{V}(s_{t+l})$ paarweise aufheben.

<!--en:8623aaa7-->
- **Geometrische Summe:** Der Term $\gamma^l \delta_{t+l}$ kommt in jeder Tiefe $n \ge l + 1$ vor. Sein Gesamtgewicht ist $(1 - \lambda) \sum_{n > l} \lambda^{n-1} = \lambda^l$.

<!--en:1dbd3f4f-->
::: {.slide title="Verallgemeinerte Vorteilsschätzung in zwei Zeilen"}

<!--en:fc59d994-->
[GAE-Implementierung]{.kicker}

<!--en:c3991695-->
Wegen $\hat{A}^{\textrm{GAE}}_t = \delta_t + \gamma\lambda\, \hat{A}^{\textrm{GAE}}_{t+1}$ lässt sich GAE mit demselben `backward_scan` berechnen wie der künftige Ertrag in Kapitel §{num:chap_reinforcement_learning}. Seine Eingabe ist $\delta_t$ mit dem Diskontierungsfaktor $\gamma\lambda$:

<!--en:1073e590-->
::: {.slide title="Verschiedene Lambda-Werte bei kurzem CartPole-Training" layout="tight"}

<!--en:61fca2e0-->
[Versuchsreihe]{.kicker}

<!--en:3a816602-->
- Bei GAE passt sich der Kritiker an den **$\lambda$-Ertrag** $G^\lambda_t = \hat{A}^{\textrm{GAE}}_t + \hat{V}(s_t)$ an. Der Akteur verwendet standardisierte Vorteile.

<!--en:b5a93611-->
- Bei knappem Budget (50 Updates zu je 4 Episoden, 5 Zufallsstartwerte) stagnieren Schätzer mit hohem Bootstrap-Anteil ($\lambda = 0$ und $\lambda = 0.5$) bei einem Median des Ertrags von $150$ bis $250$.

<!--en:3264577c-->
- Mischungen mit längerem Horizont ($\lambda \in \{0.9, 0.95, 1.0\}$) erreichen etwa doppelt so hohe Erträge (Mediane von rund $350$–$450$). Die geringere Gewichtung der Verzerrung des noch ungenauen Kritikers wiegt schwerer als das zusätzliche Belohnungsrauschen.

<!--en:3de8713a-->
::: {.slide title="Verzerrung und Varianz des Gradienten bei festen Strategien"}

<!--en:9d495338-->
[Messung bei fester Strategie]{.kicker}

<!--en:0939109b-->
Nach 40 Updates bleiben die Parameter von 8 teilweise trainierten Agenten fest. So lässt sich die Qualität des Schätzers unabhängig vom Trainingsverlauf messen. Die Messung verwendet je 200 Batches zu 4 Episoden und bestimmt relative Verzerrung und Varianz des Gradienten anhand des Mittels über 200 Batches für $\lambda = 1$:

<!--en:3ebf2459-->
Steigt bei den 8 festgehaltenen Agenten $\lambda$ von $0$ auf $1$, sinkt die Verzerrung auf $0$. Die Varianz steigt um mehr als den Faktor $10\times$. Am stärksten steigt sie zwischen $0.95$ und $1.0$, wo die Halbwertszeit von $(\gamma\lambda)^l$ von $11$ auf $69$ Schritte springt. Der Gesamtfehler $\textrm{Bias}^2 + \textrm{Var}$ ist bei **$\lambda \in [0.9, 0.95]$** am kleinsten.

<!--en:86023579-->
::: {.slide title="Warum Lambda = 0 bei einem exakten Kritiker optimal ist"}

<!--en:038d89f8-->
[Theorie und Praxis]{.kicker}

<!--en:6408d0dd-->
- **Proposition (exakter Kritiker $\hat{V} = V^\pi$):** Wegen der Markov-Eigenschaft gilt $\mathbb{E}[\delta_{t+l} \mid s_t, a_t, \delta_t, \dots, \delta_{t+l-1}] = 0$ für alle $l \ge 1$. Alle Kreuzterme verschwinden, und es folgt
  $$\mathbb{E}\big[(\hat{A}^{\textrm{GAE}}_t)^2 \mid s_t, a_t\big] = \sum_{l=0}^{\infty} (\gamma\lambda)^{2l}\, \mathbb{E}\big[\delta_{t+l}^2 \mid s_t, a_t\big]$$
  mit einem strikten Minimum bei **$\lambda = 0$**. Ein Optimum im Inneren ($\lambda \approx 0.95$) entsteht also *nur*, weil der Kritiker ungenau ist.

<!--en:b33bc58f-->
| Aufbau des Schätzers | Belohnungen | Episodenhorizont | Zusatzaufwand | Typische Anwendung |

<!--en:4de61cbf-->
| **Gelernter Kritiker mit GAE** | Belohnung in jedem Schritt | Lang oder fortlaufend | Wertnetz und Verzerrung des Kritikers | Robotik, Spiele, PPO (§{sec_ppo}) |

<!--en:568b29ad-->
| **Gruppen-Baseline (GRPO)** | Eine abschließende Bewertung | Eine generierte Antwort | $K$ parallele Rollouts pro Prompt | Schlussfolgern mit Sprachmodellen und Prüfprogrammen (§{sec_rl_sequences}) |

<!--en:b465137a-->
::: {.slide title="Verständnisfragen zu TD-Fehlern, GAE-Horizonten und Verzerrung durch den Kritiker"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:6ad34534-->
Drei Fragen zur Überprüfung des Verständnisses:

<!--en:b60050df-->
1. **Effektiver Horizont:** Bei $\gamma = 0.99$: Nach wie vielen Schritten $l$ sinkt das GAE-Gewicht $(\gamma\lambda)^l$ auf $0.5$, einmal für $\lambda = 0.95$ und einmal für $\lambda = 1.0$?

<!--en:3a8207d9-->
2. **Anpassung im Trainingsverlauf:** Gegen Ende eines langen Trainingslaufs ist der Kritiker $\hat{V}_w$ in allen besuchten Zuständen nahezu exakt. Sollte man $\lambda$ in Richtung $1$ erhöhen oder in Richtung $0$ senken?

<!--en:d30eeb70-->
3. **Konstante Überschätzung:** Angenommen, es gilt $\hat{V}(s) = V^\pi(s) + c$ mit einer positiven Konstanten $c > 0$ in jedem nichtterminalen Zustand (und $\hat{V} = 0$ in Endzuständen). Liefert $\delta_t$ weiterhin einen erwartungstreuen Beitrag zum Policy-Gradienten?

<!--en:557b98b6-->
**Lösungen:**

<!--en:41c1d72d-->
1. Die Halbwertszeit ist $l_{1/2} = \ln 2 / (-\ln(\gamma\lambda))$. Für $\lambda = 0.95$ ($\gamma\lambda = 0.9405$) beträgt sie **$l_{1/2} \approx 11.3$ Schritte**, für $\lambda = 1.0$ ($\gamma\lambda = 0.99$) **$l_{1/2} \approx 69.0$ Schritte**.

<!--en:76939824-->
2. **$\lambda$ in Richtung $0$ senken:** Sobald die Verzerrung des Kritikers verschwindet, haben künftige TD-Fehler $\delta_{t+l}$ ($l \ge 1$) den bedingten Erwartungswert null und tragen nur zur Varianz bei.

<!--en:213baf08-->
3. **Nur bei nichtterminalen Übergängen** bleibt die Gradientenschätzung erwartungstreu, sofern $\gamma c - c = (\gamma - 1)c$ zustandsunabhängig ist. Kann eine Aktion die Episode beenden ($e(s') = 0$), verzerrt $+c$ den Schritt des Akteurs *zugunsten der Fortsetzung*.

<!--en:0bda4151-->
::: {.slide title="Vorteilsschätzer zwischen TD-Lernen und Monte Carlo"}

<!--en:c5a222f8-->
[Zusammenfassung]{.kicker}

<!--en:50ad3804-->
| Schätzer | Formel für das Vorteilsgewicht $\hat{A}_t$ | Verzerrung (wenn $\hat{V} \neq V^\pi$) | Varianz einer einzelnen Ziehung | Abgeschlossene Episode nötig? |

<!--en:75377a47-->
| **Einschritt-TD ($\lambda = 0$)** | $\delta_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$ | **Am höchsten** ($\gamma \mathbb{E}[e(s_{t+1})]$) | **Am niedrigsten** ($1$ Übergang) | **Nein** (vollständig online, 1 Schritt) |

<!--en:5cac73d6-->
| **$n$-Schritt-Ertrag** | $\sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n \hat{V}(s_{t+n}) - \hat{V}(s_t)$ | Klingt gemäß $\gamma^n \mathbb{E}[e(s_{t+n})]$ ab | Wächst mit dem Horizont $n$ | **Nein** (Vorausschau über $n$ Schritte) |

<!--en:cd6d6114-->
| **GAE ($\lambda \in (0, 1)$)** | $\sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$ (`backward_scan`) | Über $1 - \lambda$ einstellbar | Ausgewogen bei $\lambda \approx 0.95$ | Segmentschnitt und Bootstrapping |

<!--en:1e07a48a-->
| **Monte Carlo ($\lambda = 1$)** | $\hat{G}_t - \hat{V}(s_t) = \sum_{l=0}^{T-t-1} \gamma^l r_{t+l} - \hat{V}(s_t)$ | **Null** (exakte Baseline) | **Am höchsten** (gesamter Restertrag) | **Ja** (oder Abbruch) |
