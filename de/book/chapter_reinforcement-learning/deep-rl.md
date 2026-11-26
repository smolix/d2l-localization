<!--en:5394ee7a-->
# Funktionsapproximation im bestärkenden Lernen

<!--en:8f944e82-->
Bisher stand jede gelernte Größe in einer Tabelle: sechzehn Zeilen für die Felder des Sees, eine Zeile für jeden Prompt der Aufgabe mit Prüfprogramm. Eine Ausnahme waren die Planer aus :numref:`sec_optimal_control`, die bekannte Bewegungsgleichungen nutzten. Stellen Sie sich nun einen Wagen vor, der eine oben am Wagen gelenkig befestigte Stange balanciert. Vier reelle Zahlen beschreiben seinen Zustand: Position und Geschwindigkeit des Wagens sowie Winkel und Winkelgeschwindigkeit der Stange. In tausend Episoden sieht der Agent dieselben vier Zahlen kein zweites Mal. Eine Tabelle für solche Zustände bräuchte bei jedem Besuch eine neue Zeile. Was der Agent bei einem Besuch lernt, würde ihm beim nächsten nichts sagen. Wir suchen eine Funktion, die ähnlichen Zuständen ähnliche Werte zuweist. Was der Agent lernt, wenn die Stange mit einer bestimmten Geschwindigkeit nach links kippt, hilft ihm dann auch bei einer etwas anderen Geschwindigkeit in derselben Richtung.

<!--en:dd86ee45-->
Das Problem ist so alt wie das Forschungsgebiet. Bellman bezeichnete das Wachstum der Tabellen mit der Anzahl der Zustandsvariablen als *Fluch der Dimensionalität* (engl. curse of dimensionality) :cite:`BellmanDPBook`. Schon 1959 bewertete Arthur Samuels Dameprogramm Brettstellungen anhand einer gewichteten Summe von Merkmalen, statt ihren Wert nachzuschlagen :cite:`Samuel.1959`. Ersetzen wir eine Tabelle durch eine parametrisierte Funktion, sprechen wir von *Funktionsapproximation*. Mit neuronalen Netzen als Funktionen erschloss dieses Verfahren dem bestärkenden Lernen (engl. reinforcement learning) Backgammon :cite:`Tesauro.1995`, Atari-Spiele :cite:`mnih2015human` und Go :cite:`Silver.Huang.Maddison.ea.2016`.

<!--en:c7c4d3c3-->
Für Policy-Gradienten ist der Schritt klein, denn die Herleitungen in :numref:`sec_policygradient` und :numref:`sec_baselines` setzen an keiner Stelle eine Tabelle voraus. REINFORCE braucht nur den Score $\nabla_\theta \log \pi_\theta(a \mid s)$. Ein Netz liefert ihn ebenso leicht wie eine Tabellenzeile. Neu ist, dass eine Aktualisierung nun die Schätzung in jedem Zustand verändert. Die gemeinsamen Parameter ermöglichen das Lernen in kontinuierlichen Zustandsräumen, schaffen aber auch eine neue Fehlerquelle. Wir vollziehen diesen Schritt an zwei kleinen Regelungsaufgaben: einem Wagen mit Stange und einem Pendel mit kontinuierlich wählbarem Drehmoment. Wir messen, wie weit eine Aktualisierung wirkt, und untersuchen zum Schluss, was der Agent damit noch nicht kann.

<!--en:06e49a03-->
## Netze anstelle von Tabellen

<!--en:a29f3fcd-->
An zwei Stellen ersetzen wir die Tabellen der vorigen Abschnitte durch Netze: bei der Strategie (engl. policy) und bei der Wertschätzung, die als Baseline diente. Bevor wir die beiden Netze implementieren, stellen wir eine Tabelle so dar, dass sich die Darstellung auf ein Netz erweitern lässt.

<!--en:76bf5dfc-->
### Tabellen als lineare Netze für One-Hot-kodierte Zustände

<!--en:1dc26328-->
`Policy.tabular` aus :numref:`sec_imitation` speichert seine Präferenzen in einer $16 \times 4$-Einbettungstabelle $\theta$. Die Präferenzen im Zustand $s$ stehen in Zeile $s$. Zeile $s$ ist gleich $\theta^\top \mathbf{e}_s$, wobei $\mathbf{e}_s$ der One-Hot-Vektor des Zustands $s$ ist. Eine Tabelle ist also ein lineares Netz mit festen One-Hot-Vektoren als Eingabemerkmalen. Der klassische Ansatz für kontinuierliche Zustände hat dieselbe Form: Er zerlegt den Zustandsraum in Quader. Dabei ist $\mathbf{e}_s$ der One-Hot-Vektor des Quaders, der $s$ enthält. Ein früher lernender Regler für den unten betrachteten Wagen mit Stange nutzte eine von Hand entworfene Einteilung in 162 solche Quader :cite:`Barto.Sutton.Anderson.1983`. Ein Netz mit einer versteckten Schicht behält diese Form bei und ersetzt die festen Merkmale durch gelernte. Die Präferenzen lauten dann $\mathbf{W}_2 \tanh(\mathbf{W}_1 \mathbf{s} + \mathbf{b}_1) + \mathbf{b}_2$, wobei $\mathbf{s}$ nun der Vektor der reellwertigen Beobachtungen selbst ist.

<!--en:03a78045-->
Die beiden Formen unterscheiden sich in der Wirkung einer Aktualisierung. Betrachten wir eine Wertschätzung $\hat{V}_w$ mit Parametern $w$. Ein Gradientenschritt der Weite $\eta_V$ für $\frac{1}{2}\big(y - \hat{V}_w(s_0)\big)^2$ bewegt die Schätzung in einem einzelnen Zustand $s_0$ auf einen Zielwert $y$ zu. Bis zur ersten Ordnung in $\eta_V$ ändert sich die Schätzung in einem beliebigen Zustand $s$ um

<!--en:466175a3-->
Die Änderung in $s$ entspricht also der angestrebten Änderung in $s_0$, skaliert mit dem Maß der Übereinstimmung beider Gradienten. Für eine Tabelle gilt $\nabla_w \hat{V}(s) = \mathbf{e}_s$. Daher ist $k(s, s_0)$ gleich $1$ für $s = s_0$ und sonst gleich $0$. Ein Gradientenschritt in einem Zustand lässt alle anderen Einträge unverändert. (Adam, der Standardoptimierer von `Policy`, kann durch sein gespeichertes Momentum auch einen Eintrag ändern, dessen Gradient im aktuellen Batch null ist. Die Aussage gilt also für Gradienten, aber nicht für jeden Optimiererschritt.) Bei einem Netz ist $k$ eine glatte Funktion der beiden Zustände, der *Tangentenkernel* (engl. tangent kernel) des Netzes :cite:`Jacot.Gabriel.Hongler.2018`. Ein Schritt in $s_0$ verändert dann auch die Schätzungen in ähnlichen Zuständen. In Aufgabe :exref:`ex_deep-rl-tangent-kernel` leiten Sie :eqref:`eq_deeprl_kernel` her. In :numref:`subsec_deeprl_coupling` messen wir die Form von $k$ für das von uns trainierte Netz.

<!--en:52713464-->
### Strategie- und Wertnetz

<!--en:c152d744-->
Der Container `Policy` aus :numref:`sec_imitation` enthält ein Strategienetz und dessen Optimierer. In :numref:`sec_baselines` war die gelernte Baseline eine separate NumPy-Tabelle. Nach jedem Batch verschob sie $\hat{V}(s_t)$ um einen Schritt der Weite $\eta_V$ in Richtung des künftigen Ertrags (engl. reward-to-go) $\hat{G}_t$, der im jeweiligen Zustand beobachtet wurde. Hier wird auch die Baseline zu einem Netz und rückt in den Container. `ActorCritic` erweitert `Policy` um das Wertmodul `value` mit eigenem Adam-Optimierer `opt_v`. `V` gibt die Werte als Array des jeweiligen Frameworks zurück. In dieser Form lässt sich der Verlust ableiten. `value_np` erwartet NumPy-Arrays und gibt solche zurück, wie die Schnittstellenmethoden von `Policy`. Der Klassenname verweist auf die beiden Bestandteile: Die Strategie heißt oft *Akteur* (engl. actor), die Wertschätzung *Kritiker* (engl. critic). Der Konstruktor `tabular` legt beide als mit Nullen initialisierte Tabellen an. So kann wie bei der Messung am Ende dieses Abschnitts eine Tabelle jedes der beiden Netze ersetzen:

<!--en:47345a73-->
Der Konstruktor `mlp` liefert Netze statt Tabellen. Beide entstehen mit `tanh_mlp`: ein Strategienetz mit `obs_dim` Eingaben und einer Präferenz pro Aktion sowie ein gleich aufgebautes Wertnetz mit einer Ausgabe. Jedes hat eine versteckte Schicht mit 64 Einheiten, eine übliche Breite für so kleine Aufgaben. Als Aktivierung dient tanh, das in einer großen Studie zu On-Policy-Verfahren für Strategienetze am besten abschnitt :cite:`Andrychowicz.Raichuk.Stanczyk.ea.2021`. Die Softmax-Funktion aus :numref:`sec_policygradient`, $\pi_\theta(a \mid s) \propto \exp(\textrm{preference}_a)$, wirkt auf die Netzausgaben wie zuvor auf eine Tabellenzeile. Der Score schließt nun die versteckte Schicht ein. Automatisches Differenzieren übernimmt die zusätzliche Anwendung der Kettenregel:

<!--en:0e0ccd61-->
Als Schichten dienen `d2l.Linear` (:numref:`sec_multihead-attention`). Dabei handelt es sich um `nnx.Linear` mit gleichverteilten Gewichten und Bias-Werten aus $[-1/\sqrt{n_\textrm{in}}, 1/\sqrt{n_\textrm{in}}]$ für $n_\textrm{in}$ Eingaben. Standardmäßig zieht Flax die Gewichte mit der LeCun-Normalinitialisierung, deren Streuung etwa 1.7-mal so groß ist, und setzt die Bias-Werte auf null. Für alle Netze in diesem und im nächsten Kapitel verwenden wir die gleichverteilte Initialisierung, weil sich diese kleinen Netze damit zuverlässiger trainieren lassen. Für das DQN aus :numref:`sec_dqn` haben wir beide Initialisierungen mit je 50 Zufallsstartwerten getestet. Mit gleichverteilter Initialisierung löste es CartPole in etwa zwei Dritteln der Läufe, mit der Standardeinstellung von Flax in etwa einem Viertel.

<!--en:6b74d3f9-->
### Regression für die Wertschätzung

<!--en:3566cc3e-->
Bei der tabellarischen Baseline bewegte jede Aktualisierung einen Eintrag auf einen beobachteten Zielwert zu. Beim Netz übernimmt das die Regression: Der Optimierer minimiert schrittweise den mittleren quadratischen Fehler zwischen $\hat{V}(s_t)$ und den Zielwerten des gesamten Batchs. `fit_value` macht `num_steps` solcher Schritte auf einen festen Zielvektor zu und passt dabei das Wertnetz `ac.value` des obigen `ActorCritic` an.

<!--en:6b1f5327-->
Wie viele Schritte braucht die Baseline pro Batch? In beiden folgenden Aufgaben sind die Zielwerte diskontierte Erträge der Größenordnung $100$. Ein Adam-Schritt verändert jeden Parameter dagegen ungefähr um die Lernrate $0.01$. Mit nur einem Schritt pro Batch hinkt die Baseline ihren Zielwerten hinterher. Bei der folgenden Pendelaufgabe messen wir den Anteil der Varianz der Zielwerte, den die Baseline erklärt, mit $1 - \textrm{Var}(\hat{G} - \hat{V}) / \textrm{Var}(\hat{G})$. Für die meisten Zufallsstartwerte erreicht dieser Anteil bis zum Ende eines Laufs nur Werte zwischen $0.4$ und $0.8$, für einige bleibt er noch niedriger. Der übrige Anteil des Gewichts $\hat{G}_t - \hat{V}(s_t)$ ist Rauschen, das eine gute Baseline beseitigen würde. Vierzig Schritte pro Batch heben den erklärten Anteil auf etwa $0.99$. Deshalb verwendet die folgende Trainingsschleife vierzig Schritte.

<!--en:78c2f05f-->
`value_np` führt das Wertnetz mit `d2l.forward_np` aus. `fit_value` führt seinen kompilierten Durchlauf `_value_pass` auf einem Batch aus, den es mit `d2l.padded` auf eine Bucket-Länge auffüllt, wie bei `policy_step`. Die Maske beschränkt den Verlust auf die tatsächlichen Zeilen. Diese Hilfsfunktionen finden sich in :numref:`sec_utils`.

<!--en:0bcc09ba-->
### Zeitgrenzen

<!--en:28234f9a-->
Beide Aufgaben in diesem Abschnitt brechen Episoden nach einer festen Zeit ab: CartPole nach 500 Schritten, Pendulum nach 200. In :numref:`sec_mdp` haben wir die Konsequenz gesehen: Die Summe der Belohnungen einer zeitbedingt abgebrochenen Episode ist nicht ihr Ertrag, denn der Prozess wäre weitergelaufen. Beim künftigen Ertrag behandeln wir den Abbruch wie ein reguläres Ende. Betrachten wir eine CartPole-Episode, in der die Stange bis zur Zeitgrenze im Gleichgewicht bleibt. Kurz vor dem Abbruch sinkt der künftige Ertrag gegen $1$. Das Wertnetz sieht dagegen eine ausbalancierte Stange wie zu jedem anderen Zeitpunkt. Es sagt etwa denselben Wert voraus wie für aufrechte Zustände zu Beginn einer Episode. Die Gewichte $\hat{G}_t - \hat{V}(s_t)$ der letzten Schritte sind dann stark negativ. Die Aktualisierung senkt also gerade die Wahrscheinlichkeit der Aktionen, die die Stange aufrecht hielten. Nehmen wir $\gamma = 0.99$, $T = 500$ und ein Wertnetz, das $1/(1-\gamma) = 100$ vorhersagt, den Wert dauerhaft erfolgreichen Balancierens. Bei $t = 490$ beträgt der künftige Ertrag $\hat{G}_t = (1 - 0.99^{10})/0.01 \approx 9.6$, was ein Gewicht nahe $-90$ ergibt. In Aufgabe :exref:`ex_deep-rl-time-limit` berechnen Sie diese Gewichte für die gesamte Episode.

<!--en:0037fd5a-->
Die Korrektur addiert den Wert der Zukunft, die durch den Abbruch der Beobachtung entgeht :cite:`Pardo.Tavakoli.Levdik.ea.2018`. Damit behandeln wir die Zeitgrenze als Vereinfachung beim Training. Wäre sie Teil der Aufgabe, müsste die verbleibende Zeit zum Zustand gehören (:numref:`sec_mdp`). Eine Episode habe die Schritte $t = 0, \ldots, T-1$. Dabei sei $s_T$ der Zustand nach der letzten Aktion, und $\mathrm{term} \in \{0, 1\}$ gebe an, ob dieser Zustand terminal ist. Der Zielwert für Schritt $t$ lautet dann

<!--en:ba2d4847-->
Bei einer regulär terminierten Episode kommt nichts hinzu. Bei einer abgebrochenen Episode erhält jeder Schritt zusätzlich den diskontierten Wert des Zustands am Ende der Aufzeichnung, als wäre die Episode von dort aus weitergelaufen. Bei $t = 490$ addiert die Korrektur im obigen Beispiel $0.99^{10} \cdot 100 \approx 90.4$. Damit gilt $y_t \approx 100$, und das Gewicht liegt wieder bei etwa $0$. Der `Batch` aus :numref:`sec_policygradient` enthält bereits die nötigen Daten: `next_obs` speichert $s_{t+1}$ für jeden Schritt, und `term` erfasst gemäß :numref:`sec_mdp` ausschließlich die Terminierung, keinen Abbruch. `truncation_bootstrap` erzeugt einen Vektor, dessen Einträge außer beim letzten Schritt jeder Episode null sind. Dort steht $\gamma (1 - \mathrm{term}) \hat{V}(s_T)$. Der Rückwärtsdurchlauf $x_t + \gamma x_{t+1} + \gamma^2 x_{t+2} + \cdots$ aus :numref:`sec_baselines` über diesen Vektor überträgt den Wert mit dem Faktor $\gamma^{T-t}$ auf jeden früheren Schritt:

<!--en:deb6ec62-->
## REINFORCE für CartPole

<!--en:2d937d47-->
Strategienetz, Wertnetz und Zielwerte stehen fest. Nun setzen wir die Trainingsschleife zusammen und erproben sie am Wagen mit der Stange.

<!--en:b53114de-->
### Die Trainingsschleife

<!--en:f4943bca-->
`train_reinforce` implementiert REINFORCE mit gelernter Baseline für beliebige Strategieobjekte. Dabei erzeugt `make_agent(seed)` den Agenten, und `env_name` benennt die Umgebung. Für jede Aktualisierung sammeln wir mit `rollout` acht Episoden und berechnen die Zielwerte $y_t$. Wir machen einen Strategieschritt mit Gewichten, die wir aus $y_t - \hat{V}(s_t)$ berechnen, und passen anschließend $\hat{V}$ mit vierzig Regressionsschritten an die Zielwerte an. Neben den Netzen gibt es vier Unterschiede zu den Läufen mit gelernter Baseline aus :numref:`sec_baselines`. Als Optimierer dient Adam, den jedes `Policy`-Objekt mitbringt, statt des einfachen SGD. Die Gewichte durchlaufen `d2l.standardize`. Diese Funktion zieht den Mittelwert des Batchs ab und teilt durch dessen Standardabweichung. Wir passen die Baseline durch Regression an, statt jeden Eintrag einzeln zu ändern. Bei abgebrochenen Episoden verwenden wir Bootstrapping.

<!--en:cc0a37c0-->
Teilen wir alle Gewichte durch dieselbe positive Zahl, ändert sich der Adam-Schritt kaum. Adam teilt nämlich jede Koordinate durch eine laufende Schätzung ihrer eigenen Skala (Aufgabe :exref:`ex_deep-rl-normalization-adam` präzisiert diese Aussage). Die Normalisierung erfüllt dennoch zwei Zwecke. Die Zentrierung wirkt wie eine zweite Baseline, die aus dem Batch selbst stammt. Die Division hält die Größenordnung der Gewichte stabil, während die Erträge im Laufe des Trainings steigen. Dieser Drift folgt der langsam abklingende Mittelwert von Adam nur mit Verzögerung.

<!--en:e5d3005c-->
Die so berechnete Aktualisierung entspricht nicht dem Gradienten der Zielfunktion $J(\theta)$. Bei $N$ Schritten im Batch und normalisierten Gewichten $\tilde{w}_t$ folgt der Strategieschritt der Richtung

<!--en:69cecd9b-->
Vier Unterschiede trennen $\hat{g}$ von $\nabla_\theta J$. Der Faktor $\gamma^t$ entfällt. Damit zählt der Score eines späten Schritts ebenso viel wie der eines frühen. Das ergibt den Schätzer mit künftigem Ertrag aus :numref:`sec_baselines`, der im Allgemeinen keiner Zielfunktion als Gradient entspricht :cite:`Nota.Thomas.2020`. Die Zielwerte $y_t$ einer abgebrochenen Episode enthalten $\hat{V}(s_T)$. Ein Fehler des Kritikers an der Abbruchstelle verzerrt somit die Gewichte. Die Zentrierung verwendet den Mittelwert des Batchs selbst, der schwach von den gezogenen Aktionen abhängt. Die Division durch $N$ und die Standardabweichung des Batchs skaliert jeden Batch anhand seiner eigenen Daten. Danach macht Adam einen Schritt annähernd fester Weite. Auf diese Kombination lässt sich kein Satz der stochastischen Approximation unmittelbar anwenden. Dass die Schleife funktioniert, zeigen daher die Trainingsläufe. Einen Konvergenzbeweis haben wir dafür nicht.

<!--en:34634cf1-->
### CartPole

<!--en:5c1537e2-->
CartPole ist die klassische Aufgabe, eine Stange zu balancieren :cite:`Barto.Sutton.Anderson.1983`. Die vier eingangs genannten Zahlen beschreiben den Zustand. Die beiden Aktionen schieben den Wagen nach links oder rechts. Jeder Schritt mit aufrechter Stange bringt die Belohnung $1$. Die Episode terminiert, wenn die Stange zu weit kippt oder der Wagen die Bahn verlässt. Nach 500 Schritten bricht die Umgebung die Episode ab. Der höchstmögliche Ertrag beträgt also 500. Gegenüber den Läufen auf dem See ändern sich drei Einstellungen: Der Konstruktor ist `ActorCritic.mlp(4, 2)` statt `Policy.tabular(16, 4)`, die Umgebung heißt `'CartPole-v1'` und der Diskontfaktor beträgt $0.99$. Dadurch steigt der effektive Horizont $1/(1-\gamma)$ aus :numref:`sec_mdp` von 20 auf 100 Schritte. `d2l.run_seeds` (:numref:`sec_utils`) führt den Trainingsgenerator einmal pro Zufallsstartwert aus und stapelt die ausgegebenen Werte. Wir trainieren mit drei Zufallsstartwerten über jeweils 60 Aktualisierungen:

<!--en:122ace41-->
Die untrainierte Strategie balanciert die Stange etwa 20 Schritte lang. Bei jedem Zufallsstartwert erreicht der Mittelwert des Batchs nach etwa 50 Aktualisierungen die Obergrenze von 500 Schritten. Er bleibt meist dort, auch wenn einzelne Batches noch darunterfallen. Bei dieser Aufgabe änderte die Korrektur an der Zeitgrenze den mittleren Ertrag in unseren Läufen kaum, weil nur erfolgreiche Episoden die Zeitgrenze erreichen. Ohne Korrektur fiel die Strategie gegen Ende des Trainings jedoch öfter unter die Obergrenze.

<!--en:22320d69-->
### Der Ersatzverlust

<!--en:851c7442-->
`train_reinforce` liefert bei jeder Aktualisierung eine zweite Zahl, den Wert `L` des Ersatzverlusts (engl. surrogate loss), den `policy_step` minimiert (:numref:`sec_policygradient`). Deep-Learning-Frameworks drücken Aktualisierungen durch skalare Verluste aus. Statt $\hat{g}$ explizit zusammenzusetzen, minimiert `policy_step` daher

<!--en:71085d95-->
bei festen Gewichten $\tilde{w}_t$. Diese liegen als NumPy-Array ohne Berechnungsgraph für den Gradienten vor. Der Gradient von $L$ ist $-\hat{g}$. Ein Abstiegsschritt für $L$ ist also ein Schritt in Richtung $\hat{g}$. Der Wert von $L$ schätzt jedoch nicht, wie gut die Strategie abschneidet. $L$ ist allein darauf ausgelegt, bei den aktuellen Parametern die richtige Ableitung zu haben. Sinkt $L$, muss der Ertrag deshalb nicht steigen. Bei den CartPole-Läufen haben wir beides aufgezeichnet:

<!--en:4e34fe16-->
Während der Ertrag von 20 auf 500 steigt, bleibt $L$ betragsmäßig unter $0.1$ und ist nach der ersten Aktualisierung fast immer negativ. Die normalisierten Gewichte haben Mittelwert null und Standardabweichung eins. Daher ist $L$ das negative Produkt aus der Standardabweichung der Log-Wahrscheinlichkeiten im Batch und ihrer Korrelation mit den Gewichten (Aufgabe :exref:`ex_deep-rl-surrogate-value`). Der Ersatzverlust ist negativ, wenn Aktionen mit positivem Gewicht bereits überdurchschnittlich wahrscheinlich sind. Sein Betrag sagt nichts über den Ertrag aus. Beobachten Sie beim Training den Ertrag sowie Größen, die die Aktualisierung selbst beschreiben, etwa das Ausmaß der Strategieänderung.

<!--en:2131d126-->
## Kontinuierliche Aktionen

<!--en:676b92e7-->
Die Softmax-Funktion benötigt eine endliche Liste von Aktionen. Viele Regelungsaufgaben haben stattdessen eine reellwertige Aktion, etwa ein Drehmoment oder einen Lenkwinkel. Die Herleitung mit der Score-Funktion in :numref:`sec_policygradient` setzt nirgends eine endliche Aktionsmenge voraus. Sie verlangt nur, dass $\log \pi_\theta(a \mid s)$ nach $\theta$ differenzierbar ist. Hat die Dichte einer Strategie über reellwertigen Aktionen diese Eigenschaft, können wir sie mit derselben Schleife trainieren.

<!--en:43196153-->
### Die Gauß-Strategie

<!--en:87ab5d6d-->
Wir kehren zum Pendel aus :numref:`sec_optimal_control` zurück (`Pendulum-v1`). Das Pendel hängt an einer motorgetriebenen Achse, und die Aktion ist ein Drehmoment $u \in [-2, 2]$. Mit $\vartheta$ bezeichnen wir den Winkel des Pendels zur aufrechten Lage. Das Symbol $\theta$ bleibt den Strategieparametern vorbehalten. Die Planer in :numref:`sec_optimal_control` kannten die Bewegungsgleichungen und differenzierten sie. Der Policy-Gradient-Agent kennt sie nicht. Er erhält nur die Beobachtung $(\cos\vartheta, \sin\vartheta, \dot{\vartheta})$ und bei jedem Schritt die Belohnung $-(\vartheta^2 + 0.1\,\dot{\vartheta}^2 + 0.001\,u^2)$. Dabei wird $\vartheta$ modulo einer vollen Umdrehung auf $[-\pi, \pi]$ abgebildet. Die Kosten sind also am niedrigsten, wenn das Pendel aufrecht stillsteht. Jede Episode beginnt bei einem zufälligen Winkel und bricht nach 200 Schritten ab. Ein ungeregeltes Pendel erzielt einen Ertrag nahe $-1200$. Ein Regler, der es aufschwingt und balanciert, erreicht etwa $-200$ oder mehr.

<!--en:2fba39b6-->
Wir verwenden eine Gauß-Strategie (engl. Gaussian policy) $\pi_\theta(a \mid s) = \mathcal{N}\big(a; \mu_\theta(s), \sigma^2\big)$, deren Netz den Mittelwert der Aktion liefert. `GaussianHead` implementiert dieses Netz. Ein *gemeinsamer Netzteil* (engl. trunk) berechnet versteckte Merkmale, standardmäßig mit einer tanh-Schicht. Über das Argument `trunk` lässt er sich durch ein beliebiges Modul mit `hidden` Ausgaben ersetzen. Eine lineare Schicht bildet die Merkmale auf den Mittelwert $\mu_\theta(s)$ ab. Mit `zero_mean=True` beginnen ihre Gewichte und ihr Bias bei null. Diese Einstellung verwenden wir unten für Pendulum. Die logarithmierte Standardabweichung $\log \sigma$ ist standardmäßig ein einzelner gelernter Vektor. Somit ist $\sigma$ in jedem Zustand gleich. Mit `state_std=True` liefert eine zweite lineare Schicht aus denselben Merkmalen die logarithmierte Standardabweichung, begrenzt auf $[-5, 2]$. Damit hängt $\sigma$ vom Zustand ab. Aufgabe :exref:`ex_deep-rl-gaussian-score` nutzt diese Option. Da $\log \sigma$ zum Strategiemodul gehört, aktualisiert der Strategieschritt diesen Parameter zusammen mit dem Mittelwert:

<!--en:4ee9aec1-->
`GaussianPolicy` ist eine Unterklasse von `ActorCritic`. Sie erzeugt einen `GaussianHead` und mit `tanh_mlp` ein Wertnetz wie das von `mlp`. Sie überschreibt die drei verteilungsabhängigen Methoden `log_prob`, `act` und `act_greedy`. Die Funktionen `rollout`, `policy_step`, `fit_value` und `train_reinforce` verwenden nur diese Schnittstelle und benötigen keine Änderung:

<!--en:640c5393-->
Die Umgebung begrenzt jedes gezogene Drehmoment auf $[-2, 2]$. Daher kann `act` das unbeschränkte Sample aus der Gauß-Verteilung zurückgeben. `log_prob` bewertet das ursprüngliche Sample. Aufgabe :exref:`ex_deep-rl-gaussian-score` zeigt, warum die Schätzung erwartungstreu bleibt. Die Aktualisierung verwendet `log_prob` nur über seinen Gradienten, den Score. Mit $z = (a - \mu)/\sigma$ lautet der Score einer Gauß-Verteilung

<!--en:a6314f54-->
Bei positivem Gewicht zieht ein Sample oberhalb des Mittelwerts $\mu$ nach oben. Ein in beliebiger Richtung weit vom Mittelwert entferntes Sample erhöht $\sigma$, ein Sample nahe dem Mittelwert senkt $\sigma$. Wir vergleichen `log_prob` mit der Formel für die Gauß-Dichte und seine Gradienten mit diesen beiden Formeln. Dazu verwenden wir $\sigma = 0.5$ und fünf zufällige Zustände mit zufälligen Aktionen. Der Mittelwert $\mu$ hängt vom Bias der Mittelwertschicht mit der Ableitung $1$ ab. Der Gradient bezüglich dieses Bias ist daher die Summe der Scores bezüglich $\mu$ über die fünf Samples:

<!--en:84a7ab02-->
Alle drei Prüfungen geben `True` aus: `log_prob` berechnet die Log-Dichte der Gauß-Verteilung, und seine Gradienten ergeben den Score für die Aktualisierung.

<!--en:658f23b4-->
### Pendulum

<!--en:8eedb550-->
Die Kurven zeigen den undiskontierten Ertrag über 200 Schritte. Die Aktualisierung verwendet dagegen $\gamma = 0.95$, was einem effektiven Horizont von 20 Schritten entspricht. Den Diskontfaktor wählen wir hier selbst. Die Aufgabe gibt ihn nicht vor. Ein Drehmoment beeinflusst die nächsten paar Dutzend Schritte weit stärker als die Kosten hundert Schritte später. Der kürzere Horizont ergibt Zielwerte mit geringerer Varianz, aber auch einer gewissen Verzerrung gegenüber der dargestellten Zielgröße. Mit $\gamma = 0.9$ verlief das Training in unseren Läufen etwa ebenso gut.

<!--en:4fc59dea-->
`pendulum_agent` setzt Gewichte und Bias der Mittelwertschicht auf null (`zero_mean=True`). Jeder Lauf beginnt daher bei $\mu = 0$ mit $\sigma = 1$, unabhängig von der Zufallsinitialisierung der anderen Schichten. Ein Mittelwert außerhalb von $[-2, 2]$ würde das Lernen zum Stillstand bringen. Die Umgebung würde fast jedes Sample auf denselben Grenzwert setzen. Die Belohnung wäre dann unabhängig vom gezogenen Sample, und die Gewichte, mit denen wir den Score multiplizieren, enthielten keine Information über die Aktion. Mit unserer Initialisierung liegt der anfängliche Mittelwert deutlich innerhalb der Grenzen. Bei Initialisierungen mit größeren Gewichten liegt er für manche Zufallsstartwerte außerhalb, und das Lernen stockt. Für On-Policy-Verfahren wird deshalb üblicherweise empfohlen, die letzte Schicht des Strategienetzes mit sehr kleinen Gewichten oder mit Nullen zu initialisieren :cite:`Andrychowicz.Raichuk.Stanczyk.ea.2021`. Die Strategie muss das Pendel erst aufschwingen lernen, bevor sie es balancieren kann. Wir trainieren daher über 300 Aktualisierungen mit jeweils acht Episoden:

<!--en:8d9cf864-->
Die Erträge beginnen bei etwa $-1200$. In jedem Lauf lernt der Agent, das Pendel aufzuschwingen und zu balancieren. Der Mittelwert über zehn Aktualisierungen überschreitet $-200$ nach etwa 150 bis 300 Aktualisierungen, also nach einigen Hunderttausend Umgebungsschritten. Sein Bestwert liegt je nach Lauf zwischen etwa $-150$ und $-200$. Spätere Aktualisierungen bleiben meist nahe diesem Niveau. Ein Lauf kann jedoch für einige Aktualisierungen um mehrere Hundert zurückfallen. Im Laufe des Trainings sinkt $\sigma$ auf etwa die Hälfte seines Anfangswerts, wie es Aufgabe :exref:`ex_deep-rl-gaussian-score` für die Nähe einer guten Aktion vorhersagt. Hier macht sich die Zeitgrenzenkorrektur bemerkbar: Jede Pendulum-Episode endet durch Abbruch. Ohne Korrektur erreichte in unseren Läufen nur etwa die Hälfte der Zufallsstartwerte bei gleichem Budget einen Ertrag von $-200$.

<!--en:c867eb8e-->
Einige Hunderttausend Schritte sind sehr viel Erfahrung für eine so kleine Aufgabe. Jeder Batch aus 1600 Schritten dient einer einzigen Aktualisierung. Danach verwerfen wir ihn. Die Gewichte stammen aus ganzen Erträgen statt aus einer gelernten Schätzung des Werts jeder Aktion. Soft Actor-Critic (:numref:`sec_sac`) erreicht $-200$ bei derselben Aufgabe innerhalb von etwa 5000 bis 11 000 Schritten. Wie ein Verfahren aus jedem Sample so viel mehr lernen kann, untersuchen wir in :numref:`chap_deep_rl`.

<!--en:abbb9e5f-->
### Gradienten der Score-Funktion und pfadweise Gradienten

<!--en:70f42736-->
Eine kontinuierliche Aktion eröffnet einen zweiten Weg, die erwartete Belohnung zu differenzieren. Betrachten wir einen einzelnen Zustand, eine Gauß-Strategie $a \sim \mathcal{N}(\mu, \sigma^2)$ und eine bekannte differenzierbare Belohnung $Q(a)$. Der Gradient von $\mathbb{E}[Q(a)]$ bezüglich $\mu$ lässt sich auf zwei äquivalente Arten schreiben:

<!--en:4cc68995-->
Die erste Gleichheit ist die in diesem Kapitel durchgehend verwendete Identität der Score-Funktion. Die zweite setzt $a = \mu + \sigma z$ ein. Dadurch wandert $\mu$ aus der Verteilung in $Q$, und wir differenzieren durch das Sample hindurch. Das ergibt den *pfadweisen* Gradienten (engl. pathwise gradient), auch Reparametrisierungsgradient genannt :cite:`Kingma.Welling.2014`. In :numref:`fig_rl_score_vs_pathwise` sehen wir, welchen Weg der Gradient bei jedem Schätzer nimmt. Bei bekanntem $Q$ sind beide erwartungstreu. Wir vergleichen ihr Rauschen für $Q(a) = -(a - 1)^2$ bei $\mu = 0$ und $\sigma = 0.5$. Der exakte Gradient beträgt dort $2$:

<!--en:6f749ca1-->
Beide Stichprobenmittelwerte liegen nahe $2$. Für ein einzelnes Sample beträgt die exakte Varianz bei der Score-Funktion $21.75$ und beim pfadweisen Schätzer $1.0$. Das Verhältnis beträgt etwa zwanzig. Addieren wir die Konstante $10$ zu $Q$, steigt die Varianz des Score-Funktionsschätzers auf $281.75$, weil $Q$ den Score multipliziert. Der pfadweise Schätzer hängt nur von $Q'$ ab und bleibt daher bei einer konstanten Verschiebung unverändert. In Aufgabe :exref:`ex_deep-rl-score-pathwise-variance` leiten Sie diese Varianzen her. Ihr Verhältnis hängt von $Q$, von $\sigma$ und von der Verschiebung ab. Die geringere Varianz hat ihren Preis: Wir brauchen $Q$ als differenzierbare Funktion der Aktion. In der Praxis übernimmt ein gelernter Kritiker die Rolle von $Q$. Nennen wir ihn $Q_w$, so führt ein Fehler in seinem Aktionsgradienten $\partial Q_w / \partial a$ zu einer Verzerrung des Policy-Gradienten.

<!--en:ac056aef-->
![Zwei Schätzer für denselben Gradienten. (a) Der Score-Funktionsschätzer nutzt nur die von der Umgebung gelieferte Belohnung. Die Umgebung muss daher nicht differenzierbar sein. (b) Der pfadweise Schätzer schreibt das Sample als $a = \mu_\theta(s) + \sigma_\theta z$, hier mit einem einzigen gelernten $\sigma_\theta$, und differenziert über $\partial Q_w / \partial a$ und $\partial a / \partial \theta$. Der Kritiker $Q_w$ muss nach der Aktion differenzierbar sein. Er lässt sich durch Regression auf wiederverwendeten Daten trainieren. Sein Fehler in $\partial Q_w / \partial a$ verzerrt den Gradienten.](../img/mdl-rl-score-vs-pathwise.svg)

<!--en:793da22d-->
Mithilfe der pfadweisen Form lässt sich bei kontinuierlicher Regelung auch die Maximierung über die Aktionen umgehen. Ein wertbasiertes Verfahren wählt die Aktion, indem es $\max_a Q(s, a)$ berechnet. Auf dem See genügt dazu ein Vergleich von vier Tabelleneinträgen. Für $a \in \mathbb{R}^d$ ist hingegen bei jedem Schritt ein Optimierungsproblem zu lösen. Das Verfahren des deterministischen Policy-Gradienten trainiert stattdessen ein Akteurnetz, das eine hochwertige Aktion ausgibt. Dazu folgt es $\partial Q_w / \partial a$ durch einen gelernten Kritiker :cite:`Silver.Lever.Heess.ea.2014`.

<!--en:500cbf8b-->
## Generalisierung über Zustände hinweg

<!--en:99aeec88-->
Zu Beginn haben wir nach einer Funktion gesucht, die ähnlichen Zuständen ähnliche Werte zuweist. Nach :eqref:`eq_deeprl_kernel` bestimmt der Kernel $k$, wie sich ein einzelner Gradientenschritt auf andere Zustände auswirkt. Wir messen diese Wirkung nun an einem frisch erzeugten Wertnetz mit demselben Aufbau wie beim CartPole-Training.

<!--en:d63ba7fb-->
### Messung der Kopplung zwischen Zuständen

<!--en:fb4097c1-->
Für die Messung erhöhen wir $\hat{V}$ in einem von 256 zufälligen Zuständen aus $[-1, 1]^4$. Dazu machen wir einen Gradientenschritt auf einen Zielwert zu, der eine Einheit über der aktuellen Schätzung liegt, und erfassen die Änderung in allen 256 Zuständen. Anschließend wiederholen wir die Messung mit einer Tabelle aus sechzehn Einträgen. Für beide Messungen ersetzen wir den Adam-Optimierer `opt_v` durch einfaches SGD (`d2l.plain_sgd`, :numref:`sec_utils`). So folgt der einzelne Schritt :eqref:`eq_deeprl_kernel`. Adam würde jeden Parameter anhand seiner eigenen laufenden Skala umskalieren. `fit_value` minimiert den mittleren quadratischen Fehler. Dessen Gradient ist doppelt so groß wie beim halbierten quadratischen Fehler. Daher ändert ein Tabellenschritt der Weite $0.25$ den Eintrag um $0.5$. Die Codezelle sortiert die 255 übrigen Testzustände nach ihrem Abstand zum aktualisierten Zustand:

<!--en:77095610-->
Der Wert im aktualisierten Zustand steigt. In den zehn nächsten Testzuständen steigt er fast ebenso stark, um mehr als 80 Prozent dieser Änderung. Mit zunehmendem Abstand nimmt die Änderung ab. In den entferntesten Testzuständen liegt sie nahe null oder ist leicht negativ: Einige Schätzungen bewegen sich in die Gegenrichtung. Die Tabelle ändert den aktualisierten Eintrag um $+0.50$ und lässt alle anderen unverändert. Wie stark die Schätzungen in den entferntesten Zuständen sinken und in wie vielen Zuständen sie sinken, hängt von der Zufallsinitialisierung des Netzes ab. Der Verlauf bleibt derselbe. :numref:`fig_rl_table_vs_network` zeigt den Unterschied für einen eindimensionalen Zustand.

<!--en:1529d9e0-->
![Reichweite einer Wertaktualisierung. Eine Tabelle mit sechzehn Einträgen und ein Netz mit 64 tanh-Einheiten schätzen den Wert eines eindimensionalen Zustands. Beide machen einen Gradientenschritt auf einen Zielwert zu, der eine Einheit über ihrer Schätzung im markierten Zustand $x_0$ liegt. Die jeweilige Schrittweite $\eta_V$ ist so gewählt, dass sich der Wert bei $x_0$ um $+0.50$ ändert. (a) Die Tabelle ändert nur einen Eintrag. (b) Das Netz verschiebt die gesamte Kurve, weil die Schätzung in jedem Zustand Parameter mit der Schätzung in $x_0$ teilt. (c) Die beiden Änderungen im Vergleich. In erster Ordnung beträgt die Änderung des Netzes $\eta_V\,\big(y - \hat{V}(x_0)\big)\,k(x, x_0)$ für den Zielwert $y$. Dabei ist $k$ das innere Produkt der Parametergradienten bei $x$ und $x_0$. Um $x_0$ steigt die Kurve zu einem positiven Buckel an. Jenseits von $x \approx 0.7$ wird sie negativ und erreicht am fernen Ende $-0.20$.](../img/mdl-rl-table-vs-network.svg)

<!--en:f9d95612-->
Der positive Buckel zeigt die Generalisierung: Erfahrung in einem Zustand fließt in die Schätzungen für ähnliche Zustände ein. Deshalb kann ein Netz CartPole lernen, ohne dass wir den Zustandsraum von Hand unterteilen. Der negative Bereich in der Ferne zeigt die Interferenz: Passen wir die Schätzungen in einer Region des Zustandsraums an, ändern sie sich auch in Regionen, die im Batch fehlen. Eine CartPole-Strategie, die gut balanciert, sammelt Batches mit fast aufrechter Stange. Das Training mit ihnen verändert die Schätzungen auch bei großen Stangenwinkeln, die im Batch nie auftreten (Aufgabe :exref:`ex_deep-rl-tangent-kernel` untersucht dies genauer). Solche Änderungen können Einbrüche in der Trainingskurve verursachen, selbst wenn jede Aktualisierung die Daten ihres eigenen Batchs gut trifft.

<!--en:11872edd-->
### Grenzen des Monte-Carlo-Agenten

<!--en:650e4767-->
Das Wertnetz in diesem Abschnitt lernt aus beobachteten Erträgen. Nur bei einem Abbruch an der Zeitgrenze enthält der Zielwert die eigene Schätzung $\hat{V}(s_T)$. Abseits der Abbruchstelle ist die Anpassung eine gewöhnliche Regression. Verfahren mit *Bootstrapping* bei jedem Schritt verlieren diese Eigenschaft, weil sie die Schätzung für den nächsten Zustand in jeden Zielwert einsetzen. Ihr Zielwert ändert sich bei jeder Aktualisierung. Die Aktualisierung folgt daher keinem Gradienten einer festen Zielfunktion mehr. Mit nichtlinearen Approximatoren oder mit Daten aus einer anderen Strategie kann die Iteration divergieren :cite:`Tsitsiklis.VanRoy.1997`.

<!--en:cf70873e-->
Wir können nun ein Strategienetz für kontinuierliche Zustände und Aktionen trainieren. Dazu verwenden wir den REINFORCE-Schätzer, eine angepasste Baseline und die Zeitgrenzenkorrektur. Der Agent hat noch drei Einschränkungen. Erstens wartet er auf das Ende jeder Episode, weil seine Zielwerte vollständige Erträge sind. In :numref:`sec_actorcritic` nutzen wir Bootstrapping, um früher zu lernen. Zweitens verwendet er jeden Batch für einen einzigen Schritt und verwirft ihn dann. In :numref:`sec_ppo` nutzen wir einen Batch für einige kontrollierte Schritte mehrfach, und in :numref:`sec_dqn` verwenden wir alte Übergänge erneut. Drittens begrenzt der Agent nicht, wie stark eine Aktualisierung die Strategie verändert. Ein großer Schritt kann sowohl den aktuellen Ertrag als auch die anschließend gesammelten Daten verschlechtern. :numref:`chap_deep_rl` greift alle drei Punkte auf und erklärt damit den Unterschied zwischen den einigen Hunderttausend Schritten hier und den zehntausend bei Soft Actor-Critic.

<!--en:337022ac-->
## Die Verfahren im Überblick

<!--en:8ae0f72f-->
Dieses und das nächste Kapitel stellen zahlreiche Verfahren vor. Zwei Fragen helfen, jedes davon einzuordnen: Was lernt es, und welche Daten darf es für seine Aktualisierungen nutzen? :numref:`fig_rl_roadmap` ordnet die Verfahren nach diesen beiden Fragen.

<!--en:57bc259f-->
![Die Verfahren dieses Kapitels und aus :numref:`chap_deep_rl`, geordnet nach dem Lerngegenstand und den für Aktualisierungen zulässigen Daten. Durchgezogene Rahmen kennzeichnen die Verfahren dieses Kapitels. Gestrichelte Rahmen und graue Namen gehören zu :numref:`chap_deep_rl`. Verhaltensklonen lernt aus einer festen Menge von Demonstrationen. DAgger lernt aus den selbst besuchten Zuständen des lernenden Agenten, für die ein Experte die Aktionen neu vorgibt. Daher steht DAgger in der On-Policy-Spalte. Wertiteration steht in der Off-Policy-Spalte, weil sie für ihre Aktualisierung den bekannten Übergangskern nutzt und keine Daten einer bestimmten Strategie benötigt. Der Pfeil vom Strategiefeld in der On-Policy-Spalte bezeichnet die Wiederverwendung neuerer Batches. Sie erfordert eine Korrektur, weil sich die Strategie seit der Datenerhebung geändert hat. Das gelernte Modell in der Off-Policy-Spalte wird an die eigenen gespeicherten Erfahrungen des Agenten angepasst. Der Agent plant in diesem Modell (:numref:`sec_model_based`).](../img/mdl-rl-roadmap.svg)

<!--en:30a34358-->
:numref:`tab_rl_map` führt die Algorithmen beider Kapitel auf. Sie zeigt, was jeder schätzt, welche Daten er nutzen kann und wo wir ihn einführen.

<!--en:67604796-->
:Die Algorithmen beider Kapitel im Überblick. „Erwähnt in“ kennzeichnet Verfahren, die dieses Buch beschreibt, aber nicht implementiert.

<!--en:20d47c76-->
| Algorithmus | Was schätzt er? | Welche Daten darf er zur Aktualisierung nutzen? | Wo? |

<!--en:caff372e-->
| Wertiteration, Strategieiteration | $V^*$ exakt durch Durchläufe mit dem Bellman-Operator, $V^\pi$ durch Strategiebewertung | keine: nutzt den Übergangskern $P$ und die Belohnung $r$ | :numref:`sec_valueiter` |

<!--en:f5870ab4-->
| Verhaltensklonen | $\pi(a \mid s)$ über die Kreuzentropie auf Demonstrationen | fester Expertendatensatz | :numref:`sec_imitation` |

<!--en:cd81608c-->
| DAgger | dieselbe Anpassung | eigene Zustände des lernenden Agenten, für die ein Experte auf Anfrage die Aktionen neu vorgibt | :numref:`sec_imitation` |

<!--en:895a3690-->
| LQR, iLQR, MPC mit gleitendem Horizont | exakte quadratische Restkosten $V_t(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}_t\mathbf{x}$ und lineare Rückführung $\mathbf{u}_t^* = -\mathbf{K}_t\mathbf{x}_t$ über die Riccati-Rekursion | keine: differenziert die bekannte Dynamik $f(\mathbf{x}, \mathbf{u})$ und die quadratischen Kosten | :numref:`sec_optimal_control` |

<!--en:7376b8a4-->
| Q-Learning (tabellarisch) | $Q^*$ aus stichprobenbasierten Backups | Übergänge einer beliebigen Verhaltensstrategie (Off-Policy) | :numref:`sec_qlearning` |

<!--en:e15e4375-->
| SARSA | $Q$ der eigenen Verhaltensstrategie einschließlich Exploration | eigene Übergänge (On-Policy) | :numref:`sec_qlearning` |

<!--en:79ff0100-->
| Planung mit Sicherheitsäquivalenz | $Q^*$ des Modells, das aus den Häufigkeiten der Samples entsteht | Übergangshäufigkeiten einer beliebigen Verhaltensstrategie | :numref:`sec_qlearning` |

<!--en:c5669b93-->
| UCB, Thompson-Sampling | Mittelwerte der Arme und deren Unsicherheit | eigene Züge am Banditen | :numref:`sec_exploration` |

<!--en:7f3a5650-->
| REINFORCE | $\nabla_\theta J$ über die Score-Funktion | ausschließlich neue Trajektorien der aktuellen Strategie | :numref:`sec_policygradient` |

<!--en:21691c52-->
| REINFORCE mit Baselines, RLOO | denselben Gradienten mit geringerer Varianz | neue Trajektorien der aktuellen Strategie | :numref:`sec_baselines` |

<!--en:20083d19-->
| GRPO | gruppenweise normalisierte Vorteile, ohne Wertnetz | neue Gruppen pro Prompt, jeweils für einige Schritte mit Begrenzung wiederverwendet | Gewichte in :numref:`sec_baselines`, Verfahren in :numref:`sec_rl_sequences` |

<!--en:b814f7bc-->
| REINFORCE mit gelernter Baseline und Netzen | $\pi_\theta$ und $\hat{V}$ als Netze | neue Trajektorien der aktuellen Strategie | :numref:`sec_deeprl` |

<!--en:06547cc1-->
| Akteur-Kritiker-Verfahren (A2C) | $\pi_\theta$ und einen Kritiker mit Bootstrapping | neue Trajektorien aus einer nahezu aktuellen Strategie | :numref:`sec_actorcritic` |

<!--en:3a8bba0e-->
| PPO | $\pi_\theta$ mit begrenztem Wahrscheinlichkeitsverhältnis | ein Batch für mehrere Schritte | :numref:`sec_ppo` |

<!--en:b8b95e96-->
| KL-regularisierte Strategieoptimierung (die Zielfunktion von RLHF) | eine gegenüber der Referenz gemäß der Belohnung verschobene Strategie | Trajektorien, die eine gelernte Belohnungsfunktion bewertet | :numref:`sec_regularized`. RLHF in großem Maßstab behandelt der Buchteil über Sprachmodelle ausführlicher |

<!--en:de6d416f-->
| DQN, Double DQN | $Q$ als Netz | Wiedergabespeicher mit älteren Erfahrungen | :numref:`sec_dqn` |

<!--en:98732b82-->
| Rainbow | DQN mit seinen einzeln untersuchten Komponenten | Wiedergabespeicher | erwähnt in :numref:`sec_deep_rl-further-reading` |

<!--en:461666ee-->
| DDPG, TD3 | einen Kritiker und einen deterministischen Akteur, der lernt, dessen Wert zu maximieren | Wiedergabespeicher (Off-Policy) | erwähnt in :numref:`sec_sac` |

<!--en:19af135c-->
| SAC | zwei weiche Kritiker und einen stochastischen Akteur mit gestauchter Ausgabe, der Belohnung plus Entropie maximiert | Wiedergabespeicher (Off-Policy) | :numref:`sec_sac` |

<!--en:ef6cc582-->
| modellprädiktive Regelung in einem gelernten Modell, Destillation der Suche in die Strategie | ein durch Regression angepasstes Modell der Dynamik und Belohnung sowie durch Suche darin gefundene Pläne | eigene gespeicherte Erfahrungen des Agenten, an die er das Modell erneut anpasst | :numref:`sec_model_based` |

<!--en:55d843ff-->
| Dyna, MBPO | dasselbe Modell zur Simulation zusätzlicher Übergänge für einen Off-Policy-Agenten | Wiedergabespeicher und im Modell simulierte Trajektorien | erwähnt in :numref:`sec_model_based` |

<!--en:c5c0534f-->
| pessimistisches Offline-Q-Learning, CQL | $Q$ mit Strafterm in datenarmen Bereichen | fester aufgezeichneter Datensatz, keinerlei Interaktion | :numref:`sec_offline` |

<!--en:6862cb7e-->
| Decision Transformer | ein auf den Ertrag konditioniertes Sequenzmodell | fester aufgezeichneter Datensatz | erwähnt in :numref:`sec_offline` |

<!--en:3e447af2-->
| MuZero, Dreamer | ein gelerntes Umgebungsmodell zum Planen oder Simulieren | eigene Interaktionen, die im Modell erneut durchgespielt werden | erwähnt in :numref:`sec_deep_rl-further-reading` |

<!--en:553da28d-->
| DPO | das regularisierte Optimum direkt aus Präferenzen | fester Präferenzdatensatz | :numref:`sec_rl_sequences`. Präferenzoptimierung in großem Maßstab behandelt der Buchteil über Sprachmodelle ausführlicher |

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:e121dd15-->
Eine Tabelle für kontinuierliche Zustände bräuchte für jeden Besuch eine Zeile. Deshalb stellen wir die Strategie und die Wertschätzung durch Netze dar. Eine Tabelle ist der Spezialfall eines linearen Netzes für One-Hot-kodierte Zustände. Die Herleitung des Policy-Gradienten benötigt nur den Score. Daher lässt sich REINFORCE mit gelernter Baseline mit drei praktischen Ergänzungen auf Netze übertragen: Wir passen die Baseline mit genügend Regressionsschritten an, damit sie ihren Zielwerten folgt. Wir normalisieren die Gewichte. Bei Episoden, die an einer Zeitgrenze abbrechen, nutzen wir den Wert des letzten Zustands für das Bootstrapping. Die Trainingsschleife folgt dem Gradienten des schrittweise gebildeten Ersatzverlusts. Das ergibt keinen erwartungstreuen Gradienten von $J$, und der Verlustwert sagt nichts über die Güte der Strategie aus. Eine Gauß-Strategie erweitert die Schleife auf kontinuierliche Drehmomente. Damit lernt der Agent in jedem Pendulum-Lauf nach einigen Hunderttausend Schritten, das Pendel aufzuschwingen und zu balancieren. Eine Aktualisierung ändert die Schätzung in jedem Zustand: Sie erhöht die Werte ähnlicher Zustände und senkt die entfernter Zustände. Der Agent wartet ganze Episoden ab, nutzt jeden Batch einmal und begrenzt seine Schritte nicht. An diesen drei Punkten setzt :numref:`chap_deep_rl` an.

<!--en:2294243a-->
## Übungen

<!--en:bf1be588-->
1. **Diskontierung an der Zeitgrenze.** `train_reinforce` liefert für jeden Batch den undiskontierten mittleren Ertrag der Episoden. Die Aktualisierung gewichtet dagegen jeden Schritt mit dem künftigen Ertrag bei $\gamma = 0.99$. Jeder CartPole-Schritt bringt die Belohnung $1$. Die Umgebung bricht jede Episode nach 500 Schritten ab, sofern sie so lange läuft.
    1. Zeigen Sie, dass der diskontierte Ertrag einer Episode der Länge $T$ gleich $(1 - \gamma^T)/(1 - \gamma)$ ist, und berechnen Sie ihn für $T \in \{20, 100, 200, 500\}$. Folgern Sie, dass undiskontierter und diskontierter Ertrag die Episoden in dieselbe Rangfolge bringen. Um welchen Faktor steigt der jeweilige Ertrag, wenn sich eine Episode von 200 auf 500 Schritte verlängert?
    1. Berechnen Sie ohne Zeitgrenzenkorrektur den künftigen Ertrag $\hat{G}_t$ bei $t \in \{0, 400, 450, 490\}$ für eine Episode, die die Zeitgrenze erreicht. Das Wertnetz sieht nur den Zustand. In einer erfolgreichen Episode ähneln die späten Zustände den frühen. Welches Vorzeichen haben die Gewichte der letzten Schritte einer erfolgreichen Episode typischerweise? Wie verändert die Strategieaktualisierung die Wahrscheinlichkeiten der dort gewählten Aktionen?
    1. `train_reinforce` addiert `batch.truncation_bootstrap(gamma, ac.value_np)` zu diesen Zielwerten. Angenommen, $\hat{V}$ sagt in jedem Zustand mit aufrechter Stange den Wert $1/(1 - \gamma)$ voraus. Zeigen Sie, dass der Zielwert $y_t$ dann bei jedem Schritt einer abgebrochenen Episode gleich $1/(1 - \gamma)$ ist, unabhängig vom Abstand zur Zeitgrenze. Warum sinkt der Zielwert gegen Ende einer Episode, die durch das Fallen der Stange terminiert, dennoch ab? Warum ist das richtig?
    1. Bei Pendulum endet jede Episode durch Abbruch, bei CartPole nur die erfolgreichen. Erklären Sie damit, warum das Weglassen der Korrektur das Lernen bei Pendulum viel stärker verlangsamt als bei CartPole.
   :exlabel:`ex_deep-rl-time-limit`

<!--en:e1c1382c-->
1. [code] **Gauß-Score.** `GaussianHead` parametrisiert eine eindimensionale Aktion durch den Mittelwert $\mu$ und die logarithmierte Standardabweichung $\ell = \log \sigma$, im Code `log_std`. Schreiben Sie $z = (a - \mu)/\sigma$ für eine gezogene Aktion $a$.
    1. Zeigen Sie, dass $\partial \log \pi / \partial \mu = z / \sigma$ und $\partial \log \pi / \partial \ell = z^2 - 1$ gelten und dass beide Ableitungen unter der Strategie den Mittelwert null haben.
    1. Sei $Q$ eine quadratische Funktion der Aktion mit zweiter Ableitung $Q''$. Zeigen Sie, dass $\mathbb{E}\big[Q(a)\,(z^2 - 1)\big] = \sigma^2 Q''$ gilt, indem Sie $\mathbb{E}[z^4] = 3$ verwenden. In welche Richtung verändert der erwartete Policy-Gradient-Schritt $\sigma$ in der Nähe eines Maximums von $Q$? Was folgt daraus für die Exploration einer Strategie, deren Mittelwert sich der besten Aktion nähert?
    1. Pendulum begrenzt jedes gezogene Drehmoment auf $[-2, 2]$, während `log_prob` das ursprüngliche Sample bewertet. Erklären Sie, warum der Score-Funktionsschätzer den Ertrag in der Umgebung mit begrenztem Drehmoment weiterhin erwartungstreu schätzt.
    1. Erzeugen Sie den Agenten aus `pendulum_agent` mit `GaussianPolicy(3, 1, state_std=True)`, sodass $\sigma$ vom Zustand abhängt. Trainieren Sie mit drei Zufallsstartwerten und `train_reinforce` wie im Pendulum-Lauf. Erfassen Sie alle 25 Aktualisierungen $\sigma$ in der Hängelage $(\cos\vartheta, \sin\vartheta, \dot{\vartheta}) = (-1, 0, 0)$ und in der aufrechten Lage $(1, 0, 0)$. Wo sinkt $\sigma$ zuerst, und wie passt das zu Teil b? Vergleichen Sie die Ertragskurve mit `runs_p`.

    *Nach Richard S. Sutton und Andrew G. Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf), Aufgabe 13.4.*
   :exlabel:`ex_deep-rl-gaussian-score`

<!--en:2a5a8454-->
1. [code] **Unstetige Belohnungen.** Die zweite Gleichheit in :eqref:`eq_score_vs_pathwise` differenziert durch $Q$ hindurch. Verwenden Sie $\mu = 0$ und $\sigma = 0.5$ wie bei der Berechnung von `g_score` und `g_path`. Nehmen Sie als Belohnung die Schwellenfunktion $Q(a) = 1$ für $a > 1$ und $Q(a) = 0$ sonst.
    1. Berechnen Sie den exakten Gradienten $\nabla_\mu \mathbb{E}[Q(a)]$. Was liefert der pfadweise Schätzer für jedes Sample? Welcher Schritt in der Herleitung der zweiten Gleichheit ist hier unzulässig?
    1. Ersetzen Sie `Q` in der Berechnung durch die glatte Stufe $Q_k(a) = 1 / \big(1 + e^{-k(a - 1)}\big)$ für $k \in \{1, 10, 100, 1000\}$. Erfassen Sie Mittelwert und Varianz von `g_score` und von `g_path`. Die Samples liegen in einfacher Genauigkeit vor. Für $k \geq 100$ kommt es deshalb bei den kleinsten Samples zu einem Überlauf des Faktors $e^{-k(a - 1)}$. Die Ableitung der Formel in dieser Schreibweise ergibt dort NaN. Implementieren Sie $Q_k$ mit einer numerisch stabilen Sigmoidfunktion. Tragen Sie beide Varianzen gegen $k$ auf. Welcher Schätzer hat bei großen $k$ die kleinere Varianz? Erklären Sie, wie die Varianz von `g_path` von $k$ über die Höhe und Breite von $Q_k'$ abhängt und wie die Varianz von `g_score` von $k$ über $Q_k$ abhängt.
    1. Pendulum begrenzt jedes gezogene Drehmoment auf $[-2, 2]$. Was liefert eine pfadweise Schätzung durch das begrenzte Drehmoment für Samples außerhalb dieses Intervalls? Warum gehen diese Samples beim Score-Funktionsschätzer aus Aufgabe :exref:`ex_deep-rl-gaussian-score`c nicht verloren?

<!--en:4fa94afd-->
1. **Wert des Ersatzverlusts.** `policy_step` wertet den Ersatzverlust :eqref:`eq_pg_surrogate_loss` auf dem Batch aus, den die aktuelle Strategie gerade gesammelt hat. Die Gewichte aus `d2l.standardize` haben den Mittelwert $0$ und die Standardabweichung $1$ über die $N$ Schritte des Batchs.
    1. Zeigen Sie, dass $L(\theta) = -s\, c$ gilt. Dabei ist $s$ die Standardabweichung von $\log \pi_\theta(a_t \mid s_t)$ über den Batch und $c$ der Korrelationskoeffizient zwischen den Gewichten und diesen Log-Wahrscheinlichkeiten. Folgern Sie, dass $|L(\theta)| \leq s$ gilt.
    1. Zeigen Sie für `GaussianPolicy` mit einer Aktionsdimension, dass $s \approx 1/\sqrt{2}$ für jedes Mittelwertnetz und jedes $\sigma$ gilt. Was folgt daraus für den Strategieverlust der Pendulum-Läufe?
    1. Wie verhält sich $s$ bei CartPole, wenn die Strategie nahezu deterministisch wird? Was folgt daraus für den Wert von $L$ gegen Ende des Trainings?
   :exlabel:`ex_deep-rl-surrogate-value`

<!--en:96643939-->
1. [code] **Normalisierung bei Adam.** ● Beim einfachen SGD ändert die Division der Gewichte durch ihre Standardabweichung die effektive Lernrate (Aufgabe :exref:`ex_baselines-step-size-confound-quantified` aus :numref:`sec_baselines`). `train_reinforce` verwendet für die Aktualisierung stattdessen Adam. Für jede Koordinate lautet der Schritt $-\eta\, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon_0)$. Dabei sind $\hat{m}_t$ und $\hat{v}_t$ verzerrungskorrigierte gleitende Mittelwerte des Gradienten $g_t$ und seines Quadrats $g_t^2$ mit den Abklingraten $\beta_1 = 0.9$ und $\beta_2 = 0.999$.
    1. Zeigen Sie für $\epsilon_0 = 0$, dass die Multiplikation jedes Gradienten mit derselben Konstanten $c > 0$ jeden Adam-Schritt unverändert lässt.
    1. Zeigen Sie die Gleichung

       $$\hat{v}_t = \frac{1 - \beta_2}{1 - \beta_2^t} \sum_{i=1}^{t} \beta_2^{t-i} g_i^2.$$

       Um welchen Faktor unterscheiden sich die Gewichte des ältesten und des neuesten quadrierten Gradienten bei $t = 300$, der Länge der Pendulum-Läufe? Nehmen Sie an, der Gradient einer Koordinate behält sein Vorzeichen bei. Nach der Hälfte eines Laufs mit 300 Aktualisierungen sinkt sein Betrag auf ein Zehntel. Um welchen Faktor ist der Schritt am Ende des Laufs kleiner als zu Beginn?
    1. Behalten Sie die Zentrierung durch `d2l.standardize` in `train_reinforce` bei, lassen Sie aber die Division weg. Damit lauten die Gewichte `A - A.mean()` mit `A = G - ac.value_np(batch.obs)`, und die beiden Varianten unterscheiden sich nur in der Skala. Protokollieren Sie die Standardabweichung der Gewichte für jeden Batch. Trainieren Sie beide Varianten mit je drei Zufallsstartwerten auf CartPole und Pendulum. Wie entwickelt sich die Skala der Gewichte bei den beiden Aufgaben? Bei welcher Aufgabe sollte das Weglassen der Division nach Teil b die Ertragskurven stärker verändern?
   :exlabel:`ex_deep-rl-normalization-adam`

<!--en:f9ec0418-->
1. **Varianz beim Score-Funktionsschätzer und beim pfadweisen Schätzer.** ● Die Samples `g_score` und `g_path` schätzen die beiden Formen aus :eqref:`eq_score_vs_pathwise` für $Q(a) = -(a - 1)^2$ bei $\mu = 0$. Verwenden Sie $\mathbb{E}[z^4] = 3$ und $\mathbb{E}[z^6] = 15$ für $z \sim \mathcal{N}(0, 1)$.
    1. Zeigen Sie, dass bei einem einzelnen Sample die Varianz des Score-Funktionsschätzers $15\sigma^2 + 14 + 1/\sigma^2$ und die des pfadweisen Schätzers $4\sigma^2$ beträgt. Berechnen Sie beide Werte bei $\sigma = 0.5$ und vergleichen Sie sie mit den ausgegebenen Varianzen.
    1. Die Addition einer Konstanten $c$ zu $Q$ entspricht der Subtraktion der Baseline $b = -c$. Setzen Sie den Score $a/\sigma^2$ in die Formel für $b^*$ aus Aufgabe :exref:`ex_baselines-optimal-constant-baseline` in :numref:`sec_baselines` ein. Zeigen Sie damit, dass $b^* = -(1 + 3\sigma^2)$ gilt und dass die kleinste Varianz des Score-Funktionsschätzers $6\sigma^2 + 8$ beträgt. Vergleichen Sie die ausgegebene Varianz für $c = 10$ mit Ihrer Formel.
    1. Betrachten Sie den Grenzübergang $\sigma \to 0$. Wie verhalten sich die beiden Varianzen mit und ohne optimale Baseline? Erklären Sie den Grenzwert $8$ der Varianz des Score-Funktionsschätzers anhand der Taylor-Entwicklung von $Q$ um $\mu$. Welcher der beiden Schätzer bleibt für einen deterministischen Akteur einsetzbar?
   :exlabel:`ex_deep-rl-score-pathwise-variance`

<!--en:d347fef3-->
1. [code] **Tangentenkernel.** ● Bei der Messung von `dV` erhöhen wir $\hat{V}$ in einem Testzustand mit einem einfachen SGD-Schritt von `fit_value`. Bezeichnen Sie mit $w$ die Parameter des Wertnetzes und mit

   $$k(s, s') = \nabla_w \hat{V}(s)^\top \nabla_w \hat{V}(s')$$

   seinen Tangentenkernel wie in :eqref:`eq_deeprl_kernel`.
    1. Zeigen Sie, dass ein Gradientenschritt der Weite $\eta$ für $\frac{1}{2}\big(y - \hat{V}(s_0)\big)^2$ die Schätzung in $s$ um $\eta\, \big(y - \hat{V}(s_0)\big)\, k(s_0, s) + O(\eta^2)$ ändert. `fit_value` minimiert stattdessen $\big(y - \hat{V}(s_0)\big)^2$. Wie ändert sich die Vorhersage?
    1. Berechnen Sie für ein neu erzeugtes `net_probe` die Gradienten $\nabla_w \hat{V}$ in den 256 Testzuständen `S` sowie die Kernel-Werte $k(s_0, s)$ für $s_0$ = `S[0]`. Machen Sie ohne Adam einen einfachen Gradientenschritt mit $\eta = 10^{-3}$ in Richtung $y = \hat{V}(s_0) + 1$. Vergleichen Sie für jeden Testzustand die gemessene Änderung mit der Vorhersage.
    1. Tragen Sie $k(s_0, s) / k(s_0, s_0)$ gegen den Abstand $\|s - s_0\|$ auf. Wie viele Testzustände haben einen negativen Kernel-Wert? Vergleichen Sie diese Zahl mit der Anzahl der Testzustände, für die die Codezelle eine Abnahme meldet.
    1. Eine CartPole-Strategie, die gut balanciert, sammelt Batches mit überwiegend fast senkrechter Stange. Erklären Sie anhand des Kernels, wie das Training mit einem solchen Batch die Wertschätzung und die Strategie bei großen Stangenwinkeln verändert, die im Batch nie vorkommen. Warum kann diese Änderung die Strategie bei solchen Winkeln verschlechtern?
   :exlabel:`ex_deep-rl-tangent-kernel`

<!--en:504906ed-->
[Diskussionen](https://d2l.discourse.group/t/36800)

<!--en:3120a937-->
Funktionsapproximation im bestärkenden Lernen<br>**Tabellen als lineare One-Hot-Netze · neuronaler Tangentenkernel · ActorCritic-MLP · Bootstrapping bei Abbruch · Gauß-Strategien · Score- und pfadweise Gradienten**

<!--en:f4dbbc9d-->
::: {.slide title="Warum kontinuierliche Regelung Funktionsapproximation erfordert"}

<!--en:b0445376-->
[Motivation und Vorstellung]{.kicker}

<!--en:2e9e7f94-->
- **Kein zweiter Besuch desselben Zustands in `CartPole-v1`:** Da $\mathbf{s}_t = (x_t, \dot{x}_t, \vartheta_t, \dot{\vartheta}_t)^\top \in \mathbb{R}^4$ kontinuierlich ist, trifft der Agent in $1{,}000$ Episoden denselben Zustandsvektor kein zweites Mal an.

<!--en:f5f8c63e-->
- **Generalisierung durch gemeinsame Gewichte:** Eine Tabelle lernt für $\vartheta + 10^{-4}$ nichts aus einer Aktualisierung bei $\vartheta$. Ein neuronales Netz nutzt dieselben Gewichte $\theta$ im gesamten Zustandsraum.

<!--en:4b4d4682-->
::: {.slide title="Tabellen als lineare One-Hot-Netze und der Tangentenkernel"}

<!--en:d055dc30-->
[Wie sich eine Aktualisierung auf andere Zustände auswirkt]{.kicker}

<!--en:5596b72b-->
- **Unabhängige Tabelleneinträge ($k(s, s_0) = \mathbf{1}(s = s_0)$):** Bei One-Hot-Merkmalen $\nabla_w \hat{V}_w(s) = \mathbf{e}_s$ bleibt jeder unbesuchte Zustand unberührt.

<!--en:d61236e9-->
- **Kopplung im Netz ($k(s, s_0) = \nabla_w \hat{V}_w(s)^\top \nabla_w \hat{V}_w(s_0)$):** Bei nahen Zuständen $s \approx s_0$ ist der Kernel positiv (Generalisierung). Auch bei entfernten Zuständen kann er von null verschieden, sogar negativ sein (Interferenz).

<!--en:62573b8e-->
::: {.slide title="Der ActorCritic-Container und MLP mit tanh"}

<!--en:5680b301-->
[Netze ersetzen Strategie- und Werttabellen]{.kicker}

<!--en:14863a98-->
- **Separate `tanh`-MLP mit je 64 Einheiten:** `ActorCritic` kombiniert ein Akteurnetz (`opt_pi`) mit einem Kritikernetz für die Baseline (`opt_v`).

<!--en:ba6f27d0-->
- **Warum `tanh` als Aktivierung:** Bei niedrigdimensionalen On-Policy-Regelungsaufgaben schneiden glatte Aktivierungsflächen besser ab als ReLU (Andrychowicz et al., 2021).

<!--en:0fcd1bc1-->
::: {.slide title="Mehrere Regressionsschritte pro Batch für das Wertnetz"}

<!--en:d5fa5963-->
[Wie die Baseline des Kritikers genau bleibt]{.kicker}

<!--en:4b55998c-->
Anders als bei der direkten Tabellenaktualisierung ($\eta_V = 0.1$) nähert sich ein Wertnetz den Zielwerten $y_t \sim 100$ durch Adam-Schritte der Weite $10^{-2}$:

<!--en:99050266-->
- **Warum 1 Gradientenschritt pro Batch zu wenig ist:** Bei Pendulum erklärt `num_steps=1` pro Batch nur $40\%$–$80\%$ der Ertragsvarianz ($1 - \textrm{Var}(y - \hat{V})/\textrm{Var}(y)$). In den Gewichten des Strategieupdates bleibt starkes Rauschen. `num_steps=40` erhöht die erklärte Varianz auf $\approx 99\%$.

<!--en:6bbae790-->
::: {.slide title="Bootstrapping bei Abbruch an der Zeitgrenze"}

<!--en:b89031a5-->
[Künstliche Strafen für lange Episoden vermeiden]{.kicker}

<!--en:d0e01247-->
- **Größe des Effekts bei CartPole ($\gamma = 0.99$):** In Schritt $t = 490$ eines erfolgreichen Laufs über 500 Schritte ($\hat{V} \approx 100$) ergibt $\hat{G}_{490} \approx 9.6$ ohne Bootstrapping einen Vorteil von $-90.4$. Das bestraft erfolgreiches Balancieren.

<!--en:76d2bffe-->
::: {.slide title="Die Trainingsschleife für REINFORCE mit Netzen"}

<!--en:22709e19-->
[Trajektorien, Bootstrapping bei Abbruch und Regression für das Wertnetz]{.kicker}

<!--en:34477627-->
`train_reinforce` sammelt 8 Episoden pro Aktualisierung, addiert `truncation_bootstrap` zu `reward_to_go` und standardisiert die empirischen Vorteile $y_t - \hat{V}_w(s_t)$. Die Schleife aktualisiert den Akteur einmal und passt den Kritiker in 40 Schritten an:

<!--en:9b1bb91c-->
- **Vier praktische Abweichungen von $\nabla_\theta J$:** (1) Ohne $\gamma^t$ zählen alle Schritte gleich. (2) Bootstrapping mit $s_T$ überträgt die Verzerrung des Kritikers an der Abbruchstelle. (3) Die Zentrierung im Batch koppelt die Samples. (4) Die Division durch das $\sigma$ des Batchs und durch $N$ skaliert die Gradienten vor Adams koordinatenweiser Normalisierung.

<!--en:7d327804-->
::: {.slide title="REINFORCE mit gelernter Baseline für CartPole"}

<!--en:ed5fad26-->
[Die Obergrenze von 500 Schritten erreichen]{.kicker}

<!--en:8d42676d-->
Das Training mit 3 Zufallsstartwerten für `ActorCritic.mlp(4, 2)` auf `CartPole-v1` mit $\gamma = 0.99$ (effektiver Horizont 100 Schritte) löst die Aufgabe innerhalb von 50 Aktualisierungen ($\approx 120{,}000$ Umgebungsschritte):

<!--en:f7c56124-->
- **Rascher Anstieg nach zufälligen Aktionen:** Untrainierte MLP lassen die Stange nach $\approx 20$ Schritten fallen. Mit allen 3 Zufallsstartwerten erreichen sie bis zur 50. Aktualisierung die Obergrenze von $500$ Schritten.

<!--en:67e97ec8-->
- **Rolle des Bootstrappings bei Abbruch:** Ohne `truncation_bootstrap` fällt der Ertrag bei späten Aktualisierungen oft unter $500$, weil erfolgreiche Trajektorien ihre letzten Schritte bestrafen.

<!--en:aaa9e489-->
::: {.slide title="Warum der Wert des Ersatzverlusts wenig über die Strategie aussagt"}

<!--en:6b4df173-->
[Das Training beim bestärkenden Lernen verfolgen]{.kicker}

<!--en:431f790d-->
Deep-Learning-Frameworks berechnen den Policy-Gradienten durch Differenzieren des skalaren **Ersatzverlusts** $L(\theta) = -\frac{1}{N}\sum_t \tilde{w}_t \log \pi_\theta(a_t \mid s_t)$. Dabei ist $\tilde{w}_t$ vom Berechnungsgraphen getrennt:

<!--en:80f5616d-->
- **Der Gradient stimmt, der Zahlenwert misst keinen Ertrag:** Während der CartPole-Ertrag von $20$ auf $500$ steigt, bleibt $L(\theta)$ nahe $-0.05$, weil die standardisierten Gewichte $\tilde{w}_t$ den Mittelwert $0$ und Varianz eins haben.

<!--en:cb27c277-->
- **Regel für die Fehlersuche:** Stets die Episodenerträge, die vom Wertnetz erklärte Varianz und die Strategieentropie beobachten. Der skalare Wert von $L(\theta)$ eignet sich dafür nicht.

<!--en:0a594cb1-->
::: {.slide title="Kontinuierliche Aktionen mit der Gauß-Strategie"}

<!--en:25522586-->
[Parametrisierung reellwertiger Drehmomente]{.kicker}

<!--en:2fec61b0-->
Bei kontinuierlicher Regelung (`Pendulum-v1`, Drehmoment $u \in [-2, 2]$) modelliert `GaussianPolicy` die Verteilung $\pi_\theta(a \mid s) = \mathcal{N}(a;\; \mu_\theta(s), \sigma^2)$ mit einem MLP für den Mittelwert $\mu_\theta(s)$ und einem lernbaren Parameter $\log\sigma$ für die logarithmierte Standardabweichung:

<!--en:b03d695d-->
Dabei gilt $z = (a - \mu_\theta(s))/\sigma$.

<!--en:7a73b2f1-->
- **Wie sich $\mu$ und $\sigma$ anpassen:** Ein positives Vorteilsgewicht $\tilde{w}_t > 0$ zieht den Mittelwert $\mu_\theta(s)$ in Richtung $a_t$ (über $z/\sigma$). Es erhöht $\sigma$, wenn ein Ausreißer mit $|z| > 1$ erfolgreich ist, und senkt $\sigma$, wenn Aktionen nahe dem Mittelwert mit $|z| < 1$ erfolgreich sind.

<!--en:487644cf-->
::: {.slide title="Aufschwingen des Pendels mit neuronalem REINFORCE"}

<!--en:1a97dd48-->
[Initialisierung mit Mittelwert null und Stichprobenbedarf]{.kicker}

<!--en:973394ea-->
Nach dem Training von `GaussianPolicy(3, 1, zero_mean=True)` auf `Pendulum-v1` über 300 Aktualisierungen mit je 8 Episoden (insgesamt $480{,}000$ Schritte) schwingt der Agent das Pendel bei allen Zufallsstartwerten auf und balanciert es:

<!--en:f8c4990f-->
- **Warum `zero_mean=True` wichtig ist:** Die Nullinitialisierung der letzten Mittelwertschicht ($\mu_0(s) = 0, \sigma_0 = 1$) verhindert, dass die Drehmomente anfangs an den Grenzen $[-2, 2]$ sättigen und die Belohnung keine Rückmeldung mehr gibt.

<!--en:d3fbf1b9-->
- **Hoher Stichprobenbedarf bei On-Policy-Verfahren im Vergleich zu §{sec_optimal_control} und §{sec_sac}:** Der iLQR-Planer aus §{sec_optimal_control} differenziert die bekannten Bewegungsgleichungen direkt. Modellfreies REINFORCE lernt allein aus trigonometrischen Beobachtungen $(\cos\vartheta, \sin\vartheta, \dot{\vartheta})$ und skalaren Belohnungen. Es braucht $150$–$300$ Aktualisierungen ($\sim 300{,}000$ Schritte), um die in §{sec_optimal_control} verwendete Schwelle von $-200$ für erfolgreiches Balancieren zu überschreiten. Das Off-Policy-Verfahren SAC (§{sec_sac}) löst die Aufgabe in $5{,}000$–$11{,}000$ Schritten.

<!--en:e0f3214b-->
::: {.slide title="Aktionsgradienten über die Score-Funktion oder den pfadweisen Ansatz"}

<!--en:9c4c2a98-->
[Zwei Wege zur Ableitung bei kontinuierlichen Aktionen]{.kicker}

<!--en:3e5e5320-->
Ist der Ertrag oder der Kritiker $Q(a)$ eine differenzierbare Funktion einer kontinuierlichen Aktion $a = \mu + \sigma z$ mit $z \sim \mathcal{N}(0, 1)$, lässt sich der Gradient über die Score-Funktion mit dem **pfadweisen Gradienten** vergleichen (Kingma & Welling, 2014):

<!--en:f09bbd6c-->
- **20× geringere Varianz und Invarianz gegenüber Verschiebung:** Für $Q(a) = -(a-1)^2$ bei $\mu = 0, \sigma = 0.5$ hat der pfadweise Schätzer die Varianz $1.00$ gegenüber $21.5$ bei der Score-Funktion (exakt: $21.75$). Bei Addition von $+10$ zu $Q(a)$ bleibt der pfadweise Schätzer unverändert, während die Varianz des Score-Funktionsschätzers auf $281$ steigt.

<!--en:4059089b-->
::: {.slide title="Informationsfluss bei Score-Funktion und pfadweiser Ableitung durch den Kritiker"}

<!--en:5f909a7c-->
[Belohnungen aus einer Blackbox oder Verzerrung durch den Kritiker]{.kicker}

<!--en:6bf6657d-->
Der Vergleich der Berechnungsgraphen zeigt, warum On-Policy-REINFORCE die Score-Funktion verwendet und kontinuierliche Off-Policy-Akteur-Kritiker-Verfahren (DDPG, TD3, SAC) pfadweise Gradienten nutzen:

<!--en:8874e5b1-->
- **Score-Funktion (a):** Behandelt die Umgebung als nicht differenzierbare Blackbox, die skalare Werte $R$ zurückgibt. Das funktioniert für diskrete Aktionen und unbekannte Dynamik ohne Verzerrung durch die Steigung eines Kritikers.

<!--en:9093c6c4-->
- **Pfadweiser Gradient (b):** Berechnet die Ableitung $\partial Q_w / \partial a \cdot \partial a / \partial \theta$ durch einen gelernten neuronalen Kritiker $Q_w(s, a)$. Die Varianz sinkt stark, doch Fehler in $\partial Q_w / \partial a$ wirken sich auf den Gradienten aus.

<!--en:3c9642fd-->
::: {.slide title="Kopplung zwischen Zuständen mit dem Tangentenkernel messen"}

<!--en:6ef0380c-->
[Generalisierung und Interferenz in einem MLP zur Wertschätzung]{.kicker}

<!--en:ff11e719-->
Ein einfacher SGD-Schritt erhöht $\hat{V}_w(s_0)$ in einem Zustand $s_0 \in [-1, 1]^4$ um $+0.50$. Die Messung am `tanh`-MLP mit 64 Einheiten zeigt die Wirkung auf 255 weitere Zufallszustände, geordnet nach ihrem euklidischen Abstand $\|s - s_0\|$:

<!--en:16c22031-->
- **Lokale Generalisierung und Drift in der Ferne:** In den 10 nächsten Zuständen in $\mathbb{R}^4$ steigen die Werte um $+0.43$ bis $+0.49$ (fast $90\%$ der bei $s_0$ gemessenen Änderung von $+0.49$ bis $+0.55$). In den 10 entferntesten Zuständen fällt die Änderung leicht negativ aus ($-0.01$ bis $-0.06$, bei $6$–$15$ von $255$ Zuständen geht der Wert *zurück*). In der Tabelle beträgt die Änderung $+0.50$ bei $s_0$ und $0.000000$ überall sonst.

<!--en:bf381d44-->
::: {.slide title="Unabhängige Tabelleneinträge und gekoppelte Netzwerte im Bild"}

<!--en:71a8d2a0-->
[Die zwei Seiten der Funktionsapproximation]{.kicker}

<!--en:2f69fceb-->
Ein eindimensionaler Schnitt durch die Wertfunktion vor und nach einer einzelnen Aktualisierung um $+0.50$ bei $x_0$ zeigt die Stärke und die Anfälligkeit des bestärkenden Lernens mit Netzen:

<!--en:0eb726c9-->
- **Generalisierung (der positive Buckel um $x_0$):** Erfahrung bei $x_0$ verbessert sofort die Informationsgrundlage für benachbarte Zustände, auch ohne sie zu besuchen.

<!--en:527fd6a1-->
- **Interferenz und die tödliche Triade:** Das Training mit Batches aus aufrechten CartPole-Zuständen verändert $\hat{V}_w(s)$ bei großen Neigungswinkeln. Zusammen mit Bootstrapping und Off-Policy-Daten (der **tödlichen Triade** aus §{sec_dqn}) kann die globale Kopplung durch gemeinsame Parameter das Lernen von Werten mit Bootstrapping zur Divergenz bringen. Tsitsiklis & Van Roy (1997) zeigten diese Divergenz.

<!--en:787fd309-->
::: {.slide title="Überblick über Kapitel §{num:chap_reinforcement_learning} und Kapitel §{num:chap_deep_rl}"}

<!--en:d9b3ea77-->
[Was jedes Verfahren lernt und welche Daten es nutzt]{.kicker}

<!--en:dd11e162-->
Zwei Achsen ordnen jeden Algorithmus aus den Kapiteln §{num:chap_reinforcement_learning} und §{num:chap_deep_rl} ein: **Welche Größe parametrisiert er** (Werte, Strategie oder Dynamikmodell) und **welche Datenverteilung bestimmt seine Aktualisierung**?

<!--en:cba37c25-->
- **Durchgezogene Rahmen (Kapitel §{num:chap_reinforcement_learning}):** exakte dynamische Programmierung mit $P$, tabellarische TD-Verfahren (Q-Learning/SARSA), Banditen (UCB/Thompson), Imitation (BC/DAgger), optimale Regelung (LQR/iLQR/MPC) und On-Policy-Verfahren mit Monte-Carlo-Schätzung des Policy-Gradienten (REINFORCE/RLOO/GRPO).

<!--en:906c2821-->
- **Gestrichelte Rahmen (Kapitel §{num:chap_deep_rl}):** Akteur-Kritiker-Verfahren mit Bootstrapping (A2C), mehrfache Batch-Nutzung mit Begrenzung (PPO), Wertnetze mit Wiedergabespeicher (DQN, SAC), gelernte Dynamikmodelle und pessimistisches bestärkendes Offline-Lernen.

<!--en:b54c51ac-->
::: {.slide title="Verständnisfragen zur Funktionsapproximation im bestärkenden Lernen"}

<!--en:929b7671-->
[Wissensprobe]{.kicker}

<!--en:dc750f4d-->
Drei Fragen zu Bootstrapping an der Zeitgrenze, Gauß-Scores und pfadweisen Gradienten:

<!--en:e4355598-->
1. **Bootstrapping bei Abbruch:** Gegeben ist Schritt $t = 490$ einer CartPole-Episode mit 500 Schritten ($\gamma = 0.99$). Für alle aufrechten Zustände gelte $\hat{V}_w(s) \approx 100$. Wie groß sind der künftige Ertrag $\hat{G}_{490}$ ohne Bootstrapping und der Zielwert $y_{490}$ mit Bootstrapping?

<!--en:005d1068-->
2. **Abnehmende Exploration bei der Gauß-Strategie:** Bei einem Gauß-Score von $\partial_{\log\sigma} \log \pi_\theta(a \mid s) = z^2 - 1$ sinkt $\sigma$ automatisch, wenn sich das Mittelwertnetz $\mu_\theta(s)$ der optimalen Aktion $a^*(s)$ nähert. Warum?

<!--en:c9c052eb-->
3. **Score-Funktion und pfadweiser Schätzer bei unstetiger Belohnung:** Ein Prüfprogramm vergibt die stufenförmige Belohnung $Q(a) = \mathbf{1}(a > 1)$. Welchen Gradienten liefert der pfadweise Schätzer $\mathbb{E}[Q'(\mu + \sigma z)]$, und warum funktioniert der Score-Funktionsschätzer weiterhin?

<!--en:557b98b6-->
**Lösungen:**

<!--en:ff5d2be7-->
1. Ohne Bootstrapping gilt $\hat{G}_{490} = \sum_{k=0}^9 0.99^k = (1 - 0.99^{10})/0.01 \approx 9.56$. Daraus ergibt sich der fälschlich negative Vorteil $9.56 - 100 = -90.44$. Die Addition von $\gamma^{10}\hat{V}_w(s_{500}) = 0.99^{10}(100) \approx 90.44$ ergibt wieder $y_{490} \approx 100.0$ und einen Vorteil von null.

<!--en:709ee633-->
2. Sobald $\mu_\theta(s) \approx a^*(s)$ gilt, erzielen Samples nahe dem Mittelwert ($|z| < 1$, also $z^2 - 1 < 0$) den höchsten Ertrag und erhalten positive Vorteilsgewichte $\tilde{w}_t > 0$. Entfernte Samples ($|z| > 1$) erhalten negative Gewichte. Beides senkt $\log\sigma$.

<!--en:d364c075-->
3. Da $Q'(a) = 0$ überall außer an der Sprungstelle $a = 1$ gilt, liefert der pfadweise Schätzer für jedes Sample $0$ und versagt vollständig. Der Score-Funktionsschätzer differenziert $\log \pi_\theta(a)$ statt $Q(a)$ und bleibt bei unstetigen Belohnungen erwartungstreu.

<!--en:0158bb90-->
::: {.slide title="Zusammenfassung: Vom tabellarischen bestärkenden Lernen zur Funktionsapproximation"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:bf227398-->
| Bestandteil | Mathematische Form | Zweck und Verhalten |

<!--en:76f449c9-->
| **Tangentenkernel** | $k(s, s_0) = \nabla_w \hat{V}_w(s)^\top \nabla_w \hat{V}_w(s_0)$ | Bestimmt, wie ein Schritt bei $s_0$ auf nahe $s$ generalisiert und in der Ferne Interferenz verursacht |

<!--en:c7c46193-->
| **Bootstrapping bei Abbruch** | $y_t = \hat{G}_t + (1 - \textrm{term}_T)\gamma^{T-t}\hat{V}_w(s_T)$ | Ergänzt den unbeobachteten künftigen Wert beim Abbruch einer Episode an der Zeitgrenze |

<!--en:e9e5ab22-->
| **Ersatzverlust** | $L(\theta) = -\frac{1}{N}\sum_t \tilde{w}_t \log \pi_\theta(a_t \mid s_t)$ | Liefert durch automatisches Differenzieren $-\hat{g}$. Sein skalarer Wert $-s_{\log\pi}\textrm{corr}$ gibt keinen Aufschluss über den Ertrag |

<!--en:d275f775-->
| **Gauß-Strategie** | $\pi_\theta(a \mid s) = \mathcal{N}(a;\; \mu_\theta(s), \sigma^2)$ | Bewertet kontinuierliche Aktionen über $\partial_\mu \log\pi = z/\sigma$ und $\partial_{\log\sigma}\log\pi = z^2 - 1$ |

<!--en:687ab006-->
| **Pfadweiser Gradient** | $\nabla_\mu \mathbb{E}[Q(a)] = \mathbb{E}_{z \sim \mathcal{N}(0, 1)}[Q'(\mu + \sigma z)]$ | $20\times$ geringere Varianz als beim Score-Funktionsschätzer, erfordert aber einen differenzierbaren Kritiker $Q_w$ |
