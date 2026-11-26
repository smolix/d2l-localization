<!--en:9ab3c867-->
# Dynamische Programmierung

<!--en:730c0ea3-->
Im vorigen Abschnitt haben wir den zugefrorenen See durch zwei Arrays und einen Diskontfaktor beschrieben. Eine Frage blieb offen. In Feld 9 ist *unten* der einzige Befehl, bei dem der Agent nicht in das benachbarte Loch rutschen kann. Allerdings rutscht er dabei in zwei von drei Fällen zur Seite. Ist diese Sicherheit die verlorenen Züge wert? Auf griffigem Eis wäre die Entscheidung klar. Der kürzeste Weg zum Ziel dauert sechs Züge, und am besten folgt man ihm. Auf rutschigem Eis führt der kürzeste Weg an den Löchern vorbei. Ein Befehl, der den Agenten ein Feld voranbringt, kann die Episode beenden, wenn der Agent ausrutscht. Welcher Befehl der beste ist, lässt sich am nächsten Feld allein nicht ablesen. Entscheidend ist, wie viel Belohnung der Agent von jedem möglichen Landefeld aus noch erwarten kann, wenn er dort weiterhin gut handelt.

<!--en:c7da81f1-->
Richard Bellman untersuchte solche mehrstufigen Entscheidungen in den 1950er-Jahren. Er fand heraus, dass diese Bewertungsgröße, der *Wert* (engl. value) eines Zustands, eine lokale Gleichung erfüllt :cite:`BellmanDPBook`. Der Wert eines Feldes ergibt sich aus der Belohnung für einen Zug plus dem diskontierten Wert des erreichten Feldes, gemittelt über die möglichen Rutschbewegungen. Die Werte hängen von der gesamten Zukunft ab, doch jeder einzelne Wert ist nur mit denen seiner Nachbarn verknüpft. Deshalb lassen sich alle Werte gemeinsam berechnen, indem wir wiederholt einen Schritt vorausschauen. Bellman nannte das Verfahren *dynamische Programmierung* (engl. dynamic programming). Es liefert die exakten Lösungen, denen sich die Lernverfahren späterer Abschnitte annähern sollen. An einem kleinen Problem können wir diese Verfahren damit überprüfen.

<!--en:a05bff06-->
Wir definieren die Werte für eine feste Strategie (engl. policy) und für die beste Strategie, leiten die Bellman-Gleichungen her und zeigen, dass die wiederholte Vorausschau konvergiert. Eine Abbruchregel gibt an, wann wir aufhören können. Auf dem rutschigen See erreicht die so gefundene Strategie das Ziel etwa sechzehnmal so oft wie der kürzeste Weg. In vier Feldern führt sie vom Ziel weg. Wir rechnen nach, warum. Zum Schluss berechnen wir, wie sich der Annäherungsbonus aus dem vorigen Abschnitt auf die beste Strategie auswirkt. Alle Berechnungen greifen auf die bekannte Übergangstabelle $P$ zu.

<!--en:63c44335-->
## Werte und Bellman-Gleichungen

<!--en:d1ffe34a-->
### Strategien und Werte

<!--en:960b970d-->
In :numref:`sec_mdp` haben wir jede Regel zur Wahl der Aktionen als Strategie bezeichnet. Allgemein ist eine *Strategie* $\pi(a \mid s)$ eine bedingte Verteilung über die Aktionen $a \in \mathcal{A}$ bei gegebenem Zustand $s \in \mathcal{S}$. Auf einem Feld des Sees könnten die Wahrscheinlichkeiten für *links*, *unten*, *rechts* und *oben* beispielsweise $(0.4, 0.2, 0.1, 0.3)$ betragen. Wir schreiben $\pi(\cdot \mid s)$ für die gesamte Verteilung im Zustand $s$. Eine *deterministische* Strategie wählt mit Wahrscheinlichkeit eins eine einzige Aktion, die wir mit $\pi(s)$ bezeichnen.

<!--en:e6a57176-->
Betrachten wir eine feste Strategie. Ein Agent startet in $s_0$, zieht in jedem Schritt $a_t \sim \pi(\cdot \mid s_t)$ und wechselt nach $s_{t+1} \sim P(\cdot \mid s_t, a_t)$. Dabei erzielt er den diskontierten Ertrag (engl. discounted return) $\sum_t \gamma^t r(s_t, a_t)$ aus :numref:`sec_mdp`. Verschiedene Durchläufe ergeben verschiedene Trajektorien. Deshalb bewerten wir den Startzustand anhand des mittleren Ertrags,

<!--en:32a62b72-->
Dies ist die *Wertfunktion* (engl. value function) von $\pi$. Da wir den Startzustand frei wählen können, ordnet $V^\pi$ jedem Zustand eine Zahl zu. Sie gibt an, wie viel Belohnung ein Agent erwarten kann, der von dort an $\pi$ folgt. Auf dem See gibt es nur beim Eintritt ins Ziel eine Belohnung. Damit ist $V^\pi(s)$ die Wahrscheinlichkeit, von $s$ aus das Ziel zu erreichen, wobei jeder Erfolg mit $\gamma^t$ gewichtet wird, wenn der Agent im Schritt $t$ ins Ziel gelangt.

<!--en:1acd58c0-->
Die Frage in Feld 9 verlangt einen Vergleich der Befehle, während $V^\pi$ über sie mittelt. Für den *Aktionswert* (engl. action value) legen wir die erste Aktion fest und folgen danach der Strategie:

<!--en:c32989e7-->
Die Summe beginnt bei $t = 1$, weil die erste Belohnung durch $a_0$ festliegt. Der Agent führt die vorgegebene Aktion einen Schritt lang aus und folgt dann für alle weiteren Schritte der Strategie. Mitteln wir den Aktionswert über die erste Aktion, die die Strategie selbst wählt, erhalten wir wieder den Zustandswert,

<!--en:975432be-->
Somit ist $V^\pi(s)$ ein gewichteter Mittelwert der Zahlen $Q^\pi(s, \cdot)$ und höchstens so groß wie die größte von ihnen. Es gilt $\max_a Q^\pi(s, a) \geq V^\pi(s)$. Gleichheit gilt nur, wenn die Strategie ihre gesamte Wahrscheinlichkeit auf Aktionen legt, die das Maximum erreichen. Jeder Verbesserungsschritt in diesem Abschnitt geht von dieser Ungleichung aus.

<!--en:75970310-->
### Die Erwartungswertform

<!--en:7b600de9-->
Die Werte sind durch unendliche Summen definiert, doch die Markov-Eigenschaft verknüpft jeden Wert mit seinen Nachbarn. Teilen wir eine Trajektorie nach dem ersten Übergang. Der erste Teil enthält die Aktion $a_0$, die Belohnung $r(s_0, a_0)$ und den Folgezustand $s_1$. Der Rest ist eine Trajektorie mit Startzustand $s_1$. Nach der Markov-Eigenschaft hängt der erwartete Ertrag des Rests nur über $s_1$ von der Vergangenheit ab und beträgt $V^\pi(s_1)$. Der Wert in $s_0$ ist daher die erwartete unmittelbare Belohnung plus der diskontierte mittlere Wert des Folgezustands:

<!--en:49fd169b-->
Der äußere Erwartungswert mittelt über die Wahl von $a_0$ durch die Strategie. Der innere mittelt bei gegebener Aktion über den Folgezustand $s_1$ der Umgebung. Für endliche Zustands- und Aktionsräume sind die Erwartungswerte Summen:

<!--en:18d825df-->
Bedingt man zusätzlich auf die erste Aktion, erhält man die entsprechende Gleichung für den Aktionswert:

<!--en:2db2357b-->
Dies sind die *Bellman-Erwartungswertgleichungen* (engl. Bellman expectation equations). Bei fester Strategie ist :eqref:`eq_dynamic_programming_val` ein System aus $|\mathcal{S}|$ linearen Gleichungen für die $|\mathcal{S}|$ Unbekannten $V^\pi(s)$. Der unendliche Horizont steckt nun in der Rekursion.

<!--en:a2ff5392-->
An einem Feld des Sees lässt sich der Aufbau dieses Systems erkennen. Betrachten wir eine Strategie, die in Feld 14 neben dem Ziel den Befehl *unten* gibt. Bei einer Bewegung nach *unten* stößt der Agent an den unteren Rand und bleibt in Feld 14. Rutscht er zur Seite, gelangt er nach links in Feld 13 oder nach rechts ins Ziel, wo er eine Belohnung von eins erhält. Jeder Ausgang hat Wahrscheinlichkeit $1/3$, und das Ziel hat als Endzustand den Wert null. Damit lautet :eqref:`eq_dynamic_programming_val` für Feld 14

<!--en:b6e924d6-->
Die Unbekannte $V^\pi(14)$ steht auf beiden Seiten, weil der Agent beim Rutschen auf demselben Feld bleiben kann. Auch $V^\pi(13)$ hängt wieder von den Werten seiner Nachbarn ab, darunter Feld 14. In einem Spielbaum könnte man die Werte einzeln von den Blättern bis zur Wurzel eintragen. Auf dem rutschigen See gibt es Schleifen. Wir müssen die Gleichungen daher gemeinsam lösen, entweder als lineares Gleichungssystem oder indem wir mit einer Schätzung beginnen und wiederholt die rechte Seite anwenden. Im Folgenden nutzen wir beide Wege.

<!--en:8af570d1-->
Auf jeder rechten Seite werten wir denselben Klammerausdruck für ein Zustand-Aktion-Paar aus. Er enthält die unmittelbare Belohnung plus den diskontierten mittleren Wert des Folgezustands. Diese Vorausschau um einen Schritt heißt *Backup*, weil sie die Werte von den Folgezuständen zum aktuellen Zustand zurückträgt. Mit den Arrays von `TabularMDP` aus :numref:`sec_mdp` lassen sich die Backups für alle Zustand-Aktion-Paare in einem Matrixprodukt berechnen. Es summiert über die letzte Achse von `P[s, a, s']` und die entsprechende Achse des Wertvektors `V`. Wir ergänzen die Klasse um die Methode `backup`. Sie gibt die Werte der Klammerausdrücke als Array mit Index $(s, a)$ zurück:

<!--en:2e4fbddf-->
### Die Optimalitätsgleichung

<!--en:3d871996-->
Eine *optimale* Strategie maximiert den Wert. Grundsätzlich könnte die beste Strategie vom Startzustand abhängen, die gesamte Vorgeschichte der Episode berücksichtigen oder zufällig wählen. Für einen endlichen MDP mit Diskontierung ist all das entbehrlich, wie bereits in :numref:`sec_mdp` festgehalten:

<!--en:36afc271-->
**Proposition (Existenz einer gleichmäßig optimalen Strategie).** *In einem endlichen MDP mit Diskontfaktor $0 \leq \gamma < 1$ gibt es eine deterministische, stationäre Strategie $\pi^*$, die in allen Zuständen zugleich optimal ist. Es gilt $V^{\pi^*}(s) \geq V^\pi(s)$ für alle $s \in \mathcal{S}$ und alle Strategien $\pi$, auch für stochastische Strategien und solche, die von der Vorgeschichte abhängen* :cite:`Puterman.1994`.

<!--en:e865eeec-->
Derselbe erwartete Ertrag definiert auch den Wert einer Strategie, die von der Vorgeschichte abhängt. Wir schreiben $V^* \equiv V^{\pi^*}$ und $Q^* \equiv Q^{\pi^*}$. Welche Gleichung erfüllt $V^*$? Zwei Vergleiche liefern die Antwort. Erstens besagt die Erwartungswertgleichung von $\pi^*$, dass $V^*(s)$ gleich dem Klammerausdruck $r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s')$ der Aktion $a = \pi^*(s)$ ist. Damit ist $V^*(s)$ höchstens so groß wie der größte Klammerausdruck. Zweitens betrachten wir eine Strategie, die einmal die Aktion $a$ im Zustand $s$ wählt und danach $\pi^*$ folgt. Ihr Wert ist gleich dem Klammerausdruck von $a$. Nach der Proposition übertrifft keine Strategie $\pi^*$. Also übersteigt kein Klammerausdruck $V^*(s)$. Zusammen zeigen beide Vergleiche, dass $V^*(s)$ gleich dem größten Klammerausdruck ist:

<!--en:6dd0deb2-->
Dies ist die *Bellman-Optimalitätsgleichung* (engl. Bellman optimality equation) :cite:`BellmanDPPaper,BellmanDPBook`. In Worten ist der Wert eines Zustands bei optimalem Spiel gleich dem Wert des besten ersten Zugs, wenn der Agent vom erreichten Feld aus optimal weiterspielt. Anders als die Erwartungswertform enthält die Gleichung keine Strategie. Ihre Unbekannte ist die Tabelle $V^*$, und wegen des Maximums ist sie nichtlinear. Der erste Vergleich zeigt auch, dass die Aktion $\pi^*(s)$ das Maximum erreicht. Kennen wir $V^*$, genügt daher ein Schritt Vorausschau mit dem Modell, um eine optimale Aktion zu finden:

<!--en:d7ccc7ef-->
Die Form von :eqref:`eq_bellman_optimality` für Aktionswerte erhalten wir, indem wir $V^*(s') = \max_{a'} Q^*(s', a')$ in den Klammerausdruck einsetzen:

<!--en:ea432ab6-->
Die beiden Formen unterscheiden sich darin, was wir zum Ableiten der optimalen Aktion benötigen. Aus $Q^*$ erhalten wir die optimale Aktion $\pi^*(s) = \mathrm{argmax}_a Q^*(s, a)$ ohne Modell. Für die Ableitung aus $V^*$ mit :eqref:`eq_optimal_policy` benötigen wir hingegen $P$ und $r$. Dieser Unterschied wird wichtig, sobald das Modell unbekannt ist. Bei bekanntem Modell können wir die Vorausschau aus :eqref:`eq_optimal_policy` auch auf mehrere Schritte ausdehnen. Brettspielprogramme suchen viele Züge weit voraus. Als Modell verwenden sie die Spielregeln :cite:`Silver.Hubert.Schrittwieser.ea.2018` oder ein selbst erlerntes Modell :cite:`Schrittwieser.Antonoglou.Hubert.ea.2020`.

<!--en:f667fe8f-->
### Backup-Diagramme

<!--en:72553858-->
Die vier Gleichungen sind gleich aufgebaut. Der Wert an der Wurzel ergibt sich aus den Werten eine Ebene darunter. :numref:`fig_rl_backups` stellt die vier Fälle nebeneinander dar. Das letzte Diagramm zeigt, wie ein lernender Agent ohne Modell vorgehen muss. Da er nicht über die Folgezustände der Umgebung mitteln kann, verwendet er den einen beobachteten Übergang.

<!--en:8a29f13b-->
![Backup-Diagramme. Offene Kreise stehen für Zustände, ausgefüllte Punkte für Zustand-Aktion-Paare. Jedes Diagramm zeigt, wie sich der Wert an der Wurzel aus den Werten eine Ebene darunter ergibt. Von links nach rechts: (a) Die Wertfunktion mittelt über die Aktionen der Strategie und die Folgezustände der Umgebung. (b) Die Aktionswertfunktion legt die erste Aktion fest und mittelt danach. (c) Der Optimalitätsoperator ersetzt den Mittelwert über die Aktionen durch ein Maximum, dargestellt als Bogen. (d) Ein stichprobenbasiertes Backup kann nicht über die Umgebung mitteln. Es verwendet den einzigen beobachteten Übergang (blau) und bildet im Folgezustand weiterhin das Maximum.](../img/mdl-rl-backups.svg)

<!--en:f7a40e5e-->
## Wertiteration

<!--en:df8e6716-->
Wir laden den rutschigen See aus :numref:`sec_mdp` als `TabularMDP` mit $\gamma = 0.95$. Für die Ausgabe legen wir außerdem für jede Aktion ein Pfeilzeichen fest:

<!--en:23685785-->
### Der Algorithmus

<!--en:a4600db9-->
Die Optimalitätsgleichung definiert $V^*$ nur implizit. Zur Berechnung fassen wir die rechte Seite von :eqref:`eq_bellman_optimality` als Abbildung auf. Der *Bellman-Optimalitätsoperator* (engl. Bellman optimality operator) $\mathcal{T}$ bildet jede Tabelle $V: \mathcal{S} \to \mathbb{R}$ auf die Tabelle der jeweils größten Klammerausdrücke ab,

<!--en:29325f5a-->
Nach der Optimalitätsgleichung gilt $V^* = \mathcal{T}V^*$. Die optimale Wertfunktion ist also ein *Fixpunkt* von $\mathcal{T}$. Am einfachsten suchen wir einen Fixpunkt, indem wir die Abbildung wiederholt anwenden. Wir beginnen mit $V_0 = 0$ und berechnen in jedem Durchlauf

<!--en:86594dc2-->
also $V_{k+1} = \mathcal{T}V_k$. Dies ist die *Wertiteration* (engl. value iteration) :cite:`BellmanDPBook`. In jedem Durchlauf berechnen wir ein Backup pro Zustand-Aktion-Paar und behalten für jeden Zustand das größte Ergebnis. Im Code genügt dafür eine Zeile, weil `backup` alle Klammerausdrücke auf einmal berechnet. Die Funktion gibt alle Iterierten zurück, damit wir verfolgen können, wie sich die Werte ändern:

<!--en:829ba9f6-->
Die Iterierten sind mehr als Zwischenergebnisse auf dem Weg zu $V^*$. Jede beantwortet eine eigene Frage. Angenommen, der Agent kann nur noch $k$ Schritte lang handeln und erhält danach keine Belohnung mehr. Bleibt kein Schritt mehr, erzielt er keinen Ertrag. Das entspricht $V_0 = 0$. Bleibt ein Schritt, erreicht er bestenfalls die größte erwartete unmittelbare Belohnung, also $\mathcal{T}V_0$. Durch Induktion folgt, dass $V_k = \mathcal{T}V_{k-1}$ der höchste erwartete Ertrag über $k$ Schritte ist. Das ist dynamische Programmierung mit *endlichem Horizont* (engl. finite horizon). Ein Problem, das nach $H$ Schritten endet, lösen wir vom Ende her. Bezeichnet $V_t$ den besten Ertrag ab Schritt $t$, dann gilt $V_H = 0$, sofern keine terminale Belohnung gezahlt wird. Andernfalls setzen wir den Endwert gleich dieser Belohnung. Für die übrigen Schritte gilt $V_t = \mathcal{T}V_{t+1}$. Gegenüber dem stationären Fall ändert sich ein Punkt. Der Wert und damit die beste Aktion hängen sowohl vom Zustand als auch von der Zeit ab. Ein sicherer Umweg, der langfristig optimal ist, lohnt sich womöglich nicht mehr, wenn nur wenige Schritte bleiben. Auch :numref:`sec_optimal_control` löst die dort behandelten Steuerungsprobleme auf diese Weise rückwärts.

<!--en:abc63e25-->
### Griffiges Eis

<!--en:e5fe5e44-->
Der einfachste Fall ist der See ohne Rutschen. Hier führt jeder Befehl genau in die angegebene Richtung. Der Agent erhält die Zielbelohnung beim Zug ins Ziel. Ein Feld, das nur einen Zug vom Ziel entfernt liegt, hat daher den Wert 1. Ein zwei Züge entferntes Feld hat den Wert $\gamma$, weil der beste Zug einen Schritt später ein Feld mit Wert 1 erreicht. Braucht der kürzeste Weg um die Löcher herum $j$ Züge, hat das Feld allgemein den Wert $V^*(s) = \gamma^{j-1}$. Ausgehend von $V_0 = 0$ findet die Wertiteration diese Werte Ring für Ring. Nach Durchlauf $k$ haben alle Felder, die höchstens $k$ Züge vom Ziel entfernt sind, bereits ihren endgültigen, von null verschiedenen Wert. Die übrigen Werte sind noch null. :numref:`fig_rl_value_wavefront` zeigt die ersten sechs Durchläufe.

<!--en:33a1636c-->
![Wertiteration auf griffigem Eis in FrozenLake. Hier lassen sich die Werte in geschlossener Form angeben. Durchlauf $k$ erreicht die Felder, die höchstens $k$ Züge vom Ziel entfernt sind. Neu erreichte Felder sind orange umrandet. Ein Feld in $j$ Zügen Entfernung hat den Wert $\gamma^{j-1}$. Das Startfeld liegt sechs Züge vom Ziel entfernt. Sein Schätzwert bleibt fünf Durchläufe lang null und beträgt ab dem sechsten $\gamma^5 = 0.774$.](../img/mdl-rl-value-wavefront.svg)

<!--en:ba1678be-->
Die folgende Codezelle führt auf griffigem Eis zehn Durchläufe aus. Nach jedem Durchlauf zählt sie die Felder mit einem von null verschiedenen Wert. Aus :eqref:`eq_optimal_policy` bestimmt sie die gierige Strategie (engl. greedy policy) `shortest`:

<!--en:93d99de8-->
Die Anzahl steigt auf 1, 3, 5, 7, 10, 11 und bleibt dann bei 11. So viele Eisfelder gibt es außer dem Ziel. Nach sechs Durchläufen sind alle Werte endgültig, und das Startfeld hat den Wert $\gamma^5 = 0.774$. Auf griffigem Eis ist die Wertiteration ein Algorithmus für kürzeste Wege. Die Werte breiten sich vom Ziel aus pro Durchlauf um einen Zug rückwärts aus. Die gierige Strategie `shortest` folgt von jedem Feld aus einem kürzesten Weg. In den Feldern 0 und 9 ergeben zwei Züge gleich kurze Wege. Hier wählt `argmax` den ersten, *unten*.

<!--en:899a5917-->
Auf rutschigem Eis ändert sich ein entscheidender Punkt. Der Agent kann auf ein früher besuchtes Feld zurückrutschen oder auf seinem Feld bleiben, wie oben in Feld 14. Damit hängen die Werte wechselseitig voneinander ab. Wir erreichen sie nicht mehr in endlich vielen Durchläufen. Konvergiert die Iteration überhaupt, wie schnell konvergiert sie, und wann können wir aufhören?

<!--en:3888aabb-->
### Konvergenz durch Kontraktion

<!--en:d95e0fd2-->
Alle drei Antworten beruhen auf einer Eigenschaft von $\mathcal{T}$. Der Operator verringert den Abstand zwischen zwei beliebigen Tabellen. Wir messen diesen Abstand in der Supremumsnorm $\|V\|_\infty = \max_{s} |V(s)|$, also als die größte Abweichung über alle Zustände.

<!--en:f6386d48-->
**Proposition (der Bellman-Operator ist eine $\gamma$-Kontraktion).** *Für zwei beliebige Tabellen $V, V': \mathcal{S} \to \mathbb{R}$ gilt*

<!--en:f9d4939d-->
**Beweis.** Wir halten $s$ fest und bezeichnen mit $f(a)$ und $g(a)$ die Klammerausdrücke in $(\mathcal{T}V)(s)$ und $(\mathcal{T}V')(s)$. Für jedes $a$ gilt $|f(a) - g(a)| = \gamma\, |\sum_{s'} P(s' \mid s, a)(V(s') - V'(s'))| \leq \gamma \|V - V'\|_\infty$, weil die Wahrscheinlichkeiten nichtnegativ sind und sich zu eins summieren. Die Differenz zweier Maxima ist durch die größte Differenz ihrer Argumente beschränkt, also $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$. Somit gilt $|(\mathcal{T}V)(s) - (\mathcal{T}V')(s)| \leq \gamma \|V - V'\|_\infty$ für jedes $s$. $\blacksquare$

<!--en:ed92fe7f-->
Nach einem Durchlauf beträgt die größte Abweichung zwischen zwei beliebigen Tabellen also höchstens das $\gamma$-Fache ihres vorherigen Werts. Wenden wir dies auf $V_k$ und auf den Fixpunkt $V^*$ von $\mathcal{T}$ an, folgt für den Fehler der Wertiteration dasselbe. Er beträgt nach jedem Durchlauf höchstens das $\gamma$-Fache des vorherigen Fehlers. :numref:`fig_rl_contraction` zeigt das an einem MDP mit zwei Zuständen. Jede Tabelle entspricht dort einem Punkt in der Ebene.

<!--en:c2e0ea41-->
![Die Wertiteration konvergiert, weil der Bellman-Operator $\mathcal T$ in der Supremumsnorm eine Kontraktion ist. Der MDP hat zwei Zustände. In beiden kann der Agent bleiben oder wechseln. Bleibt er in $s_1$, erhält er 1, sonst 0. Mit $\gamma = 0.8$ ergibt sich $V^* = (5, 4)$. Die Quadrate sind die Kugeln der Supremumsnorm mit Radius $\gamma^k r_0$ um $V^*$. Dabei gilt $r_0 = \|V_0 - V^*\|_\infty = 7$ für den Startpunkt $V_0 = (-2, 3)$. Nach Durchlauf $k$ liegt die Iterierte im $k$-ten Quadrat.](../img/mdl-rl-contraction.svg)

<!--en:b0df4595-->
Daraus ergeben sich vier Folgerungen. Erstens ist der Fixpunkt eindeutig. Sind $V$ und $V'$ beide Fixpunkte, dann gilt $\|V - V'\|_\infty = \|\mathcal{T}V - \mathcal{T}V'\|_\infty \leq \gamma \|V - V'\|_\infty$. Für $\gamma < 1$ folgt zwingend $V = V'$. Somit ist $V^*$ die einzige Lösung der Optimalitätsgleichung.

<!--en:2f50204d-->
Zweitens konvergiert die Wertiteration von jedem Startwert aus mit geometrischer Rate. Wir schreiben $e_k = \|V_k - V^*\|_\infty$ für den Fehler nach $k$ Durchläufen. Dann gilt

<!--en:e906e8b6-->
und der Fehler fällt nach höchstens $\log(e_0/10^{-6})/\log(1/\gamma)$ Durchläufen unter $10^{-6}$. Liegt $\gamma$ nahe bei eins, beträgt der Nenner ungefähr $1-\gamma$. Die Zahl der Durchläufe wächst dann proportional zum effektiven Horizont $1/(1-\gamma)$ aus :numref:`sec_mdp`.

<!--en:e78824b7-->
Drittens liefern die Iterierten selbst eine Garantie für ihre Genauigkeit. Den Fehler $e_k$ können wir nicht beobachten, weil wir $V^*$ erst berechnen wollen. Die Änderung zwischen zwei aufeinanderfolgenden Iterierten können wir hingegen beobachten. Sie liefert die folgende Fehlerschranke (Aufgabe :exref:`ex_value-iter-certificate` leitet sie her):

<!--en:a77270fc-->
Die rechte Seite hängt nur von den letzten beiden Iterierten ab. Die Wertiteration kann daher enden, sobald diese Schranke unter die vorgegebene Toleranz fällt.

<!--en:a8709a0c-->
Die vierte Folgerung betrifft Strategien. Ersetzen wir das Maximum in $\mathcal{T}$ durch den Mittelwert unter einer festen Strategie $\pi$, erhalten wir den Operator $\mathcal{T}^\pi$. Er bildet eine Tabelle $V$ auf die rechte Seite der Erwartungswertgleichung :eqref:`eq_dynamic_programming_val` ab, wobei $V$ an die Stelle von $V^\pi$ tritt. Ersetzen wir im obigen Beweis das Maximum durch einen Mittelwert, sehen wir, dass auch $\mathcal{T}^\pi$ eine $\gamma$-Kontraktion ist. Damit ist $V^\pi$ ihr eindeutiger Fixpunkt :cite:`Puterman.1994,Bertsekas.2025`. Sei nun $\pi$ eine gierige Strategie bezüglich $V^*$ wie in :eqref:`eq_optimal_policy`. Dann gilt $\mathcal{T}^\pi V^* = \mathcal{T}V^* = V^*$. Also ist $V^*$ der Fixpunkt von $\mathcal{T}^\pi$, und dieser ist $V^\pi$. Jede gierige Strategie bezüglich $V^*$ ist somit optimal, unabhängig davon, welche der maximierenden Aktionen wir bei Gleichstand wählen. Wir können die Strategie daher mit `argmax` bestimmen.

<!--en:4185711a-->
Alle vier Folgerungen setzen $\gamma < 1$ voraus. Für $\gamma = 1$ zeigt der Beweis nur, dass $\mathcal{T}$ die Abstände nicht vergrößert. Sowohl Eindeutigkeit als auch Konvergenz können verloren gehen (Aufgabe :exref:`ex_value-iter-undiscounted-lake`). Das zeigt die Strategie aus :numref:`sec_mdp`, die immer *oben* wählt und deren Episode nie endet. Bei einer Belohnung von $0.05$ pro Schritt würden ihre undiskontierten Werte mit jedem Durchlauf um $0.05$ wachsen und nie konvergieren. Probleme, bei denen jede Strategie ihre Episoden mit Wahrscheinlichkeit eins beendet, heißen stochastische Kürzeste-Wege-Probleme. Für sie bleibt die Lösung bei $\gamma = 1$ eindeutig, wenn wir die Werte der Endzustände auf null festlegen :cite:`Bertsekas.2025`.

<!--en:d729abbf-->
### Rutschiges Eis

<!--en:71d82027-->
Nun wenden wir die Wertiteration auf den rutschigen See an. Nach 1000 Durchläufen liegt die garantierte Fehlerschranke $\gamma^{1000} e_0$ unter $10^{-22}$. Wir nehmen daher die letzte Iterierte als $V^*$ und bestimmen ihre gierige Strategie $\pi^*$ nach :eqref:`eq_optimal_policy`. Die Codezelle prüft die Gleichung für Feld 14 anhand der berechneten Werte. Sie gibt die Aktionswerte $Q^*(9, \cdot)$ zur Frage in Feld 9 aus und stellt die Werte und gierigen Befehle für beide Seen grafisch dar:

<!--en:054e5584-->
Mit $V^*(13) = 0.509$ ist die Gleichung für Feld 14 erfüllt. In $\tfrac13 + 0.95\,(0.509 + 0.724)/3 = 0.724$ steht auf beiden Seiten dieselbe Zahl. Damit können wir die Frage zu Feld 9 beantworten. Der Befehl *unten* hat den Wert $0.375$, verglichen mit $0.289$ für *rechts*, den besseren der Befehle, bei denen der Agent ins Loch geraten kann. Die Sicherheit von *unten* wiegt das seitliche Rutschen also auf. Gegenüber griffigem Eis sinken durch das Rutschen alle Werte. Der Wert des Startfeldes fällt von $0.774$ auf $0.180$. Selbst Feld 14 neben dem Ziel hat nur den Wert $0.724$ statt $1$. Auch die Befehle ändern sich. In vier Feldern zeigt der optimale Befehl auf rutschigem Eis vom Ziel weg, in mehreren anderen gegen den Rand. Die Gründe für jeden dieser Befehle finden sich in :numref:`subsec_vi_slippery_ice`.

<!--en:4ca16f33-->
Wie schnell sind die Durchläufe konvergiert? Die folgende Codezelle stellt den Fehler $e_k$ der ersten 170 Durchläufe der garantierten Schranke $\gamma^k e_0$ gegenüber. Sie zeichnet die Toleranz $10^{-6}$ als gestrichelte Linie ein und prüft, dass der Fehler die Schranke nie überschreitet:

<!--en:12791b6b-->
Auf einer logarithmischen Skala erscheint eine geometrische Abnahme als Gerade. Pro Durchlauf sinkt die Schranke mit dem Faktor $\gamma = 0.95$. Von $e_0 = V^*(14) = 0.724$ ausgehend erreicht sie $10^{-6}$ im Durchlauf 264. Der gemessene Fehler sinkt schneller, pro Durchlauf auf etwa das $0.92$-Fache. Er unterschreitet die gestrichelte Linie im Durchlauf 158. Die Schranke gilt für jeden MDP mit diesem Diskontfaktor. Weder die Anzahl der Felder noch ihre Entfernung vom Ziel geht in sie ein (Aufgabe :exref:`ex_value-iter-contraction-rate` erklärt die schnellere Abnahme).

<!--en:e030a58e-->
### Abbruchregeln

<!--en:edde3ee9-->
Eine Implementierung kann den tatsächlichen Fehler nicht beobachten, da sie $V^*$ nicht kennt. Ihre Abbruchregel muss sich auf eine beobachtbare Größe stützen. Die Codezelle vergleicht zwei solche Regeln mit dem tatsächlichen Fehler. Die erste bricht ab, wenn die Änderung $\|V_k - V_{k-1}\|_\infty$ zwischen zwei Durchläufen unter $10^{-6}$ fällt, die zweite, wenn die Fehlerschranke :eqref:`eq_vi_certificate` diese Toleranz unterschreitet:

<!--en:9d227453-->
Die Regel für die Änderung bricht im Durchlauf 128 ab. Der tatsächliche Fehler beträgt dann noch $1.1 \times 10^{-5}$, mehr als das Elffache der Toleranz. Aus der beobachteten Änderung allein ist das nicht ersichtlich. Mit der Fehlerschranke endet die Iteration im Durchlauf 164, sechs Durchläufe nachdem der tatsächliche Fehler im Durchlauf 158 erstmals unter $10^{-6}$ gefallen ist. Die Schranke ist also nur leicht konservativ.

<!--en:e0aeac20-->
## Strategiebewertung und Strategieiteration

<!--en:d91f88e6-->
Die Wertiteration verbessert die Strategie implizit bei jedem Durchlauf. Eine zweite Familie von Verfahren trennt die beiden Aufgaben. Wir messen zunächst, wie gut eine gegebene Strategie ist, und verbessern sie dann. Schon die Bewertung allein ist nützlich, denn sie lässt sich auf jede Strategie anwenden, auch wenn wir sie nicht optimiert haben.

<!--en:fd6fea82-->
### Strategiebewertung

<!--en:7e44746e-->
Ersetzen wir das Maximum in jedem Durchlauf durch den Mittelwert unter der Strategie. Wir wenden also $\mathcal{T}^\pi$ anstelle von $\mathcal{T}$ an:

<!--en:43139aca-->
Dies ist die *Strategiebewertung* (engl. policy evaluation). Sie konvergiert von jedem Startwert aus gegen $V^\pi$, weil $\mathcal{T}^\pi$ eine $\gamma$-Kontraktion ist. Im Code ersetzen wir im Durchlauf der Wertiteration das Maximum durch den Mittelwert unter `pi[s, a]`. Wir wenden das Verfahren auf die Strategie an, die alle Aktionen mit gleicher Wahrscheinlichkeit wählt:

<!--en:93bafc7f-->
Die Gleichverteilungsstrategie hat den Startwert $0.0078$, etwa vier Prozent des Optimums $0.180$. Wer seine Befehle zufällig wählt, erreicht das Ziel selten, und wenn, dann spät.

<!--en:5f82ae4d-->
Ohne das Maximum sind die Gleichungen linear, und für eine Tabelle können wir sie direkt lösen. Mit $P_\pi(s' \mid s) = \sum_a \pi(a \mid s) P(s' \mid s, a)$ bezeichnen wir den Übergangskern unter $\pi$. Er gibt die Wahrscheinlichkeit an, in einem Schritt von $s$ nach $s'$ zu gelangen, wenn wir die Aktion aus $\pi$ ziehen. Mit $r_\pi(s) = \sum_a \pi(a \mid s) r(s, a)$ bezeichnen wir die zugehörige erwartete Belohnung. Dann lautet :eqref:`eq_dynamic_programming_val` in Matrixform $V^\pi = r_\pi + \gamma P_\pi V^\pi$ und hat die Lösung

<!--en:df83b985-->
Die Matrix $I - \gamma P_\pi$ ist für $\gamma < 1$ invertierbar, weil $\gamma P_\pi$ die Supremumsnorm jedes von null verschiedenen Vektors verkleinert. In Aufgabe :exref:`ex_value-iter-linear-system` deuten wir die Einträge der inversen Matrix als diskontierte Besuchszahlen. Für eine Tabelle liefert das einmalige Lösen des linearen Gleichungssystems $V^\pi$ exakt. Die iterativen Durchläufe erreichen dasselbe durch wiederholte Kontraktion. Sie eignen sich auch für Probleme, die für eine direkte Lösung zu groß sind. Wir ergänzen `TabularMDP` um die direkte Lösung:

<!--en:f181f6ec-->
Beide Verfahren ergeben $0.0078$. Mit `einsum` berechnen wir $P_\pi$, indem wir die Zeilen des Übergangskerns über die Aktionen mitteln und dabei mit den Wahrscheinlichkeiten der Strategie gewichten. Die direkte Lösung setzt für diese Arrays $\gamma < 1$ voraus. In `from_gym` führen alle Aktionen eines Endzustands wieder in diesen Zustand zurück. Für $\gamma = 1$ hat die Matrix $I - P_\pi$ daher an dieser Stelle eine Nullzeile und ist singulär. Größen für $\gamma = 1$, etwa die Wahrscheinlichkeit, das Ziel zu erreichen, berechnen wir deshalb iterativ.

<!--en:5f4d239c-->
### Strategieiteration und verallgemeinerte Strategieiteration

<!--en:fa649ff4-->
Die Bewertung und der gierige Schritt legen zusammen einen anderen Algorithmus nahe, der auf :citet:`Howard.1960` zurückgeht. Wir bewerten die aktuelle Strategie exakt, handeln bezüglich ihrer Werte gierig und wiederholen beides. Der gierige Schritt beruht auf der Ungleichung $\max_a Q^\pi(s, a) \geq V^\pi(s)$ vom Anfang des Abschnitts, die wir auf alle Zustände zugleich anwenden.

<!--en:3de83b04-->
**Proposition (Strategieverbesserung).** *Sei $\pi$ eine beliebige Strategie und $\pi'$ gierig bezüglich $V^\pi$, das heißt $\pi'(s) = \mathrm{argmax}_a \big[ r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s') \big]$. Dann gilt $V^{\pi'}(s) \geq V^\pi(s)$ in jedem Zustand. Gilt überall Gleichheit, sind beide Strategien optimal.*

<!--en:9f69dd47-->
**Beweis.** Nach Definition der gierigen Wahl ist ein Schritt mit $\pi'$, gefolgt von den Werten von $\pi$, mindestens so gut wie ein Schritt mit $\pi$. Es gilt also $\mathcal{T}^{\pi'} V^\pi = \mathcal{T} V^\pi \geq \mathcal{T}^\pi V^\pi = V^\pi$. Der Operator $\mathcal{T}^{\pi'}$ ist monoton, weil seine Koeffizienten Wahrscheinlichkeiten sind. Aus $V \leq U$ in jedem Zustand folgt $\mathcal{T}^{\pi'} V \leq \mathcal{T}^{\pi'} U$. Wiederholte Anwendung ergibt $V^\pi \leq \mathcal{T}^{\pi'} V^\pi \leq (\mathcal{T}^{\pi'})^2 V^\pi \leq \cdots$. Die Folge konvergiert gegen den eindeutigen Fixpunkt von $\mathcal{T}^{\pi'}$, also gegen $V^{\pi'}$. Gilt $V^{\pi'} = V^\pi$, so lautet die erste Zeile $\mathcal{T}V^\pi = V^\pi$. Damit löst $V^\pi$ die Optimalitätsgleichung. Deren einzige Lösung ist $V^*$, also ist $\pi$ optimal. $\blacksquare$

<!--en:a225ecce-->
Jede Runde der *Strategieiteration* (engl. policy iteration) erhöht somit den Wert in mindestens einem Zustand oder lässt alle Werte unverändert. Im zweiten Fall ist die Strategie bereits optimal. Da die Werte nie abnehmen, kann sich vor Erreichen des Optimums keine Strategie wiederholen. Ein endlicher MDP hat nur endlich viele deterministische Strategien. Die Strategieiteration endet daher nach endlich vielen Runden :cite:`Puterman.1994`. Als Abbruchkriterium dient der Gleichheitsfall der Proposition. Er tritt ein, wenn der gierige Schritt in keinem Zustand einen Gewinn bringt. Prüfen wir stattdessen, ob sich die gierige Strategie noch ändert, kann der Test bei Gleichstand versagen. Auf dem rutschigen See haben *links* und *rechts* in Feld 6 denselben Aktionswert. Rundung kann dazu führen, dass `argmax` von einer Runde zur nächsten zwischen beiden wechselt. Die folgende Schleife beginnt mit der Gleichverteilungsstrategie und bewertet jede Strategie mit `policy_value`. Sie endet, sobald keine Aktion den aktuellen Wert um mehr als den Rundungsfehler übertrifft:

<!--en:8f359bfc-->
Die Zusicherung im Code prüft, dass Strategieiteration und Wertiteration für $V^*$ auf zwölf Nachkommastellen übereinstimmen. Die nächste Codezelle zählt die Rechenschritte beider Verfahren. Ein Backup wertet den Klammerausdruck für ein Zustand-Aktion-Paar aus. Ein Durchlauf der Wertiteration kostet also $|\mathcal{S}||\mathcal{A}| = 64$ Backups:

<!--en:67311b3f-->
Die Strategieiteration musste zwei Gleichungssysteme exakt lösen. Die gierige Strategie zu den Werten der Gleichverteilungsstrategie ist bereits $\pi^*$. Die zweite Runde bestätigt nur, dass der gierige Schritt keinen Gewinn mehr bringt. Die Wertiteration benötigte 10 496 Backups, um die Genauigkeit ihrer Werte nachzuweisen. Ihre gierige Strategie war jedoch schon nach 14 Durchläufen oder 896 Backups gleich $\pi^*$. Für die Strategie zählt nur, welcher Aktionswert der größte ist. Sie kann deshalb schon lange vor den Werten feststehen. Jede Runde der Strategieiteration löst ein lineares Gleichungssystem mit 16 Unbekannten. Hier kostet das wenig. Bei vielen Zuständen wachsen die Kosten der direkten Lösung jedoch kubisch, sodass iterative Durchläufe vorzuziehen sind (Aufgabe :exref:`ex_value-iter-backup-budgets` vergleicht die Kosten).

<!--en:ea16b9c7-->
Bewertung und Verbesserung müssen nicht in abgeschlossenen Phasen abwechseln. *Verallgemeinerte Strategieiteration* (engl. generalized policy iteration) bezeichnet jedes Verfahren, das zwei Arten von Schritten verschränkt. Die einen bringen die Werte näher an $V^\pi$, die anderen die Strategie näher an die gierige Strategie :cite:`Sutton.Barto.2018`. Die Strategieiteration schließt jede Bewertung vor der Verbesserung ab. Die Wertiteration verbessert die Strategie bereits nach einem einzigen Bewertungsdurchlauf. Verfahren, die aus Erfahrung lernen, verwenden dieselben beiden Arten von Schritten. Ihre Schritte sind kleiner und stärker verrauscht, weil sie diese aus einzelnen beobachteten Übergängen berechnen, ohne das Modell zu verwenden. :numref:`fig_rl_gpi` zeigt vier solcher Verfahren.

<!--en:e2226811-->
![Verallgemeinerte Strategieiteration. Auf der Linie $V = V^\pi$ liegen die Wertschätzungen, die für ihre Strategie exakt sind. Auf der Linie $\pi = \mathrm{greedy}(V)$ liegen die Strategien, die bezüglich ihrer Schätzung gierig sind. Beide Bedingungen gelten nur bei $(V^*, \pi^*)$. Vom selben Startpunkt aus bewertet die Strategieiteration vor jeder Verbesserung bis zur ersten Linie. Die Wertiteration nähert sich ihr um einen Durchlauf. Stichprobenbasierte Backups machen kurze, verrauschte Schritte. Verfahren, die Strategie und Wertschätzung gemeinsam aktualisieren, ändern beide Koordinaten zugleich ein wenig. Die Achsen stehen für die Räume der Wertfunktionen und der Strategien. Die Darstellung ist schematisch.](../img/mdl-rl-gpi.svg)

<!--en:a819bbae-->
## Die optimale Strategie auf rutschigem Eis

<!--en:770c4729-->
Den Werten zufolge ist $\pi^*$ die beste Strategie. Wie viel besser ist sie als die naheliegende Alternative, der kürzeste Weg auf griffigem Eis? Und warum zeigt sie in vier Feldern vom Ziel weg? Wir beantworten beide Fragen exakt anhand des Modells und prüfen die Antworten durch Simulation.

<!--en:2662cbd9-->
### Monte-Carlo-Bewertung

<!--en:ea5e5cd0-->
Um eine Strategie ohne Modell zu prüfen, führen wir sie im Simulator aus und mitteln die Erträge. Dazu stellen wir die Strategie als Funktion `policy(obs, rng)` dar, die einer Beobachtung eine Aktion zuordnet. Ist die Strategie stochastisch, verwendet die Funktion für die Ziehung den Zufallszahlengenerator `rng`. Die Funktion `evaluate` gibt den mittleren diskontierten Ertrag über mehrere Episoden zurück:

<!--en:0f52be98-->
Die Variable `discount` enthält $\gamma^t$ für den aktuellen Schritt. Eine Episode endet, sobald `terminated` oder `truncated` gilt. Mit der Voreinstellung $\gamma = 1$ und der einzigen Belohnung beim Eintritt ins Ziel entspricht der mittlere Ertrag dem Anteil der Episoden, die innerhalb der vom Simulator erlaubten 100 Schritte das Ziel erreichen.

<!--en:501fa7cd-->
### Erfolgsraten

<!--en:c182f26c-->
Dieselbe Wahrscheinlichkeit lässt sich mit dem Modell berechnen. Für $\gamma = 1$ summieren $k$ Durchläufe der Strategiebewertung ab $V_0 = 0$ die erwarteten Belohnungen der ersten $k$ Schritte. Die einzige Belohnung gibt es beim Eintritt ins Ziel. Damit gibt die $k$-te Iterierte bei $s_0$ die Wahrscheinlichkeit an, innerhalb von $k$ Schritten das Ziel zu erreichen. Mit $k = 100$ erhalten wir dieselbe Grenze wie im Simulator. Wir erstellen das Modell mit $\gamma = 1$ unter dem Namen `lake1`. Wir berechnen die Wahrscheinlichkeit für $\pi^*$, für `shortest` und für die Gleichverteilungsstrategie. Zum Vergleich geben wir jeweils eine Monte-Carlo-Schätzung aus 2000 Episoden an:

<!--en:ccede963-->
Die optimale Strategie erreicht das Ziel innerhalb von 100 Schritten mit Wahrscheinlichkeit $0.730$. Die Strategie des kürzesten Weges schafft das mit Wahrscheinlichkeit $0.045$, also etwa ein Sechzehntel so oft. Wer zufällig läuft, hat eine Erfolgswahrscheinlichkeit von $0.014$. Die Abweichungen der Monte-Carlo-Schätzungen von ihren exakten Werten lassen sich durch die zufälligen Schwankungen der Stichprobe erklären. Bei 2000 Episoden beträgt der Standardfehler rund einen Prozentpunkt für $0.73$ und einen Viertelprozentpunkt für $0.014$. Die Zeitgrenze kostet $\pi^*$ etwas Erfolg. Ihre vorsichtigen Befehle führen zu langen Episoden. Einige davon würden das Ziel nach mehr als 100 Schritten erreichen, werden aber vorher abgebrochen.

<!--en:95e449f1-->
### Befehle, die vom Ziel wegführen

<!--en:4a7bea43-->
Warum schneidet der kürzeste Weg so schlecht ab? Für jedes der elf Eisfelder, auf denen der Agent handelt, stellen wir zwei Fragen. Wohin zeigt der Befehl von $\pi^*$? Und mit welcher Wahrscheinlichkeit führt der Befehl der jeweiligen Strategie den Agenten im nächsten Schritt in ein Loch? Die Richtung lesen wir an den Werten für griffiges Eis ab. Ein Befehl zeigt zum Ziel, wenn das Folgefeld auf griffigem Eis einen höheren Wert `V_calm` hat. Ist das Folgefeld das Feld selbst, zeigt der Befehl gegen den Rand. Sonst zeigt er vom Ziel weg. Ein Zug in ein Loch oder ins Ziel zählt als Bewegung vom Ziel weg, weil beide auf griffigem Eis den Wert null haben. Das Array `risk[s, a]` enthält die Wahrscheinlichkeit $\sum_{s' \textrm{ a hole}} P(s' \mid s, a)$, im nächsten Schritt in ein Loch zu fallen:

<!--en:af0e6889-->
Nur in zwei Feldern, 9 und 13, zeigt die optimale Strategie zum Ziel. In fünf Feldern, 0, 1, 3, 4 und 14, gibt sie einen Befehl gegen den Rand. In der beabsichtigten Richtung bleibt der Agent dadurch stehen. Er bewegt sich nur, wenn er in eine der beiden anderen Richtungen rutscht. In Feld 4 etwa rutscht er bei *links* am Rand entlang nach oben oder unten und nie nach rechts in Loch 5. In Feld 14 erreicht er sowohl mit *unten* als auch mit *rechts* in einem von drei Fällen das Ziel und bleibt in einem von drei Fällen in Feld 14. Im dritten Fall gelangt er mit *unten* in Feld 13 und mit *rechts* in Feld 10. Feld 10 ist weniger wert (0.40 gegenüber 0.51 im Gitter oben).

<!--en:bfcc6397-->
In vier Feldern, 2, 6, 8 und 10, zeigt die optimale Strategie vom Ziel weg. In den Feldern 8 und 10 können Befehle in Richtung Ziel in ein Loch führen, in Loch 12 unter Feld 8 oder in Loch 11 neben Feld 10. Bei *oben* in Feld 8 und *links* in Feld 10 besteht diese Gefahr nicht. Mit *links* in Feld 10 rutscht der Agent dennoch in einem von drei Fällen nach unten in Feld 14. In Feld 6 liegt auf jeder Seite ein Loch. Hier birgt jeder Befehl ein Risiko. Bei *links* und *rechts* beträgt es ein Drittel, bei *oben* und *unten* zwei Drittel. Die Strategie $\pi^*$ wählt einen der beiden gleichwertigen Befehle mit dem kleinsten Risiko. Dabei wählt `argmax` den Befehl *links*, während *rechts* zum Ziel zeigen würde. Die Strategie befiehlt also den Zug in Loch 5, der in einem von drei Fällen tatsächlich eintritt. Denn beim seitlichen Rutschen sind beide Landefelder sicher, während bei *oben* und *unten* beide Rutschrichtungen in Löcher führen. In Feld 2 kann kein Befehl in ein Loch führen. Hier gewinnt *links*, weil der Agent dabei nicht ins Eckfeld 3 rutschen kann. Es hat den niedrigsten Wert in der oberen Reihe, und jeder Weg von dort führt wieder durch Feld 2.

<!--en:3f49937f-->
In allen elf Feldern wählt die optimale Strategie einen Befehl mit dem kleinstmöglichen Risiko. Außer in Feld 6 ist dieses Risiko überall null. Die Strategie des kürzesten Weges geht in den Feldern 1, 3, 4, 8 und 10 ein Risiko von einem Drittel ein, in Feld 6 von zwei Dritteln. Wo die Episoden enden, lässt sich aus den Risiken eines einzelnen Schritts noch nicht ablesen. Das hängt auch davon ab, wie oft der Agent jedes Feld besucht. Für eine deterministische Strategie $\pi$ bezeichne $P_\pi$ den Übergangskern unter $\pi$, wobei wir die Zeilen der Endzustände auf null setzen, damit der Prozess dort endet. Bei beiden betrachteten Strategien endet jede Episode mit Wahrscheinlichkeit eins. Daher ist $I - P_\pi$ invertierbar. Startet der Agent in $s_0 = 0$, stehen die erwarteten Besuchszahlen der Zustände in derjenigen Zeile von $(I - P_\pi)^{-1}$, die zu $s_0$ gehört. Die Funktion `falls` berechnet sie, indem sie das transponierte Gleichungssystem löst. Multiplizieren wir für jedes Feld die erwartete Besuchszahl mit dem Risiko seines Befehls, erhalten wir die Wahrscheinlichkeit, dass die Episode mit einem Sturz von diesem Feld in ein Loch endet:

<!--en:daf624df-->
Ohne Zeitgrenze fällt der Agent bei optimaler Strategie mit Wahrscheinlichkeit $0.22$ in ein Loch, und zwar immer von Feld 6 aus. In den übrigen $78$ Prozent der Episoden erreicht er das Ziel. Die Grenze von 100 Schritten senkt diesen Anteil auf die oben berechneten $0.730$. Bei der Strategie des kürzesten Weges stürzt der Agent in $96$ Prozent der Episoden. Etwa ein Drittel der Stürze geht von Feld 4 auf dem kürzesten Weg aus, ein weiteres Drittel von Feld 1, das gar nicht auf diesem Weg liegt. Beim ersten Befehl, *unten* in Feld 0, rutscht der Agent in einem von drei Fällen nach rechts in Feld 1. Dort gibt `shortest` den Befehl *rechts*, bei dem er in einem von drei Fällen nach unten in Loch 5 rutscht. Die übrigen Stürze gehen von den Feldern 8, 6, 3 und 10 aus. Die Strategie des kürzesten Weges scheitert auf ihrem eigenen Weg seltener als auf den Feldern, auf die der Agent abrutscht. Die optimale Strategie meidet die Löcher in jedem Feld, in dem das mit einem Befehl möglich ist.

<!--en:4075de2a-->
## Belohnungsgestaltung und optimale Strategie

<!--en:40c0ad6b-->
In :numref:`sec_mdp` haben wir vorgeschlagen, einem lernenden Agenten auf dem See mit seinen seltenen Belohnungen zu helfen. Der Agent sollte einen Bonus von $0.3$ für jeden Zug erhalten, der die Manhattan-Distanz $\ell(s)$ zum Ziel verkürzt. Schon die Rechnung für griffiges Eis zeigte, dass es sich mit diesem Bonus lohnt, im Kreis zu laufen. Mit Wertiteration und exakter Strategiebewertung können wir nun berechnen, wie sich der Bonus auf die beste Strategie auf rutschigem Eis auswirkt. Außerdem können wir beweisen, dass die dort vorgeschlagene Form mit einem Potential die beste Strategie unverändert lässt.

<!--en:a93e1c84-->
### Ein Annäherungsbonus

<!--en:06be0f27-->
Mitteln wir über den Folgezustand, erhöht der Bonus die erwartete Belohnung um $0.3$ mal die Wahrscheinlichkeit, dem Ziel näher zu kommen:

<!--en:114450e4-->
Die Codezelle berechnet diese Belohnung für die Bonusbeträge $0.3$, $0.1$ und $0$. Für jeden so veränderten MDP findet sie durch Wertiteration die optimale Strategie und bewertet diese mit `policy_value` unter der ursprünglichen Belohnung. Die Zeile mit Bonus $0$ entspricht dem ursprünglichen Problem:

<!--en:f8689aee-->
Bei einem Bonus von $0.3$ wählt die für die veränderte Belohnung optimale Strategie in Feld 14 den Befehl *links*. Er führt nach Feld 13, 10 oder 14, aber nie ins Ziel. Mit der veränderten Belohnung beträgt ihr Startwert $2.15$ und liegt damit weit über dem ursprünglichen Optimum von $0.180$. Mit der ursprünglichen Belohnung ist ihr Wert null, weil sie das Ziel nie erreicht. Die veränderten Werte zeigen, warum. Feld 13 ist nun mehr wert als Feld 14, nämlich $1.71$ gegenüber $1.52$. Ein Zug von 13 in Richtung 14 bringt den Bonus ein, und von 14 aus kann der Agent immer wieder nach 13 zurückkehren, um ihn erneut zu erhalten. Damit haben wir die Schleife aus :numref:`sec_mdp` mit Werten versehen. Beim Pendeln zwischen Feld 13 und Feld 14 gibt es in jeder Runde den Bonus. Das ist mehr wert als Belohnung und Bonus für den einmaligen Eintritt ins Ziel. Bei einem Bonus von $0.1$ ist die Schleife weniger wert als das Ziel, und die für die veränderte Belohnung optimale Strategie erreicht das Ziel wieder. Ihr tatsächlicher Wert beträgt aber $0.115$ statt $0.180$. Es handelt sich also um eine andere, schlechtere Strategie. Ein scheinbar sinnvoller Bonus verändert die optimale Strategie vollständig, und selbst ein kleinerer Bonus verändert sie noch.

<!--en:dc46e414-->
### Potentialbasierte Belohnungsgestaltung

<!--en:ef661e97-->
Der in :numref:`sec_mdp` als Alternative vorgeschlagene Bonus ist eine Potentialdifferenz,

<!--en:51bb53d5-->
wobei $\Phi$ eine beschränkte Funktion auf den Zuständen ist. Entlang einer Schleife heben sich die Terme bis auf die Diskontierung auf. Die folgende Proposition zeigt, dass sich der Wert jeder Strategie um denselben Betrag ändert.

<!--en:9c803deb-->
**Proposition (potentialbasierte Belohnungsgestaltung).** *Es gelte $0 \leq \gamma < 1$. Die Endzustände seien wie in `TabularMDP` absorbierend, und $\tilde V^\pi$ sowie $\tilde Q^\pi$ seien die Werte unter der veränderten Belohnung. Dann gilt für jede Strategie $\pi$, jeden Zustand $s$ und jede Aktion $a$*

<!--en:ae21837e-->
*Insbesondere gelten $\tilde V^* = V^* - \Phi$ und $\tilde Q^*(s, a) = Q^*(s, a) - \Phi(s)$. Der veränderte und der ursprüngliche MDP haben dieselben optimalen Strategien* :cite:`Ng.Harada.Russell.1999`.

<!--en:7e381f09-->
**Beweis.** Entlang jeder Trajektorie bilden die Zusatzterme der ersten $T$ Schritte eine Teleskopsumme:

<!--en:8144ab1b-->
Für $T \to \infty$ verschwindet der erste Term, weil $\Phi$ beschränkt ist und $\gamma < 1$ gilt. Der veränderte Ertrag jeder Trajektorie ist daher ihr ursprünglicher Ertrag abzüglich $\Phi(s_0)$. Bilden wir unter $\pi$ den Erwartungswert, erhalten wir für den Start in $s$ die erste und für den Start mit $(s, a)$ die zweite Identität. In jedem Zustand ist die Verschiebung $\Phi(s)$ für alle Strategien und Aktionen gleich. Daher bleiben die Reihenfolge der Strategien und die maximierenden Aktionen von $Q^*(s, \cdot)$ erhalten. $\blacksquare$

<!--en:7ac37a39-->
Ein Potential verändert also jeden Wert, doch für jede Strategie fällt die Änderung gleich aus. Für die Optimierung zählt nur der Vergleich der Strategien, und dieser bleibt unverändert. Aufgabe :exref:`ex_mdp-reward-design-its-failure`c prüft $\tilde V^* = V^* - \Phi$ am Beispiel des Sees. :citet:`Ng.Harada.Russell.1999` bewiesen auch eine Umkehrung. Ist ein Zusatzterm $F(s, a, s')$ keine Potentialdifferenz, gibt es auf denselben Zuständen und Aktionen einen Übergangskern und eine Belohnung mit folgender Eigenschaft. Keine optimale Strategie des veränderten MDP ist für den ursprünglichen MDP optimal. Der Annäherungsbonus ist ein solcher Term und der rutschige See ein solcher MDP.

<!--en:1bda9a3a-->
Wozu dient ein Potential, wenn es die optimale Strategie nicht ändern kann? Für die exakte Planung brauchen wir keines, denn die Wertiteration findet mit und ohne Potential dieselbe Strategie. Das Potential soll einem lernenden Agenten helfen, der nur einzelne beobachtete Übergänge kennt. Ein zum Ziel hin ansteigendes Potential wie $\Phi = -0.3\,\ell$ verteilt die Zielbelohnung über den Weg. So erhält der Agent bei jedem Zug ein Signal und nicht erst beim letzten.

<!--en:9d3d887d-->
Der Beweis nutzt eine Eigenschaft von `TabularMDP`. Die Endzustände sind absorbierend. Deshalb laufen Trajektorien unbegrenzt weiter, und der Term $\gamma^T \Phi(s_T)$ verschwindet im Grenzwert. Ein Simulator hält dagegen beim ersten Endzustand an. Geschieht das nach $T$ Schritten, ergibt sich der veränderte Ertrag der Episode aus dem ursprünglichen Ertrag plus $\gamma^T \Phi(s_T) - \Phi(s_0)$. Der Term $\gamma^T \Phi(s_T)$ hängt davon ab, in welchem Endzustand die Episode endet und wann sie ihn erreicht. Er kann daher beeinflussen, welche Strategie die beste ist, sofern $\Phi$ nicht in jedem Endzustand null ist. Dies ist die in :numref:`sec_mdp` in Worten formulierte Bedingung. Das Potential $\Phi = -0.3\,\ell$ ist am Ziel null, an den Löchern aber negativ. Im Simulator würde es den Agenten daher für einen Sturz bestrafen, einen frühen Sturz stärker als einen späten. Die ursprüngliche Aufgabe enthält keine solche Strafe.

<!--en:9ca2e4b1-->
Alle Berechnungen in diesem Abschnitt haben auf die Tabelle $P$ zugegriffen. Jedes Backup mittelte mit den exakten Wahrscheinlichkeiten über die Folgezustände. Wir können nun für einen bekannten endlichen MDP jeden Zustand und jeden Befehl bewerten, eine optimale Strategie mit einem Nachweis ihrer Optimalität finden und prüfen, wie eine andere Belohnung die Strategie verändert. Ein lernender Agent bekommt die Tabelle nur selten. Er sieht jeweils nur den Übergang, den seine eigene Aktion bewirkt hat. In :numref:`sec_qlearning` untersuchen wir, ob sich Werte und eine optimale Strategie allein aus solchen Beobachtungen lernen lassen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:a4f5ad35-->
Die dynamische Programmierung berechnet die beste Strategie eines bekannten MDP aus den Werten seiner Zustände. Der Wert $V^\pi(s)$ ist der erwartete diskontierte Ertrag ab $s$ unter $\pi$. Die Bellman-Gleichungen verknüpfen jeden Wert mit denen seiner Folgezustände. Bei einer festen Strategie mitteln sie über die Aktionen, für die optimale Strategie bilden sie das Maximum. Für $\gamma < 1$ ist der Bellman-Operator $\mathcal{T}$ eine $\gamma$-Kontraktion. Die Wertiteration konvergiert daher von jedem Startwert aus geometrisch gegen den eindeutigen Fixpunkt $V^*$. Die Änderung zwischen zwei Durchläufen liefert eine Schranke für den Fehler. Die Strategieiteration wechselt zwischen exakter Bewertung durch Lösen eines linearen Gleichungssystems und gieriger Verbesserung. Auf dem rutschigen See erreicht die optimale Strategie das Ziel etwa sechzehnmal so oft wie die Strategie des kürzesten Weges. Sie wählt in jedem Feld den Befehl mit dem geringsten Risiko, in ein Loch zu fallen, auch wenn er vom Ziel wegzeigt. Ein Bonus für die Annäherung ans Ziel kann die optimale Strategie verändern. Ein Bonus in Form einer Potentialdifferenz lässt sie unverändert.

<!--en:2294243a-->
## Übungen

<!--en:e3d5ef90-->
1. [code] **Strategiebewertung als lineares Gleichungssystem.** Nach :eqref:`eq_policy_value_solve` gilt $V^\pi = M_\pi r_\pi$ mit $M_\pi = (I - \gamma P_\pi)^{-1}$.
    1. Zeigen Sie, dass $I - \gamma P_\pi$ für $0 \leq \gamma < 1$ invertierbar ist und dass $M_\pi = \sum_{t \geq 0} \gamma^t P_\pi^t$ gilt.
    1. Interpretieren Sie den Eintrag $M_\pi(s, s')$ anhand der Anzahl der Besuche in $s'$. Der Agent startet in $s$ und folgt $\pi$ im Prozess mit Abbruch aus Aufgabe :exref:`ex_mdp-discount-as-termination`b in :numref:`sec_mdp`. Folgern Sie, dass jede Zeile von $M_\pi$ die Summe $1/(1-\gamma)$ hat. Damit ist $(1 - \gamma) M_\pi(s, \cdot)$ eine Verteilung über die Zustände, die *diskontierte Verteilung der Zustandsbesuche* unter $\pi$ ab $s$. Schreiben Sie $V^\pi(s)$ als Produkt aus $1/(1-\gamma)$ und dem Erwartungswert von $r_\pi$ unter dieser Verteilung.
    1. Berechnen Sie $M_\pi$ für die Strategie `uniform` auf dem rutschigen See. Prüfen Sie, dass $M_\pi r_\pi$ den Startwert $0.0078$ ergibt, den `policy_evaluation` ausgibt. Episoden dauern mit dieser Strategie im Mittel weniger als acht Schritte, doch die Summe der Zeile für den Startzustand beträgt $20$. Wo liegt die restliche Wahrscheinlichkeitsmasse?
   :exlabel:`ex_value-iter-linear-system`

<!--en:0ef2dd89-->
1. **Fehlerschranke als Abbruchkriterium.** Die Schranke :eqref:`eq_vi_certificate` begrenzt den unbeobachtbaren Fehler $\|V_k - V^*\|_\infty$ anhand der beobachtbaren Änderung $\|V_k - V_{k-1}\|_\infty$.
    1. Leiten Sie die Schranke aus der Proposition zur Kontraktion in :numref:`subsec_vi_contraction` her, indem Sie das Restglied $\sum_{j \geq k} \|V_{j+1} - V_j\|_\infty$ einer Teleskopsumme abschätzen. Wie klein muss die Änderung zwischen zwei Durchläufen sein, damit der Fehler sicher unter einer gegebenen Toleranz liegt?
    1. Leiten Sie eine A-priori-Schranke her, die die Anzahl der Durchläufe vor Beginn der Wertiteration festlegt und nur $\gamma$, die Toleranz und $\|V_1 - V_0\|_\infty$ verwendet. Werten Sie die Schranke für den rutschigen See mit $V_0 = 0$ und einer Toleranz von $10^{-6}$ aus. Vergleichen Sie das Ergebnis mit den $164$ Durchläufen, die das beobachtbare Abbruchkriterium bei der Wertiteration in :numref:`subsec_vi_value_iteration` auf diesem See benötigte.
   :exlabel:`ex_value-iter-certificate`

<!--en:ee0d0cd3-->
1. **Der See ohne Diskontierung.** Die Proposition zur Kontraktion in :numref:`subsec_vi_contraction` setzt $\gamma < 1$ voraus. Dennoch konvergiert die Wertiteration auf FrozenLake auch für $\gamma = 1$. Die Löcher und das Ziel sind in FrozenLake absorbierende Zustände mit Belohnung null. Die einzige von null verschiedene Belohnung, $1$, erhält der Agent beim Eintritt ins Ziel.
    1. Wiederholen Sie den Beweis der Proposition für $\gamma = 1$. Was zeigt er noch? Zeigen Sie dann, dass der Operator $\mathcal{T}$ aus :eqref:`eq_bellman_operator` unendlich viele Fixpunkte hat, sobald er bei $\gamma = 1$ einen Fixpunkt besitzt.
    1. Zeigen Sie, dass die Wertiteration mit Startwert $V_0 = 0$ für $\gamma = 1$ eine monoton nichtfallende Folge $V_0 \leq V_1 \leq V_2 \leq \cdots$ erzeugt, die nach oben durch eins beschränkt ist und daher konvergiert. Was misst $V_k(s)$?
    1. Konstruieren Sie einen MDP mit zwei Zuständen und einer einzigen Aktion, bei dem $V_k$ für $\gamma = 1$ unbegrenzt wächst. Welche in Teil b verwendete Eigenschaft von FrozenLake fehlt ihm?
   :exlabel:`ex_value-iter-undiscounted-lake`

<!--en:926237fa-->
1. [code] **Durchläufe mit direktem Überschreiben.** ● Die Funktion `value_iteration` berechnet die gesamte Tabelle $V_{k+1}$ aus $V_k$. Ein *In-place*-Durchlauf besucht die Zustände dagegen in einer festen Reihenfolge und überschreibt $V(s)$ sofort. Spätere Zustände verwenden damit schon im selben Durchlauf die neuen Werte.
    1. Zeigen Sie, dass ein In-place-Durchlauf für jede Reihenfolge der Zustände eine $\gamma$-Kontraktion in der Supremumsnorm ist und denselben Fixpunkt hat wie $\mathcal{T}$ aus :eqref:`eq_bellman_operator`. Folgern Sie, dass die Fehlerschranke :eqref:`eq_vi_certificate` für seine Iterierten gültig bleibt.
    1. Implementieren Sie den In-place-Durchlauf und zählen Sie die Durchläufe, bis die Fehlerschranke $10^{-6}$ erreicht. Besuchen Sie die Zustände einmal in aufsteigender und einmal in absteigender Reihenfolge. Verwenden Sie die $4 \times 4$- und $8 \times 8$-Karten jeweils mit rutschigem und mit griffigem Eis. Vergleichen Sie die Anzahlen mit denen von `value_iteration`.
    1. Warum spielt die Reihenfolge auf griffigem Eis eine viel größere Rolle als auf rutschigem? Erklären Sie die Anzahlen für griffiges Eis anhand von :numref:`fig_rl_value_wavefront`.

    *Nach Sutton und Barto,
    [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf),
    zweite Auflage, Abschnitte 4.1 und 4.5.*

<!--en:f9baee31-->
1. [code] **Gemessene Kontraktionsrate.** ● In :numref:`subsec_vi_value_iteration` sinkt der Fehler der Wertiteration pro Durchlauf auf etwa das $0.92$-Fache, also schneller als mit dem garantierten Faktor $\gamma = 0.95$. Ab Durchlauf 14 ist die gierige Strategie jeder Iterierten gleich `pi_star`.
    1. Zeigen Sie, dass der Fehler von da an der linearen Rekursion $V_{k+1} - V^* = \gamma P_{\pi^*} (V_k - V^*)$ genügt. Dabei ist $P_{\pi^*}$ der Übergangskern unter $\pi^*$ aus :eqref:`eq_policy_value_solve`. Zeigen Sie außerdem, dass der Fehler an den Löchern und am Ziel null ist. Folgern Sie, dass die asymptotische Rate gleich dem Spektralradius $\rho$ von $\gamma P_{\pi^*}$ nach Einschränkung auf die Nichtendzustände ist. Berechnen Sie $\rho$.
    1. Berechnen Sie $\rho$ für $\gamma = 0.999$ mit der bei diesem Diskontfaktor optimalen Strategie. Sagen Sie voraus, wie viele Durchläufe die Wertiteration braucht, um einen Fehler von $10^{-6}$ zu erreichen. Die garantierte Schranke $\gamma^k \|V_0 - V^*\|_\infty$ benötigt mehr als $13{,}000$ Durchläufe. Prüfen Sie Ihre Vorhersage mit `value_iteration`. Welche Eigenschaft von FrozenLake bewirkt, dass $\rho$ kleiner ist als $\gamma$?
    1. Die Regel $\|V_k - V_{k-1}\|_\infty \leq 10^{-6}$ aus :numref:`subsec_vi_value_iteration` bricht im Durchlauf 128 ab. Der tatsächliche Fehler betrug dort das $11.7$-Fache der Änderung zwischen zwei Durchläufen. Dieser Faktor liegt deutlich unter dem Faktor $\gamma/(1-\gamma) = 19$ der Fehlerschranke. Zeigen Sie, dass sich für einen geometrisch mit dem Faktor $\rho$ abnehmenden Fehler der Faktor $\rho/(1-\rho)$ ergibt. Warum kann die Fehlerschranke :eqref:`eq_vi_certificate` nicht $\rho$ anstelle von $\gamma$ verwenden?
   :exlabel:`ex_value-iter-contraction-rate`

<!--en:2c72df1b-->
1. [code] **Ausmaß der Verbesserung.** ● Die Proposition zur Strategieverbesserung in :numref:`subsec_vi_policy_iteration` besagt, dass ein gieriger Schritt den Wert nicht senkt. Die folgende Identität, eine Matrixform des Lemmas zur Leistungsdifferenz :cite:`Kakade.Langford.2002`, misst den Gewinn. Für eine Strategie $\pi'$ bezeichne $\mathcal{T}^{\pi'}$ den Bewertungsoperator aus :eqref:`eq_policy_eval` für $\pi'$. Weiter sei $M_{\pi'}$ die Matrix aus Aufgabe :exref:`ex_value-iter-linear-system`.
    1. Zeigen Sie für zwei beliebige Strategien $\pi$ und $\pi'$ die Identität

       $$V^{\pi'} - V^\pi = M_{\pi'} \big( \mathcal{T}^{\pi'} V^\pi - V^\pi \big),$$

       und zeigen Sie, dass $(\mathcal{T}^{\pi'} V^\pi - V^\pi)(s) = \sum_a \pi'(a \mid s)\, \big[ Q^\pi(s, a) - V^\pi(s) \big]$ gilt.
    1. Beweisen Sie mit der Identität erneut die Proposition zur Strategieverbesserung. Zeigen Sie, dass die Verbesserung in jedem Zustand $s$ mit $Q^\pi(s, \pi'(s)) > V^\pi(s)$ strikt ist.
    1. Setzen Sie in der Identität $\pi^*$ für $\pi$ ein und zeigen Sie damit, dass $V^{\pi'} \leq V^*$ für jede stationäre Strategie $\pi'$ gilt, auch für stochastische Strategien.
    1. Prüfen Sie die Identität numerisch im Startzustand für $\pi$ = `uniform` und $\pi' = \pi^*$.
   :exlabel:`ex_value-iter-improvement-quantified`

<!--en:bbfc3b9e-->
1. [code] **Potentialbasierte Belohnungsgestaltung.** ● Die Proposition aus :numref:`subsec_vi_shaping` gilt für jedes Potential, wenn die Endzustände wie in `TabularMDP` absorbierend sind. Enden die Episoden dort wie im Simulator, gilt sie nur für Potentiale, die in den Endzuständen verschwinden.
    1. Addieren Sie eine Konstante $c$ zu jedem Eintrag von `mdp.r`, auch für die absorbierenden Zustände. Zeigen Sie, dass sich der diskontierte Ertrag jeder Trajektorie um denselben Betrag ändert, und bestimmen Sie ihn. Welches konstante Potential $\Phi$ erfüllt $\gamma \Phi(s') - \Phi(s) = c$?
    1. Berücksichtigen Sie die Konstante nun nur während der Episode. Addieren Sie $c$ ausschließlich in den Nichtendzuständen zu $r(s, a)$. Berechnen Sie für $c \in \{-0.02, -0.01, 0.01, 0.05\}$ die optimale Strategie mit `value_iteration`. Geben Sie ihre Aktion im Zustand 0 an, bestimmen Sie ihren Wert unter der ursprünglichen Belohnung mit `mdp.policy_value` und berechnen Sie wie in :numref:`subsec_vi_slippery_ice` die Wahrscheinlichkeit, innerhalb von 100 Schritten das Ziel zu erreichen. Erklären Sie die Wirkung beider Vorzeichen von $c$ anhand des konstanten Potentials und der Bedingung für die Endzustände.
    1. Ersetzen Sie den Annäherungsbonus von `shaped` durch das Potential $\Phi(s) = -0.3\, \ell(s)$. Dabei ist $\ell$ die Manhattan-Distanz `dist`. Mitteln Sie die veränderte Belohnung über den Folgezustand. Damit ergibt sich in jedem Zustand

        $$\tilde r(s, a) = r(s, a) + \sum_{s'} P(s' \mid s, a)\,[\gamma \Phi(s') - \Phi(s)].$$

       Prüfen Sie, dass die Werte $\tilde V$, die `value_iteration` zurückgibt, in jedem Zustand die Gleichung $\tilde V(s) = V^*(s) - \Phi(s)$ erfüllen. Prüfen Sie auch, dass die für die veränderte Belohnung optimale Strategie unter der ursprünglichen Belohnung denselben Wert hat wie `pi_star`. In Feld 6, wo *links* und *rechts* gleichwertig sind, können sich die beiden Strategien unterscheiden. Lassen Sie den Zusatzterm nun an den Endzuständen weg, wie es ein Simulator täte, der dort anhält. Berechnen Sie den Wert der optimalen Strategie unter der ursprünglichen Belohnung erneut. Setzen Sie schließlich $\Phi$ in den Endzuständen auf null und wiederholen Sie die Rechnung.

    *Nach Sutton und Barto,
    [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf),
    Aufgaben 3.15 und 3.16.*
   :exlabel:`ex_mdp-reward-design-its-failure`

<!--en:9a9f64f2-->
1. [extended] **Rechenbudget für Backups.** ●● Die Strategieiteration in :numref:`subsec_vi_policy_iteration` benötigte nur zwei Runden mit je einer exakten Lösung eines linearen Gleichungssystems. Die Wertiteration brauchte dagegen 164 Durchläufe, um die Genauigkeit ihrer Werte nachzuweisen. Bei vielen Zuständen ist die direkte Lösung zu teuer. Stattdessen bewerten wir die Strategie iterativ. Messen Sie die Kosten in *Backups*. Ein Backup wertet den Klammerausdruck aus :eqref:`eq_value_iteration` für ein Zustand-Aktion-Paar aus. Damit kostet ein Durchlauf der Wertiteration $|\mathcal{S}||\mathcal{A}|$ Backups und ein Bewertungsdurchlauf für eine deterministische Strategie $|\mathcal{S}|$. Die *modifizierte Strategieiteration* :cite:`Puterman.1994` vermittelt zwischen beiden Algorithmen. Sie beginnt mit $U_0 = 0$. In Runde $k$ liefert ein Durchlauf der Wertiteration zugleich die gierige Strategie $\pi_k$ bezüglich $U_k$. Darauf folgen $m$ Durchläufe von :eqref:`eq_policy_eval` für $\pi_k$, ausgehend von der aktuellen Tabelle. Mit dem Operator $\mathcal{T}^{\pi_k}$ aus :eqref:`eq_policy_eval` lässt sich eine Runde schreiben als $U_{k+1} = (\mathcal{T}^{\pi_k})^{m+1} U_k$. Für $m = 0$ ergibt sich die Wertiteration, und für $m \to \infty$ geht das Verfahren in die Strategieiteration über.
    1. Seien $V_k = \mathcal{T}^k U_0$ die Iterierten der Wertiteration. Zeigen Sie, dass aus $\mathcal{T} U_0 \geq U_0$ die Ungleichung $V_k \leq U_k \leq U_{k+1} \leq V^*$ für jedes $k$ und jedes $m$ folgt. Prüfen Sie, dass die Voraussetzung für FrozenLake erfüllt ist. Folgern Sie, dass die modifizierte Strategieiteration nach derselben Anzahl von Runden nie weiter von $V^*$ entfernt ist als die Wertiteration.
    1. Implementieren Sie die modifizierte Strategieiteration. Tragen Sie den Fehler $\|U - V^*\|_\infty$ der aktuellen Tabelle $U$ nach jedem Durchlauf gegen die bisherige Gesamtzahl der Backups auf. Verwenden Sie $m \in \{0, 1, 5, 20, 100\}$ auf den rutschigen $4 \times 4$- und $8 \times 8$-Karten mit $\gamma = 0.95$.
    1. Erklären Sie die Form der Kurven für große $m$. Warum sinkt der Fehler zwischen den gierigen Schritten nicht mehr?
    1. Wiederholen Sie Teil b für $\gamma = 0.99$. Wie ändert sich der Wert von $m$, der einen Fehler von $10^{-6}$ mit den wenigsten Backups erreicht, für $\gamma \to 1$?
   :exlabel:`ex_value-iter-backup-budgets`

<!--en:ce0b2b04-->
[Diskussion](https://d2l.discourse.group/t/36795)

<!--en:1b62fa42-->
Dynamische Programmierung<br>**Zustands- und Aktionswerte · Bellman-Erwartungswert- und Optimalitätsgleichungen · Kontraktion und Fehlerschranken · Strategieiteration · risikoaverse Steuerung**

<!--en:555c188b-->
::: {.slide title="Warum die Entfernung auf rutschigem Eis nicht ausreicht"}

<!--en:b0445376-->
[Motivation und Grundidee]{.kicker}

<!--en:8da6cdaa-->
Auf griffigem Eis führt der kürzeste Weg in 6 Zügen immer zum Ziel. Auf rutschigem Eis gelingt das mit derselben Strategie nur in **$4.5\%$** der Episoden:

<!--en:bf686d10-->
- **Gekoppelte mehrstufige Entscheidungen:** In Feld 9 kommt der Agent mit *rechts* auf griffigem Eis dem Ziel näher. Auf rutschigem Eis landet er dabei in einem von drei Fällen in Loch 5.

<!--en:b83836a8-->
- **Sicherheit oder Verzögerung:** Mit *unten* gerät der Agent nie in Loch 5, rutscht aber in zwei von drei Fällen seitwärts in Feld 8 oder 10.

<!--en:322d3848-->
- **Bellmans Einsicht:** Die Entfernung nach einem Schritt reicht nicht aus, um Aktionen zu vergleichen. Entscheidend ist der gesamte künftige Ertrag, den der Agent vom jeweiligen Feld aus noch erzielen kann.

<!--en:3bb12176-->
::: {.slide title="Zustandswertfunktion und Aktionswertfunktion"}

<!--en:92fe9cf1-->
[Bewertung von Zuständen und Befehlen]{.kicker}

<!--en:3889dc13-->
- **Deutung für FrozenLake:** $V^\pi(s)$ ist die Wahrscheinlichkeit, von $s$ aus das Ziel zu erreichen, gewichtet mit $\gamma^t$ bei Ankunft im Schritt $t$.

<!--en:cde3de2b-->
- **Aktionswert $Q^\pi(s, a)$:** Der Agent führt die Aktion $a$ im Schritt $0$ aus und folgt danach $\pi$. Es gilt $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$.

<!--en:66b77e0a-->
::: {.slide title="Bellman-Erwartungswertgleichungen und Backup mit Tensoren"}

<!--en:34c79b0f-->
[Rekursive Vorausschau um einen Schritt]{.kicker}

<!--en:ac872332-->
Die Zerlegung der Trajektorie nach dem ersten Schritt und die Markov-Eigenschaft führen von der Summe über den unendlichen Horizont zu einem linearen Gleichungssystem für benachbarte Zustände:

<!--en:2ce38032-->
- **Klammerausdruck für das Backup:** Für jeden Wertvektor $\mathbf{V} \in \mathbb{R}^{|\mathcal{S}|}$ genügt ein einziges Matrix-Vektor-Produkt, um die Vorausschau für alle Paare $(s, a)$ zu berechnen:

<!--en:f651b358-->
::: {.slide title="Die Bellman-Optimalitätsgleichungen"}

<!--en:dcdc949b-->
[Charakterisierung des optimalen Spiels]{.kicker}

<!--en:d7967fea-->
- **Optimalität der Aktionswerte:** $Q^*(s, a) = r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')$.

<!--en:51cf7fc7-->
- **Strategieableitung mit und ohne Modell:** Um $\pi^*$ aus $V^*$ abzuleiten, ist eine Vorausschau mit $P$ und $r$ nötig. Für $\pi^*(s) = \mathrm{argmax}_a Q^*(s, a)$ ist kein Modell nötig.

<!--en:087c261e-->
::: {.slide title="Exakte und stichprobenbasierte Bellman-Backups im Diagramm"}

<!--en:2f96d869-->
[Baumdiagramme für Backups]{.kicker}

<!--en:54c6716a-->
Jede Bellman-Gleichung bestimmt den Wert am Wurzelknoten aus den Werten eine Ebene darunter:

<!--en:d1e1c032-->
- **Erwartungswert und Optimalität (a–c):** $V^\pi$ und $Q^\pi$ mitteln über die Aktionen $\pi(a \mid s)$ der Strategie. Der Optimalitätsoperator ersetzt diesen Mittelwert durch $\max_a$ (als Bogen dargestellt).

<!--en:60cb5163-->
- **Stichprobenbasiertes Backup (d):** Ist $P$ unbekannt (§{sec_qlearning}), verwendet der lernende Agent einen einzigen beobachteten Übergang der Umgebung (blau), statt über $P(s' \mid s, a)$ zu summieren.

<!--en:664cb022-->
::: {.slide title="Wertiteration durch wiederholtes Anwenden des Bellman-Operators"}

<!--en:19e495e0-->
[Fixpunktiteration]{.kicker}

<!--en:eb14323f-->
Der **Bellman-Optimalitätsoperator** $\mathcal{T}$ bildet jede Wertetabelle $V$ auf das beste Ergebnis der Vorausschau um einen Schritt $(\mathcal{T}V)(s) = \max_a \big[r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)V(s')\big]$ ab:

<!--en:0712e503-->
- **Fixpunkteigenschaft:** Die optimale Wertetabelle $V^*$ erfüllt $V^* = \mathcal{T}V^*$. Die Wertiteration beginnt mit $V_0 = \mathbf{0}$ und wendet wiederholt $V_{k+1} = \mathcal{T}V_k$ an.

<!--en:aed428c2-->
::: {.slide title="Ausbreitung der Werte auf griffigem Eis"}

<!--en:7293394c-->
[Deterministischer Grenzfall der kürzesten Wege]{.kicker}

<!--en:d537fd9b-->
Bei deterministischen Übergängen breitet die Wertiteration die exakten Werte vom Ziel her rückwärts aus, pro Durchlauf um einen Ring:

<!--en:a84e6853-->
- **Geschlossene Formel für griffiges Eis:** Braucht der kürzeste Weg von einem Feld um die Löcher herum $j$ Züge, hat das Feld den exakten Wert $V^*(s) = \gamma^{j-1}$.

<!--en:64155355-->
- **Abschluss nach 6 Durchläufen:** Durchlauf $k$ liefert die endgültigen Werte aller Felder in Entfernung $k$. Das Startfeld ($j = 6$) bleibt fünf Durchläufe lang bei $0$ und erreicht im Durchlauf 6 seinen endgültigen Wert $\gamma^5 = 0.774$.

<!--en:052ba08a-->
- **Warum Rutschen eine Analyse erfordert:** Zufälliges Rutschen erzeugt Zyklen (etwa $14 \leftrightarrow 13$). Dadurch bleiben die Zustände über unendlich viele Durchläufe miteinander gekoppelt.

<!--en:d2d8b543-->
::: {.slide title="Kontraktion des Bellman-Operators in der Supremumsnorm"}

<!--en:d1b75494-->
[Warum die Wertiteration immer konvergiert]{.kicker}

<!--en:67832d9a-->
In der Supremumsnorm $\|V\|_\infty = \max_s |V(s)|$ verringert $\mathcal{T}$ den Abstand zwischen zwei beliebigen Wertetabellen auf höchstens das $\gamma$-Fache:

<!--en:9ad3ea62-->
- **Beweis mit zwei Ungleichungen:** Erstens ergibt das Mitteln über $P(\cdot \mid s, a)$ höchstens $\max_{s'} |V(s') - V'(s')|$. Zweitens gilt $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$.

<!--en:45740789-->
- **Geometrisch schrumpfende Kugeln:** Beträgt der anfängliche Abstand $r_0 = \|V_0 - V^*\|_\infty$, liegt die Iterierte nach Durchlauf $k$ im Quadrat der Supremumsnorm mit Radius $\gamma^k r_0$ um $V^*$.

<!--en:792cbb83-->
::: {.slide title="Eindeutigkeit, Konvergenzrate und Fehlerschranken"}

<!--en:52f43092-->
[Vier Folgerungen aus der Kontraktion]{.kicker}

<!--en:f7f52528-->
1. **Eindeutiger Fixpunkt:** Aus $V = \mathcal{T}V$ und $V' = \mathcal{T}V'$ folgt $V = V' = V^*$.

<!--en:f403a5dc-->
2. **Geometrische Abnahme:** Der Fehler $e_k \leq \gamma^k e_0$ sinkt pro Durchlauf auf höchstens das $\gamma$-Fache.

<!--en:c01a676b-->
3. **Beobachtbare Fehlerschranke als Abbruchkriterium:** Zwei aufeinanderfolgende Iterierte und der vom Horizont abhängige Faktor $\gamma/(1-\gamma)$ begrenzen den tatsächlichen Abstand.

<!--en:e217d5e1-->
::: {.slide title="Optimale Werte und Strategien auf griffigem und rutschigem Eis" layout="tight"}

<!--en:7194fcf9-->
[Exakte Lösung für FrozenLake]{.kicker}

<!--en:227b84e0-->
Die Wertiteration auf rutschigem Eis in FrozenLake beantwortet die Frage zu Feld 9. Sie zeigt, wie stark zufälliges Rutschen das optimale Verhalten verändert:

<!--en:5a4d2891-->
- **In Feld 9 lohnt sich Sicherheit:** $Q^*(9, \textrm{down}) = 0.375$ übertrifft $Q^*(9, \textrm{right}) = 0.289$. Um Loch 5 zu vermeiden, lohnt es sich, in zwei von drei Fällen seitwärts zu rutschen.

<!--en:bd399c64-->
- **Niedrigere Werte auf allen Feldern:** Durch das Rutschen sinkt der Startwert $V^*(s_0)$ von $0.774$ auf $0.180$. Selbst für Feld 14 direkt neben dem Ziel fällt der Wert von $1.000$ auf $0.724$.

<!--en:91639a44-->
::: {.slide title="Gemessene Konvergenz und Abbruchregeln" layout="tight"}

<!--en:5293367e-->
[Nachweis der Genauigkeit]{.kicker}

<!--en:5ac981b8-->
Der Vergleich des tatsächlichen Abstands $\|V_k - V^*\|_\infty$ mit der A-priori-Schranke $\gamma^k e_0$ und der beobachtbaren Fehlerschranke zeigt einen häufigen Fehler beim Abbruch:

<!--en:e568236a-->
- **Die Änderung allein führt zum vorzeitigen Abbruch:** Das Kriterium $\|V_k - V_{k-1}\|_\infty \leq 10^{-6}$ bricht im Durchlauf 128 ab. Der tatsächliche Fehler beträgt noch $1.1 \times 10^{-5}$ ($11\times$ über der Toleranz).

<!--en:e2bcfc15-->
- **Verlässliche Fehlerschranke:** Das Kriterium $\frac{\gamma}{1-\gamma}\|V_k - V_{k-1}\|_\infty \leq 10^{-6}$ bricht im Durchlauf 164 ab, nur 6 Durchläufe nachdem der tatsächliche Fehler im Durchlauf 158 unter $10^{-6}$ gefallen ist.

<!--en:8e5a53e1-->
::: {.slide title="Strategiebewertung durch Iteration oder direktes Lösen des Gleichungssystems" layout="tight"}

<!--en:5ceb004b-->
[Bewertung einer festen Strategie]{.kicker}

<!--en:7b898573-->
- **Lineares Bellman-Gleichungssystem:** Für eine feste Strategie $\pi$ gilt $V^\pi = r_\pi + \gamma P_\pi V^\pi$ mit $P_\pi(s' \mid s) = \sum_a \pi(a \mid s)P(s' \mid s, a)$.

<!--en:f74ed5d1-->
- **Exakte Lösung des Gleichungssystems:** Für $\gamma < 1$ liefert $V^\pi = (I - \gamma P_\pi)^{-1}r_\pi$ den Startwert $V(s_0) = 0.0078$ der Gleichverteilungsstrategie mit nur einer direkten Lösung zu Kosten von $O(|\mathcal{S}|^3)$.

<!--en:34bf1fdc-->
::: {.slide title="Strategiebewertung durch Iteration oder direktes Lösen des Gleichungssystems (Code)" layout="tight"}

<!--en:5ceb004b-2-->
[Bewertung einer festen Strategie]{.kicker}

<!--en:1ef322e5-->
::: {.slide title="Strategieverbesserung und Strategieiteration"}

<!--en:a68c6885-->
[Howards Verfahren zur monotonen Verbesserung]{.kicker}

<!--en:b2d65eab-->
Die gierige Wahl bezüglich $V^\pi$ nach $\pi'(s) = \mathrm{argmax}_a Q^\pi(s, a)$ garantiert in jedem Zustand die monotone Verbesserung $V^{\pi'}(s) \geq V^\pi(s)$:

<!--en:ffe9bd9d-->
- **Beweis durch Monotonie:** Die gierige Wahl sichert $\mathcal{T}^{\pi'}V^\pi = \mathcal{T}V^\pi \geq \mathcal{T}^\pi V^\pi = V^\pi$. Wiederholtes Anwenden des monotonen Operators $\mathcal{T}^{\pi'}$ ergibt $V^\pi \leq (\mathcal{T}^{\pi'})^k V^\pi \to V^{\pi'}$.

<!--en:c9296845-->
- **Abbruch auch bei Gleichstand:** Abbrechen, sobald überall $\max_a Q^\pi(s, a) \leq V^\pi(s) + \varepsilon$ gilt. Die Indizes von `argmax` können bei gleichwertigen Aktionen wie in Feld 6 zyklisch wechseln.

<!--en:089ab257-->
::: {.slide title="Verallgemeinerte Strategieiteration (GPI)"}

<!--en:e8a031aa-->
[Verschränkung von Bewertung und Verbesserung]{.kicker}

<!--en:e8e7fa98-->
Strategieiteration und Wertiteration sind zwei Extremfälle der **verallgemeinerten Strategieiteration**. Sie wechselt zwischen Bewertungsschritten in Richtung $V^\pi$ und gierigen Verbesserungsschritten in Richtung $\textrm{greedy}(V)$:

<!--en:2e221f6d-->
- **Strategieiteration:** Vor jedem gierigen Schritt steht die exakte Bewertung ($V = V^\pi$). Auf FrozenLake ergibt bereits der erste gierige Schritt mit den Werten der Gleichverteilungsstrategie $\pi^*$ (insgesamt 2 Gleichungssysteme).

<!--en:e7b054f2-->
- **Wertiteration:** Vor jeder Verbesserung steht ein einziger Bewertungsdurchlauf. Der Genauigkeitsnachweis für $V^*$ braucht 164 Durchläufe ($10{,}496$ Backups). Die gierige Strategie ist bereits nach Durchlauf 14 gleich $\pi^*$ ($896$ Backups).

<!--en:3966e81e-->
::: {.slide title="Auf rutschigem Eis gewinnt die optimale Strategie mit 16 zu 1"}

<!--en:f45c4686-->
[Exakte und mit Monte Carlo geschätzte Erfolgsraten]{.kicker}

<!--en:96b69d9c-->
100 Durchläufe der Strategiebewertung bei $\gamma = 1$ liefern die exakte Wahrscheinlichkeit, das Ziel innerhalb der 100 Schritte von Gymnasium zu erreichen. Die Ergebnisse stimmen mit Monte-Carlo-Schätzungen aus 2000 Episoden überein:

<!--en:020fd65c-->
- **Großer Unterschied in der Erfolgsrate:** Die für rutschiges Eis optimale Strategie $\pi^*$ erreicht das Ziel in **$73.0\%$** der Episoden. Beim kürzesten Weg für griffiges Eis sind es **$4.5\%$**, bei zufälliger Wahl **$1.4\%$**.

<!--en:6efb5c1a-->
::: {.slide title="Warum die optimale Strategie vom Ziel wegführt"}

<!--en:319135b6-->
[Risikoanalyse für die einzelnen Felder]{.kicker}

<!--en:2c7b8016-->
Die Lösung von $(I - P_\pi^\top)\,\mathbf{v} = \mathbf{e}_{s_0}$ liefert die erwarteten Besuchszahlen bis zum Ende der Episode und zeigt, von welchen Feldern der Agent in ein Loch fällt:

<!--en:890dbc6a-->
- **Der Rand schützt:** In fünf Feldern (0, 1, 3, 4, 14) gibt $\pi^*$ einen Befehl *gegen den Rand*. In der beabsichtigten Richtung bleibt der Agent stehen, und keine der beiden senkrechten Rutschrichtungen führt in ein Loch.

<!--en:027d43c0-->
- **Vom Ziel weg:** In vier Feldern (2, 6, 8, 10) zeigt $\pi^*$ vom Ziel weg. Das Risiko eines Sturzes im nächsten Schritt sinkt überall auf $0$, außer in Feld 6 zwischen den Löchern 5 und 7. Von dort gehen alle Stürze aus, insgesamt $22\%$ der Episoden.

<!--en:9b643a1a-->
::: {.slide title="Wie ein naiver Annäherungsbonus die optimale Strategie verändert"}

<!--en:b6946204-->
[Überprüfung der Belohnungsgestaltung]{.kicker}

<!--en:1795567c-->
Die Wertiteration für den veränderten MDP $\tilde{r}(s, a) = r(s, a) + b \sum_{s'} P(s' \mid s, a)\mathbf{1}(\ell(s') < \ell(s))$ zeigt, dass ein naiver Fortschrittsbonus auf rutschigem Eis die Optimalität von $\pi^*$ aufhebt:

<!--en:e4faffb7-->
- **Endlosschleife ($b = 0.3$):** Im veränderten Optimum wählt der Agent in Feld 14 *links* (`<`), um nach Feld 13 zurückzukehren und den Bonus immer wieder zu erhalten. Sein Wert mit Bonus beträgt $2.15$, sein **tatsächlicher Wert für das Erreichen des Ziels $0.000$**.

<!--en:d001c7e5-->
- **Invarianz bei Potentialdifferenzen (Ng et al., 1999):** Die Belohnung $\tilde{r} = r + \gamma\Phi(s') - \Phi(s)$ verschiebt die Werte aller Strategien um denselben Term $-\Phi(s)$ ($\tilde{V}^\pi = V^\pi - \Phi$). Damit bleibt $\pi^*$ optimal.

<!--en:95ccbec2-->
::: {.slide title="Verständnisfragen zu Bellman-Gleichungen und Kontraktion"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:7e2f748c-->
Drei Fragen zur dynamischen Programmierung beantworten:

<!--en:279e30a8-->
1. **Bellman-Gleichung für Feld 14:** Feld 14 liegt links vom Ziel am unteren Rand. Die Strategie $\pi$ wählt in Feld 14 *unten*. Den Wert $V^\pi(14)$ mithilfe der Bellman-Erwartungswertgleichung durch $V^\pi(13)$ und $V^\pi(14)$ ausdrücken.

<!--en:09bf1316-->
2. **Kontraktion und Fehlerschranke:** Es gelte $\gamma = 0.95$. Zwei aufeinanderfolgende Durchläufe der Wertiteration erfüllen $\|V_k - V_{k-1}\|_\infty = 10^{-5}$. Welche obere Schranke ergibt sich daraus für $\|V_k - V^*\|_\infty$?

<!--en:37a95d2a-->
3. **Aktionswerte und Zustandswerte:** Nach der Konvergenz liegen die optimalen Tabellen vor. Warum kann der Agent mit $Q^*$ ohne Kenntnis von $P(s' \mid s, a)$ optimal handeln, während er bei Verwendung von $V^*$ weiterhin $P$ benötigt?

<!--en:557b98b6-->
**Lösungen:**

<!--en:8f71996d-->
1. Beim Befehl *unten* stößt der Agent an den unteren Rand (und bleibt in Feld 14), rutscht nach links in Feld 13 oder nach rechts ins Ziel (Belohnung 1, Wert 0). Jeder Fall hat Wahrscheinlichkeit $1/3$. Es gilt also $V^\pi(14) = \frac{1}{3} + \frac{\gamma}{3}[V^\pi(13) + V^\pi(14)]$.

<!--en:b1dfe87c-->
2. Aus der Ungleichung für die Fehlerschranke folgt $\|V_k - V^*\|_\infty \leq \frac{\gamma}{1-\gamma}\|V_k - V_{k-1}\|_\infty = 19 \times 10^{-5} = 1.9 \times 10^{-4}$.

<!--en:b1226b0e-->
3. Da $Q^*(s, a)$ bereits über die Übergangsverteilung $P(\cdot \mid s, a)$ mittelt, genügt für die gierige Wahl $\mathrm{argmax}_a Q^*(s, a)$. Bei $V^*$ ist dagegen $r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)V^*(s')$ zu berechnen.

<!--en:ab5fb31e-->
::: {.slide title="Zusammenfassung der Verfahren der dynamischen Programmierung"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:4f8b270a-->
| Verfahren oder Operator | Rechenschritt | Rechenkosten und Garantie |

<!--en:075d9f66-->
| **Bellman-Erwartungswertoperator $\mathcal{T}^\pi$** | $(\mathcal{T}^\pi V)(s) = \sum_a \pi(a \mid s)[r + \gamma \sum_{s'} P V(s')]$ | Affine $\gamma$-Kontraktion mit Fixpunkt $V^\pi = (I - \gamma P_\pi)^{-1}r_\pi$ |

<!--en:c0cccee4-->
| **Bellman-Optimalitätsoperator $\mathcal{T}$** | $(\mathcal{T}V)(s) = \max_a [r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)V(s')]$ | Nichtlineare $\gamma$-Kontraktion in $\|\cdot\|_\infty$ mit eindeutigem Fixpunkt $V^*$ |

<!--en:d9315a92-->
| **Wertiteration** | $V_{k+1} = \mathcal{T}V_k$ ausgehend von $V_0 = \mathbf{0}$ | $O(|\mathcal{S}|^2|\mathcal{A}|)$ pro Durchlauf, Fehler $\leq \frac{\gamma}{1-\gamma}\|V_k - V_{k-1}\|_\infty$ |

<!--en:e0ca16e9-->
| **Strategieiteration** | Löse $V^{\pi_k} = \mathcal{T}^{\pi_k}V^{\pi_k}$ und setze $\pi_{k+1} = \mathrm{argmax}_a Q^{\pi_k}$ | $O(|\mathcal{S}|^3)$ pro exakter Lösung, Konvergenz auf dem See in 2 äußeren Runden |

<!--en:93c77c3d-->
| **Potentialbasierte Belohnungsgestaltung** | $\tilde{r}(s, a, s') = r(s, a) + \gamma\Phi(s') - \Phi(s)$ | Verschiebung gemäß $\tilde{Q}^*(s, a) = Q^*(s, a) - \Phi(s)$, dadurch bleibt $\pi^*$ exakt erhalten |
