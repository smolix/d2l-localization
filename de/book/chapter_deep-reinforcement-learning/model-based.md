<!--en:01465484-->
# Gelernte Modelle und Planung

<!--en:8c8a5af5-->
Soft Actor-Critic schwang das Pendel nach einigen Tausend Übergängen auf. Jeden Übergang nutzte das Verfahren auf dieselbe Weise: Die Belohnung plus der diskontierte Wert des Folgezustands diente dem Kritiker als Zielwert für eine Regression. Ein Übergang verrät aber mehr. Er zeigt, wie sich das Pendel unter einem bestimmten Drehmoment bewegt hat. Die physikalischen Gesetze dahinter gelten bei jedem Besuch. Ein Kind, das auf einer Schaukel ein paarmal angeschubst wurde, kann sagen, wo die Schaukel im nächsten Augenblick sein wird und wie kräftig man schubsen muss, damit sie höher schwingt. Könnte ein Agent die Physik des Pendels aus einigen Hundert Schritten lernen und das Aufschwingen planen, ganz ohne Kritiker und ohne Strategie (engl. policy)?

<!--en:7dc05427-->
Auf dem See haben wir das in :numref:`sec_qlearning` getan. Wir zählten die beobachteten Übergänge, schätzten daraus ein Modell und behandelten es bei der Wertiteration als exakt. In :numref:`sec_optimal_control` planten wir für die kontinuierlichen Zustände von Pendulum. Dazu differenzierten wir in iLQR die bekannten Bewegungsgleichungen. Mit modellprädiktiver Regelung mit gleitendem Horizont (engl. model predictive control, MPC) planten wir bei jedem Schritt neu. Ist die Physik unbekannt, hilft das Zählen bei den kontinuierlichen Zuständen von Pendulum nicht weiter. An seine Stelle tritt die Regression: Ein Netz lernt, aus dem aktuellen Zustand und dem Drehmoment den Folgezustand und die Belohnung vorherzusagen. Eine Suche mit gleitendem Horizont ersetzt die Wertiteration oder iLQR. Der Agent probiert im gelernten Modell viele Drehmomentfolgen aus und wendet das erste Drehmoment der besten Folge an. Suttons Dyna-Architektur verband 1990 ein gelerntes Modell mit Planung :cite:`Sutton.1990`. Mit einer solchen Suche in neuronalen Netzen als Modellen lassen sich simulierte Roboter schon mit wenigen Minuten Erfahrung steuern :cite:`Nagabandi.Kahn.Fearing.ea.2018`.

<!--en:ae482f38-->
Mit einem Modell kann ein Agent Erfahrung durch Rechenaufwand ersetzen: Auf jeden realen Schritt können Tausende simulierte Schritte folgen. Die Suche kann auch mit einem Vorschlag einer Strategie beginnen. Ihr Ergebnis dient dann der Strategie als Lernziel. So lernten AlphaZero und MuZero, Brettspiele auf übermenschlichem Niveau zu spielen :cite:`Silver.Hubert.Schrittwieser.ea.2018,Schrittwieser.Antonoglou.Hubert.ea.2020`. Auch Sprachmodelle können zur Testzeit zusätzliche Rechenleistung für eine Suche nutzen, die eine gelernte Strategie leitet. In diesem Abschnitt passen wir ein Modell von Pendulum an eine einzige Zufallsepisode an. Wir messen, wie seine Fehler im Verlauf eines Plans wachsen, planen mit dem Modell und gewinnen aus der Suche eine bessere Strategie. Dabei begleitet uns eine Frage: Wie weit über seine Daten hinaus können wir einem Modell vertrauen?

<!--en:b0da4cc8-->
## Ein Dynamikmodell aus Daten

<!--en:150b2515-->
### Regression statt Zählen

<!--en:451c1594-->
Pendulum (`Pendulum-v1`) ist der Benchmark für kontinuierliche Regelung, den wir in :numref:`sec_optimal_control` mit bekannter Physik eingeführt und in :numref:`sec_deeprl` und :numref:`sec_sac` modellfrei behandelt haben. Ein Pendel hängt an einer motorgetriebenen Achse. Sein Zustand ist $s = (\cos\vartheta, \sin\vartheta, \dot\vartheta)$, wobei $\vartheta$ den Winkel zur aufrechten Lage bezeichnet. Die Aktion ist ein Drehmoment $a \in [-2, 2]$. Jeder Schritt liefert die Belohnung $-(\vartheta^2 + 0.1\,\dot\vartheta^2 + 0.001\,a^2)$, wobei wir $\vartheta$ auf das Intervall $[-\pi, \pi]$ zurückführen. Eine Episode dauert 200 Schritte. Hängt das Pendel unten in Ruhe, verliert es $\pi^2 \approx 9.9$ Punkte pro Schritt, also etwa $2{,}000$ pro Episode. Bei zufälligen Drehmomenten beträgt der Verlust etwa $900$ bis $1{,}700$ Punkte. Ein Regler, der das Pendel aufschwingt und oben balanciert, übertrifft die Schwelle von $-200$ aus :numref:`sec_optimal_control` und :numref:`sec_deeprl`. Je nach Ausgangslage verliert er zwischen null und einigen Hundert Punkten.

<!--en:fa3f4188-->
Ein *Modell* der Umgebung sagt vorher, wie sich die Umgebung verhält. Wir bezeichnen mit $\hat f_\phi$ ein Netz mit den Parametern $\phi$. Es bildet einen Zustand und eine Aktion auf Vorhersagen für den Folgezustand und die Belohnung ab:

<!--en:95d23be1-->
Auf dem See war das Modell eine Häufigkeitstabelle. Jeden Eintrag schätzten wir allein anhand der Besuche beim jeweiligen Zustand-Aktion-Paar. Ein Netz verwendet für alle Zustände dieselben Parameter. Deshalb lernt das Modell aus einem Übergang bei einem bestimmten Winkel auch etwas über benachbarte Winkel. Wir passen es mit der Methode der kleinsten Quadrate an die gespeicherten Übergänge $(s_i, a_i, r_i, s'_i)$ an:

<!--en:44ff6cca-->
Bei der Anpassung soll das Modell für jeden gespeicherten Übergang den aufgezeichneten Folgezustand und die Belohnung reproduzieren. Bei einer Tabelle anstelle des Netzes wäre der Minimierer für jedes Paar der Mittelwert der dort beobachteten Ergebnisse. Das entspricht dem durch Zählen gewonnenen Modell aus :numref:`sec_qlearning`, reduziert auf seinen Mittelwert. Pendulum ist deterministisch, also stimmt das mittlere Ergebnis mit dem tatsächlichen überein. Für eine stochastische Umgebung bräuchten wir ein Modell der gesamten Verteilung der Folgezustände, wie es die Häufigkeiten auf dem See lieferten.

<!--en:bfa29d98-->
Zwei Entscheidungen erleichtern die Regression. Das Netz sagt die Zustandsänderung $s' - s$ vorher statt des Folgezustands $s'$ selbst. In den 0.05 Sekunden eines Schritts bewegt sich das Pendel nur wenig, und genau diese Änderung lohnt es sich zu lernen. Außerdem standardisieren wir jede Eingabe- und Zielkoordinate über die gespeicherten Daten auf Mittelwert null und Varianz eins. Denn die Koordinaten haben unterschiedliche Skalen: Kosinus und Sinus liegen in $[-1, 1]$, die Winkelgeschwindigkeit in $[-8, 8]$ und die Belohnung ungefähr in $[-16, 0]$. Ohne Standardisierung würde die Methode der kleinsten Quadrate vor allem die Belohnung anpassen.

<!--en:5624b82a-->
Zuerst legen wir die Einstellungen für den gesamten Abschnitt fest: die Drehmomentgrenze $c = 2$, die Tiefe $n = 25$ eines Plans, die Anzahl der Pläne, die die Suche ausprobiert, die Anzahl der Adam-Schritte pro Modellanpassung, die Anzahl der Zufallsstartwerte und den Zufallsstartwert, der die Anfangszustände für unseren Test festlegt.

<!--en:e857c11e-->
Unsere Daten stammen aus vollständigen Episoden. `run_episode` arbeitet in einer *Vektorumgebung* (engl. vector environment). In diesem Container von gymnasium durchlaufen mehrere Kopien von Pendulum ihre Schritte synchron. So können wir einen Regler mit einem Aufruf von zehn Anfangszuständen aus gleichzeitig ausführen. Als Regler eignet sich jede Funktion, die einen Batch von Beobachtungen auf einen Batch von Aktionen abbildet. `random_controller` liefert einen Regler, der Drehmomente aus einer Gleichverteilung zieht. `score` führt einen Regler für je eine Episode von zehn festen Anfangszuständen aus. Diese legt die Umgebung mit den Zufallsstartwerten 1000 bis 1009 fest. Die Funktion liefert den mittleren Ertrag (engl. return) und dient als Test für jeden Regler in diesem Abschnitt. Die Übergänge gelangen in einen `d2l.ReplayBufferC`, den Replay-Puffer aus :numref:`sec_sac` mit einer Gleitkommaspalte für Aktionen. Hier speichern wir darin nur Arrays und ziehen keine Batches daraus.

<!--en:43d31f38-->
`model_data` bildet aus den gespeicherten Übergängen die Eingaben $(s, a)$ und Zielwerte $(s' - s, r)$ der Regression samt ihren Mittelwerten und Standardabweichungen.

<!--en:10d38d0b-->
`DynamicsModel` ist ein mehrschichtiges Perzeptron mit zwei versteckten Schichten zu je 64 Einheiten und der glatten SiLU-Aktivierung (:numref:`sec_mlp`). Neben den Parametern gehören die Kenngrößen für die Standardisierung zu seinem Zustand. Der Vorwärtsdurchlauf standardisiert die Eingabe, macht die Standardisierung der Ausgabe rückgängig und addiert die vorhergesagte Änderung zum Zustand. `fit_model` berechnet die Kenngrößen aus allen gespeicherten Daten neu und führt dann Adam-Schritte mit Minibatches von 256 Übergängen aus. Der Optimierer entsteht einmalig zusammen mit dem Modell in `make_model`. Jede spätere Anpassung setzt also bei den bisherigen Parametern an.

<!--en:53897f86-->
Die Schichten sind vom Typ `d2l.Linear` und verwenden damit dieselbe Initialisierung wie alle JAX-Netze in diesen beiden Kapiteln (:numref:`sec_deeprl`). Die Kenngrößen liegen in einer `nnx.Variable`. So lesen die unten kompilierten Funktionen nach jeder erneuten Anpassung die aktuellen Werte. `model_step` führt einen kompilierten Adam-Schritt aus. `nnx.cached_partial` bindet das Modell und den Optimierer einmal pro Anpassung an diese Funktion.

<!--en:e7b6e9e3-->
Mit dem Zufallsstartwert 0 zeichnen wir eine Episode mit zufälligen Drehmomenten auf und passen das Modell an ihre 200 Übergänge an:

<!--en:301912b6-->
In der Zufallsepisode gehen etwa tausend Punkte verloren. Am Ende der Anpassung beträgt der Verlust nur einen kleinen Bruchteil der Varianz der Zielwerte, die nach der Standardisierung eins beträgt.

<!--en:ce341011-->
### Einen Schritt von Hand nachrechnen

<!--en:ff6286ea-->
Ein kleiner Trainingsverlust verrät uns noch nicht, was das Modell vorhersagt. Wir betrachten einen Übergang der Episode in seinen physikalischen Einheiten. `rollout_model` übergibt dem Modell einen Zustand und eine Folge von Aktionen. Jeder vorhergesagte Zustand dient als Eingabe für den nächsten Schritt. Bei einer einzigen Aktion erhalten wir genau eine Vorhersage.

<!--en:a5172b7b-->
`rollout_model` führt die Schritte mit `jax.lax.scan` in der kompilierten Funktion `_rollout_model` aus. Für jede Form der Eingaben wird diese Funktion einmal kompiliert. In diesem Abschnitt rufen wir sie nur mit wenigen verschiedenen Formen auf. Wie im anderen Reiter erhalten wir die Ergebnisse als NumPy-Arrays.

<!--en:035c72d7-->
Bei Schritt 100 beträgt der Winkel des Pendels $\vartheta = \operatorname{atan2}(-0.3394, -0.9407) = -2.795$ Radiant, also etwa $160$ Grad zur aufrechten Lage. Es schwingt mit $5.69$ Radiant pro Sekunde bei einem Drehmoment von fast null. Wie die Ausgabe zeigt, beträgt die Belohnung $-(2.795^2 + 0.1 \cdot 5.690^2 + 0.001 \cdot 0.08^2) = -11.05$. In den 0.05 Sekunden eines Schritts ändert sich der Winkel auf $\operatorname{atan2}(-0.5789, -0.8154) = -144.6$ Grad. Das Pendel hat sich also um etwa $5.6 \times 0.05 = 0.28$ Radiant oder 16 Grad gedreht. Das Modell sagt diese Bewegung bis auf einen Bruchteil eines Grads genau vorher. Bei der Belohnung liegt es um weniger als drei Zehntel daneben. Ein Schritt ist leicht vorherzusagen: Das Modell muss einen gerade beobachteten Zustand nur geringfügig korrigieren.

<!--en:8c5fef1b-->
### Wie sich Fehler fortpflanzen

<!--en:65586f12-->
Ein Plan braucht mehr als einen Schritt. Um eine Drehmomentfolge zu bewerten, muss das Modell 25 Schritte vorausschauen. Vom zweiten Schritt an ist seine Eingabe die eigene vorherige Vorhersage. Beim Training hat das Modell nur tatsächliche Zustände gesehen, nie seine eigenen, leicht fehlerhaften Vorhersagen. Deshalb verändert ein Fehler in einem Schritt die Eingabe des nächsten. Ein kleiner Winkelfehler verändert das Drehmoment der Schwerkraft und damit die Geschwindigkeit, was wiederum den Winkel weiter verändert. Wir messen diesen Effekt anhand simulierter Trajektorien *ohne Rückführung* (engl. open-loop). Dabei geben wir die in einer Episode aufgezeichneten Aktionen vor und lassen das Modell die Zustände vorhersagen. Nach $k$ Schritten vergleichen wir den vorhergesagten Winkel mit dem aufgezeichneten.

<!--en:6cef5db4-->
`kstep_error` startet bei jedem Schritt einer Episode, dem noch mindestens 25 Schritte folgen, eine simulierte Trajektorie. Die Funktion liefert das quadratische Mittel der Winkelfehler nach $k = 1, 5, 10, 25$ Schritten. Dabei berücksichtigen wir den Winkelumlauf, sodass $+179$ und $-179$ Grad zwei Grad auseinanderliegen.

<!--en:2ff95cda-->
Für jeden von fünf Zufallsstartwerten passen wir ein Modell an eine Zufallsepisode an und testen es an einer zweiten Zufallsepisode mit demselben Startwert:

<!--en:fc41ee12-->
Bei jedem Zufallsstartwert beträgt der Fehler nach zehn Schritten ein Mehrfaches des Fehlers nach einem Schritt. Nach einem Schritt liegt er bei den meisten Startwerten unter einem Grad. Nach 25 Schritten hängt der Fehler vom Zufallsstartwert ab, bleibt aber in diesen zurückgehaltenen Episoden unter etwa 20 Grad. Bei zufälligen Drehmomenten schwingt das Pendel hauptsächlich im unteren Bereich, den es auch in der Trainingsepisode besucht hat. Auf diese Einschränkung kommen wir zurück, sobald der Planer das Pendel in andere Bereiche geführt hat.

<!--en:65d58b37-->
Die Tiefe eines Plans ist eine Stellgröße wie das $n$ der $n$-Schritt-Erträge in :numref:`sec_actorcritic`. Mit einem tieferen Plan lassen sich die Folgen eines Drehmoments weiter überblicken. Das Aufschwingen dauert einige Dutzend Schritte. Endet der Plan, bevor das Pendel an Höhe gewinnt, kann er eine gute Folge von Drehmomenten zur Energiezufuhr nicht von einer schlechten unterscheiden. Mit jedem weiteren Schritt geht aber auch der Modellfehler erneut in die Vorhersage ein. Wir verwenden $n = 25$ Schritte, also 1.25 Sekunden. In :exref:`ex_model-based-horizon` variieren Sie diese Stellgröße.

<!--en:e5bcc74d-->
## Planung im Modell

<!--en:c176557c-->
### Random Shooting

<!--en:e4f32593-->
Eine Drehmomentfolge im Modell auszuprobieren kostet einige Matrixmultiplikationen und keinen einzigen realen Schritt. Die einfachste Suche nutzt das direkt. Nehmen wir an, das Pendel hängt unten in Ruhe. Wir ziehen fünf zufällige Folgen von je 25 Drehmomenten, simulieren jede im eben angepassten Modell und summieren ihre vorhergesagten Belohnungen:

<!--en:f47fca04-->
Bliebe das Pendel 25 Schritte lang in Ruhe, verlöre es etwa $25 \times 9.9 \approx 247$ Punkte. Die fünf Vorhersagen für den Ertrag liegen deutlich über diesem Ergebnis und nur wenige Punkte auseinander. Für die letzte Spalte spielen wir dieselben fünf Folgen vom selben Zustand aus in der realen Umgebung durch. Dabei verlieren wir zwischen 223 und 242 Punkten. Wenn wir das Pendel unten leicht ins Schwingen bringen, sparen wir gegenüber der Ruhelage also 5 bis 24 Punkte. Das Modell sagt dagegen eine Ersparnis von 60 bis 80 Punkten vorher. Bei jeder Folge liegt es um 40 bis 70 Punkte zu hoch. Der Grund liegt vermutlich in der Abdeckung durch die Daten. In der Zufallsepisode durchlief das Pendel den tiefsten Punkt mit hoher Geschwindigkeit. Das Modell sah es dort nie annähernd in Ruhe und sagt für diesen Zustand zu hohe Belohnungen vorher. Trotzdem ist die Folge mit dem höchsten vorhergesagten Ertrag auch in der realen Umgebung die beste der fünf. Zur Auswahl unter den Kandidaten genügt ihre Rangfolge, ihre Werte brauchen wir nicht. Wir würden also diese Folge wählen und ihr erstes Drehmoment anwenden. Schon in diesem kleinen Beispiel stellt sich die Frage, wie weit wir dem Modell vertrauen können: Seine Werte sind falsch, seine Wahl ist richtig.

<!--en:42f34c92-->
Genau das ist *Random Shooting*: Wir ziehen viele Kandidatenfolgen $(a_0^{(j)}, \ldots, a_{n-1}^{(j)})$ und sagen ihre Erträge vorher:

<!--en:4916ff17-->
Dabei ist $\hat r_t^{(j)}$ die Belohnung, die das Modell für Schritt $t$ des Kandidaten $j$ vorhersagt. Wir behalten die beste Folge. Die Summe ist undiskontiert, denn über 25 Schritte würde eine Diskontierung mit $\gamma = 0.99$ die Gewichte höchstens um etwa ein Fünftel ändern. Der Agent führt nur das erste Drehmoment der besten Folge aus. Im nächsten Schritt beobachtet er den realen Zustand und plant von dort aus neu. Das ist die *modellprädiktive Regelung* (MPC) mit gleitendem Horizont aus :numref:`sec_optimal_control`. Dort planten wir mit den bekannten Bewegungsgleichungen des Pendels, hier sucht der Agent im gelernten Modell $\hat f_\phi$. Da er bei jedem Schritt neu plant, häufen sich die Modellfehler auf der tatsächlich durchlaufenen Trajektorie nicht an: Die nächste reale Beobachtung ersetzt die Vorhersage, bevor sich deren Fehler fortpflanzen können. Die Fehler in den späteren Schritten eines Plans beeinflussen aber weiterhin, welches erste Drehmoment gewinnt.

<!--en:197007b4-->
`plan` führt diese Suche für einen ganzen Batch von $B$ Zuständen gleichzeitig aus. Für jeden Zustand erzeugt die Funktion aus einem vorgegebenen Plan 500 Kandidaten, indem sie ihn mit gaußschem Rauschen variiert. Einer der Kandidaten ist der vorgegebene Plan selbst. Die Funktion begrenzt die Drehmomente auf $[-c, c]$, simuliert alle Kandidaten mit `rollout_model` und gibt für jeden Zustand die beste Folge samt ihrem vorhergesagten Ertrag zurück. Da der Ausgangsplan zu den Kandidaten gehört, ist der vorhergesagte Ertrag des gewählten Plans mindestens so hoch wie der des Ausgangsplans. Eine Aktion kostet $500 \times 25 = 12{,}500$ Modellauswertungen.

<!--en:66b8cddb-->
### Initialisierung mit dem bisherigen Plan

<!--en:bbb23120-->
Mit welchem Plan soll die Suche beginnen? Ohne Vorwissen würde sie bei jedem Schritt neue Zufallsfolgen ziehen. Damit verwirft sie den vermutlich guten Plan aus dem vorigen Schritt und muss jedes Mal aufs Neue unter 500 Zufallsfolgen eine zum Aufschwingen finden. Aufeinanderfolgende Drehmomente stammen dann aus voneinander unabhängigen Plänen. Bei der *Initialisierung mit dem bisherigen Plan* (engl. warm start) verwenden wir den letzten Plan erneut. Sein erstes Drehmoment haben wir gerade ausgeführt. Wir rücken die übrigen 24 um einen Schritt vor und hängen ein Drehmoment von null an. Die Suche variiert diesen verschobenen Plan. So müssen wir einen guten Plan nur einmal finden und können ihn von Schritt zu Schritt verfeinern. :numref:`fig_rl_model_planning` zeigt einen Schritt dieser Suche.

<!--en:d1088e90-->
![Planung in einem gelernten Modell durch Random Shooting mit Initialisierung aus dem bisherigen Plan. (a) Im Zustand $s_t$ ziehen wir Aktionsfolgen um den um einen Schritt verschobenen bisherigen Plan (gestrichelt). Wir simulieren sie mit $\hat f_\phi$ über 25 Schritte und bewerten sie anhand der Summe der vorhergesagten Belohnungen (Zahlen rechts). Nur die erste Aktion der besten Folge (blau) wird ausgeführt. (b) Der reale Folgezustand weicht leicht vom vorhergesagten Zustand ab (offener Kreis). Von dort aus sucht der Planer erneut in der Umgebung des um einen Schritt verschobenen besten Plans aus (a). (c) Ohne Rückführung wächst der Modellfehler mit dem Horizont. Das ist ein Grund, bei jedem Schritt neu zu planen. Die vertikale Achse und die Scores stammen aus einem schematischen eindimensionalen Modell statt aus Pendulum. Ein Score summiert die Belohnungen aller Schritte einschließlich der Kosten für Geschwindigkeit und Drehmoment. Die Folge, die der aufrechten Lage am Ende am nächsten kommt, muss also nicht den höchsten Score haben.](../img/mdl-rl-model-planning.svg)

<!--en:47f308e4-->
`mpc` baut einen solchen Regler aus einem Modell auf. Beim ersten Aufruf entsteht für jede Umgebung ein Plan, der im Funktionsabschluss des Reglers gespeichert bleibt. Die Standardabweichung des Rauschens entspricht der Drehmomentgrenze $c = 2$. In separaten Läufen mit fünf Zufallsstartwerten verglichen wir diesen Regler mit einem Planer, der bei jedem Schritt 500 neue Folgen aus einer Gleichverteilung zieht. Für die Bewertung verwendeten wir zehn andere Anfangszustände. Nach 600 realen Schritten erreichte der mit dem bisherigen Plan initialisierte Planer in jedem Lauf einen Score zwischen etwa $-115$ und $-155$. Der Planer mit jeweils neuen Folgen lag zwischen etwa $-165$ und $-555$. Nach 1000 realen Schritten lag er bei jedem Zufallsstartwert immer noch 40 bis 110 Punkte zurück.

<!--en:68067265-->
### Aufschwingen nach wenigen Episoden

<!--en:b48ae6e1-->
Wie viele reale Schritte braucht der Planer? `mpc_run` beginnt mit einer Zufallsepisode. Dann passen wir dreimal das Modell an alle bisherigen Daten an, bewerten den Planer anhand der zehn Testanfangszustände und führen mit ihm eine Episode aus, um weitere Daten zu sammeln. Zum Schluss passen wir das Modell noch einmal an und bewerten den Planer erneut. Die Scores geben also an, wie gut der Planer nach 200, 400, 600 und 800 realen Schritten ist.

<!--en:152d9293-->
Die Ausgabe zeigt für fünf Zufallsstartwerte die Erträge der vier aufgezeichneten Episoden und die vier Scores:

<!--en:9b123c58-->
Was ist ein guter Score? Betrachten wir einen Regler, der das Pendel aufschwingt und oben balanciert. Von den zehn Testanfangszuständen kostet ihn der oberste fast keinen Punkt. Bei jedem der fünf Anfangszustände im unteren Bereich verliert er etwa 250 bis 320 Punkte. Im Mittel über alle zehn verliert er etwa 170 Punkte. Bewerten wir den Greedy-Akteur von Soft Actor-Critic aus :numref:`sec_sac` nach jeder Episode auf dieselbe Weise, pendelt sich sein Score bei etwa $-170$ ein. Bei einer Aktualisierung pro Schritt erreicht er $-200$ erstmals nach etwa 5000 bis 7500 Schritten.

<!--en:b7e9e816-->
Nach 600 realen Schritten, also drei Episoden, liegt der Planer in jedem Lauf zwischen etwa $-165$ und $-215$ und damit nahe an diesem Niveau. Nach 800 Schritten liegt er zwischen etwa $-170$ und $-195$. Er erreicht $-200$ erstmals nach 200 bis 800 realen Schritten, in den meisten Läufen schon nach höchstens 600. Bei drei der fünf Zufallsstartwerte liefert das Modell aus der Zufallsepisode allein bereits einen Planer mit einem Score zwischen etwa $-175$ und $-340$. Die Zufallsstrategie liegt mit $-1{,}000$ oder weniger weit darunter. Bei den beiden übrigen Zufallsstartwerten 3 und 4 scheitert der erste Planer mit einem Score um $-1{,}500$. Seine ersten ein bis zwei Episoden misslingen ebenfalls. Nach 600 realen Schritten gelingt demselben Planer das Aufschwingen.

<!--en:1d277183-->
### Abdeckung durch die Daten

<!--en:4cb692f6-->
Bei den beiden Zufallsstartwerten mit dem langsamsten Fortschritt waren auch die Verluste in der Zufallsepisode am höchsten: etwa $1{,}500$ und $1{,}700$ Punkte. Die zufälligen Drehmomente hoben das Pendel nie über die Horizontale. Wie in der Fehlertabelle oben nehmen wir für jeden Zufallsstartwert das Modell aus der einen Zufallsepisode. Diesmal testen wir es anhand der Episoden des Planers selbst statt an einer zweiten Zufallsepisode:

<!--en:3f583262-->
Bei den Zufallsstartwerten 0 bis 2 sind die Fehler in den Episoden des Planers anfangs so klein wie bei zufälligen Daten: Nach einem Schritt betragen sie einen Bruchteil eines Grads. Nach 25 Schritten erreichen sie einige Dutzend Grad. In diesen Episoden verbringt das Pendel die meiste Zeit in der Nähe der instabilen aufrechten Lage. Dort entscheidet ein kleiner Fehler im vorhergesagten Winkel darüber, auf welche Seite das Pendel fällt. Bei den Zufallsstartwerten 3 und 4 beträgt der Fehler schon nach einem Schritt mehrere Grad und nach fünf Schritten einige Dutzend Grad. Das Modell ist dort genau, wo seine Daten liegen, und liegt anderswo falsch. Auf dem See erschien ein nie erprobtes Paar dem durch Zählen gewonnenen Modell wertlos. Hier extrapoliert das Netz in den oberen Halbkreis, den es nie gesehen hat, und seine Extrapolation ist falsch. Ausgerechnet die Fehlversuche des Planers beheben das Problem. Selbst ein misslungener Versuch, das Pendel aufzuschwingen, bringt es höher als zufällige Drehmomente und liefert Daten darüber, was dort geschieht. Bei der nächsten Anpassung hat das Modell Daten aus diesen Zuständen, und der darauf aufbauende Plan funktioniert. Der Planer exploriert ohne ausdrückliche Explorationsregel: Er sucht die Zustände auf, für die das Modell Belohnung verspricht, und die neuen Daten korrigieren das Modell dort, wo es falsch lag.

<!--en:81281258-->
### Die Kosten eines realen Schritts

<!--en:3fe68179-->
Der Planer erreicht $-200$ erstmals nach 200 bis 800 realen Schritten. Legen wir denselben Score zugrunde, also den Mittelwert über die zehn Testanfangszustände, erreicht Soft Actor-Critic $-200$ bei einem Gradientenschritt pro Umgebungsschritt erstmals nach etwa 5000 bis 7500 realen Schritten. Bei zehn Gradientenschritten pro Umgebungsschritt braucht es etwa 1600 bis 4000. Diese Läufe verwenden den Code aus :numref:`sec_sac` mit gelernter Temperatur und fünf Zufallsstartwerten pro Einstellung. Die ersten tausend Schritte mit zufälligen Aktionen zählen mit. Gemessen in Umgebungsschritten braucht der Planer also etwa ein Zehntel der Schritte der ersten Variante. Auch die zweite Variante braucht ein Mehrfaches der Schritte des Planers.

<!--en:15ef39a8-->
Die zweite Messgröße ist der Rechenaufwand. Jede Aktion des Planers erfordert eine Suche mit 12 500 Modellauswertungen. Bei diesem kleinen Netz dauert das auf einem CPU-Kern einige Millisekunden. Pro Episode kommt eine erneute Anpassung mit tausend Adam-Schritten hinzu. Soft Actor-Critic macht pro realem Schritt einen oder zehn Gradientenschritte mit drei kleinen Netzen. Für die Wahl einer Aktion genügt ein Vorwärtsdurchlauf des Akteurs. Bei einem realen Roboter muss die Suche außerdem innerhalb der Abtastperiode abgeschlossen sein, hier also innerhalb von 0.05 Sekunden. Auch mit Modell hat das Lernen seinen Preis. Es verlagert die Kosten von der Erfahrung, die bei einem Roboter knapp ist, auf die Rechenleistung, die auf seinem Steuergerät knapp ist.

<!--en:4d849b71-->
## Strategieverbesserung durch Suche

<!--en:15c62343-->
### Der Plan des Akteurs als Ausgangspunkt der Suche

<!--en:017481e2-->
Der Planer hat keine Strategie. Jede Aktion erfordert eine vollständige Suche. Von ihren Ergebnissen bleibt nur erhalten, was im Modell und im verschobenen Plan des letzten Schritts steckt. Ein Strategienetz könnte mit einem Vorwärtsdurchlauf handeln und über Zustände hinweg generalisieren. Dazu braucht es aber Training. In :numref:`sec_sac` dauerte es Tausende Schritte, ein solches Netz mithilfe eines Kritikers zu trainieren. Kann die Suche stattdessen die Strategie trainieren?

<!--en:9613b242-->
Beginnen wir die Suche mit dem Plan der Strategie selbst. Der Akteur sei ein Netz, das einen Zustand auf ein Drehmoment $\bar a_\theta(s)$ abbildet. Im Modell besteht sein Plan vom Zustand $s$ aus in der Folge der Drehmomente, die er entlang seiner eigenen vorhergesagten Trajektorie wählt: zuerst $a_0 = \bar a_\theta(s)$, dann $\hat s_1$ aus $\hat f_\phi(s, a_0)$, dann $a_1 = \bar a_\theta(\hat s_1)$ und so weiter über $n$ Schritte. `actor_plan` berechnet diesen Plan für einen Batch von Zuständen. `improve` übergibt ihn an `plan` anstelle des verschobenen bisherigen Plans. Die Suche verwendet dieselben 500 Kandidaten und dieselbe Art von Störung und liefert das erste Drehmoment des besten Kandidaten.

<!--en:28d65407-->
Der eigene Plan des Akteurs gehört zu den Kandidaten. Deshalb liefert die Suche eine Aktion, deren Plan einen mindestens ebenso hohen vorhergesagten Ertrag hat wie der Plan des Akteurs. Im Modell und über $n$ Schritte schneidet die Suche mit dem Plan der Strategie als Ausgangspunkt nie schlechter ab als die Strategie selbst. Das entspricht dem Greedy-Schritt der Strategieiteration in :numref:`subsec_vi_policy_iteration`. Dort wählten wir für jeden Zustand eine Aktion, die nach den eigenen Werten der Strategie mindestens so gut war wie deren aktuelle Aktion. Hier entscheidet die Vorhersage des Modells für die nächsten 25 Schritte. Die Garantie gilt nur im Modell und nur über diesen Horizont. Ein fehlerhaftes Modell kann eine schlechtere Aktion für besser erklären, und ein Plan, der nach 25 Schritten endet, kann spätere Folgen nicht bewerten.

<!--en:65cc1a75-->
### Was der Akteur durch Destillation lernt

<!--en:3fce70ae-->
Findet die Suche bessere Aktionen als der Akteur, können wir ihn darauf trainieren, die Suche nachzuahmen. Das ist Verhaltensklonen (engl. behavior cloning) wie in :numref:`sec_imitation`, mit der Suche als Experten. Wir weisen den Zuständen mit `improve` Labels zu und passen das Drehmoment des Akteurs durch Regression an diese Labels an. Da das Drehmoment kontinuierlich ist, verwenden wir den quadratischen Fehler. Als Akteur dient `d2l.SquashedGaussianPolicy` aus :numref:`sec_sac`, ein Netz mit zwei Schichten. Es bildet den Mittelwert einer Gauß-Verteilung mit $c \tanh$ auf den zulässigen Drehmomentbereich ab. Wir trainieren nur den so begrenzten Mittelwert $\bar a_\theta(s) = c \tanh \mu_\theta(s)$, wobei $\mu_\theta$ wie dort den Mittelwert der Gauß-Verteilung bezeichnet. Diesen begrenzten Wert gibt `act_greedy` zurück. `make_actor` erzeugt den Akteur mit seinem Adam-Optimierer. `distill` führt für diese Regression 1000 Adam-Schritte aus.

<!--en:9f7a3491-->
`distill_step` ist ein kompilierter Adam-Schritt für die Regression. Einmal pro Aufruf von `distill` bindet `nnx.cached_partial` den Akteur und seinen Optimierer an diesen Schritt, wie zuvor das Modell an `model_step`.

<!--en:1da91b95-->
In jeder Runde passt `search_and_distill` (:numref:`fig_rl_search_distill`) das Modell erneut an und lässt die Suche 200 zufällig gezogenen gespeicherten Zuständen Labels zuweisen. Durch Destillation lernt der Akteur aus diesen Labels. Anschließend zeichnen wir eine reale Episode auf, in der die Suche die Aktionen wählt. Wir trainieren den Akteur nur mit den Labels der aktuellen Runde und verwerfen ältere Labels. Mit der Suche gelingt jede Episode so gut, wie es Modell und Akteur gerade erlauben. So erfassen die Daten auch die Zustände, die ein guter Regler besucht. Ergänzen wir diesen Ablauf um ein Netz, das auch den Wert jedes Zustands vorhersagt, eine Baumsuche über diskrete Züge und einen Puffer, der die Zielwerte der Suche über viele Spiele speichert, erhalten wir die Schleife von AlphaZero: mit der Suche handeln, Labels zuweisen und destillieren.

<!--en:1ac764e7-->
![Eine Runde aus Suche und Destillation. Wir passen das Modell erneut an alle gespeicherten Übergänge an. Ausgehend vom Plan des Akteurs weist die Suche 200 gespeicherten Zuständen Labels zu. Wir passen den Akteur durch Regression an diese Labels an. Eine reale Episode, in der die Suche die Aktionen wählt, liefert neue Übergänge für die nächste Runde.](../img/mdl-rl-search-distill.svg)

<!--en:38abf4f6-->
Die Schleife zeichnet pro Runde den Ertrag der mit der Suche gesammelten Episode und den Score des Akteurs auf, wenn er ganz ohne Suche handelt. Am Ende bewertet sie das Handeln mit Suche anhand der zehn Anfangszustände. Die beiden Kurven messen Verschiedenes. Die erste zeigt pro Runde eine Trainingsepisode mit deren jeweiligem Anfangszustand. Ihr Verlauf hängt daher ebenso stark vom Anfangszustand wie vom Agenten ab. Schon ein einzelner schwieriger Anfangszustand zeigt sich als Einbruch. Die zweite Kurve zeigt den Mittelwert über die zehn Testanfangszustände, also den Score, den wir sonst in diesem Abschnitt verwenden.

<!--en:a6e5b741-->
Die acht Runden beginnen mit einer Zufallsepisode. Der Akteur der Runde $r$ hat also $200 r$ reale Schritte gesehen. Wir führen drei Läufe mit verschiedenen Zufallsstartwerten aus und zeichnen den Median und die Spannweite über diese Läufe:

<!--en:76c85a93-->
### Suche und Destillation über mehrere Runden

<!--en:371312e0-->
Mit der Suche schwingt der Agent das Pendel in jedem der drei gezeigten Läufe innerhalb von zwei bis drei Runden auf. Nach acht Runden liegt sein Score über die zehn Anfangszustände zwischen etwa $-185$ und $-245$, in den meisten Läufen also nahe am Ergebnis des obigen Planers mit Initialisierung aus dem bisherigen Plan. Auch der Akteur allein verbessert sich in jedem Lauf. Nach der ersten Runde liegt sein Score zwischen etwa $-900$ und $-1{,}600$, nach der achten zwischen etwa $-190$ und $-1{,}100$. Von einer Runde zur nächsten kann sein Score um Hunderte Punkte fallen. Das Netz hat einen Teil dessen gelernt, was die Suche weiß. Handelt es allein, bleibt es hinter der Suche zurück.

<!--en:889cc4d8-->
Warum der Akteur zurückbleibt, zeigen diese Läufe nicht. Drei Ursachen kommen infrage. Die erste ist die Verteilungsverschiebung, die wir in :numref:`sec_imitation` auf dem See gemessen haben. Die Labels gehören zu Zuständen aus dem Speicher. Bis auf die Zufallsepisode stammen diese Zustände aus Läufen mit der Suche. Ahmt der Akteur die Suche nur unvollkommen nach, gerät er beim selbstständigen Handeln in Bereiche mit wenigen Labels. Dort pflanzen sich seine Fehler fort. DAgger (:numref:`subsec_il_dagger`) behob das Problem auf dem See, indem es den vom Lernenden selbst besuchten Zuständen Labels zuwies. Die zweite mögliche Ursache ist die Größe des Trainingsdatensatzes: In jeder Runde trainiert der Akteur mit 200 neuen Labels, die früheren werden verworfen. DAgger sammelt dagegen seine Labels über alle Runden, und AlphaZero behält die Zielwerte der Suche in einem Puffer. Die dritte mögliche Ursache sind die Labels selbst. Jedes ist das erste Drehmoment des besten von 500 Zufallskandidaten und damit ein verrauschter Zielwert. In :exref:`ex_model-based-dagger` untersuchen Sie die ersten beiden Ursachen getrennt. Auch AlphaZero sucht zur Testzeit. Sein Netz muss daher nur einen guten Ausgangspunkt für die Suche liefern. Suche und Netz verbessern sich dann gegenseitig: Ein besserer Akteur liefert den Ausgangspunkt für eine bessere Suche, und aus der besseren Suche lernt der Akteur.

<!--en:ba12708c-->
### Simulierte Übergänge als Daten

<!--en:8fc44cdd-->
Ein Modell kann auch ein Lernverfahren ohne Planer mit Daten versorgen. Dyna :cite:`Sutton.1990` wechselt zwischen realen Schritten und Aktualisierungen mit Übergängen, die es aus dem Modell zieht. Model-Based Policy Optimization (MBPO) :cite:`Janner.Fu.Zhang.ea.2019` überträgt das auf Soft Actor-Critic. Alle paar Hundert Schritte passt MBPO das Modell neu an. Von Zuständen im Replay-Puffer aus simuliert es den aktuellen Akteur für einige Schritte im Modell und füllt jeden Trainingsbatch größtenteils mit diesen simulierten Übergängen (:numref:`fig_rl_branched_rollouts`). Die Menge der simulierten Daten ist nur durch den Rechenaufwand begrenzt. Allerdings gehen die Modellfehler in die Zielwerte des Kritikers ein. Mit kurzen simulierten Trajektorien begrenzt MBPO die oben gemessene Fehlerfortpflanzung.

<!--en:433be11a-->
![Verzweigte simulierte Trajektorien, schematisch dargestellt. Kurze simulierte Trajektorien (orange, gestrichelt) beginnen bei realen Zuständen aus dem Replay-Puffer (blau) und umfassen hier je drei Modellschritte. Der Modellfehler, dargestellt als schattiertes Band, pflanzt sich daher nur über wenige Schritte fort. Eine lange simulierte Trajektorie vom ersten Zustand aus (grau, gepunktet) entfernt sich von den Daten.](../img/mdl-rl-branched-rollouts.svg)

<!--en:67e3c977-->
Brauchen wir damit in Pendulum weniger reale Schritte? Wir führten Soft Actor-Critic mit einer festen Temperatur von 0.2 und zehn Gradientenschritten pro Umgebungsschritt aus. Jeder Batch bestand zu 5% aus realen und zu 95% aus simulierten Übergängen aus Trajektorien von je drei Schritten. Alle 250 Schritte passten wir das Modell neu an. Bei fünf Zufallsstartwerten erreichte der Greedy-Akteur erstmals nach 1500 bis 3000 Schritten einen Score von $-200$ oder besser. Ohne Modell brauchte er bei denselben zehn Aktualisierungen pro Schritt 1500 bis 2500 Schritte. Diese Läufe verwendeten zehn andere feste Anfangszustände, bei denen ein guter Regler weniger Punkte verliert als bei unseren. Deshalb erreichen sie $-200$ früher als die oben genannten Läufe. Bei gleicher Anzahl von Aktualisierungen pro Schritt sparten die simulierten Übergänge keine realen Schritte ein. Ein großer Teil des Vorteils von MBPO gegenüber Soft Actor-Critic mit einer Aktualisierung pro Schritt kann auf die zusätzlichen Aktualisierungen pro realem Schritt zurückgehen. Durch Wiederverwendung gespeicherter Übergänge sind diese auch ohne Modell möglich. Ein modellfreies Lernverfahren erreichte dieselbe Stichprobeneffizienz wie MBPO: Es verwendete viele Aktualisierungen pro Schritt und ein Ensemble von Kritikern, um die dadurch verursachte Überschätzung zu begrenzen :cite:`Chen.Wang.Zhou.ea.2021`. In :exref:`ex_model-based-dyna` wiederholen Sie den Vergleich. Am meisten hilft ein Modell dort, wo wir es in diesem Abschnitt eingesetzt haben: beim Vorausschauen vor einer Aktion und beim Überführen der Suche in eine Strategie. Ensembles von Modellen, die ihre Unsicherheit erkennen, Modelle eines gelernten latenten Zustands und Suchverfahren, die am Ende einen gelernten Wert verwenden, führen diese Ideen weiter. :numref:`sec_deep_rl-further-reading` beschreibt sie.

<!--en:771073f7-->
Wir können nun aus einigen Hundert Schritten ein Modell der Dynamik lernen, darin planen und einer Strategie beibringen, was die Suche findet. Bei jedem Schritt vertraute der Agent dem Modell nur dort, wo seine eigenen Episoden ihn hingeführt hatten, und nur für 25 Schritte. Indem er handelte und das Modell erneut anpasste, konnte er dieses Vertrauen überprüfen. In :numref:`sec_offline` entfällt das Handeln: Die Daten bestehen aus einem festen Protokoll. Was ein Modell oder eine Wertfunktion darüber hinaus verspricht, lässt sich nicht überprüfen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ff792294-->
Ein Übergang hält fest, wie sich die Umgebung verändert. Passen wir ein Netz mit der Methode der kleinsten Quadrate an Übergänge an, $\hat f_\phi(s, a) \to (\hat s', \hat r)$, so übertragen wir das durch Zählen gewonnene Modell des Sees auf kontinuierliche Zustände. Seine Fehler nach einem Schritt sind klein, pflanzen sich aber entlang einer simulierten Trajektorie fort. In Bereichen, die die Daten nie erfasst haben, sind sie groß. Die modellprädiktive Regelung sucht im Modell nach Drehmomentfolgen, führt das erste Drehmoment aus und plant im nächsten Schritt neu. Mit dem bisherigen Plan als Ausgangspunkt schwingt ein Planer aus ein bis drei Episoden das Pendel nach wenigen Hundert realen Schritten auf. Soft Actor-Critic brauchte dafür Tausende. Dafür benötigt der Planer für jede Aktion Tausende Modellauswertungen. Beginnt die Suche mit dem Plan einer Strategie, ist der vorhergesagte Ertrag des besten Plans mindestens so hoch wie der der Strategie. In diesem Sinn verbessert die Suche die Strategie. Passen wir den Akteur durch Regression an die Aktionen der Suche an, übernimmt er einen Teil dieser Verbesserung. Handelt er allein, bleibt er hinter der Suche zurück. Seine Labels stammen hauptsächlich aus den von der Suche besuchten Zuständen, es gibt nur wenige pro Runde, und sie sind verrauscht. Welche dieser drei Ursachen am stärksten ins Gewicht fällt, zeigen die Läufe nicht. Bei gleicher Anzahl von Aktualisierungen pro Schritt verringerte das Training mit simulierten Übergängen die Zahl der realen Schritte bei Soft Actor-Critic nicht.

<!--en:2294243a-->
## Übungen

<!--en:da6d42b2-->
1. [code] **Tiefe des Plans.** Der Planer schaut $n = 25$ Schritte voraus.
    1. Sagen Sie vor dem ersten Lauf anhand der beiden Fehlertabellen dieses Abschnitts vorher, wie sich der Testscore bei $n = 10$ und bei $n = 50$ verändert. Behalten Sie die 500 Kandidaten bei.
    1. Führen Sie `mpc_run` mit beiden Tiefen für drei Zufallsstartwerte aus. Vergleichen Sie die Scores nach 600 realen Schritten mit denen für $n = 25$.
    1. Das Aufschwingen aus der unteren Ruhelage erfordert einige Dutzend Schritte gezielter Energiezufuhr. Erklären Sie das Ergebnis für $n = 10$ anhand dessen, welche Pläne sich durch ihre vorhergesagten Erträge unterscheiden lassen und welche nicht. Erklären Sie das Ergebnis für $n = 50$ anhand der Fehlertabellen. Wie ändern sich die Kosten einer Aktion mit $n$?
   :exlabel:`ex_model-based-horizon`

<!--en:03aeb26d-->
1. [code] **Wie viele Zufallsdaten braucht der Planer?** Bei den Zufallsstartwerten 3 und 4 erfasste das Modell den oberen Halbkreis erst nach Fehlversuchen des Planers selbst.
    1. Passen Sie für beide Zufallsstartwerte das Modell an eine, zwei und fünf Zufallsepisoden an. Geben Sie wie in der Tabelle zur Abdeckung den höchsten Punkt an, den das Pendel in den Daten erreicht hat (den größten Wert von $\cos\vartheta$). Geben Sie außerdem für jedes Modell den Fehler nach 25 Schritten entlang der Episoden des Planers an.
    1. Wie viele Zufallsepisoden braucht ein Planer, der nur auf Zufallsdaten beruht, um das Pendel von den meisten Testanfangszuständen aus aufzuschwingen? Vergleichen Sie das Ergebnis mit den 600 Schritten, die mit den eigenen Episoden des Planers nötig waren, und erklären Sie den Unterschied.
   :exlabel:`ex_model-based-data`

<!--en:c333293f-->
1. [code] **Die Kreuzentropiemethode.** Die *Kreuzentropiemethode* :cite:`Rubinstein.Kroese.2004` wiederholt Random Shooting für jede Aktion mehrmals und passt die Verteilung, aus der sie zieht, erneut an die besten Kandidaten an.
    1. Ändern Sie `plan` so, dass die Funktion fünf Runden durchläuft. Setzen Sie nach jeder Runde für jedes Drehmoment des Plans den Mittelwert und die Standardabweichung auf die entsprechenden Werte der besten 50 von 500 Kandidaten. Ziehen Sie die nächsten 500 Kandidaten aus dieser Gauß-Verteilung.
    1. Vergleichen Sie die Testscores nach 600 realen Schritten und die Anzahl der Modellauswertungen pro Aktion mit den Ergebnissen für Random Shooting mit Initialisierung aus dem bisherigen Plan in diesem Abschnitt.
    1. Führen Sie das Verfahren ohne diese Initialisierung aus. Beginnen Sie jeden Schritt mit einem mittleren Drehmoment von null und einer Standardabweichung von $c$. Ersetzt die Verfeinerung innerhalb eines Schritts die Verfeinerung über mehrere Schritte hinweg?
   :exlabel:`ex_model-based-cem`

<!--en:8f002f2a-->
1. [code] **Abweichungen zwischen Modellen als Maß für ihre Verlässlichkeit.** ● Ein Ensemble von Modellen kann anzeigen, wo sein Wissen nicht ausreicht :cite:`Chua.Calandra.McAllister.ea.2018`.
    1. Passen Sie für die Zufallsstartwerte 3 und 4 jeweils drei Modelle mit unterschiedlichen Zufallsstartwerten an die Zufallsepisode an. Vergleichen Sie entlang der Episoden des Planers die Abweichungen zwischen den drei Modellen (die Standardabweichung der vorhergesagten Winkel) mit dem Fehler ihrer mittleren Vorhersage nach 1, 5 und 25 Schritten. Kündigt eine große Abweichung einen großen Fehler an?
    1. Bewerten Sie jeden Kandidaten in `plan` anhand des mittleren vorhergesagten Ertrags der drei Modelle. Ziehen Sie davon das $\kappa$-Fache der über seine Schritte summierten Abweichung ab. Verwenden Sie $\kappa \in \{0, 1, 10\}$. Wie verändert sich die erste MPC-Episode bei den Zufallsstartwerten 3 und 4?
    1. Wiederholen Sie Teil b mit $-\kappa$, also mit einem Bonus für Abweichungen. Welches Vorzeichen passt zu einem Agenten, der noch Daten sammeln kann, und welches zu einem Agenten, der das nicht mehr kann?
   :exlabel:`ex_model-based-ensemble`

<!--en:33616801-->
1. [code] **Labels für die eigenen Zustände des Akteurs.** ● Die Schleife in diesem Abschnitt sammelt Episoden, in denen die Suche die Aktionen bestimmt. Pro Runde trainiert sie den Akteur nur mit den 200 Labels dieser Runde.
    1. Sammeln Sie weiterhin Daten mit der Suche, führen Sie aber die Labels zusammen. Fügen Sie die Zustände und zugehörigen Drehmomente jeder Runde zu einem wachsenden Datensatz hinzu und destillieren Sie den Akteur mit dem gesamten Datensatz. Führen Sie fünf Läufe mit verschiedenen Zufallsstartwerten aus und vergleichen Sie die Scores des Akteurs pro Runde mit denen in diesem Abschnitt.
    1. Sammeln Sie nun die Episode jeder Runde allein mit dem Akteur, also mit `actor.act_greedy`. Weisen Sie Zuständen aus seinen Episoden mithilfe von `improve` Labels zu und führen Sie diese wie in Teil (a) zusammen. Das ist DAgger aus :numref:`sec_imitation` mit der Suche als Experten. Welche der drei Varianten liefert den Akteur mit den beständigsten Ergebnissen? Was sagt der Vergleich über die im Text genannten Ursachen aus?
    1. Bei manchen Zufallsstartwerten gelingt der DAgger-Variante womöglich nie das Aufschwingen. Untersuchen Sie die Daten eines solchen Laufs und erklären Sie den Fehlschlag anhand der Tabelle zur Abdeckung in diesem Abschnitt. Wie ließen sich die beiden Arten der Datenerhebung verbinden?
   :exlabel:`ex_model-based-dagger`

<!--en:f1707e36-->
1. [extended] **Simulierte Übergänge für Soft Actor-Critic.** ●● Model-Based Policy Optimization :cite:`Janner.Fu.Zhang.ea.2019` trainiert Soft Actor-Critic mit Übergängen, die ein Modell simuliert.
    1. Kopieren Sie `train_sac` aus :numref:`sec_sac` in Ihr Notebook. Passen Sie nach der Anfangsphase alle 250 Umgebungsschritte ein `DynamicsModel` erneut an die Daten im Replay-Puffer an. Ziehen Sie 2000 Zustände aus dem Puffer und simulieren Sie von jedem aus den aktuellen Akteur für drei Schritte im Modell. Speichern Sie die simulierten Übergänge der letzten vier Modellanpassungen in einem zweiten `d2l.ReplayBufferC` und ziehen Sie 95% jedes Batchs daraus.
    1. Führen Sie für fünf Zufallsstartwerte je 4000 Schritte mit zehn Aktualisierungen pro Umgebungsschritt aus. Bewerten Sie den Greedy-Akteur alle 250 Schritte anhand der zehn Testanfangszustände. Führen Sie zum Vergleich dieselbe Schleife ohne Modell mit denselben Zufallsstartwerten aus. Geben Sie für jeden Lauf den ersten Schritt an, bei dem er $-200$ oder mehr Punkte erreicht.
    1. Warum muss der Kontrolllauf dieselbe Anzahl von Aktualisierungen pro Umgebungsschritt verwenden? Wiederholen Sie beide Varianten mit einer Aktualisierung pro Schritt und deuten Sie den Unterschied.
    1. Ändern Sie die Länge der simulierten Trajektorien auf einen Schritt und auf zehn Schritte. Welches Ergebnis erwarten Sie anhand der Fehlertabelle dieses Abschnitts, und was beobachten Sie?
   :exlabel:`ex_model-based-dyna`

<!--en:499cb71b-->
1. [code] **Vorausschau mit einer Wertfunktion.** Der Planer dieses Abschnitts summiert Belohnungen über einen festen Horizont. Was danach kommt, lässt er außer Acht. Mit einer Wertfunktion kann der Horizont einen einzigen Schritt betragen.
    1. Passen Sie für CartPole ein `DynamicsModel` mit einem vierdimensionalen Zustand an 5000 Übergänge mit zufällig gewählten Aktionen an. Bestimmen Sie anhand des Stangenwinkels und der Wagenposition nach der bekannten Regel von CartPole, wann die Episode endet.
    1. Trainieren Sie das DQN aus :numref:`sec_dqn` für 10 000 Schritte und verwenden Sie zwei Varianten der Greedy-Aktionswahl: mit $\operatorname{argmax}_a Q(s, a)$ und mit der Einschrittvorausschau $\operatorname{argmax}_a \big[\hat r(s, a) + \gamma \max_{a'} Q(\hat s'(s, a), a')\big]$. Vergleichen Sie die Erträge über 20 Episoden für fünf Zufallsstartwerte.
    1. Wiederholen Sie den Vergleich nach 40 000 Trainingsschritten. Wann hilft die Vorausschau, und wann wiegen die Modellfehler ihren Nutzen auf?
   :exlabel:`ex_model-based-cartpole`

<!--en:1353574c-->
[Diskussion](https://d2l.discourse.group/t/36872)

<!--en:b5ae5fe4-->
Modellbasiertes tiefes bestärkendes Lernen: Gelernte Weltmodelle und Planung<br>**Regression der Dynamik · Fehlerfortpflanzung in simulierten Trajektorien · MPC mit Random Shooting · Initialisierung mit dem bisherigen Plan · Strategiedestillation · Dyna und MBPO**

<!--en:8c9c520e-->
::: {.slide title="Was ein Übergang über einen skalaren Wert hinaus verrät"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:28787bf3-->
- **Mehr Information pro Lernsignal:** Modellfreies SAC gewinnt aus jedem Schritt nur einen skalaren Zielwert $y_t \in \mathbb{R}$ für den Kritiker. Ein Weltmodell lernt durch Regression den gesamten vierdimensionalen Vektor der physikalischen Größen $(s_{t+1} - s_t, r_t) \in \mathbb{R}^4$.

<!--en:80dde352-->
- **Von Tausenden Schritten zu Hunderten:** Planung in einem gelernten Dynamikmodell $\hat{f}_\phi$ schwingt das Pendel schon nach **1 bis 3 Episoden ($200$–$600$ realen Schritten)** auf.

<!--en:6769363b-->
::: {.slide title="Anpassung eines neuronalen Dynamikmodells mit kleinsten Quadraten"}

<!--en:ddf4a183-->
[Regression für das Weltmodell]{.kicker}

<!--en:c095ae31-->
- **1. Zustandsänderungen ($s' - s$) vorhersagen:** In $\Delta t = 0.05\text{ s}$ gilt $s' \approx s$. Sagt das Netz das Residuum $\Delta s = s' - s$ mit einer Skip-Verbindung $\hat{s}' = s + \Delta\hat{s}$ vorher, nutzt es seine Kapazität für die Änderung.

<!--en:5af98b90-->
- **2. Koordinaten standardisieren:** Standardisieren wir jede Eingabe- und Zieldimension, dominiert die Belohnung $r \in [-16.2, 0]$ trotz ihrer hohen Varianz nicht den $L_2$-Verlust.

<!--en:333f94f6-->
::: {.slide title="Architektur und Standardisierung von DynamicsModel" only="pytorch"}

<!--en:2faeb046-->
[Implementierung des Modells]{.kicker}

<!--en:94fac2ce-->
`DynamicsModel` verwendet ein MLP mit 2 versteckten Schichten zu je $64$ Einheiten mit glatter `SiLU`-Aktivierung. Die Kenngrößen des Datensatzes für die Standardisierung $(\mu_x, \sigma_x, \mu_y, \sigma_y)$ liegen direkt im Modul:

<!--en:67352da1-->
::: {.slide title="Architektur und Standardisierung von DynamicsModel (Fortsetzung)" only="pytorch"}

<!--en:2faeb046-2-->
[Implementierung des Modells]{.kicker}

<!--en:fddc0747-->
`forward` rechnet die Netzausgabe in die Einheiten der Daten zurück. `make_model` verbindet das Modell mit seinem Adam-Optimierer:

<!--en:6755f4f1-->
::: {.slide title="Architektur und Standardisierung von DynamicsModel (Anpassung)" only="pytorch"}

<!--en:2faeb046-3-->
[Implementierung des Modells]{.kicker}

<!--en:97c9eb9a-->
`fit_model` kopiert die Kenngrößen der Daten in das Modul und passt dann das Netz durch Regression an standardisierte Eingaben und Zielwerte an:

<!--en:1acc68c7-->
::: {.slide title="Architektur und Standardisierung von DynamicsModel" only="jax"}

<!--en:2faeb046-4-->
[Implementierung des Modells]{.kicker}

<!--en:94fac2ce-2-->
`DynamicsModel` verwendet ein MLP mit 2 versteckten Schichten zu je $64$ Einheiten mit glatter `SiLU`-Aktivierung. Die Kenngrößen des Datensatzes für die Standardisierung $(\mu_x, \sigma_x, \mu_y, \sigma_y)$ liegen direkt im Modul:

<!--en:80e846b1-->
::: {.slide title="Architektur und Standardisierung von DynamicsModel (Fortsetzung)" only="jax"}

<!--en:2faeb046-5-->
[Implementierung des Modells]{.kicker}

<!--en:683e3592-->
`__call__` rechnet die Netzausgabe in die Einheiten der Daten zurück. `make_model` und die kompilierte Funktion `model_step` richten das Training ein:

<!--en:134ad584-->
::: {.slide title="Architektur und Standardisierung von DynamicsModel (Anpassung)" only="jax"}

<!--en:2faeb046-6-->
[Implementierung des Modells]{.kicker}

<!--en:97c9eb9a-2-->
`fit_model` kopiert die Kenngrößen der Daten in das Modul und passt dann das Netz durch Regression an standardisierte Eingaben und Zielwerte an:

<!--en:cc82c449-->
::: {.slide title="Genauigkeit der Einschrittvorhersage in einer Zufallsepisode"}

<!--en:a61e0fed-->
[Einen einzelnen Schritt überprüfen]{.kicker}

<!--en:33cd0acb-->
Wir passen `DynamicsModel` mit $1{,}000$ Adam-Schritten an **eine einzige Zufallsepisode ($200$ Übergänge)** an und betrachten den Übergang $i = 100$ in physikalischen Einheiten:

<!--en:4871c124-->
Bei Schritt 100 ($\vartheta = -160.2^\circ$, $\dot\vartheta = 5.69\text{ rad/s}$) dreht sich das Pendel um $15.6^\circ$ in $0.05\text{ s}$. Der Fehler des gelernten Modells liegt beim nächsten Winkel unter **$0.4^\circ$** ($0.30^\circ$–$0.39^\circ$) und bei der Belohnung ($-11.05$) unter $0.25$.

<!--en:731181f5-->
::: {.slide title="Autoregressive Simulation und Fehlerfortpflanzung im Modell" layout="tight"}

<!--en:28e5bcb8-->
[Der Horizont als Engpass]{.kicker}

<!--en:83b7e4fc-->
Um eine mögliche Folge von $n = 25$ Drehmomenten zu bewerten, simuliert das Modell **autoregressiv** (ohne Rückführung). Jeder vorhergesagte Zustand $\hat{s}_{t+1}$ dient als Eingabe für Schritt $t+1$:

<!--en:fa8b13f1-->
::: {.slide title="Autoregressive Simulation und Fehlerfortpflanzung im Modell (Fortsetzung)" layout="tight"}

<!--en:28e5bcb8-2-->
[Der Horizont als Engpass]{.kicker}

<!--en:2312bad6-->
- **Warum Fehler anwachsen:** Das Modell hat nur mit tatsächlichen Zuständen trainiert, nie mit den eigenen leicht fehlerhaften Vorhersagen. Ein kleiner Winkelfehler verändert das vorhergesagte Drehmoment der Schwerkraft und damit die Geschwindigkeit. Über $k = 10 \to 25$ Schritte wächst der Fehler so weiter.

<!--en:e04077ab-->
::: {.slide title="Random Shooting: Rangfolge der Kandidaten im Modell"}

<!--en:e9bc7091-->
[Planung durch Suche]{.kicker}

<!--en:a681c76d-->
Im Zustand $s$ zieht **Random Shooting** insgesamt $J$ mögliche Aktionsfolgen $A^{(j)} = (a_0^{(j)}, \dots, a_{n-1}^{(j)})$. Es simuliert alle $J$ Folgen parallel in $\hat{f}_\phi$ und bewertet jede anhand ihres vorhergesagten $n$-Schritt-Ertrags $\hat{G}^{(j)} = \sum_{t=0}^{n-1} \hat{r}_t^{(j)}$:

<!--en:012c49bc-->
**Rangfolge und Kalibrierung:** Beginnt das Pendel unten in Ruhe ($\vartheta = \pi$, Ertrag bei 25 Schritten in Ruhe: $-247$), überschätzt das Modell jede Folge um $40$–$70$ Punkte. In der zufälligen Trainingsepisode durchlief das Pendel den tiefsten Punkt schnell. Dennoch **setzt das Modell die tatsächlich beste Folge auf Platz eins**.

<!--en:8c83d980-->
::: {.slide title="MPC mit gleitendem Horizont und Initialisierung mit dem verschobenen Plan"}

<!--en:18a88977-->
[Modellprädiktive Regelung]{.kicker}

<!--en:76c88157-->
- **Regelung mit gleitendem Horizont (MPC, §{sec_optimal_control}):** Wie bei der analytischen MPC aus §{sec_optimal_control} führen wir *nur das erste Drehmoment* $a_0^\star$ des besten 25-Schritt-Plans aus. Wir beobachten den realen Folgezustand $s_{t+1}$ und planen von dort aus neu. So häufen sich Modellfehler ohne Rückführung auf der realen Trajektorie nicht an.

<!--en:1d047627-->
- **Suche mit dem bisherigen Plan:** Statt in Schritt $t+1$ 500 neue Zufallsfolgen zu ziehen, rücken wir den besten Plan aus Schritt $t$ um 1 Schritt vor, hängen $0$ an und variieren ihn mit gaußschem Rauschen ($\sigma = c = 2$). Der unveränderte verschobene Plan bleibt Kandidat $0$.

<!--en:78236320-->
::: {.slide title="Implementierung von vektorisiertem Random Shooting"}

<!--en:20522ea4-->
[Implementierung der Suche]{.kicker}

<!--en:31af908f-->
Für einen Batch von $B$ Zuständen erzeugt `plan` aus einer anfänglichen Aktionsfolge über $n$ Schritte durch Störung `num_cand = 500` Kandidaten (die unveränderte Folge hat Index $0$). Die Funktion begrenzt Drehmomente auf $[-c, c]$ und wertet alle $B \times 500 \times 25$ Schritte mit einem einzigen Batch-Aufruf von `rollout_model` aus:

<!--en:8194af88-->
::: {.slide title="Implementierung der MPC mit gleitendem Horizont und bisherigem Plan"}

<!--en:1efc0615-->
[MPC-Regler]{.kicker}

<!--en:6b17dfb6-->
`mpc` speichert den besten Plan des letzten Schritts in einem Funktionsabschluss, rückt ihn um einen Schritt vor (`plans[:, 1:]` gefolgt von einem Drehmoment von null) und wendet nur das erste Drehmoment `plans[:, 0]` des neu verfeinerten Plans an:

<!--en:a96316e4-->
Über 5 Zufallsstartwerte verbessert die Initialisierung mit dem bisherigen Plan nach 600 realen Schritten die Testerträge für 10 Anfangszustände von $[-555, -165]$ (neues Random Shooting bei jedem Schritt) auf **$[-155, -115]$**.

<!--en:3fd1ee39-->
::: {.slide title="Aufschwingen in Pendulum mit 200 bis 600 Umgebungsschritten"}

<!--en:d4dc47ef-->
[Aufschwingen nach wenigen Episoden]{.kicker}

<!--en:2216ffb8-->
Nach 1 Zufallsepisode (200 Schritte) passt `mpc_run` das Modell wiederholt an alle gesammelten Übergänge an, bewertet den MPC-Planer anhand von 10 festen Anfangszuständen und sammelt mit MPC jeweils 1 neue Episode:

<!--en:bef7598e-->
- **Benchmark mit 10 Anfangszuständen (Obergrenze $\approx -170$):** Nach **600 realen Schritten (3 Episoden)** liegt der Score bei jedem Zufallsstartwert zwischen $-165$ und $-215$. Bei 3 der 5 Zufallsstartwerte gelingt das Aufschwingen schon nach **einer einzigen Zufallsepisode (200 Schritte)**.

<!--en:4cd58738-->
::: {.slide title="Extrapolation außerhalb des Datenbereichs und Selbstkorrektur"}

<!--en:969f6682-->
[Analyse der Abdeckung]{.kicker}

<!--en:95c572e4-->
Warum lag der Score bei den Zufallsstartwerten 3 und 4 nach der ersten Zufallsepisode (200 Schritte) bei $\sim -1{,}500$, während das Aufschwingen nach 600 Schritten gelang? Die Ursache zeigt sich, wenn wir das Modell aus den ersten 200 Schritten anhand der *späteren* Aufschwungtrajektorien des Planers auswerten:

<!--en:62400eca-->
**Implizite Exploration durch optimistische Modellvorhersagen:** Bei den Zufallsstartwerten 3 und 4 kam das Pendel in der ersten Zufallsepisode nie über die Horizontale. Im oberen Halbkreis lag das Modell aus den ersten 200 Schritten daher **innerhalb von 5 Schritten um $40^\circ$–$72^\circ$ daneben**. Selbst ein misslungener MPC-Aufschwung führt das Pendel in diesen zuvor unbesuchten Winkelbereich und liefert Daten zur tatsächlichen Bewegung. Damit gelingt die nächste Anpassung.

<!--en:634ba532-->
::: {.slide title="Erfahrung und Rechenaufwand bei MPC-Planer und SAC"}

<!--en:cb9ed94c-->
[Abwägung beim Systementwurf]{.kicker}

<!--en:225829ee-->
- **MPC-Planer mit Initialisierung aus dem bisherigen Plan:** Löst Pendulum in **$200$–$800$ realen Schritten**, benötigt zur Testzeit aber $500 \times 25 = 12{,}500$ Modellauswertungen ($\sim\text{ms}$) pro Regelschritt.

<!--en:b5474f30-->
- **Modellfreies SAC (`UTD = 1` bis `10`):** Benötigt **$1{,}600$–$7{,}500$ reale Schritte**, kommt aber zur Testzeit mit einem Vorwärtsdurchlauf des Akteurs ($\sim\mu\text{s}$) aus, weil die Strategiegewichte $\theta$ die Planung vorwegnehmen.

<!--en:74390922-->
::: {.slide title="Suche als Operator zur Strategieverbesserung"}

<!--en:869634f4-->
[Suche mit dem Plan des Akteurs als Ausgangspunkt]{.kicker}

<!--en:2cb82625-->
- **Monotone Verbesserung in $\hat{f}_\phi$:** Kandidat $0$ von `plan` ist die unveränderte simulierte Trajektorie des Akteurs. Deshalb erreicht der beste Kandidat mindestens den vorhergesagten Ertrag des Akteurplans.

<!--en:c1e414c0-->
::: {.slide title="Destillation der Suchergebnisse in ein Akteurnetz" layout="tight"}

<!--en:383a7832-->
[Strategiedestillation]{.kicker}

<!--en:904e23d8-->
- **Suchaufwand mit $\pi_\theta$ amortisieren:** Wir passen den deterministischen Mittelwert $c\tanh\mu_\theta(s)$ des Akteurs an die von `improve(model, actor, obs)` unter 500 Kandidaten gewählten Aktionen an. Dazu minimieren wir den $L_2$-Verlust (`distill`).

<!--en:4abed4e7-->
::: {.slide title="Destillation der Suchergebnisse in ein Akteurnetz (Code)" layout="tight"}

<!--en:383a7832-2-->
[Strategiedestillation]{.kicker}

<!--en:9c32c6e1-->
::: {.slide title="Wiederholte Suche und Destillation (AlphaZero-Schleife)"}

<!--en:c3adfeb8-->
[Suche und Destillation]{.kicker}

<!--en:6f2edf6c-->
- Jede Runde von `search_and_distill` passt $\hat{f}_\phi$ erneut an. Die vom Akteur initialisierte Suche `improve` weist 200 Pufferzuständen Labels zu. Dann passen wir $\bar{a}_\theta(s)$ an diese Labels an (`distill`) und sammeln mit der Suche 1 Episode.

<!--en:1db86e1f-->
- **Handeln mit Suche (blau):** Das Aufschwingen gelingt zuverlässig innerhalb von $2$–$3$ Runden ($400$–$600$ Schritte).

<!--en:51ccdfcb-->
- **Akteur allein (orange):** Sein Score verbessert sich von $-1{,}400$ in Richtung $-200$. Wegen der Kovariatenverschiebung (Imitation ohne Zusammenführen der Labels wie bei DAgger) und der verrauschten Labels aus dem Random Shooting mit 500 Kandidaten bleibt er hinter der Suche zur Testzeit zurück.

<!--en:b3118d92-->
::: {.slide title="Dyna und MBPO: Simulierte Übergänge oder viele Aktualisierungen pro Schritt?"}

<!--en:616b1f09-->
[Vom Modell erzeugte Daten]{.kicker}

<!--en:f5ba1b7a-->
- **Kurze verzweigte Trajektorien (Janner et al., 2019):** Trajektorien von $k = 3$ Schritten zweigen im Modell $\hat{f}_\phi$ von realen Pufferzuständen ab. So bleiben die synthetischen Daten nahe an der tatsächlichen Zustandsverteilung und driften nicht über einen langen Horizont ab.

<!--en:df39a631-->
- **Vergleich bei gleicher Aktualisierungsrate (eigene Läufe):** Bei `UTD = 10` brauchten Batches mit $95\%$ simulierten Übergängen $1{,}500$–$3{,}000$ reale Schritte bis zum Ertrag von $-200$. Mit ausschließlich realen Übergängen aus dem Puffer waren es $1{,}500$–$2{,}500$. Das Modell sparte keine realen Schritte ein. Ein modellfreies Verfahren mit einem Ensemble von Kritikern und vielen Updates pro Schritt (REDQ, Chen et al., 2021) erreichte dieselbe Stichprobeneffizienz wie MBPO.

<!--en:c0964c46-->
::: {.slide title="Verständnisfragen zu Horizont, Zustandsänderungen und Ensembles"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:d319f8f5-->
Begründen Sie Ihre Antworten auf diese drei Fragen zum Entwurf modellbasierter Verfahren:

<!--en:5310b0cb-->
1. **Abwägung beim Planungshorizont:** Welche Probleme entstehen bei Pendulum ($\Delta t = 0.05\text{ s}$), wenn Sie den Horizont für Random Shooting von $n = 25$ auf $n = 5$ verkürzen? Was passiert, wenn Sie ihn auf $n = 100$ verlängern?

<!--en:aff39415-->
2. **Warum Zustandsänderungen vorhersagen?** Warum trainieren wir das MLP darauf, standardisierte Residuen $(s' - s)/\sigma_{\Delta s}$ vorherzusagen, statt $s'$ direkt in den ursprünglichen Koordinaten vorherzusagen?

<!--en:8fb40609-->
3. **Modellensembles:** Sie trainieren ein Ensemble von $M = 5$ Dynamiknetzen $\hat{f}_{\phi_1}, \dots, \hat{f}_{\phi_5}$ mit verschiedenen zufälligen Initialisierungen an der einzigen Zufallsepisode mit Startwert 4. Wo weichen ihre Vorhersagen am stärksten voneinander ab?

<!--en:557b98b6-->
**Antworten:**

<!--en:8e1a8e21-->
1. Bei **$n = 5$ ($0.25\text{ s}$)** reicht der Horizont nicht aus, um zu erkennen, dass sich die Energiezufuhr am tiefsten Punkt $20$ Schritte später nahe der aufrechten Lage auszahlt. Bei **$n = 100$ ($5.0\text{ s}$)** pflanzen sich die Fehler ohne Rückführung so weit fort, dass die für späte Schritte vorhergesagten Belohnungen nur noch Rauschen sind.

<!--en:d84aa0c0-->
2. Bei $\Delta t = 0.05\text{ s}$ gilt $s' \approx s$. Bei direkter Vorhersage nutzt das Netz daher den Großteil seiner Kapazität, um die Identitätsabbildung $s' = s$ zu lernen. Ohne Standardisierung dominiert im mittleren quadratischen Fehler die Belohnung $r \in [-16, 0]$ ($\text{var} \sim 20$) gegenüber $\Delta\cos\vartheta$ ($\text{var} \sim 0.01$).

<!--en:2b0b516e-->
3. Im unteren Halbkreis, wo die Trainingsdaten liegen, stimmen die 5 Netze gut überein. **Im unbesuchten oberen Halbkreis weichen ihre Vorhersagen stark voneinander ab.** Die Varianz im Ensemble ist deshalb ein guter Indikator für epistemische Unsicherheit (PETS, Chua et al., 2018).

<!--en:96b6f310-->
::: {.slide title="Zusammenfassung: Modellbasiertes tiefes bestärkendes Lernen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:19c02b26-->
| Methode / Komponente | Mathematisches Prinzip | Wichtigster Vorteil | Wichtigste Einschränkung |

<!--en:17cd8704-->
| **MLP für Zustandsänderungen ($\hat{f}_\phi$)** | $\hat{s}' = s + \sigma_{\Delta s} f_\phi((s,a-\mu_x)/\sigma_x) + \mu_{\Delta s}$ | Lernt die globale Physik aus $200$ Schritten | Fehlerfortpflanzung in simulierten Trajektorien über $k$ Schritte |

<!--en:cd69d0f4-->
| **MPC mit Random Shooting** | $\operatorname{argmax}_{A^{(j)}} \sum_{t=0}^{n-1} \hat{r}_t^{(j)}$, nur $a_0^\star$ ausführen | Neuplanung verhindert, dass sich Fehler anhäufen | Hoher Rechenaufwand zur Testzeit ($J \times n$ Modellaufrufe) |

<!--en:ff913187-->
| **Initialisierung mit dem verschobenen Plan** | Kandidaten um $A^\star_{1:n}$ aus Schritt $t-1$ ziehen | Erhält gute Mehrschrittpläne über die Zeit | Kann bei einer lokal bevorzugten Trajektorie stecken bleiben |

<!--en:c38d41a2-->
| **Destillation der Suche** | Akteur $\bar{a}_\theta(s)$ durch Regression an die vom Akteur initialisierte Suche `improve(s)` anpassen | Schnelle Strategieauswertung zur Testzeit in $O(1)$ | Kovariatenverschiebung, sofern die Labels nicht zusammengeführt werden (DAgger) |

<!--en:a09c3c17-->
| **Simulierte Trajektorien bei Dyna / MBPO** | Kritiker mit simulierten $k$-Schritt-Trajektorien trainieren | Vervielfacht die Daten für Aktualisierungen des Kritikers | Reale Übergänge aus dem Puffer schneiden bei gleichem `UTD` oft ebenso gut ab |
