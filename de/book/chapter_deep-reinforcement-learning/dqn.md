<!--en:b311d0df-->
# Deep Q-Networks

<!--en:128e2fcd-->
In :numref:`sec_qlearning` lernte ein Agent Übergang für Übergang eine Tabelle von Aktionswerten (engl. action values). Dabei spielte es keine Rolle, welches Verhalten die Übergänge erzeugt hatte. In :numref:`sec_deeprl` ersetzten wir eine Strategietabelle durch ein Netz und trainierten es auf CartPole. Für Q-Learning scheint dafür eine einzige Zeile zu genügen. Das Netz gibt die beiden Aktionswerte aus, die Aktualisierungsregel bleibt dieselbe. Beim Strategiegradienten (engl. policy gradient) war der Austausch problemlos, weil die Herleitung an keiner Stelle auf die Tabelle zurückgriff. Beim Q-Learning geht die Tabelle in den Zielwert $r + \gamma \max_{a'} Q(s', a')$ ein. Ersetzen wir sie durch ein Netz, verändert jede Aktualisierung mit der Vorhersage auch den Zielwert. Eine einzige Aktualisierung ändert sogar die Werte von Zuständen, die der Agent nie besucht hat. Das Kontraktionsargument, das die Konvergenz des tabellarischen Verfahrens sicherte, greift nicht mehr. Kann ein Netz trotzdem wie die Tabelle aus beliebigen Übergängen lernen?

<!--en:c4c3377b-->
2013 gelang einer Gruppe bei DeepMind dieser Austausch. Ein Netz lernte anhand der Bildschirmpixel, sieben Atari-Spiele zu spielen. Für das Training nutzte die Gruppe gespeicherte Übergänge aus früheren Spielzügen :cite:`mnih2013playing`. Sie nannte das Netz Deep Q-Network (DQN). 2015 spielte dasselbe Netz mit denselben Hyperparametern bereits 49 Spiele, mehr als die Hälfte davon etwa so gut wie eine menschliche Testperson :cite:`mnih2015human`. Zur Berechnung der Zielwerte diente nun eine eingefrorene Kopie des Netzes. Der Speicher für Übergänge und die eingefrorene Kopie stabilisierten das Training und gehören seither zu fast jedem tiefen Off-Policy-Verfahren. Damit kann der Agent auch etwas, das in den vorigen Abschnitten noch nicht möglich war. Das Akteur-Kritiker-Verfahren verwendete jeden Batch nur einmal und PPO nur für wenige Epochen. DQN trainiert dagegen viele Male mit jedem gespeicherten Übergang.

<!--en:d6dac93a-->
Zuerst zeigen wir an einem Beispiel mit sieben Zuständen, warum der direkte Austausch scheitern kann. Die Gewichte wachsen dort ohne Schranke, obwohl kein Stichprobenrauschen auftritt. Dann implementieren wir DQN für CartPole und entfernen die eingefrorene Kopie, um zu sehen, was sie verhindert. Zum Schluss untersuchen wir die Maximierungsverzerrung (engl. maximization bias), die das Maximum im Zielwert erzeugt, und eine kleine Änderung, die sie verringert. In :numref:`sec_regularized` haben wir dieses Maximum durch ein weiches Maximum ersetzt. Hier bleibt es hart.

<!--en:d4e4bda9-->
Wir verwenden wieder CartPole mit Diskontfaktor $0.99$. Als Trainingsbudget geben wir $50{,}000$ Umgebungsschritte vor, weil sich die Episodenlänge mit der Strategie ändert. Nach jeweils zwei Umgebungsschritten macht der Agent einen Gradientenschritt mit einem Minibatch aus 128 gespeicherten Übergängen. Sein Q-Netz bildet einen Zustand über zwei versteckte Schichten mit je 128 Einheiten auf einen Wert pro Aktion ab. Die Greedy-Aktion ist die Aktion mit dem höchsten Wert. Zur Exploration dient eine $\epsilon$-Greedy-Strategie. Dabei sinkt $\epsilon$ in den ersten 10 000 Schritten linear von 1 auf 0.05, wie im Plan `linear_schedule` aus :numref:`sec_qlearning`. Die Funktion `make_env` erstellt die Umgebung. Mit ihrem optionalen Argument `copies` lässt sich jede Aktion mehrfach anbieten. Das nutzen wir im letzten Experiment dieses Abschnitts, bis dahin ist der Wert 1.

<!--en:951d39c8-->
Als Nächstes erzeugt `make_qnet(seed, copies=1)` mit festgelegtem Zufallsstartwert ein mehrschichtiges Perzeptron (MLP) mit zwei versteckten Schichten. Es bildet den 4-dimensionalen Zustand von CartPole auf $2 \times \text{copies}$ Aktionswerte ab:

<!--en:fda610a6-->
Als Schichten des Q-Netzes dienen `d2l.Linear`. Ihre Initialisierung aus einer Gleichverteilung beschreibt :numref:`sec_deeprl`. Mit der Standardinitialisierung von Flax löste dieses DQN CartPole in unseren Versuchen bei weniger Zufallsstartwerten.

<!--en:61874785-->
## Ursachen der Instabilität

<!--en:1c0358b2-->
### Der Semigradientenschritt

<!--en:aebed964-->
Ersetzen wir die Tabelle aus :numref:`sec_qlearning` durch ein Netz $Q_w$ mit Parametern $w$, bleibt die Aktualisierung ein Semigradientenschritt in Richtung des Zielwerts aus dem Bootstrapping. Wir ziehen einen Übergang $(s, a, r, s')$ und bilden den Zielwert

<!--en:a77af6e1-->
Dann halten wir $y$ fest und machen einen Optimierungsschritt, der $Q_w(s, a)$ näher an $y$ bringt. Der Indikator entfernt den Bootstrap-Wert, wenn $s'$ die Episode beendet. Jeder einzelne Schritt entspricht einer gewöhnlichen Regression auf eine feste Zahl. Die Folge der Schritte beschreibt jedoch keinen Abstieg auf einer festen Zielfunktion, weil sich die Zahl mit $w$ ändert. Das haben wir in :numref:`sec_deeprl` für alle Zielwerte aus dem Bootstrapping gesehen. Ein sinkender Verlust zeigt also nur, dass das Netz seine aktuellen Zielwerte gut trifft. Ob diese stimmen, bleibt offen.

<!--en:c48b5ef8-->
Bei einer Tabelle entspricht die erwartete Aktualisierung einem gedämpften Schritt der Wertiteration. Das Bellman-Backup ist eine Kontraktion (:numref:`sec_valueiter`). Es verkleinert den größten Fehler über alle Zustände um den Faktor $\gamma$. Ein Netz kann jedoch nicht jede Funktion darstellen, die sich aus einem Bellman-Backup ergibt. Jede Regression liefert daher die Funktion seiner Klasse, die den Zielwerten am nächsten kommt. „Am nächsten“ bemisst sich nach der Methode der kleinsten Quadrate über die Trainingsdaten. Der Fehler eines Zustands zählt proportional dazu, wie oft er in den Daten vorkommt. Bei linearen Merkmalen entspricht diese Anpassung einer Projektion. Das Bellman-Backup verkleinert den größten Fehler, die Anpassung hingegen den nach Besuchshäufigkeit gewichteten mittleren Fehler. Diese Maße unterscheiden sich. Eine Anpassung kann den gewichteten Fehler verringern und zugleich den Fehler in einem selten besuchten Zustand vergrößern. Das nächste Bellman-Backup überträgt ihn auf die Zielwerte der Zustände, die dorthin führen. Stammen die Daten von der bewerteten Strategie, stimmen die Maße hinreichend gut überein, um bei linearen Merkmalen Konvergenz zu sichern :cite:`Tsitsiklis.VanRoy.1997`. Mit Daten einer anderen Strategie kann das Verfahren selbst bei linearen Merkmalen divergieren.

<!--en:1f9d0f60-->
### Korrelierte Daten und veränderliche Zielwerte

<!--en:069d56a1-->
Zwei Kopplungen erschweren in der Praxis das Lernen mit einem Netz gegenüber dem Lernen mit einer Tabelle. Die erste liegt in den Daten. Aufeinanderfolgende Übergänge einer Episode sind fast identisch, weil sich Wagen und Stab von Schritt zu Schritt nur wenig bewegen. Trainieren wir mit den Übergängen in der Reihenfolge ihres Eintreffens, aktualisieren wir viele Male hintereinander einen kleinen Bereich des Zustandsraums. Bei einer Tabelle ändert sich nur der besuchte Eintrag, sodass die Korrelation das Lernen lediglich verlangsamt. Ein Netz ändert mit jedem Schritt seine Vorhersagen für viele Zustände (:numref:`fig_rl_table_vs_network`). Eine Reihe ähnlicher Aktualisierungen kann daher überschreiben, was das Netz anderswo gelernt hat.

<!--en:e39f7031-->
Die zweite Kopplung liegt in den Zielwerten. Der Zielwert $y$ enthält $Q_w$. Jeder Optimierungsschritt ändert daher sowohl die Vorhersage als auch die Zahl, an die wir sie anpassen. Beim Kritiker aus :numref:`sec_actorcritic` gab es dieselbe Kopplung. Nach jeder Aktualisierung passten wir ihn jedoch an neue Übergänge der aktuellen Strategie an. Fehler in den von der Strategie besuchten Zuständen kamen daher rasch zum Vorschein und ließen sich korrigieren.

<!--en:8462ae85-->
### Die tödliche Triade und Bairds Gegenbeispiel

<!--en:b7a799b3-->
Beim Q-Learning mit einem Netz kommen drei Bestandteile zusammen: *Funktionsapproximation*, *Bootstrapping* und *Off-Policy-Daten*. Letztere stammen von einer anderen Strategie als derjenigen, deren Werte wir lernen. :numref:`fig_rl_deadly_triad` ordnet die Algorithmen dieses und des vorigen Kapitels nach diesen Bestandteilen. In jedem Bereich mit zwei Bestandteilen steht ein Verfahren, das unter seinen jeweiligen Voraussetzungen konvergiert. Tabellarisches Q-Learning nutzt Bootstrapping mit Off-Policy-Daten und konvergiert unter den Bedingungen für die Schrittweite aus :numref:`sec_qlearning`. Lineares Lernen mit zeitlichen Differenzen, die einfachste Form des Kritikers im Akteur-Kritiker-Verfahren, verbindet Bootstrapping mit Funktionsapproximation. Es konvergiert bei linearen Merkmalen, wenn die Daten von der bewerteten Strategie stammen. Beim Verhaltensklonen passen wir ein Netz ohne Bootstrapping an Off-Policy-Daten an. Das ist überwachte Regression auf feste Labels. Dies sind hinreichende Bedingungen, kein allgemeiner Satz über Paare. Nichtlineares Lernen mit zeitlichen Differenzen kann sogar mit On-Policy-Daten divergieren :cite:`Tsitsiklis.VanRoy.1997`. Die Bezeichnung *tödliche Triade* (engl. deadly triad) meint das Zusammentreffen aller drei Bestandteile :cite:`Sutton.Barto.2018,vanHasselt.Doron.Strub.ea.2018`.

<!--en:cf009db2-->
Bairds Gegenbeispiel isoliert die Triade in einem linearen Problem mit sieben Zuständen :cite:`Baird.1995`. Jede Belohnung ist null. Die wahre Wertfunktion ist daher überall null und lässt sich durch $w = 0$ exakt darstellen. Die Rechnung verwendet die Erwartungswerte der Aktualisierungen mit zeitlichen Differenzen, sodass kein Stichprobenrauschen auftritt. Sie gewichtet die sieben Zustände gleich. Das ist eine Off-Policy-Verteilung, denn die bewertete Strategie wechselt immer in Zustand 7:

<!--en:9265af0e-->
Nach einer Übergangsphase nähert sich der Faktor, um den die Gewichte pro Durchlauf wachsen, einer Konstanten. Nach tausend Durchläufen ist ihre Supremumsnorm von $10$ auf $335$ gestiegen. Für Zustand 7 liefern die Gewichte den Wert $677$, obwohl sein wahrer Wert null ist. Dabei tritt kein Stichprobenfehler auf. Eine kleinere Schrittweite verlangsamt die Divergenz, verhindert sie aber nicht. Da sich die wahre Wertfunktion darstellen lässt, scheidet auch ein Approximationsfehler als Ursache aus. Bei der Off-Policy-Gewichtung gibt es in der erwarteten Aktualisierung eine Richtung, in der die Gewichte wachsen :cite:`Sutton.Barto.2018`. Gewichten wir die Zustände stattdessen nach den Besuchshäufigkeiten der bewerteten Strategie, hört die Divergenz auf. Für Zustände, die diese Strategie nie besucht, liefert die Iteration allerdings falsche Werte (Aufgabe :exref:`ex_dqn-baird-on-policy`).

<!--en:7be64a52-->
Experience-Replay und Zielnetze lassen alle Bestandteile der Triade bestehen. Sie verringern die zeitliche Korrelation und verlangsamen die Änderung der Zielwerte. Das macht das Training in der Praxis wesentlich stabiler, liefert aber weiterhin keine allgemeine Konvergenzgarantie.

<!--en:23dec802-->
![Die tödliche Triade. Jeder Kreis steht für einen Bestandteil. In jedem Bereich steht ein Algorithmus aus diesen beiden Kapiteln. Im Zentrum treffen alle drei Bestandteile zusammen. Hier liegen DQN und Offline-Q-Learning, und hier zeigt Bairds Gegenbeispiel, dass Divergenz möglich ist. Lässt man einen beliebigen Bestandteil weg, lässt sich Konvergenz begründen, jeweils unter eigenen Voraussetzungen an Schrittweite, Abdeckung und Darstellung. Die Triade warnt vor möglicher Divergenz. Sie ist kein notwendiges und hinreichendes Kriterium.](../img/mdl-rl-deadly-triad.svg)

<!--en:d64bdad9-->
## Experience-Replay und Zielnetze

<!--en:5f272508-->
DQN begegnet jeder der beiden Kopplungen mit einer eigenen Maßnahme. :numref:`fig_rl_dqn_dataflow` zeigt, wie beide zusammenspielen.

<!--en:2bfefa15-->
![Datenfluss beim tiefen Q-Learning. Die Verhaltensstrategie wählt Aktionen mit dem Online-Netz. Die Übergänge gelangen in den Replay-Puffer. Gleichverteilt gezogene Minibatches mischen die zeitliche Reihenfolge, sodass die Daten nahezu unabhängig und wiederverwendbar sind. Eine eingefrorene Kopie des Netzes berechnet die Regressionsziele. Sie übernimmt alle $C$ Schritte die neuen Gewichte. Zwischen diesen Synchronisationen bleiben die Zielwerte fest. Der Gradient fließt auf genau einem Pfad ins Online-Netz. Über die markierte Kante von der eingefrorenen Kopie fließt kein Gradient. Jeder gespeicherte Übergang trägt das Kennzeichen $\mathrm{term}$. Die Maske $1 - \mathrm{term}$ im Zielwert unterbindet das Bootstrapping nur bei einer tatsächlichen Terminierung. Der abgebildete Puffer ist voll und verdrängt seine ältesten Einträge, wie es in einem langen Lauf schließlich üblich ist. In unseren Läufen reichen die 50 000 Schritte nicht aus, um den Ring mit 200 000 Plätzen zu füllen. Daher wird kein Eintrag verdrängt.](../img/mdl-rl-dqn-dataflow.svg)

<!--en:55e44b01-->
### Experience-Replay

<!--en:2f2b808e-->
Gegen korrelierte Daten hilft ein *Replay-Puffer* (engl. experience replay buffer) :cite:`Lin.1992`. Wir speichern jeden Übergang in einem großen Array und ziehen daraus jeden Minibatch für das Training gleichverteilt. Ein Minibatch verbindet so Übergänge von verschiedenen Zeitpunkten, und jeder Übergang fließt in viele Aktualisierungen ein. Q-Learning erlaubt das, weil der Zielwert für $(s, a, r, s')$ unabhängig davon ist, welche Strategie $a$ gewählt hat. Experience-Replay hat einen Nachteil. Die meisten gespeicherten Übergänge stammen von älteren Strategien. Ein Fehler in den Zielwerten kann daher viele Aktualisierungen überdauern, bis die aktuelle Verhaltensstrategie die betroffenen Zustände erneut besucht.

<!--en:ff315f26-->
`ReplayBuffer` speichert die fünf Felder eines Übergangs in vorab reservierten Arrays. Das sind der Zustand, die Aktion, die Belohnung, der Folgezustand und das Kennzeichen `term` für die Terminierung. Die Methode `add` schreibt den Übergang auf den nächsten freien Platz. Ist das Array voll, überschreibt sie den ältesten Eintrag. Die Methode `sample` zieht `batch_size` Plätze gleichverteilt und gibt ihre Einträge als `Batch` aus :numref:`sec_policygradient` zurück:

<!--en:1f440462-->
In diesem Abschnitt fasst der Puffer $200{,}000$ Übergänge. Das übersteigt das Budget von $50{,}000$ Schritten, sodass kein Übergang verdrängt wird. Aufgabe :exref:`ex_dqn-shrink-buffer` untersucht einen kleineren Puffer. Die Methode `sample` behandelt jeden Übergang als Episodenfragment der Länge eins, weil bei einer gleichverteilten Ziehung die Reihenfolge innerhalb der Episode verloren geht. Bei einem solchen Batch ist `reward_to_go` die Belohnung selbst. Die Methode `truncation_bootstrap` aus :numref:`sec_deeprl` addiert $\gamma (1 - \mathrm{term})\, \hat{V}(s')$ für eine gegebene Wertschätzung $\hat V$. Die Summe ist damit das Einschrittziel, das `td_target` aus :numref:`sec_actorcritic` berechnet. Bei dieser Form von Experience-Replay bleibt nur das Einschrittziel, weshalb wir sie mit Einschritt-Bootstrapping kombinieren. Ein $n$-Schritt-Ziel braucht aufeinanderfolgende Übergänge. Dafür müsste der Puffer ihre Reihenfolge erhalten (Aufgabe :exref:`ex_dqn-nstep-replay`).

<!--en:c26cac42-->
### Das Zielnetz

<!--en:4e0657d7-->
Gegen veränderliche Zielwerte hilft das *Zielnetz* (engl. target network) :cite:`mnih2015human`. Eine eingefrorene Kopie $Q_{w^-}$ des Online-Netzes berechnet jeden Bootstrap-Wert:

<!--en:d6901edc-->
Wir synchronisieren die Kopie mit den Gewichten des Online-Netzes gemäß $w^- \leftarrow w$. Das geschieht alle $C$ Umgebungsschritte, hier alle 250 Schritte. Zwischen zwei Synchronisationen bleiben die Zielwerte fest, und das Online-Netz löst ein gewöhnliches Regressionsproblem. Vorhersagen fließen nur bei jeder Synchronisation in die Zielwerte zurück statt bei jedem Gradientenschritt. Der Kritiker aus :numref:`sec_actorcritic` berechnete den Bootstrap-Wert mit seinen neuesten Gewichten, weil jeder Batch aus neuen Übergängen bestand. Beim Experience-Replay fällt diese Voraussetzung weg. DQN nimmt einige Hundert Schritte alte Zielwerte in Kauf und erhält dafür ein Regressionsproblem, das sich langsam ändert.

<!--en:2aad4569-->
Zwei weitere Maßnahmen begrenzen die Größe eines Schritts. Der Huber-Verlust ist für Residuen mit einem Betrag unter 1 quadratisch und darüber linear. Seine Ableitung nach der Vorhersage hat daher höchstens den Betrag 1. Der vollständige Gradient enthält auch $\nabla_w Q_w(s, a)$. Diesen Faktor begrenzt der Huber-Verlust nicht. Deshalb begrenzen wir die Norm des Parametergradienten auf 10. Das ist der Standardwert aus Stable-Baselines3 und stammt nicht aus der Originalarbeit :cite:`Raffin.Hill.Gleave.ea.2021`. Im Code wertet `q_values` das Netz für einen als NumPy-Array vorliegenden Zustand aus, ohne Gradienten aufzuzeichnen. Die Funktion `fit_q` macht mit dem begrenzten Gradienten des Huber-Verlusts einen Schritt von $Q_w(s, a)$ in Richtung der festen Zielwerte $y$. Die Funktion `q_step` implementiert die DQN-Aktualisierung. Die Zielwerte liefert `batch.td_target` mit dem Bootstrap-Wert $\max_{a'} Q_{w^-}(s', a')$ der eingefrorenen Kopie.

<!--en:bee12a54-->
`q_values` steht für `d2l.forward_np`. Diese Funktion kompiliert für jede Netzstruktur einen Vorwärtsdurchlauf (:numref:`sec_utils`). Die Begrenzung der Norm gehört zur Kette der Optimierungsschritte. Beim Kompilieren von `q_step` wird die Berechnung von `td_target` mitkompiliert. Eine kompilierte Funktion erwartet Arrays und kann keinen `d2l.Batch` entgegennehmen. Daher erhält `q_step` die Spalten des Batchs und setzt sie intern wieder zu einem `Batch` zusammen, einzig für den Aufruf von `td_target`.

<!--en:b53114de-->
### Die Trainingsschleife

<!--en:02d20d5a-->
Die Teile des lernenden Agenten, die vom Framework abhängen, liefert `make_learner`. Dazu gehören die eingefrorene Kopie, der Optimierer und zwei Funktionen. Mit `update` wenden wir die Aktualisierungsregel `step` einmal auf einen gezogenen Batch an. Mit `sync` übertragen wir die Gewichte des Online-Netzes auf die eingefrorene Kopie.

<!--en:dec6d9e3-->
`nnx.cached_partial` bindet das Netz, seine Kopie und den Optimierer einmalig an den kompilierten Schritt. So muss die Schleife nicht bei jedem Aufruf alle Module durchlaufen.

<!--en:d48e636d-->
Die Schleife folgt einer $\epsilon$-Greedy-Strategie. Dazu nutzt sie `epsilon_greedy` aus :numref:`sec_qlearning`. Mit Wahrscheinlichkeit $\epsilon$ wählt sie eine zufällige Aktion, sonst die mit dem höchsten Wert. Sie speichert jeden Übergang. Nach einer Anlaufphase von 1000 Schritten ruft sie bei jedem zweiten Schritt `update` mit einem gezogenen Minibatch auf. Ein Schalter legt die Versuchsvariante fest. Mit `use_target=True` übernimmt die Kopie alle `sync_every` Schritte die aktuellen Gewichte. Mit `use_target=False` geschieht dies nach jedem Schritt. Der Zielwert stammt dann praktisch vom Online-Netz selbst, wie beim naiven Verfahren ohne Zielnetz. Das Argument `step` wählt die Aktualisierungsregel und `copies` die Anzahl der Kopien jeder Aktion. Am Ende jeder Episode liefert der Generator die Schrittzahl, den Ertrag der Episode und die Schätzung $\max_a Q_w(s_0, a)$ des Netzes im zentrierten Startzustand $s_0 = \mathbf{0}$.

<!--en:5fed1bb3-->
Der Puffer speichert `terminated`, niemals `truncated`. Endet eine CartPole-Episode nach 500 Schritten durch das Zeitlimit, hat ihr letzter Zustand weiterhin eine Zukunft. Wie wir in :numref:`sec_qlearning` gesehen haben, lernt das Netz bei maskiertem Bootstrap-Wert an dieser Stelle, dass das Überleben wertlos ist. Aufgabe :exref:`ex_dqn-terminated-not-done` untersucht die Folgen.

<!--en:ecb04604-->
### Vergleich mit und ohne Zielnetz

<!--en:3bbdd245-->
Brauchen wir die eingefrorene Kopie? Wir trainieren beide Varianten mit jeweils drei Zufallsstartwerten. Die Varianten unterscheiden sich nur in der Stellung des Schalters. Die folgenden Angaben darüber, wie häufig ein Ergebnis auftritt, haben wir mit zwanzig Zufallsstartwerten pro Variante überprüft. Wir erstellen die Netze außerhalb der Trainingsschleife, damit wir die trainierten Agenten in späteren Zellen auswerten können. Die Funktion `train_checked` ruft `train_dqn` auf und sammelt die Daten jeder Episode. Nach jeweils weiteren 5000 Umgebungsschritten wartet sie bis zum ersten Episodenende. Dann führt sie die *Greedy-Strategie*, die immer $\mathrm{argmax}_a Q_w(s, a)$ wählt, für zehn Episoden in einer separaten Kopie der Umgebung aus. Am Ende des Budgets wiederholt sie die Auswertung. Die Funktion `d2l.evaluate` implementiert die Monte-Carlo-Auswertung aus :numref:`sec_valueiter`. Sie mittelt die Erträge einer Strategie über mehrere Episoden. Mit `greedy` wird aus einem Q-Netz eine solche Strategie. Diese Auswertungen beeinflussen das Training nicht. Wir betrachten ihre Ergebnisse, sobald wir die Wertschätzungen verstehen.

<!--en:c5068294-->
Mit `train_checked` trainieren wir nun jede Variante aus `arms = ('DQN', 'no target network')` mit drei Zufallsstartwerten. Die Trainingsdaten pro Episode speichern wir in `runs`, die Ergebnisse der regelmäßigen Auswertung der Greedy-Strategie in `checks`:

<!--en:107c7b9d-->
Je nach Zufallsstartwert enden unterschiedlich viele Episoden. Auf der horizontalen Achse tragen wir daher Umgebungsschritte auf, die Einheit des Budgets. Die Funktion `d2l.on_grid` überträgt die Episodendaten jedes Laufs auf ein gemeinsames Raster von Umgebungsschritten. Die Funktion `d2l.plot_curves` zeichnet den Median über die Zufallsstartwerte sowie ein Band vom niedrigsten bis zum höchsten Ergebnis:

<!--en:6a4f41ed-->
Mit Zielnetz steigt der Ertrag bei jedem Zufallsstartwert von etwa 20 bei der Zufallsstrategie auf mehrere Hundert, auch wenn die Kurven stark schwanken. Der Ertrag der Verhaltensstrategie unterschätzt die Güte der gelernten Strategie. Selbst die endgültige Explorationsrate $\epsilon = 0.05$ erzwingt noch bei einem von zwanzig Schritten eine zufällige Aktion. Ohne Zielnetz übersteigt der Ertrag nie den der Zufallsstrategie und fällt bald auf etwa 9. So lange dauert eine Episode, wenn der Wagen bei jedem Schritt einen Stoß in dieselbe Richtung erhält.

<!--en:e5baa4b5-->
Die Wertschätzungen zeigen, was schiefging. Die Trainingsschleife zeichnet $\max_a Q_w(s_0, a)$ im zentrierten Startzustand auf. Wir können im Voraus angeben, welchen Wert die Schätzung am Ende erreichen sollte. Der Puffer maskiert den Bootstrap-Wert nur bei `terminated`. Die Aktualisierung nutzt daher auch am Zeitlimit von 500 Schritten Bootstrapping, und der Zustand enthält keine Uhr. Das Netz lernt somit den Wert einer fortlaufenden Aufgabe. Das Zeitlimit beendet die Datenerhebung, der Prozess selbst läuft weiter. Eine Strategie, die den Stab für immer balanciert, erzielt vom Start an $\sum_t \gamma^t = 1/(1-\gamma) = 100$. Keine Strategie erzielt mehr, denn jeder Schritt bringt 1. Die gestrichelte Linie markiert diese Obergrenze:

<!--en:774b2742-->
Die DQN-Schätzung steigt in Richtung der Obergrenze. Ohne Zielnetz übersteigt die Schätzung dagegen $10^8$. Die Werte laufen nach oben davon, weil CartPole bei jedem Schritt die Belohnung 1 liefert. Für jeden Übergang, der die Episode nicht beendet, lautet der Zielwert $1 + \gamma \max_{a'} Q_w(s', a')$. Er liegt also um eins über einer diskontierten Ausgabe des Netzes selbst. Die gemeinsamen Parameter, die $Q_w(s, a)$ in Richtung dieses Ziels anheben, erhöhen auch $Q_w(s', \cdot)$, weil $s'$ nahe bei $s$ liegt. Der Zielwert eilt der Vorhersage bei jedem Schritt voraus, und nichts bremst den Anstieg. :numref:`fig_rl_target_feedback` zeigt diesen Kreislauf und die eine Kante, an der eine eingefrorene Kopie ihn unterbricht.

<!--en:b745e16a-->
![Q-Learning ohne Zielnetz bei CartPole. Ein Gradientenschritt hebt $Q_w(s, a)$ in Richtung des Zielwerts $y$ an. Über die gemeinsamen Gewichte steigt auch $Q_w(s', \cdot)$. Der nächste Zielwert berechnet sich aus $Q_w(s', \cdot)$ und steigt mit. Der nächste Schritt folgt diesem höheren Zielwert. Berechnet stattdessen eine eingefrorene Kopie $Q_{w^-}$ den Zielwert, unterbricht sie die markierte Kante bis zur nächsten Synchronisation.](../img/mdl-rl-target-feedback.svg)

<!--en:547603cc-->
Eine eingefrorene Kopie hält den Zielwert für 125 Gradientenschritte fest. Die Werte steigen daher nur von einer Synchronisation zur nächsten. Nehmen wir an, alle Werte beginnen bei null und das Online-Netz trifft seine Zielwerte zwischen zwei Synchronisationen exakt. Dann überschreitet nach $n$ Synchronisationen kein Wert die Schranke

<!--en:7368074c-->
Dies ist der diskontierte Ertrag aus $n$ Schritten mit Belohnung 1. Nach dem ersten Gradientenschritt reicht das Budget für etwa 196 Synchronisationen. Das begrenzt den Wert des Startzustands auf etwa 86. Die gezeichnete Schätzung folgt dieser Schranke eng. Ab etwa 10 000 Schritten liegt sie einige Punkte darüber und endet bei jedem Zufallsstartwert über 86. Der Grund ist die Generalisierung des Netzes. Ein Schritt, der den Wert eines Zustands anhebt, erhöht auch die Werte seiner Nachbarn. So breiten sich die Werte schneller aus als mit einem Backup pro Synchronisation (Aufgabe :exref:`ex_dqn-sync-period`).

<!--en:f4528e64-->
Ein Zielnetz ist eine von mehreren Möglichkeiten, diese Rückkopplung zu unterbrechen. PQN :cite:`Gallici.Fellows.Ellis.ea.2025` verzichtet sowohl auf die eingefrorene Kopie als auch auf den Replay-Puffer. Es normalisiert die Aktivierungen jeder versteckten Schicht und sammelt parallel Daten aus vielen Umgebungen. In Aufgabe :exref:`ex_dqn-layernorm` erproben Sie die Normalisierung auf CartPole.

<!--en:f1b8c48d-->
### Auswertung der Greedy-Strategie

<!--en:ff7ed917-->
Die Ertragskurve allein beschreibt einen Lauf nur unzureichend. Sie enthält die explorativen Aktionen, und ihr Endwert hängt davon ab, wann der Lauf gerade aufhört. Ihren besten Wert wählen wir erst in Kenntnis der ganzen Kurve aus, was jedes Verfahren günstig aussehen lässt. Deshalb werten wir die Greedy-Strategie bei vorab festgelegten Budgets aus. Die Ergebnisse hat `train_checked` aufgezeichnet. Nach jeweils 5000 Umgebungsschritten und am Ende des Budgets messen wir den Ertrag der Greedy-Strategie über zehn Episoden:

<!--en:113dd0c2-->
Bei dieser Auswertung liegt der Ertrag der Greedy-Strategie von DQN in den ersten 30 000 Schritten meist bei etwa 50 bis 250. Gelegentlich ergibt schon eine frühe Auswertung 500. Zwischen 35 000 und 45 000 Schritten erreichen die meisten Läufe 500. Ein gutes Ergebnis ist nicht endgültig. Bei einigen Zufallsstartwerten fällt der Ertrag später wieder um mehrere Hundert Punkte. Das sehen wir bei einem Lauf in der Ausgabe und bei etwa der Hälfte der zwanzig untersuchten Läufe. Ohne Zielnetz hält die Greedy-Strategie bei keiner Auswertung länger als etwa 10 Schritte durch. Auch längeres Training hilft ihr also nicht. Obwohl DQN jeden gespeicherten Übergang viele Male zum Training nutzt, braucht es mehr Umgebungsschritte als PPO mit vier Epochen pro Batch in :numref:`sec_ppo`, bis es den Stab eine ganze Episode lang balanciert.

<!--en:a4a3e02e-->
Der Ertrag der Greedy-Strategie zeigt, wie gut die Strategie ist. Weiß das Netz aber auch, wie gut sie ist? :citet:`Hasselt.Guez.Silver.2016` vergleichen die Wertschätzung mit dem diskontierten Ertrag, den die Greedy-Strategie tatsächlich erzielt. Genau diese Größe soll die Schätzung vorhersagen. Die Funktion `audit` berechnet für ein trainiertes Netz drei Zahlen: den Ertrag der Greedy-Strategie über 100 Episoden, ihren diskontierten Ertrag über 20 Episoden (`d2l.evaluate` mit `gamma`) und die Schätzung $\max_a Q_w(s_0, a)$. Die Episoden beginnen nahe dem zentrierten Zustand. Der diskontierte Ertrag misst daher den Wert von $s_0$. Unter dem Zeitlimit von 500 Schritten erzielt selbst eine perfekte Strategie nur $(1 - \gamma^{500})/(1 - \gamma) \approx 99.3$, etwas weniger als die Obergrenze der fortlaufenden Aufgabe.

<!--en:1a6f6a89-->
Wenn die Greedy-Strategie den Stab balanciert, weicht die Schätzung um höchstens etwa acht Punkte von ihrem tatsächlichen diskontierten Ertrag ab. Meist liegt sie darunter, weil die Werte am Ende des Budgets noch steigen. Die Kurve für den Wert des Startzustands oben ist noch nicht abgeflacht. Bei etwa zwei von fünf Zufallsstartwerten versagt die Greedy-Strategie. Dann übersteigt die Schätzung den erzielten Ertrag, in etwa der Hälfte dieser Läufe um mehr als 20 Punkte. Bei zwei Aktionen liefern nur diese fehlgeschlagenen Läufe einen Hinweis darauf, dass das Netz mehr verspricht, als seine Strategie einlöst. Für eine Messung sind es zu wenige, und andere Fehler überlagern den Effekt. Ein aussagekräftigerer Test braucht ein Maximum über mehr Aktionen.

<!--en:aa00c0b7-->
## Überschätzung

<!--en:5f8d90a6-->
### Die Maximierungsverzerrung in Zahlen

<!--en:099f8fba-->
Der Zielwert :eqref:`eq_dqn_target` maximiert geschätzte Aktionswerte. Für verrauschte Schätzungen gilt $\mathbb{E}[\max_{a'}\hat{Q}(s',a')]\geq\max_{a'}\mathbb{E}[\hat{Q}(s',a')]$. Die Ungleichung ist strikt, wenn das Rauschen ändern kann, welche Aktion den höchsten Wert hat. In :numref:`sec_qlearning` haben wir diese Verzerrung an einer Tabelle gesehen. In :numref:`sec_regularized` haben wir gezeigt, dass das harte Maximum im Grenzfall $\beta\to0$ aus dem weichen Backup mit gleichverteilter Referenzstrategie hervorgeht. Bei Funktionsapproximation fließt die Verzerrung in einem Zustand in die Zielwerte der Zustände ein, die zu ihm führen :cite:`Thrun.Schwartz.1993`.

<!--en:ad2acbcb-->
In der folgenden Simulation haben vier Aktionen denselben wahren Wert null. Zu jeder Schätzung addieren wir unabhängiges Rauschen mit Varianz eins. Der Einzelschätzer (engl. single estimator) verwendet dieselben verrauschten Werte, um eine Aktion auszuwählen und zu bewerten. Der Doppelschätzer (engl. double estimator) wählt die Aktion mit einem Satz verrauschter Werte aus und bewertet sie mit einem zweiten, unabhängigen Satz :cite:`vanHasselt.2010`:

<!--en:6cbce485-->
Das Maximum der vier Schätzungen mit Rauschvarianz eins liegt im Mittel etwa eine Einheit zu hoch. Verwenden wir für Auswahl und Bewertung unabhängige Schätzungen, hat das Rauschen bei der Bewertung für jede ausgewählte Aktion den Mittelwert null. Die gemessene Verzerrung liegt nahe null. Wie :numref:`fig_rl_max_bias` zeigt, wächst die Verzerrung des Einzelschätzers mit der Anzahl der Aktionen. Beim unabhängigen Doppelschätzer bleibt sie null. Ein Sprachmodell, das sein nächstes Token unter $10^5$ Aktionen auswählt, hätte bei einem Maximum über verrauschte Schätzungen eine weitaus größere Verzerrung. Das ist einer von mehreren Gründen, weshalb das Post-Training von Sprachmodellen Strategiegradienten statt eines Q-Netzes verwendet. Ein fester Datensatz verschärft die Verzerrung zusätzlich, weil keine neue Interaktion die Bewertung einer überschätzten Aktion korrigiert (:numref:`sec_offline`).

<!--en:ec722469-->
![Das Maximum verrauschter Schätzungen ist nach oben verzerrt. Links die Verteilung des Maximums von vier unabhängigen Schätzungen desselben wahren Werts null im Vergleich zur Verteilung einer einzelnen Schätzung. Die gemessene Verzerrung des Maximums beträgt $1.03$. Rechts wächst die Verzerrung des Einzelschätzers mit der Anzahl der Aktionen. Der Doppelschätzer verwendet für Auswahl und Bewertung der Aktion zwei unabhängige Sätze von Schätzungen. Seine Verzerrung ist durchweg nicht von null zu unterscheiden.](../img/mdl-rl-max-bias.svg)

<!--en:539886ec-->
### Double DQN

<!--en:17ea875b-->
Der Doppelschätzer braucht zwei Wertschätzungen, und DQN hat bereits zwei Netze. Double DQN :cite:`Hasselt.Guez.Silver.2016` wählt die Aktion mit dem Online-Netz aus und bewertet sie mit dem Zielnetz:

<!--en:651ad81e-->
Überschätzt das Online-Netz eine Aktion zufällig, bewertet sie also ein anderes Netz. Dafür brauchen wir kein neues Netz, nur einen weiteren Vorwärtsdurchlauf des Online-Netzes für $s'$. Die beiden Netze sind allerdings voneinander abhängig. Die Kopie entspricht dem Online-Netz von höchstens 125 Gradientenschritten zuvor. Double DQN beseitigt daher nur einen Teil der Verzerrung. Beträgt der Korrelationskoeffizient zwischen der bewertenden und der auswählenden Schätzung $\rho$, bleibt das $\rho$-Fache der Verzerrung des Einzelschätzers übrig (Aufgabe :exref:`ex_dqn-max-bias-closed-form`). In `q_step` ändern sich drei Zeilen:

<!--en:06f726ac-->
### Vervielfachte Aktionen

<!--en:280c9c0a-->
Bei CartPole hat das Maximum wenig Auswahl. Der größere von zwei unabhängigen Schätzwerten mit Rauschvarianz eins hat eine Verzerrung von $1/\sqrt{\pi} \approx 0.56$, etwa halb so viel wie bei vier Aktionen. Im Vergleich mit und ohne Zielnetz wirkte bei zwei Aktionen die Verzögerung zwischen den Synchronisationen der Überschätzung entgegen. Um die Verzerrung an einem trainierten Netz zu beobachten, ohne die Aufgabe zu ändern, bieten wir jede CartPole-Aktion neunmal an. Mit `make_env(copies=9)` stehen 18 Aktionen zur Verfügung, so viele wie im vollständigen Aktionsraum von Atari. Aktion $a$ stößt den Wagen in dieselbe Richtung wie Aktion $a \bmod 2$. Das Netz aus `make_qnet(seed, copies=9)` hat 18 Ausgaben. Die Kopien ändern weder die optimalen Werte noch die Obergrenze von 100. Die 18 Ausgaben liefern für jeden von zwei Werten neun verrauschte Schätzungen. Das Maximum im Zielwert greift unter den neun Schätzungen die zufällig höchste heraus. So sehen wir den Effekt aus dem rechten Teil von :numref:`fig_rl_max_bias` an einem echten Netz. Die nächste Zelle trainiert DQN und Double DQN mit 18 Aktionen sowie Double DQN mit zwei Aktionen. Anschließend prüfen wir jedes trainierte Netz. DQN mit zwei Aktionen haben wir bereits im Vergleich mit und ohne Zielnetz trainiert:

<!--en:ce0b73aa-->
Für jede Variante und jeden Zufallsstartwert geben wir den Ertrag der Greedy-Strategie, ihren diskontierten Ertrag und den geschätzten Wert des Startzustands aus:

<!--en:6a5bbd4b-->
Für die beiden Varianten mit 18 Aktionen vergleichen wir die geschätzten Werte des Startzustands im Trainingsverlauf mit der Obergrenze:

<!--en:b9bc5cb9-->
Bieten wir jede Aktion neunmal an, liegt die DQN-Schätzung für den Startzustand am Ende bei jedem Zufallsstartwert über der Obergrenze, meist um 10 bis 30 Punkte. Balanciert die Greedy-Strategie den Stab und erzielt etwa 99, übersteigt die Schätzung den Ertrag um etwa denselben Betrag. Kein in dieser Aufgabe möglicher Wert liegt über der Obergrenze. Die Überschätzung steht daher ohne weitere Auswertung fest. Im Diagramm überschreitet die DQN-Schätzung die Obergrenze vor 40 000 Schritten und steigt weiter, während die Schätzung von Double DQN nahe der Grenze oder darunter bleibt. Am Ende liegt die Schätzung von Double DQN bei fast jedem Zufallsstartwert um 10 bis 25 Punkte niedriger, nahe der Obergrenze. Bei einem Lauf, in dem die Strategie versagt, kann sie dennoch über die Grenze hinausschießen. In einigen der zwanzig DQN-Läufe mit 18 Aktionen steigen die Werte bei 50 000 Schritten noch, weil die Verzerrung über den Bootstrap-Wert zurückwirkt. Bei zwei Aktionen stimmen die beiden Zielwerte bis auf wenige Punkte überein. Bei diesem Budget unterscheiden sich die Erträge der Greedy-Strategien beider Varianten nur im Rahmen der Streuung zwischen den Zufallsstartwerten. Bei CartPole macht der Doppelschätzer die Werte genauer, ohne zu ändern, wie gut die Greedy-Strategie den Stab balanciert.

<!--en:3c10f2ec-->
Wir können ein Wertnetz nun mit Übergängen beliebiger Strategien trainieren und jeden Übergang viele Male wiederverwenden. Eine eingefrorene Kopie hält die Zielwerte fest. Eine zweite Schätzung wirkt der Überschätzung durch das Maximum entgegen. Beide Maßnahmen setzen voraus, dass wir das Maximum über die Aktionen berechnen können. Die zwei Stoßrichtungen bei CartPole und die 18 Joystick-Aktionen bei Atari lassen sich aufzählen. Das Drehmoment eines Motors hingegen ist eine Zahl in einem Intervall. Das Argmax über dieses Intervall zu finden ist bei jedem Schritt ein eigenes Optimierungsproblem. Im nächsten Abschnitt behalten wir Experience-Replay und die eingefrorene Kopie bei. Das Maximum ersetzen wir durch ein zweites Netz, das lernt, die beste Aktion vorzuschlagen. Damit ist die Off-Policy-Spalte in :numref:`fig_rl_roadmap` vollständig.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:81c838a6-->
Ersetzen wir die Q-Tabelle durch ein Netz, verbinden wir Funktionsapproximation, Bootstrapping und Off-Policy-Daten. Bairds sieben Zustände zeigen, dass diese Kombination selbst dann zur Divergenz führen kann, wenn sich die wahren Werte darstellen lassen und kein Stichprobenrauschen auftritt. DQN behält alle drei Bestandteile bei und entschärft zwei Kopplungen. Ein Replay-Puffer bricht die Korrelation aufeinanderfolgender Übergänge auf und nutzt jeden Übergang viele Male. Ein eingefrorenes Zielnetz hält die Regressionsziele zwischen den Synchronisationen fest. Entfernen wir die eingefrorene Kopie bei CartPole, steigt der Wert des Startzustands über $10^8$. Mit der Kopie nähert er sich von Synchronisation zu Synchronisation der Obergrenze von 100. Bei den meisten Zufallsstartwerten balanciert die Greedy-Strategie den Stab nach 35 000 bis 45 000 Umgebungsschritten die vollen 500 Schritte lang. Das Maximum im Zielwert verzerrt die Werte nach oben, umso stärker, je mehr Aktionen zur Auswahl stehen. Bieten wir jede Aktion neunmal an, liegt die DQN-Schätzung am Ende bei jedem Zufallsstartwert über der Obergrenze. Double DQN wählt die Aktion mit dem Online-Netz aus und bewertet sie mit der eingefrorenen Kopie. Ein zusätzlicher Vorwärtsdurchlauf genügt so, um den größten Teil der Verzerrung zu beseitigen.

<!--en:2294243a-->
## Übungen

<!--en:3e98d4e8-->
1. [code] **Bairds Gegenbeispiel mit On-Policy-Gewichtung.** Die Berechnung von Bairds Gegenbeispiel in :numref:`subsec_dqn-deadly-triad` gewichtet die sieben Zustände mit `.mean(0)` gleich. In Matrixform lautet ein Durchlauf $\mathbf{w} \leftarrow \mathbf{w} + \alpha A \mathbf{w}$ mit $\alpha = 0.01$ und

   $$A = \Phi^\top D (\gamma P - I) \Phi,$$

   Dabei ist $D$ die Diagonalmatrix der Zustandsgewichte und $P$ die Übergangsmatrix der bewerteten Strategie. In jeder Zeile dieser Matrix entfällt die gesamte Wahrscheinlichkeit auf Zustand 7.
    1. Berechnen Sie die Eigenwerte von $A$ für $D = I/7$. Vergleichen Sie $1 + \alpha\lambda$ für den Eigenwert $\lambda$ mit dem größten Realteil mit dem Faktor, um den die Supremumsnorm von $\mathbf{w}$ pro Durchlauf wächst. Messen Sie den Faktor gegen Ende eines Laufs mit $20{,}000$ Durchläufen.
    1. Die bewertete Strategie wechselt in Zustand 7 und bleibt dort. Ihre Besuchsverteilung legt daher das gesamte Gewicht auf Zustand 7. Wiederholen Sie die Iteration mit dieser Gewichtung. Gegen welche Gewichte und Zustandswerte konvergiert sie? Wie verhalten sich diese Werte zu den wahren Werten?
    1. Mischen Sie die beiden Gewichtungen gemäß $\mathbf{d} = (1 - p)\,\mathbf{1}/7 + p\,\mathbf{e}_7$. Bestimmen Sie durch Bisektion das kleinste $p$, für das kein Eigenwert von $A$ einen positiven Realteil hat. Untersuchen Sie $\gamma \in \{0.9, 0.95, 0.99\}$. Wie hängt der Anteil der Off-Policy-Gewichtung, den die Iteration verträgt, vom Diskontfaktor ab?
   :exlabel:`ex_dqn-baird-on-policy`

<!--en:c4352e39-->
1. **Terminierung und Zeitlimit unterscheiden.** `train_dqn` speichert `float(terminated)`. Daher nutzt auch ein Übergang, der das Zeitlimit von 500 CartPole-Schritten erreicht, noch Bootstrapping. Betrachten Sie eine Implementierung, die stattdessen `float(terminated or truncated)` speichert.
    1. Welche gespeicherten Übergänge betrifft die Änderung, solange die Greedy-Strategie noch innerhalb weniger Hundert Schritte versagt? Welche betrifft sie, sobald jede Episode das Limit erreicht?
    1. Die Beobachtung enthält keine Uhr. Das Netz kann daher einen Zustand bei Schritt 499 nicht von einem ähnlich gut ausbalancierten Zustand bei Schritt 50 unterscheiden. Modellieren Sie seine Schätzung für ausbalancierte Zustände durch eine einzige Zahl $q$. Das ist der Wert einer Aufgabe mit einem Zustand, die bei jedem Schritt die Belohnung 1 liefert und nach $H$ Schritten abgebrochen wird. Mit der fehlerhaften Maske haben $H - 1$ der $H$ Übergänge einer Episode den Zielwert $1 + \gamma q$ und der letzte den Zielwert $1$. Zeigen Sie, dass $q = H / \big(H - \gamma (H - 1)\big)$ der Fixpunkt der gemittelten Aktualisierung ist. Berechnen Sie ihn für $H = 500$ und $\gamma = 0.99$. Das Diagramm zum Wert des Startzustands in :numref:`subsec_dqn-ablation` kennzeichnet $1/(1 - \gamma)$ als Obergrenze. Vergleichen Sie $q$ mit dem Wert des Startzustands einer Episode von 500 Schritten bei bekannter Restzeit, also mit $(1 - \gamma^{H})/(1 - \gamma)$. Welcher der beiden Werte ist bei der fehlerhaften Implementierung die Obergrenze für ein Netz ohne Uhr? Worauf würde eine abschließende Schätzung von $94$ hindeuten?
   :exlabel:`ex_dqn-terminated-not-done`

<!--en:56bc4aca-->
1. [code] **Maximierungsverzerrung in geschlossener Form.** Die Schätzwerte $X_1, \ldots, X_m$ für $m$ Aktionen mit demselben wahren Wert $0$ seien gemeinsam normalverteilt und hätten jeweils die Varianz $\sigma^2$.
    1. Zeigen Sie für $m = 2$ und unabhängige Schätzwerte, dass $\mathbb{E}[\max(X_1, X_2)] = \sigma/\sqrt{\pi}$ gilt. Verwenden Sie dazu $\max(x, y) = (x + y)/2 + |x - y|/2$.
    1. Die Ausgaben eines Netzes haben gemeinsame versteckte Schichten. Ihre Fehler sind daher korreliert. Zeigen Sie für $m = 2$ und eine Korrelation $c$ zwischen $X_1$ und $X_2$, dass die Verzerrung $\sigma \sqrt{(1 - c)/\pi}$ beträgt.
    1. Bei Double DQN bewertet das Zielnetz die vom Online-Netz ausgewählte Aktion. Die Fehler der beiden Netze sind korreliert. Modellieren Sie die bewertende Schätzung für Aktion $a$ durch $Y_a = \rho X_a + \sqrt{1 - \rho^2}\, Z_a$. Dabei haben $Z_1, \ldots, Z_m$ jeweils Mittelwert null und Varianz $\sigma^2$ und sind unabhängig von $X_1, \ldots, X_m$. Somit hat $Y_a$ die Varianz $\sigma^2$ und die Korrelation $\rho$ mit $X_a$. Zeigen Sie, dass die Verzerrung von $Y_{\mathrm{argmax}_a X_a}$ gleich $\rho\, \mathbb{E}[\max_a X_a]$ ist.
    1. Überprüfen Sie die Teile (a) und (c), indem Sie die Simulation aus :numref:`subsec_dqn-max-bias` anpassen. Verwenden Sie zwei Aktionen und bilden Sie die zweite Schätzung als `rho * est + np.sqrt(1 - rho**2) * noise`. Ziehen Sie das standardnormalverteilte Rauschen `noise` jedes Mal neu und untersuchen Sie $\rho \in \{0, 0.5, 0.9\}$.
   :exlabel:`ex_dqn-max-bias-closed-form`

<!--en:666f19a7-->
1. [code] **Wiederverwendung von Übergängen und Pufferkapazität.** ● Sobald der Puffer `warmup` Übergänge enthält, macht `train_dqn` jeweils einen Gradientenschritt mit `batch_size` $= 128$ gleichverteilt gezogenen Übergängen. Das geschieht alle `train_freq` $= 2$ Umgebungsschritte. Bei der Standardkapazität wird kein Übergang verdrängt.
    1. Zeigen Sie, dass ein bei Schritt $t_0 \geq 1000$ gespeicherter Übergang im gesamten Lauf im Erwartungswert etwa $64 \ln(50{,}000/t_0)$ Mal gezogen wird. Berechnen Sie diesen Ausdruck für $t_0 \in \{1000, 25{,}000, 49{,}000\}$. Aus welcher Trainingsphase stammen die Daten, die am häufigsten in die Aktualisierungen eingehen?
    1. Der volle Puffer fasst nun 500 Übergänge. Wie oft wird ein Übergang im Erwartungswert gezogen, während er im Puffer bleibt? Vergleichen Sie das Ergebnis mit Teil (a).
    1. Setzen Sie `buffer_size` auf $500$ und auf $5000$. Senken Sie `warmup` für die kleinere Kapazität auf 500, denn `train_dqn` wartet, bis der Puffer `warmup` Übergänge enthält. Trainieren Sie für jede Kapazität mit einem Zufallsstartwert. Vergleichen Sie die Ertragskurve und den Verlauf des Werts des Startzustands mit dem Standardlauf. In welcher Trainingsphase zeigt sich die Verschlechterung?
    1. Bei einem kleinen Puffer sind die Übergänge eines Minibatchs korreliert, und alte Zustände gehen verloren. Um beide Effekte zu trennen, behalten Sie die Standardkapazität bei. Ändern Sie `ReplayBuffer.sample` so, dass die Methode `batch_size` aufeinanderfolgende Übergänge ab einem gleichverteilt gezogenen Startindex zurückgibt. Vergleichen Sie diesen Lauf mit dem Lauf bei Kapazität 500. Welcher Effekt erklärt die Verschlechterung in Teil (c)?
   :exlabel:`ex_dqn-shrink-buffer`

<!--en:4d7d3a57-->
1. [code] **Das Synchronisationsintervall.** ● Die Funktion `train_dqn` kopiert die Gewichte des Online-Netzes alle `sync_every` $= 250$ Umgebungsschritte in das Zielnetz, also alle 125 Gradientenschritte.
    1. Wie :numref:`subsec_dqn-ablation` zeigt, liegt jeder Wert nach $n$ Synchronisationen höchstens bei $(1 - \gamma^n)/(1 - \gamma)$. Berechnen Sie diese Schranke am Ende des Budgets für `sync_every` $\in \{10, 50, 250, 1000, 5000\}$. Zählen Sie dabei die Synchronisationen nach dem ersten Gradientenschritt.
    1. Trainieren Sie die DQN-Variante für diese fünf Werte erneut, jeweils mit zwei Zufallsstartwerten. Zeichnen Sie den Verlauf des Werts des Startzustands auf und werten Sie die Greedy-Strategie am Ende des Budgets aus. An welchem Ende der Versuchsreihe ähnelt das Verhalten dem Lauf ohne Zielnetz? An welchem Ende begrenzt die Schranke aus Teil (a) die erreichbaren Werte?
    1. Die Schranke setzt voraus, dass alle Werte bei null beginnen und das Online-Netz seine Zielwerte zwischen den Synchronisationen exakt trifft. Zeichnen Sie bei `sync_every` $= 250$ für einen Zufallsstartwert den Wert $\max_a Q_w(s_0, a)$ unmittelbar vor jeder der ersten 40 Synchronisationen auf. Stellen Sie ihn zusammen mit der Schranke grafisch dar. Wo überschreitet der Wert erstmals die Schranke? Welche der beiden Annahmen ist dort verletzt?
   :exlabel:`ex_dqn-sync-period`

<!--en:4d8f3f9a-->
1. **n-Schritt-Ziele aus dem Replay-Puffer.** ● Mehrere Nachfolger von DQN ersetzen das Einschrittziel :eqref:`eq_dqn_target` durch ein $n$-Schritt-Ziel. Es entspricht dem $n$-Schritt-Ertrag aus :numref:`sec_actorcritic` mit einem Maximum am Ende:

   $$y^{(n)}_t = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n \max_{a'} Q_{w^-}(s_{t+n}, a'),$$

   Hier ist angenommen, dass die Episode über Schritt $t + n$ hinaus andauert.
    1. `ReplayBuffer.sample` gibt einzelne Übergänge zurück. Was muss der Puffer speichern und wie muss `sample` die Indizes wählen, damit sich $y^{(n)}_t$ bilden lässt?
    1. Geben Sie $y^{(n)}_t$ für eine Episode an, die bei Schritt $t + j$ mit $j < n$ terminiert. Wie ändert sich der Ausdruck, wenn die Episode bei Schritt $t + j$ stattdessen abgebrochen wird?
    1. Die Aktionen $a_{t+1}, \ldots, a_{t+n-1}$ stammen aus der $\epsilon$-Greedy-Verhaltensstrategie eines älteren Netzes. Geben Sie für $n = 2$ und $Q_{w^-} = Q^*$ den Ausdruck für $\mathbb{E}[y^{(2)}_t \mid s_t, a_t]$ an. Zeigen Sie, dass er nur dann gleich dem Einschrittziel von $Q^*$ ist, wenn $a_{t+1}$ mit Wahrscheinlichkeit eins eine Greedy-Aktion für $Q^*$ ist. Den Wert welcher Strategie schätzt das $n$-Schritt-Ziel?
    1. Bei den $n$-Schritt-Erträgen aus :numref:`sec_actorcritic` stellte sich die Frage aus Teil (c) nicht. Warum?
   :exlabel:`ex_dqn-nstep-replay`

<!--en:b24c3ccb-->
1. [code] **Bairds Gegenbeispiel mit Zielnetz.** ●● DQN berechnet seinen Bootstrap-Wert mit einer eingefrorenen Kopie der Gewichte. Ändern Sie die Berechnung von Bairds Gegenbeispiel in :numref:`subsec_dqn-deadly-triad`. Berechnen Sie `delta` aus einer Kopie `w_target` mit dem Ausdruck `gamma * (Phi @ w_target)[6] - Phi @ w` und setzen Sie `w_target = w.copy()` alle $k$ Durchläufe.
    1. Führen Sie $100{,}000$ Durchläufe für $k \in \{1, 10, 30, 100, 300, 1000\}$ aus. Zeichnen Sie die Supremumsnorm von $\mathbf{w}$ und den größten Fehler der Wertschätzungen $\max_i |(\Phi \mathbf{w})_i|$. Für welche $k$ konvergiert die Iteration, und wie hängt ihre Geschwindigkeit von $k$ ab? Warum nähert sich die Supremumsnorm von $\mathbf{w}$ nicht null, wenn die Werte gegen null gehen?
    1. Zeigen Sie, dass die sieben Zeilen von `Phi` linear unabhängig sind. Damit lässt sich jeder Vektor von Zielwerten durch ein geeignetes $\mathbf{w}$ exakt darstellen. Leiten Sie daraus ab: Würde die Regression zwischen zwei Synchronisationen exakt gelöst, so würde jede Synchronisation die Werte $\mathbf{v}$ auf $\gamma P \mathbf{v}$ abbilden, unabhängig von der Wahl positiver Zustandsgewichte. Erklären Sie damit das Verhalten an beiden Enden der Versuchsreihe aus Teil (a).
    1. Im Gegenbeispiel von Tsitsiklis und Van Roy haben zwei nichtterminale Zustände die Werte $w$ und $2w$ mit einem einzigen Gewicht $w$. Jede Belohnung ist null. Der erste Zustand geht in den zweiten über. Im zweiten Zustand bleibt der Prozess mit Wahrscheinlichkeit $1 - \xi$, andernfalls terminiert er. Gewichten Sie beide Zustände gleich. Zeigen Sie, dass die exakte Anpassung nach der Methode der kleinsten Quadrate an die Zielwerte eines eingefrorenen Gewichts $w_k$ die Aktualisierung $w_{k+1} = \gamma (6 - 4\xi)\, w_k / 5$ ergibt. Zeigen Sie, dass die Semigradienteniteration ohne Zielnetz unter derselben Bedingung an $\gamma$ und $\xi$ divergiert.
    1. Beide Beispiele enthalten alle drei Bestandteile der Triade. Ein Zielnetz verhindert nur in einem der beiden Beispiele die Divergenz. In welcher Eigenschaft unterscheiden sich die Funktionsklassen der beiden Beispiele?

    *Nach Richard S. Sutton und Andrew G. Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf), Beispiel 11.1.*
   :exlabel:`ex_dqn-baird-target-network`

<!--en:a9593da7-->
1. [extended] **LayerNorm anstelle eines Zielnetzes.** ●● :citet:`Gallici.Fellows.Ellis.ea.2025` beweisen, dass ein hinreichend breites Q-Netz mit LayerNorm beim Training mit einem $\ell_2$-Strafterm für die Gewichte der letzten Schicht eine Stabilitätsbedingung erfüllt. Diese gilt für das Lernen mit zeitlichen Differenzen ohne Zielnetz, selbst mit Off-Policy-Daten. In ihren Experimenten genügte LayerNorm allein in jeder Umgebung außer in Bairds Gegenbeispiel.
    1. Fügen Sie nach jeder versteckten linearen Schicht von `make_qnet` vor der ReLU eine LayerNorm-Schicht ein. Wiederholen Sie den Versuch ohne Zielnetz mit drei Zufallsstartwerten. Vergleichen Sie die Verläufe der Werte des Startzustands und die Ergebnisse der Greedy-Strategien mit beiden Varianten aus :numref:`subsec_dqn-ablation`.
    1. Ergänzen Sie dieselben Schichten auch in der DQN-Variante, die ihr Zielnetz behält. Was ändert LayerNorm dort?
    1. Divergiert ein Lauf aus Teil (a) weiterhin, ergänzen Sie einen $\ell_2$-Strafterm für die Gewichte der letzten linearen Schicht. Bestimmen Sie den kleinsten Koeffizienten, der die Divergenz verhindert. Wie wirkt sich ein großer Koeffizient auf die gelernten Werte aus?
    1. Der Satz betrifft hinreichend breite Netze und eine regularisierte Aktualisierung. Welche Annahmen erfüllt Ihr Experiment? Welche Ergebnisse aus den Teilen (a) bis (c) würden der Aussage der Arbeit widersprechen, und welche lägen lediglich außerhalb ihres Geltungsbereichs?
   :exlabel:`ex_dqn-layernorm`

<!--en:42be191d-->
[Diskussion](https://d2l.discourse.group/t/36804)

<!--en:486893f4-->
Deep Q-Networks, Replay-Puffer und Double DQN<br>**tödliche Triade · Bairds Gegenbeispiel · Experience-Replay · Zielnetze · Maximierungsverzerrung · Double DQN**

<!--en:09051ca5-->
::: {.slide title="Warum naives neuronales Q-Learning Werte über 10^8 liefert"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:cd1bcb68-->
- **Sich verstärkende Rückkopplung:** Bei CartPole ($V^\infty = 100$ für $\gamma = 0.99$) erhöht die direkte Regression von $Q_w(s, a)$ auf $1 + \gamma \max_{a'} Q_w(s', a')$ über die gemeinsamen Gewichte auch $Q_w(s', \cdot)$. Die Vorhersagen steigen über **$10^8$**, während der Ertrag der Greedy-Strategie auf $\sim 9$ fällt.

<!--en:6458e095-->
- **Der Durchbruch mit DQN (Mnih et al., 2015):** Experience-Replay und ein eingefrorenes Zielnetz $Q_{w^-}$ stabilisieren das Off-Policy-Lernen von Werten.

<!--en:1748ced5-->
::: {.slide title="Warum Semigradientenschritte und kleinste Quadrate in Konflikt geraten"}

<!--en:470751d7-->
[Ursachen der Instabilität]{.kicker}

<!--en:6e52c0f2-->
- **Veränderliche Zielwerte:** Semigradientenschritte halten $y$ fest, während sie $w$ aktualisieren. Die Schrittfolge entspricht daher keinem Gradientenabstieg auf einer festen Verlustfunktion.

<!--en:fa09cfc0-->
- **Unterschiedliche Normen:** Das Bellman-Backup kontrahiert in der Supremumsnorm $\|\cdot\|_\infty$. Die Regression mit einem Netz projiziert dagegen in einer nach Besuchshäufigkeit gewichteten $L_2(d)$-Norm. Dabei kann $\|\cdot\|_\infty$ wachsen.

<!--en:d4f25bd9-->
::: {.slide title="Die tödliche Triade und Bairds Gegenbeispiel"}

<!--en:db0f0718-->
[Theoretische Grundlagen]{.kicker}

<!--en:bc6dbafd-->
- **Die tödliche Triade (Sutton & Barto, 2018):** Treffen **Funktionsapproximation**, **Bootstrapping** und **Off-Policy-Daten** zusammen, kann das Verfahren instabil werden.

<!--en:3bd7a315-->
- **Bairds Gegenbeispiel mit 7 Zuständen (1995):** Lineare Wertfunktion $v = \Phi \mathbf{w}$ (8 Gewichte, 7 Zustände), überall Belohnung null ($V^* = \mathbf{0}$, durch $\mathbf{w} = \mathbf{0}$ exakt darstellbar) und exakte erwartete Aktualisierungen (kein Rauschen).

<!--en:824ae128-->
- Bei gleichmäßiger Off-Policy-Gewichtung der Zustände und einer bewerteten Strategie, die immer in Zustand 7 springt, divergiert $\|\mathbf{w}\|_\infty$ exponentiell ($10 \to 335$).

<!--en:800afc5b-->
::: {.slide title="Die Algorithmen innerhalb der tödlichen Triade"}

<!--en:e9c6b82f-->
[Einteilung der Algorithmen]{.kicker}

<!--en:1092ca7e-->
- **Mit je zwei Bestandteilen ist Konvergenz möglich:**
  - *Bootstrapping + Off-Policy (ohne Netz):* **Tabellarisches Q-Learning** konvergiert dank der Kontraktion in $\|\cdot\|_\infty$.
  - *Funktionsapproximation + Bootstrapping (On-Policy):* **Lineares TD($0$) / Akteur-Kritiker-Verfahren** konvergiert unter der stationären Verteilung der bewerteten Strategie (Tsitsiklis & Van Roy, 1997).
  - *Funktionsapproximation + Off-Policy (ohne Bootstrapping):* **Verhaltensklonen** ist gewöhnliches überwachtes Lernen mit festen Zielwerten.

<!--en:1334c912-->
- **Gemeinsame Schnittmenge:** DQN und bestärkendes Offline-Lernen verbinden alle drei Bestandteile und brauchen daher eine Architektur, die das Lernen stabilisiert.

<!--en:9bd23766-->
::: {.slide title="Zwei Elemente der Architektur: Experience-Replay und Zielnetze"}

<!--en:d9229db3-->
[DQN-Architektur]{.kicker}

<!--en:dcc216fe-->
- **1. Replay-Puffer:** Speichert Übergänge $(s, a, r, s', \textrm{term})$ in einem Ringpuffer und zieht Minibatches gleichverteilt. Das bricht die zeitliche Korrelation auf und nutzt jeden Übergang dutzende Male.

<!--en:0198cf3d-->
- **2. Eingefrorenes Zielnetz ($Q_{w^-}$):** Berechnet den Bootstrap-Wert $\max_{a'} Q_{w^-}(s', a')$ mit einer Kopie der Gewichte. Diese wird nur alle $C = 250$ Umgebungsschritte synchronisiert. Damit bleiben die Zielwerte für $125$ Gradientenschritte fest.

<!--en:6f93c5df-->
::: {.slide title="Experience-Replay mit vorab reserviertem Ringpuffer" layout="tight"}

<!--en:db513a2b-->
[Funktionsweise des Replay-Puffers]{.kicker}

<!--en:cfc13c85-->
- **Ringpuffer ($200{,}000$ Plätze):** Überschreibt den ältesten Übergang und schaltet mit `ptr = (ptr + 1) % cap` weiter. Zieht Minibatches aus je $128$ unkorrelierten Fragmenten der Länge 1 gleichverteilt.

<!--en:f30d83c6-->
::: {.slide title="Experience-Replay mit vorab reserviertem Ringpuffer (Code)" layout="tight"}

<!--en:db513a2b-2-->
[Funktionsweise des Replay-Puffers]{.kicker}

<!--en:fe7c93aa-->
::: {.slide title="Regression mit Zielnetz und begrenztem Gradienten des Huber-Verlusts"}

<!--en:90a8fefb-->
[Implementierung der Aktualisierung]{.kicker}

<!--en:79e8791c-->
Zwischen den Synchronisationen (alle $C = 250$ Schritte) löst $Q_w$ ein stationäres Regressionsproblem mit Huber-Verlust. Die Gradientennorm ist auf `10.0` begrenzt:

<!--en:8320039f-->
::: {.slide title="Die DQN-Trainingsschleife für CartPole" layout="tight"}

<!--en:b0a65151-->
[Trainingsschleife]{.kicker}

<!--en:d77fcefa-->
Nach $1{,}000$ Anlaufschritten folgt `train_dqn` einer $\epsilon$-Greedy-Strategie ($\epsilon: 1.0 \to 0.05$ über $10\text{k}$ Schritte). Alle `train_freq = 2` Schritte aktualisiert die Funktion das Netz und synchronisiert gemäß $w^- \leftarrow w$ alle `sync_every = 250` Schritte (bei `use_target=False` nach jedem Schritt):

<!--en:4620fd49-->
::: {.slide title="Die DQN-Trainingsschleife für CartPole (Fortsetzung)" layout="tight"}

<!--en:b0a65151-2-->
[Trainingsschleife]{.kicker}

<!--en:7df0f4b2-->
In jedem Umgebungsschritt wählt der Agent eine Aktion und speichert den Übergang. Anschließend trainiert und synchronisiert er die Netze in den festgelegten Abständen:

<!--en:7f1aa261-->
Stets `float(terminated)` speichern, niemals `terminated or truncated`. So nutzt der Agent bei Episoden, die das Zeitlimit von 500 CartPole-Schritten erreichen, weiterhin den Bootstrap-Wert für die Fortsetzung.

<!--en:405891d6-->
::: {.slide title="Ertrag der Verhaltensstrategie mit und ohne separates Zielnetz"}

<!--en:2f6749f2-->
[Vergleich mit und ohne Zielnetz]{.kicker}

<!--en:ea26f3b3-->
- Mit eingefrorenem Zielnetz, das alle $250$ Schritte synchronisiert wird (**DQN**), steigt der Ertrag bei jedem Zufallsstartwert vom Niveau der Zufallsstrategie ($\sim 20$) auf Hunderte von Schritten. Die Exploration mit $\epsilon = 0.05$ wählt weiterhin $1$ von $20$ Aktionen zufällig.

<!--en:4461ad27-->
- **Ohne Zielnetz** (`use_target=False`, Synchronisation nach jedem Schritt) bricht die Episodenlänge bei jedem Zufallsstartwert auf $\sim 9$ Schritte ein und erholt sich nicht mehr.

<!--en:fd8ab646-->
::: {.slide title="Schranke für die Wertausbreitung über mehrere Synchronisationen"}

<!--en:81350fdb-->
[Warum das Zielnetz wirkt]{.kicker}

<!--en:2a98ddb3-->
- Ohne Zielnetz erhöhen gemeinsame Gewichte $Q_w(s', \cdot)$ zusammen mit $Q_w(s, a)$. Das führt zu $\max_a Q_w(s_0, a) > 10^8$.

<!--en:094c384d-->
- **Schranke nach der Synchronisation:** Beginnen die Werte bei $0$ und treffen sie die Zielwerte zwischen den Synchronisationen exakt, gilt nach $n$ Synchronisationen mit $r=1$ für jeden Wert:
  $$Q \le \sum_{k=0}^{n-1} \gamma^k = \frac{1 - \gamma^n}{1 - \gamma}$$

<!--en:237ccba5-->
- Bei $50{,}000$ Schritten ($n \approx 196$ Synchronisationen nach der Anlaufphase) beträgt die Schranke $\approx 86$. Durch die Generalisierung des Netzes steigt $\max_a Q_w(s_0, a)$ knapp über diese Kurve in Richtung der wahren Obergrenze $100$.

<!--en:a719fa6e-->
::: {.slide title="Regelmäßige Auswertung der Greedy-Strategie bei festen Schrittzahlen"}

<!--en:024ded35-->
[Strategiebewertung]{.kicker}

<!--en:0aeed4eb-->
Die Ertragskurven im Training vermischen explorative $\epsilon$-Greedy-Aktionen mit Verbesserungen der Strategie. Die deterministische **Greedy-Strategie** ($\operatorname{argmax}_a Q_w(s, a)$) wird alle $5{,}000$ Schritte für 10 Episoden ausgewertet. So entsteht ein unverzerrtes Bild des Fortschritts:

<!--en:9f7466b0-->
- **Fortschritt mit DQN:** Der Ertrag der Greedy-Strategie liegt in den ersten Schritten bei etwa $50$–$250$. Diese Phase dauert $30{,}000$ Schritte. Die Obergrenze von $500$ Schritten erreichen die meisten Läufe zwischen $35{,}000$ und $45{,}000$ Schritten. Einige fallen vorübergehend zurück.

<!--en:f042b146-->
- **Ohne Zielnetz:** Bei keiner Auswertung steigt die Episodenlänge über $\sim 10$ Schritte, auch nicht bis zum Ende des Laufs nach $50{,}000$ Schritten.

<!--en:7122e43b-->
::: {.slide title="Wertschätzungen im Vergleich mit tatsächlich erzielten Erträgen"}

<!--en:b9106305-->
[Kalibrierung der Werte]{.kicker}

<!--en:d38551fc-->
Ein hoher Ertrag der Greedy-Strategie belegt die Güte der *Rangfolge für das Argmax*. Die Genauigkeit der *Wertvorhersagen* bleibt offen. Nach van Hasselt, Guez & Silver (2016) vergleicht `audit` daher $\max_a Q_w(s_0, a)$ mit dem tatsächlichen diskontierten Ertrag der Strategie über 500 Schritte (Obergrenze $(1 - 0.99^{500})/0.01 \approx 99.3$):

<!--en:1578fc72-->
- In erfolgreichen Läufen weicht $Q_w(s_0, a)$ höchstens $\sim 8$ Punkte von $99.3$ ab. In fehlgeschlagenen Läufen kippt der Stab früh um, und das Netz **überschätzt** den eigenen diskontierten Ertrag häufig um $20+$ Punkte.

<!--en:97b51fe1-->
::: {.slide title="Maximierungsverzerrung: Warum das Maximum zu hoch ausfällt"}

<!--en:bc589876-->
[Systematische Überschätzung]{.kicker}

<!--en:4c82cfe3-->
- Nach der Jensen-Ungleichung ist das Maximum verrauschter Schätzungen nach oben verzerrt: $\mathbb{E}[\max_a \hat{Q}(s', a)] \ge \max_a \mathbb{E}[\hat{Q}(s', a)]$ (Thrun & Schwartz, 1993).

<!--en:17577cfb-->
- Für $m = 4$ Aktionen mit wahrem Wert $0$ und unabhängigem $\mathcal{N}(0, 1)$-Rauschen liefert der Einzelschätzer $\max_a \hat{Q}(a)$ im Mittel **$+1.03$**.

<!--en:f5941ee7-->
- **Der Doppelschätzer (van Hasselt, 2010):** Die Auswahl von $a^\star = \operatorname{argmax}_a \hat{Q}^{(1)}(a)$ mit einer Schätzung und die Bewertung durch $\hat{Q}^{(2)}(a^\star)$ aus einer unabhängigen zweiten Schätzung liefern den Erwartungswert **$0.000$** für jede Anzahl $m$ von Aktionen.

<!--en:e70eeab7-->
::: {.slide title="Double DQN trennt Auswahl und Bewertung"}

<!--en:85d73890-->
[Double DQN]{.kicker}

<!--en:391ab3f7-->
- **Getrennte Auswahl und Bewertung (van Hasselt et al., 2016):** Das vorhandene Online-Netz $Q_w$ dient zur *Auswahl* von $a^\star$, das Zielnetz $Q_{w^-}$ zur *Bewertung* der Aktion.

<!--en:6c00916a-->
::: {.slide title="Überschätzung sichtbar machen mit 18 Aktionen"}

<!--en:e6766613-->
[Wertprüfung mit 18 Aktionen]{.kicker}

<!--en:afd57b88-->
Steht jede der 2 Stoßrichtungen bei CartPole $9$-mal zur Wahl (`copies=9`, 18 Aktionen wie bei Atari), bleiben $Q^*$ und die Obergrenze $V^\infty = 100$ unverändert. Doch $\max_{a'}$ wählt nun die zufällig höchste von $9$ verrauschten Ausgaben pro Stoßrichtung:

<!--en:bfea2a5c-->
- Mit 18 Aktionen liegt **Standard-DQN am Ende bei jedem Zufallsstartwert $10$ bis $30$ Punkte über der theoretischen Obergrenze von $100$**. **Double DQN** senkt die Schätzungen um $10$–$25$ Punkte und bleibt in der Nähe von $100$.

<!--en:c83ebc57-->
::: {.slide title="Double DQN hält die Schätzungen bei 18 Aktionen nahe der Obergrenze"}

<!--en:cdbe23e5-->
[Verlauf der Wertschätzungen]{.kicker}

<!--en:816c0ca8-->
- Der Verlauf von $\max_a Q_w(s_0, a)$ beim Training auf CartPole mit 18 Aktionen zeigt, dass Standard-DQN die theoretische Obergrenze von $100$ schon vor $40{,}000$ Schritten überschreitet. Die Werte steigen weiter, weil die Maximierungsverzerrung über den Bootstrap-Wert zurückwirkt.

<!--en:5de72181-->
- Der entkoppelte Zielwert von Double DQN hält die Wertschätzung nahe der wahren Obergrenze von $100$ für die fortlaufende Aufgabe.

<!--en:a712b046-->
::: {.slide title="Verständnisfragen zu Triade, Synchronisation und Tokenräumen"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:1775cdbb-->
Drei Fragen zum Entwurf der Verfahren:

<!--en:3e0b5ed6-->
1. **Bestandteile der Triade:** Welche Bestandteile der tödlichen Triade fehlen bei (a) tabellarischem Q-Learning, (b) On-Policy-REINFORCE mit einem neuronalen Netz und (c) Verhaltensklonen?

<!--en:c976ac55-->
2. **Seltenere Synchronisation:** Das Intervall `sync_every` steigt von $C = 250$ auf $C = 1{,}000$. Das Budget beträgt $50{,}000$ Schritte, die Anlaufphase umfasst $1{,}000$ Schritte. Welche theoretische Obergrenze $(1 - \gamma^n)/(1 - \gamma)$ gilt für $Q(s_0, a)$ nach $n = 49$ Synchronisationen bei $\gamma = 0.99$?

<!--en:37199cdf-->
3. **Q-Learning für große Sprachmodelle:** Warum eignet sich ein Standard-DQN nicht für das Training auf dem Vokabular eines Sprachmodells mit $m = 10^5$ Token?

<!--en:557b98b6-->
**Lösungen:**

<!--en:58007ad6-->
1. (a) Keine **Funktionsapproximation**, (b) kein **Bootstrapping** und keine **Off-Policy-Daten**, (c) kein **Bootstrapping**.

<!--en:9347ea17-->
2. Bei $n = 49$ Synchronisationen gilt $(1 - 0.99^{49}) / 0.01 \approx \mathbf{38.9}$. Sehr seltene Synchronisationen stabilisieren die Regression, bremsen aber die Ausbreitung der Werte.

<!--en:9dc336cb-->
3. Das erwartete Maximum von $m = 10^5$ unabhängigen standardnormalverteilten Rauschvariablen beträgt $\approx \mathbf{4.4\sigma}$. Die Obergrenze liegt bei $\sqrt{2 \ln 10^5} \approx 4.8\sigma$. Zudem bleibt $Q_{w^-}$ stark mit $Q_w$ korreliert. Das führt bei $10^5$ Ausgabeköpfen zu einer erheblichen Maximierungsverzerrung.

<!--en:c5f03741-->
::: {.slide title="Zusammenfassung: stabiles Off-Policy-Lernen mit tiefen Wertnetzen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:e6a8810d-->
| Bestandteil / Variante | Funktionsweise | Verhindertes Problem | Aufwand / Einschränkung |

<!--en:2c1036ed-->
| **Experience-Replay** | Minibatches gleichverteilt aus dem Ringpuffer ziehen | Zeitliche Korrelation und katastrophales Vergessen | Erfordert Off-Policy-Zielwert, nutzt Fragmente der Länge 1 |

<!--en:226bca0c-->
| **Zielnetz ($Q_{w^-}$)** | Eingefrorene Kopie alle $C = 250$ Schritte synchronisieren | Sich verstärkende Rückkopplung ($Q(s_0) > 10^8$) | Belohnung breitet sich pro Synchronisation um $1$ Schritt im Horizont aus |

<!--en:3d051f78-->
| **Terminierungsmaske** | Bootstrap-Wert bei `terminated` maskieren, bei `truncated` beibehalten | Das Netz lernt fälschlich, dass Überleben bis Schritt 500 den Wert $0$ hat | Lernt den Wert der fortlaufenden Aufgabe $V^\infty = 1/(1-\gamma) = 100$ |

<!--en:ebfe42b6-->
| **Double DQN** | $Q_{w^-}(s', \operatorname{argmax}_{a'} Q_w(s', a'))$ | Maximierungsverzerrung nach oben bei vielen Aktionen | $1$ zusätzlicher Vorwärtsdurchlauf des Online-Netzes pro Aktualisierung |
