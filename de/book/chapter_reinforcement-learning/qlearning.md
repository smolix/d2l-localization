<!--en:4fb31a04-->
# Lernen aus zeitlichen Differenzen

<!--en:b1663aa0-->
Im vorigen Abschnitt haben wir den besten Weg über den rutschigen See mithilfe seiner Übergangstabelle gefunden. Für jede Zelle und jeden Befehl haben wir über die drei möglichen Zellen summiert, in die der Agent rutschen könnte. Auf einem echten See hat der Agent keine solche Tabelle. Er kann nur handeln, beobachten, wo er landet, und die jeweilige Belohnung entgegennehmen. Bei jedem Schritt sieht er einen einzigen Folgezustand, den die Umgebung auswählt. Die übrigen Möglichkeiten und ihre Wahrscheinlichkeiten kennt er nicht. Kann er aus solchen einzelnen Beobachtungen dennoch lernen, wie gut jeder Befehl ist, und daraus die beste Strategie (engl. policy) ableiten?

<!--en:2f632049-->
Für die Vorhersage von Werten fand Richard Sutton 1988 die Antwort, für die Wahl von Aktionen Chris Watkins 1989 :cite:`Sutton.1988,Watkins.1989,Watkins.Dayan.1992`. Der beobachtete Folgezustand liefert eine Schätzung der Summe, die der Agent nicht berechnen kann. Der Agent verschiebt den Tabelleneintrag um einen kleinen Schritt in Richtung dieser Schätzung. Über viele Besuche hinweg mitteln diese Schritte die Samples. Ihr Erwartungswert entspricht dem Backup der dynamischen Programmierung. Das so entstehende Verfahren, *Q-Learning*, benötigt kein Modell der Umgebung. Ersetzt ein neuronales Netz die Tabelle, erhalten wir das Deep Q-Learning des nächsten Kapitels.

<!--en:b4e04cad-->
Das Lernen aus Samples wirft zwei Fragen auf. Die Samples sind verrauscht, daher müssen die Schritte mit zunehmender Besuchszahl kleiner werden. Wie schnell? Außerdem bleibt der Eintrag für einen Befehl unverändert, solange der Agent ihn nie ausprobiert. Der Agent muss also immer wieder Befehle ausprobieren, die seine Tabelle schlecht bewertet. Die erste Frage beantworten wir in diesem Abschnitt. Für die zweite verwenden wir die einfachste Regel: Mit geringer Wahrscheinlichkeit handelt der Agent zufällig. Was diese Regel kostet, untersucht :numref:`sec_exploration`. Zum Schluss nutzen wir dieselben Übergänge auf eine zweite Weise: Wir schätzen die Übergangstabelle selbst und planen mit dieser Schätzung.

<!--en:4daebbe4-->
## Backup aus einem Sample

<!--en:b77db790-->
Wir gehen von der Gleichung aus, die wir mit der Wertiteration gelöst haben, und fragen, für welchen Teil wir das Modell brauchen.

<!--en:cd00a1e3-->
### Vom Bellman-Operator zum einzelnen Sample

<!--en:25694936-->
Die Wertiteration (engl. value iteration) in :numref:`sec_valueiter` wiederholte dieselbe Operation, das Bellman-Backup für Optimalität. Schreiben wir den Bellman-Operator $\mathcal{T}$ für Aktionswerte (engl. action values) statt für Zustandswerte (engl. state values), so bildet er eine Tabelle $Q$ auf die folgende Tabelle ab:

<!--en:808dd994-->
Jeder Eintrag von $\mathcal{T}Q$ ist die erwartete unmittelbare Belohnung zuzüglich des diskontierten Werts, den der Agent erzielt, wenn er vom erreichten Zustand aus gierig gemäß $Q$ handelt. Der Fixpunkt dieses Operators ist $Q^*$. Von jeder Ausgangstabelle aus konvergiert die wiederholte Anwendung gegen $Q^*$, denn $\mathcal{T}$ verkleinert die größte Differenz zwischen zwei Tabellen um den Faktor $\gamma$. Das ist das Kontraktionsargument aus :numref:`sec_valueiter`, angewandt auf Aktionswerte.

<!--en:72090934-->
Das Backup braucht das Modell an einer Stelle: beim Mitteln über den Folgezustand $s'$. Einen Mittelwert können wir aus Samples schätzen. Angenommen, der Agent steht in $s$, gibt den Befehl $a$, erhält die Belohnung $r$ und landet in $s'$. Die Zahl $r + \gamma \max_{a'} Q(s', a')$ ist zufällig. Auf dem See hängt auch die Belohnung davon ab, wo der Agent landet. Mitteln wir über Folgezustand und Belohnung, ergibt sich $(\mathcal{T}Q)(s, a)$. Ein einziger Übergang liefert also eine unverzerrte Schätzung eines Eintrags des Backups.

<!--en:9f0bcfa7-->
Deshalb lernt die modellfreie Steuerung $Q$ statt $V$. Beim Backup für Zustandswerte $V(s) \leftarrow \max_a [r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s')]$ steht das Maximum außerhalb des Mittelwerts über $s'$. Ein Übergang zeigt nur, was eine Aktion bewirkt hat. Über die Klammerausdrücke der übrigen Aktionen sagt er nichts aus, also lässt sich ihr Maximum daraus nicht schätzen. In der $Q$-Form hat jede Aktion einen eigenen Eintrag. Der Mittelwert gehört zu diesem Eintrag, und das Maximum bilden wir nur über Zahlen, die schon in der Tabelle stehen. In dieser Form lässt sich auch die gierige Strategie $\mathrm{argmax}_a Q(s, a)$ aus einer einzigen Tabellenzeile ablesen, wiederum ohne Modell.

<!--en:79d5761f-->
Q-Learning verschiebt den besuchten Eintrag um den Anteil $\alpha$ der Strecke zum beobachteten Sample. Dieses Sample ist sein *Zielwert* (engl. target):

<!--en:c5d39adc-->
Die Schrittweite (engl. step size) $\alpha \in (0, 1]$ bestimmt die Größe der Änderung. Bei $\alpha = 1$ ersetzt der Zielwert den Eintrag, bei kleinem $\alpha$ mitteln wir über viele Zielwerte. Da der Zielwert auf dem Tabelleneintrag für den nächsten Zustand beruht, sprechen wir von *Bootstrapping*. Stellen wir die Gleichung um, so ist die Änderung proportional zu einer einzigen Zahl:

<!--en:f043993a-->
Die Zahl $\delta$ heißt *Temporal-Difference-Fehler* :cite:`Sutton.1988`. Sie gibt an, was das Paar einen Schritt später wert zu sein scheint, abzüglich des Werts, der vor dem Schritt in der Tabelle stand. Teil (d) von :numref:`fig_rl_backups` zeigt die Ersetzung. Das exakte Backup summiert über alle Folgezustände. Das Backup aus einem Sample folgt dem einzigen beobachteten Übergang und behält das Maximum im nächsten Zustand bei.

<!--en:a989c960-->
Bootstrapping ist eine Möglichkeit. Eine *Monte-Carlo*-Schätzung wartet bis zum Ende der Episode und verwendet den tatsächlich erzielten Ertrag (engl. return) als Zielwert für $Q(s_t, a_t)$. Dieser Zielwert schätzt den Wert der Strategie, die die Episode erzeugt hat, unverzerrt. In ihn gehen aber die Zufallseinflüsse aller späteren Übergänge ein. Ein Zielwert aus einer *zeitlichen Differenz* blickt einen Schritt voraus und vertraut für den Rest auf die Tabelle. Solange die Tabelle ungenau ist, ist er verzerrt, schwankt aber meist erheblich weniger :cite:`Sutton.1988,Sutton.Barto.2018`. Q-Learning nimmt das Maximum in den Einschritt-Zielwert auf und schätzt damit $Q^*$ statt des Werts der Strategie, die die Daten sammelt.

<!--en:8cc3e9a3-->
### Einen Übergang von Hand berechnen

<!--en:08c4aa20-->
Wir beginnen auf dem rutschigen See mit einer Tabelle voller Nullen, mit $\gamma = 0.95$ und der Einfachheit halber $\alpha = 0.5$. Bis der Agent erstmals das Ziel erreicht, ist jeder Zielwert null: Überall sonst ist die Belohnung null, ebenso jeder Eintrag, auf den sich das Bootstrapping stützen könnte. Angenommen, der erste Erfolg geht von Zelle $14$ aus. Beim Befehl *rechts* kann der Agent von dort ins Ziel rutschen, also in Zelle $15$. Das geschieht mit Wahrscheinlichkeit $1/3$. Das Ziel ist ein Endzustand. Der Zielwert besteht daher allein aus der Belohnung, und es gilt

<!--en:dcdaea37-->
Alle anderen Tabelleneinträge bleiben gleich. Später steht der Agent auf Zelle $13$, wählt *rechts* und rutscht nach $14$. Sein Zielwert ist $0 + 0.95 \cdot \max_{a'} Q(14, a') = 0.95 \cdot 0.5 = 0.475$, also gilt $Q(13, \textrm{right}) \leftarrow 0.5 \cdot 0.475 \approx 0.24$. Vor der Aktualisierung in $14$ hätte derselbe Übergang von $13$ aus den Agenten nichts gelehrt. So breitet sich der Wert vom Ziel aus rückwärts aus, bei jedem Besuch um eine Zelle entlang der Übergänge, die der Agent zufällig durchläuft. Bei der Wertiteration geschah das um eine Zelle pro Durchlauf, entlang aller Übergänge zugleich.

<!--en:50e0bd8c-->
Die Zielwerte sind verrauscht. Gibt der Agent als Nächstes auf $14$ den Befehl *rechts* und rutscht nach oben auf Zelle $10$, deren Einträge noch null sind, dann ist der Zielwert $0$ und es gilt $Q(14, \textrm{right}) \leftarrow 0.25$. Der Eintrag springt zwischen Zielwerten von $1$ und $0$, später zwischen Werten, die das Bootstrapping aus den Zellen $10$ und $14$ liefert. Erst der Mittelwert dieser Zielwerte entspricht dem Backup. Wie schnell die Schritte zum Bilden dieses Mittelwerts schrumpfen sollten, untersuchen wir weiter unten bei den Schrittweiten.

<!--en:cb983037-->
### Kleinste Quadrate und doppeltes Sampling

<!--en:e3b7ae62-->
Warum sollten uns solche verrauschten Schritte überhaupt weiterbringen? Halten wir die Tabelle fest und mitteln den Temporal-Difference-Fehler über Folgezustand und Belohnung:

<!--en:d22e6f56-->
Die erwartete Aktualisierung $Q(s, a) \leftarrow Q(s, a) + \alpha \big((\mathcal{T}Q)(s, a) - Q(s, a)\big)$ ist ein gedämpfter Schritt der Wertiteration für genau diesen Eintrag. Q-Learning führt die Wertiteration also für jeweils einen per Sampling ausgewählten Eintrag aus. Jeden Schritt überlagert ein Rauschen mit Mittelwert null.

<!--en:201cf635-->
Wer schon Regressionsmodelle angepasst hat, könnte einwenden, dass zur Anpassung einer Tabelle an Zielwerte die Methode der kleinsten Quadrate naheliegt. Bei $N$ aufgezeichneten Übergängen $(s_i, a_i, r_i, s_i')$ könnten wir den quadratischen Temporal-Difference-Fehler durch Gradientenabstieg minimieren:

<!--en:adac7b82-->
Der Gradient eines Summanden hat zwei Anteile. Einer ergibt sich über $Q(s_i, a_i)$, der andere über das Maximum im Zielwert. Die Aktualisierung :eqref:`eq_td_error` behält den ersten Anteil bei und lässt den zweiten weg, indem sie den Zielwert festhält. Sie heißt deshalb *Semigradientenschritt* und entspricht keinem Gradientenabstieg auf $\ell$.

<!--en:23f30319-->
Gerade das Weglassen des zweiten Anteils macht Q-Learning korrekt. Nehmen wir an, es liegen unendlich viele Beobachtungen vor. Jedes Paar $(s, a)$ kommt mit der Häufigkeit $d(s, a)$ vor, die der Besuchsverteilung der datensammelnden Strategie entspricht. Bedingen wir auf $(s, a)$, zerfällt der Erwartungswert jedes quadrierten Terms in das Quadrat seines Mittelwerts und die Varianz des Zielwerts:

<!--en:eefb0cad-->
Der erste Term ist das quadrierte Bellman-Residuum. Bei $Q^*$ ist er null. Der zweite ist die Varianz des Zielwerts über Folgezustand und Belohnung. Bei zufälligen Übergängen hängt sie von $Q$ ab. Wer $L$ minimiert, wägt ein kleines Residuum gegen einen weniger stark schwankenden Zielwert ab. Der Minimierer muss daher nicht $Q^*$ sein. Drei Zustände zeigen die Richtung der Verzerrung. Von $s$ aus führt die einzige Aktion mit Wahrscheinlichkeit $1/2$ jeweils nach $u_1$ mit anschließender Belohnung $1$ oder nach $u_2$ mit anschließender Belohnung $0$. Danach endet die Episode in beiden Fällen. Also gilt $Q^*(u_1) = 1$, $Q^*(u_2) = 0$ und $Q^*(s) = \gamma/2$. Die Varianz des Zielwerts in $s$ beträgt $\gamma^2 (Q(u_1) - Q(u_2))^2/4$. Der Minimierer von $L$ senkt sie, indem er die Werte der beiden Folgezustände einander annähert, weg von $1$ und $0$. Wie weit, berechnen Sie in :exref:`ex_qlearning-double-sampling-counterexample`.

<!--en:2c90a146-->
Residualgradientenverfahren folgen dem negativen vollständigen Gradienten des quadrierten Residuums und stoßen auf dieselbe Schwierigkeit :cite:`Baird.1995`. Eine unverzerrte Schätzung dieses Gradienten erfordert zwei unabhängige Folgezustände desselben Paars, einen für den Zielwert und einen für die Ableitung. Ein Agent sieht auf seiner Trajektorie aber nur einen pro Übergang. Das ist das *Problem des doppelten Samplings*. Bei deterministischen Übergängen und Belohnungen verschwindet der Varianzterm, und beide Ansätze stimmen überein. Bei neuronalen Netzen begegnet uns das Festhalten des Zielwerts wieder. Eine zweite, langsam aktualisierte Kopie des Netzes berechnet die Zielwerte: das Zielnetz aus :numref:`sec_dqn`.

<!--en:ca1b4742-->
### Schrittweiten und Konvergenz

<!--en:8902acb0-->
In unserem Rechenbeispiel erhielt ein Eintrag eine Folge verrauschter Zielwerte $y_1, y_2, \ldots$. Am einfachsten fassen wir sie zusammen, indem wir sie mitteln. Mit der Schrittweite $\alpha_n = 1/n$ beim $n$-ten Besuch des Paars lautet die Aktualisierung $Q_n = Q_{n-1} + (y_n - Q_{n-1})/n = \big((n-1) Q_{n-1} + y_n\big)/n$. Durch Induktion folgt

<!--en:d0e9f938-->
also unabhängig vom Anfangswert der Stichprobenmittelwert der Zielwerte. Wären die Zielwerte unabhängige Ziehungen mit festem Mittelwert, würde der Eintrag nach dem Gesetz der großen Zahlen gegen diesen Mittelwert konvergieren. Robbins und Monro gaben Bedingungen an, unter denen eine abnehmende Schrittweite diese Wirkung auch im allgemeinen Fall hat :cite:`Robbins.Monro.1951`:

<!--en:d2b12f99-->
Die erste Bedingung erlaubt es, mit den Schritten jede beliebige Strecke zurückzulegen. So verliert der Anfangswert seinen Einfluss. Die zweite hält das über die Schritte angesammelte Rauschen endlich. Die Schrittweite $\alpha_n = 1/n$ des Stichprobenmittelwerts erfüllt beide Bedingungen. Eine konstante Schrittweite verletzt die zweite. Der Eintrag schwankt dann dauerhaft um den Mittelwert, mit einer Varianz der Größenordnung $\alpha$ für kleine $\alpha$.

<!--en:9d0da42b-->
**Satz (Konvergenz des tabellarischen Q-Learning).** *Die Zustands- und Aktionsmengen seien endlich, die Belohnungen beschränkt, und es gelte $\gamma < 1$. Wird jedes Paar $(s, a)$ unendlich oft aktualisiert und erfüllen seine Schrittweiten die Robbins-Monro-Bedingungen, so konvergiert $Q$ mit Wahrscheinlichkeit eins gegen $Q^*$* :cite:`Watkins.Dayan.1992,Jaakkola.Jordan.Singh.1994,Tsitsiklis.1994`.

<!--en:337d4354-->
Mittelung allein beweist diesen Satz nicht, denn die Zielwerte stammen aus der Tabelle, die wir gerade lernen, und haben keinen festen Mittelwert. Auch beim Residualgradientenabstieg verschwindet der erwartete Schritt an einem Fixpunkt, allerdings am falschen. Entscheidend für Q-Learning ist die Kontraktion. Unabhängig von den aktuellen Fehlern verkleinert der erwartete Schritt den größten Fehler. Die Bedingungen an die Schrittweiten sorgen dafür, dass sich das Rauschen um diese Schritte herausmittelt.

<!--en:a4fd3e94-->
Auf Eis ohne Rutschen führt jeder Befehl in die angegebene Richtung. Jedes Paar hat genau einen Folgezustand, der Zielwert enthält also kein Übergangsrauschen. Für konstantes $\alpha$ ist die Aktualisierung dann ein gedämpftes Backup, $Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha (\mathcal{T}Q)(s, a)$. Auf alle Einträge angewandt ist es eine Kontraktion mit dem Faktor $(1 - \alpha) + \alpha \gamma < 1$. Selbst eine so große Schrittweite wie $\alpha = 0.9$ ist dann unbedenklich.

<!--en:ba485de8-->
Die Bedingungen betreffen den Grenzwert. Bei endlichem Budget gibt der Stichprobenmittelwert jedem Zielwert dasselbe Gewicht, auch den frühen Zielwerten aus der anfangs fast vollständig mit Nullen gefüllten Tabelle. Auf dem See sind diese Zielwerte viel zu klein. Bei $\alpha_n = 1/n$ kann die Zahl der von Q-Learning benötigten Aktualisierungen exponentiell in $1/(1 - \gamma)$ wachsen, hier also in $20$. Bei $\alpha_n = 1/n^\omega$ mit $1/2 < \omega < 1$ wächst sie nur polynomiell :cite:`EvenDar.Mansour.2003`. Weiter unten testen wir $\alpha_n = 1/(1 + n)$. Das entspricht dem Stichprobenmittelwert, bei dem der Anfangswert null als weiterer Zielwert zählt. Unsere Standardeinstellung $\alpha_n = 1/(1 + 0.1\, n)$ erfüllt ebenfalls beide Bedingungen. Ihre Schrittweite nähert sich dem Zehnfachen der Schrittweite des Stichprobenmittelwerts. Sie gewichtet spätere Zielwerte stärker, den $k$-ten ungefähr proportional zu $k^9$ (siehe :exref:`ex_qlearning-step-sizes`). So folgt die Schätzung den zunehmend genaueren Zielwerten.

<!--en:04feb20a-->
### Maskierung bei Terminierung

<!--en:c124d2c4-->
Bei einer episodischen Aufgabe müssen wir den Zielwert an einer Stelle korrigieren. Ein Endzustand, hier ein Loch oder das Ziel, hat keine Zukunft. Sein Fortsetzungswert ist daher null, und der Zielwert enthält nur die unmittelbare Belohnung:

<!--en:bc0244f5-->
Die Maske verwendet `terminated` statt `truncated`. Wie :numref:`sec_mdp` erläutert, stoppt ein Zeitlimit die Aufzeichnung in einem Zustand, von dem aus der Prozess weiterliefe. Sein Fortsetzungswert ist also nicht null. Würden wir einen Abbruch wie eine Terminierung behandeln, würde die Tabelle lernen, dass die letzte Zelle jeder langen Episode wertlos ist. Im Code genügt für die Korrektur ein Faktor: `gamma * (1 - terminated) * Q[s_next].max()`.

<!--en:dc02fb55-->
## Q-Learning auf dem rutschigen See

<!--en:b1653528-->
Wir wenden die Aktualisierung nun auf den rutschigen See aus :numref:`sec_mdp` mit $\gamma = 0.95$ an und vergleichen das Ergebnis mit der exakten Lösung aus :numref:`sec_valueiter`. Das Lernverfahren greift nur über `reset` und `step` auf die Umgebung zu. Das Modell `mdp`, die optimalen Werte `V_star`, die optimalen Aktionswerte `Q_star` (ein Backup von $V^*$) und die optimale Strategie `pi_star` dienen ausschließlich der Auswertung.

<!--en:1a5e768e-->
### Die Verhaltensstrategie

<!--en:d884c02c-->
Q-Learning braucht eine Regel, nach der es während des Lernens Aktionen auswählt: seine *Verhaltensstrategie* (engl. behavior policy) $\pi_e$. Wer anhand der aktuellen Tabelle gierig handelt, stößt auf zwei Probleme. In einer mit Nullen gefüllten Tabelle sind alle vier Einträge jeder Zeile gleich, und `np.argmax` gibt den ersten Index zurück. Ein gieriger Agent würde also immer *links* wählen und nie das Ziel finden. Außerdem probiert er eine Aktion mit einem zu niedrigen Eintrag nie aus. Der Eintrag bekommt keine Aktualisierung, und die Unterschätzung bleibt bestehen. Die einfachste Abhilfe besteht darin, mit einer kleinen Wahrscheinlichkeit $\epsilon$ zufällig zu handeln:

<!--en:6da01a76-->
Das ist die *$\epsilon$-gierige* Strategie. Mit Wahrscheinlichkeit $\epsilon$ exploriert sie, andernfalls wählt sie die gierige Aktion. Bei mehreren Maxima wählt sie mit gleicher Wahrscheinlichkeit eines davon. Die Funktion zieht zunächst eine gleichverteilte Zufallszahl. Je nach Ergebnis wählt sie eine zufällige Aktion oder ein zufälliges Element aus der Menge der Maximierer:

<!--en:6e07d801-->
Solange die Tabelle noch keine Information enthält, sollte der Agent viel explorieren. Ist sie schon fast richtig, sollte er überwiegend das Gelernte nutzen. Deshalb lassen wir $\epsilon$ während des Trainings sinken. Aus einem Startwert, einem Endwert und einer Schrittzahl erzeugt `linear_schedule` eine Funktion der Schrittnummer. Sie interpoliert zunächst linear und behält danach den Endwert bei:

<!--en:f0852b4a-->
### Die Lernschleife

<!--en:605dc9dd-->
`q_learning` führt die Aktualisierung :eqref:`q_learning` mit der Maske für Endzustände aus. Die Funktion übernimmt die Tabelle `Q` sowie eine Tabelle `visits` der Besuchszahlen und ändert beide direkt. Über die Funktion `alpha(n)` bestimmen die Besuchszahlen die Schrittweite für jedes Paar. Danach lässt sich an ihnen ablesen, welche Paare der Agent ausprobiert hat. Die Explorationsrate ist eine Funktion `epsilon(k)` der Episodennummer. Standardmäßig sinkt sie über die erste Trainingshälfte von $1$ auf $0.05$ und bleibt danach bei $0.05$. Der Schleifenrumpf wählt eine Aktion, beobachtet einen Übergang und ändert einen Eintrag um $\alpha\delta$. Die Funktion liefert den Ertrag jeder Episode:

<!--en:efd8d8aa-->
Für alle Codezellen dieses Abschnitts genügt NumPy, denn eine Tabelle braucht nur Indexarithmetik. Wir trainieren für jeden Zufallsstartwert eine Tabelle über $4{,}000$ Episoden und zeichnen den Ertrag jeder Episode auf:

<!--en:10fd7323-->
Jeder Kurvenpunkt mittelt über die $200$ Episoden bis einschließlich der jeweiligen Episode. Die Kurve beginnt daher bei Episode $200$. Das Band zeigt die Spannweite über die Zufallsstartwerte. Solange Erfolge selten sind, bleibt der Ertrag nahe null. Er steigt, wenn sich der Wert vom Ziel aus rückwärts ausbreitet, und flacht nahe $0.5$ ab, sobald die Explorationsrate in Episode $2{,}000$ ihre Untergrenze erreicht.

<!--en:6ea8a72c-->
### Die gelernte Tabelle und die exakte Lösung

<!--en:523fcfc0-->
Ist ein Ertrag nahe $0.5$ gut? Die optimale Strategie aus :numref:`sec_valueiter` erreicht das Ziel innerhalb der Frist von $100$ Schritten in etwa $73\%$ der Fälle. Sie maximiert den diskontierten Ertrag, der Schnelligkeit belohnt. Eine Strategie ohne Diskontierung hat daher eine um etwa einen Prozentpunkt höhere Erfolgsquote. Außerdem misst die Trainingskurve den Ertrag der Verhaltensstrategie, die noch immer im Mittel in einem von zwanzig Schritten zufällig handelt. Wie gut die Tabelle selbst ist, erfahren wir so nicht.

<!--en:51a44728-->
Da wir das Modell hier kennen, können wir die Tabelle exakt bewerten. Bei $\gamma = 1$ und einer Belohnung von $1$ ausschließlich am Ziel ist der Wert des Startzustands die Wahrscheinlichkeit, das Ziel zu erreichen. Nach $100$ Durchläufen mit Anfangswert null liefert `policy_evaluation` aus :numref:`sec_valueiter` die Wahrscheinlichkeit, das Ziel innerhalb von $100$ Schritten zu erreichen. Die Funktion `success` kapselt diese Rechnung für eine Strategie mit einer Zeile von Aktionswahrscheinlichkeiten pro Zustand. Die Maske `live` markiert die Zellen, in denen der Agent handelt. In `greedy` steht die gierige Strategie jeder Tabelle. Die Codezelle gibt für jede Tabelle die Fehler der Zustands- und Aktionswerte aus, dazu den größten Wertverlust einer gierigen Aktion unter $Q^*$, die exakten Erfolgsquoten, die Erfolgsquote von $\pi^*$ bei zufälliger Aktionswahl mit Wahrscheinlichkeit $0.05$ sowie die Schrittzahl je Lauf:

<!--en:1499f299-->
Die dritte Zeile zeigt für jeden Zufallsstartwert null. In keiner Zelle verliert eine gelernte gierige Aktion unter $Q^*$ an Wert. Bis auf Gleichstände ist daher jede gierige Strategie optimal und erreicht genau die Erfolgsquote von $73\%$, die auch $\pi^*$ erzielt. Das ist bei diesem Budget typisch, aber nicht garantiert. In etwa jedem achten Lauf ist nach $4{,}000$ Episoden noch eine gierige Aktion suboptimal, meist mit geringen Einbußen (Erfolgsquote $71\%$ bis $74\%$). Im seltenen ungünstigsten Fall bevorzugt der Agent am Start noch *rechts* und hat in weniger als der Hälfte der Episoden Erfolg. Die beiden besten Aktionen unterscheiden sich dort um weniger als $0.01$ in $Q^*$. Die Schätzungen waren in diesem Lauf zu niedrig. Eine Unterschätzung korrigiert sich langsam, weil die gierige Regel die betreffende Aktion nur bei der Exploration ausprobiert. Über alle Zufallsstartwerte hinweg weichen die Zustandswerte um höchstens etwa $0.03$ von $V^*$ ab. Der Fehler der Aktionswerte ist um ein Mehrfaches größer und liegt bei etwa $0.05$ bis $0.17$. Die Einträge für Aktionen, die die gierige Strategie nicht wählt, ändern sich hauptsächlich bei der Exploration und mitteln daher über weniger Zielwerte.

<!--en:27ce36d7-->
Muss die Strategie bei einem von zwanzig Schritten zufällig handeln, so hat selbst $\pi^*$ nur in etwa $54\%$ der Fälle Erfolg. Das Plateau der Trainingskurve nahe $0.5$ erklärt sich aus diesen Kosten der Exploration. Die Tabelle selbst ist einwandfrei.

<!--en:e8049ffc-->
Ein Lauf benötigt typischerweise etwa $96{,}000$ Schritte in der Umgebung. Die Wertiteration bestätigte $V^*$ nach $164$ Durchläufen mit je $64$ Backups, also etwa $10{,}500$ Backups. Die Zählungen messen Verschiedenes: Ein Backup liest eine ganze Zeile der Übergangstabelle, ein Schritt zeigt nur einen gezogenen Folgezustand. In :exref:`ex_qlearning-samples-backups` vergleichen Sie beides unter Berücksichtigung dieses unterschiedlichen Informationsbedarfs.

<!--en:33fcf77e-->
Die gelernten Werte und Richtungspfeile für den ersten Zufallsstartwert neben der exakten Lösung:

<!--en:a15af705-->
Die gelernten Pfeile geben die Strategie aus :numref:`sec_valueiter` wieder, die das Rutschen berücksichtigt. Nur in Zelle $6$ wählt Q-Learning *rechts* und $\pi^*$ wählt *links*. Zelle $6$ liegt zwischen den Löchern $5$ und $7$. Beide Befehle haben unter $Q^*$ denselben Wert. Die Verfahren lösen also einen exakten Gleichstand unterschiedlich auf. Am genauesten sind die Werte in häufig besuchten Zellen, gröber in der selten besuchten oberen rechten Ecke.

<!--en:038b30e9-->
### Schrittweitenfolgen im Vergleich

<!--en:6ad62b70-->
Auf Eis ohne Rutschen enthalten die Zielwerte kein Übergangsrauschen. Das obige Kontraktionsargument erlaubt daher die konstante Schrittweite $\alpha = 0.9$. Ein Achtel des Episodenbudgets genügt dann fast immer. Hier erreicht die gierige Strategie der fertigen Tabelle in jeder Episode das Ziel:

<!--en:3b3678ee-->
Ohne Rutschen braucht der Lauf etwa $3{,}500$ Schritte in der Umgebung, auf rutschigem Eis etwa $96{,}000$. In den seltenen erfolglosen Läufen hat der Agent das Ziel zu selten erreicht, als dass sich dessen Wert bis zum Start ausbreiten konnte. Dieses Scheitern der Exploration untersucht :numref:`sec_exploration`. Den größten Teil der Kosten verursacht das Übergangsrauschen, die Lernregel spielt eine geringere Rolle. Auf rutschigem Eis kommt es auf die Schrittweite an. Wir vergleichen drei Schrittweitenfolgen mit denselben Zufallsstartwerten und verfolgen dazu den Eintrag $Q(s_0, \textrm{left})$. Sein exakter Wert $0.180$ ist der Wert des Startzustands unter $\pi^*$. Die Grafik zeigt den Median über die Zufallsstartwerte und ein Band vom kleinsten bis zum größten Schätzwert:

<!--en:d6f1a061-->
Bei konstanter Schrittweite kommt die Schätzung nie zur Ruhe. Jeder Zielwert verschiebt sie um den Anteil $90\%$ ihrer Abweichung von ihm. Nach $4{,}000$ Episoden streuen die letzten Schätzungen daher weit um $0.180$, von knapp $0.05$ bis über $0.25$. Mit unserer Standardfolge weicht die Schätzung am Ende eines typischen Laufs um höchstens etwa $0.02$ vom exakten Wert ab. Auch die Folge $1/(1 + n)$ erfüllt die Robbins-Monro-Bedingungen. Bei diesem Budget bleibt jedoch jeder Lauf unter $0.03$, einem Sechstel des exakten Werts. Die frühen Zielwerte aus der mit Nullen gefüllten Tabelle behalten ihr volles Gewicht. Das ist die langsame Konvergenz, die Even-Dar und Mansour vorhersagen. Die Bedingungen garantieren den Grenzwert. Welche Folge sich bei endlichem Budget eignet, legen sie nicht fest.

<!--en:06619221-->
### Rückkopplung zwischen Schätzung und Datenerhebung

<!--en:61cd5ba2-->
Die Verhaltensstrategie hängt von den aktuellen Schätzungen ab. Schätzfehler beeinflussen daher, welche Daten der Agent als Nächstes sammelt. Ist $Q(s, a)$ zu groß, wählt die $\epsilon$-gierige Strategie $a$ häufiger und sammelt Übergänge, deren Zielwerte die Schätzung senken. Ist $Q(s, a)$ zu klein, probiert der Agent die Aktion nur bei der Exploration aus. Wie der obige Lauf zeigte, korrigiert sich der Fehler dann nur langsam. Überschätzungen korrigieren sich also von selbst, Unterschätzungen dagegen nicht. Deshalb dauerte es im Lauf mit der niedrigen Schätzung so lange, bis der Agent seine erste Aktion änderte.

<!--en:cf3e4aa1-->
An einem Fixpunkt der Aktualisierung verschwindet die erwartete Korrektur für jedes Paar, das die Verhaltensstrategie besucht. Dort gilt $Q(s, a) = (\mathcal{T}Q)(s, a)$. Das ist noch nicht die Optimalitätsgleichung, denn das Maximum in $\mathcal{T}Q$ erstreckt sich auch über nie besuchte Einträge, für die somit keine Bedingung gilt. Eine Tabelle kann auf den besuchten Paaren die Gleichungen erfüllen und dennoch falsch sein. Deshalb setzt der Konvergenzsatz voraus, dass der Agent jedes Paar ausprobiert. Lernt er aus einer vorab aufgezeichneten Folge von Übergängen, entfällt diese Rückkopplung ganz. Auf diesen Fall kommen wir im nächsten Kapitel zurück.

<!--en:453a9043-->
## Off-Policy-Lernen

<!--en:449b7134-->
Durch das Maximum im Zielwert lernt Q-Learning die Werte einer Strategie, während es einer anderen folgt. Die nächsten beiden Unterabschnitte behandeln diese Eigenschaft und die damit verbundene Verzerrung.

<!--en:7ea44568-->
### Der Off-Policy-Zielwert und SARSA

<!--en:5c7f78da-->
Das Maximum $\max_{a'}$ in :eqref:`eq_td_error` bewertet eine gierige Fortsetzung, unabhängig davon, welche Aktion die Verhaltensstrategie als Nächstes in $s'$ ausführt. Q-Learning ist daher ein *Off-Policy*-Verfahren: Es schätzt die Werte der gierigen Strategie, während die Daten von der explorierenden Strategie $\pi_e$ stammen. Die On-Policy-Alternative verwendet beim Bootstrapping die Aktion $a'$, die die Verhaltensstrategie in $s'$ tatsächlich auswählt:

<!--en:3dd16099-->
Das Verfahren heißt *SARSA*, nach dem Tupel $(s, a, r, s', a')$, das es bei jeder Aktualisierung verwendet :cite:`Rummery.Niranjan.1994,Sutton.Barto.2018`. Bei fester Verhaltensstrategie ist der Fixpunkt der erwarteten SARSA-Aktualisierung die Lösung der Bellman-Erwartungsgleichung für $\pi_e$. SARSA lernt also $Q^{\pi_e}$, die Werte der explorierenden Strategie einschließlich ihrer zufälligen Aktionen. Auf dem rutschigen See würde ein SARSA-Agent mit konstantem $\epsilon = 0.05$ den Zellen neben Löchern geringere Werte zuweisen, als $\pi^*$ dort erzielt. Seine eigenen zufälligen Schritte führen manchmal in ein Loch. Q-Learning kann dauerhaft explorieren und dennoch wie oben eine gierige Strategie liefern, die mit $\pi^*$ übereinstimmt. Dank derselben Eigenschaft kann es aus älteren Erfahrungen, von einem anderen Agenten oder aus einer festen Aufzeichnung lernen, solange die Daten die benötigten Paare der gierigen Strategie abdecken. Ersetzen wir die Tabelle durch Netze, kann diese Flexibilität das Lernen instabil machen, wie das nächste Kapitel zeigt.

<!--en:9d46ae7f-->
### Maximierungsverzerrung

<!--en:286a23f0-->
Das Maximum verrauschter Schätzwerte fällt tendenziell größer aus als das Maximum ihrer Mittelwerte. Die Maximierung bevorzugt Einträge, deren Fehler zufällig positiv ist. Da das Maximum konvex ist, gilt $\mathbb{E}[\max_{a'} \hat{Q}(s', a')] \geq \max_{a'} \mathbb{E}[\hat{Q}(s', a')]$. Jeder Zielwert, den das Bootstrapping aus einer verrauschten Zeile gewinnt, übernimmt diese *Maximierungsverzerrung* (engl. maximization bias). Die obigen Kurven für die verschiedenen Schrittweiten zeigen den Effekt. Mit der Standardfolge steigt die Schätzung von $Q(s_0, \textrm{left})$ zunächst über $0.180$ und stabilisiert sich erst danach. Ohne Rauschen wäre das unmöglich. Beginnen wir mit einer Nulltabelle und sind die Belohnungen nie negativ, kann jedes exakte Backup die Werte nur in Richtung $Q^*$ anheben. Die Wertiteration nähert sich $0.180$ also von unten. Die Mediankurve dagegen liegt bei etwa $0.22$ oder darüber, von Episode $500$ bis Episode $1{,}500$. Das Rauschen hat sie über das Maximum nach oben verschoben. Die Spitzenwerte der ausgegebenen Läufe liegen zwischen etwa $0.2$ und $0.33$. Bei zwölf Zufallsstartwerten überschießen die meisten Läufe auf diese Weise, um wenige Hundertstel bis über ein Zehntel. Bei konstanter Schrittweite rauschen die Schätzungen stärker und überschießen noch weiter. Während eines Großteils der ersten tausend Episoden liegen sie bei etwa $0.5$.

<!--en:536f2e19-->
Mit der Konvergenz der Tabelleneinträge nimmt die Verzerrung ab. Bei Funktionsapproximation kann sie bestehen bleiben, weil das Rauschen nicht verschwindet. Double Q-Learning verwendet zwei Schätzer. Einer wählt die maximierende Aktion aus, der andere bewertet sie. So entfällt die Verzerrung nach oben, allerdings können die Schätzungen zu niedrig ausfallen :cite:`vanHasselt.2010`. In :exref:`ex_qlearning-maximization-bias` implementieren Sie das Verfahren für eine Aufgabe, bei der die Verzerrung den Agenten in die Irre führt.

<!--en:1e62898a-->
## Planung in einem gelernten Modell

<!--en:a4135651-->
Q-Learning nutzt jeden Übergang einmal. Es ändert einen Eintrag und verwirft die Beobachtung. Ändert sich der Wert des Ziels erneut, steht der alte Übergang nicht mehr zur Verfügung, um die Änderung rückwärts weiterzugeben. Wir können dieselben Übergänge aber auch zählen. Dabei zähle $N(s, a, s')$ die beobachteten Übergänge von $s$ unter Aktion $a$ nach $s'$, und es gelte $N(s, a) = \sum_{s'} N(s, a, s')$. Aus diesen Anzahlen ergibt sich ein geschätztes Modell:

<!--en:f68165ab-->
also die empirische Häufigkeit jedes Folgezustands und die mittlere Belohnung jedes Paars. Bei der Wertiteration auf $(\hat{P}, \hat{r})$ behandeln wir die Schätzung wie ein exaktes Modell. Dieses Prinzip heißt *Sicherheitsäquivalenz* (engl. certainty equivalence). Es führt das Problem ohne Modell auf das Problem aus :numref:`sec_valueiter` zurück.

<!--en:798fb7e3-->
`Counts` kapselt eine Umgebung. Es leitet jeden Schritt unverändert weiter und zeichnet $N(s, a, s')$ sowie die aufsummierten Belohnungen auf. Da `q_learning` nur über `reset` und `step` auf die Umgebung zugreift, läuft es unverändert auch mit dieser Hülle und lernt dieselbe Tabelle wie ohne sie. Durch Division berechnet `learned_model` aus den Zählerständen $\hat{P}$ und $\hat{r}$ und gibt beide als `d2l.TabularMDP` zurück. Die Zeilen nie ausprobierter Paare bleiben null. Für die Planung gelten diese Paare daher als wertlose Sackgassen:

<!--en:270c1e87-->
Wie viel besser nutzt das Modell die Daten? Wir geben Q-Learning einmal ein kleines Budget von $500$ Episoden und einmal ein größeres von $2{,}000$. Jedes Budget erhält einen vollständigen Lauf mit eigener Explorationsrate im Trainingsverlauf. Auf den Übergängen jedes Laufs vergleichen wir zwei Tabellen: die von Q-Learning gelernte Tabelle und die Aktionswerte des geschätzten Modells. Letztere erhalten wir durch ein Backup der Lösung aus der Wertiteration. Für jede Tabelle liefert `grade` den Fehler der Zustandswerte gegenüber `V_star` in der Supremumsnorm sowie die mit `success` berechnete exakte Erfolgsquote ihrer gierigen Strategie:

<!--en:c43df607-->
Bei wenigen Daten, etwa $10{,}000$ Schritten, ist die durch Planung gewonnene gierige Strategie besser. Sie erreicht das Ziel typischerweise in etwa zwei von drei Fällen, die Strategie von Q-Learning etwa in jedem zweiten. In rund neun von zehn Läufen war die geplante Strategie mindestens ebenso gut. Den größten Fehler der Zustandswerte bestimmen bei diesem Budget einige selten besuchte Zellen. Meist ist er beim Modell kleiner, in den fünf hier ausgegebenen Läufen schneidet zufällig Q-Learning besser ab. Bei etwa $47{,}000$ Schritten sind beide gierigen Strategien in den meisten Läufen optimal. Die durch Planung berechneten Werte sind etwa doppelt so genau. Bei fast jedem Zufallsstartwert waren sie besser und etwa so genau wie die Tabellen von Q-Learning nach doppelt so vielen Episoden. Den Grund dafür haben wir schon beim Übergang gesehen, den wir von Hand berechnet haben. Q-Learning gibt den Wert bei jedem Besuch um einen Übergang rückwärts weiter. Seine Zielwerte stammen aus einer Tabelle, die zum Zeitpunkt des Übergangs noch fehlerhaft war. Das Modell berücksichtigt jeden Übergang, und in jedem der wenigen hundert Durchläufe der Wertiteration breitet sich die Zielbelohnung über alle gespeicherten Übergänge aus. In :exref:`ex_qlearning-batch-td-monte-carlo` sehen Sie denselben Effekt in einem Prozess mit zwei Zuständen.

<!--en:8f429352-->
Auch das Modell verursacht Kosten. Es speichert $|\mathcal{S}|^2 |\mathcal{A}|$ Zähler, hier etwa tausend Zahlen, bei einem großen Zustandsraum viel zu viele. Nach jedem neuen Batch von Daten müssen wir das Modell erneut lösen. Außerdem vertraut die Planung der Schätzung vollständig. Ein nie ausprobiertes Paar erscheint wertlos. Ein Paar, das der Agent einmal mit glücklichem Ausgang ausprobiert hat, sieht besser aus, als es ist. In :numref:`sec_model_based` passen wir ein solches Modell für Pendulum durch Regression statt durch Zählen an und planen damit.

<!--en:3364ec8f-->
Der Agent kann nun optimale Aktionswerte allein aus Beobachtungen lernen und danach handeln. Er kann aus denselben Beobachtungen auch ein Modell gewinnen und damit planen. Beides hängt davon ab, wie er Daten sammelt. Die hier verwendete $\epsilon$-gierige Regel ist grob. Muss selbst $\pi^*$ in einem von zwanzig Schritten zufällig handeln, sinkt die Erfolgsquote auf etwa $54\%$ statt $73\%$. Zudem braucht die gierige Wahl aus der Tabelle eine endliche Menge von Aktionen, über die sie maximiert. In :numref:`sec_exploration` untersuchen wir, ob eine andere Regel mit geringeren Einbußen genauso viel lernen kann. Später im Kapitel lernen wir die Strategie selbst.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:4b56f23f-->
Q-Learning ersetzt den Erwartungswert im Bellman-Backup durch einen beobachteten Übergang und ändert den besuchten Eintrag mit Schrittweite $\alpha$ in Richtung des Temporal-Difference-Fehlers $\delta$. Im Erwartungswert ist das ein gedämpfter Schritt der Wertiteration. Da der Bellman-Operator eine Kontraktion ist, konvergiert tabellarisches Q-Learning gegen $Q^*$, sofern jedes Paar unendlich oft ausprobiert wird und die Schrittweiten die Robbins-Monro-Bedingungen erfüllen. Q-Learning ist ein Semigradientenverfahren. Bei zufälligen Übergängen konvergiert der Gradientenabstieg auf dem quadratischen Temporal-Difference-Fehler gegen einen anderen Punkt. Das ist das Problem des doppelten Samplings. Auf dem rutschigen See liefern die Tabellen in einem typischen Lauf eine optimale gierige Strategie. Die Schrittweitenfolge bestimmt, wie schnell das gelingt. Das Maximum im Zielwert macht Q-Learning zum Off-Policy-Verfahren und verzerrt die Schätzungen nach oben, solange sie verrauscht sind. Zählen wir dieselben Übergänge und planen im geschätzten Modell, erhalten wir aus denselben Daten genauere Werte. Dafür müssen wir das Modell speichern und lösen.

<!--en:2294243a-->
## Übungen

<!--en:82cde62d-->
1. **Batch-TD im Vergleich mit Monte Carlo.** Ein Markov-Belohnungsprozess ist ein MDP ohne Aktionen. Zu jedem Zustand gehört eine feste gemeinsame Verteilung von Belohnung und Folgezustand. Ein solcher Prozess habe zwei Zustände $x$ und $y$, die keine Endzustände sind, sowie $\gamma = 1$. Ein Batch enthält acht Episoden. Eine führt von $x$ nach $y$ mit Belohnung $0$ und endet dann mit Belohnung $0$. Sechs starten in $y$ und enden mit Belohnung $1$. Eine startet in $y$ und endet mit Belohnung $0$. TD(0) ist die Zustandswertform von :eqref:`eq_td_error` ohne das Maximum, also $V(s) \leftarrow V(s) + \alpha \big(r + \gamma V(s') - V(s)\big)$.
    1. Berechnen Sie aus dem Batch die Monte-Carlo-Schätzungen von $V(x)$ und $V(y)$. Berechnen Sie auch die Werte, gegen die TD(0) konvergiert, wenn es den Batch mit kleiner Schrittweite viele Male durchläuft.
    1. Welches Schätzwertepaar hat den kleineren quadratischen Fehler für die acht beobachteten Erträge?
    1. Schätzen Sie die Übergangswahrscheinlichkeiten und die erwarteten Belohnungen des Prozesses durch Zählen. Lösen Sie seine Bellman-Gleichung exakt. Welches der beiden Schätzwertepaare ergibt sich aus der Lösung?
    1. Von welchem Paar erwarten Sie bessere Vorhersagen der Erträge künftiger Episoden mit Start in $x$? Unter welcher Annahme über den Prozess?

    *Nach Richard S. Sutton und Andrew G. Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf), Beispiel 6.4.*
   :exlabel:`ex_qlearning-batch-td-monte-carlo`

<!--en:c788c1fe-->
1. **Abbruch als Terminierung behandeln.** Eine Aufgabe hat einen Zustand und eine Aktion, gewährt bei jedem Schritt die Belohnung $1$ und endet nie. Ein Zeitlimit bricht jede Episode nach $H$ Schritten ab, wie bei `FrozenLake-v1` nach $100$ Schritten. Eine fehlerhafte Implementierung maskiert den Bootstrap-Term mit `terminated or truncated` statt mit `terminated`.
    1. Bei abnehmenden Schrittweiten konvergiert die Schätzung gegen den Wert, bei dem die erwartete Aktualisierung über die $H$ Übergänge einer Episode verschwindet. Bestimmen Sie diesen Wert als Funktion von $\gamma$ und $H$. Vergleichen Sie ihn mit dem korrekten Wert $1/(1 - \gamma)$ für $\gamma = 0.95$ und $H = 100$. Welchen Diskontierungsfaktor verwendet die fehlerhafte Implementierung effektiv?
    1. Nehmen Sie nun an, das Zeitlimit gehört zur Aufgabe, sodass der Ertrag tatsächlich nach $H$ Schritten endet. Zeigen Sie, dass keine einzelne Zahl den Zustand korrekt bewertet. Erweitern Sie den Zustand um die Anzahl $h$ der verbleibenden Schritte wie in :exref:`ex_mdp-time-limits` aus :numref:`sec_mdp`. Bestimmen Sie den Wert jedes Zustands $(s, h)$ und zeigen Sie, dass die Maske `terminated or truncated` dann korrekt ist.
   :exlabel:`ex_qlearning-truncation-termination`

<!--en:221a3720-->
1. **SARSA und Q-Learning bei gieriger Aktionswahl.** SARSA ersetzt das Maximum in :eqref:`eq_td_error` durch den Wert der Aktion $a'$, die die Verhaltensstrategie in $s'$ auswählt, wie in :eqref:`eq_qlearning_sarsa`. Eine Implementierung wählt $a'$ aus, bevor sie $Q(s, a)$ aktualisiert, und führt $a'$ dann im nächsten Schritt aus. Angenommen, beide Verfahren wählen ihre Aktionen gierig anhand ihrer aktuellen Tabellen und lösen Gleichstände nach derselben festen Regel auf. Sind SARSA und Q-Learning dann derselbe Algorithmus? Wählen sie dieselben Aktionen und aktualisieren sie die Einträge auf dieselbe Weise? Falls sie sich unterscheiden, konstruieren Sie einen Übergang auf dem See ohne Rutschen und Tabellenwerte, bei denen die Verfahren im nächsten Schritt unterschiedliche Aktionen wählen.

    *Nach Richard S. Sutton und Andrew G. Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf), Übung 6.12.*
   :exlabel:`ex_qlearning-greedy-sarsa`

<!--en:61ee3bbd-->
1. [code] **Gegenbeispiel zum doppelten Sampling.** ● Konstruieren Sie einen MDP mit drei Zuständen und einer Aktion pro Zustand. In Zustand $s$ liefert die Aktion die Belohnung $0$ und führt nach $u_1$ oder $u_2$, jeweils mit Wahrscheinlichkeit $1/2$. Von $u_1$ aus liefert sie die Belohnung $1$ und terminiert. Von $u_2$ aus liefert sie die Belohnung $0$ und terminiert ebenfalls. Bezeichnen Sie die drei Einträge mit $Q(s)$, $Q(u_1)$ und $Q(u_2)$.
    1. Differenzieren Sie einen Summanden der empirischen Zielfunktion :eqref:`q_learning_optimization_problem` nach den Tabelleneinträgen. Für welche Einträge ist die Ableitung von null verschieden? Welche dieser Ableitungen lässt die Semigradientenaktualisierung :eqref:`eq_td_error` weg?
    1. Geben Sie $Q^*$ für die drei Einträge an.
    1. Nehmen Sie an, $d$ sei über die drei Paare gleichverteilt. Schreiben Sie die Zielfunktion der Grundgesamtheit :eqref:`eq_double_sampling` explizit auf und bestimmen Sie ihren Minimierer als Funktion von $\gamma$ in geschlossener Form. Welche Einträge verzerrt der Minimierer gegenüber $Q^*$ und in welche Richtung?
    1. Besuchen Sie die drei Paare gleich oft. Führen Sie mit abnehmender Schrittweite sowohl die Semigradientenaktualisierung als auch den stochastischen Gradientenabstieg mit der vollständigen Ableitung aus Teil (a) aus. Bestätigen Sie, dass beide gegen den jeweils in Teil (b) und (c) vorhergesagten Punkt konvergieren.
    1. Angenommen, die Umgebung lässt sich auf $s$ zurücksetzen, sodass jeder Besuch in $s$ zwei unabhängige Folgezustände liefert. Verwenden Sie einen davon im Zielwert und den anderen in der Ableitung aus Teil (a). Zeigen Sie, dass der resultierende stochastische Gradient den Gradienten des quadrierten Bellman-Residuums unverzerrt schätzt. Bestätigen Sie durch Simulation, dass das Verfahren gegen $Q^*$ konvergiert.
   :exlabel:`ex_qlearning-double-sampling-counterexample`

<!--en:ab077ea7-->
1. [code] **Schrittweiten bei endlichem Budget.** ● Der Abschnitt vergleicht eine konstante Schrittweite mit den Folgen $\alpha = 1/(1 + c\, n)$ für $c = 0.1$ und $c = 1$. Dabei zählt $n$ die Besuche des Eintrags einschließlich des aktuellen Besuchs. Behandeln Sie für die Analyse die Zielwerte eines Eintrags als unabhängige Ziehungen mit Mittelwert $m$ und Varianz $\sigma^2$.
    1. Bestimmen Sie für konstantes $\alpha$ die Varianz der Schätzung nach vielen Aktualisierungen. Berechnen Sie diese Varianz bei $\alpha = 0.9$ für $Q(s_0, \textrm{left})$. Verwenden Sie als Zielwerte $r + \gamma V^*(s')$, wobei Sie $s'$ aus `mdp.P[0, a0]` ziehen und `V_star` aus diesem Abschnitt übernehmen. Vergleichen Sie die resultierende Standardabweichung mit der Streuung der hier ausgegebenen Endschätzungen bei konstanter Schrittweite um den exakten Wert $0.180$. Welche Annahme der Analyse ist für diesen Eintrag verletzt, und woher kommt die übrige Streuung?
    1. Schreiben Sie für $c = 1$ die Schätzung nach $n$ Besuchen als gewichtete Summe aus dem Anfangswert und den $n$ Zielwerten. Welches Gewicht erhält die Initialisierung mit null?
    1. Schreiben Sie für allgemeines $c$ das Gewicht des $k$-ten Zielwerts nach $n$ Besuchen als Produkt. Vergleichen Sie den Logarithmus des Produkts mit einem Integral, um zu zeigen, dass das Gewicht wie $k^{1/c - 1}$ in $k$ wächst, wenn $k \leq n$ groß ist. Wie gewichtet $c = 0.1$ die Zielwerte aus der frühen, noch ungenauen Tabelle?
    1. Testen Sie auf dem rutschigen See $c \in \{0.01, 0.03, 0.1, 0.3, 1\}$ mit jeweils fünf Zufallsstartwerten. Tragen Sie gegen $c$ sowohl den Endfehler $|\hat{Q}(s_0, \textrm{left}) - Q^*(s_0, \textrm{left})|$ als auch den Fehler $\|\hat{Q} - Q^*\|_\infty$ in der Supremumsnorm über alle Einträge auf. Erklären Sie die Form der Kurven mithilfe der Teile (a) und (c). Liefern beide Fehlermaße dieselbe Wahl für $c$?
   :exlabel:`ex_qlearning-step-sizes`

<!--en:0467eb2c-->
1. [code] **Beobachtete Übergänge und Backups im Vergleich.** ● Verwenden Sie `Q_star` als Ground Truth.
    1. Erfassen Sie den Fehler $\|\hat{Q} - Q^*\|_\infty$ nach jeder Episode von `q_learning` auf dem rutschigen See. Tragen Sie ihn gegen die Anzahl der Schritte in der Umgebung auf, die `visits` zählt. Tragen Sie im selben Diagramm den Fehler der Wertiteration auf. Verwenden Sie dabei $Q_k$ = `mdp.backup` der $k$-ten Iterierten von `d2l.value_iteration` und als horizontale Koordinate die Anzahl der Bellman-Backups bei $64$ Backups pro Durchlauf. Wie viele Backups benötigt die Wertiteration, um den Fehler zu erreichen, den Q-Learning nach $4000$ Episoden erzielt?
    1. Führen Sie `q_learning` auf `Counts(env)` aus und lösen Sie nach $100$, $400$, $1000$ und $4000$ Episoden wie im Abschnitt das Modell `learned_model` mit `d2l.value_iteration`. Ergänzen Sie im Diagramm bei denselben Schrittzahlen den Fehler der Aktionswerte des Modells. Wie viele Schritte in der Umgebung benötigt das Modell, um den Fehler von Q-Learning nach $4000$ Episoden zu erreichen? Wie viele Backups kostet jede Lösung des Modells?
    1. Die Wertiteration benötigt Modell-Backups, Q-Learning benötigt Schritte in der Umgebung, und das geschätzte Modell aus Teil (b) benötigt beides. Angenommen, ein Schritt in der Umgebung kostet so viel Rechenaufwand wie $b$ Backups. Bestimmen Sie anhand Ihrer Messungen, für welchen Bereich von $b$ jedes der drei Verfahren den Fehler von Q-Learning nach $4000$ Episoden zu den geringsten Gesamtkosten erreicht. Welche Verfahren stehen einem Agenten zur Verfügung, der den Übergangskern nicht auslesen kann?
   :exlabel:`ex_qlearning-samples-backups`

<!--en:631dde7a-->
1. [code] **Maximierungsverzerrung in einer Aufgabe mit zwei Zuständen.** ● Jede Episode beginnt im Zustand $s_0$ mit zwei möglichen Aktionen. Die Aktion *rechts* terminiert mit Belohnung $0$, die Aktion *links* führt in den Zustand $s_1$ mit Belohnung $0$. In $s_1$ terminiert jede der $m$ Aktionen mit einer aus $\mathcal{N}(-0.1, 1)$ gezogenen Belohnung. Daher ist *links* ein Fehler. Verwenden Sie $\gamma = 1$, die konstante Schrittweite $\alpha = 0.1$, Anfangswerte von null und `epsilon_greedy` mit $\epsilon = 0.1$.
    1. Implementieren Sie tabellarisches Q-Learning für $m = 10$. Tragen Sie gegen die Episodennummer den Anteil der Läufe auf, die *links* in $s_0$ wählen. Mitteln Sie dazu über $1000$ Läufe mit je $300$ Episoden. Welchen Anteil würden Sie bei einem $\epsilon$-gierigen Agenten beobachten, der nach $Q^*$ handelt?
    1. Double Q-Learning :cite:`vanHasselt.2010` führt zwei Tabellen $Q_1$ und $Q_2$, handelt $\epsilon$-gierig anhand von $Q_1 + Q_2$ und wirft bei jedem Übergang eine faire Münze. Bei Kopf gilt
       $$Q_1(s, a) \leftarrow Q_1(s, a) + \alpha \big( r + \gamma\, Q_2(s', \mathrm{argmax}_{a'} Q_1(s', a')) - Q_1(s, a) \big),$$
       bei Zahl verwendet es dieselbe Aktualisierung mit vertauschten Tabellen. Ergänzen Sie die Kurve des Verfahrens im Diagramm.
    1. Jede über *links* erreichbare Belohnung hat einen negativen Mittelwert. Erklären Sie, wie die Schätzung von Q-Learning für $Q(s_0, \textrm{left})$ dennoch positiv wird. Sagen Sie vorher, wie sich die Q-Learning-Kurve für $m = 1$ und für $m = 100$ verändert, und überprüfen Sie Ihre Vorhersage. Welcher Teil des Effekts bei $m = 100$ stammt aus verrauschten Schätzungen und welcher aus noch nie ausprobierten Aktionen?

    *Nach Richard S. Sutton und Andrew G. Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf), Beispiel 6.7.*
   :exlabel:`ex_qlearning-maximization-bias`

<!--en:1eaf2b87-->
1. [code] **SARSA und Q-Learning an der Klippe.** ● In `CliffWalking-v1` durchquert der Agent ein $4 \times 12$-Gitter von der linken unteren zur rechten unteren Ecke. Jeder Schritt bringt $-1$ ein. Ein Schritt in den Abgrund am unteren Rand bringt $-100$ ein und versetzt den Agenten zurück an den Start. Verwenden Sie $\gamma = 1$, die konstante Schrittweite $\alpha = 0.5$ und ein konstantes $\epsilon = 0.1$.
    1. Schreiben Sie `sarsa` als Kopie von `q_learning`. Wählen Sie darin die nächste Aktion $a'$ mit `d2l.epsilon_greedy` vor der Aktualisierung, verwenden Sie sie im Zielwert :eqref:`eq_qlearning_sarsa` und führen Sie sie anschließend aus. Trainieren Sie beide Verfahren über $500$ Episoden mit zehn Zufallsstartwerten und zeichnen Sie den Ertrag pro Episode.
    1. Folgen Sie vom Start aus der gierigen Strategie jeder fertigen Tabelle. Welchen Weg lernt jedes Verfahren, und wie lang ist er? Welches Verfahren sammelt beim Lernen mehr Belohnung, und warum fallen die Antworten unterschiedlich aus?
    1. Lassen Sie $\epsilon$ im Verlauf des Trainings auf null sinken. Was geschieht mit den beiden gelernten Wegen, und welche Werte lernt jedes Verfahren im Grenzfall?

    *Nach Richard S. Sutton und Andrew G. Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf), Beispiel 6.6.*
   :exlabel:`ex_qlearning-cliff-walking`

<!--en:f8c6172a-->
[Diskussionen](https://d2l.discourse.group/t/36797)

<!--en:356fdf80-->
Lernen aus zeitlichen Differenzen<br>**Backups aus Samples · Q-Learning und SARSA · Verzerrung beim doppelten Sampling · Robbins-Monro-Schrittweiten · Maximierungsverzerrung · Planung nach dem Prinzip der Sicherheitsäquivalenz**

<!--en:2f1f4b17-->
::: {.slide title="Handeln lernen ohne die Übergangstabelle P"}

<!--en:b0445376-->
[Motivation und Grundidee]{.kicker}

<!--en:0ab460ad-->
Die dynamische Programmierung summierte mit dem bekannten Tensor $P(s' \mid s, a)$ über alle möglichen Rutschbewegungen. Auf einem unbekannten See beobachtet der Agent pro Schritt nur den tatsächlich eingetretenen Übergang:

<!--en:13ec2826-->
- **Ein einzelnes Sample:** Der Befehl *abwärts* in Zelle 9 liefert eine Ziehung $(r, s')$ aus der Umgebung. Die vollständige Verteilung $(1/3, 1/3, 1/3)$ über die Zellen 8, 13 und 10 bleibt verborgen.

<!--en:fe810973-->
- **Stochastische Approximation:** Sutton (1988) und Watkins (1989) ersetzten den exakten Erwartungswert im Bellman-Backup durch eine Schätzung aus einem beobachteten Folgezustand und machten einen kleinen, gedämpften Schritt $\alpha$ in ihre Richtung.

<!--en:d826ecad-->
::: {.slide title="Warum modellfreie Steuerung Aktionswerte Q lernt"}

<!--en:1825611e-->
[Zustandswerte und Aktionswerte]{.kicker}

<!--en:498c3c69-->
- **Warum ein Sample für $V^*(s)$ nicht genügt:** Im Ausdruck $\max_a \mathbb{E}_{s'}[r + \gamma V(s')]$ steht $\max_a$ *außerhalb* des Erwartungswerts.

<!--en:28382a73-->
- **Warum ein Sample für $Q^*(s, a)$ genügt:** Bei festem $(s, a)$ steht $\mathbb{E}_{s'}$ *außerhalb* von $\max_{a'} Q(s', a')$. Damit liefert $r + \gamma \max_{a'} Q(s', a')$ eine unverzerrte Schätzung von $(\mathcal{T}Q)(s, a)$.

<!--en:fafe2110-->
::: {.slide title="Von exakten Bellman-Backups zu Zielwerten aus Samples"}

<!--en:ecb9103a-->
[Das Backup beim Q-Learning]{.kicker}

<!--en:0ba3b488-->
Der Vergleich von (c) und (d) zeigt, wie Q-Learning den Mittelwert über die Umgebung durch den beobachteten Übergang ersetzt. Die gierige Maximierung im nächsten Zustand $s'$ bleibt erhalten:

<!--en:6aa19915-->
- **Temporal-Difference-Fehler (TD-Fehler) $\delta_t$:** Differenz zwischen dem durch Bootstrapping gewonnenen Einschritt-Zielwert $y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$ und der aktuellen Schätzung $Q(s_t, a_t)$.

<!--en:c732830b-->
::: {.slide title="Bootstrapping und Monte-Carlo-Schätzung"}

<!--en:eb86dc8c-->
[Verzerrung und Varianz der Zielwerte]{.kicker}

<!--en:b7bc5ff4-->
- **Monte-Carlo-Zielwert ($R_t = \sum_{k \geq 0} \gamma^k r_{t+k}$):** Liegt erst am Episodenende vor. Schätzt den Wert der Verhaltensstrategie $Q^{\pi_e}$ unverzerrt, enthält aber die hohe Varianz der Übergänge aus der gesamten restlichen Episode.

<!--en:6f880448-->
- **TD-Zielwert für einen Schritt ($y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$):** Erlaubt nach jedem Übergang eine Online-Aktualisierung mit geringer Einschrittvarianz. An die Stelle des Rauschens der restlichen Episode tritt die aktuelle Verzerrung der Tabelle bei der Off-Policy-Schätzung von $Q^*$.

<!--en:c7041c83-->
::: {.slide title="Wie sich der Wert mit jedem Besuch rückwärts ausbreitet"}

<!--en:5111126b-->
[Rechenbeispiel]{.kicker}

<!--en:c1eba7bf-->
- **Erstmals im Ziel ($14 \xrightarrow{\textrm{right}} 15$):** Der Zielwert $y = 1$ für diesen Übergang ergibt die Aktualisierung $Q(14, \textrm{right}) \leftarrow 0.500$.

<!--en:7076647d-->
- **Schritt nach 14 ($13 \xrightarrow{\textrm{right}} 14$):** Bootstrapping liefert $y = 0.95(0.500) = 0.475$ und damit $Q(13, \textrm{right}) \leftarrow 0.238$.

<!--en:cefdc94f-->
- **Rutschen zur unbesuchten Zelle 10 ($14 \xrightarrow{\textrm{right}} 10$):** Der Zielwert $y = 0$ senkt den Eintrag gemäß $Q(14, \textrm{right}) \leftarrow 0.250$. Daran zeigt sich, warum $\alpha_n$ abnehmen muss.

<!--en:1f050bb8-->
::: {.slide title="Warum Q-Learning Semigradienten verwendet und keine kleinsten Quadrate"}

<!--en:d229612b-->
[Das Problem des doppelten Samplings]{.kicker}

<!--en:9733aff9-->
- **Warum der vollständige Gradient von $\delta^2$ scheitert (Baird, 1995):** Wer das empirische $\delta_i^2$ minimiert, bestraft die Übergangsvarianz $\textrm{Var}(y \mid s, a)$. Dadurch nähern sich die Werte $Q(u_1), Q(u_2)$ der beiden zufälligen Folgezustände einander an.

<!--en:bff111d8-->
- **Lösung mit Semigradienten:** Während des Schritts bleibt $y_i = r_i + \gamma \max_{a'} Q(s_i', a')$ fest, sodass $\mathbb{E}[\delta \mid s, a] = (\mathcal{T}Q)(s, a) - Q(s, a)$ gilt.

<!--en:91a7e54a-->
::: {.slide title="Robbins-Monro-Schrittweiten und Konvergenz"}

<!--en:92092676-->
[Übergangsrauschen herausmitteln]{.kicker}

<!--en:d74843d6-->
Beim $n$-ten Besuch des Paars $(s, a)$ liefert $\alpha_n = 1/n$ den exakten Stichprobenmittelwert der Zielwerte. Allgemeiner fordern Robbins und Monro (1951):

<!--en:d593e228-->
- **Asymptotischer Konvergenzsatz (Watkins und Dayan, 1992):** In einem endlichen MDP mit $\gamma < 1$ gelte für jedes Paar $(s, a)$, dass es unendlich oft besucht wird und $\alpha_n(s, a)$ die Robbins-Monro-Bedingungen erfüllt. Dann gilt $Q_n \to Q^*$ mit Wahrscheinlichkeit $1$.

<!--en:9bf2ec86-->
- **Nachteil von $\alpha_n = 1/n$ bei endlichem Budget:** Frühe Zielwerte aus dem Bootstrapping mit einer noch fast überall verschwindenden Tabelle $Q$ erhalten dasselbe Gewicht wie späte. Die Zahl der nötigen Aktualisierungen wächst exponentiell in $1/(1-\gamma)$ (Even-Dar und Mansour, 2003).

<!--en:debee8b2-->
- **Polynomielle Gewichtung als Standard:** Unsere Schrittweitenfolge $\alpha_n = 1/(1 + 0.1\,n)$ erfüllt die Robbins-Monro-Bedingungen und gewichtet den $k$-ten Zielwert proportional zu $\approx k^9$. Frühe Zielwerte von null verlieren rasch an Einfluss.

<!--en:b0f73b5f-->
::: {.slide title="Bootstrapping nur bei Terminierung ausblenden"}

<!--en:8f52446a-->
[Episodenenden richtig behandeln]{.kicker}

<!--en:a1f09f98-->
Landet der Agent in $s'$, müssen wir den Bootstrap-Term $\max_{a'} Q(s', a')$ bei tatsächlicher Terminierung (`terminated=True`) auf null setzen. Beim Abbruch durch ein Zeitlimit (`truncated=True`) behalten wir ihn bei:

<!--en:0290dce9-->
- **Loch und Ziel als Endzustände:** Der künftige Ertrag ist null. Daher blendet `(1 - terminated)` den Term `Q[s_next].max()` aus.

<!--en:74bd13bc-->
- **Zeitlimit von 100 Schritten:** Die Aufzeichnung endet auf einer begehbaren Eisfläche. Eine Maske für `truncated` würde die Werte nahe Schritt 100 fälschlich gegen null treiben.

<!--en:c66251ee-->
::: {.slide title="Epsilon-gierige Verhaltensstrategie bei Gleichstand"}

<!--en:1d01f36f-->
[Jedes Paar ausprobieren]{.kicker}

<!--en:69ef0087-->
Damit die gierige Auswahl weder in einer unbesuchten Nulltabelle stecken bleibt noch eine unterschätzte Aktion dauerhaft übergeht, exploriert die Verhaltensstrategie $\pi_e$ mit Wahrscheinlichkeit $\epsilon$:

<!--en:2ae3818e-->
- **Warum bei Gleichstand der Zufall entscheiden muss:** In einer Nulltabelle liefert `np.argmax` immer den Index `0` (*links*). Die gleichverteilte Auswahl bei Gleichstand erlaubt Exploration auch bei kleinem $\epsilon$.

<!--en:76759ca2-->
::: {.slide title="Die Trainingsschleife für tabellarisches Q-Learning"}

<!--en:908271a0-->
[Implementierung in NumPy]{.kicker}

<!--en:fcfdcd06-->
`q_learning` senkt $\epsilon$ in der ersten Trainingshälfte linear von $1.0$ auf $0.05$ und passt die Schrittweite jedes Paars an seine Besuchszahl `visits[s, a]` an:

<!--en:30342af8-->
- **Reine Indexarithmetik:** Jeder Schritt in der Umgebung aktualisiert einen einzigen Tabelleneintrag `Q[s, a]` in Zeit $O(|\mathcal{A}|)$ und braucht dafür nur `(s, a, r, s_next, terminated)`.

<!--en:46ea1a8f-->
::: {.slide title="Warum die Trainingskurve nahe 0.50 abflacht"}

<!--en:2b4e4bb4-->
[Ertrag der Verhaltensstrategie und Bewertung der gierigen Strategie]{.kicker}

<!--en:4e6e9d05-->
Bei 5 Zufallsstartwerten mit je $4{,}000$ Episoden auf dem rutschigen FrozenLake steigt der gleitende Mittelwert der Trainingserträge über 200 Episoden und flacht nahe $0.50$ ab:

<!--en:ded4fb6a-->
- **Was die Kurve misst:** Die Trainingserträge bewerten die *Verhaltensstrategie* $\pi_e$. Sie handelt im Mittel noch in einem von zwanzig Schritten zufällig ($\epsilon = 0.05$), sobald Episode $2{,}000$ erreicht ist.

<!--en:562a9e43-->
- **Kosten der Exploration:** Selbst bei der exakten optimalen Strategie $\pi^*$ sinkt bei erzwungener Zufallswahl mit Wahrscheinlichkeit $\epsilon = 0.05$ die Erfolgsquote innerhalb von 100 Schritten von **$73.0\%$ auf $54.2\%$**.

<!--en:0a8bd8c1-->
::: {.slide title="Gelernte Tabellen im Vergleich mit den exakten Werten V* und Q*"}

<!--en:7d16f0b2-->
[Exakte Bewertung mit dem Modell]{.kicker}

<!--en:12205e5f-->
Wir bewerten die gelernten gierigen Strategien $\mathrm{argmax}_a \hat{Q}(s, a)$ mit $\epsilon = 0$ durch exakte Strategiebewertung. Bei jedem Zufallsstartwert ergibt sich eine optimale Strategie:

<!--en:4a420ff4-->
- **Kein Regret bei gieriger Wahl:** Jeder Zufallsstartwert erreicht einen Regret von $0.000$ unter $Q^*$ und die volle Erfolgsquote von **$73.0\%$**. Der Median des Aufwands beträgt $\approx 96{,}000$ Schritte in der Umgebung.

<!--en:f5b9cabf-->
- **Asymmetrischer Fehler:** Die Zustandswerte $\max_a \hat{Q}(s, a)$ treffen $V^*$ bis auf $0.03$. Die Einträge suboptimaler Aktionen weichen stärker ab ($0.05$–$0.17$), weil der Agent sie nur bei zufälligen Explorationsschritten besucht.

<!--en:f11a23a3-->
::: {.slide title="Gelernte Q-Tabelle und exakte Wertiteration"}

<!--en:775e9956-->
[Die gierige Strategie im Bild]{.kicker}

<!--en:0263a7de-->
Der Vergleich der gelernten Werte $\max_a \hat{Q}(s, a)$ und der gierigen Befehle für Zufallsstartwert 0 mit der exakten Wertiteration bestätigt die Strategie, die das Rutschen berücksichtigt:

<!--en:7e587962-->
- **Exakter Gleichstand in Zelle 6:** Bei Zufallsstartwert 0 lautet der Befehl in Zelle 6 *rechts* (`>`), bei Wertiteration *links* (`<`). Beide Aktionen haben denselben Wert $Q^*(6, \textrm{left}) = Q^*(6, \textrm{right})$ und dieselbe Wahrscheinlichkeit $1/3$, in ein Loch zu rutschen.

<!--en:2ec0fa29-->
::: {.slide title="Schrittweitenfolgen auf Eis mit und ohne Rutschen" layout="tight"}

<!--en:6ba35433-->
[Konvergenz bei endlichem Budget]{.kicker}

<!--en:f553becf-->
Auf Eis ohne Rutschen sind die Übergänge deterministisch, die Übergangsvarianz der Zielwerte ist null. Bei konstantem $\alpha = 0.9$ ist die Aufgabe nach $3{,}500$ Schritten gelöst. Auf rutschigem Eis zeigt der Vergleich von $Q(s_0, \textrm{left})$ mit $Q^*(s_0, \textrm{left}) = 0.180$ die Unterschiede zwischen drei Folgen:

<!--en:05564744-->
- **Konstantes $\alpha = 0.9$:** Die Schätzung kommt nie zur Ruhe. Mit jedem Rutschereignis schwankt sie stark zwischen $0.05$ und $0.25+$.

<!--en:f1846d16-->
- **Stichprobenmittelwert $1/(1+n)$:** Bleibt unter $0.03$, weil frühe Zielwerte von null ihr volles Gewicht $1/n$ behalten.

<!--en:439d7d74-->
- **Standardfolge $1/(1+0.1n)$:** Nimmt langsam genug ab, um frühe Nullen zu vergessen. Die Schätzung nähert sich $0.180$.

<!--en:39b3f45a-->
::: {.slide title="Off-Policy-Q-Learning und On-Policy-SARSA"}

<!--en:1e9febfa-->
[Zielstrategie und Verhaltensstrategie]{.kicker}

<!--en:39bc8a82-->
- **Off-Policy (Q-Learning):** Lernt $Q^*$ für die gierige Zielstrategie, unabhängig davon, welche Verhaltensstrategie $(s, a, r, s')$ erzeugt hat.

<!--en:41d76097-->
- **On-Policy (SARSA):** Verwendet die tatsächlich als Nächstes ausgeführte Aktion $a' \sim \pi_e(\cdot \mid s')$ beim Bootstrapping und konvergiert gegen $Q^{\pi_e}$.

<!--en:a95a8b96-->
::: {.slide title="Maximierungsverzerrung und Double Q-Learning"}

<!--en:a0420b05-->
[Warum verrauschte Maxima zu groß ausfallen]{.kicker}

<!--en:6445e5cf-->
Im obigen Verlauf von $Q(s_0, \textrm{left})$ nähern sich exakte Backups vom Anfangswert null aus monoton dem Wert $0.180$ von unten. Q-Learning überschreitet ihn zwischen Episode 500 und 1 500 und erreicht $0.22$–$0.33$:

<!--en:d0e3f35c-->
- **Jensen-Ungleichung für $\max$:** Das Maximum über verrauschte Schätzungen wählt bevorzugt Aktionen mit gerade positivem Schätzfehler. Das verzerrt jeden durch Bootstrapping gewonnenen Zielwert nach oben.

<!--en:2faf54a9-->
- **Double Q-Learning (van Hasselt, 2010):** Führt zwei unabhängige Tabellen $\hat{Q}_A$ und $\hat{Q}_B$. Mit $\hat{Q}_A$ wählt es $a^* = \mathrm{argmax}_{a'} \hat{Q}_A(s', a')$, mit $\hat{Q}_B(s', a^*)$ bewertet es den Zielwert. So beseitigt es die Maximierungsverzerrung nach oben.

<!--en:f9a73ce6-->
::: {.slide title="Planung nach dem Prinzip der Sicherheitsäquivalenz in einem gelernten Modell"}

<!--en:afa21842-->
[Dateneffizienz modellfreier und modellbasierter Verfahren]{.kicker}

<!--en:5290c04b-->
Statt jeden Übergang $(s, a, r, s')$ nach einem TD-Schritt zu verwerfen, können wir die Übergänge zählen. Aus den Anzahlen $N(s, a, s')$ erhalten wir $\hat{P}(s' \mid s, a) = N(s, a, s')/N(s, a)$ und wenden die Wertiteration auf $(\hat{P}, \hat{r})$ an:

<!--en:5cac4c79-->
- **Globale und lokale Ausbreitung:** Bei 500 Episoden ($\approx 10{,}000$ Schritte) erzielt die Planung in $\hat{P}$ eine mediane Erfolgsquote von $63.9\%$ gegenüber $48.6\%$ bei Q-Learning. Jeder Durchlauf der Wertiteration gibt Belohnungen über alle aufgezeichneten Übergänge zugleich weiter.

<!--en:7cd8a964-->
::: {.slide title="Verständnisfragen zur Steuerung mit zeitlichen Differenzen"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:9da0f1b0-->
Prüfen Sie Ihr Verständnis von TD-Aktualisierungen, Schrittweiten und empirischen Modellen:

<!--en:d5a2d7fa-->
1. **TD-Aktualisierung von Hand:** Es gelte $Q(14, \textrm{right}) = 0.25$. Beim nächsten Besuch rutscht der Agent mit *rechts* ins Ziel, einen Endzustand ($r = 1$). Wie lauten bei $\alpha = 0.5$ der TD-Fehler $\delta$ und der aktualisierte Eintrag?

<!--en:9c0deb6d-->
2. **Robbins-Monro-Bedingungen:** Warum erfüllt $\alpha_n = 1/(1 + n)$ die Voraussetzungen des Konvergenzsatzes von Robbins und Monro, schneidet aber nach $4{,}000$ Episoden auf dem rutschigen FrozenLake schlecht ab?

<!--en:7f8b737e-->
3. **Schwachstelle des Modells:** Was nimmt der empirische Übergangskern $\hat{P}$ bei der Planung nach dem Prinzip der Sicherheitsäquivalenz über ein Zustand-Aktion-Paar $(s, a)$ an, wenn der Agent es nur einmal ausprobiert hat und dabei zufällig in ein Loch gerutscht ist?

<!--en:557b98b6-->
**Antworten:**

<!--en:a6beb8c3-->
1. Das Ziel ist ein Endzustand. Daher beträgt der Zielwert $y = 1.0$, der TD-Fehler ist $\delta = 1.0 - 0.25 = +0.75$, und es gilt $Q(14, \textrm{right}) \leftarrow 0.25 + 0.5(0.75) = 0.625$.

<!--en:ba413a81-->
2. $\alpha_n = 1/(1+n)$ gewichtet alle Zielwerte gleich (Stichprobenmittelwert mit $1/n$). Da das Bootstrapping die frühen Zielwerte aus einer Nulltabelle gewinnt, braucht es exponentiell viele Besuche in $1/(1-\gamma) = 20$, bis diese Nullen ihren Einfluss verlieren.

<!--en:c37e56e5-->
3. Bei $N(s, a) = 1$ gilt $\hat{P}(\textrm{hole} \mid s, a) = 1/1 = 1.0$. Die Planung vertraut der empirischen Schätzung vollständig und nimmt an, dass $(s, a)$ mit Sicherheit in ein Loch führt, bis die Exploration einen weiteren Besuch erzwingt.

<!--en:492b4da4-->
::: {.slide title="Modellfreie und modellbasierte Steuerung mit Tabellen im Überblick"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:59a03092-->
| Verfahren | Zielwert / Aktualisierungsregel | Strategietyp und wesentliche Eigenschaften |

<!--en:1868d39f-->
| **Monte Carlo** | $Q(s_t, a_t) \leftarrow Q + \alpha(R_t - Q(s_t, a_t))$ | Unverzerrter Zielwert, hohe Varianz über den gesamten Rest der Episode |

<!--en:b285a5c7-->
| **SARSA** | $Q(s, a) \leftarrow Q + \alpha(r + \gamma Q(s', a') - Q(s, a))$ | On-Policy-TD-Verfahren, lernt $Q^{\pi_e}$ einschließlich des Explorationsrisikos |

<!--en:7bc3bc82-->
| **Q-Learning** | $Q(s, a) \leftarrow Q + \alpha(r + \gamma \max_{a'} Q(s', a') - Q(s, a))$ | Off-Policy-TD-Verfahren mit Semigradienten, konvergiert unter den Robbins-Monro-Bedingungen gegen $Q^*$ |

<!--en:e037837b-->
| **Double Q-Learning** | $y = r + \gamma Q_B(s', \mathrm{argmax}_{a'} Q_A(s', a'))$ | Trennt Aktionswahl und Bewertung und beseitigt so die Maximierungsverzerrung |

<!--en:d03f4730-->
| **Empirische Planung** | Löse $V = \mathcal{T}_{\hat{P}, \hat{r}} V$ für $\hat{P} = N(s,a,s')/N(s,a)$ | Hohe Dateneffizienz, Speicherbedarf des Modells $O(|\mathcal{S}|^2|\mathcal{A}|)$ |
