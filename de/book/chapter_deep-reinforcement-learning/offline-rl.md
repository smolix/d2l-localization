<!--en:2f930a8b-->
# Offline Reinforcement Learning

<!--en:0cc7f509-->
Alle bisherigen Verfahren in diesem Kapitel konnten etwas ausprobieren. Wenn Q-Learning eine Aktion überschätzte, führte der Agent sie irgendwann aus, beobachtete das Ergebnis und korrigierte seine Tabelle. Der Planer aus :numref:`sec_model_based` vertraute seinem gelernten Modell nur in der Nähe der Zustände, die er in seinen eigenen Episoden besucht hatte. Bei Bedarf sammelte er weitere Episoden. Nehmen wir nun an, es kommen keine Episoden mehr hinzu. Ein Krankenhaus hat Behandlungsdaten aus vielen Jahren, will aber keine Experimente an seinen Patienten durchführen. Ein Unternehmen hat die Protokolle des Empfehlungssystems, das es im Vorjahr eingesetzt hat. Ein Robotiklabor hat stundenlange Aufzeichnungen von Demonstrationen mit ferngesteuerten Robotern. Lässt sich allein aus solchen Aufzeichnungen eine Strategie (engl. policy) lernen, und wie gut kann sie sein?

<!--en:ef2bae7f-->
Die Aufzeichnungen stammen von einer *Verhaltensstrategie* (engl. behavior policy) $\pi_e$. Da der Zielwert beim Q-Learning unabhängig davon ist, wer die Aktion gewählt hat, können wir Q-Learning auf die protokollierten Übergänge anwenden. Die Schwierigkeit zeigt sich im nächsten Schritt. Um die Verhaltensstrategie zu übertreffen, muss die gelernte Strategie Aktionen bevorzugen, welche die Verhaltensstrategie selten gewählt hat. Gerade deren Werte beruhen auf den wenigsten Samples. Das Maximum im Zielwert wählt die höchsten Schätzwerte aus. Diese sind oft nur durch Zufall so hoch, und kein neuer Versuch wird sie je überprüfen. Abhilfe schafft *Pessimismus*: Je weniger Daten einer Schätzung zugrunde liegen, desto weniger vertrauen wir ihr. Das ist das Spiegelbild des Optimismus, der in :numref:`sec_exploration` die Exploration antrieb.

<!--en:63bd9e2f-->
Wir arbeiten auf dem See mit rutschigem Eis aus :numref:`sec_valueiter`. Dort kennen wir das exakte Optimum und können damit jeden Wert überprüfen, den eine gelernte Tabelle vorhersagt. Zunächst fragen wir, welche Aktualisierungsregeln überhaupt mit Daten einer anderen Strategie arbeiten können. Dann lernen wir aus zwei festen Protokollen und messen, wie weit die Vorhersagen über dem liegen, was eine Strategie höchstens erzielen kann. Dabei sehen wir, welche Fehler Pessimismus behebt und wann er überflüssig ist.

<!--en:f42c105a-->
## Lernen aus einem festen Protokoll

<!--en:b3c53532-->
### Zielwerte für Daten einer anderen Strategie

<!--en:3954d209-->
Die Aktualisierungsregeln der letzten beiden Kapitel fallen in zwei Gruppen. Ein Strategiegradient ist ein Erwartungswert über die Trajektorien der aktuellen Strategie. Mit Samples einer anderen Strategie schätzen wir daher eine andere Größe. REINFORCE und Akteur-Kritiker-Verfahren gehören deshalb zu den *On-Policy*-Verfahren. PPO kann einen Batch nur deshalb über einige Epochen hinweg nutzen, weil es die Wahrscheinlichkeit jeder Aktion mit einem Wahrscheinlichkeitsverhältnis korrigiert. Der Zielwert $r + \gamma \max_{a'} \hat{Q}(s', a')$ beim Q-Learning hängt davon ab, wie die Umgebung auf $(s, a)$ reagiert hat. Welche Strategie $a$ gewählt hat, spielt keine Rolle. Ein Übergang aus einer früheren Strategie, von einem anderen Agenten oder aus einem festen Protokoll liefert also ein gültiges Sample für dasselbe Backup. Q-Learning gehört damit zu den *Off-Policy*-Verfahren. :numref:`fig_rl_data_rules` stellt die drei Arten der Datennutzung gegenüber.

<!--en:b5a22b83-->
![Drei Arten der Datennutzung. (a) On-Policy: Jeder Batch stammt von der aktuellen Strategie und ist nach der Aktualisierung veraltet. Die Wahrscheinlichkeitsverhältnisse bei PPO erlauben die erneute Nutzung über einige Epochen. (b) Off-Policy: Ein Puffer bewahrt ältere Übergänge auf, weil der Bellman-Zielwert unabhängig von der Strategie ist, die sie erzeugt hat. (c) Offline: Der Agent durchläuft einen festen Datensatz mehrfach. Kein neuer Übergang prüft die Aktionen, welche die Strategie im Einsatz wählt.](../img/mdl-rl-data-rules.svg)

<!--en:ddf5883c-->
SARSA, die On-Policy-Variante des Q-Learnings aus :numref:`sec_qlearning`, verwendet beim Bootstrapping die Aktion $a'$, welche die Verhaltensstrategie in $s'$ tatsächlich ausgeführt hat, anstelle der besten Aktion. Zur Erinnerung hier der Temporal-Difference-Fehler aus :numref:`sec_qlearning`:

<!--en:8ac89a88-->
Mitteln wir über $a' \sim \pi_e(\cdot \mid s')$, erhalten wir als Zielwert das Bellman-Erwartungswert-Backup von $\pi_e$. SARSA lernt also $Q^{\pi_e}$, die Aktionswerte der Strategie, von der die Daten stammen. Aus einem fremden Protokoll lernt SARSA nur etwas über die fremde Strategie. Q-Learning lernt $Q^*$ aus jedem Protokoll. Übung :exref:`ex_qlearning-cliff-walking` aus :numref:`sec_qlearning` vergleicht beide Verfahren am Beispiel der Klippe.

<!--en:06b062ee-->
Auch bei einem Off-Policy-Verfahren reichen beliebige Daten nicht aus. Ein Tabelleneintrag beim Q-Learning ist nur so gut wie die Übergänge, auf denen er beruht. Für jeden Zustand, den die gierige Strategie erreicht, muss ihr gewähltes Zustand-Aktion-Paar also schon erprobt sein, und zwar oft genug für eine Wertschätzung. Diese Anforderung heißt *Abdeckungsbedingung* (engl. coverage condition). Online verbessert ein Agent eine lückenhafte Abdeckung, indem er dorthin geht, wo seine Schätzungen unsicher sind. Beim *Offline Reinforcement Learning* erhält der Agent ausschließlich einen festen Datensatz :cite:`Levine.Kumar.Tucker.ea.2020`. Er muss mit dessen Abdeckung auskommen.

<!--en:382edc4e-->
### Zwei Protokolle vom See mit rutschigem Eis

<!--en:cb0e9a43-->
Der See mit rutschigem Eis hat 16 Felder und 4 Aktionen. Elf Felder sind weder Loch noch Ziel. Dem Agenten stehen also 44 Zustand-Aktion-Paare zur Verfügung, und bei jedem Zug rutscht er in zwei von drei Fällen seitwärts. Für das Experiment kennen wir die Übergangstabelle des Sees und lesen sie in ein `d2l.TabularMDP` ein. Es enthält die Arrays $P(s' \mid s, a)$ und $r(s, a)$ aus :numref:`sec_mdp`. Die lernenden Agenten haben keinen Zugriff auf die Tabelle. Wir ziehen je 40 Protokolle zweier Arten:

<!--en:0ebe2ebe-->
Die Wertiteration aus :numref:`sec_valueiter` liefert auf dem wahren Modell den Vergleichsmaßstab. Vom Startfeld aus kann keine Strategie mehr als $V^*(s_0)$ erzielen. Aus einer Tabelle von Aktionswerten erzeugt `greedy` die zugehörige gierige Strategie als Tabelle von Aktionswahrscheinlichkeiten. Wie `d2l.epsilon_greedy` verteilt die Funktion die Wahrscheinlichkeit bei Gleichstand gleichmäßig. Eine Regel, die bei Gleichstand immer die erste Aktion wählt, würde bei jeder Nullzeile nach links gehen. Mit `mdp.policy_value` bewerten wir eine solche Tabelle exakt. Die Funktion löst dazu die linearen Bellman-Gleichungen $V^\pi = (I - \gamma P_\pi)^{-1} r_\pi$ aus :numref:`sec_valueiter`. Alle Werte, die eine Strategie im Folgenden *erzielt*, berechnen wir auf diese Weise ohne Rollouts.

<!--en:4dbcc763-->
Die erste Art von Protokoll stammt von einem Agenten ohne jedes Können. In 500 Episoden wählt er seine Aktionen gleichverteilt zufällig. Er probiert überall jede Aktion aus, erreicht das Ziel aber selten. Die zweite Art stammt von einem kompetenten, aber nicht perfekten Bediener. In 100 Episoden wählt er seine Aktionen $\epsilon$-gierig nach $Q^*$ mit $\epsilon = 0.2$. Damit wählt er mit Wahrscheinlichkeit $0.8 + 0.2/4 = 0.85$ die beste Aktion. Auf die übrigen Aktionen entfällt die restliche Wahrscheinlichkeit zu gleichen Teilen. Wir nennen dieses Protokoll *eng*, weil sich seine Samples auf die bevorzugten Aktionen des Bedieners konzentrieren. `make_log` erzeugt ein Protokoll mit `d2l.rollout`. Diese Funktion führt eine Strategie über vollständige Episoden aus und liefert die Übergänge als `d2l.Batch` mit den Feldern `obs`, `act`, `rew`, `next_obs` und `term`. Der Zufallsstartwert legt das Protokoll fest. `count` ermittelt die Besuchszahl $n(s, a)$, also wie oft jedes Paar im Protokoll vorkam. Jeder Eintrag von `logs` enthält auch die Tabelle der Aktionswahrscheinlichkeiten der Verhaltensstrategie. So können wir die Strategie selbst bewerten:

<!--en:42c3e24e-->
Der Zufallsagent erzielt vom Startfeld aus $0.008$, etwa ein Zwanzigstel des Optimums $0.180$. Dafür erprobt er in seinen 500 Episoden jedes Paar, für das eine Aktion möglich ist, ein typisches Paar etwa vierzigmal. Der Bediener erzielt $0.087$, ungefähr die Hälfte des Optimums, mit einem nur halb so langen Protokoll. Seine Samples häufen sich bei den Aktionen, die er bevorzugt. Zwei Paare bleiben unerprobt, und auf ein typisches Paar entfallen nur acht Samples. Abdeckung und Können stehen hier im Widerstreit. Die beiden Protokolle veranschaulichen die beiden Extreme.

<!--en:a1df4a35-->
### Verteilungsverschiebung im Vergleich mit dem exakten Optimum

<!--en:30505c11-->
Q-Learning durchläuft ein Protokoll wie eine Umgebung. `offline_q_sweeps` mischt die protokollierten Übergänge und wendet die Aktualisierungsregel aus :numref:`sec_qlearning` mit der abnehmenden Schrittweite $1/(1 + 0.1\, n)$ aus `d2l.q_learning` an. Dabei zählt $n$ die bisherigen Aktualisierungen des jeweiligen Paares. Insgesamt sind es 200 Durchläufe:

<!--en:1de5e643-->
Jeder Durchlauf verwendet dieselben Übergänge. Daher nähert sich die Tabelle einem Fixpunkt, der allein vom Protokoll abhängt. Durch Auszählen des Protokolls schätzen wir ein Modell: $\hat{P}(s' \mid s, a)$ ist der Anteil der Versuche mit $(s, a)$, die in $s'$ endeten, und $\hat{r}(s, a)$ ihre mittlere Belohnung. Der Fixpunkt ist die Tabelle der optimalen Aktionswerte dieses *empirischen MDP*, also die Tabelle nach dem Prinzip der Sicherheitsäquivalenz aus :numref:`sec_qlearning`. `offline_q` berechnet sie direkt durch Wertiteration auf dem ausgezählten Modell. Das Argument `kappa` steuert den Strafterm, den wir in jedem Backup für das jeweilige Paar abziehen. Bis wir weiter unten den Pessimismus einführen, bleibt er null. Ein unerprobtes Paar hat eine Nullzeile, führt also nirgendwohin und hat den Wert null. Weil die Belohnungen auf diesem See nie negativ sind, ist null der kleinste mögliche Wert einer Strategie. Wir begrenzen die Werte nach unten auf null, damit der Strafterm sie nicht unter diese Grenze drückt. Bei positivem `kappa` belegt die Funktion auch unerprobte Paare mit einer unendlichen Strafe. Durch die Begrenzung erhalten sie wieder den Wert null. Der Pessimismus belässt sie also bei dem Wert, den ihnen schon die naive Tabelle zuweist:

<!--en:f948b945-->
Stimmen Schleife und direkte Lösung überein? Wir wenden beide auf das erste Protokoll der Gleichverteilungsstrategie an:

<!--en:2942d193-->
Die beiden Tabellen unterscheiden sich in jedem Eintrag um weniger als ein Hundertstel. Ihre gierigen Strategien erzielen denselben Ertrag. Die Schleife benötigt pro Protokoll Sekunden, die direkte Lösung Millisekunden. Deshalb verwenden die folgenden Experimente `offline_q`. Übung :exref:`ex_offline-rl-dataset-size` untersucht, wie der verbleibende Tabellenfehler mit wachsendem Protokoll abnimmt.

<!--en:e7c62b4e-->
Das zweite Lernverfahren ist das Verhaltensklonen (engl. behavior cloning) aus :numref:`sec_imitation`. Es bildet die Strategie nach, die das Protokoll erzeugt hat. Bei einer Tabelle ergibt das Training mit Kreuzentropie für jeden Zustand die relativen Aktionshäufigkeiten im Protokoll. Deshalb zählt `clone_policy` die Aktionen aus. In Zuständen, die im Protokoll nie vorkamen, wählt die Funktion die Aktionen gleichverteilt. Der Klon sagt voraus, was die aufgezeichnete Strategie erzielt hat, also den mittleren diskontierten Ertrag ihrer Episoden:

<!--en:26db761b-->
Für jedes der 40 Protokolle jeder Art passen wir sowohl die naive Tabelle ohne Strafterm als auch den Klon an. Wir halten jeweils fest, welchen Wert das Verfahren für das Startfeld vorhersagt und was seine Strategie dort erzielt. Für die Tabelle erledigt dies `fit_and_score` mit einem beliebigen Strafparameter `kappa`. Die `Batch`-Methode `episode_returns` liefert den diskontierten Ertrag jeder protokollierten Episode. Daraus ergibt sich die Vorhersage des Klons:

<!--en:1cdc9338-->
`summary` gibt die Mediane über die 40 Protokolle aus. Außerdem gibt die Funktion an, bei wie vielen Protokollen die Vorhersage das Optimum übersteigt und bei wie vielen sie den exakten Wert der eigenen Strategie übersteigt. `tally` zählt, bei wie vielen Protokollen die Strategie eines lernenden Agenten mehr, weniger oder genauso viel erzielt wie die eines anderen:

<!--en:d495e091-->
Bei 36 von 40 Protokollen der Gleichverteilungsstrategie sagt die naive Tabelle mehr vorher, als ihre eigene Strategie erzielt. Bei 28 sagt sie sogar mehr vorher, als irgendeine Strategie erzielen kann. Der Median der Vorhersagen liegt mit etwa $0.20$ über dem Optimum $0.18$. Bei den engen Protokollen ist die Lücke größer. Bei 39 von 40 Protokollen übersteigt die Vorhersage $V^*$. Der Median liegt bei $0.25$, während die Strategien nur etwa $0.1$ erzielen. Der Vergleich zwischen Vorhersage und tatsächlichem Ertrag gehört zur üblichen Diagnose beim Offline Reinforcement Learning. Bei tiefen Netzen kann die Lücke weit größer sein :cite:`Fujimoto.Meger.Precup.2019`.

<!--en:193df6b7-->
Woher kommen die überhöhten Vorhersagen? Wir betrachten das erste enge Protokoll. Für jedes erprobte Paar aus einem nichtterminalen Feld tragen wir auf der waagerechten Achse seine Besuchszahl $n(s, a)$ und auf der senkrechten den Fehler $\hat{Q}(s, a) - Q^*(s, a)$ der naiven Tabelle auf. Für jedes Feld markieren wir das Paar, das die gierige Strategie wählt. Bei Gleichstand markiert `argmax` die erste Aktion, während `greedy` die Wahrscheinlichkeit gleichmäßig verteilt. Ein solcher Gleichstand ist in diesen Protokollen selten:

<!--en:5b305700-->
Bei jedem Paar mit mehr als zwanzig Samples fällt die Schätzung um etwa $0.05$ bis $0.15$ zu hoch aus. Die eigenen Übergänge dieser Paare sind gut geschätzt. Ihre Zielwerte beruhen beim Bootstrapping aber auf den Werten späterer Felder. Diese enthalten wiederum die Fehler der nachfolgenden Felder, die das Maximum in jedem späteren Backup nach oben verzerrt. Bei weniger als etwa zehn Samples streuen die Fehler in beide Richtungen, von $-0.15$ bis $+0.26$. Unter den selten erprobten Aktionen, welche die gierige Strategie wählt, finden sich die beiden größten Fehler der Abbildung. Zwei gewählte Aktionen beruhen auf je zwei Samples und eine auf fünf. Zwei davon werden um $0.26$ und $0.20$ überschätzt, also um mehr als das Optimum selbst. Bei einem einzelnen Protokoll könnte das Zufall sein. Deshalb fassen wir die gewählten Paare aus allen 40 engen Protokollen zusammen. Wir vergleichen die gewählten Paare mit einem bis fünf Samples mit allen Paaren, die ebenso selten vorkommen:

<!--en:6587a646-->
Etwa ein Viertel der von der gierigen Strategie gewählten Paare beruht auf höchstens fünf Samples. Die Überschätzung beträgt bei diesen Paaren etwa $0.2$, das Zwei- bis Dreifache des typischen Fehlers bei so wenigen Samples. Das Maximum wählt unter den selten erprobten Paaren nicht zufällig aus. Es wählt gerade jene, deren Fehler zufällig positiv ist. Das ist die Maximierungsverzerrung aus :numref:`subsec_dqn-max-bias`. Das nächste Backup gibt die überhöhten Werte an die vorhergehenden Felder weiter. Die so entstehende Abweichung zwischen der Verteilung der protokollierten Aktionen und derjenigen der gelernten Strategie heißt *Verteilungsverschiebung* (engl. distribution shift). Online würde sie sich von selbst korrigieren. Der Agent würde die überschätzte Aktion ausführen und Übergänge sammeln, die ihren Wert nach unten korrigieren. Dann würde er andere Aktionen wählen. Offline bleibt das Protokoll unverändert, also bleibt auch der Fehler bestehen.

<!--en:ec50538d-->
### Der Klon und das Verknüpfen von Teiltrajektorien

<!--en:f6326055-->
Der Klon ist nahezu kalibriert. Seine Vorhersage ist der mittlere Ertrag im Protokoll selbst und damit eine erwartungstreue Schätzung des Werts der Verhaltensstrategie. Der Wert des Klons liegt nahe an diesem Wert. Bei etwa der Hälfte der Protokolle liegt die Vorhersage über dem Wert des Klons, bei den übrigen darunter. Bei den Protokollen der Gleichverteilungsstrategie ist genau diese Eigenschaft seine Schwäche. Er reproduziert eine Irrfahrt und erzielt $0.01$.

<!--en:afd2fd00-->
Bei allen 40 Protokollen der Gleichverteilungsstrategie übertrifft die naive Tabelle den Klon. Sie erzielt etwa $0.14$, drei Viertel des Optimums, obwohl die Strategie, von der ihre Daten stammen, nur $0.008$ erzielte. Das liegt am Backup für jeden einzelnen Übergang. Nehmen wir an, eine zufällige Episode erreicht Feld 10 und endet dann in einem Loch. Eine andere erreicht Feld 10 auf einem anderen Weg und rutscht weiter bis zum Ziel (:numref:`fig_rl_stitching`). Das Backup bewertet Feld 10 anhand der Fortsetzung der zweiten Episode. Diesen Wert gibt es entlang des Wegs der ersten Episode zurück. So setzt es einen Weg zusammen, den keine einzelne Episode genommen hat. Dieses *Verknüpfen von Teiltrajektorien* (engl. stitching) ermöglicht es dem Offline-Q-Learning, die Strategie zu übertreffen, aus der seine Daten stammen. Bei den engen Protokollen wird der Klon zu einer starken Baseline. Er kopiert einen Bediener, der $0.087$ erzielt, und erreicht selbst etwa $0.09$. Die naive Tabelle lässt sich durch selten erprobte Aktionen in die Irre führen und übertrifft ihn nur bei 27 von 40 Protokollen.

<!--en:78161965-->
![Verknüpfen von Teiltrajektorien auf dem See. (a) Zwei protokollierte Episoden führen über Feld 10 (umrandet). Die erste (blau) führt am linken Rand hinunter und endet in Loch 11. Die zweite (orange) folgt der obersten Reihe und führt weiter zum Ziel. (b) Das Backup für jeden Übergang verbindet den Weg der ersten Episode mit der Fortsetzung der zweiten. Diesen Weg ist keine der protokollierten Episoden gegangen.](../img/mdl-rl-stitching.svg)

<!--en:8fa19aaa-->
## Pessimismus

<!--en:4f4d0fa3-->
### Ein zweiarmiger Bandit von Hand gerechnet

<!--en:e25527a6-->
Bevor wir das Lernverfahren für den See ändern, betrachten wir den einfachsten Fall. Ein Offline-Protokoll für einen zweiarmigen Banditen enthält 100 Züge an Arm 1, der immer $1/2$ auszahlt, und einen einzigen Zug an Arm 2. Dieser zahlt den Betrag $1$ mit Wahrscheinlichkeit $1/4$ aus, sonst $0$. Nehmen wir an, der einzige Zug an Arm 2 hat $1$ ergeben. Das passiert in einem von vier Fällen. Dann hat Arm 2 mit $1$ gegenüber $0.5$ den höheren empirischen Mittelwert. Wer ihn wählt, erzielt im Mittel $0.25$.

<!--en:a8864dce-->
Für eine *untere Konfidenzschranke* (engl. lower confidence bound) ziehen wir von jedem empirischen Mittelwert $\hat{r}(a)$ einen Radius ab, der mit der Zahl der Züge $n(a)$ am jeweiligen Arm abnimmt. Für Belohnungen in $[0, 1]$ liefert die Hoeffding-Ungleichung den Radius

<!--en:cc4828e8-->
für $K$ Arme und eine Fehlerwahrscheinlichkeit $q$. Mit Wahrscheinlichkeit mindestens $1 - q$ weicht der wahre Mittelwert jedes Arms um höchstens $b(a)$ von seinem empirischen Mittelwert ab. Mit $K = 2$ und $q = 0.1$ erhalten wir $b(1) = \sqrt{\log 40 / 200} \approx 0.136$ und $b(2) = \sqrt{\log 40 / 2} \approx 1.358$. Die unteren Schranken betragen $0.5 - 0.136 = 0.364$ für Arm 1 und $1 - 1.358 = -0.358$ für Arm 2. Die Regel wählt also Arm 1 :cite:`Rashidinejad.Zhu.Ma.ea.2021`. Mitteln wir über die möglichen Auszahlungen des einzelnen Zugs, erzielt die Wahl des empirisch besten Arms $\frac34 \cdot 0.5 + \frac14 \cdot 0.25 \approx 0.44$. Die Regel mit der unteren Schranke erzielt jedes Mal $0.5$.

<!--en:878254ba-->
Pessimismus hat seinen Preis. Würde Arm 2 im Mittel $0.7$ auszahlen und läge sein empirischer Mittelwert bei $0.7$, würde die Regel ihn so lange ablehnen, bis $0.7 - \sqrt{\log 40 / (2 n_2)}$ den Wert $0.364$ übersteigt. Das geschieht erstmals nach $n_2 = 17$ Zügen. Ein pessimistischer Agent verzichtet also auf gute Aktionen, bis die Daten für sie sprechen.

<!--en:e09a48e7-->
### Der Strafterm in einer Zeile

<!--en:5e9272b0-->
Auf den lernenden Agenten auf dem See wenden wir dasselbe Prinzip an. Jeder Eintrag der naiven Tabelle mittelt $n(s, a)$ verrauschte Zielwerte. Sein Stichprobenfehler nimmt daher wie $1/\sqrt{n(s, a)}$ ab. Für $\kappa > 0$ zieht `offline_q` in jedem Backup den Strafterm $\kappa/\sqrt{n(s, a)}$ ab und begrenzt das Ergebnis nach unten auf null:

<!--en:11af50cb-->
Die Aktualisierungsregel entspricht bis auf den Strafterm der naiven Regel. Der Strafterm ist dort am größten, wo das Protokoll die wenigsten Daten liefert. Ein nie erprobtes Paar bleibt mit und ohne Strafterm bei der Untergrenze null. Das ist die tabellarische Variante der pessimistischen Wertiteration :cite:`Jin.Yang.Wang.2021`. Hier ist $\kappa$ ein frei wählbarer Skalierungsfaktor. Die Konstante der Hoeffding-Ungleichung können wir nicht übernehmen, weil die Zielwerte eines Paares beim Bootstrapping auf denselben geschätzten Folgewerten beruhen und daher voneinander abhängen.

<!--en:c577d488-->
Der Strafterm ist das Spiegelbild des Bonus bei der Regel mit oberer Konfidenzschranke aus :numref:`sec_exploration`. Diese Regel wählt den Arm mit dem größten Wert von $\hat{\mu}(a) + \kappa\sqrt{\log t / n(a)}$. Beide Terme nehmen wie $1/\sqrt{n}$ ab. Die Vorzeichen unterscheiden sich, weil die Rückmeldung unterschiedlich ausfällt. Online bewegt eine optimistische Schätzung den Agenten dazu, die Aktion zu erproben. Der Versuch korrigiert die Schätzung. Offline ist kein Versuch möglich. Deshalb ziehen wir dieselbe Unsicherheit ab und wählen eine Aktion nur dann, wenn die Daten sie stützen.

<!--en:a78be2ac-->
### Daten mit enger und mit gleichmäßiger Abdeckung

<!--en:5c2eeed9-->
Wir wenden `offline_q` erneut auf dieselben 80 Protokolle an, diesmal also das Backup :eqref:`eq_offline_pessimistic` mit $\kappa = 0.1$:

<!--en:bb964254-->
Bei den engen Protokollen bestätigt sich, was das Banditenbeispiel erwarten ließ. Die Vorhersage sinkt von $0.25$ auf etwa $0.13$ und übersteigt nur bei 4 Protokollen das Optimum. Bei 29 von 40 Protokollen erzielt die Strategie mehr als die naive Tabelle, bei einem weniger. Der Median ihres Ertrags liegt bei etwa $0.14$. Den Klon übertrifft sie bei 39 von 40 Protokollen.

<!--en:97b68437-->
Bei den Protokollen der Gleichverteilungsstrategie beseitigt derselbe Strafterm die Überschätzung. Keine Vorhersage liegt mehr über dem Optimum. Der Ertrag sinkt aber bei 28 von 40 Protokollen und steigt nur bei 4. Jede Aktion wurde mehrere Dutzend Mal erprobt. Daher gibt es kaum zufällig zu hohe Schätzungen, vor denen wir uns schützen müssten. Ein von Paar zu Paar unterschiedlicher Strafterm ändert hier nur, welche der gut belegten Aktionen am besten erscheint. Die Theorie des pessimistischen Offline-Lernens erklärt, wann wir welches Ergebnis erwarten können. Für die Garantie des Verfahrens mit unterer Konfidenzschranke genügt es, wenn das Protokoll die Paare der *optimalen* Strategie abdeckt. Es muss nicht jedes Paar abdecken :cite:`Rashidinejad.Zhu.Ma.ea.2021`. Das enge Protokoll deckt die Aktionen der optimalen Strategie gut ab, alle anderen nur spärlich. Hier hilft Pessimismus. Ein Protokoll der Gleichverteilungsstrategie deckt alles ab, und die naive Tabelle braucht keinen Schutz.

<!--en:7cc44447-->
### Wahl des Strafterms

<!--en:e7d8e68d-->
Wie groß sollte $\kappa$ sein? Ein größerer Wert macht die Tabelle vorsichtiger. Wir erproben $\kappa = 0.3$ an denselben Protokollen:

<!--en:d4261d1b-->
Bei $\kappa = 0.3$ fällt der Startwert bei fast jedem Protokoll auf null. Bei 36 der 40 engen Protokolle und bei allen 40 Protokollen der Gleichverteilungsstrategie unterliegt die Strategie der naiven Tabelle. Der Grund liegt darin, dass bei jedem Backup ein Strafterm anfällt und ein Wert eine diskontierte Summe entlang eines Pfads ist. Hat jedes Paar auf einem Pfad etwa $n$ Samples, summieren sich die Strafterme auf höchstens etwa $\kappa / (\sqrt{n}\,(1 - \gamma))$. Bei $\gamma = 0.95$ ist das das Zwanzigfache des Strafterms pro Schritt. Da die Episoden am Ziel oder in einem Loch enden, fallen auf einem Pfad tatsächlich weniger Strafterme an. Selbst die Schranke bleibt unter $V^*(s_0) = 0.18$, wenn bei $\kappa = 0.3$ jedes Paar auf dem Pfad über tausend Samples hat. Die Protokolle mit 100 Episoden enthalten am Startfeld mehrere Hundert Samples und weiter entfernt weit weniger. Mit fünfmal so vielen Daten desselben Bedieners sinkt der Strafterm pro Schritt um den Faktor $\sqrt{5}$. Die meisten Paare auf dem optimalen Pfad haben dann einige Hundert Samples, weniger als die für die Schranke nötigen tausend. Dennoch richtet, wie Übung :exref:`ex_offline-rl-how-much-pessimism` zeigt, $\kappa = 0.3$ dort kaum Schaden an. Der tatsächlich anfallende Abzug ist schon klein genug. Der Strafterm muss zur Datenmenge passen. Die Wahl von $\kappa$ ist also eine Frage der Modellauswahl.

<!--en:49991b7b-->
### Kalibrierung und Modellauswahl

<!--en:75a22ff4-->
Alle bisherigen Vergleiche nutzten das exakte Modell des Sees, das uns im Experiment zur Verfügung steht. Im Offline-Einsatz haben wir nur das Protokoll. Für jeden Wert von $\kappa$ oder jede gelernte Tabelle müssen wir allein daraus schätzen, wie gut eine in Betracht gezogene Strategie abschneiden würde. Ein Schätzer gewichtet die protokollierten Episoden neu. Dazu bildet er für jede Aktion das Verhältnis ihrer Wahrscheinlichkeiten unter der untersuchten Strategie und der Verhaltensstrategie. Das Gewicht einer Episode ist das Produkt dieser Verhältnisse über alle ihre Schritte. Wie in :numref:`sec_ppo` wächst die Varianz dieses Produkts mit dem Horizont. Für eine gierige Strategie erhalten die meisten Episoden das Gewicht null. Ein anderer Schätzer bewertet die Strategie im empirischen MDP aus einem zurückgehaltenen Teil des Protokolls. Das ist nur dort zuverlässig, wo dieser Teil die gewählten Aktionen abdeckt. Die Kalibrierung lässt sich zumindest teilweise prüfen. Eine Tabelle ist verdächtig, wenn sie mehr als den besten diskontierten Ertrag ihres eigenen Protokolls vorhersagt. Auf dem See schlägt diese Prüfung nie an. Die beste Episode jedes Protokolls erzielt etwa $0.5$ oder mehr und liegt damit über jeder Vorhersage in diesem Abschnitt. In Übung :exref:`ex_offline-rl-what-you-cannot-measure` erproben Sie zwei Schätzer an den Protokollen der Gleichverteilungsstrategie.

<!--en:c25fbbb0-->
## Jenseits der Tabelle

<!--en:6d4ab51b-->
### Nebenbedingungen für die Strategie oder Strafterme für die Werte

<!--en:2591873f-->
Bei kontinuierlichen Zuständen und Aktionen stehen uns keine Besuchszahlen zur Verfügung. Die beiden Ansätze dieses Abschnitts bleiben aber anwendbar. Wir können die Strategie nahe an den Aktionen halten, die das Protokoll ausreichend belegt, oder die Werte der übrigen Aktionen senken. Der erste Ansatz verwendet die KL-regularisierte Zielfunktion aus :numref:`sec_regularized` mit der Verhaltensstrategie als Referenz. In jedem Zustand maximiert die Strategie

<!--en:3b7c239b-->
und eine Aktion, die im Protokoll nie vorkam, erhält eine unendliche Strafe. In der Praxis ist $\pi_e$ unbekannt, und wir verwenden an ihrer Stelle das Protokoll. TD3+BC ergänzt die Zielfunktion des Akteurs beispielsweise um den quadrierten Abstand zwischen der Aktion der Strategie und der protokollierten Aktion :cite:`Fujimoto.Gu.2021`. Bis auf eine Konstante entspricht dieser Abstand im Quadrat der negativen Log-Likelihood der protokollierten Aktion unter einer Gauß-Verteilung, deren Zentrum die Aktion der Strategie ist. Der Term schätzt also anhand eines Samples die Divergenz in der umgekehrten Richtung, $D_{\textrm{KL}}(\pi_e \Vert \pi)$. Am Ende des nächsten Abschnitts lernt die direkte Präferenzoptimierung aus einem festen Datensatz von Präferenzen zwischen Antwortpaaren. Dabei gilt dieselbe Nebenbedingung, und eine Referenzstrategie übernimmt die Rolle von $\pi_e$.

<!--en:cdaba0ae-->
Der zweite Ansatz ersetzt $\kappa/\sqrt{n}$, wofür wir Besuchszahlen brauchen, durch einen Strafterm, der nur die Ausgaben des Netzes benötigt. Conservative Q-Learning (CQL) ergänzt den Temporal-Difference-Verlust um einen Term, der in einem protokollierten Zustand die Werte aller Aktionen senkt und den Wert der protokollierten Aktion anhebt. Bei diskreten Aktionen lautet der Term $\alpha_{\textrm{c}}\big(\log\sum_{a'} e^{Q(s, a')} - Q(s, a)\big)$ :cite:`Kumar.Zhou.Tucker.ea.2020`. Sein erster Teil ist ein weiches Maximum über die Aktionen. Der zugehörige Gradient senkt daher jeden Wert proportional zu seinem Softmax-Gewicht. Am stärksten sinken die Werte der scheinbar besten Aktionen, unabhängig davon, wie gut das Protokoll sie belegt. Nur der Wert der protokollierten Aktion steigt (:numref:`fig_rl_cql_push_pull`). :numref:`tab_rl_offline_families` stellt die beiden Verfahrensfamilien ihren auf Besuchszahlen beruhenden Gegenstücken in diesem Abschnitt gegenüber.

<!--en:c1c09b85-->
![Der konservative Strafterm in einem protokollierten Zustand mit sechs Aktionen, von denen $a_3$ im Protokoll steht. (a) Der Strafterm ist die Differenz zwischen dem weichen Maximum $\log\sum_{a'} e^{Q(s, a')}$ der Werte (gestrichelt) und dem Wert der protokollierten Aktion. (b) Ein Gradientenschritt der Schrittweite eins für den Strafterm senkt jeden Wert um sein Softmax-Gewicht, die größten Werte also am stärksten. Zum Wert der protokollierten Aktion addiert er eins. Nur dieser Wert steigt. Die Änderungen summieren sich zu null.](../img/mdl-rl-cql-push-pull.svg)

<!--en:9595701d-->
:Zwei Familien von Offline-Verfahren und ihre Gegenstücke auf dem See.

<!--en:c8f9ef2c-->
| Familie | Was wird eingeschränkt? | In diesem Abschnitt | Beispiel |

<!--en:6918034d-->
| Nebenbedingung für die Strategie | Die Strategie bleibt nahe an den protokollierten Aktionen | im Extremfall der Klon | TD3+BC |

<!--en:23c2e267-->
| Strafterm für die Werte | Die Werte unzureichend belegter Aktionen sinken | $\kappa/\sqrt{n}$ in `offline_q` | CQL |

<!--en:eb2221b0-->
Eine strikte Regel, die bei der Maximierung nur protokollierte Aktionen zulässt, würde auf dem See nichts ändern. Ein unerprobtes Paar hat bereits den Wert null, die Untergrenze, und setzt sich daher beim Maximieren nie durch. Die Schwierigkeit lag bei den selten erprobten Paaren, nicht bei den gänzlich unerprobten. Die weiterführende Literatur zu diesem Kapitel (:numref:`sec_deep_rl-further-reading`) vergleicht die Verfahren mit tiefen Netzen, die auf diesen beiden Ideen beruhen.

<!--en:b381d1ed-->
### Konditionierung auf den Ertrag

<!--en:83e46a25-->
Ein dritter Ansatz kommt ganz ohne Maximum aus. Decision Transformer behandelt jede protokollierte Trajektorie als Sequenz aus verbleibenden Erträgen, Zuständen und Aktionen. Er trainiert ein autoregressives Modell darauf, jede Aktion vorherzusagen. Beim Test konditioniert er das Modell auf einen gewünschten hohen Ertrag :cite:`Chen.Lu.Rajeswaran.ea.2021`. Ein kleines Netz, das mit gewöhnlichem überwachtem Lernen auf den Ertrag konditioniert, erreicht annähernd dieselbe Güte. Ein großer Teil des Gewinns geht also auf die Konditionierung auf den Ertrag zurück und nicht auf die Sequenzarchitektur :cite:`Emmons.Eysenbach.Kostrikov.ea.2022`. Bei Vergleichsaufgaben, deren Protokolle gute Trajektorien enthalten, schneiden ertragskonditionierte Verfahren etwa so gut ab wie wertbasierte. Sie fallen zurück, wenn sich eine gute Strategie erst durch Verknüpfen von Teiltrajektorien aus verschiedenen Episoden ergibt. In stochastischen Umgebungen können sie versagen, weil ein hoher protokollierter Ertrag auf Glück beruhen kann :cite:`Paster.McIlraith.Ba.2022,Brandfonbrener.Bietti.Buckman.ea.2022`.

<!--en:894ffea5-->
Schon der einfachste Fall zeigt, wie das Verfahren scheitert. In einer Aufgabe mit einem einzigen Schritt zahlt Aktion A den Betrag $1$ mit Wahrscheinlichkeit $1/2$ aus, sonst $0$. Aktion B zahlt immer $0.6$ aus. Eine Gleichverteilungsstrategie hat beide Aktionen oft protokolliert. In jeder protokollierten Episode mit Ertrag $1$ fiel die Wahl auf Aktion A. Eine auf den höchsten protokollierten Ertrag konditionierte Strategie wählt daher immer A und erzielt $0.5$, weniger als die $0.6$ von B. Die ertragskonditionierte Strategie imitiert die Episoden mit glücklichem Ausgang. Ein Agent, der Q-Learning nutzt, mittelt dagegen die Ergebnisse jeder Aktion und wählt B.

<!--en:2d03aff7-->
Wir können nun aus einem festen Protokoll eine Strategie lernen, ihre Vorhersage mit ihrem Ertrag vergleichen und eine gut belegte Schätzung von einem Glückstreffer unterscheiden. Bei engem Protokoll verbessert Pessimismus die naive Tabelle. Deckt das Protokoll alles ab, sollten wir auf ihn verzichten. Die Besuchszahlen zeigen, wie gleichmäßig ein Protokoll die Paare abdeckt. Ob es die Paare der optimalen Strategie abdeckt, verraten sie nicht. Auch $\kappa$ ist durch das Protokoll nicht festgelegt. In der Praxis geht man deshalb häufig über das reine Offline-Lernen hinaus. Man lernt zunächst aus dem Protokoll, erhebt dann einige neue Daten und stimmt das Modell damit fein ab. Eine Idee aus diesem Abschnitt begegnet uns in :numref:`sec_rl_sequences` wieder. Dort lernt DPO unter der obigen KL-Nebenbedingung ein Sprachmodell aus einem festen Datensatz von Präferenzen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:acb0f7d6-->
Beim Offline Reinforcement Learning lernt ein Agent eine Strategie aus einem festen Protokoll. Daten einer anderen Strategie sind dafür nur mit Off-Policy-Zielwerten wie beim Q-Learning nutzbar, und auch nur dort, wo das Protokoll die Paare der gelernten Strategie abdeckt. Ohne Interaktion wählt das Maximum im Backup unter den selten erprobten Aktionen gerade diejenigen, deren Schätzwerte zufällig zu hoch sind. Kein neuer Versuch korrigiert sie. Auf dem See mit rutschigem Eis sagt die naive Tabelle bei den meisten Protokollen mehr als das Optimum vorher, und ihre Strategie erzielt weit weniger als vorhergesagt. Pessimismus zieht in jedem Backup einen Unsicherheitsstrafterm $\kappa/\sqrt{n}$ ab, das Spiegelbild des Explorationsbonus. Bei engem Protokoll verbessert er die Strategie. Deckt das Protokoll bereits jede Aktion ab, kostet er Ertrag. Seine Stärke müssen wir ohne die exakte Bewertung wählen, die uns in den Experimenten zur Verfügung stand. Offline-Verfahren mit tiefen Netzen halten die Strategie nahe an den protokollierten Aktionen oder bestrafen unzureichend belegte Werte. Ertragskonditionierte Verfahren umgehen das Maximum. Dafür können sie Teiltrajektorien nur eingeschränkt verknüpfen und vertrauen auch auf Episoden, die nur durch Glück erfolgreich waren.

<!--en:2294243a-->
## Übungen

<!--en:59fd218d-->
1. [code] **Wie viel Pessimismus?** Führen Sie `offline_q` mit $\kappa \in \{0, 0.02, 0.05, 0.1, 0.2, 0.3, 1\}$ erneut aus. Verwenden Sie die Protokolle der Gleichverteilungsstrategie und die engen Protokolle sowie zwei weitere Gruppen von je 40 Protokollen: 500 Episoden des Bedieners mit $\epsilon = 0.2$ und 100 Episoden eines Bedieners mit $\epsilon = 0.1$.
    1. Tragen Sie für jede Art von Protokoll den Median des exakten Werts der gierigen Strategie gegen $\kappa$ auf. Zeichnen Sie den Wert des Klons als waagerechte Linie ein. Wo hilft der Strafterm, wo ist er unschädlich und wo schadet er?
    1. Wie verhält sich die gierige Strategie bei $\kappa = 1$, und wie gut ist ihre Vorhersage kalibriert? Was sagt dieser Fall über die Kalibrierung als Kriterium für die Wahl von $\kappa$ aus?
    1. Wie ändert sich das beste $\kappa$, wenn das Protokoll bei gleichem Bediener von 100 auf 500 Episoden wächst? Setzen Sie Ihre Antwort in Beziehung zur Schranke $\kappa / (\sqrt{n}\,(1 - \gamma))$ für die Summe der Abzüge. Erklären Sie, warum der tatsächliche Abzug kleiner ist.
    1. Welche Informationen haben Sie zur Wahl des besten $\kappa$ herangezogen, die im Offline-Einsatz fehlen?
   :exlabel:`ex_offline-rl-how-much-pessimism`

<!--en:ddbfa5ec-->
1. **Konditionierung auf Glückstreffer.** Ertragskonditioniertes überwachtes Lernen ist die einfachste tabellarische Form von Decision Transformer. Im Zustand $s$ und bei einem Zielertrag $g$ verwendet es die Aktionsverteilung aus den protokollierten Episoden, die $g$ erzielt haben: $\pi(a \mid s, g) = P_{\mathcal{D}}(a \mid s, G = g)$ :cite:`Brandfonbrener.Bietti.Buckman.ea.2022`. Betrachten Sie die Aufgabe mit einem Schritt aus dem Text, wobei Aktion A nun den Betrag $1$ mit Wahrscheinlichkeit $p$ auszahlt.
    1. Für welche Werte von $p$ führt die Konditionierung auf den höchsten protokollierten Ertrag zur optimalen Aktion?
    1. Was lernt Offline-Q-Learning aus denselben Daten, und warum führen die Episoden mit glücklichem Ausgang das Verfahren nicht in die Irre?
    1. [code] Behalten Sie auf dem See mit rutschigem Eis von jedem Protokoll der Gleichverteilungsstrategie nur die Episoden, die das Ziel erreicht haben. Klonen Sie diese mit `clone_policy` und bewerten Sie das Ergebnis exakt. Wie schneidet der Klon im Vergleich zum Klon des gesamten Protokolls und zur naiven Tabelle ab? Welche der beiden im Text genannten Grenzen der Ertragskonditionierung zeigt der Vergleich?

<!--en:b4fba1bc-->
1. [code] **Datensatzgröße und Fehleruntergrenze.** Jeder Eintrag der naiven Tabelle mittelt $n(s, a)$ Zielwerte. Das lässt Fehler erwarten, die wie $1/\sqrt{n}$ abnehmen. Ziehen Sie je fünf Protokolle der Gleichverteilungsstrategie mit $100$, $500$ und $2{,}000$ Episoden. Fassen Sie für die Paare auf nichtterminalen Feldern die Besuchszahlen $n$ und die absoluten Fehler $e = |\hat{Q} - Q^*|$ zusammen. Berechnen Sie den Median von $e\sqrt{n}$ für $n$ in den Intervallen $[1, 10)$, $[10, 30)$, $[30, 100)$ und $[100, \infty)$. Bleibt $e\sqrt{n}$ konstant? Steigt diese Größe mit $n$, nehmen die Fehler langsamer ab als $1/\sqrt{n}$. Erklären Sie das Residuum anhand des Zielwerts eines oft erprobten Paares. Was enthält dieser Zielwert außer den Samples des Paares selbst, und wie nimmt sein Fehler mit wachsendem Protokoll ab?
   :exlabel:`ex_offline-rl-dataset-size`

<!--en:3ef72739-->
1. **Pessimismus bei einem zweiarmigen Banditen.** ● Ein Offline-Protokoll für einen zweiarmigen Banditen enthält $n_1$ Züge an Arm 1, der immer $1/2$ auszahlt, und $n_2$ Züge an Arm 2. Dieser zahlt $1$ mit Wahrscheinlichkeit $1/4$ aus, sonst $0$. Bezeichnen Sie mit $r(a)$ die mittlere Belohnung von Arm $a$, mit $\hat{r}(a)$ seinen empirischen Mittelwert und mit $a^*$ den besten Arm. Bei Gleichstand fällt die Wahl auf Arm 1.
    1. Der empirisch beste Arm maximiert $\hat{r}(a)$. Berechnen Sie die Wahrscheinlichkeit, dass dies Arm 2 ist, und den erwarteten Minderertrag $r(a^*) - r(\hat{a})$ für $n_2 \in \{1, 5, 20\}$. Wie hängt das Ergebnis von $n_1$ ab?
    1. Die Regel mit unterer Konfidenzschranke aus dem Text wählt $\hat{a} = \operatorname{argmax}_a \hat{r}(a) - b(a)$ mit $b(a) = \sqrt{\log(2K/q) / (2 n(a))}$. Nehmen Sie an, dass an jedem Arm mindestens einmal gezogen wurde. Verwenden Sie die Hoeffding-Ungleichung aus Übung :exref:`ex_qlearning-ucb-logarithmic-regret` in :numref:`sec_exploration` und die boolesche Ungleichung. Zeigen Sie damit, dass $r(a^*) - r(\hat{a}) \leq 2 b(a^*)$ mit Wahrscheinlichkeit mindestens $1 - q$ gilt.
    1. Die optimistische Regel $\operatorname{argmax}_a \hat{r}(a) + b(a)$ hat die Form der Regel mit oberer Konfidenzschranke. Leiten Sie für sie die entsprechende Schranke her. Von der Zahl der Züge an welchem Arm hängt die Schranke jetzt ab? Warum ist diese Abhängigkeit online harmlos, offline aber schädlich?
    1. Nun zahle Arm 2 den Betrag $1$ mit Wahrscheinlichkeit $0.7$ aus. Setzen Sie $n_1 = 100$ und $q = 0.1$. Berechnen Sie für $n_2 \in \{5, 10, 17, 30\}$ die exakte Wahrscheinlichkeit, mit der jede der beiden Regeln Arm 2 wählt: die Wahl des empirisch besten Arms und die Regel mit unterer Schranke. Berechnen Sie auch die erwartete Belohnung jeder Regel. Vergleichen Sie mit Teil (a), in dem Arm 2 der schlechtere Arm war. Wovor schützt Pessimismus, und was kostet er?

<!--en:3768d179-->
1. [code] **Bewertung ohne Simulator.** ● In allen Vergleichen dieses Abschnitts haben wir die gelernten Strategien im wahren Modell exakt bewertet. Im tatsächlichen Offline-Einsatz ist das unmöglich. Gewöhnliches Importance-Sampling schätzt den Wert einer Zielstrategie $\pi$ aus $m$ Episoden einer Verhaltensstrategie $\pi_e$ durch $\frac{1}{m} \sum_i w_i G_i$, gewichtetes Importance-Sampling durch $\sum_i w_i G_i / \sum_i w_i$. Dabei ist $G_i$ der diskontierte Ertrag der Episode $i$ und $w_i = \prod_t \pi(a_t \mid s_t) / \pi_e(a_t \mid s_t)$ das Produkt über ihre Schritte.
    1. Betrachten Sie die gierige Strategie der naiven Tabelle aus dem ersten Protokoll der Gleichverteilungsstrategie. Wie viele der 500 Episoden dieses Protokolls erhalten ein von null verschiedenes Gewicht? Erklären Sie die Zahl anhand der Episodenlängen.
    1. Bewerten Sie nun für jedes Protokoll der Gleichverteilungsstrategie die $\epsilon$-gierige Variante der naiven Tabelle mit $\epsilon = 0.3$ anhand ihres eigenen Trainingsprotokolls. Berechnen Sie beide Schätzwerte und die effektive Stichprobengröße $(\sum_i w_i)^2 / \sum_i w_i^2$ der Gewichte. Vergleichen Sie die Schätzwerte mit dem exakten Wert `mdp.policy_value(pi)[0]`, den wir nur im Experiment berechnen können. Für eine feste Zielstrategie ist gewöhnliches Importance-Sampling erwartungstreu. Wie erklärt die Streuung der Gewichte die Fehler der beiden Schätzwerte?
    1. Hätte einer der Schätzer $\kappa$ in Übung :exref:`ex_offline-rl-how-much-pessimism` zuverlässig auswählen können?

    *Nach Richard S. Sutton und Andrew G. Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf), Abschnitt 5.5.*
   :exlabel:`ex_offline-rl-what-you-cannot-measure`

<!--en:ee3f6dde-->
1. **Conservative Q-Learning in einer Tabelle.** ●● Conservative Q-Learning :cite:`Kumar.Zhou.Tucker.ea.2020` ergänzt den quadrierten Bellman-Fehler um einen Term. Er senkt die Werte unter einer gewählten Aktionsverteilung $\mu(a \mid s)$ und hebt sie unter der empirischen Aktionsverteilung $\hat{\pi}_e(a \mid s) = n(s, a) / n(s)$ des Protokolls an. Dabei gilt $n(s) = \sum_a n(s, a)$. In einer Tabelle wählt ein Iterationsschritt
   $$
   \begin{aligned}
   \hat{Q}_{k+1} &= \operatorname*{argmin}_Q\; \alpha_{\textrm{c}} \Big( \mathbb{E}_{s \sim \mathcal{D},\, a \sim \mu(\cdot \mid s)}\big[Q(s, a)\big] - \mathbb{E}_{(s, a) \sim \mathcal{D}}\big[Q(s, a)\big] \Big) \\
   &\quad + \frac{1}{2} \mathbb{E}_{(s, a) \sim \mathcal{D}} \Big[ \big( Q(s, a) - \hat{\mathcal{T}} \hat{Q}_k(s, a) \big)^2 \Big],
   \end{aligned}
   $$
   Dabei mittelt $\mathbb{E}_{\mathcal{D}}$ über die Übergänge des Protokolls, $\alpha_{\textrm{c}} > 0$ bestimmt die Stärke des Strafterms, und $\hat{\mathcal{T}} \hat{Q}_k(s, a)$ mittelt den Q-Learning-Zielwert $r + \gamma \max_{a'} \hat{Q}_k(s', a')$ über die protokollierten Übergänge von $(s, a)$ aus. Nehmen Sie an, dass $\mu(a \mid s) > 0$ nur dort gilt, wo $n(s, a) > 0$ ist. Andernfalls ist die Zielfunktion nach unten unbeschränkt.
    1. Zeigen Sie, dass für jedes Paar mit $n(s, a) > 0$ gilt:
       $$\hat{Q}_{k+1}(s, a) = \hat{\mathcal{T}} \hat{Q}_k(s, a) - \alpha_{\textrm{c}} \Big( \frac{\mu(a \mid s)}{\hat{\pi}_e(a \mid s)} - 1 \Big).$$
       Bei welchen Paaren senkt der Strafterm den Wert, bei welchen hebt er ihn an?
    1. Sei $\mu = \pi$ für eine Strategie $\pi$. Zeigen Sie, dass ein Iterationsschritt den Mittelwert $\sum_a \pi(a \mid s) \hat{Q}_{k+1}(s, a)$ gegenüber dem entsprechenden Mittelwert des Backups um $\alpha_{\textrm{c}} \big( \sum_a \pi(a \mid s)^2 / \hat{\pi}_e(a \mid s) - 1 \big)$ senkt und dass dieser Betrag nichtnegativ ist.
    1. Wählen Sie für $\mu$ die gierige Strategie der naiven Tabelle aus dem ersten engen Protokoll. Tragen Sie für ihre Paare auf nichtterminalen Feldern den CQL-Strafterm aus Teil (a) und den Strafterm $\kappa / \sqrt{n(s, a)}$ aus `offline_q` gegen $n(s, a)$ auf. Auf welche Eigenschaft der Daten reagiert jeder Strafterm, und welche Paare behandeln die beiden unterschiedlich?

<!--en:020c9d61-->
[Diskussion](https://d2l.discourse.group/t/36806)

<!--en:00f52c3d-->
Offline Reinforcement Learning: Verteilungsverschiebung und Pessimismus<br>**Feste Datensätze · Verknüpfen von Teiltrajektorien · Extrapolation außerhalb der Verteilung · untere Konfidenzschranken · CQL und Nebenbedingungen für Strategien**

<!--en:a480379a-->
::: {.slide title="Lernen ohne neue Versuche"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:2b5ccfa1-->
- **Reines Offline-Lernen:** Klinikdaten, Empfehlungssysteme und Protokolle ferngesteuerter Roboter liefern einen festen Datensatz $\mathcal{D}$ **ohne weitere Interaktion mit der Umgebung**.

<!--en:8ebf5f4e-->
- **Unkorrigierte Maximierungsverzerrung:** Nach 100 FrozenLake-Episoden mit enger Abdeckung ($V^*(s_0) = 0.180$) sagt naives Offline-Q-Learning **$0.25$** vorher und erzielt nur **$0.11$**. Kein Rollout widerlegt die zufällig hohen Werte selten erprobter Aktionen.

<!--en:7434d8f9-->
::: {.slide title="Drei Arten der Datennutzung: On-Policy, Off-Policy und offline"}

<!--en:9bcb4542-->
[Welche Daten passen zum Zielwert?]{.kicker}

<!--en:8f4d4475-->
- **On-Policy (REINFORCE, A2C, PPO):** Erfordert Trajektorien der aktuellen Strategie $\pi_\theta$. Die Verhältnisse $\rho_t$ erlauben die Nutzung eines Batchs über wenige Epochen.

<!--en:6c181ba6-->
- **On-Policy-TD ($\delta_{\textrm{SARSA}} = r + \gamma Q(s', a') - Q(s, a)$):** Wegen $a' \sim \pi_e(\cdot \mid s')$ konvergiert SARSA auf einem festen Protokoll gegen $Q^{\pi_e}$. Es bewertet damit die Verhaltensstrategie, ohne sie zu verbessern.

<!--en:f5c0a2be-->
- **Off-Policy-Optimalität ($r + \gamma \max_{a'} Q(s', a')$):** Der Zielwert ist unabhängig von $\pi_e$. Das Verfahren strebt $Q^*$ an, sofern $\mathcal{D}$ die **Abdeckungsbedingung** für alle von der gierigen Strategie besuchten Paare erfüllt.

<!--en:b35c598f-->
::: {.slide title="Zwei feste Protokolle auf dem rutschigen Eis von FrozenLake"}

<!--en:1d93fbaf-->
[Vergleich im Experiment]{.kicker}

<!--en:d925f941-->
Auf dem rutschigen $4 \times 4$-Spielfeld von FrozenLake ($11$ nichtterminale Felder $\times 4$ Aktionen $= 44$ mögliche Paare, exakter optimaler Startwert $V^*(s_0) = 0.180$) entstehen je 40 Datensätze mit zwei sehr unterschiedlichen Verhaltensstrategien:

<!--en:b6bba99b-->
- **Zielkonflikt zwischen Abdeckung und Können:**
  - **Protokoll der Gleichverteilungsstrategie (500 zufällige Episoden):** Die Strategie erzielt einen sehr geringen Ertrag ($0.008$), erprobt aber alle $44/44$ möglichen Paare. Der Median der Besuchszahl beträgt $n(s, a) = 42$.
  - **Enges Protokoll (100 $\epsilon$-gierige Expertenepisoden, $\epsilon = 0.2$):** Die Strategie erzielt einen hohen Ertrag ($0.087$), konzentriert aber $85\%$ ihrer Aktionen auf $\pi^*$. Der Median liegt bei nur $8$ Samples pro Paar.

<!--en:d0c882ba-->
::: {.slide title="Durchläufe des tabellarischen Q-Learnings auf einem festen Protokoll"}

<!--en:dfd24976-->
[Inkrementelles Offline-Q-Learning]{.kicker}

<!--en:97670018-->
Gewöhnliches tabellarisches Q-Learning (`offline_q_sweeps`) auf einem festen Protokoll $\mathcal{D}$: In jeder der 200 Epochen die protokollierten Übergänge mischen und die abnehmende Schrittweite $\alpha_n = 1/(1 + 0.1 n(s, a))$ verwenden:

<!--en:5563fa44-->
::: {.slide title="Sicherheitsäquivalenz: Lösung des empirischen MDP"}

<!--en:634596ab-->
[Der empirische Fixpunkt]{.kicker}

<!--en:8db10f11-->
Da jeder Durchlauf dieselben protokollierten Übergänge nutzt, konvergiert inkrementelles Q-Learning gegen die optimale Wertetabelle des **empirischen MDP** aus dem Protokoll ($\hat{P}(s' \mid s, a) = n(s, a, s')/n(s, a)$, $\hat{r}(s, a) = \sum r / n(s, a)$):

<!--en:a0b2de32-->
::: {.slide title="Sicherheitsäquivalenz: Lösung des empirischen MDP im Vergleich"}

<!--en:634596ab-2-->
[Der empirische Fixpunkt]{.kicker}

<!--en:119ffd2d-->
Bei einem Protokoll mit 500 Episoden (etwa 4000 Übergängen) der Gleichverteilungsstrategie stimmen die inkrementellen Durchläufe und die direkte Lösung gut überein:

<!--en:5c02a558-->
::: {.slide title="Verhaltensklonen und Verknüpfen von Teiltrajektorien"}

<!--en:2d32e5c8-->
[Teiltrajektorien verknüpfen]{.kicker}

<!--en:deba1571-->
- **Verhaltensklonen (`clone_policy`):** Übernimmt die empirischen Häufigkeiten $\hat{\pi}_e(a \mid s) = n(s, a)/n(s)$, ohne die Strategie zu verbessern.

<!--en:5dbdfd65-->
- **Verknüpfen von Teiltrajektorien:** Bellman-Backups übertragen den Wert des Wegs von Feld 10 zum Ziel aus Episode 2 rückwärts entlang des schnellen Anfangsstücks von Episode 1.

<!--en:e7372c1b-->
::: {.slide title="Naives Offline-Q-Learning und Verhaltensklonen im Vergleich"}

<!--en:343fa0a9-->
[Empirischer Vergleich]{.kicker}

<!--en:f89024af-->
Welche **Vorhersage** macht jeder lernende Agent für das Startfeld ($\max_a \hat{Q}(s_0, a)$), und was **erzielt seine Strategie tatsächlich** ($V^\pi(s_0)$, exakt im wahren MDP berechnet)? Der Vergleich umfasst 40 Protokolle der Gleichverteilungsstrategie und 40 enge Protokolle:

<!--en:146425cf-->
- **Teiltrajektorien aus Protokollen der Gleichverteilungsstrategie verknüpfen:** Verhaltensklonen kopiert eine Irrfahrt und erzielt **$0.01$**. Naives Q-Learning verbindet gute Teilstücke verschiedener zufälliger Episoden und erzielt **$0.14$**. Damit übertrifft es den Klon bei $40/40$ Protokollen.

<!--en:c67f6ffc-->
- **Überschätzung bei engen Protokollen:** Naives Q-Learning sagt **$0.25 > V^* = 0.18$** bei $39/40$ engen Protokollen vorher, erzielt aber nur **$0.11$**. Den Ertrag des Klons von $0.09$ übertrifft es bei $27/40$ Protokollen knapp.

<!--en:bd1aa539-->
::: {.slide title="Woher kommen die überhöhten Vorhersagen?"}

<!--en:ef4e9d12-->
[Gewählte Aktionen außerhalb der Verteilung]{.kicker}

<!--en:25b030e0-->
- Der Schätzfehler $\hat{Q}(s, a) - Q^*(s, a)$ als Funktion der Anzahl der Samples $n(s, a)$ je Paar im ersten engen Protokoll zeigt, warum naives Q-Learning versagt:

<!--en:d23c2d9c-->
- Bei selten erprobten Paaren ($n \le 10$) haben die empirischen Übergangswahrscheinlichkeiten eine sehr hohe Varianz. Die Fehler streuen von $-0.15$ bis **$+0.26$**.

<!--en:5074c256-->
- Der gierige Operator $\operatorname{argmax}_a \hat{Q}(s, a)$ wählt systematisch selten erprobte Paare ($n = 2$ oder $n = 5$), deren Fehler zufällig positiv ausfällt.

<!--en:0f2fb0e9-->
::: {.slide title="Aktionswahl der gierigen Strategie über alle 40 engen Protokolle"}

<!--en:17b075e2-->
[Messung der Verteilungsverschiebung]{.kicker}

<!--en:87400ca6-->
Die gewählten Zustand-Aktion-Paare der gierigen Strategie aus allen 40 engen Datensätzen bestätigen die Auswahlwirkung des Operators $\max_a$. Er bevorzugt selten erprobte, überschätzte Aktionen:

<!--en:2d926f7b-->
**Verteilungsverschiebung beim Offline-Lernen:** Ganze **$23\%$ der von der gierigen Strategie gewählten Paare** beruhen auf $1$ bis $5$ protokollierten Übergängen. Der Median der Überschätzung beträgt **$+0.22$** und übersteigt damit den wahren optimalen Wert $V^*(s_0) = 0.180$. Da das Protokoll feststeht, widerlegt kein Online-Rollout diese überhöhten Werte.

<!--en:32ec57e8-->
::: {.slide title="Pessimismus bei Unsicherheit: ein zweiarmiger Bandit"}

<!--en:95a92155-->
[Untere Konfidenzschranken]{.kicker}

<!--en:709237c1-->
- **Naive Wahl des empirischen Maximums:** Ein einziger Glückstreffer bei Arm 2 ($\hat{r}(2) = 1.0$, wahrer Mittelwert $0.25$) führt zur falschen Wahl. Der erwartete Ertrag beträgt $0.4375$.

<!--en:bea038e5-->
- **Untere Konfidenzschranke (LCB):** Zieht den Unsicherheitsradius ab ($b(1) = 0.136$ bzw. $b(2) = 1.358$) und wählt zuverlässig den gut erprobten Arm 1 ($+0.364 > -0.358$).

<!--en:bdd5a639-->
::: {.slide title="Pessimistische Wertiteration auf dem See"}

<!--en:5b80a234-->
[Tabellarischer Pessimismus]{.kicker}

<!--en:18af06c9-->
Bei einem MDP zieht die **pessimistische Wertiteration** (Jin, Yang und Wang, 2021) in jedem Bellman-Backup den Strafterm $\kappa / \sqrt{n(s, a)}$ ab. Er ist umgekehrt proportional zur Wurzel der Besuchszahl. Die bekannte Untergrenze der Umgebung $0$ begrenzt den Wert nach unten:

<!--en:28aef53f-->
- **Spiegelbild der UCB-Exploration (§{sec_exploration}):** Online *addiert* UCB den Term $+\kappa \sqrt{\log t / n(s, a)}$, damit der Agent unsichere Paare besucht und Daten zur Korrektur sammelt. Offline wirkt der Term $-\kappa / \sqrt{n(s, a)}$ als *Abzug*, damit die Strategie unzureichend belegte Paare meidet, für die nie neue Daten hinzukommen.

<!--en:09bec38a-->
::: {.slide title="Wann Pessimismus hilft und wann er schadet"}

<!--en:764d8e2e-->
[Enge und gleichmäßige Abdeckung]{.kicker}

<!--en:810bc883-->
Die pessimistische Wertiteration mit $\kappa = 0.1$ auf allen 40 engen Protokollen und allen 40 Protokollen der Gleichverteilungsstrategie zeigt genau die theoretische Grenze des Offline-Pessimismus (Rashidinejad et al., 2021):

<!--en:c55c2199-->
- **Enge Expertenprotokolle ($\kappa = 0.1$ hilft):** Die typische Überschätzung verschwindet ($0.25 \to 0.13$). Das Verfahren übertrifft naives Q-Learning bei **$29/40$ Protokollen** und unterliegt nur bei $1$. Verhaltensklonen übertrifft es bei **$39/40$ Protokollen**.

<!--en:dd957bca-->
- **Protokolle der Gleichverteilungsstrategie ($\kappa = 0.1$ schadet):** Sind alle Paare gut abgedeckt ($n \approx 42$), gibt es keine zufälligen Ausreißer bei selten erprobten Paaren auszusondern. Unterschiedliche Strafterme je nach Besuchszahl verzerren tatsächliche Wertunterschiede. Bei $28/40$ Protokollen unterliegt das Verfahren dem naiven Q-Learning.

<!--en:ee723c4f-->
::: {.slide title="Zu viel Pessimismus und das Dilemma der Offline-Modellauswahl"}

<!--en:34a4af66-->
[Wahl von Kappa]{.kicker}

<!--en:811385d1-->
Was geschieht bei den engen Protokollen mit 100 Episoden, wenn der Strafparameter auf $\kappa = 0.3$ steigt?

<!--en:58ae8850-->
- **Über den Horizont kumulierter Abzug:** Da der negative Term $-\kappa/\sqrt{n(s,a)}$ bei *jedem* Backup eingeht, summieren sich die Abzüge entlang der Trajektorie auf bis zu $\sim \frac{\kappa}{\sqrt{n}(1 - \gamma)} = 20 \frac{\kappa}{\sqrt{n}}$. Dadurch fallen die Startwerte auf $0.00$ bei $\kappa = 0.3$.

<!--en:508d448f-->
- **Warum sich $\kappa$ offline schwer einstellen lässt:** Ohne Simulator nutzt die Off-Policy-Bewertung Wichtigkeitsgewichte $\prod_t \frac{\pi(a_t \mid s_t)}{\pi_e(a_t \mid s_t)}$ für Trajektorien. Bei einer deterministischen Strategie erhalten die meisten Episoden das Gewicht $0$. Bei den übrigen wächst die Varianz enorm.

<!--en:f6e78aab-->
::: {.slide title="Deep Offline-RL: Nebenbedingungen für Strategien und Strafterme für Werte"}

<!--en:0c75face-->
[Jenseits tabellarischer Besuchszahlen]{.kicker}

<!--en:e31b8e4f-->
- **1. Nebenbedingungen für Strategien (TD3+BC, IQL, DPO):** Halten $\pi(\cdot \mid s)$ nahe an der Verhaltensstrategie $\pi_e(\cdot \mid s)$ durch KL- oder $L_2$-Strafterme für die Aktionen.

<!--en:820058ca-->
- **2. Conservative Q-Learning (CQL, Kumar et al., 2020):** Addiert $\alpha_{\textrm{c}}\mathbb{E}_s[\log\sum_{a'} e^{Q_w(s,a')} - Q_w(s, a_{\textrm{log}})]$, um alle Aktionswerte zu senken und Spitzen außerhalb der Verteilung zu unterdrücken. Zugleich hebt es die Werte protokollierter Aktionen proportional zu $\hat{\pi}_e(a \mid s)$ an.

<!--en:a5736104-->
::: {.slide title="Sequenzmodellierung unter einer Ertragsvorgabe und ihre Grenzen"}

<!--en:7c1baf25-->
[Konditionierung auf den Ertrag]{.kicker}

<!--en:0ff26c1e-->
**Decision Transformer** (Chen et al., 2021) kommt ohne Bellman-Backups und $\max_{a'}$ aus. Er trainiert ein autoregressives Sequenzmodell auf $(g_0, s_0, a_0, g_1, s_1, a_1, \dots)$. Dieses sagt $a_t$ für einen gewünschten verbleibenden Ertrag $g_t$ vorher:

<!--en:23915724-->
- **Vorteil (keine tödliche Triade):** Rein überwachtes Lernen mit der Kreuzentropie oder dem mittleren quadratischen Fehler als Verlust auf protokollierten Sequenzen. Kein Bootstrapping und keine divergierenden Werte außerhalb der Verteilung.

<!--en:1be4844d-->
- **Grenze 1 (eingeschränktes Verknüpfen):** Bei Vorgabe eines hohen Ertrags $g$ imitiert das Modell ganze Trajektorien, die $g$ erzielt haben. Teiltrajektorien aus verschiedenen erfolglosen Episoden verknüpft es dabei nicht.

<!--en:b1b9095f-->
- **Grenze 2 (zufällige Glückstreffer):** Bei einem Banditen mit einem Schritt zahlt **Aktion A** den Betrag $1$ mit Wahrscheinlichkeit $0.5$ aus (sonst $0$, Mittelwert $0.5$). **Aktion B** zahlt immer $0.6$ aus.
  - Jede protokollierte Episode mit Ertrag $g = 1.0$ hat Aktion A gewählt.
  - Wer auf den maximalen Ertrag $g = 1.0$ konditioniert, wählt daher Aktion A ($0.50$) anstelle der optimalen Aktion B ($0.60$).

<!--en:fc9fefcc-->
::: {.slide title="Verständnisfragen: Teiltrajektorien, LCB-Schwellen und CQL"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:9e55f275-->
Grundlagen von Offline Reinforcement Learning überprüfen:

<!--en:b32c4788-->
1. **Kalibrierung und Ertrag:** Warum ist Verhaltensklonen bei den FrozenLake-Protokollen der Gleichverteilungsstrategie so gut kalibriert ($\text{pred } 0.01 \approx \text{earned } 0.01$), obwohl naives Offline-Q-Learning mehr als das Zehnfache erzielt ($\text{earned } 0.14$)?

<!--en:a4167b05-->
2. **Mindestzahl an Zügen für LCB:** Gegeben ist der zweiarmige Bandit ($b(a) = \sqrt{\log 40 / (2n(a))}$, untere Konfidenzschranke für Arm 1 $= 0.364$). Arm 2 hat den empirischen Mittelwert $\hat{r}(2) = 0.70$. Warum verwirft LCB Arm 2 bei $n(2) = 10$ Zügen und wählt ihn bei $n(2) = 17$ Zügen?

<!--en:f22563cb-->
3. **Fixpunkt beim tabellarischen CQL:** Ein Schritt des tabellarischen CQL setzt $\hat{Q}(s, a) = \hat{\mathcal{T}}\hat{Q}(s, a) - \alpha_{\textrm{c}}\big(\frac{\mu(a \mid s)}{\hat{\pi}_e(a \mid s)} - 1\big)$. Für welche Zustand-Aktion-Paare sinkt der Wert, für welche steigt er?

<!--en:557b98b6-->
**Lösungen:**

<!--en:4b6eb8ff-->
1. Verhaltensklonen schätzt den mittleren Ertrag der zufälligen Verhaltensstrategie ohne Verzerrung durch $\max_a$. Offline-Q-Learning **verknüpft** eine zufällige Episode, die ein Zwischenfeld erreicht hat, mit einer *anderen* Episode, die von dort zum Ziel gelangt ist.

<!--en:0c6e0ed9-->
2. Für $n(2) = 10$ gilt $0.70 - \sqrt{\log 40 / 20} = 0.70 - 0.430 = \mathbf{0.270 < 0.364}$. Für $n(2) = 17$ gilt $0.70 - \sqrt{\log 40 / 34} = 0.70 - 0.329 = \mathbf{0.371 > 0.364}$.

<!--en:b38d171d-->
3. Die Werte von Aktionen, die $\mu$ *häufiger* wählt als die protokollierte Verhaltensstrategie ($\mu(a \mid s) > \hat{\pi}_e(a \mid s)$), **sinken**. Die Werte der im Protokoll häufigeren Aktionen ($\mu(a \mid s) < \hat{\pi}_e(a \mid s)$) **steigen**.

<!--en:6789a403-->
::: {.slide title="Zusammenfassung: Verfahren von Offline Reinforcement Learning"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:1d065aef-->
| Verfahren | Optimierungsziel | Verknüpfen von Teiltrajektorien | Schutz vor Aktionen außerhalb der Verteilung | Verhalten bei enger und gleichmäßiger Abdeckung |

<!--en:5d1edee0-->
| **Verhaltensklonen (BC)** | Überwachtes Lernen mit $\max_\pi \mathbb{E}_{\mathcal{D}}[\log \pi(a \mid s)]$ | **Keines** | Vollständig (bleibt bei $\hat{\pi}_e$) | Gut bei engen Expertenprotokollen ($0.09$), schwach bei gleichmäßiger Abdeckung ($0.01$) |

<!--en:4944aef5-->
| **Offline-SARSA** | On-Policy-TD mit $r + \gamma Q(s', a'_{\textrm{log}})$ | **Keines** | Bewertet nur $Q^{\pi_e}$ | Konvergiert ohne Verbesserung gegen den Wert der Verhaltensstrategie |

<!--en:da9fc920-->
| **Naives Offline-Q-Learning** | Empirisches Bellman-Maximum $\max_{a'} \hat{Q}(s', a')$ | **Gut** | **Keiner** (wählt selten erprobte Paare mit Glückstreffern) | Gut bei gleichmäßiger Abdeckung ($0.14$), Überschätzung bei enger Abdeckung ($0.25$ gegenüber $0.11$) |

<!--en:626dd8a7-->
| **Pessimistisches Q-Learning / CQL** | Strafterm im Backup mit $-\kappa/\sqrt{n}$ oder Log-Sum-Exp | **Gut** | **Stark** (unterdrückt selten erprobte Paare) | Überlegen bei engen Protokollen ($0.14$, übertrifft BC bei $39/40$), schadet bei zu starkem Abzug |

<!--en:8cf430c7-->
| **Konditionierung auf den Ertrag (DT)** | Autoregressives Modell $\pi(a_t \mid s_{\le t}, g_{\le t})$ | **Eingeschränkt** | Vermeidet den Operator $\max_{a'}$ | Anfällig für Verzerrung durch zufällig erfolgreiche Trajektorien |
