<!--en:d448a3cd-->
# Exploration

<!--en:0fc3f14f-->
Im Lieblingsrestaurant wissen wir, dass das Essen gut ist. In einem neuen Restaurant könnte es besser oder schlechter sein. Wer immer ins Lieblingsrestaurant geht, erfährt es nie. Dieselbe Frage stellt sich, wenn wir wiederholt zwischen Möglichkeiten wählen, deren Wert wir erst durch Ausprobieren kennenlernen. Wie oft sollten wir etwas Neues versuchen, und wann können wir damit aufhören?

<!--en:24e1ffeb-->
William Thompson formulierte das Problem 1933 für die Medizin. Welche von zwei Behandlungen soll der nächste Patient erhalten, wenn wir wissen, wie es den bisherigen Patienten ergangen ist :cite:`Thompson.1933`? Herbert Robbins formulierte es 1952 als Problem wiederholter Ziehungen aus zwei Grundgesamtheiten. Vor jeder Ziehung ist zu entscheiden, aus welcher der beiden sie stammen soll :cite:`Robbins.1952`. In Anlehnung an den Spielautomaten, den einarmigen Banditen, hieß das Problem bald *zweiarmiger Bandit*. Bei mehr als zwei Möglichkeiten sprechen wir vom *mehrarmigen Banditen* (engl. multi-armed bandit). 1985 zeigten Tze Leung Lai und Robbins, dass die Kosten des Lernens mit der Anzahl der Versuche wachsen müssen, dass aber logarithmisches Wachstum genügt :cite:`Lai.Robbins.1985`.

<!--en:3150b4cd-->
Beim Q-Learning in :numref:`sec_qlearning` haben wir dieselbe Frage nur grob beantwortet. Der Agent lernte nur die Tabelleneinträge für Aktionen, die er ausprobierte. Damit er weiterhin auch Befehle ausprobierte, die seine Tabelle schlecht bewertete, handelte er mit Wahrscheinlichkeit $\epsilon$ zufällig. Diese Regel hat sichtbare Kosten. Muss die optimale Strategie (engl. optimal policy) auf dem rutschigen See in einem von zwanzig Schritten zufällig handeln, erreicht sie das Ziel in etwa 54% der Fälle statt in 73%. Ist das das richtige Maß an Exploration? Könnte eine andere Regel genauso viel lernen und dabei weniger verlieren?

<!--en:44da0725-->
Dieselbe Frage stellt sich beim Training von Sprachmodellen mit einem Prüfprogramm (engl. verifier). Das Modell zieht mehrere Antworten auf einen Prompt, ein Prüfprogramm bewertet sie, und das Modell lernt, die besser bewerteten Antworten häufiger zu geben. Erhalten alle Antworten denselben Score, fehlt die Richtung für die Anpassung. Nur durch den Zufall bei der eigenen Antwortwahl kann das Modell dann eine bessere Antwort finden.

<!--en:46c260f0-->
Wir untersuchen diesen Zielkonflikt zunächst an einem Problem mit nur einem Zustand, dem Banditen. Hier lassen sich die Kosten genau messen. Dann kehren wir zu Markov-Entscheidungsprozessen (engl. Markov decision processes, MDPs) zurück. Dort kann eine lange Sequenz von Aktionen nötig sein, um ein neues Zustand-Aktion-Paar auszuprobieren. Den Abschluss bildet die Aufgabe mit Prüfprogramm aus :numref:`sec_mdp`.

<!--en:ea434966-->
## Der mehrarmige Bandit

<!--en:9892452c-->
Ein Bandit ist ein MDP mit einem einzigen Zustand. In jeder Runde zieht der Agent an einem von mehreren Armen und erhält eine zufällige Belohnung (engl. reward). Seine Wahl hat keinen Einfluss auf die nächste Runde, und er erhält jede Belohnung sofort. Es bleibt also nur eine Frage: Welcher Arm ist der beste? Das kann der Agent nur durch Ziehen herausfinden. Jeder Zug an einem schlechteren Arm ist ein Zug, der ihm am besten Arm entgeht.

<!--en:1c89abee-->
### Arme und Regret

<!--en:a18a1bb2-->
Wir verwenden zehn Arme mit Bernoulli-verteilten Belohnungen. Arm $a$ liefert $1$ mit Wahrscheinlichkeit $\mu_a$ und sonst $0$. Für den besten Arm, den dritten, gilt $\mu^* = \max_a \mu_a = 0.90$, für den zweitbesten $0.60$. Der Agent kennt die $\mu_a$ nicht. Nach $n(a)$ Zügen an Arm $a$ kennt er den empirischen Mittelwert $\hat\mu(a)$, also den Anteil dieser Züge, die $1$ ergaben. Eine *Banditenregel* legt anhand der Zählerstände den nächsten Arm fest. Die Funktion `bandit_run` spielt nach einer solchen Regel `num_pulls` Runden lang. Alle Zufallszahlen stammen aus dem Generator `rng`. Für jeden Arm führt die Funktion die Zähler `wins` und `count` und übergibt sie an die Regel:

<!--en:33d84b8f-->
Die Funktion erfasst bei jedem Zug anstelle der Belohnung, wie viel dem Agenten gegenüber dem besten Arm entgeht. Die Erfolgsquote gibt an, wie gut eine Strategie ist. Bei einem lernenden Agenten interessiert uns, wie viel ihn das Lernen gekostet hat. Ein Zug an Arm $a_u$ in Runde $u$ bringt im Erwartungswert $\mu^* - \mu_{a_u}$ weniger als der beste Arm. Der *Regret* nach $t$ Zügen ist die Summe dieser Differenzen:

<!--en:d67387d9-->
Die zweite Form gruppiert die Züge nach Arm. Für jeden Arm multiplizieren wir den Abstand $\Delta_a$ zum besten Arm mit der Anzahl $n_t(a)$ seiner Züge. Der Regret gibt an, wie viel der Agent gegenüber einem Agenten verliert, der den besten Arm von Anfang an kannte. Zieht eine Regel immer an einem Arm mit Mittelwert $0.50$, verliert sie $0.40$ pro Zug und damit $800$ nach $2000$ Zügen. Ihr Regret wächst linear mit $t$. Wächst der Regret einer Regel langsamer als $t$, zieht sie immer seltener an schlechteren Armen. Ihr mittlerer Regret pro Zug geht dann gegen null.

<!--en:534e3c3d-->
### Gierige und epsilon-gierige Regeln

<!--en:a4b29c7a-->
Die Regel aus :numref:`sec_qlearning` lässt sich unverändert anwenden. `d2l.epsilon_greedy(q, epsilon, rng)` erhält eine Zeile mit geschätzten Werten, hier die zehn empirischen Mittelwerte. Mit Wahrscheinlichkeit $\epsilon$ wählt die Regel einen Arm gleichverteilt zufällig. Sonst wählt sie den Arm mit dem größten Schätzwert. Bei Gleichstand entscheidet der Zufall. Ein noch unversuchter Arm hat den empirischen Mittelwert $0$. Für $\epsilon = 0$ heißt die Regel *gierig* (engl. greedy). Bei festem $\epsilon$ bleibt die Explorationsrate für immer gleich. Eine Regel *mit abklingender Explorationsrate* (engl. annealed) senkt dagegen $\epsilon$ im Laufe der Zeit. Wir verwenden $\epsilon_t = \min(1, 20/t)$ in Runde $t$. Damit zieht die Regel in den ersten zwanzig Runden zufällig und danach immer seltener. Der Verlauf hängt allein von der Rundenzahl ab, unabhängig davon, was der Agent gelernt hat.

<!--en:d8c8135c-->
Jede dieser Regeln macht aus einer Zeile geschätzter Werte eine Wahrscheinlichkeitsverteilung über die Aktionen. Das zeigt :numref:`fig_rl_exploration` für vier Aktionswerte, einen je Bewegungsrichtung auf dem See. Die gierige Regel weist die gesamte Wahrscheinlichkeit dem größten Schätzwert zu. Die $\epsilon$-gierige Regel verteilt einen Anteil $\epsilon$ gleichmäßig auf alle Aktionen. Bei einer dritten Regel, der *Boltzmann-Exploration*, ist die Wahrscheinlichkeit jeder Aktion proportional zu $\exp(\hat Q(s, a)/\beta)$. Das ist die Softmax-Funktion aus :numref:`sec_softmax` mit Temperatur $\beta$. Bei großem $\beta$ nähert sich die Verteilung der Gleichverteilung, bei kleiner Temperatur der gierigen Regel. Anders als die $\epsilon$-gierige Regel probiert die Boltzmann-Exploration eine fast optimale Aktion häufiger aus als eine eindeutig schlechte.

<!--en:5d84e9ee-->
![Drei Explorationsregeln für dieselbe Zeile von Aktionswerten, $\hat Q(s, \cdot) = (0.20, 0.90, 0.55, 0.10)$ für die vier Bewegungsrichtungen auf dem See. Die gierige Regel weist dem größten Wert die gesamte Wahrscheinlichkeit zu. Die $\epsilon$-gierige Regel mit $\epsilon = 0.3$ gibt jeder Aktion die Wahrscheinlichkeit $0.075$ und den Rest der Aktion mit dem größten Wert. Die Softmax-Funktion mit Temperatur $0.3$ ordnet die Aktionen nach ihren Werten.](../img/mdl-rl-exploration.svg)

<!--en:19abab2e-->
Die Funktion `eps_rule` macht aus einem Verlauf $t \mapsto \epsilon_t$ eine Banditenregel, die `d2l.epsilon_greedy` auf die empirischen Mittelwerte anwendet. Die Funktion `regret_runs` spielt nach einer Regel zwanzig unabhängige Durchläufe und gibt deren Regret-Kurven zurück. Jeder Durchlauf initialisiert seinen eigenen Zufallszahlengenerator anhand der Nummer des Zufallsstroms und des Durchlaufindex. So erhalten verschiedene Regeln unabhängige Durchläufe. Die Funktion `report` gibt den Median und die Quartile des abschließenden Regrets über die Durchläufe aus. Wir geben Mediane an, weil schon wenige Durchläufe den Mittelwert dominieren können:

<!--en:7645e53f-->
Die gierige Regel schneidet am schlechtesten ab. Der Grund lässt sich genau berechnen. Alle empirischen Mittelwerte beginnen bei $0$, also entscheidet beim ersten Zug der Zufall. Ergibt er $0$, bleibt der Mittelwert des Arms bei $0$, und alle zehn Arme liegen weiter gleichauf. Ergibt der Zug $1$, wird der Mittelwert positiv und bleibt es. Ein einziger Erfolg hält den Gewinnanteil über null, während alle anderen Arme weiter den Wert $0$ haben. Die gierige Regel zieht also so lange an zufälligen Armen, bis einer eine Belohnung liefert. Danach zieht sie nur noch an diesem Arm. Bei jedem Zufallszug fällt die Wahl auf Arm $a$ mit Wahrscheinlichkeit $1/10$, und dieser liefert $1$ mit Wahrscheinlichkeit $\mu_a$. Die Regel legt sich also auf Arm $a$ mit Wahrscheinlichkeit $\mu_a / \sum_b \mu_b$ fest. Der beste Arm gewinnt diese Lotterie mit Wahrscheinlichkeit $0.90/4.87$, also in etwa jedem fünften Durchlauf. Legt sich ein Durchlauf auf Arm $a$ fest, verliert er bei fast jedem Zug $\Delta_a$ und insgesamt etwa $2000\,\Delta_a$. Der erwartete Regret beträgt somit

<!--en:e0386834-->
Der Verlust eines einzelnen Durchlaufs nimmt einen der Werte $2000\,\Delta_a$ an, von $0$ (etwa jeder fünfte Durchlauf) bis $1300$. Der Median liegt bei etwa $800$ und entspricht den Durchläufen, die sich auf einen Arm mit Mittelwert $0.50$ festlegen.

<!--en:5f9e3b6d-->
Die Regel mit festem $\epsilon = 0.1$ findet in fast jedem Durchlauf den besten Arm, hört aber nie auf zu explorieren. Bei einem Zufallszug entgeht ihr gegenüber dem besten Arm im Mittel $0.90 - 0.487 = 0.41$. Dabei ist $0.487$ der Mittelwert der zehn $\mu_a$. Allein die Exploration kostet also $0.1 \times 0.41 \approx 0.04$ pro Zug und erklärt etwa $83$ des medianen Regrets von $106$. Diese Kosten pro Zug sinken nie. Bei festem $\epsilon$ wächst der Regret linear mit $t$.

<!--en:bd4bcfd1-->
Die Regel mit abklingender Explorationsrate macht etwa $112$ Zufallszüge in $2000$ Runden, zwanzig zu Beginn und etwa $20 \ln(2000/20) \approx 92$ danach. Bei $0.41$ je Zug kosten sie insgesamt $46$. Das entspricht genau dem medianen Regret von $46$. Fast der gesamte Regret entfällt auf die Exploration. Die gierigen Züge dazwischen gehen an den besten Arm. Die Anzahl der Zufallszüge bis Runde $t$ wächst wie $20 \ln t$. Also wächst auch der Regret so: Ein $\epsilon_t$ proportional zu $1/t$ ergibt logarithmischen Regret. Dabei kommt es auf die Konstante an. Ist sie zu klein, endet die Exploration, bevor die Regel den besten Arm vom zweitbesten unterscheiden kann. Die Analyse von :citet:`Auer.CesaBianchi.Fischer.2002` garantiert logarithmischen Regret für jeden Banditen mit $m$ Armen und kleinstem Abstand $\Delta_{\min}$, sofern $\epsilon_t = \min(1, c\,m/(\Delta_{\min}^2 t))$ für ein beliebiges $c > 5$ gilt. Für unsere Arme mit $\Delta_{\min} = 0.30$ ist das fast das Dreißigfache unseres Werts $20/t$. Unsere speziell für diese Arme gewählte Konstante fällt nicht unter die Garantie.

<!--en:ab4824c9-->
## Optimismus und Ziehen aus der A-posteriori-Verteilung

<!--en:c7b5c804-->
Die $\epsilon$-gierigen Regeln explorieren blind. Ein Zufallszug trifft den schlechtesten Arm ebenso wahrscheinlich wie den zweitbesten. Ein Arm, an dem wir schon hundertmal gezogen haben, kommt ebenso wahrscheinlich wieder zum Zug wie einer, den wir erst zweimal ausprobiert haben. Eine bessere Regel richtet die Exploration auf Arme, deren Mittelwerte noch unsicher sind und die deshalb die besten sein könnten. Denn nur solche Züge können die Entscheidung ändern.

<!--en:d18f73a0-->
### Obere Konfidenzschranken

<!--en:cd9d65c3-->
Eine Möglichkeit dazu bietet der Optimismus. Wir behandeln jeden Arm so, als wäre er so gut, wie die Daten noch zulassen. Über einen unversuchten Arm wissen wir nichts, also probieren wir ihn aus. Bei einem oft gezogenen Arm ist die Schätzung präzise. Wir ziehen nur wieder an ihm, wenn sein Schätzwert hoch ist. Die Regel der *oberen Konfidenzschranke* (engl. upper confidence bound, UCB) addiert zu jedem empirischen Mittelwert einen Bonus als Maß für seine Unsicherheit. Sie zieht an dem Arm mit der größten Summe :cite:`Auer.CesaBianchi.Fischer.2002`:

<!--en:fa4ee9b1-->
Der Bonus sinkt wie $1/\sqrt{n(a)}$, also ebenso schnell wie der Fehler des Mittelwerts von $n(a)$ Belohnungen. Er wächst langsam mit $t$, solange der Agent nicht an Arm $a$ zieht. So gibt er keinen Arm endgültig auf. Er zieht an einem Arm, weil dessen Mittelwert hoch oder noch unsicher ist. Vor Anwendung der Regel ziehen wir einmal an jedem Arm, denn bei $n(a) = 0$ ist der Bonus undefiniert. :numref:`fig_rl_ucb` zeigt den Bonus und die Indizes für einen Durchlauf.

<!--en:73edbb30-->
![Optimismus bei Unsicherheit mit $\kappa = 0.5$. (a) Der Bonus $\kappa\sqrt{\log t/n(a)}$ sinkt mit der Anzahl $n(a)$ der Züge am Arm und wächst langsam mit der Zeit $t$. (b) Ein Durchlauf der UCB-Regel für die zehn Arme nach 2000 Zügen. Jeder Balken zeigt den Schätzwert $\hat\mu(a)$ eines Arms mit aufgesetztem Bonus. Die Beschriftung gibt die Anzahl der Züge an, die schwarzen Marken die wahren Mittelwerte. Fast alle Züge entfielen auf den besten Arm. Er hat den größten Index. Die Indizes der selten gewählten Arme liegen knapp darunter, sodass mit wachsendem $\log t$ jeder wieder zum Zug kommt.](../img/mdl-rl-ucb.svg)

<!--en:355b5341-->
Die Höhe des Bonus folgt aus der Hoeffding-Ungleichung :cite:`Hoeffding.1963`. Dass der Mittelwert $\hat\mu_n$ von $n$ unabhängigen Belohnungen in $[0, 1]$ den wahren Mittelwert $\mu$ um mehr als $h$ übersteigt, hat Wahrscheinlichkeit höchstens $e^{-2 n h^2}$. Dasselbe gilt für eine ebenso große Abweichung nach unten. Mit $h = \kappa \sqrt{\log t / n}$ liegt der wahre Mittelwert mit Wahrscheinlichkeit höchstens $t^{-2\kappa^2}$ über dem Index, wenn die Anzahl der Züge $n$ feststeht. Ein größeres $\kappa$ macht solche Fehlschätzungen seltener, kostet aber mehr Züge an schlechten Armen. Mit $\kappa = \sqrt{2}$ erhalten wir die Regel UCB1 von :citet:`Auer.CesaBianchi.Fischer.2002`. Ihr Regret wächst logarithmisch mit $t$ für jeden Banditen mit Belohnungen in $[0, 1]$. Dasselbe gilt für jedes $\kappa > 1$ :cite:`Bubeck.CesaBianchi.2012`. In :exref:`ex_qlearning-ucb-logarithmic-regret` leiten Sie die Schranke her.

<!--en:23ea66f3-->
### Thompson-Sampling

<!--en:81ec2e95-->
Thompsons Regel von 1933 ersetzt den Bonus durch eine Wahrscheinlichkeitsverteilung. Für jeden Arm führt sie eine A-posteriori-Verteilung (engl. posterior distribution) über seinen Mittelwert. Sie zieht aus jeder Verteilung einen Wert und wählt den Arm mit dem größten gezogenen Wert. Für Belohnungen in $\{0, 1\}$ und eine gleichförmige A-priori-Verteilung (engl. prior distribution) auf $[0, 1]$ ergibt sich nach $w$ Gewinnen in $n$ Zügen als A-posteriori-Verteilung die Beta-Verteilung $\mathrm{Beta}(1 + w, 1 + n - w)$. Bei $3$ Gewinnen in $4$ Zügen hat der Arm die A-posteriori-Verteilung $\mathrm{Beta}(4, 2)$ mit Mittelwert $2/3$, aber großer Streuung. Für einen unversuchten Arm gilt die Gleichverteilung $\mathrm{Beta}(1, 1)$. Ein daraus gezogener Wert übertrifft einen Wert aus $\mathrm{Beta}(4, 2)$ mit Wahrscheinlichkeit $1/3$. :numref:`fig_rl_thompson` zeigt eine solche Runde. Hier übertrifft der unversuchte Arm sogar einen Arm mit $18$ Gewinnen in $20$ Zügen. Mit zunehmender Zahl der Züge werden die A-posteriori-Verteilungen schmaler. Die Wahrscheinlichkeit, an einem Arm zu ziehen, entspricht jeweils seiner A-posteriori-Wahrscheinlichkeit, der beste zu sein. Auch beim Thompson-Sampling wächst der Regret logarithmisch :cite:`Agrawal.Goyal.2012`, und wir müssen keine Konstante wählen.

<!--en:2d6e704e-->
![Eine Runde Thompson-Sampling mit drei Armen: einem unversuchten Arm, dem Arm mit $3$ Gewinnen in $4$ Zügen und einem Arm mit $18$ Gewinnen in $20$ Zügen. Die Kurven zeigen die Beta-Verteilungen der Mittelwerte nach den bisherigen Zügen. Je öfter ein Arm zum Zug kam, desto schmaler die Kurve. Die Regel zieht aus jeder A-posteriori-Verteilung einen Wert (Dreiecke) und wählt den Arm mit dem größten gezogenen Wert. In dieser Runde ist das der unversuchte Arm.](../img/mdl-rl-thompson.svg)

<!--en:2b5ebcf5-->
### UCB und Thompson-Sampling im Vergleich

<!--en:5a0fa5ad-->
Die Funktion `ucb(kappa)` gibt die UCB-Regel mit Koeffizient $\kappa$ zurück, und `thompson` zieht Werte aus den Beta-Verteilungen der Mittelwerte. Wir führen UCB mit $\kappa = 0.5$ und das Thompson-Sampling aus. Diesen Wert haben wir für diese Arme gewählt und untersuchen ihn weiter unten. Anschließend plotten wir alle fünf Regeln. `d2l.plot_curves` zeichnet den Median über die Durchläufe und das Band zwischen den Quartilen, hier an 100 Zugzahlen in gleichmäßigen Abständen auf logarithmischen Achsen:

<!--en:aedd031f-->
Beide Regeln enden mit einem medianen Regret von etwa $30$ bis $40$. Das ist etwa ein Drittel des Regrets bei festem $\epsilon$ und weniger als bei abklingender Explorationsrate. Die beiden Regeln liegen nah beieinander. Ihr Unterschied ist kleiner als die Streuung über die Durchläufe. Auf doppelt logarithmischen Achsen ergibt linear wachsender Regret eine Gerade mit Steigung eins. Die gierige Regel und die Regel mit festem $\epsilon$ nähern sich dieser Steigung. Die Kurven für UCB, Thompson-Sampling und die Regel mit abklingender Explorationsrate flachen dagegen ab. Mit $2000$ Zügen lässt sich keine asymptotische Rate nachweisen, aber genau diese Krümmung erwarten wir bei logarithmischem Wachstum.

<!--en:f974cbf7-->
### Der UCB-Koeffizient

<!--en:a3216fb1-->
Der Wert $\kappa = 0.5$ liegt unterhalb des Bereichs $\kappa > 1$, den die Theorie abdeckt. Wie viel hängt vom Koeffizienten ab? Die Codezelle spielt für jeden von drei Koeffizienten vierzig Durchläufe und gibt den Median und die Extremwerte des Regrets aus:

<!--en:3a813b45-->
Nach $2000$ Zügen entscheidet der Koeffizient über das Ergebnis. UCB1 mit $\kappa = \sqrt{2}$ endet bei fast $200$, dem Fünf- bis Sechsfachen des Regrets bei $\kappa = 0.5$. Mit $\kappa = 0.15$ ist der Median mit etwa zehn am kleinsten, das Maximum jedoch groß. Bei einem so kleinen Bonus genügen ein paar unglückliche Züge am besten Arm zu Beginn. Die Regel kann sich dann auf einen schlechteren Arm festlegen und fortan pro Zug $0.3$ oder mehr verlieren. Das geschieht in etwa jedem zehnten Durchlauf. Im Median fallen diese Durchläufe nicht auf, im Mittelwert schon.

<!--en:be9bba45-->
Man könnte erwarten, dass sich das theoretisch begründete $\kappa$ bei genügend langem Horizont durchsetzt. Bei diesen Armen geschieht das nicht. In Durchläufen mit $10^5$ und mehr Zügen, die Sie in :exref:`ex_exploration-long-horizons` nachvollziehen, verursacht $\kappa = \sqrt{2}$ weiterhin sechs- bis achtmal so viel Regret wie $\kappa = 0.5$. Beide wachsen wie $\log t$. Der Faktor erklärt sich aus dem Bonus. UCB zieht so lange an einem schlechteren Arm $a$, bis sein Bonus $\kappa\sqrt{\log t / n(a)}$ auf etwa $\Delta_a$ gesunken ist, also etwa $\kappa^2 \log t / \Delta_a^2$ Züge lang. Bei langen Durchläufen nähert sich das Verhältnis der beiden Regrets daher $(\sqrt{2}/0.5)^2 = 8$. Die Theorie garantiert etwas anderes: Für $\kappa = \sqrt{2}$ wächst der Regret bei jedem Banditen mit Belohnungen in $[0, 1]$ logarithmisch. Für $\kappa = 0.5$ gibt es keine solche Garantie. Die Hoeffding-Ungleichung muss auch im ungünstigsten Fall gelten, für einen Arm mit Varianz $1/4$. Unser bester Arm hat die Varianz $0.9 \times 0.1 = 0.09$. Sein empirischer Mittelwert ist also genauer, als die Ungleichung voraussetzt, und für dieses Problem genügt ein kleinerer Bonus. Thompson-Sampling erreicht das Ergebnis der abgestimmten UCB-Regel, ohne dass wir eine Konstante abstimmen müssen.

<!--en:9c8a0c7f-->
### Eine untere Schranke für den Regret

<!--en:869a08a0-->
Wie klein kann der Regret werden? Lai und Robbins bewiesen, dass für jede Regel, die bei jedem Banditen einer Familie gut abschneidet, Folgendes gilt: Unter $t$ Zügen muss sie an jedem schlechteren Arm $a$ mindestens etwa $\ln t / D(\mu_a \,\|\, \mu^*)$ Mal ziehen :cite:`Lai.Robbins.1985`. Dabei ist

<!--en:51e34157-->
die relative Entropie zwischen den Bernoulli-Verteilungen mit Mittelwerten $p$ und $q$. Liegen die beiden Mittelwerte nahe beieinander, ist sie klein. Ein fast optimaler Arm lässt sich also nur schwer vom besten unterscheiden und muss häufiger zum Zug kommen. Für den Arm mit Mittelwert $0.60$ gilt $D(0.60 \,\|\, 0.90) = 0.31$. An ihm muss die Regel ungefähr $\ln 2000 / 0.31 \approx 24$ Mal bis Runde $2000$ ziehen. Mit :eqref:`eq_regret` wächst der Regret jeder solchen Regel mindestens wie $\ln t$. UCB und Thompson-Sampling erreichen diese Rate. In langen Durchläufen kann also keine Regel um mehr als einen konstanten Faktor besser abschneiden. Eine Regel mit festem $\epsilon$ erreicht diese Rate wegen ihres linear wachsenden Regrets nicht.

<!--en:f0d81958-->
## Exploration in MDPs

<!--en:a0a27bdb-->
Bei einem Banditen können wir in jeder Runde an jedem Arm ziehen. In einem MDP können wir ein Zustand-Aktion-Paar nur in seinem Zustand ausprobieren. Um dorthin zu gelangen, brauchen wir unter Umständen eine bestimmte Sequenz von Aktionen. Bei zufälliger Wahl fällt die Wahrscheinlichkeit einer gegebenen Sequenz geometrisch mit ihrer Länge. Stehen zwei Aktionen zur Wahl, tritt eine bestimmte Sequenz von $k$ Aktionen mit Wahrscheinlichkeit $2^{-k}$ auf.

<!--en:149c9ac2-->
### Eine Kette von Zuständen

<!--en:e0b943ec-->
An einer Kette lässt sich das genau berechnen. Sie hat $N$ Zustände in einer Reihe und beginnt in Zustand $0$. Aktion $1$ führt einen Zustand weiter, Aktion $0$ zurück zum Start. Beim Eintritt in den letzten Zustand erhält der Agent die Belohnung $1$, und die Episode endet. Andernfalls bricht sie nach $2N$ Schritten ab. Um das Ziel zu erreichen, muss der Agent Aktion $1$ in $N - 1$ aufeinanderfolgenden Schritten wählen. Schon eine einzige Aktion $0$ schickt ihn zurück zum Start (:numref:`fig_rl_chain`).

<!--en:a95581ab-->
![Die Kette mit $N$ Zuständen. Aktion $1$ (blau) führt einen Zustand weiter, Aktion $0$ (orange) führt von überall zurück zu Zustand $0$. Die einzige Belohnung ist die $1$ beim Eintritt in den letzten Zustand. Damit endet die Episode.](../img/mdl-rl-chain.svg)

<!--en:7114e044-->
Die Funktion `chain` erstellt diesen MDP als `d2l.TabularMDP(P, r, gamma)` mit den Arrays aus :numref:`sec_mdp`. Dabei ist `P[s, a]` die Verteilung des nächsten Zustands und `r[s, a]` die erwartete Belohnung. Die Funktion `success_random` berechnet die genaue Wahrscheinlichkeit, mit gleichverteilt zufälligen Aktionen innerhalb von `horizon` Schritten das Ende zu erreichen. Dazu mittelt sie den Übergangskern über die beiden Aktionen, $P_\pi(s' \mid s) = \frac12 \sum_a P(s' \mid s, a)$. Ausgehend von Zustand $0$ berechnet sie mit $P_\pi$ Schritt für Schritt die Zustandsverteilung. Der letzte Zustand ist absorbierend. Seine Wahrscheinlichkeit nach `horizon` Schritten ist also die Wahrscheinlichkeit, bis dahin angekommen zu sein:

<!--en:646f3dcb-->
Die Funktion `d2l.q_learning(seed, Q, visits, env, num_episodes, gamma, alpha, epsilon)` aus :numref:`sec_qlearning` aktualisiert die Tabelle `Q` direkt anhand der Übergänge, die sie mit $\epsilon$-gierig gewählten Aktionen beobachtet. Sie zählt die Besuche in `visits` und liefert den Ertrag (engl. return) jeder Episode. Auf die Umgebung greift sie nur über `env.reset` und `env.step` zu. Die Klasse `ModelEnv` stellt für einen `TabularMDP` diese beiden Methoden bereit. Die Methode `reset` versetzt den Agenten in Zustand $0$, und `step` zieht den nächsten Zustand aus `P[s, a]`. Die Methode meldet die Terminierung in einem Endzustand oder den Abbruch nach `max_steps` Schritten.

<!--en:bfe6f10d-->
Wir verwenden das Q-Learning aus :numref:`sec_qlearning` mit einer Änderung: Jeder Tabelleneintrag beginnt bei $1$, einer oberen Schranke für jeden Ertrag, und der Agent handelt gierig mit $\epsilon = 0$. Bei dieser *optimistischen Initialisierung* übertragen wir die Idee hinter UCB auf eine Tabelle. Ein unversuchtes Paar erscheint so gut wie überhaupt möglich. Zum Vergleich gibt die Codezelle die erwartete Anzahl der Episoden bis zum ersten Erfolg mit zufälligen Aktionen aus. Das ist der Kehrwert der Wahrscheinlichkeit aus `success_random`. Beim Q-Learning mit einer anfangs auf null gesetzten Tabelle handelt der Agent bis zum ersten Erfolg genauso, unabhängig von $\epsilon$. In jeder Zeile herrscht nämlich Gleichstand, bei dem der Zufall entscheidet (siehe :exref:`ex_qlearning-exploration-schedule`):

<!--en:3e88f275-->
Bei zufälligen Aktionen verdoppelt jeder zusätzliche Zustand ungefähr die Zahl der nötigen Episoden: $23$ bei $N = 8$, etwa $3{,}400$ bei $N = 16$ und etwa $46{,}000$ bei $N = 20$. Der optimistisch initialisierte Agent braucht nur wenige Episoden, etwa acht bei $N = 20$. Ihre Zahl wächst ungefähr proportional zu $N$. Optimismus funktioniert, weil der erste Versuch eine Spur im Tabelleneintrag hinterlässt. Bringt ein Paar beim ersten Versuch keine Belohnung, bewegt sich sein Eintrag auf den Zielwert $r + \gamma \max_{a'} Q(s', a')$ von höchstens $\gamma < 1$ zu. Damit fällt er unter die Einträge aller unversuchten Paare, die noch bei $1$ liegen. Die gierige Regel bevorzugt deshalb ein bisher unversuchtes Paar. So arbeitet sich der Agent systematisch durch die Kette, statt auf den Zufall angewiesen zu sein. Die Zahl der Episoden wächst polynomiell mit $N$ statt exponentiell.

<!--en:80cb99f6-->
### Boni nach Besuchszahl

<!--en:24add65a-->
Bei optimistischer Initialisierung bleibt ein Paar für die Exploration attraktiv, bis der erste Versuch seinen Wert senkt. In einem stochastischen MDP sagt ein Versuch wenig aus. Der Agent muss ein Paar mehrfach besuchen, bevor er seinen Wert kennt. *Boni nach Besuchszahl* (engl. count-based bonuses) übertragen die UCB-Regel auf MDPs. Zur Belohnung jedes Paars kommt ein Bonus hinzu, der mit der Zahl seiner Besuche wie $1/\sqrt{n(s, a)}$ sinkt. Die Wert-Backups übertragen den Bonus auf die Zustände, von denen aus das Paar erreichbar ist. So plant der Agent seinen Weg zu Paaren, über die er wenig weiß. Die modellbasierte Intervallschätzung nutzt dafür ein gelerntes Modell :cite:`Strehl.Littman.2008`. Beim Q-Learning mit solchen Boni wächst der Regret wie die Quadratwurzel der Schrittzahl, bis auf Faktoren, die polynomiell von der Größe des MDP abhängen :cite:`Jin.AllenZhu.Bubeck.ea.2018`. Kommt ein Netz anstelle einer Tabelle zum Einsatz, besucht der Agent einen Zustand selten zweimal. In der Praxis ersetzen Agenten die Besuchszahl durch ein gelerntes Maß dafür, wie neu ein Zustand ist. Ein Beispiel ist der Vorhersagefehler eines Netzes, das lernt, eine feste Zufallsfunktion des Zustands vorherzusagen :cite:`Burda.Edwards.Storkey.ea.2019`.

<!--en:8a841493-->
### Abdeckung auf dem See

<!--en:cc862339-->
Brauchen wir diese Verfahren auf dem See? In :numref:`sec_qlearning` haben wir Q-Learning auf dem ruhigen See angewandt, auf dem jeder Schritt in die vorgegebene Richtung führt. Mit Schrittweite $\alpha = 0.9$ über $512$ Episoden lernte der Agent eine Tabelle, mit der er bei gieriger Aktionswahl das Ziel erreichte. Wir wiederholen den Durchlauf mit $256$ Episoden und drei Verläufen der Explorationsrate. Der Agent handelt gierig mit $\epsilon = 0$, immer zufällig mit $\epsilon = 1$ oder nach der Voreinstellung von `q_learning`. Diese Voreinstellung, `d2l.linear_schedule(1.0, 0.05, 128)`, senkt $\epsilon$ in der ersten Trainingshälfte linear von $1$ auf $0.05$. In jedem von zwanzig Durchläufen erfasst die Codezelle nach jeder Episode den Anteil der $44$ Paare, die der Agent mindestens einmal ausprobiert hat ($11$ Felder, von denen aus er handeln kann, mit je vier Aktionen). Sie erfasst auch den Anteil der Trainingsepisoden, die das Ziel erreichen, und prüft, ob die gierige Strategie der fertigen Tabelle das Ziel erreicht. Dafür genügt auf diesem deterministischen See eine Episode mit `d2l.evaluate` aus :numref:`sec_valueiter`:

<!--en:5db5942b-->
Auf dem ruhigen See reicht die Zufallswahl bei Gleichstand zur Exploration mit der gierigen Regel aus. Sie probiert etwa neun von zehn Paaren aus, und mit der fertigen Tabelle erreicht der Agent bei gieriger Wahl in fast jedem Durchlauf das Ziel. Das gelingt auch mit den Tabellen der beiden anderen Verläufe. Bei $\epsilon = 1$ probiert der Agent jedes Paar aus, erreicht aber in fast keiner Trainingsepisode das Ziel. Eine gute Tabelle zu lernen und schon beim Lernen Belohnungen zu sammeln sind unterschiedliche Ziele. Mit $\epsilon$ wägen wir sie gegeneinander ab. Bei allen drei Verläufen kommt es auf den ersten Erfolg an. Bis dahin steht jeder Eintrag auf null, und bei jedem Verlauf handelt der Agent zufällig. Für denselben Startwert des Zufallszahlengenerators fällt der erste Erfolg also immer in dieselbe Episode. Das Ziel ist sechs Schritte vom Start entfernt. Irrfahrten erreichen es innerhalb weniger Dutzend Episoden. Der See ist zu klein, um die Schwierigkeit der Kette sichtbar zu machen. Auf dem rutschigen Eis vom Anfang dieses Abschnitts sagt ein einzelner Besuch wenig über ein Paar aus. Es genügt daher nicht, jedes Paar einmal auszuprobieren (siehe :exref:`ex_qlearning-exploration-schedule`).

<!--en:d23d09a6-->
## Exploration bei der Aufgabe mit Prüfprogramm

<!--en:3ef78366-->
Die Aufgabe mit Prüfprogramm aus :numref:`sec_mdp` umfasst vier Rechenaufgaben als Prompts $x$. Für alle vier stehen dieselben neun möglichen Antworten $y$ zur Wahl. Das Prüfprogramm vergibt $r(x, y) = 1$, wenn $y$ die richtige Antwort auf den Prompt ist, und sonst $0$. Die Referenzstrategie $\pi_{\textrm{ref}}$ wendet die Softmax-Funktion für jeden Prompt auf eine Zeile von Präferenzen an. Sie verteilt die Wahrscheinlichkeit gleichmäßig auf die acht gewöhnlichen Antwortmöglichkeiten. Die neunte erhält fast keine Wahrscheinlichkeit. Sie ist eine Ausweichantwort, die alle Lösungen aufzählt und die das Prüfprogramm nie akzeptiert, weil es eine exakte Übereinstimmung verlangt. Bei zwei Prompts, $2+2$ und $3 \times 4$, ist die richtige Antwort unter den Kandidaten. Bei $8 \times 8$ und $9 \times 9$ passt keiner. Im Mittel über die Prompts ist eine gezogene Antwort mit Wahrscheinlichkeit $0.062$ richtig.

<!--en:de6710df-->
In jeder Runde erhält der Agent einen Prompt, das Prüfprogramm bewertet eine Antwort, und die Episode endet. Der Prompt bildet einen Kontext, den der Agent beobachten, aber nicht beeinflussen kann. Die Antwort entspricht dem Arm. Einen Banditen, dessen bester Arm vom beobachteten Kontext abhängt, nennen wir *kontextuellen Banditen* (engl. contextual bandit). Der Agent steht für jeden Prompt vor einem eigenen Banditen. Mithilfe eines Modells kann er das Gelernte auf andere Prompts übertragen.

<!--en:af335427-->
Ein Sprachmodell exploriert bei einer solchen Aufgabe anders als die bisherigen Regeln. Ein verbreitetes Trainingsverfahren, das wir in :numref:`sec_baselines` herleiten, zieht für einen Prompt eine Gruppe von $K$ Antworten aus der aktuellen Strategie und bewertet sie. Es erhöht die Wahrscheinlichkeit der Antworten mit überdurchschnittlichem Score innerhalb der Gruppe und senkt die der unterdurchschnittlichen. Erhalten alle Antworten einer Gruppe denselben Score, liegt keine über oder unter dem Mittelwert, und die Strategie bleibt unverändert. Beantwortet die Strategie einen Prompt mit Wahrscheinlichkeit $p = \pi(\textrm{correct} \mid x)$ richtig, so erhalten alle $K$ unabhängig gezogenen Antworten einer Gruppe denselben Score mit Wahrscheinlichkeit

<!--en:d2ae0d50-->
also der Wahrscheinlichkeit, dass alle $K$ Antworten falsch sind, zuzüglich der Wahrscheinlichkeit, dass alle richtig sind. Die Codezelle zieht $1{,}000$ Gruppen je Prompt und vergleicht den Anteil der Gruppen mit einheitlichem Score mit dieser Formel. Wir betrachten die Referenzstrategie und eine Strategie, die beide lösbaren Prompts fast immer richtig beantwortet. Letztere haben wir von Hand konstruiert, indem wir die Präferenz für jede richtige Antwort um $5$ erhöht haben. Sie wurde nicht trainiert. Das Array `R[x, y]` enthält für jedes Paar den Score des Prüfprogramms:

<!--en:eb69ac0a-->
Die beobachteten Anteile stimmen mit der Formel überein. Bei jedem $K$ hat bereits die Hälfte aller Gruppen zwangsläufig einen einheitlichen Score: Bei den beiden Prompts ohne richtigen Kandidaten ist jede Antwort falsch. Auch beliebig häufiges Sampling findet keine Antwort, die die Strategie nicht erzeugen kann. Bei den zwei lösbaren Prompts liegt die Referenzstrategie mit Wahrscheinlichkeit $p \approx 0.125$ richtig. Größere Gruppen sind hier häufiger gemischt. Etwa $78\%$ der Zweiergruppen haben einheitliche Scores, ein Drittel der Achtergruppen und fast keine der Gruppen mit $32$ Antworten. Ist ein Prompt fast gelöst, häufen sich wieder Gruppen mit einheitlichem Score. Bei $K = 8$ sind es etwa zwei Drittel, diesmal, weil fast jede Antwort richtig ist. Das Lernen an diesem Prompt verlangsamt sich.

<!--en:44ba2cbb-->
Der Agent exploriert ausschließlich durch sein eigenes Sampling. Die Wahrscheinlichkeit, dass eine Gruppe mindestens eine richtige Antwort enthält, beträgt $1 - (1 - p)^K$. Das ist der *pass@$K$*-Score, mit dem man Modelle für Programmierung und logisches Schließen bewertet :cite:`Chen.Tworek.Jun.ea.2021`. Bei einem Prompt, den die Strategie meist falsch beantwortet, entspricht er annähernd der Wahrscheinlichkeit, dass die Gruppe überhaupt ein Lernsignal liefert. Sampling mit höherer Temperatur verteilt die Wahrscheinlichkeit auf mehr Antworten. Bei einem fast gelösten Prompt entstehen dadurch mehr gemischte Gruppen, allerdings auch mehr falsche Antworten. Wo kein Kandidat richtig ist, hilft das nicht.

<!--en:4b9e6986-->
Jede Regel in diesem Abschnitt lernt durch Ausprobieren, welche Aktionen gut sind. Das kostet Züge an schlechteren Armen, Episoden mit Irrfahrten oder Gruppen ohne Lernsignal. In der Kette ist ein unversuchtes Paar schwer zu erreichen. Bei der Aufgabe mit Prüfprogramm bleibt eine richtige Antwort unentdeckt, wenn die Strategie sie nie erzeugt. Ein Experte, der gute Aktionen vormacht, könnte dem Agenten all das ersparen. In :numref:`sec_imitation` lernen wir aus solchen Demonstrationen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:b8ed2967-->
Ein Agent, der aus seinen eigenen Aktionen lernt, muss auch scheinbar schlechtere Aktionen ausprobieren, um herauszufinden, ob sie tatsächlich schlechter sind. Der Regret misst, was ihn das gegenüber einem Agenten kostet, der die beste Aktion von Anfang an kannte. Beim zehnarmigen Banditen legt sich die gierige Regel auf den ersten Arm fest, der einen Gewinn liefert. Bei festem $\epsilon$ verliert sie auf Dauer pro Zug denselben Betrag. Mit einem $\epsilon$, das wie $1/t$ abklingt, mit UCB und mit Thompson-Sampling wächst der Regret nur logarithmisch mit der Zahl der Züge. Wie Lai und Robbins zeigten, ist das die bestmögliche Rate. Bei UCB bestimmt der Koeffizient den Regret für ein gegebenes Problem. Der theoretisch begründete Koeffizient liefert für jeden Banditen eine Garantie. Mit ihm lernt der Agent hier aber langsamer. In einem MDP sinkt die Wahrscheinlichkeit, eine benötigte Aktionssequenz durch Zufall zu finden, geometrisch mit deren Länge. Optimistische Werte bewegen den Agenten dagegen dazu, gezielt unversuchte Paare aufzusuchen. Bewertet ein Prüfprogramm Gruppen gezogener Antworten, liefern Gruppen mit einheitlichem Score kein Signal. Die Strategie exploriert dann allein durch ihr eigenes Sampling.

<!--en:2294243a-->
## Übungen

<!--en:0256cad4-->
1. [code] **Verlauf der Explorationsrate.** ●● Bei einer mit null initialisierten Tabelle wählt `epsilon_greedy` bei jedem Gleichstand gleichverteilt zufällig. Alle Belohnungen in FrozenLake sind nichtnegativ.
    1. Zeigen Sie, dass bis zur ersten erfolgreichen Episode jeder Tabelleneintrag genau null bleibt und jedes $\epsilon$ bis dahin dasselbe Verhalten erzeugt. Was bedeutet das für die Wahl von $\epsilon$ anhand der Episode des ersten Erfolgs?
    1. Welche Einträge sind nach dem ersten Erfolg positiv? Zeigen Sie, dass ein einmal positiver Eintrag positiv bleibt. Zeigen Sie auch, dass die gierige Regel mit $\epsilon = 0$ in einer Zeile mit positivem Eintrag nie wieder Aktionen mit Wert null ausprobiert. Im Experiment zur Abdeckung probiert diese Regel auf dem ruhigen See etwa neun von zehn Paaren aus und lernt, das Ziel zu erreichen. Sagen Sie vorher, ob sie auch auf rutschigem Eis eine gute Strategie lernt.
    1. Das optionale Argument `epsilon` von `q_learning` ist eine Funktion, die jedem Episodenindex eine Explorationsrate zuordnet. Beispiele dafür liefert `linear_schedule`. Verwenden Sie auf dem rutschigen See wie zuvor $4000$ Episoden (siehe :numref:`sec_qlearning`). Führen Sie für jede Einstellung fünf Durchläufe mit verschiedenen Zufallsstartwerten aus: Halten Sie $\epsilon$ jeweils konstant bei $0$, $0.1$, $0.5$ und $1$ und testen Sie auch den Verlauf `linear_schedule(1.0, 0.05, num_episodes // 2)`, den `q_learning` in :numref:`sec_qlearning` verwendet. Geben Sie für jede Einstellung den mittleren Trainingsertrag an, die mit `d2l.evaluate` über $1000$ Episoden gemessene Erfolgsquote der abschließenden gierigen Strategie, den Anteil ausprobierter Paare und $\max_s |\max_a \hat{Q}(s, a) - V^*(s)|$. Traf Ihre Vorhersage zu?
    1. Ordnen Sie die Einstellungen einmal nach Trainingsertrag und einmal nach Erfolgsquote der gierigen Strategie. Wo unterscheiden sich die Rangfolgen, und wie erklären Sie den Unterschied?
   :exlabel:`ex_qlearning-exploration-schedule`

<!--en:673c4d4b-->
1. **Logarithmischer Regret von UCB.** ●● Mit $\kappa = \sqrt{2}$ entspricht die Regel :eqref:`eq_ucb` dem Verfahren UCB1 von :citet:`Auer.CesaBianchi.Fischer.2002`. Nach deren Satz 1 gilt für den erwarteten Regret nach $t$ Zügen an Armen mit Belohnungen in $[0, 1]$ die obere Schranke
   $$\sum_{a:\, \Delta_a > 0} \frac{8 \ln t}{\Delta_a} + \Big(1 + \frac{\pi^2}{3}\Big) \sum_a \Delta_a, \qquad \Delta_a = \mu^* - \mu_a.$$
   Leiten Sie diese Schranke her. Nutzen Sie dazu die Hoeffding-Ungleichung. Für den Mittelwert $\hat{\mu}_n$ von $n$ unabhängigen Belohnungen in $[0, 1]$ mit Erwartungswert $\mu$ gilt
   $$P(\hat{\mu}_n \geq \mu + h) \leq e^{-2 n h^2},$$
   und entsprechend für Abweichungen nach unten.
    1. Wählen Sie einen festen suboptimalen Arm $a$ und setzen Sie $n_0 = \lceil 8 \ln t / \Delta_a^2 \rceil$. Zeigen Sie: Zieht die Regel an Arm $a$ in einem Schritt $u \leq t$ bei $n(a) \geq n_0$, dann ist entweder der Index des besten Arms höchstens $\mu^*$, oder der empirische Mittelwert von Arm $a$ übersteigt $\mu_a$ um mehr als seinen Bonus.
    1. Halten Sie die Zugzahlen beider Arme fest. Zeigen Sie, dass jedes der beiden Ereignisse in Schritt $u$ mit Wahrscheinlichkeit höchstens $u^{-4}$ eintritt. Wenden Sie die boolesche Ungleichung auf die möglichen Zugzahlen an und summieren Sie über $u$. Zeigen Sie so, dass $\mathbb{E}[n(a)] \leq n_0 + \pi^2/3$ nach $t$ Zügen gilt.
    1. Verbinden Sie Teil (b) mit der Zerlegung des Regrets aus :eqref:`eq_regret`, um den Satz zu erhalten. Berechnen Sie die Schranke für `arms` bei $t = 2000$. Vergleichen Sie sie mit dem medianen Regret von rund $200$, den der Vergleich der Koeffizienten in diesem Abschnitt für $\kappa = \sqrt{2}$ ausgibt. Welcher Beweisschritt kostet am meisten Schärfe?
    1. Wiederholen Sie das Argument für einen allgemeinen Koeffizienten $\kappa$. Für welche $\kappa$ konvergiert die Summe über die Schritte in Teil (b) noch? Was folgt daraus für unsere Wahl $\kappa = 0.5$? (Die boolesche Ungleichung über die Zugzahlen ist grob. Eine Aufteilung in geometrisch wachsende Intervalle bei :cite:`Bubeck.CesaBianchi.2012` erfasst jedes $\kappa > 1$.)
   :exlabel:`ex_qlearning-ucb-logarithmic-regret`

<!--en:dd8fd72b-->
1. **Thompsons A-posteriori-Verteilung.** ● Beim Thompson-Sampling ziehen wir den Mittelwert jedes Arms aus $\mathrm{Beta}(1 + w, 1 + n - w)$, nachdem er $w$ Gewinne in $n$ Zügen erzielt hat.
    1. Zeigen Sie, dass dies die A-posteriori-Verteilung eines Bernoulli-Mittelwerts bei gleichförmiger A-priori-Verteilung auf $[0, 1]$ ist.
    1. Arm A hat $3$ Gewinne in $4$ Zügen erzielt. Arm B ist unversucht. Zeigen Sie, dass Thompson-Sampling Arm B mit Wahrscheinlichkeit $1/3$ wählt. Zeigen Sie allgemeiner für den Vergleich mit einem unversuchten Arm, dass die Wahrscheinlichkeit für die Wahl des unversuchten Arms gleich eins minus dem A-posteriori-Mittelwert des anderen Arms ist.
    1. Zeigen Sie: Bei gegebenen Daten wählt Thompson-Sampling jeden Arm mit der A-posteriori-Wahrscheinlichkeit, dass dieser Arm den größten Mittelwert hat.
    1. [code] Ersetzen Sie die gleichförmige A-priori-Verteilung einmal durch $\mathrm{Beta}(1, 9)$, die eher schlechte Arme erwarten lässt, und einmal durch $\mathrm{Beta}(9, 1)$, die eher gute Arme erwarten lässt. Wie ändert sich der mediane Regret nach $2000$ Zügen? Welche der beiden A-priori-Verteilungen wirkt wie eine optimistische Initialisierung, und warum?
   :exlabel:`ex_exploration-thompson-posterior`

<!--en:ddd2933e-->
1. [code] **Lange Horizonte.** ●● Die Funktion `bandit_run` spielt jeweils nur einen Durchlauf.
    1. Schreiben Sie eine Variante, die viele Durchläufe gleichzeitig spielt. Führen Sie `wins` und `count` als Arrays der Form (Anzahl der Durchläufe, Anzahl der Arme). Wählen Sie für UCB und für Thompson-Sampling die Arme aller Durchläufe in einem Schritt. Prüfen Sie, ob die Mediane bei $2000$ Zügen im Rahmen der Streuung über die Durchläufe mit denen dieses Abschnitts übereinstimmen.
    1. Starten Sie jeweils $200$ Durchläufe für UCB mit $\kappa \in \{0.15, 0.3, 0.5, \sqrt{2}\}$ und für Thompson-Sampling. Spielen Sie bis zu $10^5$ Züge und plotten Sie den medianen Regret auf logarithmischen Achsen. Holt $\kappa = \sqrt{2}$ gegenüber $\kappa = 0.5$ auf?
    1. Plotten Sie für $\kappa = 0.15$ neben dem Median auch den mittleren Regret. Warum wachsen beide unterschiedlich schnell? Welche der beiden Größen beschreibt das Risiko für jemanden, der die Regel einmal anwendet?
   :exlabel:`ex_exploration-long-horizons`

<!--en:cba4edf5-->
1. [extended] **Tiefe Exploration.** ●● Betrachten Sie die Kette aus diesem Abschnitt mit den Zuständen $0, \ldots, N-1$ und Start in Zustand $0$. Aktion $0$ führt zurück zu Zustand $0$, Aktion $1$ führt einen Zustand weiter. Beim Eintritt in Zustand $N-1$ erhält der Agent die Belohnung $1$, und die Episode endet. Andernfalls bricht sie nach $2N$ Schritten ab.
    1. Zeigen Sie, dass eine Strategie mit gleichverteilter Zufallswahl in einer Episode mit Wahrscheinlichkeit höchstens $2N \cdot 2^{-(N-1)}$ erfolgreich ist. Die erwartete Zahl der Episoden bis zum ersten Erfolg beträgt damit mindestens $2^{N-2}/N$. Vergleichen Sie diese Schranke mit den exakten Werten, die die Codezelle für die Kette ausgibt. Nach Teil (a) von :exref:`ex_qlearning-exploration-schedule` handelt das $\epsilon$-gierige Q-Learning mit einer anfangs auf null gesetzten Tabelle bis zum ersten Erfolg gleichverteilt zufällig. Die Schranke gilt also unabhängig vom Wert von $\epsilon$.
    1. Verfolgen Sie die ersten Episoden des optimistisch initialisierten, gierigen Agenten auf der Kette mit $N = 4$. Welche Einträge sinken in welcher Episode? Begründen Sie anhand dieses Musters, dass die benötigte Anzahl von Episoden höchstens polynomiell mit $N$ wächst.
    1. Testen Sie beide Agenten in `gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True)`. Dort endet eine Episode spätestens nach $100$ Schritten. Vergleichen Sie den optimistisch initialisierten, gierigen Agenten mit dem auf null initialisierten Agenten unter dem Verlauf `linear_schedule(1.0, 0.05, num_episodes // 2)`. Verwenden Sie $4000$ und $20000$ Episoden mit je fünf Zufallsstartwerten. Geben Sie die Erfolgsquote der gierigen Strategie unter `d2l.evaluate` und die Anzahl der benötigten Umgebungsschritte an. Berechnen Sie als Referenz die optimale Strategie mittels Wertiteration und geben Sie ihre Erfolgsquote an. Behält der Optimismus auf einer stochastischen Karte seinen Vorteil, und was kostet er?

   Das Projekt benötigt einige Minuten CPU-Zeit.
   :exlabel:`ex_exploration-deep-exploration`

<!--en:86533589-->
1. **Gruppen ohne Lernsignal.** ● Beantwortet die Strategie einen Prompt mit Wahrscheinlichkeit $p$ richtig, dann haben alle $K$ unabhängigen Antworten einer Gruppe mit Wahrscheinlichkeit $(1 - p)^K + p^K$ denselben Score.
    1. Bei festem $K \geq 2$: Für welches $p$ ist eine gemischte Gruppe am wahrscheinlichsten, und wie wahrscheinlich ist sie dann?
    1. Zeigen Sie für kleine $p$, dass die kleinste Gruppengröße $K$, bei der eine Gruppe mit Wahrscheinlichkeit mindestens $1/2$ gemischt ist, ungefähr $\ln 2 / p$ beträgt. Berechnen Sie diesen Näherungswert für die Referenzstrategie der Aufgabe mit Prüfprogramm mit $p \approx 0.125$ sowie für $p = 0.01$.
    1. Die Erfolgswahrscheinlichkeiten $p$ der Prompts eines Trainingsdatensatzes sind auf $[0, 1]$ gleichverteilt. Welcher Anteil der Gruppen der Größe $K = 8$ hat einheitliche Scores, und welche Prompts ergeben solche Gruppen?
    1. Für die beiden Prompts von `VerifierTask` ohne richtige Antwortmöglichkeit hat jede Gruppe einheitliche Scores, bei jedem $K$ und jeder Sampling-Temperatur. Nennen Sie zwei Änderungen an der Aufgabe oder am Agenten, die ein Lernen an diesen Prompts ermöglichen würden. Was benötigt jede Änderung, das das Prüfprogramm allein nicht liefert?
   :exlabel:`ex_exploration-identical-groups`

<!--en:ed056592-->
[Diskussion](https://d2l.discourse.group/t/36871)

<!--en:fa846604-->
Mehrarmige Banditen und Exploration<br>**kumulativer Regret · gierige und epsilon-gierige Regeln · UCB und Optimismus · Thompson-Sampling · Lai-Robbins-Schranke · tiefe Exploration in MDPs · pass@K**

<!--en:38345d19-->
::: {.slide title="Kosten der Exploration: Zufallsschritte kosten 19 Prozentpunkte"}

<!--en:b0445376-->
[Motivation und Grundidee]{.kicker}

<!--en:cf1da1d3-->
- **Kosten blinder Exploration auf FrozenLake:** Muss die optimale Strategie $\pi^*$ mit Wahrscheinlichkeit $\epsilon = 0.05$ zufällig handeln, sinkt ihre Erfolgsquote innerhalb von 100 Schritten von **$73.0\%$ auf $54.2\%$**.

<!--en:7543c642-->
- **Blinde und gezielte Exploration:** Die $\epsilon$-gierige Regel probiert bekanntermaßen verheerende Aktionen ebenso oft zufällig aus wie vielversprechende Alternativen.

<!--en:e87c7a20-->
- **Exploration beim Post-Training von LLMs:** Erhalten alle $K$ gezogenen Antworten auf einen Prompt denselben Score vom Prüfprogramm, entfällt die Aktualisierung relativ zur Gruppe.

<!--en:0b47547c-->
::: {.slide title="Mehrarmige Banditen und kumulativer Regret"}

<!--en:9db6bd9d-->
[Exploration in einem einzigen Zustand untersuchen]{.kicker}

<!--en:037a7dfd-->
- **Vergleichsproblem mit 10 Armen:** Bester Arm $\mu^* = \mu_2 = 0.90$ und kleinster Abstand $\Delta_{\min} = 0.90 - 0.60 = 0.30$ (Arm 6).

<!--en:f7cbfcc9-->
- **Linearer und sublinearer Regret:** Wer immer an einem Arm mit Mittelwert $0.50$ zieht, verliert $\Delta_a = 0.40$ pro Zug. Sublinearer Regret ($\textrm{Regret}(t)/t \to 0$) erfordert, dass suboptimale Arme immer seltener zum Zug kommen.

<!--en:75fb1ffc-->
::: {.slide title="Simulation des Banditen"}

<!--en:d3104e8d-->
[Gewinne, Züge und entgangene Belohnung erfassen]{.kicker}

<!--en:89d1cf2a-->
`bandit_run` führt für jeden Arm die Zähler `wins` und `count` (der empirische Mittelwert ist also $\hat{\mu}(a) = \textrm{wins}[a]/\textrm{count}[a]$) und summiert über die Runden die erwartete Differenz $\mu^* - \mu_{a_t}$:

<!--en:10daa253-->
Die Auswertung erfasst den erwarteten Abstand `arms.max() - arms[a]` anstelle der zufälligen binären Belohnungen. Dadurch entfällt die durch diese Zufallsentscheidungen verursachte Varianz im Gütemaß. Der lernende Agent erhält weiterhin keinen Zugriff auf `arms`.

<!--en:b36372b6-->
::: {.slide title="Gierige Auswahl, epsilon-gierige Auswahl und Boltzmann-Auswahl"}

<!--en:d6146688-->
[Von Schätzwerten zu Aktionswahrscheinlichkeiten]{.kicker}

<!--en:66c3a2f0-->
Für eine Zeile von Schätzwerten $\hat{Q}(s, \cdot) = (0.20, 0.90, 0.55, 0.10)$ verteilen drei heuristische Regeln die Explorationswahrscheinlichkeit unterschiedlich:

<!--en:4b88bd16-->
- **Gierige Regel ($\epsilon = 0$):** Legt $100\%$ der Wahrscheinlichkeitsmasse auf den aktuellen empirischen Maximierer $\mathrm{argmax}_a \hat{\mu}(a)$.

<!--en:ced89ed4-->
- **$\epsilon$-gierige Regel:** Reserviert für jede Aktion die Wahrscheinlichkeit $\epsilon/m$ ($0.3/4 = 0.075$), unabhängig davon, wie schlecht sie bisher abgeschnitten hat.

<!--en:5adafb8a-->
- **Boltzmann-Regel (Softmax-Funktion) $\propto \exp(\hat{Q}/\beta)$:** Die Wahrscheinlichkeit steigt stetig mit dem Schätzwert. Fast optimale Aktionen kommen häufiger zum Zug als sehr schlechte.

<!--en:d4237843-->
::: {.slide title="Warum die gierige Regel beim ersten Arm mit Gewinn bleibt"}

<!--en:7053aee4-->
[Was ohne Exploration schiefgeht]{.kicker}

<!--en:073d6e4e-->
- **Der erste Gewinn entscheidet:** Anfangs gilt $\hat{\mu}_0(a) = 0$. Sobald ein Arm $a$ den Wert $1$ liefert, übertrifft sein empirischer Mittelwert $\hat{\mu}(a) > 0$ dauerhaft den jedes unversuchten Arms ($0$).

<!--en:9b56b62e-->
- **Gewinnchance:** Der beste Arm $\mu_2 = 0.90$ gewinnt diese erste Auslosung in nur $0.90 / 4.87 \approx 18.5\%$ der Durchläufe.

<!--en:20e1e6e4-->
::: {.slide title="Regret bei fester und abklingender Explorationsrate"}

<!--en:82da6b08-->
[Lineare und logarithmische Kosten]{.kicker}

<!--en:2293d588-->
Ein Vergleich der gierigen Regel, der Regel mit konstantem $\epsilon = 0.1$ und der Regel mit harmonisch abklingendem $\epsilon_t = \min(1, 20/(t+1))$ über 20 Durchläufe mit je $2{,}000$ Zügen zeigt die Kosten ungezielter Exploration:

<!--en:a4a5e0df-->
- **Konstantes $\epsilon = 0.1$ verursacht lineare Kosten:** Ein Zufallszug kostet den mittleren Abstand $0.90 - 0.487 = 0.413$. Damit entstehen dauerhaft Kosten von $0.1 \times 0.413 \approx 0.041$ pro Zug ($\approx 83$ des medianen Regrets von $106$).

<!--en:aeb130de-->
- **Harmonisch abklingendes $\epsilon_t = 20/t$ erreicht Regret der Ordnung $O(\ln t)$:** Insgesamt nur $20 + 20\ln(2000/20) \approx 112$ Zufallszüge (Explorationskosten $\approx 46$, medianer Regret $46$). Die Konstante $20$ muss allerdings auf $\Delta_{\min}^{-2}$ abgestimmt sein.

<!--en:29d94d1d-->
::: {.slide title="Optimismus bei Unsicherheit: obere Konfidenzschranken"}

<!--en:6749b4d5-->
[Gezielte Exploration durch Hoeffding-Boni]{.kicker}

<!--en:78ffe537-->
Die Regel der **oberen Konfidenzschranke** (UCB) addiert zu jedem empirischen Mittelwert einen Unsicherheitsbonus. So zieht sie deterministisch am Arm mit dem höchsten optimistischen Index, statt blind zu explorieren (Auer et al., 2002):

<!--en:332329d4-->
- **Grundlage ist die Hoeffding-Ungleichung:** Für $n$ unabhängige Belohnungen in $[0, 1]$ gilt $\Pr(\mu_a > \hat{\mu}_n(a) + h) \leq e^{-2nh^2}$. Mit $h = \kappa\sqrt{\log t / n(a)}$ ist die Wahrscheinlichkeit, Arm $a$ zu unterschätzen, höchstens $t^{-2\kappa^2}$.

<!--en:cb544925-->
- **Zwei Gründe für einen Zug:** Entweder ist der empirische Mittelwert $\hat{\mu}(a)$ hoch (Ausnutzung), oder die Anzahl der Züge $n(a)$ ist im Verhältnis zu $\log t$ klein (gezielte Exploration).

<!--en:23eda293-->
::: {.slide title="UCB-Indizes und Verteilung der Züge im Bild"}

<!--en:5a3f919e-->
[Wie Optimismus schlechte Arme aussortiert]{.kicker}

<!--en:6bdd8e5c-->
Zur Initialisierung kommt jeder Arm einmal zum Zug, sodass $n(a) \geq 1$ gilt. Danach konzentriert UCB fast alle Züge auf den optimalen Arm. Die Indizes der suboptimalen Arme bleiben knapp unter $\mu^*$:

<!--en:8acdfd00-->
- **Schrumpfende Unsicherheitsbalken:** In Teilbild (a) sinkt der Bonus $\kappa\sqrt{\log t / n(a)}$ mit der Anzahl der Züge wie $n(a)^{-1/2}$. Bleiben Züge aus, wächst er nur wie $\sqrt{\log t}$.

<!--en:702adfa9-->
- **Die Indizes gleichen sich aus:** In Teilbild (b) entfallen nach $2{,}000$ Zügen mit $\kappa = 0.5$ auf Arm 2 ($\mu^* = 0.90$) insgesamt $1{,}852$ Züge. Schlechte Arme kommen nicht mehr zum Zug, sobald ihre obere Konfidenzschranke unter $0.90$ fällt.

<!--en:173d6491-->
::: {.slide title="Ziehen aus der A-posteriori-Verteilung (Thompson-Sampling)"}

<!--en:6be462a9-->
[Auswahl nach bayesschen Wahrscheinlichkeiten]{.kicker}

<!--en:67f72f52-->
- **Aktualisierung beim konjugierten Beta-Bernoulli-Modell:** Aus der gleichförmigen A-priori-Verteilung $\mathrm{Beta}(1, 1)$ ergibt sich mit $w_a$ Gewinnen und $n_a - w_a$ Niederlagen die Verteilung $\mathrm{Beta}(1 + w_a, 1 + n_a - w_a)$.

<!--en:5e8d521d-->
- **Wahrscheinlichkeitsgerechte Auswahl ohne Feinabstimmung:** Pro Arm einen Wert $\tilde{\mu}_a$ ziehen und $\mathrm{argmax}_a \tilde{\mu}_a$ wählen. So entspricht die Auswahlwahrscheinlichkeit jedes Arms seiner A-posteriori-Wahrscheinlichkeit, optimal zu sein.

<!--en:fa54714e-->
::: {.slide title="Die fünf Banditenregeln im doppelt logarithmischen Diagramm" layout="tight"}

<!--en:c9aa3a51-->
[Empirische Regret-Kurven]{.kicker}

<!--en:c7c2f414-->
Auf logarithmischen Achsen unterscheiden sich Regeln mit linearem Regret (Steigung $1$) deutlich von Regeln mit logarithmischem Regret (abflachende Kurven):

<!--en:8f7e7f8c-->
- **Verbesserung um den Faktor drei:** UCB ($\kappa = 0.5$) und Thompson-Sampling enden bei einem medianen Regret von $30$ bis $38$. Das ist rund ein Drittel des Werts bei festem $\epsilon = 0.1$ ($106$) und weniger als bei abklingendem $\epsilon_t = 20/t$ ($46$).

<!--en:b109740f-->
::: {.slide title="Einfluss des UCB-Koeffizienten auf die Exploration"}

<!--en:e01a0543-->
[Garantie für den ungünstigsten Fall und empirische Varianz]{.kicker}

<!--en:92eeecce-->
Ein Vergleich verschiedener Werte von $\kappa \in \{0.15, 0.50, \sqrt{2}\}$ über je 40 Durchläufe erklärt, warum das theoretisch begründete UCB1 ($\kappa = \sqrt{2}$) bei diesen Armen hinter $\kappa = 0.5$ zurückbleibt:

<!--en:0a121051-->
- **Warum UCB1 ($\kappa = \sqrt{2}$) hier $5\times$ so viel kostet:** UCB zieht an Arm $a$, bis $\kappa\sqrt{\log t / n(a)} \approx \Delta_a$ gilt. Somit gilt $n(a) \propto \kappa^2 \log t / \Delta_a^2$. Die Hoeffding-Schranke berücksichtigt die größtmögliche Bernoulli-Varianz $1/4$. Unser bester Arm hat dagegen die Varianz $0.9 \times 0.1 = 0.09$.

<!--en:fe95726e-->
- **Warum $\kappa = 0.15$ riskant ist:** Der mediane Regret beträgt nur $8$. Nach einem unglücklichen Start bleiben einzelne Durchläufe aber bei einem suboptimalen Arm (maximaler Regret $709$). Thompson-Sampling passt sich automatisch an, ohne Abstimmung von $\kappa$.

<!--en:f7126a50-->
::: {.slide title="Die informationstheoretische untere Schranke von Lai und Robbins"}

<!--en:89bbdc51-->
[Warum kein Algorithmus logarithmischen Regret unterbieten kann]{.kicker}

<!--en:86cd5bd4-->
Lai und Robbins (1985) bewiesen: Jeder konsistente Banditenalgorithmus muss an jedem suboptimalen Arm $a$ mindestens proportional zu $\ln t$ geteilt durch die KL-Divergenz $D(\mu_a \,\|\, \mu^*)$ ziehen:

<!--en:7b397b32-->
- **Fast optimale Arme unterscheiden:** Für den zweitbesten Arm ($\mu_6 = 0.60$ gegenüber $\mu^* = 0.90$) gilt $D(0.60 \,\|\, 0.90) = 0.311$. Er braucht mindestens $\ln(2000)/0.311 \approx 24.4$ Züge (Regret $\approx 7.3$) bis $t = 2000$.

<!--en:630a2ada-->
- **Optimale Rate:** Die Summe über $\Delta_a n_t(a)$ ergibt die nicht weiter verbesserbare Schranke $\textrm{Regret}(t) = \Omega(\ln t)$. UCB und Thompson-Sampling erreichen beide die optimale Wachstumsrate $\Theta(\ln t)$.

<!--en:3ec6843f-->
::: {.slide title="Tiefe Exploration in MDPs: die Kette mit N Zuständen"}

<!--en:6022f919-->
[Wenn der Weg zu einem unbekannten Zustand k Schritte erfordert]{.kicker}

<!--en:c521a9b8-->
- **Geometrisch sinkende Erfolgswahrscheinlichkeit bei Irrfahrten:** Führt Aktion $1$ vorwärts und Aktion $0$ zurück zu Zustand $0$, so erfordert der Weg zu Zustand $N-1$ genau $N - 1$ Schritte nach rechts in Folge (Wahrscheinlichkeit $2^{-(N-1)}$ bei einer Irrfahrt).

<!--en:25bc10c4-->
- **Q-Learning mit Nullinitialisierung:** Bis zur ersten Belohnung am Ziel bleiben alle Einträge bei $0$. Die $\epsilon$-gierige Regel führt eine blinde Irrfahrt aus.

<!--en:c92ffa15-->
::: {.slide title="Optimistische Initialisierung und Irrfahrt bei wachsender Kettenlänge"}

<!--en:d056bd1d-->
[Polynomieller und exponentieller Explorationsaufwand]{.kicker}

<!--en:f82a0ff9-->
Die optimistische Initialisierung aller Tabelleneinträge mit $Q_0(s, a) = 1.0$ (einer oberen Schranke für den Ertrag) und rein gierige Auswahl ($\epsilon = 0$) beseitigen der exponentielle Zeitbedarf der Irrfahrt:

<!--en:f4bdcad1-->
- **Warum Optimismus systematisch voranführt:** Bei einem Besuch ohne Belohnung bewegt sich $Q(s, a)$ auf $\gamma \max_{a'} Q(s', a') = \gamma < 1$ zu und sinkt unter den Wert jedes unversuchten Paars ($1.0$). Die gierige Auswahl erkundet so die Kette systematisch ($8$ Episoden bei $N = 20$ gegenüber $46{,}000$ bei Zufallswahl).

<!--en:d096741b-->
::: {.slide title="Boni nach Besuchszahl und Abdeckung auf FrozenLake" layout="tight"}

<!--en:ffdb399d-->
[Von deterministischen Ketten zu stochastischen Gittern]{.kicker}

<!--en:6d787e45-->
In einem stochastischen MDP lässt sich $P(\cdot \mid s, a)$ aus einem einzigen Besuch nicht bestimmen. Auf kleinen deterministischen Gittern genügt dagegen oft schon die Zufallswahl bei Gleichstand, um die meisten Paare zu besuchen:

<!--en:2d72f2bc-->
- **UCB-Boni nach Besuchszahl:** Ein Bonus von $c / \sqrt{n(s, a)}$ auf die Belohnungen bei Bellman-Backups lenkt den Agenten zu selten besuchten Paaren. Der Regret liegt bei $\tilde{O}(\sqrt{T})$ (Jin et al., 2018).

<!--en:299f7116-->
- **Abdeckung auf dem ruhigen See:** Das Ziel ist nur 6 Schritte entfernt. Die gierige Regel mit Zufallswahl bei Gleichstand probiert $93\%$ der Paare aus, in denen der Agent handeln kann. Bei $19/20$ der Startwerte lernt sie den Weg zum Ziel und gewinnt $60\%$ der Trainingsepisoden.

<!--en:973857ef-->
::: {.slide title="Exploration mit Antwortgruppen bei der Aufgabe mit Prüfprogramm"}

<!--en:adf76019-->
[Kontextuelle Banditen und pass@K beim Post-Training von LLMs]{.kicker}

<!--en:49184dd7-->
Bei der gruppenbasierten Strategieoptimierung (etwa GRPO in §{sec_baselines}) zieht der Agent $K$ Antworten je Prompt $x$ aus $\pi_\theta(\cdot \mid x)$. Bei der Aktualisierung erhöht er die Wahrscheinlichkeit von Antworten mit überdurchschnittlichem Score innerhalb ihrer Gruppe:

<!--en:739d9774-->
- **Gruppen ohne Lernsignal:** Sind alle $K$ Antworten falsch ($(1-p)^K$) oder alle $K$ richtig ($p^K$), ist jeder Vorteil null. Die Strategie exploriert allein durch ihr eigenes Sampling mit Erfolgswahrscheinlichkeit $\text{pass@}K = 1 - (1 - p)^K$.

<!--en:d2383643-->
::: {.slide title="Verständnisfragen zu Regret-Schranken und Exploration"}

<!--en:929b7671-->
[Verständnisprüfung]{.kicker}

<!--en:94f0b26a-->
Drei Rechenaufgaben zur Exploration lösen:

<!--en:81b4334f-->
1. **Kleiner oder großer Abstand zum besten Arm:** Nach der Lai-Robbins-Schranke bei $t = 2000$ ($\ln 2000 \approx 7.6$) braucht der Arm mit Mittelwert $0.60$ ($D = 0.31$) insgesamt $\approx 24$ Züge. Für den Arm mit Mittelwert $0.25$ gilt $D(0.25 \,\|\, 0.90) \approx 1.19$. Wie viele Züge braucht er, und welcher Arm trägt mehr zum kumulativen Regret bei?

<!--en:aa307142-->
2. **Auswahlwahrscheinlichkeit beim Thompson-Sampling:** Arm A hat $3$ Gewinne in $4$ Zügen erzielt ($\mathrm{Beta}(4, 2)$), Arm B ist unversucht ($\mathrm{Beta}(1, 1)$). Mit welcher Wahrscheinlichkeit wählt Thompson-Sampling den unversuchten Arm B?

<!--en:94400018-->
3. **Gruppengröße für die Exploration bei LLMs:** Für einen lösbaren Prompt gelte $p = 0.125$. Wie groß muss die Gruppe mindestens sein (Gruppengröße $K$), damit eine Gruppe von $K$ Antworten in mindestens $50\%$ der Fälle unterschiedliche Scores enthält?

<!--en:557b98b6-->
**Antworten:**

<!--en:4c1eb88f-->
1. Der Arm mit Mittelwert $0.25$ braucht $\approx 7.6 / 1.19 \approx 6.4$ Züge. Sein Regret beträgt $6.4 \times 0.65 \approx 4.2$. Der Arm mit Mittelwert $0.60$ kostet hingegen $24.4 \times 0.30 \approx 7.3$. Fast optimale Arme verursachen insgesamt mehr Regret, weil es länger dauert, sie vom besten Arm zu unterscheiden.

<!--en:3558396f-->
2. Für $X \sim \mathrm{Beta}(4, 2)$ und $U \sim \mathrm{Uniform}(0, 1)$ gilt $\Pr(U > X) = \mathbb{E}[1 - X] = 1 - \frac{4}{4+2} = 1/3$.

<!--en:a58dc43a-->
3. Es muss $1 - (0.875^K + 0.125^K) \geq 0.5$ gelten. Für $K = 5$ ergibt sich $1 - 0.513 = 0.487 < 0.5$, für $K = 6$ dagegen $1 - 0.449 = 0.551 \geq 0.5$. Also ist $K = 6$ (die Näherung $\ln 2 / p \approx 5.5$ führt zum selben Ergebnis).

<!--en:b33f2fbb-->
::: {.slide title="Zusammenfassung: mehrarmige Banditen und Exploration"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:2470f2a1-->
| Strategie | Regel zur Aktionswahl | Wachstum des Regrets und praktische Abwägung |

<!--en:b5f12cee-->
| **Rein gierig** | $\mathrm{argmax}_a \hat{\mu}(a)$ | Linearer Regret $\Theta(t)$. Bleibt beim ersten Arm, der einen Gewinn liefert |

<!--en:1de451d9-->
| **$\epsilon$-gierig mit fester Rate** | Zufallswahl mit Wahrscheinlichkeit $\epsilon$, sonst $\mathrm{argmax}_a \hat{\mu}(a)$ | Linearer Regret $\Theta(\epsilon t)$. Verschwendet dauerhaft Züge an bekanntermaßen schlechte Arme |

<!--en:e48bfc29-->
| **Abklingendes $\epsilon_t \propto 1/t$** | Zufallswahl mit Wahrscheinlichkeit $\min(1, c/t)$ | $O(\ln t)$ für $c \gtrsim m/\Delta_{\min}^2$. Setzt jedoch die Kenntnis von $\Delta_{\min}$ voraus |

<!--en:351516df-->
| **UCB** | $\mathrm{argmax}_a [\hat{\mu}(a) + \kappa\sqrt{\log t / n(a)}]$ | $O(\ln t)$ entspricht der Rate der Lai-Robbins-Schranke. Empfindlich gegenüber der Wahl von $\kappa$ |

<!--en:85dee438-->
| **Thompson-Sampling** | $\tilde{\mu}_a \sim \mathrm{Beta}(1+w_a, 1+n_a-w_a)$ ziehen, Arm mit größtem Wert wählen | Optimale Rate $O(\ln t)$ ohne abzustimmende Hyperparameter |

<!--en:1a309ed1-->
| **Optimistische Initialisierung / Bonus** | $Q_0(s, a) = R_{\max}$ oder $r + c/\sqrt{n(s, a)}$ | Auf Ketten genügen anstelle von $\Theta(2^N)$ Schritten einer Irrfahrt $O(N)$ gezielte Durchgänge |
