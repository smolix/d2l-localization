<!--en:6ae91db4-->
# Baselines und Varianzreduktion

<!--en:33079fe0-->
Im vorigen Abschnitt haben wir eine Strategie (engl. policy) trainiert, indem wir die Log-Wahrscheinlichkeit jeder gezogenen Aktion proportional zum Ertrag (engl. return) ihrer Trajektorie erhöht haben,

<!--en:b28bd62a-->
Im Mittel ergibt diese Schätzung den Gradienten des erwarteten Ertrags $J(\theta)$. Eine einzelne Stichprobe kann erheblich davon abweichen. Bei vier Episoden pro Aktualisierung war der Fehler einer einzelnen Schätzung für eine teilweise trainierte Strategie typischerweise mehr als dreimal so groß wie die Länge des Gradienten. Ein Batch ohne Erfolg führte zu gar keiner Aktualisierung. Das liegt zu einem großen Teil an den Gewichten. Auf griffigem Eis beträgt jeder Ertrag entweder null oder $\gamma^{T-1}$. Eine Aktion bleibt also entweder unberücksichtigt oder wird belohnt. Ein Umweg in einer erfolgreichen Episode wird wahrscheinlicher, während ein Schritt in ein Loch in einer gescheiterten Episode nie unwahrscheinlicher wird.

<!--en:a3678510-->
Williams erkannte das Problem 1992 in der Arbeit, in der er REINFORCE einführte, und schlug dort eine Lösung vor. Vor der Gewichtung zieht man vom Ertrag einen Referenzwert ab, die *Baseline* :cite:`Williams.1992`. In der Statistik heißt das eine Kontrollvariate. Zieht man eine entsprechend zentrierte Größe mit bekanntem Mittelwert ab, behält der Schätzer seinen Mittelwert, während seine Varianz sinken kann. Wir gehen davon aus, dass der Score einer von der Strategie selbst gezogenen Aktion im Mittel null ist. Daraus leiten wir ab, welche Größen wir abziehen dürfen, und messen anhand exakter Gradienten, wie viel Varianz wir damit jeweils einsparen. Die Actor-Critic-Verfahren des nächsten Kapitels und die gruppenweisen Aktualisierungen beim Post-Training von Sprachmodellen verwenden diesen Schätzer mit anderen Baselines.

<!--en:b04bfb8d-->
Wir verwenden weiterhin den See mit griffigem Eis, $\gamma = 0.95$ und die Softmax-Tabelle aus :numref:`sec_policygradient`. Auf griffigem Eis wirkt sich jede Aktion sichtbar auf den Ertrag aus. Auf rutschigem Eis ist Erfolg größtenteils Glückssache, und der Ertrag sagt wenig über die einzelne Aktion aus (wie :numref:`sec_policygradient` zeigt, muss ein Batch dort bei anfänglicher Gleichverteilung für dieselbe Präzision etwa neunmal so groß sein). Für jede Aktualisierung verwenden wir vier Episoden. Das sind so wenige, dass Unterschiede in der Varianz sichtbar werden. Für jeden Trainingsvergleich verwenden wir zwanzig Zufallsstartwerte, bei der größten Schrittweite hundert. Wir brechen Episoden nach 200 Schritten ab. Nur eine Strategie, die in einer Schleife feststeckt, läuft so lange, und ihr Wert liegt nahe null. Der Abbruch ändert daher kein Gewicht messbar. Ohne ihn entstehen bei den großen Schrittweiten weiter unten Episoden mit mehreren tausend Schritten.

<!--en:61f302bc-->
## Die Identität für den Mittelwert null

<!--en:7ac04572-->
Jedes Verfahren in diesem Abschnitt ändert das Gewicht eines Scores und muss dabei den Mittelwert erhalten. Das gewährleistet eine Identität, die wir bereits beim Beweis des Policy-Gradient-Theorems (:numref:`sec_policygradient`) verwendet haben. Hier formulieren wir sie für jede differenzierbare Strategie.

<!--en:e462f16d-->
### Lemma zum Mittelwert null

<!--en:28bca877-->
Bei der Softmax-Tabelle hat der Score der Aktion $a$ im Zustand $s$ nur einen von null verschiedenen Block. Er gehört zu den Präferenzen des Zustands $s$:

<!--en:c3f18593-->
Mitteln wir über die Aktion, die die Strategie in $s$ zieht, hat die Indikatorfunktion den Mittelwert $\pi_\theta(a' \mid s)$. Damit hebt sich der zweite Term auf. Das gilt für jede differenzierbare Strategie.

<!--en:9e9cc77a-->
**Lemma.** Für jeden Zustand $s$ gilt $\ \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s)\ \nabla_\theta \log \pi_\theta(a \mid s) = 0$.

<!--en:be95708e-->
**Beweis.** $\sum_a \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) = \sum_a \nabla_\theta\, \pi_\theta(a \mid s) = \nabla_\theta \sum_a \pi_\theta(a \mid s) = \nabla_\theta 1 = 0.$ $\blacksquare$

<!--en:74eaf309-->
In jedem Zustand hat also der Score der gezogenen Aktion den Mittelwert null. Was eine Aktion an Wahrscheinlichkeitsmasse gewinnt, verlieren die anderen.

<!--en:384e71b5-->
### Die bedingte Identität

<!--en:8ce32343-->
Daraus folgt eine etwas stärkere Aussage. Sei $c$ eine beliebige Größe, die bereits feststeht, wenn der Agent den Zustand $s_t$ erreicht. Das kann eine frühere Belohnung der Trajektorie sein, eine Konstante oder eine Funktion von $s_t$ selbst. Bedingt auf den Anfang der Trajektorie $(s_0, a_0, r_0, \ldots, s_t)$ steht der Wert $c$ fest, während die Aktion $a_t$ noch zufällig ist. Da die Strategie nur $s_t$ berücksichtigt, können wir das Lemma auf den inneren Erwartungswert anwenden:

<!--en:4d785fac-->
Multiplizieren wir einen beliebigen Score im REINFORCE-Schätzer mit einer solchen Größe, erhalten wir also einen Term mit Mittelwert null. Ziehen wir die Größe vom Gewicht des Scores ab, bleibt der Mittelwert des Schätzers erhalten. Entscheidend ist der bisherige Verlauf der Trajektorie. Für eine Größe, die von $a_t$ oder von späteren Ereignissen abhängt, gilt diese Aussage nicht.

<!--en:ef33d8d7-->
## Künftiger Ertrag, Baselines und Kontrollvariaten

<!--en:659ba126-->
Die Identität erlaubt zwei Änderungen am Gewicht in Schritt $t$. Wir dürfen Terme weglassen, die vor $a_t$ feststehen, und eine Größe abziehen, die vor $a_t$ feststeht. Beides erhält den Mittelwert. Ob eine dieser Änderungen die Varianz verringert, ist eine andere Frage.

<!--en:331c31a1-->
### Künftiger Ertrag und Kausalität

<!--en:7413758e-->
Im REINFORCE-Schätzer multiplizieren wir den Score in Schritt $t$ mit dem Ertrag $R(\tau_i)$ der gesamten Trajektorie. Er enthält auch Belohnungen aus der Zeit, bevor die Aktion $a_t^i$ gezogen wurde. Diese Belohnungen stehen durch den bisherigen Verlauf fest. Nach der bedingten Identität hat ihr Produkt mit dem Score in Schritt $t$ also den Mittelwert null. Lassen wir sie weg, bleibt

<!--en:3e798be9-->
Das ist *Kausalität*: Eine Aktion kann nur die Zukunft beeinflussen, die Vergangenheit steht fest. Die innere Summe ist gleich $\gamma^t \hat{G}_t$. Dabei bezeichnet

<!--en:60db3d61-->
den *künftigen Ertrag* (engl. reward-to-go) ab Schritt $t$, also den diskontierten Ertrag der verbleibenden Episode. Der einfache Schätzer multipliziert eine Summe aus $T$ Scores mit einem Ertrag, der selbst eine Summe über $T$ Schritte ist. Gibt es bei jedem Schritt eine Belohnung, wächst seine Varianz daher rasch mit der Länge der Episode. Lassen wir frühere Belohnungen weg, verkürzt sich die Summe, mit der wir jeden Score multiplizieren. Einen Term mit Mittelwert null wegzulassen, senkt jedoch nicht zwangsläufig die Varianz. Der Term kann mit dem Rest korrelieren. Die Messungen weiter unten zeigen beide Fälle.

<!--en:633fae28-->
In der Praxis lassen Implementierungen, auch unsere, zusätzlich den Faktor $\gamma^t$ weg und gewichten den Score in Schritt $t$ allein mit $\hat{G}_t$:

<!--en:fe1073e8-->
Das Weglassen von $\gamma^t$ folgt nicht aus der Identität. Dadurch zählt ein später Besuch eines Zustands genauso viel wie ein früher. Für $\gamma < 1$ weicht deshalb der Mittelwert von :eqref:`eq_rtg` in seiner Richtung von $\nabla_\theta J$ ab. :citet:`Thomas.2014` identifizierte diese Verzerrung in Actor-Critic-Algorithmen. :citet:`Nota.Thomas.2020` zeigten, dass die Aktualisierung überhaupt kein Gradient einer Zielfunktion ist. Bei der unten untersuchten Strategie beträgt der Winkel etwa acht Grad (Kosinus $0.991$), und der Mittelwert ist etwa $1.4$-mal so lang wie der Gradient (Aufgabe :exref:`ex_baselines-reward-to-go-taken-apart`). Wir behalten diese Konvention bei, die auch die auf diesem Schätzer aufbauenden Verfahren verwenden, und messen beide Varianten.

<!--en:f78f8c78-->
Ein Rückwärtsdurchlauf pro Episode liefert den künftigen Ertrag jedes Schritts gemäß $\hat{G}_t = r_t + \gamma \hat{G}_{t+1}$, mit Startwert null nach dem letzten Schritt. `backward_scan` berechnet diese Rekursion für einen beliebigen Vektor und Faktor (in späteren Abschnitten verwenden wir andere Größen und Faktoren). Die Rekursion durchläuft jede Episode, die `Batch.episodes` aus :numref:`sec_policygradient` liefert. `reward_to_go` wendet sie mit dem Faktor $\gamma$ auf die Belohnungen an:

<!--en:f1e0afa7-->
### Baselines

<!--en:7c5f1810-->
Die zweite Operation ist die Subtraktion. Jede Größe $b$, die durch den bisherigen Verlauf der Trajektorie bis $s_t$ feststeht, dürfen wir vom künftigen Ertrag abziehen. Das kann eine Konstante oder eine Funktion $b(s_t)$ des aktuellen Zustands sein:

<!--en:aac9fe65-->
Ein solches $b$ heißt *Baseline*. Der zusätzliche Term ist das Produkt von $b(s_t)$ und dem Score. Nach der bedingten Identität hat dieses Produkt den Mittelwert null. Daher haben :eqref:`eq_pg_baseline` und :eqref:`eq_rtg` denselben Mittelwert.

<!--en:0e435392-->
Auf dem See ist der künftige Ertrag stets nichtnegativ. Ohne Baseline wird daher keine gezogene Aktion jemals unwahrscheinlicher. Ziehen wir einen Wert nahe dem typischen künftigen Ertrag ab, erhalten überdurchschnittliche Fortsetzungen positive Gewichte und unterdurchschnittliche negative. Wie kann sich das Lernen ändern, wenn der Mittelwert gleich bleibt? Über die Varianz, die wir im nächsten Unterabschnitt berechnen. Und warum ziehen wir keine Schätzung von $Q(s_t, a_t)$ ab, dem besten Prädiktor für das Gewicht? Sie hängt von $a_t$ ab. Damit greift die Identität nicht mehr und der Mittelwert ändert sich (Aufgabe :exref:`ex_baselines-admissible`).

<!--en:50a04b06-->
### Kontrollvariaten

<!--en:f39b095f-->
Das Abziehen einer Baseline ist ein Spezialfall eines allgemeinen Verfahrens zur Monte-Carlo-Schätzung. Angenommen, wir wollen $\mathbb{E}[X]$ schätzen, und jedes Sample enthält zusätzlich eine damit korrelierte Größe $Y$ mit bekanntem Erwartungswert. Für jeden Skalar $c$ definieren wir

<!--en:2c3d53da-->
Wegen $\mathbb{E}[Y-\mathbb{E}[Y]]=0$ hat der Schätzer $X_c$ denselben Erwartungswert wie $X$, und zwar für jedes $c$. Seine Varianz ist

<!--en:b6bae1d5-->
eine quadratische Funktion von $c$ mit Minimum bei

<!--en:d5203bfa-->
Setzen wir $c^*$ wieder ein, bleibt eine Varianz von $(1 - \mathrm{corr}^2)\, \textrm{Var}(X)$. Dabei ist $\mathrm{corr}$ die Korrelation zwischen $X$ und $Y$. Eine starke Korrelation ermöglicht eine erhebliche Reduktion. Bei $\mathrm{corr}=0.9$ sinkt die Varianz etwa um den Faktor fünf, bei $\mathrm{corr}=0$ bleibt sie gleich. Die Größe $Y$ heißt *Kontrollvariate* für $X$.

<!--en:60c07ccb-->
Für einen Policy-Gradienten setzen wir $X = \hat{G}_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ und $Y = \nabla_\theta \log \pi_\theta(a_t \mid s_t)$. Nach dem Lemma gilt $\mathbb{E}[Y\mid s_t]=0$, und mit $c=b(s_t)$ erhalten wir wieder :eqref:`eq_pg_baseline`. Eine Baseline ist also eine Kontrollvariate, die den Score nutzt. Der beste Koeffizient ist $c^*$, den wir für jeden Zustand getrennt berechnen. Bei einem vektorwertigen Score treten innere Produkte an die Stelle der Produkte und liefern die optimale Baseline $b^*$ aus Aufgabe :exref:`ex_baselines-optimal-constant-baseline`. :citet:`Greensmith.Bartlett.Baxter.2004` entwickeln diese Analyse für Policy-Gradienten.

<!--en:31c62bf6-->
Schon der einfachste Fall zeigt, wie groß der Effekt ist. Betrachten wir das Problem mit einem Schritt aus :numref:`fig_rl_score_ascent`. Wir ziehen eine Aktion $a$ aus der Normalverteilung $\mathcal{N}(a; \mu, 1)$ bei $\mu = 0$ und erhalten die Belohnung $R(a)=0.4+2e^{-(a-2)^2/2}$. Die erwartete Belohnung ist $J = \mathbb{E}[R] = 0.92$, der Score bezüglich $\mu$ ist $a - \mu = a$, und der exakte Gradient ist $\mathrm{d}J/\mathrm{d}\mu = 0.52$. Die Schätzung $R(a)\,a$ aus einem einzigen Sample hat diesen Gradienten als Mittelwert. Ihre Standardabweichung beträgt $1.34$, mehr als das Zweieinhalbfache der geschätzten Größe. Ziehen wir die mittlere Belohnung ab, ergibt sich $(R(a) - 0.92)\,a$, und die Standardabweichung sinkt auf $0.69$. Die beste Konstante ist $b^* = \mathbb{E}[R(a)\,a^2] / \mathbb{E}[a^2] = 1.18$. Sie liegt über der mittleren Belohnung, weil Aktionen weit weg von null große Scores und hier auch große Belohnungen haben. Mit dieser Konstante bleibt ein Anteil von $1-\mathrm{corr}^2 = 0.23$ der Varianz ohne Baseline (:numref:`fig_rl_variance_reduction`).

<!--en:95b9a87d-->
![Zwei Wege, die Streuung des Score-Funktions-Schätzers zu verringern. (a) In einer Episode mit mehreren Schritten hat das Produkt früherer Belohnungen mit dem Score einer späteren Aktion den Erwartungswert null. Daher bleibt nur der künftige Ertrag $\hat G_3$. (b) Beim Problem mit einem Schritt aus :numref:`fig_rl_score_ascent` sinkt die Standardabweichung einer Schätzung aus einem Sample durch Abziehen der mittleren Belohnung von $1.34$ auf $0.69$. Beide Histogramme sind am 1. und 95. Perzentil abgeschnitten. (c) Die Varianz als Funktion einer konstanten Baseline $b$ ist eine Parabel mit Minimum bei $b^* = 1.18$. Dort bleibt der Anteil $0.23$ der Varianz bei $b = 0$.](../img/mdl-rl-variance-reduction.svg)

<!--en:5d30aca9-->
### Der Vorteil und die gelernte Baseline

<!--en:eb3c70c4-->
Auf dem See passt keine einzelne Konstante zu jedem Zustand. Auf einem Feld neben dem Ziel ist der erwartete künftige Ertrag viel höher als am Start. Als zustandsabhängige Baseline bietet sich der Zustandswert an. Nach :numref:`sec_valueiter` gilt $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)$. Das ist der Mittelwert der Aktionswerte unter der Strategie. Der *Vorteil* (engl. advantage) einer Aktion gibt an, um wie viel sie diesen Mittelwert übertrifft,

<!--en:eb12b62b-->
Nach der eben angegebenen Beziehung beträgt der Mittelwert null. Ein positiver Vorteil kennzeichnet eine Aktion, die den Mittelwert der Strategie in $s$ übertrifft. Wegen $\mathbb{E}[\hat{G}_t \mid s_t, a_t] = Q^{\pi_\theta}(s_t, a_t)$ ist das Gewicht $\hat G_t - V^{\pi_\theta}(s_t)$ ein Sample des Vorteils der gewählten Aktion. Die Aktualisierung macht somit Aktionen wahrscheinlicher, die den Mittelwert des Zustands übertreffen, und alle anderen unwahrscheinlicher.

<!--en:1b488bcc-->
Wir kennen $V^{\pi_\theta}$ nicht, können diesen Wert aber aus denselben Batches schätzen. Dazu führen wir eine Tabelle $\hat{V}(s)$ und verschieben nach jedem Batch den Eintrag jedes besuchten Zustands in Richtung des dort beobachteten künftigen Ertrags,

<!--en:5d2afa11-->
mit der Schrittweite $\eta_V$. Das ist REINFORCE mit Baseline :cite:`Williams.1992`. Wir passen die Tabelle durch Regression an Monte-Carlo-Erträge an, also an die künftigen Erträge aus vollständigen Episoden. Bilden wir die Zielwerte aus den Vorhersagen von $\hat{V}$ für den jeweils nächsten Schritt, erhalten wir die Actor-Critic-Verfahren aus :numref:`chap_deep_rl`. Ohne Bootstrapping vereinfacht sich deren Vorteilsschätzung zu $\hat{G}_t - \hat{V}(s_t)$.

<!--en:381d306c-->
## Zentrierung, Skalierung und Normalisierung

<!--en:33986c4a-->
In der Praxis standardisiert man häufig die Gewichte jedes Batches oder vergleicht die Samples einer Gruppe miteinander. Außerdem muss jede Implementierung festlegen, wodurch sie den aufsummierten Verlust teilt. Manche Entscheidungen ändern den Schätzer, andere nur seine Skalierung. Beides lässt sich leicht verwechseln. Um die Effekte zu unterscheiden, brauchen wir eine feste Strategie, deren exakten Gradienten wir kennen.

<!--en:4cbf63b6-->
Wir erzeugen `probe` genau wie in :numref:`sec_policygradient` mit `d2l.train_reinforce_tabular` und denselben sechzehn Aktualisierungen aus jeweils sechzehn Episoden. Ihr Wert $J(\theta)=V^{\pi_\theta}(s_0)$ ist der erste Eintrag der Lösung des linearen Gleichungssystems $V = r_\pi + \gamma P_\pi V$ aus :numref:`sec_valueiter` mit $P_\pi(s' \mid s) = \sum_a \pi_\theta(a \mid s)\, P(s' \mid s, a)$ und $r_\pi(s) = \sum_a \pi_\theta(a \mid s)\, r(s, a)$. Mit `probe.probs` lesen wir $\pi_\theta$ aus der Teststrategie aus. Die Methode `mdp.policy_value(pi_np)` löst dieses System in NumPy und liefert $V^{\pi_\theta}$ für jeden Zustand, also die exakte zustandsabhängige Baseline:

<!--en:5151054a-->
`exact_gradient` differenziert `d2l.policy_objective` aus :numref:`sec_policygradient` an der Präferenztabelle der Teststrategie. Die Funktion beschreibt dasselbe System mit den Arrays des Frameworks. Sie erhält die Belohnungstabelle `r` als Argument und lässt sich daher später auch mit einer geänderten Belohnung verwenden:

<!--en:568cf6bb-->
Der ausgegebene Wert $J(\theta) = 0.313$ bei einem bestmöglichen Wert von $\gamma^5 \approx 0.77$ entspricht dem Wert der Teststrategie in :numref:`sec_policygradient`. Ihr exakter Gradient dient bei allen folgenden Messungen als Referenz.

<!--en:81cb41df-->
### Batchzentrierung und Skalierung anhand der Varianz

<!--en:4971e47a-->
Eine verbreitete Variante standardisiert die künftigen Erträge innerhalb jedes Batches. Dazu sammeln wir alle $\hat{G}_t^i$ im aktuellen Batch, berechnen ihren Mittelwert $\bar{R}$ und ihre Standardabweichung $\sigma$ und verwenden

<!--en:d9cb34ed-->
anstelle von $\hat{G}_t^i$. Die Konstante $10^{-8}$ verhindert eine Division durch null. Beim Abziehen von $\bar{R}$ verwenden wir eine Baseline. Sie ändert die relativen Gewichte der Samples und kann dadurch die Richtung der Schätzung ändern. Da $\bar{R}$ aus demselben Batch stammt, hängt dieser Mittelwert schwach von den gezogenen Aktionen ab. Im nächsten Unterabschnitt berechnen wir die resultierende Verzerrung für Gewichte pro Trajektorie, in Aufgabe :exref:`ex_baselines-batch-centering` für die hier verwendeten Gewichte pro Schritt.

<!--en:cdc80d56-->
Die Division durch $\sigma + 10^{-8}$ multipliziert die gesamte Schätzung eines Batches mit einer positiven Zahl. Innerhalb eines Batches wirkt sie somit wie eine Schrittweite, die sich an die Streuung der Belohnungen anpasst. Eine Baseline wirkt anders. Über mehrere Batches hinweg ist der Faktor zufällig und mit dem jeweiligen Batch korreliert. Daher ändert sich auch die Richtung der erwarteten Aktualisierung. Aufgabe :exref:`ex_baselines-sigma-objective` untersucht den verwandten Effekt eines $\sigma$, das von Prompt zu Prompt variiert. Auf dem See folgt aus $0\leq\hat{G}_t\leq1$, dass $\sigma\leq1/2$ gilt. Bei festgelegter Schrittweite sind die Schritte mit Normalisierung also mindestens doppelt so lang wie mit Zentrierung. Weiter unten messen wir etwa die dreifache Länge. `standardize` implementiert :eqref:`eq_pg_normalized`:

<!--en:4e24b8a9-->
### Die Leave-one-out-Baseline

<!--en:2aab77ce-->
Der Mittelwert des Batches enthält auch das Sample, von dem wir ihn abziehen. Die Leave-one-out-Zentrierung beseitigt diese Abhängigkeit, indem sie für jede Trajektorie den Mittelwert der *anderen* Trajektorien als Baseline verwendet,

<!--en:6c517835-->
Da wir die Trajektorien unabhängig ziehen, ist $b_i$ unabhängig von Trajektorie $i$. Die bedingte Identität für den Mittelwert null gilt ohne Korrektur. Der Schätzer ist bei jeder Batchgröße erwartungstreu. Seine Gewichte ergeben sich aus der Zentrierung um $\bar{R}$ durch Skalierung,

<!--en:ae166cad-->
Die einfache Zentrierung mit einem Gewicht pro Trajektorie multipliziert den Mittelwert des Schätzers mit $(n-1)/n$, halbiert ihn also bei $n = 2$. Wir prüfen beide Aussagen an einem Problem, für das sich alle Fälle aufzählen lassen. Es gibt zwei Zustände, die wir nacheinander besuchen, je zwei Aktionen und eine Tabelle für die Belohnungen. `leave_one_out` berechnet die Gewichte $R_i - b_i$ eines Batches:

<!--en:122a056d-->
Die nächste Codezelle berechnet den exakten Gradienten `g` aus den vier möglichen Trajektorien und den exakten Mittelwert beider Schätzer über alle sechzehn geordneten Batches der Größe zwei:

<!--en:6f85c8c0-->
Beide Identitäten gelten bis auf Maschinengenauigkeit. :citet:`Kool.Hoof.Welling.2019` schlugen die Leave-one-out-Baseline für REINFORCE mit mehreren Samples pro Eingabe vor. Unter dem Namen RLOO kommt sie beim Post-Training von Sprachmodellen zum Einsatz :cite:`Ahmadian.Cremer.Galle.ea.2024`. Für einen Prompt zieht man $n$ Antworten und gewichtet jede mit ihrer Belohnung abzüglich der mittleren Belohnung der übrigen $n-1$ Antworten.

<!--en:2adc7917-->
### Die Gruppe als Batch

<!--en:33ec5a6e-->
Auf dem See beginnt jede Episode auf demselben Feld. Ein Batchmittelwert reicht daher für alle Episoden. Bei der Aufgabe mit Prüfprogramm aus :numref:`sec_mdp` ist das anders. Es gibt vier Rechenaufgaben als Prompts, neun gemeinsame Antwortkandidaten und ein Prüfprogramm, das eine richtige Antwort mit eins belohnt. Die Referenzstrategie hat die Erfolgswahrscheinlichkeit $0.062$. Keine Strategie kann $0.5$ übertreffen, weil für zwei Prompts die richtige Antwort fehlt. Jeder Prompt bildet einen eigenen Startzustand mit eigener Erfolgsquote und braucht deshalb eine eigene Baseline. `verifier_step` aus :numref:`sec_policygradient` zieht $K$ Antworten pro Prompt und gewichtet den Score $\mathbf{e}_y - \pi(\cdot \mid x)$ jeder Antwort $y$ mit einer Funktion der $K$ Belohnungen. Die $K$ Antworten auf einen Prompt, die *Gruppe*, liefern die Baseline.

<!--en:248bd1dc-->
Bei Belohnungen von null und eins hat die Zentrierung um den Gruppenmittelwert eine einfache Form. Sind $k$ der $K$ Antworten richtig, beträgt der Gruppenmittelwert $k/K$. Eine richtige Antwort erhält damit das Gewicht $1 - k/K$, eine falsche das Gewicht $-k/K$. Ohne Baseline hatte eine falsche Antwort Gewicht null. Jetzt wird sie unwahrscheinlicher, und zwar umso stärker, je häufiger die Strategie diesen Prompt bereits richtig beantwortet. Teilen wir durch die Standardabweichung $\sqrt{\hat p(1 - \hat p)}$ der Gruppe mit $\hat p = k/K$, ergeben sich die beiden Gewichte

<!--en:c8732a52-->
Bei $K = 16$ und zwei richtigen Antworten erhält jede richtige Antwort etwa $+2.65$ und jede falsche etwa $-0.38$. Ein seltener Erfolg bei einem schwierigen Prompt wird also stark verstärkt. Erzielen alle Antworten einer Gruppe dieselbe Belohnung, weil alle falsch oder alle richtig sind, fehlt der Vergleich. Jede Antwort erhält dann Gewicht null. In :numref:`sec_exploration` haben wir gezählt, wie häufig das vorkommt.

<!--en:ef73bc8a-->
Das standardisierte Gewicht der Antwort $j$ in ihrer Gruppe ist

<!--en:146bad4f-->
wobei $\bar{R}$ und $\sigma$ Mittelwert und Standardabweichung der $K$ Belohnungen der Gruppe sind. Dies ist die Variante von :eqref:`eq_pg_normalized` mit einem Gewicht pro Trajektorie. Der Prompt ist der Startzustand und die Gruppe der Batch. Hier standardisieren wir eine Belohnung pro Antwort statt eines künftigen Ertrags pro Schritt. Dieses Gewicht verwendet Group Relative Policy Optimization (GRPO) :cite:`Shao.Wang.Zhu.ea.2024`. Die Funktion `group_step` entspricht `verifier_step` mit `standardize` zur Gewichtung:

<!--en:cfff1b19-->
Der Gruppenmittelwert enthält die Belohnung der Antwort selbst. Nach der Leave-one-out-Rechnung entspricht die erwartete gruppenzentrierte Aktualisierung daher dem $(K-1)/K$-Fachen des Gradienten. Für Zweiergruppen prüfen wir das exakt, ohne durch $\sigma$ zu teilen. Die Codezelle berechnet den Score jeder Antwort (`Sv`), den exakten Gradienten der Erfolgswahrscheinlichkeit jedes Prompts als Summe von neun Termen $\sum_y \pi(y \mid x)\, r(x, y)\, (\mathbf{e}_y - \pi(\cdot \mid x))$ (`gv`) und den exakten Mittelwert der zentrierten Aktualisierung über alle geordneten Antwortpaare (`u`):

<!--en:f134a051-->
Wie stark wirkt sich die Verzerrung aus? Bei $K=1$ ist der Faktor null. Der Gruppenmittelwert entspricht der Belohnung der Antwort selbst, jedes Gewicht ist null, und die Strategie ändert sich nie, obwohl ihr Gradient von null verschieden ist. Für $K \geq 2$ verkürzt der Faktor lediglich den Schritt. Beim Vergleich halten wir das Budget bei $6{,}400$ bewerteten Antworten pro Prompt fest. Kleinere Gruppen machen daher mehr Schritte ($3{,}200$ bei $K = 2$, $200$ bei $K = 32$):

<!--en:617dbd61-->
Bei $K=1$ behält die Strategie wie vorhergesagt den Referenzwert $0.062$. Für jedes $K\geq2$ erreicht sie mit demselben Budget die Obergrenze von $0.5$, und zwar bei allen zehn getesteten Zufallsstartwerten. Die Gruppe liefert den Vergleich pro Prompt, den sonst eine gelernte Wertetabelle liefern würde. Die Division durch $\sigma$ pro Prompt hat einen weiteren Effekt, der hier nicht sichtbar wird. Sie ändert, welche Prompts die Aktualisierung bevorzugt (Aufgabe :exref:`ex_baselines-sigma-objective`).

<!--en:b9ec0747-->
### Der Divisor des Verlusts

<!--en:627987d4-->
Die Doppelsumme in :eqref:`eq_rtg` läuft über Episoden und Schritte. Bei der Implementierung müssen wir ihren Divisor festlegen. Teilen wir durch die Anzahl der Episoden $n$, erhalten wir genau diese Gleichung. Teilen wir durch die Gesamtzahl der Schritte, skalieren wir jeden Batch mit dem Kehrwert seiner mittleren Episodenlänge. Diese ändert sich, während die Strategie besser wird. Teilen wir jede Episode durch ihre eigene Länge, ändern sich die relativen Gewichte der Episoden und möglicherweise auch die Richtung. Eine feste Konstante ändert nur die Skalierung. Wir berechnen alle vier Varianten für einen Batch aus vier unterschiedlich langen Episoden. `agg` summiert die Scores des Batches `b` unter der Strategie `pi` mithilfe von `d2l.tabular_score` aus :numref:`sec_policygradient`. Jeden Score gewichten wir mit seinem künftigen Ertrag mal einem Faktor `scale` pro Schritt:

<!--en:fe4b6958-->
Drei der vier Gradienten sind parallel, obwohl sich ihre Längen um eine Größenordnung unterscheiden. Die Variante mit der Länge der einzelnen Episode weicht in der Richtung ab. Die Funktion `policy_step` aus :numref:`sec_imitation` teilt durch die Gesamtzahl der Schritte, also nach der dritten Variante. Die Trainingsschleife weiter unten skaliert ihre Gewichte daher so, dass wir die erste Variante erhalten. Beim Training von Sprachmodellen stellt sich dieselbe Frage, mit Antworten als Episoden und Token als Schritten. GRPO teilt jede Antwort durch ihre eigene Länge, DAPO teilt durch die Gesamtzahl der Token :cite:`Yu.Zhang.Zhu.ea.2025`, und Dr. GRPO teilt durch eine Konstante. Nur die letzte Variante skaliert jeden Batch mit demselben Faktor :cite:`Liu.Chen.Li.ea.2025`.

<!--en:9a39e3c4-->
## Fünf Schätzer im Vergleich

<!--en:c9bad0fa-->
Wir haben nun fünf Gewichtungen für den Score in Schritt $t$. Welche verringern die Varianz, und um wie viel? Das untersuchen wir zuerst bei einer festen Strategie, deren exakten Gradienten wir kennen, und anschließend im Training.

<!--en:192ed29b-->
### Schätzer bei fester Strategie

<!--en:c4f6a8b2-->
Wir vergleichen folgende fünf Gewichtungen:

<!--en:001b7aaf-->
1. **Ertrag der Trajektorie** $R(\tau)$ wie beim erwartungstreuen REINFORCE-Schätzer zu Beginn dieses Abschnitts.

<!--en:9e6eb192-->
2. **Künftiger Ertrag** $\hat{G}_t$ wie in :eqref:`eq_rtg`, ohne den Faktor $\gamma^t$.

<!--en:08296160-->
3. **Zentrierter** künftiger Ertrag $\hat{G}_t - \bar{R}$, wobei $\bar{R}$ der Mittelwert über die Schritte des Batches ist.

<!--en:f8ae4c0d-->
4. **Normalisierter** künftiger Ertrag :eqref:`eq_pg_normalized`.

<!--en:f77174ec-->
5. **Wert-Baseline** $\hat{G}_t - V(s_t)$. Bei fester Strategie verwenden wir den exakten Wert $V^{\pi_\theta}$ aus dem linearen Gleichungssystem, das wir hier ohne Weiteres lösen können. Im Training verwenden wir die gelernte Tabelle $\hat{V}$ aus :eqref:`eq_value_baseline`.

<!--en:fd2fd487-->
Eine sechste Zeile, **kausal**, gewichtet den Score in Schritt $t$ mit $\gamma^t \hat{G}_t$. Das ist der erwartungstreue Schätzer aus der Herleitung über die Kausalität. Damit unterscheiden wir den Effekt des Weglassens früherer Belohnungen vom Effekt des Weglassens von $\gamma^t$.

<!--en:e2aef09a-->
Wir halten die Teststrategie fest, ziehen $2{,}000$ Batches zu je vier Episoden und berechnen jeden Schätzer für jeden Batch. `estimate` summiert die gewichteten Scores eines Batches mit `d2l.tabular_score` und teilt wie in :eqref:`eq_rtg` durch die Episodenzahl. `discounted` multipliziert jeden künftigen Ertrag mit $\gamma^t$, wobei $t$ ab Beginn der jeweiligen Episode zählt. `weightings` ordnet jedem Namen eine Funktion zu, die den Batch, seine künftigen Erträge und eine Wertetabelle erhält (`per_step` wiederholt den Ertrag jeder Episode für jeden ihrer Schritte). `stats` berechnet die drei ausgegebenen Spalten. `|mean|/|grad J|` und `cos` beschreiben Länge und Richtung des mittleren Schätzvektors. Die erste Größe gibt die Länge des Stichprobenmittels relativ zu $\nabla_\theta J$ an, die zweite den Kosinus zwischen beiden Vektoren. Die Spalte `noise` gibt die relative Varianz an. Dazu teilen wir die Spur der Kovarianzmatrix der gezogenen Schätzwerte durch die quadrierte Länge ihres eigenen Mittelwerts. Als Bezug wählen wir den Mittelwert des jeweiligen Schätzers, weil die normalisierten Gewichte durch Division der zentrierten Gewichte durch $\sigma$ entstehen. Die Skalierung allein soll die Bewertung eines Schätzers nicht beeinflussen.

<!--en:0b15246c-->
Auf dem See gibt es nur am Ziel eine Belohnung. Es gibt also keine früheren Belohnungen, die wir weglassen könnten. Um die Wirkung der Kausalität zu sehen, wiederholen wir die Messung in der Codezelle mit Kosten von $0.1$ für jeden Zug von einem nichtterminalen Feld aus. Jeder aufgezeichnete Schritt ist ein solcher Zug. Ziehen wir von jeder Belohnung eines Batches $0.1$ ab, erhalten wir dieselben Episoden unter der neuen Belohnung. `costly` ist der MDP mit demselben Übergangskern und der Belohnung abzüglich der Kosten. Er liefert den exakten Gradienten und $V^\pi$:

<!--en:bfb65ef8-->
Auf dem See stimmen die Zeilen für den Ertrag und die kausale Gewichtung bei jeder Ziehung überein. Da es nur am Ende einer erfolgreichen Episode eine Belohnung gibt, gilt bei jedem Schritt $\gamma^t \hat{G}_t = \gamma^{T-1}$, und dieser Wert ist der Ertrag selbst. Der künftige Ertrag unterscheidet sich davon nur durch den weggelassenen Faktor $\gamma^t$. Sein Mittelwert ist etwa $1.4$-mal so lang und sein Kosinus etwas kleiner. Mittelt man einen erwartungstreuen Schätzer über $2{,}000$ Ziehungen, liegt der Kosinus nahe $0.997$. Diese Spalte zeigt also vor allem die Richtungsabweichung. Die Spalte für das Rauschen zeigt eine Rangfolge, die sich bei anderen Zufallsstartwerten bestätigt. Die Wert-Baseline streut am wenigsten. Danach folgen gleichauf die zentrierte und die normalisierte Variante. Ertrag und künftiger Ertrag streuen am stärksten. Zentrierung beseitigt etwa 40 Prozent des Rauschens des künftigen Ertrags, die Wert-Baseline etwa die Hälfte. Zentrierung verkürzt den Mittelwert außerdem auf etwa das 0.8-Fache des Mittelwerts beim künftigen Ertrag, nahe dem Faktor $(n-1)/n = 3/4$ bei Zentrierung pro Trajektorie. Die Division durch $\sigma$ macht den Mittelwert etwa dreimal so lang und belässt das Rauschen auf dem Niveau der Zentrierung, wie bei einer Änderung der Schrittweite zu erwarten.

<!--en:9ce86b21-->
Mit Kosten pro Zug sinkt das Rauschen des Ertragsschätzers hier durch Weglassen früherer Belohnungen auf etwa ein Drittel. Zwingend ist das nicht. Die weggelassenen Terme haben den Mittelwert null, korrelieren jedoch mit dem Rest, und $\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\,\textrm{Cov}(X, Y)$ kann kleiner sein als $\textrm{Var}(X)$. Bei Kosten von $0.01$ pro Zug steigt das Rauschen durch Weglassen um etwa ein Sechstel (Aufgabe :exref:`ex_baselines-reward-to-go-taken-apart`). Auch die Reihenfolge der übrigen Zeilen gilt für diese Strategie. Bei der Gleichverteilungsstrategie hilft Zentrierung nicht (Aufgabe :exref:`ex_baselines-variance-ordering`).

<!--en:6359d511-->
### Schrittweite und Varianz im Training

<!--en:be00a4c2-->
Im Training bauen viele Schritte aufeinander auf, und die Schrittweite spielt eine Rolle. Die folgenden Varianten verwenden dieselben Zufallsstartwerte und unterscheiden sich nur in dem Gewicht, das sie an `policy_step` übergeben. Zur Aktualisierung verwenden sie einfaches SGD mit Schrittweite $\eta$. Adam würde jede Koordinate neu skalieren und die untersuchten Größenunterschiede teilweise ausgleichen. Jeder Lauf aktualisiert $\hat{V}$ gemäß :eqref:`eq_value_baseline`, obwohl nur die Variante mit Wert-Baseline diese Tabelle nutzt. `entropy` mittelt die Entropie $-\sum_a \pi(a \mid s) \log \pi(a \mid s)$ in Nat über die Zustände eines Batches. Jeder Zustand zählt also so oft, wie der Agent ihn besucht:

<!--en:91564d7a-->
`train` implementiert REINFORCE mit einfachem SGD, einer Gewichtung aus `weightings` und Schrittweite `eta`. Die übrigen Argumente sind die Umgebung und die zu Beginn des Abschnitts festgelegten Einstellungen. `policy_step` mittelt über die $N$ Schritte eines Batches, während wir bei der Tabelle für die feste Strategie über seine $n$ Episoden gemittelt haben. Multiplizieren wir jedes Gewicht mit $N/n$, wird aus dem ersten Mittelwert der zweite. Damit verwendet jede Variante den Schätzer, den wir in der Tabelle untersucht haben. Ohne diesen Faktor wäre die effektive Schrittweite $\eta\, n/N$ und würde mit kürzer werdenden Episoden wachsen. Übergeben wir den MDP als `exact`, ersetzt `train` die gelernte Tabelle durch den exakten Wert $V^{\pi_\theta}$ der Strategie vor der Aktualisierung. Wie bei der festen Strategie lösen wir dafür vor jeder Aktualisierung das Gleichungssystem neu. Diese Variante nennen wir *exakte Baseline*. Bei jeder Aktualisierung zeichnen wir die Erfolgsquote des Batches, die Länge des Parameterschritts und die Entropie entlang der besuchten Zustände auf.

<!--en:a86b3202-->
`table` kopiert die Präferenztabelle aus der Strategie. `d2l.plain_sgd` (:numref:`sec_utils`) ersetzt den Adam-Optimierer der Strategie durch einfaches SGD. `d2l.run_seeds` (:numref:`sec_utils`) ruft `train` einmal pro Zufallsstartwert auf und stapelt die Ergebnisse:

<!--en:de5ef1c3-->
`train` verarbeitet alle Zufallsstartwerte gemeinsam. Jeder hat einen eigenen Zufallszahlengenerator, eine eigene Umgebung und eine eigene Wertetabelle. Beim Erzeugen der Trajektorien durchlaufen wir die Zufallsstartwerte in einer Schleife, weil die Umgebung in Python immer nur einen Zustand weiterschaltet. Die Aktualisierung ist eine einzige JAX-Berechnung. `sgd_step` entspricht `policy_step` mit einfachem SGD für eine Präferenztabelle. `jax.vmap` wendet diese Funktion entlang einer führenden Achse für die Zufallsstartwerte an. Ein kompilierter Aufruf aktualisiert somit alle Tabellen. Jeder Lauf zieht dieselben Zufallszahlen wie bei separater Ausführung:

<!--en:e3b70bbe-->
Mit diesen Hilfsfunktionen führt `train` die REINFORCE-Schleife mit einfachem SGD für `num_updates` Batches aus. Dabei skaliert die Funktion jedes Schrittgewicht aus `weighting(batch, G, V)` mit `len(batch) / batch_episodes`, aktualisiert die Wertetabelle `V` gemäß :eqref:`eq_value_baseline` (oder berechnet $V^\pi$ exakt, wenn `exact` vorliegt) und zeichnet den mittleren Ertrag, die Norm des Parameterschritts und die Entropie der Strategie auf:

<!--en:e477fe1e-->
Die Gewichtungen sind unterschiedlich skaliert. Ein gemeinsames $\eta$ führt daher zu unterschiedlich langen Schritten, und schon die Schrittlänge beeinflusst, wie schnell ein Lauf lernt. Auch ein Abgleich der mittleren Schrittlänge löst dieses Problem nicht. In die Länge eines Schritts geht auch sein Rauschen ein. Sobald eine Strategie Erfolg hat, werden die Schritte mit Baseline kürzer. Ohne Baseline werden die ohnehin gewählten Aktionen weiterhin verstärkt. Varianten mit derselben mittleren Schrittlänge machen daher während des Lernens immer noch unterschiedlich lange Schritte. Für einen aussagekräftigen Vergleich halten wir stattdessen die erwartete Aktualisierung fest. Der exakte Wert $V^{\pi_\theta}$ steht bereits vor dem Ziehen des Batches fest, ebenso der gelernte Wert $\hat{V}$, den wir nur an frühere Batches angepasst haben. Nach der bedingten Identität liefern künftiger Ertrag, Wert-Baseline und exakte Baseline daher bei jedem $\theta$ dieselbe erwartete Aktualisierung. Für ein festes $\eta$ unterscheiden sie sich nur in ihrem Rauschen. Wir testen alle sechs Varianten bei $\eta = 2$ (`shared`) und diese drei zusätzlich bei $\eta = 16$ mit `num_seeds_large` Zufallsstartwerten (`large`). `setup` enthält die gemeinsamen Einstellungen aller Läufe. Die Läufe jeder Variante stapeln wir zu einem Array der Form (Zufallsstartwerte, Aktualisierungen, 3):

<!--en:c789bd0c-->
Die folgende Darstellung zeigt die Erfolgsquote der drei Varianten mit gleichem Mittelwert bei $\eta = 16$, geglättet über ein Fenster von zehn Aktualisierungen. Jedes Band zeigt den Interquartilbereich um den Median über die Zufallsstartwerte:

<!--en:165ba123-->
`to90` ermittelt die erste Aktualisierung, bei der die Erfolgsquote über die letzten zehn Aktualisierungen 90 Prozent erreicht. Die Toleranz gleicht Rundungsfehler bei neun von zehn aus. Für beide Schrittweiten gibt die Codezelle den Median über die Zufallsstartwerte zusammen mit dem schnellsten und dem langsamsten Lauf aus. Außerdem zeigt sie die Anzahl der Läufe, die in den 150 Aktualisierungen nie 90 Prozent erreichen, und die mittlere Schrittlänge über die Aktualisierungen, bevor ein Lauf 90 Prozent erreicht:

<!--en:06a92d0c-->
Bei der gemeinsamen Schrittweite erklärt sich die Rangfolge durch die Schrittlänge. Die Normalisierung benötigt mit etwa 30 gegenüber 50 bis 70 die wenigsten Aktualisierungen. Bis sie 90 Prozent erreicht, sind ihre Schritte jedoch etwa dreimal so lang wie bei der Zentrierung und mehr als anderthalbmal so lang wie beim künftigen Ertrag. Die Zentrierung ist etwa so schnell wie die Gewichtung mit dem einfachen Ertrag. Sie verkürzt den Schritt und ist daher langsamer als die direkte Gewichtung mit dem künftigen Ertrag. Bei dieser Schrittweite lassen sich die drei Varianten mit gleichem Mittelwert kaum unterscheiden. Jede benötigt etwa 50 Aktualisierungen.

<!--en:3c7d0e5d-->
Bei $\eta = 16$ treten Unterschiede auf. Alle drei lernen schneller als bei $\eta = 2$, doch die exakte Baseline profitiert am meisten. Ihr Median fällt von 50 auf etwa 20 Aktualisierungen, der des künftigen Ertrags von 51 auf etwa 35. Der größere Unterschied liegt bei den Fehlschlägen. Mit dem künftigen Ertrag erreicht etwa jeder siebte Lauf nie 90 Prozent, mit der exakten Baseline gelingt es allen. Fast jeder dieser erfolglosen Läufe ist stecken geblieben. Lange Schritte in verrauschte Richtungen haben die Strategie auf einem Pfad nahezu deterministisch gemacht, der bis zum Zeitlimit im Kreis führt oder in einem Loch endet. Jede spätere Episode hat Ertrag null, jedes Gewicht ist null, und die Strategie ändert sich nie wieder. Die gelernte Tabelle liegt dazwischen (Median 26 Aktualisierungen, einer von zwanzig Läufen erreicht nie 90 Prozent), weil sich die Strategie bei dieser Schrittweite schneller ändert, als $\hat{V}$ folgen kann. Die Spalte für die Schrittlänge zeigt, dass längere Schritte den Gewinn nicht erklären. Bis zum Erreichen von 90 Prozent legt die exakte Baseline pro Aktualisierung höchstens etwa ein Sechstel mehr zurück als der künftige Ertrag. Eine geringere Varianz verbessert einen kleinen Schritt nicht, denn über viele kleine Schritte mittelt sich das Rauschen heraus. Sie macht einen längeren Schritt nutzbar (Aufgabe :exref:`ex_baselines-step-size-confound-quantified`).

<!--en:01cfce7b-->
Bei $\eta = 2$ benötigt der langsamste Lauf jeder Variante zwei- bis sechsmal so viele Aktualisierungen wie der schnellste. Bei $\eta = 16$ streuen die Läufe ohne exakte Baseline noch stärker. Bei zwanzig Zufallsstartwerten können daher nahe beieinanderliegende Mediane ihre Reihenfolge tauschen. In einem separaten Experiment mit 200 Zufallsstartwerten pro Variante ziehen wir jeweils zwanzig davon zufällig. Bei $\eta = 2$ unterscheiden sich die Mediane des künftigen Ertrags und der exakten Baseline bei etwa jeder siebten Ziehung um mehr als sechs Aktualisierungen. Bei $\eta = 16$ hat die exakte Baseline in etwa 49 von 50 Ziehungen den niedrigeren Median. Ein einziger Lauf mit künftigen Erträgen, der einen anderen Verlauf nimmt, kann jedoch den Median von zwanzig Läufen um mehr als fünfzehn Aktualisierungen verschieben. Für die Varianten bei $\eta = 16$ verwenden wir daher 100 Zufallsstartwerte. Wir geben Mediane und Spannweiten an und prüfen eine Aussage mit weiteren Zufallsstartwerten, bevor wir uns auf sie verlassen :cite:`Henderson.Islam.Bachman.ea.2018,Agarwal.Schwarzer.Castro.ea.2021`.

<!--en:0c01badf-->
Wie deterministisch wird die Strategie in den Zuständen, die der Agent besucht?

<!--en:9674362e-->
Bei $\eta = 16$ sinkt der Median der über den Batch gemittelten Entropie bei beiden Varianten innerhalb der ersten 20 bis 40 Aktualisierungen von $\ln 4 \approx 1.39$ Nat bei der Gleichverteilungsstrategie auf unter $0.1$ Nat. Beim künftigen Ertrag geschieht das meist früher. Ein festgefahrener Lauf ist so deterministisch wie ein erfolgreicher. Auf dem Pfad, den er wiederholt, liegt die Entropie nahe null Nat. Ein Mittelwert über alle sechzehn Zustände bliebe über $0.4$ Nat, weil der Agent in den vier Löchern und am Ziel keine Aktion ausführt und die Entropie dort bei $\ln 4$ bleibt.

<!--en:724f49a6-->
Wir können nun angeben, welche Gewichte zulässig sind, wie viel Varianz jedes davon einspart und wie wir einen besseren Schätzer von einem längeren Schritt unterscheiden. Bisher griff jeder Schätzer auf Tabellen zu, mit sechzehn Zeilen für die Präferenzen und sechzehn Einträgen für $\hat{V}$. Ein Wagen auf einer Schiene lässt sich durch vier reelle Zahlen beschreiben und besucht nie zweimal denselben Zustand. In :numref:`sec_deeprl` ersetzen wir beide Tabellen durch Netze.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ca4c1c48-->
Der Score einer Aktion, die eine Strategie selbst zieht, hat den Mittelwert null. Dasselbe gilt für sein Produkt mit jeder Größe, die vor der Aktion feststeht. Daher dürfen wir frühere Belohnungen weglassen und eine Baseline abziehen, ohne den erwarteten Gradienten zu ändern. Eine Baseline ist eine Kontrollvariate. Mit dem Zustandswert $V^\pi$ als Baseline wird das Gewicht zu einem Sample des Vorteils. Bei der untersuchten Strategie beseitigte diese Baseline etwa die Hälfte des Rauschens. Zentrieren wir den Ertrag jeder Trajektorie um den Batch- oder Gruppenmittelwert, multipliziert das den erwarteten Gradienten mit $(n-1)/n$. Leave-one-out beseitigt diesen Faktor. Bei schrittweiser Zentrierung verkürzt sich der Gradient um einen anderen Faktor. Die Division durch die Streuung ändert die Schrittweite. Bei binären Belohnungen macht das gruppennormalisierte Gewicht von GRPO erfolglose Antworten unwahrscheinlicher. Mit einer Gruppe der Größe eins ist kein Lernen möglich. Eine gemeinsame Schrittweite begünstigt die Gewichtung mit den längsten Schritten. Vergleichen wir die exakte Wert-Baseline und den künftigen Ertrag bei gleicher Schrittweite und gleichem Mittelwert, lernen sie bei kleiner Schrittweite gleich schnell. Bei großer Schrittweite lernt die Variante mit Baseline schneller und bleibt nicht stecken.

<!--en:2294243a-->
## Aufgaben

<!--en:112c73aa-->
1. **Zulässige Baselines.** Nach der bedingten Identität für den Mittelwert null dürfen wir jede Größe, die bereits beim Erreichen des Zustands $s_t$ feststeht, von $\hat{G}_t$ abziehen. Entscheiden Sie für jede der folgenden Größen, ob ihr Abziehen in :eqref:`eq_pg_baseline` den erwarteten Gradienten erhält. Begründen Sie Ihre Antwort anhand dieser Bedingung.
    1. der Zeitindex $t$,
    1. die Summe $r_0 + \cdots + r_{t-1}$ der vor Schritt $t$ erhaltenen Belohnungen,
    1. die Anzahl $T - t$ der verbleibenden Schritte der Episode,
    1. eine Funktion des nächsten Zustands $s_{t+1}$,
    1. eine Schätzung des Aktionswerts $\hat{Q}(s_t, a_t)$.

   Berechnen Sie für den letzten Fall mit $\hat{Q} = Q^{\pi_\theta}$ den erwarteten Beitrag von Schritt $t$ nach der Subtraktion. Was sagt das Ergebnis über aktionsabhängige Baselines aus?
   :exlabel:`ex_baselines-admissible`

<!--en:c1989873-->
1. **Optimale konstante Baseline.** Halten Sie einen Zustand $s$ fest. Schreiben Sie $\boldsymbol{\psi} = \nabla_\theta \log \pi_\theta(a \mid s)$ für den Score der gezogenen Aktion und betrachten Sie den Schätzer $(\hat{G} - b)\, \boldsymbol{\psi}$ für einen einzelnen Schritt mit einer Konstanten $b$. Sein Mittelwert hängt nicht von $b$ ab. Wir messen sein Rauschen daher anhand der Spur seiner Kovarianzmatrix,

   $$\mathbb{E}\big[\|(\hat{G} - b)\, \boldsymbol{\psi}\|^2\big] - \big\|\mathbb{E}[\hat{G}\, \boldsymbol{\psi}]\big\|^2.$$

    1. Minimieren Sie die Spur bezüglich $b$ und zeigen Sie, dass sie ihr Minimum an folgender Stelle annimmt:

       $$b^* = \frac{\mathbb{E}\big[\|\boldsymbol{\psi}\|^2\, \hat{G}\big]}{\mathbb{E}\big[\|\boldsymbol{\psi}\|^2\big]}.$$

       Zeigen Sie, dass sich die Spur für jedes andere $b$ um genau $\mathbb{E}\big[\|\boldsymbol{\psi}\|^2\big]\, (b - b^*)^2$ erhöht. Zeigen Sie außerdem, dass $b^*$ dem Koeffizienten $c^*$ aus :eqref:`eq_control_variate` für $X = \hat{G}\, \boldsymbol{\psi}$ und $Y = \boldsymbol{\psi}$ entspricht, wenn innere Produkte an die Stelle der Produkte treten.
    1. Beim Problem mit einem Schritt aus :numref:`fig_rl_variance_reduction` gilt für die Aktion $a \sim \mathcal{N}(\mu, 1)$ bei $\mu = 0$. Der Score bezüglich $\mu$ ist daher $a$. Berechnen Sie $b^*$ für $R(a) = 0.4 + 2e^{-(a-2)^2/2}$ und bestätigen Sie den Wert $b^* = 1.18$. Das Produkt der Standardnormaldichte mit $e^{-(a-2)^2/2}$ ist proportional zur Dichte einer anderen Normalverteilung. Warum ist $b^*$ größer als $\mathbb{E}[R] = 0.92$?
   :exlabel:`ex_baselines-optimal-constant-baseline`

<!--en:3da7fceb-->
1. [code] **Verzerrung durch Batchzentrierung.** ● In :numref:`subsec_leave_one_out` ergibt sich der Faktor $(n-1)/n$, mit dem die Batchzentrierung pro Trajektorie den erwarteten Gradienten verkürzt, aus der Leave-one-out-Identität $R_i - b_i = \frac{n}{n-1}(R_i - \bar{R})$. Dieses Argument gilt nur für Gewichte pro Trajektorie.
    1. Leiten Sie den Faktor direkt ohne Leave-one-out-Baseline her. Schreiben Sie die zentrierte Schätzung als $\frac{1}{n}\sum_{i=1}^n \big(R(\tau_i) - \bar{R}\big)\, \mathbf{S}_i$. Dabei ist $\mathbf{S}_i$ die Summe der Scores der Trajektorie $i$ und $\bar{R}$ der mittlere Ertrag des Batches. Zerlegen Sie $\mathbb{E}[\bar{R}\, \mathbf{S}_i]$ in seine $n$ Terme und nutzen Sie die Unabhängigkeit der Trajektorien. Welche Terme haben einen von null verschiedenen Erwartungswert?
    1. Die zentrierte Variante von `train` zieht stattdessen bei jedem Schritt den Mittelwert aller künftigen Erträge im Batch ab. Erweitern Sie die Aufzählung aller Fälle auf diese Gewichtung. Für jede Trajektorie des Problems mit zwei Zuständen gilt $\hat{G}_0 = r_0 + r_1$ und $\hat{G}_1 = r_1$, da die Aufzählung ohne Diskontierung auskommt. Zeile $s$ von `S[i]` ist der Score des Schritts im Zustand $s$. Berechnen Sie den Erwartungswert der Schätzung für $n \in \{2, 4, 8\}$ und vergleichen Sie ihn anhand des Normverhältnisses und des Kosinus mit `g`.
    1. Erklären Sie den Vergleich für jeden der beiden Schritte. Bei welchen künftigen Erträgen im Batchmittelwert hat das Produkt mit dem Score des jeweiligen Schritts einen von null verschiedenen Erwartungswert?
   :exlabel:`ex_baselines-batch-centering`

<!--en:9f9c3ceb-->
1. **Die $\sigma$-normalisierte Zielfunktion.** ● GRPO teilt die zentrierten Belohnungen jeder Gruppe durch ihre Standardabweichung $\sigma$. Die Folgen dieser Wahl untersuchen :citet:`Liu.Chen.Li.ea.2025`. Betrachten Sie einen Prompt $x$ mit binärer Belohnung, $r = 1$ für eine richtige Antwort und $r = 0$ andernfalls. Die aktuelle Strategie antwortet mit Wahrscheinlichkeit $p_x$ richtig. Wählen Sie die Gruppe so groß, dass $\bar{R}$ und $\sigma$ ihren Werten in der Grundgesamtheit entsprechen, also $p_x$ und $\sqrt{p_x (1 - p_x)}$.
    1. Berechnen Sie $\sigma$ für $p_x = 0.5$ und für $p_x = 0.9$. Um welchen Faktor verstärkt die Division die Aktualisierung des zweiten Prompts relativ zum ersten?
    1. Sei $y$ die gezogene Antwort. Zeigen Sie, dass gilt:

       $$\mathbb{E}\big[(r - p_x)\, \nabla_\theta \log \pi_\theta(y \mid x)\big] = \nabla_\theta\, p_x.$$

       Folgern Sie, dass die $\sigma$-normalisierte Aktualisierung einem Gradientenaufstieg für $\sum_x h(p_x)$ mit einer Funktion $h$ mit $h'(p) = 1/\sqrt{p(1-p)}$ entspricht. Bestimmen Sie $h$.
    1. Vergleichen Sie $h$ mit der Zielfunktion $\sum_x p_x$ der unnormalisierten Aktualisierung. Welche Prompts erhalten relativ dazu mehr Gewicht? Ist diese Gewichtung in der Praxis erwünscht, wenn die Strategie die meisten Prompts eines Trainingssatzes entweder fast immer oder fast nie löst?
    1. Bei einer endlichen Gruppe von $K$ Antworten trägt ein Prompt nichts zum Gradienten bei, wenn alle $K$ Belohnungen übereinstimmen. Berechnen Sie die Wahrscheinlichkeit dieses Ereignisses für $p_x = 0.9$ und $K = 8$. Erklären Sie, wie das die Verstärkung von Prompts mit kleinem $\sigma$ begrenzt.
   :exlabel:`ex_baselines-sigma-objective`

<!--en:078414eb-->
1. [code] **Rangfolge der Varianzen für verschiedene Strategien.** ● Der Vergleich bei fester Strategie in :numref:`subsec_five_estimators` untersucht eine einzige Strategie, nämlich `probe` nach sechzehn Aktualisierungen mit $J(\theta) = 0.313$. Wiederholen Sie den Vergleich für die Strategien nach $k \in \{0, 8, 32, 64\}$ Aktualisierungen der Teststrategie. Ändern Sie dazu die Anzahl der Iterationen der Schleife, die `probe` trainiert, und berechnen Sie `theta_probe`, `pi_np`, `g_exact` und `V_pi` neu, bevor Sie die $2{,}000$ Batches ziehen.
    1. Tabellieren Sie das Rauschen der fünf Schätzer in Abhängigkeit von $k$. Gilt die Rangfolge nach sechzehn Aktualisierungen für jede Strategie? Bei welcher Strategie hilft Zentrierung nicht mehr, und warum?
    1. Wie verhalten sich die beiden Schätzer ohne Baseline, wenn sich $J(\theta)$ seinem Maximum $\gamma^5 \approx 0.77$ nähert, und warum?
    1. Zählen Sie bei $k = 0$ die Batches ohne Erfolg und erklären Sie, welchen Beitrag ein solcher Batch zu jedem der fünf Schätzer leistet.
   :exlabel:`ex_baselines-variance-ordering`

<!--en:c3fb6458-->
1. [code] **Einfluss der Schrittweite auf den Vergleich.** ● Bei gemeinsamer Schrittweite erreichte die normalisierte Variante von `train` die Erfolgsquote von 90% mit den wenigsten Aktualisierungen. Sie machte aber auch die längsten Schritte. Für einen Batch mit $\sigma > 0$ liefern die normalisierten Gewichte aus :eqref:`eq_pg_normalized` dieselbe SGD-Aktualisierung wie die zentrierten Gewichte bei Schrittweite $\eta / (\sigma + 10^{-8})$.
    1. Protokollieren Sie $\sigma$ für jeden Batch der normalisierten Variante. Tragen Sie den Median über die Zufallsstartwerte von $1/(\sigma + 10^{-8})$ gegen den Index der Aktualisierung auf und lassen Sie Batches ohne Erfolg aus. Wie ändert sich dieser Multiplikator der Schrittweite, während sich die Strategie verbessert? Was bedeutet das für den Versuch, die beiden Varianten anzugleichen, indem Sie für die normalisierte Variante $\eta$ durch den über den Lauf gemittelten Wert von $1/(\sigma + 10^{-8})$ teilen?
    1. Wiederholen Sie den Vergleich von künftigem Ertrag, Wert-Baseline und exakter Baseline bei $\eta \in \{4, 8, 32\}$. Tragen Sie für jede Variante gegen $\eta$ sowohl die mediane Anzahl der Aktualisierungen bis zu einer Erfolgsquote von 90% als auch den Anteil der Läufe auf, die diese Quote nie erreichen. Welche Schrittweite ist für jede Variante am besten? Erklären Sie, warum die beste Schrittweite bei geringerem Rauschen des Schätzers wächst.
    1. Gleichen Sie stattdessen die Schrittlängen an, erhält eine stärker verrauschte Variante ein kleineres $\eta$. Wählen Sie für jede der fünf Varianten von `shared` eine Schrittweite, bei der ihre mittlere Schrittlänge über die ersten 40 Aktualisierungen etwa $0.6$ beträgt. Protokollieren Sie die mittlere Schrittlänge für jeden Lauf getrennt vor und nach dem Erreichen von 90%. Welche Varianten machen während des Lernens längere Schritte? Warum vermischt ein Vergleich bei gleicher mittlerer Schrittlänge immer noch Varianz und Schrittweite?
   :exlabel:`ex_baselines-step-size-confound-quantified`

<!--en:6bc444ef-->
1. [code] **Optimale zustandsabhängige Baseline.** ●● In Aufgabe :exref:`ex_baselines-optimal-constant-baseline` haben Sie für einen Zustand die konstante Baseline mit minimaler Varianz hergeleitet. Wenden Sie das Ergebnis auf jeden Zustand einzeln an, erhalten Sie die optimale Baseline-Funktion von :citet:`Greensmith.Bartlett.Baxter.2004`. Sie minimiert die Varianz des Beitrags eines einzelnen Schritts. In dieser Aufgabe berechnen Sie diese Funktion für die tabellarische Softmax-Strategie und untersuchen, wann sie auch für den episodischen Schätzer `estimate` optimal ist.
    1. Zeigen Sie anhand des Softmax-Scores, dass gilt:

       $$\|\nabla_\theta \log \pi_\theta(a \mid s)\|^2 = 1 - 2\pi_\theta(a \mid s) + \sum_{a'} \pi_\theta(a' \mid s)^2.$$

       Nutzen Sie $\mathbb{E}[\hat{G}_t \mid s_t = s, a_t = a] = Q^{\pi_\theta}(s, a)$, um die optimale Baseline $b^*(s)$ durch $\pi_\theta(\cdot \mid s)$ und $Q^{\pi_\theta}(s, \cdot)$ auszudrücken.
    1. Die Schätzung für eine Episode summiert die gewichteten Scores aller ihrer Schritte. Zeigen Sie, dass bei tabellarischer Parametrisierung die Scores zweier Schritte in unterschiedlichen Zuständen orthogonal sind. Damit zerfällt die quadrierte Norm der Schätzung in einen Term pro Schritt sowie Kreuzterme für wiederholte Besuche desselben Zustands. Unter welcher Bedingung an die Episoden minimiert $b^*(s)$ die Spur der Kovarianzmatrix von `estimate` exakt? Erfüllt FrozenLake diese Bedingung?
    1. Berechnen Sie $b^*(s)$ für `probe`, wobei $Q^{\pi_\theta}$ durch `mdp.backup(V_pi)` gegeben ist. Fügen Sie die Gewichtung `G - b_star[b.obs]` zu `weightings` in :numref:`subsec_five_estimators` als `lambda b, G, V: G - b_star[b.obs]` hinzu. Vergleichen Sie ihr Rauschen mit dem der exakten Baseline, sowohl bei `probe` als auch bei der Strategie nach 64 Aktualisierungen der Teststrategie.
    1. In welchen Zuständen weicht $b^*(s)$ am stärksten von $V^{\pi_\theta}(s)$ ab? Erklären Sie anhand der Gewichte $\|\nabla_\theta \log \pi_\theta(a \mid s)\|^2$, weshalb sich $b^*(s)$ bei einem Zustand mit nahezu deterministischer Aktionswahl den Aktionswerten der selten gewählten Aktionen nähert. Setzen Sie die Differenz der beiden Baselines in Beziehung zur zusätzlichen Varianz aus Aufgabe :exref:`ex_baselines-optimal-constant-baseline`a.
   :exlabel:`ex_baselines-optimal-state-baseline`

<!--en:ad2110f7-->
1. [code] **Der künftige Ertrag im Einzelnen.** ● Gegenüber dem Ertrag der Trajektorie ändert die Gewichtung mit dem künftigen Ertrag aus :eqref:`eq_rtg` zweierlei. Sie lässt für jede Aktion die zuvor erhaltenen Belohnungen weg, was nach der bedingten Identität zulässig ist. Sie lässt aber auch den Faktor $\gamma^t$ weg, was die Identität nicht rechtfertigt.
    1. Sei $\rho(s) = \mathbb{E}\big[\sum_t \mathbf{1}(s_t = s)\big]$ die erwartete Anzahl der Besuche in $s$ während einer Episode. Wie in Aufgabe :exref:`ex_policy-gradient-discount-that-implementations-drop`a ist der Mittelwert von :eqref:`eq_rtg` gleich $\sum_s \rho(s) \sum_a \nabla_\theta \pi_\theta(a \mid s)\,
       Q^{\pi_\theta}(s, a)$, während der Policy-Gradient die Zustände mit $\sum_t \gamma^t \Pr(s_t = s)$ gewichtet. Für welches $\gamma$ stimmen beide überein?
    1. Berechnen Sie bei `probe` die Größe $\rho$, indem Sie $\rho = \mathbf{e}_{s_0} + P_\pi^\top \rho$ für die elf Felder lösen, die weder Löcher noch Ziel sind. Entnehmen Sie $Q^{\pi_\theta}$ aus `mdp.backup(V_pi)` und werten Sie den Mittelwert von :eqref:`eq_rtg` exakt aus. Vergleichen Sie den Kosinus mit `g_exact` und die Länge relativ zu `g_exact` mit den Angaben für den künftigen Ertrag in der Tabelle für die feste Strategie.
    1. Wenn Sie frühere Belohnungen weglassen, entfernen Sie aus der Schätzung einen Term $Y$ mit Mittelwert null. Die ursprüngliche Schätzung sei $X + Y$. Zeigen Sie, dass $\textrm{Var}(X) < \textrm{Var}(X + Y)$ genau dann gilt, wenn $\textrm{Var}(Y) + 2\,\textrm{Cov}(X, Y) > 0$. Wiederholen Sie den Vergleich der Zeilen für den Ertrag und die kausale Gewichtung mit Kosten von $0.01$ pro Zug statt $0.1$ sowie bei der Strategie nach 64 Aktualisierungen der Teststrategie. Verringert die Berücksichtigung der Kausalität das Rauschen immer? Erklären Sie jeden Fall anhand des Vorzeichens der Kovarianz zwischen früheren Belohnungen und künftigem Ertrag.
   :exlabel:`ex_baselines-reward-to-go-taken-apart`

<!--en:ebf297c4-->
[Diskussionen](https://d2l.discourse.group/t/36799)

<!--en:dffe2f88-->
Baselines und Varianzreduktion<br>**Score mit Mittelwert null · künftiger Ertrag · Baselines als Kontrollvariaten · Vorteil und gelerntes V · Gruppennormalisierung bei RLOO und GRPO**

<!--en:533ae521-->
::: {.slide title="Warum die Gewichtung mit dem vollen Ertrag so stark streut"}

<!--en:b0445376-->
[Motivation und Grundidee]{.kicker}

<!--en:745d5632-->
- **Alles-oder-nichts-Gewichte:** Nichtnegative Erträge $R(\tau) \in \{0, \gamma^{T-1}\}$ machen unnötige Schleifen in erfolgreichen Episoden *wahrscheinlicher*. Schritte in ein Loch erhalten Gewicht $0$ und werden nie *unwahrscheinlicher*.

<!--en:e4bc493e-->
- **Lösung mit Kontrollvariate:** Nach Abzug von $b(s_t)$ werden überdurchschnittliche Aktionen wahrscheinlicher und unterdurchschnittliche unwahrscheinlicher. Dabei bleibt $\mathbb{E}[\hat{u}]$ erhalten.

<!--en:97f9f7a7-->
::: {.slide title="Score mit Mittelwert null und bedingte Identität"}

<!--en:389cf276-->
[Warum eine Baseline keine Verzerrung erzeugt]{.kicker}

<!--en:d9be8a31-->
- **Bedingte Identität für den bisherigen Verlauf:** Für jede Größe $c(s_{0:t}, a_{0:t-1})$, die vor dem Ziehen von $a_t$ feststeht, gilt $\mathbb{E}[c\,\nabla_\theta \log \pi_\theta(a_t \mid s_t)] = \mathbf{0}$.

<!--en:5cff622d-->
- **Zulässigkeit:** Jede durch den bisherigen Verlauf bestimmte Größe darf ohne Verzerrung abgezogen werden. Eine Abhängigkeit von $a_t$ oder $s_{t+1}$ verzerrt den Erwartungswert.

<!--en:b704994c-->
::: {.slide title="Kausalität und künftiger Ertrag"}

<!--en:644fd8f5-->
[Frühere Belohnungen im Rückwärtsdurchlauf weglassen]{.kicker}

<!--en:23971d1c-->
Entfällt der Anfang $\sum_{t'<t} \gamma^{t'} r_{t'}$ von $R(\tau)$ (sein Produkt mit dem Score in Schritt $t$ hat Erwartungswert null), bleibt der **künftige Ertrag**:

<!--en:238141df-->
- **Übliche Konvention gegenüber dem exakten Gewicht $\gamma^t\hat{G}_t$:** Meist erhält Schritt $t$ das Gewicht $\hat{G}_t$ statt $\gamma^t\hat{G}_t$. Späte Besuche zählen damit so viel wie frühe. Bei der festgehaltenen Teststrategie dreht sich der Mittelwert um $\approx 8^\circ$.

<!--en:d58103e5-->
::: {.slide title="Baselines als statistische Kontrollvariaten"}

<!--en:9dcf8809-->
[Varianz des Schätzers minimieren]{.kicker}

<!--en:ee75705f-->
Wird bei der Monte-Carlo-Schätzung von $\mathbb{E}[X]$ eine korrelierte Größe $c\,(Y - \mathbb{E}[Y])$ mit Mittelwert null abgezogen, bleibt der Mittelwert erhalten. Die Varianz lässt sich verringern und beschreibt als Funktion von $c$ eine Parabel:

<!--en:0ded1da5-->
- **Optimaler Koeffizient $c^* = \textrm{Cov}(X, Y)/\textrm{Var}(Y)$:** Er senkt die Varianz auf $(1 - \textrm{corr}^2)\,\textrm{Var}(X)$ (Reduktion um den Faktor $5\times$ bei $\textrm{corr} = 0.9$).

<!--en:88f8c300-->
- **Anwendung auf Policy-Gradienten:** Mit $X = \hat{G}_t\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ und $Y = \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ (wobei $\mathbb{E}[Y \mid s_t] = \mathbf{0}$ gilt) ergibt sich der Schätzer mit zustandsabhängiger Baseline:
  $$\hat{u} = \frac{1}{n}\sum_{i=1}^n \sum_{t=0}^{T_i-1} \big(\hat{G}_t^i - b(s_t^i)\big)\,\nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i)$$

<!--en:67492c16-->
::: {.slide title="Varianzreduktion bei einer Aufgabe mit einem Schritt"}

<!--en:4e7c7a19-->
[Kausalität und Varianzparabel]{.kicker}

<!--en:675ba320-->
Beim Gauß-Problem mit einem Schritt, $a \sim \mathcal{N}(\mu, 1)$ bei $\mu = 0$ und Belohnung $R(a) = 0.4 + 2e^{-(a-2)^2/2}$ ($\mathbb{E}[R] = 0.92$, exakter Gradient $0.52$), verringert eine konstante Baseline $b$ die Streuung des Schätzers erheblich:

<!--en:22c29961-->
- **Varianzreduktion um mehr als den Faktor $4\times$:** Ohne Baseline hat $R(a)\,a$ die Standardabweichung $1.34$. Das Abziehen der mittleren Belohnung $b = 0.92$ halbiert die Streuung auf $0.69$. Beim scoregewichteten Optimum $b^* = \mathbb{E}[R(a)a^2]/\mathbb{E}[a^2] = 1.18$ bleiben nur $23\%$ der ursprünglichen Varianz.

<!--en:deb2152a-->
::: {.slide title="Zustandswerte und Vorteilsfunktion"}

<!--en:766c7fa8-->
[Eine zustandsabhängige Baseline lernen]{.kicker}

<!--en:0d30e2f8-->
- **Vorteilsschätzung aus einem Sample $\hat{G}_t - V^{\pi_\theta}(s_t)$:** Das Abziehen von $V^{\pi_\theta}(s_t)$ zentriert die Gewichte in jedem Zustand um null. Aktionen mit $Q^\pi(s_t, a_t) > V^\pi(s_t)$ werden wahrscheinlicher, schlechtere Aktionen unwahrscheinlicher.

<!--en:2269b0a4-->
- **Online gelernte Wertetabelle als Baseline:** Nach jedem Batch gilt für $\hat{V}(s)$ die Aktualisierung $\hat{V}(s_t) \leftarrow \hat{V}(s_t) + \eta_V(\hat{G}_t - \hat{V}(s_t))$.

<!--en:be258995-->
::: {.slide title="Batchzentrierung und Skalierung mit der Standardabweichung"}

<!--en:4ce1d0b2-->
[Baseline abziehen und Schrittweite skalieren]{.kicker}

<!--en:6b61fcc6-->
In der Praxis standardisiert man oft alle Gewichte aus künftigen Erträgen innerhalb eines Batches mit $\tilde{G}_t^i = (\hat{G}_t^i - \bar{R})/(\sigma + 10^{-8})$:

<!--en:8c5d4127-->
- **$\bar{R}$ als Baseline abziehen:** Die relativen Gewichte verschieben sich. Unterdurchschnittliche Trajektorien erhalten negative Gewichte. Da jede Trajektorie selbst zum Mittel beiträgt, verkürzt sich der erwartete Gradient auf das $(n-1)/n$-Fache.

<!--en:722de56f-->
- **Division durch $\sigma$ passt die Schrittweite an:** Sie multipliziert den gesamten Batchgradienten mit $1/\sigma$. Auf FrozenLake gilt wegen $\hat{G}_t \in [0, 1]$ auch $\sigma \leq 0.5$. Die Division durch $\sigma$ verlängert die SGD-Schritte dort um den Faktor $\approx 3\times$.

<!--en:806f5f6d-->
::: {.slide title="Leave-one-out (RLOO) beseitigt die Verzerrung durch Zentrierung"}

<!--en:a3d299be-->
[Erwartungstreue Batchzentrierung]{.kicker}

<!--en:0b38c952-->
Der Batchmittelwert $\bar{R} = \frac{1}{n}\sum_{j=1}^n R_j$ enthält für Trajektorie $i$ deren eigenen Ertrag $R_i$. Einfache Zentrierung verkleinert daher den erwarteten Gradienten auf das $(n-1)/n$-Fache. **Leave-one-out (RLOO)** (Kool et al., 2019, Ahmadian et al., 2024) verwendet den Mittelwert der *anderen* $n-1$ Trajektorien:

<!--en:b629872c-->
- **Erwartungstreue für jedes $n \geq 2$:** Da die unabhängig gezogenen Trajektorien $j \neq i$ nicht von den Aktionen in $\tau_i$ abhängen, gilt exakt $\mathbb{E}[b_i \nabla_\theta \log P(\tau_i; \theta)] = \mathbf{0}$.

<!--en:c95604af-->
::: {.slide title="Group Relative Policy Optimization (GRPO)"}

<!--en:9ac9a8b7-->
[Baselines ohne Kritiker für Sprachmodelle]{.kicker}

<!--en:ad635d2f-->
Beim Post-Training großer Sprachmodelle hat jeder Prompt $x$ eine eigene Schwierigkeit $p_x$. **GRPO** (Shao et al., 2024) zieht daher eine Gruppe von $K$ Antworten auf Prompt $x$ und verwendet Mittelwert und Streuung der Gruppe zur Normalisierung:

<!--en:d1035abb-->
- **Warum bei $K = 1$ das Lernen ausbleibt und bei $K \geq 2$ die Obergrenze erreicht wird:** Bei $K = 1$ gilt stets $r_1 - \bar{R}_x = 0$. Für jedes $K \geq 2$ macht die Gruppenzentrierung falsche Antworten unwahrscheinlicher ($-k/K$) und richtige wahrscheinlicher ($1 - k/K$). So erreicht die Strategie ohne Wertenetz die Obergrenze $0.500$.

<!--en:e808deff-->
::: {.slide title="Wie der Divisor des Verlusts Richtung und Länge beeinflusst"}

<!--en:f90d4612-->
[Episodenzahl, Gesamtzahl der Schritte und einzelne Episodenlänge]{.kicker}

<!--en:ce0bf540-->
Bei der Summe gewichteter Scores über Episoden unterschiedlicher Länge $T_i$ (insgesamt $N = \sum_{i=1}^n T_i$ Schritte) wirken sich vier übliche Divisoren unterschiedlich aus:

<!--en:27adba86-->
- **Drei Gradienten parallel, einer mit anderer Richtung:** Die Division durch die Episodenzahl $n$, die Gesamtzahl der Schritte $N$ (DAPO) oder eine feste Konstante (Dr. GRPO) ergibt parallele Gradienten (`cos = 1.000`) verschiedener Länge. Die Division jeder Episode durch ihre eigene Länge $T_i$ (GRPO) gewichtet kurze und lange Episoden um und ändert so die Richtung (`cos = 0.937`).

<!--en:195ffe22-->
::: {.slide title="Sechs Schätzer bei fester Strategie im Vergleich" layout="tall-output"}

<!--en:e9bf0272-->
[Länge, Richtung und relative Varianz]{.kicker}

<!--en:3f11e499-->
Anhand von $2{,}000$ Batches mit je vier Episoden lassen sich bei fester Teststrategie ($J = 0.313$, exakter Gradient $\nabla_\theta J$ und exakter Wert $V^{\pi_\theta}$ aus dem linearen Gleichungssystem) die Verzerrung und das relative Rauschen $\textrm{tr}(\textrm{Cov})/\|\textrm{mean}\|^2$ vergleichen:

<!--en:a0224a51-->
::: {.slide title="Die Varianztabelle bei fester Strategie lesen"}

<!--en:be19ce8b-->
[Welche Gewichte das Rauschen senken und wann Kausalität hilft]{.kicker}

<!--en:597cdbfe-->
- **Die Zustandswert-Baseline streut am wenigsten:** Das Abziehen von $V^{\pi_\theta}(s_t)$ senkt das relative Rauschen auf weniger als die Hälfte (von etwa $12$ auf unter $6$). Länge ($1.38$) und Richtung (`cos = 0.990`) des Mittelwerts bleiben erhalten.

<!--en:84b2f4b0-->
- **Zentrierung und Normalisierung:** Batchzentrierung senkt das Rauschen um $\approx 40\%$ und verkürzt den Mittelwert auf das $(n-1)/n$-Fache. Die Division durch $\sigma$ erhält das relative Rauschen ($7.5$) und verdreifacht die Schrittlänge ($3.28$).

<!--en:dbf80a07-->
- **Frühere Belohnungen weglassen:** Bei einer zusätzlichen Belohnung von $-0.1$ pro Schritt senkt die kausale Gewichtung das Rauschen um mehr als den **Faktor $3\times$** ($19.5 \to 5.8$).

<!--en:18f18cd3-->
::: {.slide title="Einfluss von Schrittweite und Varianz im Training trennen" layout="tall-output"}

<!--en:81050056-->
[Gemeinsame kleine Schrittweite und große Schrittweite]{.kicker}

<!--en:4cdab83a-->
Der Vergleich aller sechs Schätzer mit einfachem SGD bei $\eta = 2$ (20 Zufallsstartwerte) mit den drei Schätzern mit gleichem Mittelwert bei $\eta = 16$ (100 Zufallsstartwerte) zeigt den Einfluss der Schrittweite:

<!--en:b5fe7f6d-->
- **Bei $\eta = 2$ gewinnt die Normalisierung allein durch $3\times$ längere Schritte:** Sie erreicht $90\%$ Erfolg im Median nach $32.0$ Aktualisierungen. Ihre Schrittnorm beträgt $0.93$ gegenüber $0.32$ bei Zentrierung und $0.53$ beim künftigen Ertrag.

<!--en:2d24c3d6-->
::: {.slide title="Geringere Varianz ermöglicht größere stabile Schrittweiten"}

<!--en:bc10bc9c-->
[Warum die Wert-Baseline bei eta = 16 gewinnt]{.kicker}

<!--en:707cae07-->
Der erwartete Aktualisierungsvektor bleibt für `reward-to-go`, `value baseline` (gelerntes $\hat{V}$) und `exact baseline` ($V^{\pi_\theta}$) bei $\eta = 16$ gleich. Damit wird der Effekt der Varianzreduktion sichtbar:

<!--en:e9247ab1-->
- **Schnellere Konvergenz ohne Stillstand:** Bei $\eta = 16$ erreicht `exact baseline` eine Erfolgsquote von $90\%$ im Median nach **etwa $20$ Aktualisierungen bei $0/100$ festgefahrenen Läufen**.

<!--en:b53ca352-->
- **Vollständiger Stillstand ohne Baseline:** `reward-to-go` verfehlt $90\%$ in **etwa jedem siebten Lauf**. Ein großer verrauschter Schritt legt die Strategie auf eine Schleife mit Ertrag null fest. Alle künftigen Gewichte sind dann $0$.

<!--en:d1c04235-->
::: {.slide title="Einbruch der Strategieentropie bei großen Schrittweiten"}

<!--en:3c2cabd2-->
[Stillstand anhand der Entropie im Batch erkennen]{.kicker}

<!--en:a228dc04-->
Die über die besuchten Zustände gemittelte Shannon-Entropie $\mathcal{H}(\pi_\theta(\cdot \mid s)) = -\sum_a \pi_\theta(a \mid s)\log\pi_\theta(a \mid s)$ zeigt, wie schnell große Schritte die Strategie deterministisch machen:

<!--en:eab0e9de-->
- **Von der Gleichverteilung ($\ln 4 \approx 1.39$ Nat) zur deterministischen Strategie ($< 0.1$ Nat):** Bei $\eta = 16$ konzentrieren sich beide Varianten innerhalb von 20 bis 40 Aktualisierungen auf eine nahezu deterministische Trajektorie.

<!--en:014739af-->
- **Warum festgefahrene Läufe stecken bleiben:** Ein festgefahrener Lauf mit `reward-to-go` ist ebenso deterministisch ($\approx 0$ Nat) wie ein erfolgreicher. Er wiederholt denselben Weg in ein Loch, bei Entropie und Gradient null.

<!--en:f4fdc111-->
::: {.slide title="Verständnisfragen zu Baselines und Varianzreduktion"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:90f1de82-->
Fragen zu zulässigen Baselines, RLOO und dem Einfluss der Schrittweite:

<!--en:44a4199d-->
1. **Zulässige Baselines:** Welche Größen lassen sich von $\hat{G}_t$ in Schritt $t$ abziehen, ohne den erwarteten Gradienten zu verzerren? (a) Der Schrittindex $t$, (b) die verbleibende Episodenlänge $T - t$ oder (c) ein geschätzter Aktionswert $\hat{Q}(s_t, a_t)$?

<!--en:c4007f8b-->
2. **Zentrierung und RLOO:** Ein Batch enthält $n = 4$ Episoden. Eine erreicht das Ziel ($R_1 = 0.77$), drei enden in Löchern ($R_2 = R_3 = R_4 = 0$). Welche Gewichte erhalten die erfolgreichen und die gescheiterten Episoden bei Batchzentrierung ($R_i - \bar{R}$) und bei RLOO ($R_i - b_i$)?

<!--en:8f12cea0-->
3. **Normalisierung beurteilen:** Warum wäre es bei $\eta = 2$ falsch, allein deshalb auf eine geringere Varianz des Gradientenschätzers `normalized` gegenüber `value baseline` zu schließen, weil er die Erfolgsquote $90\%$ mit weniger Aktualisierungen erreicht?

<!--en:557b98b6-->
**Antworten:**

<!--en:13d43b8e-->
1. Nur **(a), der Schrittindex $t$**, ist zulässig, weil er vor dem Ziehen von $a_t$ feststeht. Sowohl $T - t$ als auch $\hat{Q}(s_t, a_t)$ hängen von der gezogenen Aktion $a_t$ ab und verzerren den Erwartungswert.

<!--en:ae3a026b-->
2. Bei $\bar{R} = 0.77/4 = 0.1925$ erhält die erfolgreiche Episode durch Batchzentrierung $+0.5775$, jede gescheiterte $-0.1925$. RLOO skaliert diese Gewichte mit $\frac{n}{n-1} = \frac{4}{3}$. Damit erhält die erfolgreiche Episode $0.77 - 0 = +0.77$ und jede gescheiterte $0 - 0.77/3 = -0.2567$.

<!--en:3122838b-->
3. Die Division durch $\sigma \leq 0.5$ multipliziert die effektive SGD-Lernrate mit $\approx 3\times$. Bei der festen Teststrategie hat `normalized` sogar ein höheres relatives Rauschen ($7.5$) als `value baseline` ($5.7$).

<!--en:ccb1ad59-->
::: {.slide title="Zusammenfassung zur Varianzreduktion bei Policy-Gradienten"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:cc748a01-->
| Schätzer / Verfahren | Gewichtungsformel $w_t^i$ | Erwartungswert und Varianz |

<!--en:4b0e8845-->
| **Ertrag der Trajektorie** | $R(\tau_i) = \sum_{t'=0}^{T_i-1} \gamma^{t'} r_{t'}^i$ | Erwartungstreu ($\nabla_\theta J$), höchste Varianz über die ganze Trajektorie |

<!--en:d3fdac4b-->
| **Kausaler künftiger Ertrag** | $\gamma^t \hat{G}_t^i$ mit $\hat{G}_t^i = \sum_{t' \geq t} \gamma^{t'-t} r_{t'}^i$ | Erwartungstreu ($\nabla_\theta J$), frühere Belohnungen $r_{<t}$ entfallen nach dem Lemma zum Mittelwert null |

<!--en:84727a94-->
| **Zustandswert-Baseline** | $\hat{G}_t^i - V(s_t^i)$ (Vorteilsschätzung $A_t^i$) | Erhält den Erwartungswert, senkt die relative Varianz auf dem See um $\approx 50\%$ |

<!--en:6a318c33-->
| **Leave-one-out (RLOO)** | $R(\tau_i) - \frac{1}{n-1}\sum_{j \neq i} R(\tau_j) = \frac{n}{n-1}(R_i - \bar{R})$ | Erwartungstreue Zentrierung pro Batch oder Gruppe ohne den verkleinernden Faktor $(n-1)/n$ |

<!--en:bf628a1e-->
| **GRPO-Gruppengewicht** | $(r(x, y_j) - \bar{R}_x)/(\sigma_x + \varepsilon)$ über $K \geq 2$ Ziehungen | Baseline pro Prompt ohne Kritiker und adaptive Schrittskalierung mit $\sigma_x^{-1}$ |
