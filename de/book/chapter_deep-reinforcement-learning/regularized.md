<!--en:fc3d36c5-->
# KL-regularisierte Strategieoptimierung

<!--en:d0beca46-->
Alle bisherigen Verfahren in :numref:`chap_reinforcement_learning` und in diesem Kapitel setzen voraus, dass jemand die Belohnung (engl. reward) vorgegeben hat. Bei einem zugefrorenen See oder einem Wagen mit einer Stange ist das leicht: Das Ziel soll erreicht werden, die Stange soll aufrecht bleiben. Bei den meisten Aufgaben, die Menschen wichtig sind, fällt das schwer. Niemand kann eine Formel angeben, die eine hilfreiche Antwort, einen sicheren Spurwechsel oder einen gut lesbaren Beweis bewertet. Menschen können aber zwei Versuche vergleichen und angeben, welchen sie bevorzugen. An solche Vergleiche lässt sich eine Belohnungsfunktion anpassen. So lernten die ersten weitverbreiteten Chatmodelle, hilfreiche Antworten zu geben. Zunächst passte man ein Belohnungsmodell (engl. reward model) an menschliche Vergleiche an und trainierte anschließend eine Strategie (engl. policy) anhand dieses Modells :cite:`Christiano.Leike.Brown.ea.2017,ouyang2022training`.

<!--en:319e913c-->
Eine geschätzte Belohnung wirft eine Frage auf, die sich bei einer vorgegebenen nie stellte. Das Modell ist nur dort genau, wo die Vergleichsdaten Aufschluss geben. Anderswo muss es raten, während ein Optimierer überall sucht. Findet der Optimierer einen Bereich, den die Vergleiche nie erfasst haben und in dem die geschätzte Belohnung zufällig hoch ist, weicht er dorthin aus. Die tatsächliche Qualität sinkt. In der Praxis hat sich ein Gegenmittel bewährt. Entfernt sich die Strategie von derjenigen, die die Trajektorien für die Vergleiche erzeugt hat, kostet sie das etwas: Ein Strafterm erfasst die KL-Divergenz von dieser Referenzstrategie. Die Trust-Region aus :numref:`sec_ppo` begrenzte bei jedem Schritt die Abweichung von der vorherigen Strategie. Der Strafterm hier geht in die Zielfunktion ein. Damit ändert sich neben dem Lösungsweg auch die Lösung selbst. Sie lässt sich in geschlossener Form angeben: Wir gewichten die Referenzverteilung mit der Exponentialfunktion der Belohnung neu.

<!--en:1cf39c2d-->
Zunächst bauen wir eine Gitterwelt auf, deren wahre Belohnung wir kennen. Wir schätzen die Belohnung anhand von Vergleichen und beobachten, wie ein Optimierer die Lücken ausnutzt. Dann lösen wir das Problem mit Strafterm für eine einzelne Entscheidung und erweitern die Lösung auf mehrere Schritte. Wir verfolgen, wie die tatsächliche Qualität zunächst steigt und dann sinkt, wenn sich die Strategie von ihrer Referenz entfernt.

<!--en:148cb1bb-->
## Lernen einer Belohnung aus Vergleichen

<!--en:ab86a2ac-->
Um zu sehen, was schiefgeht, brauchen wir eine Aufgabe mit bekannter wahrer Belohnung. So können wir die Belohnung vor dem lernenden Agenten verbergen und trotzdem das Ergebnis bewerten. Eine kleine Gitterwelt bietet uns eine solche Aufgabe.

<!--en:9bc01399-->
### Die Gitterwelt und ihre Referenzstrategie

<!--en:e6484a88-->
Die Welt besteht aus einem $4 \times 4$-Gitter mit deterministischen Zügen, wie :numref:`fig_rl_reward_gridworld` zeigt. Eine Episode beginnt im Feld oben links und endet beim Eintritt ins Ziel unten rechts. Dafür gibt es $+1$. Ein gewöhnlicher Zug kostet $0.04$. Ein Zug in eines der beiden benachbarten Gefahrenfelder in der zweiten Zeile, den *Gefahrenstreifen*, kostet $1.5$. Die wahre Belohnung $r^*(s, a)$ richtet sich nach dem Feld, in das die Aktion $a$ vom Zustand $s$ aus führt. Die Felder sind zeilenweise nummeriert. Der Start ist Zustand $0$, die Gefahrenfelder sind die Zustände $5$ und $6$, und das Ziel ist Zustand $15$. Die Aktionen $0$ bis $3$ führen nach links, unten, rechts und oben. Bei einem Zug gegen die Wand bleibt der Agent auf seinem Feld.

<!--en:4b24691b-->
![Die Gitterwelt dieses Abschnitts. (a) Episoden beginnen in Zustand 0 (S) und enden im Ziel, Zustand 15 (G, $+1$). Ein Zug in eines der Gefahrenfelder 5 oder 6 kostet $1.5$, jeder gewöhnliche Zug $0.04$. In jedem Feld steht, in wie vielen der 1000 Referenztrajektorien es besucht wird. (b) Datenabdeckung für jedes Zustand-Aktion-Paar. Die vier Dreiecke jedes Felds stehen für die Züge nach links, unten, rechts und oben. Ihre Schattierung gibt auf einer logarithmischen Skala an, wie viele Trajektorien das Paar enthalten. Ein Kreuz kennzeichnet ein Paar, das nie vorkommt. Die acht Züge in den Gefahrenstreifen sind umrandet.](../img/mdl-rl-reward-gridworld.svg)

<!--en:ba42d842-->
Für die Vergleiche verwenden wir Trajektorien einer festen Strategie, der *Referenzstrategie* (engl. reference policy) $\pi_{\textrm{ref}}$. Sie soll sich auskennen, aber zögerlich handeln. Dazu wenden wir die Softmax-Funktion bei der Temperatur $0.25$ auf die optimalen Aktionswerte an: $\pi_{\textrm{ref}}(a \mid s) \propto e^{Q^*(s, a)/0.25}$. Am Start haben die beiden Züge in Richtung Ziel den Wert $Q^* = 0.59$, die beiden Züge gegen die Wände den Wert $0.52$. Die Referenzstrategie wählt deshalb jeden guten Zug mit Wahrscheinlichkeit $0.28$ und jeden nutzlosen mit $0.22$. Ein Zug in ein Gefahrenfeld ist etwa $1.5$ weniger wert als die Alternativen. Die Softmax-Funktion übersetzt das in eine Wahrscheinlichkeit von etwa $e^{-6}$, sodass die Referenzstrategie diese Felder selten betritt. Die Codezelle fasst die Arrays $P[s, a, s']$ und $r^*[s, a]$ zu einem `d2l.TabularMDP` zusammen (:numref:`sec_mdp`). Sie berechnet $Q^*$ mit `d2l.value_iteration` und einem weiteren Aufruf von `backup` (:numref:`sec_valueiter`) und bildet daraus die Referenzstrategie.

<!--en:e10c125b-->
### Präferenzen und das Bradley-Terry-Modell

<!--en:327a31ba-->
Ein Vergleich verrät, welche von zwei Trajektorien eine Person bevorzugt. Wie groß der Unterschied ist, erfahren wir dabei nicht. Das Bradley-Terry-Modell :cite:`Bradley.Terry.1952` beschreibt solche Antworten statistisch. Es weist jeder Trajektorie einen skalaren Score zu, hier ihren Ertrag (engl. return) $r(\tau)$ unter der Belohnung $r$. Den Vergleich behandelt es als verrauschte Beobachtung der Differenz dieser Scores:

<!--en:7e150e45-->
Bei gleichem Ertrag wird also jede der beiden Trajektorien mit Wahrscheinlichkeit 1/2 bevorzugt. Unterscheiden sich die Erträge um eine Einheit, gewinnt die bessere Trajektorie in etwa drei von vier Vergleichen.

<!--en:a29ac1a9-->
In der Gitterwelt hängt der Ertrag linear von der Belohnung ab. Der Vektor $\mathbf{c}(\tau)$ enthalte die diskontierten Besuchszahlen einer Trajektorie mit einem Eintrag pro Zustand-Aktion-Paar. Zum Eintrag $(s, a)$ addieren wir $\gamma^t$ für jeden Schritt $t$, in dem die Trajektorie Aktion $a$ im Zustand $s$ ausführt. Dann gilt $r(\tau) = \mathbf{c}(\tau)^\top r$, wobei wir die Belohnungen als Vektor mit $64$ Einträgen schreiben. Die Anpassung von :eqref:`eq_bradley_terry` ist eine logistische Regression mit den Merkmalen $\mathbf{c}(\tau) - \mathbf{c}(\tau')$ und den Belohnungen als Gewichten. Die Funktion `sample_traj` liefert $\mathbf{c}(\tau)$ für eine Trajektorie von $\pi_{\textrm{ref}}$, die am Ziel oder nach $40$ Schritten endet. In `X` stapeln wir $1000$ solcher Vektoren. Für $6000$ zufällige Paare ziehen wir die Label per Münzwurf mit den Wahrscheinlichkeiten aus :eqref:`eq_bradley_terry` unter $r^*$. Wir schätzen $\hat r$ mittels Gradientenaufstieg auf der Log-Likelihood. Ein schwacher Gewichtszerfall (engl. weight decay) von $0.02$ sorgt dafür, dass die Schätzung auch für Paare mit wenigen Daten wohldefiniert ist. `episodes` zählt für jedes Paar die Episoden, in denen es vorkommt. `core` markiert die Paare, die in mindestens $50$ Episoden vorkommen. `into` enthält die vier Züge, die von oben und von den Seiten in den Gefahrenstreifen führen.

<!--en:4c863eed-->
Die Wurzel des mittleren quadratischen Fehlers (RMS-Fehler) beträgt $0.19$ für die sechsundvierzig Paare in `core`. Neun Paare kommen nie vor: die vier Aktionen am Ziel, wo jede Episode schon beendet ist, und die fünf in :numref:`fig_rl_reward_gridworld` mit Kreuzen markierten Paare. Ihre geschätzten Belohnungen bleiben beim Anfangswert null. Jeder Zug von oben oder von der Seite in den Gefahrenstreifen kommt in zwei bis vier der tausend Episoden vor. Die beiden Züge von unten kommen nie vor. Das reicht kaum aus, um die geschätzten Belohnungen von null wegzubewegen. Dort lagen die Gewichte anfangs, und dorthin zieht sie der Gewichtszerfall. Tatsächlich kostet jeder dieser Züge $1.5$. Ein neuronales Belohnungsmodell würde gemäß seinem induktiven Bias extrapolieren, mit besseren oder schlechteren Ergebnissen (Übung :exref:`ex_regularized-neural-reward-models`).

<!--en:ba8a869f-->
### Invarianz gegenüber Belohnungsgestaltung

<!--en:e9ce26e6-->
Ein RMS-Fehler von $0.19$ erscheint bei Schrittkosten von $0.04$ groß. Vergleiche können jedoch den größten Teil dieses Fehlers gar nicht erfassen. Die potentialbasierte Belohnungsgestaltung (engl. potential-based shaping) aus :numref:`sec_mdp` addiert $\gamma \Phi(s') - \Phi(s)$ zur Belohnung jedes Zugs von $s$ nach $s'$. Dabei ist $\Phi$ eine Funktion auf den Zuständen mit $\Phi = 0$ am Ziel. Wie wir in :numref:`sec_valueiter` bewiesen haben, bleibt die optimale Strategie dabei unverändert. Entlang einer Trajektorie bilden diese Terme eine Teleskopsumme. Für jede Trajektorie von Zustand $0$ bis zum Ziel ändert sich der Ertrag also um denselben Betrag $-\Phi(0)$. Jeder Vergleich betrifft zwei Trajektorien mit demselben Startzustand $0$. Daher fällt die Verschiebung in $r(\tau) - r(\tau')$ weg und ist für :eqref:`eq_bradley_terry` unsichtbar. (Bei jeder zehnten Trajektorie brechen wir nach vierzig Schritten ab. Dann bleibt ein letzter Term $\gamma^{40}\Phi(s_{40})$, der zu klein ist, um $\Phi$ festzulegen.) Die Vergleiche bestimmen die Belohnung also nur bis auf potentialbasierte Belohnungsgestaltung. Unter den Belohnungsfunktionen, die sie nicht unterscheiden können, bevorzugt der Gewichtszerfall diejenige mit den kleinsten Gewichten.

<!--en:b1323ab7-->
Die Codezelle bestimmt mit der Methode der kleinsten Quadrate das Potential, dessen Beitrag zur Belohnung den Schätzfehler auf `core` am besten erklärt. Das Produkt der Zeile $(s, a)$ von `F` mit $\Phi$ ergibt $\gamma \sum_{s'} P(s' \mid s, a)\, \Phi(s') - \Phi(s)$. Indem wir die letzte Spalte weglassen, setzen wir $\Phi(15) = 0$.

<!--en:fe01b9bd-->
Ziehen wir den Potentialterm ab, bleibt ein RMS-Fehler von $0.03$. Bis auf die Belohnungsgestaltung ist die Schätzung also dort gut, wo Daten vorliegen. Das Potential steigt von etwa $0.8$ am Start auf etwa $0.9$ neben dem Ziel. An den Gefahrenfeldern, die kein Paar aus `core` berührt, bleibt es unbestimmt. Ein solches Potential zieht etwa $0.9$ von der Belohnung für den Zug ins Ziel ab und etwa $(1 - \gamma) \times 0.8 = 0.04$ von der Belohnung für jeden anderen Zug. Genau das zeigen die ausgegebenen Belohnungen: Der letzte Zug bringt $0.09$ statt $1$, und ein gewöhnlicher Zug kostet $0.08$ statt $0.04$. Beide Belohnungsfunktionen ordnen jedes Paar von Trajektorien aus Zustand $0$ gleich ein. Die geschätzte Funktion hat die kleineren Gewichte. Nach dieser Schätzung mit Belohnungen nahe null sind Züge in den Gefahrenstreifen günstiger als ein gewöhnlicher Zug.

<!--en:c2b6e132-->
Bei Sprachmodellen gilt dieselbe Invarianz. Dort vergleichen wir zwei Antworten $y$ und $y'$ auf denselben Prompt $x$. Die Belohnung $r(x, y) + f(x)$ sagt für jede beliebige Funktion $f$ des Prompts sämtliche Vergleiche genauso vorher wie $r(x, y)$. Eine gelernte Belohnung ist also nur relativ zum jeweiligen Prompt bestimmt (Übung :exref:`ex_regularized-identifiability`). Auf Strategiegradientenverfahren hat das keinen Einfluss. Ziehen wir eine Funktion des Prompts von der Belohnung ab, verwenden wir damit eine Baseline, die den erwarteten Gradienten unverändert lässt (:numref:`sec_baselines`). Der Gruppenmittelwert bei GRPO ist eine solche Baseline, enthält allerdings auch die Belohnung der jeweiligen Antwort selbst. Wie der genannte Abschnitt zeigt, skaliert das den erwarteten Aktualisierungsschritt um den Faktor $(K-1)/K$ bei Gruppen von $K$ Antworten.

<!--en:ab13107e-->
## Optimierung einer geschätzten Belohnung

<!--en:ea876bc4-->
Nun suchen wir wie ein Agent die Strategie, die die geschätzte Belohnung maximiert. Da wir die Dynamik kennen, finden wir sie durch Wertiteration (engl. value iteration). Die Funktion `plan` liefert die optimale deterministische Strategie für eine Belohnungsfunktion in Form von One-Hot-Zeilen. Die Funktion `ret` liefert den exakten Ertrag einer Strategie ausgehend von Zustand $0$. Dazu verwendet sie `policy_value` aus :numref:`sec_valueiter`. Diese Funktion löst die linearen Bellman-Gleichungen $V^\pi = (I - \gamma P_\pi)^{-1} r_\pi$ in einem Schritt. Wir planen einmal mit der wahren und einmal mit der geschätzten Belohnung. Beide Pläne bewerten wir mit beiden Belohnungsfunktionen.

<!--en:0efb27ef-->
Nach der geschätzten Belohnung erzielt der Plan für $\hat r$ einen Score von $-0.03$ und übertrifft damit den Score $-0.24$ des wahren Optimums. Der geschätzte Score des wahren Optimums liegt um $0.83$ unter seinem tatsächlichen Score. Der Abstand entspricht ungefähr der Verschiebung $\Phi(0) = 0.78$ durch die Belohnungsgestaltung. Aussagekräftig sind daher nur Differenzen zwischen geschätzten Scores. Unter der wahren Belohnung erzielen die beiden Pläne $-2.11$ und $+0.59$. Der Plan für $\hat r$ folgt dem Weg $0 \to 1 \to 5 \to 6 \to 7 \to 11 \to 15$ mitten durch beide Gefahrenfelder. Dieser Weg ist so kurz wie der sichere Weg am Rand und unter $\hat r$ günstiger. Die Wertiteration hat die vorgegebene Belohnung optimiert. Der Fehler liegt in der Belohnung außerhalb des von den Referenzdaten erfassten Bereichs. Wir ziehen $24$ Präferenzdatensätze mit unterschiedlichen Zufallsstartwerten. Der Plan für $\hat r$ durchquert bei $19$ Datensätzen die Gefahrenfelder und erreicht bei $4$ weiteren nie das Ziel. Ein Datensatz liefert das wahre Optimum.

<!--en:a8c22151-->
Dieses Versagen hat einen Namen. Nach Goodharts Gesetz verliert eine Messgröße ihren Wert, sobald sie selbst zum Ziel wird. Erzielt eine Strategie bei einer vorgegebenen oder geschätzten Belohnung einen hohen Score und verfehlt dabei das eigentliche Ziel, spricht man im bestärkenden Lernen von *Reward-Hacking*. Zahlreiche Beispiele finden sich in :cite:`Krakovna.Uesato.Mikulik.ea.2020`. Der Annäherungsbonus aus :numref:`sec_valueiter`, der die optimale Strategie endlos im Kreis laufen ließ, beruhte auf demselben Mechanismus. Damals war der Fehler von Hand eingebaut. Hier ist er statistisch bedingt und tritt dort auf, wo Daten fehlen.

<!--en:14cc3d0d-->
## Die KL-regularisierte Zielfunktion

<!--en:9b48f23f-->
Der geschätzten Belohnung können wir nur dort vertrauen, wo die Referenzstrategie tatsächlich war. Eine Strategie, die nahe an der Referenz bleibt, bleibt im Bereich der Daten. Deshalb belegen wir Abweichungen mit einer Strafe. Die KL-Divergenz bietet sich als Maß an. Sie ist ein Erwartungswert unter der Strategie selbst und lässt sich daher aus deren eigenen Samples schätzen. Wählt die Strategie eine Aktion, die bei der Referenz nie vorkommt, ist die Divergenz unendlich. Betrachten wir eine einzelne Entscheidung mit Belohnung $r(a)$ auf einer endlichen Aktionsmenge, einer Referenz $\pi_{\textrm{ref}}$ und einem Koeffizienten $\beta > 0$. Dann lautet die *KL-regularisierte Zielfunktion* (engl. KL-regularized objective)

<!--en:a905877c-->
Der erste Term ist die erwartete Belohnung. Der zweite zieht für jedes Nat Divergenz von der Referenz $\beta$ Belohnungseinheiten ab. Der Koeffizient $\beta$ gewichtet den Strafterm. Die Referenz bleibt fest.

<!--en:929a3a5f-->
### Geschlossene Form des Optimums für eine Entscheidung

<!--en:e9a2c8e1-->
**Proposition.** Für $\beta > 0$ besitzt die Zielfunktion :eqref:`eq_kl_objective` genau eine maximierende Strategie:

<!--en:92915056-->
Der Maximalwert beträgt $\beta \log Z$.

<!--en:37fb2ef8-->
**Beweis.** Nach Konstruktion gilt $r(a) = \beta \log\big( \pi^\star(a) Z / \pi_{\textrm{ref}}(a) \big)$. Setzen wir dies in die Zielfunktion ein, so folgt für jede Strategie $\pi$:

<!--en:2bbcbce9-->
Nach der in :numref:`chap_mdl-information-theory` bewiesenen Gibbs-Ungleichung ist die verbleibende Divergenz nichtnegativ. Sie ist genau dann null, wenn $\pi = \pi^\star$ gilt. $\blacksquare$

<!--en:505d003f-->
Das Optimum ergibt sich also, indem wir die Referenz mit dem Exponentialfaktor $e^{r/\beta}$ umgewichten und erneut normieren. Der erreichte Wert ist der Logarithmus der Partitionsfunktion. :numref:`fig_rl_kl_tilting` zeigt das Produkt für fünf Aktionen. Die Referenzwahrscheinlichkeit verläuft dort absichtlich nicht monoton mit der Belohnung. Ihr Einbruch bei der zweiten Aktion bleibt bei schwacher Umgewichtung sichtbar. Für das Optimum zählt also neben der Rangfolge der Belohnungen auch das Gewicht jeder Aktion in der Referenz.

<!--en:6b6b1e4a-->
![Das Optimum der Zielfunktion mit Strafterm als Produkt. Jede Zeile zeigt das Produkt der Referenz $\pi_{\textrm{ref}} = (0.30,\ 0.10,\ 0.25,\ 0.20,\ 0.15)$ mit dem Faktor $e^{r/\beta}$ für die Belohnung $r = (0,\ 0.5,\ 1,\ 2,\ 3)$. Nach erneuter Normierung ergibt sich $\pi^\star$, in der oberen Zeile für $\beta = 2$ und in der unteren für $\beta = 0.2$. Der Einbruch der Referenz bei $a_2$ ist oben auch in $\pi^\star$ erkennbar. Unten überwiegt die Umgewichtung. Am Rand stehen die beiden Grenzfälle: Für $\beta \to \infty$ ergibt sich die Referenz, für $\beta \to 0$ eine Punktmasse auf der besten Aktion.](../img/mdl-rl-kl-tilting.svg)

<!--en:3912cca1-->
Zwei Grenzfälle vervollständigen die Aussage. Erstens übernimmt das Optimum den Träger der Referenz. Gilt $\pi_{\textrm{ref}}(a) = 0$, so hat jede Strategie, die der Aktion $a$ positive Masse zuweist, unendliche Divergenz. Nach :eqref:`eq_kl_optimum` hat $a$ unabhängig von ihrer Belohnung die Wahrscheinlichkeit null. Zweitens konzentriert sich das Optimum für $\beta \to 0$ auf die beste Aktion im Träger von $\pi_{\textrm{ref}}$. Teilen sich mehrere Aktionen den höchsten Wert, ist der Grenzwert die auf diese Aktionen eingeschränkte und neu normierte Referenz.

<!--en:ef79f783-->
Wir prüfen die Proposition numerisch. Dazu lösen wir :eqref:`eq_kl_objective` mit einem iterativen Verfahren, das die geschlossene Form nicht verwendet. Die Funktion `solve_kl` führt einen Gradientenaufstieg mit exponentiertem Gradienten aus. Sie multipliziert die aktuelle Strategie mit der Exponentialfunktion des Produkts aus Schrittweite und Gradient $r - \beta(\log(\pi/\pi_{\textrm{ref}}) + 1)$ und normiert das Ergebnis wieder auf den Simplex. Der Summand $+1$ ist für jede Aktion gleich und fällt bei der Normierung weg. Daher lässt ihn der Code weg.

<!--en:d0946f5e-->
Sinkt $\beta$ von $5$ auf $0.2$, steigt die erwartete Belohnung von $1.38$ auf $2.99$ und die Divergenz von $0.02$ auf $1.84$ Nat. Die beiden Zeilen in :numref:`fig_rl_kl_tilting` zeigen die Optima für $\beta = 2$ mit einer Divergenz von $0.15$ Nat und für $\beta = 0.2$. Wir vergleichen nun jede konvergierte Näherung mit :eqref:`eq_kl_optimum` und den erreichten Zielfunktionswert mit $\beta \log Z$. Die Funktion `pi_star` wertet die geschlossene Form aus. Im Exponenten zieht sie die größte Belohnung ab, um einen Überlauf zu verhindern. Bei der Normierung fällt diese Verschiebung weg.

<!--en:aaaa20c5-->
Die Iteration liefert für jedes $\beta$ die geschlossene Lösung bis auf Gleitkommagenauigkeit. Der Zielfunktionswert stimmt auf sechs Nachkommastellen mit $\beta \log Z$ überein.

<!--en:ee151e74-->
### Vier Folgerungen aus der geschlossenen Form

<!--en:ef367dee-->
**Ohne Strafterm ist das Optimum eine Punktmasse.** Im Grenzfall $\beta \to 0$ in :eqref:`eq_kl_optimum` fällt die gesamte Masse auf die Aktion mit der höchsten Belohnung. Bei einem Gleichstand verteilt sie sich gemäß der Referenz auf die besten Aktionen. Dies ist das unregularisierte Problem, dessen Lösung der Argmax ist. Die Aussage betrifft das Optimum. Der Verlauf eines einzelnen Trainings kann davon abweichen. Die Entropie der PPO-Strategie in :numref:`sec_ppo` sank während des Trainings. Das passt zu einer Annäherung an das fast deterministische Optimum der PPO-Zielfunktion mit schwacher Entropieregularisierung. Bei stärkerer Gewichtung des Strafterms wird eine stochastische Strategie optimal.

<!--en:fd0b6592-->
**Der Koeffizient interpoliert zwischen Referenz und Argmax.** Wir werten die geschlossene Form für ein sehr großes und ein sehr kleines $\beta$ aus:

<!--en:413ca2b2-->
Ein großes $\beta$ reproduziert die Referenz, ein kleines $\beta$ führt zur gierigen Auswahl. Jede Strategie dazwischen ist für einen bestimmten Wert von $\beta$ das exakte Optimum.

<!--en:09d1f14c-->
**Bei gleichverteilter Referenz wird der Strafterm zum Entropiebonus.** Für die Gleichverteilung als Referenz gilt $D_{\textrm{KL}}(\pi \Vert \textrm{uniform}) = \log |\mathcal{A}| - H(\pi)$. Der Strafterm ergibt also einen Entropiebonus $\beta H(\pi)$ plus eine Konstante. Damit wird :eqref:`eq_kl_objective` zur Zielfunktion für *maximale Entropie*. Die geschlossene Form vereinfacht sich zur Softmax-Funktion der Belohnung:

<!--en:fb4e22b1-->
Diese Form haben die Softmax-Strategien aus :numref:`sec_policygradient`. Die Boltzmann-Regel zur Exploration aus :numref:`sec_exploration` mit der Temperatur $\beta$ ist also das exakte Optimum einer einstufigen Zielfunktion mit Entropiebonus.

<!--en:3a4635a7-->
**Das Optimum ist eine A-posteriori-Verteilung.** Die geschlossene Lösung :eqref:`eq_kl_optimum` hat die Form der Bayes-Regel. Dabei ist $\pi_{\textrm{ref}}$ die A-priori-Verteilung, $e^{r/\beta}$ eine unnormierte Likelihood, $\pi^\star$ die A-posteriori-Verteilung und $Z$ die Normierungskonstante. Verfahren, die Steuerung als Inferenz auffassen, bauen auf dieser Identität auf. Sie deuten Belohnungen als Log-Likelihoods eines Ereignisses namens Optimalität und die regularisierte Strategieoptimierung als Variationsinferenz :cite:`Levine.2018,Korbak.Perez.Buckley.2022`.

<!--en:5c57fe22-->
### Die Grenzkurve für eine einzelne Entscheidung

<!--en:a75526a0-->
Jeder Wert von $\beta$ liefert ein Optimum mit einer erwarteten Belohnung und einer Divergenz von der Referenz. Variieren wir $\beta$, erhalten wir für jede Divergenz die beste erreichbare Belohnung. Das ergibt die *Grenzkurve von Belohnung und Divergenz* (engl. reward-divergence frontier):

<!--en:0b136b8a-->
Die Grenzkurve beginnt bei der mittleren Belohnung der Referenz von $1.15$. Sie endet bei der Punktmasse auf der besten Aktion mit einer Divergenz von $\log(1/0.15) = 1.90$ Nat. Die Kurve steigt und ist konkav. Ihre Steigung beträgt $\beta$ am Optimum zum Koeffizienten $\beta$ (Übung :exref:`ex_regularized-frontier-is-concave`). Jedes weitere Nat bringt also weniger zusätzliche Belohnung. Die beiden Zeilen in :numref:`fig_rl_kl_tilting` entsprechen zwei Punkten dieser Kurve. Beim Best-of-$N$-Sampling ziehen wir $N$ Antworten aus der Referenz und behalten die mit der höchsten Belohnung. So bewegen wir uns ganz ohne Training entlang derselben Achse. Die Divergenz von der Referenz beträgt höchstens $\log N - (N-1)/N$, also etwa $1.8$ Nat für $N = 16$ :cite:`Gao.Schulman.Hilton.2023`. Für die Gitterwelt zeichnen wir eine Kurve derselben Art. Dort bewerten wir die Strategien anhand der wahren Belohnung statt anhand der Belohnung, die wir optimieren.

<!--en:86785d58-->
### Trust-Region und Strafterm im Vergleich

<!--en:bed9e39b-->
Sowohl die Trust-Region als auch der Strafterm mit fester Referenz verwenden die KL-Divergenz. Sie erfüllt dabei unterschiedliche Aufgaben.

<!--en:eefab72b-->
> **Zwei Anwendungen der KL-Divergenz.** Eine **Trust-Region** (:numref:`sec_ppo`) misst die Divergenz von der *vorherigen Iterierten*. Sie begrenzt jeden Aktualisierungsschritt und sichert so eine lokale Näherung ab. Ihre Referenz ändert sich nach jedem Schritt. Ein **Strafterm mit fester Referenz** (dieser Abschnitt) ist Teil der Zielfunktion und verschiebt das Optimum zu :eqref:`eq_kl_optimum`. Das Fine-Tuning von Sprachmodellen nach dem Vorbild von PPO verwendet beides. Eine begrenzte Ersatzzielfunktion relativ zur Strategie, aus der die Samples stammen, stabilisiert die Schritte. Ein Strafterm für die Abweichung vom eingefrorenen Ausgangsmodell steuert die endgültige Strategie :cite:`ouyang2022training`.

<!--en:1b3b3608-->
Die geschlossene Form zeigt, wohin eine Trust-Region allein führt. Angenommen, jeder Aktualisierungsschritt löst sein Trust-Region-Problem exakt. Dabei gewichten wir die Divergenz von der vorherigen Iterierten mit $\tilde{\beta}$, also $\pi_{k+1} = \operatorname{argmax}_\pi \mathbb{E}_\pi[r] - \tilde{\beta} D_{\textrm{KL}}(\pi \Vert \pi_k)$, und beginnen bei $\pi_0 = \pi_{\textrm{ref}}$. Nach der Proposition multipliziert jeder Schritt die Strategie mit $e^{r/\tilde{\beta}}$. Nach $k$ Schritten folgt daher

<!--en:fa291fc4-->
also die geschlossene Form mit dem Koeffizienten $\tilde{\beta}/k$. Mit wachsendem $k$ geht der Koeffizient gegen null, und die Iterierten nähern sich der Punktmasse auf der besten Aktion. Eine Trust-Region bestimmt den Weg. Nur eine feste Referenz ändert den Grenzwert der Iteration (Übung :exref:`ex_regularized-moving-reference`).

<!--en:c8649d96-->
### Die Richtung der Divergenz

<!--en:5fd7646b-->
Der Strafterm $D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}})$ ist in der Einteilung aus :numref:`sec_mdl-fwd-vs-rev-kl` die *Rückwärtsdivergenz*. Sie ist ein Erwartungswert unter $\pi$ und lässt sich aus den Samples der Strategie selbst schätzen. Die geschlossene Form zeigt, was mit einer Aktion $a$ geschieht, deren Belohnung um $\Delta(a) > 0$ unter der besten liegt. Relativ zur besten Aktion beträgt ihre Wahrscheinlichkeit unter $\pi^\star$ das entsprechende Verhältnis der Referenzwahrscheinlichkeiten mal $e^{-\Delta(a)/\beta}$. Der Strafterm mit Vorwärtsdivergenz $D_{\textrm{KL}}(\pi_{\textrm{ref}} \Vert \pi)$ ist unendlich, sobald die Strategie eine Aktion ausschließt, die bei der Referenz vorkommt. Im Optimum behält jede Aktion eine Wahrscheinlichkeit von mindestens der Größenordnung $\beta / \Delta(a)$ mal ihrer Referenzwahrscheinlichkeit (Übung :exref:`ex_regularized-forward-kl-penalty`). In beiden Richtungen hat jede Aktion der Referenz für jedes $\beta > 0$ positive Wahrscheinlichkeit. Der Unterschied zeigt sich darin, wie schnell die Wahrscheinlichkeiten schlechter Aktionen bei sinkendem $\beta$ abklingen: beim Strafterm mit Rückwärtsdivergenz exponentiell in $1/\beta$, beim Strafterm mit Vorwärtsdivergenz nur linear in $\beta$.

<!--en:c2e395fc-->
## Das weiche Backup

<!--en:47ba30c4-->
In der Gitterwelt treffen wir mehr als eine Entscheidung. In einem Markov-Entscheidungsprozess (MDP) fällt der Strafterm bei jedem Schritt an. Ausgehend von Zustand $0$ lautet die Zielfunktion damit

<!--en:b4ee811c-->
In jedem Zustand steht die Strategie vor einer einzelnen Entscheidung der Art, die wir gerade untersucht haben. Mit welcher Belohnung sollen wir die Referenz für diese Entscheidung umgewichten?

<!--en:98f74941-->
### Ein Schritt vor dem Ende

<!--en:a8b4f3bf-->
Bei der letzten Entscheidung einer Episode entspricht der Wert einer Aktion ihrer Belohnung. Die Proposition gilt unmittelbar: Das Optimum ist dort die mit $e^{r/\beta}$ umgewichtete Referenz, und der Zustand hat den Wert $\beta \log Z$. Einen Schritt früher ist eine Aktion so viel wert wie ihre Belohnung plus der Wert des Zustands, in den sie führt. Dieser Wert ist das, was die optimale Strategie *dort* unter Berücksichtigung des Strafterms erzielt, also $\beta \log Z$. Er liegt unter der besten dort verfügbaren Belohnung. Die optimale Strategie wählt die beste Aktion im letzten Schritt nur mit einer gewissen Wahrscheinlichkeit und muss zudem die Strafe zahlen.

<!--en:79cc524d-->
Ein Beispiel mit zwei Schritten verdeutlicht den Unterschied. Die Entscheidung zwischen fünf Aktionen aus :numref:`fig_rl_kl_tilting` mit $\beta = 2$ bilde den letzten Schritt einer Episode. Ihr Wert beträgt $\beta \log Z = 1.44$, obwohl die beste Belohnung $3$ beträgt. Einen Schritt früher kann der Agent in diesen Entscheidungszustand wechseln, ohne sofort etwas zu erhalten, oder mit einer sicheren Belohnung von $2$ aufhören. Die Referenz wählt beides mit gleicher Wahrscheinlichkeit, und es gibt keine Diskontierung. Die Codezelle gewichtet die erste Entscheidung einmal anhand des weichen Werts $1.44$ und einmal anhand der besten Belohnung $3$ um. Für beide zweistufigen Strategien wertet sie die Zielfunktion mit Strafterm aus.

<!--en:00962bd3-->
Die Umgewichtung anhand des weichen Werts erreicht das Optimum, also $\beta \log Z$ für die erste Entscheidung. Wer mit der besten Belohnung umgewichtet, setzt den Wert der zweiten Entscheidung mit $3$ an, obwohl die Strategie dort nach Abzug der Strafe nur $1.44$ erzielt. Sie wählt diesen Zustand daher zu oft und verliert etwa $0.15$. Die Umgewichtung muss den Wert verwenden, der jede spätere Belohnung *und* jede spätere Strafe berücksichtigt.

<!--en:f8b3ad01-->
### Weiche Wertiteration

<!--en:2951065b-->
Bestimmen wir diesen Wert für jeden Zustand, erhalten wir die allgemeine Form. Dabei sei $Q_\beta(s, a)$ der Wert der Aktion $a$ im Zustand $s$ unter der Zielfunktion mit Strafterm und $V_\beta(s)$ der Wert des Zustands $s$. Wir wenden die Proposition in jedem Zustand an, wobei $Q_\beta(s, \cdot)$ die Rolle der Belohnung übernimmt. Dann gilt

<!--en:c0f3833b-->
und die optimale Strategie lautet $\pi_\beta(a \mid s) \propto \pi_{\textrm{ref}}(a \mid s)\, e^{Q_\beta(s, a)/\beta}$. Dies sind die Bellman-Optimalitätsgleichungen aus :numref:`sec_valueiter` mit einer Änderung. An die Stelle des Maximums über die Aktionen tritt ein mit der Referenz gewichtetes Log-Sum-Exp, ein *weiches Maximum* (engl. soft maximum). Für $\beta \to 0$ nähert es sich dem harten Maximum über den Träger der Referenz. Bei verrauschten Schätzungen der Aktionswerte ist das harte Maximum nach oben verzerrt, wie :numref:`sec_qlearning` zeigt. Das Log-Sum-Exp ist konvex und daher ebenfalls nach oben verzerrt, allerdings weniger stark. Bei gleichverteilter Referenz erhalten wir so die weiche Wertiteration des bestärkenden Lernens mit maximaler Entropie.

<!--en:b5fdcdf6-->
Die Wertiteration löste die harten Gleichungen durch wiederholte Anwendung von $V \leftarrow \max_a \textrm{backup}(V)$ mit dem Startwert $V = 0$. Derselbe Durchlauf mit dem weichen Maximum löst :eqref:`eq_soft_backup`. Der Beweis, dass der Durchlauf mit hartem Maximum kontrahiert (:numref:`subsec_vi_contraction`), verwendete nur zwei Eigenschaften des Maximums. Es ist monoton, und addiert man zu jedem Eintrag eine Konstante, wächst das Maximum um dieselbe Konstante. Das weiche Maximum besitzt beide Eigenschaften. Daher ist auch der Durchlauf mit weichem Maximum eine $\gamma$-Kontraktion, für jedes $\beta > 0$ und jede Referenz (Übung :exref:`ex_regularized-soft-backup-transplanted`). Die Funktion `soft_max` berechnet für jede Zeile eines Arrays von Aktionswerten das weiche Maximum. Zur numerischen Stabilität zieht sie zuvor das jeweilige Zeilenmaximum ab. Zusammen mit dem weichen Maximum liefert sie die umgewichtete Strategie. Bei jedem Durchlauf wendet `soft_v` zuerst `backup` und dann `soft_max` an.

<!--en:fb3276ac-->
### Wahrer Ertrag in Abhängigkeit vom KL-Budget

<!--en:3a317166-->
Nun bestimmen wir die Grenzkurve für die Gitterwelt mit einer Änderung. Der Optimierer sieht die geschätzte Belohnung $\hat r$, wir bewerten mit der wahren. Wir durchlaufen $36$ Werte von $\beta$, beginnend bei $10$ bis hinunter zu $10^{-2.5}$. Für jeden Wert berechnet `soft_v` das exakte Optimum $\pi_\beta$ für $\hat r$ mit Strafterm. Wir erfassen den wahren und den geschätzten Ertrag dieser Strategie sowie ihre Divergenz von der Referenz. Als Divergenz verwenden wir die Größe aus dem Strafterm der MDP-Zielfunktion, also $\mathbb{E}_{\pi_\beta}\big[\sum_t \gamma^t D_{\textrm{KL}}\big(\pi_\beta(\cdot \mid s_t) \Vert \pi_{\textrm{ref}}(\cdot \mid s_t)\big)\big]$ ausgehend von Zustand $0$. Dies ist die KL-Divergenz zwischen den Trajektorienverteilungen von $\pi_\beta$ und $\pi_{\textrm{ref}}$, wenn jede Episode zusätzlich nach jedem Schritt mit Wahrscheinlichkeit $1 - \gamma$ endet (Übung :exref:`ex_regularized-trajectory-divergence`). Der Code gewichtet die Divergenzen `kl` der einzelnen Zustände mit `visits`, den erwarteten diskontierten Besuchszahlen pro Zustand ausgehend von Zustand $0$. Dazu genügt es, ein lineares Gleichungssystem mit der Übergangsmatrix von $\pi_\beta$ zu lösen. Der Code erfasst auch das größte Residuum von :eqref:`eq_soft_backup` über den gesamten Durchlauf.

<!--en:b89dce5e-->
Jeder Punkt im Diagramm ist ein exaktes Optimum der Zielfunktion mit Strafterm. Das größte Bellman-Residuum liegt unter $10^{-10}$. Der geschätzte Ertrag steigt durchgehend, wie es für exakte Optima sein muss. Sei $\beta_1 < \beta_2$, und seien $J_1, J_2$ die geschätzten Erträge und $D_1, D_2$ die Divergenzen der zugehörigen Optima. Aus der Optimalität folgen $J_1 - \beta_1 D_1 \geq J_2 - \beta_1 D_2$ und $J_2 - \beta_2 D_2 \geq J_1 - \beta_2 D_1$. Addition liefert $(\beta_2 - \beta_1)(D_1 - D_2) \geq 0$, also $D_1 \geq D_2$. Aus der ersten Ungleichung folgt dann $J_1 \geq J_2$.

<!--en:086a5e43-->
Der wahre Ertrag verhält sich anders. Vom Referenzwert $-0.12$ steigt er auf $+0.53$ bei $3.7$ Nat, nahe am wahren Optimum von $+0.59$. Dann fällt er auf $-1.83$, während sich die Strategie auf den Gefahrenstreifen konzentriert. Eine mäßige Optimierung der geschätzten Belohnung verbessert den wahren Ertrag. Eine stärkere Optimierung nutzt die Züge aus, die in den Daten nie erfasst wurden. Bei $24$ Präferenzdatensätzen liegt der Höchstwert zwischen etwa $+0.5$ und dem Optimum $+0.59$ und wird bei wenigen Nat erreicht. In $21$ Fällen fällt der wahre Ertrag danach. Bei den übrigen drei Datensätzen fällt er im untersuchten Bereich nicht. In einem Fall ist das geschätzte Optimum das wahre. In zwei Fällen sind die geschätzten Werte des sicheren und des gefährlichen Wegs fast gleich. Daher behält die optimale Strategie mit Strafterm bis hinunter zu $\beta = 10^{-2.5}$ den sicheren Weg bei. Wie weit der Höchstwert von der Referenz entfernt liegt und wie tief der Ertrag danach fällt, hängt von den Kosten der Gefahr ab und davon, wie gut die Vergleiche sie erfassen. :citet:`Gao.Schulman.Hilton.2023` beobachteten denselben Anstieg und Abfall bei Sprachmodellen. Sie optimierten die Modelle anhand eines gelernten Belohnungsmodells und bewerteten sie mit einem dafür zurückgehaltenen Belohnungsmodell.

<!--en:3436b4dd-->
Der Optimierer kann den Höchstwert des wahren Ertrags nicht erkennen. Er sieht nur die geschätzte Kurve, und diese steigt monoton. In der Praxis wählen wir $\beta$ daher anhand einer Bewertung, die vom Belohnungsmodell für das Training unabhängig ist. Dazu eignen sich zurückgehaltene Vergleiche, menschliche Urteile oder ein Prüfprogramm. Bei gleichverteilter Referenz wird derselbe Strafterm zum Entropiebonus. In :numref:`sec_sac` entwickeln wir dafür ein Off-Policy-Akteur-Kritiker-Verfahren, das die Belohnung mit Entropiebonus für kontinuierliche Aktionen maximiert. Sein Kritiker bewertet die eigene Strategie anhand der Belohnung mit Entropiebonus, anstatt das Log-Sum-Exp-Backup zu iterieren. Nach der Identität im Beweis der Proposition stimmen beide Werte überein, wenn die Strategie die umgewichtete Referenz ist. In :numref:`sec_rl_sequences` wenden wir die geschlossene Form :eqref:`eq_kl_optimum` auf die Aufgabe mit Prüfprogramm aus :numref:`chap_reinforcement_learning` an. Sie sagt voraus, ab welchem $\beta$ sich eine Ausweichantwort nicht mehr lohnt. Alle bisherigen Verfahren haben aus neuen Samples der aktuellen Strategie oder anhand eines bekannten Modells gelernt. In :numref:`sec_dqn` untersuchen wir, ob ein Netz Aktionswerte aus gespeicherten Übergängen beliebiger Strategien lernen kann, wie es die Q-Tabelle in :numref:`sec_qlearning` tat.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ee16d8bf-->
Eine aus Vergleichen geschätzte Belohnung ist nur dort genau, wo Vergleichsdaten vorliegen. Vergleiche zwischen Trajektorien mit gemeinsamem Start können eine potentialbasierte Belohnungsgestaltung nicht erkennen. Paare, die in den Daten nie vorkommen, behalten den Anfangswert der Schätzung. Ein Optimierer nutzt diese Lücken aus. In der Gitterwelt führt der Plan für die geschätzte Belohnung durch Gefahrenfelder, die die Referenzstrategie selten betreten hat. Eine Strafe für die KL-Divergenz von der Referenz verändert die Zielfunktion. Ihr exaktes Optimum ist die mit $e^{r/\beta}$ umgewichtete Referenz, der zugehörige Wert beträgt $\beta \log Z$. Bei gleichverteilter Referenz wird die Strafe zum Entropiebonus. In einem MDP richtet sich die Umgewichtung nach dem weichen Aktionswert. Dieser erfüllt eine Bellman-Gleichung, in der ein Log-Sum-Exp das Maximum ersetzt. Verringern wir das Gewicht des Strafterms, steigt der geschätzte Ertrag monoton. Der wahre Ertrag erreicht dagegen bei mäßiger Divergenz seinen Höchstwert und fällt dann wieder. Das Gewicht müssen wir daher anhand einer Bewertung wählen, die der Optimierer nicht sieht.

<!--en:2294243a-->
## Übungen

<!--en:7c33e22b-->
1. **Identifizierbarkeit und Skala.** Wir passen das Bradley-Terry-Modell :eqref:`eq_bradley_terry` an Vergleiche zweier Antworten $y$ und $y'$ auf denselben Prompt $x$ an.
    1. Zeigen Sie, dass das Ersetzen von $r(x, y)$ durch $r(x, y) + f(x)$ für jede Funktion $f$ des Prompts die Likelihood jedes solchen Datensatzes unverändert lässt. Verbinden Sie umgekehrt zwei Antworten auf $x$ durch eine Kante, wenn der Datensatz sie miteinander vergleicht. Zeigen Sie, dass sich zwei Belohnungsfunktionen auf jeder Zusammenhangskomponente dieses Graphen nur um eine Konstante unterscheiden, wenn sie jedem verglichenen Paar dieselbe Präferenzwahrscheinlichkeit zuweisen. Was ist identifizierbar, wenn der Graph zu einem Prompt mehrere Komponenten hat?
    1. Angenommen, die Bewertenden vergeben die Label gemäß $P(y \succ y') = \sigma\big(\omega\, (r^*(x, y) - r^*(x, y'))\big)$ mit einer Konsistenz $\omega > 0$. Wir passen :eqref:`eq_bradley_terry` mit der Maximum-Likelihood-Methode an unbegrenzt viele Daten an. Welche Belohnung ergibt die Schätzung?
    1. Zeigen Sie, dass das Optimum :eqref:`eq_kl_optimum` der geschätzten Belohnung beim Koeffizienten $\beta$ mit dem Optimum für $r^*$ beim Koeffizienten $\beta / \omega$ übereinstimmt. Angenommen, $\beta$ wurde anhand eines Belohnungsmodells abgestimmt, das an die Label einer Gruppe von Bewertenden angepasst wurde. Wie stark ist die Strafe beim selben $\beta$ für ein Belohnungsmodell, das die Label einer weniger konsistenten Gruppe verwendet, also einer Gruppe mit kleinerem $\omega$?
   :exlabel:`ex_regularized-identifiability`

<!--en:04925396-->
1. [code] **Divergenz von Trajektorienverteilungen.** Sei $p_\pi(\tau)$ die Wahrscheinlichkeit, dass eine Strategie $\pi$ ausgehend von Zustand $0$ die Trajektorie $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ mit festem Horizont $T$ erzeugt.
    1. Zeigen Sie, dass gilt:
       $$D_{\textrm{KL}}\big(p_\pi \Vert p_{\pi_{\textrm{ref}}}\big) = \mathbb{E}_\pi\Big[\sum_{t=0}^{T-1} D_{\textrm{KL}}\big(\pi(\cdot \mid s_t) \Vert \pi_{\textrm{ref}}(\cdot \mid s_t)\big)\Big].$$
       Welche Faktoren von $p_\pi(\tau)$ kürzen sich heraus? Warum lässt sich die Divergenz deshalb aus Trajektorien von $\pi$ ohne Übergangsmodell schätzen?
    1. Die Trajektorie ende nun nach jedem Schritt unabhängig von den Aktionen mit Wahrscheinlichkeit $1 - \gamma$. Zeigen Sie, dass die Divergenz zwischen den beiden Trajektorienverteilungen dann die diskontierte Summe der Divergenzen pro Zustand ergibt, die der Durchlauf in :numref:`subsec_kl_budget` an `kls` als `visits @ kl` anhängt.
    1. Betrachten Sie die Strategie $\pi_\beta$, bei der der wahre Ertrag in diesem Durchlauf seinen Höchstwert erreicht. Mitteln Sie $\mathbf{c}(\tau)^\top \log(\pi_\beta / \pi_{\textrm{ref}})$ über die Trajektorien. Berechnen Sie dazu den Logarithmus eintragsweise und ordnen Sie seine Einträge wie in $\mathbf{c}(\tau)$ als Vektor an. Verwenden Sie 2000 Trajektorien aus `sample_traj(mdp, pi, rng, T=400)` und vergleichen Sie den Mittelwert mit der am Höchstwert ausgegebenen Divergenz von $3.7$ Nat. Für manche Trajektorien ergeben sich negative Werte. Warum widerspricht das der Gibbs-Ungleichung nicht?
   :exlabel:`ex_regularized-trajectory-divergence`

<!--en:c36bea5f-->
1. [code] **Weiches Backup auf dem See.** Der Beweis in :numref:`subsec_vi_contraction` nutzte eine Eigenschaft des Maximums, nämlich $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$.
    1. Sei $m_\beta(f) = \beta \log \sum_a \pi_{\textrm{ref}}(a)\, e^{f(a) / \beta}$. Zeigen Sie, dass $m_\beta$ monoton ist und dass $m_\beta(f + c) = m_\beta(f) + c$ für jede Konstante $c$ gilt. Leiten Sie daraus ab, dass $m_\beta$ dieselbe Eigenschaft wie das Maximum besitzt. Folgern Sie, dass das Backup :eqref:`eq_soft_backup` eine $\gamma$-Kontraktion für jedes $\beta > 0$ und jede Referenz ist.
    1. Wenden Sie `soft_v` auf den rutschigen See aus :numref:`sec_valueiter` an. Bauen Sie ihn als `d2l.TabularMDP` auf und verwenden Sie eine gleichverteilte Referenz. Plotten Sie $V_\beta(s_0)$ gegen $\beta$ auf einem logarithmischen Gitter von $10^{-4}$ bis $10^2$. Welchen beiden in :numref:`sec_valueiter` berechneten Werten nähert sich die Kurve an ihren Enden? Erklären Sie anhand der regularisierten Zielfunktion, warum die Kurve monoton ist.
    1. Für welchen ungefähren Wert von $\beta$ liegt $V_\beta(s_0)$ auf halbem Weg zwischen den beiden Grenzwerten? Vergleichen Sie diesen Koeffizienten mit den Differenzen zwischen den optimalen Aktionswerten $Q^*(s_0, a)$ auf dem See.
   :exlabel:`ex_regularized-soft-backup-transplanted`

<!--en:59667d0f-->
1. [extended] **Neuronale Belohnungsmodelle.** ● Die lineare Schätzung von `r_hat` lässt die Werte der Paare ohne Daten nahe null. Ein neuronales Belohnungsmodell extrapoliert dagegen gemäß seinem induktiven Bias. Ersetzen Sie die 64 Gewichte `w` durch die Ausgaben eines mehrschichtigen Perzeptrons mit zwei versteckten Schichten zu je 64 ReLU-Einheiten und einer Ausgabe pro Zustand-Aktion-Paar. Setzen Sie die Zeile für das Ziel auf null. Trainieren Sie das Modell mit denselben Daten `D` und denselben Labels `y` für 2000 Schritte, jeweils mit dem gesamten Datensatz. Verwenden Sie die negative Log-Likelihood des Bradley-Terry-Modells und Adam mit der Lernrate $0.01$.
    1. Trainieren Sie ein Modell mit einer One-Hot-Kodierung von Zustand und Aktion als Eingabe. Trainieren Sie ein zweites Modell mit den beiden auf $[0, 1]$ skalierten Gitterkoordinaten des Zustands und einer One-Hot-Kodierung der Aktion. Verwenden Sie jeweils vier Zufallsstartwerte. Geben Sie für jedes Modell den RMS-Fehler auf `core`, die geschätzten Belohnungen der in `into` aufgeführten Züge in die Gefahrenfelder und den wahren Ertrag von `plan` an.
    1. Wiederholen Sie den Durchlauf aus :numref:`subsec_kl_budget` für jede geschätzte Belohnung. Welche Darstellung benötigt den Strafterm? Welche Eigenschaft ihrer Eingabe erklärt den Unterschied?
   :exlabel:`ex_regularized-neural-reward-models`

<!--en:3547baff-->
1. **Steigung der Grenzkurve.** ● Setzen Sie $u = 1 / \beta$. Dabei bezeichnen $R(u)$ und $D(u)$ die erwartete Belohnung und die Divergenz $D_{\textrm{KL}}(\pi^\star \Vert \pi_{\textrm{ref}})$ des Optimums :eqref:`eq_kl_optimum`.
    1. Zeigen Sie, dass $R(u) = \frac{d}{du} \log Z$ und $R'(u) = \textrm{Var}_{\pi^\star}(r)$ gelten. Zeigen Sie dann anhand des Werts $\beta \log Z$ aus der Proposition, dass $D'(u) = u\, \textrm{Var}_{\pi^\star}(r)$ gilt.
    1. Folgern Sie, dass die durch `front` und `kl_f` beschriebene Grenzkurve die Steigung $\beta$ im Punkt $\pi^\star$ hat. Zeigen Sie auch, dass sie steigt und konkav ist, sofern $r$ auf dem Träger von $\pi_{\textrm{ref}}$ nicht konstant ist.
    1. Zeigen Sie, dass $\pi^\star$ auch das Optimierungsproblem $\max_\pi \mathbb{E}_\pi[r]$ unter der Nebenbedingung $D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}}) \leq D(u)$ löst. Welcher Größe des Problems mit Nebenbedingung entspricht $\beta$?
    1. Das Best-of-$N$-Sampling zieht $N$ unabhängige Samples aus $\pi_{\textrm{ref}}$ und behält das mit der höchsten Belohnung. Die Belohnung habe unter der Referenz eine stetige Verteilung. Zeigen Sie, dass die Divergenz der Best-of-$N$-Verteilung von der Referenz dann $\log N - (N-1)/N$ beträgt. Betrachten Sie den Banditen mit fünf Aktionen, `p_ref` und `r`, dessen Belohnungen verschieden und aufsteigend geordnet sind. Best-of-$N$ wählt die Aktion $a$ mit Wahrscheinlichkeit $F(a)^N - F(a-1)^N$. Dabei ist $F(a)$ die kumulierte Referenzwahrscheinlichkeit aller Aktionen bis einschließlich $a$. Plotten Sie die erwartete Belohnung gegen die Divergenz für $N = 1, \ldots, 16$ zusammen mit `front` und `kl_f`. Wie nahe kommt das Verfahren der Grenzkurve? Bleibt seine Divergenz unter $\log N - (N-1)/N$?
   :exlabel:`ex_regularized-frontier-is-concave`

<!--en:693808f2-->
1. **Eine veränderliche Referenz.** ● Die Trust-Region aus :numref:`sec_ppo` misst die Divergenz von der vorherigen Iterierten. In diesem Abschnitt haben wir gezeigt, dass die idealisierte Aktualisierung $\pi_{k+1} = \operatorname{argmax}_\pi\ \mathbb{E}_\pi[r] - \tilde{\beta}\, D_{\textrm{KL}}(\pi \Vert \pi_k)$ mit dem Startwert $\pi_0 = \pi_{\textrm{ref}}$ zum Argmax konvergiert.
    1. Fügen Sie nun wie beim Fine-Tuning nach dem Vorbild von PPO den Strafterm mit fester Referenz aus :eqref:`eq_kl_objective` hinzu. Damit maximiert $\pi_{k+1}$ den Ausdruck
       $$\mathbb{E}_\pi[r] - \beta\, D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}}) - \tilde{\beta}\, D_{\textrm{KL}}(\pi \Vert \pi_k).$$
       Zeigen Sie, dass bis auf eine Konstante gilt:
       $$\log \pi_{k+1} = \frac{\beta \log \pi_{\textrm{ref}} + \tilde{\beta} \log \pi_k + r}{\beta + \tilde{\beta}},$$
       und bestimmen Sie den Grenzwert von $\pi_k$. Welcher Koeffizient bestimmt den Grenzwert? Wie schnell nähert sich die Iteration ihm an?
    1. Zeigen Sie, dass die Rekursion aus Teil a dem Aktualisierungsschritt von `solve_kl` bei geeigneter Schrittweite entspricht. Drücken Sie diese Schrittweite durch $\beta$ und $\tilde{\beta}$ aus. Welche Schrittweiten von `solve_kl` führen zur Konvergenz, lassen sich aber nicht als Trust-Region interpretieren?
   :exlabel:`ex_regularized-moving-reference`

<!--en:e63cf181-->
1. [code] **Strafterm mit KL-Vorwärtsdivergenz.** ● In :numref:`subsec_kl_direction` vergleichen wir die Rückwärtsdivergenz aus :eqref:`eq_kl_objective` mit der Vorwärtsdivergenz. Ersetzen Sie im Strafterm die Divergenz durch die Vorwärtsdivergenz:
   $$\max_\pi\ \mathbb{E}_{a \sim \pi}\big[r(a)\big] - \beta\, D_{\textrm{KL}}\big(\pi_{\textrm{ref}} \Vert \pi\big).$$
    1. Führen Sie einen Multiplikator $m$ für die Nebenbedingung $\sum_a \pi(a) = 1$ ein und zeigen Sie, dass das Optimum $\pi(a) = \beta\, \pi_{\textrm{ref}}(a) / (m - r(a))$ lautet. Dabei ist $m > \max_a r(a)$ der einzige Wert, für den sich diese Wahrscheinlichkeiten zu eins summieren.
    1. Sei $\Delta(a) = \max_{a'} r(a') - r(a) > 0$ der Abstand der Belohnung einer suboptimalen Aktion $a$ zur besten Belohnung. Wie schnell geht für $\beta \to 0$ das Verhältnis $\pi(a) / \pi_{\textrm{ref}}(a)$ bei Verwendung der Vorwärtsdivergenz gegen null, wie schnell bei der Rückwärtsdivergenz?
    1. Berechnen Sie für den Banditen mit fünf Aktionen, `p_ref` und `r`, das Optimum bei Vorwärtsdivergenz durch Bisektion bezüglich $m$. Bestimmen Sie für beide Richtungen jeweils den Wert von $\beta$, bei dem die erwartete Belohnung im Optimum $2.9$ beträgt. Vergleichen Sie die beiden Optima für jede Aktion. Welche Aktionen erhalten bei Vorwärtsdivergenz mehr Wahrscheinlichkeit, und wie passt das zu :numref:`subsec_kl_direction`?
   :exlabel:`ex_regularized-forward-kl-penalty`

<!--en:0f5b6bb5-->
[Diskussion](https://d2l.discourse.group/t/36803)

<!--en:60efa131-->
KL-regularisierte Strategieoptimierung und Reward-Hacking<br>**Bradley-Terry-Präferenzen · Invarianz gegenüber potentialbasierter Belohnungsgestaltung · Goodharts Gesetz · exponentiell umgewichtetes Optimum · weiche Wertiteration**

<!--en:62e11ffe-->
::: {.slide title="Wenn die Belohnung aus Vergleichen gelernt werden muss"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:c9a7dd85-->
- **Ablauf von RLHF:** Mit einer Referenzstrategie $\pi_{\textrm{ref}}$ Trajektorien erzeugen, ein Belohnungsmodell $\hat{r}$ an paarweise menschliche Vergleiche $\tau \succ \tau'$ anpassen und anhand von $\hat{r}$ eine Strategie optimieren.

<!--en:000286a9-->
- **Reward-Hacking (Goodharts Gesetz):** Da $\hat{r}$ nur auf dem Träger von $\pi_{\textrm{ref}}$ genau ist, nutzt eine unregularisierte Strategie blinde Flecken außerhalb dieses Bereichs aus, sofern $-\beta D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}})$ sie nicht an die Referenz bindet.

<!--en:0a23d353-->
::: {.slide title="Eine 4×4-Gitterwelt mit verborgener wahrer Belohnung"}

<!--en:1d93fbaf-->
[Versuchsaufbau]{.kicker}

<!--en:92ce2519-->
- Zum Prüfen der gelernten Belohnung und ihrer Ausnutzung anhand der Ground Truth dient eine deterministische $4 \times 4$-Gitterwelt ($\gamma = 0.95$) mit Startzustand $0$ (S) und absorbierendem Zielzustand $15$ (G, $+1.0$).
  - Gewöhnliche Züge bringen $-0.04$, ein Zug in eines der beiden **Gefahrenfelder** ($5$ und $6$) bringt **$-1.50$**.

<!--en:6a599203-->
- Die Referenzstrategie $\pi_{\textrm{ref}}(a \mid s) \propto \exp(Q^*(s, a)/0.25)$ ist kompetent, aber zögerlich: Sie meidet den Gefahrenstreifen ($p \approx e^{-6}$). Einzelne Zugänge nutzt sie nur in $2$–$4$ von $1{,}000$ Trajektorien.

<!--en:0ba5d7c3-->
::: {.slide title="Anpassung eines Belohnungsmodells durch Bradley-Terry-Regression"}

<!--en:fe8f9b16-->
[Lernen aus Präferenzen]{.kicker}

<!--en:0029343d-->
- **Lineare Abhängigkeit von diskontierten Besuchszahlen:** Mit $c_{s,a}(\tau) = \sum_{t:(s_t,a_t)=(s,a)} \gamma^t$ beträgt der Ertrag einer Trajektorie $r(\tau) = \mathbf{c}(\tau)^\top \mathbf{r}$.

<!--en:d66b1b22-->
- **Logistische Regression:** Die Schätzung von $\hat{\mathbf{r}}$ aus $6{,}000$ Vergleichen entspricht einer $L_2$-regularisierten logistischen Regression ($\lambda_{\textrm{wd}} = 0.02$) mit den Differenzen der Besuchszahlen $\mathbf{c}(\tau) - \mathbf{c}(\tau')$.

<!--en:b1640cab-->
::: {.slide title="Wo die geschätzte Belohnung genau ist und wo sie versagt"}

<!--en:39eac7ed-->
[Datenabdeckung]{.kicker}

<!--en:c76fc4fb-->
Die geschätzte Belohnung $\hat{r}$ für alle 64 Zustand-Aktion-Paare zeigt einen deutlichen Unterschied zwischen häufig und selten besuchten Bereichen:

<!--en:87bb6358-->
**Blinder Fleck bei wenigen Daten:** Züge in die Gefahrenfelder ($5$ und $6$) kommen nur in $2$–$4$ von $1{,}000$ Episoden vor. Der Gewichtszerfall hält ihre geschätzten Belohnungen nahe **$0.00$**, obwohl sie tatsächlich **$-1.50$** einbringen.

<!--en:63dd8438-->
::: {.slide title="Warum Vergleiche potentialbasierte Belohnungsgestaltung nicht erkennen"}

<!--en:0eddf6f1-->
[Identifizierbarkeit und Belohnungsgestaltung]{.kicker}

<!--en:74c5a004-->
Warum beträgt der RMS-Fehler für die 46 häufig besuchten Paare in `core` ganze $0.193$, obwohl ein gewöhnlicher Schritt nur $0.04$ kostet?

<!--en:4698005e-->
- **Invarianz gegenüber potentialbasierter Belohnungsgestaltung:** Die Zusatzterme $\gamma \Phi(s') - \Phi(s)$ (mit $\Phi(\textrm{goal}) = 0$) bilden entlang jeder Trajektorie von $s_0$ bis zum Ziel eine Teleskopsumme. So verschiebt sich jeder Ertrag $r(\tau)$ um dieselbe Konstante $-\Phi(s_0)$.

<!--en:deeb709f-->
- Bei jedem Vergleich $r(\tau) - r(\tau')$ fällt die konstante Verschiebung $-\Phi(s_0)$ vollständig weg. Paarweise Präferenzen bestimmen $r$ also nur bis auf ein Potential $\Phi$:

<!--en:d56ea940-->
- Der Gewichtszerfall wählt den Vertreter mit der kleinsten Norm ($\Phi(0) \approx 0.78$). Gewöhnliche Züge bringen dann $\approx -0.08$. So erscheinen die unbeobachteten Züge in den Gefahrenstreifen ($\hat{r} \approx 0$) *günstiger* als normale Schritte.

<!--en:6f9406e3-->
::: {.slide title="Reward-Hacking in der Praxis: der Weg durch die Gefahrenfelder"}

<!--en:b1153ea4-->
[Goodharts Gesetz]{.kicker}

<!--en:1ba78b97-->
Die exakte Wertiteration (`plan`) liefert für die geschätzte Belohnung $\hat{r}$ eine deterministische Strategie. Ihre Bewertung sowohl unter $\hat{r}$ als auch unter der verborgenen wahren Belohnung $r^*$ ergibt:

<!--en:00918c00-->
- **Ausnutzen des blinden Flecks:** Unter $\hat{r}$ erzielt der kürzeste Weg $0 \to 1 \to 5 \to 6 \to 7 \to 11 \to 15$ durch beide Gefahrenfelder einen Score von $-0.03$ und übertrifft damit die $-0.24$ des tatsächlich optimalen Wegs. Er bringt jedoch nur **$-2.11$** unter $r^*$.

<!--en:0c50eab9-->
- Bei 24 unabhängigen Präferenzdatensätzen führt die Planung mit $\hat{r}$ in 19 Fällen durch die Gefahrenfelder und in 4 Fällen in eine Endlosschleife.

<!--en:0ea9374f-->
::: {.slide title="Die KL-regularisierte Zielfunktion für eine Entscheidung"}

<!--en:1eacb2e4-->
[KL-Regularisierung]{.kicker}

<!--en:7df0ddae-->
- **Exponentiell umgewichtetes Optimum:** Für jedes $\beta > 0$ gilt $\pi^\star(a) = \frac{1}{Z}\,\pi_{\textrm{ref}}(a)\,e^{r(a)/\beta}$ mit der Partitionsfunktion $Z = \sum_{a'} \pi_{\textrm{ref}}(a')\,e^{r(a')/\beta}$.

<!--en:046f5163-->
- **Exakter Optimalwert:** Nach der Gibbs-Ungleichung erreicht jede Strategie $\pi$ den Zielfunktionswert $\beta \log Z - \beta D_{\textrm{KL}}(\pi \Vert \pi^\star) \le \beta \log Z$.

<!--en:a321ffd1-->
::: {.slide title="Exponentielle Umgewichtung bei verschiedenen Temperaturen"}

<!--en:31c05391-->
[Geometrie der Umgewichtung]{.kicker}

<!--en:b4fd3b32-->
- Die optimale Strategie $\pi^\star(a) \propto \pi_{\textrm{ref}}(a) e^{r(a)/\beta}$ **gewichtet die Referenzverteilung neu**. Die Rangfolge der Belohnungen allein bestimmt die Aktionswahl also nicht.
  - **Erhaltung des Trägers:** Für jede Aktion mit $\pi_{\textrm{ref}}(a) = 0$ gilt weiterhin $\pi^\star(a) = 0$, unabhängig von $r(a)$.
  - **Schwache Umgewichtung ($\beta = 2$):** Der Einbruch von $\pi_{\textrm{ref}}$ bei $a_2$ bleibt in $\pi^\star$ sichtbar ($D_{\textrm{KL}} = 0.15\text{ nats}$).
  - **Starke Umgewichtung ($\beta = 0.2$):** Die Masse konzentriert sich auf die Aktion mit der höchsten Belohnung, $a_5$ ($D_{\textrm{KL}} = 1.84\text{ nats}$).

<!--en:b2c31340-->
::: {.slide title="Numerische Prüfung mit exponentiertem Gradienten"}

<!--en:7433f6af-->
[Überprüfung]{.kicker}

<!--en:abb9a2c7-->
Der iterative Gradientenaufstieg mit exponentiertem Gradienten $\pi \leftarrow \pi \odot \exp\big(\eta(r - \beta \log(\pi/\pi_{\textrm{ref}}))\big)$ bestätigt die geschlossene Lösung $\pi^\star$ und den logarithmischen Partitionsfunktionswert $\beta \log Z$ bis auf Maschinengenauigkeit:

<!--en:fcf3c642-->
::: {.slide title="Vier strukturelle Folgerungen aus der geschlossenen Form"}

<!--en:53d05ada-->
[Gemeinsame Eigenschaften]{.kicker}

<!--en:1620a229-->
| Eigenschaft / Grenzfall | Spezialfall von $\pi^\star(a) = \pi_{\textrm{ref}}(a) e^{r(a)/\beta} / Z$ | Bedeutung für das bestärkende Lernen |

<!--en:cf62aa33-->
| **1. Unregularisierter Grenzfall ($\beta \to 0$)** | Konzentration auf $\operatorname{argmax}_{a \in \textrm{supp}(\pi_{\textrm{ref}})} r(a)$ | Ohne Strafterm ist die optimale Strategie bei eindeutiger bester Aktion deterministisch |

<!--en:728a8b23-->
| **2. Starke Strafe ($\beta \to \infty$)** | $\pi^\star(a) \to \pi_{\textrm{ref}}(a)$ mit $D_{\textrm{KL}} \to 0$ | $\beta$ ermöglicht den stetigen Übergang von $\pi_{\textrm{ref}}$ zur gierigen Argmax-Auswahl |

<!--en:4117a45c-->
| **3. Gleichverteilte Referenz ($\pi_{\textrm{ref}} = 1/|\mathcal{A}|$)** | $-D_{\textrm{KL}}(\pi \Vert \textrm{unif}) = H(\pi) - \log |\mathcal{A}| \implies \pi^\star = \textrm{softmax}(r/\beta)$ | Bestärkendes Lernen mit maximaler Entropie (§{sec_sac} SAC) und Boltzmann-Exploration |

<!--en:469539a3-->
| **4. Bayessche Variationsform** | A-priori-Verteilung $\pi_{\textrm{ref}}(a) \times$ Likelihood $e^{r(a)/\beta} / Z = \textrm{Posterior } \pi^\star(a)$ | Steuerung als Inferenz und Direct Preference Optimization (§{sec_rl_sequences}) |

<!--en:4bb0eb0d-->
::: {.slide title="Die konkave Grenzkurve von Belohnung und Divergenz"}

<!--en:d58e6851-->
[Pareto-Front]{.kicker}

<!--en:f31ba800-->
- Die Variation von $\beta$ von $10^{1.5}$ bis hinunter zu $10^{-1.5}$ liefert die **Grenzkurve von Belohnung und Divergenz**. Sie zeigt die maximale erwartete Belohnung für jedes KL-Budget $D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}})$.

<!--en:ca92b4a5-->
- Die Grenzkurve ist streng monoton steigend und konkav mit Steigung $\beta$. Sie endet bei $\log(1/\pi_{\textrm{ref}}(a^*)) = \log(1/0.15) = 1.90\text{ nats}$.

<!--en:1f231e5f-->
- **Best-of-$N$-Sampling:** $N$ Samples aus $\pi_{\textrm{ref}}$ ziehen und das mit dem höchsten Score behalten. Das führt entlang derselben Achse zu $D_{\textrm{KL}} \le \log N - (N-1)/N$ ($\approx 1.83\text{ nats}$ bei $N=16$).

<!--en:cec7dca3-->
::: {.slide title="Trust-Region, Strafterm mit fester Referenz und Richtung der KL-Divergenz"}

<!--en:55980d3b-->
[Wesentliche Unterschiede]{.kicker}

<!--en:efb1c591-->
- **Veränderliche Trust-Region (§{sec_ppo}) und Strafe mit fester Referenz (§{sec_regularized}):** Eine Strafe für die Divergenz von der *vorherigen Iterierten* $\pi_k$ kumuliert die Umgewichtungen gemäß $\pi_k(a) \propto \pi_{\textrm{ref}}(a) e^{k\,r(a)/\tilde{\beta}}$. Die Iteration konvergiert weiterhin zur gierigen Argmax-Auswahl. Nur eine **eingefrorene** Referenz $\pi_{\textrm{ref}}$ ändert den Grenzwert.

<!--en:053aa4f3-->
- **Rückwärts- und Vorwärtsdivergenz:** Die Rückwärtsdivergenz $D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}})$ unterdrückt schlechte Aktionen exponentiell ($\propto e^{-\Delta(a)/\beta}$). Bei der Vorwärtsdivergenz bleibt eine lineare Untergrenze ($\approx \beta / \Delta(a)$).

<!--en:dda07cdb-->
::: {.slide title="Mehrstufige Entscheidungen: Warum mit dem weichen Wert umgewichten?"}

<!--en:77a6a3fc-->
[Von Banditen zu MDPs]{.kicker}

<!--en:1d675221-->
In einem MDP geht bei jedem Schritt der Term $-\beta D_{\textrm{KL}}(\pi(\cdot \mid s_t) \Vert \pi_{\textrm{ref}}(\cdot \mid s_t))$ in die Zielfunktion ein. Wie viel ist dann der nächste Zustand $s'$ wert?

<!--en:56269d08-->
- **Beispiel mit zwei Schritten:** Die Entscheidung zwischen 5 Aktionen bei $\beta = 2$ sei der letzte Schritt. Die beste Aktion bringt $r_{\max} = 3.0$, doch die optimale Strategie mit Strafterm erzielt den weichen Wert $V_{\textrm{next}} = \beta \log Z = 1.44$.

<!--en:dc6d0d97-->
- Einen Schritt früher wählt der Agent zwischen dem Übergang in diesen Zustand ($r=0$) und dem Beenden der Episode ($r=2$).

<!--en:26beb1ea-->
- **Folgerung:** Eine Umgewichtung der ersten Entscheidung anhand des harten Maximums $3.0$ führt zu oft in den nächsten Zustand. Der Zielfunktionswert sinkt ($1.592 < 1.740$). Jeder Schritt muss den **weichen Wert** $\beta \log Z$ verwenden, der alle künftigen Belohnungen *und* KL-Strafen berücksichtigt.

<!--en:82061e68-->
::: {.slide title="Weiche Wertiteration in Markov-Entscheidungsprozessen"}

<!--en:63e30298-->
[Das weiche Bellman-Backup]{.kicker}

<!--en:9a4ac249-->
Die geschlossene Form ersetzt in jedem Zustand das harte $\max_a$ der Wertiteration durch ein mit der Referenz gewichtetes **Log-Sum-Exp**:

<!--en:c9412cea-->
Da $m_\beta(Q) = \beta \log \sum_a \pi_{\textrm{ref}}(a) e^{Q(a)/\beta}$ monoton und verschiebungsäquivariant ist ($m_\beta(Q + c) = m_\beta(Q) + c$), ist der weiche Bellman-Operator eine **$\gamma$-Kontraktion** bezüglich $\|\cdot\|_\infty$ für jedes $\beta > 0$.

<!--en:01fdfcc2-->
::: {.slide title="Wahrer und geschätzter Ertrag in Abhängigkeit vom KL-Budget"}

<!--en:def8f603-->
[Kurve der Überoptimierung]{.kicker}

<!--en:eeb397a5-->
- Die Anwendung von `soft_v` auf die geschätzte Belohnung $\hat{r}$ bei 36 Temperaturen $\beta \in [10^{-2.5}, 10^1]$ ergibt die klassische **Kurve der Überoptimierung bei RLHF** (Gao et al., 2023).
  - **Geschätzter Ertrag (orange):** steigt monoton, wenn $\beta$ sinkt und die KL-Divergenz der Trajektorien wächst.
  - **Wahrer Ertrag (blau):** steigt von $-0.12$ ($\pi_{\textrm{ref}}$) auf **$+0.53$ bei $3.7\text{ nats}$** (nahe $V^* = +0.59$). Dann bricht er auf **$-1.83$** ein, weil die Strategie den kaum erfassten Gefahrenstreifen betritt.

<!--en:90573b49-->
::: {.slide title="Verständnisfragen zu Belohnungsgestaltung, Schwellenwerten und Best-of-N"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:d1855d6f-->
Drei Fragen mithilfe der Theorie aus diesem Abschnitt beantworten:

<!--en:6a85c1aa-->
1. **Invarianz gegenüber einer Prompt-Baseline:** Beim Lernen von LLMs aus Präferenzen lässt sich $r(x, y)$ durch $r(x, y) + f(x)$ mit einer beliebigen Funktion $f(x)$ des Prompts ersetzen. Warum bleiben sowohl die Bradley-Terry-Likelihood als auch die umgewichtete Strategie $\pi^\star(y \mid x)$ unverändert?

<!--en:9dc8b00d-->
2. **Schwellenwert für ein Schlupfloch:** Bei einem Prompt $x$ bevorzugt die Referenzstrategie eine normale gegenüber einer „Ausweichantwort“ mit Log-Odds von $4$ ($\pi_{\textrm{ref}}(\textrm{hedge})/\pi_{\textrm{ref}}(\textrm{normal}) = e^{-4}$). Ein fehlerhaftes Belohnungsmodell gibt der Ausweichantwort jedoch $+1$ mehr Belohnung. Für welche Werte von $\beta$ bevorzugt $\pi^\star$ die normale Antwort?

<!--en:2d312fd4-->
3. **Wahl von $\beta$:** Warum kann ein Algorithmus, der $\hat{r}(s,a) - \beta D_{\textrm{KL}}$ optimiert, den Höchstwert des wahren Ertrags nie anhand seiner eigenen Trainingsmetriken erkennen?

<!--en:557b98b6-->
**Lösungen:**

<!--en:b59d0328-->
1. Beide Antworten eines Vergleichs beziehen sich auf denselben Prompt $x$. Daher fällt $f(x)$ in $r(x,y) - r(x,y')$ weg. In $\pi^\star(y \mid x)$ lässt sich $e^{f(x)/\beta}$ als gemeinsamer Faktor ausklammern, der sich mit demselben Faktor in der Partitionsfunktion $Z(x)$ kürzt.

<!--en:11c99464-->
2. Für das umgewichtete Chancenverhältnis gilt $e^{-4} e^{1/\beta} < 1 \iff 1/\beta < 4 \iff \mathbf{\beta > 0.25}$.

<!--en:4c3f9ad7-->
3. Der geschätzte Ertrag $\hat{J}(\pi_\beta)$ steigt nachweislich monoton, wenn $\beta$ sinkt. Um den wahren Höchstwert zu erkennen, brauchen wir eine unabhängige, vom Training getrennte Bewertung oder ein Prüfprogramm.

<!--en:a33515aa-->
::: {.slide title="Zusammenfassung: KL-Regularisierung und Optimierung anhand von Präferenzen"}

<!--en:c5a222f8-->
[Wesentliche Ergebnisse]{.kicker}

<!--en:262c078a-->
| Konzept / Bestandteil | Mathematische Form | Aufgabe und Verhalten in der Praxis |

<!--en:617e5225-->
| **Bradley-Terry-Modell** | $P(\tau \succ \tau') = \sigma(r(\tau) - r(\tau'))$ | Schätzt die Belohnung $\hat{r}$ aus paarweisen Vergleichen bis auf Belohnungsgestaltung mit Potential $\Phi$ |

<!--en:7968d417-->
| **Reward-Hacking** | $\operatorname{argmax}_\pi J_{\hat{r}}(\pi)$ | Nutzt selten beobachtete Zustand-Aktion-Paare aus, deren Kosten $\hat{r}$ nicht erfasst hat |

<!--en:d81e9729-->
| **Umgewichtetes Optimum ($\pi^\star$)** | $\pi^\star(a) = \pi_{\textrm{ref}}(a) e^{r(a)/\beta} / Z$ | Exakter Maximierer von $\mathbb{E}_\pi[r] - \beta D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}})$ mit Wert $\beta \log Z$ |

<!--en:e877f9c3-->
| **Weiches Bellman-Backup** | $V_\beta(s) = \beta \log \sum_a \pi_{\textrm{ref}}(a \mid s) e^{Q_\beta(s,a)/\beta}$ | $\gamma$-Kontraktion, die die KL-Regularisierung auf MDPs mit mehreren Entscheidungsschritten erweitert |

<!--en:14f63fd9-->
| **Abwägung beim KL-Budget** | Mittleres $\beta$ ($3.7\text{ nats}$ in der Gitterwelt) | Erzielt eine echte Verbesserung ($+0.53$) vor dem Einbruch außerhalb der Verteilung ($-1.83$) |
