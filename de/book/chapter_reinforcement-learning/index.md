<!--en:b8107830-->
# Bestärkendes Lernen

<!--en:52cecb39-->
Bisher haben wir jedes Modell in diesem Buch mit einem festen Datensatz trainiert. Keine Vorhersage des Modells hatte Einfluss darauf, welches Beispiel als Nächstes kam. Betrachten wir nun ein Sprachmodell, das eine Mathematikaufgabe löst. Sein Lösungsweg umfasst tausend Token. Erst am Ende prüft ein Programm, ob die Antwort stimmt. Kein Token hat ein eigenes Label, und jedes Token verändert, was das Modell als Nächstes schreiben kann. Einem Regler, der einen Stab auf einem fahrenden Wagen balanciert, geht es ähnlich. Jeder Stoß verändert den Winkel, den er als Nächstes korrigieren muss. Seine einzige Belohnung (engl. reward) ist die Zeit, die der Stab aufrecht bleibt. :numref:`fig_rl_agent_env` zeigt diesen Kreislauf zwischen dem *Agenten*, der die Entscheidungen trifft, und der *Umgebung* (engl. environment), auf die er einwirkt.

<!--en:4af4acea-->
![Der Kreislauf zwischen einem Agenten und seiner Umgebung. Im Schritt $t$ wählt der Agent gemäß seiner Strategie $\pi$ eine Aktion $a_t$ (blau). Die Umgebung liefert gemäß ihren Übergangswahrscheinlichkeiten $P$ und ihrer Belohnungsfunktion $r$ eine Belohnung $r_t$ und den nächsten Zustand $s_{t+1}$ (orange). Aus der wiederholten Interaktion ergibt sich die Trajektorie $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$.](../img/mdl-rl-agent-env.svg)

<!--en:f2184032-->
Das wirft zwei Fragen auf. Welchen Entscheidungen lässt sich eine späte Belohnung zuschreiben? Und wie kann ein Agent gute Entscheidungen finden, wenn ihm niemand zeigt, welche das sind? Richard Bellman beantwortete die erste Frage in den 1950er-Jahren für den Fall, dass die Regeln bekannt sind. Er bewertete jeden Zustand anhand der noch zu erwartenden Belohnung. So zerlegte er einen langen Planungshorizont in eine Folge von Rechnungen, die jeweils nur einen Schritt vorausblicken :cite:`BellmanDPBook`. In den 1980er-Jahren lernten Andrew Barto, Richard Sutton und Chris Watkins diese Zustandswerte durch Versuch und Irrtum :cite:`Barto.Sutton.Anderson.1983,Sutton.1988,Watkins.1989`. Ronald Williams zeigte, wie sich die Entscheidungsregel, die *Strategie* (engl. policy), direkt anhand der Belohnungen verbessern lässt, die der Agent bei seinen Versuchen erhält :cite:`Williams.1992`. Da der Agent seine Daten selbst sammelt, verändern seine Fehler auch die Daten, aus denen er als Nächstes lernt. Das erklärt, warum Nachahmung allein zu kurz greift, wenn ein Modell nie übt, eigene Fehler zu korrigieren.

<!--en:18e46521-->
Wir gehen in drei Etappen vor. Zunächst beschreiben wir Markov-Entscheidungsprozesse (engl. Markov decision processes, MDP) formal (:numref:`sec_mdp`). Bei bekannten Regeln berechnen wir optimale Strategien exakt (:numref:`sec_valueiter`). Wir lernen Aktionswerte (engl. action values) aus zufällig gezogenen Übergängen (:numref:`sec_qlearning`) und bestimmen den Regret der Exploration (:numref:`sec_exploration`). In der zweiten Etappe untersuchen wir, was selbst Fachleute über lange Zeithorizonte mit ihren Demonstrationen nicht zeigen (:numref:`sec_imitation`). Wir betrachten, welche Regelverfahren die kontinuierliche dynamische Programmierung bei bekannter Physik liefert. Dazu gehören der linear-quadratische Regler (engl. linear-quadratic regulator, LQR), der iterative LQR (iLQR) und die modellprädiktive Regelung (engl. model predictive control, MPC) mit gleitendem Horizont (:numref:`sec_optimal_control`). Zum Schluss optimieren wir Strategien direkt mit Score-Funktions-Gradienten (:numref:`sec_policygradient`) und senken deren Varianz mit Baselines und gruppenrelativen Gewichten (:numref:`sec_baselines`). Wir ersetzen Nachschlagetabellen durch neuronale Netze und geben dann einen Überblick über die Verfahren dieses und des nächsten Kapitels (:numref:`sec_deeprl`).

<!--en:a2aa64c4-->
Alle Beispiele laufen in PyTorch und JAX. Die meisten sind klein genug, um jede Schätzung mit einer exakten Lösung zu vergleichen. :numref:`chap_deep_rl` schließt daran mit Kritikern an, die durch Bootstrapping lernen, sowie mit Experience-Replay, erlernten Dynamikmodellen und festen Offline-Datensätzen. Der Teil über Sprachmodelle behandelt das präferenzbasierte Post-Training ausführlicher.

<!--en:5dd576ec-->
## Materialien und weiterführende Literatur {.unnumbered}

<!--en:0124b728-->
Die folgenden frei zugänglichen Materialien bieten theoretische Darstellungen, Kurse, Implementierungen und Hinweise für empirische Untersuchungen.

<!--en:311cd888-->
**Lehrbücher**

<!--en:cabb7b05-->
- Sutton und Barto geben in *Reinforcement Learning: An Introduction* :cite:`Sutton.Barto.2018` eine umfassende Einführung in wertbasierte und strategiebasierte Verfahren, Planung und Funktionsapproximation. Für ihre grundlegenden Beiträge zum bestärkenden Lernen (engl. reinforcement learning) erhielten sie den Turing Award 2024.

<!--en:1725f7f1-->
- Szepesvári stellt in *Algorithms for Reinforcement Learning* :cite:`Szepesvari.2010` die Algorithmen dieses Kapitels und ihre Garantien auf weniger als hundert Seiten vor.

<!--en:2a027965-->
- Bertsekas entwickelt in *A Course in Reinforcement Learning* :cite:`Bertsekas.2025` das bestärkende Lernen aus der Sicht der optimalen Regelung und der dynamischen Programmierung.

<!--en:fe304d58-->
- Agarwal, Jiang, Kakade und Sun geben in *Reinforcement Learning: Theory and Algorithms* :cite:`Agarwal.Jiang.Kakade.ea.2019` die Wachstumsraten der Stichprobenkomplexität an, die wir in diesem Kapitel qualitativ beschreiben.

<!--en:e1e65e47-->
- Lattimore und Szepesvári behandeln in *Bandit Algorithms* :cite:`Lattimore.Szepesvari.2020` die gesamte Theorie, die :numref:`sec_exploration` zugrunde liegt.

<!--en:e53a40b7-->
- Zu :numref:`sec_optimal_control` finden sich Herleitungen des linear-quadratischen Reglers (LQR) bei :citet:`Kalman.1960` und :citet:`Anderson.Moore.2007`. Die differentielle dynamische Programmierung :cite:`Mayne.1966` und der iterative LQR :cite:`Li.Todorov.2004` erweitern die Riccati-Rückwärtsrekursion auf die nichtlineare Trajektorienoptimierung. :citet:`Rawlings.Mayne.Diehl.2017` behandeln die modellprädiktive Regelung mit gleitendem Horizont und Nebenbedingungen. :citet:`Tedrake.2023` behandelt unteraktuierte Roboter und Trajektorienoptimierung.

<!--en:3e1e13d6-->
**Kurse**

<!--en:13842fe7-->
- [Berkeley CS285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) behandelt auf Graduiertenniveau Policy-Gradienten, wertbasierte Verfahren, modellbasiertes Lernen und bestärkendes Offline-Lernen.

<!--en:6bc85c3c-->
- [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/) kommt diesem Kapitel thematisch am nächsten. Der Kurs behandelt die Grundlagen, die Exploration und Policy-Gradienten mit ihren Garantien.

<!--en:f4d069ff-->
- [Stanford CS224R: Deep Reinforcement Learning](https://cs224r.stanford.edu/) widmet sich der Praxis und behandelt auch Nachahmung und Offline-Verfahren.

<!--en:982afea8-->
- [CMU 10-703: Deep Reinforcement Learning and Control](https://cmudeeprl.github.io/703website/) verbindet die Algorithmen mit der Regelungstechnik und ihrer Anwendung auf Agenten mit einem physischen Körper.

<!--en:4947b987-->
- [MIT 6.7920: Reinforcement Learning: Foundations and Methods](https://web.mit.edu/6.7920/www/) entwickelt die Theorie der dynamischen Programmierung, die wir in :numref:`sec_valueiter` einführen.

<!--en:ce8a4e45-->
**Kommentierte Implementierungen**

<!--en:217fb7cd-->
- [CleanRL](https://github.com/vwxyzjn/cleanrl) :cite:`Huang.Dossa.Ye.ea.2022` implementiert die wichtigsten nach :numref:`sec_deeprl` behandelten Algorithmen für tiefes bestärkendes Lernen in jeweils einer Datei und prüft sie mit Benchmarks.

<!--en:ed5acff1-->
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) :cite:`Raffin.Hill.Gleave.ea.2021` stellt dieselben Algorithmen als Bibliothek bereit. Die zugehörige Sammlung enthält abgestimmte Hyperparameter.

<!--en:76ce63c4-->
- [Gymnasium](https://gymnasium.farama.org/) :cite:`Towers.Kwiatkowski.Terry.ea.2024` dokumentiert die Schnittstelle zur Umgebung, die wir in diesen Kapiteln durchgehend verwenden.

<!--en:c6b3100d-->
**Empirische Praxis**

<!--en:beeca0d6-->
- [Spinning Up in Deep RL](https://spinningup.openai.com/) :cite:`Achiam.2018` ist ein kurzer Lehrgang von der Score-Funktion bis zu SAC. Er ist so aufgebaut, dass man den Code parallel zur Lektüre ausführen kann.

<!--en:e4f655c4-->
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) :cite:`Huang.Dossa.Raffin.ea.2022` dokumentiert Implementierungsentscheidungen, die in den Beschreibungen des Algorithmus fehlen, und zeigt, wie sie die Güte von PPO und die Reproduzierbarkeit beeinflussen.

<!--en:b7b234df-->
Grundlagen des bestärkenden Lernens<br>**Markov-Entscheidungsprozesse · dynamische Programmierung · zeitliche Differenzen · Exploration · Nachahmung · optimale Regelung · Policy-Gradienten · Varianzreduktion**

<!--en:649a629f-->
::: {.slide title="Aktionen bestimmen, welche Daten der Agent als Nächstes sieht"}

<!--en:8619d0ac-->
[Der Kreislauf sequenzieller Entscheidungen]{.kicker}

<!--en:d613a313-->
Beim überwachten Lernen steht der Datensatz fest. Bei sequenziellen Entscheidungen verändert der Agent die künftigen Zustände und bestimmt damit, welche Daten als Nächstes eintreffen:

<!--en:690f2004-->
- **Strategie $\pi(a_t \mid s_t)$:** Ordnet dem aktuellen Zustand $s_t$ eine Aktion $a_t$ zu (etwa ein Motordrehmoment, einen Zug im Gitter oder ein generiertes Token).

<!--en:d78787d6-->
- **Dynamik der Umgebung $P(s_{t+1} \mid s_t, a_t)$:** Bestimmt den Folgezustand stochastisch und liefert eine skalare Belohnung $r_t$.

<!--en:8a74da1b-->
- **Rückkopplung im geschlossenen Kreis:** Die wiederholte Interaktion erzeugt eine Trajektorie $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$, deren Verteilung von $\pi$ abhängt.

<!--en:b64e2aef-->
::: {.slide title="Zuordnung des Erfolgs bei später Belohnung und Suche durch Versuch und Irrtum"}

<!--en:cf471910-->
[Zwei zentrale Herausforderungen]{.kicker}

<!--en:d511f929-->
- **Zeitliche Zuordnung des Erfolgs:** Welche früheren Entscheidungen waren gut, welche schlecht, wenn die Belohnung erst am Ende einer langen Aktionsfolge eintrifft?

<!--en:26aa3d27-->
- **Bellman-Prinzip (1957):** Bewerten wir jeden Zustand nach seinem erwarteten diskontierten künftigen Ertrag, so genügt für ein Problem mit Horizont $1/(1-\gamma)$ eine Folge von Vorausschaurechnungen über jeweils einen Schritt.

<!--en:1019ab95-->
- **Exploration ohne Lehrer:** Aktionen mit hohem Wert nutzen und zugleich Aktionen erproben, deren Wert noch unsicher ist.

<!--en:84417a2a-->
::: {.slide title="Benchmark 1: Auf glattem Eis ist der kürzeste Weg nicht mehr der beste"}

<!--en:4a73abd6-->
[Benchmark für einen tabellarischen MDP]{.kicker}

<!--en:a18fa7ed-->
Unser erster durchgängiger Benchmark ist die $4 \times 4$ Felder große Umgebung `FrozenLake-v1`. Sie ist klein genug für eine Lösung bis auf Maschinengenauigkeit, doch die zufälligen Übergänge lassen einfache gierige Heuristiken scheitern:

<!--en:3396d493-->
- **Übergangskern auf glattem Eis:** Bei jedem Befehl bewegt sich der Agent mit Wahrscheinlichkeit $1/3$ in die beabsichtigte Richtung. Mit Wahrscheinlichkeit $1/3$ je Seite rutscht er seitlich im rechten Winkel dazu.

<!--en:e58ddad8-->
- **Kürzester Weg und Optimum bei Rutschgefahr:** Ohne Rutschen ist der Weg mit 6 Zügen der kürzeste. Auf glattem Eis führt er nur in **$4.5\%$** der Episoden zum Ziel, weil der Agent in benachbarte Löcher rutscht.

<!--en:9e23e6ba-->
- **Risikominimierende Regelung:** Mit der optimalen Strategie $\pi^*$ erreicht der Agent in **$73.0\%$** der Episoden das Ziel. In vier Feldern weist sie vom Ziel *weg*, um das Risiko eines Sturzes in ein Loch zu minimieren.

<!--en:6e4f071c-->
::: {.slide title="Benchmark 2: Sprachgenerierung als MDP mit Prüfprogramm"}

<!--en:015b2589-->
[Benchmark für das Post-Training von Sprachmodellen]{.kicker}

<!--en:80215e22-->
- **Zusammenfassung zu einem Schritt:** Bei deterministischem Anhängen der Token und terminaler Belohnung $r(x, y) \in \{0, 1\}$ gilt eine ganze Antwort $y$ als eine einzige Aktion zum Prompt $x$.

<!--en:ee8e9eb3-->
- **Referenz und Obergrenze:** Die gleichverteilte Referenzstrategie mit Softmax-Funktion besteht die Prüfung in $6.2\%$ der Fälle (Erfolgswahrscheinlichkeit $\approx 1/8$ bei `2+2` und `3*4`, $0$ bei den unlösbaren Aufgaben `8*8` und `9*9`). Die Obergrenze liegt bei $50\%$.

<!--en:c0cf26e0-->
- **Gefahr durch Reward-Hacking:** Eine neunte Antwort lässt alle Möglichkeiten offen (*„4, 12, 64 oder 81“*). Ein exaktes Prüfprogramm vergibt dafür $0$, ein großzügiges dagegen $1.0$ bei jedem Prompt, weil ihm eine passende Teilzeichenkette genügt.

<!--en:b9813a17-->
::: {.slide title="Überblick über Kapitel §{num:chap_reinforcement_learning}: Von exakten Tabellen zu neuronalen Kritikern"}

<!--en:deaa4fa5-->
[Architekturen und Algorithmen im Kapitelverlauf]{.kicker}

<!--en:c7209fe6-->
- **Von bekannter zu unbekannter Physik (§§{sec_mdp}–§{num:sec_exploration}):** MDPs formulieren, die Bellman-Kontraktion iterieren, $Q^*$ aus einzelnen Samples lernen und den Regret der Exploration mit UCB und Thompson-Sampling beschränken.

<!--en:20140ac7-->
- **Demonstrationen und kontinuierliche Physik (§§{sec_imitation}–§{num:sec_optimal_control}):** Die Verteilungsverschiebung beim Klonen und ihre Folgen der Ordnung $O(\varepsilon T^2)$ untersuchen (DAgger). Kontinuierliche lineare und nichtlineare Regelungsprobleme mit Riccati-LQR, iLQR und MPC mit gleitendem Horizont lösen.

<!--en:92eabaa7-->
- **Policy-Gradienten und neuronale Kritiker (§§{sec_policygradient}–§{num:sec_deeprl}):** Gradienten für stochastische Strategien mithilfe von REINFORCE, Baselines, RLOO, GRPO und MLP-Kritikern mit `tanh` berechnen.

<!--en:9d54645d-->
::: {.slide title="Verständnisfragen zur Formulierung sequenzieller Entscheidungsprobleme"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:ceb8c25b-->
Beantworten Sie diese drei grundlegenden Fragen, bevor Sie mit §{sec_mdp} beginnen:

<!--en:49386d38-->
1. **Zuordnung des Erfolgs und Diskontierung:** Welcher Diskontierungsfaktor $\gamma$ ergibt einen effektiven Planungshorizont $1/(1-\gamma)$ von $50$ Schritten, und welches Gewicht erhält eine Belohnung, die erst nach $50$ Schritten eintrifft?

<!--en:e757b251-->
2. **Stochastische Regelung:** In FrozenLake mit glattem Eis liegt Loch 5 direkt über Feld 9. Warum wählt die optimale Strategie auf Feld 9 *unten*, obwohl der Agent bei *unten* in zwei von drei Fällen zur Seite rutscht?

<!--en:096f282a-->
3. **Festlegung der Belohnung:** Warum versagt bei der Rechenaufgabe mit 4 Prompts ein großzügiges Prüfprogramm, das die richtige Antwort schon als Teilzeichenkette akzeptiert?

<!--en:557b98b6-->
**Antworten:**

<!--en:045bda33-->
1. Aus $1/(1-\gamma) = 50$ folgt $\gamma = 0.98$. Eine Belohnung am Ende dieses Horizonts erhält das Gewicht $0.98^{50} \approx e^{-1} \approx 0.368$.

<!--en:8a8ac5e9-->
2. Bei den Befehlen *links*, *rechts* und *oben* gerät der Agent von Feld 9 aus jeweils mit Wahrscheinlichkeit $1/3$ in Loch 5. Beim Befehl *unten* beträgt das Risiko, im nächsten Schritt in ein Loch zu fallen, $0$. Der Agent landet auf einem der sicheren Felder 8, 13 oder 10.

<!--en:e0662bdd-->
3. Der Optimierer nutzt die Spezifikation aus und legt sich bei keinem Prompt fest. Mit der Antwort *„eine der Zahlen 4, 12, 64 oder 81“* erhält er jedes Mal $1.0$, ohne die Aufgabe zu lösen.
