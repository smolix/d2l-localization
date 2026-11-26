<!--en:a7888498-->
# Markov-Entscheidungsprozesse

<!--en:27dedb7f-->
Stellen Sie sich einen Roboter vor, der einen zugefrorenen See überquert, ein Programm, das ein Brettspiel spielt, oder ein Sprachmodell, das Token für Token eine Antwort schreibt. Jede Entscheidung beeinflusst, was als Nächstes geschieht. Der Roboter rutscht in eine andere Zelle, auf dem Spielbrett entsteht eine andere Stellung, die Antwort nimmt eine andere Wendung. Die Belohnung (engl. reward) kommt vielleicht erst am Ende, und derselbe Befehl kann Verschiedenes bewirken. Bevor wir fragen können, wie man gut handelt, müssen wir das Problem formulieren können. Was weiß der Entscheidungsträger in jedem Schritt, was kann er tun, was folgt daraus und was möchte er erreichen?

<!--en:605bae1d-->
Bellman formulierte diese Beschreibung 1957 in einer Arbeit über Markov-Entscheidungsprozesse :cite:`BellmanMDP`. Seine Idee war, alles Vergangene, was für die Zukunft zählt, in einem *Zustand* (engl. state) zusammenzufassen. Dann hängen die Folgen einer Aktion nur von dem Zustand ab, in dem sie ausgeführt wird. Diese eine Annahme macht das Problem lösbar. Hängt die Zukunft allein vom gegenwärtigen Zustand ab, genügt für jeden Zustand eine Zahl, die angibt, wie gut es ist, dort zu sein. Das daraus entstandene Modell, der *Markov-Entscheidungsprozess* (engl. Markov decision process, MDP), gehört heute zu den Standardwerkzeugen des Operations Research :cite:`Puterman.1994`. Alle Verfahren in diesem und im nächsten Kapitel arbeiten mit diesem Modell, von der dynamischen Programmierung bis zum Training von Sprachmodellen mit einer Belohnung.

<!--en:14dcc2bc-->
Wir betrachten einen zugefrorenen See mit sechzehn Zellen. Er ist klein genug, um jede Größe von Hand nachzurechnen. Dennoch zeigt sich hier, was Entscheidungen über mehrere Schritte hinweg schwierig macht. Auf glattem Eis rutscht der Agent oft an eine andere Stelle als beabsichtigt. Wir lesen die Übergangstabelle des Sees aus einem Simulator aus und benennen ihre Bestandteile. Dann definieren wir den Ertrag (engl. return), den ein Agent maximiert, und die Diskontierung, die ihn endlich hält. Wir fragen, was ein Zustand enthalten muss, und untersuchen einen scheinbar hilfreichen Bonus, mit dem der Agent dafür belohnt wird, im Kreis zu laufen. Zum Abschluss betrachten wir ein zweites, noch kleineres Beispiel. Ein Sprachmodell löst Rechenaufgaben, und ein Prüfprogramm bewertet seine Antworten.

<!--en:5af97cd0-->
In diesem Kapitel verwenden wir die Symbole aus :numref:`tab_rl_notation`. Die Symbole für diesen Abschnitt definieren wir im Folgenden. Die übrigen führen wir in späteren Abschnitten jeweils bei der ersten Verwendung ein.

<!--en:cbdc8075-->
:Symbole der Kapitel zum bestärkenden Lernen. Die Zeilen ab $V^\pi$ gehören zu späteren Abschnitten, die Zeilen ab $\lambda$ zu :numref:`chap_deep_rl`. Die Explorationsrate $\epsilon$ und die Wahrscheinlichkeit einer Abweichung pro Schritt $\varepsilon$ haben unterschiedliche Symbole, ebenso eine Antwort $y$ und ein Zielwert $y_t$. Ein $\mu$ mit einem anderen Index oder Argument als 0 bezeichnet einen Mittelwert (eines Arms oder einer Gauß-Verteilung).

<!--en:b85a0299-->
| Symbol | Bedeutung | Eingeführt in |

<!--en:0307601f-->
| $\mathcal{S}$, $\mathcal{A}$ | Menge der Zustände und Menge der Aktionen | :numref:`sec_mdp` |

<!--en:079160cc-->
| $s_t$, $a_t$, $r_t$ | Zustand, Aktion und Belohnung in Schritt $t$ | :numref:`sec_mdp` |

<!--en:a40610e8-->
| $P(s' \mid s, a)$ | Übergangskern: Wahrscheinlichkeit, $s'$ von $s$ aus mit Aktion $a$ zu erreichen | :numref:`sec_mdp` |

<!--en:d156a5bb-->
| $r(s, a)$ | erwartete Belohnung für Aktion $a$ im Zustand $s$ | :numref:`sec_mdp` |

<!--en:5213c7cd-->
| $\mu_0$ | Verteilung der Startzustände (bei Sprachmodellen die Verteilung der Prompts) | :numref:`sec_mdp` |

<!--en:43c1aabd-->
| $\tau$ | Trajektorie $(s_0, a_0, r_0, s_1, \ldots)$ | :numref:`sec_mdp` |

<!--en:231f46fa-->
| $\gamma$ | Diskontierungsfaktor | :numref:`sec_mdp` |

<!--en:8ceb56e8-->
| $R(\tau)$ | Ertrag einer Trajektorie, $\sum_t \gamma^t r_t$ | :numref:`sec_mdp` |

<!--en:bb29ec48-->
| $T$ | Anzahl der Schritte einer Episode oder Zeitlimit | :numref:`sec_mdp` |

<!--en:8a813b15-->
| $\pi(a \mid s)$, $\pi(s)$ | Strategie: Wahrscheinlichkeit für Aktion $a$ im Zustand $s$ bzw. Aktion einer deterministischen Strategie | :numref:`sec_mdp`, :numref:`sec_valueiter` |

<!--en:225a85fc-->
| $\ell(s)$ | Manhattan-Distanz von Zelle $s$ zum Ziel | :numref:`sec_mdp` |

<!--en:e3d7dd82-->
| $\Phi(s)$ | Potential eines Bonus zur Belohnungsformung | :numref:`sec_mdp` |

<!--en:87877971-->
| $x$, $y$, $r(x, y)$ | Prompt, Antwort und Score des Prüfprogramms für die Antwort | :numref:`sec_mdp` |

<!--en:5a674f8a-->
| $y_t$ | Zielwert einer Wertaktualisierung: Schätzung des Ertrags ab Schritt $t$ | :numref:`sec_mdp`, :numref:`sec_deeprl` |

<!--en:91a41a6a-->
| $V^\pi$, $Q^\pi$ | erwarteter Ertrag unter $\pi$ ausgehend von einem Zustand bzw. einem Zustand-Aktion-Paar | :numref:`sec_valueiter` |

<!--en:e6ded195-->
| $V^*$, $Q^*$ | dieselben Größen unter einer optimalen Strategie | :numref:`sec_valueiter` |

<!--en:8c11aace-->
| $\mathcal{T}$ | Bellman-Operator: wendet einen Schritt Vorausschau auf einen Vektor von Werten an | :numref:`sec_valueiter` |

<!--en:fa206ca0-->
| $\delta$, $\delta_t$ | Temporal-Difference-Fehler eines Übergangs bzw. des Übergangs in Schritt $t$ | :numref:`sec_qlearning` |

<!--en:82c31a67-->
| $\alpha$ | Schrittweite bei Temporal-Difference-Updates | :numref:`sec_qlearning` |

<!--en:4c029b2c-->
| $\epsilon$ | Explorationsrate der $\epsilon$-gierigen Aktionswahl | :numref:`sec_qlearning` |

<!--en:e11f752d-->
| $\pi_e$ | Verhaltensstrategie, die während des Lernens die Aktionen auswählt | :numref:`sec_qlearning` |

<!--en:3927874f-->
| $\kappa$ | Gewicht des Unsicherheitsbonus | :numref:`sec_exploration` |

<!--en:584b347b-->
| $K$ | Anzahl der pro Prompt gezogenen Antworten | :numref:`sec_exploration` |

<!--en:e1b8ce27-->
| $\pi_\theta$, $\theta$ | Strategie mit Parametern $\theta$ | :numref:`sec_mdp` (Präferenztabelle der Aufgabe mit Prüfprogramm), :numref:`sec_imitation` |

<!--en:3f8ff7e5-->
| $d^\pi_t$, $d^\pi_\gamma$ | Verteilung von $s_t$ unter $\pi$ bzw. ihr diskontierter Mittelwert über $t$ | :numref:`sec_imitation`, :numref:`sec_policygradient` |

<!--en:eea187c1-->
| $\varepsilon$ | Wahrscheinlichkeit pro Schritt, dass eine geklonte Strategie anders entscheidet als der Experte | :numref:`sec_imitation` |

<!--en:b347be6c-->
| $\mathbf{x}_t$, $\mathbf{u}_t$, $\mathbf{A}$, $\mathbf{B}$, $\mathbf{Q}$, $\mathbf{R}$ | kontinuierlicher Zustand und Stellgröße, Matrizen der linearen Dynamik und der quadratischen Kosten | :numref:`sec_optimal_control` |

<!--en:397f59fe-->
| $\mathbf{P}_t$, $\mathbf{K}_t$ | Riccati-Matrix der Restkosten und Verstärkungsmatrix der linearen Zustandsrückführung | :numref:`sec_optimal_control` |

<!--en:7cc70e5c-->
| $J(\theta)$ | erwarteter Ertrag von $\pi_\theta$ | :numref:`sec_policygradient` |

<!--en:8f564ab1-->
| $\eta$ | Schrittweite bei der Aktualisierung der Strategie | :numref:`sec_policygradient` |

<!--en:d11059f4-->
| $\hat{G}_t$ | künftiger Ertrag: diskontierte Summe der Belohnungen ab Schritt $t$ | :numref:`sec_baselines` |

<!--en:1276adcb-->
| $b$ | Baseline, die vom Ertrag abgezogen wird | :numref:`sec_baselines` |

<!--en:ac46b7c5-->
| $\eta_V$ | Schrittweite bei der Aktualisierung der wertbasierten Baseline | :numref:`sec_baselines` |

<!--en:7fb3e3d5-->
| $A^\pi$ | Vorteil $Q^\pi - V^\pi$ | :numref:`sec_baselines` |

<!--en:f7956911-->
| $\lambda$ | Gewicht des $\lambda$-Ertrags | :numref:`sec_actorcritic` |

<!--en:e9240bb3-->
| $w$, $w^-$ | Parameter eines Wertnetzes bzw. seiner eingefrorenen oder nach Polyak gemittelten Kopie | :numref:`sec_actorcritic`, :numref:`sec_dqn` |

<!--en:84e64aef-->
| $n$ | Tiefe eines $n$-Schritt-Ziels oder eines Plans | :numref:`sec_actorcritic`, :numref:`sec_model_based` |

<!--en:34d191f4-->
| $\rho_t$ | Wahrscheinlichkeitsverhältnis $\pi_\theta(a_t \mid s_t) / \pi_{\theta_{\textrm{old}}}(a_t \mid s_t)$ in Schritt $t$ | :numref:`sec_ppo` |

<!--en:89798223-->
| $\epsilon_{\textrm{clip}}$ | halbe Breite des Intervalls, auf das das Verhältnis begrenzt wird | :numref:`sec_ppo` |

<!--en:6f8a7324-->
| $\Delta_{\textrm{KL}}$ | KL-Budget pro Aktualisierung | :numref:`sec_ppo` |

<!--en:9f5ed2c0-->
| $\pi_{\textrm{ref}}$, $\beta$ | Referenzstrategie sowie Gewicht des KL-Strafterms bzw. Temperatur bei Soft Actor-Critic | :numref:`sec_regularized`, :numref:`sec_sac` |

<!--en:30678428-->
| $\hat{r}$ | an Daten angepasste Belohnung | :numref:`sec_regularized`, :numref:`sec_model_based` |

<!--en:a997f39f-->
| $\nu$ | Rate der Polyak-Mittelung eines Zielnetzes | :numref:`sec_sac` |

<!--en:0ede9852-->
| $\hat{f}_\phi$ | gelerntes Übergangsmodell | :numref:`sec_model_based` |

<!--en:91e0cf09-->
## Das Modell

<!--en:64dba2ad-->
:numref:`fig_rl_gridworld` zeigt den See. Wir verwenden FrozenLake aus der Gymnasium-Sammlung :cite:`Towers.Kwiatkowski.Terry.ea.2024` in allen Abschnitten dieses Kapitels, die mit Tabellen arbeiten. Der Agent startet in der Zelle oben links und versucht, das Ziel unten rechts zu erreichen. Dort erhält er eine Belohnung von eins. Vier Zellen sind Löcher. Sobald der Agent ein Loch oder das Ziel betritt, endet der Versuch. Auf *glattem* Eis bewegt sich der Agent bei einem Befehl mit Wahrscheinlichkeit $1/3$ in die gewählte Richtung und mit Wahrscheinlichkeit $1/3$ in jede der beiden Richtungen senkrecht dazu. Würde er den Rand des Gitters überschreiten, bleibt er an Ort und Stelle.

<!--en:dddc666a-->
![Der See. (a) S bezeichnet den Start, G das Ziel. Die grauen Zellen mit H sind Löcher. Jede Zelle trägt ihren Zustandsindex. Ohne Rutschen führt der kürzeste Weg in sechs Zügen zum Ziel. (b) Ein Befehl auf glattem Eis. Aus Zustand $s = 9$ führt die Aktion *unten* mit Wahrscheinlichkeit $1/3$ in jede der Zellen 13, 8 und 10. Der beabsichtigte Zug ist eines von drei gleich wahrscheinlichen Ergebnissen. Ihre Wahrscheinlichkeiten summieren sich zu eins.](../img/mdl-rl-gridworld.svg)

<!--en:afd6c398-->
Betrachten wir den Agenten in Zelle 9, drei Züge vom Ziel entfernt. Bei drei seiner vier Befehle rutscht er in je einem von drei Fällen in das Loch in Zelle 5. Nur bei *unten* ist das ausgeschlossen. Allerdings rutscht er bei *unten* in zwei von drei Fällen seitwärts, in Zelle 8 oder 10. Lohnt sich die Sicherheit von *unten* trotz der Schritte, die der Agent durch seitliches Rutschen verliert? Um die Befehle zu vergleichen, müssen wir wissen, wohin jeder führen kann, welche Möglichkeiten der Agent dort hat und was er am Ende erhält.

<!--en:c86607c2-->
### Zustände, Aktionen und Übergangskern

<!--en:de0edf06-->
Der *Zustand* des Sees ist die Zelle, in der sich der Agent befindet. Die Zustandsmenge $\mathcal{S} = \{0, 1, \ldots, 15\}$ nummeriert die Zellen zeilenweise. Die *Aktionen* sind die vier Befehle *links*, *unten*, *rechts* und *oben*. Sie sind mit den Zahlen 0 bis 3 kodiert und bilden die Menge $\mathcal{A}$. Auf glattem Eis legt eine Aktion eine Verteilung über die Folgezustände fest. Gymnasium speichert diese Verteilung in einer Tabelle, die wir direkt auslesen können:

<!--en:be322be4-->
Die Ausgabe enthält drei Ergebnisse der Form (Wahrscheinlichkeit, Folgezustand, Belohnung, terminiert). Beim Befehl *unten* in Zelle 9 erreicht der Agent mit Wahrscheinlichkeit $1/3$ jede der Zellen 8, 13 und 10, wie in :numref:`fig_rl_gridworld`. Mathematisch beschreibt diese Tabelle den *Übergangskern* (engl. transition kernel) $P(s' \mid s, a)$. Er gibt an, mit welcher Wahrscheinlichkeit der Agent Zustand $s'$ erreicht, wenn er in Zustand $s$ die Aktion $a$ ausführt. Für jedes Paar $(s, a)$ bilden die Zahlen $P(\cdot \mid s, a)$ eine Verteilung, also gilt $\sum_{s' \in \mathcal{S}} P(s' \mid s, a) = 1$.

<!--en:88ada88a-->
Der vierte Bestandteil ist die *Belohnung*. Der Simulator vergibt eins für den Schritt ins Ziel und null für jeden anderen Schritt. Die Belohnung hängt also davon ab, wo der Schritt endet. Wir verwenden ihren Mittelwert über die möglichen Zielzellen, die *erwartete Belohnung* $r(s, a)$ der Aktion $a$ im Zustand $s$. Wie :exref:`ex_mdp-expected-rewards` zeigt, ändert die Mittelung keinen erwarteten Ertrag. Zusammen bilden diese vier Objekte den MDP:

<!--en:dd4a73a5-->
Die ersten drei beschreiben die Umgebung. Die Belohnung beschreibt die Aufgabe. Wer die Aufgabe stellt, legt auch die Belohnung fest. Auf diesem Unterschied beruht der Abschnitt zum Entwurf der Belohnung. Manche Darstellungen ergänzen das Tupel um den unten definierten Diskontierungsfaktor oder lassen eine zufällige Belohnung oder eine Abhängigkeit vom Folgezustand zu. Dann entspricht unsere Größe $r(s, a)$ der erwarteten Belohnung. Die einfachere Form schränkt die Allgemeinheit der Ergebnisse in diesem Kapitel nicht ein.

<!--en:c44b9194-->
## Ertrag, Diskontierung und Horizont

<!--en:236a4f98-->
Der Agent beginnt in einem Zustand $s_0$, den wir aus einer *Verteilung der Startzustände* (engl. start-state distribution) $\mu_0$ ziehen. Beim See liegt deren gesamte Masse auf Zelle 0. In jedem Schritt $t$ befindet sich der Agent im Zustand $s_t$, führt Aktion $a_t$ aus, erhält die Belohnung $r_t$ und wechselt in einen Folgezustand $s_{t+1} \sim P(\cdot \mid s_t, a_t)$. Die Folge dieser Größen heißt *Trajektorie*,

<!--en:82c4e386-->
Die Belohnung $r_t$ einer Trajektorie ist die tatsächlich vom Simulator vergebene Belohnung. Die erwartete Belohnung $r(s_t, a_t)$ ist ihr Mittelwert bei gegebenem $s_t$ und $a_t$.

<!--en:0ceb0d36-->
Eine Regel für die Wahl der Aktionen heißt *Strategie* (engl. policy). Mit $\pi(a \mid s)$ bezeichnen wir die Wahrscheinlichkeit, dass sie Aktion $a$ im Zustand $s$ wählt. Die einfachste Strategie wählt unabhängig vom Zustand jeden der vier Befehle mit Wahrscheinlichkeit $1/4$. Wir führen sie einmal von Zelle 0 aus und geben jeden Schritt aus:

<!--en:13c36f33-->
Das Protokoll zeigt, wie der Übergangskern wirkt. Bei $t=0$ gibt der Agent den Befehl *rechts* und bleibt in Zelle 0, weil er auf dem Eis gegen die obere Wand rutscht. Bei $t=2$ wählt er *links* und rutscht nach *unten* in Zelle 4. Bei $t=8$ wählt er in Zelle 10 *rechts* und gerät in das Loch in Zelle 11. Damit endet der Versuch mit `terminated=True`. Da kein Schritt ins Ziel geführt hat, sind alle Belohnungen null. Um eine Trajektorie zu bewerten, die das Ziel erreicht, müssen wir zunächst festlegen, wie viel eine späte Belohnung zählt.

<!--en:89cd0f49-->
### Diskontierung und effektiver Horizont

<!--en:1e7e4a89-->
Warum genügt es nicht, die Belohnungen zu addieren? Eine Trajektorie kann unendlich lang sein. Mit der Strategie, stets *oben* zu wählen, verlässt der Agent nie die oberste Zeile des Sees. Bei *oben* stößt er entweder gegen die Wand oder rutscht seitlich entlang der Zeile. Bei einer positiven Belohnung in jedem Schritt wächst die Summe entlang einer solchen Trajektorie unbegrenzt. Außerdem ist eine sofortige Belohnung meist mehr wert als die gleiche Belohnung zu einem späteren Zeitpunkt. Beide Überlegungen führen zu einem *Diskontierungsfaktor* $0 \leq \gamma < 1$, der die Belohnung in Schritt $t$ mit $\gamma^t$ gewichtet. Der *Ertrag* einer Trajektorie ist die diskontierte Summe

<!--en:db6a06ea-->
Bei $\gamma = 0.95$ hat eine Belohnung nach einem Schritt das Gewicht $0.95$, nach zwanzig Schritten nur noch etwa $0.36$. Der Agent soll so handeln, dass sein Ertrag im Mittel möglichst groß ist. Das Mittel umfasst die zufälligen Übergänge, den zufälligen Startzustand und die Entscheidungen des Agenten. Wir maximieren diesen *erwarteten Ertrag*, weil er dem mittleren Ertrag vieler unabhängiger Versuche entspricht. Für einen endlichen MDP mit $\gamma < 1$ lässt sich die Frage klar beantworten :cite:`Puterman.1994`. Es gibt eine Strategie, die für alle Startzustände zugleich optimal ist. Wir können sie deterministisch und zeitunabhängig wählen, mit einem Befehl pro Zelle. Die optimale Strategie hängt also nicht davon ab, wo der Agent startet. Daher verlieren wir nichts, wenn wir die Verteilung der Startzustände auf eine Zelle konzentrieren.

<!--en:bbce0b99-->
Die Diskontierung bestimmt auch, wie weit der Agent vorausschaut. Gilt für jede Belohnung $|r_t| \leq r_{\max}$, so bleibt der Ertrag durch die Diskontierung endlich. Auch der Anteil nach Schritt $k$ ist beschränkt:

<!--en:1fc8faae-->
Die erste Schranke hält die Zielgröße endlich. Nach der zweiten können die Belohnungen nach Schritt $k$ den Ertrag höchstens um den Bruchteil $\gamma^k$ seines maximal möglichen Betrags verändern. Das legt nahe, die Reichweite der Diskontierung in Schritten zu messen, und zwar durch den *effektiven Horizont* $1/(1-\gamma)$. Liegt $\gamma$ nahe bei eins, gilt $\gamma^{1/(1-\gamma)} \approx e^{-1}$. Pro Horizont fallen die Gewichte also etwa um den Faktor $e$. Bei einer konstanten Folge von Belohnungen entfallen dann 95 Prozent des diskontierten Gewichts auf die ersten $\log(0.05)/\log(\gamma)$ Schritte, also auf etwa drei Horizonte. Die folgende Tabelle nennt beide Schrittzahlen für vier Diskontierungsfaktoren. Sie zeigt außerdem das Gewicht $\gamma^5$ der Belohnung am Ziel, wenn der Agent ohne Rutschen den kürzesten Weg von sechs Zügen nimmt:

<!--en:7846e600-->
Bei $\gamma = 0.95$, dem Diskontierungsfaktor in diesem Kapitel, beträgt der Horizont 20 Schritte. Auf die ersten 59 Schritte entfallen 95 Prozent des Gewichts, und das in sechs Zügen erreichte Ziel ist $\gamma^5 \approx 0.774$ wert. Bei $\gamma = 0.5$ ist dasselbe Ziel nur etwa $0.03$ wert. :numref:`fig_rl_return_discount` zeigt diese Größen. Die Diskontierung gehört zur Aufgabe. Wer die Aufgabe entwirft, wählt auch den Faktor. Ein kleines $\gamma$ macht den Agenten kurzsichtig. Wie :exref:`ex_mdp-planning-discount` zeigt, kann es aber die Planung mit einem ungenauen Modell zuverlässiger machen.

<!--en:cf703a0f-->
![Die Diskontierung legt einen effektiven Horizont fest. (a) Das Gewicht $\gamma^t$ einer Belohnung nach $t$ Schritten fällt unter $0.05$: bei $t = 5$ für $\gamma = 0.5$, bei $t = 29$ für $\gamma = 0.9$, bei $t = 59$ für $\gamma = 0.95$ und erst bei $t = 299$ für $\gamma = 0.99$. (b) Der Horizont $1/(1-\gamma)$ auf einer logarithmischen Achse: zwei Schritte bei $\gamma = 0.5$, zwanzig bei $\gamma = 0.95$ und hundert bei $\gamma = 0.99$.](../img/mdl-rl-return-discount.svg)

<!--en:649924ce-->
### Das Modell in Arrayform

<!--en:a2a719fd-->
Die dynamische Programmierung greift immer wieder auf den Übergangskern, die Belohnung und den Diskontierungsfaktor zu. In allen späteren Abschnitten mit tabellarischen Verfahren prüfen wir die Lernergebnisse anhand dieser exakten Berechnungen. Das von Gymnasium bereitgestellte Wörterbuch mit Tupeln ist dafür unhandlich. Wir möchten $P$ in einem Array mit den Indizes $(s, a, s')$ und $r$ in einem Array mit den Indizes $(s, a)$ speichern. So lässt sich ein Mittelwert über die Folgezustände als ein einziges Matrixprodukt berechnen. `TabularMDP` fasst die drei Objekte zusammen. Die Methode `from_gym` erzeugt die Arrays aus der Tabelle des Simulators. Sie summiert die Wahrscheinlichkeiten der Ergebnisse, die in denselben Folgezustand führen, und mittelt die Belohnung über die Ergebnisse. Das ergibt die erwartete Belohnung $r(s, a)$. Wir speichern die Klasse in der Bibliothek `d2l`, da sie in allen Abschnitten des Kapitels mit tabellarischen Verfahren zum Einsatz kommt:

<!--en:a2a7d7f0-->
Wir erstellen das Modell des Sees mit $\gamma = 0.95$ und geben die Zeile für den Befehl aus, den wir oben aus dem Simulator ausgelesen haben:

<!--en:2f35ef82-->
Aus der Liste von Tupeln ist eine dicht gespeicherte Zeile mit sechzehn Zahlen geworden. In den Spalten 8, 10 und 13 steht $1/3$, sonst null. Wir prüfen für jedes Array eine Eigenschaft. Jede Zeile von $P$ muss die Summe eins haben, und die Belohnung darf nur dort von null abweichen, wo ein Schritt ins Ziel möglich ist:

<!--en:25a97a23-->
Jede Zeile von $P$ hat die Summe eins. Das Array der Belohnungen enthält nur Nullen, außer in Zelle 14, der einzigen Eiszelle neben dem Ziel. Dort führen *unten*, *rechts* und *oben* jeweils mit Wahrscheinlichkeit $1/3$ ins Ziel und haben damit die erwartete Belohnung $1/3$. Bei *links* führt keines der drei möglichen Ergebnisse ins Ziel. Auf dem See erhält der Agent nur für den Schritt ins Ziel eine Belohnung.

<!--en:cbbae1ed-->
### Episoden, Terminierung und Abbruch

<!--en:6adc01af-->
Ein Zustand, der den Prozess beendet, heißt *Endzustand* (engl. terminal state). Auf dem See sind die Löcher und das Ziel Endzustände. Ein Durchlauf vom Start bis zu einem Endzustand heißt *Episode*. Dort beendet der Simulator die Episode und muss zurückgesetzt werden. Die Tabelle von Gymnasium und damit auch `TabularMDP` speichert jeden Endzustand als absorbierenden Zustand. Jede Aktion führt mit Belohnung null in denselben Zustand zurück. Somit ist jede Zeile von $P$ eine Verteilung. Beide Beschreibungen liefern dieselben Erträge, weil im absorbierenden Zustand keine weitere Belohnung hinzukommt.

<!--en:15606595-->
Endet jede Episode nach endlich vielen Schritten, bleibt der Ertrag auch ohne Diskontierung endlich. Dann ist $\gamma = 1$ zulässig. Der Ertrag ist in diesem Fall die ungewichtete Summe der Belohnungen einer Episode mit $T$ Schritten. Auf dem See endet nicht jede Episode. Wählt der Agent stets *oben*, bewegt er sich für immer in der obersten Zeile. Dennoch ist $\gamma = 1$ hier unproblematisch, denn die Belohnungen sind nichtnegativ und eine Trajektorie bringt höchstens eins ein. Das gilt nicht allgemein. Bei einer Belohnung von $+0.05$ pro Schritt hätte dieselbe Strategie, stets *oben* zu wählen, einen unendlichen undiskontierten Ertrag. Für Aufgaben ohne Ende, etwa die Heizungsregelung eines Gebäudes, verwendet man daher $\gamma < 1$. Ein weiteres gängiges Ziel bei solchen Aufgaben ist die Maximierung der langfristigen mittleren Belohnung pro Schritt. Darauf gehen wir in diesem Kapitel nicht ein.

<!--en:f3d3f4e5-->
Gymnasium unterscheidet zwei Fälle, in denen eine Episode endet. Bei `terminated=True` hat der Prozess einen Endzustand erreicht. Es kann nichts mehr geschehen, und der Wert einer Fortsetzung ist null. Bei `truncated=True` endet die Aufzeichnung, meist wegen eines Zeitlimits. Der Agent befindet sich aber in einem Zustand, von dem aus der Prozess weiterliefe. `gym.make` versieht FrozenLake mit einem solchen Limit. Nach 100 Schritten endet die Episode mit `truncated=True`, selbst wenn der Agent auf einer Eisfläche steht, von der aus das Ziel noch erreichbar ist. :numref:`fig_rl_termination` zeigt beide Fälle. Die Summe der Belohnungen einer abgebrochenen Episode ist daher nicht ihr Ertrag. Es fehlt alles, was nach dem Abbruch käme. Ein Verfahren, das den Wert einer Fortsetzung ab dem letzten Zustand schätzt, muss die beiden Fälle unterscheiden. Nach einer Terminierung ist dieser Wert null. Nach einem Abbruch zählt der *Wert* $V(s)$ des letzten Zustands, also der erwartete Ertrag ab $s$. Eine genaue Definition gibt :numref:`sec_valueiter`. Alle Verfahren in diesem Kapitel, die den Wert einer Fortsetzung schätzen, setzen ihn nur bei `terminated` auf null.

<!--en:4f5a818d-->
![Zwei Arten, wie eine Episode endet. (a) Terminierung: Der Agent gerät in das Loch $s_8$, einen Endzustand. Danach folgt nichts. Der Wert der Fortsetzung $V(s_8)$ ist also null. Der Zielwert für den letzten Schritt $y_7$, also der Ertrag ab diesem Schritt, besteht allein aus der letzten Belohnung $r_7$. (b) Abbruch: Nach 100 Schritten endet die Aufzeichnung, während der Agent auf einer Eiszelle steht, von der aus der Prozess weiterliefe. Die nicht aufgezeichnete Zukunft hat den Wert $V(s_{100})$. Dieser Wert einer Fortsetzung ab dem letzten aufgezeichneten Zustand ist von null verschieden. Eine Schätzung des Ertrags ab dem letzten Schritt, $y_{99} = r_{99} + \gamma V(s_{100})$, muss ihn daher berücksichtigen.](../img/mdl-rl-termination.svg)

<!--en:283218a4-->
## Wahl des Zustands

<!--en:3c45858c-->
Der Übergangskern $P(s' \mid s,a)$ ist allein durch den aktuellen Zustand und die Aktion bedingt. Das ist die *Markov-Eigenschaft*. Ist der gegenwärtige Zustand bekannt, liefert die Vorgeschichte keine weitere Information über den nächsten Zustand. Dadurch bleibt das Modell klein. Weil die Zukunft nur vom gegenwärtigen Zustand abhängt, genügt für jeden Zustand eine Zahl, die angibt, wie gut es ist, dort zu sein. Der Weg dorthin spielt keine Rolle. Beim See gilt die Eigenschaft aufgrund der Konstruktion. Der Simulator zieht die nächste Zelle aus der oben ausgelesenen Zeile `P[s][a]` und entscheidet bei jedem Schritt aufs Neue zufällig, wohin der Agent rutscht. Ob die Eigenschaft allgemein gilt, hängt davon ab, was wir als Zustand bezeichnen.

<!--en:94306244-->
### Zustandserweiterung

<!--en:5c56b9e1-->
Nehmen wir an, der aufgezeichnete Zustand $s_t$ ist der Ort eines Fahrzeugs, und die Aktion $a_t$ ist seine Beschleunigung. Der nächste Ort hängt auch von der aktuellen Geschwindigkeit ab. Diese geht aus dem aktuellen Ort allein nicht hervor, lässt sich aber aus zwei aufeinanderfolgenden Orten erschließen. Der nächste Ort hängt also von zwei früheren Orten ab,

<!--en:35cb0b0a-->
und der Ort allein erfüllt die Markov-Eigenschaft nicht. Definieren wir den Zustand als Paar aus Ort und Geschwindigkeit, ist das Problem gelöst. Nach der newtonschen Mechanik bestimmen das aktuelle Paar und die vorgegebene Beschleunigung den nächsten Ort und die nächste Geschwindigkeit. Die Markov-Eigenschaft hängt also sowohl vom physikalischen System als auch von unserer Zustandsdarstellung ab.

<!--en:11e7e849-->
Dieses Vorgehen heißt *Zustandserweiterung* (engl. state augmentation). Wir erweitern den Zustand so weit, bis er alle für die Vorhersage der Zukunft nötigen Informationen enthält. Der erweiterte Zustand kann groß sein. Die benötigten Informationen können unbekannt, zu umfangreich zum Speichern oder gar nicht beobachtbar sein. Ein einfaches Beispiel ist ein Zeitlimit, das zur Aufgabe gehört. Sind 100 Schritte erlaubt, kann der beste Befehl in einer Zelle davon abhängen, wie viele Schritte noch übrig sind. Ergänzen wir den Zustand um diese Anzahl, gilt die Markov-Eigenschaft wieder (siehe :exref:`ex_mdp-time-limits`).

<!--en:69345b9d-->
### Teilweise Beobachtbarkeit

<!--en:999b9e52-->
Die Zustandserweiterung setzt voraus, dass die zusätzlichen Variablen beobachtbar sind. Beim Poker sieht ein Spieler die Karten der anderen nicht. Erhält der Agent eine Beobachtung $o_t$, die nur einen Teil des Zustands erkennen lässt, liegt ein *teilweise beobachtbarer* (engl. partially observable) MDP vor. Eine exakte Lösung führt den *Informationszustand* (engl. belief) des Agenten mit. Das ist die A-posteriori-Verteilung des versteckten Zustands, bedingt auf alle bisherigen Beobachtungen (siehe :exref:`ex_mdp-belief-states`). In der Praxis nähern Agenten sie mithilfe eines Gedächtnisses an. Frühe Atari-Agenten stapelten die letzten Videobilder, weil ein einzelnes Bild zwar Orte zeigt, aber keine Geschwindigkeiten. Heutige Agenten führen einen rekurrenten Zustand mit oder nutzen Attention über ihre gesamte Vorgeschichte, so wie ein Sprachmodell über seinen Kontext. Im restlichen Kapitel setzen wir voraus, dass der Agent einen Markov-Zustand beobachtet.

<!--en:31c60771-->
## Entwurf der Belohnung

<!--en:301f3310-->
Die Zustände, die Aktionen und der Übergangskern beschreiben die Umgebung. Die Belohnung beschreibt die Aufgabe. Wer diese entwirft, wählt die Belohnung. Ein Optimierer maximiert genau die vorgegebene Belohnung und nutzt dabei auch Abweichungen von der eigentlichen Absicht aus :cite:`Krakovna.Uesato.Mikulik.ea.2020`. Am See lässt sich das mit wenigen Rechenschritten zeigen.

<!--en:0cf57a43-->
Bei FrozenLake sind die Belohnungen *spärlich*. Außer beim Schritt ins Ziel ist jede Belohnung null. Ein Agent, der das Ziel noch nie erreicht hat, hat bisher nur Nullen erhalten. Es liegt nahe, ihm auf dem Weg zu helfen. Sei $\ell(s)$ die Manhattan-Distanz von Zelle $s$ zum Ziel, also die Anzahl der nötigen Züge, wenn kein Loch im Weg liegt. Für jeden Zug, der die Distanz verkürzt, vergeben wir einen Bonus von $0.3$:

<!--en:e45df25a-->
Der Bonus scheint den Agenten zum Ziel zu führen, lässt sich aber auch kassieren, ohne je anzukommen. Betrachten wir die beiden Zellen 13 und 14 nahe dem Ziel in der untersten Zeile mit $\ell = 2$ und $\ell = 1$. Nehmen wir vorerst an, dass der Agent nicht rutscht. Der Zug von 14 nach 13 bringt nichts und kostet nichts. Der Rückweg bringt $0.3$. Geht der Agent von Zelle 14 ins Ziel, erhält er mit Belohnung und Bonus insgesamt $1.3$. Damit endet die Episode. Pendelt er stattdessen zwischen 14 und 13, erhält er auf Dauer bei jedem zweiten Schritt $0.3$. Bei $\gamma = 0.95$ ist das $0.3\gamma/(1 - \gamma^2) \approx 2.9$ wert. Der Fehler liegt darin, dass der Agent für die Annäherung einen Bonus erhält, für den Rückweg aber nichts bezahlt. So fällt auf jedem Rundweg insgesamt ein positiver Betrag an. :citet:`Randlov.Alstrom.1998` stießen bei einem simulierten Fahrrad auf dasselbe Problem. Es wurde für die Annäherung an ein Ziel belohnt und lernte, im Kreis zu fahren.

<!--en:9987b983-->
Damit der Agent auf einem Rundweg keinen Bonus ansammeln kann, muss jeder Schritt vom Ziel weg so viel kosten, wie der Rückweg einbringt. Boni der Form

<!--en:63299a58-->
mit einer Funktion $\Phi: \mathcal{S} \to \mathbb{R}$, dem *Potential*, haben diese Eigenschaft. Auf einem Rundweg zurück zum Ausgangspunkt heben sich die Differenzen $\Phi(s') - \Phi(s)$ auf. Bis auf den Effekt der Diskontierung bringt ein Rundweg also nichts ein. Diese Konstruktion heißt *potentialbasierte Belohnungsformung* (engl. potential-based shaping). Bei $\Phi(s) = -0.3\,\ell(s)$ bringt der Zug von 13 nach 14 einen Bonus von $0.315$, der Rückweg kostet $0.27$. Pendelt der Agent von Zelle 14 aus unbegrenzt hin und her, erhält er insgesamt $0.3$. Der Schritt ins Ziel bringt weiterhin $1.3$. Mit der angepassten Belohnung ist das Ziel demnach um genau jene Eins besser als das Pendeln, die schon die ursprüngliche Belohnung vergibt. :numref:`fig_rl_potential_shaping` stellt die beiden Boni für diesen Rundweg gegenüber. Auch der potentialbasierte Bonus hilft beim Lernen. Das Potential verteilt die Zielbelohnung entlang des Wegs, sodass der Agent schon lange vor seinem ersten Erfolg ein Lernsignal erhält.

<!--en:3a841128-->
![Zwei Boni in den Zellen 13 und 14 neben dem Ziel G, ohne Rutschen. Jeder Pfeil trägt die angepasste Belohnung des Zugs. (a) Der Annäherungsbonus bringt $0.3$ für den Zug zum Ziel hin und nichts für den Rückweg. Unbegrenztes Pendeln von Zelle 14 aus ist also $2.9$ wert, mehr als die $1.3$ für den Schritt ins Ziel. (b) Das Potential $\Phi = -0.3\,\ell$ vergütet den Zug zum Ziel hin mit $0.315$ und zieht für den Rückweg $0.27$ ab. Pendeln ist jetzt nur noch $0.3$ wert. Der Schritt ins Ziel ist die bessere Wahl.](../img/mdl-rl-potential-shaping.svg)

<!--en:b00a378b-->
Eine Bedingung bleibt. In den Endzuständen muss das Potential null sein. Der Simulator hält an, sobald er einen Endzustand erreicht. Ein von null verschiedenes Potential ließe dort einen letzten Term übrig, der vom Endzustand und vom Zeitpunkt des Eintreffens abhängt. Ein solcher Term kann ändern, welches Verhalten optimal ist. Unser Potential $\Phi = -0.3\,\ell$ ist am Ziel null, an den Löchern aber nicht. Dort müssen wir es also auf null setzen. In :numref:`sec_valueiter` berechnen wir, wie der Annäherungsbonus die optimale Strategie auf glattem Eis beeinflusst. Dort beweisen wir auch, dass der potentialbasierte Bonus die optimale Strategie erhält.

<!--en:a5abee46-->
## Zwei Spezialfälle

<!--en:a6ab53fd-->
Zwei Arten von MDP begegnen uns in diesem Kapitel mehrfach, weil bei jeder eine der Schwierigkeiten des Sees entfällt. Gibt es nur einen Zustand, besteht jede Runde darin, eine Aktion zu wählen und eine Belohnung zu beobachten. Keine Aktion beeinflusst die Zukunft. Das ist der *mehrarmige Bandit* (engl. multi-armed bandit). Der Name erinnert an eine Reihe von Spielautomaten, deren Auszahlungsraten ein Spieler durch Spielen herausfinden muss. Hier stellt sich allein die Frage, welche Aktionen man ausprobieren sollte. Dieser Frage geht :numref:`sec_exploration` nach. Im zweiten Fall haben Aktionen Folgen für die Zukunft, aber die Übergänge sind deterministisch. Dieser Fall beschreibt ein Sprachmodell, das eine Antwort schreibt.

<!--en:72ba5299-->
### Textgenerierung als MDP

<!--en:322cd6d2-->
Ein Sprachmodell, das Token für Token eine Antwort schreibt, lässt sich als MDP mit einfacher Dynamik beschreiben. Der Zustand besteht aus dem Prompt und den bisher geschriebenen Token. Die Aktion ist das nächste Token, und der Übergang hängt dieses Token mit Wahrscheinlichkeit eins an den Zustand an. Die Belohnung kommt einmalig, sobald die Antwort vollständig ist. Ein *Prüfprogramm* (engl. verifier) vergleicht die Antwort mit einer bekannten Lösung, oder ein gelerntes Modell weist ihr einen Score zu. Zwei Eigenschaften unterscheiden diesen MDP vom See. Die Übergänge sind deterministisch, also kommt der gesamte Zufall aus der Strategie. Da der Zustand das ganze Präfix umfasst, kommt kein Zustand zweimal vor. Die Zustände bilden einen Baum mit dem Prompt als Wurzel.

<!--en:225d68f2-->
Bei deterministischen Übergängen, einer einzigen Belohnung am Ende und ohne erneuten Besuch eines Zustands hängt der erwartete Ertrag nur von den Wahrscheinlichkeiten ab, die die Strategie den vollständigen Antworten zuweist. Daher können wir die ganze Antwort ohne Informationsverlust als eine einzige Aktion auffassen, die beim Prompt gewählt wird. Der MDP reduziert sich auf einen Schritt. Der Prompt $x$ aus einer Verteilung $\mu_0$ über die Prompts ist der Zustand, eine vollständige Antwort $y$ ist die Aktion und der Score $r(x, y)$ des Prüfprogramms ist die Belohnung. Deshalb können Verfahren für Sprachmodelle ganze Antworten bewerten und für Aktualisierungen nutzen. In :numref:`sec_rl_sequences` kehren wir zur Sicht auf einzelne Token zurück.

<!--en:b5194d2b-->
### Die Aufgabe mit Prüfprogramm

<!--en:d96a5d55-->
Wir veranschaulichen die Reduktion auf einen Schritt an einem Beispiel, das klein genug für eine exakte Berechnung ist. Es ist das zweite durchgehende Beispiel dieses Kapitels. Zu vier Rechenaufgaben stehen dieselben neun Antworten zur Wahl. Das Prüfprogramm vergibt eins, wenn die Antwort genau mit der Lösung der Aufgabe übereinstimmt, und sonst null. Für zwei der Prompts, `8*8` und `9*9`, ist keine der angebotenen Antworten richtig. Daher kann keine Strategie im Mittel über die Prompts einen höheren Score als $0.5$ erzielen. Die neunte Antwort, „eine der Zahlen 4, 12, 64 oder 81“, nennt alle Lösungen zugleich. Das Prüfprogramm verlangt eine exakte Übereinstimmung und lehnt diese *Ausweichantwort* ab. Ein großzügigeres Prüfprogramm, dem es genügt, dass die Lösung irgendwo in der Antwort steht, würde sie bei jedem Prompt belohnen. Hier begegnet uns das Problem des Belohnungsentwurfs in anderer Form.

<!--en:4ffc4ec3-->
Die Strategie steht hier für ein vortrainiertes Sprachmodell. Eine Tabelle $\theta$ enthält Präferenzen mit einer Zeile je Prompt und einer Spalte je Antwort. Wie bei der Softmax-Regression wandelt die Softmax-Funktion jede Zeile in Wahrscheinlichkeiten um, $\pi(y \mid x) \propto \exp \theta_{xy}$. Die Referenztabelle $\theta_{\textrm{ref}}$ enthält überall null, außer in der Spalte für die Ausweichantwort. Dort steht $-4$. Die Referenzstrategie liefert somit jede der acht gewöhnlichen Antworten mit einer Wahrscheinlichkeit von etwa $1/8$ und die Ausweichantwort mit etwa $0.2$ Prozent. Die Klasse `VerifierTask` fasst diese Bestandteile zusammen. Ihre Methode `verify(x, y)` prüft die Antwort. Ihr Attribut `R` speichert die Belohnungen des Prüfprogramms in einer Tabelle mit einer Zeile je Prompt und einer Spalte je Antwort. Die Methode `policy(theta)` wendet die Softmax-Funktion auf jede Zeile einer Präferenztabelle an. Die Methode `success(pi)` liefert den erwarteten Score einer gezogenen Antwort, gemittelt über die vier Prompts. Das ist der erwartete Ertrag dieses einstufigen MDP mit einer Gleichverteilung $\mu_0$ über die Prompts. Über das optionale Argument `reward` lässt sich die Strategie mit einem anderen Prüfprogramm bewerten. Beim Training ändert sich die Tabelle. Hier bewerten wir nur die Referenzstrategie:

<!--en:94495f9c-->
Eine aus der Referenzstrategie gezogene Antwort besteht die Prüfung mit Wahrscheinlichkeit $0.062$. Bei jedem der beiden lösbaren Prompts hat die richtige Antwort eine Wahrscheinlichkeit von etwa $1/8$. Die beiden übrigen Prompts tragen nichts bei. Damit liegt der Mittelwert bei etwa $1/16$. Das Training soll die Lücke zwischen $0.062$ und dem Höchstwert von $0.5$ schließen.

<!--en:561478c3-->
Wir können nun ein Entscheidungsproblem formulieren. Für den See genügen zwei Arrays und ein Diskontierungsfaktor, für die Aufgabe mit Prüfprogramm eine Tabelle von Scores. In beiden Fällen können wir den Ertrag einer Trajektorie angeben. Ein solches Modell lässt sich für jeden Zustand und jede Aktion abfragen. Das nutzen *modellbasierte* Verfahren. *Modellfreie* Verfahren sehen nur die Übergänge, die der Agent selbst erlebt. Die Frage aus Zelle 9 bleibt offen. Um *unten* mit *rechts* zu vergleichen, brauchen wir den Ertrag, zu dem jeder der beiden Befehle führen kann. Dieser hängt davon ab, was der Agent in jeder möglichen Folgezelle tut. In :numref:`sec_valueiter` berechnen wir diese Werte für alle Zellen zugleich.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:af89cb47-->
Ein Markov-Entscheidungsprozess beschreibt Entscheidungen über mehrere Schritte hinweg durch Zustände, Aktionen, einen Übergangskern und eine Belohnung, $(\mathcal{S}, \mathcal{A}, P, r)$. Er setzt voraus, dass der gegenwärtige Zustand alles Vergangene erfasst, was für die Zukunft zählt. Der Agent maximiert den erwarteten diskontierten Ertrag. Der Diskontierungsfaktor $\gamma$ hält den Ertrag endlich und legt einen effektiven Horizont von $1/(1-\gamma)$ Schritten fest. Pro Horizont sinken die Gewichte etwa um den Faktor $e$, wenn $\gamma$ nahe bei eins liegt. Ein Endzustand beendet den Prozess, ein Abbruch lediglich die Aufzeichnung. Nur bei Terminierung entfällt daher der Wert einer Fortsetzung. Erfüllt der naheliegende Zustand die Markov-Eigenschaft nicht, lässt sie sich durch eine Zustandserweiterung wiederherstellen, sofern die fehlende Information beobachtbar ist. Die Belohnung legen wir beim Entwurf fest. Ein Bonus, den der Agent auf einem Rundweg ansammeln kann, verändert sein Verhalten. Ein potentialbasierter Bonus lässt sich so nicht ansammeln. Die durchgehenden Beispiele des Kapitels sind FrozenLake und die Aufgabe mit Prüfprogramm und vier Prompts, bei der sich ein MDP auf Tokenebene auf einen Schritt reduziert.

<!--en:2294243a-->
## Übungen

<!--en:81f1eb35-->
1. **Was der Zustand enthalten muss.** In jedem der folgenden Systeme erfüllt der naheliegende Zustand die Markov-Eigenschaft nicht. Nennen Sie jeweils die kleinste Ergänzung, mit der die Eigenschaft wieder gilt. Geben Sie an, ob der Agent sie beobachten kann.
    1. Der Zustand eines Lieferroboters ist sein Standort. Bei jeder Bewegung verbraucht er je nach Gelände und Last unterschiedlich viel Akkuladung. Ist der Akku leer, bleibt er stehen.
    1. Der Zustand eines Empfehlungssystems ist das Nutzerprofil. Das Interesse des Nutzers an einem Thema sinkt mit der Anzahl der jüngsten Empfehlungen zu diesem Thema.
    1. Der Zustand eines Schachprogramms ist die Anordnung der Figuren auf dem Brett. Die Partie endet remis, sobald eine Stellung zum dritten Mal auftritt oder jeder Spieler fünfzig Züge lang weder eine Figur geschlagen noch einen Bauern gezogen hat.

   In welchen der drei Systeme kann die Ergänzung mit der Länge der Episode wachsen, und was begrenzt dort ihr Wachstum?
   :exlabel:`ex_mdp-state-must-contain`

<!--en:00f5c7c9-->
1. [code] **Episoden aus dem Übergangskern.** Für die Beispieltrajektorie $\tau$ dieses Abschnitts ziehen wir mit `rng.integers(4)` jede Aktion gleichverteilt. Unter dieser Strategie $\pi$ hat der Folgezustand die Verteilung

   $$P_\pi(s' \mid s) = \frac{1}{4} \sum_{a \in \mathcal{A}} P(s' \mid s, a),$$

   die sich mit `mdp.P.mean(axis=1)` berechnen lässt.
    1. `from_gym` addiert Wahrscheinlichkeiten mit `+=`, weil in Gymnasium mehrere Ergebnisse eines Zustand-Aktion-Paars zum selben Folgezustand führen können. Finden Sie alle Paare, für die das gilt. Erklären Sie anhand von :numref:`fig_rl_gridworld`, warum es genau diese Paare sind. Angenommen, die Zeile zum Füllen von `P` verwendete die Zuweisung `=`. Welche Zeilen von `mdp.P` hätten dann eine andere Summe als eins, und welche Summe hätten sie?
    1. Sei $d^\pi_t$ die Verteilung von $s_t$, ausgehend von $d^\pi_0 = \mu_0$, der Punktmasse auf Zustand 0. Es gilt also $d^\pi_{t+1}(s') = \sum_{s} d^\pi_t(s) P_\pi(s' \mid s)$. Zeichnen Sie für $t \leq 30$ die Wahrscheinlichkeit, dass der Agent das Ziel erreicht hat, die Wahrscheinlichkeit, dass er in ein Loch gefallen ist, und die Wahrscheinlichkeit, dass die Episode noch läuft. Welche Grenzwerte ergeben sich für $t \to \infty$?
    1. Lassen Sie das Zeitlimit von `env` außer Acht und zeigen Sie, dass die mittlere Episodenlänge gleich $\sum_{t \geq 0} \Pr(s_t \textrm{ is not terminal})$ ist. Führen Sie 10 000 Episoden von `env` mit der Zufallsstrategie aus. Vergleichen Sie den Anteil der Episoden, die das Ziel erreichen, und die mittlere Episodenlänge mit den Vorhersagen von $d^\pi_t$. Das Limit von 100 Schritten in `env` beendet eine zufällige Episode mit einer Wahrscheinlichkeit unter $10^{-8}$ und beeinflusst daher den Vergleich nicht.
   :exlabel:`ex_mdp-kernel-predicts-episodes`

<!--en:ebe371a2-->
1. **Erwartete Belohnungen.** In Gymnasium hängt die Belohnung vom Übergang ab. Der Schritt ins Ziel bringt eins. `from_gym` ersetzt diese Belohnung $\rho(s, a, s')$ durch ihren Erwartungswert $r(s, a) = \sum_{s'} P(s' \mid s, a)\, \rho(s, a, s')$.
    1. Zeigen Sie für jede Strategie und jeden Startzustand, dass der erwartete diskontierte Ertrag gleich bleibt, ob Schritt $t$ nun $\rho(s_t, a_t, s_{t+1})$ oder $r(s_t, a_t)$ einbringt. Erweitern Sie das Argument auf eine Belohnung, die bei jedem Besuch von $(s, a)$ unabhängig von allem anderen aus einer Verteilung mit Mittelwert $r(s, a)$ gezogen wird.
    1. Welche FrozenLake-Episoden haben unter der Konvention $r(s, a)$ einen positiven Ertrag? Schätzen Sie für die gleichverteilte Zufallsstrategie dieses Abschnitts aus 10 000 Episoden die Wahrscheinlichkeit eines positiven Ertrags unter jeder der beiden Konventionen. Welche Eigenschaften des Ertrags legt das Modell :eqref:`eq_mdp` demnach fest, und welche lässt es offen?
   :exlabel:`ex_mdp-expected-rewards`

<!--en:24d7d548-->
1. **Diskontierung als Terminierung.** Der diskontierte Ertrag :eqref:`eq_mdp_discounted_return` gewichtet die Belohnung in Schritt $t$ mit $\gamma^t$.
    1. Wie in diesem Abschnitt beschrieben, entfallen 95 Prozent des diskontierten Gewichts auf die ersten $\log(0.05) / \log(\gamma)$ Schritte. Das entspricht etwa drei effektiven Horizonten. Zeigen Sie, dass für jeden Anteil $p \in (0, 1)$ die Ungleichung

        $$\frac{\ln(1/p)}{\ln(1/\gamma)} \leq \frac{\ln(1/p)}{1 - \gamma},$$

       gilt und dass der Quotient der beiden Seiten für $\gamma \to 1$ gegen eins geht. Erklären Sie mithilfe der Ungleichung die Schrittzahlen in :numref:`fig_rl_return_discount` bei $\gamma = 0.5$, $0.9$ und $0.99$.
    1. Ändern Sie einen MDP so ab, dass der Prozess nach jeder Belohnung mit Wahrscheinlichkeit $1 - \gamma$ in einen absorbierenden Zustand mit Belohnung null übergeht und sonst gemäß $P$ fortfährt. Zeigen Sie, dass der erwartete undiskontierte Ertrag jeder Strategie im abgeänderten Prozess gleich ihrem erwarteten diskontierten Ertrag im ursprünglichen Prozess ist und dass die erwartete Anzahl der erhaltenen Belohnungen $1 / (1 - \gamma)$ beträgt. Deuten Sie die Arrays von `TabularMDP` in diesem Sinne. Was bedeuten die Zeilensummen von $\gamma P$?
   :exlabel:`ex_mdp-discount-as-termination`

<!--en:f59f66b4-->
1. **Zeitlimits.** `gym.make` versieht FrozenLake mit einem Zeitlimit. Nach 100 Schritten endet die Episode mit `truncated=True` (:numref:`subsec_mdp-termination`). Nehmen Sie an, das Limit gehört zur Aufgabe. Die Zielgröße ist die Wahrscheinlichkeit, das Ziel innerhalb von 100 Schritten zu erreichen. Modellieren Sie den Ablauf des Zeitlimits als Übergang in einen absorbierenden Zustand. Zeigen Sie, dass die Verteilung des Folgezustands dann von der Anzahl $h$ der verbleibenden Schritte abhängt. Der Ort allein ist also kein Markov-Zustand, das Paar $(s, h)$ hingegen schon :cite:`Pardo.Tavakoli.Levdik.ea.2018`. Warum kann die beste Aktion an einem Ort von $h$ abhängen?
   :exlabel:`ex_mdp-time-limits`

<!--en:634f8d2a-->
1. **Aktionen ohne Folgen.** Ein mehrarmiger Bandit hat nur einen Zustand. Daher kann keine Aktion die Zukunft beeinflussen.
    1. Nehmen Sie allgemeiner an, dass der Übergangskern unabhängig von der Aktion ist, also $P(s' \mid s, a) = P(s' \mid s)$ für alle $a$ gilt. Zeigen Sie für jedes $\gamma$, dass die Strategie, die in jedem Zustand eine Aktion aus $\arg\max_a r(s, a)$ wählt, den erwarteten diskontierten Ertrag von jedem Startzustand aus maximiert.
    1. Konstruieren Sie einen MDP mit zwei Zuständen und zwei Aktionen, in dem diese gierige Strategie für kleine $\gamma$ optimal ist, für große $\gamma$ jedoch nicht. Bestimmen Sie den Wert von $\gamma$, bei dem die optimale Aktion wechselt.

<!--en:598dc235-->
1. **Informationszustände.** ● In einem teilweise beobachtbaren MDP erhält der Agent eine Beobachtung $o_t$ mit Wahrscheinlichkeit $O(o_t \mid s_t)$ anstelle des Zustands $s_t$. Sein *Informationszustand* $\beta_t(s) = \Pr(s_t = s \mid o_0, a_0, \ldots, a_{t-1}, o_t)$ ist die A-posteriori-Verteilung des versteckten Zustands, bedingt auf alle bisherigen Beobachtungen :cite:`Kaelbling.Littman.Cassandra.1998`.
    1. Leiten Sie aus der Markov-Eigenschaft der versteckten Zustände die Aktualisierungsregel her:

        $$\beta_{t+1}(s') \propto O(o_{t+1} \mid s') \sum_{s \in \mathcal{S}} P(s' \mid s, a_t)\, \beta_t(s).$$

    1. Zeigen Sie, dass der Informationszustand ein Markov-Zustand ist. Bei gegebenem $\beta_t$ und $a_t$ hängen die Verteilung von $\beta_{t+1}$ und die erwartete Belohnung nicht von früheren Beobachtungen ab. Welchen Zustandsraum und welchen Übergangskern hat der resultierende MDP auf den Informationszuständen?
    1. In einem T-Labyrinth nach :citet:`Bakker.2001` sieht der Agent am Start einen Hinweis, *links* oder *rechts*, jeweils mit Wahrscheinlichkeit $1/2$. Dann durchquert er mit einer Zelle pro Schritt einen Gang aus $N$ Zellen, in denen er stets dieselbe Beobachtung macht, und erreicht eine Abzweigung. Biegt er dort in die angezeigte Richtung ab, erhält er die Belohnung eins, sonst null. Zeigen Sie, dass eine Strategie auf Grundlage der $k$ jüngsten Beobachtungen höchstens mit Wahrscheinlichkeit $1/2$ Erfolg hat, sofern nicht $k \geq N + 2$ gilt. Wie lautet der Informationszustand an der Abzweigung, und wie viel Speicher genügt, um optimal zu handeln? Warum reicht es bei Pong, einige Bilder zu stapeln, hier aber nicht?
   :exlabel:`ex_mdp-belief-states`

<!--en:cf6f3336-->
1. [extended] **Planungshorizont und Modellfehler.** ●● Die Aufgabe legt den Diskontierungsfaktor $\gamma$ der Zielgröße fest. Ein Planungsverfahren kann seine Strategie jedoch mit einem kleineren *Diskontierungsfaktor für die Planung* $\gamma'$ berechnen. :citet:`Jiang.Kulesza.Singh.ea.2015` zeigten, dass bei einem aus Daten geschätzten Modell $\gamma' < \gamma$ eine bessere Strategie liefern kann als $\gamma' = \gamma$. Sei $V^\pi_\gamma(s)$ der erwartete diskontierte Ertrag ab $s$ unter der Strategie $\pi$ mit Diskontierungsfaktor $\gamma$. In Teil (b) verwenden Sie die Wertiteration und die exakte Strategiebewertung aus :numref:`sec_valueiter` (`d2l.value_iteration` und `TabularMDP.policy_value`).
    1. Die Belohnungen mögen in $[0, r_{\max}]$ liegen, und es gelte $\gamma' \leq \gamma$. Zeigen Sie, dass für jede Strategie $\pi$ und jeden Zustand $s$ die Ungleichung

        $$0 \leq V^\pi_\gamma(s) - V^\pi_{\gamma'}(s) \leq \frac{(\gamma - \gamma')\, r_{\max}}{(1 - \gamma)(1 - \gamma')},$$

       gilt. Folgern Sie daraus, dass die für $\gamma'$ optimale Strategie im wahren Modell unter $\gamma$ höchstens diesen Betrag einbüßt.
    1. Wiederholen Sie das Experiment in kleinerem Maßstab mit `TabularMDP`. Ziehen Sie 100 zufällige MDPs mit 10 Zuständen und 2 Aktionen. Wählen Sie für jedes Zustand-Aktion-Paar 5 Folgezustände gleichverteilt ohne Zurücklegen. Weisen Sie ihnen gleichverteilt aus $[0, 1]$ gezogene Gewichte zu und normieren Sie diese. Ziehen Sie jedes $r(s, a)$ gleichverteilt aus $[0, 1]$. Schätzen Sie für jeden MDP und jedes $n \in \{2, 5, 10, 20\}$ jeweils 20 Übergangskerne $\hat P$ aus je $n$ gezogenen Folgezuständen pro Paar. Behalten Sie das wahre $r$ bei. Berechnen Sie für jedes geschätzte Modell die optimale Strategie für $\gamma' \in \{0, 0.1, \ldots, 0.9, 0.99\}$. Bewerten Sie diese im wahren Modell bei $\gamma = 0.99$ und mitteln Sie den Verlust $V^*_\gamma(s) - V^\pi_\gamma(s)$ über Zustände, Übergangskerne und MDPs. Dabei ist $V^*_\gamma$ der Wert der optimalen Strategie im wahren Modell. Die Wertiteration benötigt etwa $\log(10^{-8})/\log\gamma'$ Durchläufe, bei $\gamma' = 0.99$ rund 2000 und bei kleineren Werten deutlich weniger. Mit dieser Anzahl dauert das Experiment einige Minuten CPU-Zeit. Zeichnen Sie den Verlust als Funktion von $\gamma'$ für jedes $n$.
    1. Welches $\gamma'$ minimiert den Verlust für das jeweilige $n$, und wie ändert sich dieser Wert mit $n$? Erklären Sie den Kurvenverlauf anhand zweier Verlustquellen, der in dieser Aufgabe beschränkten Verzerrung durch den Horizont und des Fehlers von $\hat P$. Welche der beiden wächst mit $\gamma'$?
   :exlabel:`ex_mdp-planning-discount`

<!--en:4f43f6a4-->
[Diskussionen](https://d2l.discourse.group/t/36794)

<!--en:b6686ba3-->
Markov-Entscheidungsprozesse<br>**Zustände, Aktionen, Übergangskern und Belohnung · diskontierter Ertrag und Horizont · Terminierung und Abbruch · Zustandserweiterung · potentialbasierte Belohnungsformung · Aufgabe mit Prüfprogramm**

<!--en:985603f0-->
::: {.slide title="Handeln unter Unsicherheit: das Dilemma in Zelle 9"}

<!--en:b0445376-->
[Motivation und Anschauung]{.kicker}

<!--en:fc83efa1-->
Um Entscheidungen über mehrere Schritte zu optimieren, müssen wir festlegen, was der Agent weiß, welche Befehle er geben kann und wie die zufälligen Übergänge ablaufen:

<!--en:d859fc2f-->
- **Glattes 4×4-Gitter:** Von Zelle 0 (`S`) aus sucht der Agent das Ziel in Zelle 15 (`G`, Belohnung $+1$). Dabei muss er vier Löcher (`H`) meiden, die den Versuch beenden.

<!--en:153b42ea-->
- **Zufällige Übergänge:** Bei jedem Befehl bewegt sich der Agent mit Wahrscheinlichkeit $1/3$ in die gewählte Richtung und mit Wahrscheinlichkeit $1/3$ in jede der beiden Richtungen senkrecht dazu.

<!--en:8f760edd-->
- **Abwägung in Zelle 9:** Bei *links*, *rechts* und *oben* rutscht der Agent jeweils mit Wahrscheinlichkeit $1/3$ in Loch 5. Nur bei *unten* meidet er das Loch sicher, rutscht dafür aber in zwei von drei Fällen seitwärts.

<!--en:8432c8b7-->
::: {.slide title="Der Markov-Entscheidungsprozess als Tupel"}

<!--en:6f14e397-->
[Mathematische Formulierung]{.kicker}

<!--en:54d4d680-->
- **Zustände $\mathcal{S}$ und Aktionen $\mathcal{A}$:** FrozenLake hat $|\mathcal{S}| = 16$ Zellindizes und $|\mathcal{A}| = 4$ Richtungsbefehle.

<!--en:068ab1d5-->
- **Übergangskern $P(s' \mid s, a)$:** Die bedingten Wahrscheinlichkeiten für die Zielzellen erfüllen $\sum_{s'} P(s' \mid s, a) = 1$.

<!--en:4c3b8539-->
- **Erwartete Belohnung $r(s, a)$:** unmittelbare Belohnung, gemittelt über die Folgezustände, $r(s, a) = \sum_{s'} P(s' \mid s, a)\,\rho(s, a, s')$.

<!--en:1d526c48-->
::: {.slide title="Den Übergangskern aus Gymnasium auslesen"}

<!--en:96ab43f8-->
[Darstellung im Simulator]{.kicker}

<!--en:4ecf196c-->
Gymnasium stellt den tabellarischen Übergangskern als verschachteltes Wörterbuch `env.unwrapped.P[s][a]` mit Tupeln der Form `(probability, next_state, reward, terminated)` bereit:

<!--en:192b9736-->
Der Befehl *unten* (`1`) in Zustand 9 hat drei gleich wahrscheinliche Ergebnisse (Zellen 8, 13 und 10), jedes mit Wahrscheinlichkeit $1/3$ und unmittelbarer Belohnung $0.0$.

<!--en:115e18ce-->
::: {.slide title="Der MDP in vektorisierter Form mit dichten Tensoren"}

<!--en:e58167df-->
[Darstellung als Arrays]{.kicker}

<!--en:da88802e-->
- **Dichte Tensoren:** Für Wertaktualisierungen mit Matrizen speichert `TabularMDP.from_gym` die Einträge `P[s, a, s']` in der Form `(16, 4, 16)` und `r[s, a]` in der Form `(16, 4)`.

<!--en:5221e36d-->
- **Übergänge am Rand aufsummieren:** Führen mehrere Rutschbewegungen in dieselbe Randzelle, addiert `P[s, a, s_next] += p` ihre Wahrscheinlichkeiten.

<!--en:833efca6-->
::: {.slide title="Normierung des Übergangskerns und Verteilung der Belohnungen prüfen"}

<!--en:0a79fcef-->
[Plausibilitätsprüfung des Modells]{.kicker}

<!--en:aa23fb87-->
Zwei Eigenschaften bestätigen die Konstruktion der Arrays. Für jedes $(s, a)$ summiert sich der zugehörige Ausschnitt von `P` zu $1$, und `r[s, a]` ist nur dort von null verschieden, wo ein Schritt in die Zielzelle 15 führen kann:

<!--en:c3056e1d-->
- **Abprallen am Rand:** In einer Ecke können sowohl ein Befehl in Richtung einer Wand als auch eine der zugehörigen Rutschbewegungen den Agenten gegen eine Wand führen. Der Agent bleibt dann mit Wahrscheinlichkeit $2/3$ in der aktuellen Zelle.

<!--en:d7d8225f-->
- **Erwartete Belohnung in Zelle 14:** Bei *unten*, *rechts* oder *oben* erreicht der Agent von Zelle 14 aus die Zielzelle 15 mit Wahrscheinlichkeit $1/3$. Damit ist die erwartete Belohnung $r(14, a) = 1/3$. Bei *links* erreicht er das Ziel nie in einem Schritt.

<!--en:b2f17616-->
::: {.slide title="Trajektorien, Strategien und zufällige Durchläufe"}

<!--en:4ad50307-->
[Interaktion zwischen Agent und Umgebung]{.kicker}

<!--en:d08a0fdd-->
Eine stochastische Strategie $\pi(a \mid s)$ legt für jeden Zustand die Wahrscheinlichkeiten der Aktionen fest. So entsteht eine zufällige Trajektorie $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$:

<!--en:1a25b9ac-->
- **Beobachtete und erwartete Übergänge:** Bei $t=0$ rutscht der Agent beim Befehl *rechts* gegen die obere Wand und bleibt bei $s'=0$. Bei $t=8$ gerät er in Loch 11 (`terminated=True`).

<!--en:67c664a9-->
::: {.slide title="Diskontierter Ertrag und geometrische Restabschätzung"}

<!--en:a6c323ba-->
[Optimierungsziel]{.kicker}

<!--en:7337fa79-->
- **Endliche Zielgröße:** Gelten $|r_t| \leq r_{\max}$ und $\gamma \in [0, 1)$, konvergiert die unendliche Summe. Ihr Rest nach Schritt $k$ nimmt proportional zu $\gamma^k$ ab.

<!--en:d4fdbd9e-->
- **Für alle Startzustände optimale Strategie:** Jeder endliche MDP mit $\gamma < 1$ besitzt eine deterministische stationäre Strategie $\pi^*(s)$, die von jedem Startzustand aus optimal ist.

<!--en:f7659dfd-->
::: {.slide title="Der effektive Planungshorizont"}

<!--en:b4bbc95a-->
[Wie weit die Diskontierung reicht]{.kicker}

<!--en:08942fc7-->
Der Diskontierungsfaktor legt einen effektiven Horizont von $H_{\textrm{eff}} = 1/(1 - \gamma)$ Schritten fest, innerhalb dessen Belohnungen erhebliches Gewicht haben:

<!--en:6491a753-->
- **Exponentieller Abfall:** Für $\gamma$ nahe $1$ sinken die Gewichte pro effektivem Horizont auf den Anteil $\gamma^{1/(1-\gamma)} \approx e^{-1} \approx 0.368$.

<!--en:ca2985fa-->
- **Zeitfenster für 95% des Gewichts:** Bei konstanter Belohnung entfallen $95\%$ des gesamten diskontierten Gewichts auf die ersten $\lceil\ln(0.05)/\ln\gamma\rceil \approx 3\,H_{\textrm{eff}}$ Schritte.

<!--en:c3940d5c-->
- **Referenzwert des Kapitels ($\gamma = 0.95$):** Der effektive Horizont beträgt 20 Schritte. Auf die ersten 59 Schritte entfallen $95\%$ des Gewichts.

<!--en:7301a3ee-->
::: {.slide title="Wie der Horizont vom Diskontierungsfaktor abhängt"}

<!--en:5dba7865-->
[Numerischer Vergleich]{.kicker}

<!--en:0e18f280-->
Der Vergleich von vier Diskontierungsfaktoren zeigt, wie stark $\gamma$ die Bewertung des kürzesten Weges ohne Rutschen beeinflusst (6 Züge, Zielbelohnung bei $t = 5$):

<!--en:cedb66c1-->
Ein kleiner Diskontierungsfaktor ($\gamma = 0.5$) bewertet das in 6 Zügen erreichte Ziel mit nur $\gamma^5 = 0.031$. Bei $\gamma = 0.95$ ist es $0.774$ wert. Ein kleineres $\gamma$ macht die Planung weniger empfindlich gegen Modellfehler, kann aber langfristig schlechtere Entscheidungen begünstigen.

<!--en:eba265ce-->
::: {.slide title="Terminierung und Abbruch am Zeitlimit"}

<!--en:5eb6a811-->
[Fehlerquelle am Ende einer Episode]{.kicker}

<!--en:f0898607-->
Gymnasium unterscheidet zwischen dem Ende des Prozesses (`terminated=True`) und dem Abbruch am Zeitlimit (`truncated=True` nach 100 Schritten):

<!--en:d83534a9-->
- **Terminierung ($s_8 \in \textrm{Holes}$):** Der Prozess in der Umgebung ist zu Ende. Für den künftigen Ertrag ab $s_8$ gilt $V(s_8) = 0$. Der letzte Zielwert ist daher $y_7 = r_7$.

<!--en:261db889-->
- **Abbruch (bei $t = 100$):** Die Aufzeichnung endet auf sicherem Eis. Die nicht aufgezeichnete Fortsetzung hat den Wert $V(s_{100}) \neq 0$. Daher muss $y_{99} = r_{99} + \gamma V(s_{100})$ gelten.

<!--en:d4593531-->
::: {.slide title="Absorbierende Endzustände und Aufgaben ohne Diskontierung"}

<!--en:7dc05b13-->
[Konventionen für Endzustände]{.kicker}

<!--en:691410c0-->
Eine einheitliche Darstellung der Endzustände sorgt dafür, dass Matrixgleichungen und Durchläufe mit endlichem Horizont übereinstimmen:

<!--en:d41c86f0-->
- **Absorbierende Zustände:** `TabularMDP` modelliert jedes Loch und das Ziel als Übergang in denselben Zustand ($P(s \mid s, a) = 1$) mit $r(s, a) = 0$. Damit summiert sich jede Zeile von $P$ zu $1$, und die Erträge der Trajektorien bleiben erhalten.

<!--en:15083e60-->
- **Wann $\gamma = 1$ zulässig ist:** Endet jede Strategie nach endlich vielen Schritten oder sind die Belohnungen nichtnegativ und ist ihre Summe auf jeder Trajektorie nach oben beschränkt (bei FrozenLake höchstens $1$ pro Episode), bleiben die undiskontierten Erträge endlich.

<!--en:612ed16c-->
- **Wann der Ertrag bei $\gamma = 1$ divergiert:** Mit einem Überlebensbonus von $+0.05$ pro Schritt erzielt die Strategie, stets *oben* zu wählen, einen unendlichen undiskontierten Ertrag. Für solche fortlaufenden Aufgaben braucht man $\gamma < 1$.

<!--en:6374a682-->
::: {.slide title="Markov-Eigenschaft und Zustandserweiterung"}

<!--en:2ba494ab-->
[Wahl der Zustandsdarstellung]{.kicker}

<!--en:2ef7bfa7-->
- **Warum der Ort allein nicht genügt:** Bei einem Fahrzeug mit Beschleunigung $a_t$ hängt der nächste Ort $p_{t+1}$ von der Geschwindigkeit $v_t \propto p_t - p_{t-1}$ ab.

<!--en:a783043b-->
- **Zustandserweiterung:** Erweitert man den Zustand auf $\tilde{s}_t = (p_t, v_t)^\top$ (oder ergänzt die verbleibende Zeit $H - t$), erhält man wieder einen Markov-Übergangskern erster Ordnung.

<!--en:740c0fa8-->
::: {.slide title="Teilweise Beobachtbarkeit, Informationszustände und Gedächtnis"}

<!--en:da9c06f4-->
[Wenn der vollständige Zustand verborgen bleibt]{.kicker}

<!--en:262bb964-->
- **Exakter bayesscher Informationszustand:** Führt man $\beta_t(s) = \Pr(s_t = s \mid o_{0:t}, a_{0:t-1})$ nach der Bayes-Regel fort, erhält man wieder einen kontinuierlichen Markov-Zustand.

<!--en:763bd3c5-->
- **Bildfolgen in der Regelung:** Aus den letzten 4 Videobildern können Filter die Geschwindigkeiten mithilfe endlicher Differenzen erschließen.

<!--en:f90ecc76-->
- **Kontext bei Rekurrenz und Attention:** Moderne Agenten und große Sprachmodelle nutzen Attention über dem bisherigen Verlauf der Trajektorie $(o_0, a_0, \ldots, o_t)$.

<!--en:09e1dc50-->
::: {.slide title="Reward Hacking: Fortschrittsbonus für das Laufen im Kreis"}

<!--en:708af532-->
[Fehlerquelle beim Entwurf der Belohnung]{.kicker}

<!--en:6fb7e396-->
Auf FrozenLake erhält der Agent nur selten eine terminale Belohnung. Nehmen wir an, er erhält zusätzlich $+0.3$ für jeden Schritt, der die Manhattan-Distanz $\ell(s)$ zum Ziel verkürzt:

<!--en:364eb77b-->
- **Einseitiger Anreiz:** Ein Schritt zum Ziel bringt $+0.3$ (Zellwechsel $13 \to 14$, bei dem $\ell$ von $2$ auf $1$ fällt). Ein Schritt zurück kostet $0$ ($14 \to 13$).

<!--en:9d590409-->
- **Pendeln lohnt sich mehr als Ankommen:** Ohne Rutschen bringt bei $\gamma = 0.95$ der Schritt von Zelle 14 ins Ziel einmalig $1 + 0.3 = 1.3$. Beim endlosen Pendeln $14 \leftrightarrow 13$ bringt jeder zweite Schritt $0.3$:
  $$R_{\textrm{loop}} = 0.3\big(\gamma + \gamma^3 + \gamma^5 + \cdots\big) = \frac{0.3\,\gamma}{1 - \gamma^2} \approx 2.92 > 1.30$$

<!--en:56417ce4-->
::: {.slide title="Invarianz bei potentialbasierter Belohnungsformung"}

<!--en:828a00ef-->
[Dichte Belohnungen bei gleicher optimaler Strategie]{.kicker}

<!--en:796173b7-->
- **Teleskopsumme (Ng et al., 1999):** Bei $\Phi(s) = -0.3\,\ell(s)$ bringt der Schritt $13 \to 14$ einen Bonus von $+0.315$. Für den Rückweg $14 \to 13$ beträgt der Bonus $-0.270$.

<!--en:2c29a324-->
- **Die optimale Strategie bleibt erhalten:** Unbegrenztes Pendeln bringt höchstens $-\Phi(14) = +0.3$. Der Schritt ins Ziel ($1.3$) liegt damit um $+1.0$ vorn.

<!--en:7e002c5c-->
- **Randbedingung:** Für alle Endzustände muss $\Phi(s_{\textrm{term}}) = 0$ gelten.

<!--en:1b5a5b15-->
::: {.slide title="Sprachgenerierung als MDP mit einem einzigen Schritt"}

<!--en:a33cd51c-->
[Vom Tokenbaum zum kontextuellen Banditen]{.kicker}

<!--en:0b49ed94-->
Bei der autoregressiven Sprachgenerierung mit abschließender Antwortprüfung vereinfacht sich die Struktur des allgemeinen MDP in zweierlei Hinsicht:

<!--en:5cfbea39-->
- **Baumförmiger MDP auf Tokenebene:** Der Zustand $s_t = (x, y_{<t})$ besteht aus Prompt und bisherigem Tokenpräfix. Die Aktion $a_t = y_t$ ist das nächste Token. Der Übergang hängt $y_t$ deterministisch an. Das Prüfprogramm bewertet nur die fertige Antwort.

<!--en:e37d0423-->
- **Reduktion auf einen Schritt:** Die Übergänge sind deterministisch, die Belohnung kommt nur am Ende, und kein Präfix tritt erneut auf. Daher hängt der erwartete Ertrag nur von der Sequenzwahrscheinlichkeit $\pi_\theta(y \mid x)$ ab.

<!--en:240a7166-->
- **Äquivalenz zum kontextuellen Banditen:** Der Prompt $x \sim \mu_0$ ist der Startzustand, die vollständige Antwort $y$ eine einzige Makroaktion und der Score des Prüfprogramms $r(x, y) \in \{0, 1\}$ die unmittelbare Belohnung.

<!--en:d117e0bc-->
::: {.slide title="Aufgabe mit Prüfprogramm: Referenzstrategie und erreichbarer Höchstwert"}

<!--en:9d2ab2b5-->
[Zweites durchgehendes Beispiel]{.kicker}

<!--en:49ff27e8-->
`VerifierTask` verbindet 4 Rechenaufgaben mit denselben 9 möglichen Antworten (8 Zahlen und die Ausweichantwort *„eine der Zahlen 4, 12, 64 oder 81“*). Die tabellarische Softmax-Strategie lautet $\pi(y \mid x) \propto \exp(\theta_{x,y})$:

<!--en:9473de5a-->
- **Warum die Referenzstrategie den Score $0.062$ erzielt:** Mit $\theta_{\textrm{ref}} = 0$ für die 8 Zahlen und $-4$ für die Ausweichantwort beträgt die Wahrscheinlichkeit einer richtigen Antwort $\approx 1/8$ bei `2+2` und `3*4` und $0$ bei den unlösbaren Prompts `8*8` und `9*9` (Mittelwert $\approx 1/16 = 0.062$, Höchstwert $0.500$).

<!--en:81639ade-->
::: {.slide title="Verständnisfragen zu MDP-Struktur und Belohnungsentwurf"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:9084a7af-->
Prüfen Sie Ihr Verständnis von MDP-Modellierung, Abbruch und Belohnungsformung:

<!--en:0727bb81-->
1. **Wahl des Markov-Zustands:** Warum bildet ein einzelnes Videobild von Pong keinen Markov-Zustand? Welche minimale Zustandserweiterung stellt die Markov-Eigenschaft wieder her?

<!--en:0df16030-->
2. **Terminierung und Abbruch:** Nach 100 Schritten wird eine Episode abgebrochen, während der Agent auf Zelle 14 steht. Warum wäre es ein Fehler, den Wert der Fortsetzung auf null zu setzen (`done = terminated or truncated`)?

<!--en:08c9ee49-->
3. **Randbedingung für das Potential:** Bei $\Phi(s) = -0.3\,\ell(s)$ müssen wir für die vier Löcher ausdrücklich $\Phi(s) = 0$ setzen, bevor wir einzelne Episoden simulieren. Warum?

<!--en:557b98b6-->
**Antworten:**

<!--en:2b48a34a-->
1. Ein Bild zeigt den Ort des Balls. Für seine Geschwindigkeit fehlen Richtung und Tempo. Aus 2 aufeinanderfolgenden Bildern liefert die endliche Differenz die Geschwindigkeit, und die Markov-Eigenschaft gilt wieder.

<!--en:020428df-->
2. Zelle 14 grenzt ans Ziel. Die Fortsetzung hat den positiven Wert $V(14) > 0$. Setzt man ihn beim Abbruch auf null, lernt der Agent fälschlich, Zelle 14 sei bei Schritt 100 eine wertlose Sackgasse.

<!--en:f4fe98a4-->
3. Ohne $\Phi(\textrm{hole}) = 0$ bleibt beim Ende der Episode in einem Loch der Term $\gamma^T\Phi(s_T) = -0.3\gamma^T\ell(s_T) < 0$ übrig. Er hängt davon ab, in welches Loch der Agent wann fällt, und verfälscht die optimale Strategie.

<!--en:952a4910-->
::: {.slide title="Zusammenfassung: Markov-Entscheidungsprozesse"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:367ddc80-->
| Begriff | Mathematische Form | Bedeutung für die Praxis |

<!--en:648ed503-->
| **MDP-Tupel** | $(\mathcal{S}, \mathcal{A}, P, r)$ | Beschreibt Zustands- und Aktionsmenge, Übergangskern $P(s' \mid s, a)$ und erwartete Belohnung $r(s, a)$ |

<!--en:7fedc4c6-->
| **Diskontierter Ertrag** | $R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t$ | Beschränkt Erträge bei unendlichem Horizont durch $r_{\max}/(1-\gamma)$ und legt den Horizont $1/(1-\gamma)$ fest |

<!--en:9d904109-->
| **Terminierung und Abbruch** | $y_{T-1} = r_{T-1} + \gamma(1 - \textrm{term})V(s_T)$ | Der Wert der Fortsetzung entfällt nur am tatsächlichen Prozessende, nie beim Abbruch am Zeitlimit |

<!--en:b514e788-->
| **Markov-Zustand** | $P(s_{t+1} \mid s_t, a_t, s_{<t}) = P(s_{t+1} \mid s_t, a_t)$ | Lässt sich durch Ergänzen von Geschwindigkeiten, Restzeit oder Informationen aus der Vorgeschichte herstellen |

<!--en:80707c61-->
| **Potentialbasierte Belohnungsformung** | $\tilde{r} = r + \gamma\Phi(s') - \Phi(s)$, $\Phi(\textrm{term}) = 0$ | Auf Rundwegen entsteht eine Teleskopsumme. Dadurch bleibt die optimale Strategie auch bei häufigen Fortschrittssignalen erhalten |

<!--en:16d02d10-->
| **Aufgabe mit Prüfprogramm** | $x \sim \mu_0,\; y \sim \pi_\theta(\cdot \mid x),\; r(x, y)$ | Reduziert die Textgenerierung großer Sprachmodelle mit deterministischen Übergängen auf einen kontextuellen Banditen mit einem Schritt |
