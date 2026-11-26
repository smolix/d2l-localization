<!--en:ff4a4c24-->
# Tiefes bestärkendes Lernen

<!--en:b4a2a305-->
Der letzte Agent aus :numref:`chap_reinforcement_learning` schwang nach einigen Hunderttausend Schritten ein Pendel auf. Im Simulator kostet das wenig. Ein Roboterarm wäre vorher verschlissen. Auch ein Sprachmodell, das anhand seiner eigenen Antworten lernt, kann es sich nicht leisten, jeden Batch von Antworten nach einer einzigen Aktualisierung zu verwerfen. Der Agent wartete das Ende jeder Episode ab und nutzte jeden Batch nur einmal. Wie stark eine Aktualisierung seine Strategie (engl. policy) veränderte, blieb unbegrenzt. Wie viel lässt sich aus einem einzelnen Sample lernen? Und was geht schief, wenn wir versuchen, mehr daraus zu lernen?

<!--en:d7be8a07-->
Vier Ideen geben darauf eine Antwort. Jede bringt eine eigene Fehlerquelle mit sich. Ein Agent kann den verbleibenden Ertrag einer Episode mit einer erlernten Funktion schätzen, statt das Ende abzuwarten :cite:`Samuel.1959,Sutton.1988`. Anfangs ist die Schätzung verzerrt. Der Agent kann Daten einer früheren Strategie erneut verwenden, indem er sie neu gewichtet oder erneut einspielt :cite:`Lin.1992`. Die Daten veralten jedoch, und die Zielwerte ändern sich. Er kann bei jeder Aktualisierung begrenzen, wie stark sich das Verhalten der Strategie ändert :cite:`Kakade.Langford.2002,Schulman.Wolski.Dhariwal.ea.2017`. Das beschränkt die Größe jedes einzelnen Schritts. Wohin die Schritte insgesamt führen, bleibt offen. Der Agent kann ein Modell der Umgebung lernen und darin planen :cite:`Sutton.1990`. Dabei übernimmt er die Fehler des Modells, und bei jedem Schritt fällt Rechenaufwand an. Ein erlernter Zielwert, ein gemeinsames Netz für alle Zustände und Daten einer anderen Strategie können zusammen bewirken, dass die Werte divergieren. Bei einer Tabelle tritt diese Divergenz nie auf. Auf diesen Ideen beruhen Regler, Spielagenten und das Post-Training von Sprachmodellen.

<!--en:6a9175ed-->
Wir entwickeln diese Ideen in drei Etappen. Zunächst betrachten wir die On-Policy-Optimierung und die regularisierte Strategieoptimierung. In :numref:`sec_actorcritic` müssen wir das Ende der Episode nicht mehr abwarten. Wir nutzen dazu einen Kritiker mit Bootstrapping und schätzen Vorteile (engl. advantages) über mehrere Schritte. In :numref:`sec_ppo` verwenden wir jede Trajektorie für mehrere Aktualisierungen. Dabei begrenzen wir das Verhältnis der Aktionswahrscheinlichkeiten. In :numref:`sec_regularized` binden wir die Strategie an eine Referenzverteilung, wenn wir die Belohnung (engl. reward) selbst aus Vergleichen lernen. Als Nächstes betrachten wir Off-Policy-Replay und Planung. In :numref:`sec_dqn` stabilisieren eine eingefrorene Netzkopie als Zielnetz und Double DQN das neuronale Q-Learning mit erneut eingespielten Übergängen. In :numref:`sec_sac` kombinieren wir zwei Kritiker mit einem Akteur für kontinuierliche Aktionen, der nach dem Prinzip der maximalen Entropie lernt. So gelingt das Aufschwingen des Pendels in wenigen Tausend Schritten. In :numref:`sec_model_based` lernen wir die physikalische Dynamik aus wenigen Hundert Schritten und planen im erlernten Modell. Zum Schluss betrachten wir Situationen ohne Online-Exploration oder ohne Unsicherheit über die Zustandsübergänge. In :numref:`sec_offline` lernen wir konservativ aus einem festen Datenprotokoll. In :numref:`sec_rl_sequences` passen wir die Verfahren dieses Kapitels an das Post-Training von Sprachmodellen auf Tokenebene an.

<!--en:0a3d8bbd-->
Auf CartPole vergleichen wir die Verfahren für diskrete Aktionen anhand der Anzahl der Umgebungsschritte. Auf Pendulum erproben wir Soft Actor-Critic und das erlernte Modell. Bei der Aufgabe mit Prüfprogramm und in zwei kleinen Gitterwelten können wir die Ergebnisse exakt anhand geschlossener Lösungen überprüfen. Alle Beispiele laufen sowohl in PyTorch als auch in JAX in wenigen Minuten auf der CPU. :numref:`fig_rl_roadmap` und :numref:`tab_rl_map` ordnen alle Verfahren beider Kapitel ein. :numref:`sec_deep_rl-further-reading` gibt einen Überblick über weiterführende Ansätze. Der Buchteil über Sprachmodelle behandelt das präferenzbasierte Post-Training im großen Maßstab.

<!--en:fcb2e89d-->
Tiefes bestärkendes Lernen: Dateneffizienz und Funktionsapproximation<br>**Kritiker mit Bootstrapping · Trust-Regionen · KL-Strafterme · Replay-Puffer · Weltmodelle · Offline-Protokolle · Post-Training von LLMs**

<!--en:2a2457a4-->
::: {.slide title="Warum naive Strategiegradienten Interaktionen verschwenden"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:5b0222ea-->
- **Warten auf das Episodenende:** Mit Monte-Carlo-Erträgen $\hat{G}_t$ kann das Lernen erst beginnen, wenn die Episode endet oder das Zeitlimit erreicht ist.

<!--en:d2c602f2-->
- **Erfahrung nur einmal nutzen:** Nach einem einzigen Gradientenschritt verwerfen wir den gesamten Batch von $1{,}600$ Übergängen.

<!--en:4b61e0fb-->
- **Unbegrenzte Parameterschritte:** Ein einziger verrauschter Schritt kann die Aktionswahrscheinlichkeiten beliebig stark verschieben. Die Strategie kann dadurch unwiderruflich kollabieren.

<!--en:08485b7a-->
::: {.slide title="Vier Wege, aus jedem Sample mehr zu lernen"}

<!--en:1388122f-->
[Grundideen]{.kicker}

<!--en:6c7c55fc-->
- **1. Bootstrapping mit einem Kritiker (§{sec_actorcritic}, §{sec_dqn}–§{num:sec_sac}):** Der Kritiker ersetzt den verbleibenden Ertrag $\hat{G}_{t+1}$ durch eine Schätzung. Anfangs hat sie eine Verzerrung $e(s')$.

<!--en:4b7976f8-->
- **2. Frühere Übergänge erneut nutzen (§{sec_ppo}, §{sec_dqn}–§{num:sec_offline}):** Wir sparen neue Trajektorien, müssen aber mit einer Verteilungsverschiebung und veränderlichen Zielwerten rechnen.

<!--en:309b5e65-->
- **3. Strategieänderungen begrenzen (§{sec_ppo}–§{num:sec_regularized}, §{sec_rl_sequences}):** Die Schrittweite lässt sich über das Wahrscheinlichkeitsverhältnis oder die KL-Divergenz zur Referenzstrategie $\pi_{\textrm{ref}}$ beschränken.

<!--en:c929d493-->
- **4. Im Weltmodell planen (§{sec_model_based}):** Simulationen zur Testzeit ersetzen reale Versuche. Dabei summieren sich die Fehler entlang der simulierten Trajektorie.

<!--en:1c8c8cec-->
::: {.slide title="Die tödliche Triade: Wann Wertschätzungen eines Netzes divergieren"}

<!--en:c6154d07-->
[Grundlegendes Risiko]{.kicker}

<!--en:d6fc516a-->
- **1. Bootstrapping:** Die Zielwerte $y = r + \gamma \hat{V}_w(s')$ ändern sich mit den Vorhersagen des Netzes selbst.

<!--en:bcc0f0d4-->
- **2. Funktionsapproximation:** Gemeinsame Parameter $w$ koppeln alle Zustände miteinander.

<!--en:8d228ef2-->
- **3. Off-Policy-Daten:** Die Übergänge stammen aus einem Replay-Puffer oder einem festen Protokoll statt aus der aktuellen Strategie. Schon bei Bairds linearem Markov-Entscheidungsprozess führt das zu $\|\mathbf{w}\|_\infty \to \infty$ (§{sec_dqn}).

<!--en:d8bf700d-->
::: {.slide title="Einteilung der Verfahren des tiefen bestärkenden Lernens"}

<!--en:abaa9b03-->
[Algorithmen im Überblick]{.kicker}

<!--en:d85c6b8e-->
- **Zwei Achsen:** Jedes Verfahren aus Kapitel §{num:chap_reinforcement_learning} und §{num:chap_deep_rl} lässt sich danach einordnen, *was es lernt* (Strategie, Aktionswert oder Dynamikmodell) und *welche Daten es zur Aktualisierung nutzt* (On-Policy-Daten, Off-Policy-Replay oder feste Offline-Protokolle).

<!--en:29887adb-->
- **Von links nach rechts:** Auf reine On-Policy-Trajektorien (§{sec_actorcritic}) folgen begrenzte Aktualisierungen mit demselben Batch über mehrere Epochen (§{sec_ppo}), Off-Policy-Replay-Puffer (§{sec_dqn}–§{num:sec_model_based}) und reine Offline-Datensätze (§{sec_offline}–§{num:sec_rl_sequences}).

<!--en:ec7026bf-->
::: {.slide title="Wie viel die Verfahren in den Vergleichsaufgaben an Dateneffizienz gewinnen"}

<!--en:f83b166d-->
[Versuchsreihe]{.kicker}

<!--en:09d2058f-->
- **Kontinuierliches Pendulum-v1 (Ziel: $\ge -200$):** Der Datenbedarf sinkt von $\sim 300\text{k}$ Schritten (REINFORCE) auf $2.4\text{k}$–$11\text{k}$ (SAC, §{sec_sac}) und schließlich auf $200$–$800$ Schritte (MPC mit Weltmodell, §{sec_model_based}).

<!--en:4f979d80-->
- **Diskretes CartPole-v1 (Höchstwert $500$):** PPO über mehrere Epochen (§{sec_ppo}) halbiert den Datenbedarf gegenüber Updates über eine einzige Epoche ($\gtrsim 50\text{k} \to \sim 25\text{k}$ Schritte). DQN (§{sec_dqn}) löst die Aufgabe in $\sim 40\text{k}$ Schritten.

<!--en:03895502-->
- **Exakte Prüfung mit Tabellen und Prüfprogramm:** Der Vergleich mit Optima in geschlossener Form zeigt Reward-Hacking (§{sec_regularized}, §{sec_rl_sequences}) und Überschätzung beim Offline-Lernen (§{sec_offline}).

<!--en:258bf3c6-->
::: {.slide title="Acht Leitfragen zu Kapitel §{num:chap_deep_rl}"}

<!--en:123f19cf-->
[Lernleitfaden]{.kicker}

<!--en:bf63b610-->
| Abschnitt | Leitfrage | Mathematischer Kern |

<!--en:c46cf16b-->
| **§{sec_actorcritic} Actor-Critic und GAE** | Muss eine Strategie bis zum Episodenende warten? | TD-Fehler $\delta_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$ und rückwärts berechnete Summe $\sum_l (\gamma\lambda)^l \delta_{t+l}$ |

<!--en:258a6ca9-->
| **§{sec_ppo} Trust-Regionen und PPO** | Wie viele Aktualisierungen erlaubt ein Batch, ohne die Strategie zu gefährden? | Wichtigkeitsgewicht $\rho_t(\theta)$ pro Schritt und begrenzte Ersatzzielgröße $\min(\rho_t \hat{A}_t, \textrm{clip}(\rho_t)\hat{A}_t)$ |

<!--en:d89282f4-->
| **§{sec_regularized} KL-regularisiertes bestärkendes Lernen** | Wie weit darf eine Strategie von den Präferenzdaten abweichen? | Exponentielle Umgewichtung der Referenzverteilung $\pi^\star(a) \propto \pi_{\textrm{ref}}(a) e^{r(a)/\beta}$ und weiches Backup $V_\beta = \beta \log \sum_a \pi_{\textrm{ref}} e^{Q_\beta/\beta}$ |

<!--en:6d4eaf3b-->
| **§{sec_dqn} Deep Q-Networks** | Kann ein Netz aus jedem beliebigen Übergang Werte lernen? | Experience-Replay mit gleichverteilter Ziehung, eingefrorenes Zielnetz $Q_{w^-}$ und Entkopplung von Auswahl und Bewertung durch Double DQN |

<!--en:7ea30964-->
| **§{sec_sac} Soft Actor-Critic** | Wie lernen wir eine kontinuierliche Regelung in $3{,}000$ Schritten? | Zielterm für maximale Entropie $-\beta \log \pi$, Jacobi-Matrix der Stauchung einer Gaußverteilung, zwei Kritiker $\min_j Q_j$ |

<!--en:47c64869-->
| **§{sec_model_based} Weltmodelle und MPC** | Kann ein erlerntes physikalisches Modell das Aufschwingen planen? | Regression der Zustandsänderung $\hat{f}_\phi(s,a)$, Random Shooting mit Initialisierung aus dem bisherigen Plan, Strategiedestillation |

<!--en:d5eb5c1c-->
| **§{sec_offline} Offline-RL** | Was lässt sich ohne neue Versuche lernen? | Verknüpfung beobachteter Teiltrajektorien im empirischen Markov-Entscheidungsprozess und pessimistischer Strafterm $-\kappa/\sqrt{n(s,a)}$ (CQL / TD3+BC) |

<!--en:71cf2b77-->
| **§{sec_rl_sequences} Bestärkendes Lernen für Sequenzen** | Welche Verfahren bleiben anwendbar, wenn die Aktionen Token sind? | Zuordnung der Erfolgsbeiträge zu einzelnen Token, überprüfbare Belohnungen, GRPO mit $k_3$-KL-Verlust und DPO |
