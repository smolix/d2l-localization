<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:c45e1dc7-->
In jedem Abschnitt dieses Kapitels haben wir ein Verfahren kennengelernt und an einer kleinen Aufgabe gemessen, wie gut es funktioniert. In der neueren Literatur begegnen uns Varianten dieser Verfahren. Meist ändern sie einen Aspekt des Updates: wie es berechnet wird, welche Daten es verwenden darf oder welche Größe es begrenzt. Wir stellen die gängigsten Varianten in der Reihenfolge des Kapitels vor. Für jede erläutern wir, welches Problem sie löst und was sie dafür ändert. Den Abschluss bilden Quellen zum Nachschlagen und ein Projekt, in dem Sie die vollständige Trainingsschleife für das Post-Training aus :numref:`sec_rl_sequences` aufbauen.

<!--en:e16fc97d-->
## Strategieoptimierung in der Praxis

<!--en:3a647a0d-->
PPO aus :numref:`sec_ppo` erreicht bei CartPole die maximale Punktzahl. Um veröffentlichte Ergebnisse auf schwierigeren Aufgaben zu reproduzieren, muss man jedoch viele kleine Details der Implementierung ändern, die in den Arbeiten selten zur Sprache kommen. :citet:`Huang.Dossa.Raffin.ea.2022` haben 37 solcher Entscheidungen den jeweiligen Zeilen im Referenzcode zugeordnet. Die folgenden wirken sich am stärksten auf das Update aus:

<!--en:e3d9467b-->
| Detail | Auswirkung |

<!--en:d331c4dc-->
| Normalisierung von Beobachtungen und Belohnungen | laufend aktualisierte Statistiken halten die Eingaben des Netzes und die Zielwerte des Kritikers in der Größenordnung von eins |

<!--en:b34efdf3-->
| Normalisierung der Vorteile pro Minibatch | die Standardisierung pro Batch aus :numref:`sec_baselines` wird auf jeden Minibatch einzeln angewandt |

<!--en:2cfda119-->
| schrittweise Absenkung der Lernrate auf null | die Schritte werden gegen Ende des Trainings kleiner |

<!--en:203d32d7-->
| orthogonale Initialisierung, kleine Anfangsgewichte im Strategiekopf | die anfängliche Strategie ist nahezu gleichverteilt, sodass die ersten Wahrscheinlichkeitsverhältnisse nahe bei 1 bleiben |

<!--en:d2749ec0-->
| Begrenzung der globalen Gradientennorm | damit ist auch jeder Schritt im Parameterraum begrenzt |

<!--en:d6bc0311-->
| gemeinsamer Verlust für Akteur und Kritiker | Strategie-, Wert- und Entropieterm nutzen denselben Durchlauf pro Minibatch, während :numref:`sec_ppo` zuerst den Kritiker anpasst |

<!--en:2b7af87a-->
| Clipping beim Verlust der Wertfunktion | ein Band begrenzt, wie stark sich die Schätzung des Kritikers ändern darf. Nur ein Teil der Implementierungen verwendet es |

<!--en:eadfa0d7-->
| Early Stopping anhand der geschätzten KL-Divergenz | die Epochen für einen Batch enden, sobald die gemessene KL-Divergenz einen Schwellenwert überschreitet |

<!--en:daea6e43-->
Zwei Studien untersuchten, welche dieser Entscheidungen ins Gewicht fallen. :citet:`Engstrom.Ilyas.Santurkar.ea.2020` zeigten, dass die Wahrscheinlichkeitsverhältnisse bei PPO ohne diese Details ungehindert aus dem Clipping-Band laufen. Mit denselben Details schneidet TRPO etwa so gut ab wie PPO. Ein Großteil des Vorteils von PPO geht also auf den Code rund um die Zielfunktion zurück. :citet:`Andrychowicz.Raichuk.Stanczyk.ea.2021` trainierten Hunderttausende Agenten über ein Raster solcher Einstellungen und leiteten daraus Empfehlungen ab. Die Normalisierung der Beobachtungen half bei den meisten Aufgaben. Keine der beiden Studien fand einen Nutzen für das Clipping beim Verlust der Wertfunktion. Wenn Sie neben :numref:`sec_ppo` eine gepflegte Implementierung in einer einzigen Datei lesen, können Sie die Einträge der Tabelle im Code nachvollziehen.

<!--en:973bddb1-->
Die Arbeit zu PPO beschreibt eine zweite Möglichkeit, den Schritt zu begrenzen :cite:`Schulman.Wolski.Dhariwal.ea.2017`. Ein KL-Strafterm erfasst die Abweichung von der alten Strategie (engl. policy). Liegt die KL-Divergenz nach einem Batch über dem Zielwert, verdoppelt sich das Gewicht des Strafterms, liegt sie darunter, halbiert es sich. In den Experimenten der Autoren schnitt dieser Strafterm schlechter ab als das Clipping. Er ist eng mit dem Strafterm mit fester Referenz aus :numref:`sec_regularized` verwandt.

<!--en:f4fdeda7-->
Beim verteilten Training sammeln viele Akteure Daten mit Kopien der Strategie, die dem lernenden Agenten um mehrere Updates hinterherhinken. Das V-trace-Ziel von IMPALA :cite:`Espeholt.Soyer.Munos.ea.2018` korrigiert die $n$-Schritt-Ziele aus :numref:`sec_actorcritic` mit Wichtigkeitsverhältnissen, die nach oben auf eins begrenzt sind. Die Begrenzung hält die Varianz in Schranken. Dafür schätzen die Zielwerte den Wert einer Strategie, die zwischen den Strategien der Akteure und der des lernenden Agenten liegt.

<!--en:a0168576-->
Läuft der Simulator selbst auf der GPU, führen Tausende Umgebungen ihre Schritte parallel aus. Die Kosten bemessen sich dann an der tatsächlich verstrichenen Zeit statt an der Anzahl der Samples :cite:`Makoviychuk.Wawrzyniak.Guo.ea.2021`. Für Roboter mit Beinen und für die geschickte Handhabung von Objekten ist PPO mit sehr großen Batches und Domänenrandomisierung (engl. domain randomization) dann das Standardverfahren :cite:`Rudin.Hoeller.Reist.ea.2022`. Führt reale Hardware die Schritte aus, kommen bevorzugt die weiter unten beschriebenen Off-Policy- und modellbasierten Verfahren zum Einsatz.

<!--en:af00877e-->
## Lernen von Wertfunktionen nach DQN

<!--en:eeb56867-->
Das DQN aus :numref:`sec_dqn` benötigt Zehntausende Schritte bei CartPole. Indem es das Maximum über verrauschte Schätzungen bildet, überschätzt es die Werte. Jedes Nachfolgeverfahren ändert einen Bestandteil. Ein Mehrschrittziel setzt den $n$-Schritt-Ertrag (engl. return) aus :numref:`sec_actorcritic` in den Zielwert für Q-Learning ein. So erreicht eine Belohnung (engl. reward) frühere Zustände in weniger Updates. Beim priorisierten Experience-Replay (engl. prioritized replay) :cite:`Schaul.Quan.Antonoglou.ea.2016` ist die Wahrscheinlichkeit, einen Übergang zu ziehen, proportional zu seinem TD-Fehler. Eine neue Gewichtung gleicht die dadurch entstehende Verzerrung aus. Ein Dueling-Netz :cite:`Wang.Schaul.Hessel.ea.2016` teilt die Ausgabe in einen Zustandswert und einen Vorteil (engl. advantage) je Aktion auf. So tragen die Übergänge jeder Aktion dazu bei, den gemeinsamen Zustandswert zu lernen.

<!--en:a747ee68-->
Verteilungsbasierte Verfahren lernen mit einer Bellman-Gleichung für Verteilungen die gesamte Verteilung des Ertrags statt nur seines Mittelwerts :cite:`Bellemare.Dabney.Munos.2017`. :citet:`Lyle.Castro.Bellemare.2019` zeigten, dass diese Verfahren bei tabellarischer und linearer Darstellung dieselben Erwartungswerte lernen wie mit dem gewöhnlichen Update. Ihr Vorteil ergibt sich also daraus, wie das informationsreichere Ziel ein tiefes Netz trainiert. Rainbow :cite:`Hessel.Modayil.vanHasselt.ea.2018` kombinierte sechs dieser Änderungen, darunter den Doppelschätzer aus :numref:`sec_dqn`. Die Ablationsstudie auf Atari ergab, dass priorisiertes Experience-Replay und Mehrschrittziele am meisten beitrugen, der Doppelschätzer und der Dueling-Kopf am wenigsten. Ein späteres Ergebnis weist in dieselbe Richtung wie das von Lyle et al. Verteilt man den skalaren Zielwert auf benachbarte Intervalle des diskretisierten Ertrags und trainiert den Kritiker mit einem Kreuzentropieverlust, gelingt der Übergang zu großen Netzen auch dort, wo der quadratische Verlust versagt :cite:`Farebrother.Orbay.Vuong.ea.2024`. Die weiter unten beschriebenen Verfahren Dreamer und TD-MPC2 trainieren ihre Wertfunktionen mit einem solchen Verlust.

<!--en:8ed574de-->
:citet:`vanHasselt.Doron.Strub.ea.2018` maßen unter vielen Trainingsbedingungen, wie oft die tödliche Triade (engl. deadly triad) aus :numref:`sec_dqn` bei tiefen Q-Netzen Probleme verursacht. Unbeschränkte Divergenz war selten. Häufig stiegen die Schätzungen jedoch vorübergehend über jeden erreichbaren Ertrag. Mehrschrittziele und Zielnetze machten solche Phasen seltener. Zwei neuere Ansätze verzichten ganz auf die Gegenmaßnahmen aus :numref:`sec_dqn`. PQN :cite:`Gallici.Fellows.Ellis.ea.2025` trainiert mit Batches aus vielen parallelen Umgebungen und verwendet Schichtnormalisierung im Netz. Es kommt ohne Replay-Puffer und ohne Zielnetz aus. Streaming-Verfahren :cite:`Elsayed.Vasan.Mahmood.2024` lernen aus jedem Übergang genau einmal, sobald er eintrifft. Dazu normalisieren sie Beobachtungen und Belohnungen und begrenzen die Größe jedes Updates. Experience-Replay und eine eingefrorene Kopie des Netzes sind also nur eine Möglichkeit, mit korrelierten Daten und veränderlichen Zielwerten umzugehen.

<!--en:6ccc93a2-->
## Kontinuierliche Regelung seit 2018

<!--en:6a0c1161-->
TD3 :cite:`Fujimoto.vanHoof.Meger.2018` führte drei Änderungen gegenüber DDPG :cite:`Lillicrap.Hunt.Pritzel.ea.2016` ein, dessen deterministischer Akteur selbst dort höheren Schätzwerten folgt, wo sie auf Fehler des Kritikers zurückgehen. Erstens verwendet TD3 zwei Kritiker, wie sie später auch :numref:`sec_sac` einsetzte. Zweitens aktualisiert TD3 den Akteur seltener als die Kritiker. Drittens addiert TD3 Rauschen zur Aktion im Zielwert, damit der Akteur keine schmale Spitze in der Wertschätzung des Kritikers ausnutzen kann.

<!--en:5e87cef5-->
Die meisten Fortschritte gelangen seither durch mehr Gradientenschritte pro Umgebungsschritt, ohne dass die Schätzungen des Kritikers aus dem Ruder liefen. Damit steigt das Verhältnis von Updates zu Daten aus :numref:`sec_sac`. REDQ :cite:`Chen.Wang.Zhou.ea.2021` verwendet ein großes Ensemble von Kritikern und nimmt das Minimum der Schätzungen zweier zufällig gewählter Mitglieder. Das ermöglicht viele Updates pro Schritt. DroQ :cite:`Hiraoka.Imagawa.Hashimoto.ea.2022` ersetzt das Ensemble durch zwei Kritiker mit Dropout und Schichtnormalisierung. CrossQ :cite:`Bhatt.Palenicek.Belousov.ea.2024` bildet eine Ausnahme. Es erreicht mit einem Update pro Schritt dieselbe Leistung wie REDQ und kommt ohne Zielnetze aus. Sein Kritiker verwendet Batchnormalisierung und verarbeitet die aktuellen und die nächsten Zustand-Aktion-Paare in einem einzigen Batch. Damit gelten für beide dieselben Statistiken. BRO :cite:`Nauman.Ostaszewski.Jankowski.ea.2024` vergrößert den Kritiker zu einem breiten Residualnetz mit Schichtnormalisierung und ergänzt ihn um optimistische Exploration. Mit PQN haben diese Verfahren die Normalisierung innerhalb des Kritikers gemeinsam.

<!--en:dbd6958e-->
Neben der Überschätzung begrenzt ein weiterer Effekt die Zahl der Updates pro Umgebungsschritt. :citet:`Nikishin.Schwarzer.DOro.ea.2022` stellten fest, dass sich Agenten bei vielen Updates pro Schritt zu stark an ihre ersten Daten anpassen und aus späteren Daten nur noch wenig lernen. Diese Übergewichtung früher Erfahrungen (engl. primacy bias) lässt sich ausgleichen: Setzt man die Netze regelmäßig zurück und behält den Replay-Puffer bei, lernen die Agenten wieder. Mit solchen Rücksetzungen lässt sich die Zahl der Updates pro Schritt wesentlich weiter erhöhen, bevor sich die Erträge nicht mehr verbessern :cite:`DOro.Schwarzer.Nikishin.ea.2023`.

<!--en:4671e4cb-->
## Modelle, simulierte Trajektorien und Suche

<!--en:ef7763ae-->
Der Planer aus :numref:`sec_model_based` wertet sein Modell für jeden realen Schritt Tausende Male aus und vertraut ihm nur in der Nähe der Daten. Die folgenden Verfahren nutzen ein gelerntes Modell anders. Dyna :cite:`Sutton.1990` simuliert damit zusätzliche Übergänge für einen modellfrei lernenden Agenten. Das durch Zählen geschätzte Modell aus :numref:`sec_qlearning` liefert so zwischen den realen Schritten Daten für Q-Learning-Updates. MBPO :cite:`Janner.Fu.Zhang.ea.2019` hält die simulierten Trajektorien kurz. Sie beginnen bei Zuständen aus dem Replay-Puffer und reichen nur wenige Schritte weit. MBPO mischt sie in die Batches von Soft Actor-Critic. So pflanzen sich Modellfehler nur über wenige Schritte fort und bleiben klein. REDQ aus dem vorigen Abschnitt erreichte die Dateneffizienz (engl. sample efficiency) von MBPO ohne Modell. Das legt nahe, dass ein Großteil des Vorteils von MBPO auf die höhere Zahl von Updates pro realem Schritt zurückging.

<!--en:ae2866c9-->
PETS :cite:`Chua.Calandra.McAllister.ea.2018` passt ein Ensemble probabilistischer Netze an und plant mit der Kreuzentropiemethode anhand von Stichproben aus diesem Ensemble. Wo viele Daten vorliegen, stimmen die Mitglieder überein, anderswo weichen ihre Vorhersagen voneinander ab. Diese Abweichungen zeigen also, wo ein Plan unzuverlässig ist. Dreamer :cite:`Hafner.Pasukonis.Ba.ea.2025` lernt aus Bildern ein rekurrentes Dynamikmodell in einem ebenfalls gelernten latenten Raum. Akteur und Kritiker trainiert Dreamer ausschließlich mit simulierten latenten Trajektorien. Dabei verwendet das Verfahren für viele Anwendungsgebiete dieselben Hyperparameter.

<!--en:4ca316a1-->
Die Suche mit einem gelernten Wert schließt den Kreis aus :numref:`sec_model_based`. AlphaZero :cite:`Silver.Schrittwieser.Simonyan.ea.2017,Silver.Hubert.Schrittwieser.ea.2018` sucht mit den Spielregeln als Modell und bewertet die Blätter des Suchbaums mit einem Wertnetz. Die Entscheidungen der Suche überträgt es durch Destillation auf die Strategie. MuZero :cite:`Schrittwieser.Antonoglou.Hubert.ea.2020` lernt ein Modell, das nur Belohnungen, Werte und Strategien vorhersagt, keine Beobachtungen. In diesem Modell führt es die Suche aus. TD-MPC2 :cite:`Hansen.Su.Wang.2024` plant in einem latenten Modell über kurze Horizonte und schließt jeden Plan mit einem gelernten Wert ab.

<!--en:e1a51a38-->
Ein gelerntes Modell hilft auch dann, wenn keine neuen Daten erhoben werden dürfen. Voraussetzung ist, dass der offline lernende Agent dem Modell dort misstraut, wo wenige Daten vorliegen. MOPO :cite:`Yu.Thomas.Yu.ea.2020` und MOReL :cite:`Kidambi.Rajeswaran.Netrapalli.ea.2020` verringern die simulierten Belohnungen dort, wo die Vorhersagen eines Ensembles von Modellen voneinander abweichen. Damit übertragen sie den Pessimismus aus :numref:`sec_offline` auf ein Modell.

<!--en:ddb2a44d-->
## Offline-Verfahren für große Probleme

<!--en:3e2199a9-->
Der Strafterm aus :numref:`sec_offline` nimmt umgekehrt proportional zur Quadratwurzel der Häufigkeit ab, mit der ein Zustand-Aktion-Paar in den aufgezeichneten Daten vorkommt. Bei kontinuierlichen Aktionen ist diese Häufigkeit für fast jede von einer gelernten Strategie vorgeschlagene Aktion null. Tiefe Offline-Verfahren ersetzen die Häufigkeit deshalb durch eine Größe, die ein Netz schätzen kann. Sie unterscheiden sich darin, was sie einschränken:

<!--en:0812710f-->
| Verfahren | Was schränkt es ein? | Wie? |

<!--en:03f34a7b-->
| BCQ :cite:`Fujimoto.Meger.Precup.2019` | die Aktionen, die die Strategie wählen darf | ein generatives Modell der aufgezeichneten Aktionen schlägt Kandidaten vor, unter denen die Strategie wählt |

<!--en:617d5a47-->
| CQL :cite:`Kumar.Zhou.Tucker.ea.2020` | die Werte | ein Verlustterm senkt $Q$ für die Aktionen der Strategie und erhöht den Wert für die aufgezeichneten Aktionen |

<!--en:6727e182-->
| IQL :cite:`Kostrikov.Nair.Levine.2022` | die Aktionen, die das Backup abfragt | $V$ wird an ein oberes Expektil von $Q$ angepasst, in das nur aufgezeichnete Aktionen eingehen. Eine mit den Vorteilen gewichtete Regression liefert die Strategie |

<!--en:d4c9ebf6-->
| TD3+BC :cite:`Fujimoto.Gu.2021` | den Abstand der Strategie von den Daten | zum Verlust des Akteurs von TD3 kommt ein Term für das Verhaltensklonen hinzu |

<!--en:4facf895-->
Die andere Familie bilden Sequenzmodelle, die auf den Ertrag konditioniert sind. :numref:`sec_offline` behandelt diese Modelle und ihre Grenzen.

<!--en:bb6913cb-->
Die meisten dieser Verfahren passen eine Gauß-Strategie oder eine deterministische Strategie an. Demonstrationen folgen jedoch oft keinem dieser Muster. Menschen lösen dieselbe Aufgabe auf unterschiedliche Weise. Passt man eine einzige Gauß-Verteilung an zwei Lösungswege an, mittelt sie diese zu einer Aktion, die niemand ausgeführt hat. Diffusionsstrategien :cite:`Chi.Feng.Du.ea.2023` klonen die Demonstrationen mit den generativen Modellen aus :numref:`chap_diffusion`, die mehrere Modi zugleich darstellen können. In der Robotik sind Diffusionsstrategien inzwischen die übliche Form des Verhaltensklonens (engl. behavior cloning). Offline-Verfahren verwenden ein solches Modell dann als Strategie und begrenzen wie oben beschrieben ihre Abweichung von den aufgezeichneten Daten :cite:`Wang.Hunt.Zhou.2023`. Oft schließt sich das im Folgenden beschriebene Online-Fine-Tuning an.

<!--en:136660bc-->
Bei vielen Anwendungen liegen bereits aufgezeichnete Daten vor, und es lassen sich in geringem Umfang neue Daten erheben. Cal-QL :cite:`Nakamoto.Zhai.Singh.ea.2023` kalibriert die pessimistischen Werte so, dass sie nie unter die Werte einer Referenzstrategie fallen, etwa der Strategie, mit der die Daten erhoben wurden. So verhindert Cal-QL den Einbruch des Ertrags, den zu niedrige Werte sonst zu Beginn des Online-Fine-Tunings verursachen. RLPD :cite:`Ball.Smith.Kostrikov.ea.2023` überspringt das Offline-Training und setzt von Anfang an online einen Off-Policy-Agenten ein. Die Hälfte jedes Batchs stammt aus den aufgezeichneten Daten. Die Kritiker verwenden die zuvor beschriebene Normalisierung.

<!--en:9fa65a88-->
## Varianten des Post-Trainings

<!--en:f4229ff7-->
:numref:`sec_rl_sequences` setzt das Update für das Post-Training aus dem Gruppengewicht aus :numref:`sec_baselines`, dem begrenzten Wahrscheinlichkeitsverhältnis von PPO und dem KL-Strafterm aus :numref:`sec_regularized` zusammen. Dort haben wir auch DAPO, Dr. GRPO und GSPO kennengelernt. Beim Training großer Modelle für schlussfolgerndes Denken :cite:`Shao.Wang.Zhu.ea.2024,DeepSeekAI.2025` traten mit diesem Update Probleme auf, die in der Aufgabe mit nur einem Token nicht sichtbar werden. Die Varianten gehen auf diese Probleme ein. DAPO :cite:`Yu.Zhang.Zhu.ea.2025` mittelt den Verlust zudem direkt über alle Token eines Batchs, ohne zuerst den Mittelwert für jede Antwort zu bilden. Eine lange Antwort trägt damit proportional zu ihrer Länge bei. DAPO passt auch die Belohnung für Antworten an, die an der Längengrenze abgeschnitten werden. Dr. GRPO :cite:`Liu.Chen.Li.ea.2025` führt zwei Verzerrungen auf die beiden Divisionen zurück, die es weglässt. Die Division durch die Antwortlänge verringert die Strafe pro Token bei langen falschen Antworten. Dadurch werden solche Antworten im Training länger. Die Division durch die Standardabweichung der Gruppe gibt denjenigen Prompts das größte Gewicht, deren Antworten fast alle richtig oder fast alle falsch sind. GSPO :cite:`Zheng.Liu.Li.ea.2025` wurde entwickelt, um das Training großer Mixture-of-Experts-Modelle zu stabilisieren, deren Wahrscheinlichkeitsverhältnisse pro Token besonders stark rauschen.

<!--en:11549c34-->
Prozessbelohnungen ändern, woher die Belohnung kommt. :citet:`Lightman.Kosaraju.Burda.ea.2024` trainierten ein Belohnungsmodell mit menschlichen Labels für einzelne Denkschritte. Es beurteilte Lösungen besser als ein Modell, das nur mit Labels für die Endergebnisse trainiert wurde. Eine gelernte Prozessbelohnung bewertet jeden Schritt einzeln, wie es auch die Präfixwerte aus :numref:`sec_rl_sequences` tun. Sie bringt aber auch die ausnutzbaren Fehler einer gelernten Belohnung mit sich, die :numref:`sec_regularized` gemessen hat.

<!--en:9d0a786d-->
## Über das Kapitel hinaus

<!--en:05b55ee2-->
Vier Annahmen dieser beiden Kapitel können verletzt sein. Lernen mehrere Agenten gleichzeitig, gehören die anderen jeweils zur Umgebung des einzelnen Agenten. Damit ist nichts mehr stationär. Die Kombination von Spiel gegen sich selbst und Suche erreichte im No-Limit-Poker mit zwei Spielern übermenschliche Spielstärke :cite:`Brown.Sandholm.2017`. Erhält der Agent Beobachtungen statt des Zustands, gilt die Markov-Eigenschaft aus :numref:`sec_mdp` für die Beobachtungen nicht mehr. Die Strategie braucht dann ein Gedächtnis, etwa ein rekurrentes Netz oder eine Wahrscheinlichkeitsverteilung über die möglichen Zustände :cite:`Kaelbling.Littman.Cassandra.1998`. Wenn sich Aufgaben ändern, kann ein Agent lernen, sich mit wenigen Episoden an eine neue Aufgabe anzupassen (Meta-Lernen). Oder er lernt Teilstrategien, die über viele Schritte handeln, und zugleich eine Strategie, die zwischen ihnen wählt (hierarchisches Lernen). Sind die Belohnungen so selten, dass zufällige Aktionen fast nie eine erzielen, erkunden die $\epsilon$-Greedy-Strategie und ein Entropiebonus die Umgebung zu langsam. Der Agent muss gezielt aufsuchen, was er noch nicht gesehen hat. Ein Netz führt keine Besuchszahlen, aus denen sich der Bonus aus :numref:`sec_exploration` berechnen ließe. Tiefe Verfahren verwenden deshalb eine Größe, die ein Netz liefern kann: die Pseudo-Besuchszahl eines Dichtemodells :cite:`Bellemare.Srinivasan.Ostrovski.ea.2016` oder den Vorhersagefehler eines Netzes. Dieses Netz sagt entweder den nächsten Zustand :cite:`Pathak.Agrawal.Efros.ea.2017` oder die Ausgabe eines festen, zufällig initialisierten Netzes :cite:`Burda.Edwards.Storkey.ea.2019` vorher. Die jeweilige Größe kommt dort zur Belohnung hinzu, wo der Agent wenig Erfahrung hat. Alternativ folgt der Agent eine ganze Episode lang einem zufällig gezogenen Mitglied aus einem Ensemble von Wertfunktionen. Die Mitglieder kommen dort zu unterschiedlichen Schätzungen, wo wenige Daten vorliegen :cite:`Osband.Blundell.Pritzel.ea.2016`. Murphys Übersichtsarbeit :cite:`Murphy.2025` behandelt alle vier Fälle.

<!--en:2619697c-->
## Nachschlagewerke, Benchmarks und Codesammlungen

<!--en:a398a79c-->
Zu den Grundlagen aus :numref:`chap_reinforcement_learning` gehören die Implementierungen von CleanRL mit je einer Datei pro Verfahren :cite:`Huang.Dossa.Ye.ea.2022` und die Zusammenstellung der Implementierungsdetails von PPO :cite:`Huang.Dossa.Raffin.ea.2022`. Die folgenden Quellen ergänzen einzelne Abschnitte dieses Kapitels:

<!--en:4ec45e61-->
- Die Kapitel 11 bis 13 bei Sutton und Barto :cite:`Sutton.Barto.2018` behandeln den $\lambda$-Ertrag und die Berechtigungsspuren (engl. eligibility traces) aus :numref:`sec_actorcritic` sowie die darauf aufbauenden Akteur-Kritiker-Verfahren. Sie untersuchen auch die tödliche Triade aus :numref:`sec_dqn` und zeigen, was sich beweisen lässt, sobald einer ihrer drei Bestandteile wegfällt. Kapitel 8 entwickelt Dyna und die Planung mit einem gelernten Modell (:numref:`sec_model_based`).

<!--en:5ced3d7c-->
- Die kontrollierte Studie von :citet:`Engstrom.Ilyas.Santurkar.ea.2020` ergänzt :numref:`sec_ppo`. Bei gleichen Implementierungsentscheidungen schneiden PPO und TRPO fast gleich gut ab. Ein Großteil des praktischen Vorteils von PPO beruht also auf Details, die in der Zielfunktion nicht zum Ausdruck kommen.

<!--en:13768bd2-->
- Der [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) :cite:`Raffin.Hill.Gleave.ea.2021` liefert auf die jeweilige Umgebung abgestimmte Hyperparameter für PPO (:numref:`sec_ppo`), DQN (:numref:`sec_dqn`) und SAC (:numref:`sec_sac`).

<!--en:79721658-->
- [rliable](https://github.com/google-research/rliable) :cite:`Agarwal.Schwarzer.Castro.ea.2021` liefert Konfidenzintervalle aus einem geschichteten Bootstrap-Verfahren und Leistungsprofile. Damit lassen sich Trainingsläufe beim tiefen bestärkenden Lernen auch bei wenigen Zufallsstartwerten vergleichen.

<!--en:47df5cc5-->
- Die Übersichtsarbeit von :citet:`Levine.Kumar.Tucker.ea.2020` knüpft an :numref:`sec_offline` an und führt von der Untersuchung des tabellarischen Falls zu tiefen Offline-Verfahren.

<!--en:8307001c-->
- [The RLHF Book](https://rlhfbook.com/) :cite:`Lambert.2026` entwickelt die Verfahren für das Post-Training, deren Update wir in :numref:`sec_regularized` und :numref:`sec_rl_sequences` zusammengesetzt haben, für Sprachmodelle in voller Größe.

<!--en:2c81a0f1-->
## Ein Projekt

<!--en:99499977-->
:numref:`sec_rl_sequences` setzt das Update für das Post-Training zusammen und wendet es auf eine einzelne Gruppe an. In diesem Projekt bauen Sie die Trainingsschleife um dieses Update und prüfen sie anhand der Ergebnisse, die sich in diesem Kapitel exakt berechnen lassen.

<!--en:01c88719-->
1. **Die Aufgabe.** Verwenden Sie die vier Prompts der Aufgabe mit Prüfprogramm. Beantworten Sie diese wie in :numref:`sec_rl_sequences` mit jeweils zwei Zifferntoken, sodass `04`, `12`, `64` und `81` die richtigen Antworten sind. Die Strategie ist eine Tabelle von Präferenzen: für die Zehnerziffer je Prompt und für die Einerziffer je Kombination aus Prompt und Zehnerziffer. Die Referenzstrategie ist gleichverteilt.

<!--en:254869dd-->
2. **Das Update.** Erweitern Sie `grpo_step` auf Antworten mit zwei Token. Ziehen Sie für jeden Prompt eine Gruppe von 16 Antworten, speichern Sie die Log-Wahrscheinlichkeiten ihrer Token und standardisieren Sie die Belohnungen innerhalb der Gruppe. Führen Sie dann $E$ Epochen Gradientenaufstieg mit Schrittweite 1 aus. Verwenden Sie dafür die begrenzte Ersatzzielfunktion pro Token mit $\epsilon_{\textrm{clip}} = 0.2$. Mitteln Sie diese über die Token der Gruppe und ziehen Sie das $\beta$-Fache des $k_3$-Terms aus :numref:`sec_rl_sequences` ab. Protokollieren Sie für jede Epoche den Anteil außerhalb des Bands.

<!--en:f1987787-->
3. **Die Kontrolle.** In der ersten Epoche ist jedes Wahrscheinlichkeitsverhältnis gleich 1 und der Anteil außerhalb des Bands gleich null. Für $E = 1$ entspricht das Update also dem REINFORCE-Schritt mit innerhalb der Gruppe standardisierten Gewichten.

<!--en:78a18370-->
4. **Zielwert für die Wiederverwendung.** Bei $\beta = 0$ und gleicher Schrittweite sollten vier Epochen pro Gruppe mit etwa halb so vielen gezogenen Antworten wie bei einer Epoche eine Erfolgsquote von 0.9 erreichen. Das sollte für jeden getesteten Zufallsstartwert gelten. Zugleich sollte der Anteil außerhalb des Bands von der ersten bis zur vierten Epoche steigen.

<!--en:dd0af492-->
5. **Zielwert für den Strafterm.** Verwenden Sie bei der Aufgabe mit einem Token das nachsichtige Prüfprogramm aus :numref:`sec_rl_sequences`. Setzen Sie das Leave-one-out-Gewicht aus :numref:`sec_baselines` und $\beta = 1/2$ ein und trainieren Sie, bis sich die Strategie stabilisiert. Beim $k_3$-Verlust sollte die Ausweichantwort für `8*8` etwa die Hälfte der Wahrscheinlichkeit behalten. Das ist der Maximierer der Zielfunktion mit der KL-Divergenz in der anderen Richtung (Aufgabe :exref:`ex_rl-sequences-kl-loss`). Geht der Strafterm in die Belohnung ein, sollte die Wahrscheinlichkeit der Ausweichantwort unter zwei Prozent bleiben. Das entspricht dem Optimum nach exponentieller Umgewichtung. Erklären Sie, warum das Leave-one-out-Gewicht für diese beiden Zahlen erforderlich ist und was sich durch eine Standardisierung innerhalb der Gruppe ändern würde.

<!--en:829dcf23-->
[Diskussionen](https://d2l.discourse.group/t/36873)
