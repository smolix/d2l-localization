<!--en:8e3f998d-->
# Optimierung in der Praxis

<!--en:060e9ca8-->
In unseren bisherigen kleinen Experimenten dauerte ein Trainingslauf nur wenige Sekunden. Wir konnten ganze Raster von Lernraten (engl. learning rates) durchprobieren und einen Lauf wiederholen, sobald ein Update zur Divergenz führte. Dauert ein Trainingslauf im produktiven Einsatz dagegen Tage oder Wochen, müssen die meisten Einstellungen vorab feststehen. Eine einzige ungebremste Gradientenspitze kann Stunden an Arbeit zunichtemachen. Und der letzte Checkpoint darf nicht davon abhängen, welchen Minibatch wir zufällig zuletzt ziehen.

<!--en:39b19ec8-->
Selbst bei extrem großen Modellen beruht zuverlässiges Training auf drei Maßnahmen. Ein Beispiel ist PaLM mit $540$ Milliarden Parametern: Beim Training auf $6{,}144$ TPU-Chips traten trotz aktivem Gradienten-Clipping rund $20$ plötzliche Verlustspitzen auf :cite:`chowdhery2022palm`. Die drei Maßnahmen sind:

<!--en:01e4dbf6-->
1. **Schutz vor vorübergehenden Instabilitäten:** **Gradienten-Clipping** anhand der globalen Norm (:numref:`sec_rnn-scratch`) begrenzt ungewöhnlich große Schritte beim stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD). Bei AdamW schützt es den Speicher für das zweite Moment $\mathbf{v}_t$ vor kurzzeitigen Gradientenspitzen. Hinzu kommt die Möglichkeit, zu einem früheren Checkpoint zurückzukehren.

<!--en:9253ca9b-->
2. **Glättung der Iterierten:** Ein **exponentieller gleitender Mittelwert** (engl. exponential moving average, EMA) der Modellgewichte beseitigt schnelle Schwankungen innerhalb der Rauschkugel (engl. noise ball). Dazu muss die Lernrate nicht bis auf null sinken.

<!--en:c96c8651-->
3. **Systematische Kalibrierung mit kurzen Probeläufen:** Wir wählen die Batchgröße (engl. batch size) anhand des Gerätedurchsatzes und der Größe $b_{\text{noise}}$ (:numref:`sec_batch_size`). Für den Optimierer übernehmen wir bewährte Standardwerte. Die maximale Lernrate kalibrieren wir in einem kurzen Lauf ($10\%$ der Trainingsdauer), bevor wir den gesamten Lernratenplan mit Aufwärmphase und anschließendem Abfall ausführen (:numref:`sec_scheduler`).

<!--en:b4c5e360-->
In diesem Abschnitt untersuchen wir diese drei Maßnahmen und verbinden sie zu einer vollständigen Trainingskonfiguration für ein bislang nicht abgestimmtes Modell mit $3.3$ Millionen Parametern und $4$ Blöcken: `TinyLM`.

<!--en:87bd8eb7-->
## Gradienten-Clipping und Wiederaufnahme des Trainings nach Spitzen

<!--en:36d967f9-->
Das **Gradienten-Clipping** anhand der globalen Norm kennen wir bereits von den rekurrenten Netzen in :numref:`sec_rnn-scratch`. Es fasst die Gradienten aller Parameter zu einem einzigen Vektor $\mathbf{g}$ zusammen und skaliert ihn, sobald seine euklidische Norm $\|\mathbf{g}\|_2$ einen Schwellenwert $\theta$ überschreitet:

<!--en:2fab7803-->
Da der Skalierungsfaktor $\theta / \|\mathbf{g}\|_2$ für alle Schichten gilt, bleibt die Richtung des Gesamtgradienten beim Clipping erhalten. Seine euklidische Norm beträgt danach höchstens $\theta$. Diesen Gradienten erhält der Optimierer.

<!--en:349acad3-->
### Clipping als Schutz oder als Dauerbremse

<!--en:dae9801e-->
In :numref:`sec_adam` divergierte SGD mit Momentum $0.9$ auf `TinyLM` gegen `inf`, und zwar bei $\eta = 1.0$ innerhalb der ersten $20$ Schritte. Ist $\eta = 1.0$ für die gesamte Verlustlandschaft zu groß oder nur für die ersten Schritte, in denen die frisch initialisierten Gewichte große Gradienten erzeugen? Zunächst laden wir den Datenstrom von `WikiText103` ($b = 64$, Fensterlänge $64$):

<!--en:fee13756-->
Anschließend trainiert `run(seed, lr, max_norm)` das Modell `TinyLM` über $2{,}000$ SGD-Schritte mit Momentum $0.9$. Dabei begrenzt die Funktion die Gradientennorm auf `max_norm` und gibt für jeden Schritt den Verlust und die Gradientennorm vor dem Clipping zurück:

<!--en:743056b6-->
Wir testen nun $(\eta, \theta) \in \{(1.0, \infty), (1.0, 1.0), (0.3, \infty), (3.0, 1.0)\}$ mit zwei Zufallsstartwerten. Wir protokollieren, bei wie vielen Schritten das Clipping greift ($\|\mathbf{g}_t\|_2 > \theta$), bei welchem Schritt es zuletzt greift und wie groß der Median der Gradientennorm ist:

<!--en:4e44306d-->
Bei $\eta = 1.0$ greift das Clipping mit dem Schwellenwert $\theta = 1.0$ nur während der ersten paar Dutzend Schritte (alle vor Schritt $50$). In den verbleibenden $1{,}950$ Schritten greift es kein einziges Mal mehr (Median der Gradientennorm: $\sim 0.25$). Damit gelingt das Training: Statt `inf` beträgt der Verlust nur $\sim 1.61\text{--}1.64$ Nat. Hier dient das Clipping als **Schutz**. Es greift bei $<2\%$ der Schritte, verhindert damit eine anfängliche Divergenz und fängt Gradientenrauschen mit schweren Verteilungsrändern ab :cite:`Zhang.Karimireddy.Veit.ea.2020`. Bei $\eta = 3.0$ dagegen greift das Clipping bei jedem Schritt, solange die Gradientennorm endlich bleibt. Das Training scheitert trotzdem: Der Verlust stagniert bei $100\text{--}200$ Nat oder erreicht durch Überlauf `inf`. Wirkt das Clipping dauerhaft als **Bremse**, ist die Lernrate $\eta$ zu groß :cite:`Godbole.Dahl.Gilmer.ea.2023`. Setzen Sie $\theta$ in der Praxis auf etwa $3\text{--}5\times$ den Median der Gradientennorm im stationären Zustand (typischerweise $\theta = 1.0$). Beobachten Sie den Anteil der Schritte, bei denen das Clipping greift.

<!--en:261fb223-->
### Clipping bei AdamW und Wiederaufnahme des Trainings nach Spitzen

<!--en:b20c82c5-->
Bei AdamW (:numref:`sec_adam`) ist der Schritt selbst bei einem einzelnen ungewöhnlich großen Gradienten $g \gg \sqrt{v_{t-1}}$ bereits begrenzt. Er beträgt höchstens $\eta(1-\beta_1)/\sqrt{1-\beta_2}$ ($\approx 3.2\,\eta$ bei $\beta_2 = 0.999$ oder $0.45\,\eta$ bei $\beta_2 = 0.95$). **Clipping schützt bei AdamW den Zustand $\mathbf{v}_t$ für das zweite Moment**, statt den unmittelbar anstehenden Schritt zu begrenzen. Eine unbegrenzte Gradientenspitze der Größe $100\times$ Normalwert trägt das $(1-\beta_2) \times 10^4$-Fache des üblichen Gradientenquadrats zu $v_t$ bei. Dadurch wächst der Nenner $\sqrt{\hat{v}_t}$, und das Lernen kommt für die nächsten $\sim 1/(1-\beta_2)$ Schritte praktisch zum Erliegen ($1{,}000$ Schritte bei $\beta_2 = 0.999$, $20$ Schritte bei $\beta_2 = 0.95$).

<!--en:a9c6171a-->
Trotz Gradienten-Clipping und zusätzlicher Regularisierung der Logits kann der Verlust plötzlich in die Höhe schnellen. Ein Beispiel für eine solche Regularisierung ist der $z$-Verlust $10^{-4}\log^2(\sum_j e^{o_j})$ von PaLM (Übung :exref:`ex_practice-logit-drift`). Das übliche Vorgehen :cite:`chowdhery2022palm,zhang2022opt` ist dann: **Kehren Sie zu einem Checkpoint zurück, der $\sim 100$ Schritte vor der Spitze gespeichert wurde, und überspringen Sie die $200\text{--}500$ Minibatches um den auslösenden Batch herum**. Dazu müssen Sie regelmäßig die Modellgewichte, die Momente $(\mathbf{m}_t, \mathbf{v}_t)$ des Optimierers, den Schrittzähler des Lernratenplans und die aktuelle Position im Datenstrom speichern.

<!--en:d7a7e8d2-->
## Mittelung der Gewichte (EMA)

<!--en:6ff92b20-->
Bei konstanter oder von null verschiedener Lernrate schwanken die stochastischen Iterierten fortwährend innerhalb der Rauschkugel um das Optimum (:numref:`sec_sgd`). Für eine Richtung mit Krümmung $\lambda_{\text{curv}}$ beträgt die Autokorrelationszeit $\sim 1/(\eta\lambda_{\text{curv}})$ Schritte. Wir berechnen einen **exponentiellen gleitenden Mittelwert (EMA)** $\bar{\mathbf{x}}_t$ der aktuellen Parameter $\mathbf{x}_t$ mit einem Abklingkoeffizienten $\alpha \in [0.99, 0.9999]$ :cite:`Polyak.Juditsky.1992`:

<!--en:c4f42261-->
Dieser Mittelwert erfasst jeweils die letzten $1/(1-\alpha)$ Schritte. So gleichen sich bei der Auswertung die Schwankungen in der Rauschkugel aus, ohne dass sich die Trajektorie des Trainings ändert.

<!--en:4696c73e-->
Wir vergleichen SGD bei konstanter Lernrate ($\eta = 0.3$) mit und ohne EMA der Gewichte ($\alpha = 0.999$, Fenster von $1{,}000$ Schritten $\approx 4.2$ Epochen, Beginn nach Epoche $5$, damit der Mittelwert anfangs nicht hinterherhinkt) mit einem Kosinus-Abfall (engl. cosine decay) der Lernrate. Wir trainieren $15$ Epochen ($3{,}525$ Schritte) lang das Modell `FashionCNN`. Zunächst laden wir `FashionMNIST` ($b = 256$) und definieren die Hilfsfunktion `test_acc` für die Validierungsgenauigkeit:

<!--en:6b98a3f9-->
Als Nächstes definieren wir die Hilfsfunktion `warmup_cosine` für den Lernratenplan und einen EMA der Gewichte:

<!--en:a5506f4b-->
In PyTorch berechnet `torch.optim.swa_utils.AveragedModel` den EMA. Hier müssen wir daher nur den Lernratenplan definieren.

<!--en:1e8688a2-->
`ema_of_weights` ist eine optax-Transformation, die den EMA im Optimiererzustand speichert. `averaged` erzeugt eine Kopie des Modells mit diesen Gewichten.

<!--en:f6379990-->
Die Trainingsschleife `train_fashion` verwendet gewöhnlichen SGD mit dem Lernratenplan `schedule(t)` über `num_epochs=15` Epochen. Am Ende jeder Epoche erfasst sie die Genauigkeit auf den zurückgehaltenen Testdaten, sowohl für die aktuellen Gewichte als auch für deren EMA (ab Epoche `start=5`):

<!--en:5b76685d-->
Wir führen `train_fashion` für zwei Zufallsstartwerte sowohl mit konstanter Lernrate ($\eta = 0.3$) als auch mit Kosinus-Abfall ($\eta: 0.3 \to 0$) aus. Wir vergleichen die Spannweite der Genauigkeit der aktuellen Gewichte in den letzten fünf Epochen mit der Genauigkeit des EMA am Ende des Trainings:

<!--en:5d7c9265-->
Tragen wir für den ersten Zufallsstartwert die Testgenauigkeit für jede Epoche auf, sehen wir, wie der EMA bei konstanter Lernrate die Schwankungen der aktuellen Gewichte in der Rauschkugel ausgleicht:

<!--en:abd3eb0a-->
Bei konstanter Lernrate schwankt die Testgenauigkeit der aktuellen Gewichte in den letzten fünf Epochen um einige Prozentpunkte. In einzelnen ungünstigen Epochen fällt sie sogar deutlich stärker ab. Die Testgenauigkeit des EMA steigt dagegen gleichmäßig am oberen Rand dieses Bereichs auf etwa $91\%$. Damit erreicht sie die Endgenauigkeit des Laufs mit Kosinus-Abfall, obwohl die Lernrate durchgehend konstant bleibt. Auf dem Plateau eines WSD-Plans lässt sich mit dem EMA der Gewichte das Ergebnis eines Trainingszweigs mit Abkühlphase kostengünstig annähern (:numref:`subsec_scheduler-branching`): Er erfordert keine zusätzlichen Schritte. Vollständig ersetzen kann er diesen Zweig allerdings nicht. Bei Sprachmodellen erzielt die Mittelung auf dem Plateau nur einen Teil des Gewinns durch die Abkühlphase :cite:`Hagele.Bakouch.Kosson.ea.2024`.

<!--en:4458ad3a-->
## Eine vollständige Trainingskonfiguration am Beispiel

<!--en:a5efbb5a-->
Wir führen nun die Ergebnisse des Kapitels zusammen und konfigurieren einen Trainingslauf über $3{,}000$ Schritte. Wir verwenden ein bislang nicht abgestimmtes Modell mit $4$ Blöcken, Breite $256$ und $4$ Köpfen: `TinyLM` mit $3.3$ Millionen Parametern. :numref:`tab_practice_recipes` fasst die veröffentlichten Optimiererkonfigurationen für das Pre-Training von vier führenden Modellen zusammen. In :numref:`tab_practice_symptoms` können wir für typische Symptome im Trainingsprotokoll die passenden Gegenmaßnahmen nachschlagen.

<!--en:d46501c2-->
:Veröffentlichte Optimiererkonfigurationen aus vier Läufen zum Pre-Training großer Sprachmodelle.

<!--en:34906255-->
| Modell | Optimierer | $\beta_1, \beta_2$ | Maximale Lernrate und Verlauf | Aufwärmphase | Clipping-Schwelle $\theta$ | Gewichtszerfall $\lambda$ | Batchgröße |

<!--en:f972e312-->
| Llama 3 405B :cite:`Grattafiori.Dubey.Jauhri.ea.2024` | AdamW | – | $8{\times}10^{-5}$, Kosinus-Abfall auf $8{\times}10^{-7}$ | 8000 Schritte | – | – | Anstieg: 4 Mio. $\to$ 16 Mio. Token |

<!--en:6db75bf6-->
| DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` | AdamW | 0.9, 0.95 | $2.2{\times}10^{-4}$, WSD mit Kosinus-Abfall auf $2.2{\times}10^{-5}$ | 2000 Schritte | 1.0 | 0.1 | Anstieg: 3072 $\to$ 15 360 Sequenzen |

<!--en:00aefb2f-->
| OLMo 2 7B :cite:`OLMo.2025` | AdamW | 0.9, 0.95 | $3{\times}10^{-4}$, Kosinus-Abfall auf $3{\times}10^{-5}$ | 2000 Schritte | 1.0 | 0.1 (ohne Einbettungen) | 1024 Sequenzen |

<!--en:747ef2b9-->
| Kimi K2 :cite:`Kimi.Team.2025` | MuonClip | – | $2{\times}10^{-4}$, WSD mit Kosinus-Abfall auf $2{\times}10^{-5}$ | 500 Schritte | – | 0.1 | 67 Mio. Token |

<!--en:0d0dc48b-->
:Diagnose von Optimierungsproblemen anhand protokollierter Größen (Verlust, Gradientennorm vor dem Clipping, Anteil der Schritte mit Clipping und relative Schrittweite $\|\Delta\mathbf{W}\|_{\text{RMS}}/\|\mathbf{W}\|_{\text{RMS}}$).

<!--en:4e72506d-->
| Symptom im Protokoll | Diagnose | Wichtigste Gegenmaßnahme |

<!--en:46cd5967-->
| Verlust divergiert in den ersten Schritten, Gradientennormen explodieren | Schrittweite überschreitet die durch die anfängliche lokale Schärfe bestimmte Grenze $2/L$ | Lineare Aufwärmphase oder Clipping anhand der globalen Norm ergänzen |

<!--en:8cb1edae-->
| Verlust stagniert bei Adam auf hohem Niveau, anfängliche relative Gewichtsänderungen $>10\%$ | Erster Schritt beim Kaltstart macht die Initialisierung zunichte | Lineare Aufwärmphase ergänzen ($1\%\text{--}5\%$ der Schritte) |

<!--en:485f3061-->
| Verlust stagniert bei Adam trotz Aufwärmphase auf hohem Niveau | Maximale Lernrate $\eta_0$ zu hoch | $\eta_0$ um den Faktor $2\text{--}3\times$ senken |

<!--en:9be8f6a7-->
| Clipping greift während des gesamten Trainings bei $>20\%\text{--}50\%$ der Schritte | Clipping wirkt als Dauerbremse | Zuerst $\eta_0$ senken, dann $\theta$ anpassen |

<!--en:770325ae-->
| Verlust erreicht früh ein Plateau mit hoher Varianz zwischen den Schritten | Stationäre Rauschkugel beim aktuellen Verhältnis $\eta / b$ | Lernrate gemäß $\eta_t \to 0$ senken, EMA der Gewichte auswerten oder $b$ erhöhen |

<!--en:0d0cd917-->
| Verlust sinkt bis zum letzten Schritt gleichmäßig mit konstanter Steigung | Budget für den insgesamt zurückgelegten Weg $B_T$ zu klein | Maximale Lernrate $\eta_0$ oder Gesamtschrittzahl $T$ erhöhen |

<!--en:e983568b-->
Nach dem Vorbild von :numref:`tab_practice_recipes` wählen wir für unser Modell mit $3.3\text{M}$ Parametern, `TinyLM`, folgende Einstellungen:

<!--en:3669c16f-->
1. **Batchgröße:** $b = 256$ Sequenzen. Hier erreicht der Hardwaredurchsatz sein Maximum. Die Batchgröße ist mit der Rauschskala des Gradienten nach der Aufwärmphase vergleichbar ($b_{\text{noise}} \sim 100\text{--}200$).

<!--en:7c87a104-->
2. **Optimierer und Schutzmaßnahmen:** Wir verwenden AdamW mit $(\beta_1, \beta_2) = (0.9, 0.95)$. Der entkoppelte Gewichtszerfall (engl. decoupled weight decay) mit $\lambda = 0.1$ wirkt ausschließlich auf die zweidimensionalen Matrizen der versteckten Schichten (:numref:`subsec_adamw-what-not-to-decay`). Hinzu kommen Gradienten-Clipping anhand der globalen Norm bei $\theta = 1.0$ und ein EMA der Gewichte mit kurzem Fenster ($\alpha = 0.995$).

<!--en:c44b9075-->
3. **Kalibrierung der maximalen Lernrate:** Drei Probeläufe über je $300$ Schritte ($10\%$ der Trainingsdauer) mit $\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}\}$ ergeben $\eta_0 = 3 \times 10^{-3}$ als besten Wert. Bei $<5\%$ der Schritte greift das Clipping. Damit bestätigt sich, dass $\theta = 1.0$ nur als Schutz dient.

<!--en:7c368a3b-->
4. **Lernratenplan:** Auf eine lineare Aufwärmphase von $60$ Schritten ($2\%$ von $T = 3{,}000$) folgt ein Kosinus-Abfall auf null. Damit liegt der Validierungsverlust (engl. validation loss) um $0.05\text{--}0.06$ Nat unter dem der Baseline mit konstanter Lernrate. Die Spannweite zwischen den Zufallsstartwerten beträgt $<0.01$ Nat.

<!--en:aa305fa2-->
Zunächst definieren wir `tiny_lm(seed)`. Die Funktion erzeugt das Modell mit $4$ Blöcken, Breite $256$ und $4$ Köpfen, `TinyLM`. Mit `sample_lm(b)` ziehen wir zufällige Trainingsfenster, um die Rauschskala zu schätzen:

<!--en:46534a27-->
Die Trainingsroutine `train(seed, schedule, num_steps)` konfiguriert AdamW mit $(\beta_1, \beta_2) = (0.9, 0.95)$ und einem Gewichtszerfall von $0.1$, der nur auf zweidimensionale Matrizen außerhalb der Einbettungen wirkt. Sie ergänzt Clipping anhand der globalen Norm (`max_norm=1.0`) und einen EMA der Gewichte (`alpha=0.995`). Sie gibt die Verlustwerte, das Modell mit den aktuellen Gewichten, das EMA-Modell und den Anteil der Schritte mit Clipping zurück:

<!--en:1ed556d6-->
Bei $b = 256$ messen wir `d2l.noise_scale` zum Zeitpunkt der Initialisierung. In drei Probeläufen über je $300$ Schritte ($10\%$ der Trainingsdauer) testen wir $\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}\}$ und bestimmen so die maximale Lernrate `peak`. Wir schätzen `noise_scale` nach $300$ Schritten erneut:

<!--en:b2f61dd7-->
Abschließend führen wir den vollständigen Lauf über $3{,}000$ Schritte mit zwei Zufallsstartwerten aus. Wir vergleichen unseren Lernratenplan (`warmup_cosine(peak, 60, 3000)`) und den zugehörigen EMA der Gewichte mit einer Baseline, deren Lernrate konstant bei `peak` liegt:

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:9fcc0871-->
**Gradienten-Clipping anhand der globalen Norm** ($\mathbf{g} \leftarrow \min(1, \theta/\|\mathbf{g}\|_2)\mathbf{g}$) wirkt als **Schutz**, wenn es nur bei einem kleinen Anteil ($<5\%$) früher oder ungewöhnlicher Schritte greift. Es verhindert eine anfängliche Divergenz bei SGD und schützt bei AdamW den Speicher für das zweite Moment $\mathbf{v}_t$ vor Gradientenspitzen. Greift das Clipping dagegen ständig, ist $\eta$ zu groß und das Clipping wirkt als **Bremse**. Der **exponentielle gleitende Mittelwert (EMA)** der Modellgewichte ($\bar{\mathbf{x}}_t = \alpha\bar{\mathbf{x}}_{t-1} + (1-\alpha)\mathbf{x}_t$) glättet stationäre Schwankungen in der Rauschkugel, ohne die Trajektorie des Trainings zu ändern. Bei konstanter Lernrate erreicht er die Genauigkeit eines vollständigen Kosinus-Abfalls. Für zuverlässiges Training in großem Maßstab wählen wir Batchgrößen nahe $b_{\text{noise}}$, die die Hardware auslasten. Wir verwenden AdamW mit $(\beta_1, \beta_2) = (0.9, 0.95)$ und entkoppeltem Gewichtszerfall $\lambda = 0.1$ ausschließlich für zweidimensionale Gewichtsmatrizen. Hinzu kommen Clipping anhand der globalen Norm bei $\theta = 1.0$, die Kalibrierung der maximalen Lernrate in kurzen Probeläufen und eine lineare Aufwärmphase mit anschließender Kosinus- oder WSD-Abkühlphase.

<!--en:2294243a-->
## Übungen

<!--en:235de320-->
1. [code] **Clipping als Schutz oder als Bremse.** Testen Sie $\theta \in \{0.01, 0.1, 0.5, 1.0, 4.0, \infty\}$ bei $\eta = 1.0$ mit `run` auf `TinyLM`. Erfassen Sie den Verlust am Ende des Trainings und den Anteil der Schritte mit Clipping. Vergleichen Sie die Ergebnisse mit Läufen, in denen Sie das Clipping durch eine lineare Aufwärmphase von $20$ oder $100$ Schritten ersetzen.
   :exlabel:`ex_practice-clipping-regimes`

<!--en:49367563-->
1. **Drift der Softmax-Logits und der $z$-Verlust von PaLM.** Betrachten Sie die Ausgabe-Logits $\mathbf{o} \in \mathbb{R}^V$ und die Partitionsfunktion $Z = \sum_{j=1}^V e^{o_j}$. Zeigen Sie, dass die Komponenten des Gradienten der Kreuzentropie in der Summe null ergeben ($\mathbf{1}^\top \nabla_{\mathbf{o}}\ell_{\text{CE}} = 0$). Zeigen Sie ferner, dass der zusätzliche Strafterm $10^{-4}\log^2 Z$ :cite:`chowdhery2022palm` einen rückstellenden Gradienten $2 \times 10^{-4}(\log Z)\,\mathbf{p}$ erzeugt. Dieser zieht $\log Z$ in Richtung $0$, ohne die Invarianz von $\mathbf{p} = \operatorname{softmax}(\mathbf{o})$ gegenüber einer einheitlichen Verschiebung der Logits zu ändern.
   :exlabel:`ex_practice-logit-drift`

<!--en:d24be433-->
1. [code] **Fenster und Startzeitpunkt des EMA der Gewichte.** Variieren Sie den Abklingkoeffizienten $\alpha \in \{0.9, 0.99, 0.999, 0.9999\}$ des EMA und die Startepoche `start` $\in \{0, 5, 10\}$ in `train_fashion` bei konstantem $\eta = 0.3$. Setzen Sie die Verläufe der Genauigkeit in Beziehung zur effektiven Fensterlänge $1/(1-\alpha)$, ausgedrückt in Epochen.
   :exlabel:`ex_practice-averaging-window`

<!--en:c5909aa5-->
1. [code] **Ablationsstudie zur Beispielkonfiguration.** ● Verwenden Sie das Modell mit $4$ Blöcken und Breite $256$ (`TinyLM` aus `train`). Testen Sie mit je zwei Zufallsstartwerten (a) einen späten linearen Abfall gemäß WSD im Vergleich zum Kosinus-Abfall, (b) den Verzicht auf die Aufwärmphase oder auf das Gradienten-Clipping, (c) eine maximale Lernrate von $10^{-3}$ oder $10^{-2}$ und (d) die Auswertung des EMA der Gewichte für die Baseline mit konstanter Lernrate bei $\alpha \in \{0.995, 0.999\}$.
   :exlabel:`ex_practice-worked-configuration`

<!--en:aafc6861-->
[Diskussion](https://d2l.discourse.group/t/36767)

<!--en:3d62da30-->
Diagnose und Konfiguration der Optimierung in der Praxis<br>**Gradienten-Clipping anhand der globalen Norm · Schutz oder Bremse · Wiederaufnahme nach Spitzen bei AdamW · EMA der Gewichte · vollständige Trainingskonfiguration**

<!--en:3fe1acab-->
::: {.slide title="Drei Maßnahmen für zuverlässiges Pre-Training im produktiven Einsatz"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:81ff8147-->
- **1. Schutz vor vorübergehenden Instabilitäten:** Clipping anhand der globalen Norm ($\theta = 1.0$) verhindert bei SGD eine anfängliche Divergenz in Bereichen hoher Krümmung und schützt bei AdamW das zweite Moment $\mathbf{v}_t$ vor Spitzen. Hinzu kommen $\beta_2 = 0.95$ und die Rückkehr zu einem früheren Checkpoint.

<!--en:ce8b0df6-->
- **2. Glättung der Iterierten (EMA der Gewichte):** Die exponentielle Mittelung der aktuellen Gewichte gleicht späte Schwankungen in der Rauschkugel aus, ohne die Lernrate gemäß $\eta_t \to 0$ abzusenken.

<!--en:01eae6e3-->
- **3. Kalibrierung mit kurzen Probeläufen:** Maximale Lernrate $\eta_0$ und $b_{\text{noise}}$ vor dem vollständigen Lauf anhand von $10\%$ der Schritte ($300$ von $3{,}000$) kalibrieren

<!--en:c92723d8-->
::: {.slide title="Gradienten-Clipping anhand der globalen Norm erhält die Richtung"}

<!--en:8f9cfca1-->
[Funktionsweise des Clippings]{.kicker}

<!--en:096e6590-->
Gradienten-Clipping anhand der globalen Norm fasst die Gradienten aller Parameter zu einem einzigen Vektor $\mathbf{g} \in \mathbb{R}^d$ zusammen. Es skaliert ihn, sobald seine euklidische Norm $\|\mathbf{g}\|_2$ einen Schwellenwert $\theta > 0$ überschreitet:

<!--en:48e67730-->
- **Die Größenverhältnisse zwischen den Schichten bleiben erhalten:** Elementweises Clipping verzerrt die Richtung in jeder Schicht hin zu einem Vorzeichenvektor. Clipping anhand der globalen Norm multipliziert dagegen den Gradienten jedes Parameters mit demselben Skalar $\theta / \|\mathbf{g}\|_2 < 1$.

<!--en:f618c7fc-->
- **Die SGD-Schrittweite ist begrenzt:** Beim gewöhnlichen SGD hat jedes Parameterupdate höchstens die euklidische Länge $\eta\theta$.

<!--en:f72a19f7-->
::: {.slide title="Zwei Hypothesen zur Divergenz von SGD bei η = 1.0 im Test"}

<!--en:4bc3ab9a-->
[Experiment: Schutz oder Bremse?]{.kicker}

<!--en:be0ef14f-->
- Warum gelang mit SGD und Momentum ($\beta = 0.9$) das Training von `TinyLM` bei $\eta = 0.3$, während es innerhalb von $20$ Schritten bei $\eta = 1.0$ divergierte?

<!--en:87fb9e83-->
- **Hypothese A: vorübergehend hohe Schärfe zu Beginn.** $\eta = 1.0$ ist nur in den ersten Schritten zu groß. Clipping bei $\theta = 1.0$ greift in den **ersten paar Dutzend Schritten**, danach nicht mehr (**Schutz**).

<!--en:62dc0546-->
- **Hypothese B: überall zu hohe Lernrate.** $\eta$ überschreitet überall $2/L$. Clipping greift bei **fast jedem Schritt** ($>50\text{--}100\%$, **Dauerbremse**).

<!--en:ec703f37-->
::: {.slide title="Clipping greift bei <2% der Schritte und rettet den Lauf mit η = 1.0"}

<!--en:10b5deac-->
[Empirische Diagnose des Clippings bei TinyLM]{.kicker}

<!--en:3a478c34-->
- Läufe über $2{,}000$ Schritte mit $(\eta, \theta) \in \{(1.0, \infty), (1.0, 1.0), (0.3, \infty), (3.0, 1.0)\}$ und zwei Zufallsstartwerten bestätigen **Hypothese A** für $\eta = 1.0$:
  - **Bei $\eta = 1.0, \theta = \infty$:** Divergenz gegen `inf` innerhalb von $20$ Schritten.
  - **Bei $\eta = 1.0, \theta = 1.0$ (Schutz):** Clipping greift bei nur **$14\text{--}17$ Schritten (alle vor Schritt $30$)**. In den verbleibenden $1{,}970$ Schritten greift es nie wieder (Median: $\|\mathbf{g}_t\|_2 \approx 0.25$). Der Verlust sinkt auf **$1.61\text{--}1.63$ Nat** und liegt damit unter dem Wert bei $\eta = 0.3$ ohne Clipping.
  - **Bei $\eta = 3.0, \theta = 1.0$ (Bremse):** Clipping greift bei **jedem einzelnen Schritt**, bis ein Überlauf auftritt oder der Verlust oberhalb von $100$ Nat stagniert.

<!--en:302a8528-->
::: {.slide title="Schutz und Bremse im Trainingsprotokoll unterscheiden"}

<!--en:9d552a1e-->
[Wahl der Clipping-Schwelle θ]{.kicker}

<!--en:9e916c15-->
- **Wirksamer Schutz ($\eta = 1.0, \theta = 1.0$):**
  - Greift bei **$<2\%\text{--}5\%$ der Schritte** (Übergangsphase zu Beginn oder seltene Spitzen). Median im stationären Zustand: $\|\mathbf{g}_t\|_2 \approx 0.25 \approx \theta/4$.

<!--en:65914da3-->
- **Dauerbremse ($\eta = 3.0, \theta = 1.0$):**
  - Greift bei **$>50\%\text{--}100\%$ der Schritte**. Damit führt der Optimierer einen normierten Abstieg mit fester Schrittweite aus.

<!--en:bcd57174-->
- **Faustregel:** Greift das Clipping bei $>20\%\text{--}50\%$ der Schritte, **zuerst die Lernrate $\eta$ senken**, dann $\theta$ anpassen.

<!--en:ea97656f-->
::: {.slide title="Warum Clipping bei AdamW das zweite Moment v_t schützt"}

<!--en:6f43d5c0-->
[Auswirkungen von Gradientenspitzen bei AdamW]{.kicker}

<!--en:75cdb908-->
- **Der unmittelbare Schritt ist bereits begrenzt:** Bei AdamW geht eine Spitze $g_t \gg \sqrt{v_{t-1}}$ sowohl in $\mathbf{m}_t$ als auch in $\sqrt{\mathbf{v}_t}$ ein. Damit gilt $|\Delta x_t| \approx \eta\frac{1-\beta_1}{\sqrt{1-\beta_2}} \le 3.16\eta$.

<!--en:8b30696d-->
- **Clipping schützt den Zustand $\mathbf{v}_t$ im Nenner:**
  - Eine unbegrenzte Spitze der Größe $100\times$ Normalwert trägt $(1-\beta_2)10^4$ zu $v_t$ bei. Für $\beta_2 = 0.999$ gilt für $v_t$ ein Anstieg auf $11\times$ Ausgangswert. Die effektiven Lernraten sinken auf $30\%$ des üblichen Werts, und zwar für **$2{,}300$ Schritte**.
  - Bei einer Schwelle von $\theta = 4\times$ Median der Norm bleibt der zusätzliche Beitrag unter $2\%$. Mit **$\beta_2 = 0.95$** erholt sich der Optimierer $20\times$ so schnell ($\sim 120$ Schritte).

<!--en:a3db385c-->
::: {.slide title="Wiederaufnahme nach Spitzen und z-Verlust der Softmax-Funktion"}

<!--en:661cfd78-->
[Umgang mit Instabilitäten im laufenden Training]{.kicker}

<!--en:d77d8557-->
- **Ursache von Spitzen im laufenden Training (PaLM, Chowdhery et al., 2022):** Ein Parameterzustand am Rand der Stabilität trifft auf einen bestimmten ungewöhnlichen Datenbatch.

<!--en:837bfb9f-->
- **Vorgehen zur Wiederaufnahme:**
  1. Zu einem Checkpoint $\sim 100$ Schritte vor der Spitze **zurückkehren** (Gewichte, $\mathbf{m}_t, \mathbf{v}_t$ und Schritt $t$ wiederherstellen).
  2. Die $200\text{--}500$ Minibatches um die Spitze herum **überspringen**.

<!--en:d11d8b7f-->
- **$z$-Verlust der Softmax-Funktion ($10^{-4}\log^2 Z$):** Bewirkt $\log Z \to 0$ durch den rückstellenden Gradienten $2{\times}10^{-4}(\log Z)\mathbf{p}$, ohne $\mathbf{p}$ zu verändern.

<!--en:1fbb1496-->
::: {.slide title="Mittelung der Gewichte (EMA): Glättung der Schwankungen in der Rauschkugel"}

<!--en:adb5b7fb-->
[Mittelung der Iterierten nach Polyak und Ruppert]{.kicker}

<!--en:676ebd52-->
- Bei jeder von null verschiedenen Lernrate $\eta$ entsprechen die aktuellen Gewichte $\mathbf{x}_t$ einer einzelnen Zufallsziehung aus der stationären Rauschkugel (§{sec_sgd}).

<!--en:044b8a13-->
- Der **exponentiellen gleitenden Mittelwert (EMA)** $\bar{\mathbf{x}}_t$ der aktuellen Parameter mit dem Abklingkoeffizienten $\alpha \in [0.99, 0.9999]$ berechnet sich wie folgt (Polyak und Juditsky, 1992):
  $$\bar{\mathbf{x}}_t = \alpha\,\bar{\mathbf{x}}_{t-1} + (1 - \alpha)\,\mathbf{x}_t$$
  Der EMA mittelt über die jeweils letzten **$\tau_{\text{EMA}} = \frac{1}{1 - \alpha}$ Schritte**, ohne die Trajektorie des Trainings zu verändern.

<!--en:405079fa-->
- **Wahl von $\alpha$ und Startschritt:** $\tau_{\text{EMA}}$ muss größer sein als die Autokorrelationszeit $\frac{1}{\eta\lambda_{\text{curv}}}$ in der Rauschkugel (z. B. $\alpha = 0.999 \implies 1{,}000$ Schritte). Die Mittelung sollte erst nach dem anfänglichen raschen Abstieg beginnen, damit frühe Gewichte mit hohem Verlust den Mittelwert nicht hinterherhinken lassen.

<!--en:6c9f392d-->
::: {.slide title="EMA und aktuelle Gewichte bei konstanter Lernrate auf FashionCNN"}

<!--en:5d926074-->
[Schwankungen zwischen Epochen ausgleichen]{.kicker}

<!--en:683d83ae-->
- Training von `FashionCNN` über $15$ Epochen ($235$ Schritte/Epoche $= 3{,}525$ Schritte) mit gewöhnlichem SGD bei konstanter Lernrate $\eta = 0.3$:
  - **Aktuelle Gewichte (blau):** Sie schwanken in der Rauschkugel. In den letzten fünf Epochen schwankt die **Testgenauigkeit um mehrere Prozentpunkte** unterhalb von $\approx 91\%$. In einzelnen Epochen fällt sie noch tiefer.
  - **EMA der Gewichte ($\alpha = 0.999$, ab Epoche 5, orange):** Die Genauigkeit steigt am oberen Rand des Bereichs gleichmäßig auf **etwa $91\%$ Testgenauigkeit** und erreicht damit den Wert des vollständigen Laufs mit Kosinus-Abfall (grün).

<!--en:542af48b-->
::: {.slide title="Der EMA der Gewichte erreicht die Genauigkeit des Kosinus-Abfalls bei konstantem η"}

<!--en:dea8463b-->
[Quantitativer Vergleich mit zwei Zufallsstartwerten]{.kicker}

<!--en:66fec9d0-->
- SGD bei konstanter Lernrate ($\eta = 0.3$) und mit Kosinus-Abfall ($\eta: 0.3 \to 0$) im Vergleich, jeweils mit zwei Zufallsstartwerten auf `FashionCNN`:
  - **Bei konstanter Lernrate:** Der EMA verbessert die abschließende Testgenauigkeit gegenüber dem verrauschten Checkpoint der aktuellen Gewichte um etwa $0.5\text{--}2$ Prozentpunkte. Damit erreicht er die Genauigkeit eines vollständigen Kosinus-Abfalls von **$\approx 91\%$**.
  - **Nach Kosinus-Abfall auf null:** Die aktuellen Gewichte liegen bereits im Zentrum der Rauschkugel ($\eta_T = 0$). Ein zusätzlicher EMA bringt dann keinen Gewinn oder liegt um wenige Zehntelprozentpunkte zurück.

<!--en:7d3a7ff8-->
- **Wann lohnt sich der EMA der Gewichte?** Bei konstanter Lernrate oder auf dem WSD-Plateau liefert er in jedem Schritt ein nutzbares Modell, ohne auf einen Trainingszweig mit Abkühlphase warten zu müssen (bei Sprachmodellen erzielt er einen Teil des Gewinns durch die Abkühlphase, Hägele et al., 2024).

<!--en:a081d708-->
::: {.slide title="Gemeinsamkeiten veröffentlichter Konfigurationen für das Pre-Training führender Modelle"}

<!--en:e4fdd2d1-->
[Referenzkonfigurationen aus der Industrie]{.kicker}

<!--en:0b1217c4-->
- **DeepSeek-V3 671B, OLMo 2 7B und Kimi K2 1T stimmen überall dort überein, wo sie einen Wert angeben (für Llama 3 405B fehlen Angaben zu $\beta$, $\theta$ und $\lambda$):**
  - **Momente:** $(\beta_1, \beta_2) = (0.9, 0.95)$ für eine schnelle Erholung nach Spitzen.
  - **Entkoppelter Gewichtszerfall:** $\lambda = 0.1$ nur für 2D-Matrizen (ausgenommen sind Bias-Terme und 1D-Normierungsparameter, bei OLMo 2 auch Einbettungen).
  - **Clipping anhand der globalen Norm:** $\theta = 1.0$.
  - **Lernratenplan und Batchgröße:** $500\text{--}8{,}000$ Schritte Aufwärmphase, dann Kosinus- oder WSD-Abkühlphase auf $0.01\text{--}0.1\eta_0$. Die Batchgröße steigt schrittweise auf mehrere Millionen Token.

<!--en:50f2ec01-->
::: {.slide title="Diagnose anhand typischer Fehlerbilder im Trainingsprotokoll"}

<!--en:4046c42d-->
[Von Messwerten zu Maßnahmen]{.kicker}

<!--en:40ae6857-->
- **Immer 4 Kurven protokollieren:** Verlust, Gradientennorm $\|\mathbf{g}_t\|_2$ vor dem Clipping, Clipping-Indikator ($\|\mathbf{g}_t\|_2 > \theta$) und relative Schrittweite $\|\Delta\mathbf{W}\|_{\text{RMS}}/\|\mathbf{W}\|_{\text{RMS}}$:
  - **Divergenz in Schritt 1–20 oder Sprung von $>10\%$ in Schritt 1:** Lineare Aufwärmphase ($1\text{--}5\%$) und Clipping ($\theta = 1.0$) ergänzen.
  - **Hoher Verlust trotz Aufwärmphase oder Clipping bei $>20\%$ der Schritte:** Maximale Lernrate $\eta_0$ zu hoch $\to$ **$\eta_0$ um den Faktor $2\text{--}3\times$ senken**.
  - **Plateau mit starken Schwankungen:** Lernrate gemäß $\eta_t \to 0$ senken, EMA der Gewichte verwenden oder $b$ erhöhen.
  - **Konstante Steigung bei Schritt $T$:** Zu kleines Budget $B_T \to$ $\eta_0$ erhöhen oder $T$ verlängern.

<!--en:6ccd0c1c-->
::: {.slide title="Fallstudie: Konfiguration eines TinyLM mit 3.3 Mio. Parametern"}

<!--en:a43d38d8-->
[Ergebnisse aus allen 10 Abschnitten]{.kicker}

<!--en:abde942d-->
Konfiguration für einen Trainingslauf über $3{,}000$ Schritte für ein neues Modell mit $4$ Blöcken, Breite $256$ und $4$ Köpfen: `TinyLM` ($3.3\text{M}$ Parameter, $7\times$ so groß wie unser Modell mit $2$ Blöcken). So entfällt die Wiederholung des vollständigen Laufs über $3{,}000$ Schritte für ein ganzes Parameterraster:

<!--en:b9b8eaca-->
1. **Batchgröße ($b = 256$ Sequenzen):** Hier erreicht der GPU-Durchsatz sein Maximum. Die Batchgröße ist vergleichbar mit $b_{\text{noise}} \sim 100$ nach der Aufwärmphase.

<!--en:3d6fc6b5-->
2. **Optimierer und Schutzmaßnahmen:** AdamW mit $(\beta_1, \beta_2) = (0.9, 0.95)$, entkoppelter Gewichtszerfall $\lambda = 0.1$ nur für 2D-Matrizen, Clipping anhand der globalen Norm bei $\theta = 1.0$ und EMA der Gewichte ($\alpha = 0.995$).

<!--en:a90f6af5-->
3. **Kalibrierung der maximalen Lernrate in Probeläufen ($10\%$ des Budgets):** Jeweils $\eta \in \{10^{-3}, 3{\times}10^{-3}, 10^{-2}\}$ über $300$ Schritte testen.

<!--en:39bc6a07-->
4. **Lernratenplan:** Lineare Aufwärmphase über $60$ Schritte ($2\%$), dann Kosinus-Abfall auf null.

<!--en:2c5be643-->
::: {.slide title="Kalibrierung der maximalen Lernrate und Schätzung der Rauschskala in 300 Schritten"}

<!--en:d3df39b0-->
[Kalibrierung über 10% der Trainingsdauer]{.kicker}

<!--en:e567b1b3-->
- Vor dem Lauf über $3{,}000$ Schritte stehen drei Probeläufe mit je $300$ Schritten (insgesamt $900$ Schritte, $30\%$ eines vollständigen Laufs) für die Lernraten $\eta \in \{10^{-3}, 3{\times}10^{-3}, 10^{-2}\}$:
  - **$\eta = 10^{-3}$:** Langsam ($\approx 1.88$ Nat, Clipping bei $2\text{--}3\%$ der Schritte).
  - **$\eta = 3 \times 10^{-3}$:** **Optimal ($\approx 1.75$ Nat, Clipping bei $3\%$ der Schritte, also wirksamer Schutz)**.
  - **$\eta = 10^{-2}$:** Der Verlust stagniert bei $\approx 2.5$ Nat (Clipping bei $6\%$ der Schritte).

<!--en:f770ecc5-->
- **Kontrolle der Rauschskala:** $b_{\text{noise}}$ wächst von $\approx 2$ Sequenzen bei Schritt 0 auf $\sim 150$ Sequenzen bei Schritt 300. Das bestätigt, dass $b = 256$ nahe dem Knick der Kurve bei der kritischen Batchgröße liegt.

<!--en:f849cf79-->
::: {.slide title="Überprüfung über alle 3000 Schritte mit verschiedenen Zufallsstartwerten"}

<!--en:08781ffc-->
[Beispielkonfiguration und Baseline mit konstanter Lernrate]{.kicker}

<!--en:f17a0e1d-->
- Test des vollständigen Lernratenplans mit Aufwärmphase und Kosinus-Abfall über $3{,}000$ Schritte bei der kalibrierten maximalen Lernrate $\eta_0 = 3 \times 10^{-3}$ mit zwei unabhängigen Zufallsstartwerten:
  - **Beispielkonfiguration:** Der Validierungsverlust beträgt **$\approx 1.25$ Nat**. Gegenüber der Baseline mit konstanter Lernrate ($\approx 1.30$ Nat) sinkt er um **$0.05\text{--}0.06$ Nat**. Der Unterschied beträgt ein Mehrfaches der Spannweite von $<0.01$ Nat zwischen den Zufallsstartwerten.
  - **Überprüfung der Schutzwirkung:** Clipping greift nur bei **$0.5\text{--}0.6\%$ der Schritte** ($\sim 15\text{--}18$ von $3{,}000$ Schritten).
  - **EMA nach dem Abfall:** Der Verlust stimmt bis auf $0.001$ Nat mit dem der aktuellen Gewichte überein, weil der Kosinus-Abfall die Lernrate bereits gemäß $\eta_T \to 0$ gesenkt hat.

<!--en:17efe40a-->
::: {.slide title="Verständnisfragen: Clipping, Wiederaufnahme nach Spitzen und EMA der Gewichte"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:a2e0fcc8-->
Für jedes Trainingsszenario eine Diagnose stellen und eine Lösung angeben:

<!--en:16a1e25d-->
1. **Diagnose eines stagnierenden AdamW-Laufs:** Beim Modell mit $3.3\text{M}$ Parametern, `TinyLM`, stagniert der Verlust bei der Lernrate $\eta = 10^{-2}$ und der Clipping-Schwelle $\theta = 1.0$ nahe $2.5$ Nat. Das Clipping greift bei $6\%$ der Schritte, doppelt so oft wie bei den kleineren Lernraten. Ein Kollege schlägt vor, die Clipping-Schwelle auf $\theta = 0.2$ zu senken, um die Schritte zu verkürzen. Kommt das Training dadurch wieder voran?

<!--en:83fc1f30-->
2. **Erholungszeit des zweiten Moments bei AdamW:** Angenommen, eine unbegrenzte Gradientenspitze der Größe $100\times$ Normalwert trifft auf AdamW und multipliziert $v_t$ mit $1 + (1-\beta_2)10^4$. Wie viele Schritte braucht es, bis der zusätzliche Beitrag zu $v_t$ auf den Anteil $e^{-3} \approx 5\%$ seines Anfangswerts abgeklungen ist, einmal für $\beta_2 = 0.999$ und einmal für $\beta_2 = 0.95$?

<!--en:19c807f0-->
3. **Wann bringt der EMA der Gewichte keinen Gewinn?** Warum ist in der abschließenden Tabelle zum Lauf über $3{,}000$ Schritte der Validierungsverlust für die aktuellen Gewichte der Beispielkonfiguration und ihren EMA bis auf $0.001$ Nat gleich?

<!--en:71dd5bec-->
::: {.slide title="Verständnisfragen: Clipping, Wiederaufnahme nach Spitzen und EMA der Gewichte (Antworten)"}

<!--en:929b7671-2-->
[Wissen überprüfen]{.kicker}

<!--en:557b98b6-->
**Antworten:**

<!--en:54002266-->
1. **Nein.** AdamW dividiert $\hat{\mathbf{m}}_t$ durch $\sqrt{\hat{\mathbf{v}}_t}$. Skaliert man $\mathbf{g}_t$ durch eine niedrigere globale Clipping-Schwelle $\theta = 0.2$, ändern sich daher Zähler und Nenner im selben Verhältnis. Die Schrittweite pro Koordinate im stationären Zustand bleibt bei $\approx \eta$. Abhilfe: **die Lernrate $\eta$ senken** (z. B. auf $3 \times 10^{-3}$).

<!--en:983b6fdf-->
2. Ein Abklingen auf den Anteil $e^{-3}$ dauert $3\,\tau_2 = \frac{3}{1 - \beta_2}$ Schritte: **$3{,}000$ Schritte bei $\beta_2 = 0.999$** gegenüber nur **$60$ Schritten bei $\beta_2 = 0.95$** (Erholung um den Faktor $50\times$ schneller).

<!--en:4be3e303-->
3. Der konfigurierte Lauf endete mit einem **Kosinus-Abfall auf $\eta_T = 0$**. Die Rauschkugel war deshalb in den letzten Schritten bereits auf null geschrumpft. Der EMA der Gewichte lohnt sich besonders bei der Auswertung von **Checkpoints ohne Abkühlphase aus Läufen mit konstanter Lernrate oder vom WSD-Plateau**.

<!--en:61e3245e-->
::: {.slide title="Zusammenfassung: Diagnose und Konfiguration der Optimierung in der Praxis"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:7918086e-->
| Maßnahme | Mathematische Regel oder Wirkungsweise | Zielwerte im Trainingsprotokoll | Nutzen im produktiven Einsatz |

<!--en:b68fc8b2-->
| **Clipping anhand der globalen Norm** | $\mathbf{g} \leftarrow \min\!\left(1, \frac{\theta}{\|\mathbf{g}\|_2}\right)\mathbf{g}$ ($\theta = 1.0$) | Greift bei $<2\%\text{--}5\%$ der Schritte (Schutz). Bei $>20\%$ die Lernrate $\eta$ senken | Begrenzt frühe SGD-Schritte und schützt $\mathbf{v}_t$ von AdamW vor Spitzen |

<!--en:671d0021-->
| **$\beta_2 = 0.95$ und Rückkehr zum Checkpoint** | $\tau_2 = \frac{1}{1-\beta_2} = 20$ Schritte (gegenüber $1{,}000$ bei $0.999$) | Bei einer Spitze um $\sim 100$ Schritte zurückgehen und $200\text{--}500$ Batches überspringen | Verhindert, dass eine einzelne Spitze das Lernen über $\sqrt{\hat{\mathbf{v}}_t}$ für $2{,}300$ Schritte lähmt |

<!--en:e3fe1f97-->
| **Mittelung der Gewichte (EMA)** | $\bar{\mathbf{x}}_t = \alpha\bar{\mathbf{x}}_{t-1} + (1-\alpha)\mathbf{x}_t$ ($\alpha \in [0.995, 0.9999]$) | Fensterlänge $\frac{1}{1-\alpha} > \frac{1}{\eta\lambda_{\text{curv}}}$. Nach dem anfänglichen Abstieg beginnen | Erreicht bei konstanter Lernrate die Genauigkeit eines Laufs mit Kosinus-Abfall |

<!--en:298f512f-->
| **Kalibrierung in Probeläufen** | $\eta_0$ über $10\%$ der Trainingsdauer variieren ($300$ von $3{,}000$ Schritten) | $\eta_0$ um den Faktor $3\times$ unter der Stagnationsschwelle wählen und $b_{\text{noise}}$ prüfen | Kalibriert die maximale Lernrate mit $30\%$ des Aufwands eines vollständigen Trainingslaufs |

<!--en:c2525580-->
| **Bewährte Standardkonfiguration** | AdamW $(0.9, 0.95)$, $\lambda = 0.1$ nur für Matrizen, Clipping bei $1.0$ | $1\text{--}5\%$ lineare Aufwärmphase, dann Kosinus- oder WSD-Abfall auf $\le 0.1\eta_0$ | Zuverlässige Baseline für Transformer und moderne tiefe Netze |
