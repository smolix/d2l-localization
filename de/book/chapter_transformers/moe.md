<!--en:34366326-->
# Mixture of Experts (MoE)

<!--en:24e56bf8-->
In einem dichten Transformer (:numref:`sec_scaling-laws`) nutzt jedes Token jeden Parameter. Verdoppeln wir die Modellkapazität $N \approx 12 L d^2$, verdoppeln sich sowohl die Trainingskosten ($6 N$ FLOPs pro Token) als auch der Rechenaufwand bei der Inferenz ($2 N$ FLOPs pro Token). Können wir die Anzahl der gespeicherten Parameter $N_{\textrm{total}}$ eines Modells erhöhen, ohne zugleich die Anzahl der pro Token genutzten aktiven Parameter (engl. active parameters) $N_{\textrm{active}}$ zu erhöhen?

<!--en:bd9da652-->
Aus :numref:`sec_transformer-block` wissen wir, dass das Feedforward-Netz zwei Drittel der Parameter eines dichten Blocks enthält ($8 d^2$ von $12 d^2$). Es verarbeitet **positionsweise** jeden Tokenvektor $\mathbf{x} \in \mathbb{R}^d$ unabhängig von den anderen. Eine **dünn besetzte Mixture of Experts (MoE)** nutzt diese Unabhängigkeit :cite:`Jacobs.Jordan.Nowlan.ea.1991,Shazeer.Mirhoseini.Maziarz.ea.2017,Lepikhin.Lee.Xu.ea.2021,fedus2022switch` (:numref:`fig_moe-layer`). In dieser Schicht treten an die Stelle des einzelnen FFN eines Blocks $E$ parallele FFN als **Experten** $\{\textrm{FFN}_1, \dots, \textrm{FFN}_E\}$. Ein kleiner linearer **Router** $\mathbf{W}_r \in \mathbb{R}^{d \times E}$ leitet jedes Token $\mathbf{x}$ nur an seine Top-$k$-Experten weiter ($1 \le k \ll E$).

<!--en:248a3558-->
## Bedingte Berechnung mit gespeicherten und aktiven Parametern

<!--en:aa179bba-->
Betrachten wir einen MoE-Transformer mit $L$ Schichten. Jeder Block hat $P_{\textrm{attn}}$ Attention-Parameter, $E$ vom Router auswählbare SwiGLU-Experten mit innerer Breite $w$ ($P_{\textrm{exp}} = 3 d w$ Parameter pro Experte) und $E_{\textrm{shared}}$ stets aktive gemeinsame Experten (engl. shared experts) :cite:`Dai.Deng.Zhao.ea.2024`. Alle $E + E_{\textrm{shared}}$ Experten müssen im Speicher liegen ($N_{\textrm{total}}$). Pro Token arbeiten jedoch nur $k + E_{\textrm{shared}}$ Experten ($N_{\textrm{active}}$).

<!--en:85cc1f35-->
Die Parameterzahlen $N_{\textrm{total}}$ und $N_{\textrm{active}}$ bestimmen den Bedarf an unterschiedlichen Systemressourcen:

<!--en:da877742-->
- Die **FLOPs pro Token (Training und Prefill)** wachsen proportional zu **$N_{\textrm{active}}$** ($C_{\textrm{train}} \approx 6 N_{\textrm{active}} D$).

<!--en:ba734586-->
- Die erforderliche **VRAM-Kapazität** wächst proportional zu **$N_{\textrm{total}}$**, weil alle $E$ Experten im Gerätespeicher liegen müssen.

<!--en:df01466a-->
- **Bandbreite bei der Dekodierung im Batch:** Bei einer Batchgröße von $B = 1$ liest ein Token nur die Gewichte von $k$ Experten aus dem HBM ($N_{\textrm{active}}$ Byte an Gewichten). In einem Batch mit $B$ Token aktivieren verschiedene Token jedoch unterschiedliche Experten. Sobald $B \gg E / k$ gilt, erhält fast jeder der $E$ Experten mindestens ein Token des Batchs. Jeder Dekodierschritt liest dann fast **$N_{\textrm{total}}$** Byte an Gewichten aus dem HBM und benötigt $2 B N_{\textrm{active}}$ FLOPs.

<!--en:eb75457c-->
## Eine Mixture-of-Experts-Schicht

<!--en:3e736b30-->
### Top-$k$-Routing

<!--en:6534e9aa-->
Für jedes Token $\mathbf{x} \in \mathbb{R}^d$ berechnet der Router die Wahrscheinlichkeiten $\mathbf{p}(\mathbf{x}) = \textrm{softmax}(\mathbf{x}\mathbf{W}_r) \in \Delta^{E-1}$. Er wählt die Menge $T(\mathbf{x}) \subset \{1, \dots, E\}$ der $k$ Experten mit den höchsten Scores und übernimmt deren ursprüngliche Wahrscheinlichkeiten als Gewichte, also $g_e(\mathbf{x}) = p_e(\mathbf{x})$. Bei einer erneuten Normierung über $T(\mathbf{x})$ erhielte ein Top-1-Router keinen Gradienten (siehe Übung :exref:`ex_moe-router-gradient`). Aus den gewichteten Ausgaben bildet der Router die Summe:

<!--en:2c19b664-->
![Eine Mixture-of-Experts-Schicht mit acht Experten, von denen pro Token zwei aktiv sind. Der Router bewertet jeden Experten. Die beiden Experten mit den höchsten Wahrscheinlichkeiten verarbeiten das Token. Ihre Ausgaben werden mit den jeweiligen Wahrscheinlichkeiten gewichtet und addiert. Die anderen sechs Experten liegen im Speicher, führen für dieses Token aber keine Berechnung aus.](../img/mdl-transformers-moe.svg)

<!--en:5ed65233-->
### Verteilung der Token auf Expertenpuffer mit fester Kapazität

<!--en:307a0d85-->
Beim kompilierten oder verteilten Training reserviert jeder Experte $e \in \{1, \dots, E\}$ vorab einen Puffer mit $S = \lceil \gamma \cdot n k / E \rceil$ Plätzen für einen Batch aus $n$ Token. Dabei ist $n k / E$ die Last pro Experte bei gleichmäßiger Verteilung und $\gamma \ge 1$ der **Kapazitätsfaktor** (engl. capacity factor). Für $\gamma = E$ gilt $S = n k$, sodass keine Zuweisung entfällt. Entfallen mehr als $S$ Zuweisungen von Token auf den Experten $e$, werden die überzähligen Zuweisungen **verworfen**. Ihr Beitrag zu :eqref:`eq_moe-layer` ist null. Die betreffenden Token gelangen über die residuale Skip-Verbindung durch den umgebenden Block. :numref:`fig_mdl-transformers-moe-capacity` zeigt diese Verteilung am Beispiel von sechs Token.

<!--en:2f9ece0f-->
![Verteilung von sechs Token auf vier Experten mit Top-2-Routing. Der Kapazitätsfaktor $\gamma = 1.25$ ergibt $S = 4$ Plätze pro Experte. Die Zuweisungen reihen sich bei jedem Experten in der Reihenfolge der Token ein. Die fünfte und sechste Zuweisung an Experte 1 überschreiten seine $S$ Plätze und entfallen. Token 5 und 6 nutzen deshalb nur ihren jeweils zweiten Experten. Entfallen alle Zuweisungen eines Tokens, bleibt nur sein Residuum.](../img/mdl-transformers-moe-capacity.svg)

<!--en:95562f67-->
`MoELayer` implementiert den Router, die Verteilung auf Puffer mit je $S$ Plätzen und die Regel zum Verwerfen überzähliger Zuweisungen:

<!--en:1c2d7f25-->
Zwei kleine Hilfsfunktionen, `moe_layers(model)` und `active_params(model)`, sammeln alle `MoELayer`-Untermodule eines Netzes und zählen die Parameter, die am Vorwärtsdurchlauf eines einzelnen Tokens beteiligt sind (der Anteil $k / E$ der Expertenparameter zuzüglich aller gemeinsamen Parameter):

<!--en:d1b18641-->
Da `d2l.TransformerBlock` (:numref:`sec_transformer-block`) über `ffn_factory` eine Funktion entgegennimmt, genügt ein einziger Lambda-Ausdruck, um das dichte FFN durch eine Top-2-Schicht vom Typ `MoELayer` mit 8 Experten zu ersetzen:

<!--en:25f1d284-->
## Routing-Kollaps und Lastausgleich

<!--en:b9e50d4b-->
### Ungleiche Last und positive Rückkopplung

<!--en:93613a2b-->
Schon bei der Initialisierung leiten die zufälligen Router-Gewichte mehr Token an manche Experten weiter als an andere. Beim Training ohne Lastausgleich (engl. load balancing) verstärkt eine **positive Rückkopplung** die Asymmetrie bis zum **Routing-Kollaps** (engl. routing collapse) :cite:`Shazeer.Mirhoseini.Maziarz.ea.2017`. Die Experten mit mehr Token in Schritt 1 erhalten größere Gradientenupdates. Dadurch senken sie in Schritt 2 den Verlust bei der Sprachmodellierung schneller. In Schritt 3 erhalten sie über $\nabla_{\mathbf{W}_r}\mathcal{L}$ höhere Router-Logits. Ohne Gegenmaßnahme können am Ende wenige Experten die meisten Top-$k$-Plätze belegen, während die übrigen zu wenige Token erhalten, um etwas zu lernen.

<!--en:d4a80d0c-->
### Zwei Lösungen für den Lastausgleich mit Hilfsverlust oder Bias-Regler

<!--en:7c05ef14-->
1. **Switch-Hilfsverlust zum Lastausgleich** :cite:`fedus2022switch`. Sei $f_e = \frac{1}{n k} \sum_{t=1}^n \mathbb{I}[e \in T(\mathbf{x}_t)]$ der empirische Anteil der Plätze, die der Router dem Experten $e$ in einem Batch mit $n$ Token zuweist. Sei $\bar{p}_e = \frac{1}{n}\sum_{t=1}^n p_e(\mathbf{x}_t)$ die mittlere Wahrscheinlichkeit, die der Router dem Experten $e$ zuweist. Wir addieren $\alpha \mathcal{L}_{\textrm{aux}}$ zum Trainingsverlust. Dabei gilt:

   $$
   \mathcal{L}_{\textrm{aux}} \;=\; E \sum_{e=1}^{E} f_e \, \bar{p}_e,
   $$
   :eqlabel:`eq_moe-aux`

   Dieser Hilfsverlust bestraft eine hohe Wahrscheinlichkeit $\bar{p}_e$ für stark ausgelastete Experten (großes $f_e$). Er erreicht sein Minimum $\mathcal{L}_{\textrm{aux}} = 1$, wenn $f_e = 1/E$ für alle $e$ gilt.

<!--en:847f4a3b-->
2. **Dynamische Regelung des Router-Bias ohne Hilfsverlust** (:citet:`Wang.Gao.Zeng.ea.2024`, eingesetzt in DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024`). Der Gradient $\nabla \mathcal{L}_{\textrm{aux}}$ drängt die Tokenrepräsentationen selbst dann zu gleichmäßigem Routing, wenn Spezialisierung den Verlust bei der Sprachmodellierung senken würde. Dieses Verfahren gleicht die Last daher ohne zusätzlichen Gradienten aus (DeepSeek-V3 behält daneben nur einen sehr kleinen Ausgleichsverlust pro Sequenz bei). Jeder Experte hat dazu einen nicht differenzierbaren Bias $b_e$. Der Router addiert ihn *nur* bei der Top-$k$-Auswahl ($T(\mathbf{x}) = \textrm{argtop}_k(\mathbf{p}(\mathbf{x}) + \mathbf{b})$). Die Gate-Gewichte $g_e(\mathbf{x})$ bleiben also unverändert. Nach jedem Schritt aktualisiert ein Regler den Bias anhand des Vorzeichens der Abweichung mit der Schrittweite $\eta_b$:

   $$
   b_e \;\leftarrow\; b_e \;+\; \eta_b \, \textrm{sign}\!\left(\frac{1}{E} - f_e\right).
   $$
   :eqlabel:`eq_moe-bias`

<!--en:26e114bc-->
### Verfahren zum Lastausgleich im Vergleich

<!--en:3f0fa347-->
Wir ergänzen `balance(rate)` in `MoELayer` und vergleichen fünf Konfigurationen eines bytebasierten `GPT` mit 2 Blöcken ($E = 8, k = 1$). Jeder Experte ist ein FFN voller Breite, sodass pro Token genau 1 Experte aktiv ist und damit genau 1 dichtes FFN zum Einsatz kommt. Über 800 Schritte auf WikiText-103 vergleichen wir (1) die Initialisierung, (2) das Training ohne Lastausgleich, (3) einen **eingefrorenen Router** als Kontrolle, ob der Kollaps eine Rückkopplung zwischen Router und Experten voraussetzt, (4) den Switch-Hilfsverlust ($\alpha = 0.01$) und (5) die Bias-Regelung ohne Hilfsverlust ($\eta_b = 10^{-2}$):

<!--en:3c250a3e-->
Um die Lastverteilung auf dem Validierungsdatensatz zu untersuchen, erfasst `evaluate_moe` die Kreuzentropie auf diesen Daten, die sortierten Tokenanteile der Experten und den Anteil verworfener Zuweisungen. Mit `measure` vergleichen wir den Validierungsverlust beim eingestellten Kapazitätsfaktor mit dem bei einem Puffer, der alle Zuweisungen aufnimmt ($S = n k$, keine verworfenen Zuweisungen):

<!--en:f152b65b-->
Als Nächstes laden wir `WikiText103` und definieren `small_gpt`, ein bytebasiertes `GPT` mit 2 Blöcken und der Breite $d = 128$. Bei `capacity_factor=None` verwendet es ein dichtes FFN, andernfalls eine Top-1-Schicht vom Typ `MoELayer` mit 8 Experten. Außerdem definieren wir `freeze_router` für den Kontrollversuch mit eingefrorenem Router:

<!--en:7a6ad75b-->
Die Funktion `balancing_run` trainiert ein neu initialisiertes `small_gpt` mit dem Kapazitätsfaktor $\gamma = 1.25$ für `num_steps=800` Schritte nach dem gewählten Verfahren zum Lastausgleich und gibt die Zusammenfassung von `measure` zurück:

<!--en:43f73fe0-->
Für jede der fünf Varianten starten wir zwei unabhängige Läufe mit unterschiedlichen Zufallsstartwerten. Die Tabelle zeigt die Validierungsverluste mit und ohne Verwerfen von Zuweisungen bei Pufferüberlauf, die höchste Expertenlast und den Anteil verworfener Zuweisungen:

<!--en:faf6c013-->
Wir mitteln die nach Größe sortierten Tokenanteile der Experten über beide MoE-Schichten und tragen sie zusammen mit der Referenzlinie für die Gleichverteilung bei $1/8 = 12.5\%$ auf. So wird sichtbar, wie beide Verfahren zum Lastausgleich einen Routing-Kollaps verhindern:

<!--en:178bd4bc-->
Ohne Lastausgleich (`no balancing`) erhalten die 3 am stärksten ausgelasteten Experten etwa $70\%$ aller verteilten Token, der am wenigsten ausgelastete dagegen fast keine. Bei einem Kapazitätsfaktor von $\gamma = 1.25$ entfällt ein Viertel bis ein Drittel aller Zuweisungen. Mit eingefrorenem Router (`frozen router`) ist die Last am Ende etwas gleichmäßiger verteilt. Das frisch initialisierte Modell (`initialization`) weist allerdings schon ein ähnlich großes Ungleichgewicht auf wie das ohne Lastausgleich trainierte Modell. Bei dieser Modellgröße geht ein großer Teil des Ungleichgewichts also auf die Eingaben zurück. Mit zwei Läufen pro Tabellenzeile lässt sich dieser Einfluss nicht von der Rückkopplung zwischen lernenden Experten und Router-Updates trennen (siehe Übung :exref:`ex_moe-feedback`). Sowohl der Switch-Hilfsverlust als auch der Bias-Regler ohne Hilfsverlust halten den Anteil jedes Experten nahe am Idealwert $1/E = 12.5\%$. Selbst der höchste Anteil bleibt unter dem Doppelten dieses Werts. Bei $\gamma = 1.25$ entfallen höchstens einige Prozent der Zuweisungen.

<!--en:d9f18bcb-->
## Mixture-of-Experts-Sprachmodelle

<!--en:dc45bb73-->
### MoE-GPT und dichtes GPT bei gleicher Anzahl aktiver Parameter

<!--en:a5bbc824-->
Wir vergleichen unser MoE-GPT mit Lastausgleich, 8 Experten und Top-1-Routing ($N_{\textrm{active}} = 0.43\textrm{M}$, $N_{\textrm{total}} = 2.26\textrm{M}$) mit einem dichten GPT. Beide haben **genau gleich viele aktive Parameter und benötigen gleich viele FLOPs pro Token** ($N_{\textrm{active}} = N_{\textrm{total}} = 0.43\textrm{M}$). Der Vergleich über 6000 Trainingsschritte auf WikiText-103 zeigt, welchen Vorteil die bedingte Aktivierung der Experten bei gleichem Rechenaufwand (IsoFLOP) bietet:

<!--en:7cad27bf-->
Wir trainieren beide Modelle in je zwei unabhängigen Läufen für $6{,}000$ Schritte, beim MoE-Modell mit dem Switch-Hilfsverlust ($\alpha = 0.01$). Alle $250$ Schritte messen wir den Validierungsverlust. So sehen wir, wann das dünn besetzte Modell das dichte Referenzmodell überholt:

<!--en:cfcab959-->
Das MoE-Modell liegt anfangs hinter dem dichten Modell und überholt es innerhalb der ersten 2000 Schritte. Am Ende ist seine Kreuzentropie auf dem Validierungsdatensatz um etwa 0.05 Nat niedriger, bei denselben Kosten in FLOPs pro Token.

<!--en:44ce90f9-->
### Feingranulare und gemeinsame Experten in veröffentlichten Modellen

<!--en:8ddbd8ab-->
Neuere MoE-Spitzenmodelle (:numref:`tab_moe-experts`) verfeinern das grobe Top-2-Routing von Mixtral 8x7B mit seinen 8 Experten auf zweierlei Weise (:numref:`fig_moe-fine-grained`):

<!--en:0bb64234-->
1. **Feingranulare Unterteilung der Experten** (engl. fine-grained expert segmentation). Wir teilen jeden der $E$ Experten voller Breite in $m$ Experten der Breite $w / m$. Der Router weist jedes Token $m k$ der insgesamt $m E$ Experten zu. Die Anzahl der gespeicherten und der aktiven Parameter bleibt jeweils gleich. Die Zahl der möglichen Kombinationen aktiver Experten steigt dagegen zum Beispiel von $\binom{16}{2} = 120$ auf $\binom{64}{8} \approx 4.4 \times 10^9$ bei $m = 4$. Veröffentlichte Modelle gehen weiter. Sie verwenden $128$–$256$ Experten, von denen $4$–$8$ aktiv sind. Dadurch sinkt auch der Anteil aktiver Experten.

<!--en:173fe92a-->
2. **Gemeinsame Experten** :cite:`Dai.Deng.Zhao.ea.2024`. Wir reservieren $E_{\textrm{shared}} \ge 1$ schmale Experten, die ohne Routing *jedes* Token verarbeiten. Sie übernehmen häufig benötigte Transformationen für Grammatik und Formatierung. So können sich die vom Router ausgewählten Experten spezialisieren, ohne dieselben Transformationen mehrfach zu erlernen.

<!--en:af320cbc-->
![Drei Möglichkeiten, dieselben Expertenparameter zu nutzen. Jedes Rechteck steht für einen Experten. Seine Breite ist proportional zu dessen Größe. Hervorgehoben sind die für ein Token ausgewählten Experten. Oben nutzt ein Token 2 von 16 Experten. In der Mitte verteilen sich dieselben Parameter auf 64 Experten mit einem Viertel der Breite, von denen ein Token 8 nutzt. Unten wird einer der schmalen Experten zum gemeinsamen Experten für alle Token. Jedes Token wählt 7 der übrigen 63. Alle drei Varianten speichern gleich viele Parameter und nutzen pro Token gleich viele davon.](../img/mdl-transformers-moe-fine.svg)

<!--en:1192adf1-->
:Konfigurationen der vom Router auswählbaren Experten in veröffentlichten MoE-Sprachmodellen.

<!--en:09647520-->
| Modell | Experten pro Schicht ($E$) | Davon aktiv ($k$) | Expertenbreite / $d$ | Gemeinsamer Experte | Gespeicherte / aktive Parameter |

<!--en:8775ba36-->
| Mixtral 8x7B :cite:`Jiang.Sablayrolles.Roux.ea.2024` | 8 | 2 | 3.5 | keiner | 46.7B / 12.9B = 3.6 |

<!--en:45fd708e-->
| DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` | 256 | 8 | 0.29 | 1 | 671B / 37B = 18 |

<!--en:a43da2b6-->
| Qwen3-235B :cite:`Yang.Li.Yang.ea.2025` | 128 | 8 | 0.38 | keiner | 235B / 22B = 11 |

<!--en:477bf832-->
| gpt-oss-120b :cite:`OpenAI.2025` | 128 | 4 | 1.0 | keiner | 117B / 5.1B = 23 |

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:e431bc08-->
Schichten mit **Mixture of Experts (MoE)** (:eqref:`eq_moe-layer`) ersetzen ein dichtes FFN durch $E$ Experten-FFNs. Pro Token aktiviert der Router nur die $k \ll E$ Experten mit den höchsten Scores. Dies entkoppelt die **gespeicherten Parameter $N_{\textrm{total}}$** (Speicherkapazität) von den **aktiven Parametern $N_{\textrm{active}}$** (FLOPs pro Token). Ohne Lastausgleich ist die Last sehr ungleich verteilt. Ein großer Teil dieses Ungleichgewichts besteht schon bei der Initialisierung. Positive Rückkopplung zwischen Router und Experten kann es bis zum **Routing-Kollaps** verstärken. Sowohl der **Switch-Hilfsverlust** $\mathcal{L}_{\textrm{aux}} = E \sum_e f_e \bar{p}_e$ (:eqref:`eq_moe-aux`) als auch **dynamische Bias-Terme des Routers ohne Hilfsverlust** $b_e \leftarrow b_e + \eta_b \textrm{sign}(1/E - f_e)$ (:eqref:`eq_moe-bias`) halten die Lastanteile der Experten nahe bei $1/E$. Bei gleicher Anzahl aktiver Parameter und gleicher Zahl an Trainings-FLOPs erreicht ein MoE-Sprachmodell mit Lastausgleich eine niedrigere Kreuzentropie auf dem Validierungsdatensatz als ein dichtes GPT. Neuere MoE-Architekturen mit offen zugänglichen Gewichten (:numref:`tab_moe-experts`) unterteilen die Experten noch **feingranularer** ($E \in [128, 256], k \in [4, 8]$). So ergeben sich mehr Kombinationen für die Spezialisierung, oft ergänzt um einen stets aktiven **gemeinsamen Experten** (:numref:`fig_moe-fine-grained`).

<!--en:2294243a-->
## Übungen

<!--en:db75659d-->
1. **Router-Gradient bei erneuter Normierung für $k = 1$ und $k > 1$.** Viele MoE-Schichten normieren die ausgewählten Wahrscheinlichkeiten erneut, indem sie diese durch ihre Summe teilen: $g_e(\mathbf{x}) = p_e(\mathbf{x}) / \sum_{j \in T(\mathbf{x})} p_j(\mathbf{x})$.
   1. Zeigen Sie, dass dieselbe Normierung für $k = 1$ den konstanten Wert $g_{e^*}(\mathbf{x}) = p_e(\mathbf{x}) / p_e(\mathbf{x}) \equiv 1$ ergibt. Damit gilt $\partial \textrm{MoE}(\mathbf{x}) / \partial \mathbf{W}_r = \mathbf{0}$, und der Router erhält nie einen Gradienten aus dem Verlust der eigentlichen Aufgabe.
   2. Prüfen Sie im Code, dass unsere `MoELayer` dieses Problem vermeidet, indem sie jeden ausgewählten Experten mit seiner ursprünglichen Wahrscheinlichkeit gewichtet (also $g_{e^*}(\mathbf{x}) = p_{e^*}(\mathbf{x})$ wie beim Switch Transformer :cite:`fedus2022switch`).
   :exlabel:`ex_moe-router-gradient`

<!--en:e3657d89-->
2. **Minimum und Gradient des Switch-Hilfsverlusts.** Betrachten Sie $\mathcal{L}_{\textrm{aux}} = E \sum_{e=1}^E f_e \bar{p}_e$ (:eqref:`eq_moe-aux`). Dabei behandeln wir $f_e$ durch einen Gradientenstopp als Konstante, und es gilt $\sum_e \bar{p}_e = 1$.
   1. Berechnen Sie $\partial \mathcal{L}_{\textrm{aux}} / \partial s_{t,e}$ bezüglich des Router-Logits von Token $t$, also $s_{t,e} = (\mathbf{x}_t \mathbf{W}_r)_e$. Zeigen Sie, dass der Gradient genau dann verschwindet, wenn alle empirischen Lastanteile gleich sind ($f_1 = \dots = f_E = 1/E$).
   2. Beweisen Sie mit der Cauchy-Schwarz-Ungleichung, dass aus $\bar{p}_e = f_e$ die Ungleichung $\mathcal{L}_{\textrm{aux}} \ge 1$ folgt, mit Gleichheit genau dann, wenn $f_e = 1/E$ gilt. Bestimmen Sie den Maximalwert $E$ beim vollständigen Kollaps auf einen einzigen Experten ($f_1 = \bar{p}_1 = 1$).

<!--en:318fbb8d-->
3. **Positive Rückkopplung und ungleiche Eingabeverteilung auseinanderhalten.** Vergleichen Sie in `balancing_run` die Variante `frozen router` mit `no balancing` nach den Schritten $100, 200, 400, 800$. Betrachten Sie den Tokenanteil des am stärksten ausgelasteten Experten. Im frisch initialisierten bytebasierten Modell beträgt er bereits etwa ein Drittel. Weicht dieser Anteil mit trainierbarem Router stärker vom Anfangswert ab als mit eingefrorenem Router? Verwenden Sie für diese Beurteilung genügend Zufallsstartwerte. Was geschieht, wenn Sie alle Ausgabeprojektionen `W_2` der Experten mit null initialisieren, sodass alle Experten in den ersten Schritten identische Ausgaben liefern?
   :exlabel:`ex_moe-feedback`

<!--en:e5a146b3-->
4. **Einfluss der Schrittweite beim Bias-Regler ohne Hilfsverlust.** Führen Sie `balancing_run` mit `bias_rate` aus und verwenden Sie die Werte $\eta_b \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$. Tragen Sie den höchsten Tokenanteil eines Experten und den Validierungsverlust über den Trainingsschritten auf. Warum schwankt die Expertenlast bei zu großer Schrittweite des Reglers ($\eta_b = 10^{-1}$) von Schritt zu Schritt?
   :exlabel:`ex_moe-bias-controller`

<!--en:41fd2669-->
5. **Überanpassung von MoE-Modellen bei sehr kleinen Datenmengen.** Wiederholen Sie den Vergleich über 6000 Schritte zwischen `dense` und `MoE` (:numref:`subsec_moe-gpt`). Beschränken Sie `data.train` auf die ersten $65{,}536$ Byte ($64\textrm{ KB}$), sodass die Modelle denselben Text mehrere hundert Mal durchlaufen. Tragen Sie den Trainingsverlust und den Validierungsverlust für `dense` und `MoE` auf. Warum lernt das MoE-Modell mit $2.26\textrm{M}$ Parametern das kleine Korpus viel schneller auswendig und zeigt früher Überanpassung als das dichte Modell mit $0.43\textrm{M}$ Parametern, obwohl beide genau gleich viele FLOPs pro Token benötigen?
   :exlabel:`ex_moe-data-regime`

<!--en:6fc416a1-->
6. **Gemeinsame Experten und feinere Unterteilung bei festem Budget.** Implementieren Sie eine `SharedMoELayer`, die 1 stets aktiven gemeinsamen Experten der Breite $w$ ausführt und zusätzlich die Top-$k$ von $E$ auswählbaren Experten der Breite $w$. Vergleichen Sie drei FFN-Konfigurationen in `small_gpt` bei gleicher gespeicherter und aktiver Gesamtbreite der Experten ($16 \times 170$ gespeichert, $2 \times 170$ aktiv pro Block): (a) grob mit $E = 8, k = 1, w = 340$, (b) mittel mit $E = 16, k = 2, w = 170$ und (c) fein mit gemeinsamem Experten und $E_{\textrm{shared}} = 1, E = 15, k = 1, w = 170$. Vergleichen Sie die Validierungsverluste nach 800 Schritten.
   :exlabel:`ex_moe-granularity`

<!--en:1af279af-->
[Diskussion](https://d2l.discourse.group/t/36779)

<!--en:273e2ae0-->
Transformer mit dünn besetzter Mixture of Experts (MoE)<br>**gespeicherte und aktive Kapazität · Top-k-Routing · Puffer fester Kapazität · Hilfsverlust oder Bias-Regelung · feingranulare und gemeinsame Experten**

<!--en:af715f66-->
::: {.slide title="Kapazität und FLOPs entkoppeln"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:27131ec9-->
- **Dichter Transformer:** Jedes Token wird mit jeder Gewichtsmatrix multipliziert ($N_{\text{active}} = N_{\text{total}}$).

<!--en:c9350e15-->
- **Dünn besetzte Mixture of Experts (MoE):** Ersetzt das FFN jedes Blocks ($8d^2$ von $12d^2$ Parametern) durch $E$ parallele FFN als **Experten**. Jedes Token gelangt nur zu seinen Top-$k$-Experten ($1 \le k \ll E$).

<!--en:e552c184-->
- **Dimensionen heutiger Spitzenmodelle:** `DeepSeek-V3` speichert **$671\text{B}$** Parameter und aktiviert pro Token nur **$37\text{B}$**.

<!--en:91b963aa-->
::: {.slide title="Top-k-Routing in einer Mixture-of-Experts-Schicht"}

<!--en:77065343-->
[Aufbau einer MoE-Schicht]{.kicker}

<!--en:2904a6ec-->
- **Linearer Router ($\mathbf{W}_r \in \mathbb{R}^{d \times E}$):** Bildet das Token $\mathbf{x} \in \mathbb{R}^d$ auf die Wahrscheinlichkeiten $\mathbf{p}(\mathbf{x}) = \mathrm{softmax}(\mathbf{x}\mathbf{W}_r) \in \Delta^{E-1}$ für die Experten ab.

<!--en:9c114451-->
- **Dünn besetzte Top-$k$-Auswahl:** Wählt die Indexmenge $\mathcal{T}_k(\mathbf{x}) \subset \{1, \dots, E\}$ der $k$ Experten mit den höchsten Scores und bildet die gewichtete Summe:
  $$\mathrm{MoE}(\mathbf{x}) = \sum_{e \in \mathcal{T}_k(\mathbf{x})} p_e(\mathbf{x}) \, \mathrm{FFN}_e(\mathbf{x})$$

<!--en:61921192-->
- **Keine Berechnung bei den übrigen Experten:** Die verbleibenden $E - k$ Experten liegen im Speicher, ohne für das Token $\mathbf{x}$ auch nur einen FLOP auszuführen.

<!--en:7267810b-->
::: {.slide title="Gespeicherte (N_total) und aktive (N_active) Parameter"}

<!--en:90b25bbf-->
[Parameterbilanz]{.kicker}

<!--en:daa4bf74-->
Für ein MoE-Modell mit $L$ Blöcken, $E$ vom Router auswählbaren SwiGLU-Experten der inneren Breite $w$ ($3 d w$ Gewichte pro Experte) und $E_{\text{shared}}$ stets aktiven gemeinsamen Experten gilt:

<!--en:9636f9fa-->
| Ressource / Flaschenhals | Maßgebliche Größe | Begründung |

<!--en:87c40101-->
| **FLOPs bei Training und Prefill** | **$N_{\text{active}}$** | Pro Token arbeiten nur $k + E_{\text{shared}}$ Experten ($C \approx 6 N_{\text{active}} D$) |

<!--en:778bb19f-->
| **VRAM-Kapazität der GPU** | **$N_{\text{total}}$** | Alle $E + E_{\text{shared}}$ Experten müssen im HBM der GPU liegen |

<!--en:330811ea-->
| **Bandbreite bei der Dekodierung im Batch** | **$N_{\text{active}} \to N_{\text{total}}$** | Bei einer Batchgröße von $B \gg E/k$ nutzen die Token zusammen fast alle $E$ Experten |

<!--en:3dc567eb-->
::: {.slide title="Wie der Router bei diskreter Top-k-Auswahl lernt"}

<!--en:f575a4a5-->
[Router-Gradienten]{.kicker}

<!--en:0da5a734-->
- **Differenzierbares Gate $p_e(\mathbf{x})$:** Für jeden ausgewählten Experten $e \in \mathcal{T}_k(\mathbf{x})$ liefert die Kettenregel die Ableitung nach dem skalaren Gewicht $p_e(\mathbf{x})$:
  $$\frac{\partial \mathcal{L}}{\partial p_e(\mathbf{x})} = \bigl(\nabla_{\mathrm{MoE}(\mathbf{x})} \mathcal{L}\bigr)^\top \mathrm{FFN}_e(\mathbf{x})$$

<!--en:3ce8554a-->
- **Problem der erneuten Normierung bei $k = 1$:** Normiert man für **$k = 1$** erneut, gilt $g_{e^*} = p_{e^*} / p_{e^*} \equiv 1$. Daraus folgt $\partial \mathrm{MoE}(\mathbf{x}) / \partial \mathbf{W}_r = \mathbf{0}$, und der Router lernt nie.

<!--en:bbbf4e86-->
::: {.slide title="Puffer mit fester Kapazität und Verwerfen von Token"}

<!--en:e69063d2-->
[Kompilierte Tokenverteilung]{.kicker}

<!--en:83ebf274-->
- **Statische Pufferkapazität $S$ ($\gamma \ge 1$):** Kompilierte GPU- und TPU-Kernel reservieren einen statischen Puffer der Form $(E, S, d)$ mit
  $$S = \left\lceil \gamma \cdot \frac{n k}{E} \right\rceil$$
  Bei $\gamma = 1.25$ bleiben gegenüber der gleichmäßigen Last $25\%$ Reserve.

<!--en:8af8fa7c-->
- **Überlauf und Verwerfen von Token:** Eine Zuweisung mit Warteschlangenposition $\text{rank} \ge S$ landet auf einem ungenutzten Überlaufplatz. Das Token durchläuft den Residualstrom unverändert.

<!--en:672a83d2-->
::: {.slide title="Vektorisierte Verteilung, SwiGLU im Batch und Zusammenführen der Ergebnisse" only="pytorch"}

<!--en:c6695647-->
[Implementierung von MoELayer]{.kicker}

<!--en:43ebfe90-->
`MoELayer.forward` bestimmt die Warteschlangenposition jedes Tokens über kumulierte Summen der One-Hot-Kodierungen. Die Methode verteilt die Token auf `buf` (Platz `E * S` fängt den Überlauf ab), führt alle $E$ Experten mit einem einzigen `bmm` im Batch aus und sammelt die Ergebnisse ein:

<!--en:b4d27d1a-->
::: {.slide title="Vektorisierte Verteilung, SwiGLU im Batch und Zusammenführen der Ergebnisse" only="jax"}

<!--en:c6695647-2-->
[Implementierung von MoELayer]{.kicker}

<!--en:ea3ed559-->
`MoELayer.__call__` bestimmt die Warteschlangenposition jedes Tokens über kumulierte Summen der One-Hot-Kodierungen. Die Methode verteilt die Token auf `buf` (`mode='drop'` ignoriert den Platz `E * S` außerhalb des gültigen Bereichs), führt alle $E$ Experten parallel aus und sammelt die Ergebnisse ein (`mode='fill', fill_value=0`):

<!--en:5088346f-->
::: {.slide title="MoELayer in einen TransformerBlock einbinden"}

<!--en:c6695647-3-->
[Implementierung von MoELayer]{.kicker}

<!--en:a9080f6a-->
::: {.slide title="Warum Eingabeverteilung und Rückkopplung zu ungleicher Last führen"}

<!--en:cb26526d-->
[Routing-Kollaps]{.kicker}

<!--en:fbf5fe9d-->
- **1. Ungleiche Verteilung der Eingaben ($t = 0$):** Häufige Bytes (`' '` macht $18.9\%$ von WikiText-103 aus, mehr als $\gamma / E = 15.6\%$) und gemeinsame Mittelwerte der Repräsentationen begünstigen in den anfänglichen Logits 1–2 Experten.

<!--en:5ab3ab07-->
- **2. Positive Rückkopplung verstärkt den Vorsprung:** Die am stärksten ausgelasteten Experten erhalten mehr Gradientenupdates und senken den Verlust schneller. Über $\nabla_{\mathbf{W}_r}\mathcal{L}$ steigt ihr $p_e(\mathbf{x})$. Den kaum genutzten Experten bleiben so womöglich zu wenige Token zum Lernen.

<!--en:6c9e9e07-->
::: {.slide title="Lösung 1: Switch-Hilfsverlust zum Lastausgleich"}

<!--en:be4be804-->
[Lastausgleich mit Hilfsverlust]{.kicker}

<!--en:2e72ebb6-->
- Der Switch Transformer (Fedus et al., 2022) ergänzt den Strafterm $\alpha \mathcal{L}_{\text{aux}}$ ($\alpha = 0.01$):
  $$\mathcal{L}_{\mathrm{aux}} = E \sum_{e=1}^{E} f_e \, \bar{p}_e$$

<!--en:269efbe3-->
- Für $f_e$ gilt ein Gradientenstopp, während $\bar{p}_e$ differenzierbar ist. Daher senkt $\partial \mathcal{L}_{\text{aux}} / \partial \bar{p}_e = E f_e$ die Wahrscheinlichkeit überlasteter Experten ($f_e > 1/E$). Bei gleichmäßigem Routing erreicht der Hilfsverlust sein Minimum $\mathcal{L}_{\text{aux}} = 1$.

<!--en:6e564aec-->
::: {.slide title="Lösung 2: dynamische Regelung des Router-Bias ohne Hilfsverlust"}

<!--en:10321f9c-->
[Lastausgleich ohne Hilfsverlust]{.kicker}

<!--en:89ca26c4-->
- **Bias-Regler ohne Hilfsverlust bei DeepSeek-V3 (Wang et al., 2024):**
  $$\mathcal{T}_k(\mathbf{x}) = \operatorname{argtop}_k\bigl(\mathbf{p}(\mathbf{x}) + \mathbf{b}\bigr), \qquad b_e \leftarrow b_e + \eta_b \, \mathrm{sign}\!\left(\frac{1}{E} - f_e\right)$$

<!--en:5e3ccfba-->
- Verschiebt *nur* die Auswahlschwelle von $\operatorname{argtop}_k$. Die Gate-Gewichte $p_e(\mathbf{x})$ in $\sum_{e \in \mathcal{T}_k} p_e(\mathbf{x})\,\mathrm{FFN}_e(\mathbf{x})$ bleiben völlig unverändert (keine Beeinflussung des Gradienten).

<!--en:d41c67a5-->
::: {.slide title="Fünf Routing-Varianten auf WikiText-103 im Vergleich"}

<!--en:76b1be05-->
[Ablationsstudie zum Lastausgleich]{.kicker}

<!--en:d448e109-->
Ein bytebasiertes GPT mit 2 Blöcken, 8 Experten und Top-1-Routing ($d = 128, \gamma = 1.25$) durchläuft $800$ Trainingsschritte mit fünf Routing-Varianten (je zwei Zufallsstartwerte):

<!--en:060e7293-->
- **Ungleichgewicht zu Beginn und ohne Lastausgleich:** Bei `initialization` und `no balancing` erhält der am stärksten ausgelastete Experte $2.4\text{--}3.9\times$ so viele Token wie bei gleichmäßiger Verteilung ($30\%\text{--}49\%$ aller Token). **$20\%\text{--}36\%$** der Zuweisungen entfallen.

<!--en:acd2e83c-->
- **Beide Verfahren verhindern den Kollaps:** Mit `auxiliary loss` ($\alpha = 0.01$) oder `bias controller` ($\eta_b = 0.01$) erreicht kein Experte eine Last von $2\times$ seines Idealanteils. Höchstens wenige Prozent der Zuweisungen werden verworfen. Der Validierungsverlust sinkt um etwa **$0.1\text{--}0.25\text{ nats/byte}$**.

<!--en:83d09c9d-->
::: {.slide title="Tokenanteile der Experten nach Verfahren"}

<!--en:b9b5f205-->
[Lastverteilung]{.kicker}

<!--en:f83096b3-->
- **Experten nach Tokenanteil (1 höchste, 8 niedrigste Last):** Bei `no balancing` (`C0`) erhalten die ersten 3 Experten etwa $70\%$ aller Token, der letzte fast keine.

<!--en:1703a36c-->
- **Gleichmäßige Auslastung ($1/8 = 12.5\%$):** Beim Switch-Hilfsverlust (`C1`) und beim Bias-Regler ohne Hilfsverlust (`C2`) liegen die Anteile aller 8 Experten nahe an der waagerechten schwarzen Linie bei $12.5\%$.

<!--en:8142aabb-->
::: {.slide title="IsoFLOP-Vergleich zwischen MoE mit 8 Experten und einem dichten GPT"}

<!--en:0be646cb-->
[Aufbau des kontrollierten Vergleichs]{.kicker}

<!--en:46779968-->
Ist ein Modell mit bedingt aktiven Experten einem dichten Transformer überlegen, wenn **die Anzahl aktiver Parameter und die FLOPs pro Token exakt übereinstimmen**? Im Vergleich stehen ein dichtes GPT aus 2 Blöcken und ein Top-1-MoE-GPT mit 8 Experten ($\gamma = 2$). Das Training umfasst $6{,}000$ Schritte ($49\text{M}$ Byte):

<!--en:3865b339-->
- **Gleicher Rechenaufwand ($0.43\text{M}$ aktive Parameter pro Token):** In beiden Modellen benötigt jedes Token genau gleich viele Matrizenmultiplikationen für Attention und FFN.

<!--en:9ebf95a8-->
- **$5.3\times$ so viele gespeicherte Parameter:** Das MoE-Modell speichert $2.26\text{M}$ Parameter ($8$ Experten-FFNs pro Schicht statt $1$). Beim Kapazitätsfaktor $\gamma = 2$ entfällt auch dann keine Zuweisung, wenn zu Beginn viele häufige Bytes aufeinanderfolgen.

<!--en:1b75025a-->
::: {.slide title="MoE überholt dichtes GPT bei gleicher Zahl an FLOPs pro Token"}

<!--en:b8b7c7a0-->
[IsoFLOP-Lernkurven]{.kicker}

<!--en:e2a002ac-->
- **Anfänglicher Rückstand (Schritte $0\text{--}1{,}500$):** Das MoE-Modell liegt zunächst leicht zurück. Jeder der $8$ Experten erhält pro Schritt nur $\sim 1/8$ der Gradientenupdates, und seine Ausgabe wird mit $p_e(\mathbf{x}) \approx 1/8$ skaliert.

<!--en:672714c0-->
- **Überholpunkt und Gewinn von $\sim 0.05\text{-nat}$:** Innerhalb der ersten $2{,}000$ Schritte überholen die spezialisierten Experten das dichte FFN. Über die Zufallsstartwerte hinweg liegt ihr Verlust am Ende **um $0.05\text{ nats/byte}$ niedriger**. Derselbe Gewinn erfordert beim dichten Modell dreimal so viele FLOPs.

<!--en:13d5aabe-->
::: {.slide title="Feingranulare Unterteilung und gemeinsame Experten"}

<!--en:e7ab229e-->
[Aufbau moderner MoE-Modelle]{.kicker}

<!--en:96680bd3-->
- **1. Feingranulare Unterteilung (DeepSeekMoE, Dai et al. 2024):** Aus $E = 16$ Experten voller Breite ($k = 2$) werden $m E = 64$ Experten mit je einem Viertel der Breite ($m k = 8$). Die Anzahl gespeicherter und aktiver Parameter bleibt jeweils gleich. Die Zahl der aktiven Kombinationen wächst von $\binom{16}{2} = 120$ auf $\binom{64}{8} \approx 4.4 \times 10^9$.

<!--en:9777922c-->
- **2. Stets aktive gemeinsame Experten:** $E_{\text{shared}} \ge 1$ Experten verarbeiten *jedes* Token und übernehmen häufige syntaktische Transformationen. So vermeiden die vom Router ausgewählten Experten redundante Arbeit.

<!--en:0fb83113-->
::: {.slide title="Expertenkonfigurationen führender MoE-Modelle mit offenen Gewichten"}

<!--en:3ec0f888-->
[Spitzenmodelle im Vergleich]{.kicker}

<!--en:d27b198d-->
- **Grobes Routing (Mixtral 8x7B, 2024):** $E = 8$ Experten voller Breite, Top-$2$ aktiv ($47\text{B}$ insgesamt / $13\text{B}$ aktiv, Kapazitätsverhältnis $3.6\times$).

<!--en:a603e28f-->
- **Feingranulare Experten (Qwen3, DeepSeek-V3, gpt-oss):** Bei $128\text{--}256$ schmalen Experten ($w/d \in [0.29, 1.0]$) mit Top-$4$- oder Top-$8$-Routing beträgt **das Verhältnis von gespeicherter zu aktiver Kapazität $11\times\text{--}23\times$**. DeepSeek-V3 ergänzt einen gemeinsamen Experten.

<!--en:ea4ae731-->
::: {.slide title="Verständnisfragen zu MoE-Kapazität, Verlust und Datenbudget"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:7c4e6378-->
Drei Rechenaufgaben zum Aufbau von MoE-Modellen:

<!--en:0dbfb6ae-->
1. **Anteil aktiver Expertenparameter:** Eine Schicht hat $E = 64$ vom Router auswählbare Experten der Breite $w$. Pro Token aktiviert sie $k = 6$ davon. Hinzu kommen $E_{\text{shared}} = 2$ stets aktive gemeinsame Experten derselben Breite $w$. Welcher Anteil der Expertenparameter dieser Schicht kommt für ein einzelnes Token zum Einsatz?

<!--en:01eade3f-->
2. **Berechnung des Switch-Hilfsverlusts:** In einer Schicht mit $4$ Experten ($E = 4$) entfallen in einem Minibatch $50\%$ der Plätze auf Experte 1 ($f_1 = \bar{p}_1 = 1/2$) und je $1/6$ auf die Experten 2, 3 und 4 ($f_e = \bar{p}_e = 1/6$). Welchen Wert hat $\mathcal{L}_{\text{aux}} = E \sum_{e=1}^4 f_e \bar{p}_e$?

<!--en:76c26a90-->
3. **Token pro Experte:** Wie viele Trainingsbytes durchlaufen in unserem Lauf über $6{,}000$ Schritte auf WikiText-103 ($64 \times 128 = 8{,}192$ Byte pro Schritt) das dichte FFN? Wie viele durchlaufen bei gleichmäßiger Verteilung jeden einzelnen Experten einer Top-1-MoE-Schicht mit 8 Experten?

<!--en:557b98b6-->
**Lösungen:**

<!--en:0d4880a8-->
1. **$(k + E_{\text{shared}}) / (E + E_{\text{shared}}) = (6 + 2) / (64 + 2) = 8 / 66 \approx \mathbf{12.1\%}$** (zuzüglich der kleinen $d \times 64$-Router-Matrix).

<!--en:8f149bd5-->
2. **$\mathcal{L}_{\text{aux}} = 4 \times \bigl(\frac{1}{2}\cdot\frac{1}{2} + 3 \cdot \frac{1}{6}\cdot\frac{1}{6}\bigr) = 4 \times \bigl(\frac{1}{4} + \frac{1}{12}\bigr) = 4 \times \frac{1}{3} \approx \mathbf{1.333}$** (über dem Minimum von $1.000$ bei gleichmäßiger Last).

<!--en:8e804441-->
3. **$6{,}000 \times 8{,}192 \approx \mathbf{49.2\text{M}\text{ bytes}}$** durchlaufen das dichte FFN. Jeder Top-1-Experte verarbeitet dagegen nur $49.2\text{M} / 8 \approx \mathbf{6.1\text{M}\text{ bytes}}$. Das erklärt, warum MoE-Modelle bei sehr kleinen Datensätzen schnell zur Überanpassung neigen und für das Pre-Training große Korpora benötigen.

<!--en:98a07d21-->
::: {.slide title="Zusammenfassung: Transformer mit dünn besetzter Mixture of Experts"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:b3f25dfb-->
- **Bedingte Berechnung mit Top-$k$-Routing:** $\mathrm{MoE}(\mathbf{x}) = \sum_{e \in \mathcal{T}_k(\mathbf{x})} p_e(\mathbf{x})\,\mathrm{FFN}_e(\mathbf{x})$ entkoppelt die gespeicherte Kapazität $N_{\text{total}}$ von den FLOPs pro Token, die von $N_{\text{active}}$ abhängen.

<!--en:c795ea8c-->
- **Statische Puffer und Lastausgleich:** Feste Kapazität $S = \lceil \gamma n k / E \rceil$ in Verbindung mit dem Switch-Hilfsverlust oder der dynamischen Regelung des Bias $b_e$ ohne Hilfsverlust.

<!--en:b0be0b78-->
- **Feingranulare Experten:** Führende MoE-Modelle unterteilen ihre Experten nach dem Muster $E \to mE, k \to mk$. DeepSeek-V3 ergänzt einen gemeinsamen Experten.
