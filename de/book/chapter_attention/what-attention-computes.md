<!--en:07a5ff2e-->
# Attention-Schaltkreise

<!--en:8fb72da2-->
Wie in :numref:`sec_queries-keys-values` erläutert, liefert eine Attention-Heatmap eine Hypothese darüber, was eine Schicht berechnet. Ein Beweis ist sie nicht. Die Heatmap zeigt, auf welche Positionen ein Kopf zugreift. Das ist nur die Hälfte seiner Berechnung. Was er zur Ausgabe beiträgt, hängt von seiner Wertprojektion und seiner Ausgabeprojektion ab. Konzentriert sich ein Kopf auf ein Token, kann er dessen Identität kopieren, ein einzelnes Merkmal aus einem Unterraum extrahieren oder einen vernachlässigbar kleinen Vektor beitragen.

<!--en:5483a8f3-->
Um zu verstehen, was trainierte Attention-Köpfe berechnen, übernehmen wir die mechanistische Beschreibung durch Schaltkreise von :citet:`Elhage.Nanda.Olsson.ea.2021`. Wir wenden sie auf das reine Attention-Sprachmodell `TinyCharLM` (:numref:`sec_positional-information`) an. Jeder Kopf lässt sich durch zwei Operatoren vom Rang $d_h$ auf dem *Residualstrom* (engl. residual stream) beschreiben. Der bilineare *QK-Schaltkreis* (engl. QK circuit) bestimmt, auf welche Positionen der Kopf zugreift. Der lineare *OV-Schaltkreis* (engl. OV circuit) bestimmt, wie die Vektoren an diesen Positionen die Ausgabe-Logits verändern. Mit dieser Zerlegung zeigen wir, warum ein Modell mit einer Schicht Bigramme und Skip-Trigramme darstellen und damit auch einzelne Token kopieren kann. Muster im Kontext vervollständigen ($[\mathrm{A}][\mathrm{B}] \ldots
[\mathrm{A}] \to [\mathrm{B}]$) kann es hingegen nicht. In einem Modell mit zwei Schichten kann ein *Vorgängerkopf* (engl. previous-token head) mit einem *Kopf zum Abgleichen und Kopieren* (engl. match-and-copy head) einen *Induktionskopf* (engl. induction head) bilden :cite:`Olsson.Elhage.Nanda.ea.2022`. Abschließend prüfen wir diese Hypothese, indem wir die Gewichte untersuchen und durch Ablation einzelner Köpfe deren kausalen Beitrag messen.

<!--en:a0d5299d-->
## Schaltkreise und Residualstrom

<!--en:97dde2ec-->
### QK- und OV-Schaltkreise

<!--en:6520a2c0-->
In `TinyCharLM` (:numref:`subsec_tinycharlm`) durchlaufen die Token-Einbettungen $\mathbf{e}_{x_i} \in \mathbb{R}^d$ ($d = 128$) insgesamt $N$ Residualblöcke. Jeder Block besteht aus $H = 4$ Köpfen für kausale Attention mit der Breite $d_h =
d/H = 32$ und kommt ohne MLP, Normalisierung und Bias-Terme aus. Am Ende steht die Ausgabeprojektion $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}$, deren Gewichte an die Einbettung gekoppelt sind.

<!--en:fc500027-->
Schreiben wir die Token-Vektoren als Spaltenvektoren $\mathbf{h}_j \in \mathbb{R}^d$. Jeder Kopf projiziert $\mathbf{h}_j$ mit $\mathbf{W}_v \in \mathbb{R}^{d_h \times d}$ und bildet die gewichtete Summe $\sum_{j \le i} \alpha_{ij} \mathbf{W}_v
\mathbf{h}_j$. Wir zerlegen die Ausgabematrix des Blocks in $H$ Spaltenblöcke $\mathbf{W}_o \in \mathbb{R}^{d \times d_h}$, einen pro Kopf. Damit lassen sich Verkettung und anschließende Projektion als Summe unabhängiger Beiträge der Köpfe schreiben:

<!--en:a0c5bf54-->
Entsprechend ist der Score vor der Softmax-Funktion zwischen der Abfrage an Position $i$ und dem Schlüssel an Position $j$ gleich $(\mathbf{W}_q \mathbf{h}_i)^\top (\mathbf{W}_k \mathbf{h}_j) /
\sqrt{d_h}$ mit $\mathbf{W}_q, \mathbf{W}_k \in \mathbb{R}^{d_h \times d}$. Zwei Matrizen der Größe $d \times d$ mit Rang höchstens $d_h$ beschreiben somit jeden Kopf vollständig :cite:`Elhage.Nanda.Olsson.ea.2021`:

<!--en:0ad8fe51-->
* Der **QK-Schaltkreis** $\mathbf{W}_{\mathrm{QK}}$ (bei RoPE $\mathbf{W}_q^\top
  \mathbf{R}_{j-i} \mathbf{W}_k$, siehe :numref:`subsec_pos-rope`) bestimmt, *wohin* der Kopf seine Attention richtet.

<!--en:b1ec01dc-->
* Der **OV-Schaltkreis** $\mathbf{W}_{\mathrm{OV}}$ bestimmt, *welchen* Vektor der Kopf an die Zielposition schreibt, wenn er auf eine Quellposition zugreift.

<!--en:884a1a25-->
Untersuchen wir die Produkte $\mathbf{W}_{\mathrm{QK}}$ und $\mathbf{W}_{\mathrm{OV}}$ anstelle der einzelnen Faktoren, entfällt die Eichfreiheit. Die Faktoren $\mathbf{W}_o \mathbf{G}^{-1}$ und $\mathbf{G}\mathbf{W}_v$ ergeben genau dasselbe $\mathbf{W}_{\mathrm{OV}}$ für jede invertierbare Matrix $\mathbf{G} \in \mathbb{R}^{d_h \times d_h}$. Zugleich wird der Flaschenhals sichtbar, der durch die Beschränkung auf einen Unterraum vom Rang $d_h$ entsteht.

<!--en:7e9f97de-->
### Repräsentationen im Residualstrom über mehrere Schichten

<!--en:9dc74be1-->
Setzen wir $\mathbf{h}_i^{(0)} = \mathbf{e}_{x_i}$. Dann aktualisiert Block $\ell \in \{1, \ldots,
N\}$ die Repräsentation an Position $i$ durch Addition:

<!--en:0f0d95f2-->
Der Ausgabekopf mit gekoppelten Gewichten liefert die Logits $\mathbf{z}_i = \mathbf{E}\,
\mathbf{h}_i^{(N)}$. Die Folge $\mathbf{h}_i^{(0)}, \ldots,
\mathbf{h}_i^{(N)}$ ist der *Residualstrom* an Position $i$ (:numref:`fig_residual-stream`). Er dient als gemeinsamer Kommunikationskanal, aus dem jeder Kopf liest und zu dem er seinen Beitrag addiert.

<!--en:e75097b9-->
![Der Residualstrom. An jeder Position durchläuft ein Vektor die Schichten von der Einbettung bis zu den Ausgabe-Logits. Der QK-Schaltkreis bestimmt die Attention-Gewichte $\alpha_{3j}$. Der OV-Schaltkreis bestimmt, welchen Beitrag jeder berücksichtigte Vektor zum Residualstrom an der Zielposition leistet.](../img/mdl-attention-residual-stream.svg)

<!--en:8f2af948-->
### Bigramme, Skip-Trigramme und die Grenzen einer einzelnen Schicht

<!--en:47f2be2d-->
Entwickeln wir :eqref:`eq_stream-update` nach der Tiefe, erkennen wir, welche Funktionen $N$ Attention-Schichten darstellen können:

<!--en:30bdff45-->
* **Kein Block ($N = 0$):** $\mathbf{z}_i = \mathbf{E}\,\mathbf{e}_{x_i}$. Damit lassen sich die Logits für Token-Bigramme nur so weit modellieren, wie es die symmetrische, positiv semidefinite Gram-Matrix $\mathbf{E}\mathbf{E}^\top$ zulässt (Aufgabe :exref:`ex_what-attention-computes-tied-bigrams`).

<!--en:510045ab-->
* **Ein Block ($N = 1$):**

  $$
  \mathbf{z}_i = \mathbf{E}\,\mathbf{e}_{x_i} + \sum_{m=1}^{H} \sum_{j \leq i} \alpha_{ij}^{m}\; \mathbf{E}\,\mathbf{W}_{\mathrm{OV}}^{m}\,\mathbf{e}_{x_j}.
  $$
  :eqlabel:`eq_one-layer-paths`

  Der Term jedes Kopfes realisiert eine *Skip-Trigramm*-Regel der Form $x_j \ldots
  x_i \to x_{\textrm{next}}$. Zum Beispiel kann der Kopf von *„in“* aus auf ein früheres *„keep“* zugreifen und das Logit für *„mind“* erhöhen. Er kann auch *kopieren*, indem er das Logit für $x_j$ selbst erhöht.

<!--en:da2791f3-->
* **Zwei Blöcke ($N = 2$) und der Induktionskopf:** Angenommen, der Kontext enthält $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}]$ und das Modell soll $[\mathrm{B}]$ durch Nachschlagen im Kontext vorhersagen. Mit einem Block gelingt dies nicht durch inhaltlichen Abgleich. Um $[\mathrm{B}]$ von Position $j$ zu kopieren, muss ein Kopf auf Position $j$ zugreifen. In :eqref:`eq_one-layer-paths` hängt der Schlüssel an Position $j$ aber nur von $\mathbf{e}_{x_j} = \mathbf{e}_{[\mathrm{B}]}$ ab und enthält keine Information über den Vorgänger $x_{j-1} = [\mathrm{A}]$.

  Zwei Blöcke lösen die Aufgabe durch *K-Komposition* (:numref:`fig_induction-circuit`, Aufgabe :exref:`ex_what-attention-computes-paths`). In Block 1 greift ein *Vorgängerkopf* mit dem Versatz $-1$ auf das vorherige Token zu und schreibt $x_{j-1} =
  [\mathrm{A}]$ in den Residualstrom $\mathbf{h}_j^{(1)}$ an Position $j$. In Block 2 gleicht ein *Induktionskopf* seine Abfrage $[\mathrm{A}]$ mit diesem Schlüsselmerkmal in $\mathbf{h}_j^{(1)}$ ab. Über seinen OV-Schaltkreis kopiert er $x_j = [\mathrm{B}]$ in die Vorhersage-Logits.

<!--en:ec65f413-->
![Der Induktionsschaltkreis erstreckt sich über zwei Schichten. In Schicht 1 schreibt ein Vorgängerkopf den Vorgänger jedes Tokens in dessen Residualstrom. Der Schlüssel an der Position von B kann so „folgt auf A“ signalisieren. In Schicht 2 gleicht der QK-Schaltkreis des Induktionskopfs das aktuelle A mit diesem Signal ab. Sein OV-Schaltkreis kopiert B in die Vorhersage.](../img/mdl-attention-induction-circuit.svg)

<!--en:988058d1-->
## Kopieren und Induktion

<!--en:dadd9db6-->
### Eine Aufgabe mit wiederholten Sequenzen

<!--en:fd2b32de-->
Um zu prüfen, ob der Gradientenabstieg diesen zweischichtigen Schaltkreis findet, trainieren wir mit synthetischen Sequenzen aus *wiederholten zufälligen Token*. Für jede Sequenz ziehen wir ein zufälliges Muster der Länge $L \in [L_{\min}, L_{\max}]$ gleichverteilt aus einem Vokabular der Größe $64$ und wiederholen es bis zur Länge $64$. Da wir jedes Muster neu ziehen, ist die erste Kopie ($t < L$) unvorhersagbar (Kreuzentropie bei zufälligem Raten: $\log 64
\approx 4.16$ Nat). Das gilt auch für das Token an Position $t = L$, wo das Muster von vorn beginnt. Im Kontext hat sich das Muster noch nicht wiederholt, und das Modell kann $L$ nicht kennen. Daher zählen wir die Zielpositionen $t \le L$ zur ersten Kopie. Die späteren Kopien ($t > L$) lassen sich durch Nachschlagen im Kontext vorhersagen.

<!--en:e7df75f3-->
`eval_copy` misst die Kreuzentropie für die erste Kopie ($t \le L$), die Kreuzentropie für spätere Kopien ($t > L$) und die Top-1-Genauigkeit für spätere Kopien:

<!--en:af51fdbb-->
### Eine Lösung mit festem Versatz

<!--en:3fc218c8-->
Bevor wir variable Perioden untersuchen, nehmen wir eine feste Musterperiode von $L = 32$ an. Ein Modell mit *einem Block* (`num_blks=1, bias=True`) erreicht für die zweite Kopie mühelos eine Genauigkeit von $100\%$:

<!--en:b3b2bfad-->
Wie kann eine einzige Schicht den Nachfolger kopieren, wenn unsere Analyse für inhaltsbasierte Induktion zwei Schichten verlangt? Bei konstantem $L = 32$ stimmt das Zieltoken an Position $i + 1$ immer mit dem Token an der um einen festen Versatz zurückliegenden Position $j = i - 31$ überein. Mit konstanten, durch den Bias bestimmten Abfragen und Schlüsseln kann ein einziger RoPE-Kopf allein anhand des relativen Versatzes $-31$ zugreifen, ganz ohne inhaltlichen Abgleich (Aufgabe :exref:`ex_what-attention-computes-shortcut-bias`). Testen wir `model_shortcut` mit Mustern der Periode $L = 24$, bei denen das Ziel beim Versatz $-23$ liegt, erkennen wir die Abkürzung sofort. Die Attention bleibt beim Versatz $-31$, und die Genauigkeit fällt auf Zufallsniveau ($1/64 \approx 0.02$):

<!--en:a3bd48f8-->
### Ein Induktionsschaltkreis aus zwei Blöcken

<!--en:19b936d7-->
Ziehen wir die Musterperiode gleichverteilt aus $L \in \{16, \ldots, 32\}$, entfällt jede Abkürzung über einen festen Versatz. Wir trainieren drei unabhängige Modelle mit zwei Blöcken und als Kontrolle ein Modell mit einem Block (alle mit `bias=False`) über $5{,}000$ Schritte:

<!--en:dad9bbc2-->
Mit zwei Blöcken erreichen alle drei Läufe bei späteren Kopien eine Genauigkeit von mehr als $90\%$ (Verlust etwa $0.2$ bis $0.4$ Nat). Mit einem Block stagniert der Verlust oberhalb von $3$ Nat ($\sim 10\%$ Genauigkeit). Das Modell erhöht lediglich die Logits der Unigramme, die im Kontext vorkommen. Bei der ersten Kopie liegt der Verlust am Ende jedes Laufs leicht *über* dem Zufallswert $\log 64$. Alle vier Modelle haben gelernt, bereits im Kontext aufgetretene Token zu bevorzugen. In der ersten Periode ist diese Sicherheit unbegründet. Wiederholt sich dort zufällig ein Token, sagt ein Modell mit zwei Blöcken dessen früheren Nachfolger vorher. Außerdem zeigt die Trainingskurve nach einem anfänglichen Plateau einen abrupten Phasenübergang:

<!--en:74cb450f-->
## Belege für den Schaltkreis

<!--en:75d397ff-->
Um zu prüfen, ob `model_two` tatsächlich den zweistufigen Induktionsschaltkreis aus :numref:`fig_induction-circuit` realisiert, untersuchen wir (1) die Attention-Karten, (2) die OV-Kopiermatrizen und (3) die Auswirkungen der Ablation von Köpfen.

<!--en:7d300a69-->
### Attention-Muster der trainierten Köpfe

<!--en:a3111dc1-->
Für eine Testsequenz der Periode $L = 24$ sagt unser Schaltkreis voraus, dass mindestens ein Kopf in Block 1 eine Position zurückblickt ($j = i - 1$). Die Köpfe in Block 2 sollten auf das Induktionsziel $j = i + 1 - L = i - 23$ zugreifen, sobald $i \ge
24$ gilt:

<!--en:a5a6cfd6-->
Wir quantifizieren beide Rollen für variable Musterlängen $L \in \{16, \ldots,
32\}$ über alle drei Läufe. Dazu berechnet `head_scores` für jeden Kopf das mittlere Attention-Gewicht auf dem vorherigen Token ($j = i - 1$) und auf dem Induktionsziel ($j = i + 1 - L$ für $i \ge L$):

<!--en:8d20f638-->
In jedem Lauf spezialisieren sich ein bis drei Köpfe in Block 1 als Vorgängerköpfe. Die Köpfe in Block 2 konzentrieren ihre Attention auf das Induktionsziel $j = i + 1

<!--en:f21f8c94-->
- L$.

<!--en:1c548db7-->
### Die Kopiermatrix

<!--en:3b1569b2-->
Der Zugriff auf Position $j$ führt nur dann zur Vorhersage von $x_j$, wenn der OV-Schaltkreis von Block 2 das Logit des Tokens $x_j$ erhöht. Wir definieren

<!--en:ed1a3e57-->
als die Erhöhung des Logits für das Token $b$, wenn die Summe der OV-Schaltkreise eines Blocks den mittleren Vektor $\bar{\mathbf{h}}_a$ des Residualstroms an Positionen mit Token $a$ liest (oder die ursprüngliche Einbettung $\mathbf{e}_a$). Bei einem kopierenden OV-Schaltkreis liegt das Zeilenmaximum auf der Diagonalen ($b = a$), und zwar in fast allen $64$ Zeilen (bei Zufall beträgt der Anteil $1/64 \approx 0.02$):

<!--en:5c509a52-->
Wir werten `diagonal_share(copy_matrix(...))` für alle drei Trainingsläufe mit zwei Blöcken aus, sowohl für den gemittelten Residualstrom $\bar{\mathbf{h}}_a$ aus Block 1 als auch für die ursprünglichen Token-Einbettungen $\mathbf{e}_a$. Außerdem untersuchen wir das Kontrollmodell mit einem Block. Der OV-Schaltkreis bildet jeweils eine Kopiermatrix, deren Zeilenmaxima fast alle auf der Diagonalen liegen:

<!--en:81054cb5-->
Zwei Einzelheiten sind aufschlussreich. (1) *Auch* das Kontrollmodell mit einem Block lernt eine diagonale OV-Kopiermatrix. OV-Kopieren ohne K-Komposition mit dem QK-Schaltkreis reicht also nicht aus, um Nachfolger im Kontext zu finden (siehe Aufgabe :exref:`ex_what-attention-computes-k-composition`). (2) Block 2 kopiert zuverlässiger aus dem Residualstrom $\bar{\mathbf{h}}_a$ von Block 1 als aus den ursprünglichen Einbettungen $\mathbf{e}_a$ allein. Bei diesen schwankt der Anteil der Zeilenmaxima auf der Diagonalen von Lauf zu Lauf stark.

<!--en:ae39b506-->
### Ablation von Köpfen

<!--en:44e946a9-->
Um zu prüfen, welche Köpfe kausal notwendig sind, setzt `ablate` den Spaltenblock $\mathbf{W}_o^m$ der Ausgabeprojektion für ausgewählte Köpfe eines Blocks auf null. Wir vergleichen drei Eingriffe. In Block 1 entfernen wir entweder die Vorgängerköpfe (Score $> 0.2$) oder die übrigen Köpfe. In Block 2 entfernen wir die Induktionsköpfe nach und nach, beginnend mit dem stärksten:

<!--en:2f113c13-->
Wir wenden `ablate` auf alle drei trainierten Modelle mit zwei Blöcken an. Damit messen wir, wie stark sich Kreuzentropie und Genauigkeit bei späteren Kopien verschlechtern, wenn wir die Vorgängerköpfe aus Block 1 oder die stärksten Induktionsköpfe aus Block 2 entfernen:

<!--en:b721adcb-->
Entfernen wir die Vorgängerköpfe aus Block 1, fällt die Genauigkeit bei späteren Kopien unter $0.2$ und damit auf das Niveau des Modells mit einem Block. Das bestätigt, dass die Vorgängerköpfe notwendig sind. Interessanterweise scheitert das Retrieval auch, wenn wir die *anderen* Köpfe aus Block 1 entfernen. Da `copy_matrix` auf den von Block 1 transformierten Residualstrom angewiesen war, hängt Block 2 auch von diesen Köpfen ab. Zudem verändert das Nullsetzen in einem Netz ohne Normalisierung die Normen im Residualstrom (siehe Aufgabe :exref:`ex_what-attention-computes-circuit-ablation` zur Mittelwertablation). In Block 2 übernehmen die vier Köpfe die Induktion teilweise redundant. Erst wenn mehrere Köpfe ausfallen, sinkt die Genauigkeit auf Zufallsniveau.

<!--en:6a9930c4-->
## In-Context-Lernen

<!--en:c68500e1-->
### Generalisierung auf unbekannte Muster

<!--en:8684f784-->
Da es $64^{16} \approx 8 \times 10^{28}$ mögliche Muster gibt, ist jede Testsequenz neu. In der Trainingsverteilung sind alle Token-Paare gleich wahrscheinlich. Sobald ein Token und sein Nachfolger im Kontextfenster einmal gemeinsam aufgetreten sind, verknüpft der Induktionsschaltkreis sie miteinander. Das ist eine elementare Form des *In-Context-Lernens* (engl. in-context learning):

<!--en:52f1826f-->
Die Genauigkeit springt von $1/64$ bei der ersten Kopie (auch beim erneuten Beginn des Musters an Position $t = 24$, wo dem inhaltlichen Abgleich noch kein wiederholtes Token vorliegt) auf etwa $0.8$ an Position $25$ und wenige Positionen später auf über $0.9$.

<!--en:efbdf9fd-->
Woher kommt der verbleibende Fehler nach dem erneuten Beginn des Musters? Kommt ein Token $K > 1$-mal im Muster aus $L$ Token vor und hat es verschiedene Nachfolger, ist der Abgleich allein anhand des aktuellen Tokens mehrdeutig. Wer gleichverteilt unter den $K$ Vorkommen wählt, trifft mit Wahrscheinlichkeit $1/K$ den richtigen Nachfolger. Für $L = 24$ und $p = 1/64$ wiederholt sich ein Token im Muster mit Wahrscheinlichkeit $1 - (63/64)^{23} \approx 0.30$. Für den Abgleich einzelner Token mit gleichverteilter Auswahl ergibt sich eine erwartete Gesamtgenauigkeit von

<!--en:ba2e09f9-->
und eine Genauigkeit von $\approx 0.47$ allein für wiederholte Token. Wir teilen die Testpositionen nach $K = 1$ (einmalig) und $K > 1$ (wiederholt) auf. In den drei Läufen beträgt die Genauigkeit bei einmaligen Token mindestens $0.97$, bei wiederholten etwa $0.7$ bis $0.9$. Letzteres liegt deutlich über dem Referenzwert von $0.47$ für den Abgleich einzelner Token. Eine plausible Erklärung, die wir hier nicht überprüfen, ist der zusätzliche Abgleich längerer $n$-Gramme im Kontext zusammen mit einer Präferenz für bestimmte RoPE-Versätze (Aufgabe :exref:`ex_what-attention-computes-paths`):

<!--en:b32a8533-->
### Induktionsköpfe in größeren Modellen

<!--en:0c8550c3-->
:citet:`Olsson.Elhage.Nanda.ea.2022` zeigten, dass sich Induktionsköpfe in Sprachmodellen unterschiedlicher Größe während eines abrupten Phasenübergangs bilden. Gleichzeitig entsteht die Fähigkeit zum allgemeinen In-Context-Lernen (Aufgabe :exref:`ex_what-attention-computes-text`). Attention-Schichten können nicht nur kopieren und Muster durch unscharfen Abgleich vervollständigen. Bei Regressionsaufgaben können sie auch Gradientenschritte im Kontext ausführen (Aufgabe :exref:`ex_what-attention-computes-in-context-regression`, :cite:`Oswald.Niklasson.Randazzo.ea.2023`). Ob vortrainierte Sprachmodelle auf diese Weise im Kontext lernen, ist allerdings umstritten :cite:`Shen.Mishra.Khashabi.2024`.

<!--en:9f19e80d-->
Unser Modell kommt ohne Normalisierung und ohne MLP aus. Ein echter Transformer-Block (:numref:`chap_transformers`) enthält beides. Dadurch ändert sich die Methode, die Grundidee bleibt jedoch dieselbe. Die Schichtnormalisierung skaliert den Residualstrom anhand seiner eigenen Norm. Die Abbildung vom Residualstrom auf die Logits ist also nur dann linear, wenn der Skalierungsfaktor feststeht. Bei der Analyse von Schaltkreisen fixiert man ihn auf seinen beobachteten Wert und behandelt den Rest wie zuvor. MLPs führen zusätzliche Pfade ein, die kein Attention-Muster durchlaufen. Ein MLP kann die Ausgabe eines Kopfes umschreiben, bevor ein späterer Kopf sie liest. Schließlich arbeiten Induktionsköpfe in trainierten Sprachmodellen mit unscharfem Abgleich. Sie vervollständigen $[\mathrm{A}^*][\mathrm{B}^*] \ldots [\mathrm{A}] \to [\mathrm{B}]$ auch für Token, die den früheren nur ähneln und nicht mit ihnen identisch sind.

<!--en:2c4350f5-->
## Attention-Gewichte als Erklärungen

<!--en:7f9ed73a-->
Unsere Untersuchung zeigt, warum Attention-Heatmaps allein den Mechanismus nicht erklären können:

<!--en:1c4952b0-->
* **Wichtige Köpfe können unauffällige Attention-Karten haben:** Die Köpfe in Block 1, die keine Vorgängerköpfe sind, wirkten in den Heatmaps untätig. Entfernten wir sie, scheiterte jedoch das Retrieval in Block 2.

<!--en:74426518-->
* **Selbst stark konzentrierte Attention kann wenig bewirken:** In vortrainierten Sprachmodellen legen untätige Köpfe häufig den Großteil ihres Gewichts auf das erste Token (einen *Attention-Sink*). Die Norm seines OV-Vektors liegt nahe null :cite:`Xiao.Tian.Chen.ea.2024,Gu.Pang.Du.ea.2025`.

<!--en:c4d031e1-->
:numref:`tab_induction-evidence` fasst zusammen, auf welchen Ebenen wir die Belege für den Induktionsschaltkreis geprüft haben.

<!--en:53e70893-->
:Belege für die Deutung als Induktionskopf, nach Untersuchungsebene.

<!--en:513c843a-->
| Untersuchung in diesem Abschnitt | Untersuchungsebene | Gestützte Schlussfolgerung | Verbleibende Einschränkung |

<!--en:7bd7c8ae-->
| Verlust bei späteren Kopien | Verhalten | Zwei Blöcke lösen die Aufgabe mit wiederholten Token, ein Block allein schafft das nicht | Identifiziert keine Komponente |

<!--en:dc1997fb-->
| Änderung der Periode | Eingriff in die Eingabe | Das Modell mit fester Periode nutzt die Position statt des Inhalts | Ändert neben der Eingabe auch die Aufgabe |

<!--en:43077ecc-->
| Scores der Köpfe | Aktivierung | Die Köpfe greifen mit den vorhergesagten Versätzen zu | Erfasst die Transformation der Werte nicht |

<!--en:3bf5165d-->
| Kopiermatrix | Gewichte und mittlere Aktivierungen | Die OV-Schaltkreise von Block 2 kopieren Token | Auch in einem Modell ohne erfolgreiches Retrieval vorhanden |

<!--en:1bce30ca-->
| Ablation von Köpfen | Eingriff in Komponenten | Für die Vorhersage späterer Kopien sind ganz Block 1 einschließlich der Köpfe ohne Vorgängerfunktion und die Induktionsköpfe in Block 2 nötig | Das Nullsetzen verschiebt den Residualstrom aus seiner Trainingsverteilung |

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:063b2e90-->
Jeder Attention-Kopf wirkt über zwei Matrizen vom Rang höchstens $d_h$ auf den Residualstrom. Der **QK-Schaltkreis** $\mathbf{W}_q^\top \mathbf{W}_k$ bestimmt, auf welche Positionen er zugreift, und der **OV-Schaltkreis** $\mathbf{W}_o \mathbf{W}_v$, welchen Vektor er an die Zielposition schreibt. Eine einzige Attention-Schicht kann Bigramme und Skip-Trigramme darstellen und damit auch einzelne Token kopieren. Um neue Muster $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}] \to [\mathrm{B}]$ im Kontext zu vervollständigen, braucht es mindestens zwei Schichten. Erst dann können ein **Vorgängerkopf** und ein **Induktionskopf** zum Abgleichen und Kopieren über den Residualstrom zusammenwirken. Beim Training auf wiederholten Zufallsmustern mit variabler Periode durchläuft ein Modell mit zwei Blöcken einen abrupten Phasenübergang. Danach übersteigt die Genauigkeit im Kontext $90\%$. Bei fester Periode kann ein Modell mit einem Block hingegen eine störanfällige Abkürzung über die Position nutzen. Um herauszufinden, was ein Kopf tatsächlich berechnet, müssen wir daher die Attention-Muster und die OV-Gewichte untersuchen und die kausale Wirkung durch Ablation von Köpfen prüfen.

<!--en:2294243a-->
## Übungen

<!--en:b68fb3bc-->
1. **Bigramme mit gekoppelten Gewichten.** Ohne Blöcke ist das Logit für Token $b$ nach Token $a$ gleich $\mathbf{e}_b^\top \mathbf{e}_a$. Allein die Geometrie der Einbettungen bestimmt also, welchen Nachfolger das Modell bevorzugt. Wir nennen $b$ den *bevorzugten Nachfolger* von $a$, wenn $\mathbf{e}_b^\top \mathbf{e}_a$ das größte Logit der betreffenden Zeile ist. Nehmen Sie an, dass alle Zeilenmaxima eindeutig sind.
    1. Zeigen Sie, dass der bevorzugte Nachfolger jedes Tokens das Token selbst ist, wenn alle Einbettungen dieselbe Norm haben.
    1. Lassen Sie die Annahme gleicher Normen fallen. Zeigen Sie, dass ein Token $b \neq a$ nur dann der bevorzugte Nachfolger von $a$ sein kann, wenn $\|\mathbf{e}_b\| >
       \|\mathbf{e}_a\|$ gilt. Folgern Sie, dass die bevorzugten Nachfolger keinen Zyklus $a_1 \to a_2 \to \cdots \to a_k \to a_1$ mit $k \geq 2$ bilden können.
    1. Angenommen, in einem Korpus ist der häufigste Nachfolger des Zeichens `t` das Zeichen `h`, der von `h` ist `e`, der von `e` ist das Leerzeichen und der des Leerzeichens ist `t`. Zeigen Sie, dass kein Modell ohne Block alle vier als bevorzugte Nachfolger reproduziert, unabhängig von seiner Trainingsdauer.
    1. In einem Modell mit einem Block greift jeder Kopf an der ersten Position nur auf diese Position zu. Schreiben Sie die dortigen Logits in der Form $\mathbf{E}\mathbf{M}
       \mathbf{e}_{x_0}$ mit einer Matrix $\mathbf{M}$ aus den OV-Schaltkreisen. Erklären Sie, warum keine der beiden Einschränkungen aus den vorigen Teilaufgaben für $\mathbf{E}\mathbf{M}\mathbf{E}^\top$ gilt.
   :exlabel:`ex_what-attention-computes-tied-bigrams`

<!--en:40336399-->
1. **Pfade durch zwei Blöcke.** Halten Sie für ein Modell mit $N$ Blöcken zu je $H$ Köpfen die Attention-Muster auf einer Sequenz aus $T$ Token fest. Bezeichnen Sie mit $\mathbf{A}^{\ell m}$ die Matrix der Größe $T \times T$ mit den Einträgen $\alpha_{ij}^{\ell m}$.
    1. Entwickeln Sie die Logits aus :eqref:`eq_stream-update` für $N = 2$ zu einer Summe von Pfadtermen. Zeigen Sie, dass der Pfad durch Kopf $m$ von Block 1 und Kopf $m'$ von Block 2 wie ein einzelner Kopf mit dem Muster $\mathbf{A}^{2m'}\mathbf{A}^{1m}$ und dem OV-Schaltkreis $\mathbf{W}_{\mathrm{OV}}^{2m'}\mathbf{W}_{\mathrm{OV}}^{1m}$ wirkt. Geben Sie eine obere Schranke für den Rang dieses Schaltkreises an.
    1. Bestimmen Sie die Anzahl der Pfadterme für $N$ Blöcke und berechnen Sie diese Anzahl für `model_two`.
    1. In einem trainierten Modell sind die Muster von Block 2 nicht fest. Über den QK-Schaltkreis von Block 2 hängen sie von $\mathbf{h}^{(1)}$ ab. Setzen Sie $\mathbf{h}_i^{(1)}$ und $\mathbf{h}_j^{(1)}$ in den Score aus :eqref:`eq_qkov` ein. Ordnen Sie die Terme danach, ob die Abfrageseite, die Schlüsselseite oder beide die Ausgabe eines Kopfes aus Block 1 enthalten. Die entsprechenden Terme heißen *Q-Komposition* und *K-Komposition*. Der Pfad aus der ersten Teilaufgabe entspricht der *V-Komposition* :cite:`Elhage.Nanda.Olsson.ea.2021`. Auf welcher Art von Term beruht der Induktionsschaltkreis aus :numref:`fig_induction-circuit`?
    1. Zeigen Sie, dass zwei Blöcke das aus zwei Token bestehende Präfix $(x_{i-1}, x_i)$ abgleichen können, wenn Block 1 Köpfe mit den Versätzen $-1$ und $-2$ enthält. Benennen Sie für jede Hälfte des Abgleichs den zugehörigen Kompositionsterm.
   :exlabel:`ex_what-attention-computes-paths`

<!--en:4af21001-->
1. [code] **Die Abkürzung ohne Bias-Terme.** Das Modell mit festem Versatz aus :numref:`subsec_fixed-offset-shortcut` hat Bias-Terme in den Projektionen.
    1. Die Abfrage und der Schlüssel eines Kopfes seien vor der Rotation an jeder Position dieselben konstanten Vektoren $\mathbf{q}$ und $\mathbf{k}$. Zeigen Sie mit $\mathbf{R}_i^\top \mathbf{R}_j = \mathbf{R}_{j-i}$, dass der Score nur vom Versatz $j - i$ abhängt. Erklären Sie, wie sich $\mathbf{q}$ und $\mathbf{k}$ so wählen lassen, dass der Score beim Versatz $-31$ am größten ist.
    1. Ohne Bias-Terme ist jede Abfrage und jeder Schlüssel eine lineare Funktion des Residualstroms. Was müssen die Token-Einbettungen enthalten, damit ein Kopf nahezu konstante Abfragen und Schlüssel erzeugen kann?
    1. Trainieren Sie das Modell mit einem Block aus dem Text dreimal mit `bias=False`. Geben Sie die Genauigkeit bei der zweiten Kopie und das Attention-Gewicht jedes Kopfes 31 Positionen zurück an. Wie zuverlässig bildet sich die Abkürzung ohne Bias-Terme, und hält sie dem Test mit Periode 24 stand?
   :exlabel:`ex_what-attention-computes-shortcut-bias`

<!--en:3fe27ab2-->
1. **Gewichte ohne Auswirkung.** Die Ausgabe eines Kopfes bei der Abfrage $i$ ist $\mathbf{W}_{\mathrm{OV}} \sum_j \alpha_{ij}\, \mathbf{h}_j$ gemäß :eqref:`eq_stream-update`. Durch die Softmax-Funktion ist jedes Gewicht $\alpha_{ij}$ eines unmaskierten Schlüssels positiv. :citet:`Jain.Wallace.2019` stellten fest, dass in trainierten Textklassifikatoren selbst recht unterschiedliche Attention-Verteilungen oft dieselbe Vorhersage liefern. Daraus schlossen sie, dass Attention-Gewichte die Vorhersagen nicht erklären.
    1. Betrachten Sie alle Attention-Verteilungen über $n$ Schlüssel, die dieselbe Kopfausgabe liefern wie $\boldsymbol{\alpha}_i$. Zeigen Sie, dass diese Menge in einer Umgebung von $\boldsymbol{\alpha}_i$ ein Stück eines affinen Unterraums der Dimension mindestens $n - 1 - d_h$ enthält. Berechnen Sie die Schranke für die letzte Abfrage einer Eingabe aus 63 Token für `model_two`.
    1. Die Alternativen aus der ersten Teilaufgabe sind wie die von :citet:`Jain.Wallace.2019` jeweils für eine einzelne Eingabe konstruiert. :citet:`Wiegreffe.Pinter.2019` argumentierten, eine Alternative spreche nur dann gegen eine Erklärung, wenn ein einziges trainiertes Modell für jede Eingabe solche Alternativen erzeugt und dabei nahezu dieselben Vorhersagen trifft. Erklären Sie, warum die Dimensionsschranke aus der ersten Teilaufgabe nichts über die Existenz eines solchen Modells aussagt. Beschreiben Sie ein Experiment, das diese Frage für `model_two` entscheiden würde.

<!--en:e95788ec-->
1. [code] **Null- und Mittelwertablation.** Die Ablationen aus :numref:`subsec_head-ablation` setzen die Ausgabe eines Kopfes auf null.
    1. Erklären Sie, warum das Nullsetzen der zu einem Kopf gehörenden Spalten von `proj` genau den Beitrag dieses Kopfes zum Residualstrom entfernt. Welche Attention-Muster bleiben beim Entfernen eines Kopfes aus Block 1 unverändert, welche können sich ändern?
    1. Beim Training ist die Ausgabe eines Kopfes nie null. Das Nullsetzen versetzt den Residualstrom daher in Zustände, die das Modell noch nie gesehen hat. Die *Mittelwertablation* ersetzt die Ausgabe $\mathbf{o}_m$ von Kopf $m$ durch ihren Mittelwert $\bar{\mathbf{o}}_m$ über Positionen und Sequenzen. Implementieren Sie die Mittelwertablation. Erstellen Sie ein `TinyCharLM` mit `bias=True` und kopieren Sie die Gewichte von `model_two` hinein. Setzen Sie dabei die Bias-Terme von `qkv` auf null. Setzen Sie für Kopf $m$ die zugehörigen Spalten von `proj` auf null. Setzen Sie den Bias von `proj` auf das Produkt dieser Spalten vor dem Nullsetzen mit $\bar{\mathbf{o}}_m$. Wiederholen Sie die Ablationen aus dem Text. Scheitert das Retrieval weiterhin, wenn Sie in Block 1 die Köpfe entfernen, die keine Vorgängerköpfe sind? Welche der beiden Deutungen aus :numref:`subsec_head-ablation` stützt dieses Ergebnis?
    1. Entfernen Sie nur den zweitstärksten Induktionskopf von Block 2. Addieren Sie die Verlustanstiege bei späteren Kopien für die beiden Einzelablationen. Vergleichen Sie diese Summe mit dem Anstieg, wenn Sie beide Köpfe gemeinsam entfernen. Was sagt der Vergleich darüber aus, wie sich die Köpfe die Arbeit teilen?
    1. Welche Einschränkungen aus :numref:`tab_induction-evidence` hebt die Mittelwertablation auf, welche bleiben bestehen?
   :exlabel:`ex_what-attention-computes-circuit-ablation`

<!--en:88d071d9-->
1. [code] **K-Komposition in den Gewichten.** ● Der Abgleich im Induktionskopf gelingt, weil der QK-Schaltkreis von Block 2 liest, was der OV-Schaltkreis von Block 1 in den Residualstrom geschrieben hat.
    1. Betrachten Sie jedes Paar aus Kopf $m$ von Block 1 und Kopf $m'$ von Block 2 in `model_two`. Berechnen Sie die Matrix der Größe $64 \times 64$
       $$\mathbf{E}\, \mathbf{W}_{\mathrm{QK}}^{2m'}\, \mathbf{W}_{\mathrm{OV}}^{1m}\, \mathbf{E}^\top,$$
       Ihr Eintrag $(a, b)$ gibt den Score für ein Abfragetoken $a$ und einen Schlüssel mit Vorgänger $b$ an. Messen Sie den Anteil der Zeilen, deren größter Eintrag auf der Diagonalen liegt, analog zum Kopiertest aus :eqref:`eq_copy-matrix`.
    1. Vergleichen Sie das Ergebnis mit dem direkten Pfad $\mathbf{E}\,
       \mathbf{W}_{\mathrm{QK}}^{2m'}\, \mathbf{E}^\top$ derselben Köpfe aus Block 2. Welche Kopfpaare leisten den Abgleich? Stimmen sie mit den auf Attention beruhenden Scores von `head_scores` überein?
    1. Die Berechnung lässt die Rotation außer Acht, die RoPE innerhalb des QK-Schaltkreises ausführt. Beziehen Sie für einen festen Versatz die relative Rotation $\mathbf{R}_{j-i}$ ein. Prüfen Sie, ob sich dadurch beim Versatz $-23$ die Rangfolge der Paare ändert.
   :exlabel:`ex_what-attention-computes-k-composition`

<!--en:d7c763f0-->
1. [code] **Ein Kopf, der einen Gradientenschritt ausführt.** ● Induktionsköpfe lernen durch Kopieren aus dem Kontext. Ein Modell kann auch aus Beispielen mit Label im Kontext eine Regel ableiten. Eine Sequenz enthalte Paare $(\mathbf{x}_i,
   y_i)$, $i = 1, \ldots, n$, mit standardnormalverteiltem $\mathbf{x}_i \in
   \mathbb{R}^p$ und $y_i = \mathbf{w}^\top \mathbf{x}_i$ für einen Vektor $\mathbf{w}$, den wir für jede Sequenz neu ziehen. Darauf folge eine Abfrage $\mathbf{x}_q$, deren Label vorherzusagen ist.
    1. Zeigen Sie, dass ein Schritt des Gradientenabstiegs auf $\frac{1}{2}\sum_i (y_i -
       \mathbf{w}^\top \mathbf{x}_i)^2$ mit Startwert $\mathbf{w} =
       \mathbf{0}$ und Lernrate $\eta$ die Vorhersage $\hat{y}_q = \eta
       \sum_i y_i\, \mathbf{x}_i^\top \mathbf{x}_q$ liefert. Deuten Sie dies als Attention ohne Softmax-Funktion. Was übernimmt die Rolle der Abfrage, der Schlüssel und der Werte? Vergleichen Sie dies mit der linearen Attention aus :numref:`sec_windowed-linear-attention` mit der Merkmalsabbildung $\phi(\mathbf{x}) = \mathbf{x}$.
    1. Ordnen Sie jedem Beispiel das Token $\mathbf{z}_i = [\mathbf{x}_i; y_i] \in
       \mathbb{R}^{p+1}$ und der Abfrage das Token $\mathbf{z}_q =
       [\mathbf{x}_q; 0]$ zu. Wählen Sie die Abbildungen für Abfragen, Schlüssel und Werte eines Kopfes ohne Softmax-Funktion so, dass seine Ausgabe am Abfragetoken gleich $\hat{y}_q$ ist. Warum spielt es keine Rolle, dass das Abfragetoken auch auf sich selbst zugreift? Implementieren Sie den Kopf und vergleichen Sie seine Ausgabe mit dem Ergebnis des Gradientenschritts. Hängen die Gewichte des Kopfes von $\mathbf{w}$ ab?
    1. Zeigen Sie für $\eta = 1/n$, dass der Schritt $\mathbf{w}_1 =
       \big(\frac{1}{n}\sum_i \mathbf{x}_i \mathbf{x}_i^\top\big)
       \mathbf{w}$ ergibt und dass $\mathbb{E}\,\|\mathbf{w}_1 - \mathbf{w}\|^2 =
       \frac{p+1}{n}\|\mathbf{w}\|^2$ gilt. Hinweis: Für standardnormalverteiltes $\mathbf{x}$ gilt $\mathbb{E}[(\mathbf{x}^\top \mathbf{w})^2
       \|\mathbf{x}\|^2] = (p+2)\|\mathbf{w}\|^2$. Folgern Sie, dass der erwartete quadratische Fehler von $\hat{y}_q$ bezogen auf die Varianz von $y_q$ gleich $(p+1)/n$ ist.
    1. Messen Sie für $p = 8$ den relativen Fehler über 500 Sequenzen bei $n =
       10$, 100 und 1000. Vergleichen Sie ihn mit der Vorhersage $9/n$. Wie schneidet der Kopf bei $n =
       10$ im Vergleich zur konstanten Vorhersage null ab?
   :exlabel:`ex_what-attention-computes-in-context-regression`

<!--en:dabe3f69-->
1. [extended] **Induktionsköpfe in Texten.** ●● :citet:`Olsson.Elhage.Nanda.ea.2022` bringen die Entstehung von Induktionsköpfen in Sprachmodellen mit einem abrupten Zugewinn beim In-Context-Lernen in Verbindung. Sie messen ihn als Rückgang des Verlusts von frühen zu späten Positionen im Kontext. Prüfen Sie dies am Modell auf Byte-Ebene aus :numref:`subsec_pos-experiment`.
    1. Trainieren Sie `d2l.TinyCharLM` mit `pos='rope'` auf den WikiText-103-Bytes aus :numref:`subsec_pos-experiment` in 30 Runden zu je 100 Schritten mit `d2l.train_steps`. Halten Sie nach jeder Runde zwei Größen fest. Die erste ist der höchste Score eines Kopfes in Block 2 für das Induktionsziel. Berechnen Sie ihn mit `head_scores` auf wiederholten Zufallssequenzen aus dem Textvokabular mit Musterlängen von 16 bis 32. Die zweite Größe ist der Zugewinn durch den Kontext: der Validierungsverlust an den Zielpositionen 0 bis 15 abzüglich des Verlusts an den Positionen 96 bis 127.
    1. Tragen Sie beide Größen gegen den Trainingsschritt auf. Steigen der Induktions-Score und der Zugewinn durch den Kontext im selben Zeitfenster?
    1. Entfernen Sie nach dem Training die Köpfe aus Block 2 mit den höchsten Induktions-Scores mithilfe von `ablate`. Messen Sie den Zugewinn durch den Kontext und den Validierungsverlust erneut. Wie viel tragen diese Köpfe zum Zugewinn bei?
    1. Die wiederholten Zufallssequenzen weisen außer der Wiederholung keine Struktur auf. In Texten dagegen wiederholen sich Wörter und Wortgruppen in unregelmäßigen Abständen. Welche Ihrer Ergebnisse würden sich ändern, wenn der Text keine wiederholten Wörter enthielte, und warum?
   :exlabel:`ex_what-attention-computes-text`

<!--en:03980222-->
[Diskussion](https://d2l.discourse.group/t/36773)

<!--en:9e19b826-->
Attention-Schaltkreise: Residualstrom, QK/OV-Zerlegung und Induktionsköpfe<br>**Residualstrom · QK- und OV-Schaltkreise · Skip-Trigramme · zweischichtige Induktionsköpfe · kausale Ablation von Köpfen · In-Context-Lernen**

<!--en:1bfbdb4d-->
::: {.slide title="Warum eine Attention-Heatmap allein einen Kopf nicht erklärt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:fb9908ea-->
- **Eine Abkürzung über die Position kann Retrieval nachahmen:** Ein Modell mit einem Block lernt auf Sequenzen aus 64 Token, in denen sich ein Zufallsmuster aus 32 Token wiederholt. Bei der zweiten Kopie erreicht es **$100\%$ Genauigkeit**, indem es $31$ Positionen zurückblickt.

<!--en:10bc21cc-->
- **Ein Test mit Periode 24 deckt die Abkürzung auf:** Bei Mustern aus 24 Token fällt die Genauigkeit auf **$1.6\%$ (Zufallsniveau)**, weil die Köpfe weiterhin starr auf den Versatz $-31$ zugreifen.

<!--en:52559156-->
::: {.slide title="Die Ausgabe der Multi-Head-Attention als Summe über die Köpfe"}

<!--en:04d93862-->
[Additive Zerlegung nach Köpfen]{.kicker}

<!--en:fb205b2d-->
- Die Zerlegung von $\mathbf{W}_o \in \mathbb{R}^{d \times H d_h}$ in $H$ Spaltenblöcke $\mathbf{W}_o^m \in \mathbb{R}^{d \times d_h}$ ergibt für die Multi-Head-Attention eine Summe aus $H$ unabhängigen Aktualisierungen:
  $$\sum_{m=1}^H \mathbf{W}_o^m \mathbf{u}_m = \sum_{m=1}^H \sum_{j \le i} \alpha_{ij}^m\,\big(\mathbf{W}_o^m \mathbf{W}_v^m\big)\,\mathbf{h}_j$$

<!--en:fe47a2e6-->
- **Nur das Produkt $\mathbf{W}_o^m \mathbf{W}_v^m$ zählt:** Die Wertprojektion $\mathbf{W}_v^m$ und der zugehörige Spaltenblock $\mathbf{W}_o^m$ der Ausgabeprojektion wirken ausschließlich über ihr Produkt vom Rang $d_h$.

<!--en:2cc4a47a-->
::: {.slide title="Zwei Matrizen vom Rang d_h beschreiben jeden Kopf: QK und OV"}

<!--en:8475a0cd-->
[QK- und OV-Schaltkreise]{.kicker}

<!--en:4a6019dd-->
- Zwei Matrizen beschreiben jeden Attention-Kopf der Breite $d_h$ vollständig. Beide haben die Größe $d \times d$ und den Rang $\le d_h$ (Elhage et al., 2021):
  $$\mathbf{W}_{\mathrm{QK}} = \mathbf{W}_q^\top \mathbf{W}_k, \qquad \mathbf{W}_{\mathrm{OV}} = \mathbf{W}_o \mathbf{W}_v$$

<!--en:5eeec445-->
- **QK-Schaltkreis ($\mathbf{W}_{\mathrm{QK}}$):** Bilinearform, die bestimmt, *wohin* der Kopf seine Attention richtet ($\alpha_{ij}$).

<!--en:8c094dee-->
- **OV-Schaltkreis ($\mathbf{W}_{\mathrm{OV}}$):** Linearer Operator, der bestimmt, *welcher* Vektor $\mathbf{W}_{\mathrm{OV}}\mathbf{h}_j$ an der Zielposition $i$ addiert wird.

<!--en:f02695e8-->
::: {.slide title="Der Residualstrom als gemeinsamer Kommunikationsbus"}

<!--en:50f47a36-->
[Der Residualstrom]{.kicker}

<!--en:40e02f78-->
- Ausgehend von der Token-Einbettung $\mathbf{h}_i^{(0)} = \mathbf{e}_{x_i}$ addiert jeder Block $\ell = 1, \ldots, N$ die Ausgaben seiner Köpfe direkt zum bisherigen Vektor an Position $i$:
  $$\mathbf{h}_i^{(\ell)} = \mathbf{h}_i^{(\ell-1)} + \sum_{m=1}^H \sum_{j \le i} \alpha_{ij}^{\ell m}\,\mathbf{W}_{\mathrm{OV}}^{\ell m}\,\mathbf{h}_j^{(\ell-1)}$$

<!--en:27aba8c1-->
- **Gemeinsamer Lese- und Schreibkanal:** Über seine QK- und OV-Schaltkreise liest jeder Kopf aus Unterräumen von $\mathbf{h}^{(\ell-1)}$ und addiert sein Ergebnis zu $\mathbf{h}^{(\ell)}$. Am Ende liefert die gekoppelte Ausgabeprojektion die Logits $\mathbf{z}_i = \mathbf{E}\,\mathbf{h}_i^{(N)}$.

<!--en:19d34c3d-->
![Jeder Kopf liest über seinen QK-Schaltkreis aus früheren Residualströmen und addiert über seinen OV-Schaltkreis seinen Beitrag.](../img/mdl-attention-residual-stream.svg){.nostretch width=100%}

<!--en:d2c382f2-->
::: {.slide title="Was Modelle mit null oder einer Attention-Schicht darstellen können"}

<!--en:f7765c9c-->
[Ausdrucksstärke in Abhängigkeit von der Tiefe]{.kicker}

<!--en:9453bd78-->
- Die Entwicklung des Residualstroms für $N = 1$ liefert einen Bigrammterm ohne Schicht und Skip-Trigrammterme mit einer Schicht:
  $$\mathbf{z}_i = \mathbf{E}\,\mathbf{e}_{x_i} + \sum_{m=1}^H \sum_{j \le i} \alpha_{ij}^m\,\mathbf{E}\,\mathbf{W}_{\mathrm{OV}}^m\,\mathbf{e}_{x_j}$$

<!--en:16e7e78b-->
- **Kein Block ($N = 0$):** Die symmetrische Tabelle $\mathbf{E}\mathbf{E}^\top$ kann keine asymmetrischen Zyklen darstellen.

<!--en:059fe3a8-->
- **Ein Block ($N = 1$):** Skip-Trigramme $x_j \ldots x_i \to x_{\textrm{next}}$ sind möglich. Neue Muster $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}] \to [\mathrm{B}]$ kann der Block **nicht** vervollständigen, weil der Schlüssel von $[\mathrm{B}]$ nichts über seinen Vorgänger $[\mathrm{A}]$ weiß.

<!--en:cf3141ad-->
::: {.slide title="Zwei Schichten bilden durch K-Komposition einen Induktionskopf"}

<!--en:b7379725-->
[Der zweischichtige Induktionsschaltkreis]{.kicker}

<!--en:d1b9888c-->
- Die Vervollständigung $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}]$ $\to$ $[\mathrm{B}]$ erfordert **zwei über den Schlüsselpfad verknüpfte Schichten (K-Komposition)**:

<!--en:cae5455d-->
- **Schicht 1 (Vorgängerkopf):** Mithilfe von RoPE bevorzugt der Kopf einen Versatz von $1$ Schritt zurück ($j - 1$). Er schreibt $\mathbf{W}_{\mathrm{OV}}^{1}\,\mathbf{e}_{[\mathrm{A}]}$ in den Residualstrom $\mathbf{h}_j^{(1)}$ von $[\mathrm{B}]$ an Position $j$.

<!--en:e2490500-->
- **Schicht 2 (Induktionskopf):** Der QK-Schaltkreis gleicht die aktuelle Abfrage $\mathbf{e}_{[\mathrm{A}]}$ mit dem zuvor geschriebenen Schlüsselmerkmal in $\mathbf{h}_j^{(1)}$ ab ($\mathbf{e}_{[\mathrm{A}]}^\top \mathbf{W}_{\mathrm{QK}}^{2}\mathbf{W}_{\mathrm{OV}}^{1}\mathbf{e}_{[\mathrm{A}]}$). Der OV-Schaltkreis kopiert $x_j = [\mathrm{B}]$ in die Ausgabe-Logits.

<!--en:aa649f97-->
![Ein zweischichtiger Induktionsschaltkreis verknüpft einen Vorgängerkopf in Schicht 1 mit einem Kopf zum Abgleichen und Kopieren in Schicht 2.](../img/mdl-attention-induction-circuit.svg){.nostretch width=100%}

<!--en:5331eb5f-->
::: {.slide title="Synthetischer Benchmark mit wiederholten zufälligen Token-Mustern"}

<!--en:49023aac-->
[Retrieval im Kontext isoliert betrachtet]{.kicker}

<!--en:2a268c83-->
Findet der Gradientenabstieg diesen Schaltkreis? In jeder Sequenz wiederholt sich ein zufälliges Muster der Länge $L \in [L_{\min}, L_{\max}]$ bis zur Gesamtlänge von $64$ Token:

<!--en:9f9025b6-->
- **Unvorhersagbare erste Kopie ($t \le L$):** Jedes Muster ist neu. Daher bleibt die Vorhersage in der ersten Periode auf Zufallsniveau (Kreuzentropie $\log 64 \approx 4.16\text{ nats}$, Genauigkeit $1/64 \approx 1.6\%$). Das gilt auch für das erste Token der nächsten Periode bei $t = L$, da sich das Muster noch nicht wiederholt hat.

<!--en:2fcd1fab-->
- **Vorhersagbare spätere Kopien ($t > L$):** Sobald sich das Muster wiederholt, lässt sich jedes Token vorhersagen, indem man sein früheres Vorkommen im Kontext nachschlägt.

<!--en:8bda1cd7-->
::: {.slide title="Eine mögliche Fehlerquelle: die Abkürzung über einen festen Versatz"}

<!--en:03853197-->
[Warum feste Perioden in die Irre führen]{.kicker}

<!--en:cf794029-->
Was lernt ein Modell mit **einem Block** (`num_blks=1, bias=True`) beim Training auf Sequenzen mit einer *festen* Musterperiode $L = 32$?

<!--en:0f4800bc-->
**Fehlerquelle bei der Deutung:** Bei $L = 32$ erreicht das Modell mit einem Block eine Genauigkeit von $1.00$ (Verlust etwa $0.1\text{ nats}$). Unter RoPE greift es mit konstanten, durch den Bias bestimmten Abfragen und Schlüsseln ausschließlich auf den festen Versatz $j - i = -31$ zu. Beim Test mit Periode $L = 24$ (Ziel beim Versatz $-23$) liegt weiterhin mehr als $0.9$ der Attention-Masse seiner Köpfe beim Versatz $-31$. Die Genauigkeit fällt auf $0.02$ und damit auf Zufallsniveau.

<!--en:3fc4e0a1-->
::: {.slide title="Variable Perioden erzwingen inhaltsbasierte Induktion"}

<!--en:6b1f0238-->
[Experiment mit einem und zwei Blöcken]{.kicker}

<!--en:83be1b7a-->
Eine für jede Sequenz gleichverteilt aus $L \in \{16, \ldots, 32\}$ gezogene Periode verhindert jede Abkürzung über einen festen Versatz. Drei Modelle mit zwei Blöcken und ein Kontrollmodell mit einem Block (`bias=False`) durchlaufen je $5{,}000$ Adam-Schritte:

<!--en:cb4da30b-->
- **Mit zwei Blöcken gelingt das Retrieval:** In allen drei Läufen mit unterschiedlichen Startwerten fällt der Verlust unter $1\text{ nat}$, und zwar zwischen Schritt $1{,}000$ und Schritt $3{,}000$. Er sinkt auf etwa $0.2$ bis $0.4\text{ nats}$, die Genauigkeit bei späteren Kopien steigt auf mehr als $90\%$.

<!--en:ab685f42-->
- **Ein Block reicht nicht aus:** Der Verlust des Kontrollmodells stagniert oberhalb von $3.5\text{ nats}$ (etwa $10\%$ Genauigkeit). Es kann lediglich die Logits der Unigramme erhöhen, die irgendwo im Präfix vorkamen.

<!--en:035588c3-->
::: {.slide title="Phasenübergang bei der Bildung von Induktionsköpfen"}

<!--en:e7406901-->
[Trainingsdynamik]{.kicker}

<!--en:3476cd74-->
- **Abrupter Rückgang des Verlusts:** Weit über tausend Schritte bleibt das Modell mit zwei Blöcken auf einem Plateau nahe $4\text{ nats}$, knapp unter dem Verlust bei zufälligem Raten. Sobald die OV-Ausgabe von Block 1 zum QK-Schaltkreis von Block 2 passt, fällt der Verlust bei späteren Kopien um mehr als $3\text{ nats}$.

<!--en:3fccc64c-->
- **Nebeneffekt bei der ersten Kopie:** Genau am Phasenübergang *steigt* der Verlust bei der ersten Kopie leicht über $\log 64 \approx 4.16$. Sobald die Induktion einsetzt, sagt das Modell bei einer zufälligen Wiederholung innerhalb der ersten Periode den früheren Nachfolger des Tokens vorher.

<!--en:d6b70a45-->
::: {.slide title="Ebene 1: Untersuchung der Attention-Heatmaps"}

<!--en:a33ee4fe-->
[Aktivierungsmuster]{.kicker}

<!--en:c2c76712-->
- Ein Test von `model_two` mit einer Sequenz der Periode $L = 24$ zeigt zwei unterschiedliche geometrische Muster in den beiden Blöcken:

<!--en:7ece2c6d-->
- **Block 1 (obere Reihe):** Mehrere Köpfe bilden bei $j = i - 1$ einen Streifen direkt unterhalb der Hauptdiagonalen (**Vorgängerköpfe**).

<!--en:92758ec7-->
- **Block 2 (untere Reihe):** Sobald bei $i \ge 24$ die zweite Kopie beginnt, bilden alle vier Köpfe bei $j = i + 1 - L = i - 23$ einen zur Diagonalen parallelen Streifen (**Induktionsköpfe**).

<!--en:4e7cc7bd-->
::: {.slide title="Scores für Vorgänger- und Induktionsköpfe" layout="tall-output"}

<!--en:d8f4914a-->
[Gemittelte Attention-Kennzahlen]{.kicker}

<!--en:0db6a195-->
`head_scores` mittelt für jeden Kopf die Attention-Gewichte auf dem vorherigen Token ($j = i - 1$) und dem variablen Induktionsziel ($j = i + 1 - L$ für $i \ge L$) über $256$ Sequenzen mit $L \in \{16, \ldots, 32\}$:

<!--en:88a56f13-->
- **Klare Arbeitsteilung:** Block 1 hat in jedem Lauf $1$ bis $3$ Vorgängerköpfe (Scores von etwa $0.3$ bis $0.99$). Der Induktions-Score beträgt dort höchstens $0.02$. In Block 2 konzentrieren die Köpfe bis zu $\approx 0.6$ ihrer Attention-Masse direkt auf das wechselnde Ziel $i + 1 - L$ (bei gleichverteilter Attention $\approx 0.025$).

<!--en:5d95262c-->
::: {.slide title="Ebene 2: Analyse der OV-Kopiermatrix"}

<!--en:2c4e4ec6-->
[Schaltkreisanalyse im Gewichtsraum]{.kicker}

<!--en:c01b2198-->
- Der Zugriff auf Position $j$ sagt Token $x_j$ nur vorher, wenn der OV-Schaltkreis von Block 2 das Logit von $x_j$ erhöht:
  $$C_{ab} \stackrel{\textrm{def}}{=} \mathbf{e}_b^\top \mathbf{W}_{\mathrm{OV}}\,\bar{\mathbf{h}}_a$$

<!--en:e96216e6-->
- **Kennzeichen des Kopierens:** Kopiert der OV-Schaltkreis die Identität des Tokens, liegt der größte Eintrag der Zeile $a$ auf der Diagonalen ($b = a$). Das gilt für fast $100\%$ der $64$ Token (gegenüber $1/64 \approx 0.02$ bei Zufallsinitialisierung).

<!--en:586f71ce-->
::: {.slide title="Die OV-Kopiermatrix und zwei wichtige Einzelheiten" layout="tight"}

<!--en:d10a7fc2-->
[Ergebnisse der Gewichtsanalyse]{.kicker}

<!--en:f9c0b043-->
- **1. Residualstrom und Einbettungen:** Die OV-Matrix von Block 2 kopiert fast perfekt (Diagonalenanteil $0.9$ bis $1.0$), wenn sie den Residualstrom $\bar{\mathbf{h}}_a$ von Block 1 liest. Weniger zuverlässig (Anteil zwischen $0.3$ und fast $1.0$) kopiert sie die ursprünglichen Einbettungen $\mathbf{e}_a$. Block 2 liest also auch Merkmale, die die OV-Schaltkreise von Block 1 transformiert haben (**V-Komposition**).

<!--en:56a41d84-->
- **2. Kopieren $\neq$ Retrieval:** *Auch* das Modell mit einem Block hat eine fast perfekte diagonale OV-Kopiermatrix (Anteil $\approx 1.00$). Dennoch erreicht es nur $10\%$ Genauigkeit, weil sein QK-Schaltkreis das Ziel nicht finden kann.

<!--en:11ee6dbd-->
::: {.slide title="Ebene 3: Kausale Ablation von Köpfen"}

<!--en:698dec90-->
[Eingriffe in einzelne Komponenten]{.kicker}

<!--en:550489bb-->
Korrelationen in Attention-Karten oder Gewichten belegen keine Kausalität. Um zu prüfen, welche Köpfe kausal notwendig sind, setzt `ablate` für ausgewählte Köpfe den Spaltenblock $\mathbf{W}_o^m$ der Ausgabeprojektion auf null:

<!--en:dbe88289-->
- **Gezielte Entfernung eines Summanden:** Block $\ell$ addiert $\sum_{m=1}^H \mathbf{W}_o^m \mathbf{u}_m$ zum Residualstrom. Das Nullsetzen von $\mathbf{W}_o^m$ entfernt genau den Beitrag von Kopf $m$. Die übrigen Köpfe des Blocks bleiben intakt.

<!--en:1bf71c52-->
- **Versuchsaufbau:** In Block 1 entfallen entweder die Vorgängerköpfe (Score $> 0.2$) oder alle *anderen* Köpfe. In Block 2 entfallen schrittweise die jeweils $k \in \{1, 2, 3, 4\}$ stärksten Induktionsköpfe.

<!--en:dbf62909-->
::: {.slide title="Was die Ablation über Notwendigkeit und Redundanz zeigt" layout="tall-output"}

<!--en:9396f57c-->
[Deutung der Eingriffe]{.kicker}

<!--en:4e79ed2a-->
- **1. Block 1 braucht die Vorgängerköpfe:** Ohne sie fällt die Genauigkeit von über $0.9$ auf unter $0.2$.

<!--en:ec1d9ea7-->
- **2. Überraschenderweise braucht Block 1 auch die „untätigen“ Köpfe:** Ohne sie sinkt die Genauigkeit unter $0.15$. Der OV-Kopierschaltkreis von Block 2 ist auf ihre Transformation des Residualstroms angewiesen. Zudem verschiebt das Nullsetzen die Normen des Stroms aus ihrer Trainingsverteilung.

<!--en:7e284b42-->
- **3. Redundanz in Block 2:** Ohne den stärksten Induktionskopf bleibt die Genauigkeit bei $0.7$ bis $0.9$. Alle vier Köpfe teilen sich die Induktion. Erst wenn alle vier wegfallen, sinkt die Genauigkeit auf Zufallsniveau.

<!--en:feb78033-->
::: {.slide title="In-Context-Lernen und Generalisierung auf unbekannte Muster"}

<!--en:aa006219-->
[Mustervervollständigung nach einmaliger Beobachtung]{.kicker}

<!--en:0f3c7333-->
- Bei $64^{24} \approx 2.2 \times 10^{43}$ möglichen Mustern der Länge $24$ hat das Modell keines der Testmuster im Training gesehen. Alle Token-Übergänge sind a priori gleich wahrscheinlich.

<!--en:33c9b25f-->
- **Erste Kopie ($t < 24$):** Die Genauigkeit liegt auf dem Zufallsniveau von $1/64 \approx 0.016$.

<!--en:d253f70d-->
- **Zweite Kopie ($t \ge 25$):** Die Genauigkeit steigt nach *einem einzigen* Vorkommen im Kontext sofort auf etwa $0.8$ und binnen weniger Positionen auf über $0.90$. (Genau beim erneuten Beginn an Position $t = 24$ liegt für den inhaltlichen Abgleich noch kein wiederholtes Token vor.)

<!--en:51737fdf-->
::: {.slide title="Wiederholte Token als Ursache des verbleibenden Fehlers"}

<!--en:2362901c-->
[Über den Abgleich einzelner Token hinaus]{.kicker}

<!--en:1e62e7eb-->
Warum bleibt die Genauigkeit nach dem erneuten Beginn des Musters einige Prozentpunkte unter $100\%$? Kommt ein Token $K > 1$-mal im Muster aus $L$ Token vor und hat es verschiedene Nachfolger, ist der Abgleich allein anhand des aktuellen Tokens mehrdeutig:

<!--en:a5c9ca9c-->
Für einmalige Token im Muster ($K = 1$, $70\%$ der Positionen) erreicht `model_two` eine **Genauigkeit von $0.97$ bis $1.00$**, für wiederholte Token ($K > 1$, $30\%$ der Positionen) etwa **$0.7$ bis $0.9$**. Damit übertrifft das Modell die Genauigkeit von $0.47$ beim gleichverteilten Abgleich einzelner Token. Es nutzt dazu Präfixe aus mehreren Token und die von RoPE bevorzugten Abstände.

<!--en:798a8699-->
::: {.slide title="Gradientenabstieg im Kontext durch lineare Attention"}

<!--en:164612c3-->
[Über das Kopieren hinaus]{.kicker}

<!--en:799acb9d-->
- Ein einzelner Kopf mit linearer Attention kann für eine Regressionsaufgabe $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ im Kontext auch **einen Schritt des Gradientenabstiegs** ausführen (Oswald et al., 2023):
  $$\mathbf{w}_1 = \eta \sum_{i=1}^n y_i\,\mathbf{x}_i \;\Longrightarrow\; \hat{y}_q = \sum_{i=1}^n (\mathbf{x}_q^\top \mathbf{x}_i)\,(\eta y_i)$$

<!--en:1beea685-->
- Mit den Beispielen in der Form $\mathbf{z}_i = [\mathbf{x}_i; y_i]$ und der Abfrage in der Form $\mathbf{z}_q = [\mathbf{x}_q; 0]$ ergibt sich $\hat{y}_q$ in einem einzigen Vorwärtsdurchlauf ohne Aktualisierung der Modellgewichte.

<!--en:3dc26334-->
::: {.slide title="Ebenen der mechanistischen Beweisführung"}

<!--en:2f9a0648-->
[Methoden zur Interpretation]{.kicker}

<!--en:ef6d62ff-->
Die Untersuchung der Induktionsköpfe zeigt, warum Belege auf mehreren Ebenen nötig sind, um festzustellen, was ein Schaltkreis berechnet:

<!--en:2afb821a-->
| Untersuchung | Untersuchungsebene | Was sie zeigt | Was offenbleibt |

<!--en:165f6123-->
| **1. Verlust bei späteren Kopien** | Verhalten | Zwei Blöcke lösen die Wiederholungsaufgabe mit variabler Periode, ein Block allein schafft das nicht | Identifiziert weder interne Komponente noch Algorithmus |

<!--en:198df1a6-->
| **2. Periodenwechsel ($32$ $\to$ $24$)** | Eingriff in die Eingabe | Deckt beim Modell mit einem Block die Abkürzung über den festen Versatz $-31$ auf | Verändert zugleich die Eingabeverteilung |

<!--en:ece3421d-->
| **3. Scores der Attention-Köpfe** | Untersuchung der Aktivierungen | Block 1 greift auf $i-1$ zu, Block 2 auf $i+1-L$ | Zeigt, *wohin* die Köpfe blicken, lässt aber offen, *was* ihre Werte übertragen |

<!--en:d67fa1b7-->
| **4. OV-Kopiermatrix ($\mathbf{C}$)** | Gewichte und Residualstrom | Der OV-Schaltkreis von Block 2 erhöht das Logit des angesteuerten Tokens | Auch im erfolglosen Kontrollmodell mit einem Block diagonal ($\approx 1.00$) |

<!--en:3fbcb68d-->
| **5. Null- und Mittelwertablation** | Kausaler Eingriff | Beide Gruppen in Block 1 und die Köpfe von Block 2 sind kausal notwendig | Nullablation verschiebt die Normen im Residualstrom (durch Mittelwertablation überprüfen) |

<!--en:cb69810b-->
::: {.slide title="Verständnisfragen zu Schaltkreisen, Komposition und Attention-Sinks"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:46d623a0-->
Verständnis der QK- und OV-Schaltkreise und der Eingriffe in ihre Funktionsweise prüfen:

<!--en:e0190e48-->
1. **Q-, K- und V-Komposition:** Ein Kopf in Block 2 berechnet den Score $\mathbf{h}_i^{(1)\top}\mathbf{W}_{\mathrm{QK}}^{2}\mathbf{h}_j^{(1)}$. Warum braucht der Induktionsschaltkreis die **K-Komposition** ($\mathbf{e}_{x_i}^\top \mathbf{W}_{\mathrm{QK}}^{2} \mathbf{W}_{\mathrm{OV}}^{1} \mathbf{e}_{x_{j-1}}$) statt der Q-Komposition?

<!--en:be054e6e-->
2. **Attention-Sinks:** In vortrainierten LLMs legen viele Köpfe für fast alle Abfragen $> 80\%$ ihres Attention-Gewichts auf das allererste Token (`<bos>`). Welche beiden Größen gilt es zu messen, bevor sich behaupten lässt, `<bos>` trage entscheidende semantische Information?

<!--en:80a9ac2a-->
3. **Bigrammzyklen ohne Block:** Warum kann ein Modell ohne Block mit gekoppelten Gewichten für Einbettung und Ausgabe und den Logits $\mathbf{z}_i = \mathbf{E}\,\mathbf{e}_{x_i}$ niemals für jeden Übergang `t` $\to$ `h` $\to$ `e` $\to$ `␣` $\to$ `t` das jeweils folgende Token als wahrscheinlichsten Nachfolger vorhersagen?

<!--en:557b98b6-->
**Lösungen:**

<!--en:cb459024-->
1. Die Abfrage an der Zielposition $i$ ist das aktuelle Token $x_i = [\mathrm{A}]$ selbst ($\mathbf{e}_{x_i}$). Der *Schlüssel* an der Quellposition $j$ ($x_j = [\mathrm{B}]$) muss dagegen die OV-Ausgabe von Block 1 enthalten, die den Vorgänger $x_{j-1} = [\mathrm{A}]$ kodiert.

<!--en:5f082b48-->
2. (1) Die Norm des OV-Ausgabevektors $\|\mathbf{W}_{\mathrm{OV}}\mathbf{h}_{\langle\mathrm{bos}\rangle}\|$. Bei einem **Attention-Sink**, der ungenutzte Wahrscheinlichkeitsmasse der Softmax-Funktion aufnimmt, liegt sie nahe null. (2) Die Änderung des Verlusts bei Ablation des Kopfes.

<!--en:03759624-->
3. Damit $b \neq a$ das Logit von $a$ für sich selbst, also $\mathbf{e}_a^\top\mathbf{e}_a$, übertrifft, muss nach der Cauchy-Schwarz-Ungleichung $\|\mathbf{e}_b\| > \|\mathbf{e}_a\|$ gelten. In einem geschlossenen Zyklus $\|\mathbf{e}_{\texttt{t}}\| < \|\mathbf{e}_{\texttt{h}}\| < \|\mathbf{e}_{\texttt{e}}\| < \|\mathbf{e}_{\texttt{␣}}\| < \|\mathbf{e}_{\texttt{t}}\|$ ist das unmöglich.

<!--en:983eec02-->
::: {.slide title="Zusammenfassung: Attention-Schaltkreise und Induktionsköpfe"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:aeaee6e3-->
| Konzept oder Schaltkreis | Mathematischer Ausdruck | Rang oder erforderliche Tiefe | Funktion in Sequenzmodellen |

<!--en:1e60822f-->
| **QK-Schaltkreis** | $\mathbf{W}_{\mathrm{QK}} = \mathbf{W}_q^\top \mathbf{W}_k$ (oder $\mathbf{W}_q^\top \mathbf{R}_{j-i}\mathbf{W}_k$) | Rang $\le d_h$ ($d \times d$) | Bestimmt, *wohin* ein Kopf seine Attention richtet ($\alpha_{ij}$) |

<!--en:fdced10b-->
| **OV-Schaltkreis** | $\mathbf{W}_{\mathrm{OV}} = \mathbf{W}_o \mathbf{W}_v$ | Rang $\le d_h$ ($d \times d$) | Bestimmt, *welcher* Vektor $\mathbf{W}_{\mathrm{OV}}\mathbf{h}_j$ an die Zielposition $i$ geschrieben wird |

<!--en:c27e0c05-->
| **Residualstrom** | $\mathbf{h}_i^{(\ell)} = \mathbf{h}_i^{(\ell-1)} + \sum_{m, j \le i} \alpha_{ij}^{\ell m}\mathbf{W}_{\mathrm{OV}}^{\ell m}\mathbf{h}_j^{(\ell-1)}$ | Gemeinsamer Vektorbus der Breite $d$ | Spätere Schichten können die Ausgaben früherer Köpfe lesen und verknüpfen |

<!--en:de8c0681-->
| **Skip-Trigramm** | $\mathbf{E}\,\mathbf{W}_{\mathrm{OV}}^m\,\mathbf{e}_{x_j}$ gewichtet mit $\alpha_{ij}^m$ | $N = 1$ Attention-Schicht | Realisiert $x_j \ldots x_i \to x_{\textrm{next}}$ und das Kopieren einzelner Token |

<!--en:e6cb6b33-->
| **Induktionsschaltkreis** | K-Komposition $\mathbf{e}_{x_i}^\top \mathbf{W}_{\mathrm{QK}}^{2m'} \mathbf{W}_{\mathrm{OV}}^{1m} \mathbf{e}_{x_{j-1}}$ | **$N \ge 2$ Attention-Schichten** | Vervollständigt neue Muster im Kontext: $[\mathrm{A}][\mathrm{B}] \ldots [\mathrm{A}]$ $\to$ $[\mathrm{B}]$ |

<!--en:e0c67418-->
| **Kopiermatrix und Ablation** | $C_{ab} = \mathbf{e}_b^\top \mathbf{W}_{\mathrm{OV}}\bar{\mathbf{h}}_a$, Null- oder Mittelwertablation über $\mathbf{W}_o^m$ | Gewichte und Eingriff | Unterscheidet kausal wirksame Schaltkreise von Attention-Sinks und Abkürzungen |
