<!--en:400213f5-->
# Self-Attention mit mehreren Köpfen

<!--en:cc5898f1-->
In den Beispielen aus :numref:`sec_attention-scoring-functions` stammten die Abfragen (engl. query) aus einer Sequenz, die Schlüssel (engl. key) und Werte (engl. value) aus einer anderen. Ein Sprachmodell, das nur aus einem Decoder besteht, hat nur eine Sequenz. Um jedes Token in seinen Kontext einzuordnen, muss das Modell Abfragen, Schlüssel und Werte aus dieser Sequenz selbst ableiten.

<!--en:8d797dd7-->
Betrachten wir den Satz *„Alice gave Bob the book“*. Um *gave* zu verstehen, muss klar sein, wer gibt (*Alice*), wer etwas erhält (*Bob*) und welcher Gegenstand den Besitzer wechselt (*book*). Bei der *Self-Attention* :cite:`Cheng.Dong.Lapata.2016,Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017` projiziert jedes Token seine Darstellung in eine Abfrage, einen Schlüssel und einen Wert. So kann *gave* die anderen Wörter im Satz abfragen und deren Werte zusammenführen.

<!--en:1a0dcb62-->
Ein einzelner Attention-Zugriff liefert jedoch nur *eine* Wahrscheinlichkeitsverteilung pro Abfrage und fasst die Werte zu einem einzigen Vektor zusammen. Wie kann ein Token mehrere verschiedene Argumente abrufen, ohne sie zu einem Mittelwert zusammenzufassen? Zunächst verfolgen wir an unserem Satz aus fünf Wörtern, wie ein einzelner Self-Attention-Kopf arbeitet. Dann beweisen wir eine informationstheoretische untere Schranke, die erklärt, warum ein Kopf zwei unabhängige Werte nicht rekonstruieren kann. Schließlich konstruieren wir die *Multi-Head-Attention* und die *Grouped-Query-Attention* (GQA).

<!--en:0c2e5121-->
## Self-Attention

<!--en:3f14d969-->
Bei der Self-Attention übernimmt jede Tokendarstellung $\mathbf{x}_i \in \mathbb{R}^d$ (Zeile $i$ von $\mathbf{X} \in \mathbb{R}^{n \times d}$) durch die gelernten linearen Projektionen $\mathbf{W}_q, \mathbf{W}_k \in
\mathbb{R}^{d \times d_h}$ und $\mathbf{W}_v \in \mathbb{R}^{d \times d_v}$ drei verschiedene Rollen. Die *Abfrage* $\mathbf{x}_i\mathbf{W}_q$ gibt an, was das Token sucht. Der *Schlüssel* $\mathbf{x}_i\mathbf{W}_k$ gibt an, was es anderen Abfragen anbietet. Der *Wert* $\mathbf{x}_i\mathbf{W}_v$ enthält den Inhalt, den das Token bei seiner Auswahl zurückgibt. Mit skalierter Skalarprodukt-Attention erhalten wir einen Attention-*Kopf* der Breite $d_h$:

<!--en:2f2a914f-->
### Ein Kopf am Beispiel eines Satzes

<!--en:e3e04557-->
Um :eqref:`eq_self-attention` an einem Beispiel nachzuvollziehen, stellen wir jedes Wort in *„Alice gave Bob the book“* durch einen 9-dimensionalen Vektor dar. Er verkettet zwei One-Hot-Kodierungen. Die ersten vier Komponenten geben die grammatische Rolle an (Person, Handlung, Gegenstand oder Funktionswort), die letzten fünf das Wort selbst. Wir wählen $\mathbf{W}_q$ und $\mathbf{W}_k$ so, dass sie nur die vier Komponenten der Rolle auslesen und auf $d_h = 2$ Dimensionen abbilden. Mit $\mathbf{W}_v$ lesen wir die fünf Komponenten für das Wort aus:

<!--en:23607dad-->
| Rolle | Abfrage | Schlüssel |

<!--en:18adebdb-->
| Person (Alice, Bob) | $(1, -1)$ | $(1, 1)$ |

<!--en:1081ffe0-->
| Handlung (*gave*) | $(1, 1)$ | $(1, -1)$ |

<!--en:e4a3936a-->
| Gegenstand (book) | $(1, -1)$ | $(1, 1)$ |

<!--en:d3658a0d-->
| Funktionswort („the“) | $(0, 0)$ | $(0, 0)$ |

<!--en:a236f130-->
Für *gave* ergibt das Skalarprodukt der Abfrage $(1, 1)$ den Wert $2$ mit dem Schlüssel $(1, 1)$ jedes Substantivs und $0$ mit den Schlüsseln von *gave* und *„the“*. Nach Division durch $\sqrt{2}$ beträgt der skalierte Score für jedes Substantiv $1.41$ und sonst $0$. Die Softmax-Funktion liefert daraus die Gewichte $e^{1.41}/(3e^{1.41} + 2) \approx 0.29$ für jedes Substantiv und $0.07$ für *gave* und *„the“* (:numref:`fig_self-attention`).

<!--en:53fe71d5-->
![Ein Self-Attention-Kopf für „Alice gave Bob the book“ mit den im Code oben von Hand festgelegten Abbildungen ($d_h = 2$). Jedes Token liefert eine Abfrage, einen Schlüssel und einen Wert. Für die Abfrage von *gave*, $(1, 1)$, ergibt sich mit jedem Schlüssel der Score $\mathbf{q} \cdot \mathbf{k}/\sqrt{2}$. Die Softmax-Funktion wandelt die Scores $1.41$ und $0$ in die Gewichte $0.29$ und $0.07$ um. Die Ausgabe von *gave* ist die gewichtete Summe der Werte, die die fünf Wörter kennzeichnen.](../img/mdl-attention-self-attention.svg)

<!--en:a840f332-->
Umgekehrt erzielt die Abfrage $(1, -1)$ jedes Substantivs bei *gave* den Score $1.41$ (Gewicht $0.51$) und sonst $0$. Die Abfrage von *„the“* ist der Nullvektor, sodass alle fünf Wörter dasselbe Gewicht $0.20$ erhalten. Mit einer kausalen Maske kann *gave* nur auf *Alice* und sich selbst zugreifen. Nach erneuter Normierung ergeben ihre Scores $(1.41, 0)$ die Gewichte $(0.80, 0.20)$.

<!--en:17631875-->
### Getrennte Abbildungen für Abfragen und Schlüssel

<!--en:2c869620-->
Warum verwenden wir getrennte Matrizen $\mathbf{W}_q$ und $\mathbf{W}_k$? Diente eine einzige Projektion $\mathbf{u}_i = \mathbf{x}_i\mathbf{W}$ sowohl als Abfrage als auch als Schlüssel, wäre die Score-Matrix $s_{ij} = \mathbf{u}_i^\top \mathbf{u}_j / \sqrt{d_h}$ symmetrisch und positiv semidefinit. Nach der Cauchy-Schwarz-Ungleichung gälte $s_{ij} \le
\sqrt{s_{ii}\, s_{jj}} \le \max(s_{ii}, s_{jj})$. Zwei Dinge wären dann unmöglich. Ein Verb könnte ein Substantiv nicht stärker gewichten als das Substantiv das Verb, denn es gilt $s_{ij} = s_{ji}$. Auch könnten zwei Token einander nie beide höher bewerten als sich selbst. Ein Verb kann Substantive nur dann sich selbst vorziehen, wenn diese die größere Norm haben. Dann ziehen die Substantive aber sich selbst dem Verb vor. Getrennte Projektionen für Abfragen, Schlüssel und Werte entkoppeln drei Fragen: *Wer sucht?*, *Wer wird gefunden?* und *Welcher Inhalt wird weitergegeben?*

<!--en:99a0ff66-->
## Grenzen eines einzelnen Kopfes

<!--en:07b351d6-->
In unserem Satz aus fünf Wörtern fasste *gave* die Wörter *Alice*, *Bob* und *book* zu einem einzigen gewichteten Mittelwert zusammen. Wie stark schränkt uns eine einzige Attention-Verteilung pro Abfrage ein? Um diesen Flaschenhals zu quantifizieren, betrachten wir die *Aufgabe, beide Werte zu kopieren* (engl. copy-both task). An zwei Quellpositionen stehen unabhängige standardnormalverteilte Wertvektoren $\mathbf{v}_1,
\mathbf{v}_2 \in \mathbb{R}^{d_v}$. Ihre Schlüssel sind fest und hängen nur von der Position ab. Eine einzige Abfrage soll durch lineares Auslesen die Verkettung $\mathbf{t} = [\mathbf{v}_1;
\mathbf{v}_2] \in \mathbb{R}^{2d_v}$ ausgeben.

<!--en:37b9ab15-->
Da die Schlüssel nur von der Position abhängen, vergibt ein einzelner Kopf feste Softmax-Gewichte $(\alpha_1, \alpha_2)$ mit $\alpha_1 + \alpha_2 = 1$. Zum linearen Auslesen steht nur eine Mischung $\mathbf{m} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2
\in \mathbb{R}^{d_v}$ zur Verfügung.

<!--en:859946c7-->
**Proposition.** Ein Attention-Kopf habe zwei feste, rein positionsabhängige Schlüssel und unabhängige standardnormalverteilte Werte $\mathbf{v}_1, \mathbf{v}_2 \in
\mathbb{R}^{d_v}$. Beim linearen Auslesen beträgt der kleinstmögliche relative quadratische Fehler beim Kopieren beider Werte $1/2$, und zwar für jedes Paar von Softmax-Gewichten $(\alpha_1, \alpha_2)$.

<!--en:ece735db-->
**Beweis.** Gegeben sei $\mathbf{m} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2$. Der Schätzer für $\mathbf{v}_1$ mit dem kleinsten mittleren quadratischen Fehler ist der bedingte Erwartungswert $\mathbb{E}[\mathbf{v}_1 \mid \mathbf{m}] =
\tfrac{\alpha_1}{\alpha_1^2 + \alpha_2^2}\, \mathbf{m}$. Er ist linear in $\mathbf{m}$. Damit verbleibt eine Restvarianz von $1 - \tfrac{\alpha_1^2}{\alpha_1^2

<!--en:50fc153a-->
+ \alpha_2^2} = \tfrac{\alpha_2^2}{\alpha_1^2 + \alpha_2^2}$ pro Koordinate. Aus Symmetriegründen verbleibt bei der Schätzung von $\mathbf{v}_2$ eine Restvarianz von $\tfrac{\alpha_1^2}{\alpha_1^2 + \alpha_2^2}$ pro Koordinate. Zusammen ergeben die Restvarianzen $1$ bei einer Gesamtvarianz der Zielgröße von $2$ pro Koordinate. Der relative quadratische Fehler beträgt also $1/2$, unabhängig von $(\alpha_1, \alpha_2)$. $\blacksquare$

<!--en:f7b1f6f6-->
Um beide Werte zu kopieren, brauchen wir (1) mindestens zwei linear unabhängige *Gewichtungen* der Schlüssel und (2) eine ausreichend große *Ausgabedimension* mit Platz für alle $2d_v$ abgerufenen Koordinaten (:exref:`ex_multihead-what-the-bound-counts`). Wir überprüfen die Schranke von $\sqrt{1/2}
\approx 0.707$ für die Wurzel des relativen mittleren quadratischen Fehlers bei einem Kopf. Dann zeigen wir, dass zwei Köpfe mit verschiedenen Gewichtungen ausreichen. Selbst bei weichen Gewichtungen wie $(0.88, 0.12)$ und $(0.12, 0.88)$ lässt sich die $2 \times 2$-Mischmatrix beim linearen Auslesen bis auf Gleitkommagenauigkeit invertieren (:numref:`fig_one-head-averages`).

<!--en:2f2c9c93-->
![Eine Gewichtung pro Abfrage im Vergleich zu zweien. Beim Kopieren beider Werte verwendet ein einzelner Kopf eine einzige Softmax-Gewichtung. Seine Ausgabe mischt daher die beiden angeforderten Werte. Zwei Köpfe mit unterschiedlichen Gewichtungen liefern zwei Mischungen, aus denen sich beide Werte linear auslesen lassen.](../img/mdl-attention-one-head-averages.svg)

<!--en:ebcc7705-->
## Multi-Head-Attention

<!--en:da5595ab-->
### Projektionen für die einzelnen Köpfe

<!--en:2d09fb9b-->
Die *Multi-Head-Attention* :cite:`Vaswani.Shazeer.Parmar.ea.2017` führt $H$ unabhängige Attention-Köpfe parallel aus. Jeder Kopf besitzt eigene Projektionen für Abfragen, Schlüssel und Werte $\mathbf{W}_i^{(q)}, \mathbf{W}_i^{(k)}, \mathbf{W}_i^{(v)}
\in \mathbb{R}^{d \times d_h}$:

<!--en:212ce8a9-->
Anschließend verkettet sie die Ausgaben und wendet eine abschließende lineare Projektion $\mathbf{W}_o \in \mathbb{R}^{H d_h \times d}$ an (:numref:`fig_multi-head-attention`):

<!--en:498d5312-->
![Multi-Head-Attention. Jeder Kopf liest dieselben Abfragen, Schlüssel und Werte, projiziert sie in seinen eigenen Unterraum und berechnet dort die Attention. Nach der Verkettung führt eine einzige Ausgabeprojektion die Ausgaben der Köpfe zusammen.](../img/mdl-attention-multi-head.svg)

<!--en:4a9bfc4f-->
### Parameteranzahl und FLOP-Zahl

<!--en:60cddb2b-->
Mit der Kopfbreite $d_h = d/H$ bleibt die Gesamtdimension aller Köpfe gleich $H d_h = d$. Stellen wir die $H$ Abfrageprojektionen nebeneinander, bilden sie eine einzige $d \times d$-Matrix. Dasselbe gilt für die Schlüssel-, Wert- und Ausgabeprojektionen. Zählen wir eine Multiplikation mit anschließender Addition als $2$ FLOPs, ergibt sich für eine Self-Attention-Schicht mit mehreren Köpfen über $n$ Token folgender Aufwand:

<!--en:8eea6b05-->
Für eine Schicht von GPT-2 small ($d = 768$, $n = 1{,}024$) sind das $4.8$ GFLOPs für die Projektionen, $3.2$ GFLOPs für die Scores und das Mischen der Werte sowie $2.4 \times 10^6$ Parameter. Bemerkenswerterweise hängen weder die Parameteranzahl noch die führenden Terme der FLOP-Zahl von der Anzahl der Köpfe $H$ ab. Teilen wir die Breite $d$ auf $H$ Köpfe auf, schrumpft der Aufwand für jedes Skalarprodukt auf das $1/H$-Fache. Dafür berechnen wir $H$ solcher Produkte. Nur der Zusatzaufwand von $O(Hn^2)$ für die Softmax-Funktion wächst mit $H$. Umgekehrt kann $d_h$ nicht beliebig klein werden. Bei $d_h = 1$ sind die skalaren Scores $q k_j$ monoton in $k_j$. Eine Abfrage kann sich also nur auf die extremalen Schlüssel konzentrieren (:exref:`ex_multihead-narrow-heads`). In der Praxis ist $d_h \in \{64, 128\}$ üblich.

<!--en:518a9550-->
### Vergleich mit Faltung und Rekurrenz

<!--en:87722c23-->
:numref:`fig_cnn-rnn-self-attention` und :numref:`tab_cnn-rnn-attn` vergleichen die 1D-Faltung (:numref:`chap_cnn`), die Rekurrenz (:numref:`sec_rnn`) und die Self-Attention als Sequenz-zu-Sequenz-Schichten für $n$ Token der Breite $d$.

<!--en:7facde17-->
![Verbindungen in Sequenzschichten für fünf Token. Die hervorgehobenen Kanten zeigen, wie die Eingaben in einer Schicht die dritte Ausgabe erreichen. Eine Faltung mit Kerngröße $k = 3$ erfasst die Nachbarn, eine Rekurrenz reicht die Information schrittweise entlang der Kette weiter und Self-Attention verbindet jede Position direkt mit jeder anderen. Unter jedem Teilbild steht, wie lang der Pfad zwischen weit entfernten Token ist und ob die Schicht die Ausgaben parallel berechnet.](../img/mdl-attention-cnn-rnn-self-attention.svg)

<!--en:59403698-->
:Kosten pro Schicht für die Abbildung von $n$ Token der Dimension $d$ auf $n$ Token der Dimension $d$.

<!--en:83a8eff7-->
| Schichttyp | Rechenaufwand | Sequenzielle Operationen | Maximale Pfadlänge |

<!--en:9e063211-->
| Faltung (Kerngröße $k$) | $O(knd^2)$ | $O(1)$ | $O(n/k)$ |

<!--en:dc1110ce-->
| Rekurrenz | $O(nd^2)$ | $O(n)$ | $O(n)$ |

<!--en:30537b51-->
| Self-Attention | $O(nd^2+n^2d)$ | $O(1)$ | $O(1)$ |

<!--en:9dbc3cf4-->
Self-Attention verbindet $O(1)$ sequenzielle Schritte (volle Parallelität beim Training) mit einer maximalen Pfadlänge von $O(1)$ zwischen zwei beliebigen Token. Damit entfällt die Kette von $O(n)$ Schritten, die der Gradient bei der Rekurrenz durchläuft (:numref:`sec_bptt`). Der Preis dafür ist der quadratische Term $4n^2 d$ für die Scores und das Mischen der Werte. Er übersteigt die Projektionskosten von $8nd^2$, sobald $n > 2d$ gilt ($n > 1{,}536$ für GPT-2 small).

<!--en:869d223f-->
### Implementierung

<!--en:a10417bb-->
In `MultiHeadAttention` berechnet eine einzige lineare Schicht alle $H$ Kopfprojektionen auf einmal. Dann formt `split_heads` den Tensor von `(batch, length, H * d_h)` in `(batch, H, length, d_h)` um. So wertet `DotProductAttention` alle Köpfe in einer einzigen Batchoperation aus (:numref:`fig_head-split`). Wie bei modernen Transformern üblich, verzichten die Projektionen standardmäßig auf additive Bias-Terme (`bias=False`). Über den Parameter `num_kv_heads` können mehrere Köpfe dieselben Schlüssel und Werte nutzen (:numref:`subsec_kv-head-sharing`).

<!--en:c0a58c1c-->
![Köpfe als Slices einer Projektion. Die projizierten Abfragen von $n$ Token zerfallen spaltenweise in $H$ Slices der Breite $d_h = d/H$. Diese bilden eine eigene Kopfachse. Jeder Kopf berechnet seine Attention unabhängig. Vor der Ausgabeprojektion $\mathbf{W}_o$ liegen die Ausgaben der Köpfe wieder nebeneinander. Für Schlüssel und Werte gilt dieselbe Aufteilung.](../img/mdl-attention-head-split.svg)

<!--en:fd005a61-->
Die beiden Hilfsfunktionen `split_heads` und `merge_heads` (:numref:`fig_head-split`) formen Tensoren um und vertauschen Achsen. Vor der Attention zerlegen wir die letzte Dimension $H d_h$, sodass eine eigene Kopfachse entsteht (Form `(batch, num_heads, length, width)`). Danach verketten wir die Ausgaben der $H$ Köpfe wieder:

<!--en:5e388f4e-->
### Gelernte Köpfe beim Kopieren beider Werte

<!--en:3f1d9e60-->
Wir testen nun `MultiHeadAttention` mit der Breite $d = 16$ an der Aufgabe, beide Werte zu kopieren. Die Wertvektoren mit $d_v = 8$ ergänzen wir mit Nullen auf die Breite $16$. Als Schlüssel dienen $\mathbf{e}_1,
\mathbf{e}_2$ und als Abfrage $\mathbf{e}_3$.

<!--en:31a7ba14-->
Für jede Kopfanzahl $H \in \{1, 2, 4\}$ trainieren wir mit zwei verschiedenen Zufallsstartwerten. Alle Modelle haben dieselbe Gesamtbreite $d = 16$ und dieselbe Parameteranzahl $1{,}024$. Wir vergleichen ihren relativen Rekonstruktionsfehler und das Attention-Gewicht $\alpha_1$, das jeder Kopf dem ersten Wertvektor $\mathbf{v}_1$ zuweist:

<!--en:d114c806-->
Alle Konfigurationen haben $4 \times 16^2 = 1{,}024$ Parameter. Wie unsere Proposition vorhersagt, bleibt der Fehler für $H = 1$ an der Schranke $\sqrt{1/2} \approx 0.71$ stehen. Für $H \in \{2, 4\}$ lernen die Köpfe hingegen unterschiedliche Gewichtungen, und der Fehler sinkt auf $\sim 10^{-7}$.

<!--en:beca7c1b-->
### Gemeinsam genutzte Schlüssel und Werte

<!--en:464dc5de-->
Bei der autoregressiven Generierung speichern wir frühere Schlüssel und Werte zwischen. So kann jedes neue Token auf alle vorherigen Positionen zugreifen, ohne deren Projektionen erneut zu berechnen (:numref:`sec_kv-cache`). Die *Grouped-Query-Attention* (GQA) :cite:`Ainslie.Lee-Thorp.Jong.ea.2023` verkleinert diesen Cache um den Faktor $H / H_\textrm{kv}$. Dazu nutzt jede Gruppe von $H /
H_\textrm{kv}$ Query-Köpfen einen gemeinsamen Schlüssel-Wert-Kopf. Mit $H_\textrm{kv} = 1$ erhalten wir die *Multi-Query-Attention* (MQA) :cite:`Shazeer.2019`.

<!--en:bd0eb6e6-->
Beim Kopieren beider Werte erfassen die gemeinsam genutzten Wertköpfe höchstens $H_\textrm{kv}
d_h$ Koordinaten pro Wert. Um einen $d_v$-dimensionalen Wert zu rekonstruieren, muss daher $H_\textrm{kv} d_h \ge d_v$ gelten (:exref:`ex_multihead-sharing-keys-and-values`). Für $d = 16$ und $d_v = 8$ erfüllen die Konfigurationen $(H, H_\textrm{kv}) = (2, 1)$ und $(4, 2)$ die Bedingung mit $H_\textrm{kv} d_h =
8$. Sie lösen die Aufgabe bis auf fp32-Genauigkeit. Bei $(4, 1)$ gilt hingegen $H_\textrm{kv}
d_h = 4$. Trotz unterschiedlicher gelernter Gewichtungen in den vier Query-Köpfen liegt der Fehler wieder an der Schranke von $0.71$:

<!--en:b43e5cd7-->
## Cross-Attention

<!--en:55b15d40-->
Bei der Self-Attention dient dieselbe Sequenz $\mathbf{X}$ als Eingabe für Abfragen, Schlüssel und Werte (`attention(X, X, X)`). Die *Cross-Attention* bezieht ihre Abfragen aus einer Zielsequenz $\mathbf{A}$, die Schlüssel und Werte aus einer Quellsequenz $\mathbf{B}$ (`attention(A, B, B)`). Sie erzeugt eine Ausgabezeile pro Token von $\mathbf{A}$. Encoder-Decoder-Transformer nutzen Cross-Attention, um beim Übersetzen den kodierten Ausgangssatz zu lesen. In multimodalen Modellen können Texttoken damit visuelle Merkmale abfragen :cite:`alayrac2022flamingo`.

<!--en:69f61a19-->
Selbst mit mehreren Köpfen erkennt Self-Attention jedoch die Wortreihenfolge nicht. Vertauschen wir *Alice* und *Bob* in *„Alice gave Bob the book“*, bleibt die Ausgabe bei *gave* gleich. Wie wir die Reihenfolge der Sequenz einbringen, behandelt :numref:`sec_positional-information`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:656c96ac-->
Self-Attention projiziert jedes Token einer Sequenz in eine Abfrage, einen Schlüssel und einen Wert. So verbindet sie alle Positionspaare in $O(1)$ sequenziellen Schritten bei einer Pfadlänge von $O(1)$ und einem Aufwand von $O(nd^2 + n^2d)$ FLOPs. Ein einzelner Kopf bildet pro Abfrage nur eine konvexe Kombination der Werte. Soll er zwei unabhängige Werte gleichzeitig abrufen, beträgt sein relativer quadratischer Fehler deshalb mindestens $1/2$. Multi-Head-Attention überwindet diesen Flaschenhals. Sie führt $H$ Köpfe parallel in Unterräumen der Breite $d_h = d/H$ aus und kombiniert deren Ausgaben mit $\mathbf{W}_o$. Dafür braucht sie $4d^2$ Parameter und $8nd^2 + 4n^2d$ FLOPs, unabhängig von $H$. Bei Grouped-Query-Attention und Multi-Query-Attention nutzen Gruppen von Abfragen insgesamt $H_\textrm{kv} < H$ gemeinsame Schlüssel-Wert-Köpfe. Dadurch schrumpft der Cache bei der Generierung um den Faktor $H/H_\textrm{kv}$. Die Fähigkeit zum Abruf bleibt erhalten, sofern die gemeinsame Wertbreite $H_\textrm{kv} d_h$ für die abgerufenen Merkmale ausreicht.

<!--en:2294243a-->
## Übungen

<!--en:f62bba76-->
1. [code] **Schmale Köpfe.** Für $d_h = d/H$ hängt der führende Rechenaufwand in :eqref:`eq_multihead-flops` nicht von $H$ ab. Bis zu $H = d$ scheinen zusätzliche Köpfe also nichts zu kosten. Betrachten Sie einen Kopf ohne Maske mit den Scores $\mathbf{q}^\top \mathbf{k}_j / \sqrt{d_h}$ für die Schlüssel $\mathbf{k}_1,
   \ldots, \mathbf{k}_m \in \mathbb{R}^{d_h}$. Eine Abfrage heiße scharf auf dem Schlüssel $j$, wenn sich ihre Gewichte auf $\mathbf{k}_j$ konzentrieren, sobald man die Abfrage mit einem gegen unendlich wachsenden Faktor skaliert.
    1. Zeigen Sie für $d_h = 1$ und paarweise verschiedene skalare Schlüssel, dass die Gewichte jeder Abfrage monoton in $k_j$ sind. Folgern Sie, dass die Abfragen eines solchen Kopfes nur auf dem größten oder dem kleinsten Schlüssel scharf sein können.
    1. Zeigen Sie, dass sich für $d_h = 2$ die Schlüssel und Abfragen so wählen lassen, dass jede der $n$ Abfragen auf einem Schlüssel ihrer Wahl scharf ist. Dies soll für jede Zuordnung von Abfragen zu Schlüsseln gelten.
    1. Geben Sie anhand dieser beiden Ergebnisse an, welche Attention-Muster eine Schicht verliert, wenn sich $H$ dem Wert $d$ nähert.
    1. Messen Sie die Laufzeit des Vorwärtsdurchlaufs von `MultiHeadAttention` bei Dropout 0 für Self-Attention mit $d = 256$ und $n = 512$. Verwenden Sie einen Minibatch von $8$ Sequenzen und testen Sie $H \in \{1, 2, 4, 8, 16, 32, 64\}$. Führen Sie zunächst einen Aufwärmlauf aus. Warten Sie bei der Messung auf das Ergebnis, bevor Sie die Uhr anhalten. Geben Sie das Laufzeitverhältnis der langsamsten zur schnellsten Konfiguration an. Bestimmen Sie die Kosten, die mit $H$ wachsen und in :eqref:`eq_multihead-flops` fehlen. Prüfen Sie, ob sie den Verlauf erklären.
   :exlabel:`ex_multihead-narrow-heads`

<!--en:2bd25bf1-->
1. [code] **Köpfe als Slices.** `MultiHeadAttention` berechnet für Abfragen, Schlüssel und Werte jeweils eine Projektion der Breite $d$ und teilt sie mit `split_heads` in Köpfe auf.
    1. Kopf $i$ erhält je einen Slice der Ausgabemerkmale von `W_q`, `W_k` und `W_v`. Bestimmen Sie, welche Eingabemerkmale von `W_o` die Ausgabe dieses Kopfes erhalten. Schreiben Sie einen Vorwärtsdurchlauf, der in einer Schleife über die Köpfe läuft, die jeweiligen Projektionsausschnitte anwendet und `d2l.DotProductAttention` einmal pro Kopf aufruft. Prüfen Sie mit zufälligen Eingaben und einer Padding-Maske, ob das Ergebnis mit `MultiHeadAttention` bei Dropout 0 übereinstimmt. Erweitern Sie Ihre Schleife auf `num_kv_heads < num_heads`.
    1. Ersetzen Sie den Funktionsrumpf von `split_heads` durch `X.reshape(X.shape[0], num_heads, X.shape[1], -1)`. Der zurückgegebene Tensor hat dieselbe Form. Beschreiben Sie, welche Zahlen jeder „Kopf“ nun enthält. Zeigen Sie anhand zufälliger Eingaben, dass sich die Ausgabe ändert, und erklären Sie, warum keine Prüfung der Form den Fehler erkennen kann. Für welche Längen und Breiten der Eingabe stimmen die beiden Versionen gegebenenfalls überein?

<!--en:88289fc0-->
1. **Permutation der Quelle.** Bei der Cross-Attention stammen die Abfragen aus $\mathbf{A} \in \mathbb{R}^{n \times d}$, die Schlüssel und Werte aus $\mathbf{B} \in \mathbb{R}^{m \times d}$. Beide enthalten keine Positionsinformation. Sei $\boldsymbol{\Pi}$ eine $m \times m$-Permutationsmatrix.
    1. Ersetzen Sie $\mathbf{B}$ sowohl im Schlüssel- als auch im Wertargument durch $\boldsymbol{\Pi}\mathbf{B}$. Zeigen Sie, dass die Ausgabe von `MultiHeadAttention` ohne Maske gleich bleibt. Zeigen Sie auch, dass sich die Ausgabe im Allgemeinen ändert, wenn Sie nur das Schlüsselargument ersetzen.
    1. Nun markiere eine Padding-Maske nur die ersten $\ell$ Zeilen von $\mathbf{B}$ als gültig. Bei welchen Permutationen bleibt die Ausgabe gleich, wenn die Maske fest bleibt?
   :exlabel:`ex_multihead-permuting-the-source`

<!--en:eb00465c-->
1. [code] **Fast identische Köpfe.** ● Zwei Köpfe mit den Gewichten $(0.5 + \epsilon, 0.5 - \epsilon)$ und $(0.5 - \epsilon, 0.5 +
   \epsilon)$ liefern zwei verschiedene Mischungen. Durch lineares Auslesen lässt sich die Mischung daher für jedes $\epsilon > 0$ exakt umkehren.
    1. Schreiben Sie die $2 \times 2$-Mischmatrix auf, berechnen Sie ihre Eigenwerte und Eigenvektoren und geben Sie ihre Konditionszahl als Funktion von $\epsilon$ an.
    1. Addieren Sie zu beiden Mischungen unabhängiges gaußsches Rauschen mit der Standardabweichung $\sigma = 0.01$. Messen Sie den Fehler mit `readout_error` für $\epsilon \in \{0.4, 0.1, 0.03, 0.01, 0.003, 0.001\}$. Tragen Sie den Fehler gegen $\epsilon$ auf logarithmischen Achsen auf. Lesen Sie die Steigung im geraden Teil der Kurve ab und erklären Sie sie mit der Konditionszahl aus (a).
    1. In einer frisch initialisierten Schicht gewichtet jeder Kopf die beiden Positionen beim Kopieren beider Werte annähernd mit $(0.5, 0.5)$. Ändern Sie `train_copy_both` so, dass die Funktion alle 20 Schritte den relativen Fehler und die Gewichte jedes Kopfes aufzeichnet. Stellen Sie beides für $H = 2$ grafisch dar. Erklären Sie anhand der Kondition der Mischmatrix, wie die Fehlerkurve in den ersten paar hundert Schritten verläuft. Wiederholen Sie den Versuch für $H = 8$. Ordnen Sie die Köpfe am Ende drei Gruppen zu: Köpfe mit annähernder One-Hot-Gewichtung, Köpfe mit weichen, aber unterschiedlichen Gewichtungen und Köpfe, die einen anderen Kopf nahezu duplizieren.
   :exlabel:`ex_multihead-nearly-identical-heads`

<!--en:9c826093-->
1. **Wovon die Schranke abhängt.** ● Die Proposition in :numref:`subsec_one-mixture-bottleneck` setzt feste Softmax-Gewichte voraus. Für seine Fehlerschranke genügt jedoch eine schwächere Voraussetzung. Zum Auslesen steht nur ein einziger Vektor $\mathbf{m} \in \mathbb{R}^{d_v}$ zur Verfügung, aus dem die Zielgröße $\mathbf{t} \in \mathbb{R}^{2d_v}$ entstehen muss.
    1. Sei $\mathbf{m} \in \mathbb{R}^{d_v}$ eine beliebige Funktion von $\mathbf{v}_1$ und $\mathbf{v}_2$. Sie darf auch eine Mischung sein, deren Gewichte von den Werten abhängen. Sei $\mathbf{W}_o \in \mathbb{R}^{d_v
       \times 2d_v}$ eine beliebige Matrix. Zeigen Sie, dass folgende Ungleichung gilt:
       $$\mathbb{E}\,\|\mathbf{m}\mathbf{W}_o - \mathbf{t}\|^2 \geq \tfrac{1}{2}\, \mathbb{E}\,\|\mathbf{t}\|^2.$$
       Folgern Sie, dass die Schranke auch bei inhaltsabhängigen Schlüsseln, zusätzlichen Schichten oder einer Residualverbindung gilt, solange das Auslesen linear ist und dafür $d_v$ Zahlen zur Verfügung stehen.
    1. Eine Schicht der Breite $d$ mit zwei Köpfen der Breite $d_h = d/2$ erhält die Werte $\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^{d/2}$, mit Nullen ergänzt auf die Breite $d$. Geben Sie für jeden Kopf die Attention-Gewichte sowie Wertprojektionen $\mathbf{W}_i^{(v)}$ und eine Ausgabeprojektion $\mathbf{W}_o$ an, die die Aufgabe exakt lösen. Vergleichen Sie Ihre Gewichte mit den in :numref:`subsec_copy-both-learned` ausgegebenen Gewichten.
    1. Verallgemeinern Sie die Proposition auf $k$ Quellpositionen mit unabhängigen standardnormalverteilten Werten $\mathbf{v}_1, \ldots, \mathbf{v}_k \in
       \mathbb{R}^{d_v}$ und der Zielgröße $[\mathbf{v}_1; \ldots;
       \mathbf{v}_k]$. Jeder der $H$ Köpfe gewichtet die Werte mit festen Gewichten. Die Gewichte von Kopf $i$ bilden die Zeile $i$ einer Matrix $\mathbf{A} \in \mathbb{R}^{H \times k}$. Zum linearen Auslesen stehen alle $H$ Mischungen zur Verfügung. Zeigen Sie, dass der kleinstmögliche relative quadratische Fehler durch folgenden Ausdruck gegeben ist:
       $$1 - \frac{\operatorname{rank}(\mathbf{A})}{k},$$
       Zeigen Sie außerdem, dass Softmax-Gewichte trotz ihrer durchweg positiven Einträge $\operatorname{rank}(\mathbf{A}) = \min(H, k)$ erreichen können.
   :exlabel:`ex_multihead-what-the-bound-counts`

<!--en:a3a272ee-->
1. [code] **Gemeinsame Schlüssel und Werte beim Kopieren beider Werte.** ● Eine Schicht der Breite $d$ habe $H$ Query-Köpfe der Breite $d_h = d/H$ und $H_\textrm{kv} < H$ Schlüssel-Wert-Köpfe.
    1. Zeigen Sie mit dem Zählargument aus :exref:`ex_multihead-what-the-bound-counts`, dass die Schicht die Aufgabe nur lösen kann, wenn $H_\textrm{kv} d_h \ge d_v$ gilt. Zeigen Sie, dass diese Bedingung auch ausreicht.
    1. Trainieren Sie mit `train_copy_both(2, 1)` eine Schicht, in der zwei Query-Köpfe einen Schlüssel-Wert-Kopf gemeinsam nutzen. Ermitteln Sie, welche Parameter sich zwischen den beiden Köpfen unterscheiden und welche sie gemeinsam nutzen. Erklären Sie anhand der Aufgabe, beide Werte zu kopieren, warum der Unterschied an der von Ihnen gefundenen Stelle liegen muss.
   :exlabel:`ex_multihead-sharing-keys-and-values`

<!--en:8c0660d6-->
[Diskussion](https://d2l.discourse.group/t/36770)

<!--en:fb0a1644-->
Self-Attention, Multi-Head-Attention und gemeinsame Nutzung von Schlüsseln und Werten<br>**Self-Attention · Flaschenhals einer einzigen Mischung · parallele Köpfe in Unterräumen · FLOPs und Parameteranzahl · GQA und MQA**

<!--en:629dbed1-->
::: {.slide title="Warum eine Sequenz sich selbst auf mehrere Arten abfragen muss"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:913159b7-->
- **Mehrere syntaktische Argumente:** Um das Verb *gave* in *„Alice gave Bob the book“* zu verstehen, muss klar sein, wer gibt (*Alice*), wer etwas erhält (*Bob*) und um welchen Gegenstand es geht (*book*).

<!--en:b1a1ec0c-->
- **Flaschenhals eines einzelnen Kopfes:** Ein Attention-Kopf berechnet *eine* Wahrscheinlichkeitsverteilung pro Abfrage. Er mittelt alle berücksichtigten Argumente zu einem einzigen Vektor ($64$ Gleitkommazahlen bei GPT-2 small).

<!--en:289c384d-->
- **Parallele Köpfe in Unterräumen:** Mit $H$ unabhängigen parallelen Zugriffen kann ein Token mehrere verschiedene Argumente gleichzeitig abrufen.

<!--en:23178c32-->
::: {.slide title="Self-Attention mit drei gelernten Projektionen pro Token"}

<!--en:bbfaf4b0-->
[Mathematische Form der Self-Attention]{.kicker}

<!--en:860561a9-->
- **Drei verschiedene Rollen aus einer Eingabe $\mathbf{X} \in \mathbb{R}^{n \times d}$:**
  $$\textrm{SelfAttention}(\mathbf{X}) = \mathrm{softmax}\!\left(\frac{(\mathbf{X}\mathbf{W}_q)(\mathbf{X}\mathbf{W}_k)^\top}{\sqrt{d_h}}\right)\mathbf{X}\mathbf{W}_v$$

<!--en:99dc0daf-->
- **Abfrage ($\mathbf{q}_i = \mathbf{x}_i \mathbf{W}_q$):** Welche Kontextmerkmale sucht Position $i$?

<!--en:1ea92cb9-->
- **Schlüssel ($\mathbf{k}_j = \mathbf{x}_j \mathbf{W}_k$) und Wert ($\mathbf{v}_j = \mathbf{x}_j \mathbf{W}_v$):** Was bietet Position $j$ an und welchen Inhaltsvektor gibt sie bei ihrer Auswahl zurück?

<!--en:8852c9e9-->
::: {.slide title="Warum Abfragen und Schlüssel getrennte Matrizen brauchen"}

<!--en:ecf8bbbc-->
[Asymmetrie des Zugriffs]{.kicker}

<!--en:75e169c3-->
- **Einschränkung durch Symmetrie und positive Semidefinitheit ($\mathbf{W}_q = \mathbf{W}_k = \mathbf{W}$):** Mit $\mathbf{u}_i = \mathbf{x}_i \mathbf{W}$ ist die Score-Matrix $s_{ij} = \mathbf{u}_i^\top \mathbf{u}_j / \sqrt{d_h}$ symmetrisch und positiv semidefinit.

<!--en:cd99b9cb-->
- **Schranke für den gegenseitigen Score nach Cauchy-Schwarz:**
  $$s_{ij} \le \sqrt{s_{ii}\,s_{jj}}$$

<!--en:2cc5405c-->
- **Sprachliche Einschränkung:** Die Scores sind symmetrisch ($s_{ij} = s_{ji}$). Wegen $s_{ij} \le \max(s_{ii}, s_{jj})$ können ein Verb und ein Substantiv nicht *beide* das jeweils andere Wort sich selbst vorziehen. Getrennte Matrizen $\mathbf{W}_q \neq \mathbf{W}_k$ heben beide Einschränkungen auf.

<!--en:28f93a01-->
::: {.slide title="Rechenbeispiel für einen Kopf an einem Satz aus fünf Wörtern"}

<!--en:97c38671-->
[Self-Attention Schritt für Schritt]{.kicker}

<!--en:02917136-->
Jedes Wort in *„Alice gave Bob the book“* erhält einen $9\text{D}$-Vektor. Er verkettet die $4\text{D}$-One-Hot-Kodierung der grammatischen Rolle und die $5\text{D}$-One-Hot-Kodierung des Wortes. Die Kopfbreite beträgt $d_h = 2$:

<!--en:861bbde8-->
| Token | Grammatische Rolle | Abfrage $\mathbf{x}_i\mathbf{W}_q$ | Schlüssel $\mathbf{x}_i\mathbf{W}_k$ |

<!--en:8f58c532-->
| *Alice, Bob* | Person | $(1, -1)$ | $(1, 1)$ |

<!--en:a78be9d0-->
| *gave* | Handlung (Verb) | $(1, 1)$ | $(1, -1)$ |

<!--en:a27ce4fd-->
| *book* | Gegenstand | $(1, -1)$ | $(1, 1)$ |

<!--en:cbe2064b-->
| *the* | Funktionswort | $(0, 0)$ | $(0, 0)$ |

<!--en:f527de4c-->
- **Scores für *gave*:** Das Skalarprodukt seiner Abfrage $(1, 1)$ beträgt $2$ für den Schlüssel $(1, 1)$ jedes Substantivs und $0$ für die Schlüssel von *gave* und *the*.

<!--en:3b0018f6-->
- **Skalierung und Softmax-Funktion:** Nach Division durch $\sqrt{2}$ lauten die Scores $1.41$ für die drei Substantive und $0$ für die übrigen Wörter. Die Gewichte betragen $e^{1.41}/(3e^{1.41}+2) \approx \mathbf{0.29}$ pro Substantiv und $\mathbf{0.07}$ für *gave* und *the*.

<!--en:ea20b190-->
::: {.slide title="Ein Kopf für „Alice gave Bob the book“ im Bild"}

<!--en:064c597f-->
[Geometrie und kausale Maskierung]{.kicker}

<!--en:aba99a9a-->
![Ein Self-Attention-Kopf berechnet die Scores von „gave“ für alle fünf Wörter.](../img/mdl-attention-self-attention.svg){.nostretch width=90%}

<!--en:e9abe0bc-->
::: {.slide title="Berechnung eines Kopfes für „Alice gave Bob the book“"}

<!--en:064c597f-2-->
[Geometrie und kausale Maskierung]{.kicker}

<!--en:92f06108-->
::: {.slide title="Flaschenhals eines Kopfes beim Kopieren beider Werte"}

<!--en:b8e68af2-->
[Informationstheoretische Grenze]{.kicker}

<!--en:cb15d985-->
In unserem Satz erhielt *gave* die Mischung $0.29\,\mathbf{v}_{\textrm{Alice}} + 0.29\,\mathbf{v}_{\textrm{Bob}} + 0.29\,\mathbf{v}_{\textrm{book}}$ mit gleichen Gewichten für die drei Substantive. Wie stark das die Trennung mehrerer Argumente einschränkt, zeigt die **Aufgabe, beide Werte zu kopieren**:

<!--en:bb10351d-->
- **Aufbau:** Zwei Quellpositionen enthalten unabhängige gaußverteilte Werte $\mathbf{v}_1, \mathbf{v}_2 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_v})$ mit festen, rein positionsabhängigen Schlüsseln. Eine Abfrage soll durch lineares Auslesen $\mathbf{t} = [\mathbf{v}_1; \mathbf{v}_2] \in \mathbb{R}^{2d_v}$ ausgeben.

<!--en:141693ee-->
- **Flaschenhals einer einzigen Mischung:** Bei festen Schlüsseln vergibt ein einzelner Kopf feste Gewichte $(\alpha_1, \alpha_2)$. Zum Auslesen steht nur *ein* $d_v$-dimensionaler Vektor $\mathbf{m} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2$ bereit.

<!--en:3cb86014-->
![Ein Kopf liefert eine einzige Mischung, zwei Köpfe liefern zwei linear unabhängige Mischungen.](../img/mdl-attention-one-head-averages.svg){.nostretch width=100%}

<!--en:71066c75-->
::: {.slide title="Satz über den Verlust der halben Varianz bei einem Kopf"}

<!--en:4967cfe6-->
[Beweis der unteren Schranke]{.kicker}

<!--en:48151b9e-->
- **Optimaler Bayes-Schätzer:** Gegeben sei $\mathbf{m} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2$. Der Schätzer für $\mathbf{v}_1$ mit dem kleinsten mittleren quadratischen Fehler lautet $\mathbb{E}[\mathbf{v}_1 \mid \mathbf{m}] = \frac{\alpha_1}{\alpha_1^2 + \alpha_2^2}\,\mathbf{m}$. Es verbleibt die Restvarianz $\frac{\alpha_2^2}{\alpha_1^2 + \alpha_2^2}$.

<!--en:878adbad-->
- **Konstanter Gesamtfehler:** Bei der Schätzung von $\mathbf{v}_2$ verbleibt die Restvarianz $\frac{\alpha_1^2}{\alpha_1^2 + \alpha_2^2}$. Die Restvarianzen summieren sich auf $1$ bei einer Zielvarianz von $2$. Der relative mittlere quadratische Fehler beträgt **$1/2$** (Wurzel $\sqrt{1/2} \approx 0.707$) für *jedes* Gewichtspaar $(\alpha_1, \alpha_2)$.

<!--en:1c8e1201-->
- **Exakte Umkehrung mit zwei Köpfen:** Zwei verschiedene Gewichtungen bilden eine invertierbare $2 \times 2$-Mischmatrix.

<!--en:6d32d664-->
::: {.slide title="Zwei Köpfe kehren die Mischung bis auf Maschinengenauigkeit um"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:4b35ed12-->
Zwei Köpfe mit verschiedenen Gewichtungen (selbst mit weichen wie $(0.88, 0.12)$ und $(0.12, 0.88)$) liefern zwei linear unabhängige Mischungen $[\mathbf{m}_1; \mathbf{m}_2] \in \mathbb{R}^{2d_v}$. Daraus lassen sich die Werte mit `float32`-Genauigkeit linear auslesen:

<!--en:867609d4-->
Um $k$ unabhängige $d_v$-dimensionale Werte zu rekonstruieren, sind zwei Dinge zugleich nötig: (1) **mindestens $k$ linear unabhängige Attention-Gewichtungen** ($\operatorname{rank}(\mathbf{A}) = k$) und (2) **insgesamt mindestens $k\,d_v$ Ausgabekoordinaten** als Platz für die Werte.

<!--en:35d1c339-->
::: {.slide title="Multi-Head-Attention durch parallele Zugriffe in Unterräumen"}

<!--en:63cf87bb-->
[Architektur mit mehreren Köpfen]{.kicker}

<!--en:72ec7b7c-->
- **Projektionen in Unterräume ($i = 1, \ldots, H$):** Jeder Kopf $i$ projiziert $\mathbf{X} \in \mathbb{R}^{n \times d}$ auf die Breite $d_h = d/H$. Dazu dienen die Matrizen $\mathbf{W}_i^{(q)}, \mathbf{W}_i^{(k)}, \mathbf{W}_i^{(v)} \in \mathbb{R}^{d \times d_h}$:
  $$\mathbf{Z}_i = \mathrm{softmax}\!\left(\frac{(\mathbf{X}\mathbf{W}_i^{(q)})(\mathbf{X}\mathbf{W}_i^{(k)})^\top}{\sqrt{d_h}}\right)\mathbf{X}\mathbf{W}_i^{(v)}$$

<!--en:f5f4b316-->
- **Verkettung und Ausgabeprojektion:** Die Ausgaben aller $H$ Köpfe zu einer Matrix der Form $n \times H d_h$ verketten und mit $\mathbf{W}_o \in \mathbb{R}^{H d_h \times d}$ über die Köpfe hinweg mischen:
  $$\textrm{MHA}(\mathbf{X}) = [\mathbf{Z}_1, \ldots, \mathbf{Z}_H]\,\mathbf{W}_o \in \mathbb{R}^{n \times d}$$

<!--en:097d2bc0-->
![Multi-Head-Attention führt H parallele Zugriffe in Unterräumen aus und projiziert die verketteten Ausgaben.](../img/mdl-attention-multi-head.svg){.nostretch width=100%}

<!--en:2ae790b2-->
::: {.slide title="Parameteranzahl und FLOP-Zahl sind unabhängig von H"}

<!--en:0b56ca18-->
[Rechenaufwand]{.kicker}

<!--en:f70d4a42-->
- **Führende Terme (mit $d_h = d / H$):**
  $$\textrm{Params} = 4d^2, \qquad \textrm{FLOPs} = 8nd^2 + 4n^2 d$$

<!--en:652f4034-->
- **Eine Schicht von GPT-2 small ($d = 768, n = 1{,}024$):** $2.36\text{M}$ Parameter, $4.83\text{ GFLOPs}$ für Projektionen und $3.22\text{ GFLOPs}$ für Scores und das Mischen der Werte, sowohl bei $H = 1$ als auch bei $H = 12$.

<!--en:b6e0d5c7-->
- **Warum nicht $H = d$ ($d_h = 1$)?** Bei $d_h = 1$ sind die skalaren Scores $q_i k_j$ monoton in $k_j$. Abfragen können sich daher nur auf die extremalen Schlüssel konzentrieren.

<!--en:08ed6180-->
::: {.slide title="Faltung, Rekurrenz und Attention im Vergleich"}

<!--en:4bd45711-->
[Schichttypen für Sequenzen]{.kicker}

<!--en:3ab708a0-->
| Schichttyp | Rechenaufwand in FLOPs | Sequenzielle Schritte | Maximale Pfadlänge |

<!--en:425f27f4-->
| **1D-Faltung (Kerngröße $k$)** | $O(knd^2)$ | $O(1)$ | $O(n/k)$ |

<!--en:638c75be-->
| **Rekurrenz (RNN)** | $O(nd^2)$ | $O(n)$ | $O(n)$ |

<!--en:93ee534e-->
| **Self-Attention** | $O(nd^2 + n^2d)$ | **$O(1)$** | **$O(1)$** |

<!--en:5aed5527-->
- Self-Attention erreicht beim parallelen Training eine Tiefe von $O(1)$. Der direkte Pfad zwischen zwei Token hat die Länge $O(1)$. Der Term $4n^2d$ übersteigt $8nd^2$ erst für $n > 2d$.

<!--en:370a67c4-->
![Verbindungen, Pfadlänge und Parallelität bei Faltung, Rekurrenz und Self-Attention.](../img/mdl-attention-cnn-rnn-self-attention.svg){.nostretch width=100%}

<!--en:fa2722e1-->
::: {.slide title="Vektorisierung der Köpfe durch Umformen und Transponieren von Tensoren"}

<!--en:db66b284-->
[Kopfachse als Batchdimension]{.kicker}

<!--en:ddaf6310-->
- Anstelle von $H$ getrennten Matrizenmultiplikationen in einer Python-Schleife lassen sich die Projektionen für alle Köpfe auf einmal berechnen, in der Form `(batch, n, H * d_h)`.

<!--en:55afb9a1-->
- **`split_heads`:** Formt die Merkmalsachse zu `(batch, n, H, d_h)` um und vertauscht die Achsen `1` und `2`. In der neuen Form `(batch, H, n, d_h)` dient `H` als Batchdimension.

<!--en:334952a6-->
- **`merge_heads`:** Tauscht die Achsen `1` und `2` zurück und fasst `(H, d_h)` zu `H * d_h` zusammen. Danach folgt die Projektion mit $\mathbf{W}_o$.

<!--en:6af436e6-->
![Aufteilung eines projizierten Tensors der Form (n, d) in H Köpfe der Breite d_h = d/H.](../img/mdl-attention-head-split.svg){.nostretch width=100%}

<!--en:ce33121d-->
::: {.slide title="Implementierung von MultiHeadAttention"}

<!--en:f18ec381-->
[Implementierung des Moduls]{.kicker}

<!--en:ca0b4701-->
Ein Modul deckt sowohl Multi-Head-Attention (`num_kv_heads = num_heads`) als auch Grouped-Query-Attention (`num_kv_heads < num_heads`) ab. Der Konstruktor legt die Größe der Schlüssel- und Wertprojektionen anhand von `num_kv_heads` fest:

<!--en:df85fa8b-->
::: {.slide title="Implementierung von MultiHeadAttention (Fortsetzung)"}

<!--en:f18ec381-2-->
[Implementierung des Moduls]{.kicker}

<!--en:03e5b86b-->
Der Vorwärtsdurchlauf teilt die Projektionen in Köpfe auf, wiederholt jeden Schlüssel-Wert-Kopf für die zugehörige Abfragegruppe und berechnet die Attention:

<!--en:9c538dbd-->
::: {.slide title="Fehlerquelle beim direkten Umformen in (B, H, n, d_h)"}

<!--en:8d193ffc-->
[Unbemerkter Fehler im Tensor]{.kicker}

<!--en:9095a8ae-->
- **Richtig (umformen, dann Achsen 1 und 2 vertauschen):** Der $d$-Vektor jedes Tokens zerfällt in $H$ Blöcke der Breite $d_h$.

<!--en:97d83595-->
- **Fehlerhaft (`X.reshape(B, H, n, d_h)`):** Entlang der Sequenzachse $n$ liegen die Daten zeilenweise im Speicher. Beim direkten Umformen gelangen die ersten $n/H$ *Token* in „Kopf 0“.

<!--en:8c07284f-->
- **Irreführende Übereinstimmung der Form:** Beide Varianten liefern `(B, H, n, d_h)`. Eine Prüfung der Form kann den Fehler daher nicht erkennen.

<!--en:cd3e01aa-->
::: {.slide title="Trainierte Schichten mit mehreren Köpfen beim Kopieren beider Werte"}

<!--en:63718f6d-2-->
[Empirische Überprüfung]{.kicker}

<!--en:34c1f832-->
Training von `MultiHeadAttention` ($d = 16$, Werte der Dimension $d_v = 8$ mit Nullen auf $16$ ergänzt, $4 \times 16^2 = 1{,}024$ Parameter bei allen Modellen) für $2{,}000$ Adam-Schritte für die Aufgabe, beide Werte zu kopieren:

<!--en:b7a9f3b2-->
- **$H = 1$ Kopf:** Der relative Fehler bleibt bei $0.71 \approx \sqrt{1/2}$ stehen, die Attention-Gewichte liegen nahe $(0.50, 0.50)$. Das entspricht unserem Satz.

<!--en:c5aa7129-->
- **$H \in \{2, 4\}$ Köpfe:** Die Köpfe spezialisieren sich von selbst. Einer legt ein Gewicht von $> 0.80$ auf $\mathbf{v}_1$, ein anderer eines von $< 0.20$ auf $\mathbf{v}_1$. Der relative Fehler sinkt bis auf den Rundungsfehler von `float32` ($\sim 10^{-7}$).

<!--en:6e99c590-->
::: {.slide title="Grouped-Query-Attention (GQA) und Multi-Query-Attention (MQA)"}

<!--en:88e40e22-->
[Gemeinsame Schlüssel-Wert-Köpfe]{.kicker}

<!--en:6067d522-->
- **Flaschenhals bei der Generierung:** Beim autoregressiven Dekodieren bleiben frühere Schlüssel und Werte im GPU-Speicher ($2 \cdot H \cdot d_h$ Gleitkommazahlen pro Token und Schicht).

<!--en:5e53a177-->
- **GQA ($H_{\textrm{kv}} < H$) und MQA ($H_{\textrm{kv}} = 1$):** Je $H / H_{\textrm{kv}}$ Query-Köpfe teilen sich einen Schlüssel-Wert-Kopf. Der KV-Cache schrumpft um den Faktor $H / H_{\textrm{kv}}$ (z. B. $36\text{ KiB} \to 12\text{ KiB}$ pro Token bei GPT-2 small mit 16 Bit und $12 \to 4$ KV-Köpfen).

<!--en:e79e3c2e-->
- **Erforderliche Unterraumdimension:** Die gemeinsamen Wertköpfe erfassen $H_{\textrm{kv}} d_h$ Koordinaten pro Token. Zum Kopieren $d_v$-dimensionaler Werte muss daher $H_{\textrm{kv}} d_h \ge d_v$ gelten:

<!--en:11758bf4-->
![Zuordnung von Query-Köpfen zu Schlüssel-Wert-Köpfen bei MHA, GQA und MQA.](../img/mdl-transformers-gqa.svg){.nostretch width=100%}

<!--en:eb302660-->
::: {.slide title="Gemeinsame Schlüssel-Wert-Köpfe beim Kopieren beider Werte"}

<!--en:63a4df4c-->
[Experimente mit GQA und MQA]{.kicker}

<!--en:3d36885e-->
Der Versuch mit $(H, H_{\textrm{kv}}) \in \{(2, 1), (4, 2), (4, 1)\}$ mit $d = 16$ und $d_v = 8$ überprüft die erforderliche Unterraumdimension $H_{\textrm{kv}} d_h \ge d_v$:

<!--en:656fed8b-->
- **Ausreichende Wertbreite ($(2, 1)$ und $(4, 2)$):** In beiden Fällen gilt $H_{\textrm{kv}} d_h = 8 = d_v$. Beide Konfigurationen kopieren die zwei Werte bis auf Maschinengenauigkeit ($\sim 10^{-7}$) und benötigen dafür nur $768$ Parameter anstelle von $1{,}024$.

<!--en:17410334-->
- **Flaschenhals bei den Werten ($(4, 1)$):** Vier Query-Köpfe lernen vier verschiedene Gewichtungen. Gemeinsam stehen ihnen aber nur $H_{\textrm{kv}} d_h = 1 \times 4 = 4 < 8$ Wertkoordinaten zur Verfügung, sodass der Fehler wieder an der Schranke von $0.71$ liegt.

<!--en:cf220479-->
::: {.slide title="Self-Attention und Cross-Attention im Vergleich"}

<!--en:0bb26711-->
[Herkunft der Eingaben]{.kicker}

<!--en:2bb81190-->
- **Self-Attention (`attention(X, X, X)`):** Abfragen, Schlüssel und Werte stammen aus derselben Sequenz $\mathbf{X} \in \mathbb{R}^{n \times d}$. Das ergibt eine $n \times n$-Score-Matrix.

<!--en:1a3e9b40-->
- **Cross-Attention (`attention(A, B, B)`):** Die Abfragen stammen aus der Zielsequenz $\mathbf{A} \in \mathbb{R}^{n \times d}$, die Schlüssel und Werte aus der Quellsequenz $\mathbf{B} \in \mathbb{R}^{m \times d}$. Das ergibt eine $n \times m$-Score-Matrix.

<!--en:2d72d086-->
- **Invarianz gegenüber Permutationen der Quelle:** Bei Cross-Attention ohne Maske lassen sich die Quellzeilen gemäß $\mathbf{B} \mapsto \boldsymbol{\Pi}\mathbf{B}$ umordnen. Die Ausgabe bleibt für jede Abfrage aus $\mathbf{A}$ gleich.

<!--en:21aa31d3-->
::: {.slide title="Verständnisfragen zu Köpfen, FLOPs und GQA-Kapazität"}

<!--en:929b7671-->
[Verständnis überprüfen]{.kicker}

<!--en:805ea78e-->
Fragen zur Skalierung und zu den Kapazitätsgrenzen der Multi-Head-Attention:

<!--en:5f1d915c-->
1. **Gleicher Rechenaufwand:** Eine Schicht habe die innere Breite $d = 4{,}096$ und $H = 32$ Köpfe ($d_h = 128$). Bei welcher Sequenzlänge $n$ entspricht die quadratische FLOP-Zahl für Scores und Wertmischung ($4n^2 d$) der linearen FLOP-Zahl für die QKVO-Projektionen ($8nd^2$)? Verschiebt sich dieser Punkt, wenn $H$ auf $64$ steigt?

<!--en:ef42c541-->
2. **Allgemeine Schranke beim Kopieren von $k$ Werten:** Eine einzelne Abfrage verwendet $H$ Köpfe mit fester Attention-Gewichtsmatrix $\mathbf{A} \in \mathbb{R}^{H \times k}$, um $k$ unabhängige gaußverteilte Werte $[\mathbf{v}_1; \ldots; \mathbf{v}_k]$ abzurufen. Wie klein kann der relative quadratische Fehler werden?

<!--en:9fe3f06f-->
3. **Invarianz gegenüber der Wortreihenfolge:** Nach dem Vertauschen von *Alice* und *Bob* lautet der Satz *„Bob gave Alice the book“*. Ändert sich die Ausgabe der Self-Attention ohne Maske an der Position von *gave*, wenn die Schicht $H = 12$ Köpfe hat?

<!--en:557b98b6-->
**Lösungen:**

<!--en:95eb8e22-->
1. **$n = 2d = 8{,}192$ Token** ($4n^2d = 8nd^2 \iff n = 2d$). Steigt $H$ auf $64$, bleiben beide führenden FLOP-Zahlen gleich und damit auch die Sequenzlänge, bei der sie übereinstimmen.

<!--en:205a0e31-->
2. **$1 - \operatorname{rank}(\mathbf{A}) / k$** (sofern die Ausgabe mit $H d_h \ge k d_v$ genügend Platz bietet). Für $H = 1$ und $k = 2$ folgt aus $\operatorname{rank}(\mathbf{A}) = 1$ der Fehler $1 - 1/2 = 1/2$.

<!--en:abe3e9d3-->
3. **Nein.** Beim Vertauschen zweier Token wechseln in jedem Kopf ihre Schlüssel zusammen mit ihren Werten die Position. Die gewichtete Summe bei *gave* bleibt also gleich. Das motiviert die Einführung von Positionskodierungen (§{sec_positional-information}).

<!--en:0492d443-->
::: {.slide title="Zusammenfassung der Self-Attention mit mehreren Köpfen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:f1d94201-->
| Konzept / Variante | Mathematische Form / Dimension | Parameter und FLOPs | Fähigkeit / Flaschenhals |

<!--en:b08a16e8-->
| **Ein Kopf ($H=1$)** | $\mathrm{softmax}\!\left(\frac{\mathbf{X}\mathbf{W}_q(\mathbf{X}\mathbf{W}_k)^\top}{\sqrt{d_h}}\right)\mathbf{X}\mathbf{W}_v$ | $3d d_h$ Parameter, $O(nd d_h + n^2 d_h)$ | 1 Mischung pro Abfrage, mittlerer quadratischer Fehler $\ge 1/2$ beim Kopieren von 2 Werten |

<!--en:9bf608b4-->
| **Getrennte Matrizen $\mathbf{W}_q \neq \mathbf{W}_k$** | Asymmetrische Score-Matrix $\mathbf{Q}\mathbf{K}^\top$ | Entkoppelt die Unterräume für Abfragen und Schlüssel | Die Schranke $s_{ij} \le \sqrt{s_{ii}s_{jj}}$ für positiv semidefinite Score-Matrizen bei gemeinsamer Abbildung entfällt |

<!--en:cdfc0045-->
| **Multi-Head-Attention (MHA)** | $[\mathbf{Z}_1, \ldots, \mathbf{Z}_H]\mathbf{W}_o$, $d_h = d/H$ | **$4d^2$ Parameter, $8nd^2 + 4n^2d$ FLOPs** | $H$ verschiedene Mischungen pro Abfrage bei Kosten unabhängig von $H$ |

<!--en:3d0b6926-->
| **Grouped-Query-Attention (GQA)** | $H$ Query-Köpfe teilen sich $H_{\textrm{kv}}$ KV-Köpfe | $(2 + 2 H_{\textrm{kv}}/H)d^2$ Parameter | Verkleinert den KV-Cache bei der Generierung um Faktor $H/H_{\textrm{kv}}$, erfordert $H_{\textrm{kv}}d_h \ge d_v$ |

<!--en:a1ea2100-->
| **Multi-Query-Attention (MQA)** | Alle $H$ Query-Köpfe teilen sich $H_{\textrm{kv}} = 1$ KV-Kopf | $(2 + 2/H)d^2$ Parameter | Maximale Verkleinerung des KV-Caches um den Faktor $H\times$ |

<!--en:a2172bd7-->
| **Cross-Attention** | `attention(A, B, B)` ($n$ Abfragen, $m$ Schlüssel) | $O((n+m)d^2 + nmd)$ FLOPs | Leitet Informationen von der Quelle $\mathbf{B}$ zum Ziel $\mathbf{A}$ |
