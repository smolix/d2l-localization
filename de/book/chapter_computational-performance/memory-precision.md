<!--en:9804346c-->
# Speicher und Genauigkeit

<!--en:1b1b676a-->
In den vorigen Abschnitten ging es darum, wie schnell ein Trainingsschritt abläuft. Reicht der GPU-Speicher dafür nicht aus, lässt sich der Schritt überhaupt nicht ausführen. Wenn wir unser GPT mit 1 Milliarde Parametern mit Adam in fp32 trainieren, speichern wir pro Parameter vier fp32-Werte: das Gewicht, seinen Gradienten und die beiden Momentschätzungen von Adam. Damit sind bereits $16N \approx 15.1$ GiB der 24 GiB einer RTX 4090 belegt, bevor wir das erste Token einlesen. Die verbleibenden rund 9 GiB müssen die *Aktivierungen* aufnehmen, die der Vorwärtsdurchlauf für die Backpropagation speichert. Sie belegen mehrere Megabyte pro Token. So passen nur zwei oder drei Sequenzen mit je 1024 Token auf die Karte. Große Sprachmodelle trainiert man dagegen mit globalen Batches von einer halben Million bis zu mehreren Millionen Token :cite:`brown2020language`.

<!--en:08d5b1b3-->
Der Speicherbedarf eines Trainingsschritts besteht aus einem festen Anteil für die Parameter, der proportional zu $N$ ist, und einem Anteil für die Aktivierungen, der proportional zur Tokenanzahl $n$ ist. Wir berechnen beide Anteile für unser GPT und vergleichen das Ergebnis mit dem gemessenen Spitzenbedarf. Anschließend untersuchen wir drei Verfahren, die beim Aktivierungsspeicher oder bei der Mikrobatchgröße ansetzen: Bei *gemischter Genauigkeit* (engl. mixed precision) rechnen wir mit 16 Bit. Beim *Aktivierungs-Checkpointing* berechnen wir Aktivierungen im Rückwärtsdurchlauf neu, statt sie zu speichern. Bei der *Gradientenakkumulation* (engl. gradient accumulation) zerlegen wir einen großen globalen Batch in Mikrobatches, die wir nacheinander verarbeiten. Bei allen drei Verfahren belegt der Parameterzustand weiterhin $16N$ Byte. Wie wir diesen Zustand auf mehrere GPUs verteilen, behandelt :numref:`sec_sharding`.

<!--en:4706dd73-->
## Speicherbedarf eines Trainingsschritts

<!--en:85e21b5b-->
### Speicherbedarf pro Parameter

<!--en:0858951e-->
Wie viel Speicher ein Parameter benötigt, hängt vom Optimierer ab. Einfacher SGD in fp32 speichert Gewicht und Gradient (8 Byte pro Parameter). Bei SGD mit Momentum kommt die Geschwindigkeit hinzu (12 Byte). Adam speichert zusätzlich zwei Momentschätzungen (**16 Byte pro Parameter**, :numref:`subsec_memory_during_training`). Beim Training mit gemischter Genauigkeit bleiben alle vier Werte in fp32. Während des Schritts kommt eine 16-Bit-Kopie des Gewichts hinzu (**18 Byte pro Parameter**).

<!--en:b0871f33-->
Unser Referenzmodell ist `d2l.GPT` (:numref:`sec_gpt`) mit der Breite $d = 2048$, 16 Köpfen, $L = 18$ SwiGLU-Blöcken mit vorgeschalteter Normalisierung, einem Vokabular der Größe $V = 50{,}257$ und erlernten Positionseinbettungen. Jeder Block enthält rund $12d^2 \approx 50$ Millionen Parameter (rund 0.9 Milliarden in 18 Blöcken, dazu rund 0.1 Milliarden in der auch für die Ausgabe genutzten Tokeneinbettung). Auf einem Meta-Gerät, das nur die Tensorformen erfasst, können wir $N$ bestätigen, ohne 4 GB für Gewichte zu reservieren (:numref:`subsec_count_without_alloc`):

<!--en:f27a828b-->
### Speicherbedarf pro Token

<!--en:aa99f4a4-->
Im Vorwärtsdurchlauf speichern die Schichten die Zwischenaktivierungen, die der Rückwärtsdurchlauf benötigt. Im Rückwärtsdurchlauf geben sie diese Aktivierungen Schicht für Schicht frei, sobald sie die Parametergradienten berechnet haben (:numref:`fig_memory_anatomy`). Der Speicherbedarf erreicht daher zu Beginn des Rückwärtsdurchlaufs sein Maximum.

<!--en:eb40a02d-->
![Speicherbedarf eines Trainingsschritts mit Adam in Einheiten von $4N$ Byte. Die Gewichte ($4N$) und die Momentschätzungen von Adam ($8N$) bleiben den ganzen Schritt über im Speicher. Im Vorwärtsdurchlauf sammeln sich Aktivierungen an. Im Rückwärtsdurchlauf werden sie frei, während die Gradienten ($4N$) hinzukommen. Der Spitzenbedarf liegt zu Beginn des Rückwärtsdurchlaufs.](../img/mdl-perf-memory-anatomy.svg)

<!--en:88d1591a-->
Drei Regeln bestimmen, was ein Transformer-Block für die Backpropagation speichert:

<!--en:106d41ae-->
1. Eine lineare Schicht $\mathbf{y} = \mathbf{W}\mathbf{x}$ speichert ihre Eingabe $\mathbf{x}$, denn es gilt $\nabla_{\mathbf{W}} = (\nabla_{\mathbf{y}})\mathbf{x}^\top$.

<!--en:ef0dbfb3-->
2. Eine elementweise Nichtlinearität $f$ speichert ihre Eingabe oder ihre Ausgabe.

<!--en:fd2cd25b-->
3. Ein elementweises Produkt $\mathbf{p} \odot \mathbf{q}$ speichert beide Faktoren.

<!--en:1a34f87a-->
Wir zählen die Werte pro Token in Einheiten der Breite $d = 2048$:

<!--en:6a22f190-->
* **Attention-Teilschicht:** Die RMSNorm vor der Attention speichert ihre Eingabe ($d$), die QKV-Projektion die normalisierte Ausgabe ($d$). Die fusionierte kausale Attention speichert $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ ($3d$) und ihre Ausgabe ($d$). Hinzu kommt nur ein skalarer logarithmischer Normierungsfaktor pro Kopf und Token anstelle der $T \times T$-Score-Matrix. Die Ausgabeprojektion speichert die Attention-Ausgabe ($0$ bei gemeinsam genutztem Speicher oder $d$, wenn das Zusammenführen der Köpfe eine Kopie erfordert). Insgesamt ergibt das $6d\text{--}7d$.

<!--en:618cc827-->
* **SwiGLU-FFN-Teilschicht:** Die RMSNorm vor dem FFN speichert ihre Eingabe ($d$). Gate- und Aufwärtsprojektion nutzen dieselbe normalisierte Ausgabe ($d$). Für $\textrm{SiLU}(\mathbf{g})
  \odot \mathbf{u}$ und die Abwärtsprojektion werden das Gate $\mathbf{g}$, das aktivierte Gate $\textrm{SiLU}(\mathbf{g})$, das Ergebnis der Aufwärtsprojektion $\mathbf{u}$ und das Produkt der beiden letzten Vektoren gespeichert. Zusammen enthalten diese vier Vektoren 4 × $\tfrac{8}{3}d \approx 10.7d$ Werte. Insgesamt ergibt das rund $12.7d$.

<!--en:0a944e88-->
Insgesamt speichert ein fusionierter Block $18.7d\text{--}19.7d \approx \mathbf{20d}$ Werte pro Token (**160 KiB** in fp32 bei $d = 2048$, gegenüber $16d$ bei GPT-3-Blöcken ohne Gate in :citet:`Korthikanti.Casper.Lym.ea.2022`). Die fusionierte Attention berechnet ihre $T \times T$-Gewichte im Rückwärtsdurchlauf direkt auf dem Chip neu. Daher wächst keine der hier gezählten Größen mit der Sequenzlänge $T$. Schließlich speichert der Kreuzentropieverlust $V = 50{,}257$ logarithmierte Wahrscheinlichkeiten in fp32 pro Token (**196 KiB**), mehr als ein ganzer Transformer-Block.

<!--en:8e38d508-->
Um diese Rechnung empirisch zu prüfen, definieren wir zunächst `saved_bytes`. Die Funktion summiert die Bytes der Tensoren, die für den Rückwärtsdurchlauf von `gpt_loss` im Speicher bleiben:

<!--en:af24509a-->
Wir werten `saved_bytes` für GPT-Modelle mit 1 und mit 2 Blöcken bei den Batchgrößen $b \in \{1, 2\}$ aus. Aus den Differenzen bestimmt `bytes_per_token`, wie viel Aktivierungsspeicher pro Token auf einen zusätzlichen Transformer-Block entfällt (`block`) und wie viel auf Einbettungen, Ausgabekopf und Verlust (`head`):

<!--en:411b2c65-->
Da `scaled_dot_product_attention`, RMSNorm und SiLU jeweils als ein einziger fusionierter Kernel laufen, speichert PyTorch genau das theoretische Minimum: $18.7d$ Werte (149 KiB) pro Block und 212 KiB für Einbettung, Ausgabeprojektion und Verlust.

<!--en:be07bff6-->
Vor der XLA-Kompilierung erfasst `jax.vjp` pro Block $29.0d$ Werte (232 KiB). Der Grund ist, dass `nnx.RMSNorm` und `jax.nn.silu` aus mehreren elementweisen Operationen bestehen. Deren Zwischenergebnisse bleiben als Arrays in den ursprünglichen VJP-Residuen erhalten. Ob das kompilierte Programm sie alle behält, entnehmen wir weiter unten dem Speicherplan von XLA.

<!--en:0b255017-->
### Speicherbudget

<!--en:7a2b720e-->
Wir addieren den Speicher für den Parameterzustand ($16N$ Byte, wenn die Gradienten über mehrere Mikrobatches erhalten bleiben), $a$ Byte pro Token und Block sowie $a_\textrm{out}$ Byte pro Token für Einbettung, Ausgabekopf und Verlust. So erhalten wir das Speicherbudget für einen Schritt mit $n$ Token:

<!--en:a6e49bce-->
Für unser GPT mit 18 Blöcken auf einer Karte mit 24 GiB ($16N = 15.1$ GiB für den Parameterzustand, 8.9 GiB für Aktivierungen) ergibt die reine Zählung folgende Vorhersage:

<!--en:da210c66-->
$La + a_\textrm{out} \approx 2.8$ MiB/Token (rund 3200 Token oder 3 Sequenzen mit je 1024 Token).

<!--en:0ce8986b-->
$La + a_\textrm{out} \approx 4.3$ MiB/Token (rund 2100 Token oder 2 Sequenzen mit je 1024 Token).

<!--en:61553658-->
### Gemessener Spitzenbedarf

<!--en:fe8cd518-->
Wie viel Speicher belegt der Rückwärtsdurchlauf vorübergehend zusätzlich zu den gespeicherten Aktivierungen? Dazu messen wir bei einer Variante unseres GPT mit 2 Blöcken den Spitzenbedarf oberhalb des Gewichtsspeichers für 1024 bis 8192 Token.

<!--en:bc769094-->
Bei kleinen Tokenanzahlen fällt der feste Zusatzbedarf noch ins Gewicht. Sobald er vernachlässigbar ist, wächst der Spitzenbedarf proportional zur Tokenanzahl. Den Zuwachs pro Token vergleichen wir mit unserer Rechnung.

<!--en:ba423f7c-->
Der gemessene Spitzenbedarf wächst um etwa 905 KiB pro Token, also um 393 KiB mehr als die errechneten 511 KiB. Dieser Mehrbedarf entsteht zu Beginn des Rückwärtsdurchlaufs, solange noch alle Aktivierungen des Vorwärtsdurchlaufs im Speicher liegen. Der Rückwärtsdurchlauf der Kreuzentropie legt je einen Gradienten bezüglich der logarithmierten Wahrscheinlichkeiten und der Logits an. Diese beiden temporären Tensoren enthalten je $V = 50{,}257$ fp32-Werte pro Token ($8V \approx 393$ KiB/Token zusätzlich zu den gezählten gespeicherten Tensoren).

<!--en:6bbad403-->
Zwischen 4096 und 8192 Token wächst der von XLA geplante Spitzenbedarf um etwa 690 KiB pro Token. Das entspricht den 692 KiB aus der Zählung der ursprünglichen VJP-Residuen. Das kompilierte Programm benötigt also pro Token so viel Speicher, wie `jax.vjp` erfasst hat. Beim Verlust entsteht kein messbarer Mehrbedarf, sodass wir den errechneten Wert als Budget verwenden können.

<!--en:aa013246-->
Mit der PyTorch-API für den Verlauf der Speicherbelegung können wir alle Speicheranforderungen und -freigaben eines Adam-Schritts mit 4096 Token sichtbar machen:

<!--en:c11a5f45-->
Wir summieren die Speicheränderungen in Byte mit ihrem Vorzeichen (`alloc` minus `free_completed`) über `trace` und unterteilen den Verlauf anhand von `forward_end` und `backward_end`. So sehen wir, wie viel Speicher in jeder der drei Phasen des Schritts jeweils belegt ist:

<!--en:9e9f3465-->
Die Aufzeichnung beginnt bei $12N$ (Gewichte und zwei Adam-Momente). Die Belegung steigt im Vorwärtsdurchlauf von Block zu Block linear an und zeigt Spitzen bei den Logits und zu Beginn des Rückwärtsdurchlaufs. Anschließend sinkt sie stufenweise, während Speicher für Aktivierungen frei wird und Speicher für Gradienten hinzukommt.

<!--en:79bbe576-->
## Gemischte Genauigkeit

<!--en:cd8e49be-->
Beim Training mit gemischter Genauigkeit (:numref:`sec_numerics`) bleiben Hauptgewichte, Gradienten, Momentschätzungen des Optimierers, Normalisierungen, Residualstrom und Verlust in fp32. Matrixmultiplikationen und Zwischenaktivierungen berechnen wir dagegen mit 16-Bit-Gleitkommazahlen. Dafür eignet sich bf16, dessen 8-Bit-Exponent denselben Wertebereich abdeckt wie bei fp32 und das daher ohne Loss-Scaling auskommt, oder fp16 mit Loss-Scaling (:numref:`subsec_loss_scaling`).

<!--en:a4c55054-->
Die beiden Eingaben aus dem Residualstrom für die Normalisierungen eines Blocks bleiben in fp32 (4 Byte pro Wert). Von den rund $20d$ gespeicherten Werten pro Block wechseln die übrigen 18 Vektoren der Blockbreite zu bf16 (2 Byte pro Wert). Damit sinkt der Speicherbedarf der Aktivierungen eines Blocks auf etwa das $(2 \cdot 4 + 18 \cdot 2) / (20 \cdot 4) = 0.55$-Fache des fp32-Bedarfs:

<!--en:ab3fc123-->
Der Aktivierungsspeicher eines Blocks sinkt auf das $0.58\text{--}0.60$-Fache des fp32-Werts. Während des Schritts halten wir jedoch bf16-Kopien der Gewichte im Speicher. Dadurch steigt der feste Parameterspeicher von $16N = 15.1$ GiB auf $18N = 17.0$ GiB. Für Aktivierungen bleiben nur 7.0 GiB. Die Einsparung bei den Aktivierungen und der Mehrbedarf für die Gewichtskopien heben sich weitgehend auf. Auf eine Karte mit 24 GiB passen so nur 15–25% mehr Token.

<!--en:e4019812-->
Gemischte Genauigkeit mit bf16 verbessert vor allem die Geschwindigkeit. Auf der RTX 4090 erreichen die Tensor Cores mit bf16 einen Durchsatz von 165 TFLOP/s, doppelt so viel wie die 82.6 TFLOP/s mit tf32 und fp32. Wir untersuchen getrennt, was die Tensor Cores bewirken (tf32 gegenüber skalarem fp32) und was die 16-Bit-Operanden bewirken (bf16 gegenüber tf32). Dazu vermessen wir einen SGD-Schritt eines rechenbegrenzten dreischichtigen MLP der Breite 8192 mit einem Batch aus 2048 Beispielen. Bei $1.1 \times 10^{12}$ FLOPs pro Schritt ergibt das Roofline-Modell eine Mindestdauer von 13.3 ms bei 82.6 TFLOP/s und 6.7 ms bei 165 TFLOP/s. Hinzu kommen rund 1.3 ms für die bandbreitenbegrenzte SGD-Aktualisierung.

<!--en:e44d74d0-->
Als Nächstes zählen wir für die Matrixprodukte im Vorwärts- und Rückwärtsdurchlauf des MLP die Multiplikationen mit anschließender Addition (`macs`) und die Gleitkommaoperationen insgesamt (`flops`). Wir definieren `report`, um für jedes Zahlenformat die Laufzeit in Millisekunden und den Durchsatz in TFLOP/s zu messen:

<!--en:c8b05511-->
Nun führen wir `report` für reines skalares `'fp32'`, für `'tf32'` auf Tensor Cores und für gemischte Genauigkeit mit `'bf16'` aus:

<!--en:6f1afe07-->
Reines fp32 erreicht auf skalaren Einheiten rund 60% der Spitzenleistung von 82.6 TFLOP/s, weil jede Multiplikation mit Addition und jede Adressberechnung um Plätze bei der Befehlsausgabe konkurrieren. Mit tf32 erreichen die Tensor Cores rund 90% derselben Spitzenleistung. Dadurch sinkt die Schrittzeit um ein Drittel, obwohl die gespeicherten Datentypen gleich bleiben. Der Wechsel von tf32 zu bf16 beschleunigt den Schritt nochmals um den Faktor $1.5\text{--}1.75\times$. Die Laufzeit nähert sich der Untergrenze einschließlich Aktualisierung von $6.7 + 1.3 = 8.0$ ms, da die SGD-Aktualisierung in fp32 und die Typumwandlungen der Gewichte weiterhin gleich viel Zeit beanspruchen.

<!--en:bbf3745d-->
## Aktivierungs-Checkpointing

<!--en:ad36e6e4-->
Damit ein Vielfaches der bisherigen Tokenanzahl auf die Karte passt, muss der Schritt weniger Aktivierungen speichern. Beim *Aktivierungs-Checkpointing* (:numref:`subsec_activation_checkpointing`, :cite:`Chen.Xu.Zhang.ea.2016`) speichern wir im Vorwärtsdurchlauf nur die Eingabe jedes Transformer-Blocks ($d$ fp32-Werte oder $4d$ Byte pro Token). Im Rückwärtsdurchlauf berechnen wir die internen Aktivierungen Block für Block neu. Das kostet einen zusätzlichen Vorwärtsdurchlauf (rund 33% mehr Rechenaufwand, also $8N$ FLOPs/Token statt $6N$). Dafür sinkt der Spitzenbedarf pro Token über $L$ Blöcke von $L a + a_\textrm{out} + \Delta$ auf höchstens

<!--en:7bc14ba6-->
Dabei bezeichnet $a$ den Aktivierungsspeicher des einen Blocks, den wir gerade neu berechnen. $\Delta$ gibt an, um wie viel der gemessene Spitzenbedarf den errechneten Wert übersteigt (:numref:`subsec_mp-measuring`).

<!--en:c69ad1b9-->
Beim blockweisen Checkpointing speichern die 18 Blöcke zusammen nur $18 \times 8 = 144$ KiB/Token, weniger als Ausgabeschicht und Verlust allein. Damit steigt das Tokenbudget auf mehr als das Dreifache (10 000–15 000 Token). Mit Checkpointing passen in bf16 sogar etwas *weniger* Token in den Speicher als in fp32. Die zusätzlichen $2N$ Byte für bf16-Gewichtskopien überwiegen die Einsparung bei den Aktivierungen des einzigen neu berechneten Blocks. Nun beanspruchen die Ausgabe-Logits und der Kreuzentropieverlust den größten Teil des Speichers pro Token. Berechnen wir die Ausgabeprojektion und den Verlust abschnittsweise für Gruppen von Token in einem eigenen Checkpoint, entfällt dieser Engpass bei nur $2dV / (6N) \approx 3\%$ zusätzlichen FLOPs (Aufgabe :exref:`ex_mp-chunked-loss`).

<!--en:3c1a4739-->
Vermessen wir das blockweise Checkpointing für 4 aufeinanderfolgende Blöcke mit zwei Sequenzen zu je 1024 Token. Für die vier Blockeingaben und die abschließende Ausgabe erwarten wir zusammen $5 \times 4d = 40$ KiB/Token:

<!--en:05212c61-->
Mit Checkpointing belegen die gespeicherten Tensoren genau die vorhergesagten 40 KiB/Token. Der Spitzenbedarf für Aktivierungen sinkt deutlich, und die Schrittzeit steigt um rund 25% (unter der Obergrenze von 33% für einen zusätzlichen Vorwärtsdurchlauf). Bei der *selektiven Neuberechnung* (engl. selective recomputation) :cite:`Korthikanti.Casper.Lym.ea.2022` speichern wir die Ergebnisse der kostspieligen Matrixmultiplikationen. Statt eines ganzen Blocks berechnen wir nur die günstigen elementweisen Aktivierungen wie das Gate-Produkt von SwiGLU neu (Aufgabe :exref:`ex_mp-selective`).

<!--en:5e8834c9-->
## Gradientenakkumulation

<!--en:1f014ea7-->
Mit der *Gradientenakkumulation* erreichen wir einen globalen Batch $B_\textrm{global} = m \, B_\textrm{micro}$ mit Hunderttausenden von Token, ohne mehr Aktivierungen zu speichern. Wir teilen den globalen Batch in $m$ Mikrobatches, die wir nacheinander verarbeiten. Dabei akkumulieren wir ihre Gradienten, indem wir den mittleren Verlust jedes Mikrobatches vor der Gradientenberechnung durch $m$ teilen. Erst am Ende führen wir einen einzigen Optimiererschritt aus.

<!--en:4cb30a07-->
Die Gradientenakkumulation liefert rechnerisch dasselbe Ergebnis wie ein einziger großer Batch. Nach dem ersten Mikrobatch lesen wir jedoch für jeden weiteren den bisherigen Gradientenakkumulator und den neuen Gradienten und schreiben ihre Summe ($12N$ Byte Speicherverkehr). Die Verarbeitung des Mikrobatches selbst erfordert $6Nn$ FLOPs für $n$ Token. Bezogen auf die Rechenzeit bei Spitzenleistung kostet dieser Speicherverkehr einen zusätzlichen Anteil von

<!--en:45e40acb-->
Dabei ist $I^* = P/\beta$ der Knickpunkt der Roofline (:numref:`sec_perf_model`). Bei tf32 ($I^* = 82$) und Mikrobatches mit $n = 1{,}024$ Token beträgt der Mehraufwand rund 16% für jeden der $m - 1$ Mikrobatches, deren Gradienten wir hinzuaddieren. Bei $n = 4{,}096$ Token sinkt er auf 4%.

<!--en:bf3c5453-->
An unserem GPT mit 2 Blöcken prüfen wir die numerische Übereinstimmung und messen den Speicher- und Zeitaufwand. Dazu verarbeiten wir 4096 Token einmal als 1 Batch und einmal als $m = 4$ Mikrobatches mit je 1024 Token:

<!--en:31bd2f6f-->
Der Spitzenbedarf oberhalb des Gewichtsspeichers sinkt ungefähr auf die Hälfte. Die $4N = 0.77$ GiB für die Gradienten bleiben über die Mikrobatches hinweg belegt, während der Aktivierungsspeicher auf ein Viertel sinkt. Die Schrittzeit steigt um 10–20%, weil die Additionen im Akkumulator hinzukommen und die Tensor Cores bei kleineren Mikrobatches etwas weniger effizient arbeiten. Optimal ist daher immer der größte Mikrobatch, der in den Speicher passt.

<!--en:e9e15d84-->
## Wahl des Verfahrens zur Speichernutzung

<!--en:f6230ec4-->
:numref:`tab_mp-techniques` zeigt, welches Verfahren sich für welchen Engpass eignet.

<!--en:f18fdde4-->
:Verfahren nach Engpass, mit ihrer Wirkung auf :eqref:`eq_mp-budget` und ihren Kosten.

<!--en:1f1ef127-->
| Engpass | Verfahren | Wirkung auf den Speicherbedarf | Kosten |

<!--en:7dea8a7f-->
| Matrixprodukte bestimmen die Schrittzeit | gemischte Genauigkeit mit bf16 | Aktivierungen $\times 0.6$, fester Zustand $+2N$ für Gewichtskopien | Rundung auf 8 Bit Signifikandengenauigkeit (fp16 benötigt zudem Loss-Scaling) |

<!--en:e53d652a-->
| Ein sinnvoll großer Mikrobatch passt nicht in den Speicher | Aktivierungs-Checkpointing | gespeicherte Aktivierungen pro Block von $a$ auf $4d$ Byte/Token | ein zusätzlicher Vorwärtsdurchlauf (rund 25–33% mehr Zeit) |

<!--en:bb4dd239-->
| Ausgabeschicht und Verlust dominieren | abschnittsweiser Kreuzentropieverlust | $a_\textrm{out}$ und $\Delta$ jeweils nur für eine Gruppe von Token | ein zusätzlicher Durchlauf durch die Ausgabeprojektion (rund 3% mehr FLOPs) |

<!--en:a1b222f3-->
| Der globale Batch übersteigt die Kapazität eines Mikrobatches | Gradientenakkumulation | keine Änderung, der globale Batch wächst auf das $m$-Fache | Speicherverkehr des Akkumulators $2I^*/n$, erneutes Lesen der Gewichte für jeden Mikrobatch |

<!--en:c352707d-->
| Schon der Parameterzustand von $16N$ Byte füllt die Karte | Partitionierung von Gewichten, Gradienten und Optimiererzustand (:numref:`sec_sharding`) | $16N/k$ auf jeder der $k$ GPUs | kollektive Kommunikation zwischen GPUs |

<!--en:f5931152-->
Bei unserem GPT mit 1 Milliarde Parametern passen mit gemischter Genauigkeit in bf16 zwei oder drei Sequenzen zu je 1024 Token ohne Checkpointing auf eine einzelne RTX 4090. Akkumulieren wir 256 Mikrobatches mit je zwei Sequenzen, erreichen wir einen globalen Batch mit 524 288 Token. Bei $6N$ FLOPs/Token benötigt ein globaler Schritt jedoch $3.2 \times 10^{15}$ FLOPs. Auf einer Karte dauert das mindestens 19 s pro Schritt, also mehr als drei Wochen für 100 000 Schritte (52 Milliarden Token). Eine einzelne GPU bietet genug Speicher für einen Schritt unseres GPT. Um das Training abzuschließen, brauchen wir jedoch mehrere GPUs (:numref:`sec_multi_gpu`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:6d3927bb-->
Der Speicherbedarf eines Trainingsschritts setzt sich aus einem festen Parameteranteil ($16N$ Byte für Adam in fp32, $18N$ mit bf16-Gewichtskopien) und einem Aktivierungsanteil pro Token zusammen. Was gespeichert wird, folgt aus drei Ableitungsregeln: Lineare Schichten behalten ihre Eingaben, elementweise Funktionen ihre Eingaben oder Ausgaben und Produkte beide Faktoren. So speichert ein fusionierter SwiGLU-Transformer-Block rund $20d$ Werte pro Token. Der Kreuzentropieverlust speichert zusätzlich $V$ logarithmierte Wahrscheinlichkeiten. Temporäre Tensoren für den Rückwärtsdurchlauf der Verlustfunktion können den Spitzenbedarf über diesen errechneten Wert heben. Deshalb messen wir ihn. Gemischte Genauigkeit beschleunigt Matrixprodukte auf Tensor Cores, während die Tokenkapazität nahezu gleich bleibt. Aktivierungs-Checkpointing speichert nur die Blockeingaben ($4d$ Byte/Token) und kostet rund 25% mehr Zeit. Durch Gradientenakkumulation bilden wir große globale Batches aus Mikrobatches, die in den Speicher passen. Der zusätzliche Speicherverkehr des Akkumulators kostet einen Zeitanteil von $2I^*/n$.

<!--en:2294243a-->
## Übungen

<!--en:6a9ac8d8-->
1. **Zeitpunkt des Spitzenbedarfs.** In :numref:`fig_memory_anatomy` erreicht der Speicherbedarf eines Trainingsschritts zu Beginn des Rückwärtsdurchlaufs sein Maximum. Betrachten Sie $n$ identische, aufeinanderfolgende Schichten. Die Parameter jeder Schicht belegen $w$ Byte, und ihr Vorwärtsdurchlauf speichert $a$ Byte an Aktivierungen. Nehmen Sie an, dass zu Beginn des Rückwärtsdurchlaufs noch keine Gradienten existieren. Jede Schicht gibt im Rückwärtsdurchlauf ihre $a$ Byte an Aktivierungen frei und belegt $w$ Byte für Parametergradienten.
    1. Geben Sie an, wie viele Byte Aktivierungen und Gradienten belegen, nachdem der Rückwärtsdurchlauf $j$ der $n$ Schichten verarbeitet hat. Zeigen Sie, dass der Spitzenbedarf für $a > w$ am Anfang des Rückwärtsdurchlaufs liegt und für $a < w$ am Ende.
    1. Ein Block unseres GPT enthält rund $12d^2$ Parameter und speichert rund $20d$ fp32-Werte pro Token. Oberhalb welcher Tokenanzahl pro Schritt liegt der Spitzenbedarf zu Beginn des Rückwärtsdurchlaufs?
    1. Bei der Gradientenakkumulation über $m$ Mikrobatches bleiben die Gradienten des ersten Mikrobatches im Speicher, während die weiteren Mikrobatches verarbeitet werden. Bestimmen Sie den Spitzenbedarf mit Akkumulation. Für welche Werte von $m$ liegt er unter dem Spitzenbedarf eines Schritts mit dem gesamten Batch?
   :exlabel:`ex_mp-peak`

<!--en:39dc01fc-->
2. [code] **Speicherbedarf des Optimiererschritts.** Erstellen Sie das GPT mit zwei Blöcken aus :numref:`subsec_mp-measuring` mit drei Konfigurationen von `torch.optim.Adam`: `foreach=False`, der Standardeinstellung und `fused=True`.
    1. Führen Sie mit jedem Optimierer zwei Schritte aus. Messen Sie beim zweiten Schritt, wie viele Byte während `opt.step()` maximal zusätzlich belegt sind. Geben Sie den Wert in Einheiten der Gradientengröße $4N$ an.
    1. Erklären Sie die drei Messwerte anhand der Art, wie die Implementierungen die Parametertensoren aktualisieren: einzeln, durch gebündelte elementweise Operationen auf mehreren Tensoren oder durch einen einzigen fusionierten Kernel. Wann kann der Optimiererschritt den Spitzenbedarf des gesamten Trainingsschritts bestimmen?

<!--en:7183ce76-->
2. [code] **Speicherbedarf des Optimiererschritts.** Erweitern Sie die kompilierte Gradientenberechnung aus :numref:`subsec_mp-measuring` mit `optax.adam` zu einer vollständigen Adam-Aktualisierung.
    1. Kompilieren Sie den Schritt für 512 Token und geben Sie `argument_size_in_bytes`, `output_size_in_bytes` und `temp_size_in_bytes` aus `memory_analysis()` in Einheiten von $4N$ an.
    1. Kompilieren Sie den Schritt erneut. Erfassen Sie dabei mit `donate_argnums` die Parameter und den Optimiererzustand. Geben Sie `alias_size_in_bytes` an und erklären Sie, wie sich die Freigabe der Eingabepuffer zur Wiederverwendung auf den Spitzenbedarf auswirkt.

<!--en:c12d2ed9-->
3. **Gradientenbegrenzung bei der Akkumulation.** Die Begrenzung der Gradientennorm auf $\tau$ ersetzt $\mathbf{g}$ durch $\mathbf{g} \min(1, \tau / \|\mathbf{g}\|)$. Die Mikrobatchgradienten $\mathbf{g}_1, \ldots, \mathbf{g}_m$ seien mit $1/m$ skaliert, sodass ihre Summe $\mathbf{g}$ den Gradienten des globalen Batches ergibt.
    1. Angenommen, eine Schleife begrenzt jeden Gradienten $\mathbf{g}_i$, bevor sie ihn zum Akkumulator addiert. Welche Norm kann der akkumulierte Gradient höchstens erreichen? Wann liefert die Begrenzung pro Mikrobatch dasselbe Ergebnis wie die globale Begrenzung?
    1. Beim Loss-Scaling in fp16 mit dem Faktor $S$ enthält der Akkumulator nach dem letzten Mikrobatch $S \mathbf{g}$. In welcher Reihenfolge muss die Schleife die Skalierung durch Division durch $S$ rückgängig machen, den Gradienten begrenzen und den Optimiererschritt ausführen? Was geht schief, wenn sich $S$ zwischen den Mikrobatches ändert?

<!--en:f0c6f35e-->
4. [code] **Skalierungsbereich für fp16-Gradienten.** ● Mit seinem 5-Bit-Exponenten kann fp16 positive normalisierte Zahlen von $2^{-14}$ bis 65 504 und subnormale Zahlen bis hinunter zu $2^{-24}$ darstellen. Beim Loss-Scaling :cite:`Micikevicius.Narang.Alben.ea.2018` multiplizieren wir den Verlust mit $S$ vor dem Aufruf von `backward()` und dividieren die Gradienten anschließend durch $S$. Verwenden Sie `mlp`, `X_mlp` und `y_mlp` aus :numref:`subsec_mp-mixed` mit Referenzgradienten, die vollständig in fp32 berechnet wurden.
    1. Plotten Sie ein Histogramm von $\log_2 |g|$ für alle von null verschiedenen Einträge der Referenzgradienten. Bestimmen Sie die Skalierungsfaktoren $S$, bei denen 99.9% der skalierten Einträge $Sg$ im normalisierten Wertebereich von fp16 liegen.
    1. Berechnen Sie für $S = 2^0, 2^2, \ldots, 2^{24}$ die Gradienten mit `torch.autocast('cuda', dtype=torch.float16)`, dividieren Sie sie durch $S$ und plotten Sie den relativen Fehler $\|\hat{\mathbf{g}} - \mathbf{g}\| / \|\mathbf{g}\|$. Wiederholen Sie den Versuch mit `torch.bfloat16`. Für welchen Bereich von $S$ ist fp16 genauer als bf16 und warum?
    1. Wie verschiebt sich der gültige Bereich von $S$, wenn die Batchgröße von 2048 auf 32 768 steigt?

<!--en:3a0107f7-->
4. [code] **Skalierungsbereich für fp16-Gradienten.** ● Mit seinem 5-Bit-Exponenten kann fp16 positive normalisierte Zahlen von $2^{-14}$ bis 65 504 und subnormale Zahlen bis hinunter zu $2^{-24}$ darstellen. Beim Loss-Scaling :cite:`Micikevicius.Narang.Alben.ea.2018` multiplizieren wir den Verlust vor der Differentiation mit $S$ und dividieren die Gradienten anschließend durch $S$. Verwenden Sie `mlp_params`, `X_mlp`, `y_mlp` und `loss_fn` aus :numref:`subsec_mp-mixed` mit Referenzgradienten, die vollständig in fp32 berechnet wurden.
    1. Plotten Sie ein Histogramm von $\log_2 |g|$ für alle von null verschiedenen Einträge der Referenzgradienten. Bestimmen Sie die Skalierungsfaktoren $S$, bei denen 99.9% der skalierten Einträge $Sg$ im normalisierten Wertebereich von fp16 liegen.
    1. Berechnen Sie für $S = 2^0, 2^2, \ldots, 2^{24}$ die Gradienten mit `jnp.float16`, dividieren Sie sie durch $S$ und plotten Sie den relativen Fehler $\|\hat{\mathbf{g}} - \mathbf{g}\| / \|\mathbf{g}\|$. Wiederholen Sie den Versuch mit `jnp.bfloat16`. Für welchen Bereich von $S$ ist fp16 genauer als bf16 und warum?
    1. Wie verschiebt sich der gültige Bereich von $S$, wenn die Batchgröße von 2048 auf 32 768 steigt?

<!--en:32a07fbf-->
5. [code] **Selektive Neuberechnung.** ● Beim blockweisen Checkpointing speichern wir nur die Eingabe jedes Blocks. Übergeben wir `torch.utils.checkpoint.create_selective_checkpoint_contexts` als Argument `context_fn` an `checkpoint`, speichert das Verfahren die Ausgaben ausgewählter Operationen (etwa `torch.ops.aten.mm.default`) zwischen und berechnet den Rest neu.
    1. Listen Sie für einen Block unseres GPT die gespeicherten Tensoren und die neu berechneten Matrixprodukte für drei Strategien auf: vollständiges blockweises Checkpointing, Zwischenspeichern jeder `mm`-Ausgabe und Zwischenspeichern aller `mm`-Ausgaben außer denen der Gate- und Aufwärtsprojektion des FFN.
    1. Implementieren Sie die letzten beiden Strategien für die 4 aufeinanderfolgenden Blöcke aus :numref:`subsec_mp-checkpointing`. Vergleichen Sie die gespeicherten Byte pro Token, den Spitzenbedarf und die Schrittzeit.
   :exlabel:`ex_mp-selective`

<!--en:44364a75-->
5. [code] **Selektive Neuberechnung.** ● `jax.checkpoint` akzeptiert als `policy`-Argument beispielsweise `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`. Diese Strategie speichert die Ergebnisse von Multiplikationen mit Gewichtsmatrizen und berechnet elementweise Operationen neu.
    1. Listen Sie für einen Block unseres GPT die gespeicherten Residuen und die neu berechneten Matrixprodukte für drei Fälle auf: ohne Checkpointing, mit `jax.checkpoint` in der Standardeinstellung und mit `dots_with_no_batch_dims_saveable`. Prüfen Sie Ihre Angaben mit `jax.ad_checkpoint.print_saved_residuals`.
    1. Ergänzen Sie den Benchmark mit 4 Blöcken aus :numref:`subsec_mp-checkpointing` um diese Strategie. Vergleichen Sie die gespeicherten Byte pro Token, den geplanten Spitzenbedarf und die Schrittzeit.
   :exlabel:`ex_mp-selective`

<!--en:46a8d21a-->
6. [extended] **Abschnittsweise Verlustberechnung.** ●● Beim blockweisen Checkpointing benötigen Ausgabeschicht und Kreuzentropieverlust unseres GPT pro Token mehr Speicher als alle 18 Blöcke zusammen. Die Verlustfunktion legt nämlich pro Token $V = 50{,}257$ logarithmierte Wahrscheinlichkeiten und deren Gradienten an.
    1. Implementieren Sie die Ausgabeprojektion und den Kreuzentropieverlust so, dass sie Token in Gruppen der Größe $n_c = 1024$ verarbeiten. Legen Sie für jede Gruppe einen Checkpoint an, damit Logits und logarithmierte Wahrscheinlichkeiten jeweils nur für eine Gruppe im Speicher liegen. Prüfen Sie, ob Verlust und Gradienten mit denen der ursprünglichen Berechnung übereinstimmen.
    1. Sagen Sie für unser GPT mit 18 Blöcken auf einer Karte mit 24 GiB die maximale Mikrobatchgröße bei blockweisem Checkpointing und abschnittsweiser Verlustberechnung voraus. Messen Sie am GPT mit 2 Blöcken den Speicherzuwachs pro Token und die Schrittzeit.
   :exlabel:`ex_mp-chunked-loss`

<!--en:3ec681b7-->
[Diskussion](https://d2l.discourse.group/t/36790)

<!--en:8963bac5-->
Speicherhierarchie, Genauigkeit und Checkpointing<br>**Speicherbudgets für Parameter und Aktivierungen · gemischte Genauigkeit (`bf16`/`tf32`) · Aktivierungs-Checkpointing · Gradientenakkumulation**

<!--en:5ada6313-->
::: {.slide title="Ein GPT mit 1 Mrd. Parametern belegt schon vor dem ersten Token 15 von 24 GiB"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:b2e03f3d-->
- **Fester Parameterzustand ($16N = 15.1\text{ GiB}$):** Bei einem GPT mit 18 Blöcken und 1 Mrd. Parametern lassen `fp32`-Gewichte ($4N$), Gradienten ($4N$) und Adam-Momente ($8N$) nur $8.9\text{ GiB}$ auf einer RTX 4090 mit $24\text{ GiB}$ frei.

<!--en:f2c10b3e-->
- **Aktivierungen pro Token ($\sim 3\text{--}4\text{ MiB/tok}$):** Mit gespeicherten Aktivierungen aus dem Vorwärtsdurchlauf aller 18 Blöcke passen nur **2–3 Sequenzen** mit je $1{,}024$ Token in einen Mikrobatch.

<!--en:e1cca555-->
- **Größere Batches:** Für globale Batches mit $0.5\text{M+}$ Token sind gemischte Genauigkeit, Aktivierungs-Checkpointing und Gradientenakkumulation nötig.

<!--en:b4948744-->
::: {.slide title="Fester Trainingszustand bei Adam: 16 Byte pro Parameter"}

<!--en:689800b4-->
[Parameterspeicher]{.kicker}

<!--en:296a64c2-->
Der Speicherbedarf von Modellparametern, Gradienten und Optimiererzustand hängt nur von der Parameteranzahl $N$, dem Optimierer und der Genauigkeit ab. Er ist unabhängig von der Batchgröße:

<!--en:35341447-->
| Optimierer und Genauigkeit | Gespeicherte Arrays pro Parameter | Byte pro Parameter | Zustand eines GPT mit 1.01 Mrd. Parametern |

<!--en:d1c7b31e-->
| **Einfacher SGD (`float32`)** | Gewicht ($4\text{ B}$) + Gradient ($4\text{ B}$) | $8\text{ B}$ | $7.5\text{ GiB}$ |

<!--en:6effbed1-->
| **SGD mit Momentum (`float32`)** | Gewicht + Gradient + Geschwindigkeit ($4\text{ B}$) | $12\text{ B}$ | $11.3\text{ GiB}$ |

<!--en:2329f2cf-->
| **Adam / AdamW (`float32`)** | Gewicht + Gradient + erstes und zweites Moment ($8\text{ B}$) | **$16\text{ B}$** | **$15.1\text{ GiB}$** |

<!--en:fa514dd9-->
| **Adam mit gemischter Genauigkeit (`bf16`)** | Alle vier in `fp32` ($16\text{ B}$) + Gewichtskopie in `bf16` ($2\text{ B}$) | **$18\text{ B}$** | **$17.0\text{ GiB}$** |

<!--en:d32982ca-->
::: {.slide title="Lebensdauer der Aktivierungen: Spitzenbedarf zu Beginn des Rückwärtsdurchlaufs"}

<!--en:e4fb8700-->
[Aktivierungsspeicher]{.kicker}

<!--en:ebadfbc5-->
- **Vorwärts speichern, rückwärts freigeben:** Jede Schicht speichert im Vorwärtsdurchlauf die Aktivierungen für ihr Vektor-Jacobi-Produkt im Rückwärtsdurchlauf. Bei der Backpropagation berechnet sie damit die Parametergradienten und gibt die Aktivierungen frei.

<!--en:4310cb60-->
- **Zeitpunkt der höchsten Belegung:** Der gesamte belegte Speicher erreicht sein Maximum genau **zwischen der Verlustberechnung im Vorwärtsdurchlauf und dem Beginn des Rückwärtsdurchlaufs**.

<!--en:6e9ca554-->
::: {.slide title="Drei Ableitungsregeln bestimmen, welche Aktivierungen gespeichert werden"}

<!--en:1bce2285-->
[Speicher für die Backpropagation]{.kicker}

<!--en:dfccf841-->
- **Lineare Schicht ($\mathbf{Y} = \mathbf{X}\mathbf{W}$):** Sie muss die Eingabe $\mathbf{X}$ speichern, denn der Gewichtsgradient $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^\top(\nabla_{\mathbf{Y}}\mathcal{L})$ enthält als Faktor die transponierte Eingabe $\mathbf{X}$.

<!--en:104f3254-->
- **Elementweise Nichtlinearität ($\mathbf{y} = f(\mathbf{x})$):** Sie speichert $\mathbf{x}$ (oder $\mathbf{y}$), um die Ableitung $f'(\mathbf{x})$ auszuwerten.

<!--en:30c8a8bb-->
- **Hadamard-Produkt ($\mathbf{z} = \mathbf{p} \odot \mathbf{q}$):** Es speichert **beide** Faktoren $\mathbf{p}$ und $\mathbf{q}$. Deshalb speichert allein SwiGLU vier versteckte Vektoren ($\approx 10.7d$ Werte/Token).

<!--en:114f4d96-->
::: {.slide title="Speicherbilanz eines Transformer-Blocks: rund 20d Werte pro Token"}

<!--en:87933759-->
[Bilanz pro Token]{.kicker}

<!--en:e4ccda04-->
- **Normalisierung und Attention ($8d\text{--}9d$):** Zwei RMSNorm-Schichten und die Residualeingaben speichern $4d$ Werte. Die fusionierte FlashAttention speichert $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}$ ($4d\text{--}5d$ Werte, ohne die $T \times T$-Matrix).

<!--en:48502b17-->
- **SwiGLU-FFN ($\approx 10.7d$):** Es speichert das Gate $\mathbf{g}$, $\textrm{SiLU}(\mathbf{g})$, die Aufwärtsprojektion $\mathbf{u}$ und das Produkt der letzten beiden, jeweils mit Breite $\frac{8}{3}d$. Pro Block ergibt das **$a \approx 18.7d\text{--}20d$** ($149\text{--}160\text{ KiB/tok}$ in `fp32`).

<!--en:637359c8-->
- **Ausgabekopf ($a_{\textrm{out}} + \Delta$):** Abschließende Normalisierung, $V$ logarithmierte Wahrscheinlichkeiten in `fp32` ($212\text{ KiB/tok}$) und der Mehrbedarf $\Delta$, um den der gemessene Spitzenwert diese Summe übersteigt.

<!--en:a3355120-->
::: {.slide title="Gespeicherte Bytes nachmessen: 18.7d Werte pro fusioniertem Block" only="pytorch"}

<!--en:d876201b-->
[Autograd-Hook für gespeicherte Tensoren]{.kicker}

<!--en:13b32ee9-->
Den Hook `torch.autograd.graph.saved_tensors_hooks` registrieren und die Differenzen zwischen 1 und 2 Sequenzen sowie zwischen 1 und 2 Blöcken bilden. Das zeigt exakt, wie viele Byte pro Token ein Block ($a$) und der Ausgabekopf ($a_{\textrm{out}}$) speichern:

<!--en:7d0ef0b7-->
- **Exakte Übereinstimmung mit der Theorie:** PyTorch führt `scaled_dot_product_attention`, `RMSNorm` und `F.silu` jeweils als einen fusionierten Kernel aus. Daher speichert jeder Block das theoretische Minimum von **$18.7d = 149\text{ KiB/token}$**.

<!--en:637865ed-->
::: {.slide title="Gespeicherte Bytes nachmessen: Residuen vor der Fusion und XLA im Vergleich" only="jax"}

<!--en:89a2d393-->
[Speicherbedarf der VJP-Residuen]{.kicker}

<!--en:517e7e7f-->
Die ursprünglichen VJP-Residuen untersuchen, die `jax.vjp` vor der XLA-Kompilierung liefert. Aus den Differenzen zwischen 1 und 2 Sequenzen sowie zwischen 1 und 2 Blöcken ergibt sich der Speicherbedarf der Residuen vor der Kompilierung:

<!--en:65ebe162-->
- **Warum `jax.vjp` vor der Kompilierung $29.0d$ Werte ($232\text{ KiB/token}$) erfasst:** `nnx.RMSNorm` und `jax.nn.silu` bestehen in Python aus mehreren elementweisen Operationen. Ihre Zwischenergebnisse erscheinen daher im unkompilierten VJP-Baum. Bei `jax.jit` fusioniert XLA später viele dieser Operationen und berechnet ihre Ergebnisse neu.

<!--en:33097bb3-->
::: {.slide title="Speicherbudget eines Schritts auf einer GPU mit 24 GiB"}

<!--en:0531592f-->
[Speicherbudget]{.kicker}

<!--en:fab1402d-->
Zum festen Parameterzustand ($16N$ Byte, wenn die Gradienten über mehrere Mikrobatches erhalten bleiben) kommen $L$ Blöcke mit je $a$ Byte/Token und der Ausgabekopf mit $a_{\textrm{out}}$ Byte/Token. Das ergibt für $n$ Token folgendes Speicherbudget:

<!--en:082055f1-->
- **Untergrenze durch die Parameter ($N = 1.01 \times 10^9$):** Bei $16N = 15.1\text{ GiB}$ bleiben nur $M_{\textrm{GPU}} - 16N = \mathbf{8.9\text{ GiB}}$ für Aktivierungen und temporäre Tensoren des Rückwärtsdurchlaufs auf einer RTX 4090 mit $24\text{ GiB}$.

<!--en:2f073487-->
- **Maximale Tokenanzahl pro Mikrobatch ($L = 18$ Blöcke):** Bei $La + a_{\textrm{out}} \approx 2.8\text{--}4.3\text{ MiB/token}$ passen höchstens **$2{,}000\text{--}3{,}200$ Token** in einen Mikrobatch (2 oder 3 Sequenzen der Länge $1{,}024$).

<!--en:2a7f20b3-->
- **Doppelte Tiefe mit 36 Blöcken ($N = 1.92\text{B}$):** Schon der feste Bedarf beträgt $16N = 28.6\text{ GiB} > 24\text{ GiB}$. Selbst für ein einziges Token ist der Zustand von $16N$ Byte auf mehrere GPUs zu verteilen (§{sec_sharding}).

<!--en:6b60deb1-->
::: {.slide title="Gemessener Spitzenbedarf: 8V zusätzliche Byte für die Kreuzentropie" only="pytorch"}

<!--en:b0470c61-->
[Spitzenbedarf und errechneter Bedarf]{.kicker}

<!--en:45ae52a1-->
Die Messung des tatsächlichen Spitzenbedarfs (`torch.cuda.max_memory_allocated`) für $1{,}024$ bis $8{,}192$ Token bei einem GPT mit 2 Blöcken zeigt einen Mehrbedarf von $\Delta \approx 393\text{ KiB/token}$ gegenüber dem errechneten Bedarf der gespeicherten Aktivierungen:

<!--en:71f36b1e-->
- **Ursprung der zusätzlichen $393\text{ KiB/token}$:** Ganz zu Beginn von `backward()` liegen noch alle Aktivierungen des Vorwärtsdurchlaufs und die $V$ logarithmierten Wahrscheinlichkeiten im Speicher. Der Rückwärtsdurchlauf der Kreuzentropie legt zwei `float32`-Gradiententensoren der Breite $V = 50{,}257$ an (für logarithmierte Wahrscheinlichkeiten und Logits: $2 \times 4V = 8V \approx 393\text{ KiB/token}$).

<!--en:7de684d1-->
::: {.slide title="Geplanter Spitzenbedarf mit memory_analysis() von XLA" only="jax"}

<!--en:b0470c61-2-->
[Spitzenbedarf und errechneter Bedarf]{.kicker}

<!--en:733cae82-->
In JAX plant XLA die Zuweisung statischer Puffer im Voraus. Mit `compile().memory_analysis()` für Platzhalter vom Typ `ShapeDtypeStruct` lässt sich der Spitzenbedarf vorhersagen, ohne einen einzigen GPU-Tensor anzulegen:

<!--en:2d7f2cfb-->
- **Der Plan bestätigt die Rechnung:** Zwischen $4{,}096$ und $8{,}192$ Token steigt der geplante Spitzenbedarf um etwa $690\text{ KiB/token}$. Das entspricht den $692\text{ KiB}$ aus der ursprünglichen VJP-Zählung ($2 \times 29d$ plus Ausgabekopf). Damit gilt $\Delta \approx 0$, und der errechnete Bedarf eignet sich als Budget.

<!--en:113f4caf-->
::: {.slide title="Speicherbelegung im Vorwärtsdurchlauf, Rückwärtsdurchlauf und Optimiererschritt" only="pytorch"}

<!--en:53e9a906-->
[Aufzeichnung der Speicherbelegung]{.kicker}

<!--en:2a61ecbb-->
Alle Ereignisse des CUDA-Caching-Allocators während eines Adam-Schritts mit $4{,}096$ Token aufzeichnen. So wird das Sägezahnmuster sichtbar, das durch Aktivierungen und Gradienten entsteht:

<!--en:b969d80e-->
- **Drei Phasen:** Die Belegung beginnt bei $12N$ (Gewichte und zwei Adam-Momente) und steigt im **Vorwärtsdurchlauf** stufenweise von Block zu Block. Spitzen entstehen bei den Logits und bei den ersten Verlustgradienten im **Rückwärtsdurchlauf**. Wenn Aktivierungen frei werden, sinkt die Belegung stufenweise. Im **Optimiererschritt** kommen kurzzeitig temporäre Tensoren hinzu.

<!--en:ea73696e-->
::: {.slide title="Gemischte Genauigkeit: Referenzzustand in fp32, Berechnung in bf16"}

<!--en:ecb40386-->
[Gemischte Genauigkeit]{.kicker}

<!--en:51338541-->
- **Matrixprodukte linearer Schichten und der Attention in `bfloat16` ($2\text{ B}$):** Der Durchsatz der Tensor Cores verdoppelt sich auf $165\text{ TFLOP/s}$. Der Aktivierungsspeicher pro Block sinkt um $\sim 40\%$.

<!--en:abc82dc4-->
- **Referenzzustand und Reduktionen in `float32` ($4\text{ B}$):** Kleine Aktualisierungen ($\eta g \sim 10^{-5}w$) bleiben beim Runden erhalten. RMSNorm, Residualberechnung und Kreuzentropie behalten ihre Genauigkeit.

<!--en:cbb7eb5b-->
- **Kein Loss-Scaling nötig:** `bfloat16` hat denselben 8-Bit-Exponenten wie `float32` (Wertebereich $\sim 10^{\pm 38}$).

<!--en:701a8a6a-->
::: {.slide title="bf16 spart 40% Aktivierungsspeicher pro Block und benötigt 2N Byte mehr für den Zustand"}

<!--en:77e24226-->
[Genauigkeit und Kapazität]{.kicker}

<!--en:d75918ce-->
Bei gemischter Genauigkeit mit `bfloat16` bleiben die beiden Eingaben aus dem Residualstrom pro Block in `float32` ($4\text{ B}$). Die übrigen $18$ Vektoren der Blockbreite, die zu den $\sim 20d$ gespeicherten Werten beitragen, wechseln zu `bfloat16` ($2\text{ B}$). Für die Blockaktivierungen ergibt sich daher das $(2 \cdot 4 + 18 \cdot 2) / (20 \cdot 4) = 0.55$-Fache ihres Speicherbedarfs in `float32`:

<!--en:52d8a3bc-->
- **Warum die Tokenkapazität kaum wächst ($\sim 15\text{--}25\%$):** Die zwischengespeicherten `bfloat16`-Gewichtskopien erhöhen den festen Parameterspeicher von $16N = 15.1\text{ GiB}$ auf $18N = 17.0\text{ GiB}$. Damit sinkt der Platz für Aktivierungen ($8.9 \to 7.0\text{ GiB}$). Die vielen logarithmierten Wahrscheinlichkeiten des Verlusts in `float32` ($a_{\textrm{out}}$) beanspruchen weiterhin genauso viel Speicher.

<!--en:58b27eb4-->
::: {.slide title="Beschleunigung durch Tensor Cores: skalares fp32, tf32 und bf16"}

<!--en:3f095723-->
[Genauigkeit und Durchsatz]{.kicker}

<!--en:c78cfbc1-->
Gemischte Genauigkeit verbessert vor allem die **Geschwindigkeit**. Ein Benchmark misst einen SGD-Schritt eines rechenbegrenzten MLP mit 3 Schichten der Breite 8192 ($1.1 \times 10^{12}\text{ FLOPs/step}$, mindestens $13.3\text{ ms}$ bei $82.6\text{ TFLOP/s}$ und $6.7\text{ ms}$ bei $165.2\text{ TFLOP/s}$, dazu $1.3\text{ ms}$ für die SGD-Aktualisierung in `fp32`). So lässt sich getrennt messen, was die Tensor Cores und was die 16-Bit-Operanden bewirken:

<!--en:3ea05093-->
- **Skalares `fp32` $\to$ `tf32` auf Tensor Cores ($1.45\times$):** Gleiche theoretische Obergrenze von $82.6\text{ TFLOP/s}$ und weiterhin 32 Bit pro gespeichertem Wert. Die Kachelbefehle der Tensor Cores erreichen jedoch $\sim 90\%$ Auslastung gegenüber $\sim 60\%$ auf skalaren ALUs.

<!--en:7b5ae469-->
- **Tensor Cores mit `tf32` $\to$ `bf16` ($1.5\text{--}1.75\times$):** Die Zeit für die Matrixmultiplikationen halbiert sich auf $\sim 6.7\text{ ms}$. Die bandbreitenbegrenzte SGD-Aktualisierung in `fp32` ($1.3\text{ ms}$) und die Typumwandlungen der Gewichte dauern gleich lange.

<!--en:8a4c6133-->
::: {.slide title="Aktivierungs-Checkpointing: O(L) Eingaben, interne Aktivierungen für einen Block"}

<!--en:777658c7-->
[Aktivierungs-Checkpointing]{.kicker}

<!--en:4bb996ef-->
- **Vorwärtsdurchlauf:** Er speichert nur die `fp32`-Eingabe an jeder der $L = 18$ Blockgrenzen ($18 \times 8\text{ KiB} = \mathbf{144\text{ KiB/tok}}$ statt $2.7\text{ MiB/tok}$).

<!--en:014cd40b-->
- **Rückwärtsdurchlauf:** Der Vorwärtsdurchlauf von Block $\ell$ wird unmittelbar vor der Backpropagation durch Block $\ell$ wiederholt. So bleiben jeweils nur die internen Aktivierungen **eines einzigen Blocks** ($a$) im Speicher, bei $\le 33\%$ zusätzlichen FLOPs.

<!--en:2fbd2f14-->
::: {.slide title="Tokenkapazität je nach Genauigkeit und Checkpointing"}

<!--en:c78a0932-->
[Kapazitätsvergleich]{.kicker}

<!--en:d94f017b-->
Für unser GPT mit 18 Blöcken und 1 Mrd. Parametern zeigt die Auswertung von $M(n) \le 24\text{ GiB}$, wie viel Platz Checkpointing für Aktivierungen schafft:

<!--en:fc2624b9-->
- **Mehr als $3\times$ so große Mikrobatches:** Blockweises Checkpointing erhöht die Kapazität von $\sim 2{,}800$ Token auf **$\sim 10{,}000\text{--}15{,}000$ Token**.

<!--en:c076d340-->
- **Überraschendes Zusammenspiel:** Bei vollständigem blockweisem Checkpointing passen mit `bf16` etwas *weniger* Token in den Speicher als mit `fp32`. Die zusätzlichen $2N = 1.9\text{ GiB}$ für `bf16`-Gewichtskopien wiegen schwerer als die $60\text{ KiB/token}$, die beim einzigen gerade neu berechneten Block frei werden.

<!--en:74dda592-->
::: {.slide title="Checkpointing in der Messung: 40 KiB/Token und 25% mehr Zeit"}

<!--en:38d4e3d0-->
[Checkpointing im Benchmark]{.kicker}

<!--en:da563da9-->
Bei 4 aufeinanderfolgenden Blöcken ($d = 2048$ in `float32`) speichert Checkpointing nur die $4$ Blockeingaben und die abschließende Ausgabe ($5 \times 4d = 40\text{ KiB/token}$):

<!--en:d9957b97-->
- **Exakter Speicherbedarf, geringer Zeitaufwand:** Der gespeicherte Umfang sinkt genau auf den Wert der Vorhersage von $40\text{ KiB/token}$, der Spitzenbedarf sinkt deutlich und die Schrittzeit steigt um nur **$\sim 25\%$**. Sie bleibt unter der FLOP-basierten Schranke von $33\%$, weil fusionierte Kernel im Rückwärtsdurchlauf die elementweise Neuberechnung mit übernehmen.

<!--en:c9c4191c-->
::: {.slide title="Selektive Neuberechnung und abschnittsweiser Kreuzentropieverlust"}

<!--en:7c61c4c7-->
[Gezielte Neuberechnung]{.kicker}

<!--en:1d050890-->
Sobald blockweises Checkpointing den Speicherbedarf aller 18 Blöcke auf $144\text{ KiB/token}$ gesenkt hat, helfen zwei Verfeinerungen bei den verbleibenden Engpässen:

<!--en:ab201ab3-->
- **Selektive Neuberechnung (Korthikanti et al., 2022):** Matrixmultiplikationen verursachen die meisten FLOPs eines Blocks. Elementweise Operationen beanspruchen dagegen den meisten Aktivierungsspeicher ($\sim 5d$ Werte für aktiviertes Gate und Produkt bei SwiGLU oder $T \times T$ Attention-Scores ohne Fusion). Die Ausgaben von `mm` zwischenzuspeichern und nur günstige elementweise Operationen neu zu berechnen spart viel Speicher bei fast unveränderter FLOP-Anzahl.

<!--en:5e1bf855-->
- **Abschnittsweise Ausgabeprojektion und Kreuzentropieverlust:** Ausgabe-Logits und Verlust ($a_{\textrm{out}} + \Delta \approx 605\text{ KiB/token}$) benötigen nun **$4\times$ so viel Speicher pro Token wie alle 18 Blöcke mit Checkpointing zusammen**. Ausgabeprojektion und Kreuzentropie für Gruppen von je $1{,}024$ Token innerhalb eines Checkpoints auswerten. Damit entfällt die Speicherspitze der Breite $V$ bei nur $2dV / (6N) \approx 3\%$ zusätzlichen FLOPs.

<!--en:8a2795f9-->
::: {.slide title="Gradientenakkumulation und der Mehraufwand von 2I*/n für Speicherzugriffe"}

<!--en:21a8cff2-->
[Gradientenakkumulation]{.kicker}

<!--en:81c1314d-->
- **Große globale Batches ($B_{\textrm{global}} = m\,B_{\textrm{micro}}$):** $\frac{1}{m}\nabla\mathcal{L}_i$ über $m$ aufeinanderfolgende Mikrobatches mit je $n$ Token summieren und dann einen einzigen Adam-Schritt ausführen.

<!--en:22c48356-->
- **Mehraufwand von $2I^*/n$ für Speicherzugriffe:** Jeder weitere Mikrobatch liest die bisherigen Gradienten ($4N$) und die neuen Gradienten ($4N$) und schreibt ihre Summe ($4N$). Stets den größten Mikrobatch der Größe $n$ verwenden, der ohne zusätzliches Checkpointing in den Speicher passt.

<!--en:daa4fdef-->
::: {.slide title="Akkumulation in der Messung: halber Spitzenbedarf, gleiche Gradienten"}

<!--en:0f3f0df9-->
[Gradientenakkumulation im Benchmark]{.kicker}

<!--en:e15ef5d1-->
$4{,}096$ Token beim GPT mit 2 Blöcken in $m = 4$ Mikrobatches mit je $1{,}024$ Token aufteilen. Das ergibt genau den Gradienten des gesamten Batches und senkt den Aktivierungsspeicher auf ein Viertel:

<!--en:b4997332-->
- **Spitzenbedarf und Schrittzeit:** Der Spitzenbedarf oberhalb des Gewichtsspeichers sinkt etwa auf die Hälfte (auf $4N = 0.77\text{ GiB}$ für den dauerhaften Gradientenpuffer plus die Aktivierungen für $1{,}024$ Token). Die Schrittzeit steigt um $\sim 10\text{--}18\%$, wie $2I^*/n$ vorhersagt.

<!--en:e7c89f71-->
::: {.slide title="Verständnisfragen zu Speicherbudget, Neuberechnung und Akkumulation"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:8ad5e826-->
Die Formeln für Speicherbedarf und Laufzeit auf zwei Fragen zu unserem GPT mit 18 Blöcken und 1 Mrd. Parametern anwenden ($d = 2048$, Knickpunkt für `bf16` bei $I^* = 164\text{ FLOP/byte}$):

<!--en:65729c2b-->
1. Ziel ist ein globaler Batch von $524{,}288$ Token auf einer RTX 4090 mit $24\text{ GiB}$ in `bfloat16`. Mikrobatches mit **2 Sequenzen** ($n = 2{,}048$ Token, $m = 256$) passen *ohne* Checkpointing in den Speicher. Für **4 Sequenzen** ($n = 4{,}096$ Token, $m = 128$) ist blockweises Checkpointing nötig ($\sim 25\%$ Mehraufwand für die Neuberechnung). Mit welcher Variante ist der globale Schritt schneller fertig?

<!--en:d974a683-->
2. Wie viel Rechenzeit benötigt ein globaler Schritt auf einer einzelnen RTX 4090 bei $6N\text{ FLOPs/token}$ und einer Spitzenleistung von $165\text{ TFLOP/s}$ mindestens, wenn er $524{,}288$ Token umfasst?

<!--en:557b98b6-->
**Lösungen:**

<!--en:e453d95f-->
1. **2 Sequenzen ohne Checkpointing ($m = 256$) sind schneller:** Der Mehraufwand für den Akkumulator beträgt $\frac{2 I^*}{n} = \frac{2 \times 164}{2048} \approx \mathbf{16\%}$. Bei 4 Sequenzen mit Checkpointing kommen zu $8\%$ für den Akkumulator noch $\sim 25\%$ für die Neuberechnung hinzu (insgesamt $\approx \mathbf{33\%}$ Mehraufwand). Checkpointing nur aktivieren, wenn kein ausreichend großer Mikrobatch in den Speicher passt.

<!--en:7b26f5c7-->
2. **$6 \times (1.01 \times 10^9) \times 524{,}288 \approx 3.18 \times 10^{15}\text{ FLOPs/step}$**. Das erfordert **$\ge 19.2\text{ s/step}$** bei $100\%$ der Spitzenleistung ($\sim 22\text{ s}$ mit Akkumulation), also über 25 Tage für $100{,}000$ Schritte. Eine GPU bietet genug Speicher für den Schritt. Für das vollständige Training sind mehrere GPUs nötig.

<!--en:1dfabcaf-->
::: {.slide title="Zusammenfassung: Speicherbedarf und Optimierungsverfahren"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:72bbff5b-->
| Verfahren / Komponente | Wirkung auf den Speicherbedarf $M(n)$ eines Schritts | Zeit- und Rechenaufwand | Einsatz |

<!--en:902599cc-->
| **Fester Adam-Zustand** | $16N\text{ B}$ (`fp32`), $18N\text{ B}$ (`bf16`) | pro Schritt konstant | Grundbedarf für Parameter, Gradienten und 2 Momentpuffer |

<!--en:588f79c8-->
| **Blockaktivierungen ($a$)** | $\approx 20d$ Werte/Token ($160\text{ KiB}$ in `fp32`) | regulärer Vorwärts- und Rückwärtsdurchlauf | Eingaben linearer Schichten ($4d$) + Attention ($4d$) + SwiGLU ($11d$) |

<!--en:2615a70a-->
| **Gemischte Genauigkeit mit `bf16`** | Blockaktivierungen $\times 0.6$, Zustand $+2N$ | **$1.5\text{--}1.75\times$ schneller** | immer auf GPUs mit Tensor Cores (kein Loss-Scaling nötig) |

<!--en:d473abf8-->
| **Blockweises Checkpointing** | Speicher für die Blöcke $La \to L \cdot 4d + a$ | $+25\text{--}33\%$ Schrittzeit | wenn ein sinnvoll großer Mikrobatch mit $n$ Token nicht in $M_{\textrm{GPU}}$ passt |

<!--en:a178e4a6-->
| **Abschnittsweise Kreuzentropie** | beseitigt die Speicherspitze der Breite $V$ und Größe $a_{\textrm{out}} + 8V$ | $+3\%$ FLOPs ($2dV$) | wenn das Vokabular der Größe $V$ den Speicherbedarf mit Checkpointing bestimmt |

<!--en:a3500e18-->
| **Gradientenakkumulation** | verarbeitet $B_{\textrm{global}} = m \, B_{\textrm{micro}}$ mit einem Speicherbedarf von $M(B_{\textrm{micro}})$ | $+\frac{2I^*}{n}$ für Speicherzugriffe des Akkumulators | für große globale Batches bei begrenzter GPU-Kapazität |
