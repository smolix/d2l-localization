<!--en:919572c4-->
# Fallstudie: Optimierung eines Transformers

<!--en:10bac504-->
In den bisherigen Abschnitten haben wir jeweils eine Ursache für langsame Trainingsschritte untersucht und gezeigt, wie sie sich beheben lässt: Startkosten auf dem Host, Speicherverkehr durch kleine Kernel, breite Zahlenformate und die Grenzen einer einzelnen GPU. In der Praxis bremsen mehrere Ursachen denselben Schritt. Beseitigen wir eine davon, verlagert sich der Engpass. Wir gehen nun wie in der Praxis von einem einzelnen Modell aus. Wie stark lässt sich sein Trainingsschritt beschleunigen, und in welcher Reihenfolge sollten wir die Änderungen vornehmen?

<!--en:5f655c8f-->
Wir verwenden das GPT-Modell aus :numref:`sec_gpt` und verdoppeln seine Breite auf 512 (19 Millionen Parameter). Ein Trainingsschritt mit 64 Fenstern zu je 128 Byte ($8{,}192$ Token) benötigt etwa $6 \times 1.9 \times 10^{7} \times 8{,}192 \approx 9 \times 10^{11}$ FLOPs: pro Parameter und Token zwei im Vorwärtsdurchlauf und vier im Rückwärtsdurchlauf. Auf einer RTX 4090 ($165\text{ TFLOP/s}$ in bf16) dauert die reine Rechnung weniger als $6\text{ ms}$. Ein einfacher fp32-Schritt mit AdamW dauert etwa $25\text{ ms}$. Allein das Zahlenformat erklärt rund $6\text{ ms}$ der Differenz von $19\text{ ms}$ (in tf32 erreichen die Tensor Cores den halben bf16-Durchsatz und benötigen $11\text{ ms}$). In den verbleibenden rund $13\text{ ms}$ arbeiten die Tensor Cores unterhalb ihrer Spitzenleistung oder liegen brach.

<!--en:372e98c0-->
Wir folgen der Methode aus :numref:`sec_perf_model`: zählen, vorhersagen, messen und die Abweichung erklären. Vor jeder Änderung liefert das Laufzeitprofil des bisherigen Schritts eine Obergrenze für den möglichen Gewinn. Danach vergleichen wir Vorhersage und Messung. Bei jeder Messung behalten wir die vorherigen Änderungen bei. Zum Schluss prüfen wir, ob das Modell mit dem schnellsten Schritt weiterhin lernt, und schlüsseln die verbleibende Laufzeit auf.

<!--en:b189bb62-->
## Versuchsaufbau

<!--en:e48e6dae-->
Wir beurteilen jede Änderung in diesem Abschnitt anhand derselben Messung. Modell, Daten und Durchsatzbenchmark bleiben daher während des gesamten Versuchs gleich.

<!--en:7cd743b6-->
### Modell und Daten

<!--en:22c84e42-->
Wir trainieren `d2l.GPT` (:numref:`sec_gpt`). Das Modell hat sechs Blöcke mit vorgeschalteter Normalisierung und je acht Attention-Köpfen. Es verwendet Rotary Position Embedding, RMSNorm, SwiGLU und eine Eingabeeinbettung, die ihre Gewichte mit der Ausgabeschicht teilt. Aus WikiText-103 ziehen wir an zufälligen Positionen Fenster von je 128 Byte. Das Vokabular umfasst alle 256 Bytewerte (`d2l.WikiText103`, :numref:`sec_test_models`).

<!--en:b619a793-->
Wir verdoppeln die Breite von 256 auf 512, damit die Matrixprodukte deutlich im rechenbegrenzten Bereich liegen (engl. compute-bound, :numref:`sec_perf_model`). Die Multiplikation einer Aktivierungsmatrix der Größe $n \times d$ mit einer Gewichtsmatrix der Größe $d \times d'$ benötigt $2ndd'$ FLOPs und bewegt $b(nd + dd' + nd')$ Byte ($b$ Byte je Element). Bei $n = 8{,}192$ Token und einer Breite von $d = 512$ erreichen die Projektionen der Blöcke $125\text{--}185\text{ FLOP/byte}$ in tf32 ($b = 4$) und $250\text{--}370\text{ FLOP/byte}$ in bf16 ($b = 2$). Das liegt weit über den Knickpunkten der RTX 4090 ($82\text{ FLOP/byte}$ für tf32 und $164\text{ FLOP/byte}$ für bf16). Ein schnelleres Zahlenformat verkürzt daher unmittelbar die Rechenzeit für diese Produkte. Bei einer Breite von 256 liegen sie teils oberhalb, teils unterhalb des Knickpunkts (Aufgabe :exref:`ex_ft-width`).

<!--en:88dd9141-->
Das Trainingskorpus von 270 MB passt als eindimensionales Byte-Array in den GPU-Speicher. Die nächste Zelle kopiert es auf die GPU, definiert `windows` zum Ziehen von Minibatches direkt auf dem Gerät ohne Übertragung vom Host und erstellt das Modell.

<!--en:d3da2a04-->
Bei 16 Byte pro Parameter (Gewicht, Gradient und zwei Adam-Momente) belegt der Trainingszustand für 19 Millionen Parameter nur $0.3\text{ GiB}$. Wir benötigen weder Gradientenakkumulation noch Partitionierung (:numref:`sec_memory_precision`, :numref:`sec_sharding`). Da das Korpus auf der GPU liegt, können wir den Trainingsschritt auch unabhängig von der Eingabepipeline untersuchen (:numref:`sec_hardware`).

<!--en:48bb432b-->
### Durchsatzmessung

<!--en:f2f7d6b5-->
Wir messen den Trainingsdurchsatz (engl. training throughput) in Token pro Sekunde über den gesamten Schritt hinweg. Die Messung umfasst das Ziehen des Minibatchs, den Vorwärts- und Rückwärtsdurchlauf sowie das Update des Optimierers:

<!--en:dc883e31-->
| Einstellung | Wert |

<!--en:8ce23b56-->
| Hardware | 1 von 4 GPUs des Typs RTX 4090 (2 und 4 für Datenparallelität) |

<!--en:23c950f8-->
| Modell | `d2l.GPT`: 6 Blöcke, Breite 512, 8 Köpfe, Byte-Vokabular, Kontextlänge 128 |

<!--en:286470b8-->
| Daten | in jedem Schritt auf der GPU zufällig gezogene Fenster von je 128 Byte aus WikiText-103 |

<!--en:cb5dd6bc-->
| Optimierer | AdamW ($\text{lr} = 10^{-3}$, Gewichtszerfall $0.01$) |

<!--en:c1181ceb-->
| Rechengenauigkeit | fp32 mit tf32-Matrixprodukten bis zur Umstellung auf bf16 |

<!--en:db43d119-->
| Zeitmessung | `d2l.Benchmark`: 20 Aufwärmschritte, dann 3 Messfenster zu je 50 Schritten mit Gerätesynchronisation an den Fenstergrenzen |

<!--en:563919d3-->
| Ergebnis | Median der Token/s über die 3 Fenster, Verhältnis zur letzten beibehaltenen Konfiguration und relative Streuung zwischen den Fenstern |

<!--en:ba0fd0f2-->
`tokens_per_second` setzt diese Regeln mit `d2l.Benchmark` um (:numref:`sec_perf_model`). `record` speichert jedes Ergebnis und gibt den Beschleunigungsfaktor gegenüber der letzten beibehaltenen Konfiguration sowie die relative Streuung über die drei Messfenster aus ($\text{range} / \text{median}$). Konfigurationen mit `kept=False` werden als Varianten gespeichert, auf denen spätere Konfigurationen nicht aufbauen.

<!--en:0edb1c45-->
Schritte mit hohem Arbeitsaufwand auf dem Host reagieren empfindlich auf dessen Hintergrundlast. Ihre Messwerte streuen stärker als bei kompilierten oder als Graph erfassten Schritten. Die Host-Last schwankt von Tag zu Tag stärker als zwischen aufeinanderfolgenden Fenstern desselben Laufs. Wir geben Beschleunigungsfaktoren daher auf eine Nachkommastelle an und vertrauen kleinen Gewinnen von wenigen Prozent nur, wenn das Laufzeitprofil sie vorhersagt.

<!--en:64e136d2-->
## Eine einzelne GPU

<!--en:ac4893f1-->
Ausgehend von der Implementierung aus :numref:`sec_gpt` optimieren wir den Trainingsschritt zunächst auf einer einzelnen GPU. Dabei nehmen wir jeweils nur eine Änderung vor.

<!--en:004cc0c6-->
### Baseline

<!--en:e2521df9-->
Als Baseline dient der Standardtrainingsschritt aus :numref:`sec_gpt` mit AdamW: Vorwärtsdurchlauf, Kreuzentropieverlust, Rückwärtsdurchlauf und Parameterupdate.

<!--en:0cb5b086-->
Der Schritt läuft im Eager-Modus: Python und Autograd veranlassen für jede Operation im Vorwärts- und Rückwärtsdurchlauf einzelne Kernelstarts. AdamW (`fused=True`) aktualisiert alle Parameter und Momente in wenigen fusionierten Kerneln. `make_step` fasst den Trainingsschritt in einer Funktion zusammen. Ihre Argumente `forward` und `bf16` ermöglichen die Änderungen in den nächsten Unterabschnitten.

<!--en:ca239df1-->
`train_step` wird mit `nnx.jit` kompiliert, wie in `d2l.train_steps` (:numref:`sec_test_models`). Die Funktion erhält Modell und Optimierer als Argumente und berechnet den Kreuzentropieverlust aus fp32-Logits.

<!--en:add6011f-->
Wie erwartet streuen die Messwerte der Baseline wegen der Host-Arbeit pro Schritt etwa zehnmal so stark wie bei späteren Konfigurationen. Um die Laufzeit aufzuschlüsseln, zeichnet `profile` fünf Schritte mit `d2l.kernel_times` auf. Die Ausgabe von `d2l.show_profile` (:numref:`subsec_perf-profiler`) zeigt die gesamte Kernel-Laufzeit auf dem Gerät im Vergleich zur verstrichenen Schrittzeit, den Anteil der Matrixprodukte einschließlich fusionierter Attention-Kernel und die Kernel mit der längsten Gesamtlaufzeit. Das Laufzeitprofil speichern wir für die abschließende Bilanz.

<!--en:e7b580e3-->
Das Gerät führt fast den gesamten Schritt über Kernel aus. Matrixprodukte beanspruchen etwa zwei Drittel der Zeit. Da der Host meist Schritt hält, begrenzt die Arbeit auf dem Gerät die Geschwindigkeit.

<!--en:4ee7e692-->
Das verbleibende Drittel verteilt sich auf viele kurze Kernel für SwiGLU, rotierende Positionseinbettungen, Additionen im Residualstrom und RMSNorm. Jeder liest aus dem Gerätespeicher und schreibt in ihn. Die Speicherbandbreite begrenzt daher ihre Geschwindigkeit (:numref:`subsec_comp-fusion`).

<!--en:4ffed5a8-->
Da `nnx.jit` den gesamten Schritt kompiliert, hat XLA die elementweisen Operationen bereits mit den umliegenden Kerneln fusioniert. Auf diese fusionierten Kernel entfällt das Drittel der Gerätezeit außerhalb der Matrixprodukte. Ihre Laufzeit hängt weiterhin von der bewegten Datenmenge ab.

<!--en:6e21ac95-->
### Startkosten auf dem Host und kleine Kernel

<!--en:64ec4c6f-->
Zwei Maßnahmen verkürzen die Zeit außerhalb der Matrixprodukte. Das Laufzeitprofil liefert für jede eine obere Schranke für den Gewinn. Diese Schranke folgt aus *Amdahls Gesetz* :cite:`Amdahl.1967`: Beschleunigt eine Änderung den Anteil $f$ eines Schritts um den Faktor $s$, so beträgt der Beschleunigungsfaktor für den ganzen Schritt $1/(1 - f + f/s)$. Er bleibt unter $1/(1 - f)$, wie groß $s$ auch wird. Für die beiden Maßnahmen bedeutet das:

<!--en:e9d55357-->
1. *Geringere Startkosten auf dem Host* verkürzen die Leerlaufzeiten, in denen das Gerät auf den Host wartet (:numref:`sec_perf_model`). Der Leerlaufanteil begrenzt den möglichen Gewinn.

<!--en:b48edb4d-->
2. *Kernel-Fusion* fasst Folgen bandbreitenbegrenzter elementweiser Kernel zusammen, sodass Zwischenergebnisse auf dem Chip bleiben. Die Gerätezeit außerhalb der Matrixprodukte begrenzt den möglichen Gewinn.

<!--en:50679eec-->
Unsere Baseline fällt in den zweiten Fall: Die GPU liegt nur wenige Prozent der Schrittzeit brach, während Kernel ohne Matrixmultiplikation $8.8\text{ ms}$ von insgesamt $24.5\text{ ms}$ Gerätezeit beanspruchen. Selbst wenn die Kompilierung all diese Kernel und alle Leerlaufzeiten beseitigte, blieben die $15.6\text{ ms}$ für Matrixprodukte. Der Beschleunigungsfaktor beträgt also höchstens $1.6\times$. `torch.compile` (:numref:`sec_compilation`) fusioniert elementweise Operationen im Vorwärts- und Rückwärtsdurchlauf. Die Matrixprodukte und das fusionierte Update des Optimierers bleiben erhalten. Mit `mode='reduce-overhead'` und `dynamic=False` erfasst der Compiler außerdem CUDA-Graphen für feste Tensorformen (:numref:`subsec_comp-overhead`). Damit entfallen die wenigen Prozent Wartezeit auf den Host. Zugleich verhindern wir, dass der Kernel-Start durch den Host später zum Engpass wird, wenn die Gerätezeit sinkt. Zunächst prüfen wir, ob das kompilierte Modell dieselben Logits liefert wie im Eager-Modus.

<!--en:95653a90-->
XLA hat unseren Schritt bereits fusioniert, und die GPU wartet nur während 2% der Schrittzeit. Weniger Arbeit auf dem Host kann derzeit also höchstens 2% einsparen. Wir beseitigen sie dennoch: Ihre festen Kosten pro Schritt würden spürbar, sobald spätere Optimierungen die Gerätezeit verkürzen. Bei jedem Aufruf durchläuft `nnx.jit` die Python-Objekte für Modell und Optimierer, um ihre Arrays zu extrahieren und zu aktualisieren. Wir messen die Kosten dieses Durchlaufs mit einer Funktion, die nichts tut. Dazu verwenden wir zunächst `nnx.jit`, dann `nnx.jit(..., graph=False)` (Baummodus) und schließlich `nnx.jit_partial(..., graph=False)`. Die letzte Variante bindet Modell und Optimiererzustand einmalig, sodass wir pro Schritt nur den Minibatch übergeben.

<!--en:efbe0e70-->
Die Logits des kompilierten Modells stimmen auf vier signifikante Stellen mit der Referenz überein. Die kleinen Unterschiede entstehen, weil die fusionierten Kernel in fp32 in anderer Reihenfolge runden und die tf32-Matrixprodukte diese Abweichungen verstärken. Der Durchsatz steigt um über 10%. Der Beschleunigungsfaktor bleibt damit deutlich unter der Obergrenze von $1.6\times$.

<!--en:ef5be11b-->
Ein Aufruf ohne Operation über `nnx.jit` beansprucht fast $9\text{ ms}$ auf dem Host, mehr als ein Drittel der Gerätezeit pro Schritt. Diese Kosten bleiben nur verborgen, solange der Host dem Gerät voraus ist. Der Wechsel zum Baumdurchlauf beseitigt den größten Teil davon, die Bindung mit `nnx.jit_partial` fast den gesamten Rest. Als Nächstes prüfen wir, ob `nnx.jit_partial` denselben Verlust und dieselben Gewichtsänderungen liefert wie `nnx.jit`.

<!--en:806378e2-->
Die aktualisierten Gewichte stimmen exakt überein. Der Durchsatz steigt um etwa 1%. Das liegt innerhalb der Schranke von 2% und der Streuung der Baseline.

<!--en:31c0eecb-->
Ein zweites Laufzeitprofil zeigt, wie sich der Schritt verändert hat:

<!--en:a3c71f7b-->
Das Gerät ist weiterhin fast den gesamten Schritt über beschäftigt. Auf die Matrixprodukte entfallen nun rund zwei Drittel der Gerätezeit oder mehr.

<!--en:9b24db13-->
Die Gerätezeit pro Schritt sank von $24.5\text{ ms}$ auf $22.3\text{ ms}$, und die Pausen zwischen den Kerneln verschwanden. Die Fusion durch Inductor sparte $2.2\text{ ms}$ ein (ein Viertel der Zeit außerhalb der Matrixprodukte). Die Wiedergabe des CUDA-Graphen beseitigte etwa $1\text{ ms}$ Wartezeit zwischen Kernelstarts durch den Host.

<!--en:87e85438-->
Die Kernel-Laufzeiten auf dem Gerät bleiben gleich, da `nnx.jit_partial` nur die Startkosten auf dem Host gesenkt hat.

<!--en:71b3041c-->
### Geringere Rechengenauigkeit

<!--en:7b0effe6-->
Die GPU ist ausgelastet, und Matrixprodukte beanspruchen den größten Teil der Gerätezeit. Als Nächstes beschleunigen wir daher die Matrixprodukte. Der Wechsel von tf32 zu bf16 verdoppelt den Spitzendurchsatz der Tensor Cores auf der RTX 4090 auf $165\text{ TFLOP/s}$ und halbiert den Speicherbedarf der bf16-Aktivierungen (:numref:`sec_memory_precision`). Entfällt der Anteil $f \in [2/3, 0.7]$ der Schrittzeit auf die Matrixprodukte und verdoppelt sich nur deren Geschwindigkeit, so sinkt die Schrittzeit auf den Anteil $1 - f/2$. Das entspricht einem Beschleunigungsfaktor von etwa $1.5\times$. Er nähert sich $2\times$, wenn auch die übrigen Kernel nur halb so viele Byte bewegen.

<!--en:42c8a73f-->
Wir verwenden gemischte Genauigkeit (engl. mixed precision) nach dem Verfahren aus :numref:`subsec_mp-mixed`. Matrixprodukte einschließlich Attention und die dazwischenliegenden elementweisen Operationen laufen in bf16. Hauptgewichte, Gradienten, Adam-Momente, Normalisierungen, Verlust und Residualstrom bleiben in fp32. So gehen kleine Änderungen eines Blocks im Residualstrom nicht durch Rundung auf die 8-Bit-Mantisse von bf16 verloren. Da bf16 denselben Exponentenbereich wie fp32 hat, ist kein Loss-Scaling nötig. Vor der Zeitmessung prüfen wir, ob der bf16-Vorwärtsdurchlauf innerhalb der bf16-Genauigkeit (etwa 1%) mit den fp32-Logits übereinstimmt.

<!--en:257a48f7-->
`torch.autocast` wählt die Rechengenauigkeit für jede Operation einzeln (:numref:`sec_numerics`). Matrixprodukte und Eingaben der Attention werden in bf16 umgewandelt. Einbettungen, Normalisierungen, Residualstrom und die Softmax-Funktion für den Verlust bleiben in fp32.

<!--en:15fe7903-->
`d2l.to_bf16` (:numref:`subsec_mp-mixed`) setzt für die linearen Schichten `dtype=jnp.bfloat16`. Dabei bleiben `param_dtype` und alle anderen Schichten (Einbettungen, Normalisierungen und die Ausgabeprojektion mit geteilten Gewichten) in fp32.

<!--en:b3a4b55f-->
Die bf16-Logits stimmen bis auf 1% mit den fp32-Logits überein.

<!--en:30c686b4-->
Der Durchsatz steigt um etwa den Faktor $1.75\times$ und damit stärker als die vorhergesagten $1.55\times$. Die Vorhersage halbiert lediglich die Zeit für Matrixprodukte, die zuvor 71% ausmachte. In bf16 erreichen die GEMMs jedoch eine höhere Auslastung als in tf32. Zudem bewegen die fusionierten Kernel zwischen den Produkten bf16-Aktivierungen von 2 Byte statt Gleitkommazahlen von 4 Byte. `F.scaled_dot_product_attention` verwendet bereits seit der Baseline einen fusionierten FlashAttention-Kernel.

<!--en:1e5b2b97-->
Der Durchsatz steigt um den Faktor $1.45\times$. Das liegt knapp unter dem Faktor $1.48\times$, den wir allein aus der Halbierung des Matrixproduktanteils von 65% erwarten. Die übrigen Kernel sind demnach nicht schneller geworden. Ein Grund dafür ist die Attention: Standardmäßig nutzt `jax.nn.dot_product_attention` die Implementierung von XLA, die die vollständige Attention-Matrix in den Gerätespeicher schreibt (:numref:`sec_attention-at-scale`). Da die Aktivierungen nun mit 16 Bit vorliegen, können wir zum fusionierten FlashAttention-Kernel von cuDNN wechseln (`implementation='cudnn'`). Er hält die Attention-Scores auf dem Chip.

<!--en:999eb9ad-->
Wir erstellen `model16` als Instanz von `FusedGPT` mit denselben, in bf16 umgewandelten Parametern. Anschließend prüfen wir, ob seine Logits weiterhin bis auf 1% mit der fp32-Referenz `high` übereinstimmen, und messen den Trainingsschritt:

<!--en:a19098d0-->
Die fusionierte Attention von cuDNN erhöht den Durchsatz um ein weiteres Drittel. Die Logits stimmen weiterhin bis auf 1% überein. Zusammen erzielen bf16 und fusionierte Attention einen Beschleunigungsfaktor von fast $2\times$.

<!--en:6b78e7c5-->
### Größere Minibatches

<!--en:ec0b5f63-->
Ein größerer Minibatch verteilt die festen Kosten jedes Schritts auf mehr Token. Da das Gerät bereits den ganzen Schritt über beschäftigt ist, können wir nur solche Gerätekosten umlegen, die mit der Parameteranzahl wachsen und von der Tokenzahl unabhängig sind:

<!--en:a2b4f68a-->
- AdamW liest Gewichte, Gradienten und zwei Momente und schreibt die Gewichte und beide Momente ($28\text{ bytes/param}$ in fp32).

<!--en:811c3291-->
- Bei gemischter Genauigkeit wandeln wir die fp32-Hauptgewichte im Vorwärtsdurchlauf in bf16 und die bf16-Gewichtsgradienten im Rückwärtsdurchlauf wieder in fp32 um ($12\text{ bytes/param}$).

<!--en:0daa2ff8-->
Zusammen bewegen diese Operationen $40\text{ bytes/param}$, also $0.76\text{ GB/step}$. Bei einer Speicherbandbreite von $1.0\text{ TB/s}$ beansprucht diese feste Arbeit rund $0.75\text{ ms}$. Das entspricht etwa $1/16$ eines bf16-Schritts von $12\text{--}13\text{ ms}$ bei Batchgröße 64. Vergrößern wir den Minibatch um den Faktor $8\times$ von 64 auf 512 Fenster ($65{,}536$ Token), sparen wir pro Token $7/8$ dieser festen Kosten ein. Wir erwarten daher rund 5% mehr Durchsatz.

<!--en:e03485ba-->
Selbst ein kleiner Gewinn auf einer GPU lohnt sich, denn Datenparallelität skaliert nur dann gut, wenn die Schrittzeit pro GPU wesentlich länger ist als der Allreduce der Gradienten (:numref:`subsec_ft-dp`). Allerdings senken größere Batches die Dateneffizienz, sobald sie die kritische Batchgröße überschreiten (:numref:`sec_batch_size`, Aufgabe :exref:`ex_ft-target`). Da der Aktivierungsspeicher linear mit der Batchgröße wächst, messen wir zusätzlich zu den Gewichten und dem Optimiererzustand den maximalen weiteren Speicherbedarf (:numref:`subsec_mp-measuring`).

<!--en:a236c7f5-->
Bei Batchgröße 512 beanspruchen die Aktivierungen etwa ein Drittel der $24\text{ GiB}$ GPU-Speicher. Der Durchsatz steigt um einige Prozent (in Aufgabe :exref:`ex_ft-fixed` trennen Sie anhand verschiedener Batchgrößen die festen Kosten pro Schritt von den Kosten pro Token). Ein drittes Laufzeitprofil zeigt die Aufteilung bei Batchgröße 512:

<!--en:08549914-->
Das Gerät ist den ganzen Schritt über beschäftigt. Matrixprodukte einschließlich fusionierter Attention beanspruchen knapp 60% der Zeit. Der Rest entfällt auf fusionierte, bandbreitenbegrenzte Kernel für RMSNorm, SwiGLU, rotierende Positionseinbettungen und die Umwandlung zwischen fp32 und bf16.

<!--en:d3e5e3e9-->
### Aktivierungs-Checkpointing

<!--en:388405c0-->
Aktivierungs-Checkpointing (:numref:`subsec_mp-checkpointing`) spart Speicher, indem es im Vorwärtsdurchlauf nur die Eingabe jedes Blocks speichert und im Rückwärtsdurchlauf die internen Aktivierungen erneut berechnet. Bei Batchgröße 512 belegen wir nur ein Drittel des GPU-Speichers. Checkpointing ist hier also unnötig und kann den Schritt nur verlangsamen. Die Verlangsamung lässt sich auf zwei Arten vorhersagen:

<!--en:3a5cc03e-->
1. *Anhand der FLOP-Zahl*: Wiederholen wir den Vorwärtsdurchlauf, steigt der Rechenaufwand eines Blocks pro Token von $6$ auf $8\text{ FLOPs/param}$. Wir erwarten daher einen Anteil von $6/8 = 0.75$ des Durchsatzes ohne Checkpointing.

<!--en:aafccd39-->
2. *Anhand der Dauer des Vorwärtsdurchlaufs*: Wir messen die Dauer $t_{\text{fwd}}$ eines separaten Vorwärtsdurchlaufs bei Batchgröße 512 und berechnen $1 / (1 + t_{\text{fwd}} / t_{\text{step}})$.

<!--en:37418439-->
Wir vergleichen `t_forward` mit der gesamten Schrittzeit `t_step` bei Batchgröße 512. Der Quotient `1 / (1 + t_forward / t_step)` gibt an, welchen Anteil des Durchsatzes wir erwarten, wenn sich der Vorwärtsdurchlauf für jeden Block wiederholt:

<!--en:280ca79a-->
Beide Schätzungen stimmen überein: Der Vorwärtsdurchlauf beansprucht etwa ein Drittel des Schritts. Bei einer Wiederholung sollten also mindestens drei Viertel des Durchsatzes erhalten bleiben. Tatsächlich erwarten wir etwas mehr, da Einbettungen und Ausgabeprojektion vom Checkpointing ausgenommen sind. Wir kapseln nun jeden Block in `Checkpointed`, prüfen die Logits auf Gleichheit und messen sowohl den Durchsatz (`kept=False`) als auch den Aktivierungsspeicher.

<!--en:218b5197-->
`Checkpointed` ruft `torch.utils.checkpoint.checkpoint` auf.

<!--en:62cdf357-->
`Checkpointed` ruft `nnx.remat` auf.

<!--en:de8f8c35-->
Checkpointing senkt den Aktivierungsspeicher um rund zwei Drittel. Dabei bleibt mehr Durchsatz erhalten, als die separat gemessene Dauer des Vorwärtsdurchlaufs erwarten ließ.

<!--en:e5cedc44-->
Der Durchsatz sinkt auf $0.85\times$ und liegt damit über der Schranke von $0.77\times$. Die Aufschlüsselung nach Kerneln zeigt: Die erneute Berechnung der Matrixprodukte in den Blöcken kostet zusätzlich $14\text{ ms}$, fast so viel wie die $17\text{ ms}$ für alle Produkte des Vorwärtsdurchlaufs einschließlich Ausgabeprojektion. Für die übrigen Kernel kommen hingegen nur $3\text{ ms}$ statt $13\text{ ms}$ hinzu. Inductor fusioniert die erneut berechneten elementweisen Operationen direkt mit den Kerneln, die ihre Ergebnisse im Rückwärtsdurchlauf verwenden. So entsteht bei der Neuberechnung fast kein zusätzlicher Speicherverkehr.

<!--en:89a12121-->
Der Durchsatz sinkt auf $0.79\times$ und liegt damit über der Schranke von $0.76\times$, weil auch XLA die neu berechneten elementweisen Operationen mit dem Rückwärtsdurchlauf fusioniert (Aufgabe :exref:`ex_ft-remat`).

<!--en:42d40006-->
Checkpointing ist unverzichtbar, wenn das Modell oder die gewählte Batchgröße sonst den Gerätespeicher überfordern würde. Hier steht ausreichend Speicher zur Verfügung. Wir verzichten daher bei den folgenden Konfigurationen auf Checkpointing.

<!--en:51b150e0-->
## Datenparallelität

<!--en:2a9c3ff4-->
Bei unserem schnellsten Schritt auf einer GPU bleiben drei der vier RTX 4090 des Rechners ungenutzt. Bevor wir mehrere GPUs einsetzen, sagen wir mit dem Kostenmodell aus :numref:`subsec_mg-accounting` den Beschleunigungsfaktor durch Datenparallelität (engl. data parallelism) vorher.

<!--en:88d16aaa-->
### Vorhergesagter Beschleunigungsfaktor

<!--en:bde51d0f-->
Bei Datenparallelität (:numref:`sec_multi_gpu`) hält jede der $k$ GPUs eine Kopie des Modells und berechnet Gradienten für ihren eigenen Minibatch. Ein Allreduce summiert die Gradienten aller GPUs. Er umfasst $S$ Byte in $c$ Gradientenbündeln:

<!--en:aed69938-->
Unser PCIe-Rechner unterstützt keinen direkten Zugriff zwischen GPUs. Die Daten fließen über den Host-Speicher mit einer Busbandbreite von rund $2.4\text{ GB/s}$ (:numref:`sec_multi_gpu_concise`), entsprechend $\beta_{\textrm{link}} \approx 0.42\text{ ns/byte}$. Bei schwacher Skalierung (fester Batchgröße pro GPU) beträgt der Latenzbeitrag der Bündel $c\,\alpha$ weniger als eine Millisekunde. Gegenüber einer Schrittzeit von mehreren zehn Millisekunden ist er vernachlässigbar. Ohne Überlappung vereinfacht sich die Schrittzeit damit zu

<!--en:55ffa62c-->
und der Durchsatz steigt um den Faktor $k\, t_{\textrm{step}}(1) / (t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k))$. Eine zweite GPU erhöht den Durchsatz nur, wenn $t_{\textrm{allreduce}}(2) < t_{\textrm{step}}(1)$.

<!--en:1ec601a0-->
Für unser GPT mit 19 Millionen Parametern belegen die fp32-Gradienten $S = 76\text{ MB}$. Daraus ergeben sich $t_{\textrm{allreduce}}(2) \approx S\beta_{\textrm{link}} \approx 31\text{ ms}$ auf zwei GPUs und $t_{\textrm{allreduce}}(4) \approx 1.5\,S\beta_{\textrm{link}} \approx 47\text{ ms}$ auf vier:

<!--en:213a189d-->
- *Kompilierter fp32-Schritt bei Batchgröße 64 pro GPU* ($t_{\textrm{step}}(1) \approx 22\text{ ms}$): Der Allreduce dauert länger als der Schritt selbst. Nach :eqref:`eq_ft-dp` erwarten wir auf 2 GPUs eine *Verlangsamung* ($\sim 0.85\times$) und auf 4 GPUs nur den Faktor $\sim 1.3\times$.

<!--en:6a721c0d-->
- *Kompilierter bf16-Schritt bei Batchgröße 512 pro GPU* ($t_{\textrm{step}}(1) \approx 90\text{--}100\text{ ms}$): Der Schritt dauert wesentlich länger als der Allreduce. Wir erwarten den Faktor $\sim 1.5\times$ auf 2 GPUs und $2.6\text{--}2.7\times$ auf 4 GPUs.

<!--en:fce541e2-->
Da die Frameworks den Allreduce einzelner Gradientenbündel mit dem Rückwärtsdurchlauf überlappen (:numref:`fig_ddp_overlap`), liefert :eqref:`eq_ft-dp` meist eine untere Schranke für den Beschleunigungsfaktor. Der Abstand nach oben zeigt, wie viel Kommunikation die Berechnungen im Rückwärtsdurchlauf verdecken. Eine Garantie gibt es dafür nicht: Die Kopierkernel von NCCL konkurrieren mit dem Rückwärtsdurchlauf um dieselbe GPU (:numref:`subsec_mgp-scaling`). Diese Konkurrenz kann den Vorteil aufzehren.

<!--en:6d8d56d0-->
### Gemessener Beschleunigungsfaktor

<!--en:c0f81e82-->
`data_parallel(k, bf16, batch_size)` misst auf $k$ GPUs den Trainingsdurchsatz und auf denselben GPUs separat die Allreduce-Zeit für einen Gradientenpuffer von $76\text{ MB}$.

<!--en:602ec740-->
`DistributedDataParallel` führt mit `d2l.torchrun` einen Prozess je GPU aus (:numref:`sec_multi_gpu_concise`). Dabei überlappen sich die Allreduces der Gradientenbündel mit dem Rückwärtsdurchlauf. Die Kompilierung verzichtet auf CUDA-Graphen, da die Gerätezeit ausreicht, um die Startkosten auf dem Host zu verdecken. Für $k = 1$ gelten dieselben Messbedingungen.

<!--en:e596b4bf-->
JAX steuert alle $k$ GPUs aus einem Prozess (:numref:`sec_multi_gpu_concise`). Modell, Optimierer und Korpus werden auf dem gesamten `Auto`-Gerätegitter repliziert. Jeden Minibatch der Größe $k \times \text{batch\_size}$ partitioniert JAX entlang der Achse `'data'`. XLA fügt den Allreduce der Gradienten automatisch ein. Der in der ersten Zelle aktivierte Scheduler überlappt ihn mit dem Rückwärtsdurchlauf, um die Latenz zu verdecken.

<!--en:be9d7406-->
Wir führen `data_parallel` für den kurzen fp32-Schritt (`batch_size=64`) und den langen bf16-Schritt (`batch_size=512`) auf $k \in \{1, 2, 4\}$ GPUs aus. Dabei vergleichen wir die Vorhersage ohne Überlappung aus :eqref:`eq_ft-dp` mit dem gemessenen Beschleunigungsfaktor:

<!--en:63110679-->
Der separat gemessene Allreduce dauert $31\text{ ms}$ auf 2 GPUs und $48\text{ ms}$ auf 4 GPUs. Das stimmt bis auf eine Millisekunde mit unserer Schätzung aus der Übertragungsrate überein. Beide gegenläufigen Vorhersagen treffen zu. Bei Batchgröße 64 sind 2 GPUs langsamer als 1 GPU, und 4 GPUs erreichen weniger als den Faktor $1.5\times$. Bei Batchgröße 512 in bf16 erreichen 4 GPUs rund $3\times$ so viel Durchsatz wie eine einzelne GPU. Das ist der größte Gewinn durch eine einzelne Änderung in diesem Abschnitt. Durch die Überlappung der Kommunikation liegt der gemessene Beschleunigungsfaktor über der Untergrenze ohne Überlappung aus :eqref:`eq_ft-dp`. (Auf einem NVLink-Rechner mit $900\text{ GB/s}$ je Richtung dauert ein Allreduce von $76\text{ MB}$ nur einen Bruchteil einer Millisekunde. Dort skaliert das Training selbst bei Batchgröße 64 annähernd linear.)

<!--en:003645d0-->
## Ergebnisse

<!--en:252c39a3-->
Wir stellen nun alle Optimierungen für eine einzelne GPU gegenüber, prüfen, ob das Modell mit dem endgültigen Schritt weiterhin korrekt lernt, und schlüsseln dessen verbleibende Laufzeit auf.

<!--en:2027c912-->
### Vergleich der aufeinanderfolgenden Optimierungen

<!--en:3e1a2793-->
Die folgende Tabelle zeigt die Konfigurationen für eine GPU in ihrer Reihenfolge. Jeder Faktor unter `step` bezieht sich auf die Zeile darüber. `total` gibt den kumulierten Beschleunigungsfaktor gegenüber der Baseline an. Unter dem Trennstrich steht Checkpointing als Variante, die wir nicht beibehalten haben. Für `TFLOP/s` rechnen wir den Durchsatz mit $6\text{ FLOPs}$ pro Parameter und Token in eine Rechenrate um. Für `MFU` teilen wir diese durch die bf16-Spitzenleistung der RTX 4090 von $165.2\text{ TFLOP/s}$ (*Auslastung durch Modell-FLOPs*, engl. model FLOPs utilization, :citet:`chowdhery2022palm`). In Aufgabe :exref:`ex_ft-mfu` kommen die FLOPs für die Attention-Scores hinzu.

<!--en:45c016b6-->
Zusammen verdoppeln die Optimierungen auf einer GPU den Trainingsdurchsatz. Wir erreichen rund 50% der bf16-Spitzenleistung der RTX 4090. Fast der gesamte Gewinn stammt aus schnelleren Matrixprodukten (bf16) und geringerem Speicherverkehr zwischen ihnen (Kernel-Fusion). Der Abbau der Startkosten auf dem Host und der größere Minibatch tragen jeweils einige Prozent bei, wie die Laufzeitprofile vorhersagten.

<!--en:2913e8a1-->
Jeder Beschleunigungsfaktor hängt von der Reihenfolge der Änderungen (Aufgabe :exref:`ex_ft-order`) und vom Verhältnis der Hardwareleistungen ab. Auf einer schnelleren GPU oder bei einem kleineren Modell ist die Gerätezeit kürzer. Dann wirkt sich der Wegfall der Startkosten auf dem Host viel stärker aus.

<!--en:e4b7c04f-->
### Überprüfung des Trainings

<!--en:9b5dbffe-->
Ein höherer Durchsatz ist wertlos, wenn eine Optimierung unbemerkt das Training beeinträchtigt. Wir prüfen das mit zwei identisch initialisierten Modellen. Eines trainieren wir mit dem Baseline-Schritt, das andere mit dem endgültigen kompilierten bf16-Schritt. Beide erhalten dieselben 200 Minibatches mit je 512 Fenstern ($\sim 13\text{M}$ Token). Wir erhöhen die Lernrate in den ersten 50 Schritten linear, um späte Verlustspitzen bei AdamW zu verhindern (:numref:`sec_gpt`). Sowohl beim mittleren Verlust der letzten 20 Schritte als auch beim Validierungsverlust dürfen die Modelle höchstens um $0.15\text{ nats/byte}$ voneinander abweichen (`d2l.lm_perplexity`, :numref:`sec_test_models`). Außerdem prüfen wir, ob beide die Bigramm-Baseline von $2.45\text{ nats/byte}$ aus :numref:`sec_gpt` unterbieten.

<!--en:f5256a3c-->
Wir vergleichen den Trainingsverlust beider Modelle in den Schritten 1, 50, 100 und 200 sowie den Mittelwert der letzten 20 Schritte (`late`) und die logarithmierte Perplexität auf den Validierungsdaten (`validation`). Wir prüfen, ob beide Läufe unter der Bigramm-Schwelle von $2.45\text{ nats/byte}$ liegen und höchstens um $0.15\text{ nats/byte}$ voneinander abweichen:

<!--en:93c33b05-->
Beide Kurven beginnen bei $\log 256 \approx 5.55$, liegen über alle 200 Schritte dicht beieinander und erreichen nahezu identische Validierungsverluste deutlich unter der Bigramm-Schwelle. Nach 200 Schritten mit Batchgröße 512 ($13\text{M}$ Token) ist der Verlust allerdings höher als nach 1000 Schritten mit Batchgröße 64 und kosinusförmig abnehmender Lernrate ($8\text{M}$ Token) in :numref:`sec_gpt`. Der reine Tokendurchsatz und die Zeit bis zum Zielverlust sind also unterschiedliche Größen (Aufgabe :exref:`ex_ft-target`).

<!--en:cc8329d4-->
### Zeitbilanz des endgültigen Schritts

<!--en:0ea6bf23-->
Abschließend rechnet `account` die Laufzeitprofile der Baseline und des endgültigen Schritts in Millisekunden pro $8{,}192$ Token um. Wir vergleichen die tatsächlich verstrichene Gesamtzeit mit der idealen Rechenzeit bei Spitzenleistung, der gemessenen Kernel-Laufzeit für Matrixprodukte und der Zeit für die übrigen Kernel.

<!--en:4eba1428-->
Pro $8{,}192$ Token benötigt der endgültige Schritt rund $7\text{ ms}$ für Matrixprodukte. Diese erreichen über 80% der bf16-Spitzenleistung, bei der die Rechnung nur $5.7\text{ ms}$ dauern würde. Weitere $5\text{ ms}$ entfallen auf die bandbreitenbegrenzten Kernel zwischen den Produkten. Das ist eine typische Aufteilung nach dem Roofline-Modell: Die Kontraktionen leisten über 99% der FLOPs in rund 60% der Zeit :cite:`Ivanov.Dryden.Ben-Nun.ea.2021`. Bei einer Aufteilung von $7\text{ ms}$ zu $5\text{ ms}$ genügt weder schnellere Arithmetik allein (zum Beispiel fp8) noch der Wegfall des Speicherverkehrs der übrigen Kernel, um die Schrittzeit zu halbieren. Für einen weiteren Beschleunigungsfaktor von $2\times$ brauchen wir beides.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:95040ba4-->
Ein Kernel-Laufzeitprofil zeigt, was einen Trainingsschritt begrenzt und wie viel jede mögliche Optimierung höchstens einspart. Geringere Startkosten auf dem Host sparen höchstens die Leerlaufzeit des Geräts ein, Kernel-Fusion höchstens die Gerätezeit außerhalb der Matrixprodukte. Verdoppelt ein anderes Zahlenformat den Durchsatz der Tensor Cores, liegt der Beschleunigungsfaktor zwischen $1/(1 - f/2)$ und $2$, wenn Matrixprodukte den Anteil $f$ der Schrittzeit beanspruchen. Sobald ein Engpass entfällt, verschieben sich die übrigen Zeitanteile. Deshalb prüfen wir jede Optimierung auf numerische Gleichwertigkeit und messen ihre Wirkung zusammen mit den vorherigen Optimierungen. Auf einer einzelnen RTX 4090 lief der Trainingsschritt unseres GPT mit 19 Millionen Parametern doppelt so schnell und erreichte 45–50% MFU. Fast der gesamte Gewinn stammte aus bf16-Matrixprodukten und Kernel-Fusion. Im endgültigen Schritt verteilte sich die Zeit etwa im Verhältnis 60/40 auf Matrixkontraktionen nahe der Spitzenleistung und bandbreitenbegrenzte Kernel. Das Kostenmodell für Datenparallelität $t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k)$ sagte schließlich beide Effekte vorher: die Verlangsamung auf zwei GPUs beim kurzen fp32-Schritt ($22\text{ ms}$ für den Schritt gegenüber $31\text{ ms}$ für den PCIe-Allreduce) und den Beschleunigungsfaktor von rund $3\times$ auf vier GPUs beim bf16-Schritt mit Batchgröße 512 ($90\text{--}100\text{ ms}$ für den Schritt gegenüber $48\text{ ms}$ für den Allreduce).

<!--en:2294243a-->
## Übungen

<!--en:94c0ab12-->
1. [code] **Auslastung durch Modell-FLOPs.** Das GPT-Modell aus :numref:`subsec_ft-subject` hat $L = 6$ Blöcke der Breite $d = 512$ und die Kontextlänge $T = 128$. Abgesehen von der Rundung bei SwiGLU enthält jeder Block $12d^2$ Gewichte in seinen Matrixprodukten. Das Training kostet etwa $6\text{ FLOPs}$ pro Gewicht und Token.
    1. Das Produkt $\mathbf{Q}\mathbf{K}^\top$ für die Attention-Scores und die Aggregation der Werte sind Matrixprodukte ohne Gewichtsparameter. Zählen Sie ihre Trainings-FLOPs pro Token und Block. Rechnen Sie für den Rückwärtsdurchlauf mit dem doppelten Aufwand des Vorwärtsdurchlaufs und zählen Sie maskierte Einträge mit. Zeigen Sie, dass das Verhältnis zu den FLOPs der gewichteten Produkte $T/(6d)$ beträgt. Berechnen Sie es für $T = 128$ und $T = 1{,}024$.
    1. Berechnen Sie die Spalten `TFLOP/s` und `MFU` der Vergleichstabelle erneut, diesmal einschließlich der Attention-FLOPs. Bestimmen Sie auch für jede Konfiguration den Anteil an der Spitzenleistung ihres Zahlenformats ($165\text{ TFLOP/s}$ in bf16, $83\text{ TFLOP/s}$ in tf32).
    1. Nennen Sie zwei Konfigurationen oder Kernel aus :numref:`subsec_ft-rungs`, bei denen sich die theoretischen Modell-FLOPs von den tatsächlich auf der Hardware ausgeführten FLOPs unterscheiden. Geben Sie die Richtung jeder Abweichung an.
   :exlabel:`ex_ft-mfu`

<!--en:060568ca-->
1. [code] **Feste Kosten eines Schritts.** Modellieren Sie die bf16-Schrittzeit bei einer Batchgröße von $B$ Fenstern als $t(B) = a + bB$ mit festen Kosten $a$ pro Schritt und Kosten $b$ pro Fenster.
    1. Messen Sie die Schrittzeit für $B = 32, 64, 128, 256, 512$. Schätzen Sie $a$ und $b$ mit der Methode der kleinsten Quadrate. Geben Sie an, welcher Anteil der Schrittzeit bei $B = 64$ auf $a$ entfällt.
    1. Für AdamW-Updates und die Umwandlung der Gewichte zwischen fp32 und bf16 haben wir $\sim 0.75\text{ ms/step}$ geschätzt. Welchen Anteil von $a$ erklärt das? Ermitteln Sie weitere von $B$ unabhängige Arbeiten und überprüfen Sie Ihre Antwort anhand eines Laufzeitprofils bei $B = 32$.
    1. Sagen Sie mit Ihrem angepassten linearen Modell den Durchsatz bei $B = 1{,}024$ und $B = 2{,}048$ vorher. Prüfen Sie, welche Batchgrößen in $24\text{ GiB}$ GPU-Speicher passen, und messen Sie deren Durchsatz.
   :exlabel:`ex_ft-fixed`

<!--en:04793dfa-->
1. [code] **Kosten der Neuberechnung.** Das Aktivierungs-Checkpointing in :numref:`subsec_ft-rungs` lieferte einen höheren Durchsatz, als sowohl die FLOP-Zahl als auch die separate Zeitmessung für den Vorwärtsdurchlauf erwarten ließen.
    1. Erfassen Sie mit `d2l.kernel_times` Laufzeitprofile für den separaten Vorwärtsdurchlauf und den Schritt mit Checkpointing. Vergleichen Sie die durch Checkpointing hinzugekommenen Kernel mit denen des separaten Vorwärtsdurchlaufs. Welche treten unverändert auf, welche sind mit Kerneln des Rückwärtsdurchlaufs fusioniert?
    1. Messen Sie die Zeit für den Vorwärtsdurchlauf allein durch die sechs Blöcke (ohne Einbettungen und Ausgabeschicht) und berechnen Sie die Untergrenze für den Durchsatz erneut.
    1. Wenden Sie Checkpointing nur auf jeden zweiten Block an. Sagen Sie vor dem Lauf anhand Ihrer bisherigen Messungen Durchsatz und Speicherbedarf vorher. Messen und vergleichen Sie anschließend.
   :exlabel:`ex_ft-remat`

<!--en:83fc8f10-->
1. [code] **Kausalitätsprüfung.** Angenommen, ein fehlerhafter Attention-Kernel ignoriert unbemerkt die kausale Maske.
    1. Erklären Sie, weshalb weder der Trainingsverlust noch der Verlust auf den Validierungsfenstern noch die Bigramm-Schwelle von $2.45\text{ nats/byte}$ in :numref:`subsec_ft-waterfall` diesen Fehler erkennen ließen, wenn Baseline und optimiertes Modell den fehlerhaften Kernel verwendeten.
    1. Schreiben Sie einen Test, der das letzte Eingabetoken jeder Sequenz eines Minibatchs verändert. Prüfen Sie darin, ob die Logits an allen früheren Positionen innerhalb der numerischen Toleranz unverändert bleiben. Führen Sie den Test mit `learner` und mit einer Kopie aus, deren Attention die kausale Maske weglässt.

<!--en:e4357aea-->
1. **Durchsatz und Zeit bis zum Zielverlust.** Nach dem Modell aus :numref:`sec_batch_size` benötigt das Training bei Batchgröße $b$ insgesamt $E(b) = E_{\min}(1 + b / b_{\textrm{noise}})$ Trainingsbeispiele, um einen vorgegebenen Verlust zu erreichen. Dabei ist $b_{\textrm{noise}}$ die Skala des Gradientenrauschens.
    1. Sei $\rho > 1$ das Durchsatzverhältnis von Batchgröße 512 zu Batchgröße 64. Zeigen Sie, dass Batchgröße 512 den Zielverlust genau dann in kürzerer Zeit erreicht, wenn
       $$b_{\textrm{noise}} > \frac{512 - 64\rho}{\rho - 1},$$
       und berechnen Sie diese Schwelle für $\rho = 1.2$, $\rho = 1.7$ und den Messwert von $\rho$ aus :numref:`subsec_ft-waterfall`.
    1. Ordnen Sie jede Änderung aus :numref:`subsec_ft-rungs`, die Datenparallelität mit schwacher Skalierung und den Wechsel zu Muon (:numref:`sec_muon`) danach ein, ob sie $E(b)$, den Tokendurchsatz oder beides verändert.
   :exlabel:`ex_ft-target`

<!--en:e2ef6486-->
1. [code] **Breite 256.** Mit `num_hiddens=256` entspricht `d2l.GPT` der in :numref:`sec_gpt` trainierten Architektur.
    1. Bei der Ausgabeprojektion werden Matrizen der Größen $8{,}192 \times 256$ und $256 \times 256$ multipliziert. Berechnen Sie für diese Projektion $\mathbf{W}_o$ die arithmetische Intensität in tf32 mit 4 Byte und bf16 mit 2 Byte. Tragen Sie beide Werte in die Roofline der RTX 4090 ein (:numref:`fig_roofline`) und sagen Sie den Beschleunigungsfaktor durch bf16 für dieses Produkt vorher.
    1. Führen Sie die Baseline und die ersten beiden Optimierungen aus :numref:`subsec_ft-rungs` mit `num_hiddens=256` aus. Erfassen Sie deren Laufzeitprofile mit `profile` und `account`. Vergleichen Sie den gemessenen Beschleunigungsfaktor für die Matrixprodukte mit Ihrer Roofline-Vorhersage.
   :exlabel:`ex_ft-width`

<!--en:dcf15536-->
1. [code] **Reihenfolge der Änderungen.** Jeder Beschleunigungsfaktor in der Vergleichstabelle hängt von den zuvor angewandten Optimierungen ab.
    1. Sagen Sie anhand des Laufzeitprofils der Baseline und der Schranke aus :numref:`subsec_ft-precision` den Beschleunigungsfaktor vorher, wenn Sie bf16 direkt auf die unkompilierte Baseline anwenden.
    1. Messen Sie diesen Beschleunigungsfaktor. Vergleichen Sie ihn mit Ihrer Vorhersage und mit der Zeile `+ bf16` der Vergleichstabelle. Wenden Sie anschließend die Kompilierung (zweite Tabellenzeile) auf die bf16-Konfiguration an. Erklären Sie anhand der Kernel-Laufzeitprofile, wie sich beide Faktoren verschoben haben.
   :exlabel:`ex_ft-order`

<!--en:3fecd44e-->
[Diskussion](https://d2l.discourse.group/t/36793)

<!--en:6367036c-->
Fallstudie: Schritt für Schritt zu einem schnelleren Transformer<br>**Amdahl-Schranken aus Laufzeitprofilen · Fusion und CUDA-Graphen · bf16 und FlashAttention · Batchgröße · Skalierung auf mehrere GPUs · MFU**

<!--en:4942df79-->
::: {.slide title="Ein GPT-Schritt braucht 6 ms Rechenzeit und dauert 25 ms"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:d8e61ac9-->
- **Grenze der Hardware ($5.7\text{ ms}$):** Ein Schritt des GPT mit 6 Blöcken der Breite 512 ($N = 19\text{M}$, $n = 8{,}192$ Token) erfordert $6Nn \approx 0.93\text{ TFLOP}$. Dafür genügen $5.7\text{ ms}$ bei $165.2\text{ TFLOP/s}$ (`bf16`).

<!--en:5f6a1f28-->
- **Gemessene `fp32`-Baseline ($\sim 25\text{ ms}$):** Der Schritt dauert $> 4\times$ so lange wie die reine Rechnung in `bf16` ($\sim 23\%$ MFU).

<!--en:f5812e3c-->
- **Die Differenz von $19\text{ ms}$:** Rund $6\text{ ms}$ erklärt der unterschiedliche Durchsatz der Tensor Cores in `tf32` und `bf16`. Die übrigen $\sim 13\text{ ms}$ entfallen auf unfusionierte bandbreitenbegrenzte Kernel und Pausen zwischen Kernelstarts.

<!--en:83e754d7-->
::: {.slide title="Modellbreite 512: Matrixprodukte oberhalb von I*"}

<!--en:b5d4a230-->
[Einordnung in die Roofline]{.kicker}

<!--en:dc7dce8d-->
- **Oberhalb des Knickpunkts:** Bei $d = 512$ und $n = 8{,}192$ Token erreichen die Blockprojektionen $I = 125\text{--}185\text{ FLOP/B}$ in `tf32` ($> I^* = 82$) und $250\text{--}370\text{ FLOP/B}$ in `bf16` ($> I^* = 164$).

<!--en:9f2dd1c3-->
- **Warum Breite 512 zählt:** Jede GEMM liegt deutlich im rechenbegrenzten Bereich. Verdoppelt sich die Spitzenleistung $P$ der Tensor Cores beim Wechsel von `tf32` zu `bf16`, halbiert sich unmittelbar die GEMM-Laufzeit.

<!--en:7326e33b-->
::: {.slide title="Versuchsaufbau und Zeitmessung für den gesamten Trainingsschritt"}

<!--en:8bf89114-->
[Vergleichbare Messungen]{.kicker}

<!--en:f8177a47-->
Für vergleichbare Ergebnisse bleiben Modell, Datenpipeline und Messverfahren auf allen Optimierungsstufen gleich:

<!--en:9e4b131a-->
- **Keine Störeinflüsse durch Host-Transfers:** Das gesamte Byte-Korpus von WikiText-103 ($270\text{ MB}$) liegt im DRAM der GPU. Jeder Schritt zieht zufällige Fenster von je $128$ Byte direkt auf dem Gerät.

<!--en:8680a8c1-->
- **Kompakter Parameterzustand:** Bei $16\text{ B/param}$ belegt der AdamW-Zustand für $19\text{M}$ Parameter nur $0.3\text{ GiB}$. Gradientenakkumulation und FSDP-Partitionierung sind damit überflüssig.

<!--en:d5793cd5-->
- **Synchronisierte Messung über mehrere Fenster (`tokens_per_second`):** Auf $20$ Aufwärmschritte folgen $3$ Messfenster mit je $50$ Schritten zwischen Gerätesynchronisationen. Die Ausgabe enthält den Median in $\text{tokens/s}$, den Beschleunigungsfaktor gegenüber der Vorstufe und die Streuung der Fenster ($\textrm{range}/\textrm{median}$).

<!--en:685a067c-->
- **Korrektheitsprüfung vor der Messung:** Vor dem Benchmark steht auf jeder Stufe die Prüfung auf numerische Übereinstimmung (mit `allclose` für Logits oder Gewichtsänderungen).

<!--en:146c8788-->
::: {.slide title="Schranken für den nächsten Gewinn aus dem bisherigen Laufzeitprofil"}

<!--en:82dd7802-->
[Vorhersageverfahren]{.kicker}

<!--en:bbba3651-->
- **Schranken für Kernelstart und Fusion:** Ohne Wartezeiten zwischen Kernelstarts beträgt der Beschleunigungsfaktor höchstens $t_{\textrm{step}}/t_{\textrm{busy}}$. Bei Fusion der Kernel außerhalb der Matrixprodukte ($t_{\textrm{other}}$) liegt die Obergrenze bei $t_{\textrm{step}}/t_{\textrm{products}}$.

<!--en:81b2f320-->
- **Rechengenauigkeit (`tf32` $\to$ `bf16`):** Beanspruchen GEMMs den Anteil $f$ der Schrittzeit, liefert der doppelte GEMM-Durchsatz einen Faktor $\ge \frac{1}{1 - f/2}$ (bis zu $2.0\times$, wenn sich auch der Speicherverkehr der Aktivierungen halbiert).

<!--en:28ab3e72-->
- **Batchgröße und Neuberechnung:** Die Vergrößerung $B \to 8B$ verteilt nur die festen Kosten von $40\text{ B/param}$ auf mehr Token ($\le 1.05\times$). Die Neuberechnung kostet $+2N\text{ FLOPs/tok}$ ($\ge 0.75\times$).

<!--en:fc6fa6e9-->
::: {.slide title="Stufe 0 (Baseline): Laufzeitprofil des fp32-Schritts im Eager-Modus" only="pytorch"}

<!--en:0605a104-->
[Laufzeitprofil der Baseline]{.kicker}

<!--en:dda1a5a5-->
Das Laufzeitprofil zeigt die `float32`-Baseline im Eager-Modus (`tf32`-Matrixprodukte, fusioniertes AdamW) bei Batchgröße $64$ ($8{,}192$ Token/Schritt). So verteilen sich die $\sim 25\text{ ms}$:

<!--en:7fb32258-->
- **Diagnose:** Die GPU arbeitet während fast $100\%$ der Schrittzeit von $25\text{ ms}$. Die Startkosten begrenzen den Schritt also nicht. Matrixprodukte beanspruchen $\sim 64\%$ ($15.6\text{ ms}$), kurze bandbreitenbegrenzte Kernel für elementweise Operationen und Normalisierungen die übrigen $\sim 36\%$ ($8.8\text{ ms}$).

<!--en:a4a97a53-->
- **Amdahl-Schranke für die Kompilierung:** Selbst wenn Fusion und CUDA-Graphen *alle* Kernel außerhalb der Matrixprodukte und alle Startpausen beseitigten, sänke die Schrittzeit höchstens wie folgt: $25\text{ ms} \to 15.6\text{ ms}$ (**Beschleunigungsfaktor $\le 1.6\times$**).

<!--en:a3489b5a-->
::: {.slide title="Stufe 0 (Baseline): Laufzeitprofil des JIT-kompilierten fp32-Schritts" only="jax"}

<!--en:0605a104-2-->
[Laufzeitprofil der Baseline]{.kicker}

<!--en:91fc2468-->
In JAX ist die Baseline `train_step` bereits mit `nnx.jit` kompiliert. XLA hat daher die Folgen elementweiser Operationen bereits fusioniert:

<!--en:7e984dcb-->
- **Diagnose:** Die GPU arbeitet während $98\%$ der Schrittzeit von $\sim 24\text{ ms}$. Matrixprodukte beanspruchen $\sim 65\%$, die fusionierten XLA-Kernel zusammen mit der unfusionierten XLA-Attention $\sim 35\%$.

<!--en:53786ec2-->
- **Amdahl-Schranke für den Abbau der Host-Kosten:** Das Gerät wartet nur während $2\%$ der Schrittzeit auf den Host. Weniger Startaufwand in Python spart auf dieser Stufe **höchstens $2\%$** ein. Sein Wegfall verhindert aber einen Engpass auf späteren Stufen mit kürzerer Gerätezeit.

<!--en:bab59cb4-->
::: {.slide title="Stufe 1: Fusion durch Inductor und Wiedergabe von CUDA-Graphen" only="pytorch"}

<!--en:30e3014a-->
[Compiler und Graphwiedergabe]{.kicker}

<!--en:c5f8198c-->
Die Kompilierung des Modells mit `torch.compile(model, dynamic=False, mode='reduce-overhead')` verbindet die Operatorfusion von Inductor mit der Wiedergabe von CUDA-Graphen für feste Tensorformen:

<!--en:b54fa128-->
- **Ergebnis (Beschleunigungsfaktor $1.16\times$):** Die Logits stimmen bis auf $1.5 \times 10^{-5}$ überein. Die Fusion spart $2.2\text{ ms}$ bei den Kerneln außerhalb der Matrixprodukte ($24.5 \to 22.3\text{ ms}$). CUDA-Graphen beseitigen $\sim 1\text{ ms}$ Wartezeit zwischen Kernelstarts.

<!--en:10160fb3-->
::: {.slide title="Stufe 1: jit_partial erspart den PyTree-Durchlauf" only="jax"}

<!--en:3110f490-->
[Geringere Startkosten auf dem Host]{.kicker}

<!--en:6cffac25-->
Ein Aufruf ohne Operation über `nnx.jit` zeigt: Das Durchlaufen des Python-Referenzgraphen von `nnx.Module` und `nnx.Optimizer` kostet fast **$9\text{ ms}$ CPU-Zeit pro Schritt**. Das fällt nur deshalb nicht auf, weil der GPU-Schritt $24\text{ ms}$ dauert:

<!--en:768fbf32-->
- **`nnx.jit_partial(..., graph=False)`:** Im Baummodus werden Modell und Optimiererzustand einmalig gebunden. Jeder Schritt übergibt nur den Minibatch `(X, Y)`. Die Startzeit auf dem Host sinkt von $\sim 9\text{ ms}$ auf $0.23\text{ ms}$. Der Durchsatz steigt bereits jetzt um $\sim 1\%$, und die Untergrenze von $9\text{ ms}$ durch den Host entfällt auf allen weiteren Stufen.

<!--en:356278b0-->
::: {.slide title="Laufzeitprofil nach Stufe 1: Zwei Drittel der Zeit für Matrixprodukte"}

<!--en:9dbfbf5d-->
[Zweites Laufzeitprofil]{.kicker}

<!--en:9360717d-->
Nach Stufe 1 arbeitet die GPU während $99\text{--}100\%$ der Schrittzeit. Matrixprodukte beanspruchen $f \approx 65\text{--}71\%$ der Gerätezeit:

<!--en:cc2e20e9-->
- **Vorhersage für Stufe 2 (gemischte Genauigkeit mit `bfloat16`):** Der Wechsel von `tf32` ($82.6\text{ TFLOP/s}$) zu `bfloat16` ($165.2\text{ TFLOP/s}$) halbiert die Zeit für Matrixprodukte auf den Tensor Cores.

<!--en:6ef42472-->
- **Theoretische Schranken:** Verdoppelt sich *nur* die Geschwindigkeit der Matrixprodukte ($f \approx 0.68$), beträgt der Beschleunigungsfaktor $\frac{1}{1 - f/2} \approx \mathbf{1.52\times}$. Halbieren die übrigen Kernel ihren Speicherverkehr durch 2-Byte-Aktivierungen ebenfalls, nähert sich der Faktor **$2.0\times$**.

<!--en:150996e3-->
::: {.slide title="Stufe 2: Gemischte Genauigkeit mit bf16 beschleunigt den Schritt um den Faktor 1.75×" only="pytorch"}

<!--en:cdef7d8b-->
[Geringere Rechengenauigkeit]{.kicker}

<!--en:33aebc4e-->
Mit `torch.autocast('cuda', torch.bfloat16)` laufen lineare Schichten und fusionierte FlashAttention (`F.scaled_dot_product_attention`) in `bfloat16`. Hauptgewichte, Normalisierungen, Residualstrom und Verlust bleiben in `float32`:

<!--en:191eb09d-->
- **Warum $1.74\times$ im Intervall $[1.55\times, 2.0\times]$ liegt:** Die Logits stimmen bis auf $0.2\%$ überein (Prüfkriterium: $< 1\%$). Der Beschleunigungsfaktor übertrifft die allein aus Matrixprodukten abgeleitete Schranke von $1.55\times$. GEMMs in `bfloat16` nutzen die Tensor Cores besser aus als in `tf32`. Zudem bewegen die fusionierten Kernel zwischen den Produkten 2-Byte-Tensoren in `bfloat16` anstelle von 4-Byte-Tensoren in `float32`.

<!--en:ef455a2d-->
::: {.slide title="Warum bf16 die Amdahl-Schranke für Matrixprodukte allein übertrifft" only="pytorch"}

<!--en:1b42d278-->
[Rechengenauigkeit und Bandbreite]{.kicker}

<!--en:b257f906-->
Der Beschleunigungsfaktor von $1.75\times$ durch `torch.autocast('cuda', torch.bfloat16)` zeigt bei genauerer Betrachtung, wie geringere Rechengenauigkeit in beiden Bereichen der Roofline zugleich Zeit spart:

<!--en:f8ce5e5f-->
- **Rechenbegrenzte Kernel (GEMMs und FlashAttention):** Der Spitzendurchsatz der Tensor Cores verdoppelt sich ($82.6 \to 165.2\text{ TFLOP/s}$). Die kleineren 16-Bit-Operanden entlasten die Register und verringern Bankkonflikte im gemeinsam genutzten Speicher.

<!--en:98cdb17f-->
- **Bandbreitenbegrenzte Kernel (SwiGLU, rotierende Positionseinbettungen, Tensoransichten):** Halb so große Aktivierungselemente ($4\text{ B} \to 2\text{ B}$) halbieren die im DRAM gelesene und geschriebene Datenmenge für jeden Zwischentensor zwischen linearen Projektionen.

<!--en:31c9f879-->
- **Was in `float32` bleibt und den vollen Faktor $2.0\times$ verhindert:** Hauptgewichte, Gradienten, AdamW-Momente ($28\text{ B/param}$), RMSNorm-Reduktionen, Additionen im Residualstrom und logarithmierte Wahrscheinlichkeiten für die Kreuzentropie behalten aus Gründen der numerischen Stabilität 32 Bit.

<!--en:b5d11595-->
::: {.slide title="Stufe 2a: Gemischte Genauigkeit mit bf16 beschleunigt den Schritt um den Faktor 1.45×" only="jax"}

<!--en:cdef7d8b-2-->
[Geringere Rechengenauigkeit]{.kicker}

<!--en:d3d90e17-->
`d2l.to_bf16` setzt `dtype=jnp.bfloat16` für alle linearen Projektionen. Hauptparameter, Normalisierungen, Einbettungen und Residualstrom bleiben in `float32`:

<!--en:cac5dbca-->
- **Warum `bf16` allein den Faktor $1.45\times$ liefert (nahe der Schranke $1.48\times$ für Matrixprodukte allein):** Standardmäßig nutzt `jax.nn.dot_product_attention` eine XLA-Implementierung, die die gesamte Attention-Matrix der Größe $T \times T$ ins DRAM schreibt.

<!--en:0815153d-->
::: {.slide title="Stufe 2b: Fusionierte FlashAttention von cuDNN bringt einen weiteren Faktor 1.35×" only="jax"}

<!--en:a672c839-->
[Attention auf dem Chip]{.kicker}

<!--en:6cf45914-->
Q, K und V liegen nun in `bfloat16` mit 16 Bit vor. Damit lässt sich `jax.nn.dot_product_attention` auf den fusionierten FlashAttention-Kernel von cuDNN umstellen (`implementation='cudnn'`). Er berechnet die Attention kachelweise im SRAM auf dem Chip, ohne die Score-Matrix der Größe $T \times T$ ins DRAM zu schreiben:

<!--en:d7abde49-->
- **Gemeinsamer Gewinn durch `bf16` und FlashAttention ($\approx 1.96\times$):** Ohne den externen Speicherverkehr für die Attention-Matrix beschleunigt sich der Schritt um einen weiteren Faktor **$1.35\times$**. Zusammen mit der geringeren Rechengenauigkeit ergibt sich ein Faktor von fast $2\times$.

<!--en:25a9ab6c-->
::: {.slide title="Stufe 3: Vergrößerung des Minibatchs auf 512 Fenster"}

<!--en:05d74ae3-->
[Verteilung der festen Kosten]{.kicker}

<!--en:baa2b45f-->
Die GPU ist bereits zu $100\%$ ausgelastet. Wächst der Batch um den Faktor $8\times$ ($64 \to 512$ Fenster, $65{,}536$ Token/Schritt), lassen sich nur solche Gerätekosten umlegen, die mit der Parameteranzahl $N = 19\text{M}$ wachsen und von der Tokenzahl $n$ unabhängig sind:

<!--en:b31a5396-->
- **Fester Speicherverkehr pro Schritt ($40\text{ B/param} = 0.76\text{ GB}$):** Das fusionierte AdamW liest und schreibt insgesamt $28\text{ B/param}$ (Gewicht, Gradient und zwei Momente in `fp32`). Bei gemischter Genauigkeit wechseln Gewichte und Gradienten zwischen `fp32` und `bf16` ($12\text{ B/param}$). Zusammen dauert das $\sim 0.75\text{ ms}$ bei $1\text{ TB/s}$ ($\sim 6\%$ eines Schritts von $12.5\text{ ms}$ bei Batchgröße 64).

<!--en:9f9bf690-->
- **Vorhersage (Gewinn $\le 5\%$):** Die Einsparung von $7/8$ der festen Kosten von $0.75\text{ ms}$ je ursprünglicher Tokenmenge verspricht einige Prozent mehr Durchsatz. Zugleich verlängert sich der Schritt auf $\sim 95\text{ ms}$, sodass das Training trotz Allreduce auf mehrere GPUs skalieren kann.

<!--en:9c4a6f4d-->
::: {.slide title="Laufzeitprofil bei Batchgröße 512: 60% Matrixprodukte, 40% Speicherverkehr"}

<!--en:8f9e500b-->
[Drittes Laufzeitprofil]{.kicker}

<!--en:d35547f2-->
Bei Batchgröße $512$ belegen die Aktivierungen $\sim 8\text{ GiB}$. Damit bleiben zwei Drittel der $24\text{ GiB}$ GPU-Speicher frei. Das Laufzeitprofil zeigt die endgültige Zeitverteilung auf einer GPU:

<!--en:13bde0bc-->
- **Zeitanteile:** Die GPU arbeitet während $100\%$ der Schrittzeit von $\sim 91\text{--}100\text{ ms}$. Matrixprodukte und fusionierte Attention beanspruchen **$\sim 57\text{--}60\%$** des Schritts. Auf fusionierte bandbreitenbegrenzte Kernel (RMSNorm, SwiGLU, RoPE und Umwandlungen zwischen `fp32` und `bf16`) entfallen die übrigen **$\sim 40\%$**.

<!--en:eb46cc5a-->
::: {.slide title="Variante: Vorhersage der Kosten für Aktivierungs-Checkpointing"}

<!--en:af7c9de0-->
[Vorhersage für Checkpointing]{.kicker}

<!--en:747a1da8-->
Batchgröße $512$ passt gut in $24\text{ GiB}$. Was geschieht bei Aktivierungs-Checkpointing für jeden Block (Variante mit `kept=False`)? Zwei Abschätzungen für den verbleibenden Durchsatz:

<!--en:bb2f7699-->
1. **Anhand der FLOP-Zahl:** Die Wiederholung des Vorwärtsdurchlaufs jedes Blocks bei der Backpropagation erhöht den Rechenaufwand der Blöcke ($6N \to 8N\text{ FLOPs/token}$). Erwarteter Durchsatzfaktor: $6/8 = \mathbf{0.75\times}$.

<!--en:1ec3079a-->
2. **Anhand der separaten Vorwärtslaufzeit ($t_{\textrm{fwd}}$):** Aus der Dauer des Vorwärtsdurchlaufs allein ergibt sich ein erwarteter Durchsatzfaktor von $\frac{1}{1 + t_{\textrm{fwd}} / t_{\textrm{step}}} \approx \mathbf{0.76\text{--}0.77\times}$.

<!--en:5ddd43a2-->
::: {.slide title="Checkpointing im Test: Fusionierte Neuberechnung übertrifft die Untergrenze"}

<!--en:4e066523-->
[Messung mit Checkpointing]{.kicker}

<!--en:a8256335-->
Kapselt `Checkpointed` alle 6 Transformer-Blöcke, sinkt der Aktivierungsspeicher um zwei Drittel ($7\text{--}9\text{ GiB} \to 2.6\text{--}3.2\text{ GiB}$). Dabei bleibt der Durchsatzfaktor **$0.79\text{--}0.85\times$** erhalten. Er liegt über beiden Untergrenzen: $0.75\times$ aus der FLOP-Zahl und $0.77\times$ aus der Zeitmessung für den Vorwärtsdurchlauf:

<!--en:8560b614-->
- **Warum Neuberechnen weniger kostet als ein separater Vorwärtsdurchlauf:** Die Matrixprodukte der Blöcke müssen vollständig neu berechnet werden ($+14\text{ ms}$). Die erneut berechneten elementweisen Kernel (SwiGLU, RoPE, Normalisierungen) fusioniert der Compiler direkt mit den Kerneln des Rückwärtsdurchlaufs, die ihre Ergebnisse verwenden. Dadurch entfällt der größte Teil der $13\text{ ms}$ für den separaten DRAM-Verkehr.

<!--en:9f9d86d2-->
::: {.slide title="Kumulierte Gewinne auf einer GPU: rund 2× so schnell, 45–50% MFU"}

<!--en:432b13ec-->
[Ergebnisse auf einer GPU]{.kicker}

<!--en:9b18928c-->
Für den Vergleich aller Stufen auf einer GPU rechnet die Tabelle den Tokendurchsatz in erreichte `TFLOP/s` ($6N\text{ FLOPs/token}$) und in die **Auslastung durch Modell-FLOPs** ($\textrm{MFU} = \textrm{achieved FLOP/s} / 165.2\text{ TFLOP/s}$) um:

<!--en:3188afaf-->
- **Ergebnis:** Zusammen **verdoppeln die Optimierungen den Durchsatz ($\sim 2.0\text{--}2.1\times$)** und erreichen auf der RTX 4090 **$\sim 45\text{--}50\%$ MFU**. Fast der gesamte Gewinn stammt von den Tensor Cores in `bfloat16` sowie der Fusion von Operatoren und Attention-Kerneln.

<!--en:cc46e389-->
::: {.slide title="Skalierung auf mehrere GPUs aus Schritt- und Übertragungszeiten vorhersagen"}

<!--en:d023c261-->
[Kostenmodell der Datenparallelität]{.kicker}

<!--en:fd3e0741-->
- **Feste Kosten des Allreduce von $76\text{ MB}$:** Über PCIe mit Umweg über den Host ($\beta_{\textrm{link}} \approx 0.42\text{ ns/B}$) dauert die Synchronisation von $19\text{M}$ Gradienten in `fp32` jeweils $31\text{ ms}$ ($k=2$) und $47\text{ ms}$ ($k=4$).

<!--en:06b282ed-->
- **Zwei gegenläufige Vorhersagen:** Bei Batchgröße $64$ ($t_{\textrm{step}}(1) \approx 22\text{ ms} < 31\text{ ms}$) **verlangsamt** $k=2$ das Training ($\sim 0.85\times$). Bei Batchgröße $512$ ($t_{\textrm{step}}(1) \approx 95\text{ ms}$) erreicht $k=4$ den Faktor **$\approx 2.6\text{--}2.7\times$**.

<!--en:13e31eb0-->
::: {.slide title="Gemessene Skalierung: Verlust bei Batchgröße 64, Faktor 3× bei Batchgröße 512"}

<!--en:f5d70ecf-->
[Messungen auf mehreren GPUs]{.kicker}

<!--en:de0b14f3-->
Ein Test mit Datenparallelität auf $k = 1, 2, 4$ GPUs prüft beide gegenläufigen Vorhersagen unter Praxisbedingungen:

<!--en:c787a631-->
**Beide Vorhersagen treffen zu:** Der separat gemessene Allreduce von $76\text{ MB}$ dauert $31\text{ ms}$ ($k=2$) und $48\text{ ms}$ ($k=4$). Bei Batchgröße $64$ verlangsamt die zweite GPU das Training ($\sim 0.85\text{--}0.95\times$). Bei Batchgröße $512$ in `bfloat16` erreichen $4$ GPUs **den Faktor $\sim 2.9\text{--}3.2\times$ gegenüber einer GPU** ($\sim 1.9\text{--}2.3\text{M tokens/s}$ oder **Faktor $6\text{--}7\times$ gegenüber der Baseline auf einer GPU**). Die Überlappung der Gradientenbündel mit dem Rückwärtsdurchlauf erklärt den Vorsprung gegenüber der Untergrenze ohne Überlappung.

<!--en:31f4b0f0-->
::: {.slide title="Trainingstest: Konvergiert das optimierte Modell wie die Baseline?"}

<!--en:1b718c47-->
[Numerische Überprüfung]{.kicker}

<!--en:b3724792-->
Ein auf einer GPU um den Faktor $2\times$ schnellerer Schritt ist nur dann brauchbar, wenn das Modell dieselbe Repräsentation lernt wie die `float32`-Baseline. Zwei identisch initialisierte Modelle (eines mit `baseline`, das andere mit `final`, also `+ batch 512` in `bf16`) erhalten im Training exakt dieselben $200$ Minibatches mit je $512$ Fenstern ($\sim 13\text{M}$ Token, lineares Anheben der Lernrate über 50 Schritte):

<!--en:cc303f68-->
- **Prüfung bestanden:** Beide Kurven beginnen bei $\ln 256 \approx 5.55$, weichen über alle Schritte um höchstens $0.05\text{ nats/byte}$ voneinander ab (insgesamt $200$ Schritte) und erreichen denselben Validierungsverlust ($1.5\text{--}1.6$, deutlich unter der Bigramm-Schwelle von $2.45$).

<!--en:f8e15f48-->
::: {.slide title="Durchsatz und Zeit bis zum Zielverlust"}

<!--en:0724e344-->
[Grenzen der Optimierung]{.kicker}

<!--en:1b2049e1-->
- **Kritische Batchgröße ($b_{\textrm{noise}}$):** Bei Batchgröße $b$ benötigt das Training insgesamt $E(b) = E_{\min}(1 + b/b_{\textrm{noise}})$ Token, um einen vorgegebenen Validierungsverlust zu erreichen.

<!--en:5c9e839a-->
- **Abnehmender Nutzen jenseits von $b_{\textrm{noise}}$:** Bei $b \gg b_{\textrm{noise}}$ kommen die Parameter pro Schritt trotz genauerer Gradientenschätzungen kaum weiter als bei kleineren Batches. Insgesamt sind mehr Token nötig.

<!--en:439dbb09-->
- **Faustregel:** Beschleunigungen bei gleicher Batchgröße (`compile`, `bf16`, FlashAttention, Datenparallelität mit starker Skalierung) verkürzen stets die Zeit bis zum Zielverlust. Eine Vergrößerung des Batchs hilft nur, solange $b \lesssim b_{\textrm{noise}}$.

<!--en:996ef7e6-->
::: {.slide title="Zeitbilanz des endgültigen Schritts in Millisekunden"}

<!--en:3f4b8d66-->
[Abschließende Bilanz]{.kicker}

<!--en:3d782c87-->
In Millisekunden pro $8{,}192$ Token aufgeschlüsselt zeigen die Kernel-Laufzeitprofile, wie sich jede Millisekunde der Baseline und des endgültigen Schritts zur idealen Rechenzeit bei Spitzenleistung der Tensor Cores verhält ($5.7\text{ ms}$ in `bf16`, $11.3\text{ ms}$ in `tf32`):

<!--en:9147372d-->
- **Warum der endgültige Schritt $\sim 50\%$ MFU erreicht ($11.5\text{--}12\text{ ms}$ gegenüber idealen $5.7\text{ ms}$):**
  - **Matrixprodukte ($\sim 7\text{ ms}$):** Sie erreichen **$> 80\%$ der Hardware-Spitzenleistung (ideale Zeit: $5.7\text{ ms}$)** und erledigen dabei $> 99\%$ aller Rechenoperationen.
  - **Bandbreitenbegrenzte Kernel ($\sim 5\text{ ms}$):** Normalisierungen, SwiGLU, RoPE, Typumwandlungen und AdamW leisten $< 1\%$ der FLOPs. Der Datentransport durch das DRAM kostet sie jedoch **$\sim 40\%$ der Schrittzeit**.

<!--en:a2c01b60-->
::: {.slide title="Verständnisfragen: Amdahl-Schranken, Skalierung und weitere Optimierungen"}

<!--en:929b7671-->
[Verständnistest]{.kicker}

<!--en:fca6ba74-->
Drei Fragen anhand der endgültigen Zeitbilanz beantworten: $7\text{ ms}$ für Matrixprodukte nahe der `bf16`-Spitzenleistung plus $5\text{ ms}$ für bandbreitenbegrenzte Kernel außerhalb der Matrixprodukte $= 12\text{ ms}$ pro $8{,}192$ Token.

<!--en:79c0bc38-->
1. Angenommen, die Matrixprodukte wechseln von `bfloat16` ($165\text{ TFLOP/s}$) auf `fp8` ($330\text{ TFLOP/s}$). Ihre Rechenzeit halbiert sich ($7\text{ ms} \to 3.5\text{ ms}$), während die $5\text{ ms}$ für die übrigen Kernel erhalten bleiben. Wie groß ist der Beschleunigungsfaktor für den Schritt?

<!--en:1e38a92b-->
2. Angenommen, speziell von Hand geschriebene große Kernel beseitigen durch Fusion *alle* $5\text{ ms}$ für den Speicherverkehr außerhalb der Matrixprodukte. Die `bfloat16`-Matrixprodukte dauern weiterhin $7\text{ ms}$. Wie groß ist der Beschleunigungsfaktor für den Schritt?

<!--en:3beb5653-->
3. Was wäre für einen weiteren **Beschleunigungsfaktor von $2\times$ auf einer GPU** nötig ($12\text{ ms} \to 6\text{ ms}$)?

<!--en:557b98b6-->
**Lösungen:**

<!--en:8315a2cc-->
1. **Allein durch `fp8`-Matrixprodukte:** Die Schrittzeit sinkt ($12\text{ ms} \to 3.5 + 5.0 = 8.5\text{ ms}$). Der Beschleunigungsfaktor beträgt **$12 / 8.5 \approx 1.41\times$** (weit unter $2\times$).

<!--en:2c0988c6-->
2. **Allein durch Wegfall aller Kernel außerhalb der Matrixprodukte:** Die Schrittzeit sinkt ($12\text{ ms} \to 7.0\text{ ms}$). Der Beschleunigungsfaktor beträgt **$12 / 7.0 \approx 1.71\times$** (ebenfalls unter $2\times$).

<!--en:ef94fe5f-->
3. **Beides ist zugleich erforderlich:** Von der Schrittzeit von $12\text{ ms}$ sind $58\%$ rechenbegrenzt ($7\text{ ms}$) und $42\%$ bandbreitenbegrenzt ($5\text{ ms}$). Für eine Halbierung auf $6\text{ ms}$ braucht es **sowohl** schnellere `fp8`-Matrixprodukte ($7 \to 3.5\text{ ms}$) **als auch** weniger Speicherverkehr für Aktivierungen durch schmalere Formate oder Fusion ($5 \to 2.5\text{ ms}$).

<!--en:b533abd1-->
::: {.slide title="Zusammenfassung: Schrittweise Optimierung eines Transformers" only="pytorch"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:4490ca8b-->
| Optimierungsstufe | Engpass | Vorhersage nach Amdahl | Gemessener Faktor | Kumulierter Beschleunigungsfaktor / MFU |

<!--en:ae2491a1-->
| **0. Baseline (`fp32`/`tf32`)** | Referenz ($26\text{ ms}$ / $8192$ Token) | – | $1.00\times$ | $1.00\times$ ($\sim 22\%$ MFU) |

<!--en:cc89345a-->
| **1. `torch.compile`** | Speicherverkehr außerhalb der Matrixprodukte und Kernelstart durch den Host | $\le 1.6\times$ | $1.16\times$ | $1.16\times$ ($\sim 25\%$ MFU) |

<!--en:b0f8332c-->
| **2. `bf16` (FlashAttention bereits aktiv)** | GEMMs auf Tensor Cores | $1.5\text{--}2.0\times$ | $1.74\times$ | $\approx 2.0\times$ ($\sim 44\%$ MFU) |

<!--en:e2f24d11-->
| **3. Batchgröße $64 \to 512$** | Feste Kosten von $40\text{ B/param}$ pro Schritt ($0.75\text{ ms}$) | $\approx 1.05\times$ | $1.02\times$ | **$\approx 2.1\times$ ($\sim 45\%$ MFU)** |

<!--en:79ab3dbb-->
| **Variante: Checkpointing** | Aktivierungsspeicher ($8.6 \to 3.2\text{ GiB}$) | $\ge 0.77\times$ | $0.85\times$ | Spart Speicher (nicht beibehalten) |

<!--en:d097122b-->
| **4. Datenparallelität auf 4 GPUs** | Leistungsgrenze einer GPU (Allreduce: $48\text{ ms}$) | $\approx 2.7\times$ (Batchgröße 512) | **$\approx 2.9\times$** gegenüber 1 GPU | **$\approx 6\times$ gegenüber der Baseline auf 1 GPU** |

<!--en:d26f54f7-->
::: {.slide title="Zusammenfassung: Schrittweise Optimierung eines Transformers" only="jax"}

<!--en:c5a222f8-2-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:4490ca8b-2-->
| Optimierungsstufe | Engpass | Vorhersage nach Amdahl | Gemessener Faktor | Kumulierter Beschleunigungsfaktor / MFU |

<!--en:6ff94da4-->
| **0. Baseline (`fp32`/`tf32`)** | Referenz ($24\text{ ms}$ / $8192$ Token) | – | $1.00\times$ | $1.00\times$ ($\sim 23\%$ MFU) |

<!--en:b76bb8c8-->
| **1. `jit_partial`** | Kernelstart durch den Host | $1.02\times$ | $1.01\times$ | $1.01\times$ ($\sim 24\%$ MFU) |

<!--en:df8dda23-->
| **2. `bf16` und fusionierte Attention** | GEMMs auf Tensor Cores und Attention der Größe $T \times T$ | $1.5\text{--}2.0\times$ | $1.45\times$, dann $1.35\times$ | $\approx 2.0\times$ ($\sim 47\%$ MFU) |

<!--en:6c871822-->
| **3. Batchgröße $64 \to 512$** | Feste Kosten von $40\text{ B/param}$ pro Schritt ($0.75\text{ ms}$) | $\approx 1.05\times$ | $1.06\times$ | **$\approx 2.1\times$ ($\sim 50\%$ MFU)** |

<!--en:634cb76f-->
| **Variante: Checkpointing** | Aktivierungsspeicher ($7.4 \to 2.6\text{ GiB}$) | $\ge 0.76\times$ | $0.79\times$ | Spart Speicher (nicht beibehalten) |

<!--en:37d80bd8-->
| **4. Datenparallelität auf 4 GPUs** | Leistungsgrenze einer GPU (Allreduce: $48\text{ ms}$) | $\approx 2.6\times$ (Batchgröße 512) | **$\approx 3.2\times$** gegenüber 1 GPU | **$\approx 7\times$ gegenüber der Baseline auf 1 GPU** |
