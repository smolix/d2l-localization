<!--en:bd41d179-->
# Partitionierung des Trainingszustands

<!--en:4cf89bb8-->
Wie wir in :numref:`sec_memory_precision` gesehen haben, belegt Adam beim Training in fp32 16 Byte pro Parameter. Je vier Byte entfallen auf das Gewicht, seinen Gradienten und die beiden Momentschätzungen von Adam. Bei unserem GPT mit 18 Blöcken und einer Milliarde Parametern belegt dieser feste Trainingszustand bereits 15.1 GiB einer Karte mit 24 GiB, bevor das Modell auch nur ein Token liest. Bei Datenparallelität (:numref:`sec_multi_gpu`, :numref:`sec_multi_gpu_concise`) liegen alle 16 Byte auf jeder GPU. Verdoppeln wir die Tiefe unseres GPT auf 36 Blöcke (1.92 Milliarden Parameter, 28.6 GiB Trainingszustand), können vier GPUs mit je 24 GiB das Modell mit Datenparallelität ebenso wenig trainieren wie eine einzelne GPU.

<!--en:f94066ad-->
Können $k$ GPUs ein Modell trainieren, das auf keine einzelne Karte passt? Ja, denn eine GPU braucht sämtliche Gewichte eines Blocks nur während seines Vorwärts- oder Rückwärtsdurchlaufs. Den vollständigen Gradienten oder den gesamten Optimiererzustand benötigt sie nie. Jede der $k$ GPUs kann daher einen Anteil von $1/k$ der Gewichte, Gradienten und Momente des Optimierers speichern. Unmittelbar vor der Ausführung eines Blocks trägt sie seine Gewichte zusammen. Sobald seine Gradienten vorliegen, summieren die GPUs sie mit Reduce-Scatter und teilen das Ergebnis unter sich auf.

<!--en:8cc4f943-->
Zunächst prüfen wir auf einem Meta-Gerät, das nur die Tensorformen benötigt, ob der Trainingszustand unseres GPT mit 36 Blöcken tatsächlich $16N$ Byte belegt und damit die Speicherkapazität von 24 GiB einer einzelnen RTX 4090 übersteigt:

<!--en:4617892b-->
## Partitionierung von Optimiererzustand, Gradienten und Gewichten

<!--en:b6eaf86a-->
Bei herkömmlicher Datenparallelität speichern alle $k$ GPUs identische Kopien der Momentschätzungen von Adam und führen dieselben Optimierungsschritte aus. Die Verfahrensfamilie *ZeRO* (*Zero Redundancy Optimizer*) :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` beseitigt diese Redundanz durch Partitionierung (engl. sharding) in drei aufeinander aufbauenden Stufen (:numref:`tab_mgp-zero`):

<!--en:c943fe60-->
* **ZeRO-Stufe 1** partitioniert nur den Optimiererzustand. Jede GPU speichert die beiden Momente von Adam für einen Anteil von $1/k$ der Parameter ($8 + 8/k$ Byte pro Parameter). An die Stelle eines Allreduce ($=\text{reduce-scatter} + \text{all-gather}$, :numref:`subsec_mg-ring`) tritt zunächst ein Reduce-Scatter der Gradienten. Jede GPU erhält so den summierten Gradienten für ihren Anteil von $1/k$ und aktualisiert ihre eigenen Parameter. Anschließend tragen die GPUs die aktualisierten Gewichte mit All-Gather zusammen. Insgesamt übertragen sie genau so viele Byte wie bei Datenparallelität.

<!--en:2371aa8f-->
* **ZeRO-Stufe 2** partitioniert zusätzlich die Gradienten ($4 + 12/k$ Byte pro Parameter). Sobald sich im Rückwärtsdurchlauf ein Gradientenpuffer gefüllt hat, führen die GPUs darauf ein Reduce-Scatter aus. Jede GPU gibt den Anteil von $1 - 1/k$ der Gradienten frei, für den andere GPUs zuständig sind. Auch hier fallen gegenüber Datenparallelität keine zusätzlichen Übertragungen an.

<!--en:146dff2a-->
* **ZeRO-Stufe 3** oder *Fully Sharded Data Parallelism* (FSDP) :cite:`Zhao.Gu.Varma.ea.2023` partitioniert auch die Gewichte ($16/k$ Byte pro Parameter). Bevor ein Block im Vorwärts- oder Rückwärtsdurchlauf an die Reihe kommt, trägt ein All-Gather alle seine Gewichte zusammen. Unmittelbar nach der Berechnung gibt jede GPU die fremden Gewichtspartitionen frei. Ein Reduce-Scatter verteilt die summierten Gradienten des Blocks (:numref:`fig_fsdp_lifecycle`).

<!--en:d07e7ede-->
:Trainingszustand pro GPU am Ende des Rückwärtsdurchlaufs mit Adam in fp32 auf $k$ GPUs sowie für das GPT mit 36 Blöcken und 1.92 Milliarden Parametern bei $k = 2$.

<!--en:f160e345-->
| Inhalt jeder GPU | Gewichte | Gradienten | Adam-Zustand | Byte pro Parameter | GPT, $k = 2$ |

<!--en:4978c5e7-->
| Datenparallelität | alle | alle | vollständig | 16 | 28.6 GiB |

<!--en:6fb3025f-->
| ZeRO-Stufe 1 | alle | alle | $1/k$ | $8 + 8/k$ | 21.4 GiB |

<!--en:654ca3e8-->
| ZeRO-Stufe 2 | alle | $1/k$ | $1/k$ | $4 + 12/k$ | 17.9 GiB |

<!--en:6891dea4-->
| ZeRO-Stufe 3 (FSDP) | $1/k$ | $1/k$ | $1/k$ | $16/k$ | 14.3 GiB |

<!--en:b6fcf1f3-->
![Gewichte eines Blocks auf einer GPU im Verlauf eines FSDP-Schritts. Zwischen zwei Berechnungen ist der Block partitioniert. Jede GPU hält einen Anteil von $1/k$. Die anderen Blöcke bleiben währenddessen partitioniert. Ein All-Gather setzt den ganzen Block für den Vorwärtsdurchlauf zusammen. Anschließend wird er freigegeben. Ein zweites All-Gather setzt ihn für den Rückwärtsdurchlauf zusammen. Danach wird er erneut freigegeben, und durch ein Reduce-Scatter erhält jede GPU ihren Anteil an den Gradienten.](../img/mdl-perf-fsdp-lifecycle.svg)

<!--en:da477b88-->
FSDP benötigt pro Schritt zwei All-Gather-Operationen für den Vorwärts- und Rückwärtsdurchlauf und ein Reduce-Scatter mit jeweils $S = 4N$ Byte. Das sind drei kollektive Phasen gegenüber zwei bei einem Ring-Allreduce. FSDP überträgt damit **$1.5\times$ so viele Daten wie bei Datenparallelität**. :numref:`tab_collectives` fasst die vier wichtigsten kollektiven Operationen beim parallelen Training zusammen. :numref:`fig_collectives` zeigt, welche Daten sie jeweils zwischen zwei GPUs übertragen.

<!--en:51637ac0-->
![Datenaustausch zwischen zwei GPUs bei den vier kollektiven Operationen, jeweils vorher und nachher. GPU 0 hat anfangs die Blöcke $a_0, a_1$ und GPU 1 die Blöcke $b_0, b_1$. Bei All-Gather hat GPU $g$ anfangs nur ihre eigene Partition, also Block $g$. Blau kennzeichnet die ursprünglichen Blöcke, Grün die Summen.](../img/mdl-perf-collectives.svg)

<!--en:48924402-->
:Kollektive Operationen beim datenparallelen Training und beim Training mit partitioniertem Zustand.

<!--en:cf05ac75-->
| Kollektive Operation | Ergebnis auf jeder GPU | Einsatz |

<!--en:e0ced2ce-->
| Allreduce | die vollständige Summe | Gradienten bei Datenparallelität |

<!--en:3aa03358-->
| Reduce-Scatter | ein Anteil von $1/k$ der Summe | Gradienten bei ZeRO und FSDP |

<!--en:3fbf4ab3-->
| All-Gather | alle Partitionen, aneinandergereiht | Gewichte bei ZeRO und FSDP |

<!--en:d1e3cb62-->
| All-to-All | je ein anderer Slice von jeder GPU | Token-Routing bei Mixture of Experts (:numref:`sec_moe`) |

<!--en:8c105a55-->
## Speicherbedarf und Kommunikation

<!--en:4706d640-->
Schätzen wir den maximalen Speicherbedarf und die Schrittdauer unseres GPT mit 36 Blöcken bei FSDP auf $k = 2$ GPUs. Wir verwenden pro GPU 2 Sequenzen mit je 1024 Token und Aktivierungs-Checkpointing für jeden Block (:numref:`subsec_mp-checkpointing`).

<!--en:3b593ca0-->
* **Speicherbedarf:** Der partitionierte Trainingszustand belegt $16N/2 = 14.3$ GiB pro GPU. Mit Checkpointing benötigen die Aktivierungen etwa 1 MiB pro Token (etwa 2 GiB für 2048 Token). Hinzu kommen ein oder zwei Blöcke, deren Gewichte gerade vollständig vorliegen (jeweils etwa 0.2 GB). Damit liegt der Spitzenbedarf bei höchstens **rund 17 GiB** pro GPU und somit deutlich unter 24 GiB.

<!--en:fa979f07-->
* **Kommunikation:** In jeder der drei kollektiven Phasen sendet jede GPU $(k-1)S/k$ Byte. Für die gesamte Kommunikationszeit ergibt sich

  $$
  t_\textrm{comm} = \frac{3(k-1)}{k}\,S\,\beta_\textrm{link}.
  $$
  :eqlabel:`eq_fsdp_comm`

  Entscheidend für die Skalierung von FSDP ist folgender Zusammenhang: **Bei Verteilung auf mehr GPUs sinkt der Speicherbedarf für den Zustand pro GPU um den Faktor $k$. Die Datenmenge, die jede GPU pro Schritt sendet, steigt hingegen von $1.5S$ (bei $k = 2$) auf bis zu $3S$ (für große $k$).** Nur wenn jede GPU mehr Token verarbeitet oder die Verbindung schneller ist, sinkt der Zeitanteil der Kommunikation an einem Schritt.

<!--en:c38baa04-->
Zunächst messen wir $\beta_\textrm{link}$ für ein All-Gather von 256 MiB auf $k = 2$ GPUs:

<!--en:e71ed503-->
Mit dem Messwert für $\beta_\textrm{link}$ und der tf32-Spitzenleistung der RTX 4090 von 82.6 TFLOP/s (`peak_flops`) berechnen wir unsere theoretischen Vorhersagen für den Speicherbedarf des Zustands pro GPU ($16N/k$), die Kommunikationszeit ($3(k-1)S\beta_\textrm{link}/k$) und die Rechenzeit mit Checkpointing ($8NbT/P$):

<!--en:d8e4559b-->
Bei einer Busbandbreite von rund 2.2 GB/s über den Hauptspeicher des Hosts dauert die Übertragung von $1.5S = 11.5$ GB pro Schritt etwa 5 s. Die Berechnungen für die 2048 Token dauern hingegen nur etwa 0.4 s. Da 0.4 s Rechenzeit die 5 s Kommunikation nicht verdecken können, erwarten wir eine Schrittdauer von rund 5 s.

<!--en:ccdcccea-->
## Training mit partitioniertem Zustand auf zwei GPUs

<!--en:33e0eb2b-->
Nun führen wir einen Trainingsschritt für unser partitioniertes GPT mit 36 Blöcken auf zwei GPUs aus. Wir bauen das Modell so auf, dass keine GPU jemals den gesamten Zustand von 28.6 GiB auf einmal speichern muss.

<!--en:104ab75c-->
In PyTorch erstellen wir das Modell auf dem Gerät `'meta'`. Auf jeden Block mit Aktivierungs-Checkpointing und auf das übergeordnete Modell wenden wir `fully_shard` aus `torch.distributed.fsdp` an. Mit `model.to_empty(device=device)` legen wir nur die Partitionen des jeweiligen Rangs im Speicher an und initialisieren sie direkt dort.

<!--en:982e1a82-->
Wir starten `TRAIN_FSDP` in $k$ Prozessen mit `d2l.torchrun`. Wir erfassen den maximal belegten Speicher (`peak_bytes`) sowie die Schrittdauern (`times`) und die NCCL-Kommunikationszeit (`comm`) von Rang 0:

<!--en:b010f490-->
In JAX partitioniert `fsdp_sharding` jeden Parametertensor entlang der ersten Achse, deren Länge durch $k$ teilbar ist (`P('data')`). Die Funktion `create` wendet diese Vorgabe an. So initialisiert jede GPU nur ihre eigenen Partitionen von `model` und `opt`:

<!--en:f2a1ce6c-->
Für das Aktivierungs-Checkpointing wenden wir `nnx.remat` auf jeden Block an. In `loss_fn` geben wir vor, dass die Ausgabeaktivierungen jedes Blocks entlang der Batchachse partitioniert bleiben (`data`). So erzeugt XLA die All-Gather- und Reduce-Scatter-Operationen für FSDP automatisch. Die Puffer von `model` und `opt` geben wir in `train_step` zur Wiederverwendung frei, damit der aktualisierte Zustand die alten Partitionen direkt überschreibt:

<!--en:abe6c0d1-->
Vor der Ausführung kompilieren wir `train_step` mit einem entlang der Batchachse partitionierten Tokenbatch `X` der Form $(kb, T)$. Dann fragen wir mit `compiled.memory_analysis()` ab, wie viel Speicher der zur Wiederverwendung freigegebene Zustand und die temporären Puffer pro GPU belegen:

<!--en:e7939c2f-->
Um zu prüfen, ob XLA die erwarteten kollektiven Operationen für FSDP erzeugt hat, analysiert `collective_megabytes` den HLO-Text des kompilierten Programms. Für jede kollektive Operation summiert die Funktion die Ergebnisgrößen in Byte pro GPU:

<!--en:273274b6-->
Die Auswertung der kollektiven Operationen in `compiled` bestätigt die Vorhersage. `all-gather` (zweimal pro Schritt) und `reduce-scatter` (einmal pro Schritt) übertragen jeweils die berechneten Datenmengen:

<!--en:166f61d5-->
Nach einem Aufwärmdurchlauf von `train_step` messen wir die Dauer von drei Wiederholungen. Anhand eines Laufzeitprofils erfassen wir die NCCL-Kommunikationszeit (`comm`) und den maximal belegten Gerätespeicher (`peak_bytes`):

<!--en:1148a2a5-->
Zum Schluss geben wir den maximal belegten Speicher pro GPU, den Median der Schrittdauer und die Laufzeit der NCCL-Kommunikationskernel auf GPU 0 aus:

<!--en:a083b8f3-->
Auf beiden GPUs belegt das Modell mit 36 Blöcken in der Spitze rund 16 GiB pro Karte. Damit bleibt es unter unserer Obergrenze von 17 GiB. Ein Schritt dauert etwa 5.5 s und weicht damit um weniger als ein Sechstel von unserer theoretischen Vorhersage von 5 s ab. GPU 0 verbringt über 90% der Schrittdauer in NCCL-Kommunikationskerneln. Über NVLink 5 (900 GB/s pro Richtung) dauert die Übertragung derselben 11.5 GB nur rund 13 ms, weniger als ein Dreißigstel der Rechenzeit von 0.4 s.

<!--en:57aba994-->
Der Vergleich zeigt auch, *wann* sich die Partitionierung lohnt. Bei unserem GPT mit 18 Blöcken passt der Trainingszustand zusammen mit einem ausreichend großen Mikrobatch auf eine GPU. Ein solches Modell trainiert mit Datenparallelität und Gradientenakkumulation (engl. gradient accumulation) wesentlich schneller als mit FSDP über PCIe. Im ersten Fall kommunizieren die GPUs nur einmal pro globalem Schritt. FSDP hingegen trägt die Gewichte bei jedem Mikrobatch zweimal erneut zusammen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:d634531c-->
Bei Datenparallelität liegen alle $16N$ Byte des Adam-Zustands in fp32 auf jeder GPU. ZeRO partitioniert diesen Zustand über $k$ GPUs. Stufe 1 partitioniert die Momente des Optimierers ($8 + 8/k$ Byte pro Parameter), Stufe 2 zusätzlich die Gradienten ($4 + 12/k$ Byte pro Parameter). Beide übertragen pro Schritt genau dieselben $2(k-1)S/k$ Byte wie ein Ring-Allreduce. Stufe 3 (FSDP) partitioniert auch die Gewichte ($16/k$ Byte pro Parameter). Für den Vorwärts- und den Rückwärtsdurchlauf trägt FSDP die Gewichte jedes Blocks zusammen und verteilt die summierten Gradienten mit Reduce-Scatter. Dabei überträgt es $1.5\times$ so viele Daten wie bei Datenparallelität ($3(k-1)S/k$ Byte pro GPU). Dank der Partitionierung können zwei GPUs mit je 24 GiB ein GPT mit 36 Blöcken trainieren, dessen Zustand von 28.6 GiB auf keine der beiden Karten allein passt.

<!--en:2294243a-->
## Übungen

<!--en:93e21771-->
1. **Kosten der Partitionierung.** ● Geben Sie das Kommunikationsvolumen pro GPU und Schritt in Einheiten von $S = 4N$ an. Das ist die Anzahl der Byte für einen vollständigen Satz von Gewichten oder Gradienten in fp32.
    1. Eine Variante der ZeRO-Stufe 3 hält jeden zusammengesetzten Block vom Vorwärtsdurchlauf bis zum Ende seines Rückwärtsdurchlaufs im Speicher. Welche kollektive Operation entfällt dadurch? Wie viel der in :numref:`tab_mgp-zero` angegebenen Speicherersparnis geht verloren?
    1. FSDP mit gemischter Genauigkeit verwendet bf16 für das All-Gather der Gewichte und das Reduce-Scatter der Gradienten. Die lokalen Partitionen der Referenzgewichte und des Adam-Zustands bleiben auf jeder GPU in fp32. Vergleichen Sie die pro Schritt übertragene Datenmenge mit der bei Datenparallelität in fp32 und den Speicherbedarf pro Parameter mit dem Wert in der letzten Zeile von :numref:`tab_mgp-zero`.
    1. Betrachten Sie das GPT mit 36 Blöcken auf $k = 2$ GPUs beim gemessenen Wert von $\beta_\textrm{link}$. Wie viele Token pro GPU müsste ein Schritt verarbeiten, damit die Rechenzeit mit Checkpointing ($8N$ FLOPs pro Token bei 83 TFLOP/s) der Kommunikationszeit aus :eqref:`eq_fsdp_comm` entspricht? Passen die zugehörigen Aktivierungen auf die Karte? Warum hilft Gradientenakkumulation nicht, wenn FSDP die Gewichte für jeden Mikrobatch erneut zusammenträgt?

<!--en:1f9dc34e-->
2. [extended] **Partitionierter Optimiererzustand.** ● Implementieren Sie ZeRO-Stufe 1 :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` selbst. Gehen Sie von `TRAIN_DDP` (:numref:`sec_multi_gpu_concise`) aus und verwenden Sie `torch.optim.Adam`.
    1. Verteilen Sie die Parametertensoren so auf die Ränge, dass jeder Rang ungefähr den Anteil $1/k$ der Parameter hält und Adam nur auf seine eigenen Tensoren anwendet. Lassen Sie den zuständigen Rang jeden aktualisierten Tensor mit `dist.broadcast` an alle Ränge senden. Prüfen Sie nach mehreren Schritten, ob die Parameter mit denen des vollständigen DDP-Trainings mit Adam übereinstimmen, und vergleichen Sie mit `torch.distributed.optim.ZeroRedundancyOptimizer`.
    1. Messen Sie bei $k = 2$ und $k = 4$ den belegten GPU-Speicher und seinen Spitzenwert mit und ohne Partitionierung. Vergleichen Sie die insgesamt übertragene Datenmenge beim DDP-Allreduce mit anschließendem Versand durch die zuständigen Ränge mit der bei Reduce-Scatter und All-Gather in Stufe 1.

<!--en:d13a5945-->
3. [code] **Tensorparallelität und Datenparallelität.** ● Betrachten Sie den MLP-Block $\mathbf{Y} = \phi(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$ mit $\mathbf{X} \in \mathbb{R}^{n \times d}$, $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$ und $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$ auf $k$ Geräten. Bei Tensorparallelität (:numref:`fig_splitting`, :cite:`shoeybi2019megatron`) wird $\mathbf{W}_1$ spaltenweise und $\mathbf{W}_2$ zeilenweise aufgeteilt.
    1. Zeigen Sie, dass jedes Gerät ohne Kommunikation eine Teilausgabe der Form $n \times d$ berechnen kann und dass die Summe der $k$ Teilausgaben $\mathbf{Y}$ ergibt. Welche kollektive Operation benötigt der Vorwärtsdurchlauf und welche der Rückwärtsdurchlauf für $\nabla_{\mathbf{X}}$?
    1. Vergleichen Sie, wie viele Werte jedes Gerät pro Schritt bei Tensorparallelität und bei Datenparallelität sendet. Bestimmen Sie die Tokenanzahl $n$, unterhalb derer Tensorparallelität weniger Daten überträgt. Warum beschränkt sich Tensorparallelität trotzdem auf schnelle NVLink-Verbindungen innerhalb eines Knotens?
    1. Verwenden Sie `parallelize_module` mit `ColwiseParallel()` und `RowwiseParallel()` auf einem eindimensionalen `DeviceMesh`. Zählen Sie mit `CommDebugMode` die kollektiven Operationen im Vorwärts- und Rückwärtsdurchlauf.
   :exlabel:`ex_mgp-tensor`

<!--en:8f3c0a63-->
2. [extended] **Partitionierter Optimiererzustand.** ● Implementieren Sie ZeRO-Stufe 1 :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` deklarativ. Gehen Sie von `train_step` (:numref:`sec_multi_gpu_concise`) aus und verwenden Sie `optax.adam`. Halten Sie die Modellparameter auf allen Geräten repliziert (`P()`) und verteilen Sie die Blätter von `nnx.state(opt)` mit `fsdp_sharding`.
    1. Stellen Sie sicher, dass die Parameter am Ende von `train_step` weiterhin repliziert sind. Verwenden Sie bei Bedarf `jax.sharding.reshard`. Prüfen Sie über mehrere Schritte, ob die Parameterupdates mit denen bei repliziertem Adam-Zustand übereinstimmen.
    1. Untersuchen Sie den kompilierten Schritt mit `collective_megabytes`. Vergleichen Sie seine kollektiven Operationen sowie den laufenden und maximalen Speicherbedarf bei $k = 2$ und $k = 4$ mit :numref:`tab_mgp-zero`.

<!--en:7d3b57ef-->
3. [code] **Tensorparallelität und Datenparallelität.** ● Betrachten Sie den MLP-Block $\mathbf{Y} = \phi(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$ mit $\mathbf{X} \in \mathbb{R}^{n \times d}$, $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$ und $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$ auf $k$ Geräten. Bei Tensorparallelität (:numref:`fig_splitting`, :cite:`shoeybi2019megatron`) wird $\mathbf{W}_1$ spaltenweise (`P(None, 'data')`) und $\mathbf{W}_2$ zeilenweise (`P('data', None)`) aufgeteilt.
    1. Zeigen Sie, dass jedes Gerät ohne Kommunikation eine Teilausgabe der Form $n \times d$ berechnen kann und dass die Summe der $k$ Teilausgaben $\mathbf{Y}$ ergibt. Welche kollektive Operation benötigt der Vorwärtsdurchlauf und welche der Rückwärtsdurchlauf für $\nabla_{\mathbf{X}}$?
    1. Vergleichen Sie, wie viele Werte jedes Gerät pro Schritt bei Tensorparallelität und bei Datenparallelität sendet. Bestimmen Sie die Tokenanzahl $n$, unterhalb derer Tensorparallelität weniger Daten überträgt. Warum beschränkt sich Tensorparallelität trotzdem auf schnelle NVLink-Verbindungen innerhalb eines Knotens?
    1. Platzieren Sie $\mathbf{W}_1$ mit `P(None, 'data')`, $\mathbf{W}_2$ mit `P('data', None)` und $\mathbf{X}$ repliziert auf einem `Auto`-Mesh. Kompilieren Sie den Vorwärts- und Rückwärtsdurchlauf per JIT und untersuchen Sie die erzeugten kollektiven Operationen.
   :exlabel:`ex_mgp-tensor`

<!--en:f98d8942-->
4. **Vorausladen des nächsten Blocks.** Im Vorwärtsdurchlauf mit FSDP bezeichne $g$ die Dauer des All-Gather für einen Block und $c$ seine Rechenzeit. Ohne Überlappung benötigen $L$ Blöcke die Zeit $L(g + c)$. Beim Vorausladen beginnt das All-Gather für Block $l + 1$, während Block $l$ rechnet.
    1. Zeigen Sie, dass das Vorausladen die Dauer des Vorwärtsdurchlaufs auf $g + (L-1)\max(g, c) + c$ verkürzt. Erklären Sie, warum kein Ablaufplan die Zeit $L\max(g, c)$ unterschreiten kann.
    1. Berechnen Sie $g$ und $c$ für einen Block des GPT mit 36 Blöcken ($50$ Millionen Parameter, 2048 Token pro GPU, 83 TFLOP/s), einmal für unsere PCIe-Verbindung und einmal für NVLink 5 (900 GB/s). Geben Sie für beide Verbindungen an, wie viel Zeit das Vorausladen spart.

<!--en:3ee51ae5-->
[Diskussion](https://d2l.discourse.group/t/36886)

<!--en:214c5972-->
ZeRO und Fully Sharded Data Parallelism (FSDP)<br>**Speicherengpass durch 16N Byte Zustand · ZeRO-Stufen 1–3 · All-Gather und Reduce-Scatter · Gewichte im FSDP-Schritt · Kommunikation und Speicherbedarf**

<!--en:39dbafcc-->
::: {.slide title="Wenn allein der Parameterzustand mit 16N Byte den GPU-Speicher übersteigt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:974ac1de-->
Die Verdopplung der Tiefe des GPT mit einer Milliarde Parametern von $18$ auf $36$ Blöcke (Breite $d = 2048$) ergibt ein Modell mit $1.92\text{B}$ Parametern. Sein Adam-Zustand in `float32` ($16N$ Byte) passt nicht mehr auf eine mit $24\text{ GiB}$ ausgestattete RTX 4090:

<!--en:842c633b-->
- **Warum Datenparallelität hier nicht hilft:** Herkömmliches DDP repliziert alle $16\text{ B/param}$ ($28.6\text{ GiB}$) auf jeder GPU. Auch vier Karten mit je $24\text{ GiB}$ haben daher ebenso wie eine einzelne Karte zu wenig Speicher, bevor sie auch nur ein Token verarbeiten.

<!--en:05f1f61d-->
::: {.slide title="Warum Kopien des Zustands auf k GPUs redundant sind"}

<!--en:5e1073f6-->
[Grundgedanke]{.kicker}

<!--en:9cd5c2c2-->
- **Gewichte ($\mathbf{W}_\ell$, $4\text{ B/param}$):** Nur nötig, während Block $\ell$ seinen Vorwärts- oder Rückwärtsdurchlauf ausführt.

<!--en:e8f295e3-->
- **Gradienten ($\nabla\mathbf{W}_\ell$, $4\text{ B/param}$):** Nur vom Rückwärtsdurchlauf des Blocks $\ell$ bis zum Optimierungsschritt nötig.

<!--en:534d7df8-->
- **Adam-Momente ($\mathbf{m}, \mathbf{v}$, $8\text{ B/param}$):** Auf sie greift Adam **nur bei der elementweisen Aktualisierung** am Ende des Schritts zu. Jede der $k$ GPUs muss nur einen Anteil von $1/k$ der Parameter speichern und aktualisieren.

<!--en:5d82e95a-->
::: {.slide title="Speicherbedarf pro GPU in den drei ZeRO-Stufen"}

<!--en:8c793f18-->
[ZeRO im Überblick]{.kicker}

<!--en:fc8e69f6-->
- **ZeRO-1 ($8 + 8/k\text{ B/param}$):** Verteilt die Adam-Momente $\mathbf{m}, \mathbf{v}$ auf $k$ GPUs ($21.4\text{ GiB}$ bei $k=2$ für unser GPT mit $1.92\text{B}$ Parametern).

<!--en:2f13cd68-->
- **ZeRO-2 ($4 + 12/k\text{ B/param}$):** Partitioniert Momente und Gradienten ($17.9\text{ GiB}$ bei $k=2$). Das Kommunikationsvolumen beträgt wie bei DDP $2\,\frac{k-1}{k}S$.

<!--en:cf1ff427-->
- **ZeRO-3 / FSDP ($16/k\text{ B/param}$):** Partitioniert Gewichte, Gradienten und Momente ($14.3\text{ GiB}$ bei $k=2$, damit bleiben $\sim 10\text{ GiB}$ für Aktivierungen).

<!--en:40665bcf-->
::: {.slide title="Kollektive Operationen beim Training mit partitioniertem Zustand"}

<!--en:67dff45d-->
[Kollektive Operationen]{.kicker}

<!--en:66919a60-->
- **Ring-Allreduce = Reduce-Scatter + All-Gather:** Die Summenbildung mit einem Datenvolumen von $\frac{2(k-1)}{k}S$ zerfällt in zwei kollektive Phasen mit je $\frac{k-1}{k}S$ Byte.

<!--en:cdd8936a-->
- **Reduce-Scatter ($\text{Full } S \to \text{Summed } S/k$):** Summiert die Gradienten aller GPUs und übergibt Partition $j$ an die zuständige GPU $j$.

<!--en:0cf1cef3-->
- **All-Gather ($\text{Shard } S/k \to \text{Full } S$):** Setzt auf allen GPUs die vollständigen Gewichte zusammen, unmittelbar vor der Berechnung einer Schicht oder nach dem partitionierten Optimierungsschritt.

<!--en:218963e7-->
::: {.slide title="ZeRO-Stufe 1: Partitionierte Momente ohne zusätzliche Kommunikation"}

<!--en:f2456114-->
[ZeRO-Stufe 1]{.kicker}

<!--en:ae23fc50-->
- **Adam zwischen den beiden Hälften des Allreduce:** (1) Reduce-Scatter der Gradienten ($\frac{k-1}{k}S$), (2) den eigenen Anteil von $1/k$ der Gewichte mit dem eigenen Anteil von $1/k$ der Adam-Momente aktualisieren, (3) All-Gather der aktualisierten Gewichte ($\frac{k-1}{k}S$).

<!--en:cea32378-->
- **Kein zusätzliches Byte übertragen:** Das Datenvolumen pro GPU beträgt insgesamt $\frac{2(k-1)}{k}S$, genau wie bei herkömmlichem DDP. Der Speicherbedarf für den Adam-Zustand sinkt um den Faktor $k\times$.

<!--en:f2093eb9-->
::: {.slide title="ZeRO-Stufe 2: Fortlaufendes Reduce-Scatter der Gradienten"}

<!--en:8d7acada-->
[ZeRO-Stufe 2]{.kicker}

<!--en:6c2247a7-->
Warum alle $4N$ Byte der Gradienten bis zum Ende des Rückwärtsdurchlaufs speichern, wenn jede GPU nur einen Anteil von $1/k$ der Parameter aktualisiert? **ZeRO-Stufe 2** gibt die Gradienten bereits während des Rückwärtsdurchlaufs frei:

<!--en:f90afe4c-->
- **Reduce-Scatter je Gradientenpuffer während `backward()`:** Sobald die Backpropagation die Gradienten für einen Block oder Puffer berechnet hat, starten die GPUs für diesen Puffer ein **Reduce-Scatter**.

<!--en:9c3b5f14-->
- **Sofortige Freigabe fremder Partitionen:** Jede GPU behält nur ihren Anteil von $1/k$ am summierten Gradienten. Den übrigen Anteil von $(k-1)/k$ gibt sie sofort frei, bevor der nächste Block im Rückwärtsdurchlauf an die Reihe kommt.

<!--en:27e2a04b-->
- **Gleiches Kommunikationsvolumen wie bei DDP:** Im gesamten Rückwärtsdurchlauf und beim anschließenden All-Gather der aktualisierten Gewichte überträgt jede GPU insgesamt $\frac{2(k-1)}{k}S$ Byte. Dabei belegt der vollständige Gradiententensor mit seinen $4N$ Byte nie auf einmal Speicher.

<!--en:f3e5680d-->
::: {.slide title="ZeRO-Stufe 3 (FSDP): Gewichte blockweise zusammentragen und freigeben"}

<!--en:939754cb-->
[Fully Sharded Data Parallel]{.kicker}

<!--en:8f34fb2e-->
- **Partitionierung der Gewichte (gesamter Zustand $16N/k$):** Zwischen zwei Berechnungen speichert jede GPU von den Gewichten jedes Blocks nur den Anteil $1/k$.

<!--en:ecf1f68b-->
- **Ablauf pro Block:** Bevor Block $\ell$ im **Vorwärtsdurchlauf** an die Reihe kommt, setzt ein **All-Gather** seine Gewichte in einem temporären Puffer zusammen. Sobald Block $\ell$ fertig ist, wird der Puffer freigegeben. Im **Rückwärtsdurchlauf** setzt ein zweites **All-Gather** die Gewichte von Block $\ell$ erneut zusammen. Es folgen ein **Reduce-Scatter** für die Gradienten von Block $\ell$ und die sofortige Freigabe der Gewichte.

<!--en:f3069360-->
::: {.slide title="Kommunikationsaufwand von FSDP: 1.5× so viele Daten wie bei Datenparallelität"}

<!--en:cd44412f-->
[Kommunikationsmodell]{.kicker}

<!--en:84b3ef16-->
- **$1.5\times$ so viele Daten wie bei DDP:** Da FSDP die zusammengetragenen Gewichte nach dem Vorwärtsdurchlauf freigibt, benötigt es im Rückwärtsdurchlauf ein zusätzliches All-Gather ($3$ kollektive Phasen statt $2$).

<!--en:bd6d5aef-->
- **Skalierungsgesetz für wachsendes $k$:** Mit mehr GPUs sinkt der dauerhafte Speicherbedarf pro GPU ($16N/k \downarrow$). Das Kommunikationsvolumen pro GPU steigt von $1.5\,S$ ($k=2$) auf bis zu $3S$ ($k \to \infty$).

<!--en:d3dd03aa-->
::: {.slide title="Bandbreitenmessung für All-Gather auf zwei RTX 4090"}

<!--en:37794ebb-->
[Kalibrierung der Verbindung]{.kicker}

<!--en:60816398-->
Vor dem Lauf des GPT mit 36 Blöcken auf $k = 2$ GPUs liefert ein Bandbreitentest den Wert von $\beta_{\textrm{link}}$ für ein All-Gather von $256\text{ MiB}$ auf zwei RTX 4090:

<!--en:af7e39d8-->
- **Gleiche Übertragungsrate über den Host-Speicher:** Jede GPU sendet einen Anteil von $(k-1)/k = 1/2$ der $256\text{ MiB}$ großen Ausgabe ($128\text{ MiB}$). Die Busbandbreite von $\sim 2.2\text{--}2.3\text{ GB/s}$ ($\beta_{\textrm{link}} \approx 0.43\text{--}0.45\text{ ns/B}$) entspricht der Übertragungsrate des isolierten Allreduce aus §{sec_multi_gpu_concise}.

<!--en:378ce7de-->
::: {.slide title="Speicherbedarf, Kommunikations- und Rechenzeit für das GPT mit 1.9 Mrd. Parametern"}

<!--en:649eb15a-->
[Vorhersage für einen Schritt]{.kicker}

<!--en:d0078a9c-->
Gesucht sind der maximale Speicherbedarf, die Kommunikationszeit und die Rechenzeit eines FSDP-Schritts für unser GPT mit 36 Blöcken und $1.92\text{B}$ Parametern ($S = 4N = 7.7\text{ GB}$). Der Schritt verwendet $k = 2$ GPUs mit je $b = 2$ Sequenzen zu $T = 1{,}024$ Token und Aktivierungs-Checkpointing pro Block:

<!--en:ffae7134-->
- **Vorhersage für den Speicherbedarf ($\le 17\text{ GiB} < 24\text{ GiB}$):** Partitionierter Zustand $16N/2 = 14.3\text{ GiB}$ + gespeicherte Aktivierungen ($\sim 1\text{ MiB/token} \times 2{,}048 \approx 2\text{ GiB}$) + 1–2 gerade zusammengesetzte Blöcke ($\sim 0.4\text{ GiB}$) $\approx 16.7\text{ GiB}$.

<!--en:061ae86e-->
- **Vorhersage für die Schrittdauer ($\sim 5\text{ s}$ Kommunikation gegenüber $0.4\text{ s}$ Rechenzeit):** Bei $k = 2$ überträgt jede GPU $1.5\,S = 11.5\text{ GB}$ über die PCIe-Verbindung mit $2.2\text{ GB/s}$ ($\sim 5.0\text{ s}$). Gegenüber dieser Übertragungszeit fallen die $0.4\text{ s}$ für die Berechnung in `tf32` kaum ins Gewicht ($8N$ FLOPs/Token mit Checkpointing).

<!--en:297a373a-->
::: {.slide title="Initialisierung auf dem Meta-Gerät und fully_shard in PyTorch" only="pytorch"}

<!--en:98bce545-->
[FSDP in PyTorch]{.kicker}

<!--en:43f4b79d-->
Damit beim Start keine einzelne GPU den gesamten Zustand von $28.6\text{ GiB}$ speichern muss, entsteht das Modell auf dem Gerät `'meta'`, das keinen Speicher für die Daten belegt. Jeder Block wird mit `fully_shard` partitioniert. Mit `to_empty` entstehen nur die lokalen Partitionen im Speicher:

<!--en:9a4fa79a-->
- **Warum `fully_shard` jeden Block einzeln kapselt:** So trägt ein All-Gather jeweils nur die Gewichte eines Blocks mit $50\text{M}$ Parametern ($\sim 200\text{ MB}$) zusammen. Das gesamte Modell umfasst $1.92\text{B}$ Parameter.

<!--en:a01b2fe4-->
::: {.slide title="Speichermangel auf dem Host beim FSDP-Start mit mehreren Prozessen vermeiden" only="pytorch"}

<!--en:081c51a0-->
[Initialisierung]{.kicker}

<!--en:e6288fc8-->
Startet `torchrun` auf einem Host $k$ Arbeitsprozesse, kann eine unbedachte Modellinitialisierung den Hauptspeicher des Hosts oder den GPU-Speicher erschöpfen, noch bevor FSDP startet:

<!--en:d6b2012a-->
- **Fehler 1: Speicher auf der GPU vor `fully_shard` belegen:** Der Aufruf von `d2l.GPT(**config).to(device)` vor `fully_shard` belegt $4N = 7.7\text{ GB}$ auf der Karte. Bei größeren Modellen ($7\text{B+}$) bricht CUDA sofort wegen Speichermangels ab.

<!--en:25bbfbd2-->
- **Fehler 2: $k$ Kopien im Hauptspeicher des Hosts:** Erstellen und initialisieren alle $k = 8$ Prozesse gleichzeitig ein Modell von $30\text{ GB}$ im Hauptspeicher, belegen sie insgesamt $8 \times 30 = 240\text{ GB}$ im DRAM des Hosts.

<!--en:9474ff4b-->
- **Vorgehen im Produktivbetrieb:** Modell auf `torch.device('meta')` erstellen $\to$ jeden Block mit `fully_shard` partitionieren $\to$ lokalen GPU-Speicher für den Anteil $1/k$ mit `model.to_empty(device=device)` reservieren $\to$ die partitionierten Tensoren vom Typ `DTensor` direkt initialisieren oder Checkpoints direkt in diese Tensoren einlesen.

<!--en:bdf836d1-->
::: {.slide title="Deklarative Partitionierung von Parametern und Aktivierungen in JAX" only="jax"}

<!--en:4307d10f-->
[FSDP in JAX]{.kicker}

<!--en:9256a6b5-->
In JAX kommt FSDP ohne manuell programmierte All-Gather- oder Reduce-Scatter-Operationen aus. Dazu jeden Parametertensor entlang der ersten Achse partitionieren, deren Länge durch $k$ teilbar ist (`fsdp_sharding`). Für die Ausgabeaktivierungen jedes Blocks die Partitionierung entlang der Batchachse beibehalten (`P('data')`):

<!--en:d7362e20-->
- Die Blockgewichte sind entlang `'data'` partitioniert, die Blockaktivierungen *ebenfalls* entlang `'data'`. Daher erzeugt XLA vor der Matrixmultiplikation automatisch ein All-Gather für die Gewichte jedes Blocks und im Rückwärtsdurchlauf ein Reduce-Scatter für seine Gewichtsgradienten.

<!--en:b42d90d7-->
- **Datenvolumen wie vorhergesagt:** Die HLO-Ergebnisgrößen (beim All-Gather $2 \times$ so groß wie die gesendete Datenmenge für $k=2$) betragen beim All-Gather $2S \approx 15.0\text{ GB}$ (gesendet: $S = 7.5\text{ GB}$) und beim Reduce-Scatter $S/2 \approx 3.8\text{ GB}$ (gesendet: $3.8\text{ GB}$). Insgesamt werden $11.3\text{ GB} \approx 1.5\,S$ gesendet.

<!--en:c803ae9c-->
::: {.slide title="Speicherplanung und Pufferwiederverwendung pro GPU in XLA" only="jax"}

<!--en:6be3765f-->
[Speicherplan des Compilers]{.kicker}

<!--en:4b51524b-->
Eine Abfrage mit `compiled.memory_analysis()` zeigt schon vor dem ersten Schritt, dass der Speicherbedarf nach dem statischen Speicherplan von XLA deutlich unter $24\text{ GiB}$ pro GPU liegt:

<!--en:30c55f2f-->
- **Puffer zur Wiederverwendung freigeben (`donate_argnums=(0, 1)`):** Die bisherigen Puffer für die partitionierten Gewichte und Adam-Momente ($10.7\text{ GiB} = 12N/2$) dürfen mit dem aktualisierten Zustand überschrieben werden. Müssten der alte und der neue Zustand während des Optimierungsschritts gleichzeitig im Speicher liegen, würden sie $2 \times 10.7 = 21.4\text{ GiB}$ belegen. Der Speicher würde nicht ausreichen.

<!--en:f7c44630-->
::: {.slide title="Gemessener FSDP-Schritt: 16 GiB Spitzenbedarf, 90% Kommunikation"}

<!--en:dbb2c465-->
[Experimentelle Überprüfung]{.kicker}

<!--en:c62bb76c-->
Ein Trainingsschritt des GPT mit 36 Blöcken und $1.92\text{B}$ Parametern auf zwei mit je $24\text{ GiB}$ ausgestatteten RTX 4090 bestätigt die Vorhersagen für Speicherbedarf und Laufzeit:

<!--en:02b7140a-->
- **Speicher reicht aus:** Jede GPU belegt in der Spitze **$\sim 16\text{ GiB}$** und bleibt damit unter der Obergrenze von $17\text{ GiB}$. Gemeinsam trainieren sie ein Modell mit einem Zustand von $28.6\text{ GiB}$, der auf keine Karte allein passt.

<!--en:8f6ea900-->
- **Kommunikation dominiert bei PCIe:** Jeder Schritt dauert **$\sim 5.5\text{ s}$**. Auf GPU 0 entfallen über **$90\%$ der Schrittdauer ($\sim 5.0\text{ s}$) auf NCCL-Kommunikationskernel** und nur $\sim 0.4\text{ s}$ auf Berechnungen.

<!--en:b31c925b-->
::: {.slide title="FSDP über PCIe mit Umweg über den Host und über NVLink 5"}

<!--en:1c7e2089-->
[Einfluss der Verbindung]{.kicker}

<!--en:92c65c19-->
- **PCIe 4.0 über den Host-Speicher ($\sim 2.2\text{ GB/s}$):** Die Übertragung von $1.5\,S = 11.5\text{ GB}$ dauert $\sim 5.0\text{ s}$, die Berechnung nur $0.40\text{ s}$. Die Verbindung begrenzt die Schrittdauer (Anteil der Rechenzeit: $\sim 7\%$).

<!--en:580b942d-->
- **NVLink 4/5 ($450\text{--}900\text{ GB/s}$):** Die Übertragung derselben $11.5\text{ GB}$ dauert $13\text{--}26\text{ ms}$. Das ist kürzer als die Rechenzeit für den gesamten Schritt auf diesen GPUs ($14\text{--}32\text{ ms}$ bei maximaler `bf16`-Rechenleistung von B200/H100).

<!--en:5f61af62-->
- **Blöcke vorausladen:** Läuft das All-Gather für Block $\ell+1$ parallel zur Berechnung von Block $\ell$, lässt sich die Kommunikationszeit bei $g \le c$ vollständig verdecken.

<!--en:bd7a3f83-->
::: {.slide title="Wann partitionieren, wann replizieren und Gradienten akkumulieren?"}

<!--en:69e9d074-->
[Entscheidungshilfe]{.kicker}

<!--en:4c286512-->
- **Wenn $16N$ Byte in den Speicher passen (z. B. GPT mit 1 Mrd. Parametern auf $24\text{ GiB}$):** **DDP mit Gradientenakkumulation** verwenden. Die GPUs kommunizieren nur **einmal pro globalem Schritt**, nachdem alle $m$ Mikrobatches verarbeitet sind.

<!--en:9c457b05-->
- **Warum FSDP mit Gradientenakkumulation über PCIe langsamer ist:** FSDP gibt die Gewichte nach jedem Block frei. Dadurch muss es **für jeden einzelnen Mikrobatch zweimal** alle Gewichte zusammentragen ($2m\times$ zusätzlicher PCIe-Datenverkehr).

<!--en:a9e82535-->
- **Reihenfolge:** DDP $\to$ ZeRO-1/2 $\to$ ZeRO-3 (FSDP). Erst dann zur nächsten Stufe wechseln, wenn der Speicherbedarf es erfordert.

<!--en:82d02841-->
::: {.slide title="Verständnisfragen: ZeRO-Speicherbedarf, FSDP-Datenvolumen und Verbindungen"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:dedd14db-->
Drei Fälle der Partitionierung durchrechnen ($S = 4N$: Größe einer vollständigen Kopie der Gewichte oder Gradienten in `float32`, gemessen in Byte):

<!--en:f15dc987-->
1. $N = 4 \times 10^9$ Parameter ($16N = 64\text{ GB}$ Adam-Zustand in `float32`) auf $k = 4$ GPUs: Wie viele GB Zustand hält jede GPU nach dem Rückwärtsdurchlauf bei **DDP**, **ZeRO-1**, **ZeRO-2** und **ZeRO-3 (FSDP)**?

<!--en:f7419112-->
2. FSDP mit gemischter Genauigkeit behält seine Partitionen in `float32`. Für das All-Gather der Gewichte und das Reduce-Scatter der Gradienten verwendet es `bfloat16` ($2\text{ B}$). Wie viele Byte überträgt jede GPU pro Schritt im Vergleich zu DDP in `float32`?

<!--en:a2522324-->
3. FSDP läuft zunächst auf $k = 2$ GPUs, dann auf $k = 64$ GPUs. Um welchen Faktor ändern sich jeweils der Speicherbedarf des Zustands pro GPU und das Kommunikationsvolumen pro GPU und Schritt?

<!--en:557b98b6-->
**Lösungen:**

<!--en:3da1bef2-->
1. **DDP ($16N$):** $\mathbf{64\text{ GB}}$, **ZeRO-1 ($(8 + 8/k)N = 10N$):** $\mathbf{40\text{ GB}}$, **ZeRO-2 ($(4 + 12/k)N = 7N$):** $\mathbf{28\text{ GB}}$, **ZeRO-3 / FSDP ($16N/k = 4N$):** $\mathbf{16\text{ GB}}$ pro GPU.

<!--en:c582273f-->
2. **Weniger Datenverkehr als bei DDP in `float32`:** Der Wechsel auf `bfloat16` halbiert die Bytezahl pro Wert bei den kollektiven Operationen. Für $S_{\textrm{comm}}$ gilt nun $4N \to 2N = S/2$. Die 3 FSDP-Phasen übertragen daher $\frac{3(k-1)}{k} \frac{S}{2} = \mathbf{1.5\,\frac{k-1}{k}S}$ Byte. Das sind $25\%$ *weniger* Byte als bei DDP in `float32` mit $\frac{2(k-1)}{k}S$ Byte.

<!--en:89b4cc07-->
3. **Der Speicherbedarf sinkt um den Faktor $32\times$** ($16N/2 \to 16N/64$), während **das Kommunikationsvolumen pro GPU fast um den Faktor $2\times$ steigt** (von $3(1/2)S = 1.5\,S$ bei $k=2$ auf $3(63/64)S \approx 2.95\,S$ bei $k=64$).

<!--en:9a53155d-->
::: {.slide title="Zusammenfassung: ZeRO-Stufen 1–3, Speicherbedarf und Kommunikation bei FSDP"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:d851a9dd-->
| Strategie | Auf $k$ GPUs verteilt | Speicher pro Parameter (Adam in `fp32`) | Kollektive Operationen pro Schritt | Gesendete Byte pro GPU ($S = 4N$) |

<!--en:2e6299c4-->
| **Herkömmliches DDP** | nur Minibatch | $16\text{ B}$ | 1 Allreduce | $\frac{2(k-1)}{k}S < 2S$ |

<!--en:7123e5aa-->
| **ZeRO-Stufe 1** | Momente des Optimierers ($\mathbf{m}, \mathbf{v}$) | $8 + \frac{8}{k}\text{ B}$ | 1 Reduce-Scatter + 1 All-Gather | $\frac{2(k-1)}{k}S < 2S$ |

<!--en:62977595-->
| **ZeRO-Stufe 2** | Momente + Gradienten ($\nabla \mathbf{W}$) | $4 + \frac{12}{k}\text{ B}$ | Reduce-Scatter je Gradientenpuffer + 1 All-Gather | $\frac{2(k-1)}{k}S < 2S$ |

<!--en:4432a1a5-->
| **ZeRO-Stufe 3 (FSDP)** | Momente + Gradienten + Gewichte ($\mathbf{W}$) | **$\frac{16}{k}\text{ B}$** | 2 All-Gather + 1 Reduce-Scatter | **$\frac{3(k-1)}{k}S < 3S$** |

<!--en:7014f3c6-->
| **FSDP mit Kommunikation in `bf16`** | gesamter Zustand (Referenzpartitionen in `fp32`) | $\frac{16}{k}\text{ B}$ (+ aktueller Block) | 2 All-Gather in `bf16` + 1 Reduce-Scatter in `bf16` | $\frac{1.5(k-1)}{k}S < 1.5S$ |
