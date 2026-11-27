<!--en:c310ce27-->
# Verteiltes Modelltraining

<!--en:31fbbddd-->
In den Abschnitten zum Multi-GPU-Training in :numref:`chap_performance` haben wir ein ResNet und ein GPT auf den vier GPUs eines Rechners trainiert. Jede GPU erhielt dazu eine Kopie des Modells. Wir haben auch berechnet, wie sich durch Aufteilen des Trainingszustands (engl. training state) Platz für ein Modell schaffen lässt, das auf keine einzelne GPU passt. Modelle, die wir für den praktischen Einsatz feinabstimmen, sind oft noch größer. Trainieren wir ein Sprachmodell mit 7 Milliarden Parametern mit Adam, brauchen allein die Gewichte, Gradienten und der Optimiererzustand (engl. optimizer state) etwa 112 GB. Dabei ist noch keine einzige Aktivierung gespeichert. Das übersteigt die 80 GB einer H100. Ein solches Modell auf einer einzigen GPU von Grund auf zu trainieren, würde Jahre dauern.

<!--en:b56d6631-->
Wie verteilen wir Zustand und Rechenarbeit eines Trainingslaufs auf viele Beschleuniger, gegebenenfalls auf mehreren Rechnern, um dasselbe Ergebnis in kürzerer Zeit zu erhalten? Jede Lösung teilt etwas auf: den Batch, den Trainingszustand, die Schichten oder die Sequenz. Jede Aufteilung erfordert zusätzliche Kommunikation. Mit jedem weiteren Arbeitsprozess gibt es eine weitere Stelle, an der der Lauf scheitern kann. Deshalb messen wir zunächst auf einem einzelnen Gerät eine Baseline und wählen dann das einfachste Verfahren, das den beobachteten Engpass beim Speicher oder beim Durchsatz beseitigt. Wir ordnen die Verfahren nach dem Engpass, den sie jeweils beheben, und zeigen, welche Bibliotheken sie mit Stand November 2026 implementieren. Zum Schluss betrachten wir, wie sich ein Trainingslauf über mehrere Wochen in Gang halten lässt.

<!--en:088b5d9f-->
## Formen der Parallelität

<!--en:0ff34f78-->
### Stufen der Skalierung

<!--en:d7bd8d9e-->
Jedes der folgenden Verfahren überwindet eine Grenze des vorherigen, erfordert aber zusätzliche Kommunikation. Deshalb setzen wir sie der Reihe nach ein, als *Stufen der Skalierung* (:numref:`fig_tools_training_ladder`).

<!--en:55fbf3be-->
![In der Praxis führt die Skalierung vom einzelnen Beschleuniger über Replikation und Partitionierung bis zur Kombination mehrerer Formen der Parallelität.](../img/tools-training-ladder.svg)

<!--en:19311187-->
1. **Ein einzelner Beschleuniger** liefert die Referenzwerte: die Verlustkurve, die Anzahl der Token oder Beispiele pro Sekunde, den maximalen Speicherbedarf und einen zuverlässigen Checkpoint.

<!--en:385fa20b-->
1. **Datenparallelität (DDP)** repliziert das Modell auf jeder GPU. Jede GPU verarbeitet einen anderen Teil des Batchs. Nach jedem Rückwärtsdurchlauf werden die Gradienten gemittelt. DDP ist die richtige Wahl, solange Modell, Gradienten und Optimiererzustand auf ein Gerät passen.

<!--en:e2839387-->
1. **Vollständig partitionierte Datenparallelität** (engl. fully sharded data parallelism, FSDP) entspricht Stufe 3 der ZeRO-Familie. Sie verteilt Parameter, Gradienten und Optimiererzustand auf die GPUs. Die Parameter jeder Schicht führt sie nur so lange zusammen, wie die Schicht gebraucht wird (:numref:`subsec_mgp-fsdp`). Der Speicherbedarf für den Zustand pro Gerät sinkt um den Faktor der GPU-Anzahl. Das Kommunikationsvolumen beträgt etwa das 1.5-Fache von DDP.

<!--en:2d06d5b3-->
1. **Tensorparallelität, Pipeline-Parallelität, Kontextparallelität und Expertenparallelität** teilen unterschiedliche Teile des Modells auf: einzelne Schichten, Gruppen aufeinanderfolgender Schichten, die Sequenzdimension bzw. die Experten einer Mixture of Experts (:numref:`sec_moe`). Große Systeme kombinieren mehrere dieser Verfahren, weil keine Achse allein weit genug skaliert.

<!--en:6c259180-->
In PyTorch brauchen DDP und FSDP ein Startprogramm und einen Wrapper:

<!--en:4d947696-->
In `train.py` aktiviert `DistributedDataParallel(model)` die Datenparallelität mit DDP. Für FSDP übernimmt das `fully_shard(model)` aus `torch.distributed.fsdp`. Diese API heißt FSDP2 und speichert jeden partitionierten Parameter als `DTensor`. PyTorch empfiehlt sie gegenüber dem älteren Wrapper `FullyShardedDataParallel`. In JAX legt ein `NamedSharding` für jedes Array die Verteilung auf einem Gerätegitter fest. `jax.jit` kompiliert die kollektiven Operationen, die sich aus dieser Verteilung ergeben (:numref:`sec_multi_gpu_concise`).

<!--en:9ffe9991-->
Bei der Tensorparallelität tauschen die GPUs innerhalb jeder Schicht Aktivierungen aus, bei der Pipeline-Parallelität nur an den Grenzen zwischen den Stufen. Das *Ultra-Scale Playbook* von Hugging Face stützt sich auf mehr als 4000 Läufe mit bis zu 512 GPUs. Dort sank der Durchsatz typischerweise um etwa 43%, wenn die Tensorparallelität von einem Knoten mit 8 GPUs auf zwei Knoten ausgedehnt wurde. Bei der Pipeline-Parallelität betrug der Rückgang unter denselben Bedingungen 14%, wenn jeder Schritt nur wenige Mikrobatches umfasste. Beschränken Sie die Tensorparallelität auf einen Knoten, wo NVLink die Daten überträgt. Nutzen Sie für die kollektiven Operationen mit dem größten Datenvolumen die schnellsten Verbindungen.

<!--en:a82e512b-->
Auf jeder Stufe gilt dieselbe Gleichung. Die Batchgröße, die der Optimierer sieht, beträgt

<!--en:00a1b670-->
also die Batchgröße pro Gerät mal der Anzahl der datenparallelen Ränge mal der Anzahl der Mikrobatches, über die wir pro Optimiererschritt die Gradienten akkumulieren. Ändern wir die Anzahl der GPUs, ohne die Batchgröße pro Gerät oder die Lernrate anzupassen, ändert sich daher das Optimierungsproblem. Für die Daten gelten zwei weitere Regeln: Verteilte Sampler dürfen keine Beispiele doppelt liefern, und bei der Auswertung müssen wir die Metriken mit den richtigen Nennern zusammenführen. Meldet ein Lauf mit mehreren GPUs eine andere Genauigkeit, liegt das meist an einem Verstoß gegen eine dieser Regeln.

<!--en:1b9865b8-->
### Speicherbedarf beim Training

<!--en:567a9a13-->
Bei den meisten Trainingsläufen reicht eine GPU irgendwann nicht mehr aus, weil der Trainingszustand nicht in ihren Speicher passt. Deshalb berechnen wir zuerst das Speicherbudget (:numref:`fig_tools_training_memory`).

<!--en:1d117875-->
![Partitionierung, Checkpointing, Gradientenakkumulation und Auslagerung verringern unterschiedliche Anteile des Speicherbedarfs und verursachen unterschiedliche Kosten.](../img/tools-training-memory.svg)

<!--en:d6d93568-->
Der maximale Speicherbedarf beim Training setzt sich aus Parametern, Gradienten, Optimiererzustand, Aktivierungen, Kommunikationspuffern und dem Zusatzbedarf der Speicherverwaltung zusammen. Trainieren wir ein Modell mit 7 Milliarden Parametern in BF16 mit Adam, belegen die Parameter und die Gradienten jeweils 14 GB. Die beiden FP32-Momente von Adam benötigen 56 GB. Beim Training mit gemischter Genauigkeit (engl. mixed precision) kommen 28 GB für die FP32-Masterkopie der Parameter hinzu. Das ergibt insgesamt 112 GB oder 16 Byte pro Parameter. Unser vereinfachtes Rechenmodell lässt die Masterkopie von 28 GB weg. Übernehmen wir die Zahlenwerte in GB als GiB, bleiben $14 + 14 + 56 = 84\text{ GiB}$ für den Modellzustand. Hinzu kommen 18 GiB für Aktivierungen und 5 GiB für temporäre Puffer (insgesamt $107\text{ GiB}$). Trotz der Vereinfachungen ordnet das Modell die Strategien richtig ein. Für die Berechnung ist keine GPU nötig:

<!--en:c47bf712-->
Ohne weitere Maßnahmen benötigt das Training 107 GiB, mehr als auf eine H100 mit 80 GB passen. DDP mit Aktivierungs-Checkpointing benötigt auf jeder GPU immer noch 95 GiB: Es repliziert die 84 GiB des Zustands, nur der Aktivierungsspeicher schrumpft. FSDP mit Checkpointing auf acht GPUs teilt den Zustand durch acht und benötigt 22 GiB pro Gerät.

<!--en:f52d10b5-->
Jedes der gängigen Verfahren verringert einen anderen Summanden und verursacht andere Kosten. **Gemischte Genauigkeit** senkt die Anzahl der Byte, die ausgewählte Tensoren und Operationen benötigen. **Gradientenakkumulation** verarbeitet pro Schritt einen kleineren Mikrobatch. Der globale Batch aus :eqref:`eq_global_batch` bleibt dabei erhalten. So sinkt der Aktivierungsspeicher, während der Modellzustand gleich groß bleibt. **Aktivierungs-Checkpointing** berechnet ausgewählte Aktivierungen während der Backpropagation erneut. Speicherersparnis und zusätzlicher Rechenaufwand hängen davon ab, welche Aktivierungen gespeichert bleiben. **Auslagerung auf CPU oder NVMe**, etwa mit ZeRO-Offload von DeepSpeed, verschiebt Zustand aus dem Gerätespeicher und erfordert in jedem Schritt Übertragungen zwischen Host und Gerät. **LoRA und QLoRA** trainieren kleine Adapter mit niedrigem Rang anstelle aller Gewichte. Damit entfällt ein Großteil der Gradienten und des Optimiererzustands. QLoRA speichert die eingefrorenen Gewichte zudem mit 4 Bit, sodass sich ein Modell mit 7 Milliarden Parametern auf einer einzigen Karte mit 16 GB feinabstimmen lässt. Kombinieren Sie die Verfahren gezielt: Auch für sich genommen sinnvolle Optimierungen können die Kompilierung, die zeitliche Überlappung von Operationen oder die numerischen Eigenschaften beeinträchtigen.

<!--en:259e3520-->
## Softwarebibliotheken

<!--en:7d406ca9-->
Für das Training auf mehreren Knoten implementiert kaum jemand diese Verfahren von Hand. Bibliotheken übernehmen das, und die Wahl der Bibliothek richtet sich vor allem nach der Größe des Trainingslaufs. Die folgenden Projekte wurden im November 2026 aktiv gepflegt und waren weit verbreitet. Da sich diese Softwarelandschaft schnell verändert, lohnt sich vor der Wahl eines Projekts ein Blick auf seine jüngsten Veröffentlichungen.

<!--en:33f5f6db-->
**In PyTorch integrierte Funktionen.** Neben DDP und FSDP2 bietet PyTorch APIs für Tensorparallelität und Kontextparallelität auf der Grundlage von `DTensor` sowie eine API für Pipeline-Parallelität, `torch.distributed.pipelining`. Die Dokumentation kennzeichnet diese APIs weiterhin als experimentell oder als Alpha-Versionen. Die meisten der folgenden übergeordneten Werkzeuge erzeugen Aufrufe dieser Grundoperationen.

<!--en:fe24493f-->
**Die Werkzeuge von Hugging Face.** `accelerate` startet dasselbe Skript mit DDP, FSDP2 oder DeepSpeed. Für die Auswahl genügt eine Konfigurationsdatei, der Code bleibt gleich. Darauf baut `transformers.Trainer` auf. `peft` implementiert LoRA und QLoRA. `trl` bietet Trainer für überwachtes Fine-Tuning, DPO und Reinforcement Learning (RL) nach dem Vorbild von GRPO. Diese Werkzeuge decken die meisten Fine-Tuning-Aufgaben auf einem Knoten oder einem kleinen Cluster ab.

<!--en:073dadb5-->
**DeepSpeed.** DeepSpeed bietet die ZeRO-Stufen 1–3 sowie die Auslagerung auf CPU und NVMe. Die erste Stufe partitioniert den Optimiererzustand, die zweite zusätzlich die Gradienten und die dritte auch die Parameter. ZeRO führte die Partitionierung ein, die FSDP2 inzwischen direkt unterstützt. FSDP2 kann Zustand auch in den CPU-Speicher auslagern. DeepSpeed bleibt die übliche Wahl, wenn ein Trainingslauf Daten auf NVMe auslagert oder ein anderes Framework auf DeepSpeed aufbaut.

<!--en:3ffc5da7-->
**Megatron Core und TorchTitan.** Beide Bibliotheken dienen dem Pre-Training in großem Maßstab. Megatron Core von NVIDIA implementiert Tensorparallelität, Pipeline-Parallelität, Kontextparallelität und Expertenparallelität mit Training in FP8 und FP4. Megatron-LM ist das zugehörige Referenzsystem für das Training. Das Gegenstück auf PyTorch-Basis ist TorchTitan. Dieses kompakte Referenzsystem kombiniert FSDP2 mit Tensorparallelität, Pipeline-Parallelität, Kontextparallelität und Expertenparallelität. Es enthält unter anderem Modelldefinitionen für Llama 3 405B und DeepSeek-V3. Veröffentlichte Messungen reichen bis zu 512 GPUs. Beide Bibliotheken eignen sich für Modelle, die mehrere Formen der Parallelität zugleich erfordern.

<!--en:55007933-->
**Benutzeroberflächen für das Fine-Tuning.** Unsloth ermöglichte zunächst durch fusionierte Kernel schnelles LoRA- und QLoRA-Training auf einer GPU. Heute unterstützt es auch vollständiges Fine-Tuning, GRPO und mehrere GPUs. Der Geschwindigkeitsgewinn hängt vom Modell, der Sequenzlänge und der Hardware ab. Bei Axolotl steuert eine einzige YAML-Datei das vollständige oder parametereffiziente Fine-Tuning auf einem oder mehreren Knoten. Als Backends dienen FSDP2 und DeepSpeed. LLaMA-Factory bietet einen ähnlichen Funktionsumfang mit einer Weboberfläche. Das PyTorch-Team stellte die aktive Entwicklung von `torchtune` im Jahr 2025 ein.

<!--en:dc2faa5a-->
**JAX.** MaxText ist das Referenzsystem für Pre-Training und Post-Training auf TPUs und GPUs. Marin trainiert mit Levanter und dokumentiert seine Trainingsläufe von den Rohdaten bis zum fertigen Modell, einschließlich fehlgeschlagener Experimente. Marin veröffentlicht Code, Datenpipelines und Protokolle, damit andere die Läufe reproduzieren können.

<!--en:7c97c656-->
**Post-Training mit Reinforcement Learning.** veRL kombiniert das Training mit FSDP, FSDP2 oder Megatron mit der Erzeugung von Rollouts durch vLLM oder SGLang. OpenRLHF baut auf Ray, DeepSpeed und vLLM auf.

<!--en:d0d1298e-->
### Werkzeuge nach Systemgröße

<!--en:9ff3d72f-->
:Trainingswerkzeuge nach Systemgröße (Stand November 2026)

<!--en:a4983b78-->
| Systemgröße | Typische Aufgabe | Geeignete Werkzeuge |

<!--en:c8af2005-->
| 1 GPU, ≥ 8 GB | LoRA-/QLoRA-Fine-Tuning mit ≤ 8 Mrd. Parametern | Unsloth oder PEFT + Trainer |

<!--en:991f291b-->
| 1 Knoten, 2–8 GPUs | Vollständiges Fine-Tuning mit bis zu etwa 30 Mrd. Parametern, DDP/FSDP2 | Accelerate oder Axolotl |

<!--en:0a41747f-->
| Wenige Knoten | Großes Fine-Tuning, kleines Pre-Training | FSDP2 + torchrun, DeepSpeed |

<!--en:ae7e196c-->
| Viele Knoten | Pre-Training in großem Maßstab, MoE | Megatron Core, TorchTitan, MaxText |

<!--en:9bb29cd5-->
| RL-Post-Training | GRPO-/DPO-Pipelines | TRL (klein), veRL/OpenRLHF (groß) |

<!--en:25d603b1-->
Die Modellgrößen ergeben sich aus den 16 Byte pro Parameter beim vollständigen Fine-Tuning: Acht GPUs mit je 80 GB fassen den Zustand von etwa 30 Milliarden Parametern und lassen noch Platz für Aktivierungen. Wählen Sie die erste Tabellenzeile, die den gemessenen Anforderungen genügt. Je weiter unten eine Zeile steht, desto mehr Konfiguration erfordert sie.

<!--en:d6d63535-->
## Fehlertoleranz bei langen Trainingsläufen

<!--en:154ed00e-->
Ein Trainingslauf, der am zehnten Tag abstürzt und sich nicht fortsetzen lässt, hat nichts erbracht. Je mehr Arbeitsprozesse beteiligt sind und je länger ein Lauf dauert, desto alltäglicher werden Hardware-, Netzwerk- und Softwareausfälle. Deshalb müssen wir die Wiederherstellung ebenso sorgfältig entwerfen und testen wie das Training selbst.

<!--en:4e6556dd-->
* **Checkpoints vollständig und atomar schreiben.** Ein Checkpoint für die Wiederaufnahme enthält Modell- und Optimiererzustand, Lernratenplan und Schrittzähler, die Position des Datenladers sowie den Zustand des Zufallszahlengenerators. Schreiben Sie zunächst an einen temporären Pfad und benennen Sie die Datei erst nach Abschluss um. So kann ein Absturz des Trainingsprozesses während des Schreibens den vorherigen Checkpoint nicht zerstören. Bewahren Sie mehrere Checkpoints auf, bis Sie die Wiederherstellung getestet haben. Mit partitionierten Checkpoints (je eine Partition pro Rang) entfällt das Zusammenführen eines Modells, das für den Speicher jedes einzelnen Hosts zu groß ist.

<!--en:f24f9752-->
* **Die Wiederherstellung proben.** Brechen Sie den Lauf absichtlich ab und setzen Sie ihn mit neuen Arbeitsprozessen fort. Prüfen Sie, ob sich Verlust, Schrittzähler und Position im Datenstrom wie ohne Unterbrechung entwickeln. Testen Sie die Wiederherstellung, bevor Sie auf Spot-Instanzen setzen (:numref:`sec_cloud_instances`). Die Ersparnis hängt davon ab, dass sich der Lauf nach jeder Unterbrechung fortsetzen lässt.

<!--en:99bf43c1-->
* **Die Beschleuniger mit genügend Daten versorgen.** Messen Sie den Aufwand für das Laden der Daten getrennt vom Rechenaufwand: Lesezugriffe auf das Speichermedium, Dekodierung und Datenaugmentation, Tokenisierung und Übertragung vom Host zum Gerät. Wartet eine GPU auf Eingaben, bleibt ihr Durchsatz niedrig. Begrenzt die Datenzufuhr das Training, konkurrieren weitere GPUs um dieselben Ressourcen, ohne das Training zu beschleunigen.

<!--en:7ec275cf-->
* **Konvergenz und Geschwindigkeit überwachen.** Eine hohe Tokenrate sagt nichts über den Verlust aus. Protokollieren Sie beides sowie die Zeit für kollektive Operationen und Checkpointing. Reproduzieren Sie Fehler im verteilten Training zuerst mit einem Prozess, dann auf einem Knoten und schließlich auf mehreren. Geben Sie in jeder Protokollzeile Rang und Host an.

<!--en:2ec91de0-->
Mit diesen Maßnahmen lässt sich ein Trainingslauf fortsetzen. Dennoch kosten jeder Checkpoint und jeder Neustart Zeit. In den Übungen berechnen Sie, welches Checkpoint-Intervall sich aus diesen Kosten bei gegebener Ausfallrate ergibt.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:dffc5f25-->
Verteiltes Training teilt den Batch, den Trainingszustand, die Schichten oder die Sequenz auf mehrere Arbeitsprozesse auf. Messen Sie zunächst die Baseline auf einer GPU. Verwenden Sie DDP, solange der Modellzustand in den Speicher passt, und wechseln Sie andernfalls zu FSDP oder ZeRO. Ergänzen Sie Tensorparallelität, Pipeline-Parallelität, Kontextparallelität oder Expertenparallelität erst, wenn die Modellgröße das verlangt. Beschränken Sie die kommunikationsintensive Tensorparallelität auf einen NVLink-Knoten. Der maximale Speicherbedarf setzt sich aus unterschiedlichen Anteilen zusammen. Partitionierung, Aktivierungs-Checkpointing, Gradientenakkumulation, Auslagerung auf den Host und LoRA verringern jeweils einen bestimmten Anteil. Die Kosten für Kommunikation, erneute Berechnungen oder Übertragungszeit lassen sich abschätzen. Im November 2026 sind Unsloth oder PEFT auf einer GPU üblich, Accelerate oder Axolotl mit FSDP2 auf einem Knoten und Megatron Core, TorchTitan oder MaxText für Pre-Training auf mehreren Knoten. Für das Post-Training mit Reinforcement Learning sind TRL und veRL verbreitet. Mehrwöchige Trainingsläufe gelangen zuverlässig zum Abschluss, wenn wir Checkpoints vollständig und atomar schreiben, die Wiederherstellung im Voraus proben, die Datenpipelines mit der Berechnung Schritt halten und wir die Konvergenz zusammen mit dem Durchsatz protokollieren.

<!--en:2294243a-->
## Übungen

<!--en:2c003954-->
1. **Globale Batchgröße.** Ein Pre-Training verwendet Sequenzen mit je 4096 Token und einen globalen Batch von $2^{22} \approx 4.2$ Millionen Token, also 1024 Sequenzen pro Optimiererschritt. Jede GPU fasst höchstens 4 Sequenzen pro Mikrobatch.
    1. Wählen Sie für $N_{\textrm{ranks}} \in \{32, 64, 128, 256\}$ die Größen $B_{\textrm{device}}$ und $N_{\textrm{accumulation}}$ in :eqref:`eq_global_batch` so, dass der globale Batch bei 1024 Sequenzen bleibt und möglichst wenige Akkumulationsschritte nötig sind. Wie viele Ränge können bei reiner Datenparallelität höchstens mit diesem globalen Batch trainieren?
    1. Ein Mikrobatch mit 4 Sequenzen benötige die Zeit $t_{\textrm{mb}}$. Der Allreduce der Gradienten am Ende jedes Optimiererschritts dauere $t_{\textrm{ar}}$ und überlappe sich nicht mit der Berechnung. Geben Sie die Zeit pro Optimiererschritt und die Anzahl der Token pro Sekunde und GPU als Funktionen von $N_{\textrm{ranks}}$ an. Es gelte $t_{\textrm{ar}} = 2t_{\textrm{mb}}$. Um welchen Faktor verkürzt sich die Trainingszeit für eine feste Anzahl von Token, wenn sich die Anzahl der Ränge von 32 auf 64 bzw. von 128 auf 256 verdoppelt? Was begrenzt den Beschleunigungsfaktor mit zunehmender Anzahl der Ränge?
    1. Ein Kollege verlegt das Training von 64 auf 128 GPUs und behält $B_{\textrm{device}}$ und $N_{\textrm{accumulation}}$ bei. Wie ändern sich der globale Batch und die Anzahl der Optimiererschritte für eine feste Anzahl von Token? Für adaptive Verfahren wie Adam skaliert :numref:`sec_batch_size` die Lernrate mit der Quadratwurzel der Batchgröße: $\eta(b) = \sqrt{b/b_0}\,\eta_0$. Dabei ist $\eta_0$ auf die Batchgröße $b_0$ abgestimmt. Welche Lernrate ergibt sich nach dieser Regel, wenn der größere Batch erhalten bleibt? Was geschieht mit einer Aufwärmphase, deren Dauer in Schritten angegeben ist?
   :exlabel:`ex_training-systems-global-batch`

<!--en:8a871c5d-->
1. [code] **Auswertung mit Padding in den Datenpartitionen.** Ein Datensatz mit 10 001 Beispielen wird auf 8 Rängen ausgewertet. Rang $r$ durchläuft `DistributedSampler(dataset, num_replicas=8, rank=r, shuffle=False)`. Da `num_replicas` und `rank` explizit übergeben werden, kann ein einzelner Prozess die Sampler für alle Ränge erzeugen, ohne eine Prozessgruppe zu starten.
    1. Sammeln Sie für `dataset = list(range(10001))` die Indizes aller 8 Sampler. Wie viele Indizes gibt es, welche Beispiele kommen doppelt vor, und um wie viele Prozentpunkte können die Duplikate die gemessene Genauigkeit höchstens verschieben?
    1. Wiederholen Sie den Versuch mit `drop_last=True`. Welche Beispiele werden nie ausgewertet?
    1. Implementieren Sie eine exakte Auswertung, bei der die Ränge die 10 001 Beispiele ohne Padding untereinander aufteilen. Jeder Rang zählt seine richtigen Vorhersagen und seine Beispiele. Anschließend summieren die Ränge die Zählerstände mit `torch.distributed.all_reduce`. Testen Sie die Reduktion in einem einzelnen Prozess: Simulieren Sie die 8 Ränge mit zufälligen Vorhersagen und ersetzen Sie den Allreduce durch eine Summe über die Ränge. Warum liefert der Mittelwert der 8 pro Rang berechneten Genauigkeiten für diese Partitionen kein exaktes Ergebnis?
   :exlabel:`ex_training-systems-padded-evaluation`

<!--en:eb54014a-->
1. [code] **Bitgenaue Wiederaufnahme.** Ein fortgesetzter Lauf ist korrekt, wenn er dieselben Parameter berechnet wie ein ununterbrochener Lauf. Bauen Sie auf der CPU ein kleines Experiment auf: Trainieren Sie ein zweischichtiges MLP mit Dropout auf 1024 zufälligen Beispielen mit je 20 Merkmalen. Verwenden Sie Adam mit einem `LambdaLR`-Lernratenplan für die Aufwärmphase. Ein `DataLoader` mit Batchgröße 32 und `shuffle=True` liefert die Beispiele. Sein eigener `torch.Generator` erhält für das Mischen in jeder Epoche einen Startwert.
    1. Trainieren Sie zunächst 64 Schritte ohne Unterbrechung. Trainieren Sie in einem zweiten Lauf 40 Schritte und speichern Sie einen Checkpoint. Erzeugen Sie Modell, Optimierer, Lernratenplaner und Datenlader mit einem anderen Startwert neu. Stellen Sie den Checkpoint wieder her und trainieren Sie die verbleibenden 24 Schritte. Sorgen Sie dafür, dass die Parameter am Ende beider Läufe bitgenau übereinstimmen. Welches Objekt hält in diesem Experiment welchen der in :numref:`subsec_training_long_run` aufgeführten Zustandsbestandteile? Wie setzt der Datenlader bei der Wiederherstellung mitten in einer Epoche wieder ein?
    1. Lassen Sie bei der Wiederherstellung der Reihe nach jeweils einen Zustandsbestandteil weg. Geben Sie jeweils die größte Parameterabweichung nach den 64 Schritten an. Welche Auslassungen wären in der Verlustkurve sichtbar, welche blieben unbemerkt?
    1. Simulieren Sie einen Absturz während `torch.save`, indem Sie nur die erste Hälfte der Byte eines Checkpoints an seinen endgültigen Pfad schreiben. Laden Sie die Datei. Schreiben Sie dann jeden Checkpoint in eine temporäre Datei im selben Verzeichnis und verschieben Sie diese mit `os.replace` an den endgültigen Pfad. Warum enthält dieser Pfad beim zweiten Verfahren immer einen vollständigen Checkpoint, entweder den vorherigen oder den neuen, unabhängig vom Zeitpunkt des Prozessabsturzes? Bei einem Stromausfall kann die Umbenennung bereits auf dem Datenträger gespeichert sein, bevor der Dateiinhalt gesichert ist. Welche Aufrufe braucht das Verfahren, um auch einen Stromausfall zu überstehen?
   :exlabel:`ex_training-systems-resume-parity`

<!--en:e5fa1b8d-->
1. **Auslagerung des Optimierers.** ● ZeRO-Offload :cite:`Ren.Rajbhandari.Aminabadi.ea.2021` hält für die Vorwärts- und Rückwärtsdurchläufe 16-Bit-Parameter auf der GPU. Die FP32-Masterparameter und die beiden Momente von Adam liegen im Hostspeicher, wo die CPU das Update berechnet. Jeder Schritt überträgt die 16-Bit-Gradienten zum Host und die aktualisierten 16-Bit-Parameter zurück. Die Veröffentlichung verwendet FP16, das ebenso viel Speicher belegt wie BF16. Betrachten Sie ein Modell mit 7 Milliarden Parametern auf einer GPU mit einer Dauerleistung von 400 TFLOP/s. Über PCIe ist sie mit 50 GB/s in jeder Richtung an einen Host angebunden, dessen Speicher 200 GB/s liefert.
    1. Berechnen Sie die Zeit pro Schritt für die beiden nacheinander ausgeführten Übertragungen.
    1. Der Speicherverkehr begrenzt die Geschwindigkeit des CPU-Updates. Zählen Sie, wie viele Byte pro Parameter ein Adam-Update bei dieser Speicheraufteilung liest und schreibt, und schätzen Sie die Dauer des Updates.
    1. Vorwärts- und Rückwärtsdurchlauf kosten zusammen etwa $6N$ FLOPs pro Token bei einem Modell mit $N$ Parametern (:numref:`sec_scaling-laws`). Wie viele Token muss ein Schritt enthalten, damit bei Ausführung ohne Überlappung weniger als 20% der Schrittzeit auf Übertragungen und Update entfallen?
    1. Beim um einen Schritt verzögerten Parameterupdate von ZeRO-Offload berechnet die CPU das Update aus den Gradienten von Schritt $k$. Gleichzeitig berechnet die GPU den Vorwärts- und Rückwärtsdurchlauf von Schritt $k + 1$ mit den Parametern, die aus den Gradienten von Schritt $k - 1$ hervorgingen. Unter welcher Bedingung kostet das Update dann keine zusätzliche Zeit? Was ändert die Verzögerung an der Optimierung?
   :exlabel:`ex_training-systems-optimizer-offload`

<!--en:468cec61-->
1. **Wiederaufnahme mit weniger Arbeitsprozessen.** ● Ein Training mit FSDP auf 64 Rängen verwendet $B_{\textrm{device}} = 4$ Sequenzen und $N_{\textrm{accumulation}} = 4$. Es schreibt partitionierte Checkpoints und muss nach dem Ausfall zweier Knoten mit 48 Rängen fortgesetzt werden.
    1. Ordnen Sie die in :numref:`subsec_training_long_run` aufgeführten Inhalte eines Checkpoints für die Wiederaufnahme zwei Gruppen zu: Welcher Zustand ist auf allen Rängen gleich, welcher unterscheidet sich je nach Rang? Geben Sie für jeden rangspezifischen Bestandteil an, was bei der Wiederherstellung geschehen muss, wenn sich die Anzahl der Ränge ändert.
    1. Zeigen Sie, dass sich der globale Batch mit 1024 Sequenzen auf 48 Rängen durch keine Wahl von $B_{\textrm{device}}$ und $N_{\textrm{accumulation}}$ beibehalten lässt. Vergleichen Sie zwei Lösungsansätze: einen anderen globalen Batch und unterschiedlich große Mikrobatches auf den einzelnen Rängen. Was ändert sich jeweils an der Optimierung? Was verlangt jeder Ansatz für die Normierung des Verlusts?
    1. Entwerfen Sie eine Datenreihenfolge, mit der der fortgesetzte Lauf bei jeder beliebigen Anzahl von Rängen jedes verbleibende Beispiel der Epoche genau einmal sieht. Warum genügt es nicht, für jeden Rang seine Position in der eigenen Partition zu speichern?
   :exlabel:`ex_training-systems-elastic-resume`

<!--en:1f7c36d7-->
1. [extended] **Pipeline-Ablaufpläne.** ●● Eine Pipeline mit $p$ Stufen trainiert mit $m$ Mikrobatches pro Optimiererschritt. Der Vorwärtsdurchlauf eines Mikrobatchs durch eine Stufe dauert $t_f$, sein Rückwärtsdurchlauf dauert $t_b = 2t_f$. Eine Stufe kann den Vorwärtsdurchlauf für Mikrobatch $j$ beginnen, sobald die vorherige Stufe damit fertig ist. Den Rückwärtsdurchlauf für $j$ kann sie beginnen, sobald die nächste Stufe ihren Rückwärtsdurchlauf für $j$ beendet hat. Eine Stufe hält die Aktivierungen von $j$ vom Vorwärtsdurchlauf bis zum Rückwärtsdurchlauf. GPipe :cite:`Huang.Cheng.Bapna.ea.2019` führt alle Vorwärtsdurchläufe vor allen Rückwärtsdurchläufen aus. Der 1F1B-Ablaufplan von :citet:`Narayanan.Shoeybi.Casper.ea.2021` beginnt mit einigen Vorwärtsdurchläufen zum Aufwärmen. Danach wechseln sich je ein Vorwärts- und ein Rückwärtsdurchlauf ab. Der relative Leerlauf eines Ablaufplans ist die Differenz zwischen seiner Schrittzeit und der idealen Zeit $m(t_f + t_b)$, geteilt durch die ideale Zeit.
    1. Schreiben Sie einen Simulator, der die Reihenfolge der Operationen auf jeder Stufe als Eingabe erhält. Er soll jede Operation ausführen, sobald ihre Stufe frei ist und ihre Eingaben bereitstehen. Geben Sie die Schrittzeit aus und für jede Stufe die Höchstzahl der Mikrobatches, deren Aktivierungen sie gleichzeitig hält.
    1. Erzeugen Sie für beliebige $p$ und $m$ die Reihenfolgen für GPipe und 1F1B. Leiten Sie dabei her, wie viele Vorwärtsdurchläufe jede Stufe bei 1F1B zum Aufwärmen benötigt. Simulieren Sie die Abläufe für $p \in \{2, 4, 8\}$ und $m \in \{1, 4, 8, 32\}$. Leiten Sie aus den Ergebnissen den relativen Leerlauf und den Aktivierungsspeicher beider Ablaufpläne als Funktionen von $p$ und $m$ her.
    1. Bei der verschachtelten Zuordnung von :citet:`Narayanan.Shoeybi.Casper.ea.2021` hält jedes Gerät $v$ nicht aufeinanderfolgende Gruppen von Schichten. Ein Mikrobatch durchläuft dann $pv$ Stufen und besucht jedes Gerät $v$-mal, jeweils zu einem Anteil von $1/v$ der Kosten. Simulieren Sie diese Zuordnung für $v \in \{1, 2, 4\}$ und führen Sie dabei zuerst alle Vorwärtsdurchläufe aus. Wie hängen der relative Leerlauf und die Anzahl der Punkt-zu-Punkt-Übertragungen pro Schritt von $v$ ab?
    1. Bei Datenparallelität vom Grad $d$ über mehrere Kopien der Pipeline und Mikrobatches mit je $b$ Sequenzen ergibt der globale Batch mit 1024 Sequenzen aus Übung :exref:`ex_training-systems-global-batch` den Wert $m = 1024/(db)$. Berechnen Sie für $p = 8$ und $b = 1$ den relativen Leerlauf von 1F1B bei $d = 8$, 16 und 64. Was begrenzt bei festem globalem Batch den Grad der Datenparallelität eines Trainingslaufs mit Pipeline?
   :exlabel:`ex_training-systems-pipeline-schedules`

<!--en:27107f33-->
[Diskussion](https://d2l.discourse.group/t/36846)

<!--en:4b9e61de-->
Verteiltes Modelltraining: DDP, FSDP/ZeRO, 3D-Parallelität und Fehlertoleranz<br>**Stufen der Skalierung · DDP und FSDP2 im Vergleich · Tensorparallelität und Pipeline-Parallelität · Invariante der globalen Batchgröße · Partitionierung des Speicherbedarfs · ZeRO-Offload · atomare Checkpoints**

<!--en:4333579a-->
::: {.slide title="Warum das Training auf einem Beschleuniger an Grenzen stößt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:a10504d6-->
- **Speichergrenze bei 7 Mrd. Parametern auf einer GPU ($112\text{ GB}$):** Adam mit gemischter Genauigkeit hält BF16-Gewichte ($14\text{ GB}$), BF16-Gradienten ($14\text{ GB}$), FP32-Mastergewichte ($28\text{ GB}$) und die FP32-Momente $m, v$ von Adam ($56\text{ GB}$). Schon ohne Aktivierungen übersteigt das die $80\text{ GB}$ einer H100.

<!--en:010aba3a-->
- **Vier Achsen der Partitionierung:** Aufteilen lassen sich der Batch (DDP), der Trainingszustand (FSDP/ZeRO), die Matrizen der Schichten bzw. die Stufen (TP/PP) oder die Sequenz bzw. die Experten (CP/EP).

<!--en:12aa0b35-->
- **Faustregel:** Mit einer geprüften 1-GPU-Baseline beginnen und jeweils eine Stufe weitergehen.

<!--en:7b517e9c-->
::: {.slide title="Stufen der Skalierung beim verteilten Training"}

<!--en:30faf367-->
[Schrittweise Skalierung]{.kicker}

<!--en:a69f9ac3-->
Mit jeder Stufe steigen Kommunikationsvolumen und technischer Aufwand. Daher diese Reihenfolge einhalten:

<!--en:bdd325e9-->
1. **Baseline auf einem Beschleuniger:** Die Konvergenz des Verlusts, den Durchsatz in Token/s, den maximalen Speicherbedarf und die Wiederherstellung aus einem Checkpoint prüfen.

<!--en:049fa70e-->
2. **Datenparallelität (DDP):** Das Modell auf jeder GPU replizieren, den Batch aufteilen und die Gradienten einmal pro Schritt mit `all_reduce` mitteln. DDP verwenden, solange der Zustand auf eine GPU passt.

<!--en:021cf049-->
3. **Vollständig partitionierte Datenparallelität (FSDP / ZeRO-3):** Parameter, Gradienten und Optimiererzustände auf $W$ GPUs verteilen. Die Parameter jeder Schicht bei Bedarf mit `all_gather` zusammenführen ($\sim 1.5\times$ so viel Datenverkehr wie bei DDP).

<!--en:e65169f9-->
4. **Kombinierte 3D-/4D-Parallelität (TP / PP / CP / EP):** Einzelne Schichten, Gruppen von Schichten, Sequenzen oder Experten aufteilen, wenn eine einzelne Schicht oder Aktivierung die Speicherkapazität eines Geräts übersteigt.

<!--en:7a5961de-->
::: {.slide title="Datenparallelität (DDP) im Vergleich zu FSDP2 / NamedSharding"}

<!--en:4fe76198-->
[Stufen 2 und 3]{.kicker}

<!--en:2b492f7e-->
- **Stufe 2: DDP (`DistributedDataParallel`):** Repliziert den gesamten Zustand ($16N$) auf jeder GPU und mittelt die gebündelten Gradienten mit `all_reduce` (Datenvolumen $\approx 2S$). Die beste Wahl, solange der Zustand in den Speicher passt.

<!--en:8969e8c6-->
- **Stufe 3: FSDP2 (`fully_shard` / `DTensor` und `NamedSharding` in JAX):** Teilt Parameter, Gradienten und Adam-Momente durch $W$ ($16N/W$). Die Operationen `all_gather` und `reduce_scatter` überlappen sich schichtweise ($\approx 3S = 1.5\times$ so viel Datenverkehr wie bei DDP).

<!--en:feb12591-->
::: {.slide title="Tensorparallelität, Pipeline-Parallelität, Kontextparallelität und Expertenparallelität"}

<!--en:bc8ffa14-->
[Stufe 4: Modell- und Sequenzachsen]{.kicker}

<!--en:7dd7bf96-->
- **Aufteilung innerhalb und zwischen Schichten:** Tensorparallelität (TP) teilt die Matrizen innerhalb jedes Blocks auf (2 kollektive Operationen pro Schicht auf dem kritischen Pfad). Bei der Pipeline-Parallelität (PP) werden Aktivierungen an den Stufengrenzen von Punkt zu Punkt übertragen.

<!--en:5faef273-->
- **Regel für die Hardwaretopologie (*Ultra-Scale Playbook*):** **TP mit ihrem häufigen Datenaustausch auf einen NVLink-Knoten mit 8 GPUs** beschränken ($450\text{--}900\text{ GB/s}$, bei knotenübergreifender TP ändert sich der Durchsatz um $-43\%$). **PP + FSDP/DDP über Knotengrenzen hinweg** skalieren.

<!--en:83d4d72a-->
::: {.slide title="Pipeline-Ablaufpläne: Leerlauf und Speicherbedarf bei GPipe und 1F1B"}

<!--en:b208fad1-->
[Funktionsweise der Pipeline]{.kicker}

<!--en:a9bf4ba1-->
- **GPipe (erst alle $m$ Vorwärts-, dann alle $m$ Rückwärtsdurchläufe):** Kurze Leerlaufzeiten setzen $m \gg p$ voraus. Stufe 1 muss daher **die Aktivierungen aller $m$ Mikrobatches gleichzeitig** halten (Speicherbedarf $O(m)$).

<!--en:1d99dac9-->
- **1F1B-Ablaufplan:** Nach der Aufwärmphase wechseln sich je ein Vorwärts- und ein Rückwärtsdurchlauf ab. Der relative Leerlauf bleibt bei $(p-1)/m$, doch die gleichzeitig gespeicherten Aktivierungen stammen aus **$\le p$ Mikrobatches**.

<!--en:3ce7aa9b-->
::: {.slide title="Die globale Batchgröße als Invariante auf allen Stufen"}

<!--en:b7775888-->
[Auswirkungen auf die Optimierung]{.kicker}

<!--en:d909aaf0-->
- **Dynamik der Optimierung erhalten:** Eine Verdopplung von $N_{\text{ranks}}$ gemäß $64 \to 128$, ohne zugleich $N_{\text{accumulation}}$ zu halbieren, **verdoppelt $B_{\text{global}}$** unbemerkt. Die Anzahl der Optimiererschritte halbiert sich.

<!--en:bcd952de-->
- **Wenn $B_{\text{global}}$ wächst:** Die Lernrate von Adam mit $\sqrt{B/B_0}$ skalieren und die Aufwärmphase in **Token** statt in Schritten angeben.

<!--en:982da7b6-->
::: {.slide title="Verteilte Sampler und exakte Aggregation von Metriken"}

<!--en:ee33801a-->
[Fehlerquellen bei der Auswertung]{.kicker}

<!--en:b2201c61-->
- **Fehlerquelle Padding bei `DistributedSampler`:** Das standardmäßige Padding wiederholt die ersten Beispiele, damit jeder Rang $\lceil N/W \rceil$ Elemente erhält. Das verzerrt die Validierungsmetriken.

<!--en:d74770ad-->
- **Exakte verteilte Auswertung:** Die Beispiele ohne Padding verteilen, lokal `(correct_r, count_r)` mit ganzen Zahlen erfassen und Zähler und Nenner einmal mit `all_reduce(SUM)` summieren.

<!--en:8b290817-->
::: {.slide title="Zusammensetzung des maximalen Speicherbedarfs beim Training"}

<!--en:0531592f-->
[Speicherbudget]{.kicker}

<!--en:5cbb6d02-->
Der maximale Speicherbedarf pro GPU setzt sich aus Modellzustand, gespeicherten Aktivierungen, temporären Kommunikationspuffern und der Fragmentierung durch die Speicherverwaltung zusammen:

<!--en:c26dbe97-->
- **Adam-Zustand bei gemischter Genauigkeit für ein Modell mit 7 Mrd. Parametern ($16\text{ B/param} = 112\text{ GB}$):**
  - BF16-Parameter: $2N = 14\text{ GB}$
  - BF16-Gradienten: $2N = 14\text{ GB}$
  - FP32-Masterparameter: $4N = 28\text{ GB}$
  - Erstes und zweites Moment von Adam in FP32 ($m, v$): $8N = 56\text{ GB}$

<!--en:f825af10-->
- **Variable Anteile:** Der Aktivierungsspeicher wächst linear mit $B_{\text{device}} \times T$. Zwischenpuffer für kollektive Operationen benötigen zusätzlich $4\text{--}8\text{ GiB}$.

<!--en:966b2c09-->
::: {.slide title="Speicherbedarf pro Gerät: DDP und FSDP im Vergleich" layout="tight"}

<!--en:866d757c-->
[Ausführbares Speichermodell]{.kicker}

<!--en:2786e9c1-->
Selbst ohne die Masterkopie der Gewichte mit $28\text{ GB}$ ($84\text{ GiB}$ Zustand $+ 18\text{ GiB}$ Aktivierungen $+ 5\text{ GiB}$ temporäre Puffer) zeigt das Skript, warum 80 GB pro GPU bei DDP für ein vollständiges Fine-Tuning mit 7 Mrd. Parametern nicht ausreichen:

<!--en:5c1dbd2b-->
- **1 GPU ohne weitere Maßnahmen ($107.0\text{ GiB}$):** Der Bedarf übersteigt die $80\text{ GB}$ einer H100.

<!--en:61b01a6c-->
- **DDP auf 8 GPUs mit Aktivierungs-Checkpointing ($95.3\text{ GiB/GPU}$):** Checkpointing senkt den Aktivierungsspeicher von $18\text{ GiB}$ auf $6.3\text{ GiB}$, aber DDP **repliziert den gesamten Modellzustand von $84\text{ GiB}$** auf jeder GPU. Der Speicher reicht weiterhin nicht aus.

<!--en:fb57d3ee-->
- **FSDP auf 8 GPUs mit Aktivierungs-Checkpointing ($21.8\text{ GiB/GPU}$):** Die Partitionierung teilt den Zustand durch $8$ und senkt den Bedarf pro GPU auf $10.5\text{ GiB}$. Für größere Mikrobatches bleiben fast $60\text{ GiB}$ frei.

<!--en:858b6f84-->
::: {.slide title="Speicherbedarf pro Gerät: DDP und FSDP im Vergleich (Code)" layout="tight"}

<!--en:866d757c-2-->
[Ausführbares Speichermodell]{.kicker}

<!--en:ed77d72f-->
::: {.slide title="Fünf unabhängige Verfahren zum Speichersparen"}

<!--en:8e48918b-->
[Kosten der Speicheroptimierung]{.kicker}

<!--en:ba6c0ae5-->
- **Partitionierung und Checkpointing:** FSDP2/ZeRO teilt den Zustand von $16N$ durch $W$ (Kommunikationsvolumen $\sim 1.5\times$). Aktivierungs-Checkpointing benötigt $+25\text{--}33\%$ FLOPs mehr und speichert nur die Eingabe jedes Blocks (weiterhin $O(L)$, aber nur $d$ Werte pro Token und Block).

<!--en:b62044f9-->
- **Akkumulation, Auslagerung und Adapter:** Gradientenakkumulation verkleinert den Mikrobatch $B_{\text{device}}$. CPU-Auslagerung überträgt Mastergewichte und Adam-Zustand über PCIe. LoRA/QLoRA friert die Basisgewichte ein, sodass $>98\%$ des Speicherbedarfs für Gradienten und Optimiererzustand entfallen.

<!--en:1278d0c7-->
::: {.slide title="Zeitbedarf bei Auslagerung des Optimierers auf die CPU (ZeRO-Offload)"}

<!--en:3ab8c8de-->
[Roofline für Host und Gerät]{.kicker}

<!--en:64d40f17-->
- **Kosten der Auslagerung pro Schritt ($1.54\text{ s}$ bei 7 Mrd. Parametern):** Der Versand von $14\text{ GB}$ BF16-Gradienten und der Empfang von $14\text{ GB}$ BF16-Gewichten über PCIe mit $50\text{ GB/s}$ dauern $0.56\text{ s}$. Hinzu kommen $0.98\text{ s}$, um $196\text{ GB}$ ($28\text{ B/param}$) im DDR-Speicher der CPU mit $200\text{ GB/s}$ zu übertragen.

<!--en:0fb8ac44-->
- **Amortisierung und verzögertes Update:** Damit die Auslagerung $< 20\%$ der Schrittzeit bei $400\text{ TFLOP/s}$ beansprucht, sind $\ge 58{,}700\text{ tok/step}$ nötig (oder ein um einen Schritt verzögertes CPU-Update).

<!--en:04ff772f-->
::: {.slide title="Die einfachste Software für die jeweilige Systemgröße (November 2026)"}

<!--en:71e8c4d2-->
[Softwarelandschaft]{.kicker}

<!--en:f3656b79-->
- **Von 1 GPU bis zu 1 Knoten (2–8 GPUs):** **Unsloth** oder **HF PEFT** für LoRA/QLoRA auf einer GPU und **HF Accelerate**, **Axolotl** oder **LLaMA-Factory** für DDP/FSDP2 auf einem Knoten verwenden.

<!--en:fe21b2ca-->
- **Mehrere Knoten und RL-Post-Training:** **PyTorch FSDP2 (`fully_shard`)** oder **DeepSpeed** auf wenigen Knoten, **TorchTitan / Megatron Core / MaxText** auf 64–512 und mehr GPUs sowie **TRL / veRL / OpenRLHF** für DPO/GRPO verwenden.

<!--en:27fd5c3f-->
::: {.slide title="Fehlertoleranz durch vollständiges und atomares Checkpointing"}

<!--en:9ae90677-->
[Wiederanlauf nach Hardwareausfällen]{.kicker}

<!--en:99fcbc1a-->
- **Checkpoint für bitgenaue Wiederherstellung:** Die partitionierten Modellgewichte, Adam-Momente, den **Zustand des Lernratenplaners**, Schritt- und Tokenzähler, die deterministische **Position im Datenstrom** und die **Zustände der Pseudozufallszahlengeneratoren** pro Rang speichern.

<!--en:79d99a8b-->
- **Absturzsicherheit durch atomares Umbenennen:** `ckpt` nie direkt überschreiben. In `ckpt.tmp` schreiben, `f.flush()` und `os.fsync()` aufrufen, die Zieldatei mit **`os.replace("ckpt.tmp", "ckpt")`** ersetzen und `fsync` für das Verzeichnis aufrufen.

<!--en:4e319fc5-->
::: {.slide title="Betrieb langer verteilter Trainingsläufe"}

<!--en:07746822-->
[Produktivbetrieb]{.kicker}

<!--en:34318867-->
Vier Vorgehensweisen entscheiden darüber, ob ein mehrwöchiger Trainingslauf zum Abschluss kommt, stecken bleibt oder unbemerkt divergiert:

<!--en:a5c1c392-->
- **1. Wiederherstellung vor Trainingsbeginn proben:**
  - Den Lauf bei Schritt 40 absichtlich abbrechen und mit neuen Arbeitsprozessen fortsetzen. Für Schritt 41 prüfen, ob Verlust, Gradientennorm und Datenbatch mit denen eines ununterbrochenen Laufs bitgenau übereinstimmen.
  - Die Daten global nach dem Beispielindex ordnen. Dann funktioniert die **Wiederaufnahme mit geänderter Prozesszahl** auch dann, wenn nach dem Ausfall zweier Knoten mit je 8 GPUs nur 48 GPUs verbleiben.

<!--en:ebca2309-->
- **2. Eingabepipelines getrennt von der GPU-Rechenarbeit vermessen:**
  - Leseraten des Speichermediums, Dekodierraten für Bilder und Text sowie Kopierraten vom Host zum Gerät einzeln messen. Bei einem E/A-Engpass verschärfen weitere GPUs nur die Konkurrenz um Speicherzugriffe.

<!--en:d4c6663d-->
- **3. Neben dem Systemdurchsatz auch die Konvergenz überwachen:**
  - `(step, loss, grad_norm, tok/s, collective_ms, ckpt_ms)` mit **Rang und Hostnamen** in jeder Protokollzeile festhalten. Fehler schrittweise eingrenzen: $1\text{ process}$ $\to$ $1\text{ node}$ $\to$ $N\text{ nodes}$.

<!--en:0cbe457f-->
::: {.slide title="Verständnisfragen zu Partitionierung, Batch-Invarianten und Wiederherstellung"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:82bb2da5-->
Drei Szenarien zum verteilten Training:

<!--en:5cf54973-->
1. Beim Pre-Training bleibt der globale Batch bei $B_{\text{global}} = 1{,}024$ Sequenzen (je 4096 Token, insgesamt $\approx 4.2\text{M}$ Token). Jede GPU fasst höchstens $B_{\text{device}} = 4$ Sequenzen pro Mikrobatch. Wie viele Schritte der Gradientenakkumulation $N_{\text{accumulation}}$ sind auf $N_{\text{ranks}} = 64$ GPUs mindestens nötig? Wie viele GPUs können diesen globalen Batch allein mit Datenparallelität höchstens verarbeiten?

<!--en:a990bf3a-->
2. Ein Kollege stellt aus einem Checkpoint Modellgewichte, Adam-Momente und die Position des Datenladers wieder her. Er vergisst den Zustand des Lernratenplaners `LambdaLR`. Was geschieht mit dem Trainingslauf? Löst das eine Ausnahme aus?

<!--en:4d68d2ac-->
3. Warum überträgt FSDP2 pro Schritt etwa $1.5\times$ so viele Daten wie DDP?

<!--en:ea89a84b-->
::: {.slide title="Verständnisfragen zu Partitionierung, Batch-Invarianten und Wiederherstellung (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:9cf4ce3b-->
1. Auf 64 GPUs mit $B_{\text{device}} = 4$ gilt $N_{\text{accumulation}} = 1024 / (64 \times 4) = \mathbf{4}$. Die maximale Anzahl rein datenparalleler GPUs (bei $B_{\text{device}} = 1, N_{\text{accumulation}} = 1$) beträgt **$N_{\text{ranks}} = 1{,}024$ GPUs** (oder $256$ GPUs, wenn $B_{\text{device}} = 4$ erhalten bleibt).

<!--en:228b99e7-->
2. **Es wird keine Ausnahme ausgelöst**, aber die Lernrate springt auf Schritt 0 zurück. Damit beginnt die Aufwärmphase erneut oder die Lernrate springt auf ihren Höchstwert $\eta_{\max}$. **Der Verlust steigt sprunghaft an oder die Genauigkeit sinkt unbemerkt.**

<!--en:f3292fb3-->
3. Bei $N$ Parametern mit einem Gesamtumfang von $S = 2N$ Byte führt DDP einmal `all_reduce` für die Gradienten aus (pro GPU $\approx 2S$ Byte gesendet bzw. empfangen). FSDP führt im Vorwärtsdurchlauf ein `all_gather` für die Gewichte aus ($S$), im Rückwärtsdurchlauf erneut ein `all_gather` für die Gewichte ($S$) und ein `reduce_scatter` für die Gradienten ($S$). Das ergibt **$3S$ Byte ($1.5\times$ so viel wie bei DDP)**.

<!--en:5074c86a-->
::: {.slide title="Zusammenfassung: Trainingssysteme mit mehreren GPUs und verteiltes Training"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:c9292749-->
| Systemaspekt | Grundregel oder Gleichung | Empfehlung für die Praxis |

<!--en:69072e13-->
| **Stufen der Skalierung** | 1 GPU $\to$ DDP $\to$ FSDP2 $\to$ TP + PP + CP/EP | DDP verwenden, solange der Zustand auf eine GPU passt. Danach den Zustand mit FSDP2 partitionieren. TP auf einen NVLink-Knoten begrenzen. |

<!--en:af500e38-->
| **Globale Batchgröße** | $B_{\text{global}} = B_{\text{device}} \cdot N_{\text{ranks}} \cdot N_{\text{accumulation}}$ | Bei jeder Änderung der GPU-Anzahl $N_{\text{accumulation}}$ oder $\eta(B)$ anpassen, damit die Dynamik der Optimierung gleich bleibt. |

<!--en:7e8b7a09-->
| **Speicherbudget** | $M_{\text{state}} = 16N\text{ B}$ (BF16-Gewichte und -Gradienten + FP32-Mastergewichte und Adam-Momente) | FSDP ($\div W$), Aktivierungs-Checkpointing und LoRA/QLoRA gezielt kombinieren, um mit dem verfügbaren GPU-Speicher auszukommen. |

<!--en:02391c7e-->
| **Pipeline-Ablaufplanung** | Relativer Leerlauf $= (p - 1)/m$ | **1F1B** statt GPipe verwenden, um den Speicherbedarf für ausstehende Aktivierungen auf $p$ Mikrobatches zu begrenzen. |

<!--en:85329bb7-->
| **Verteilte Auswertung** | $\text{Acc} = (\sum_r c_r) / (\sum_r n_r)$ | Doppelte Beispiele durch Padding bei der Auswertung vermeiden. Zähler und Nenner getrennt mit `all_reduce` summieren. |

<!--en:781c7364-->
| **Fehlertoleranz** | Atomares Schreiben: `tmp` $\to$ `fsync` $\to$ `os.replace` | Modell, Optimierer, Lernratenplaner, Position im Datenstrom und Zustand des Zufallszahlengenerators speichern. Vor langen Läufen oder Läufen auf Spot-Instanzen die Wiederherstellung proben. |
