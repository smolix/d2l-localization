<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:cfd5025b-->
In diesem Kapitel haben wir jede Idee an einem konkreten Beispiel untersucht. Wir haben eine Roofline mit einer einzigen Speicherbandbreite betrachtet, eine GPU mit ihren Verbindungen, einen Compiler pro Framework und drei Möglichkeiten, auf einer einzelnen Karte Speicher zu sparen. Hinzu kamen ein kollektiver Algorithmus, ein Verfahren zur Aufteilung des Modellzustands auf mehrere GPUs und ein Transformer, den wir auf einem einzelnen Rechner beschleunigt haben. Zwei frei zugängliche Bücher ergänzen das gesamte Kapitel. [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (Austin et al., 2025) leitet Rooflines und Kosten kollektiver Operationen für TPUs und GPUs her und entwickelt eine Notation für die Partitionierung (engl. sharding). [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Hugging Face, 2025) gibt anhand von Messungen einen Überblick über Speicherbudgets, Datenparallelität, ZeRO und Parallelität über mehrere Rechner hinweg. Eine kurze Einführung in die drei Fälle aus :numref:`sec_perf_model` gibt der Essay von Horace He :cite:`He.2022`.

<!--en:300df661-->
## Messung und Modellierung

<!--en:f256a612-->
### Genauere Leistungsmodelle

<!--en:896b73c9-->
Die Roofline aus :numref:`sec_perf_model` :cite:`Williams.Waterman.Patterson.2009` kann die Leistung sowohl unterschätzen als auch überschätzen. Manche Kernel überschreiten ihre Bandbreitengrenze (engl. bandwidth roof). Passen die Tensoren in den L2-Cache mit 72 MB oder in den gemeinsamen Speicher, stehen ihre Daten schneller zur Verfügung, als die Bandbreite $\beta$ erwarten lässt. Eine Roofline, die den Cache berücksichtigt :cite:`Ilic.Pratas.Sousa.2014`, zeigt für jede Ebene der Speicherhierarchie eine eigene Bandbreitengrenze. :citet:`Yang.Wang.Farrell.ea.2020` lesen für Deep-Learning-Kernel aus Hardwarezählern ab, wie viele Bytes jede Ebene überträgt. Andere Kernel bleiben unter der Grenze, weil sie zu wenige Bytes gleichzeitig übertragen (Gesetz von Little, :numref:`subsec_hw-gpu`). :citet:`Volkov.2016` zeigt, dass unabhängige Befehle innerhalb eines Threads zusätzliche Warps ersetzen können. Frühere GPU-Leistungsmodelle wenden das Gesetz von Little falsch an und liegen deshalb bei der Vorhersage des Durchsatzes um bis zu den Faktor 1.7 daneben.

<!--en:12f50225-->
### Zeitmessung auf dem Gerät und systemweite Profiler

<!--en:977e410a-->
`d2l.Benchmark` (:numref:`sec_perf_model`) wartet mit dem Ablesen der Uhr, bis die Warteschlange leer ist. Das würde eine laufende Trainingsschleife verlangsamen. CUDA-Ereignisse (`torch.cuda.Event`) erfassen Zeitstempel im GPU-Stream, ohne die Warteschlange zu leeren. Das gemessene Intervall schließt allerdings Zeiten ein, in denen die GPU auf den Host wartet. Der Profiler aus :numref:`subsec_perf-profiler` unterscheidet für einen Prozess zwischen Rechenzeit und Leerlaufzeit und exportiert Ablaufprotokolle für Perfetto. Nsight Systems von NVIDIA zeigt für alle Prozesse die Threads des Betriebssystems, Speicherkopien und NCCL-Aufrufe auf einer gemeinsamen Zeitachse. Nsight Compute misst die Leistung eines einzelnen Kernels und vergleicht sie mit beiden Grenzen der Roofline.

<!--en:e288fbfc-->
### Die Eingabepipeline

<!--en:e06cf234-->
Wer mit Bild- oder Audiodaten trainiert, stößt oft zuerst auf einen Engpass beim Datenlader (engl. data loader), noch bevor sich der Zusatzaufwand durch Python bemerkbar macht. Die GPU wartet, während die CPU den nächsten Minibatch dekodiert und die Daten augmentiert. Hat der Host weniger freie Kerne, als die GPU benötigt (:numref:`subsec_hw-cpu`), lässt sich die Dekodierung auf die GPU verlagern (z. B. mit NVIDIA DALI). :citet:`Mohan.Phanishayee.Raniwala.ea.2021` untersuchen, wie lange verschiedene Trainingsläufe auf Daten warten. Sie vermeiden einen Großteil der Wartezeit durch einen auf das Training abgestimmten Cache und nutzen die Vorverarbeitung für mehrere Hyperparametereinstellungen gemeinsam.

<!--en:62ae0657-->
### Auslastung und Zeit bis zum angestrebten Verlust

<!--en:f311bb68-->
Wie am Ende der Fallstudie (:numref:`subsec_ft-account`) schlüsselten :citet:`Ivanov.Dryden.Ben-Nun.ea.2021` die Laufzeit für BERT nach Kerneln auf. Sie fusionierten die bandbreitenbegrenzten Kernel zwischen den Matrixprodukten und beschleunigten so eine Encoder-Schicht um den Faktor $1.3\times$. Die Spalte `MFU` der Fallstudie zeigt die Kennzahl, die für große Trainingsläufe üblicherweise angegeben wird :cite:`chowdhery2022palm`. Die *Hardware-FLOPs-Auslastung* (engl. hardware FLOPs utilization, HFU) zählt auch erneut ausgeführte FLOPs. Beim Aktivierungs-Checkpointing übersteigt sie daher die MFU :cite:`Korthikanti.Casper.Lym.ea.2022`. Nur die MFU erlaubt einen fairen Systemvergleich, weil sie ausschließlich die vom Modell benötigten Rechenoperationen zählt.

<!--en:7ebe969a-->
Der Durchsatz in Token pro Sekunde bewertet jede Änderung positiv, die einen Schritt verkürzt, selbst wenn das Training dadurch mehr Schritte bis zur Konvergenz benötigt. Der modded-nanoGPT-Speedrun :cite:`Jordan.2024` bewertet Änderungen deshalb nach der tatsächlich verstrichenen Zeit, bis ein GPT beim Training auf acht GPUs einen vorgegebenen Validierungsverlust erreicht. Er kombiniert kompilierte Kernel und Zahlenformate geringer Genauigkeit mit dem Muon-Optimierer (:numref:`sec_muon`).

<!--en:a2dc8fea-->
## Hardware und Zahlenformate

<!--en:a8b82c6f-->
### Weitere Beschleuniger

<!--en:7cc73e82-->
Die Roofline-Analyse gilt unverändert auch für Beschleuniger, die über die Auswahl in :numref:`tab_gpu_specs` hinausgehen. Der Instinct MI355X von AMD, der TPU7x von Google, der Gaudi 3 von Intel und der Trainium2 von AWS kombinieren gestapelten HBM mit Recheneinheiten für Matrixoperationen. Ihre bf16-Knickpunkte reichen von etwa 230 FLOP/Byte (Trainium2) bis über 450 (Gaudi 3). Der Faktor gegenüber unserer RTX 4090 beträgt damit $1.4\times$ bis $3\times$, ausgehend von deren 164 FLOP/Byte. Damit gewinnen Operatorfusion und schmale Zahlenformate noch an Bedeutung.

<!--en:11d5e0c1-->
### Gemeinsamer Speicher von CPU und GPU

<!--en:261372a5-->
Bei der Dekodierung einer einzelnen Sequenz ist die Geschwindigkeit höchstens so groß wie die Speicherbandbreite geteilt durch die Anzahl der Bytes für die Gewichte (:numref:`subsec_hw-two-workloads`), sofern die Gewichte in den Speicher passen. Beim M3 Ultra von Apple teilen sich CPU und GPU bis zu 512 GB Speicher mit einer Bandbreite von 819 GB/s. Ein Modell mit 70 Milliarden Parametern in 8 Bit (rund 70 GB) passt nicht auf eine Karte mit 24 GB. Beim M3 Ultra erlaubt die Bandbreitengrenze dagegen etwa ein Dutzend Token pro Sekunde. Solche Rechner mit gemeinsamem Speicher bieten eine hohe Speicherkapazität bei geringerer maximaler Rechenleistung. Sie eignen sich daher eher für die Inferenz mit einer einzelnen Sequenz als für das Training.

<!--en:12a312f2-->
### Zahlenformate mit blockweiser Skalierung im Detail

<!--en:9ca89418-->
Die blockweisen Skalierungsfaktoren aus :numref:`subsec_hw-gpu` benötigen nur wenige zusätzliche Bits. Bei den OCP-Formaten mit Mikroskalierung :cite:`Rouhani.Zhao.More.ea.2023` teilen sich 32 Werte einen Skalierungsfaktor von 8 Bit, der eine Zweierpotenz darstellt. Das kostet zusätzlich 0.25 Bit pro Wert. Bei NVFP4 von NVIDIA teilen sich 16 Werte einen fp8-Skalierungsfaktor (0.5 Bit pro Wert). Hinzu kommt ein fp32-Skalierungsfaktor pro Tensor. Damit lässt sich der größte Wert jedes Blocks nahe bei 6 abbilden, der Obergrenze des fp4-Wertebereichs. Bei einer Zweierpotenz als Skalierungsfaktor ist das nur bis auf einen Faktor zwei möglich. In beiden Fällen benötigt ein Gewicht oder eine Aktivierung mit 4 Bit insgesamt höchstens 4.5 Bit.

<!--en:4234eebd-->
## Compiler und Kernel

<!--en:4246a638-->
### Fusionierte Kernel programmieren

<!--en:342bd9f6-->
Die Compiler aus :numref:`sec_compilation` fusionieren Ketten elementweiser Operationen und Reduktionen (:cite:`Ansel.Yang.He.ea.2024` für PyTorch 2). Sie finden jedoch nicht von selbst Umformulierungen eines Algorithmus, die wie FlashAttention mit Kacheln arbeiten. Bei Triton :cite:`Tillet.Kung.Cox.2019` und bei Pallas in JAX ist die Tensorkachel die Programmiereinheit. Beim Programmieren legt man die Kachelung und die Pufferung auf dem Chip fest. Der Compiler verwaltet Threads, Vektorbefehle und gemeinsamen Speicher. FlashAttention-2 :cite:`Dao.2024` parallelisiert die Berechnung über die Kacheln der Abfragen. FlashAttention-3 :cite:`Shah.Bikshandi.Zhang.ea.2024` überlappt auf Hopper-GPUs die Datenübertragung mit den Produkten auf den Tensor Cores und der Berechnung von Exponentialfunktionen. Es unterstützt zudem fp8.

<!--en:090623ed-->
### Kompilierzeit bei großen Modellen

<!--en:d17843bc-->
Einen kleinen Trainingsschritt zu kompilieren (:numref:`subsec_comp-wholestep`) dauert Sekunden, bei einem großen Modell kann die Kompilierung pro Prozess und Form Minuten dauern. Bei der regionalen Kompilierung kompiliert man einen wiederkehrenden Block wie eine Transformer-Schicht nur einmal für alle seine Kopien (`torch.compiler.nested_compile_region`). Persistente Caches stellen die Kompilate mehreren Prozessen zur Verfügung (`torch.compiler.save_cache_artifacts`, `jax_compilation_cache_dir`). Das Autotuning (in PyTorch mit `mode='max-autotune'`, in XLA integriert) misst zur Kompilierzeit die Laufzeiten verschiedener Kachelungen für Matrixprodukte. Die erste Kompilierung dauert dadurch länger, die einzelnen Schritte werden schneller (:numref:`subsec_comp-wholestep`).

<!--en:6f6ab4d5-->
### Wiedergabe von Kernelstarts im Training

<!--en:578fa2da-->
Ein aufgezeichneter CUDA-Graph legt Formen und Speicheradressen fest (:numref:`subsec_comp-overhead`). Die CUDA-Graphbäume hinter `mode='reduce-overhead'` zeichnen Vorwärts- und Rückwärtsdurchlauf einmal pro Form in einem gemeinsamen Speicherpool auf. Die nächste Wiedergabe überschreibt Tensoren, die über mehrere Schritte erhalten bleiben sollen (etwa einen protokollierten Verlust), sofern man sie nicht vorher klont. Erkennt die Laufzeitumgebung eine Schrittgrenze nicht automatisch, markiert `torch.compiler.cudagraph_mark_step_begin` sie explizit.

<!--en:f418163a-->
### Export und frühere Ansätze für Berechnungsgraphen

<!--en:5ae5488c-->
Ein erfasster Graph lässt sich auch außerhalb von Python ausführen. `torch.export` erfasst einen vollständigen Graphen und lässt keine Graphunterbrechung zu. Eine JAX-Funktion mit Just-in-time-Kompilierung lässt sich in StableHLO übersetzen und damit auf Mobilgeräten, im Web und in Laufzeitumgebungen für Inferenzdienste bereitstellen. In früheren Frameworks wie Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` und TensorFlow 1 :cite:`Abadi.Barham.Chen.ea.2016` musste man statische symbolische Graphen explizit aufbauen. Tracing-Compiler gewinnen denselben Graphen für das gesamte Programm aus gewöhnlichem, unmittelbar ausgeführtem Code :cite:`Frostig.Johnson.Leary.2018`.

<!--en:243c0297-->
## Speicher

<!--en:ea815006-->
### Kleinerer Optimiererzustand

<!--en:5899876d-->
Die beiden Momentschätzungen von Adam belegen 8 der 16 Bytes pro Parameter aus :numref:`subsec_mp-anatomy`. Statt die Momente auf mehrere GPUs zu verteilen (:numref:`sec_sharding`), quantisiert 8-Bit-Adam :cite:`Dettmers.Lewis.Shleifer.ea.2022` beide blockweise. So benötigen sie zusammen nur 2 Bytes pro Parameter, ohne dass die Genauigkeit sinkt. Adafactor :cite:`Shazeer.Stern.2018` lässt das erste Moment weg und zerlegt das zweite Moment in Zeilen- und Spaltensummen.

<!--en:ed1ab25e-->
### Auslagerung in den Host-Speicher und auf Laufwerke

<!--en:11df9f73-->
Der DRAM des Hosts bietet eine Größenordnung mehr Kapazität als der GPU-Speicher und ist um eine Größenordnung günstiger. ZeRO-Offload :cite:`Ren.Rajbhandari.Aminabadi.ea.2021` hält die fp32-Hauptgewichte und die Momente von Adam im Host-Speicher. Der Optimierer aktualisiert die Parameter auf der CPU. Auf der GPU bleiben nur Gewichte und Gradienten mit jeweils 16 Bit. ZeRO-Infinity :cite:`Rajbhandari.Ruwase.Rasley.ea.2021` erweitert die Auslagerung auf NVMe-Laufwerke.

<!--en:f5658cf5-->
### Auswahl der neu zu berechnenden Werte

<!--en:33201ec9-->
Nur die Eingaben der Blöcke zu speichern (:numref:`subsec_mp-checkpointing`, :cite:`Chen.Xu.Zhang.ea.2016`) ist eine von mehreren Möglichkeiten, Speicherbedarf und Aufwand für die Neuberechnung gegeneinander abzuwägen. Bei einer festen Anzahl gespeicherter Zustände in einer Kette minimiert binomiales Checkpointing :cite:`Griewank.Walther.2008` den Aufwand für die Neuberechnung. Checkmate :cite:`Jain.Jain.Nrusimha.ea.2020` löst das Problem für allgemeine Berechnungsgraphen mit ganzzahliger linearer Optimierung. Moderne Compiler nutzen die selektive Neuberechnung auch automatisch (in PyTorch durch den Min-Cut-Partitionierer von AOTAutograd) oder nach expliziten Regeln (`jax.checkpoint_policies` in JAX).

<!--en:3062a30b-->
### Fragmentierung und Vorabreservierung

<!--en:ef607908-->
Wenn sich die Tensorformen von Schritt zu Schritt ändern, kann die Speicherverwaltung von PyTorch durch das Zwischenspeichern freien Speicher in nicht zusammenhängende Fragmente zerlegen („reserviert, aber nicht belegt“). Mit `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` können Segmente an Ort und Stelle wachsen. Die Speicherverwaltung von JAX reserviert beim Start standardmäßig 75% des GPU-Speichers. Mit `XLA_PYTHON_CLIENT_PREALLOCATE=false` lässt sich die Vorabreservierung deaktivieren, mit `XLA_PYTHON_CLIENT_ALLOCATOR=vmm` die virtuelle Speicherverwaltung von CUDA nutzen.

<!--en:afbbe1cd-->
## Kommunikation und Skalierung

<!--en:850e4cd9-->
### Kollektive Algorithmen jenseits des Rings

<!--en:be199326-->
Der Ring-Allreduce (:numref:`subsec_mg-ring`) minimiert die Gesamtzahl übertragener Bytes, benötigt aber $2(k-1)$ aufeinanderfolgende Übertragungsschritte. Bei Baumalgorithmen wächst die Anzahl dieser Schritte wie $\log_2 k$. NCCL kombiniert zwei komplementäre Binärbäume, sodass jedes Gerät in einem Baum ein Blatt und im anderen ein innerer Knoten ist. Ist $k$ eine Zweierpotenz, übertragen rekursive Halbierung und Verdopplung in $\log_2 k$ Schritten ebenso viele Bytes wie der Ring :cite:`Thakur.Rabenseifner.Gropp.2005`.

<!--en:5f7128b4-->
### Parameterserver und asynchrone Aktualisierungen

<!--en:20c318c1-->
Bevor Bibliotheken für kollektive Operationen zum Standard wurden, fasste man beim verteilten Training Gradienten auf Parameterservern zusammen :cite:`Dean.Corrado.Monga.ea.2012,Li.Andersen.Park.ea.2014`. Das ermöglichte asynchrone Aktualisierungen und lokalen stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD) :cite:`Stich.2019`. Veraltete Gradienten verändern die Dynamik der Optimierung. Deshalb sind synchrone Verfahren wie Allreduce und ZeRO (:cite:`Li.Zhao.Varma.ea.2020,Rajbhandari.Rasley.Ruwase.ea.2020`) heute bei dicht besetzten Modellen Standard. Für sehr große, dünn besetzte Tabellen in Empfehlungssystemen sind Parameterserver weiterhin verbreitet (:numref:`chap_recsys`).

<!--en:8defd137-->
### Schnittstellen für Parallelität und die Verteilung auf mehrere Rechner

<!--en:31051db4-->
In PyTorch steht für jede Parallelisierungsstrategie auf einem Gerätegitter eine Schnittstelle bereit, die sich mit den anderen kombinieren lässt (`DistributedDataParallel`, `fully_shard`, `torch.distributed.tensor.parallel`, `torch.distributed.pipelining`). In JAX beschreibt man alle Strategien deklarativ durch die Platzierung von Arrays auf einem Gitter mit benannten Achsen. GSPMD oder Shardy übernimmt die Kompilierung :cite:`Xu.Lee.Chen.ea.2021`. Der Buchteil über Sprachmodelle behandelt Tensor-, Pipeline- und Expertenparallelität über mehrere Rechner hinweg :cite:`shoeybi2019megatron,Narayanan.Shoeybi.Casper.ea.2021,Lepikhin.Lee.Xu.ea.2021` (:numref:`chap_lm_parallelism`) sowie das Training mit fp8 :cite:`Micikevicius.Stosic.Burgess.ea.2022,Liu.Feng.Xue.ea.2024` (:numref:`sec_lm_low_precision`). Dort behandeln wir auch kontinuierliches Batching und PagedAttention :cite:`Yu.Jeong.Kim.ea.2022,Kwon.Li.Zhuang.ea.2023` (:numref:`chap_lm_serving`) sowie die Quantisierung der Gewichte (:numref:`chap_lm_compression`).

<!--en:191246a0-->
[Diskussion](https://d2l.discourse.group/t/36854)
