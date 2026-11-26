<!--en:412e09c0-->
# Datenparallelität in der Praxis

<!--en:0ee7f281-->
In :numref:`sec_multi_gpu` verlangsamte unsere selbst geschriebene datenparallele Schleife LeNet auf zwei GPUs aus zwei Gründen. Ein einzelner Python-Thread startete die Kernel auf beiden GPUs nacheinander. Außerdem wartete der Allreduce auf das Ende des gesamten Rückwärtsdurchlaufs (engl. backward pass), obwohl die Gradienten der letzten Schichten schon viel früher vorlagen.

<!--en:34458d1e-->
Werkzeuge für datenparalleles Training im Produktivbetrieb beseitigen beide Engpässe. Sie starten entweder einen Host-Prozess pro GPU oder kompilieren ein einziges Programm, das sie auf alle GPUs verteilen. Außerdem fassen sie Gradienten zu *Bündeln* (engl. buckets) zusammen. Sobald ein Bündel im Rückwärtsdurchlauf voll ist, startet sein Allreduce. Wie viel vom Gradientenaustausch lässt sich so durch Berechnung überdecken? Um das zu testen, wechseln wir von LeNet zu ResNet-18 und verwenden Fashion-MNIST-Bilder der Größe $64 \times 64$. Pro Bild fallen 2.2 Milliarden Multiplikationsadditionen an, also rund 50 FLOPs pro Byte des 45 MB großen Gradienten. Ein Schritt mit 256 Bildern lastet eine RTX 4090 damit rund 100 ms lang aus. Anhand eines einzelnen Schritts prüfen wir, was der parallele Trainingsschritt berechnet. Wir messen die schwache und die starke Skalierung (engl. weak scaling und strong scaling) auf einer, zwei und vier GPUs und vergleichen die Ergebnisse mit unserem Kostenmodell.

<!--en:607aceb6-->
## Datenparallelität im Produktivbetrieb

<!--en:c7dc4357-->
### Das Ausführungsmodell

<!--en:ea6097a1-->
Jedes moderne Framework steuert $k$ GPUs nach einem von zwei Ausführungsmodellen an. Beide folgen dem Prinzip *Single Program, Multiple Data* (SPMD):

<!--en:88ca6568-->
1. **Mehrere Prozesse ($k$ Kopien + Bibliothek für kollektive Operationen):** Ein Startprogramm erzeugt $k$ unabhängige Python-Prozesse, einen pro GPU. Jeder Prozess hat einen eigenen *Rang* (engl. rank) aus $\{0, \ldots, k-1\}$ und einen eigenen CPU-Thread. Die Prozesse synchronisieren ihre Gradienten über NCCL. In jedem Prozess kann beliebiger Python-Code laufen.

<!--en:491cd6bd-->
2. **Ein Programm, das der Compiler partitioniert:** Wir schreiben einen Schritt für ein einzelnes Gerät und geben an, wie sich seine Eingaben auf ein Gerätegitter (engl. device mesh) verteilen. Der Compiler teilt die Berechnung auf, fügt die kollektiven Operationen ein und plant ihre Ausführung automatisch :cite:`Xu.Lee.Chen.ea.2021`.

<!--en:ced4e3e3-->
PyTorch verwendet mehrere Prozesse. `torchrun` startet $k$ Prozesse, und `DistributedDataParallel` (DDP) kapselt das Modell. Da ein Notebook in einem einzigen Prozess läuft, schreibt unsere Hilfsfunktion `torchrun` ein eigenständiges Skript in eine temporäre Datei. Anschließend führt sie `torchrun --standalone --nproc-per-node=k` aus und liest die JSON-Ausgabe jedes Rangs ein.

<!--en:db9b0d11-->
In JAX übernimmt der Compiler die Partitionierung. `nnx.jit` kompiliert einen Trainingsschritt für ein einzelnes Gerät. Kennzeichnen wir die Parameter als repliziert (`P()`) und die Eingabe als entlang der Batchachse partitioniert (`P('data')`), fügt der Shardy/GSPMD-Partitionierer automatisch den Allreduce für die Gradienten ein.

<!--en:f7aad6d2-->
### Überlappung von Kommunikation und Rückwärtsdurchlauf

<!--en:d27667dd-->
Frameworks für den Produktivbetrieb fassen die Gradienten in umgekehrter Schichtreihenfolge zu *Bündeln* von einigen Dutzend Megabyte zusammen. In dieser Reihenfolge liefert die Backpropagation die Gradienten. Sobald der letzte Gradient eines Bündels vorliegt, startet dessen asynchroner Allreduce :cite:`Li.Zhao.Varma.ea.2020` (:numref:`fig_ddp_overlap`).

<!--en:363b5d96-->
![Oben wartet der Allreduce auf das Ende des gesamten Rückwärtsdurchlaufs. Unten bilden die Gradienten Bündel. Sobald der Rückwärtsdurchlauf ein Bündel gefüllt hat, startet dessen Allreduce. Nur der Allreduce des letzten Bündels läuft nach Ende des Rückwärtsdurchlaufs. Die früheren Allreduces sind nur dann vollständig überdeckt, wenn jeder von ihnen noch während des Rückwärtsdurchlaufs endet.](../img/mdl-perf-ddp-overlap.svg)

<!--en:05f02a8e-->
Bei der Wahl der Bündelgröße wägen wir die Latenz $\alpha$ pro Aufruf gegen die mögliche Überlappung ab. Bei winzigen Bündeln fällt $\alpha$ viele Male an. Bei einem einzigen großen Bündel kann der Allreduce erst nach dem Rückwärtsdurchlauf starten. Im idealen *Bündelmodell* sind die Verbindungen zwischen den GPUs schnell genug, um jedes Bündel zu übertragen, bevor das nächste voll ist. Außerdem bremst die Kommunikation die gleichzeitig laufenden Kernel des Rückwärtsdurchlaufs nicht. Unter diesen Annahmen überdeckt der Rückwärtsdurchlauf alle Allreduces bis auf den letzten.

<!--en:7d28b52b-->
DDP schließt das erste Bündel bei 1 MiB und alle weiteren bei 25 MiB (`bucket_cap_mb`). Für die 45 MB an Gradienten von ResNet-18 bildet DDP drei Bündel von rund 9, 26 und 9 MB. Das letzte Bündel mit 9 MB enthält die Gradienten der frühen Schichten, die zuletzt anfallen.

<!--en:3af6d37e-->
XLA fasst Allreduces für Gradienten bis zu einem Schwellenwert von 30 MiB zusammen (`--xla_gpu_all_reduce_combine_threshold_bytes`). Für die 45 MB an Gradienten von ResNet-18 entstehen zwei asynchrone Paare aus `all-reduce-start` und `all-reduce-done`, die XLA zeitgleich mit den Berechnungen des Rückwärtsdurchlaufs einplant.

<!--en:c6d66c16-->
### Der Trainingsschritt

<!--en:98c1ff7a-->
Wir definieren die ResNet-18-Variante für kleine Bilder aus :numref:`sec_resnet`. Mit `#@save` speichern wir sie für :numref:`chap_cv`. Wir bestimmen die Größe $S$ ihrer Gradienten in Byte (45 MB in fp32):

<!--en:c4b82ea0-->
Als Nächstes laden wir Fashion-MNIST und bestimmen für ResNet-18 die Parameteranzahl `num_params` und das Datenvolumen `S` der Gradienten in fp32:

<!--en:3b6eb283-->
Wir bestimmen auch die Gradientenbündel `buckets`, die DDP in umgekehrter Reihenfolge mit Schwellenwerten von 1 MiB und 25 MiB füllt, bevor es die Allreduces startet.

<!--en:2a8c1699-->
Nun definieren wir den datenparallelen Trainingsschritt für $64 \times 64$ Pixel große Fashion-MNIST-Bilder:

<!--en:bbdb1767-->
`TRAIN_DDP` kapselt `resnet18` in `DDP` und misst die Laufzeit von `train_step` über alle Prozesse hinweg (wahlweise mit `model.no_sync()` und einem Laufzeitprofil der Kernel).

<!--en:45b09c1b-->
`setup` repliziert das Modell und den Datensatz auf dem Gerätegitter `'data'`. `train_step` führt ein JIT-kompiliertes SGD-Update mit den partitionierten Minibatch-Indizes `idx` aus.

<!--en:6b9ff784-->
### Prüfung anhand eines Trainingsschritts

<!--en:9588c6d3-->
Anders als LeNet verwendet ResNet-18 die Batchnormalisierung (:numref:`sec_batch_norm`). Sie normalisiert jeden Kanal anhand seines Mittelwerts und seiner Varianz im Minibatch. Was geschieht, wenn wir einen Minibatch mit 256 Beispielen auf zwei GPUs aufteilen? Verwendet jede GPU die Statistiken ihrer eigenen 128 Beispiele, wie bei der Gradientenakkumulation über zwei Mikrobatches zu je 128 Beispielen auf einer GPU? Oder synchronisieren die GPUs die globalen Statistiken aller 256 Beispiele? Vergleichen wir das Update auf zwei GPUs mit beiden Referenzrechnungen auf einer GPU.

<!--en:3f4081f7-->
PyTorch DDP normalisiert standardmäßig die Teilmenge jeder GPU anhand ihrer lokalen Statistiken. Das Update stimmt bis auf etwa $10^{-7}$ mit dem Ergebnis der Akkumulation über beide Hälften überein. Vom Update mit den Statistiken des gesamten Batchs weicht es um mehrere Prozent ab. Die Umstellung auf `nn.SyncBatchNorm` synchronisiert die globalen Statistiken, erfordert aber zusätzliche kollektive Operationen (Übung :exref:`ex_mgp-syncbn`).

<!--en:e698ab3f-->
Die Batchachse ist entlang der Gitterachse `'data'` im Modus `Auto` partitioniert. Deshalb berechnet der XLA-Partitionierer die Mittelwerte und Varianzen für die Batchnormalisierung global über den gesamten Minibatch. Der Schritt entspricht damit dem Schritt mit 256 Beispielen auf einer GPU. Neben den beiden großen Gradientenpuffern fügt XLA kleine Allreduces für die kanalweisen Statistiken der Batchnormalisierung ein:

<!--en:18691108-->
Wir kompilieren `train_step` für das Gitter aus mehreren GPUs und werten das Ergebnis mit `allreduce_megabytes` aus. Neben den kleinen kanalweisen Allreduces für die Batchnormalisierung finden wir zwei große Allreduces für die Gradienten (`big`, der letzte Puffer ist als `last_bucket` erfasst):

<!--en:f42cdf9b-->
## Skalierung auf vier GPUs

<!--en:273845bf-->
Wir messen nun die schwache Skalierung (256 Beispiele pro GPU) und die starke Skalierung (insgesamt 512 Beispiele) auf $k \in \{1, 2, 4\}$ GPUs.

<!--en:f437de37-->
Zunächst messen wir $\beta_\textrm{link}$ mit einem isolierten NCCL-Allreduce über 256 MB. Bei schwacher Skalierung vergleichen wir zwei Vorhersagen für die nicht überdeckte Kommunikationszeit $t_{\textrm{step}}(k) - t_{\textrm{step}}(1)$:

<!--en:e9dfabbd-->
1. **Keine Überlappung:** Der gesamte Ring-Allreduce über $S = 45$ MB trägt zur Schrittzeit bei ($\frac{2(k-1)}{k} S \beta_\textrm{link}$).

<!--en:0fb0c95f-->
2. **Vollständige Überlappung der Bündel:** Nach Ende des Rückwärtsdurchlaufs trägt nur noch das letzte Bündel beziehungsweise der letzte Puffer ($last\_bucket$) zur Schrittzeit bei.

<!--en:bf36961e-->
Der Aufruf von `allreduce_beta()` liefert die gemessenen Verbindungskosten `beta_link`. Daraus berechnen wir `exposed[k]` für $k \in \{2, 4\}$ GPUs nach beiden Modellen, einmal ohne Überlappung ($2(k-1)S\beta_\textrm{link}/k$) und einmal mit vollständiger Überlappung (`last_bucket`):

<!--en:b6403425-->
Nun messen wir die schwache Skalierung bei 256 Beispielen pro GPU (`step_time(k, 256)`) auf $k \in \{1, 2, 4\}$ GPUs. Wir vergleichen den gemessenen Durchsatz, die Skalierungseffizienz und den nicht überdeckten Zeitaufwand `weak[k] - weak[1]` mit beiden Vorhersagen:

<!--en:fabf84c6-->
Bei einer Busbandbreite von rund 2.4 GB/s dauert ein Allreduce über 45 MB ohne Überlappung rund 18 ms auf zwei GPUs und rund 28 ms auf vier GPUs. Bei schwacher Skalierung erreichen vier GPUs $3.1\text{--}3.3\times$ so viel Durchsatz wie eine GPU (77–83% Skalierungseffizienz). Das kommt der Vorhersage *ohne Überlappung* sehr nahe und bleibt weit hinter der Vorhersage des Bündelmodells mit vollständiger Überlappung zurück.

<!--en:8c2f51ca-->
Bei starker Skalierung bleibt der globale Batch bei 512 Beispielen. Jede der $k$ GPUs verarbeitet also $512/k$ Beispiele, während das Kommunikationsvolumen gleich bleibt. Bei $k = 4$ sinkt die Effizienz weiter:

<!--en:b5d509d0-->
### Annahmen des Bündelmodells

<!--en:4ca093b8-->
Warum hat die Überlappung so wenig Kommunikationszeit eingespart? Das Bündelmodell beruht auf drei Annahmen. (1) Der Allreduce jedes Bündels startet während des Rückwärtsdurchlaufs. (2) NCCL erreicht dieselbe Übertragungsrate wie im isolierten Benchmark. (3) Die Rechenkernel des Rückwärtsdurchlaufs werden nicht langsamer, wenn NCCL gleichzeitig läuft. Mit Laufzeitprofilen von GPU 0 prüfen wir alle drei Annahmen in drei Konfigurationen: 1 GPU, 2 GPUs ohne Kommunikation und 2 GPUs mit Kommunikation.

<!--en:beb7c03b-->
Wir führen `profile_step` mit $b = 256$ Beispielen pro GPU in den drei Konfigurationen aus: eine GPU, zwei GPUs ohne Gradientensynchronisierung und zwei GPUs mit Allreduce. So können wir für die Laufzeit eines Schritts die Rechenzeit auf dem Gerät und die Laufzeit der NCCL-Kernel getrennt betrachten:

<!--en:6292a865-->
Ohne Kommunikation dauert ein Schritt auf zwei GPUs genau so lange wie auf einer GPU. Die NCCL-Kernel laufen 20–22 ms, was zu den 18 ms im isolierten Benchmark passt. Die dritte Annahme trifft jedoch nicht zu: **Läuft NCCL gleichzeitig mit dem Rückwärtsdurchlauf, brauchen die Rechenkernel selbst 8–13 ms länger**. Auf GPUs ohne direkten Peer-to-Peer-Zugriff kopiert NCCL die Daten mit eigenen GPU-Kerneln über den Host-Speicher. Diese Kernel belegen Streaming Multiprocessors (SMs) und Speicherbandbreite, die sonst dem Rückwärtsdurchlauf zur Verfügung stünden.

<!--en:44901085-->
Eine undokumentierte NCCL-Option (`NCCL_SHM_USE_CUDA_MEMCPY=1`, verfügbar in NCCL 2.28–2.30) kann diese Kopiervorgänge über den Host-Speicher in isolierten Mikrobenchmarks an die DMA-Kopiereinheiten der GPU übertragen:

<!--en:deda2485-->
Dieselbe Option führt in DDP zu einem Deadlock, wenn sich die Allreduces mit den Berechnungen des Rückwärtsdurchlaufs überlappen.

<!--en:e5ef6ff2-->
Auf Rechenzentrums-GPUs mit NVLink-Verbindung (450–900 GB/s pro Richtung, :numref:`subsec_hw-interconnects`) dauert der Allreduce über die 45 MB von ResNet-18 deutlich weniger als eine Millisekunde. Damit skaliert das Training nahezu linear. Selbst bei idealer Skalierung repliziert die Datenparallelität jedoch alle $16N$ Byte des Parameter- und Optimiererzustands auf jeder Karte. In :numref:`sec_sharding` verteilen wir diesen Zustand auf mehrere GPUs.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:36948aa3-->
Bei der Datenparallelität im Produktivbetrieb arbeiten $k$ GPUs entweder mit $k$ unabhängigen, über NCCL synchronisierten Prozessen oder mit einem einzigen, vom Compiler partitionierten Programm. Beide Ansätze fassen Gradienten zu Bündeln von 25–30 MiB zusammen, damit sich die Allreduces mit dem Rückwärtsdurchlauf überlappen. Der Vergleich eines Trainingsschritts mit Referenzrechnungen auf einer GPU zeigt, ob die Batchnormalisierung lokale Statistiken der jeweiligen Teilmenge verwendet oder mit zusätzlichen kollektiven Operationen globale Statistiken berechnet. Unsere vier RTX 4090 kommunizieren bei einer Busbandbreite von 2.4 GB/s über den Host-Speicher. Bei schwacher Skalierung erreichen sie $3.1\text{--}3.3\times$ so viel Durchsatz wie eine GPU. Die Laufzeitprofile zeigen, dass die NCCL-Kernel beim Datentransfer über den Host mit den Rechenkerneln um SMs konkurrieren. Dadurch geht der größte Teil der Zeit wieder verloren, die die Überlappung von Kommunikation und Rückwärtsdurchlauf einspart.

<!--en:2294243a-->
## Übungen

<!--en:7e19e56e-->
1. [code] **Aufteilung der Daten.** In `TRAIN_DDP` entnimmt der Prozess mit Rang $r$ einer gemeinsamen Zufallspermutation den Slice ab Position $(sk + r)b$ für Schritt $s$.
    1. Entfernen Sie `rank` aus der Berechnung des Startindex, sodass alle Prozesse dieselben $b$ Beispiele verwenden. Sagen Sie voraus, wie sich `step_time` bei $k = 2$ und der gemittelte Gradient verhalten. Überprüfen Sie Ihre Vorhersage, indem Sie auf jedem Rang eine Prüfsumme des Minibatchs ausgeben.
    1. Stellen Sie den ursprünglichen Startindex wieder her, aber geben Sie jedem Rang einen anderen Zufallsstartwert. Welche Eigenschaft der Minibatch-Aufteilung geht verloren?
    1. Schlagen Sie eine einfache Laufzeitprüfung vor, die bereits im ersten Schritt identische Teilmengen auf verschiedenen Rängen erkennt.

<!--en:3de9337c-->
1. [code] **Aufteilung der Daten.** In `minibatches` legt `P('data')` die Verteilung der Indizes fest.
    1. Verwenden Sie stattdessen `P()`, damit jede GPU alle Indizes erhält. Sagen Sie voraus, welche kollektiven Operationen das kompilierte `train_step` enthält und wie sich `step_time(2, 256)` im Vergleich zu `step_time(1, 256)` verhält. Prüfen Sie beide Vorhersagen.
    1. Stellen Sie `P('data')` wieder her. Legen Sie in `train_step` für `X[idx]` die Verteilung `NamedSharding(mesh, P(None, 'data'))` fest, sodass die GPUs das Bild entlang der Höhe aufteilen. Welche Randwerte müssen benachbarte GPUs für eine $3 \times 3$-Faltung austauschen? Welche kollektive Operation erscheint im kompilierten HLO-Code?

<!--en:77f4c861-->
2. **Extrapolation der schwachen Skalierung.** Nehmen Sie an, die Laufzeit eines Schritts auf $k$ GPUs ist die Summe aus der Laufzeit $t_1$ auf einer GPU und der nicht überdeckten Kommunikationszeit. Letztere sei proportional zum Datenverkehr $2(k-1)S/k$ im Ring.
    1. Drücken Sie die Effizienz bei schwacher Skalierung $E(k) = t_1 / t_{\textrm{step}}(k)$ durch eine einzige Konstante aus. Bestimmen Sie die Konstante anhand des Messwerts für $E(2)$ und vergleichen Sie die Vorhersage für $E(4)$ mit dem Messwert in :numref:`subsec_mgp-scaling`.
    1. Bei Gradientenakkumulation über $a$ Mikrobatches mit je 256 Beispielen pro GPU kommunizieren die GPUs nur beim letzten Mikrobatch. Bestimmen Sie das kleinste $a$, das $E(4)$ auf 95% anhebt. Geben Sie an, wie sich die globale Batchgröße ändert.

<!--en:553c3576-->
3. [code] **Bündelgröße.** ● Modellieren Sie die Überlappung der Bündel (:numref:`fig_ddp_overlap`) unter folgenden Annahmen. Ein Rückwärtsdurchlauf der Dauer $t_b$ liefert $n$ gleich große Bündel zu den Zeitpunkten $i\,t_b/n$ ($i = 1, \ldots, n$). Die Allreduces laufen nacheinander und dauern jeweils $\alpha + \tau/n$, wobei $\tau = S\,\beta_\textrm{link}$ gilt.
    1. Zeigen Sie, dass für die nicht überdeckte Kommunikationszeit nach Ende des Rückwärtsdurchlaufs gilt:
       $$t_{\textrm{exposed}}(n) = \max\Bigl(\alpha + \frac{\tau}{n},\;
       n\alpha + \tau - \frac{n-1}{n}\,t_b\Bigr),$$
       Bestimmen Sie das optimale $n$ für die beiden Fälle $\tau < t_b$ und $\tau > t_b$.
    1. Variieren Sie `bucket_cap_mb` in `DDP` von 1 MB bis 100 MB bei $k = 2$. Vergleichen Sie die gemessene nicht überdeckte Kommunikationszeit mit der Vorhersage des Modells.
   :exlabel:`ex_mgp-buckets`

<!--en:bf6b432e-->
4. [code] **Synchronisierte Batchnormalisierung.** ● Die Prüfung anhand eines Trainingsschritts in :numref:`subsec_mgp-production` hat gezeigt, dass DDP jede Teilmenge standardmäßig mit lokalen Statistiken normalisiert.
    1. Konvertieren Sie `model` in `CHECK_DDP` mit `nn.SyncBatchNorm.convert_sync_batchnorm`, bevor Sie das Modell in `DDP` kapseln. Wiederholen Sie die Prüfung. Mit welcher Referenz auf einer GPU stimmt das Ergebnis jetzt überein?
    1. Zählen Sie die zusätzlichen kollektiven Operationen, die `SyncBatchNorm` pro Schritt einfügt. Messen Sie `step_time` bei $k = 2$ und $k = 4$. Erklären Sie, warum die Laufzeit der zusätzlichen Operationen das Produkt aus ihrem Datenvolumen in Byte und $\beta_\textrm{link}$ übersteigt.
   :exlabel:`ex_mgp-syncbn`

<!--en:8e21cfd8-->
3. [code] **Bündelgröße.** ● Modellieren Sie die Überlappung der Bündel (:numref:`fig_ddp_overlap`) unter folgenden Annahmen. Ein Rückwärtsdurchlauf der Dauer $t_b$ liefert $n$ gleich große Bündel zu den Zeitpunkten $i\,t_b/n$ ($i = 1, \ldots, n$). Die Allreduces laufen nacheinander und dauern jeweils $\alpha + \tau/n$, wobei $\tau = S\,\beta_\textrm{link}$ gilt.
    1. Zeigen Sie, dass für die nicht überdeckte Kommunikationszeit nach Ende des Rückwärtsdurchlaufs gilt:
       $$t_{\textrm{exposed}}(n) = \max\Bigl(\alpha + \frac{\tau}{n},\;
       n\alpha + \tau - \frac{n-1}{n}\,t_b\Bigr),$$
       Bestimmen Sie das optimale $n$ für die beiden Fälle $\tau < t_b$ und $\tau > t_b$.
    1. Variieren Sie `--xla_gpu_all_reduce_combine_threshold_bytes` von 1 MB bis 100 MB. Starten Sie für jeden Wert einen neuen Prozess, zählen Sie die resultierenden Allreduces für die Gradienten und vergleichen Sie die gemessene nicht überdeckte Kommunikationszeit mit der Vorhersage des Modells.
   :exlabel:`ex_mgp-buckets`

<!--en:20d756af-->
4. [code] **Statistiken der Batchnormalisierung.** ● Bei `nnx.jit` mit Gitterachsen im Modus `Auto` berechnet die Batchnormalisierung globale Statistiken über alle GPUs.
    1. Erklären Sie die 74 kleinen Allreduces in `allreduce_megabytes(compiled)` anhand der Anzahl der `BatchNorm`-Schichten in `resnet18` und der Berechnungen im Vorwärts- und Rückwärtsdurchlauf.
    1. Rufen Sie vor dem Kompilieren `model.eval()` auf, damit `BatchNorm` gleitende Mittelwerte anstelle der Batchstatistiken verwendet. Prüfen Sie, ob die kleinen Allreduces verschwinden, und messen Sie, wie viel Zeit pro Schritt sie bei $k = 1, 2, 4$ kosten.
   :exlabel:`ex_mgp-syncbn`

<!--en:423a7b67-->
[Diskussion](https://d2l.discourse.group/t/36792)

<!--en:3f48dac5-->
Distributed Data Parallel (DDP) in der Praxis: Bündelung und Überlappung<br>**SPMD mit mehreren Prozessen oder per Compiler · Gradientenbündelung · Überlappung mit dem Rückwärtsdurchlauf · Verhalten von BatchNorm · Skalierung und Konkurrenz um SMs**

<!--en:c39f4f82-->
::: {.slide title="Vom einfachen LeNet zum rechenintensiven ResNet-18" only="pytorch"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:26271afe-->
- **Warum LeNet in §{sec_multi_gpu} ausgebremst wurde:** Ein Python-Thread startete die Kernel auf beiden GPUs nacheinander ($k \cdot h_{\textrm{step}}$). Der Allreduce wartete auf das Ende von `backward()`.

<!--en:766d3bf2-->
- **Abhilfe durch DDP im Produktivbetrieb:** DDP startet **1 Prozess pro GPU** (`torchrun`). Die Allreduces für die einzelnen Bündel beginnen *während* `backward()`.

<!--en:6fb6c76f-->
- **Benchmark mit ResNet-18 ($64 \times 64$, $b=256$):** Jede RTX 4090 ist für $\sim 100\text{ ms/step}$ mit $11.2\text{M}$ Parametern ($S = 45\text{ MB}$) ausgelastet. So lässt sich die tatsächliche Überlappung mit der Kommunikation messen.

<!--en:baf56e1b-->
::: {.slide title="Vom einfachen LeNet zum rechenintensiven ResNet-18" only="jax"}

<!--en:e02fdbdc-2-->
[Motivation]{.kicker}

<!--en:26271afe-2-->
- **Warum LeNet in §{sec_multi_gpu} ausgebremst wurde:** Ein Python-Thread startete die Kernel auf beiden GPUs nacheinander ($k \cdot h_{\textrm{step}}$). Der Allreduce wartete auf das Ende von `backward()`.

<!--en:01ea7f28-->
- **Abhilfe durch den Compiler:** Er kompiliert **1 Programm für mehrere GPUs** (`jax.jit`). Die von XLA eingefügten Allreduces laufen *während* des Rückwärtsdurchlaufs.

<!--en:6fb6c76f-2-->
- **Benchmark mit ResNet-18 ($64 \times 64$, $b=256$):** Jede RTX 4090 ist für $\sim 100\text{ ms/step}$ mit $11.2\text{M}$ Parametern ($S = 45\text{ MB}$) ausgelastet. So lässt sich die tatsächliche Überlappung mit der Kommunikation messen.

<!--en:b26faf98-->
::: {.slide title="SPMD mit mehreren Prozessen: ein Python-Prozess pro GPU" only="pytorch"}

<!--en:728ab2d9-->
[Ausführungsmodelle]{.kicker}

<!--en:e8d87c33-->
- **SPMD mit mehreren Prozessen (`torchrun` + `DDP`):** Es laufen $k$ unabhängige Python-Prozesse (1 pro GPU). Autograd-Hooks starten während `backward()` asynchrone NCCL-Allreduces für die Bündel.

<!--en:39fdb034-->
::: {.slide title="Vom Compiler partitioniertes SPMD: ein Programm für k GPUs" only="jax"}

<!--en:728ab2d9-2-->
[Ausführungsmodelle]{.kicker}

<!--en:104e3863-->
- **Vom Compiler partitioniertes SPMD (`jax.jit` + `Mesh`):** Ein Prozess erfasst per Tracing eine mit `NamedSharding` annotierte Funktion für ein einzelnes Gerät. XLA (GSPMD/Shardy) fügt in HLO kollektive NCCL-Operationen ein und plant sie überlappend ein.

<!--en:d99e3656-->
::: {.slide title="Wie DDP die Engpässe der selbst geschriebenen Schleife beseitigt" only="pytorch"}

<!--en:88567b84-->
[PyTorch DDP]{.kicker}

<!--en:12285187-->
- **Paralleler Kernel-Start durch den Host ($1 \cdot h_{\textrm{step}}$):** Jeder der $k$ von `torchrun` gestarteten Prozesse steuert gleichzeitig seine eigene GPU an. Damit entfällt der Engpass von $k \cdot h_{\textrm{step}}$ durch einen einzelnen Thread.

<!--en:5ef8f6b9-->
- **Kollektive NCCL-Operationen auf Ring oder Baum:** Beim sternförmigen Sammeln und Verteilen mit Aufwand $O(kS)$ ist der zentrale Knoten der Engpass. Bandbreitenoptimale kollektive NCCL-Operationen kommen mit $\frac{2(k-1)}{k}S$ aus.

<!--en:24610f06-->
- **Rückwärts-Hooks für die überlappende Ausführung:** Sie starten asynchrone Allreduces für $25\text{ MiB}$ große, in umgekehrter Reihenfolge gefüllte Bündel, während für frühere Schichten noch die Backpropagation läuft.

<!--en:227417f6-->
::: {.slide title="Wie der Compiler die Datenparallelität durch Partitionierung automatisiert" only="jax"}

<!--en:d7ab51d6-->
[JIT-Kompilierung mit Partitionierung in JAX]{.kicker}

<!--en:f007d007-->
- **Code für ein einzelnes Gerät:** Wie gewohnt `train_step(model, opt, X, y, idx)` schreiben, ohne `shard_map`, `ravel_pytree` oder `lax.psum` von Hand aufzurufen.

<!--en:02f501bb-->
- **Automatisch erzeugte kollektive Operationen:** Sind die Gewichte als repliziert (`P()`) und die Batchindizes als partitioniert (`P('data')`) gekennzeichnet, fügt XLA automatisch Allreduces für die Gradienten ein.

<!--en:24644b2a-->
- **Asynchrone Bündelung bis 30 MiB:** XLA fasst Allreduces bis $30\text{ MiB}$ zusammen und plant Berechnungen des Rückwärtsdurchlaufs zwischen `all-reduce-start` und `all-reduce-done` ein.

<!--en:51ec55b7-->
::: {.slide title="Ein Prozess pro GPU mit torchrun und DDP" only="pytorch"}

<!--en:923104f1-->
[Ablauf mit mehreren Prozessen]{.kicker}

<!--en:dd16d8cb-->
`torchrun --standalone --nproc-per-node=k` startet $k$ Prozesse. Jeder Prozess bindet sich über `LOCAL_RANK` an die ihm zugewiesene GPU und kapselt das Modell in `DDP`:

<!--en:43426f53-->
Bei der Initialisierung verteilt DDP die Anfangswerte der Parameter und Puffer von Rang $0$ an alle Prozesse. Für jeden Parameter registriert DDP Autograd-Hooks, die während `backward()` die NCCL-Allreduces für die Bündel starten.

<!--en:627c56f7-->
::: {.slide title="Deklarative Verteilung auf dem Gerätegitter mit NamedSharding" only="jax"}

<!--en:081419f0-->
[Partitionierung auf dem Gerätegitter]{.kicker}

<!--en:966ce62e-->
In JAX genügt ein `train_step` wie für eine einzelne GPU. Das Modell wird repliziert (`P()`), die Minibatch-Indizes werden partitioniert (`P('data')`). Das Gerätegitter verwendet den Modus `Auto`:

<!--en:ae45aa85-->
`model` ist repliziert (`P()`), `idx` hingegen partitioniert (`P('data')`). Die Ableitung des über `'data'` gemittelten Verlusts erfordert automatisch ein Allreduce über `'data'`, damit die aktualisierten Parameter auf allen Geräten übereinstimmen.

<!--en:fe5a93a6-->
::: {.slide title="Überlappung der Allreduces für Gradientenbündel mit der Backpropagation"}

<!--en:be09af21-->
[Überlappung der Kommunikation]{.kicker}

<!--en:6fbdd9fc-->
- **Gradienten in umgekehrter Reihenfolge:** Die Backpropagation berechnet die Gradienten von der letzten Ausgabeschicht zurück bis zur ersten Eingabeschicht.

<!--en:49953467-->
- **Überlappende Kommunikation und Berechnung:** Die Gradienten bilden in umgekehrter Parameterreihenfolge **Bündel** von $25\text{--}30\text{ MiB}$. Der Allreduce jedes Bündels läuft in einem separaten Kommunikationsstrom, während für frühere Schichten noch die Backpropagation läuft.

<!--en:5efbf928-->
::: {.slide title="Wahl der Bündelgröße und ideales Bündelmodell"}

<!--en:e65677b7-->
[Bündelmodell]{.kicker}

<!--en:d9fba47b-->
- **Viele kleine Bündel ($c \gg 1$) oder 1 großes Bündel ($c = 1$):** Bei kleinen Bündeln fällt die Latenz Dutzende Male an, insgesamt $c\alpha$. Ein einziges großes Bündel muss auf das Ende des Rückwärtsdurchlaufs warten.

<!--en:1253f937-->
- **Günstige Bündelgröße von 25–30 MiB:** Bei idealer Überlappung (schnelle Verbindungen, keine Konkurrenz um SMs) überdeckt die Backpropagation alle Bündel bis auf das letzte mit der Größe $S_{\textrm{last}}$.

<!--en:0ca0117e-->
::: {.slide title="Bündelung in umgekehrter Reihenfolge in PyTorch DDP (1 und 25 MiB)" only="pytorch"}

<!--en:976c295c-->
[DDP-Bündel]{.kicker}

<!--en:1a6f49c1-->
PyTorch DDP schließt das erste Bündel bei $1\text{ MiB}$, damit die Kommunikation fast sofort beginnt. Für alle weiteren gilt `bucket_cap_mb = 25` ($25\text{ MiB} = 26.2\text{ MB}$):

<!--en:654afe78-->
- **Bündel von ResNet-18 (insgesamt $S = 44.7\text{ MB}$):** Beim Durchlaufen von `reversed(list(net.parameters()))` entstehen drei Bündel mit **$9.4\text{ MB}$, $26.2\text{ MB}$ und $9.0\text{ MB}$**. Schon ein einzelner $9.4\text{ MB}$ großer Gewichtstensor einer späten `Conv2d`-Schicht überschreitet die erste Schwelle von $1\text{ MiB}$.

<!--en:ba768dda-->
- **Vorhersage bei idealer Überlappung:** Enden die Allreduces der ersten beiden Bündel ($9.4$ und $26.2\text{ MB}$) während der Backpropagation, bleibt nur das letzte Bündel mit **$S_{\textrm{last}} = 9.0\text{ MB}$** übrig. Es enthält die zuletzt berechneten Gradienten der frühesten Schichten. Die nicht überdeckte Kommunikation sinkt um den Faktor **$5\times$**.

<!--en:d6ce97df-->
::: {.slide title="Asynchrone Bündelung von Allreduces in XLA (30 MiB)" only="jax"}

<!--en:d3936478-->
[Bündelung kollektiver Operationen in XLA]{.kicker}

<!--en:22148851-->
In JAX fasst der XLA-Compilerdurchlauf `all-reduce-combiner` die Allreduces für Gradienten bis zur Schwelle von $30\text{ MiB}$ zusammen (`--xla_gpu_all_reduce_combine_threshold_bytes`):

<!--en:5c9d7cd5-->
- **Puffer von ResNet-18 (insgesamt $S = 44.7\text{ MB}$):** XLA bündelt die Parametergradienten für zwei asynchrone Paare aus `all-reduce-start` und `all-reduce-done` mit **$25.8\text{ MB}$** und **$18.9\text{ MB}$**. Die Faltungen des Rückwärtsdurchlaufs plant XLA zwischen `start` und `done` ein.

<!--en:3db25691-->
- **Vorhersage bei idealer Überlappung:** Endet die Übertragung des ersten Puffers mit $25.8\text{ MB}$ während der Backpropagation für frühere Schichten, bleibt nach dem Rückwärtsdurchlauf nur der letzte Puffer mit **$S_{\textrm{last}} = 18.9\text{ MB}$** übrig.

<!--en:95737bc5-->
::: {.slide title="Prüfung anhand eines Schritts: lokale Batchnormalisierung in PyTorch DDP" only="pytorch"}

<!--en:f43a71ed-->
[Verhalten der Batchnormalisierung]{.kicker}

<!--en:59d1fba6-->
Anders als LeNet enthält ResNet-18 `BatchNorm2d`-Schichten. Für einen DDP-Schritt auf 2 GPUs mit $256$ Bildern gibt es zwei Referenzrechnungen: (a) eine GPU mit allen $256$ Bildern und (b) eine GPU, die Gradienten über zwei Hälften zu je $128$ Bildern akkumuliert. Der Vergleich zeigt, was `BatchNorm2d` berechnet:

<!--en:2ee742e6-->
- **Standardmäßig lokale Statistiken:** PyTorch DDP normalisiert die $128$ Bilder jeder GPU anhand ihres **eigenen lokalen Mittelwerts und ihrer Varianz**. Das Update stimmt bis auf $10^{-7}$ mit der Akkumulation über beide Hälften überein. Von der globalen Batchnormalisierung über 256 Bilder weicht es um $\sim 6\%$ der Update-Größe ab.

<!--en:fda4ab83-->
::: {.slide title="Prüfung anhand eines Schritts: globale Batchnormalisierung im Auto-Gerätegitter von JAX" only="jax"}

<!--en:f43a71ed-2-->
[Verhalten der Batchnormalisierung]{.kicker}

<!--en:fdb3a971-->
Für einen `nnx.jit`-Schritt auf 2 GPUs mit $256$ Bildern gibt es zwei Referenzrechnungen: (a) eine GPU mit allen $256$ Bildern und (b) eine GPU, die Gradienten über zwei Hälften zu je $128$ Bildern akkumuliert. Der Vergleich zeigt, welche Statistiken die Batchnormalisierung verwendet:

<!--en:fcdbcc9f-->
- **Globale Batchstatistiken bei `Auto`-Partitionierung:** Die Batchdimension ist entlang `'data'` im Modus `AxisType.Auto` partitioniert. XLA verteilt die Reduktionen für Mittelwert und Varianz der Batchnormalisierung **global auf alle GPUs**. Das Update stimmt mit dem für 256 Bilder auf einer GPU bis auf $\sim 10^{-5}$ überein. Gegenüber zwei unabhängigen Hälften beträgt die Abweichung $1.7 \times 10^{-2}$.

<!--en:8271517a-->
::: {.slide title="Lokale Batchnormalisierung und SyncBatchNorm im Vergleich" only="pytorch"}

<!--en:6e5c7db9-->
[Synchronisierung der Statistiken]{.kicker}

<!--en:42b5f2ad-->
Wann macht die lokale Batchnormalisierung einen Unterschied, und was kostet ihre Synchronisierung?

<!--en:75a578c3-->
- **Wann die lokale Batchnormalisierung gut funktioniert:** Bei großen Teilmengen pro GPU mit $b = B/k$ ($b \ge 32\text{--}64$ Bilder/GPU) sind die lokalen Mittelwerte und Varianzen Schätzwerte mit geringer Varianz. Eine Kommunikationsbarriere im Vorwärtsdurchlauf entfällt.

<!--en:c08b1a43-->
- **Wann `nn.SyncBatchNorm` nötig ist:** Bei der Objekterkennung oder Segmentierung mit winzigen Batches ($b = 1\text{--}4$ hochauflösende Bilder/GPU) schwanken die lokalen Statistiken zu stark. `nn.SyncBatchNorm.convert_sync_batchnorm(model)` fügt im Vorwärts- und Rückwärtsdurchlauf **für jede Schicht mit Batchnormalisierung** einen Allreduce ein. So entstehen Dutzende Synchronisierungspunkte, deren Kosten die Latenz $\alpha$ bestimmt.

<!--en:4ce2697d-->
::: {.slide title="Kollektive Operationen im kompilierten HLO-Code untersuchen" only="jax"}

<!--en:8b56306d-->
[Kollektive Operationen des Compilers]{.kicker}

<!--en:60128aae-->
Die Analyse des kompilierten HLO-Texts (`compiled.as_text()`) mit `allreduce_megabytes` bestätigt sowohl die beiden gebündelten Gradientenpuffer als auch die globalen Reduktionen der Batchnormalisierung:

<!--en:fa62ed27-->
- **Zwei große Puffer und 74 kleine Aufrufe:** Die beiden großen Allreduces ($25.8\text{ MB}$ und $18.9\text{ MB}$) übertragen die $44.7\text{ MB}$ an Parametergradienten. Die $74$ kleinen Allreduces (insgesamt $0.10\text{ MB}$) synchronisieren kanalweise Mittelwerte, Varianzen und Aktivierungsgradienten für alle 20 Schichten mit Batchnormalisierung im Vorwärts- und Rückwärtsdurchlauf.

<!--en:535d16b0-->
::: {.slide title="Unbemerkte Fehler bei der Datenpartitionierung im verteilten Training"}

<!--en:aa8e1c30-->
[Fehlerquellen in der Praxis]{.kicker}

<!--en:155555e4-->
- **Fehler 1 (identische Teilmengen auf allen Rängen):** Fehlt `rank` im Startindex des Slice oder werden die Indizes mit `P()` repliziert, rechnen alle $k$ GPUs mit denselben $b$ Beispielen. So bleiben $k-1$ GPUs ungenutzt.

<!--en:877cda21-->
- **Fehler 2 (Mischen ohne gemeinsamen Zufallsstartwert):** Unabhängige Zufallspermutationen ziehen über die Ränge hinweg Beispiele *mit Zurücklegen*, anstatt die Daten der Epoche zu partitionieren.

<!--en:1c7303e5-->
- **Prüfung in Schritt 1:** Bei gemeinsamem Zufallsstartwert für die Epoche per Zusicherung prüfen, ob die Minibatches auf verschiedenen Rängen unterschiedliche Prüfsummen haben.

<!--en:f9b47f8f-->
::: {.slide title="Grenzen der nicht überdeckten Kommunikation mit und ohne Bündelung"}

<!--en:f71d221d-->
[Theoretische Vorhersagen]{.kicker}

<!--en:ac235475-->
Ein isolierter NCCL-Allreduce liefert den Messwert für $\beta_{\textrm{link}}$ bei $256\text{ MB}$ Datenvolumen ($\sim 2.4\text{ GB/s}$ Busbandbreite, also $\beta_{\textrm{link}} \approx 0.42\text{ ns/B}$). Daraus ergeben sich zwei theoretische Grenzen für die nicht überdeckte Kommunikationszeit $t_{\textrm{step}}(k) - t_{\textrm{step}}(1)$ bei schwacher Skalierung:

<!--en:d5005e2e-->
- **Zwei Vorhersagen:** Ohne Überlappung sind **$\sim 18\text{ ms}$** ($k=2$) und **$\sim 28\text{ ms}$** ($k=4$) zu erwarten. Bei idealer Überlappung bleibt nur das letzte Bündel bzw. der letzte Puffer übrig (**$4\text{--}8\text{ ms}$** bei $k=2$, **$5\text{--}12\text{ ms}$** bei $k=4$).

<!--en:ff262940-->
::: {.slide title="Schwache Skalierung auf vier GPUs: 3.1–3.3× so viel Durchsatz"}

<!--en:71a06e07-->
[Messung der schwachen Skalierung]{.kicker}

<!--en:0b28927f-->
Bei fester Batchgröße von $b = 256$ Bildern pro GPU lassen sich auf $k \in \{1, 2, 4\}$ GPUs der Durchsatz bei schwacher Skalierung und die nicht überdeckte Kommunikationszeit $t_{\textrm{step}}(k) - t_{\textrm{step}}(1)$ messen:

<!--en:1a5b0948-->
Vier GPUs erreichen **$3.1\text{--}3.3\times$ so viel Durchsatz wie eine GPU** ($77\text{--}83\%$ Skalierungseffizienz). Die gemessene nicht überdeckte Kommunikationszeit entspricht der **Vorhersage ohne Überlappung** ($\sim 18\text{ ms}$ bei $k=2$, $\sim 28\text{ ms}$ bei $k=4$) und liegt über der Vorhersage des idealen Bündelmodells.

<!--en:c8950dcf-->
::: {.slide title="Starke Skalierung auf vier GPUs: sinkender Rechenanteil"}

<!--en:e9389891-->
[Messung der starken Skalierung]{.kicker}

<!--en:d16540a6-->
Starke Skalierung bei fester globaler Batchgröße $B = 512$ ($b = 512/k$ pro GPU) für $k \in \{1, 2, 4\}$:

<!--en:d325cd8d-->
- **Warum die Effizienz bei starker Skalierung für $k = 4$ schneller sinkt:** Bei $k = 4$ verarbeitet jede GPU nur $128$ Bilder. Damit halbiert sich die lokale Rechenzeit des Rückwärtsdurchlaufs gegenüber $b = 256$. Für die $45\text{ MB}$ an Gradienten bleibt das Übertragungsvolumen des Allreduce mit $\frac{2(k-1)}{k}S = 1.5\,S$ unverändert.

<!--en:b794134a-->
::: {.slide title="Warum spart die Überlappung so wenig Zeit? Laufzeitprofil von GPU 0"}

<!--en:c7cfc960-->
[Prüfung der Modellannahmen]{.kicker}

<!--en:a25d5f46-->
Die Allreduces starten bereits während `backward()`. Warum entspricht die nicht überdeckte Kommunikationszeit trotzdem der Formel ohne Überlappung? Dazu dienen Laufzeitprofile von GPU 0 aus drei Läufen: (1) 1 GPU, (2) 2 GPUs ohne Kommunikation (`no_sync` / `shard_map`) und (3) 2 GPUs mit vollständiger Kommunikation:

<!--en:b7bbd273-->
**Annahme 3 des Bündelmodells trifft nicht zu:** Die Kommunikationskernel von NCCL überlappen sich zeitlich mit der Backpropagation. **Die parallel zu NCCL laufenden Rechenkernel des Rückwärtsdurchlaufs brauchen aber $8\text{--}13\text{ ms}$ länger**. Damit geht der größte Teil der eingesparten Zeit wieder verloren.

<!--en:a6a406dd-->
::: {.slide title="Konkurrenz um SMs durch NCCL-Kopiervorgänge über den Host" only="pytorch"}

<!--en:f08d41f0-->
[Engpass beim Datentransport]{.kicker}

<!--en:338f8c3e-->
Warum verlangsamt ein NCCL-Allreduce auf unserer Workstation die gleichzeitig laufenden Kernel für Faltungen?

<!--en:2927e363-->
- **Datentransport mit NCCL ohne Peer-to-Peer-Zugriff (P2P):** Die RTX 4090 für den Verbrauchermarkt unterstützt weder NVLink noch PCIe-P2P. NCCL überträgt jeden Datenblock über den gemeinsamen Host-DRAM (`SHM`). Dazu nutzt es **eigene GPU-Kernel auf den SMs**, die über PCIe lesen und schreiben und die Blöcke auf den SMs summieren.

<!--en:b29c6dd8-->
- **Konkurrenz um Ressourcen:** Diese NCCL-Kernel konkurrieren im Rückwärtsdurchlauf direkt mit den Kerneln für Faltungen und GEMM um **Streaming Multiprocessors** und **DRAM-Speicherbandbreite**.

<!--en:5f0a1604-->
Übernehmen die DMA-Kopiereinheiten den Transfer über den Host (`NCCL_SHM_USE_CUDA_MEMCPY=1`), erreicht der isolierte Allreduce $4\text{--}5\times$ so viel Bandbreite. Bei Überlappung mit dem Rückwärtsdurchlauf kommt es in PyTorch DDP jedoch zu einem Deadlock. Transportoptionen immer in der tatsächlichen Trainingsschleife prüfen.

<!--en:ca762d90-->
::: {.slide title="Konkurrenz um SMs durch NCCL-Kopiervorgänge über den Host" only="jax"}

<!--en:f08d41f0-2-->
[Engpass beim Datentransport]{.kicker}

<!--en:338f8c3e-2-->
Warum verlangsamt ein NCCL-Allreduce auf unserer Workstation die gleichzeitig laufenden Kernel für Faltungen?

<!--en:2927e363-2-->
- **Datentransport mit NCCL ohne Peer-to-Peer-Zugriff (P2P):** Die RTX 4090 für den Verbrauchermarkt unterstützt weder NVLink noch PCIe-P2P. NCCL überträgt jeden Datenblock über den gemeinsamen Host-DRAM (`SHM`). Dazu nutzt es **eigene GPU-Kernel auf den SMs**, die über PCIe lesen und schreiben und die Blöcke auf den SMs summieren.

<!--en:b29c6dd8-2-->
- **Konkurrenz um Ressourcen:** Diese NCCL-Kernel konkurrieren im Rückwärtsdurchlauf direkt mit den Kerneln für Faltungen und GEMM um **Streaming Multiprocessors** und **DRAM-Speicherbandbreite**.

<!--en:75df2a19-->
Übernehmen die DMA-Kopiereinheiten den Transfer über den Host (`NCCL_SHM_USE_CUDA_MEMCPY=1`), erreicht der isolierte Allreduce $4\text{--}5\times$ so viel Bandbreite. Der isolierte Mikrobenchmark sagt jedoch nichts über die Trainingsschleife mit Überlappung aus. Transportoptionen immer in der tatsächlichen Trainingsschleife prüfen.

<!--en:f2f2fd7e-->
::: {.slide title="PCIe-Übertragung über den Host und NVLink-Verbindungen"}

<!--en:0c8a12a8-->
[Einfluss der GPU-Verbindung]{.kicker}

<!--en:b31cad40-->
- **PCIe 4.0 über Host-DRAM auf Verbraucherkarten ($\sim 2.4\text{ GB/s}$ Busbandbreite):** Ein Allreduce über $45\text{ MB}$ benötigt bei $k=4$ $\sim 28\text{ ms}$ und konkurriert um SMs. Die Effizienz bei schwacher Skalierung beträgt $77\text{--}83\%$ ($3.1\text{--}3.3\times$).

<!--en:9e6c2e94-->
- **NVLink 4/5 im Rechenzentrum ($450\text{--}900\text{ GB/s}$):** Eigene Übertragungseinheiten erledigen denselben Allreduce über $45\text{ MB}$ in $< 0.2\text{ ms}$. Die Skalierungseffizienz erreicht $>99\%$ ($\approx 4.0\times$).

<!--en:7a261daa-->
- **Nächste Grenze:** Auch mit NVLink repliziert DDP alle $16N$ Byte des Zustands auf jeder GPU (§{sec_sharding}).

<!--en:94b9cc06-->
::: {.slide title="Verständnisfragen zu Bündelung, Batchnormalisierung und Skalierung" only="pytorch"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:1ee91468-->
Drei Fragen zur Datenparallelität im Produktivbetrieb:

<!--en:7284b3af-->
1. In welcher Parameterreihenfolge bündelt DDP die Gradienten? Warum würde die Schichtreihenfolge des Vorwärtsdurchlaufs ($1, 2, \ldots, L$) die Überlappung mit der Kommunikation verhindern?

<!--en:4b6706f7-->
2. Auf unseren vier RTX 4090 ergab die schwache Skalierung bei $b = 256$ Bildern/GPU eine Effizienz von $E(4) = t_1 / t_{\textrm{step}}(4) \approx 80\%$ ($t_1 \approx 100\text{ ms}$ Rechenzeit + $28\text{ ms}$ nicht überdeckte Allreduce-Zeit). Welche Effizienz ergibt sich, wenn vor dem Allreduce die Gradienten über $a = 5$ Mikrobatches zu je $256$ Bildern pro GPU akkumuliert werden (`model.no_sync()` für die ersten $4$ Mikrobatches)? Gesucht ist $E(4)$.

<!--en:df806d12-->
3. Ein Vergleich eines Trainingsschritts auf 2 GPUs mit dem Training auf 1 GPU verwendet ein Netz mit `BatchNorm`. Mit welcher Referenz auf einer GPU stimmt `DDP` in der Standardeinstellung überein?

<!--en:557b98b6-->
**Lösungen:**

<!--en:71f82d96-->
1. **Umgekehrte Parameterreihenfolge ($L, L-1, \ldots, 1$):** Denn `backward()` liefert die Gradienten von Schicht $L$ zurück bis Schicht $1$. Der Allreduce eines in Vorwärtsreihenfolge gefüllten Bündels mit Schicht $1$ könnte erst ganz am Ende von `backward()` starten.

<!--en:518d6017-->
2. **$\mathbf{E(4) \approx 95\%}$:** Die lokale Rechenzeit steigt um den Faktor $5\times$ auf $5 \times 100 = 500\text{ ms}$. Der einzige Allreduce dauert weiterhin $28\text{ ms}$. Das ergibt $E(4) = \frac{500}{500 + 28} \approx 94.7\%$, allerdings bei einem um den Faktor $5\times$ größeren globalen Batch $B = 5 \times 4 \times 256 = 5{,}120$.

<!--en:eaf3cd9c-->
3. **`DDP`** verwendet standardmäßig lokale Statistiken, genau wie **1 GPU bei der Akkumulation über zwei Hälften der Größe $B/2$**.

<!--en:2de9da18-->
::: {.slide title="Verständnisfragen zu Bündelung, Batchnormalisierung und Skalierung" only="jax"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:1ee91468-2-->
Drei Fragen zur Datenparallelität im Produktivbetrieb:

<!--en:e624febf-->
1. In welcher Parameterreihenfolge sollten die Gradienten gebündelt werden? Warum würde die Schichtreihenfolge des Vorwärtsdurchlaufs ($1, 2, \ldots, L$) die Überlappung mit der Kommunikation verhindern?

<!--en:a1825221-->
2. Auf unseren vier RTX 4090 ergab die schwache Skalierung bei $b = 256$ Bildern/GPU eine Effizienz von $E(4) = t_1 / t_{\textrm{step}}(4) \approx 80\%$ ($t_1 \approx 100\text{ ms}$ Rechenzeit + $28\text{ ms}$ nicht überdeckte Allreduce-Zeit). Welche Effizienz ergibt sich, wenn vor dem Allreduce die Gradienten über $a = 5$ Mikrobatches zu je $256$ Bildern pro GPU akkumuliert werden? Gesucht ist $E(4)$.

<!--en:d91f24c3-->
3. Ein Vergleich eines Trainingsschritts auf 2 GPUs mit dem Training auf 1 GPU verwendet ein Netz mit `BatchNorm`. Mit welcher Referenz auf einer GPU stimmt die Partitionierung im Modus `Auto` überein?

<!--en:557b98b6-2-->
**Lösungen:**

<!--en:71f82d96-2-->
1. **Umgekehrte Parameterreihenfolge ($L, L-1, \ldots, 1$):** Denn `backward()` liefert die Gradienten von Schicht $L$ zurück bis Schicht $1$. Der Allreduce eines in Vorwärtsreihenfolge gefüllten Bündels mit Schicht $1$ könnte erst ganz am Ende von `backward()` starten.

<!--en:518d6017-2-->
2. **$\mathbf{E(4) \approx 95\%}$:** Die lokale Rechenzeit steigt um den Faktor $5\times$ auf $5 \times 100 = 500\text{ ms}$. Der einzige Allreduce dauert weiterhin $28\text{ ms}$. Das ergibt $E(4) = \frac{500}{500 + 28} \approx 94.7\%$, allerdings bei einem um den Faktor $5\times$ größeren globalen Batch $B = 5 \times 4 \times 256 = 5{,}120$.

<!--en:a2264ab8-->
3. **Die Partitionierung im Modus `Auto`** fügt GPU-übergreifende Reduktionen für die Statistiken der Batchnormalisierung ein. Das entspricht **1 GPU mit dem gesamten Batch $B$**.

<!--en:41573c56-->
::: {.slide title="Zusammenfassung: Datenparallelität im Produktivbetrieb und Überlappung" only="pytorch"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:245fd9ed-->
- **SPMD-Ausführung:** `torchrun` + `DDP` ($k$ Prozesse) beseitigen den Engpass von $k \cdot h_{\textrm{step}}$ durch einen einzelnen Thread.

<!--en:6de04781-->
- **Bündelung in umgekehrter Reihenfolge:** Gradientenbündel von $25\text{--}30\text{ MiB}$ in umgekehrter Parameterreihenfolge ermöglichen die überlappende Ausführung von NCCL-Allreduces und Backpropagation.

<!--en:8854fcb2-->
- **Prüfung und Ressourcenkonkurrenz:** Anhand eines einzelnen Trainingsschritts auf Fehler bei Batchnormalisierung und Partitionierung prüfen. Bei PCIe-Übertragungen über den Host die Konkurrenz um SMs mit einem Laufzeitprofil untersuchen.

<!--en:d278285f-->
::: {.slide title="Zusammenfassung: Datenparallelität im Produktivbetrieb und Überlappung" only="jax"}

<!--en:c5a222f8-2-->
[Das Wichtigste]{.kicker}

<!--en:92c2b61f-->
- **SPMD-Ausführung:** `jax.jit` + `NamedSharding` (1 kompiliertes Programm) beseitigen den Engpass von $k \cdot h_{\textrm{step}}$ durch einen einzelnen Thread.

<!--en:6de04781-2-->
- **Bündelung in umgekehrter Reihenfolge:** Gradientenbündel von $25\text{--}30\text{ MiB}$ in umgekehrter Parameterreihenfolge ermöglichen die überlappende Ausführung von NCCL-Allreduces und Backpropagation.

<!--en:8854fcb2-2-->
- **Prüfung und Ressourcenkonkurrenz:** Anhand eines einzelnen Trainingsschritts auf Fehler bei Batchnormalisierung und Partitionierung prüfen. Bei PCIe-Übertragungen über den Host die Konkurrenz um SMs mit einem Laufzeitprofil untersuchen.
