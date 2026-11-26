<!--en:89775820-->
# Rechenleistung

<!--en:549c6289-->
Bisher sind wir bei jedem Modell gleich vorgegangen. Wir haben das Netz programmiert, die Trainingsschleife gestartet und gewartet. Die Dauer eines Schritts und den Speicherbedarf haben wir als gegeben hingenommen. Das funktioniert nicht mehr, wenn ein Trainingslauf Tage dauert, ein Modell nicht in den Speicher der Grafikkarte passt oder eine zweite GPU das Training verlangsamt. Betrachten wir den GPT aus :numref:`sec_gpt`. Wir verdoppeln seine Breite, damit die GPU mehr zu tun hat. Teilen wir die Anzahl der Rechenoperationen pro Trainingsschritt durch die Spitzenrechenleistung der GPU, erhalten wir die kürzestmögliche Dauer eines Schritts. Der gemessene Schritt dauert jedoch um ein Mehrfaches länger. Wo bleibt die übrige Zeit?

<!--en:9065c484-->
In der Rechnerarchitektur stellte sich diese Frage lange vor dem Deep Learning. Schon in den 1990er-Jahren waren Prozessoren dem Speicher so weit voraus, dass viele Programme die meiste Zeit auf Daten warteten. 2009 fassten Williams, Waterman und Patterson die Konsequenz im *Roofline*-Modell zusammen :cite:`Williams.Waterman.Patterson.2009`. Die Geschwindigkeit eines Programms hängt sowohl davon ab, wie schnell die Recheneinheiten arbeiten, als auch davon, wie schnell der Speicher die benötigten Bytes liefert. Steuert Python eine GPU an, kommt eine dritte Grenze hinzu. Der Host braucht Zeit, um jede einzelne Operation zu starten. Welche Änderung einen Schritt beschleunigt, hängt davon ab, welche dieser drei Grenzen ihn bremst. Diese Sichtweise erklärt Entscheidungen aus früheren Kapiteln, etwa den Einsatz großer Minibatches und von 16-Bit-Zahlen (:numref:`sec_numerics`). Sie liefert uns auch eine Methode. Wir zählen die Rechenoperationen und die übertragenen Bytes eines Schritts, sagen seine Dauer vorher, messen sie und erklären die Abweichung.

<!--en:c6fc2e2b-->
Die Grundlage für das gesamte Kapitel legt :numref:`sec_perf_model`. Dort führen wir die *arithmetische Intensität* ein und wenden die Methode an. Die arithmetische Intensität gibt an, wie viele Gleitkommaoperationen (engl. floating-point operations, FLOPs) eine Operation pro übertragenem Byte ausführt. Der Abschnitt zur Hardware (:numref:`sec_hardware`) erklärt, woraus sich die Spitzenwerte für Rechenleistung und Bandbreite ergeben. Die meisten Fachleute nutzen bei der Arbeit mit einer einzelnen GPU täglich die Kompilierung (:numref:`sec_compilation`) und die Verfahren zur Speichernutzung aus :numref:`sec_memory_precision`. In den nächsten drei Abschnitten kommen weitere GPUs hinzu. Zunächst erarbeiten wir die Datenparallelität von Grund auf (:numref:`sec_multi_gpu`) und setzen sie in der Praxis ein (:numref:`sec_multi_gpu_concise`). Dann teilen wir Gewichte, Gradienten und Optimiererzustand auf mehrere Karten auf (:numref:`sec_sharding`). So können wir auch Modelle trainieren, die nicht in den Speicher einer einzelnen GPU passen. In der abschließenden Fallstudie (:numref:`sec_fast_transformer`) untersuchen wir den eingangs betrachteten GPT auf einer und auf vier GPUs. Wir schlüsseln in Millisekunden auf, wie sich die Zeit für einen Trainingsschritt zusammensetzt.

<!--en:db7377ea-->
Wir setzen voraus, dass Sie die Laufzeit von GPU-Code messen und den maximalen Speicherbedarf ablesen können (:numref:`sec_use_gpu`). Außerdem sollten Ihnen bf16 und tf32 (:numref:`sec_numerics`) sowie der GPT und sein KV-Cache (:numref:`sec_gpt`, :numref:`sec_kv-cache`) vertraut sein. Alle Messungen stammen von einem einzigen Rechner mit vier RTX 4090, PyTorch 2.11 und JAX 0.12. Auf Ihrem Rechner werden Sie andere Zeiten messen. Die Anzahl der FLOPs und der Bytes bleibt gleich, und aus beiden Größen lässt sich die Laufzeit mit derselben Methode vorhersagen.

<!--en:d3fc9c8f-->
Rechenleistung – Roofline-Modelle, Hardware, Compiler und Skalierung auf mehrere GPUs<br>**arithmetische Intensität · Grenzen durch Roofline und Kernel-Start · Speicher und Rechengenauigkeit · Fusion und CUDA-Graphen · kollektive Operationen und FSDP**

<!--en:9cfc653f-->
::: {.slide title="Vier Engpässe bestimmen Trainingszeit und Kapazität"}

<!--en:acc8203e-->
[Das zentrale Problem]{.kicker}

<!--en:83f99cd3-->
- **Rechenleistung ($P$) und Bandbreite ($\beta$):** Matrixmultiplikationen der Größe $8192^2$ in `bf16` (GEMMs) lasten die Tensor Cores bei rund $165\text{ TFLOP/s}$ aus. Elementweise Additionen ($I = 0.25$) erreichen nur $0.25\text{ TFLOP/s}$.

<!--en:c1bba8c6-->
- **Kernel-Start durch den Host ($h$):** Im Eager-Modus kostet der Kernel-Start aus Python $10\text{--}100\,\mu\text{s}$ pro Operation. Zwischen den jeweils nur wenige Mikrosekunden dauernden Aufrufen wartet die GPU.

<!--en:98072254-->
- **Speicherkapazität ($C$):** Ein GPT mit einer Milliarde Parametern und Adam in `fp32` belegt $16N \approx 15.1\text{ GiB}$ auf einer Karte mit $24\text{ GiB}$, noch bevor die Aktivierungen eines einzigen Tokens im Speicher liegen.

<!--en:d4347d14-->
::: {.slide title="Drei Untergrenzen für die Laufzeit und der Knickpunkt der Roofline"}

<!--en:fdea0be7-->
[Gemeinsames analytisches Modell]{.kicker}

<!--en:37d5a07f-->
- **Arithmetische Intensität ($I = \textrm{FLOPs}/\textrm{bytes}$):** Anzahl der Rechenoperationen pro Byte, das zwischen dem externen DRAM und den Registern oder dem SRAM auf dem Chip übertragen wird.

<!--en:f13f4db5-->
- **Knickpunkt ($I^* = P/\beta$):** Hier trifft die Bandbreitengrenze $I\beta$ auf die Grenze der Rechenleistung $P$ ($I^* \approx 164\text{ FLOP/B}$ in `bf16` auf der RTX 4090, $\sim 295$ auf der H100).

<!--en:81da9a50-->
- **Skalierung auf mehrere GPUs:** Bei der Verteilung auf $k$ GPUs kommen die Latenz der kollektiven Kommunikation und die Übertragungszeit hinzu, insgesamt $\alpha + \frac{2(k-1)}{k}S\beta_{\textrm{link}}$.

<!--en:f822e7b2-->
::: {.slide title="Fünf Hardwarekonstanten kennzeichnen unseren Benchmark-Rechner" only="pytorch"}

<!--en:58ad8689-->
[Referenzrechner]{.kicker}

<!--en:64333825-->
- **Rechenleistung und Bandbreite ($P, \beta, I^*$):** Die vier Karten mit je $24\text{ GiB}$ (RTX 4090) erreichen pro Karte $P = 165.2\text{ TFLOP/s}$ (`bf16`) und eine DRAM-Bandbreite von $\beta = 1.01\text{ TB/s}$ ($I^* \approx 164\text{ FLOP/B}$).

<!--en:aa13136d-->
- **Start durch den Host ($h$):** Der Start einer einzelnen Operation kostet $\sim 13\,\mu\text{s}$.

<!--en:c64da95f-->
- **Verbindung zwischen GPUs ($\beta_{\textrm{link}}$):** Allreduce über PCIe 4.0 mit Zwischenspeicherung im Host-Speicher erreicht eine Busbandbreite von $\sim 2.4\text{ GB/s}$ ($0.42\text{ ns/B}$).

<!--en:07e28794-->
::: {.slide title="Fünf Hardwarekonstanten kennzeichnen unseren Benchmark-Rechner" only="jax"}

<!--en:58ad8689-2-->
[Referenzrechner]{.kicker}

<!--en:64333825-2-->
- **Rechenleistung und Bandbreite ($P, \beta, I^*$):** Die vier Karten mit je $24\text{ GiB}$ (RTX 4090) erreichen pro Karte $P = 165.2\text{ TFLOP/s}$ (`bf16`) und eine DRAM-Bandbreite von $\beta = 1.01\text{ TB/s}$ ($I^* \approx 164\text{ FLOP/B}$).

<!--en:8931d8c6-->
- **Start durch den Host ($h$):** Ohne JIT-Kompilierung kostet der Start einer einzelnen Operation $50\text{--}100\,\mu\text{s}$.

<!--en:c64da95f-2-->
- **Verbindung zwischen GPUs ($\beta_{\textrm{link}}$):** Allreduce über PCIe 4.0 mit Zwischenspeicherung im Host-Speicher erreicht eine Busbandbreite von $\sim 2.4\text{ GB/s}$ ($0.42\text{ ns/B}$).

<!--en:84b25b7f-->
::: {.slide title="Aufbau von Kapitel §{num:chap_performance}"}

<!--en:ddfd1d8f-->
[Kapitelübersicht]{.kicker}

<!--en:044ce76f-->
- **§{sec_perf_model}–§{num:sec_hardware} Roofline und Hardware:** Arithmetische Intensität, synchronisierte Zeitmessung, Speicherhierarchie, SMs, Tensor Cores und die konstante Dauer von $20\text{ ms}$ für einen vollständigen Speicherdurchlauf.

<!--en:c521751c-->
- **§{sec_compilation}–§{num:sec_memory_precision} Compiler und Speicher:** Symbolisches Tracing, Fusion von Operatoren ($9\times$), CUDA-Graphen, Speicherbedarf des Adam-Zustands ($16N$), `bf16`, Neuberechnung von Aktivierungen und Gradientenakkumulation.

<!--en:cfd0074a-->
- **§{sec_multi_gpu}–§{num:sec_fast_transformer} Mehrere GPUs und Fallstudie:** Ring-Allreduce, Bündelung von Gradienten in DDP, Partitionierung mit ZeRO/FSDP und schrittweise Beschleunigung des GPT um den Faktor $6\times$.

<!--en:83b000ef-->
::: {.slide title="Wie Sie dieses Kapitel durcharbeiten"}

<!--en:123f19cf-->
[Hinweise zum Studium]{.kicker}

<!--en:e740867f-->
- **Erst vorhersagen, dann messen:** Notieren Sie vor dem Ausführen jeder Benchmark-Zelle die Anzahl der FLOPs, die Zahl der übertragenen Bytes und die vorhergesagte Untergrenze $\max(h, \textrm{FLOPs}/P, \textrm{bytes}/\beta)$. Weicht die Messung von Ihrer Rechnung ab, weist die Differenz auf eine übersehene Speicherübertragung, eine verborgene Synchronisationsbarriere auf dem Host oder konkurrierende Zugriffe auf die SMs hin.

<!--en:51ab4ecd-->
- **Zuerst die numerische Äquivalenz prüfen:** Verändert eine Optimierung unbemerkt Gradienten, Statistiken der Batchnormalisierung oder die kausale Maskierung, liegt ein Fehler vor. In jedem Abschnitt ergänzen wir die Laufzeitmessungen durch Gradientenvergleiche für einen einzelnen Schritt (`allclose`) und prüfen im vollständigen Trainingslauf, ob der Validierungsverlust die Vorgaben erfüllt.

<!--en:6ba34608-->
**Als Nächstes (§{sec_perf_model} Das Leistungsmodell):** Wir definieren die arithmetische Intensität, leiten die Roofline-Schranke her, vermeiden Messfehler bei asynchroner Ausführung und analysieren das Laufzeitprofil eines vollständigen MLP-Trainingsschritts.
