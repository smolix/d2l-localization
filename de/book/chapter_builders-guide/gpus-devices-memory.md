<!--en:a74121a0-->
# GPUs, Geräte und Speicher

<!--en:916eade1-->
Seit :numref:`sec_oo-design` nimmt unser `Trainer` das Argument `num_gpus` entgegen und ignoriert es höflich. Bei den winzigen Modellen der früheren Kapitel fiel das kaum ins Gewicht, denn auf der CPU dauerte das Training nur Sekunden. Bei größeren Modellen entscheidet die Gerätezuordnung (engl. device placement) jedoch darüber, ob wir Sekunden oder Stunden warten. Eine moderne Desktop-GPU multipliziert zwei $4096 \times 4096$-Matrizen in weniger als 2 Millisekunden. Eine CPU braucht dafür mindestens eine Zehntelsekunde, wie wir in :numref:`subsec_async_dispatch` messen. Diese Rechenleistung hat eine technische Voraussetzung. Die GPU verfügt über eigenen Speicher (bei einer Desktop-Karte etwa 24 GB), und jeder Tensor, mit dem sie rechnet, muss dort liegen.

<!--en:50553a47-->
Passt das gewünschte Modell überhaupt in den Speicher? Unser Beispielmodell mit 8 Milliarden Parametern braucht 16 GB für seine Gewichte in bf16. Für die Inferenz reicht der Speicher also gut aus. Beim Training mit Adam in fp32 belegt jedoch schon der statische Zustand 128 GB (:numref:`sec_parameters`), bevor wir auch nur eine Aktivierung speichern. Das übersteigt die Kapazität einer einzelnen Karte mit 24 GB oder 80 GB bei Weitem. Ob ein Modell auf ein Gerät passt und ob die GPU durchgehend rechnet oder auf Daten wartet, hängt davon ab, wo jeder Tensor liegt und wie viele Bytes er belegt.

<!--en:c5e4f454-->
In diesem Abschnitt beantworten wir drei Fragen. Wie legen wir Tensoren und Modelle auf CPUs und GPUs ab und kopieren sie von einem Gerät auf ein anderes? Wie funktioniert die asynchrone Ausführung auf der GPU, und wie messen wir die Laufzeit von GPU-Kerneln genau? Was belegt beim Training den GPU-Speicher, und wie spart *Aktivierungs-Checkpointing* mit zusätzlichem Rechenaufwand einen großen Teil dieses Speichers ein? Anschließend bauen wir die Gerätezuordnung in den `Trainer` ein, den wir im weiteren Verlauf des Buchs verwenden. In :numref:`sec_parameters` und :numref:`sec_numerics` haben wir gezählt, wie viele Bytes ein Parameter oder eine Zahl belegt. Hier legen wir diese Bytes auf konkreter Hardware ab. Das Training mit mehreren GPUs behandeln wir in :numref:`chap_performance`. Ohne GPU läuft jede Codezelle auf der CPU. So lässt sich die Korrektheit auch dann prüfen, wenn für die Zeitmessung eine GPU nötig ist.

<!--en:55d9abc9-->
## Geräte und Gerätezuordnung

<!--en:d81f641b-->
### Geräteobjekte

<!--en:b6c53d98-->
Jeder Tensor liegt auf einem bestimmten *Gerät* (engl. device), entweder auf der CPU mit dem Arbeitsspeicher des Host-Rechners oder auf einer einzelnen GPU mit ihrem eigenen Speicher. Der Terminalbefehl `nvidia-smi` listet alle NVIDIA-GPUs Ihres Rechners auf. Für jede GPU zeigt er den gesamten und den belegten Speicher sowie die aktiven Prozesse an. Die vier Frameworks benennen Geräte etwas unterschiedlich, folgen aber demselben Prinzip. Jede Operation läuft auf einem einzigen Gerät, also müssen alle ihre Operanden dort liegen.

<!--en:fd176641-->
`npx.cpu()` und `npx.gpu(i)` bezeichnen die CPU und die GPU $i$ (vor MXNet 2.0 hieß die Geräteangabe `ctx`).

<!--en:50ad4bdc-->
`torch.device('cpu')` und `torch.device(f'cuda:{i}')` bezeichnen die CPU und die GPU $i$.

<!--en:7dd4df34-->
`tf.device('/CPU:0')` und `tf.device(f'/GPU:{i}')` legen fest, auf welchem Gerät die Operationen innerhalb des jeweiligen Geltungsbereichs laufen.

<!--en:3b6b0920-->
`jax.devices('cpu')[0]` und `jax.devices('gpu')[i]` liefern Verweise auf die jeweiligen Hardwaregeräte.

<!--en:908bf274-->
Für die Geräteabfrage schreiben wir drei kleine Hilfsfunktionen, die wir im weiteren Verlauf des Buchs importieren. Eine liefert die CPU, eine die GPU $i$ und die dritte die Anzahl verfügbarer GPUs.

<!--en:9e7663db-->
Ein Geräteobjekt lässt sich immer erzeugen. Erst wenn wir dort einen Tensor ablegen, prüft das Framework, ob die Hardware vorhanden ist.

<!--en:ff4f3abe-->
Standardmäßig nutzt TensorFlow *Soft Placement*. Ist die angeforderte GPU nicht vorhanden, weicht TensorFlow ohne Meldung auf die CPU aus.

<!--en:88afa312-->
`jax.devices('gpu')` fragt sofort die vorhandene Hardware ab und löst einen Fehler aus, wenn keine GPU vorhanden ist.

<!--en:359a59a9-->
Aufbauend auf `num_gpus` definieren wir zwei Hilfsfunktionen. Sie wählen eine GPU, sofern sie verfügbar ist, und andernfalls die CPU. So läuft derselbe Notebook-Code unverändert auf Laptops und GPU-Servern.

<!--en:a3d4d8c0-->
Andere Beschleuniger wie die GPUs von Apple Silicon (`mps`) und die TPUs von Google verwenden eigene Gerätekennungen. Die folgenden Regeln für die Gerätezuordnung gelten auch für sie. In diesem Buch konzentrieren wir uns auf CPUs und CUDA-GPUs.

<!--en:0502bb7b-->
### Arrays auf Geräten

<!--en:7783326f-->
Das Attribut `device` gibt für jeden Tensor an, auf welchem Gerät er liegt:

<!--en:c5512ceb-->
MXNet legt Tensoren standardmäßig auf der CPU an, sofern wir kein Gerät über `device=` angeben.

<!--en:011cba51-->
PyTorch legt Tensoren standardmäßig auf der CPU an, sofern wir kein Gerät über `device=` angeben.

<!--en:0197a876-->
TensorFlow legt Tensoren standardmäßig auf der ersten GPU an, sofern eine verfügbar ist. Mit `with tf.device(...):` lässt sich das Gerät explizit festlegen.

<!--en:cfd40027-->
JAX legt Arrays standardmäßig auf der ersten GPU an, sofern eine verfügbar ist. Mit `jax.device_put` lässt sich das Gerät explizit festlegen.

<!--en:2b5ccf37-->
Wenn wir den Speicher direkt auf dem Zielgerät zuweisen, entfällt eine zusätzliche Kopie von der CPU zur GPU. Wir legen `X` auf der ersten GPU und `Y` auf der zweiten GPU ab und versuchen, beide zu addieren. Fehlt eine der GPUs, verwenden wir an ihrer Stelle die CPU:

<!--en:d9fba4f3-->
Beim Addieren von Tensoren auf unterschiedlichen Geräten tritt sofort ein Fehler auf. So kommt es nicht versehentlich zu Übertragungen zwischen Geräten.

<!--en:fb299e23-->
Außerhalb eines expliziten Geltungsbereichs für die Gerätezuordnung kopiert TensorFlow Operanden von anderen Geräten ohne Meldung auf das Standardgerät. Das lässt sich mit `tf.config.experimental.set_device_policy('explicit')` unterbinden.

<!--en:64b4456c-->
### Kopieren zwischen Geräten

<!--en:0e64e58e-->
Tensoren zwischen Geräten zu übertragen dauert sehr viel länger, als mit ihnen zu rechnen. Den Unterschied messen wir in :numref:`subsec_async_dispatch`. Würde das Framework Operanden von anderen Geräten stillschweigend kopieren, blieben gravierende Ursachen für lange Laufzeiten verborgen. Deshalb kopieren wir `X` explizit auf das Gerät von `Y` (:numref:`fig_copyto`), bevor wir beide addieren.

<!--en:10c2249b-->
![X liegt auf GPU 0 und Y auf GPU 1. Mit der Kopie Z von X auf GPU 1 (gestrichelt) lässt sich Y + Z vollständig auf GPU 1 berechnen.](../img/bg-copyto.svg)

<!--en:0d398bd5-->
Jetzt berechnet die zweite GPU die Summe. Das Ergebnis liegt dort neben den beiden Operanden. Für die Praxis gilt daher folgende Regel. Übertragen Sie die Daten einmal zu Beginn der Berechnung und führen Sie alle Operationen der inneren Schleife auf demselben Gerät aus. Einen Tensor auf das Gerät zu verschieben, auf dem er schon liegt, bewirkt praktisch nichts.

<!--en:4481e7eb-->
Die Methode `to_device` gibt das Array selbst zurück, wenn es bereits auf dem Zielgerät liegt. `copyto` erzeugt immer eine Kopie.

<!--en:ceaed4ec-->
### Modelle auf einem Gerät

<!--en:e9a6ff47-->
Der Vorwärtsdurchlauf (engl. forward pass) verknüpft die Eingaben mit den Modellparametern. Deshalb muss das ganze Modell auf demselben Gerät liegen wie seine Minibatches. Dank des Modulbaums aus :numref:`sec_model_construction` kann jedes Framework alle Parameter mit einem einzigen Aufruf verschieben oder initialisieren.

<!--en:caa51f7c-->
Liegen Eingaben und Parameter auf demselben Gerät, läuft auch der Vorwärtsdurchlauf dort und liefert seine Ausgabe auf diesem Gerät.

<!--en:2d2aba5b-->
Beachten Sie zwei Regeln für die Gerätezuordnung. Verschieben Sie das Modell auf das Zielgerät oder initialisieren Sie es dort, *bevor* Sie den Optimierer erzeugen. So liegen auch die Momente des Optimierers auf diesem Gerät. Legen Sie neue Tensoren innerhalb von `forward` auf `X.device` an, damit sie nicht standardmäßig auf der CPU landen.

<!--en:e4d81c1f-->
## Asynchrone Ausführung

<!--en:30ce4cc6-->
### Die Befehlswarteschlange

<!--en:222b5f47-->
Wenn Python eine GPU-Operation aufruft, führt die GPU sie asynchron aus. Python reiht jeden Kernelstart in die Befehlswarteschlange (engl. command queue) des Geräts ein und kehrt sofort zurück. Während die GPU die Befehle der Reihe nach abarbeitet, kann die CPU schon die nächsten vorbereiten. Diese Pipeline verbirgt den Zeitaufwand von Python, hat aber eine wichtige Folge für die Zeitmessung. Die Laufzeit einer Python-Schleife misst das *Einreihen* der Befehle, nicht ihre *Ausführung*. Wir reihen 32 Produkte von $4096 \times 4096$-Matrizen ein und messen zweimal die verstrichene Zeit. Die erste Messung endet, sobald Python alle Aufträge eingereiht hat, die zweite, sobald das Gerät alle Produkte berechnet hat.

<!--en:01664299-->
MXNet stößt Berechnungen auf CPU und GPU asynchron an. `npx.waitall()` blockiert, bis die Berechnung abgeschlossen ist.

<!--en:91d24936-->
JAX stößt Berechnungen auf CPU und GPU asynchron an. `B.block_until_ready()` blockiert, bis die Berechnung abgeschlossen ist.

<!--en:16430a33-->
GPU-Kernel laufen asynchron, CPU-Operationen synchron. `torch.cuda.synchronize()` wartet, bis die GPU alle eingereihten Aufträge abgearbeitet hat.

<!--en:cf43f427-->
GPU-Kernel laufen asynchron, CPU-Operationen synchron. `tf.test.experimental.sync_devices()` wartet, bis die GPU alle eingereihten Aufträge abgearbeitet hat.

<!--en:2525d7ad-->
Python braucht für das Einreihen aller 32 Matrizenmultiplikationen nur einen winzigen Bruchteil der tatsächlichen Rechenzeit. Wann muss die CPU warten? Nur wenn Python einen konkreten Wert auf dem Host benötigt. Das geschieht bei der Ausgabe eines Tensors, beim Umwandeln eines Skalars in eine Python-Gleitkommazahl, bei der Konvertierung nach NumPy oder wenn ein Tensor über eine Verzweigung in einer `if`-Anweisung entscheidet. An jedem solchen *Synchronisierungspunkt* wartet Python, bis die GPU alle Befehle bis zu diesem Wert abgearbeitet hat. Danach steht die GPU still, bis Python wieder Arbeit liefert (:numref:`fig_bg_async_queue`). Wer den Verlust nach jedem Minibatch als Python-Zahl protokolliert, erzwingt in jedem Schritt eine solche Pause. Deshalb sammelt unser `Trainer` die Verluste auf dem Gerät in einem `d2l.Averager` und wandelt sie nur einmal pro Epoche in Python-Gleitkommazahlen um (:numref:`sec_oo-design`).

<!--en:87df6748-->
![Python reiht die Operationen op1 bis op4 ein und läuft weiter, während die GPU sie der Reihe nach ausführt. Das Auslesen des Verlusts als Python-Zahl ist ein Synchronisierungspunkt. Die CPU wartet, bis die GPU ihre Warteschlange abgearbeitet hat. Dann läuft Python weiter, und die GPU wartet auf neue Aufträge.](../img/bg-async-queue.svg)

<!--en:2dbb6705-->
Aus der asynchronen Ausführung ergeben sich vier Regeln für verlässliche GPU-Benchmarks. (1) Mit einigen Aufrufen zum Aufwärmen schließen wir einmalige Kosten für Speicherzuweisung, Einrichtung der Bibliotheken und JIT-Kompilierung aus. (2) Wir synchronisieren das Gerät unmittelbar *vor* dem Start und unmittelbar *vor* dem Ende der Zeitmessung. (3) Wir mitteln über mehrere Wiederholungen. (4) Mit `time.perf_counter` erreichen wir eine Auflösung von unter einer Millisekunde. (Die Beispielzelle oben enthielt eine Aufwärmphase und eine Synchronisierung, gab aber die Gesamtzeit für 32 Produkte an. Unsere Hilfsklasse `Benchmark` setzt alle vier Regeln für dieses und die folgenden Kapitel automatisch um.)

<!--en:cd2324f3-->
JAX wartet auf einzelne Arrays. Deshalb muss die vermessene Funktion ihr Ergebnis zurückgeben. `Benchmark` wartet auf den Rückgabewert des letzten Aufrufs, nicht auf das gesamte Gerät.

<!--en:0d29e549-->
Mit `Benchmark` prüfen wir gleich zwei frühere Aussagen. Wie viel schneller ist eine GPU bei großen Matrizenmultiplikationen als eine CPU? Wie lange dauert die Übertragung vom Host auf das Gerät im Vergleich zur Berechnung? Wir messen die Zeit für ein $4096 \times 4096$-Matrixprodukt auf der CPU (fp32) und auf der GPU mit tf32 (:numref:`sec_numerics`) sowie mit 16 Bit Genauigkeit. Außerdem messen wir die Kopierzeit für eine einzelne $4096 \times 4096$-Matrix vom Arbeitsspeicher der CPU auf die GPU.

<!--en:193954e5-->
MXNet verwendet für fp32-Matrixprodukte standardmäßig tf32. Da die Operation bf16 nicht unterstützt, berechnen wir das 16-Bit-Produkt hier in fp16. Es läuft auf denselben Tensor Cores mit derselben Geschwindigkeit.

<!--en:c1a51f33-->
TensorFlow verwendet für fp32-Matrixprodukte standardmäßig tf32.

<!--en:3e6cfe6c-->
Auf dieser GPU verwendet JAX mit der standardmäßigen Genauigkeit für Matrixprodukte tf32.

<!--en:ef8714f0-->
Eine moderne GPU berechnet das $4096 \times 4096$-Produkt in weniger als 2 ms. Die CPU benötigt etwa 100 ms bis 2 s. Je nach Anzahl der CPU-Kerne ergibt das eine Beschleunigung von $100\times$ bis $1000\times$. Mit 16 Bit Genauigkeit halbiert sich die GPU-Zeit noch einmal ungefähr. Damit erreichen die Tensor Cores den in :numref:`sec_numerics` angekündigten Geschwindigkeitsgewinn.

<!--en:0155e3f0-->
Beachten Sie die letzte Zeile. Allein das *Kopieren* der Matrix vom CPU-Speicher auf die GPU dauert drei- bis sechsmal so *lange* wie die anschließende Multiplikation. Dabei führt die GPU pro kopiertem Element 4096 Multiplikationen mit anschließender Addition aus. Beim Kopieren vom Host gelangen die Daten standardmäßig zuerst in *auslagerungsgesperrten* Host-Speicher (engl. pinned memory), auf den die DMA-Einheit der GPU direkt zugreifen kann. Legen wir die Batches vorab in solchem Speicher ab, entfällt diese zusätzliche Kopie, und Übertragung und Berechnung können sich zeitlich überlappen (:numref:`sec_hardware`).

<!--en:4a829697-->
## GPU-Speicher

<!--en:d885ec77-->
### Drei Sichten auf den Speicher

<!--en:dcae694e-->
Wie viel GPU-Speicher belegt ein Trainingsskript? Je nachdem, auf welcher Ebene wir nachfragen, erhalten wir drei verschiedene Zahlen. Speicher direkt vom CUDA-Treiber anzufordern kostet Zeit. Deshalb verwenden Frameworks intern einen *Speicherallokator mit Cache* (engl. caching allocator). Er fordert große Speicherblöcke vom Treiber an und teilt sie den Tensoren stückweise zu. Frei gewordene Blöcke behält er zur Wiederverwendung. Daraus ergeben sich drei ineinander verschachtelte Sichten auf den Speicher (:numref:`fig_bg_allocator`). Die innerste erfasst die Bytes, die aktuell vorhandene Tensoren belegen. Die nächste umfasst den größeren Speichervorrat, den der Allokator des Frameworks reserviert. Die äußerste erfasst den gesamten Speicher, den der CUDA-Treiber dem Prozess zugewiesen hat. Diesen Wert zeigt `nvidia-smi` an.

<!--en:4a027632-->
![Die drei Sichten auf den Speicher sind ineinander verschachtelt. Der vom Treiber zugewiesene Prozessspeicher, den nvidia-smi meldet, enthält die vom Allokator reservierten Blöcke. In diesen liegen die aktuell vorhandenen Arrays.](../img/bg-allocator.svg)

<!--en:440bcbe9-->
`npx.gpu_memory_info(0)` fragt beim CUDA-Treiber ab, wie viele Bytes nach `gpu().empty_cache()` auf dem gesamten Gerät frei und wie viele insgesamt vorhanden sind.

<!--en:bf510a4a-->
`torch.cuda.memory_allocated()` meldet die aktuell von Tensoren belegten Bytes. `torch.cuda.memory_reserved()` gibt die Größe des Speichervorrats an, den der Allokator mit Cache reserviert hat.

<!--en:57f94f2d-->
`tf.config.experimental.get_memory_info('GPU:0')` meldet unter `'current'` und `'peak'`, wie viele Bytes im GPU-Allokator von TensorFlow aktuell und maximal belegt sind.

<!--en:ae4b5222-->
`gpu().memory_stats()` meldet `'bytes_in_use'` und `'peak_bytes_in_use'` für den von JAX vorab reservierten GPU-Speicher.

<!--en:4f2a0a91-->
Wir legen einen float32-Tensor mit 1 GiB an und löschen ihn sofort wieder. Beobachten wir, wie die Zähler darauf reagieren:

<!--en:27321a35-->
MXNet meldet nur den vom Treiber zugewiesenen Speicher. Der Wert sinkt, sobald `empty_cache()` den Block zurückgibt.

<!--en:de4e0091-->
Wenn wir `big` löschen, sinkt der von Tensoren belegte Speicher um 1 GiB. Der Allokator des Frameworks behält den frei gewordenen Block von 1 GiB jedoch in seinem reservierten Vorrat. Deshalb ändert sich die Anzeige von `nvidia-smi` nicht. (Erzwingen Sie innerhalb einer Trainingsschleife nie die Rückgabe solcher Blöcke an den CUDA-Treiber. Sonst kosten neue Speicheranforderungen beim Treiber in jedem Schritt Zeit.) Die äußerste Sicht, die des Treibers, setzt die feste Speichergrenze. Der Treiber kann nur so viel RAM zuweisen, wie auf der Karte vorhanden ist. Um herauszufinden, *wo* unser Code Speicher belegt, betrachten wir die Zähler für aktuell belegte Bytes. Besonders wichtig ist der *Spitzenwert* während eines Aufrufs. Er entscheidet, ob ein Schritt gelingt oder mit einem Speicherfehler (engl. out of memory, OOM) abbricht.

<!--en:08924997-->
Mit `torch.cuda.reset_peak_memory_stats()` und `torch.cuda.max_memory_allocated()` messen wir für jeden Aufruf exakt, um wie viele Bytes die Tensorbelegung in der Spitze zunimmt:

<!--en:a0ad8633-->
XLA plant den Speicher für kompilierte Funktionen statisch. Deshalb gibt `compiled.memory_analysis().temp_size_in_bytes` an, wie viele Bytes für temporären Arbeitsspeicher und Aktivierungen zusätzlich zu den Ein- und Ausgaben nötig sind:

<!--en:d52c9932-->
In Python bietet der C++-Speicherallokator mit Cache von MXNet keine Zähler für die aktuelle oder maximale Tensorbelegung. Den Gerätespeicher können wir mit `nvidia-smi` oder mit den Profiling-Werkzeugen des Frameworks untersuchen.

<!--en:beaeb034-->
### Speicherbedarf beim Training

<!--en:a31fdc85-->
In :numref:`sec_parameters` haben wir den statischen Mindestspeicherbedarf für das Training mit Adam in fp32 hergeleitet. Er beträgt $16N$ Bytes für $N$ Parameter ($4N$ für Gewichte, $4N$ für Gradienten und $8N$ für Adam-Momente). Im Trainingsschritt kommt eine vierte Größe hinzu, die *Aktivierungen*. Die Backpropagation muss Zwischenergebnisse des Vorwärtsdurchlaufs so lange speichern, bis der Rückwärtsdurchlauf (engl. backward pass) sie verarbeitet. Anders als die statischen $16N$ wächst der Aktivierungsspeicher linear mit der Minibatchgröße und der Schichtbreite. Wir können den Bedarf vor der Messung berechnen. Betrachten wir ein MLP mit drei versteckten Schichten der Breite 4096 und 1024-dimensionalen Eingaben ($N \approx 3.8 \times 10^7$ Parameter, also $4N \approx 144$ MiB) bei einer Minibatchgröße von 4096. Jede der drei versteckten Schichten gibt einen $4096 \times 4096$-Tensor in fp32 mit 64 MiB aus. Zusammen belegen die Ausgaben 192 MiB, also schon ein Drittel mehr als die Gewichte selbst.

<!--en:95436ba6-->
Da MXNet in Python keinen Zähler für die aktuelle Tensorbelegung pro Schritt bereitstellt, enthält dieser Unterabschnitt keinen MXNet-Code. Die folgende Rechnung mit $16N$ plus Aktivierungsspeicher gilt unverändert.

<!--en:6a1af672-->
An den Messwerten lässt sich ein Trainingsschritt genau verfolgen. Zunächst liegen Gewichte ($4N$ Bytes) und Eingaben im Speicher. Im Vorwärtsdurchlauf kommen die gespeicherten Aktivierungen hinzu (rund 192 MiB, also mehr als die Gewichte). Der Rückwärtsdurchlauf verarbeitet die Aktivierungen und gibt sie frei, während die Gradienten entstehen ($4N$). Im ersten Optimierungsschritt reserviert Adam Speicher für seine beiden Momente ($8N$). :numref:`fig_bg_step-memory` zeigt diese vier Phasen ohne den 16 MiB großen Eingabebatch. Ab dem zweiten Schritt steigt und fällt die Speicherbelegung innerhalb jedes Schritts. Der Vorwärtsdurchlauf speichert die Aktivierungen zusätzlich zu den Gewichten und Adam-Momenten. Der Rückwärtsdurchlauf ersetzt sie durch Gradienten. Die Spitzenbelegung beträgt höchstens die statischen $16N$ zuzüglich der Aktivierungen. Gibt die Schleife die alten Gradienten vor dem nächsten Vorwärtsdurchlauf frei, sinkt die Spitze um bis zu $4N$. Wenn die Aktivierungen den verfügbaren Speicher sprengen, hilft es am schnellsten, die Minibatchgröße zu verringern. Dadurch sinkt der Aktivierungsspeicher, ohne dass wir die Architektur ändern müssen. Steigt die Belegung hingegen *von Schritt zu Schritt* stetig an, suchen Sie nach einem Speicherleck, durch das alte Berechnungsgraphen erhalten bleiben. Ein typischer Fehler ist es, Verlusttensoren an eine Python-Liste anzuhängen, ohne sie vorher vom Graphen zu lösen.

<!--en:0e0babbf-->
`optimizer.zero_grad()` gibt die Gradienten standardmäßig frei (`set_to_none=True`). Ruft die Schleife diese Methode vor dem Vorwärtsdurchlauf auf, belegt der statische Zustand nur $12N$ Bytes statt $16N$, solange die Aktivierungen im Speicher liegen.

<!--en:f6f79d8d-->
![Speicherbelegung nach jeder Phase des ersten Trainingsschritts für das MLP der Breite 4096 bei einer Minibatchgröße von 4096. Der Eingabebatch ist nicht mitgezählt. Im Vorwärtsdurchlauf kommen zu den Gewichten ($4N$) die drei gespeicherten Aktivierungen hinzu. Der Rückwärtsdurchlauf gibt die Aktivierungen frei und hinterlässt die Gradienten ($4N$). Im Optimierungsschritt kommen die beiden Adam-Momente hinzu ($8N$).](../img/bg-step-memory.svg)

<!--en:d3e5e3e9-->
### Aktivierungs-Checkpointing

<!--en:7e8d23b1-->
Auch über die Architektur können wir den Aktivierungsspeicher verringern. Die gewöhnliche Backpropagation speichert die Zwischenausgabe jeder Schicht, um sie im Rückwärtsdurchlauf ein einziges Mal zu lesen. Beim *Aktivierungs-Checkpointing* (auch *Gradienten-Checkpointing* oder *Rematerialisierung* genannt, :cite:`Chen.Xu.Zhang.ea.2016`) verwerfen wir im Vorwärtsdurchlauf die meisten Zwischenaktivierungen. Wir behalten nur die Eingaben an den Grenzen ausgewählter Segmente. Im Rückwärtsdurchlauf berechnen wir die inneren Aktivierungen jedes Segments bei Bedarf neu. (Trotz der ähnlichen Bezeichnung hat diese Technik im Arbeitsspeicher nichts mit den auf Datenträger geschriebenen Checkpoints aus :numref:`sec_read_write` zu tun.)

<!--en:2a9e7017-->
Teilen wir $K$ gleichartige Schichten in Segmente mit je $s$ Schichten auf, müssen wir nur die Eingaben an $K/s$ Segmentgrenzen und $s$ Aktivierungen im gerade bearbeiteten Segment speichern. Das ergibt etwa $K/s + s$ Tensoren statt $K$. Mit $s = \sqrt{K}$ sinkt der maximale Aktivierungsspeicher von $O(K)$ auf $O(\sqrt{K})$. Dafür braucht die Backpropagation pro Segment einen zusätzlichen Vorwärtsdurchlauf (:numref:`fig_bg_activation_checkpoint`). Eine lineare Schicht berechnet im Rückwärtsdurchlauf zwei Matrixprodukte, eines für den Eingabegradienten und eines für den Gewichtsgradienten. Im Vorwärtsdurchlauf berechnet sie nur eines. Der zusätzliche Vorwärtsdurchlauf erhöht den Rechenaufwand pro Schritt damit von rund 3 auf 4 Vorwärtsdurchläufe. Je nach Hardware und Zusammensetzung der Schichten ergibt das als grobe Obergrenze etwa 33% Mehraufwand.

<!--en:d5da1496-->
![(a) Die gewöhnliche Backpropagation speichert die Aktivierungen jeder Schicht. (b) Checkpointing speichert nur die Segmenteingaben (ausgefüllt) und berechnet die inneren Aktivierungen jedes Segments (offen) im Rückwärtsdurchlauf neu. Bei $K$ Schichten in Segmenten von je $\sqrt{K}$ Schichten sinkt die Zahl der gespeicherten Aktivierungen von $O(K)$ auf $O(\sqrt{K})$. Das kostet einen zusätzlichen Vorwärtsdurchlauf.](../img/bg-activation-checkpoint.svg)

<!--en:019550d8-->
Aktivierungs-Checkpointing eignet sich besonders für tiefe Netze aus vielen gleichartigen Blöcken, etwa unser Residual-MLP (:numref:`sec_model_construction`) und moderne Transformer. Wir definieren `ResidualBlock` mit Gewichten aus $\mathcal{N}(0, 0.01^2)$ und Bias-Termen von null. Die Hilfsfunktion `run_stack` fasst die Blöcke zu Segmenten zusammen und erlaubt wahlweise Checkpointing.

<!--en:b1c9c0f5-->
MXNet bietet in Python keine Schnittstelle für Checkpointing einzelner Segmente. Nur für hybridisierte Graphen gibt es den prozessweiten Schalter `MXNET_BACKWARD_DO_MIRROR=1`. Deshalb enthält dieser Unterabschnitt keinen MXNet-Code.

<!--en:b4655b18-->
`torch.utils.checkpoint.checkpoint(segment, X, use_reentrant=False)` verwirft die Zwischenaktivierungen in `segment`. Die Funktion sichert den Zustand des Zufallszahlengenerators und stellt ihn wieder her, damit das Dropout konsistent bleibt. Sie führt `segment` während `backward()` erneut aus.

<!--en:6fe66306-->
`tf.recompute_grad(segment)` umschließt `segment` so, dass `GradientTape` nur die Eingabe an der Segmentgrenze speichert. Im Rückwärtsdurchlauf berechnet es die inneren Aktivierungen neu.

<!--en:5130abfd-->
`jax.checkpoint` (oder `jax.remat`) transformiert eine reine Funktion, die wir mit `nnx.split` erhalten. Dadurch speichert `jax.grad` nur die Eingaben an den Segmentgrenzen und berechnet die inneren Aktivierungen bei der Backpropagation neu.

<!--en:461e5fa6-->
Prüfen wir zunächst die Korrektheit. Das erneute Berechnen der Aktivierungen darf am mathematischen Ergebnis nichts ändern. An acht Blöcken, die wir für das Checkpointing in zwei Segmente aufteilen, prüfen wir, ob alle Gradienten mit denen der gewöhnlichen Backpropagation übereinstimmen.

<!--en:665d079e-->
Die Korrektheit können wir auf jedem Gerät prüfen. Um den Zusammenhang zwischen Speicherbedarf und Rechenzeit zu messen, verwenden wir auf einer GPU ein tieferes Netz aus 16 Blöcken (Breite 1024, Batchgröße 8192) mit je vier Blöcken pro Segment. Wir erfassen den *Speicherbedarf pro Schritt* (maximaler Speicherbedarf zusätzlich zu Parametern und Eingaben) und messen die verstrichene Zeit mit `Benchmark`.

<!--en:646ba6e9-->
Wie bei allen Zeitmessungen mit TensorFlow kompilieren wir den Schritt mit `tf.function`.

<!--en:71f477ab-->
Als Speicherbedarf pro Schritt zählen wir den Speicher, den der kompilierte Schritt zusätzlich zu seinen Eingaben einplant. Dazu gehören temporäre Arrays, vor allem Aktivierungen, und die Ausgabe, also die Gradienten. `peak_bytes_in_use` eignet sich hier nicht, da JAX diesen Wert nicht zurücksetzen kann und frühere Zellen ihn bereits erhöht haben.

<!--en:4c312377-->
Ohne Checkpointing hält der Rückwärtsdurchlauf die Aktivierungen aller 16 Blöcke gleichzeitig im Speicher. Bei Segmenten aus je vier Blöcken genügen die Eingaben an vier Segmentgrenzen und ein aktives Segment. Der Speicherbedarf pro Schritt sinkt damit von rund 1.1 GiB auf 0.4–0.5 GiB, während die Rechenzeit um etwa ein Viertel steigt. Das liegt innerhalb unserer geschätzten Obergrenze von 33%. Wegen dieses günstigen Verhältnisses gehört Aktivierungs-Checkpointing beim Training großer Transformer und tiefer Modelle für die Bildverarbeitung zum Standard.

<!--en:235e276d-->
## Trainer mit automatischer Gerätezuordnung

<!--en:491c6993-->
Jetzt können wir `num_gpus` in `d2l.Trainer` nutzen. Um unsere Regeln umzusetzen, genügen zwei Schritte. Vor dem Training legen wir die Modellparameter auf dem Zielgerät ab. Zu Beginn jedes Trainingsschritts übertragen wir den jeweiligen Minibatch auf dieses Gerät.

<!--en:dd8b2c31-->
Mit `min(num_gpus, d2l.num_gpus())` ist `num_gpus` eine Obergrenze. Auf Rechnern ohne GPU bleibt `self.gpus` leer, und das Training läuft automatisch auf der CPU. Darauf beruht jeder Aufruf von `Trainer(num_gpus=1)` im weiteren Verlauf des Buchs.

<!--en:d2aef107-->
Wie viel schneller trainiert eine GPU ein kleines Modell tatsächlich? Unser $4096 \times 4096$-Matrixprodukt lief auf der GPU etwa $100\times$ so schnell oder schneller. Eine vollständige Trainingsschleife umfasst aber auch das Laden der Daten, das Anstoßen der Operationen durch Python und kleinere Schichten. Wir messen die Trainingszeit eines Fashion-MNIST-Klassifikators aus zwei Residualblöcken der Breite 256 (:numref:`subsec_activation_checkpointing`). Wir initialisieren die Gewichte aus $\mathcal{N}(0, 0.01^2)$ und die Bias-Terme mit null.

<!--en:e49206c6-->
Da der Abschnitt zum Checkpointing keinen MXNet-Code enthält, definiert die Zelle die Gluon-Variante des Residualblocks aus :numref:`sec_model_construction`. Die Methode `initialize` des Modells zieht alle Gewichte aus $\mathcal{N}(0, 0.01^2)$.

<!--en:f2de6b31-->
Wir vergleichen auf Fashion-MNIST die Laufzeit je einer Epoche nach der Aufwärmphase mit `num_gpus=0` und `num_gpus=1`:

<!--en:f6b3a4c8-->
Der gemessene Geschwindigkeitsgewinn zeigt, was in der Praxis zu erwarten ist. Bei diesem kleinen Fashion-MNIST-Netz läuft eine Epoche auf der GPU höchstens ein paarmal so schnell wie auf der CPU. Das liegt weit unter dem Faktor $100\times$ bei der Multiplikation von $4096 \times 4096$-Matrizen. Hier hat die größte Matrix nur die Form $784 \times 256$. Die GPU ist damit bei Weitem nicht ausgelastet. Die meiste Zeit kosten das Laden der Daten und die Kernelstarts. GPUs lohnen sich, sobald große Faltungen, Attention-Köpfe und breite lineare Schichten den Rechenaufwand bestimmen, wie wir ab dem nächsten Kapitel sehen.

<!--en:8d88d85b-->
Dieses kleine, mit SGD trainierte Modell passt bequem in den Speicher, und mehrere Gigabytes bleiben frei. Kehren wir zu unserem Modell mit 8 Milliarden Parametern zurück. Eine kleinere Batchgröße und Aktivierungs-Checkpointing senken den Aktivierungsspeicher. Der statische Mindestbedarf von 128 GB für fp32-Gewichte, Gradienten und Adam-Momente bleibt jedoch bestehen. Denken Sie daran, dass `Trainer` nicht vorab warnen kann, wenn der Speicher für einen Schritt nicht ausreicht. Deshalb bleibt es unerlässlich, den Speicherbedarf vorab zu überschlagen. Wie wir die Speichergrenze eines einzelnen Geräts durch Parallelität über mehrere GPUs hinweg überwinden, behandeln wir in :numref:`chap_performance`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:f3e5932f-->
Jeder Tensor und jeder Modellparameter liegt auf einem bestimmten Gerät. Jede Operation setzt voraus, dass alle ihre Operanden auf demselben Gerät liegen. Übertragungen zwischen Host und Gerät dauern viel länger als die Berechnung auf der GPU. Legen Sie deshalb das Modell einmal vorab auf dem Zielgerät ab, übertragen Sie die Batches zu Beginn jedes Schritts und führen Sie innere Schleifen auf dem Gerät aus. GPUs arbeiten ihre Befehlswarteschlange asynchron ab. Python wartet daher nur an Synchronisierungspunkten, etwa bei der Umwandlung eines Skalars in eine Python-Gleitkommazahl oder bei der Konvertierung nach NumPy. Verlässliche Benchmarks erfordern eine Aufwärmphase, explizite Synchronisierung und wiederholte Messungen. Der Trainingsspeicher setzt sich aus dem statischen Modell- und Optimiererzustand sowie den batchabhängigen Aktivierungen zusammen. Aktivierungs-Checkpointing senkt den Aktivierungsspeicher von $O(K)$ auf $O(\sqrt{K})$ und kostet ungefähr einen zusätzlichen Vorwärtsdurchlauf. Unser `Trainer` übernimmt die Gerätezuordnung für eine einzelne GPU und weicht automatisch auf die CPU aus, wenn keine GPU vorhanden ist.

<!--en:2294243a-->
## Übungen

<!--en:848dadc6-->
1. [code] **Spitzenbelegung des Speichers.** Betrachten Sie wie in :numref:`subsec_memory_during_training` ein mit Adam trainiertes vierschichtiges MLP mit den Breiten 1024, 4096, 4096, 4096 und 10 Ausgaben. Berechnen Sie nach dem dort beschriebenen Verfahren die erwartete Spitzenbelegung eines Trainingsschritts für die Batchgrößen 64, 256 und 1024. Berücksichtigen Sie Gewichte, Gradienten, Optimiererzustand und die für den Rückwärtsdurchlauf gespeicherten Aktivierungen. Messen Sie anschließend die Spitzenbelegung auf einer GPU. Wo weicht Ihre Vorhersage von der Messung ab, und was haben Sie in der Rechnung ausgelassen?

<!--en:517b9ee4-->
2. **Speicherberechnung beim Checkpointing.** Arbeiten Sie den Ablauf aus :numref:`fig_bg_activation_checkpoint` für sechzehn aufeinanderfolgende Blöcke vom Typ `ResidualBlock` auf Papier aus. Verwenden Sie die Breite 1024 und die Batchgröße 8192.
    1. Benennen Sie die Arrays innerhalb eines Blocks, die für den Rückwärtsdurchlauf erhalten bleiben müssen, und zählen Sie ihre Einträge.
    1. Berechnen Sie den maximalen Aktivierungsspeicher während des Rückwärtsdurchlaufs ohne Checkpointing sowie mit Segmenten aus einem, zwei, vier und acht Blöcken. Geben Sie für jeden Ablauf an, welcher Anteil der Vorwärtsberechnung sich wiederholt.
    1. Welche Segmentgröße würden Sie hier wählen? Wie verhält sie sich zur Faustregel, pro Segment etwa $\sqrt{K}$ Schichten vorzusehen (siehe :numref:`subsec_activation_checkpointing`)?

<!--en:8157a5e5-->
3. [code] **Asynchrones Anstoßen von Berechnungen.** ● Verwenden Sie für beide Teilaufgaben die Regeln zur Zeitmessung aus :numref:`subsec_async_dispatch`.
    1. Messen Sie für $n$ von 256 bis 8192 die Zeit für die Übertragung einer $n \times n$-Matrix von der CPU zur GPU und zurück. Messen Sie auch die Zeit für ein $n \times n$-Matrixprodukt auf dem Gerät. Bestimmen Sie das $n$, bei dem beide Zeiten gleich sind. Gibt es in diesem Bereich keinen Schnittpunkt, extrapolieren Sie aus dem Wachstum der beiden Zeiten mit $n$. Sagen Sie den Schnittpunkt anhand der veröffentlichten Bandbreite vom Host zur GPU und des Durchsatzes Ihrer GPU für Matrixprodukte vorher. Erklären Sie mögliche Abweichungen zwischen Vorhersage und Messung.
    1. Falls Sie zwei GPUs haben, messen Sie zunächst auf einer GPU die Zeit für 1000 Produkte zweier $4096 \times 4096$-Matrizen. Stoßen Sie dann aus derselben Schleife je 500 Produkte auf zwei GPUs an. Messen Sie den Geschwindigkeitsgewinn. Erklären Sie, wie Python durch das asynchrone Anstoßen beide Geräte mit Arbeit versorgt, und benennen Sie Ursachen für Abweichungen von der linearen Skalierung. Sichern Sie das Experiment mit `num_gpus() >= 2` ab.

<!--en:1fe84b97-->
4. [code] **Synchronisierungspunkte.** ● Betrachten Sie das Training von `ResMLPClassifier` aus :numref:`subsec_device_aware_trainer` auf einer GPU.
    1. Listen Sie alle Synchronisierungspunkte eines Trainingsschritts auf. Lesen Sie dazu `Trainer.fit_epoch`, `Trainer.prepare_batch` und `Averager` sowie den Modellcode. Entscheiden Sie für jeden Punkt, ob er für die korrekte Funktion der Schleife unvermeidlich ist oder nur durch die Messwerterfassung entsteht und vermeidbar wäre. Begründen Sie jede Einordnung.
    1. Angenommen, die Schleife liest den Verlust nach jedem Schritt als Python-Gleitkommazahl aus. Sagen Sie anhand Ihrer Liste vorher, wie stark das Auslesen den standardmäßig konfigurierten `ResMLPClassifier` und den `ResMLPClassifier(num_hiddens=2048, num_blocks=8)` verlangsamt. Messen Sie anschließend für jedes Modell die Dauer einer Epoche mit und ohne Auslesen und erklären Sie die vier Messwerte.

    *Nach der PyTorch-Dokumentation zur [CUDA-Semantik](https://docs.pytorch.org/docs/stable/notes/cuda.html).*

<!--en:aa598418-->
[Diskussionen](https://d2l.discourse.group/t/36758)

<!--en:6941aca3-->
GPUs, Geräte und Speicherbudgets<br>**Gerätezuordnung · Engpässe beim Kopieren über PCIe · asynchrone Befehlswarteschlangen · 3 Sichten auf den Speicher · Aktivierungs-Checkpointing mit $O(\sqrt{K})$ Speicher**

<!--en:10b40618-->
::: {.slide title="Wie Gerätezuordnung und Speicherbudget die Modellgröße begrenzen"}

<!--en:c98a85b9-->
[Hardwaredurchsatz und HBM-Kapazität]{.kicker}

<!--en:b739f97d-->
- **Eigener GPU-Speicher (HBM):** Ein Modell mit 8 Mrd. Parametern passt zur Inferenz auf eine GPU mit `24 GB` (`bf16`, `16 GB` für Gewichte). Beim Adam-Training in `fp32` belegt schon der statische Zustand `128 GB`, bevor eine einzige Aktivierung hinzukommt.

<!--en:599a3e5f-->
- **Alle Operanden auf demselben Gerät:** Übertragungen zwischen Geräten über PCIe dauern $3\text{--}6\times$ so lange wie eine Matrizenmultiplikation. Alle Eingaben, Gewichte und Puffer müssen auf demselben Gerät liegen.

<!--en:3251a36f-->
- **Asynchrone Befehlswarteschlangen:** Python reiht GPU-Kernel ein und kehrt sofort zurück. Das Auslesen eines Skalars auf dem Host bei jedem Schritt hält die Pipeline auf.

<!--en:dc7f138f-->
::: {.slide title="Geräteabfrage mit automatischem Ausweichen auf die CPU"}

<!--en:6c00c008-->
[Geräteabstraktion]{.kicker}

<!--en:f836778e-->
Jeder Tensor liegt auf einem bestimmten Gerät (`cpu()`, `gpu(i)`). Die Hilfsfunktion `try_gpu(i)` wählt die GPU `i`, sofern sie vorhanden ist, und andernfalls die CPU. So laufen Notebooks überall unverändert:

<!--en:5e5af379-->
::: {.slide title="Fehler bei Operanden auf unterschiedlichen Geräten" except="tensorflow"}

<!--en:7586b4ac-->
[Keine stillschweigenden Kopien zwischen Geräten]{.kicker}

<!--en:88f388f3-->
- Liegt `X` auf GPU 0 und `Y` auf GPU 1 (oder der CPU), löst `X + Y` sofort einen Laufzeitfehler aus. So bremsen implizite PCIe-Kopien die Trainingsschleife nie aus:

<!--en:6a1d23ef-->
::: {.slide title="Gerätezuordnung und implizite Kopien in TensorFlow" only="tensorflow"}

<!--en:b596db73-->
[Regeln für die Gerätezuordnung in TensorFlow]{.kicker}

<!--en:de09343c-->
- Mit `set_device_policy('explicit')` weist TensorFlow Operanden auf unterschiedlichen Geräten zurück. Dadurch bremsen implizite PCIe-Übertragungen innere Schleifen nie aus:

<!--en:f58d36fd-->
::: {.slide title="Explizite Gerätewechsel durch einmaliges Kopieren vor der Berechnung"}

<!--en:5083407f-->
[Tensoren zwischen Geräten verschieben]{.kicker}

<!--en:0744ea13-->
Um Tensoren von unterschiedlichen Geräten zu verknüpfen, `X` explizit auf das Gerät von `Y` kopieren:

<!--en:9e91418b-->
- **Datenübertragung:** Daten einmal an der Schleifengrenze übertragen und alle Operationen der inneren Schleife auf demselben Gerät ausführen.

<!--en:c54e3b79-->
::: {.slide title="Der gesamte Modulbaum auf dem Zielgerät"}

<!--en:7c3225c8-->
[Parameter und Eingaben auf demselben Gerät]{.kicker}

<!--en:5e36afea-->
- Das Modell auf die Ziel-GPU verschieben oder dort initialisieren, **bevor** der Optimierer entsteht. So liegen auch die Adam-Momente auf der GPU. Neue Tensoren innerhalb von `forward` auf `X.device` anlegen:

<!--en:91e04f59-->
::: {.slide title="Asynchrone GPU-Ausführung mit der Befehlswarteschlange"}

<!--en:56be30d2-->
[Wie Python eine GPU ansteuert]{.kicker}

<!--en:3d59c9af-->
Python wartet bei `B = A @ A` nicht auf das Ergebnis. Es reiht einen Kernelstart in die **Befehlswarteschlange** der GPU ein und läuft weiter:

<!--en:8a89ed54-->
- **Einreihen und Rechnen:** Die CPU reiht 32 Produkte von $4096 \times 4096$-Matrizen innerhalb weniger Millisekunden ein. Die GPU braucht mehrere Dutzend Millisekunden, um die Warteschlange abzuarbeiten.

<!--en:9e3dcbdd-->
::: {.slide title="Synchronisierungspunkte halten die CPU-GPU-Pipeline auf"}

<!--en:1a664a18-->
[Unterbrechungen der Pipeline vermeiden]{.kicker}

<!--en:adce65fb-->
- **Wann die CPU warten muss:** Aufrufe von `loss.item()`, `float(loss)`, `print(tensor)`, `.numpy()` oder eine Verzweigung anhand eines Tensorwerts (`if loss < best:`) blockieren die CPU, bis die GPU alle eingereihten Aufträge erledigt hat. Danach wartet die GPU, bis Python neue Arbeit liefert.

<!--en:460adef1-->
- **Empfohlene Vorgehensweise:** Verluste der einzelnen Schritte in einem Tensor auf dem Gerät aufsummieren und erst am Ende der Epoche in eine Python-Gleitkommazahl umwandeln.

<!--en:e577977b-->
::: {.slide title="Vier Regeln für genaue GPU-Benchmarks"}

<!--en:7ad4674a-->
[Laufzeit asynchroner Kernel messen]{.kicker}

<!--en:0036085e-->
- **Aufwärmphase und zweimalige Synchronisierung:** Aufwärmiterationen ausführen, vor dem Start von `perf_counter()` synchronisieren und unmittelbar vor dem Ende der Zeitmessung erneut synchronisieren.

<!--en:25b23459-->
::: {.slide title="Eine Hilfsklasse für Benchmarks nach diesen Regeln"}

<!--en:7ad4674a-2-->
[Laufzeit asynchroner Kernel messen]{.kicker}

<!--en:6d757652-->
::: {.slide title="4096×4096-Matrixprodukt und PCIe-Kopie vom Host auf das Gerät"}

<!--en:5dd66761-->
[Rechendurchsatz und Speicherbandbreite]{.kicker}

<!--en:e99b77dd-->
Eine $4096 \times 4096$-Matrizenmultiplikation auf der CPU (`fp32`) und auf der GPU (`tf32` und 16 Bit) sowie eine Kopie von der CPU zur GPU vermessen. Die Zeiten zeigen einen grundlegenden Hardwareengpass beim Deep Learning:

<!--en:c2094a41-->
- **Die GPU rechnet $100\text{--}1000\times$ so schnell:** Tensor Cores schaffen $4096^3 \approx 6.9 \times 10^{10}$ Multiplikationen mit anschließender Addition in weniger als `2 ms` (mit 16 Bit Genauigkeit halbiert sich die Zeit noch einmal).

<!--en:ae723c7d-->
- **PCIe-Kopien dauern $3\text{--}6\times$ so lange wie das Matrixprodukt:** Die Übertragung von $4096^2 \times 4\text{ B} = 64\text{ MiB}$ über PCIe dauert länger als die anschließenden $4096$ Multiplikationen mit Addition pro Element auf der GPU.

<!--en:d676be16-->
::: {.slide title="Drei verschachtelte Sichten auf den GPU-Speicher"}

<!--en:f9eac136-->
[Aktuelle Tensoren, Speicherallokator mit Cache und CUDA-Treiber]{.kicker}

<!--en:617eb86b-->
Wie viel GPU-Speicher nutzt ein Prozess? Je nach abgefragter Ebene der Speicherhierarchie ergeben sich drei verschiedene Antworten:

<!--en:a1f202d1-->
- **Warum `del big` die Anzeige von `nvidia-smi` unverändert lässt:** Frameworks verwenden einen **Speicherallokator mit Cache**. Er hält frei gewordene GPU-Speicherblöcke zur sofortigen Wiederverwendung bereit. So entfallen langsame Aufrufe des CUDA-Treibers.

<!--en:8c4647a2-->
::: {.slide title="Speicherbelegung im Verlauf eines Trainingsschritts" except="mxnet"}

<!--en:5dd358e4-->
[Gewichte + Aktivierungen + Gradienten + Momente des Optimierers]{.kicker}

<!--en:91d9bccb-->
Die GPU-Speicherbelegung während eines Trainingsschritts zeigt, wann die vier Speicheranteile hinzukommen. Das Beispiel ist ein MLP mit 3 versteckten Schichten ($1024 \to 4096 \to 4096 \to 4096 \to 10$, Gewichte mit $4N \approx 144\text{ MiB}$, Batchgröße $B = 4096$):

<!--en:0c208b2e-->
- **Aktivierungen übertreffen die Gewichte ($+192\text{ MiB}$):** Die drei $4096 \times 4096$-Ausgaben der versteckten Schichten ($3 \times 64\text{ MiB} = 192\text{ MiB}$) liegen während `forward` im Speicher. `backward` gibt sie frei, wenn Gradienten ($+144\text{ MiB}$) entstehen. Hinzu kommen die Adam-Momente ($+288\text{ MiB}$).

<!--en:d4c2a4bf-->
::: {.slide title="Speicherbelegung im Verlauf eines Trainingsschritts" only="mxnet"}

<!--en:5dd358e4-2-->
[Gewichte + Aktivierungen + Gradienten + Momente des Optimierers]{.kicker}

<!--en:993006ea-->
- **Statischer Parameterzustand ($16N \approx 576\text{ MiB}$):** Gewichte ($4N = 144\text{ MiB}$), Gradienten ($4N = 144\text{ MiB}$) und Adam-Momente ($8N = 288\text{ MiB}$).

<!--en:da7fdbe1-->
- **Dynamische Aktivierungen im Vorwärtsdurchlauf ($\approx 192\text{ MiB}$):** Drei $4096 \times 4096$-Tensoren in `fp32` als Ausgaben der versteckten Schichten ($3 \times 64\text{ MiB}$). Die Spitzenbelegung liegt am Übergang zum Rückwärtsdurchlauf.

<!--en:e0e66f5a-->
::: {.slide title="Passt das Modell mit 8 Mrd. Parametern auf eine GPU mit 24 GB oder 80 GB?"}

<!--en:9a205237-->
[Statischer und dynamischer Speicherbedarf]{.kicker}

<!--en:ac96ef2b-->
- **Inferenz (`16 GB` in `bf16`, `32 GB` in `fp32`):** In `bf16` passt das Modell auf eine einzelne GPU mit `24 GB`. Für KV-Cache und Aktivierungen bleiben `8 GB`.

<!--en:12aa9087-->
- **Training (`38.3 GB` nur für das Training des Ausgabekopfs, `128 GB` für das gesamte Modell mit Adam in `fp32`):** Fine-Tuning nur des Ausgabekopfs passt auf eine GPU mit `80 GB`. Für das gesamte Modell mit Adam in `fp32` sind $\ge 2$ GPUs nötig.

<!--en:26cf7121-->
- **Speichermangel durch Aktivierungen ($O(K \cdot B \cdot T \cdot d)$):** Zuerst die Mikrobatchgröße $B$ halbieren, dann **Aktivierungs-Checkpointing** einschalten.

<!--en:fa5d83dc-->
::: {.slide title="Aktivierungs-Checkpointing mit O(√K) Speicher für 1 zusätzlichen Durchlauf"}

<!--en:21644f0c-->
[Weniger Aktivierungsspeicher durch mehr Rechenarbeit]{.kicker}

<!--en:4416c3b2-->
**Aktivierungs-Checkpointing** (Chen et al., 2016) speichert nur die Eingaben an Segmentgrenzen und berechnet den Rest während `backward` neu:

<!--en:d3aecc7b-->
- **Spitzenbedarf für Aktivierungen: $O(K) \to O(\sqrt{K})$:** Bei $K$ Schichten in Segmenten der Größe $s$ liegen $K/s$ Tensoren an Segmentgrenzen und $s$ innere Tensoren im Speicher (insgesamt $K/s + s$, minimal bei $s = \sqrt{K}$ mit $2\sqrt{K}$ Tensoren).

<!--en:4d911c2f-->
- **Begrenzter Rechenmehraufwand:** Die Neuberechnung aller Segmente kostet insgesamt einen zusätzlichen Vorwärtsdurchlauf (rund $20\text{--}30\%$ mehr Zeit pro Schritt).

<!--en:c0bc9a8d-->
::: {.slide title="Aktivierungs-Checkpointing für einzelne Segmente" except="mxnet"}

<!--en:3577dafb-->
[Checkpointing für eine Folge von Residualblöcken]{.kicker}

<!--en:21797b75-->
- `ResidualBlock`-Module in Segmente mit `segment_size = 4` einteilen und auf jedes Segment die Checkpointing-Funktion des Frameworks anwenden. So ergeben sich exakte Gradienten bei einem Speicherbedarf von $O(\sqrt{K})$.

<!--en:ecff7885-->
::: {.slide title="Gleiche Gradienten mit Checkpointing und gewöhnlicher Backpropagation" except="mxnet"}

<!--en:3577dafb-2-->
[Checkpointing für eine Folge von Residualblöcken]{.kicker}

<!--en:7b1a1e83-->
- Die Neuberechnung der Aktivierungen darf das Ergebnis nicht ändern. Bei acht Blöcken in zwei Segmenten stimmt jeder Gradient mit dem der gewöhnlichen Backpropagation überein:

<!--en:04292743-->
::: {.slide title="Messung: >50% weniger Speicher pro Schritt bei rund 25% mehr Zeit" except="mxnet"}

<!--en:f5963e5e-->
[16 Residualblöcke auf einer GPU]{.kicker}

<!--en:1b677c70-->
Bei 16 Residualblöcken ($d = 1024$, Batchgröße $B = 8192$, vier Blöcke pro Segment) senkt Aktivierungs-Checkpointing die Spitzenbelegung pro Schritt von $\sim 1.1\text{ GiB}$ auf $\sim 0.45\text{--}0.5\text{ GiB}$:

<!--en:deaf5da7-->
::: {.slide title="Begrenzung des Aktivierungsspeichers ohne Schnittstelle für einzelne Segmente" only="mxnet"}

<!--en:d9b13bc2-->
[Speicherbedarf der Aktivierungen begrenzen]{.kicker}

<!--en:74dac4f5-->
- **Lineare Abhängigkeit von der Batchgröße (akkumulierte Gradienten):** Eine Halbierung der physischen Minibatchgröße $B$ halbiert den Aktivierungsspeicher. Die Akkumulation über 2 halbe Batches erhält den effektiven Trainingsschritt.

<!--en:65be1c3b-->
- **Analytische $\sqrt{K}$-Regel für Checkpointing:** Die Aufteilung von $K = 16$ Blöcken in Segmente mit je $s = \sqrt{16} = 4$ Blöcken senkt die Zahl der gespeicherten Aktivierungstensoren von $16$ auf $16/4 + 4 = 8$.

<!--en:3638837e-->
::: {.slide title="Automatische Gerätezuordnung in d2l.Trainer"}

<!--en:935f0267-->
[Der Trainer mit automatischer Gerätezuordnung]{.kicker}

<!--en:f90988b2-->
- `d2l.Trainer` verschiebt das Modell auf `self.gpus[0]`, und zwar in `prepare_model`. Jeden Minibatch verschiebt er in `prepare_batch` dorthin. Ohne GPU nutzt er die CPU.

<!--en:ba315cbc-->
::: {.slide title="Laufzeit des Trainers mit automatischer Gerätezuordnung auf CPU und GPU"}

<!--en:935f0267-2-->
[Der Trainer mit automatischer Gerätezuordnung]{.kicker}

<!--en:6f6eea49-->
- **Warum kleine Modelle nur mäßig von der GPU profitieren:** Bei einem Fashion-MNIST-MLP der Breite 256 haben die Matrizen höchstens die Form $784 \times 256$. Die meiste Zeit kosten das Laden der Daten und die Latenz der Kernelstarts aus Python. Sobald breite Schichten, Faltungen und Attention den Rechenaufwand bestimmen, erreichen GPUs den Faktor $100\times$.

<!--en:9e5865e3-->
::: {.slide title="Verständnisfragen zu asynchronen Warteschlangen und Aktivierungsspeicher"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:7185e946-->
Wie verhält sich das System in diesen drei Fällen?

<!--en:4e5555d5-->
1. Eine Trainingsschleife führt bei jedem Minibatch auf einer GPU `if loss < best_loss: best_loss = loss` aus. Warum mindert diese einzelne `if`-Anweisung die GPU-Auslastung, und wie lässt sie sich umschreiben?

<!--en:2588e01e-->
2. Ein tiefer Transformer besteht aus $K = 36$ gleichartigen Blöcken. Wie viele Aktivierungstensoren der Blockeingaben liegen während des Rückwärtsdurchlaufs maximal im Speicher, wenn Checkpointing entfällt ($s = 36$), jeder Block einen Checkpoint erhält ($s = 1$) oder die optimale Segmentgröße $s = \sqrt{36} = 6$ gilt?

<!--en:4a47e46c-->
3. Beim Löschen eines GPU-Tensors von `2 GiB` sinkt die vom Framework gemeldete Speicherbelegung um `2 GiB`. Die Anzeige von `nvidia-smi` ändert sich jedoch nicht. Warum?

<!--en:557b98b6-->
**Lösungen:**

<!--en:b707bb8d-->
1. **Die Auswertung eines Python-Werts vom Typ `bool` (`loss < best_loss`) erzwingt bei jedem Schritt eine Synchronisierung von CPU und GPU.** Dabei arbeitet die GPU ihre Befehlswarteschlange vollständig ab. Abhilfe schafft die Aktualisierung `best_loss = minimum(best_loss, loss)` auf dem Gerät. Den Wert nur einmal pro Epoche auslesen.

<!--en:98883d91-->
2. **Maximal gespeicherte Blockaktivierungen ($K/s + s$):** Ohne Checkpointing sind es $36$, bei $s = 1$ sind es $36/1 + 1 = 37$ und bei optimaler Größe $s = 6$ nur $36/6 + 6 = 12$ (um den Faktor $3\times$ weniger).

<!--en:cb00f757-->
3. **Der Speicherallokator mit Cache behält frei gewordene Blöcke in seinem reservierten Vorrat.** Er nutzt sie für künftige Tensoren, ohne sie an den CUDA-Treiber zurückzugeben (`nvidia-smi`).

<!--en:1cd6e238-->
::: {.slide title="Zusammenfassung zu GPUs, Geräten und Speicherbudgets"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:67063ea2-->
- **Gemeinsames Gerät und asynchrone Warteschlangen:** Modell vor dem Erzeugen des Optimierers auf die GPU verschieben, jeden Batch einmal zu Beginn des Schritts übertragen und Wartezeiten durch Synchronisierung mit dem Host bei jedem Schritt vermeiden.

<!--en:2f51346d-->
- **Benchmarks und Sichten auf den Speicher:** Vor GPU-Zeitmessungen immer aufwärmen und unmittelbar vor Beginn und Ende synchronisieren. Zwischen aktuell von Tensoren belegten Bytes, dem Speichervorrat des Allokators mit Cache und der Anzeige von `nvidia-smi` unterscheiden.

<!--en:99902994-->
- **Aktivierungs-Checkpointing:** Bei $K$ Schichten mit $s = \sqrt{K}$ Schichten pro Segment sinkt der Aktivierungsspeicher von $O(K)$ auf $O(\sqrt{K})$. Der zusätzliche Rechenaufwand beträgt $\sim 25\text{--}33\%$.
