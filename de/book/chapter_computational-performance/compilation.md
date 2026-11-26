<!--en:95e8cfb0-->
# Berechnungsgraphen und Kompilierung

<!--en:44e0f922-->
In :numref:`sec_perf_model` dauerte die Aktivierungskette `gelu_ish` mit acht Operationen acht- bis neunmal so lange wie eine einzelne Addition auf demselben Tensor. Jede Operation lief als eigener GPU-Kernel, las ihre Eingaben aus dem DRAM und schrieb ihre Ausgabe dorthin zurück (18 Array-Transfers statt 2). Bei einem kleinen Tensor ist dieselbe Kette aus einem zweiten Grund langsam. Python stößt jede Operation einzeln an, und der Host braucht für den Start jedes Kernels länger als die GPU für dessen Ausführung.

<!--en:b6b2f57c-->
Beide Kosten entstehen bei der *sofortigen Ausführung* (engl. eager execution). Dabei läuft jede Operation einzeln, sobald Python sie erreicht. Übergeben wir das gesamte Programm vor der Ausführung einem Compiler, können wir beide Kosten senken. Die *Operatorfusion* (engl. operation fusion) fasst Ketten elementweiser Operationen zu einem Kernel zusammen. Er liest die Eingabe einmal und schreibt die Ausgabe einmal. Das senkt die Kosten bei begrenzter Bandbreite. Die *Wiedergabe von Kernelstarts* (engl. kernel replay) zeichnet eine Folge von Kernelstarts auf und wiederholt sie mit einem einzigen Aufruf des Hosts. Damit entfällt der Aufwand für die einzelnen Starts. Ein großes, rechenbegrenztes Matrixprodukt allein profitiert von keinem der beiden Verfahren. Es lohnt sich jedoch, die elementweisen Operationen davor und danach zu kompilieren.

<!--en:5a3b2d23-->
## Grapherfassung

<!--en:22dadb52-->
### Berechnungsgraphen

<!--en:55d99e6e-->
Die Operationen eines neuronalen Netzes bilden einen gerichteten *Berechnungsgraphen* (engl. computation graph). Seine Knoten sind elementare Operationen, seine Kanten transportieren Tensoren (:numref:`fig_compute_graph`). Bei sofortiger Ausführung zeichnet Autograd (:numref:`sec_autograd`) während des Vorwärtsdurchlaufs die Operationen für den Rückwärtsdurchlauf auf. Für die Optimierung des Vorwärtsdurchlaufs selbst kommt diese Aufzeichnung zu spät. Bis sie vorliegt, hat jede Operation bereits die Zeit $h$ für den Start durch den Host benötigt und Daten aus dem GPU-Speicher gelesen und dorthin zurückgeschrieben. Um das Programm zu optimieren, brauchen wir die *Grapherfassung* (engl. graph capture). Der Berechnungsgraph muss vorliegen, bevor die Operationen laufen.

<!--en:27d8c413-->
![Berechnungsgraph eines zweischichtigen Netzes mit Verlustfunktion. Das automatische Differenzieren durchläuft ihn rückwärts. Liegt der gesamte Graph bereits vor dem Vorwärtsdurchlauf vor, kann ein Compiler ihn umformen.](../img/mdl-perf-compute-graph.svg)

<!--en:aa47bc8e-->
### Tracing

<!--en:7cb3ecc9-->
Moderne Frameworks erfassen Graphen durch *Tracing*. Dazu führen sie die Python-Funktion einmal mit symbolischen *Platzhaltern* (engl. placeholders) aus. Diese haben eine Form und einen Datentyp, aber keine konkreten Zahlenwerte (:numref:`fig_compile_pipelines`). Jede Tensoroperation auf einem Platzhalter ergibt einen Knoten im Graphen. Gewöhnlichen Python-Kontrollfluss, der nur von statischen Formen oder Schleifengrenzen abhängt, löst das Framework beim Tracing auf. Ein Compiler übersetzt den erfassten Graphen anschließend in fusionierte GPU-Kernel und speichert sie zwischen. Spätere Aufrufe mit passenden Eingabeformen und Datentypen verwenden sie erneut.

<!--en:1c591a15-->
![Grapherfassung. Beim ersten Aufruf zeichnet das Framework die Funktion anhand von Platzhaltern auf. Ein Compiler übersetzt den Graphen in Kernel, die spätere Aufrufe mit denselben Formen und Datentypen wiederverwenden. Eine Verzweigung anhand eines Tensorwerts lässt sich so nicht aufzeichnen. Eine neue Form oder ein neuer Datentyp kann eine neue Erfassung auslösen.](../img/mdl-perf-compile-pipelines.svg)

<!--en:8db0d260-->
In PyTorch fängt `torch.compile` über TorchDynamo den Python-Bytecode ab :cite:`Ansel.Yang.He.ea.2024` und zeichnet die Tensoroperationen als Graphen auf. AOTAutograd erfasst vorab den Graphen für den Rückwärtsdurchlauf. Inductor kompiliert beide Graphen zu fusionierten Triton- und C++-Kerneln. Dynamo fügt *Prüfbedingungen* (engl. guards) für Eingabeformen und Datentypen hinzu. Sind sie erfüllt, springen spätere Aufrufe direkt zu den kompilierten Kerneln.

<!--en:5d9858b6-->
In JAX zeichnet `jax.jit` die Funktion mit abstrakten Tracer-Objekten in einer typisierten Zwischendarstellung auf (*jaxpr*, einsehbar mit `jax.make_jaxpr`) :cite:`Frostig.Johnson.Leary.2018`. Der XLA-Compiler übersetzt sie in fusionierte GPU-Kernel und speichert diese nach Eingabeform und Datentyp zwischen.

<!--en:cca46009-->
### Wertabhängige Verzweigungen

<!--en:5be46261-->
Da Platzhalter keine Zahlenwerte enthalten, kann ein Tracer eine datenabhängige Python-Verzweigung wie `if x.sum() > 0:` nicht auswerten. Bei der Grapherfassung gibt es drei Möglichkeiten:

<!--en:ac20f079-->
1. Den Zweig **festschreiben**, den eine konkrete Beispieleingabe nimmt. Das ist schnell, liefert aber ohne Fehlermeldung falsche Ergebnisse, wenn spätere Eingaben den anderen Zweig nehmen müssten.

<!--en:dacbdbae-->
2. Die Erfassung mit einer Fehlermeldung **ablehnen**. Dann muss die Verzweigung als elementare Graphoperation formuliert werden, entweder als *Auswahl* (sie wertet beide Zweige aus und wählt einen aus) oder als *bedingte Operation*.

<!--en:d69e06ac-->
3. Durch eine *Graphunterbrechung* (engl. graph break) **auf Python zurückgreifen**. Der erste Graph endet an der Bedingung. Python wertet die Verzweigung anhand des tatsächlich berechneten Werts aus, dann geht die Ausführung in einem zweiten kompilierten Graphen weiter.

<!--en:a7fa1d29-->
Bei Verzweigungen anhand statischer Tensorformen (`if x.shape[0] > 4:`) steht der Zweig schon beim Tracing fest. Sie brauchen keine Sonderbehandlung. Sehen wir uns an, wie unser Framework mit einer datenabhängigen Verzweigung umgeht.

<!--en:a7adc669-->
Standardmäßig fügt `torch.compile` eine Graphunterbrechung ein. Graph 1 berechnet `x * 2` und `x.sum() > 0`. Dann übernimmt Python, wertet das `if` aus und startet Graph 2 für den gewählten Zweig. Mit `fullgraph=True` (oder mit `torch.export`) sind Graphunterbrechungen unzulässig. Schreiben wir die Verzweigung mit `torch.where(x.sum() > 0, x + 1, x - 1)` um, entsteht ein einziger Graph. Das ältere `torch.jit.trace` schreibt den Zweig seiner Beispieleingabe fest.

<!--en:d59ca368-->
`jax.jit` lehnt die Erfassung mit dem Fehler `TracerBoolConversionError` ab. Ersetzen wir das Python-`if` durch `jax.lax.select` (oder durch `jax.lax.cond`, wenn nur ein Zweig ausgewertet werden soll), erhalten wir ein einziges jaxpr.

<!--en:91208d38-->
### Formen und Neukompilierung

<!--en:c72c3a61-->
Compiler stimmen Kernel auf Tensorformen ab, etwa die Kachelgrößen, Schleifengrenzen und Speicheranordnung. Ändert sich die Eingabeform, etwa durch einen kürzeren letzten Minibatch einer Epoche oder durch variable Sequenzlängen, muss der Compiler erneut kompilieren, sofern er die Dimensionen nicht symbolisch verallgemeinert. In der Praxis füllen wir die Eingaben mit Padding auf einige wenige vorgegebene Formen auf oder markieren die variablen Achsen als dynamisch.

<!--en:f606d72a-->
Dynamo kompiliert beim ersten Aufruf ($n = 8$) und verwendet denselben Graphen beim zweiten Aufruf ($n = 8$) erneut. Bei $n = 16$ kompiliert es mit einer symbolischen Dimension (`SymInt` `s77`) neu. Diese deckt dann auch $n = 32$ ohne weitere Kompilierung ab.

<!--en:913152c4-->
`jax.jit` spezialisiert sich ausschließlich auf statische Formen. Die Python-Anweisung `print` läuft beim Tracing für die Längen 8, 16 und 32 (drei Aufzeichnungen bei vier Aufrufen). Der zweite Aufruf mit Länge 8 trifft auf den Eintrag im Cache.

<!--en:7fbec394-->
## Operatorfusion

<!--en:287b41dd-->
### Berechnung des Speicherverkehrs

<!--en:83aba589-->
Bei sofortiger Ausführung startet `gelu_ish(x)` acht elementweise Kernel. Sechs unäre Operationen lesen und schreiben je ein Array. Zwei binäre Operationen lesen je zwei Arrays und schreiben eines. Insgesamt werden $10 + 8 = 18$ Arrays von je $nb$ Byte übertragen. Ein fusionierter Kernel liest `x` einmal, hält alle sieben Zwischenwerte in Registern und schreibt die Ausgabe einmal (:numref:`fig_fusion`). Dadurch sinkt der Speicherverkehr um den Faktor

<!--en:fffde429-->
Selbst der fusionierte Kernel führt in fp32 nur etwa 12 FLOPs pro 8 Byte aus ($I \approx 1.5 \ll 82$ FLOP/Byte). Damit sind beide Varianten bandbreitenbegrenzt, und die Fusion sollte `gelu_ish` um etwa den Faktor $9\times$ beschleunigen.

<!--en:86dc3a23-->
![Speicherverkehr von `gelu_ish`. Bei sofortiger Ausführung lesen acht Kernel insgesamt zehn Arrays und schreiben acht. Der fusionierte Kernel liest die Eingabe einmal, hält die sieben Zwischenergebnisse in Registern und schreibt das Ergebnis einmal.](../img/mdl-perf-fusion.svg)

<!--en:d7d6149b-->
Beide Varianten übertragen 0.6–0.9 TB/s. Die kompilierte Variante ist um den Faktor $8\times$ bis $11\times$ schneller, nahe am aus der Bytezahl vorhergesagten Faktor $9\times$. Der gesamte Gewinn entsteht dadurch, dass 16 der 18 DRAM-Transfers entfallen.

<!--en:f853597c-->
Wie viel bringt die Fusion der Aktivierung in einem Transformer-Block? Betrachten wir ein GPT mit einer Milliarde Parametern, der Breite $d = 2048$ und einer versteckten FFN-Schicht der Breite $4d = 8192$ in bf16. Für $M$ Token überträgt die unfusionierte Aktivierung aus 8 Operationen $18 \cdot 4dM \cdot 2 = 144\,dM$ Byte. Die beiden FFN-Matrixprodukte vor und nach der Aktivierung benötigen $2 \cdot 2 \cdot M \cdot d \cdot 4d = 16\,d^2M$ FLOPs. Das Verhältnis der Laufzeiten ist unabhängig von der Tokenzahl $M$:

<!--en:5e925a0c-->
für $I^* = 164$ FLOP/Byte und $d = 2048$. Ohne Fusion benötigt die Aktivierung fast drei Viertel der Zeit der beiden Matrixprodukte, mit Fusion weniger als ein Zehntel.

<!--en:85242db0-->
### Fusionsmuster

<!--en:e4d3ed11-->
Compiler fusionieren regelmäßig zwei Arten von Operationen:

<!--en:df4e3d0d-->
* **Ketten elementweiser Operationen** (Aktivierungen, Additionen in Residualverbindungen, Dropout-Masken, Aktualisierungen durch den Optimierer). Jeder Thread berechnet das Element $i$ der Ausgabe in Registern.

<!--en:32cbb421-->
* **Zeilenweise Reduktionen** (Softmax-Funktion, LayerNorm, RMSNorm). Jede Zeile passt in den SRAM auf dem Chip. Der Kernel liest sie einmal, berechnet Reduktion und Normalisierung auf dem Chip und schreibt die Ausgabe einmal.

<!--en:30a0468f-->
Compiler erfinden keine neuen Algorithmen. Sie können weder ein rechenbegrenztes Matrixprodukt beschleunigen noch gekachelte Algorithmen wie FlashAttention entdecken (:numref:`sec_attention-at-scale`, :cite:`Dao.Fu.Ermon.ea.2022`). FlashAttention vermeidet es, die $T \times T$-Attention-Matrix im Speicher anzulegen. Dazu skaliert der von Hand in CUDA oder Triton geschriebene Algorithmus die Softmax-Funktion fortlaufend neu :cite:`Tillet.Kung.Cox.2019`.

<!--en:e0e5282d-->
## Kompilierung eines Trainingsschritts

<!--en:cdb85962-->
Kompilieren wir nun einen vollständigen Trainingsschritt mit Vorwärtsdurchlauf, quadratischem Verlust, Rückwärtsdurchlauf und SGD-Update. Wir verwenden ein MLP mit 4 Schichten der Breite 1024 und `gelu_ish`-Aktivierungen für einen Batch mit 512 Beispielen.

<!--en:108b545e-->
Schätzen wir die Dauer eines Schritts. Die vier Matrixprodukte im Vorwärtsdurchlauf und die sieben im Rückwärtsdurchlauf benötigen etwa $11 \times 2
\cdot 512 \cdot 1024^2 \approx 1.2 \times 10^{10}$ FLOPs (0.14 ms bei 82.6 TFLOP/s in tf32). Die erste Schicht braucht keinen Eingabegradienten. Beim Lesen und Aktualisieren der 16 MiB Parameterdaten werden rund 96 MiB übertragen (0.10 ms bei 1 TB/s). Insgesamt arbeitet das Gerät also **etwa 0.25 ms**. Bei sofortiger Ausführung fallen dagegen über 100 einzelne Operationen mit je $h \ge 10$ µs an (> 1 ms Start-Overhead auf dem Host). Dieser Overhead begrenzt die Geschwindigkeit des Schritts.

<!--en:6c0e6c27-->
Die Kompilierung kostet beim ersten Aufruf $t_\textrm{compile}$ und spart pro Schritt $t_\textrm{eager} - t_\textrm{compiled}$. Damit amortisiert sie sich nach

<!--en:39d3d2ee-->
Schritten.

<!--en:1ae65b56-->
In der Standardeinstellung beschleunigt `torch.compile` den Schritt um weniger als den Faktor $2\times$. Er dauert noch immer 5- bis 10-mal so lange wie die eigentliche Arbeit auf dem Gerät (0.25 ms). Zwischen Vorwärts- und Rückwärtsdurchlauf einerseits und Parameterupdate andererseits verursacht `torch.autograd.grad` eine Graphunterbrechung. Der von Inductor erzeugte Python-Code startet jeden Kernel weiterhin einzeln.

<!--en:c9f8115f-->
`jax.jit` kompiliert Vorwärtsdurchlauf, Rückwärtsdurchlauf und Update zu einem einzigen XLA-Programm. Ein Aufruf startet das gesamte Programm. Mit rund 0.25 ms entspricht die Laufzeit unserer Abschätzung. Das Programm läuft etwa um den Faktor $100\times$ schneller als JAX ohne JIT-Kompilierung.

<!--en:f58957fb-->
## Wiedergabe von Kernelstarts

<!--en:97c4a0df-->
Matrizenmultiplikationen lassen sich durch Fusion nicht zusammenfassen. Laufen die verbleibenden Kernel nur kurz, bestimmen die einzelnen Kernel-Starts durch den Host die Schrittdauer. Betrachten wir eine Folge von 60 linearen Schichten der Breite 256 ohne Bias mit $\tanh$-Aktivierungen für einen Batch mit 64 Beispielen (120 Operationen pro Aufruf). Jede Schicht benötigt nur $8.4 \times 10^6$ FLOPs und überträgt 384 KiB (< 0.5 µs nach der Roofline, tatsächlich etwa 1–3 µs auf dem Gerät unter Berücksichtigung der hardwarebedingten Mindestzeit für den Kernelstart). Bei sofortiger Ausführung fällt dagegen pro Operation $h$ an (insgesamt $120\,h$).

<!--en:a6caae63-->
NVIDIA-GPUs unterstützen *CUDA-Graphen*, mit denen die einzelnen Kernel-Starts durch den Host entfallen. Das Laufzeitsystem zeichnet die Folge von Kernelstarts samt festen Pufferadressen einmal auf. Danach gibt es die gesamte Folge mit einem einzigen Start durch den Host wieder. Da Pufferadressen und Formen in der Aufzeichnung feststehen, werden die Eingaben vor der Wiedergabe in feste Puffer kopiert. Die nächste Wiedergabe überschreibt die Ausgaben.

<!--en:4e6fc1af-->
![Zeitlicher Ablauf auf Host und Gerät für eine Folge kurzer Kernel. Bei einzelnen Starts wartet die GPU vor jedem Kernel auf den Host und bleibt zwischen den Kerneln untätig. Ein CUDA-Graph gibt die gesamte Folge nach einem einzigen Start wieder. So laufen die Kernel unmittelbar hintereinander.](../img/mdl-perf-kernel-replay.svg)

<!--en:80357d2a-->
:numref:`fig_kernel_replay` vergleicht die beiden Zeitabläufe. Die Kernel selbst benötigen in beiden Fällen gleich viel Zeit. Bei der Wiedergabe entfallen die Leerlaufzeiten dazwischen.

<!--en:9111087b-->
In der Standardeinstellung ist `torch.compile` (`fused`) kaum schneller als die sofortige Ausführung. Inductor überlässt die Matrixprodukte cuBLAS und kann deshalb die $\tanh$-Aktivierungen nicht in sie integrieren. Python startet weiterhin alle Kernel einzeln, sodass die GPU nur während etwa 15% der Aufrufdauer arbeitet. Aktivieren wir CUDA-Graphen mit `mode='reduce-overhead'`, genügt ein Start für die Wiedergabe der gesamten Folge. Die Aufrufdauer sinkt um etwa den Faktor $7\times$ auf die rund 0.3 ms tatsächlicher GPU-Laufzeit.

<!--en:b71a7efd-->
Mit `mode='reduce-overhead'` sinkt auch die Dauer des MLP-Trainingsschritts aus :numref:`subsec_comp-wholestep` auf unter eine Millisekunde. Die GPU arbeitet für die vorhergesagten 0.25 ms. Die restliche Zeit verbringt Python an der Graphunterbrechung bei `autograd.grad`.

<!--en:460b5f69-->
`jax.jit` fusioniert jede $\tanh$-Aktivierung direkt mit dem folgenden Matrixprodukt und gibt die kompilierte Folge als XLA-Befehlspuffer (CUDA-Graph) wieder. Das ist um etwa den Faktor $30\times$ schneller als die sofortige Ausführung mit Einzelstarts. Die GPU arbeitet fast während der gesamten Aufrufdauer von rund 0.25 ms.

<!--en:70f26cee-->
## Grenzen der Kompilierung

<!--en:b55fbcfa-->
Kompilierung lohnt sich nur, wenn ein Programm mehr als $n^*$ Schritte läuft und viel Zeit für bandbreitenbegrenzte Operationsketten oder den Start-Overhead auf dem Host benötigt. In der Praxis sind drei Einschränkungen wichtig:

<!--en:37625f26-->
* **Verzweigungen anhand von Tensorwerten:** Das Protokollieren eines skalaren Verlusts, ein Test wie `if loss < 0.1:` oder eine datenabhängige Indizierung (`x[x > 0]`) erzwingen Graphunterbrechungen oder lösen Fehler aus. Lesen Sie Werte auf dem Host außerhalb kompilierter Funktionen aus. Verwenden Sie für Verzweigungen im Graphen `where` / `select` oder `cond`.

<!--en:d85daa43-->
* **Wechselnde Formen:** Variable Sequenzlängen oder ein kurzer letzter Minibatch einer Epoche können wiederholte Neukompilierungen auslösen. Lassen Sie den letzten Minibatch weg oder füllen Sie ihn mit Padding auf. Gruppieren Sie unterschiedlich lange Sequenzen in Längenklassen.

<!--en:797b1e6a-->
* **Fehlersuche:** Kompilierte Kernel laufen nicht Zeile für Zeile in Python. Prüfen und debuggen Sie den Code zuerst bei sofortiger Ausführung und kompilieren Sie ihn anschließend.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:df5d929c-->
Bei sofortiger Ausführung läuft jede Operation einzeln. Für jeden Kernel fällt die Latenz $h$ für den Start durch den Host an. Jede elementweise Operation liest aus dem DRAM und schreibt dorthin zurück. Beim Tracing entsteht vor der Ausführung anhand symbolischer Platzhalter ein Berechnungsgraph. Für datenabhängige Verzweigungen sind Auswahloperationen oder bedingte Graphoperationen nötig, sonst kommt es zu Graphunterbrechungen. Formänderungen lösen eine Neukompilierung aus, sofern die Dimensionen nicht symbolisch sind. Operatorfusion fasst Ketten elementweiser Operationen und zeilenweise Reduktionen auf dem Chip zu einzelnen Kerneln zusammen. Bei `gelu_ish` sinken Speicherverkehr und Laufzeit dadurch auf ein Neuntel. Bei kleinen Modellen und Batches beseitigt die Wiedergabe von CUDA-Graphen den Start-Overhead auf dem Host für die einzelnen Kernel.

<!--en:2294243a-->
## Übungen

<!--en:762f77d9-->
1. [code] **Wertabhängige Verzweigungen.** Die Funktion `f` aus :numref:`subsec_comp-capture` unterbricht den Graphen an ihrer `if`-Anweisung.
    1. Kompilieren Sie `f` mit `fullgraph=True` und lesen Sie die Fehlermeldung. Schreiben Sie die Verzweigung mit `torch.cond` um. Prüfen Sie mit `torch._dynamo.explain`, ob ein einziger Graph entsteht, und vergleichen Sie `torch.cond` mit `torch.where`.
    1. Messen Sie die Laufzeiten des kompilierten Originals und der kompilierten `torch.where`-Variante mit `d2l.Benchmark` für eine Eingabe mit 8 Elementen und eine Eingabe der Form $4000 \times 4000$. Erklären Sie den Unterschied für jede Größe.
    1. Beschreiben Sie zwei Zweige, bei denen `torch.where` (das beide Zweige auswertet) mehr kostet als eine Graphunterbrechung.

<!--en:0a7fd4f0-->
2. [code] **Formänderungen.** `torch.compile` prüft die Eingabeformen und kompiliert erneut, wenn eine Prüfbedingung verletzt ist (:numref:`subsec_comp-capture`).
    1. Kompilieren Sie `torch.tanh(x) * 2 + 1` mit `recording_backend` und rufen Sie die Funktion für Vektoren der Längen $8, 16, 32, \ldots, 8192$ auf. Zählen Sie die Kompilierungen und untersuchen Sie die Gründe mit `torch._logging.set_logs(recompiles=True)`.
    1. Wiederholen Sie den Versuch mit `dynamic=False` und dann mit `torch._dynamo.mark_dynamic(x, 0)` vor dem ersten Aufruf. Wie viele Kompilierungen gibt es jeweils, und wann ist `mark_dynamic` vorzuziehen?

<!--en:a478c077-->
1. [code] **Wertabhängige Verzweigungen.** Beim Tracing ersetzen symbolische Platzhalter die Argumentwerte (:numref:`subsec_comp-capture`).
    1. Schreiben Sie die Verzweigung in `f` einmal mit `jnp.where` und einmal mit `jax.lax.cond` um. Untersuchen Sie beide jaxprs mit `jax.make_jaxpr`. Welche Operationen führt jede Variante zur Laufzeit aus?
    1. Wenden Sie `jax.vmap` auf die `jax.lax.cond`-Variante für einen Batch von Vektoren an und untersuchen Sie ihr jaxpr. Was ist aus der bedingten Operation geworden und warum?
    1. Kompilieren Sie eine Funktion mit JIT, die anhand von `x.shape[0] > 4` verzweigt, und rufen Sie sie für Vektoren der Längen 3, 5 und 6 auf. Warum gelingt das, und wie oft findet Tracing statt?

<!--en:7fb6b456-->
2. [code] **Formänderungen.** Eine mit JIT kompilierte Funktion ist auf die Formen ihrer Argumente spezialisiert (:numref:`subsec_comp-capture`).
    1. Kompilieren Sie mit JIT eine Funktion, die `jnp.tanh(x).mean()` zurückgibt und beim Tracing einen Python-Zähler erhöht. Rufen Sie sie für Vektoren der Längen 100, 200 und 300 jeweils dreimal auf. Vergleichen Sie die Dauer des ersten Aufrufs mit der Dauer der Aufrufe, die den Cache nutzen.
    1. Füllen Sie jede Eingabe mit Padding auf die Länge 512 auf. Übergeben Sie eine Maske oder die Länge, sodass eine einzige Aufzeichnung für alle drei Längen den exakten Mittelwert ohne die Padding-Elemente berechnet.
    1. Vergleichen Sie zwei Möglichkeiten, die Länge zu übergeben. Verwenden Sie einmal ein beim Tracing erfasstes Array und markieren Sie die Länge einmal mit `static_argnums`.

<!--en:e7515bd3-->
3. [code] **Fusion nach einem Matrixprodukt.** ● Die Kompilierung lässt ein großes Matrixprodukt unverändert, fusioniert aber die nachfolgende Kette elementweiser Operationen. Betrachten Sie `gelu_ish(a @ a)` für eine $n \times n$-Matrix `a` in bf16 auf einer GPU mit Spitzenleistung $P$, Bandbreite $\beta$ und Knickpunkt $I^* = P/\beta$. Modellieren Sie das Produkt als rechenbegrenzt ($2n^3/P$ Sekunden) und die Kette als bandbreitenbegrenzt ($m = 18$ Arrays mit je $n^2$ Elementen zu je $b = 2$ Byte bei sofortiger Ausführung, 2 Arrays nach der Fusion).
    1. Leiten Sie die Schranke $S(n)$ für die Beschleunigung durch Kompilierung gegenüber der sofortigen Ausführung als Funktion von $n, m, b, I^*$ her. Werten Sie sie für die RTX 4090 ($I^* \approx 164$) bei $n = 1024, 4096, 16384$ aus und bestimmen Sie das $n$, für das $S(n) = 1.1$ gilt.
    1. Messen Sie die Laufzeit von `gelu_ish(a @ a)` bei sofortiger und bei kompilierter Ausführung für $n = 256, 512, \ldots, 8192$. Plotten Sie die gemessene Beschleunigung im Vergleich zu $S(n)$ und erklären Sie die Abweichungen.

<!--en:54b376df-->
4. **Fusion einer Reduktion.** ● Eine numerisch stabile zeilenweise Softmax-Funktion für eine $R \times d$-Matrix in fp32 lässt sich mit fünf sofort ausgeführten Operationen berechnen. Diese bestimmen das Zeilenmaximum, subtrahieren es, wenden die Exponentialfunktion an, bilden die Zeilensumme und dividieren durch sie.
    1. Zählen Sie die DRAM-Lese- und Schreibzugriffe für fünf einzeln ausgeführte Kernel und für einen einzigen fusionierten Kernel, der jede Zeile im SRAM hält. Berücksichtigen Sie die zu $Rd$ proportionalen Terme und geben Sie den Beschleunigungsfaktor bei begrenzter Bandbreite an.
    1. Bestimmen Sie für eine Zeile, die nicht in den SRAM passt, die Mindestzahl der Durchläufe für zwei Verfahren. Ein fusionierter Kernel berechnet Maximum und Summe in getrennten Durchläufen. Eine Online-Reduktion führt $m_j = \max(m_{j-1}, x_j)$ und $s_j = s_{j-1} e^{m_{j-1} - m_j} + e^{x_j - m_j}$ in einem einzigen Durchlauf mit. Prüfen Sie, dass $s_d = \sum_{j=1}^d e^{x_j - m_d}$ gilt.

<!--en:1472d075-->
5. [code] **Kompilierung des Optimierers.** ● Schreiben Sie den Schritt aus :numref:`subsec_comp-wholestep` mit `torch.optim.Adam` um (`zero_grad`, `backward`, `step`).
    1. Untersuchen Sie den Schritt mit `torch._dynamo.explain`. Wie viele Graphen erfasst das Verfahren und wo treten Unterbrechungen auf?
    1. Zählen Sie, wie viele Arrays von der Größe des Parameterbestands ein SGD-Update und ein Adam-Update jeweils mindestens lesen und schreiben (:numref:`sec_adam`). Messen Sie für beide Optimierer die Schrittdauer bei sofortiger und kompilierter Ausführung. Erklären Sie, warum die Kompilierung des Updates bei einem mehr bringt als beim anderen.

<!--en:8e057904-->
5. [code] **Konstanten beim Tracing.** Der Python-Rumpf einer mit JIT kompilierten Funktion läuft nur beim Tracing (:numref:`subsec_comp-capture`).
    1. Kompilieren Sie mit JIT eine Funktion, die `x + np.random.randn()` zurückgibt. Rufen Sie sie zweimal auf, untersuchen Sie ihr jaxpr und erklären Sie die beiden Ausgaben.
    1. Angenommen, eine mit JIT kompilierte Dropout-Funktion liest einen PRNG-Schlüssel aus einer globalen Python-Variablen. Erklären Sie, was in den folgenden Schritten passiert, wenn die globale Variable einen neuen Wert erhält. Schreiben Sie die Funktion so um, dass jeder Aufruf eine neue Maske zieht.

<!--en:56479ab6-->
6. [code] **Wann der Start-Overhead überwiegt.** ● Untersuchen Sie an den 60 Schichten aus :numref:`subsec_comp-overhead`, ab wann die Wiedergabe von Kernelstarts keine Rolle mehr spielt.
    1. Messen Sie die Laufzeit bei sofortiger Ausführung und bei Wiedergabe für Tiefen $L \in [10, 200]$ und Breite 256. Passen Sie $t(L) = c + sL$ an die Messwerte an und vergleichen Sie die Steigungen.
    1. Leiten Sie aus $\max(\textrm{FLOPs}/P, \textrm{bytes}/\beta) = h$ ab, bei welcher Breite die Ausführung des Matrixprodukts einer Schicht so lange dauert wie sein Start. Messen Sie dann die Laufzeiten bei sofortiger Ausführung und bei Wiedergabe für Tiefe 60 und Breiten von 64 bis 4096. Vergleichen Sie die gemessene Übergangsbreite mit Ihrer Vorhersage.
    1. Erstellen Sie mit `d2l.kernel_times` und `d2l.show_profile` ein Laufzeitprofil für die sofortige Ausführung der Schichten. Vergleichen Sie den Zeitanteil, in dem das Gerät arbeitet, und die verstrichene Zeit pro Start mit dem Profil der kompilierten Variante.

<!--en:6c738e4b-->
[Diskussion](https://d2l.discourse.group/t/36789)

<!--en:291b8372-->
Compiler, Operatorfusion und Grapherfassung<br>**Grapherfassung und Tracing · wertabhängige Verzweigungen und dynamische Formen · Operatorfusion · Kompilierung ganzer Trainingsschritte · CUDA-Graphen**

<!--en:95a244b6-->
::: {.slide title="Warum eine Aktivierung mit acht Operationen 9× so viele Bytes wie nötig überträgt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:51b17eea-->
- **Überflüssige DRAM-Zugriffe:** Bei sofortiger Ausführung startet `gelu_ish(x)` für seine 8 Operationen 8 separate Kernel. Sie lesen 10 Arrays und schreiben 8 ($18 \times 61\text{ MiB} = 1.07\text{ GiB}$ gegenüber mindestens $122\text{ MiB}$).

<!--en:325b12e9-->
- **Kernel-Start durch den Host:** Bei kleinen Tensoren fällt in Python die Startlatenz $h$ ($10\text{--}100\,\mu\text{s}$) für jeden der 8 Kernel an.

<!--en:978513f3-->
- **Abhilfe durch den Compiler:** Operatorfusion hält Zwischenergebnisse in Registern ($9\times$ weniger DRAM-Verkehr). Bei der Wiedergabe eines CUDA-Graphen genügt 1 Aufruf des Hosts.

<!--en:e4292703-->
::: {.slide title="Autograd bei sofortiger Ausführung und Grapherfassung vor der Ausführung"}

<!--en:efcc5bd7-->
[Berechnungsgraphen]{.kicker}

<!--en:759545f6-->
- **Warum Autograd bei sofortiger Ausführung zu spät kommt:** Die Aufzeichnung für den Rückwärtsdurchlauf entsteht dynamisch *während* des Vorwärtsdurchlaufs. Bis sie vorliegt, sind für jede Vorwärtsoperation bereits die Startlatenz $h$ auf dem Host und der Schreibzugriff für die Ausgabe im GPU-DRAM angefallen.

<!--en:7cf22f1d-->
- **Grapherfassung im Voraus:** Um `add` mit `relu` zu fusionieren oder die Wiederverwendung von Speicher zu planen, muss der Compiler die Graphen für den Vorwärts- und Rückwärtsdurchlauf *vor* der Ausführung der Kernel erfassen.

<!--en:f7a1941e-->
::: {.slide title="Symbolisches Tracing und Kompilierungspipelines" only="pytorch"}

<!--en:97d34cad-->
[Tracing mit Platzhaltern]{.kicker}

<!--en:edbaa7ac-->
- **So funktioniert Tracing:** Beim ersten Aufruf führt das Framework die Python-Funktion mit symbolischen **Platzhaltern** (Tracern) aus. Sie enthalten `shape` und `dtype`, aber keine Zahlenwerte. Das Framework zeichnet jede elementare Tensoroperation im Graphen auf und rollt statische Python-Schleifen aus.

<!--en:06427ed4-->
- **Übersetzung und Zwischenspeicherung (`torch.compile`):** TorchDynamo fängt Bytecode ab, AOTAutograd zeichnet den Rückwärtsdurchlauf auf und Inductor erzeugt fusionierte Triton- und C++-Kernel mit Prüfbedingungen für Form und Datentyp.

<!--en:1629898a-->
::: {.slide title="Symbolisches Tracing und Kompilierungspipelines" only="jax"}

<!--en:97d34cad-2-->
[Tracing mit Platzhaltern]{.kicker}

<!--en:edbaa7ac-2-->
- **So funktioniert Tracing:** Beim ersten Aufruf führt das Framework die Python-Funktion mit symbolischen **Platzhaltern** (Tracern) aus. Sie enthalten `shape` und `dtype`, aber keine Zahlenwerte. Das Framework zeichnet jede elementare Tensoroperation im Graphen auf und rollt statische Python-Schleifen aus.

<!--en:813d698d-->
- **Übersetzung und Zwischenspeicherung (`jax.jit`):** `jax.jit` zeichnet ein typisiertes `jaxpr` auf. XLA übersetzt es in fusionierte GPU-Kernel und speichert sie nach Eingabeform und Datentyp zwischen.

<!--en:2aaed29f-->
::: {.slide title="Drei Möglichkeiten für den Umgang mit datenabhängigen Verzweigungen beim Tracing" only="pytorch"}

<!--en:e9cfaa0d-->
[Kontrollfluss]{.kicker}

<!--en:e2c608a3-->
- **1. Festschreiben (`torch.jit.trace`):** Der Tracer zeichnet nur den Zweig einer Beispieleingabe auf. Das ist schnell, liefert aber **unbemerkt falsche Ergebnisse**, wenn neue Eingaben den anderen Zweig nehmen.

<!--en:099f78fb-->
- **2. Ablehnen (`fullgraph=True`):** Beim Tracing tritt ein Fehler auf. Ein einziges fusioniertes Programm erfordert elementare Graphoperationen (`torch.where` oder `torch.cond`).

<!--en:3e503c19-->
- **3. Auf Python zurückgreifen (Standard bei `torch.compile`):** Eine **Graphunterbrechung** synchronisiert die GPU mit Python, damit Python vor dem Start von Graph 2 das `if` auswerten kann.

<!--en:95539374-->
::: {.slide title="Drei Möglichkeiten für den Umgang mit datenabhängigen Verzweigungen beim Tracing" only="jax"}

<!--en:e9cfaa0d-2-->
[Kontrollfluss]{.kicker}

<!--en:f0340d49-->
- **1. Festschreiben:** Der Tracer zeichnet nur den Zweig einer Beispieleingabe auf. Das ist schnell, liefert aber **unbemerkt falsche Ergebnisse**, wenn neue Eingaben den anderen Zweig nehmen.

<!--en:aa74b647-->
- **2. Ablehnen (`jax.jit`):** Beim Tracing tritt ein Fehler auf. Ein einziges fusioniertes Programm erfordert elementare Graphoperationen (`jnp.where`/`lax.select` oder `lax.cond`).

<!--en:b94bcc63-->
- **3. Auf Python zurückgreifen:** Eine **Graphunterbrechung** synchronisiert die GPU mit Python, damit Python vor dem Start von Graph 2 das `if` auswerten kann.

<!--en:45eb0d3f-->
::: {.slide title="Graphunterbrechungen und torch.where in torch.compile" only="pytorch" layout="tall-output"}

<!--en:30d41c86-->
[Wertabhängige Verzweigungen in PyTorch]{.kicker}

<!--en:43f98baa-->
Bei einer Funktion mit einem datenabhängigen `if x.sum() > 0:` zeigt `torch._dynamo.explain`, wie TorchDynamo das Programm an der Verzweigung aufteilt:

<!--en:341b715a-->
- **Zwei Graphen, eine Unterbrechung:** Graph 1 berechnet `x * 2` und den booleschen Skalar `x.sum() > 0`. Python wartet auf das Gerät, liest den Wahrheitswert und startet Graph 2 (`x + 1` oder `x - 1`).

<!--en:893703c4-->
- **Umformulierung als einzelner Graph:** Der Ersatz von `if` durch `torch.where(x.sum() > 0, x + 1, x - 1)` (oder `torch.cond`) beseitigt die Synchronisierungsbarriere mit dem Host. Die Berechnung passt in einen einzigen Graphen (`fullgraph=True`).

<!--en:1bd2b4f7-->
::: {.slide title="TracerBoolConversionError und lax.select in jax.jit" only="jax"}

<!--en:98a6e086-->
[Wertabhängige Verzweigungen in JAX]{.kicker}

<!--en:9d846b3c-->
In JAX löst die Umwandlung des beim Tracing aufgezeichneten booleschen Arrays `x.sum() > 0` in einen Python-`bool` innerhalb von `jax.jit` sofort `TracerBoolConversionError` aus:

<!--en:e91d572f-->
- **`jax.lax.select` (oder `jnp.where`):** Beide einfachen elementweisen Zweige laufen in einem einzigen `jaxpr`. Das Gerät wählt die Ausgabe ohne Synchronisierung mit dem Host.

<!--en:75921dac-->
- **`jax.lax.cond`:** Kompiliert eine echte bedingte Operation, wenn nur der gewählte Zweig laufen soll (etwa um eine teure Berechnung zu überspringen).

<!--en:bf151a78-->
::: {.slide title="Prüfbedingungen für Formen und symbolische Dimensionen in Dynamo" only="pytorch"}

<!--en:38a5ae77-->
[Dynamische Formen]{.kicker}

<!--en:e01a3cba-->
Kompilierte Kernel stimmen Kachelgrößen und Schleifengrenzen auf Tensorformen ab. Ändert sich die Eingabelänge, scheitert die Prüfung und der Compiler kompiliert erneut:

<!--en:1d1ebe6a-->
- **Automatische symbolische Verallgemeinerung:** Dynamo kompiliert einmal für $n = 8$ und verwendet den gespeicherten Kernel beim zweiten Aufruf mit $n = 8$ erneut. Bei $n = 16$ kompiliert es mit einer symbolischen ganzzahligen Dimension (`SymInt` `s77`) neu. Diese deckt auch $n = 32$ ohne weitere Kompilierung ab.

<!--en:c7d6ada6-->
::: {.slide title="Spezialisierung auf statische Formen und Zwischenspeicherung in jax.jit" only="jax"}

<!--en:8b31df1d-->
[Statische Formen]{.kicker}

<!--en:5cbdc3cf-->
`jax.jit` speichert kompilierte XLA-Programme zwischen. Als Schlüssel dienen die genaue statische Form (`shape`) und der Datentyp (`dtype`) jedes Arguments:

<!--en:e59c6465-->
- **Seiteneffekte nur beim Tracing:** Die Python-Anweisung `print` läuft beim ersten Aufruf für jede neue Länge ($8, 16, 32$). Der zweite Aufruf mit Länge $8$ verwendet direkt das kompilierte Programm aus dem Cache, ohne Python erneut auszuführen.

<!--en:122ef6af-->
::: {.slide title="Sequenzen unterschiedlicher Länge ohne Neukompilierung"}

<!--en:a8f5cb57-->
[Längenklassen und Padding]{.kicker}

<!--en:2ca990ea-->
- **Variable Längen ohne Padding ($100, 128, 75$):** Jede neue Sequenzlänge erzwingt eine mehrsekündige Kompilierungspause und unterbricht so immer wieder die Trainingsschleife.

<!--en:c941a3ce-->
- **Längenklassen in Zweierpotenzen oder Vielfachen von 64:** Füllt Padding mit maskierten Token die Längen $100$ und $75$ auf die Klassenlänge $128$ auf, muss der Compiler **nur 1 Kernel** kompilieren und zwischenspeichern.

<!--en:17435d92-->
- **Letzter Batch einer Epoche:** Bei Kompilierung für statische Formen oder Wiedergabe von CUDA-Graphen stets `drop_last=True` übergeben (oder den letzten Batch mit Padding auffüllen).

<!--en:8e7c1fad-->
::: {.slide title="Operatorfusion spart Zugriffe auf den externen DRAM"}

<!--en:abb92aa8-->
[Operatorfusion]{.kicker}

<!--en:df199337-->
- **Zwischenwerte in Registern:** Ein fusionierter Kernel lädt jedes Element $x_i$ einmal aus dem DRAM, berechnet alle $8$ Operationen von `gelu_ish` in Registern auf dem Chip und schreibt das Endergebnis einmal in den DRAM:
  $$\textrm{Traffic Reduction} \;=\; \frac{\textrm{bytes moved (eager)}}{\textrm{bytes moved (fused)}} \;=\; \frac{(10\textrm{ reads} + 8\textrm{ writes})\, n b}{(1\textrm{ read} + 1\textrm{ write})\, n b} \;=\; 9\times$$

<!--en:ecfacf5c-->
::: {.slide title="Fusioniertes gelu_ish läuft wie aus der Bytezahl vorhergesagt 9× so schnell"}

<!--en:6b0b22da-->
[Messung der Fusion]{.kicker}

<!--en:7ec80de5-->
Auch nach der Fusion benötigt `gelu_ish` nur $\sim 12\text{ FLOPs}$ pro $8\text{ B}$ in `float32` ($I \approx 1.5 \ll 82\text{ FLOP/byte}$). Damit bleiben die sofort ausgeführte und die kompilierte Variante bandbreitenbegrenzt:

<!--en:9a5eef42-->
Die Übereinstimmung immer zuerst mit `allclose` prüfen, da die Fusion die Reihenfolge von Gleitkommaadditionen ändern kann. Beide Varianten übertragen $0.6\text{--}0.9\text{ TB/s}$. Der kompilierte Kernel ist **um den Faktor $8\text{--}11\times$ schneller**, wie das Verhältnis $9\times$ der Bytezahlen erwarten lässt.

<!--en:f020f291-->
::: {.slide title="Fusion der Aktivierung in einem Block eines GPT mit 1 Mrd. Parametern"}

<!--en:0f41375a-->
[Fusion im FFN eines Transformers]{.kicker}

<!--en:fb020f12-->
- **Unabhängig von der Tokenzahl $M$:** Für $M = 8{,}192$ Token ($d = 2048$, versteckte Breite $4d = 8192$) benötigen die beiden `bf16`-GEMMs des FFN insgesamt $\sim 3.3\text{ ms}$ bei $165\text{ TFLOP/s}$.

<!--en:fbf2ba05-->
- **Sofort ausgeführte und fusionierte Aktivierung:** Ohne Fusion überträgt `gelu_ish` mit 8 Operationen $2.25\text{ GiB}$ ($\sim 2.4\text{ ms}$ bzw. **$72\%$ der GEMM-Laufzeit**). Durch Fusion sinkt der Speicherverkehr um den Faktor $9\times$ auf $256\text{ MiB}$ ($\sim 0.27\text{ ms}$, $< 8\%$ der GEMM-Laufzeit).

<!--en:b0aecc63-->
::: {.slide title="Was Compiler fusionieren und was von Hand gekachelte Kernel leisten"}

<!--en:6f0bb91f-->
[Reichweite der Fusion]{.kicker}

<!--en:8fcb6ff5-->
- **Automatische Fusion:** Ketten elementweiser Operationen (Aktivierungen, Additionen in Residualverbindungen, AdamW) und zeilenweise Reduktionen im SRAM (RMSNorm, zeilenweise Softmax-Funktion) benötigen nur noch 1 Lese- und 1 Schreibzugriff.

<!--en:74389371-->
- **GEMMs aus Bibliotheken:** Dichte Matrixprodukte ($\mathbf{X}\mathbf{W}$) erreichen mit Kacheln von cuBLAS/CUTLASS bereits nahezu $P$. Optional lassen sich Bias und Aktivierung in die abschließenden Rechenschritte integrieren.

<!--en:54566e14-->
- **Spezielle gekachelte Algorithmen:** Um die $T \times T$-Attention-Matrix im DRAM zu vermeiden, skaliert FlashAttention die Softmax-Funktion fortlaufend über K/V-Kacheln hinweg neu.

<!--en:9b7a3c79-->
::: {.slide title="Zeitbedarf eines vollständigen Trainingsschritts für ein MLP mit 4 Schichten"}

<!--en:50434989-->
[Vorhersage für den gesamten Schritt]{.kicker}

<!--en:f84656f2-->
Ein vollständiger Trainingsschritt (Vorwärtsdurchlauf, MSE-Verlust, Rückwärtsdurchlauf und SGD-Update) für ein MLP mit 4 Schichten der Breite $1024$, `gelu_ish`-Aktivierungen und einem Batch der Größe $B = 512$ in `float32` (`tf32`-Tensor-Cores, $P = 82.6\text{ TFLOP/s}$):

<!--en:2211e57e-->
- **Untergrenze durch Rechenleistung ($0.14\text{ ms}$):** Vier Matrixprodukte im Vorwärtsdurchlauf und sieben im Rückwärtsdurchlauf (die erste Schicht benötigt keinen Eingabegradienten) erfordern $11 \times 2 \cdot 512 \cdot 1024^2 \approx 1.18 \times 10^{10}\text{ FLOPs}$. Das dauert $0.14\text{ ms}$ bei $82.6\text{ TFLOP/s}$.

<!--en:7c74e082-->
- **Untergrenze durch Bandbreite ($0.10\text{ ms}$):** Bei $16\text{ MiB}$ Parameterdaten und $2\text{ MiB}$ Aktivierungen übertragen $\sim 6$ Durchläufe insgesamt $\sim 96\text{ MiB}$. Das dauert $0.10\text{ ms}$ bei $1\text{ TB/s}$ (insgesamt $\sim \mathbf{0.25\text{ ms}}$ Arbeit auf dem Gerät).

<!--en:0aeeeb76-->
- **Untergrenze durch Einzelstarts ($> 1.0\text{ ms}$):** Die sofortige Ausführung startet $> 100$ kleine Operationen mit je $h \ge 10\,\mu\text{s}$ Startlatenz. Damit **begrenzen die Startkosten** die Geschwindigkeit des Schritts um den Faktor $5\text{--}10\times$.

<!--en:330c08a8-->
::: {.slide title="Kompilierung des Trainingsschritts mit Graphunterbrechungen und Kernelstarts" only="pytorch"}

<!--en:8600c8d3-->
[Kompilierung des gesamten Schritts]{.kicker}

<!--en:0741e018-->
Mit `torch.compile(train_step)` in der Standardeinstellung beschleunigt sich der Trainingsschritt des MLP mit 4 Schichten gegenüber der sofortigen Ausführung um weniger als den Faktor $2\times$. Er dauert weiterhin $5\text{--}10\times$ so lange wie die Mindestzeit von $0.25\text{ ms}$ auf dem Gerät:

<!--en:df1727e7-->
- **Warum der Start-Overhead bei `torch.compile` in der Standardeinstellung überwiegt:** Der Aufruf von `torch.autograd.grad` in `train_step` verursacht eine Graphunterbrechung (`graphs: 2`). Der Python-Code von Inductor startet jeden kompilierten Kernel weiterhin einzeln von der CPU aus.

<!--en:9d21f68c-->
::: {.slide title="Kompilierung des Trainingsschritts zu einem XLA-Programm" only="jax"}

<!--en:8600c8d3-2-->
[Kompilierung des gesamten Schritts]{.kicker}

<!--en:9533b79f-->
`jax.jit(train_step)` erfasst den Vorwärtsdurchlauf, `jax.value_and_grad` und das SGD-Update mit `jax.tree.map` in einem einzigen ausführbaren XLA-Programm. Ein Aufruf startet das gesamte Programm:

<!--en:d0322705-->
- **Die Abschätzung von $0.25\text{ ms}$ trifft zu:** Der kompilierte Schritt dauert $\sim 0.25\text{ ms}$. Er ist damit rund **um den Faktor $100\times$ schneller** als die sofortige Ausführung in JAX ohne JIT. Die Kompilierung amortisiert sich nach wenigen hundert Schritten.

<!--en:7a6b5f11-->
::: {.slide title="Schrittzahl n* bis zur Amortisation der Kompilierung"}

<!--en:1daffbcc-->
[Amortisation der Kompilierzeit]{.kicker}

<!--en:1e58277c-->
Grapherfassung, Optimierung und Codeerzeugung für die Kernel kosten beim ersten Schritt vorab die Zeit $t_{\textrm{compile}}$. Deshalb lohnt sich die Kompilierung erst nach $n^*$ Schritten:

<!--en:38972ab9-->
- **Rechenbeispiel:** Dauert die Kompilierung $t_{\textrm{compile}} = 30\text{ s}$ und spart sie $t_{\textrm{eager}} - t_{\textrm{compiled}} = 10\text{ ms/step}$, amortisiert sie sich bei $n^* = 3{,}000\text{ steps}$. Bei einem Training mit $100{,}000$ Schritten fällt das kaum ins Gewicht. Eine kurze Auswertung mit $200$ Schritten dauert insgesamt länger.

<!--en:8f28f083-->
- **Dauerhafter Kompilierungscache:** Compiler können kompilierte Binärprogramme auf der Festplatte zwischenspeichern. Bei späteren Prozessstarts sinkt $t_{\textrm{compile}}$ dadurch von mehreren zehn Sekunden auf einen Sekundenbruchteil.

<!--en:38d8e85d-->
::: {.slide title="Grenzen der Fusion bei schmalen Schichten durch Start-Overhead"}

<!--en:f9da750b-->
[Durch Start-Overhead begrenzte Ausführung]{.kicker}

<!--en:1bd5f11e-->
Operatorfusion kann aufeinanderfolgende Matrizenmultiplikationen nicht zusammenfassen. Als Beispiel dient ein Netz mit 60 linearen $256 \times 256$-Schichten und $\tanh$-Aktivierungen für einen Batch der Größe $B = 64$ ($120$ Operationen pro Vorwärtsdurchlauf):

<!--en:c155121c-->
- **Arbeit auf dem Gerät pro Schicht ($< 2\,\mu\text{s}$):** Jede Schicht benötigt $2 \cdot 64 \cdot 256^2 = 8.4 \times 10^6\text{ FLOPs}$ und überträgt $384\text{ KiB}$ ($< 0.5\,\mu\text{s}$ nach der Roofline bzw. $\sim 1\text{--}2\,\mu\text{s}$ unter Berücksichtigung der hardwarebedingten Mindestzeit für den Kernelstart).

<!--en:8579a314-->
- **Startkosten auf dem Host ($120\,h$):** Selbst für $60$ fusionierte Kernel kosten die Einzelstarts aus Python $60 \times 13\,\mu\text{s} \approx 0.8\text{ ms}$ auf der CPU. Dem stehen $\sim 0.15\text{ ms}$ Arbeit auf der GPU gegenüber. Die GPU bleibt während $> 80\%$ der Zeit untätig.

<!--en:3a052097-->
::: {.slide title="CUDA-Graphen einmal aufzeichnen und mit einem Host-Aufruf wiedergeben"}

<!--en:5f680546-->
[Wiedergabe von Kernelstarts]{.kicker}

<!--en:e0b06ad8-->
- **Erfassung:** Ein Aufwärmdurchlauf in einem eigenen Ausführungsstrom zeichnet jeden Kernel-Deskriptor, jede Gitterform und jeden genauen GPU-Speicherzeiger in einem statischen Graphen auf.

<!--en:857648b6-->
- **Wiedergabe (`cudaGraphLaunch`):** Ein CPU-Aufruf übergibt den gesamten Graphen mit über 60 Kerneln an den Hardware-Scheduler der GPU. Der Overhead $h$ für Python und Treiber pro Kernel entfällt.

<!--en:fc774bb0-->
- **Statische Vorgaben:** Formen und Pufferadressen stehen fest. Dynamische Formen und Synchronisierungsbarrieren mit dem Host (`item()`, `print`) sind im Graphen unzulässig.

<!--en:c400a1d0-->
::: {.slide title="7× schnellere Ausführung der 60 Schichten durch Wiedergabe eines CUDA-Graphen" only="pytorch"}

<!--en:cdf86c15-->
[Messung der Wiedergabe]{.kicker}

<!--en:5ddde116-->
Der Vergleich von `eager`, `torch.compile` in der Standardeinstellung (`fused`) und `torch.compile(..., mode='reduce-overhead')` (`replayed`) für die 60 Schichten trennt den Effekt der Fusion von dem der Wiedergabe:

<!--en:432b7d5a-->
Mit `torch.compile` in der Standardeinstellung arbeitet die GPU nur während $\sim 15\%$ der Aufrufdauer. Python startet die cuBLAS- und Triton-Kernel jeder Schicht weiterhin einzeln. Mit `mode='reduce-overhead'` laufen alle Kernel nach einem einzigen Start. Die Aufrufdauer sinkt **um den Faktor $7\times$** auf die $\sim 0.3\text{ ms}$, in denen das Gerät tatsächlich arbeitet.

<!--en:770dd06d-->
::: {.slide title="30× schnellere Ausführung der 60 Schichten mit XLA-Befehlspuffern" only="jax"}

<!--en:cdf86c15-2-->
[Messung der Wiedergabe]{.kicker}

<!--en:df320349-->
In JAX fusioniert `jax.jit` jede $\tanh$-Aktivierung direkt mit dem Kernel für das folgende Matrixprodukt und gibt die kompilierte Folge über einen XLA-Befehlspuffer (CUDA-Graph) wieder:

<!--en:48aed660-->
Ein einziger Python-Aufruf startet alle 60 Schichten. Sie laufen in $0.25\text{ ms}$ ($\sim 2\,\mu\text{s}$ pro Operation, die GPU arbeitet während $95\%$ der Aufrufdauer). Das ist rund **um den Faktor $30\times$ schneller** als die sofortige Ausführung mit Einzelstarts pro Schicht.

<!--en:ab0b7e60-->
::: {.slide title="Wiedergabe des MLP-Trainingsschritts in unter 1 Millisekunde" only="pytorch"}

<!--en:8d056c00-->
[Wiedergabe des Trainingsschritts]{.kicker}

<!--en:a4acdd5a-->
Mit `mode='reduce-overhead'` entstehen für den Trainingsschritt des MLP mit 4 Schichten CUDA-Graphen für den Vorwärts- und Rückwärtsdurchlauf sowie für das Update. Die Schrittdauer sinkt auf unter eine Millisekunde:

<!--en:3bf48f5a-->
- **Verbleibende Differenz:** Die GPU arbeitet für die vorhergesagten $0.25\text{ ms}$. Die restlichen $\sim 0.5\text{ ms}$ verbringt Python an der Graphunterbrechung bei `autograd.grad` zwischen der Wiedergabe der beiden Graphen.

<!--en:467c6b09-->
::: {.slide title="Wann sich Kompilierung in der Praxis nicht eignet"}

<!--en:aa8e1c30-->
[Fallstricke in der Praxis]{.kicker}

<!--en:b933f104-->
Vor dem Kompilieren eines Modells oder Trainingsschritts vier mögliche Probleme prüfen:

<!--en:572685f2-->
- **Kurze Läufe ($N_{\textrm{steps}} < n^*$):** Verarbeitet ein Skript nur einige Dutzend Batches zur Auswertung, übersteigt die Kompilierzeit $t_{\textrm{compile}}$ die gesamte eingesparte Laufzeit.

<!--en:d63c5674-->
- **Wechselnde Formen bei jedem Schritt:** Wächst eine Sequenz pro Schritt um $1$ Token, muss die kompilierte Funktion bei jedem Aufruf neu kompiliert werden, sofern die Eingabe nicht durch Padding auf feste Klassenlängen gebracht wird.

<!--en:f94069f3-->
- **Lesezugriffe des Hosts und Protokollierung im Graphen:** `print(loss)` oder `if loss.item() < 0.1:` im kompilierten Schritt erzwingen Graphunterbrechungen oder Tracing-Fehler. Die Protokollierung außerhalb der kompilierten Schrittfunktion vornehmen.

<!--en:1f035f14-->
- **Zeilenweise Fehlersuche:** Aufruflisten und Haltepunkte in fusionierten Kerneln lassen sich Python-Zeilen nicht einzeln zuordnen. Die numerische Korrektheit immer zuerst bei sofortiger Ausführung prüfen.

<!--en:3add741d-->
::: {.slide title="Verständnisfragen zu Beschleunigung durch Fusion, Wiedergabe und Padding"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:39d9d799-->
Drei Fragen zum Compilerentwurf:

<!--en:f0114d27-->
1. Die zeilenweise Softmax-Funktion für eine $R \times d$-Matrix in `float32` benötigt bei sofortiger Ausführung $5$ Kernel: `m = max(x)` (1 Lesezugriff), `y = x - m` (1 Lese-, 1 Schreibzugriff), `e = exp(y)` (1 Lese-, 1 Schreibzugriff), `s = sum(e)` (1 Lesezugriff), `p = e / s` (1 Lese-, 1 Schreibzugriff). Jede Zeile passt in den SRAM auf dem Chip. Welchen Beschleunigungsfaktor ergibt der Speicherverkehr bei Fusion zu 1 Kernel?

<!--en:75bb7db9-->
2. Ein Schritt startet $400$ kleine Kernel mit je $h = 8\,\mu\text{s}$ Startzeit auf dem Host. Die gesamte Laufzeit der Kernel auf der GPU beträgt $1.5\text{ ms}$. Was begrenzt die Geschwindigkeit, und wie stark kann die Wiedergabe eines CUDA-Graphen den Schritt höchstens beschleunigen?

<!--en:975ef343-->
3. Ein Schritt besteht aus einem einzigen $8192 \times 8192$-Matrixprodukt in `bfloat16`. Wie stark beschleunigt die Kompilierung den Schritt?

<!--en:557b98b6-->
**Lösungen:**

<!--en:e0349d1b-->
1. **Beschleunigung um den Faktor $\mathbf{4\times}$:** Die 5 einzeln ausgeführten Kernel benötigen $5\text{ reads} + 3\text{ writes} = 8 Rd$ Transfers (ohne die $O(R)$ Skalare für die Zeilen). Eine einzige fusionierte Reduktion im SRAM benötigt $1\text{ read} + 1\text{ write} = 2 Rd$ Transfers ($8/2 = 4\times$).

<!--en:51d80e70-->
2. **Durch Start-Overhead begrenzt:** Die Starts durch den Host dauern $400 \times 8\,\mu\text{s} = 3.2\text{ ms} > 1.5\text{ ms}$. Bei der Wiedergabe des CUDA-Graphen genügt ein einziger Start. Damit sinkt die Schrittdauer ($3.2\text{ ms} \to 1.5\text{ ms}$) (**Beschleunigung um den Faktor $\approx 2.1\times$**).

<!--en:659f4fa9-->
3. **$1.0\times$ (keine Beschleunigung):** Die $8192^2$-GEMM ist bereits rechenbegrenzt und erreicht mit einem einzigen cuBLAS-Kernel fast $165\text{ TFLOP/s}$. Es gibt weder elementweise Operationen, deren Fusion Transfers einsparen könnte, noch Leerlaufzeiten zwischen Kernelstarts.

<!--en:f4457fb8-->
::: {.slide title="Zusammenfassung zu Grapherfassung, Operatorfusion und Wiedergabe" only="pytorch"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:563b900a-->
| Compilerverfahren | Einsatzbereich | Funktionsweise | Benchmark aus diesem Kapitel |

<!--en:8263733d-->
| **Symbolisches Tracing** | Voraussetzung | Python einmal mit Platzhaltern für Form und Datentyp ausführen (`TorchDynamo`) | Erfasst statische Schleifen. Wertabhängige Verzweigungen benötigen `torch.where` oder `torch.cond` |

<!--en:59a0788b-->
| **Spezialisierung auf Formen** | Voraussetzung | Kachelgrößen an Eingabeformen anpassen (Prüfbedingungen für `SymInt`) | Padding und Längenklassen vermeiden Neukompilierung bei unterschiedlich langen Sequenzen |

<!--en:7daea1cf-->
| **Operatorfusion** | **Bandbreitenbegrenzt** | Elementweise Operationsketten und zeilenweise Reduktionen im SRAM auf 1 Lese- und 1 Schreibzugriff im DRAM reduzieren | Speicherverkehr für `gelu_ish` mit 8 Operationen: $18 \to 2$ Arrays (**Beschleunigung um den Faktor $9\times$**) |

<!--en:dc4b89d7-->
| **Wiedergabe von CUDA-Graphen** | **Durch Start-Overhead begrenzt** | Deskriptoren und Zeiger für Kernelstarts aufzeichnen. 1 Aufruf des Hosts gibt alle Starts wieder | 60 Schichten der Breite 256 laufen **um den Faktor $7\times$ schneller** |

<!--en:b65fbe32-->
| **Amortisationsschritt $n^*$** | Kostenprüfung | $n^* = t_{\textrm{compile}} / (t_{\textrm{eager}} - t_{\textrm{compiled}})$ | Mehrere Sekunden Kompilierzeit amortisieren sich über $> 10^3$ Trainingsschritte |

<!--en:6cd48167-->
::: {.slide title="Zusammenfassung zu Grapherfassung, Operatorfusion und Wiedergabe" only="jax"}

<!--en:c5a222f8-2-->
[Das Wichtigste]{.kicker}

<!--en:563b900a-2-->
| Compilerverfahren | Einsatzbereich | Funktionsweise | Benchmark aus diesem Kapitel |

<!--en:b929d424-->
| **Symbolisches Tracing** | Voraussetzung | Python einmal mit Platzhaltern für Form und Datentyp ausführen (`jaxpr`) | Erfasst statische Schleifen. Wertabhängige Verzweigungen benötigen `lax.select` oder `lax.cond` |

<!--en:1d31f496-->
| **Spezialisierung auf Formen** | Voraussetzung | Kachelgrößen an Eingabeformen anpassen (Cache für statische Formen) | Padding und Längenklassen vermeiden Neukompilierung bei unterschiedlich langen Sequenzen |

<!--en:7daea1cf-2-->
| **Operatorfusion** | **Bandbreitenbegrenzt** | Elementweise Operationsketten und zeilenweise Reduktionen im SRAM auf 1 Lese- und 1 Schreibzugriff im DRAM reduzieren | Speicherverkehr für `gelu_ish` mit 8 Operationen: $18 \to 2$ Arrays (**Beschleunigung um den Faktor $9\times$**) |

<!--en:d373e0ed-->
| **Wiedergabe von CUDA-Graphen** | **Durch Start-Overhead begrenzt** | Deskriptoren und Zeiger für Kernelstarts aufzeichnen. 1 Aufruf des Hosts gibt alle Starts wieder | 60 Schichten der Breite 256 laufen **um den Faktor $30\times$ schneller** |

<!--en:b65fbe32-2-->
| **Amortisationsschritt $n^*$** | Kostenprüfung | $n^* = t_{\textrm{compile}} / (t_{\textrm{eager}} - t_{\textrm{compiled}})$ | Mehrere Sekunden Kompilierzeit amortisieren sich über $> 10^3$ Trainingsschritte |
