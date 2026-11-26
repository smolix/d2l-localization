<!--en:59c23e2b-->
# Grundlagen des Trainings mit mehreren GPUs

<!--en:05736597-->
Bisher haben wir bei jedem Trainingslauf nur eine GPU genutzt, obwohl unser Build-Rechner vier hat. Am einfachsten nutzen wir alle $k$ GPUs mit *Datenparallelität* (engl. data parallelism). Jede GPU erhält eine vollständige Kopie des Modells und $1/k$ jedes Minibatchs. Mit einem *Allreduce* summieren wir die Gradienten über alle GPUs und wenden auf jeder Karte dasselbe Parameterupdate an.

<!--en:a71dde4f-->
Auf unserem Rechner tauschen die GPUs Daten über PCIe 4.0 aus, mit einem Zwischenstopp im DRAM des Hosts (:numref:`subsec_hw-interconnects`). Damit ist jede Verbindung auf nominell 32 GB/s begrenzt und rund $30\times$ langsamer als der Speicher auf der Karte mit 1 TB/s. Eine zweite GPU kann einen Schritt nur verkürzen, wenn die Rechenzeit, die sie der ersten abnimmt, den zusätzlichen Zeitaufwand für den Gradientenaustausch und die Kernel-Starts durch den Host übersteigt. In diesem Abschnitt implementieren wir Datenparallelität selbst und leiten den *Ring-Allreduce* her, den Bibliotheken für kollektive Kommunikation verwenden. Aus einzeln gemessenen Konstanten erstellen wir ein Kostenmodell, das die Schrittdauer auf mehreren GPUs vorhersagt. Wir prüfen unser Modell mit LeNet auf Fashion-MNIST. An den sehr kurzen Schritten fallen die festen Kosten für den Host und die Kommunikation besonders deutlich auf.

<!--en:6f57d25f-->
In PyTorch aktivieren wir außerdem tf32-Matrizenmultiplikationen und die automatische Abstimmung der Faltungen durch cuDNN. So erreichen Kernel für feste Tensorformen ihre Geschwindigkeit im eingeschwungenen Zustand.

<!--en:6bd38a5b-->
## Drei Möglichkeiten der Aufteilung

<!--en:abd15161-->
Ein Modell und sein Minibatch lassen sich auf drei grundlegende Arten auf mehrere Geräte verteilen (:numref:`fig_splitting`).

<!--en:80373bdc-->
* **Datenparallelität** repliziert das Modell auf jedem Gerät, verteilt den Minibatch auf die Geräte und summiert die Gradienten einmal pro Schritt. Das funktioniert für jedes Modell, das auf eine einzelne GPU passt. Allerdings hält jedes Gerät weiterhin den gesamten Trainingszustand von Adam in fp32, also $16N$ Byte (:numref:`sec_memory_precision`).

<!--en:b0c529d2-->
* **Pipeline-Parallelität** verteilt aufeinanderfolgende Schichten auf die Geräte. An den Grenzen zwischen den Stufen gehen Aktivierungen vorwärts und Gradienten rückwärts (Übung :exref:`ex_mg-pipeline`).

<!--en:81c691df-->
* **Tensorparallelität** verteilt einzelne Gewichtsmatrizen auf die Geräte, wie schon 2012 bei AlexNet auf zwei GPUs mit je 3 GB :cite:`Krizhevsky.Sutskever.Hinton.2012`. Die Geräte kommunizieren mehrmals pro Schicht (Übung :exref:`ex_mgp-tensor` in :numref:`sec_sharding`).

<!--en:10cc3507-->
![Drei Möglichkeiten, ein Modell über zwei Geräte zu verteilen. Die Farbe jeder Zelle kennzeichnet ihr Gerät. Bei Datenparallelität liegt das Modell auf beiden Geräten, der Batch ist aufgeteilt. Pipeline-Parallelität unterteilt die Schichtenfolge, Tensorparallelität die Breite jeder Schicht.](../img/mdl-perf-splitting.svg)

<!--en:5b69a774-->
## Datenparallelität von Hand

<!--en:3dd20e5e-->
Ein datenparalleler Schritt auf $k$ Geräten besteht aus vier Phasen (:numref:`fig_data_parallel`). Wir teilen den Minibatch in $k$ gleich große Partitionen (engl. shards) und berechnen den Gradienten jeder Partition auf ihrem Gerät. Dann summieren wir die $k$ Gradienten per Allreduce, sodass jedes Gerät die Summe erhält. Schließlich wenden wir auf jedem Gerät dasselbe Parameterupdate an. Alle Modellkopien starten mit identischen Parametern und erhalten identische Updates. So bleiben sie synchron, ohne dass wir die Gewichte erneut kopieren müssen.

<!--en:e294a3f3-->
![Ein datenparalleler Schritt auf zwei GPUs. Nach der Aufteilung des Minibatchs folgen auf jeder Modellkopie identische Vorwärts- und Rückwärtsdurchläufe. Ein Allreduce summiert die Gradienten. Anschließend erhält jede Kopie dasselbe Update.](../img/mdl-perf-data-parallel.svg)

<!--en:be48cf4e-->
### Modell und Daten

<!--en:4cdae828-->
Wir schreiben LeNet (:numref:`sec_lenet`, mit ReLU-Aktivierungen und He-Initialisierung) als Funktion einer expliziten Liste von Parametertensoren. So sehen wir jeden Kopiervorgang zwischen Geräten. Die 61 706 fp32-Parameter belegen 247 KB, die 60 000 Trainingsbilder von Fashion-MNIST belegen 188 MB. Der Datensatz ist damit klein genug für eine Kopie auf jeder GPU. Das Laden der Daten durch den Host geht also nie in unsere Zeitmessungen ein.

<!--en:707c5cc7-->
Wir prüfen, ob mindestens zwei GPUs verfügbar sind, und kopieren den Trainingsteil von Fashion-MNIST auf beide Geräte (`train_data`). Für alle folgenden Experimente ziehen wir einmal gemeinsame Anfangswerte `base` für die Parameter.

<!--en:284f4fee-->
In JAX liegen geräteübergreifende Arrays auf einem *Gerätegitter* (engl. mesh) mit benannten Achsen. Die Verteilung legen `NamedSharding` und `PartitionSpec` fest. Dabei partitioniert `P('data')` die erste Array-Achse entlang der Gitterachse `'data'`, während `P()` das gesamte Array auf jedem Gerät repliziert.

<!--en:24c16821-->
### Broadcast und Allreduce

<!--en:56eaa878-->
Ein *Broadcast* kopiert die Anfangswerte der Parameter einmal auf alle Geräte. Ein *Allreduce* summiert je einen Tensor pro Gerät, sodass die Summe auf jedem Gerät vorliegt.

<!--en:e8dd4a50-->
Wir schreiben einen einfachen Allreduce in *Sternform*. Gerät 0 sammelt und summiert alle Tensoren und kopiert dann die Summe zurück.

<!--en:ed1ea4a6-->
Wir verwenden `jax.shard_map` mit `jax.lax.psum`, das die Berechnung an NCCL delegiert.

<!--en:7b090196-->
Wir testen `allreduce`, indem wir `[1.0, 1.0]` auf Gerät 0 und `[2.0, 2.0]` auf Gerät 1 ablegen. Dann prüfen wir, ob beide Geräte die elementweise Summe `[3.0, 3.0]` erhalten.

<!--en:e6ff71be-->
### Der datenparallele Schritt

<!--en:b61d3f92-->
Bei $k$ gleich großen Partitionen berechnen wir für jede den Gradienten ihres mittleren Verlusts. Der Mittelwert dieser $k$ Gradienten ist gleich dem Gradienten des gesamten Minibatchs (Übung :exref:`ex_mg-normalization`). Jeder Allreduce-Aufruf verursacht unabhängig von der Nachrichtengröße eine feste Latenz. Deshalb verketten wir die 10 Gradiententensoren von LeNet zu einem einzigen flachen Vektor und benötigen nur einen Allreduce pro Schritt. Mit `communicate=False` überspringen wir den Allreduce und messen die reine Rechenzeit auf den Geräten.

<!--en:6b9ff784-->
### Prüfung anhand eines einzelnen Schritts

<!--en:9bca71b5-->
Vor der Zeitmessung prüfen wir, ob ein datenparalleler Schritt auf zwei GPUs dasselbe Parameterupdate ergibt wie ein Schritt auf einer GPU mit denselben 256 Beispielen. Wir rechnen strikt in fp32. Unterschiede zwischen Faltungsalgorithmen unter tf32 könnten sonst Fehler verdecken, etwa eine fehlende Division durch $k$.

<!--en:429026e7-->
Die Updates auf einer und auf zwei GPUs stimmen bis auf den fp32-Rundungsfehler (etwa $10^{-7}$) überein.

<!--en:d5c96c4c-->
## Ring-Allreduce

<!--en:af82eaaf-->
Beim Stern-Allreduce läuft der gesamte Datenverkehr über Gerät 0. Für einen Gradienten muss der zentrale Knoten je $(k-1)S$ Byte senden und empfangen, wenn der Gradient die Größe $S$ hat. Mit jedem weiteren Gerät steigt die Last am zentralen Knoten, während die übrigen Geräte jeweils nur $S$ Byte pro Richtung übertragen (:numref:`fig_star_vs_ring`, links).

<!--en:46a1ceb4-->
![Datenverkehr beim Stern- und beim Ring-Allreduce für $k = 4$. Der zentrale Knoten des Sterns sendet und empfängt je $(k-1)S$ Byte. Im Ring sendet und empfängt jedes Gerät je $2(k-1)S/k$ Byte, also weniger als $2S$ für jedes $k$.](../img/mdl-perf-star-vs-ring.svg)

<!--en:8fc151ff-->
Der *Ring-Allreduce* :cite:`Patarasuk.Yuan.2009,Sergeev.Del-Balso.2018` kommt ohne zentralen Knoten aus. Er ordnet die $k$ Geräte in einem logischen Ring an und verteilt den Datenverkehr gleichmäßig. Dazu zerlegt er auf jedem Gerät den $S$ Byte großen Gradienten in $k$ Blöcke zu je $S/k$ Byte. Anschließend folgen zwei Phasen mit jeweils $k - 1$ Schritten (:numref:`fig_ring_allreduce`).

<!--en:319ceb4a-->
1. **Reduce-Scatter:** In jedem der $k - 1$ Schritte sendet jedes Gerät einen Block an seinen Nachfolger und addiert den Block seines Vorgängers zur lokalen Kopie. Nach $k - 1$ Schritten hält jedes Gerät die vollständige Summe für einen anderen Block.

<!--en:2c314b1d-->
2. **All-Gather:** In weiteren $k - 1$ Schritten wandern die fertigen Blöcke durch den Ring, bis jedes Gerät alle $k$ summierten Blöcke hält.

<!--en:58410af4-->
Jedes Gerät sendet pro Phase $(k-1)S/k$ Byte, insgesamt also

<!--en:8d2c9861-->
Das Sendevolumen pro Gerät beträgt $S$ bei $k = 2$. Es bleibt strikt unter $2S$, unabhängig davon, wie viele Geräte dem Ring beitreten. Beide Phasen sind auch einzeln nützlich. Beim Training mit Partitionierung (:numref:`sec_sharding`) verteilt Reduce-Scatter die summierten Gradienten, und All-Gather setzt die partitionierten Gewichte zusammen.

<!--en:9c031675-->
![Ring-Allreduce auf vier GPUs, Schritt für Schritt. Reduce-Scatter summiert die Beiträge entlang des Rings, bis pro Gerät ein Block vollständig ist. Anschließend verteilt All-Gather die fertigen Blöcke. Jede Zelle gibt an, wie viele der vier Beiträge ein Block enthält. Bei 4 ist der Block vollständig. Eine umrandete Zelle kennzeichnet den Block, den eine GPU in diesem Schritt empfangen hat. Der Ring im letzten Teilbild zeigt, welche GPU an welche sendet.](../img/mdl-perf-ring-allreduce.svg)

<!--en:ddb14c22-->
Wir implementieren beide Phasen des Rings in NumPy. Dabei verfolgen wir die aufsummierten Blöcke auf jedem Gerät und zählen mit `sent` die insgesamt pro Gerät gesendeten Byte.

<!--en:77aaad12-->
Wir testen `ring_allreduce` mit Zufallsvektoren aus je 1024 Gleitkommazahlen auf $k \in \{2, 4, 8, 16\}$ Geräten. Der Test bestätigt, dass jedes Gerät die exakte Summe erhält und dabei $2(k-1)S/k$ Byte sendet.

<!--en:4d58eb34-->
Das Sendevolumen pro Gerät bleibt zwar durch $2S$ beschränkt, doch der Ring benötigt $2(k-1)$ aufeinanderfolgende Übertragungsschritte. Bei jedem fällt eine Anlauflatenz an. Für kleine Nachrichten auf vielen GPUs wechseln Bibliotheken für kollektive Kommunikation daher zu Baumalgorithmen der Tiefe $\log_2 k$ (Übung :exref:`ex_mg-latency`).

<!--en:a0b49ef3-->
## Kommunikationskosten und Skalierung

<!--en:c1c8b928-->
### Ein Kostenmodell für einen Schritt

<!--en:6778fe69-->
Nach dem Latenz-Bandbreiten-Modell :cite:`Hockney.1994,Thakur.Rabenseifner.Gropp.2005` verursacht jeder Allreduce-Aufruf eine feste Anlauflatenz $\alpha$. Hinzu kommt die Zeit $\beta_{\textrm{link}}$ pro Byte, das ein Gerät sendet. Sie ist der Kehrwert der *Busbandbreite* (engl. bus bandwidth) des Allreduce. Ohne Überlappung ergibt sich für einen Schritt auf $k$ Geräten mit globaler Batchgröße $B$ und $c$ Allreduce-Aufrufen folgende Dauer:

<!--en:373725b9-->
Dabei gilt $c = 0$ für $k = 1$. Ein zweites Gerät verkürzt den Schritt nur dann, wenn $t_{\textrm{compute}}(B) - t_{\textrm{compute}}(B/2) > c\alpha + S\,\beta_{\textrm{link}}$ gilt.

<!--en:57c2f1e9-->
### Latenz und Bandbreite des Allreduce

<!--en:4955fec1-->
Wir messen $\alpha$ mit einem Allreduce für eine einzelne Gleitkommazahl und $\beta_{\textrm{link}}$ mit einem Allreduce für $2^{26}$ Gleitkommazahlen (268 MB), jeweils auf zwei GPUs.

<!--en:c49eb477-->
Wir prüfen, ob :eqref:`eq_dp_cost` den Zeitgewinn durch Gradientenfusion (engl. gradient fusion) vorhersagt. Dazu summieren wir die 10 Gradiententensoren von LeNet (insgesamt $S = 247$ KB) zunächst mit einzelnen Allreduce-Aufrufen ($c = 10$). Anschließend verketten wir die Tensoren zu einem flachen Vektor und summieren ihn mit einem einzigen Aufruf ($c = 1$). Die Zeiten sollten sich um $9\alpha$ unterscheiden.

<!--en:5ebeb0fe-->
Bei einem Gradienten von 247 KB kosten allein die neun zusätzlichen Nachrichten ($9\alpha$) mehr Zeit als die Übertragung sämtlicher Nutzdaten. Deshalb bündeln Bibliotheken für datenparalleles Training im Produktiveinsatz kleine Gradienten vor der Übertragung in großen Puffern (:numref:`sec_multi_gpu_concise`).

<!--en:7de54855-->
### LeNet auf einer und auf zwei GPUs

<!--en:1e23d648-->
Bevor wir LeNet auf zwei GPUs ausführen, messen wir auf einer GPU, wie viel Zeit die einzelnen Operationen für einen Minibatch mit 256 Beispielen benötigen.

<!--en:e997b6f6-->
Die GPU ist während des größten Teils des Schritts untätig. Bei Batchgröße 256 dominiert bei LeNet der Verwaltungsaufwand. Die Schrittdauer entspricht daher der Zeit $h_{\textrm{step}}$ für die Kernel-Starts durch den Host. Halbieren wir die Batchgröße auf einer GPU von 256 auf 128, ändert sich die Schrittdauer folglich kaum. Steuert ein einzelner Host-Thread zwei GPUs an, benötigt er dagegen rund $2 h_{\textrm{step}}$ (:numref:`fig_one_launcher`).

<!--en:c32b8cd2-->
Die Startzeit $k\,h_{\textrm{step}}$ eines einzelnen Host-Threads bildet eine Untergrenze für die Schrittdauer. Berücksichtigen wir sie neben der Rechenzeit $t_{\textrm{dev}}(B/k)$ auf dem Gerät, erhalten wir aus :eqref:`eq_dp_cost` das verfeinerte Modell

<!--en:3d2e3ee2-->
![Der Host begrenzt die Geschwindigkeit eines Schritts auf zwei GPUs. Er stößt zuerst die Arbeit für GPU 0 und dann die für GPU 1 an. Während der Host eine GPU bedient, wartet die andere. Bei kurzen Kerneln benötigt ein Schritt daher etwa $2h_{\textrm{step}}$ auf dem Host zuzüglich der Zeit für den Gradientenaustausch.](../img/mdl-perf-one-launcher.svg)

<!--en:52a3b2b4-->
Um beide Vorhersagen für das gesamte Training zu prüfen, schreiben wir eine Trainingsschleife `train`. Für jede vorab erzeugte Permutation der Beispiele in `orders` durchläuft sie eine Epoche. Sie gibt die mittlere Schrittdauer (ohne die erste Epoche) und die Testgenauigkeit zurück.

<!--en:baaf108c-->
Wir führen `train` nun fünf Epochen lang mit globaler Batchgröße 256 auf $k \in \{1, 2\}$ GPUs aus. Die gemessene Schrittdauer auf zwei GPUs `t_step[2]` vergleichen wir mit der einfachen Vorhersage :eqref:`eq_dp_cost` und mit dem Modell für einen einzelnen Host-Thread :eqref:`eq_dp_one-launcher`.

<!--en:81bfecd0-->
Beide Läufe erreichen dieselbe Testgenauigkeit. Das Modell :eqref:`eq_dp_one-launcher` sagt präzise vorher, um wie viel länger der Schritt auf zwei GPUs dauert. Bei Batchgröße 256 kostet der Gradientenaustausch für LeNet nur etwa 0.1 ms. Der Engpass entsteht, weil ein einzelner Host-Thread beide GPUs mit Arbeit versorgt.

<!--en:a925d5e9-->
### Starke und schwache Skalierung sowie größere Minibatches

<!--en:3271f9db-->
Für Benchmarks mit mehreren GPUs gibt es zwei Konventionen. Bei *starker Skalierung* (engl. strong scaling) bleibt die globale Batchgröße $B$ konstant, während $k$ wächst. Ziel ist es, denselben Schritt schneller abzuschließen. Bei *schwacher Skalierung* (engl. weak scaling) bleibt die Batchgröße $b$ pro GPU konstant, und die globale Batchgröße wächst auf $kb$. Ziel ist ein $k\times$ so hoher Durchsatz in Beispielen pro Sekunde, wobei größere globale Batches die Dynamik der Optimierung verändern (:numref:`sec_batch_size`). Für starke Skalierung gilt das *Gesetz von Amdahl* :cite:`Amdahl.1967`. Der nicht parallelisierbare Anteil des Schritts begrenzt die Beschleunigung unabhängig von der Zahl der GPUs. Hier besteht er aus den Kernel-Starts durch den Host und dem Gradientenaustausch. Für schwache Skalierung gilt das *Gesetz von Gustafson* :cite:`Gustafson.1988`. Wächst die Arbeit mit der Anzahl der Prozessoren, sinkt der relative Anteil der seriellen Arbeit, und der Durchsatz kann weiter steigen.

<!--en:d63a11c9-->
Nach :eqref:`eq_dp_one-launcher` erhöht eine zweite GPU bei schwacher Skalierung den Durchsatz ($2\,t(1)/t(2) > 1$), wenn folgende Bedingung gilt:

<!--en:8b28db30-->
Eine zweite GPU bringt keinen Vorteil, solange $t_{\textrm{dev}}(b) \le h_{\textrm{step}}$ gilt. Sobald $t_{\textrm{dev}}(b)$ sowohl $2h_{\textrm{step}}$ als auch die Zeit für den Gradientenaustausch übersteigt, nähert sich der Beschleunigungsfaktor 2. Wir prüfen dies für $b \in \{256, 4096, 32768\}$ Beispiele pro GPU.

<!--en:2a68a27e-->
Bei $b = 256$ bringt die zweite GPU keine Beschleunigung. Bei $b = 32{,}768$ dauert die Berechnung auf dem Gerät erheblich länger als $2h_{\textrm{step}}$ und als der Austausch der 247 KB an Gradienten. Wie vorhergesagt verdoppelt sich der Durchsatz nahezu.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:2259d3b8-->
Bei Datenparallelität liegt das Modell auf jedem der $k$ Geräte. Jedes Gerät berechnet Gradienten für $1/k$ des Minibatchs. Vor dem Update summieren die Geräte ihre Gradienten per Allreduce. Der Ring-Allreduce zerlegt die Summierung in einen Reduce-Scatter und einen anschließenden All-Gather. Dabei sendet jedes Gerät $2(k-1)S/k < 2S$ Byte in $2(k-1)$ Übertragungsschritten. Unser Kostenmodell für einen Schritt :eqref:`eq_dp_one-launcher` berücksichtigt die Rechenzeit auf dem Gerät, den Startaufwand $k\,h_{\textrm{step}}$ eines einzelnen Host-Threads, die Latenzkosten $c\alpha$ der Aufrufe und die Übertragungskosten $2(k-1)S\beta_{\textrm{link}}/k$. Es erklärt, warum das Verketten kleiner Gradiententensoren $9\alpha$ einspart. Es zeigt auch, warum eine zweite GPU das Training erst dann beschleunigt, wenn die Berechnung pro GPU länger dauert als die Kernel-Starts durch den Host und die Kommunikation.

<!--en:2294243a-->
## Übungen

<!--en:3664279c-->
1. **Datenparallelität und Pipeline-Parallelität im Vergleich.** Betrachten Sie ein Modell aus $L$ linearen Schichten ohne Bias mit Breite $d$. Es wird in fp32 auf zwei GPUs mit Batchgröße $B$ trainiert. Bei Datenparallelität summiert ein Allreduce alle Gewichtsgradienten. Bei zweistufiger Pipeline-Parallelität (:numref:`fig_splitting`) liegen auf jeder GPU $L/2$ Schichten. Über die Grenze zwischen den Stufen fließen $B \times d$ Aktivierungen vorwärts und ebenso viele Gradienten rückwärts.
    1. Bestimmen Sie für beide Verfahren die Anzahl der pro Schritt übertragenen Byte und berechnen Sie die Werte für $L = 32, d = 4096, B = 256$. Unter welcher Bedingung an $(B, L, d)$ überträgt die Datenparallelität weniger Byte?
    1. Teilen Sie den Batch in $m$ Mikrobatches für die Pipeline auf :cite:`Huang.Cheng.Bapna.ea.2019`. Leiten Sie für jede GPU einer zweistufigen Pipeline den Zeitanteil her, in dem sie untätig ist, und berechnen Sie ihn für $m = 1$ und $m = 8$.
   :exlabel:`ex_mg-pipeline`

<!--en:7d330d70-->
2. **Normalisierung der Gradienten.** Vergleichen Sie zwei datenparallele Verfahren für einen in $k$ Partitionen aufgeteilten Batch mit $B$ Beispielen. Verfahren A summiert auf jedem Gerät die Verluste aller lokalen Beispiele, summiert die Gradienten über die Geräte und dividiert durch $B$. Verfahren B mittelt auf jedem Gerät die Verluste der lokalen Beispiele und bildet dann den Mittelwert der $k$ Gerätegradienten.
    1. Zeigen Sie, dass beide Verfahren exakt den mittleren Gradienten des gesamten Batchs ergeben, wenn jede Partition $B/k$ Beispiele enthält.
    1. Ein Sprachmodell mittelt den Verlust über Token. Zwei Partitionen enthalten 3000 beziehungsweise 1000 Token. Bestimmen Sie das relative Gewicht, das Verfahren B einem Token in jeder Partition zuweist. Ändern Sie Verfahren B so ab, dass es auch bei unterschiedlich vielen Token exakt ist.
   :exlabel:`ex_mg-normalization`

<!--en:4a712844-->
3. **Beschleunigung bei starker Skalierung.** Modellieren Sie die Rechenzeit für eine Partition als $t_{\textrm{compute}}(s) = t_0 + s\,t_1$ und vernachlässigen Sie $\alpha$. Nach :eqref:`eq_dp_cost` gilt dann bei fester globaler Batchgröße $B$
   $$t_{\textrm{step}}(k) = t_0 + \frac{B\,t_1}{k} + \frac{2(k-1)}{k}\,S\,\beta_{\textrm{link}}.$$
    1. Zeigen Sie, dass die Bedingung $R(k) = t_{\textrm{step}}(1) /
       t_{\textrm{step}}(k) > 1$ für ein $k > 1$ genau dann gilt, wenn sie für alle $k > 1$ gilt. Bestimmen Sie $\lim_{k \to \infty} R(k)$.
    1. Berechnen Sie für $N = 11.2$ Millionen fp32-Parameter, $t_0 = 2$ ms, $t_1 = 0.2$ ms pro Beispiel und $B = 256$ die Beschleunigungsfaktoren $R(2)$, $R(4)$ und $\lim_{k \to \infty} R(k)$ bei Busbandbreiten von 20 GB/s und 5 GB/s. Bestimmen Sie die untere Grenze der Busbandbreite, ab der eine Beschleunigung möglich ist.
    1. Berechnen Sie $R(4)$ und $\lim_{k \to \infty} R(k)$ bei 5 GB/s erneut, wenn die Geräte Gradienten in bf16 übertragen ($S$ halbiert sich).

<!--en:af35d421-->
4. [code] **Ring-Allreduce von Hand.** ●
    1. Vergleichen Sie Stern- und Ring-Allreduce für $k$ Geräte mit je $S$ Byte. Wie viele Byte werden insgesamt übertragen, und wie viele sendet und empfängt das am stärksten belastete Gerät? Bestimmen Sie das erwartete Verhältnis der Laufzeiten für zwei Fälle. Im ersten arbeiten alle $k$ bidirektionalen Verbindungen gleichzeitig, im zweiten laufen alle Übertragungen nacheinander über einen einzigen Bus.
    1. Implementieren Sie `ring_allreduce(data)` in PyTorch mit `chunk` und expliziten Kopien durch `.to()`. Überprüfen Sie das Ergebnis anhand von `allreduce` für $k \in \{2, 3, 4\}$. Messen Sie die Laufzeiten beider Verfahren für 64 Millionen Gleitkommazahlen pro Gerät bei $k = 2$ und $k = 4$.

<!--en:62688d4b-->
4. [code] **Ring-Allreduce von Hand.** ●
    1. Vergleichen Sie Stern- und Ring-Allreduce für $k$ Geräte mit je $S$ Byte. Wie viele Byte werden insgesamt übertragen, und wie viele sendet und empfängt das am stärksten belastete Gerät? Bestimmen Sie das erwartete Verhältnis der Laufzeiten für zwei Fälle. Im ersten arbeiten alle $k$ bidirektionalen Verbindungen gleichzeitig, im zweiten laufen alle Übertragungen nacheinander über einen einzigen Bus.
    1. Implementieren Sie `ring_allreduce` innerhalb von `jax.shard_map` mithilfe von `jax.lax.ppermute` und `jax.lax.axis_index`. Überprüfen Sie das Ergebnis anhand von `jax.lax.psum` für $k \in \{2, 4\}$ und messen Sie die Laufzeit für 64 Millionen Gleitkommazahlen pro Gerät.

<!--en:635629a4-->
5. **Latenz beim Allreduce.** ● Dauert ein Übertragungsschritt für $n$ Byte insgesamt $\alpha_{\textrm{hop}} + n\,\beta_{\textrm{link}}$, so benötigt ein Ring-Allreduce mit $2(k-1)$ Schritten die Zeit
   $$t_{\textrm{ring}} = 2(k-1)\left(\alpha_{\textrm{hop}} + \frac{S}{k}\,\beta_{\textrm{link}}\right),$$
   Ein Allreduce auf einem Binärbaum summiert aufwärts und verteilt die Summe per Broadcast abwärts, in insgesamt $2\log_2 k$ Schritten zu je $S$ Byte. Er benötigt die Zeit
   $$t_{\textrm{tree}} = 2\log_2 k \left(\alpha_{\textrm{hop}} + S\,\beta_{\textrm{link}}\right).$$
    1. Vergleichen Sie, wie die Latenz- und Bandbreitenterme beider Algorithmen mit $k$ skalieren. Bestimmen Sie für $k = 8$, $\alpha_{\textrm{hop}} = 10$ µs und eine Busbandbreite von 20 GB/s die Nachrichtengröße $S$, unterhalb derer der Baum schneller ist.
    1. Bestimmen Sie bei $k = 2$ die Nachrichtengröße $S_{1/2}$ für die halbe Übertragungsleistung, bei der die feste Latenz gleich der Übertragungszeit ist. Drücken Sie sie durch $\alpha$ und $\beta_{\textrm{link}}$ aus. Berechnen Sie den Wert aus Ihren Messungen in :numref:`subsec_mg-accounting` und vergleichen Sie ihn mit den Puffergrößen für die Gradientenbündelung in :numref:`sec_multi_gpu_concise`.
   :exlabel:`ex_mg-latency`

<!--en:50e76aef-->
[Diskussion](https://d2l.discourse.group/t/36791)

<!--en:8fa21526-->
Kollektive Kommunikation und Datenparallelität von Grund auf<br>**Formen der Parallelität · Broadcast und Allreduce · Ring-Allreduce · Kostenmodell für einen Schritt · starke und schwache Skalierung**

<!--en:6036df64-->
::: {.slide title="Wann verkürzt eine zweite GPU den Schritt?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:0130dcac-->
- **Eingesparte Rechenarbeit:** Bei einem Minibatch mit $B$ Beispielen auf $k$ GPUs verarbeitet jede GPU im Vorwärts- und Rückwärtsdurchlauf nur $B/k$ Beispiele.

<!--en:a88ab654-->
- **Kosten für Kommunikation und Kernel-Starts:** Alle $k$ GPUs müssen ihre je $S$ Byte großen Gradienten über PCIe summieren ($\le 32\text{ GB/s}$ gegenüber $1\text{ TB/s}$ im DRAM der Karte). Eine Schleife in einem einzigen Host-Thread benötigt dafür zusätzlich $k \cdot h_{\textrm{step}}$ für die Kernel-Starts.

<!--en:6b7ec50b-->
- **Bedingung für eine Beschleunigung:** Eine zweite GPU verkürzt den Schritt **genau dann**, wenn die eingesparte Rechenzeit größer ist als die Übertragungszeit zwischen den GPUs plus dem zusätzlichen Zeitaufwand für Kernel-Starts.

<!--en:f97bd025-->
::: {.slide title="Drei Möglichkeiten, das Training auf Geräte zu verteilen"}

<!--en:35e676d9-->
[Formen der Parallelität]{.kicker}

<!--en:0c46952d-->
- **Datenparallelität (DP):** Jede GPU hält das gesamte Modell ($16N$ Byte für Adam in `fp32`). Die GPUs teilen den Minibatch entlang der Batchachse auf und tauschen die Gradienten einmal pro Schritt per **Allreduce** aus.

<!--en:1733e9c3-->
- **Pipeline- und Tensorparallelität (PP / TP):** Ist ein Modell für eine Karte zu groß, übernehmen mehrere GPUs jeweils einen Teil seiner Tiefe (Gruppen von Schichten) oder Breite (einzelne Gewichtsmatrizen). An den Schichtgrenzen tauschen sie Aktivierungen aus.

<!--en:b2f1f30a-->
::: {.slide title="Ablauf eines datenparallelen Trainingsschritts"}

<!--en:58145270-->
[Datenparallelität]{.kicker}

<!--en:a3376b57-->
- **Vier Phasen pro Schritt:** (1) Den Minibatch mit $B$ Beispielen in $k$ disjunkte Partitionen zu je $B/k$ Beispielen zerlegen. (2) Auf jeder GPU unabhängig Vorwärts- und Rückwärtsdurchlauf ausführen. (3) Die $k$ lokalen Gradienten per **Allreduce** summieren. (4) Auf jeder GPU denselben Optimierungsschritt anwenden.

<!--en:e0992976-->
- **Die Gewichte bleiben synchron:** Alle Modellkopien starten mit denselben per Broadcast verteilten Parametern und wenden dieselben Gradientenupdates an. Daher bleiben ihre Gewichte auch ohne weiteren Broadcast stets identisch.

<!--en:5c85d900-->
::: {.slide title="Die Grundoperationen Broadcast und Allreduce"}

<!--en:c1bde34a-->
[Kollektive Grundoperationen]{.kicker}

<!--en:e8e87cfb-->
Datenparallelität beruht auf zwei kollektiven Kommunikationsoperationen über $k$ Geräte.

<!--en:a48921b1-->
- **`broadcast(params, devices)`:** Kopiert die anfänglichen Modellparameter zu Trainingsbeginn einmal von Gerät $0$ auf alle $k$ Geräte.

<!--en:7fd2e9d9-->
- **`allreduce(grads)`:** Summiert je einen Tensor $x_j$ von jedem Gerät $j \in \{0, \ldots, k-1\}$. Die vollständige Summe $\sum_{j=0}^{k-1} x_j$ liegt anschließend auf **jedem** Gerät.

<!--en:c7b24d55-->
::: {.slide title="Exakte Gradientenmittelung über k gleich große Partitionen"}

<!--en:8a5e0542-->
[Mathematische Invariante]{.kicker}

<!--en:250ec16c-->
- **Die Division durch $k$ ist erforderlich:** Jedes Gerät berechnet lokal den Gradienten $\mathbf{g}_j$ des mittleren Verlusts über $B/k$ Beispiele. Da `allreduce` die **Summe** dieser Gradienten liefert, ergibt die Division durch $k$ den exakten globalen Gradienten.

<!--en:fcab0862-->
- **Unbemerkter Fehler ohne Division:** Fehlt der Faktor $1/k$, steigt die effektive Lernrate unbemerkt um den Faktor $k$ auf $k\eta$. Die Konvergenz verschlechtert sich, ohne dass das Programm abbricht.

<!--en:0f836653-->
::: {.slide title="Gradientenprüfung nach einem Schritt auf 1 und 2 GPUs"}

<!--en:2796e9c9-->
[Überprüfung]{.kicker}

<!--en:44de2b4d-->
Vor der Zeitmessung einer Schleife auf mehreren GPUs einen SGD-Schritt mit $B = 256$ Bildern von Fashion-MNIST auf $1$ GPU und auf $2$ GPUs ausführen. Dabei strikt in `float32` ohne `tf32`-Rundung rechnen und mit exakt denselben Gewichten starten.

<!--en:449a1999-->
Die Parameterupdates auf einer und auf zwei GPUs stimmen bis auf die Gleitkommarundung in `float32` überein ($\sim 10^{-7}$ bei einer Updategröße von $\sim 10^{-2}$). Damit sind die Aufteilung in Partitionen, die Skalierung des Verlusts, der Allreduce und die Mittelung mit dem Faktor $1/k$ korrekt.

<!--en:4265deab-->
::: {.slide title="Warum der zentrale Knoten den Stern-Allreduce begrenzt"}

<!--en:03d92745-->
[Stern- und Ringtopologie]{.kicker}

<!--en:26ed6763-->
- **Stern-Allreduce ($O(kS)$ am zentralen Knoten):** Gerät $0$ sammelt $(k-1)$ Gradiententensoren der Größe $S$, summiert sie und sendet $(k-1)$ Kopien zurück. Es sendet und empfängt je $12\text{ GB}$ bei $k=4$ und $S=4\text{ GB}$.

<!--en:c1434770-->
- **Ring-Allreduce ($< 2S$ pro GPU):** Alle $k$ GPUs reichen Blöcke der Größe $S/k$ in einem logischen Ring über $2(k-1)$ Schritte weiter. Jede sendet $\frac{2(k-1)}{k}S$ Byte ($6\text{ GB}$ bei $k=4$).

<!--en:c48463af-->
- **Optimale Bandbreitennutzung:** Das Datenvolumen pro GPU bleibt unter $2S$, selbst für $k \to \infty$.

<!--en:c07dd61c-->
::: {.slide title="Ring-Allreduce mit Reduce-Scatter und anschließendem All-Gather"}

<!--en:fd58ed5e-->
[Ring-Allreduce]{.kicker}

<!--en:f45c6ee7-->
- **Zwei Phasen mit je $k - 1$ Schritten und Blöcken der Größe $S/k$:**
  1. **Reduce-Scatter ($k-1$ Übertragungsschritte):** Jede GPU sendet einen Block der Größe $S/k$ an ihren Nachbarn im Ring und addiert den empfangenen Block zu ihrem lokalen Pufferinhalt. Nach $k-1$ Übertragungsschritten hält jede GPU die **vollständige Summe für eine andere $1/k$-Partition**.
  2. **All-Gather ($k-1$ Übertragungsschritte):** Die fertigen Partitionen der Größe $S/k$ wandern ohne weitere Addition durch den Ring, bis jede GPU alle $k$ summierten Partitionen hält.

<!--en:4b347960-->
::: {.slide title="Prüfung des Sendevolumens (< 2S) und der Anzahl aufeinanderfolgender Übertragungsschritte"}

<!--en:e400b837-->
[Simulation des Rings]{.kicker}

<!--en:c44b87b5-->
In jedem der $2(k-1)$ Ringschritte sendet jede GPU einen Block von $S/k$ Byte. Das gesamte Sendevolumen pro GPU beträgt also $\frac{2(k-1)}{k} S < 2S$.

<!--en:25a8f556-->
- **Bandbreite und Latenz:** Für große Tensoren nutzt der Ring-Allreduce die Bandbreite optimal ($S$ ist unabhängig von $k$, auch wenn $k$ wächst). Er benötigt jedoch $2(k-1)$ aufeinanderfolgende Übertragungsschritte mit je einer Anlauflatenz von $\alpha_{\textrm{hop}}$. Für kleine Nachrichten auf vielen GPUs verwendet NCCL Bäume mit $2\log_2 k$ Übertragungsschritten.

<!--en:327b9a1f-->
::: {.slide title="Das Latenz-Bandbreiten-Modell für einen parallelen Schritt"}

<!--en:d5c9f8ce-->
[Kostenmodell für einen Schritt]{.kicker}

<!--en:0399a6cd-->
Nach dem Latenz-Bandbreiten-Modell von Hockney kostet jeder Allreduce-Aufruf eine feste Anlauflatenz $\alpha$. Hinzu kommt für jedes pro Gerät gesendete Byte die Übertragungszeit $\beta_{\textrm{link}}$, der Kehrwert der *Busbandbreite* des Allreduce.

<!--en:d6c28674-->
- **$t_{\textrm{compute}}(B/k)$:** Lokale Zeit für Vorwärtsdurchlauf, Rückwärtsdurchlauf und Optimierungsschritt auf einer Partition mit $B/k$ Beispielen.

<!--en:76af33da-->
- **$c\,\alpha$:** Gesamte Anlauflatenz für $c$ separate Allreduce-Aufrufe pro Schritt ($c = 0$ bei $k = 1$).

<!--en:50128092-->
- **$\frac{2(k-1)}{k} S \beta_{\textrm{link}}$:** Übertragungszeit für insgesamt $S$ Byte an Gradienten pro GPU.

<!--en:0cca0108-->
- **Wann $k = 2$ schneller ist als $k = 1$:** Eine zweite GPU verkürzt den Schritt nur dann, wenn $t_{\textrm{compute}}(B) - t_{\textrm{compute}}(B/2) > c\alpha + S\beta_{\textrm{link}}$ gilt.

<!--en:e786ebde-->
::: {.slide title="Messung der Latenz α und des Bandbreitenparameters β_link für kollektive Kommunikation"}

<!--en:37794ebb-->
[Kalibrierung der Verbindung]{.kicker}

<!--en:e844cf9e-->
Zur Messung von $\alpha$ dient ein Allreduce für eine Gleitkommazahl ($4\text{ B}$), zur Messung von $\beta_{\textrm{link}}$ ein Allreduce für $2^{26}$ Gleitkommazahlen ($268\text{ MB}$), jeweils auf zwei RTX 4090.

<!--en:df4de7d7-->
- **Warum die Busbandbreite deutlich unter $32\text{ GB/s}$ liegt:** Die RTX 4090 für den Endkundenmarkt unterstützt weder NVLink noch direkten Peer-to-Peer-Zugriff über PCIe. Deshalb gelangt jedes Byte auf dem Weg zwischen den GPUs über PCIe in das DRAM des Hosts und von dort erneut über PCIe zur Ziel-GPU.

<!--en:52d411b4-->
::: {.slide title="Warum kleine Gradiententensoren verkettet werden müssen"}

<!--en:c132dbe1-->
[Nachrichtenanzahl und Datenvolumen]{.kicker}

<!--en:b7f6d050-->
LeNet hat $10$ Parametertensoren (5 Gewichtstensoren, 5 Bias-Vektoren) mit insgesamt $61{,}706$ Parametern in `float32` ($S = 247\text{ KB}$). Einzelne Allreduce-Aufrufe für jeden Tensor ($c = 10$) übertragen exakt dieselbe Datenmenge wie ein einziger Aufruf für den verketteten flachen Vektor ($c = 1$), nämlich $247\text{ KB}$.

<!--en:541113b8-->
**Die zusätzlichen Nachrichten kosten Zeit ($9\alpha$):** Für einen Gradienten von $247\text{ KB}$ gilt $S\beta_{\textrm{link}} \lesssim 100\,\mu\text{s}$. Die Anlauflatenz von $9$ zusätzlichen Allreduce-Aufrufen übersteigt also die Zeit für die Übertragung aller Gradienten. Kleine Gradienten vor der Übertragung immer zusammenfassen oder bündeln.

<!--en:17ac7de7-->
::: {.slide title="Laufzeitanalyse von LeNet auf einer GPU bei dominierendem Verwaltungsaufwand"}

<!--en:fa4587d3-->
[Laufzeitprofil eines lokalen Schritts]{.kicker}

<!--en:f803ccc6-->
Vor der Laufzeitmessung für LeNet auf zwei GPUs mit globaler Batchgröße $B = 256$ zunächst die lokale Berechnung (`communicate=False`) auf einer GPU analysieren.

<!--en:b8ff765a-->
- **Die Kernel-Starts durch den Host dominieren:** Die GPU rechnet nur während eines kleinen Teils des Schritts. Bei LeNet mit $B = 256$ **dominiert der Verwaltungsaufwand**. Die Startzeit $h_{\textrm{step}}$ auf dem Host bestimmt die Schrittdauer auf einer GPU, die Rechenzeit der GPU ist nachrangig.

<!--en:803b901b-->
::: {.slide title="Ein einzelner Host-Thread als Engpass bei doppeltem Startaufwand"}

<!--en:b30a8bf2-->
[Grenze durch Kernel-Starts auf dem Host]{.kicker}

<!--en:c3e0796f-->
- **Bei dominierendem Verwaltungsaufwand bringt eine Halbierung der Batchgröße nichts:** Auf $1$ GPU dauert ein Schritt mit $b = 128$ fast genauso lange ($h_{\textrm{step}}$) wie mit $b = 256$.

<!--en:1405e196-->
- **Ein Host-Thread steuert $k$ GPUs:** Startet eine einzelne Python-Schleife die Kernel erst auf GPU 0 und dann auf GPU 1, wächst die Startzeit auf dem Host gemäß $k \cdot h_{\textrm{step}}$.

<!--en:c67dc654-->
::: {.slide title="LeNet auf zwei GPUs: Vorhersage und Messung der längeren Schrittdauer"}

<!--en:a1c34195-->
[Ursache der längeren Schrittdauer]{.kicker}

<!--en:f281d737-->
Der Ersatz von $t_{\textrm{compute}}(B/k)$ durch $\max(k\,h_{\textrm{step}},\, t_{\textrm{dev}}(B/k))$ berücksichtigt sowohl die Kernel-Starts durch einen einzelnen Host-Thread als auch die Ausführung auf dem Gerät.

<!--en:0fbb5176-->
Bei globaler Batchgröße $B = 256$ dauert ein Schritt auf zwei GPUs **fast $2\times$ so lange** wie auf einer GPU. Der Austausch der $247\text{ KB}$ an Gradienten spielt mit $\sim 0.1\text{ ms}$ kaum eine Rolle. Der einzelne Host-Thread benötigt dagegen $2h_{\textrm{step}}$, um die Arbeit auf beiden Karten zu starten.

<!--en:35e03507-->
::: {.slide title="Konventionen für starke und schwache Skalierung"}

<!--en:c8157c1f-->
[Skalierungsgesetze]{.kicker}

<!--en:941bd99a-->
- **Starke Skalierung (feste globale Batchgröße $B$, $b = B/k$):** Die Gradientenstatistik bleibt exakt erhalten. Ziel ist, die reale Schrittdauer um den Faktor $k\times$ zu verkürzen. Mit wachsendem $k$ sinkt jedoch der Rechenaufwand pro GPU im Verhältnis zur Kommunikation.

<!--en:4b5af0f2-->
- **Schwache Skalierung (feste Batchgröße $b$ pro GPU, $B = kb$):** Der Rechenaufwand pro GPU bleibt konstant. Angestrebt wird der $k\times$ so hohe Durchsatz ($\text{examples/s}$), begrenzt durch die kritische Batchgröße $b_{\textrm{noise}}$.

<!--en:4d0e9f05-->
::: {.slide title="Größere Minibatches überwinden die Skalierungsschwelle"}

<!--en:215ac1d4-->
[Fixkosten verteilen]{.kicker}

<!--en:389de016-->
Wächst die Batchgröße $b$ pro GPU gemäß $256 \to 4{,}096 \to 32{,}768$, übersteigt die Ausführungszeit $t_{\textrm{dev}}(b)$ auf dem Gerät schließlich sowohl $2h_{\textrm{step}}$ als auch die Zeit $\alpha + S\beta_{\textrm{link}}$ für den Gradientenaustausch.

<!--en:7d12281b-->
- **Drei Bereiche:** Bei $b = 256$ gilt $t_{\textrm{dev}} \ll h_{\textrm{step}}$ (kein Gewinn). Bei $b = 4{,}096$ beginnt $t_{\textrm{dev}}$, die Zeit $h_{\textrm{step}}$ zu übersteigen. Bei $b = 32{,}768$ dauert die Berechnung auf dem Gerät deutlich länger als die Kernel-Starts durch den Host und der Allreduce für $247\text{ KB}$. So erreichen zwei GPUs bei schwacher Skalierung eine **Beschleunigung um den Faktor $\sim 2\times$ ($1.97\text{--}1.98\times$)**.

<!--en:692ecd07-->
::: {.slide title="Verständnisfragen zu Allreduce-Datenverkehr und Beschleunigung mit mehreren GPUs"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:e8e390cf-->
Zwei Fragen zur Skalierung mithilfe des Kostenmodells für mehrere GPUs beantworten.

<!--en:694faba1-->
1. Ein GPT-Modell mit einer Milliarde Parametern ($N = 10^9$, Gradienten in `float32` mit $S = 4\text{ GB}$) trainiert auf $k = 8$ GPUs. Wie viele Byte sendet die am stärksten belastete GPU pro Schritt bei einem **Stern-Allreduce** und bei einem **Ring-Allreduce**?

<!--en:3dda73ed-->
2. Ein Trainingsschritt dieses GPT-Modells mit einer Milliarde Parametern verarbeitet global $32{,}768$ Token ($6N\text{ FLOPs/token}$). Jede GPU erreicht dabei $35\text{ TFLOP/s}$ in `bfloat16`. Wie groß ist ohne Überlappung der Kommunikation der Beschleunigungsfaktor bei starker Skalierung mit $k = 2$ GPUs gegenüber $1$ GPU, wenn die Busbandbreite zwischen den GPUs (a) $10\text{ GB/s}$ oder (b) $2.5\text{ GB/s}$ beträgt?

<!--en:557b98b6-->
**Lösungen:**

<!--en:d54608a4-->
1. **Der zentrale Knoten des Sterns sendet $(k-1)S = 7 \times 4 = \mathbf{28\text{ GB}}$** (und empfängt $28\text{ GB}$). **Im Ring sendet jede GPU $\frac{2(k-1)}{k}S = \frac{14}{8} \times 4 = \mathbf{7\text{ GB}}$** (um den Faktor $4\times$ geringere Spitzenlast pro Gerät, strikt $< 2S = 8\text{ GB}$).

<!--en:c882b4d1-->
2. **Rechenzeit auf 1 GPU:** $\frac{6 \times 10^9 \times 32{,}768}{3.5 \times 10^{13}} \approx \mathbf{5.6\text{ s}}$. **Rechenzeit auf 2 GPUs:** $2.8\text{ s}$ zuzüglich $S\beta_{\textrm{link}}$ (bei $k=2$ sendet jede GPU $\frac{2(1)}{2}S = 4\text{ GB}$).
   - **(a) Bei $10\text{ GB/s}$:** $t_{\textrm{comm}} = 0.4\text{ s} \implies t_{\textrm{step}}(2) = 3.2\text{ s} \implies \textrm{Speedup} = 5.6 / 3.2 = \mathbf{1.75\times}$.
   - **(b) Bei $2.5\text{ GB/s}$:** $t_{\textrm{comm}} = 1.6\text{ s} \implies t_{\textrm{step}}(2) = 4.4\text{ s} \implies \textrm{Speedup} = 5.6 / 4.4 = \mathbf{1.27\times}$.

<!--en:4e361800-->
::: {.slide title="Zusammenfassung: Kollektive Kommunikation, Ring-Allreduce und Kostenmodelle"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:81b0acd6-->
| Konzept / Verfahren | Formel / Regel | Wert / Verhalten | Rolle beim Training mit mehreren GPUs |

<!--en:64aa1861-->
| **Datenparallelität** | $\mathbf{g} = \frac{1}{k}\sum_{j=1}^k \mathbf{g}_j$ | Repliziert den Trainingszustand der Größe $16N$ auf $k$ GPUs | Zerlegt den Batch der Größe $B$ in $k$ Partitionen zu je $B/k$ Beispielen und summiert die Gradienten einmal pro Schritt |

<!--en:ecbfabea-->
| **Prüfung nach einem Schritt** | $\|\Delta \boldsymbol{\theta}_{k=2} - \Delta \boldsymbol{\theta}_{k=1}\|_\infty$ | $\le 10^{-7}$ bei strikter `fp32`-Arithmetik | Erkennt eine fehlende Skalierung mit $1/k$ oder Fehler bei der Partitionierung |

<!--en:7db98ac0-->
| **Ring-Allreduce** | Reduce-Scatter + All-Gather | $\frac{2(k-1)}{k}S < 2S$ Byte/GPU | Verteilt den Datenverkehr gleichmäßig auf $k$ GPUs in $2(k-1)$ Übertragungsschritten |

<!--en:8814ff88-->
| **Kostenmodell für einen Schritt** | $\max(kh_{\textrm{step}}, t_{\textrm{dev}}) + c\alpha + \frac{2(k-1)}{k}S\beta_{\textrm{link}}$ | $\beta_{\textrm{link}} = 1 / \textrm{bus BW}$ | Sagt vorher, ob zusätzliche GPUs einen Schritt verkürzen oder verlängern |

<!--en:c4cdbb08-->
| **Gradientenfusion** | $c$ Tensoren verketten $\to 1$ Puffer | Spart $(c - 1)\alpha$ an Latenz | Vermeidet bei kleinen Gradienten die Anlaufkosten pro Tensor |

<!--en:934a4b19-->
| **Starke und schwache Skalierung** | Festes $B$ bzw. festes $b = B/k$ | Beschleunigungsfaktor $t(1)/t(k)$ bzw. $k t(1)/t(k)$ | Unterscheidet kürzere Einzelschritte von höherem Gesamtdurchsatz |
