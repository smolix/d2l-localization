<!--en:9a8660c5-->
# Das Leistungsmodell

<!--en:dfdf5910-->
Bei jedem Netz, das wir bisher trainiert haben, entfällt der größte Teil der Rechenarbeit auf Matrixprodukte. Dabei haben wir stillschweigend angenommen, dass die GPU sie nahezu mit Spitzenleistung berechnet. Diese Annahme trifft oft nicht zu. Auf der RTX 4090, auf der die Notebooks zu diesem Buch laufen, kommt das bf16-Produkt zweier Matrizen der Größe $8192 \times 8192$ bis auf wenige Prozent an die Spitzenleistung von 165 TFLOP/s heran. Ein Produkt der Größe $256 \times 256$ erreicht bei korrekter Zeitmessung weniger als zwei Prozent davon, obwohl Operation und Chip dieselben sind. Ein Trainingsschritt besteht aus vielen Operationen, darunter auch kleinen. Erreichen die meisten nur einen Bruchteil der Spitzenleistung, bringt eine schnellere GPU oder eine zweite Karte wenig. Die meiste Zeit vergeht mit Warten statt mit Rechnen.

<!--en:9e3c7216-->
Für jede GPU-Operation gibt es drei mögliche Engpässe. Der erste sind die Recheneinheiten, der zweite ist die Speicherbandbreite, mit der sie Daten erhalten, und der dritte sind die festen Kosten, die auf dem Host beim Start aus Python anfallen. Welcher der ersten beiden Engpässe greift, lässt sich auf dem Papier bestimmen. Dazu zählen wir die FLOPs, die die Operation ausführt, und die Bytes, die sie überträgt. Mit einer einzigen gemessenen Konstanten, der Zeit $h$ für den Start durch den Host, können wir auch den dritten Engpass vorhersagen. Im gesamten Kapitel zählen wir FLOPs und Bytes, sagen die Laufzeit vorher, messen sie und erklären die Abweichung.

<!--en:cf8ca2bf-->
In PyTorch setzen wir außerdem die Rechengenauigkeit für die Matrizenmultiplikation in float32 auf `'high'`. So nutzen fp32-Matrixprodukte die tf32-Tensor-Cores:

<!--en:b26d632e-->
Auf GPUs von NVIDIA berechnet JAX fp32-Matrixprodukte bereits standardmäßig auf den tf32-Tensor-Cores.

<!--en:621580d5-->
## Arithmetische Intensität und Roofline

<!--en:a07f3c91-->
Schon vor der ersten Zeitmessung können wir aus den Formen der beteiligten Tensoren vorhersagen, welche Ressource eine Operation begrenzt. In :numref:`subsec_kv-cache-prefill-decode` haben wir damit kurz erklärt, warum der Prefill Token viel schneller verarbeitet als das autoregressive Dekodieren. Hier arbeiten wir diesen Gedanken aus.

<!--en:88ee42f5-->
### FLOPs und Bytes zählen

<!--en:19d328b3-->
Jede Tensoroperation beansprucht die GPU auf zweierlei Weise. Sie rechnet, wobei wir jede Addition und jede Multiplikation als eine *Gleitkommaoperation* zählen (engl. floating-point operation, FLOP). Außerdem überträgt sie Bytes zwischen dem Hauptspeicher der GPU und den Prozessoren auf dem Chip.

<!--en:789d5633-->
Für beides hat die GPU jeweils eine eigene Obergrenze. Der Hauptspeicher der RTX 4090 liefert höchstens $\beta = 1.008 \times 10^{12}$ Byte/s (rund 1 TB/s). Ihre Spitzenrechenleistung $P$ hängt vom Zahlenformat und von der Recheneinheit ab. Bei der nominellen Boost-Taktfrequenz von 2.52 GHz erreichen bf16-Matrixprodukte auf den Tensor Cores $P = 165 \times 10^{12}$ FLOP/s. tf32-Matrixprodukte und gewöhnliche fp32-Rechenoperationen erreichen die Hälfte (82.6 TFLOP/s). Die tatsächliche Taktfrequenz kann je nach Temperatur und Leistungsaufnahme der Karte um einige Prozent höher oder niedriger liegen. Wir betrachten die Obergrenze daher als erreicht, wenn ein Messwert um höchstens etwa 5% nach oben oder unten von $P$ abweicht.

<!--en:1c5e19c2-->
Welche Grenze wir zuerst erreichen, hängt vom Verhältnis der beiden Anforderungen ab, der *arithmetischen Intensität*

<!--en:538f18c6-->
Betrachten wir das Matrixprodukt $\mathbf{X}\mathbf{W}$ mit $\mathbf{X} \in \mathbb{R}^{B \times D}$ und $\mathbf{W} \in \mathbb{R}^{D \times F}$. Es entspricht einer linearen Schicht, die einen Batch mit $B$ Zeilen verarbeitet. Das Produkt erfordert $2BDF$ FLOPs (eine Multiplikation und eine Addition pro Term des inneren Produkts). Im günstigsten Fall liest die GPU $\mathbf{X}$ und $\mathbf{W}$ je einmal und schreibt die Ausgabe einmal (insgesamt $BD + DF + BF$ Zahlen). Bei $b$ Byte pro Zahl ergibt sich die Intensität

<!--en:cd9c9bfe-->
Der mittlere Ausdruck zeigt, dass die kleinste der drei Dimensionen $(B, D, F)$ die Intensität bestimmt. Ist die Batchgröße $B$ wesentlich kleiner als die Schichtdimensionen, wächst die Intensität linear mit $B$. Jedes aus dem Speicher gelesene Gewicht kommt in allen $B$ Zeilen zum Einsatz.

<!--en:227bf092-->
Betrachten wir einen GPT mit einer Milliarde Parametern und der Breite $D = F = 2048$ in bf16 ($b = 2$). Ein Minibatch aus 8 Sequenzen mit je 1024 Token hat im Training $B = 8192$ Zeilen. Das ergibt $I \approx 910$ FLOP/Byte nach :eqref:`eq_matmul_intensity`, begrenzt durch die Breite $D$. Bei der autoregressiven Generierung einer einzelnen Sequenz gelten $B = 1$ und $I \approx 1$ FLOP/Byte. Zwischen Training und Dekodierung unterscheidet sich die Intensität derselben Schicht um fast drei Größenordnungen. Am unteren Ende liegt eine elementweise Operation wie $y_i = x_i + 1$. Unabhängig von der Tensorgröße führt sie 1 FLOP pro $2b$ übertragenen Bytes aus ($I = 0.25$ FLOP/Byte in bf16).

<!--en:acd2838d-->
### Das Roofline-Modell

<!--en:d862bfcd-->
Das *Roofline-Modell* :cite:`Williams.Waterman.Patterson.2009` liefert aus der Intensität eine obere Schranke für den Durchsatz. Eine Operation ist sowohl durch die Spitzenrechenleistung $P$ begrenzt als auch durch die Rate $I\beta$, die die Datenversorgung aus dem Speicher zulässt:

<!--en:a392fb84-->
In doppelt logarithmischer Darstellung steigt die Bandbreitengrenze als Gerade bis zum *Knickpunkt* (engl. ridge point) $I^* = P/\beta$ (:numref:`fig_roofline`). Dort trifft sie auf die waagerechte Grenze der Rechenleistung. Unterhalb von $I^*$ ist eine Operation *bandbreitenbegrenzt* (engl. bandwidth-bound). Der Speicher liefert zu langsam Daten für die Recheneinheiten. Zusätzliche Rechnungen mit Daten, die bereits auf dem Chip liegen, kosten daher keine Zeit. Oberhalb von $I^*$ ist die Operation *rechenbegrenzt* (engl. compute-bound). Die Recheneinheiten sind ausgelastet, und die Operation wird auch dann nicht schneller, wenn sie weniger Bytes überträgt. Im gesamten Kapitel bezeichnet $\beta$ die Bandbreite des Hauptspeichers (DRAM). Passen die Operanden in den 72 MB großen L2-Cache auf dem Chip, kann der Durchsatz diese Grenze überschreiten.

<!--en:71c501f1-->
![Die Roofline der RTX 4090. Der erreichbare Durchsatz ist das Minimum aus der Spitzenrechenleistung $P$ und dem Produkt aus arithmetischer Intensität und Speicherbandbreite $\beta$. Die Punkte markieren die Beispiele aus dem Text: $x + 1$ in bf16, eine lineare Schicht der Breite 2048 für ein einzelnes Token und dieselbe Schicht für einen Minibatch mit 8192 Token. Der Knickpunkt liegt bei etwa 164 FLOP/Byte.](../img/mdl-perf-roofline.svg)

<!--en:eab7f042-->
### Der Knickpunkt

<!--en:e9c96704-->
Für bf16-Produkte auf den Tensor Cores der RTX 4090 liegt der Knickpunkt bei $I^* = 165.2 / 1.008 \approx 164$ FLOP/Byte, für tf32-Produkte ($P = 82.6$ TFLOP/s) bei $I^* \approx 82$ FLOP/Byte. Auf modernen GPUs für Rechenzentren liegt er noch höher (etwa 280–300 FLOP/Byte). Die Rechenleistung der Tensor Cores in FLOP/s ist schneller gestiegen als die HBM-Bandbreite. Nur große Matrixprodukte überschreiten diese Schwellen. Elementweise Operationen, Normalisierungen und die Dekodierung einzelner Token liegen weit darunter.

<!--en:e0804497-->
Für das Produkt quadratischer bf16-Matrizen ($B = D = F = n$) gilt $I = n / 3$. Nach dem Roofline-Modell müsste also jedes Produkt mit $n \gtrsim 500$ rechenbegrenzt sein. Um diese Vorhersage zu prüfen, müssen wir die Laufzeit auf der GPU genau messen.

<!--en:dda822cb-->
## Laufzeitmessung auf der GPU

<!--en:20176669-->
Da der Start von GPU-Operationen asynchron ist (:numref:`subsec_async_dispatch`), reiht Python Kernel in die Warteschlange ein und fährt sofort fort. Messen wir die verstrichene Zeit um einen Kernel-Aufruf ohne Synchronisierung, erfassen wir nur die Zeit, die Python zum Einreihen benötigt. Die Ausführungszeit auf der GPU entgeht uns. Für eine genaue Messung führen wir zunächst Aufwärmläufe auf dem Gerät aus, synchronisieren vor Beginn und Ende der Messung und bilden den Mittelwert über viele Wiederholungen. So arbeitet `d2l.Benchmark`.

<!--en:a16bb7f5-->
### Synchronisierungspunkte

<!--en:47308057-->
Python wartet nur dann auf die GPU, wenn es einen Wert vom Gerät benötigt. Das geschieht etwa beim Umwandeln eines Tensors in einen Python-Skalar oder ein NumPy-Array, beim Ausgeben seines Werts oder bei einer Verzweigung anhand dieses Werts (`if loss < 0.1:`). Dasselbe gilt für Operationen, bei denen die Ausgabeform von Tensorwerten abhängt, etwa bei der Auswahl mit einer booleschen Maske (`x[x > 0]`).

<!--en:92032801-->
Benötigt der Host die Zeit $h$ zum Starten eines Schritts und die GPU die Zeit $t_\textrm{dev}$ für dessen Ausführung, überlappen sich die beiden Vorgänge in einer unsynchronisierten Schleife. Lesen wir dagegen in jedem Schritt einen Skalar aus, laufen Start und Ausführung nacheinander ab. Hinzu kommt die Latenz $\sigma$ für die Rückfrage beim Gerät:

<!--en:725cdca6-->
Wir vergleichen zwei Schleifen mit je 1000 Schritten. Beide multiplizieren eine Matrix der Größe $1024 \times 1024$ mit sich selbst und summieren die Ergebnisse auf der GPU auf. Die erste wandelt die laufende Summe bei jedem Schritt in einen Python-`float` um, die zweite liest sie nur einmal am Ende aus.

<!--en:4a1949db-->
Ohne Auslesen bei jedem Schritt benötigt der Host rund 40 µs für den Start der drei Aufrufe pro Schritt. Das dauert länger als die rund 26 µs für die tf32-Rechnung auf der GPU, die daher auf den Host warten muss. Lesen wir in jedem Schritt `float(s)` aus, müssen Host und Gerät abwechselnd arbeiten. Die Schleife dauert dann etwa 1.6-mal so lange.

<!--en:27de2bbf-->
Ohne Auslesen bei jedem Schritt benötigt der Host rund 0.1 ms für den Start jedes JIT-kompilierten Aufrufs. Lesen wir `float(s)` in jedem Schritt aus, kommt bei jedem Schleifendurchlauf die Latenz $\sigma$ für die Rückfrage beim Gerät hinzu. Die Schleife dauert dadurch mehr als dreimal so lange.

<!--en:dcc891cf-->
Trainingsschleifen sollten Metriken daher auf dem Gerät aufsummieren und nur zu den Protokollierungszeitpunkten auslesen.

<!--en:e3edefcc-->
### Ein Hilfswerkzeug für Laufzeitmessungen

<!--en:78984368-->
Im Folgenden messen wir Laufzeiten mit `d2l.Benchmark`. Aufwärmaufrufe sorgen dafür, dass die Kompilierung und das Laden der Kernel vor der Messung abgeschlossen sind. Anschließend misst das Werkzeug die Zeit für `repeats` Aufrufe zwischen zwei Synchronisierungsbarrieren auf dem Gerät. Als ersten Test betrachten wir ein bf16-Produkt der Größe $4096 \times 4096$. Seine Intensität von $I \approx 1365$ FLOP/Byte liegt weit über dem Knickpunkt. Die Roofline sagt daher eine Laufzeit von $2 \cdot 4096^3 / P \approx 0.83$ ms vorher.

<!--en:2ac18b01-->
Die gemessene Laufzeit weicht um höchstens etwa 10% von der Vorhersage von 0.83 ms ab.

<!--en:6423eb8e-->
## Drei Fälle

<!--en:b317d80d-->
Die Gesamtlaufzeit eines GPU-Programms verteilt sich auf rechenbegrenzte Kernel, bandbreitenbegrenzte Kernel und Leerlaufzeiten, in denen das Gerät auf den Host wartet :cite:`He.2022`. In drei Experimenten betrachten wir diese Fälle einzeln.

<!--en:8bfe044d-->
### Bandbreitengrenze und Grenze der Rechenleistung

<!--en:9a4cd0d8-->
Zunächst multiplizieren wir eine Aktivierungsmatrix mit $B$ Zeilen mit einer bf16-Gewichtsmatrix der Form $d \times d$ für $d = 16{,}384$ (512 MiB). Nach :eqref:`eq_matmul_intensity` liegt die Intensität ungefähr bei $B$ und erreicht den Knickpunkt bei etwa $B = 164$. Unterhalb des Knickpunkts muss jeder Aufruf 512 MiB an Gewichten lesen (rund 0.53 ms bei 1.008 TB/s), gleichgültig, welchen Wert $B$ hat. Oberhalb des Knickpunkts bestimmen die $2Bd^2$ FLOPs die Laufzeit. Sie wächst linear mit $B$ (0.83 ms bei $B = 256$, 3.32 ms bei $B = 1024$).

<!--en:a912d1a0-->
Mit derselben Gewichtsmatrix erreichen wir beide Grenzen. Von $B = 1$ bis $B = 128$ bleibt die Laufzeit trotz einer Zunahme der Rechenarbeit um den Faktor $128\times$ nahezu konstant bei 0.6 ms. Ab $B = 256$ wächst sie linear mit $B$, während die GPU nahe der Spitzenrechenleistung arbeitet. Verarbeiten wir 1024 Token auf einmal, steigt die Rechenarbeit gegenüber einem einzelnen Token um den Faktor $1{,}024\times$, die Laufzeit jedoch um weniger als den Faktor $6\times$.

<!--en:8b8c8f75-->
### Elementweise Operationen

<!--en:9f21b610-->
Nun bleiben wir weit unterhalb des Knickpunkts und variieren die Rechnung pro Element eines fp32-Tensors der Form $8192 \times 8192$ (256 MiB). Wir testen Addition, Multiplikation, Sinus und Sigmoidfunktion. Jede Operation liest 256 MiB und schreibt 256 MiB (insgesamt 512 MiB, also 0.53 ms bei 1.008 TB/s).

<!--en:5e2a1a99-->
Alle vier Operationen dauern rund 0.6 ms und erreichen damit eine effektive Speicherbandbreite von rund 0.9 TB/s. Da der Speicherverkehr die Laufzeit bestimmt, kosten transzendente Funktionen (`sin`, `sigmoid`) nicht mehr Zeit als `x + 1`.

<!--en:50f8c537-->
Bei sofortiger Ausführung vervielfacht die Verkettung elementweiser Operationen den Speicherverkehr. Die folgende tanh-Näherung für GELU (`gelu_ish`) verkettet 8 elementweise Operationen. Sechs einstellige Operationen lesen und schreiben je einen Tensor (12 Tensorübertragungen). Zwei zweistellige Operationen (`x + 0.04 * x**3` und die abschließende Multiplikation mit `0.5 * x`) lesen je zwei Tensoren und schreiben einen (6 Tensorübertragungen). Insgesamt werden 18 Tensoren zu je 256 MiB übertragen. Gegenüber einer einzelnen Addition steigt die Datenmenge um den Faktor $9\times$.

<!--en:a6250733-->
Wie vorhergesagt dauert die Kette aus einzelnen Kerneln um den Faktor $8\times$ bis $9\times$ länger als `x + 1`. Die *Kernel-Fusion* (:numref:`sec_compilation`) fasst die Kette zu einem einzigen Kernel zusammen. Dieser liest `x` einmal und schreibt die Ausgabe einmal.

<!--en:ebaa1b9d-->
### Matrixprodukte verschiedener Größen

<!--en:656764b8-->
Erreicht das Produkt quadratischer bf16-Matrizen die Spitzenrechenleistung, sobald $I = n / 3$ bei etwa $n = 500$ den Wert 164 überschreitet? Variieren wir $n$ von 256 bis 8192.

<!--en:79ab6795-->
Wir tragen den gemessenen Durchsatz in TFLOP/s gegen die arithmetische Intensität $n / 3$ auf und zeichnen zum Vergleich die Roofline-Schranke :eqref:`eq_roofline` ein.

<!--en:04a31947-->
Geben wir die genaue Latenz `t` pro Aufruf (in Mikrosekunden) und den Durchsatz `tf` für jede Matrixgröße `n` aus, erkennen wir die konstante Untergrenze der Laufzeit bei kleinen $n$:

<!--en:93d5fcbf-->
Bei $n = 4096$ und $8192$ erreicht das Produkt die Grenze von 165 TFLOP/s bis auf die 5%, um die der Takt schwankt. Bei kleinen Matrizen liegen die Punkte dagegen weit unterhalb der Roofline. In doppelt logarithmischer Darstellung steigen sie mit der Steigung 3 an. Das ist charakteristisch für eine konstante Zeit pro Aufruf bei einem Rechenaufwand proportional zu $n^3$. Von $n = 256$ bis $1024$ steigt der Aufwand um den Faktor $64\times$, die Zeit pro Aufruf ändert sich aber um weniger als den Faktor $2\times$.

<!--en:0e0869a9-->
Um herauszufinden, ob diese konstante Zeit auf dem Host oder dem Gerät verstreicht, messen wir die Laufzeit für ein Produkt der Größe $256 \times 256$ und für `x + 1` mit derselben Form. Mit der Hilfsfunktion `d2l.kernel_times`, die wir in :numref:`subsec_perf-profiler` entwickeln, messen wir, wie lange die GPU selbst pro Aufruf beschäftigt ist.

<!--en:21a340a2-->
In beiden Fällen arbeitet die GPU nur wenige Mikrosekunden. Die übrige Zeit entfällt auf den Start durch den Host, also die Zeit $h$ für Python, Framework und Treiber. Bei einer Matrizenmultiplikation der Größe $256 \times 256$ kostet die Python-Zeile, die sie anfordert, am meisten Zeit. Reihen wir Aufrufe unmittelbar nacheinander ein, arbeiten Host und Gerät überlappend. Pro Aufruf vergehen dann $\max(h, 2n^3/P)$. Bei kleinem $n$ ist das Gerät fertig, bevor der Host den nächsten Aufruf startet. Das Programm ist dann *durch die Startkosten begrenzt* (engl. overhead-bound). Der Übergang, ab dem die Rechnung länger dauert als der Start, liegt bei $n \approx (hP/2)^{1/3}$.

<!--en:6286e9f8-->
Für $h \approx 13$ µs liegt der Übergang bei etwa $n = 1{,}000$.

<!--en:df6dc6e8-->
Der Start einer einzelnen Operation dauert $h \approx 50\text{--}100$ µs. Für einen ganzen JIT-kompilierten Trainingsschritt fällt diese Zeit nur einmal an. Der Übergang liegt daher bei etwa $n = 1{,}600\text{--}2{,}000$.

<!--en:0d6e17db-->
Aus allen drei Untergrenzen ergibt sich unsere grundlegende Schranke für die Laufzeit einer einzeln gestarteten Operation:

<!--en:91a6772b-->
Bei $h = 10$ µs dauert jede Operation mit weniger als $hP \approx 1.6 \times 10^9$ FLOPs und weniger als $h\beta \approx 10$ MB übertragenen Daten nur so lange wie ihr Start. Außerdem wächst die Übergangsgröße $n \approx (hP/2)^{1/3}$ mit $P$. Schnellere GPUs brauchen also größere Operationen, damit die Startkosten die Laufzeit nicht begrenzen.

<!--en:bd6f952d-->
### Diagnose und Abhilfe

<!--en:f92155c2-->
Ein vollständiger Trainingsschritt umfasst alle drei Fälle. Welchem wir den Schritt zuordnen, hängt davon ab, worauf der größte Teil der Zeit entfällt (:numref:`fig_regimes`). :numref:`tab_perf_regimes` fasst für jeden Fall das typische Symptom und die üblichen Gegenmaßnahmen zusammen.

<!--en:147facc7-->
![Schematische Darstellung der drei Fälle. Welche Maßnahme einen Schritt beschleunigt, hängt davon ab, wie sich seine Laufzeit auf rechenbegrenzte Kernel, bandbreitenbegrenzte Kernel und Leerlauf verteilt. Jeder Balken nennt eine typische Maßnahme.](../img/mdl-perf-regimes.svg)

<!--en:1ae3d8f8-->
:Symptome und übliche Gegenmaßnahmen für die drei Fälle.

<!--en:e3bb30ac-->
| Fall | Symptom | Übliche Gegenmaßnahmen |

<!--en:f7a75e80-->
| rechenbegrenzt | Gerät ausgelastet, Intensität oberhalb des Knickpunkts, Laufzeit steigt mit der Anzahl der FLOPs | weniger Rechenoperationen, ein Zahlenformat mit schnelleren Tensor Cores, mehr Geräte |

<!--en:eba49ef1-->
| bandbreitenbegrenzt | Gerät ausgelastet, Intensität unterhalb des Knickpunkts, Laufzeit steigt mit der Anzahl übertragener Bytes | größerer Batch für eine höhere Intensität der Produkte, schmaleres Zahlenformat, Fusion, Neuberechnung statt Speicherung, Algorithmen mit Wiederverwendung der Daten auf dem Chip |

<!--en:b7d24a65-->
| durch Startkosten begrenzt | Gerät wartet zwischen Kerneln, Zeit pro Aufruf unabhängig von der Größe | größerer Batch für weniger Aufrufe pro Beispiel, gesamtes Programm aufzeichnen und ohne Python erneut ausführen |

<!--en:fd55bd6d-->
## Laufzeitprofil eines Trainingsschritts

<!--en:f0375286-->
Ein *Profiler* zeichnet die Dauer jedes GPU-Kernels auf. So können wir die Zeit, in der das Gerät arbeitet, mit der Gesamtlaufzeit vergleichen und nach Kernel-Typ aufschlüsseln. Sagen wir vor der Messung die Laufzeit eines SGD-Schritts für ein MLP ($4096 \to 4096 \to 4096 \to 10$) vorher. Der Minibatch umfasst 1024 Beispiele in fp32, und die Matrixprodukte verwenden tf32 ($P = 82.6$ TFLOP/s).

<!--en:c6a296af-->
### Vorhersage der Schrittdauer

<!--en:38133422-->
Jede der beiden Schichten der Größe $4096 \times 4096$ benötigt $2 \cdot 1024 \cdot 4096^2 \approx 3.44 \times 10^{10}$ FLOPs für ein Matrixprodukt im Vorwärtsdurchlauf. Ein Trainingsschritt berechnet fünf solche Produkte: zwei im Vorwärtsdurchlauf, zwei für die Gewichtsgradienten und eines für den Eingabegradienten der zweiten versteckten Schicht. Bei $I \approx 340$ FLOP/Byte (oberhalb des tf32-Knickpunkts von 82) dauert jedes Produkt $3.44 \times 10^{10} / (82.6 \times 10^{12}) \approx 0.42$ ms. Die SGD-Aktualisierung liest jedes der 33.6 Millionen fp32-Gewichte und seinen Gradienten und schreibt das neue Gewicht ($12 \times 33.6 \times 10^6 \approx 0.40$ GB). Bei 1 TB/s dauert das rund 0.40 ms. Vernachlässigen wir die kleine Ausgabeschicht und die ReLUs, ergibt sich für die Schrittdauer

<!--en:4aa48a5d-->
`kernel_times` führt eine Funktion unter dem Profiler des Frameworks `steps`-mal aus. Für jeden gestarteten Kernel gibt die Funktion zurück, wie viele Millisekunden das Gerät pro Aufruf im Mittel benötigt.

<!--en:670d6498-->
Der Profiler des Frameworks kann auch den zeitlichen Ablauf aufzeichnen und exportieren. Die Aufzeichnung lässt sich unter <https://ui.perfetto.dev> ansehen.

<!--en:33c9ecc7-->
`prof.export_chrome_trace` schreibt in `kernel_times` die zeitliche Aufzeichnung des Profils.

<!--en:ba509a88-->
`jax.profiler.trace` schreibt die Aufzeichnung in das dafür vorgesehene Verzeichnis.

<!--en:161ec600-->
### Auswertung des Laufzeitprofils

<!--en:35a8955c-->
`show_profile` vergleicht die Arbeitszeit des Geräts mit der Gesamtdauer des Schritts, berechnet den Zeitanteil der Kernel für Matrixprodukte und listet die Kernel mit den größten Zeitanteilen auf.

<!--en:8c644e58-->
Wir wenden `show_profile` nun auf die Kernel-Laufzeiten für einen Schritt unseres dreischichtigen MLP an (`kernel_times(step)`). Die Arbeitszeit des Geräts vergleichen wir mit der gemessenen Gesamtdauer `1000 * wall`:

<!--en:46de3966-->
Die gemessene Schrittdauer weicht um höchstens etwa 10% von den vorhergesagten 2.5 ms ab. Die GPU arbeitet fast den gesamten Schritt über. Mehr als drei Viertel ihrer Arbeitszeit entfallen auf Matrixprodukte (*gemm*-Kernel).

<!--en:24b833bb-->
Die drei `gemm`-Zeilen fassen die fünf großen Matrixprodukte zusammen. `multi_tensor_apply_kernel` führt die SGD-Aktualisierung im Batch in den vorhergesagten 0.4 ms aus.

<!--en:39fb1ca7-->
Beachten Sie, dass eine separate Aktualisierung durch den Optimierer unabhängig von der Batchgröße eine feste Zeit benötigt. Bei Batchgröße 128 sinkt die Laufzeit der fünf Produkte auf jeweils rund 0.07 ms (die Produkte sind dann bandbreitenbegrenzt). Damit dauert die Aktualisierung mit 0.4 ms länger als alle fünf Produkte zusammen.

<!--en:844ba5af-->
XLA integriert die SGD-Aktualisierung direkt in die Produkte für die Gewichtsgradienten (`gemm_fusion_dot`). Vor dem Schreiben des aktualisierten Gewichts zieht XLA den skalierten Gradienten ab. Damit entfallen das separate Schreiben und erneute Lesen des Gradienten.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:61a21e71-->
Für die Laufzeit einer GPU-Operation gilt $t \gtrsim \max(h,\, \textrm{FLOPs}/P,\, \textrm{bytes}/\beta)$. Dabei ist $h$ der Zeitaufwand für den Start durch den Host, $P$ die Spitzenrechenleistung und $\beta$ die Speicherbandbreite. Die arithmetische Intensität $I = \textrm{FLOPs}/\textrm{bytes}$ setzt Rechenarbeit und Datenübertragung ins Verhältnis. Unterhalb des Knickpunkts $I^* = P/\beta$ (rund 164 FLOP/Byte für bf16 auf der RTX 4090) ist eine Operation bandbreitenbegrenzt, oberhalb ist sie rechenbegrenzt. Bei einer linearen Schicht wächst die Intensität mit der Batchgröße von rund 1 FLOP/Byte bei der Dekodierung eines einzelnen Tokens auf Hunderte FLOP/Byte im Training. Genaue Laufzeitmessungen erfordern eine Synchronisierung des Geräts außerhalb der Schleifen und Aufwärmläufe (`d2l.Benchmark`). Mit `d2l.kernel_times` und `d2l.show_profile` prüfen wir die rechnerischen Vorhersagen für jeden einzelnen Kernel.

<!--en:2294243a-->
## Übungen

<!--en:681f09d4-->
1. **Die kleinste Dimension.** Der exakte Ausdruck für die Intensität des Produkts $\mathbf{X}\mathbf{W}$ in :eqref:`eq_matmul_intensity` gilt für alle Formen, solange die Näherung für $B \ll D, F$ noch nicht angewendet wurde. Sei $m = \min(B, D, F)$.
    1. Zeigen Sie, dass die Intensität zwischen $2m/(3b)$ und $2m/b$ liegt. Folgern Sie daraus für eine GPU mit einem Knickpunkt bei 164 FLOP/Byte, dass im Roofline-Modell jedes bf16-Matrixprodukt mit einer Dimension unter 164 bandbreitenbegrenzt ist.
    1. Wenden Sie die Schranke auf drei bf16-Produkte an: die letzte Schicht des MLP aus :numref:`subsec_perf-profiler` ($4096 \to 10$ bei einer Batchgröße von 4096), die Attention-Scores $\mathbf{Q}\mathbf{K}^\top$ eines Kopfes mit $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{T \times 64}$ und eine $1 \times 1$-Faltung (:numref:`subsec_1x1`) von 64 auf 64 Kanäle für einen Batch aus 128 Bildern mit je $32 \times 32$ Pixeln. Welche Dimensionen müssten jeweils wachsen, damit die Produkte den Knickpunkt erreichen?
    1. Eine Multi-Head-Attention-Schicht berechnet die Scores aller Köpfe gemeinsam in einem Kernel. Ändert dieses Batching der Köpfe die arithmetische Intensität der Score-Berechnung?

<!--en:68357fd0-->
2. [code] **Batchgröße und Knickpunkt.** Betrachten Sie das Produkt $\mathbf{X}\mathbf{W}$ mit $\mathbf{X} \in \mathbb{R}^{B \times 8192}$ und $\mathbf{W} \in \mathbb{R}^{8192 \times 8192}$ in bf16.
    1. Bestimmen Sie mit dem exakten Ausdruck aus :eqref:`eq_matmul_intensity` die Batchgröße, bei der das Produkt den Knickpunkt Ihrer GPU erreicht. Vergleichen Sie das Ergebnis mit der Näherung für kleine $B$.
    1. Sagen Sie mit :eqref:`eq_perf-time` die Laufzeit des Produkts für $B = 1$ und $B = 64$ gleichzeitig dekodierte Sequenzen vorher. Messen Sie beide Zeiten mit `d2l.Benchmark`. Bestimmen Sie, bis zu welcher Batchgröße zusätzliche Sequenzen pro Schritt fast keine zusätzliche Zeit kosten.

<!--en:08624ba7-->
3. [code] **Verborgene Barrieren.** Jede der folgenden Zeilen läuft einmal pro Schritt einer Trainingsschleife. Dabei sind `x` und `g` GPU-Tensoren, und `c` ist eine Python-Gleitkommazahl: `x[x > 0].sum()`, `print(x.shape)`, `x.argmax()`, `if g.norm() > c: g *= c / g.norm()` und `x * torch.tensor(0.5, device=x.device)`.
    1. Entscheiden Sie für jede Zeile, ob der Host auf das Gerät warten muss, und begründen Sie Ihre Antwort.
    1. Schreiben Sie jede Zeile, die eine Synchronisierung auslöst, so um, dass sie dasselbe Ergebnis ohne Barriere auf dem Host berechnet.
    1. Prüfen Sie Ihre Antworten mit `torch.cuda.set_sync_debug_mode('warn')`.

<!--en:229fb9ef-->
3. [code] **Verborgene Barrieren.** Jede der folgenden Zeilen läuft einmal pro Schritt einer Trainingsschleife. Dabei sind `x` und `g` GPU-Arrays, und `c` ist eine Python-Gleitkommazahl: `x[x > 0].sum()`, `print(x.shape)`, `jnp.argmax(x)`, `if jnp.linalg.norm(g) > c: g = g * c / jnp.linalg.norm(g)` und `x * jnp.asarray(0.5)`.
    1. Entscheiden Sie für jede Zeile, ob der Host auf das Gerät warten muss, und begründen Sie Ihre Antwort.
    1. Schreiben Sie jede Zeile, die eine Synchronisierung auslöst, so um, dass sie dasselbe Ergebnis ohne Barriere auf dem Host berechnet.
    1. Prüfen Sie Ihre Antworten innerhalb von `with jax.transfer_guard_device_to_host('disallow'):`.

<!--en:a344c0b2-->
4. **Diagnose anhand von Laufzeiten.** Angenommen, `d2l.Benchmark` liefert für vier bf16-Operationen auf der RTX 4090 ($P = 165$ TFLOP/s, $\beta = 1.008$ TB/s) die folgenden Zeiten pro Aufruf. Operation A multipliziert zwei Matrizen der Form $8192 \times 8192$ in 7.4 ms. Operation B wendet die Softmax-Funktion zeilenweise auf eine Matrix der Form $8192 \times 8192$ an und braucht dafür 0.30 ms (rund 5 FLOPs pro Element, ein Lese- und ein Schreibzugriff). Operation C multipliziert zwei Matrizen der Form $256 \times 256$ in 10 µs, Operation D zwei Matrizen der Form $1536 \times 1536$ in 0.10 ms.
    1. Berechnen Sie für jede Operation die arithmetische Intensität und die Anteile von $P$ und $\beta$, die sie erreicht. Ordnen Sie jede Operation nach zwei Kriterien ein: der Lage ihrer Intensität relativ zum Knickpunkt und der Grenze, der sich ihr Durchsatz nähert. Wo unterscheiden sich die beiden Einordnungen?
    1. Betrachten Sie jede Operation, die keiner der beiden Grenzen nahekommt. Unterscheiden Sie zwei mögliche Ursachen: die Latenz $h$ für den Start durch den Host und eine zu geringe Anzahl von Kacheln für die Thread-Blöcke, um alle Streaming-Multiprozessoren auszulasten. Was würde das Zeitdiagramm eines Profilers jeweils zeigen?

<!--en:cfefce55-->
5. [code] **Ein Pipeline-Modell für den Start durch den Host.** ● Modellieren Sie einen GPU-Aufruf mit drei Zeitspannen: der Zeit $h$ für das Einreihen durch den Host, der Ausführungszeit $t_\textrm{dev}$ auf dem Gerät und der Latenz $\sigma$ der abschließenden Barriere.
    1. Zeigen Sie, dass $R$ unmittelbar nacheinander eingereihte Aufrufe mit einer einzigen anschließenden Barriere insgesamt $R \max(h, t_\textrm{dev}) + \min(h, t_\textrm{dev}) + \sigma$ dauern. Welche Zeit liefert `d2l.Benchmark` für $t_\textrm{dev} < h$?
    1. Betrachten Sie die Schleifen aus :numref:`subsec_perf-measuring`. Wie verhält sich die Laufzeit der Schleife, die in jedem Schritt liest, zur Laufzeit der Schleife, die nur einmal liest? Leiten Sie das Verhältnis in Abhängigkeit von $h$, $t_\textrm{dev}$ und $\sigma$ her.
    1. Führen Sie beide Schleifen für Matrizen mit den Seitenlängen $256, 512, 1024, 2048, 4096$ aus. Schätzen Sie $h$, $\sigma$ und $t_\textrm{dev}$. Bestimmen Sie, ab welcher Ausführungszeit auf dem Gerät pro Schritt das Lesen eines Skalars in jedem Schritt weniger als 10% zusätzliche Zeit kostet.

<!--en:be94d524-->
6. [code] **Kacheln und Wellen.** ● Eine GPU besteht aus $n_\textrm{SM}$ Streaming-Multiprozessoren (SMs, $n_\textrm{SM} = 128$ auf der RTX 4090). Nehmen Sie an, ein Kernel für Matrixprodukte zerlegt eine Ausgabe der Form $n \times n$ in Kacheln der Form $128 \times 128$. Jeder SM berechnet jeweils eine Kachel mit der Rate $P / n_\textrm{SM}$. Die Berechnung läuft in Wellen von höchstens $n_\textrm{SM}$ Kacheln ab.
    1. Leiten Sie den nach diesem Modell erreichbaren Anteil am Spitzendurchsatz als Funktion von $n$ und $n_\textrm{SM}$ her. Vergleichen Sie das Ergebnis mit Ihrer Messreihe in :numref:`subsec_perf-sweep`.
    1. Sagen Sie den Anteil am Spitzendurchsatz für $n = 2048, 2176, 2304,
       2432, 2560$ vorher (aufeinanderfolgende Vielfache von 128) und messen Sie ihn. Erklären Sie, wo die Messungen vom einfachen Wellenmodell abweichen.
    1. Beschreiben Sie zwei Möglichkeiten, wie ein Kernel ein Produkt mit $n = 512$ in mehr Arbeitseinheiten zerlegen kann, sodass alle SMs mitrechnen. Geben Sie jeweils den zusätzlichen Aufwand an.

<!--en:30a6603f-->
[Diskussion](https://d2l.discourse.group/t/36787)

<!--en:55c4c1bd-->
Das Roofline-Modell: Grenzen durch Rechenleistung, Bandbreite und Startkosten<br>**arithmetische Intensität · Roofline-Schranke · Knickpunkt · Zeitmessung bei asynchroner Ausführung · Laufzeitprofile von Kerneln**

<!--en:a0f57dd4-->
::: {.slide title="Warum erreichen Matrixprodukte mal 2% und mal 100% der Spitzenleistung?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:45eea63b-->
- **Großes quadratisches Produkt ($8192 \times 8192$):** Erreicht dauerhaft fast $100\%$ der Spitzenleistung der RTX 4090 von $165.2\text{ TFLOP/s}$ in `bfloat16`.

<!--en:d376f65f-->
- **Kleines quadratisches Produkt ($256 \times 256$):** Erreicht **weniger als $2\%$ der Spitzenleistung**. Bei derselben Grundoperation sinkt die Auslastung um mehr als den Faktor $50\times$.

<!--en:43afe09e-->
- **Bedeutung für die Praxis:** Der Nutzen einer schnelleren GPU oder einer zweiten Karte hängt davon ab, welcher Engpass den Schritt begrenzt: **Rechenleistung ($P$)**, **Speicherbandbreite ($\beta$)** oder **Start durch den Host ($h$)**.

<!--en:1a7692d0-->
::: {.slide title="FLOPs und Bytes zählen, arithmetische Intensität bestimmen"}

<!--en:de2b3858-->
[Arithmetische Intensität]{.kicker}

<!--en:38e385eb-->
- **Unterschied zwischen Speicher und Recheneinheiten:** Die Übertragung von $1\text{ B}$ aus dem DRAM dauert $1/\beta \approx 1\text{ ps}$. In dieser Pikosekunde schaffen die Tensor Cores $P/\beta \approx 164\text{ FLOPs}$.

<!--en:0e1eaa35-->
- **Elementweiser Benchmark ($y_i = x_i + 1$):** Pro Element fällt $1\text{ FLOP}$ an. Die Operation liest $2\text{ B}$ und schreibt $2\text{ B}$ in `bf16`. Das ergibt unabhängig von der Größe $I = 0.25\text{ FLOP/B}$.

<!--en:20fbc4d8-->
::: {.slide title="Arithmetische Intensität der Matrizenmultiplikation in einer linearen Schicht"}

<!--en:d03bbf68-->
[Matrizenmultiplikation]{.kicker}

<!--en:55680adb-->
- **Die kleinste Dimension entscheidet:** Die kleinste der drei Größen Tokenzahl $B$ im Batch, Eingabebreite $D$ und Ausgabebreite $F$ begrenzt die Wiederverwendung der Daten.

<!--en:3e19d291-->
- **Wiederverwendung der Gewichte im Batch:** Für $B \ll D, F$ entfällt der größte Teil des DRAM-Verkehrs auf das Lesen von $\mathbf{W}$ ($bDF$ Byte). Jedes Gewicht kommt in allen $B$ Zeilen des Batches zum Einsatz.

<!--en:25f52557-->
::: {.slide title="Die Intensität einer GPT-Schicht variiert um den Faktor 1000"}

<!--en:6644cf3b-->
[Training und Dekodierung]{.kicker}

<!--en:21d67f0d-->
- **Dekodierung eines einzelnen Tokens ($B = 1$):** Die GPU liest $8.4\text{ MB}$ an Gewichten und verwendet jedes Gewicht nur einmal ($I \approx 1.0\text{ FLOP/B}$, stark bandbreitenbegrenzt).

<!--en:288741ad-->
- **Dekodierung im Batch ($B = 64$):** Dieselben Gewichte dienen $64$ parallelen Sequenzen ($I \approx 60\text{ FLOP/B}$). Bei gleicher Schrittdauer steigt der Durchsatz auf $\sim 60\times$ den bisherigen Wert.

<!--en:607f77ec-->
- **Training ($B = 8{,}192$):** Die Schichtbreite $D = 2048$ ist jetzt die kleinste Dimension ($I \approx 910 \gg I^* = 164$). Die Tensor Cores sind ausgelastet.

<!--en:55d8f406-->
::: {.slide title="Roofline-Schranke und Knickpunkt"}

<!--en:eacee862-->
[Das Roofline-Modell]{.kicker}

<!--en:119ec426-->
- Der Durchsatz einer Operation mit arithmetischer Intensität $I$ ist sowohl durch die Bereitstellung der Operanden aus dem Speicher ($I\beta$) als auch durch die Leistung der Recheneinheiten ($P$) begrenzt:
  $$\textrm{Throughput} \;\leq\; \min(P,\; I \beta)$$

<!--en:8bb7fee2-->
- **Unterhalb des Knickpunkts ($I < I^*$):** Der Kernel ist **bandbreitenbegrenzt**. Die Recheneinheiten warten auf den DRAM. Zusätzliche Rechnungen im selben Kernel mit bereits geladenen Registerwerten kosten daher fast keine Zeit.

<!--en:f3d74edc-->
- **Oberhalb des Knickpunkts ($I > I^*$):** Der Kernel ist **rechenbegrenzt**. Die Recheneinheiten sind ausgelastet. Nur schnellere Zahlenformate oder weniger FLOPs helfen.

<!--en:a5c4d808-->
::: {.slide title="Knickpunkt der RTX 4090 bei 164 FLOP/Byte in bf16"}

<!--en:f6875d79-->
[Schwellenwerte der Hardware]{.kicker}

<!--en:60d2d297-->
Der **Knickpunkt** $I^* = P/\beta$ gibt an, welche arithmetische Intensität mindestens erforderlich ist, um die Spitzenrechenleistung zu erreichen:

<!--en:dd156e83-->
- **Beschleuniger in Rechenzentren haben noch höhere Knickpunkte:** Auf einer H100 ($P = 989\text{ TFLOP/s}$, $\beta = 3.35\text{ TB/s}$) gilt $I^* \approx 295\text{ FLOP/byte}$. Die FLOP/s der Tensor Cores sind schneller gestiegen als die HBM-Bandbreite.

<!--en:7e57fa08-->
- **Quadratische Matrizenmultiplikation ($B = D = F = n$ in `bf16`):** Es gilt $I = \frac{2n^3}{2(3n^2)} = \frac{n}{3}$. Daher sagt das Roofline-Modell voraus, dass quadratische Produkte für $n \gtrsim 3 I^* \approx 500$ rechenbegrenzt sind.

<!--en:5ff51373-->
- **Einfluss des Caches:** Hier bezeichnet $\beta$ die Bandbreite des externen DRAM. Passen die Operanden in den $72\text{ MB}$ großen L2-Cache auf dem Chip, kann der Durchsatz die Bandbreitengrenze des DRAM übersteigen.

<!--en:bb9e2b45-->
::: {.slide title="Asynchroner Start und die Tücken der GPU-Zeitmessung"}

<!--en:1dd824ef-->
[Genaue Laufzeitmessung]{.kicker}

<!--en:d1b9b025-->
- **Asynchrone Ausführung:** Python reiht Kernel-Deskriptoren in die Warteschlange des GPU-Treibers ein. Während die GPU diese abarbeitet, fährt Python sofort mit der nächsten Anweisung fort.

<!--en:3352965d-->
- **Warum einfache Zeitmessungen täuschen:** Wer `time.perf_counter()` vor und nach einem unsynchronisierten Kernel-Aufruf abfragt, misst nur die Zeit $h$ für das Einreihen auf der CPU ($\sim 10\,\mu\text{s}$). Die Ausführungszeit $t_{\textrm{dev}}$ auf der GPU bleibt unberücksichtigt.

<!--en:fd25c35b-->
::: {.slide title="Verborgene Barrieren zwischen Host und Gerät erzwingen eine serielle Ausführung"}

<!--en:c3c23369-->
[Fallstricke bei der Synchronisierung]{.kicker}

<!--en:bbf5afdb-->
Fordert Python einen Tensorwert von der GPU an, wartet der Host, bis alle eingereihten Kernel fertig sind. Die Übertragung vom Gerät zum Host verursacht die zusätzliche Latenz $\sigma$. Host und Gerät können nicht mehr überlappend arbeiten:

<!--en:d1756fd0-->
**Verborgene Barrieren in Trainingsschleifen:** `float(loss)`, `loss.item()`, `print(x)`, Verzweigungen auf dem Host wie `if loss < 0.1:` und datenabhängige Formen (`x[x > 0]`). Metriken stets auf der GPU aufsummieren und nur zu den Protokollierungszeitpunkten synchronisieren.

<!--en:5e830498-->
::: {.slide title="Genaue Laufzeitmessung mit Aufwärmläufen und Barrieren"}

<!--en:40595b68-->
[Vorgehen bei der Messung]{.kicker}

<!--en:25919570-->
- `d2l.Benchmark` beginnt mit **Aufwärmdurchläufen**. So fallen die JIT-Kompilierung, das Anlegen des cuBLAS-Handles und das Laden der Kernel vor der Messung an.

<!--en:9ce867ea-->
- Für die Zeitmessung über `repeats` Aufrufe setzt das Werkzeug je eine explizite **Synchronisierungsbarriere auf dem Gerät** vor Beginn und Ende der Messung.

<!--en:7019f73f-->
- **Überprüfung (Matrizenmultiplikation der Größe $4096 \times 4096$ in `bf16`):** Bei einer Intensität von $I = 4096/3 \approx 1365 \gg 164$ sagt die Roofline $2 \cdot 4096^3 / P \approx 0.83\text{ ms}$ vorher. Die Abweichung vom Messwert liegt bei etwa $\sim 10\%$ oder darunter.

<!--en:a49884c2-->
::: {.slide title="Über den Knickpunkt hinweg erreicht eine Matrix beide Grenzen"}

<!--en:d1125891-->
[Bandbreitengrenze und Grenze der Rechenleistung]{.kicker}

<!--en:3ce085b6-->
Das Produkt aus $B$ Aktivierungszeilen und einer Gewichtsmatrix der Form $16384 \times 16384$ in `bfloat16` ($512\text{ MiB}$) hat die Intensität $I \approx B$. Sie überschreitet den Knickpunkt $I^* = 164$ zwischen $B = 128$ und $B = 256$:

<!--en:622ad622-->
Von $B = 1$ bis $B = 128$ bleibt die Laufzeit fast konstant bei $0.6\text{ ms}$ (der Zeit für das Lesen von $512\text{ MiB}$ bei $\sim 0.9\text{ TB/s}$), obwohl die Rechenarbeit um den Faktor $128\times$ steigt. Oberhalb von $B = 256$ wächst die Laufzeit linear mit $B$ bei nahezu maximaler Rechenleistung.

<!--en:8ace8f34-->
::: {.slide title="Elementweise Kernel erreichen die Bandbreitengrenze"}

<!--en:1210151a-->
[Bandbreitenbegrenzte Operationen]{.kicker}

<!--en:a878f070-->
Auf einer Matrix der Form $8192 \times 8192$ in `float32` ($256\text{ MiB}$) lesen die einstelligen Operationen (`x + 1`, `x * 1.5`, `sin(x)`, `sigmoid(x)`) jeweils $256\text{ MiB}$ und schreiben $256\text{ MiB}$ (insgesamt $512\text{ MiB}$, also $0.53\text{ ms}$ bei $1.008\text{ TB/s}$):

<!--en:3011fd20-->
- **An der Bandbreitengrenze kostet zusätzliche Rechenarbeit keine Zeit:** Wegen $I \le 1 \ll 82\text{ FLOP/byte}$ dauern transzendente Operationen (`sin`, `sigmoid`) mit $\sim 0.6\text{ ms}$ ($\sim 0.9\text{ TB/s}$) genauso lange wie eine einfache Addition.

<!--en:752373a8-->
::: {.slide title="Verkettete elementweise Operationen vervielfachen bei sofortiger Ausführung den Speicherverkehr"}

<!--en:fbe7a6e9-->
[Ketten ohne Fusion]{.kicker}

<!--en:eaeac70e-->
Bei sofortiger Ausführung startet jede der 8 Operationen der GELU-Näherung `gelu_ish` einen eigenen Kernel. Jeder Kernel liest seine Eingaben aus dem DRAM und schreibt seine Ausgabe in den DRAM zurück:

<!--en:c512ce43-->
Sechs einstellige Operationen übertragen $6 \times 2 = 12$ Tensoren, zwei zweistellige weitere $2 \times 3 = 6$ Tensoren ($18$ Array-Übertragungen gegenüber $2$ für `x + 1`). Die Kette braucht bei sofortiger Ausführung **um den Faktor $8\text{--}9\times$ länger**, bis ein Compiler sie zu einem Kernel zusammenfasst (§{sec_compilation}).

<!--en:e07742ae-->
::: {.slide title="Kleine Matrixprodukte bleiben weit unter der Roofline"}

<!--en:a570da7c-->
[Messreihe über mehrere Matrixgrößen]{.kicker}

<!--en:0ca3d57e-->
- Quadratische `bfloat16`-Produkte von $n = 256$ bis $8192$ zeigen, ob die Recheneinheiten ab der theoretischen Schwelle $n \approx 500$ ($I = n/3 = 164$) ausgelastet sind.

<!--en:da8ed78c-->
- Bei $n = 4096$ und $8192$ nähert sich der gemessene Durchsatz der Grenze von $165\text{ TFLOP/s}$.

<!--en:15b862de-->
- Bei kleinem $n$ liegt der Durchsatz dagegen weit unter der Roofline. Er steigt **in doppelt logarithmischer Darstellung mit Steigung $3$**. Das ist typisch für eine konstante Zeit pro Aufruf, während die FLOPs proportional zu $2n^3$ zunehmen.

<!--en:baeee84b-->
::: {.slide title="Bei kleinen Operationen dominiert die Startlatenz auf dem Host"}

<!--en:64caa0c3-->
[Begrenzung durch Startkosten]{.kicker}

<!--en:9f5664f3-->
Der Vergleich der Gesamtlaufzeit pro Aufruf (`d2l.Benchmark`) mit der tatsächlichen Ausführungszeit des GPU-Kernels (`d2l.kernel_times`) zeigt für einen Tensor der Form $256 \times 256$ in `bfloat16`, worauf die konstante Zeit entfällt:

<!--en:9784eace-->
Die GPU benötigt für beide Operationen jeweils nur $1\text{--}4\,\mu\text{s}$. Die restlichen $10\text{--}100\,\mu\text{s}$ entfallen auf die **Latenz $h$ für den Start durch den Host** (Python-Interpreter, Aufrufverwaltung des Frameworks und Start durch den CUDA-Treiber). Bei einem Produkt der Größe $256 \times 256$ kostet die Python-Zeile, die es anfordert, am meisten Zeit.

<!--en:bb99d80f-->
::: {.slide title="Drei Untergrenzen für die Laufzeit und der Übergang zur Rechenbegrenzung" only="pytorch"}

<!--en:2fc8e397-->
[Gemeinsames Laufzeitmodell]{.kicker}

<!--en:67141a66-->
- **Die größte Untergrenze bestimmt den Fall:** begrenzt durch Startkosten ($h$), Rechenleistung ($\textrm{FLOPs}/P$) oder Bandbreite ($\textrm{bytes}/\beta$).

<!--en:3cab3273-->
- **Übergangsgröße ($n^* \approx (hP/2)^{1/3}$):** Bei quadratischen Matrizen dauert die Rechnung ab $n^* \approx 1{,}000$ länger als der Start durch den Host ($h \approx 13\,\mu\text{s}$ pro Operation).

<!--en:12741b2c-->
- **Schnellere GPUs brauchen breitere Matrizen:** Steigt die GPU-Geschwindigkeit um den Faktor $14\times$, wächst $n^*$ um den Faktor $14^{1/3} \approx 2.4\times$.

<!--en:c08eec94-->
::: {.slide title="Drei Untergrenzen für die Laufzeit und der Übergang zur Rechenbegrenzung" only="jax"}

<!--en:2fc8e397-2-->
[Gemeinsames Laufzeitmodell]{.kicker}

<!--en:67141a66-2-->
- **Die größte Untergrenze bestimmt den Fall:** begrenzt durch Startkosten ($h$), Rechenleistung ($\textrm{FLOPs}/P$) oder Bandbreite ($\textrm{bytes}/\beta$).

<!--en:9f5d1c60-->
- **Übergangsgröße ($n^* \approx (hP/2)^{1/3}$):** Beim Einzelstart quadratischer Matrizenmultiplikationen dauert die Rechnung ab $n^* \approx 1{,}800$ länger als der Start durch den Host ($h \approx 70\,\mu\text{s}$).

<!--en:12741b2c-2-->
- **Schnellere GPUs brauchen breitere Matrizen:** Steigt die GPU-Geschwindigkeit um den Faktor $14\times$, wächst $n^*$ um den Faktor $14^{1/3} \approx 2.4\times$.

<!--en:dbb1d5dd-->
::: {.slide title="Die drei Engpässe erkennen und beheben"}

<!--en:3b07d39f-->
[Diagnose und Abhilfe]{.kicker}

<!--en:2e67c36c-->
| Fall | Befund im Laufzeitprofil und bei der Zeitmessung | Übliche Gegenmaßnahmen |

<!--en:5620b02e-->
| **Rechenbegrenzt** | GPU zu $\approx 100\%$ ausgelastet, $I > I^*$, Laufzeit $\propto \textrm{FLOPs}$ | geringere Rechengenauigkeit (`bf16`/`fp8`), weniger FLOPs, mehr GPUs |

<!--en:c7a9d01d-->
| **Bandbreitenbegrenzt** | GPU zu $\approx 100\%$ ausgelastet, $I < I^*$, Laufzeit $\propto \textrm{bytes}$ | Operatorfusion, größerer Batch $B$, schmalere Datentypen, Verarbeitung in Kacheln auf dem Chip |

<!--en:da23afbf-->
| **Durch Startkosten begrenzt** | GPU wartet zwischen Kerneln, Laufzeit $\approx h$ unabhängig von der Größe | gesamten Schritt kompilieren, CUDA-Graph erneut ausführen, größere Batches |

<!--en:62f83cfc-->
::: {.slide title="Vorhersage und Laufzeitprofil eines vollständigen MLP-Trainingsschritts"}

<!--en:3f240642-->
[Zeitaufwand pro Schritt]{.kicker}

<!--en:44f325a6-->
Ein SGD-Schritt eines MLP mit 3 Schichten ($4096 \to 4096 \to 4096 \to 10$) mit Batchgröße $B = 1024$ in `float32` auf `tf32`-Tensor-Cores ($P = 82.6\text{ TFLOP/s}$, $I^* = 82\text{ FLOP/byte}$):

<!--en:5730e2a8-->
- **Rechnerische Vorhersage ($2.5\text{ ms}$):** Fünf große Matrizenmultiplikationen der Form $1024 \times 4096 \times 4096$ (2 im Vorwärtsdurchlauf, 2 für Gewichtsgradienten, 1 für Eingabegradienten) dauern bei $I \approx 340 > 82$ insgesamt $5 \times \frac{2 \cdot 1024 \cdot 4096^2}{82.6 \times 10^{12}} = 5 \times 0.42 = 2.1\text{ ms}$. Hinzu kommen $0.40\text{ ms}$ für die Übertragung von $12\text{ B} \times 33.6\text{M} = 0.40\text{ GB}$ bei der SGD-Aktualisierung.

<!--en:967ba346-->
Das Profil bestätigt unsere rechnerische Vorhersage von $2.5\text{ ms}$ bis auf $10\%$. Die GPU arbeitet während fast $100\%$ des Schritts. Auf `gemm`-Kernel entfallen $> 75\%$ der Arbeitszeit des Geräts.

<!--en:d40a784f-->
::: {.slide title="Verständnisfragen zur Roofline-Schranke und zur Begrenzung durch Startkosten"}

<!--en:929b7671-->
[Wissenskontrolle]{.kicker}

<!--en:ec5c0e37-->
Gegeben ist eine RTX 4090 (`bfloat16`-Spitzenleistung $P = 165\text{ TFLOP/s}$, Bandbreite $\beta = 1.0\text{ TB/s}$, Knickpunkt $I^* = 164\text{ FLOP/byte}$, Start durch den Host $h = 10\,\mu\text{s}$). Welche Untergrenze greift für jede der folgenden `bfloat16`-Operationen, und wie lange dauert sie mindestens?

<!--en:3e3c3a62-->
1. Elementweise Addition `x + 1` für $10^8$ Zahlen ($200\text{ MB}$ lesen und $200\text{ MB}$ schreiben).

<!--en:2d1090e2-->
2. Anwendung einer linearen Schicht der Form $2048 \times 2048$ ($8.4\text{ MB}$ an Gewichten) auf ein einzelnes Token ($B = 1$).

<!--en:4d5bd2c6-->
3. Multiplikation zweier Matrizen der Form $4096 \times 4096$ ($1.37 \times 10^{11}\text{ FLOPs}$, insgesamt $100\text{ MB}$ Datenverkehr).

<!--en:557b98b6-->
**Lösungen:**

<!--en:53873e0a-->
1. **Bandbreitenbegrenzt ($I = 0.25 < 164$):** Übertragung von $4 \times 10^8\text{ B}$ bei $1.0\text{ TB/s}$ $\to$ **$t \ge 0.40\text{ ms}$** ($\gg h = 10\,\mu\text{s}$).

<!--en:b731650f-->
2. **Durch Startkosten begrenzt ($h > \textrm{bytes}/\beta > \textrm{FLOPs}/P$):** Die Übertragung von $8.4\text{ MB}$ dauert $8.4\,\mu\text{s}$, die Rechnung $0.05\,\mu\text{s}$. Der Start durch den Host mit der Dauer $h$ bestimmt die Laufzeit $\to$ **$t \ge 10\,\mu\text{s}$**.

<!--en:fa2e8759-->
3. **Rechenbegrenzt ($I = 1365 > 164$):** $1.37 \times 10^{11}\text{ FLOPs}$ bei $165\text{ TFLOP/s}$ $\to$ **$t \ge 0.83\text{ ms}$** (gegenüber $0.10\text{ ms}$ für den Speicherzugriff).

<!--en:06869d16-->
::: {.slide title="Zusammenfassung der Roofline und der drei begrenzenden Faktoren"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:1c7d1f3f-->
| Begriff oder Größe | Formel oder Definition | Wert auf der RTX 4090 | Bedeutung für die Ausführung im Deep Learning |

<!--en:54e5c391-->
| **Arithmetische Intensität ($I$)** | $\textrm{FLOPs} / \textrm{bytes moved}$ | $0.25$ (Addition) bis $> 10^3$ (GEMM) | Unterscheidet Operationen, die auf den Speicher warten, von solchen, die die Recheneinheiten auslasten |

<!--en:68af5e41-->
| **Intensität der Matrizenmultiplikation ($\mathbf{X}\mathbf{W}$)** | $\approx 2B/b$ für $B \ll D, F$ | $\sim 1$ ($B=1$) bis $910$ ($B=8192$) | Erklärt, warum Batching lineare Schichten beschleunigt |

<!--en:32ee85ff-->
| **Roofline-Schranke** | $\textrm{Throughput} \le \min(P, I\beta)$ | $P = 165\text{ TF/s}$, $\beta = 1\text{ TB/s}$ | Obere Schranke für die erreichbaren FLOP/s bei Intensität $I$ |

<!--en:ba811457-->
| **Knickpunkt ($I^*$)** | $P / \beta$ | $164\text{ FLOP/B}$ (`bf16`), $82$ (`tf32`) | Minimale Intensität, bei der die Tensor Cores ausgelastet sind |

<!--en:75edda06-->
| **Drei Untergrenzen für die Schrittdauer** | $t \gtrsim \max(h, \textrm{FLOPs}/P, \textrm{bytes}/\beta)$ | $h \approx 13\text{--}100\,\mu\text{s}$ | Zeigt, ob Startkosten, Rechenleistung oder Bandbreite eine Operation begrenzen |

<!--en:3c627904-->
| **Synchronisierungsregel** | Aufwärmläufe und Barriere außerhalb der Schleife | `d2l.Benchmark`, `kernel_times` | Verhindert, dass Host und Gerät nur nacheinander arbeiten oder die Messung nur das Einreihen erfasst |
