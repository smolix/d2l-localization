<!--en:b8374a83-->
# Hardware

<!--en:036ff085-->
Im vorigen Abschnitt haben wir eine GPU durch zwei Konstanten beschrieben, ihre Spitzenrechenleistung $P$ und ihre Speicherbandbreite (engl. memory bandwidth) $\beta$. Eine Operation mit $I$ FLOPs pro Byte erreicht höchstens $\min(P, I\beta)$. Auf der RTX 4090 gilt für bf16-Matrixprodukte $P = 165$ TFLOP/s, die Speicherbandbreite beträgt $\beta \approx 1$ TB/s. In der Zeit, in der die Karte ein Byte aus dem Speicher liest, kann sie 165 Gleitkommaoperationen ausführen. Woher kommen diese beiden Zahlen, und warum dauert die Übertragung eines Bytes so viel länger als ein FLOP?

<!--en:8c1d525f-->
Betrachten wir ein GPT mit einer Milliarde Parametern, das Text Token für Token erzeugt (:numref:`sec_kv-cache`). Seine Gewichte belegen in bf16 insgesamt 2 GB. Für jedes neue Token muss das Modell alle Gewichte lesen. Für die Berechnung eines Tokens ($2 \times 10^9$ FLOPs) brauchen die Tensor Cores 12 µs. Das Lesen der 2 GB an Gewichten dauert bei den erreichten 0.9 TB/s dagegen 2.2 ms. Eine einzelne Sequenz lässt sich daher mit höchstens rund 450 Token/s dekodieren. Die Recheneinheiten bleiben über 99% der Zeit untätig.

<!--en:e523b495-->
Drei physikalische Gegebenheiten erklären diesen Unterschied. Daten müssen Chipgrenzen überwinden. Die Zahl der Leitungen zwischen Chips wächst langsamer als die Zahl der Transistoren auf dem Chip. Und ein Byte zu übertragen kostet um Größenordnungen mehr Energie, als mit ihm zu rechnen. Zusammengenommen erklärt das, warum der Knickpunkt (engl. ridge point) $P/\beta$ im Bereich einiger Hundert liegt und über mehrere Hardwaregenerationen gestiegen ist. Als Beispiel dient uns die Workstation, auf der die Notebooks dieses Buchs laufen (vier GPUs vom Typ RTX 4090, Stand Ende 2026). :citet:`Hennessy.Patterson.2011` behandeln die Rechnerarchitektur ausführlich. Die Auswahl der Hardware besprechen wir in :numref:`sec_hardware_buyers`.

<!--en:e4c07350-->
## Die Speicherhierarchie

<!--en:27b5060f-->
Jedes Byte, auf das ein Programm zugreift, liegt in einer Speicherhierarchie (engl. memory hierarchy). Je schneller der Speicher, desto kleiner ist seine Kapazität (:numref:`fig_memory_hierarchy`). Register und SRAM-Caches auf dem Chip fassen rund 0.1 GB und liefern einige Dutzend TB/s. Das DRAM außerhalb des GPU-Chips fasst einige Dutzend bis einige Hundert GB bei 1–8 TB/s. Rechenzentrumskarten verwenden dafür *High Bandwidth Memory* (HBM), Karten für den Endkundenmarkt GDDR. Das DRAM des Hosts fasst Hunderte GB bei rund 0.5 TB/s pro CPU-Sockel. NVMe-Flashspeicher fasst Terabytes bei rund 14 GB/s. Noch weiter außen liegt der Netzwerkspeicher.

<!--en:c4043676-->
![Die Speicherhierarchie Ende 2026. Jede Stufe nach unten fasst um Größenordnungen mehr Bytes und liefert pro Sekunde um Größenordnungen weniger Bytes.](../img/mdl-perf-memory-hierarchy.svg)

<!--en:2a80943d-->
### Bandbreite und Latenz

<!--en:46abf130-->
:numref:`fig_bandwidth_ladder` ordnet die Datenpfade eines Trainingslaufs nach ihrer Bandbreite. Sie erstrecken sich über fast drei Größenordnungen, von 14 GB/s (NVMe) bis 8 TB/s (HBM der NVIDIA B200). Für jede Stufe gilt dieselbe Regel: **Die Bandbreite ist die Anzahl der Datenleitungen mal der Bitrate pro Leitung**. Die RTX 4090 greift über 384 Leitungen mit je 21 Gb/s auf ihren GDDR6X-Speicher zu (rund 1 TB/s). Ihr PCIe-4.0-×16-Steckplatz hat pro Richtung 16 Lanes mit je 16 Gb/s (32 GB/s). Bei etwa gleicher Geschwindigkeit pro Leitung stehen also $24\times$ weniger Leitungen zur Verfügung.

<!--en:7e21786d-->
Wie viele Leitungen Platz finden, hängt von der Entfernung ab. HBM-Stapel sitzen im GPU-Gehäuse wenige Millimeter vom Siliziumchip entfernt. Tausende kurze Leitungen verbinden sie mit dem Chip. Auf die Platine passen Hunderte von Leiterbahnen, in Steckplätze oder Kabel einige Dutzend Lanes. **Mit jedem Schritt weg vom Chip sinkt die Anzahl der Leitungen und damit die Bandbreite um etwa eine Größenordnung.** Eine B200 liest aus dem HBM mit 8 TB/s. Andere GPUs erreicht sie über NVLink 5 mit 0.9 TB/s (rund 36 Lanes mit je 200 Gb/s), den Host über PCIe 5.0 mit 64 GB/s.

<!--en:3450b9d9-->
![Bandbreite der Datenpfade eines Trainingslaufs Ende 2026. Die Angaben gelten jeweils für eine Richtung. Orange markiert sind die Verbindungen zum Speicher und zum Host auf unserer Workstation.](../img/mdl-perf-bandwidth-ladder.svg)

<!--en:48e46416-->
Die Bandbreite bestimmt den Durchsatz bei kontinuierlicher Übertragung. Die *Latenz* ist die Zeit, bis das erste Byte ankommt (:numref:`fig_latency_ladder`, mit Mikrobenchmarks der RTX 4090 von :citet:`Luo.Fan.Li.ea.2024`). Der Start einer einzelnen Operation im Eager-Modus aus Python beansprucht 10–100 µs auf dem Host. Davon entfallen 1–5 µs auf den eigentlichen Kernelstart in der Hardware. In 10 µs kann die RTX 4090 schon 10 MB übertragen. Bei kleinen Operationen überwiegt daher der Aufwand für den Start gegenüber der Zeit für Speicherzugriffe oder Berechnungen.

<!--en:55464b1b-->
![Latenzen auf einer logarithmischen Skala. Der hervorgehobene Eintrag zeigt die Zeit, die eine aus Python gestartete Framework-Operation auf dem Host beansprucht. Der Kernelstart macht einen Teil davon aus.](../img/mdl-perf-latency-ladder.svg)

<!--en:72a70315-->
Wie Latenz und Bandbreite zusammenspielen, hängt vom *Zugriffsmuster* (engl. access pattern) ab. DRAM erreicht seine Nennbandbreite nur bei aufeinanderfolgenden Adressen. Lesen wir einen 1 GiB großen fp32-Vektor mit `x[idx]` aus. Für `idx` verwenden wir einmal die identische Permutation und einmal eine Zufallspermutation. Der GPU-Speicher überträgt Daten in *Sektoren* von 32 Byte. Jeder zufällige Lesezugriff auf 4 Byte holt daher einen ganzen Sektor von 32 Byte. Der Zugriff auf Index- und Ausgabevektor bleibt sequenziell (insgesamt etwa $3\times$ so viele übertragene Bytes).

<!--en:9e6597e0-->
Mit der identischen Permutation erreicht der Gather-Zugriff rund 0.9 TB/s. Mit der Zufallspermutation dauert er rund 37 ms ($8\text{--}11\times$ so lange). Selbst wenn wir für jeden Lesezugriff einen vollen Sektor von 32 Byte anrechnen, ergibt das nur ein Drittel der Spitzenbandbreite. Neben den 32-Byte-Sektoren spielt eine größere Einheit eine Rolle. DRAM liefert die Daten aus geöffneten *Zeilen* von einigen Kilobyte. Aufeinanderfolgende Adressen nutzen dieselbe offene Zeile. Bei zufälligen Adressen fällt immer wieder die Latenz für das Öffnen einer neuen Zeile an. Auch die Caches für die Adressübersetzung verzeichnen Fehlzugriffe.

<!--en:3d6edfdb-->
Dichte Matrixprodukte und das Nachschlagen von Einbettungen nutzen die Bandbreite effizient. Letzteres liest pro Token zusammenhängende Zeilen von Tausenden von Bytes. Feinkörnige Zufallszugriffe wie bei dünn besetzten Matrizen, Graphdurchläufen oder Hashtabellen erreichen dagegen nur einen Bruchteil von $\beta$. Allgemein dauert die Übertragung eines zusammenhängenden Blocks von $n$ Bytes bei einer Anlauflatenz von $\alpha$ insgesamt $t(n) = \alpha + n/\beta$. Bei $n_{1/2} = \alpha\beta$ erreicht sie die Hälfte der Spitzenbandbreite :cite:`Hockney.1994`. Für ein NVMe-Laufwerk ($\alpha \approx 50\text{--}100$ µs, $\beta = 14$ GB/s) gilt $n_{1/2} \approx 1$ MB. Deshalb mischen wir Datensätze auf dem Laufwerk in Blöcken von Megabytegröße, statt einzelne Beispiele zu mischen.

<!--en:bb94955d-->
## Rechenleistung und Speicherbandbreite

<!--en:da2b5f00-->
### Fläche und Umfang

<!--en:8aba480d-->
Recheneinheiten füllen das Innere eines Siliziumchips. Ihre Zahl wächst mit der Fläche, also wie $O(n^2)$. Die I/O-Kontakte liegen dagegen am *Chiprand* (engl. shoreline). Ihre Zahl wächst mit der Kantenlänge, also wie $O(n)$ (:numref:`fig_shoreline`). Kleinere Transistoren haben denselben Effekt. Halbiert sich die Strukturgröße, vervierfacht sich die Dichte der Logik, während der Abstand der Lötkontakte am Chiprand viel langsamer schrumpft.

<!--en:8e1beade-->
![Die Rechenleistung wächst mit der Fläche, die Zahl der I/O-Kontakte mit dem Umfang. Verdoppelt sich die Seitenlänge des Chips, vervierfacht sich seine Rechenleistung. Der Platz für I/O verdoppelt sich nur.](../img/mdl-perf-shoreline.svg)

<!--en:930c1ef2-->
Beim Chipentwurf lässt sich der Engpass am Chiprand auf zwei Arten entschärfen:

<!--en:47d2f167-->
1. **HBM auf einem Silizium-Interposer:** Neben dem GPU-Chip liegen mehrere DRAM-Chips übereinander. Tausende kurze, dicht gepackte Leitungen auf dem Interposer ersetzen einige Hundert Leiterbahnen auf der Platine. So erreicht eine B200 8 TB/s gegenüber rund 0.5 TB/s bei einem CPU-Sockel.

<!--en:eca8a252-->
2. **Mehrere Chips in einem Gehäuse:** In der Lithografie begrenzt das *Belichtungsfeld* (engl. reticle limit) die Fläche eines einzelnen Chips auf etwa $858\,\textrm{mm}^2$. Spitzenbeschleuniger wie die B200 verbinden zwei Chips dieser maximalen Größe in einem Gehäuse. So verdoppelt sich der verfügbare Umfang und reicht für acht HBM-Stapel.

<!--en:4fadae63-->
### Entwicklung über die Hardwaregenerationen

<!--en:5a052362-->
Bei NVIDIAs Rechenzentrums-GPUs stieg von der V100 (2017) über A100 und H100 bis zur B200 (2024) die Rechenleistung für dichte 16-Bit-Operationen in FLOP/s pro Generation um etwa den Faktor $2.6\times$, die Speicherbandbreite um etwa den Faktor $2.0\times$ und die Kapazität um etwa den Faktor $1.8\times$ :cite:`Gholami.Yao.Kim.ea.2024`. Dadurch verdoppelte sich der Knickpunkt von 139 FLOP/Byte auf der V100 auf 295 auf der H100 (281 auf der B200). Eine Operation mit einer Intensität von 200 FLOP/Byte war auf der V100 rechenbegrenzt, ist auf einer H100 oder B200 aber bandbreitenbegrenzt.

<!--en:56e34a57-->
### Energie pro Operation

<!--en:8a2d8354-->
Eine zweite feste Obergrenze für die Bandbreite ergibt sich aus der verfügbaren elektrischen Leistung. In einem 45-nm-CMOS-Prozess (:numref:`fig_energy_ladder`, :citet:`Horowitz.2014`) kostet eine fp32-Multiplikation 3.7 pJ und eine 8-Bit-Ganzzahladdition 0.03 pJ. Das Lesen von 64 Bit aus DRAM außerhalb des Chips kostet dagegen 1.3–2.6 nJ, etwa so viel Energie wie **500 fp32-Multiplikationen**. Lesezugriffe auf SRAM im Chip liegen dazwischen (je nach Entfernung 10–100 pJ). Moderne Fertigungsprozesse senken alle diese Werte, doch ihre Rangfolge bleibt bestehen. Daten zu bewegen kostet weit mehr Energie, als mit ihnen zu rechnen. Kernelfusion, schmalere Zahlenformate und das erneute Berechnen von Aktivierungen sparen aus demselben Grund elektrische Leistung wie Zeit.

<!--en:f095d206-->
![Energie pro Operation in einem 45-nm-Prozess. Ein 64-Bit-Lesezugriff auf DRAM kostet etwa so viel wie 500 fp32-Multiplikationen.](../img/mdl-perf-energy-ladder.svg)

<!--en:cbaa3d48-->
## Die GPU

<!--en:7a32f066-->
### Streaming Multiprocessors

<!--en:a146d685-->
Eine GPU nutzt ihre Chipfläche für Tausende von Recheneinheiten. Diese sind in *Streaming Multiprocessors* organisiert (SMs, 128 auf der RTX 4090). Jeder SM führt Threads in Gruppen von je 32 im Gleichschritt aus. Diese Gruppen heißen *Warps*. Um die DRAM-Latenz zu verbergen, braucht ein SM keine riesigen Caches. Er kann in jedem Takt den Warp wechseln. Während ein Warp auf den Speicher wartet, rechnen andere weiter.

<!--en:3d5148b0-->
Nach dem Gesetz von Little :cite:`Little.1961` gilt für einen Speicherpfad mit einer Bandbreite von $\beta = 1.0$ TB/s und einer Latenz von 250 ns: Um den Pfad auszulasten, müssen ständig $1.0\,\textrm{TB/s} \times 250\,\textrm{ns} \approx 250$ KB gleichzeitig unterwegs sein. Das entspricht rund 2 KB pro SM oder 16 Warps pro SM, bei denen jeder Thread auf einen Lesezugriff von 4 Byte wartet. Bietet ein Kernel zu wenig Parallelität, um 250 KB gleichzeitig anzufordern, begrenzt die Latenz seinen Durchsatz statt der Bandbreite.

<!--en:055566b3-->
### Tensor Cores und Zahlenformate

<!--en:ae6bb8ee-->
*Tensor Cores* multiplizieren kleine Matrixblöcke und akkumulieren das Ergebnis in einem einzigen Befehl. Ähnlich wie beim systolischen Array einer TPU verteilt sich so der Aufwand für das Laden und Dekodieren des Befehls auf einen ganzen Block :cite:`Kung.1988,Jouppi.Young.Patil.ea.2017`. Auf einer H100 erreichen die Tensor Cores mit bf16 einen Durchsatz von 989 TFLOP/s gegenüber 67 TFLOP/s für skalare fp32-Operationen ($15\times$). Auf der Endkundenkarte RTX 4090 beträgt das Verhältnis $2\times$.

<!--en:9ac905a3-->
Den Spitzendurchsatz der RTX 4090 können wir direkt aus ihren technischen Daten ableiten. 128 SMs $\times$ 128 fp32-Einheiten $= 16{,}384$ Einheiten. Bei einem Boost-Takt von 2.52 GHz führt jede Einheit pro Takt eine kombinierte Multiplikation und Addition aus (2 FLOPs):

<!--en:2e904c80-->
Die Tensor Cores erreichen mit tf32 dieselben 82.6 TFLOP/s, mit bf16 (und fp32-Akkumulation) $2\times = 165.2$ TFLOP/s und mit fp8 nominell $4\times = 330.4$ TFLOP/s. Den Durchsatz für tf32 und bf16 messen wir in diesem Kapitel. Der fp8-Durchsatz ist ein Nennwert. Keines unserer Codebeispiele nutzt dieses Format.

<!--en:a4e9d5bf-->
![Gleitkommaformate Bit für Bit. fp32, tf32 und bf16 haben einen 8-Bit-Exponenten und damit denselben Wertebereich. tf32 ist ein Rechenmodus der Tensor Cores. Seine Werte belegen im Speicher 32 Bit. Bei jedem anderen Schritt zum nächstschmaleren Format halbiert sich die Anzahl der Bytes pro Wert.](../img/mdl-perf-float-formats.svg)

<!--en:4632738e-->
Wie :numref:`fig_float_formats` zeigt, wirkt sich ein schmaleres Format auf zwei Arten auf die Roofline aus:

<!--en:4684f3ff-->
1. **Eine Halbierung der Bitbreite halbiert die übertragene Datenmenge.** Bandbreitenbegrenzte Kernel werden dadurch bis zu $2\times$ schneller, und die Gewichte benötigen nur halb so viel Speicher. Die Ausnahme ist tf32. In diesem Modus rechnen die Tensor Cores mit 19-Bit-Werten (10-Bit-Mantisse), im Speicher liegen die Werte aber weiterhin als 32-Bit-Gleitkommazahlen.

<!--en:710b060c-->
2. **Eine Halbierung der Bitbreite verdoppelt in der Regel die Spitzenrechenleistung $P$ der Tensor Cores.**

<!--en:227b7338-->
Formate mit höchstens 8 Bit brauchen explizite Skalierungsfaktoren. fp8 nutzt E4M3 für Aktivierungen und Gewichte im Vorwärtsdurchlauf und E5M2 für Gradienten. Dabei gilt jeweils ein Skalierungsfaktor pro Tensor oder Block :cite:`Micikevicius.Stosic.Burgess.ea.2022`. Bei 4 Bit (fp4 / E2M1, nur 15 darstellbare Werte von $-6$ bis $6$) versagt die Skalierung pro Tensor bei Ausreißern. Deshalb verwendet die Hardware *Mikroskalierung* (engl. microscaling). Je ein Skalierungsfaktor gilt für einen Block von 16 (NVFP4) oder 32 (OCP MX) Elementen :cite:`Rouhani.Zhao.More.ea.2023`.

<!--en:6cb67747-->
### Technische Daten der Beschleuniger

<!--en:d14e0128-->
:numref:`tab_gpu_specs` vergleicht zwei GPUs für Rechenzentren mit zwei Endkundenkarten.

<!--en:03b7cff0-->
:Vier weitverbreitete GPUs. Angegeben ist der Durchsatz der Tensor Cores bei dichten Matrixprodukten mit fp32-Akkumulation und Boost-Takt in TFLOP/s (TF). Für die beiden Endkundenkarten stehen bei fp8 und fp4 Nennwerte. Für fp8 ist der doppelte bf16-Durchsatz angegeben. Der fp4-Wert der RTX 5090 ist die Spitzenangabe des Herstellers und kein Messwert. Die Bandbreite der Verbindungen gilt für eine Richtung.

<!--en:e2ddb403-->
| | H100 SXM | B200 | RTX 4090 (bei uns) | RTX 5090 |

<!--en:5048ecd2-->
| Speicherkapazität $C$ | 80 GB HBM3 | 180 GB HBM3e | 24 GB GDDR6X | 32 GB GDDR7 |

<!--en:cc557107-->
| Speicherbandbreite $\beta$ | 3.35 TB/s | 8 TB/s | 1.0 TB/s | 1.8 TB/s |

<!--en:1551fc5e-->
| bf16 | 989 TF | 2250 TF | 165 TF | 210 TF |

<!--en:943efba4-->
| fp8 | 1979 TF | 4500 TF | 330 TF | 419 TF |

<!--en:a2858d50-->
| fp4 | – | 9000 TF | – | 1676 TF |

<!--en:b34b5278-->
| Knickpunkt (bf16) | 295 FLOP/B | 281 FLOP/B | 164 FLOP/B | 117 FLOP/B |

<!--en:a65c09fa-->
| Verbindung zu anderen GPUs | NVLink 4, 450 GB/s | NVLink 5, 900 GB/s | PCIe 4.0 ×16 über den Host, $\le 32$ GB/s | PCIe 5.0 ×16 über den Host, $\le 64$ GB/s |

<!--en:4d30628a-->
| Leistungsaufnahme | 700 W | 1000 W | 450 W | 575 W |

<!--en:9c5cd8ef-->
## Datenpfade außerhalb der GPU

<!--en:f4e34623-->
### Der Host

<!--en:766b1cca-->
Während die GPU rechnet, übernimmt die Host-CPU weiterhin drei Aufgaben, die das Training ausbremsen können. Sie startet Kernel ($h$ pro Aufruf), führt die Pipeline zum Laden und Vorverarbeiten der Daten aus und stellt die Daten für den Transfer über PCIe zur GPU bereit.

<!--en:8d1e251e-->
Die *Kopiereinheit* (engl. copy engine) der GPU kann Daten per DMA über PCIe übertragen, während die SMs rechnen. Das geht nur aus *auslagerungsgesperrtem* Host-Speicher (engl. pinned memory). Bei gewöhnlichem *auslagerbarem* Speicher muss die CPU die Daten zuerst in einen internen auslagerungsgesperrten Puffer kopieren. Vergleichen wir die Übertragungsraten für ein 256 MiB großes Array in beiden Speicherarten.

<!--en:d65690a3-->
Mit auslagerungsgesperrtem Speicher erreicht der Transfer rund 23 GB/s. Das sind etwa drei Viertel der theoretischen Obergrenze von 32 GB/s für PCIe 4.0 ×16. Der Transfer ist damit $40\times$ langsamer als der Zugriff auf den GPU-Speicher. Mit auslagerbarem Speicher dauert er wegen der zusätzlichen Kopie im Host-Speicher noch länger. Datensätze, die in den GPU-Speicher passen, sollten dort bleiben. Minibatches vom Host sollten aus auslagerungsgesperrtem Speicher kommen, sodass sich Übertragung und Berechnung überlappen.

<!--en:db32fa18-->
Ein `DataLoader` mit `pin_memory=True` stellt seine Minibatches in auslagerungsgesperrtem Speicher bereit.

<!--en:b46f1c32-->
Arrays mit dem Speichertyp `'pinned_host'` liegen in auslagerungsgesperrtem Speicher.

<!--en:22a7fd3a-->
### Verbindungen zwischen GPUs

<!--en:369927ea-->
GPUs in Rechenzentren kommunizieren über dedizierte NVLink-Verbindungsnetze (450 GB/s pro Richtung auf der H100, 900 GB/s auf der B200). Die RTX 4090 für den Endkundenmarkt unterstützt weder NVLink noch direkten Peer-to-Peer-Zugriff (P2P) über PCIe. Wie :numref:`fig_pcie_topology` zeigt, läuft auf unserer Workstation mit vier GPUs jedes Byte von einer GPU zur anderen über zwei PCIe-4.0-Verbindungen. Die erste führt zum DRAM des Hosts, die zweite von dort zur Ziel-GPU. Der Durchsatz ist damit auf die Kopierrate für auslagerungsgesperrten Speicher von rund 23 GB/s begrenzt ($40\times$ langsamer als NVLink 5).

<!--en:6391251c-->
![Unsere Workstation laut `nvidia-smi topo`. Die vier RTX 4090 sitzen paarweise hinter zwei PCIe-Host-Bridges und verfügen weder über Peer-to-Peer-Zugriff noch über NVLink. Der hervorgehobene Pfad zeigt den Weg jedes Bytes von einer GPU zur anderen.](../img/mdl-perf-pcie-topology.svg)

<!--en:878b1873-->
Bei unserem GPT mit einer Milliarde Parametern ($S = 4$ GB an fp32-Gradienten) dauert eine einmalige Übertragung der Gradienten bei 23 GB/s auf unserer Workstation mindestens 0.17 s, über NVLink 5 dagegen < 5 ms. Die 0.17 s sind mehr als die Hälfte der 0.3 s Rechenzeit für einen Schritt mit 8192 Token bei maximalem bf16-Durchsatz.

<!--en:9c20bb79-->
## Arbeitslasten im Roofline-Modell

<!--en:9eaa867a-->
Die Inferenz eines Sprachmodells umfasst zwei Bereiche der Roofline (:numref:`subsec_kv-cache-prefill-decode`). Beim *Prefill* verarbeitet das Modell einen Prompt mit $n$ Token parallel. Es erreicht die Intensität $I \approx n$ (bis zur Schichtbreite) und arbeitet an der Rechengrenze. Die *Dekodierung* erzeugt pro Schritt ein Token und liest dafür alle $2N$ Bytes der bf16-Gewichte sowie den KV-Cache. Das ergibt eine Intensität von $I \approx 1$ FLOP/Byte.

<!--en:cda2d309-->
### Dekodierung einzelner Sequenzen und ganzer Batches

<!--en:a6d22476-->
Die Dekodierung einer einzelnen Sequenz ist bandbreitenbegrenzt. Für ihre Tokenrate gilt daher die Obergrenze

<!--en:c819778b-->
Füllen die Gewichte eines Modells die gesamte Speicherkapazität $C$ der Karte, entspricht die maximale Tokenrate für eine einzelne Sequenz dem Kehrwert der *Zeit für einen vollständigen Speicherdurchlauf* (engl. memory sweep time) $C/\beta$. So lange dauert es, jedes Byte des GPU-DRAMs einmal zu lesen. Zwischen den vier GPUs in :numref:`tab_gpu_specs` unterscheiden sich Kapazität und Bandbreite um den Faktor $7\text{--}8\times$. Dennoch liegt $C/\beta$ durchweg nahe bei **20 ms** (24 ms auf H100 und RTX 4090, 22.5 ms auf der B200, 18 ms auf der RTX 5090). Jedes dichte Modell, das seine GPU füllt, dekodiert eine einzelne Sequenz also mit rund 40–55 Token/s. Das gilt für eine Endkundenkarte ebenso wie für ein Spitzenmodell im Rechenzentrum. Diese Grenze setzt voraus, dass das Modell für jedes Token jedes Gewicht liest. Ein Mixture-of-Experts-Modell (:numref:`sec_moe`) liest nur die Experten, die sein Router für das jeweilige Token auswählt. Es dekodiert daher schneller, als $C/\beta$ vermuten lässt.

<!--en:7695a731-->
Fassen wir $B$ gleichzeitig verarbeitete Sequenzen zu einem Batch zusammen, verteilt sich jeder Lesezugriff auf die Gewichte auf $B$ Token. Bezogen auf die Gewichte steigt die Intensität damit auf $I \approx B$ FLOP/Byte, bis $B$ den Knickpunkt erreicht (siehe die Messung in :numref:`subsec_perf-roofline-test`). Jede Sequenz liest allerdings weiterhin ihren eigenen KV-Cache (:numref:`subsec_kv-cache-batching`).

<!--en:0d18d98a-->
### Faustregeln

<!--en:6ddc5a12-->
:numref:`tab_rules_of_thumb` fasst die wichtigsten Hardwarekonstanten zusammen, die wir im weiteren Verlauf dieses Kapitels verwenden.

<!--en:a97066dd-->
:Faustregeln für die weitverbreiteten GPUs aus :numref:`tab_gpu_specs`. Die Tabelle zeigt typische Größenordnungen, die Werte unserer Workstation (vier RTX 4090 ohne Peer-to-Peer-Zugriff) und ihre praktische Bedeutung. Die neuesten GPUs für Rechenzentren übertreffen die angegebenen Obergrenzen.

<!--en:4c142f60-->
| Größe | typische Größenordnung | unsere Workstation | praktische Bedeutung |

<!--en:4194e0ec-->
| Host-Zeit pro Aufruf, $h$ | 10 µs bis 0.1 ms | 13–100 µs (:numref:`sec_perf_model`) | kleine Operationen lasten die GPU nicht aus |

<!--en:b7ce8508-->
| bf16-Spitzendurchsatz, $P$ | 0.17–2.3 PFLOP/s | 165 TFLOP/s | begrenzt die Geschwindigkeit großer Matrixprodukte |

<!--en:6548fdf7-->
| GPU-Speicherbandbreite, $\beta$ | 1–8 TB/s | rund 0.9 TB/s (Nennwert 1.0) | begrenzt die Geschwindigkeit elementweiser Operationen und der Dekodierung |

<!--en:9b4c845f-->
| Knickpunkt, $P/\beta$ | 120–300 FLOP/Byte | 164 FLOP/Byte | unterhalb davon bestimmt die Speicherbandbreite die Laufzeit |

<!--en:6f1cf57e-->
| Zeit für einen vollständigen Speicherdurchlauf, $C/\beta$ | rund 20 ms | 24 ms | das größte dichte Modell, das in den Speicher passt, dekodiert 1 Sequenz mit rund 40–55 Token/s |

<!--en:e5f16b92-->
| Host zur GPU, auslagerungsgesperrt | 32–64 GB/s (PCIe 4/5) | rund 23 GB/s | Trainingsdaten auf der GPU oder in auslagerungsgesperrtem Speicher halten |

<!--en:98d11524-->
| GPU zu GPU | 450–900 GB/s (NVLink) | $\le 23$ GB/s über den Host (:numref:`sec_multi_gpu`) | bestimmt die Dauer jedes Gradientenaustauschs |

<!--en:f1c3dc0f-->
| Bytes pro Parameter (Adam) | 16 (fp32), 18 (gemischte Präzision) | 16–18 | bestimmt, welche Modellgröße in den Speicher passt (:numref:`sec_memory_precision`) |

<!--en:a653e3ca-->
| Bytes pro Token bei der Dekodierung | $2N$ + KV-Cache (bf16) | | Token/s $\lesssim \beta / \textrm{bytes per token}$ |

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:b1f04222-->
Die Bandbreite ist die Anzahl der Leitungen mal der Datenrate pro Leitung. Leitungen verlassen den Chip an seinem Rand, Recheneinheiten füllen seine Fläche. Daten über die Chipgrenze zu bewegen kostet zudem mehrere Hundert Mal so viel Energie, wie mit ihnen zu rechnen. Deshalb liegt der Knickpunkt $P/\beta$ bei einigen Hundert FLOP/Byte und ist über die Hardwaregenerationen gestiegen. Der Speicher überträgt Sektoren von 32 Byte aus offenen DRAM-Zeilen. Feinkörnige zufällige Lesezugriffe erreichen daher nur einen Bruchteil von $\beta$. Um $\beta$ auszuschöpfen, müssen über alle Warps hinweg $\beta \times \textrm{latency}$ Bytes gleichzeitig unterwegs sein. Schmalere Zahlenformate halbieren den Speicherverkehr und verdoppeln in der Regel $P$. Für die Dekodierung einer einzelnen Sequenz gilt die Obergrenze $\beta / (2N + \textrm{KV bytes})$. Die Zeit für einen vollständigen Speicherdurchlauf $C/\beta$ liegt bei modernen GPUs durchweg bei rund 20 ms. Das größte dichte Modell, das auf die jeweilige Karte passt, dekodiert eine einzelne Sequenz daher mit rund 40–55 Token/s.

<!--en:2294243a-->
## Übungen

<!--en:2c201d4b-->
1. [code] **Latenz und Bandbreite messen.** ● Das Modell der Blockübertragung $t(n) = \alpha + n/\beta$ mit $n_{1/2} = \alpha\beta$ (:numref:`subsec_hw-bytes`) beschreibt einen einzelnen synchronen Aufruf.
    1. Sagen Sie $n_{1/2}$ für die elementweise Multiplikation `x * 1.5` aus :numref:`subsec_perf-regimes` voraus. Verwenden Sie dazu $\alpha = h$ und $\beta$ aus Ihren Messungen in :numref:`sec_perf_model`.
    1. Messen Sie die Laufzeit von `x * 1.5` für Tensorgrößen von 1 MB bis 2 GB. Tragen Sie die erreichte Bandbreite gegen die Größe auf, mit logarithmischer Skalierung auf beiden Achsen. Setzen Sie die Übergänge zwischen den Bereichen in Beziehung zu :numref:`fig_latency_ladder` und zur Kapazität des L2-Caches auf dem Chip in :numref:`fig_memory_hierarchy`.
    1. Wiederholen Sie die Messung für den Transfer vom Host zur GPU aus :numref:`subsec_hw-cpu` für Datenmengen von 4 KB bis 256 MB. Bestimmen Sie $\alpha$ und die asymptotische Bandbreite durch Anpassung des Modells an die Messwerte.

<!--en:a983328c-->
2. **Leitungen und Datenraten.** Nach :numref:`subsec_hw-bytes` entspricht die Bandbreite der Anzahl der Leitungen mal der Datenrate pro Leitung.
    1. Ein DDR5-6400-DIMM hat 64 Datenleitungen mit je 6.4 Gb/s. Ein HBM3e-Stapel hat 1024 Datenleitungen, eine B200 trägt acht solcher Stapel. Berechnen Sie die Bandbreite eines DIMMs und die Anzahl der DIMMs, die ein CPU-Sockel für 0.5 TB/s braucht. Berechnen Sie außerdem die Datenrate pro Leitung, die sich aus den 8 TB/s der B200 in :numref:`tab_gpu_specs` ergibt. Welcher Faktor erklärt den Unterschied zwischen dem CPU-Sockel und der B200?
    1. Neben jedem der beiden Chips einer B200 liegen vier der acht HBM-Stapel. Die SMs auf jedem Chip lesen gleich oft aus beiden Speicherhälften, insgesamt mit 8 TB/s. Welche Bandbreite muss die Brücke zwischen den Chips im Gehäuse in jeder Richtung bieten?

<!--en:5079d506-->
3. **Wenn die Eingabepipeline zu wenig Daten liefert.** Angenommen, die GPU kann beim Training 2 GB/s an dekodierten Bildern verarbeiten. Jeder CPU-Arbeitsprozess durchläuft nacheinander zwei Stufen. Er liest komprimierte JPEG-Bilder mit 1 GB/s von einem NVMe-Laufwerk (komprimierte Daten) und dekodiert sie auf einem CPU-Kern mit 0.3 GB/s (dekodierte Ausgabe, Vergrößerung um den Faktor $8\times$). Vor der Berechnung kopiert die Trainingsschleife jeden dekodierten Batch synchron mit 12 GB/s aus auslagerbarem Host-Speicher auf die GPU.
    1. Mit welcher Rate (in dekodierten Bytes/s) liefert ein Arbeitsprozess Batches, und welche Stufe begrenzt die Rate?
    1. Wie viele parallele Arbeitsprozesse braucht die GPU, um durchgehend mit Daten versorgt zu sein?
    1. Um welchen Anteil verlängert das synchrone Kopieren aus auslagerbarem Speicher jeden Schritt? Was ändert sich beim Wechsel zu asynchronen Kopien aus auslagerungsgesperrtem Speicher mit 24 GB/s?

<!--en:22ca7d18-->
4. **Knickpunkt einer Verbindung.** Der Knickpunkt $P/\beta$ gilt für jeden Datenpfad mit Bandbreite $\beta$. Er gibt an, wie viele FLOPs ein Gerät ausführen kann, während ein Byte diesen Pfad durchläuft.
    1. Berechnen Sie $P/\beta$ in bf16 für eine RTX 4090, einmal für ihr eigenes DRAM und einmal für den GPU-zu-GPU-Pfad über den Host mit 23 GB/s. Wiederholen Sie die Rechnung für eine B200 mit ihrem HBM und für eine Richtung von NVLink 5 (900 GB/s).
    1. Beim datenparallelen Training (:numref:`sec_multi_gpu`) hält jede GPU ein Modell mit $N$ Parametern und verarbeitet pro Schritt $T$ Token mit rund $6N$ FLOPs pro Token. Ein Allreduce summiert die bf16-Gradienten und verschickt dabei rund $4N$ Bytes pro GPU. Nehmen Sie an, Kommunikation und Berechnung überlappen sich nicht. Zeigen Sie, dass die Kommunikation genau dann höchstens so lange dauert wie die Berechnung, wenn $T$ einen Schwellenwert erreicht oder übersteigt, der proportional zum Knickpunkt der Verbindung und unabhängig von $N$ ist.
    1. Berechnen Sie diese Tokenschwelle für NVLink 5, für das Kopieren aus auslagerungsgesperrtem Speicher mit 23 GB/s und für die 2.4 GB/s, die NCCL auf unserer Workstation standardmäßig über den Host erreicht (:numref:`sec_multi_gpu_concise`).

<!--en:de47107f-->
5. **Wenn der Kernelstart die Dekodierung begrenzt.** Beim Dekodieren einer einzelnen Sequenz im Eager-Modus startet die CPU jeden Kernel jedes Blocks, während die GPU die Gewichte des Blocks liest. Angenommen, jeder der $L$ Blöcke der Breite $d$ enthält $12d^2$ bf16-Gewichte und startet $K = 12$ Kernel pro Token. Pro Kernel dauert der Start auf dem Host $h = 8$ µs.
    1. Zeigen Sie, dass es von $L$ unabhängig ist, ob die Startzeit auf dem Host oder die Lesezeit der Gewichte den Durchsatz begrenzt. Bestimmen Sie die Breite $d^*$, bei der beide Zeiten gleich sind, für die RTX 4090 und die B200.
    1. Berechnen Sie beide Grenzen für GPT-2 Small ($d = 768$, $L = 12$) auf der RTX 4090. Welche Grenze bestimmt den Durchsatz?
    1. Wie verschiebt sich $d^*$ von der RTX 4090 zur B200? Was bedeutet das für den Nutzen der Aufzeichnung und Wiedergabe von CUDA-Graphen (:numref:`sec_compilation`) auf schnelleren GPUs?

<!--en:da2d37bc-->
6. [code] **Gather-Zugriffe auf zufällig angeordnete Blöcke.** ● Wiederholen Sie den Gather-Benchmark aus :numref:`subsec_hw-bytes`. Permutieren Sie diesmal zusammenhängende Blöcke von $k$ Elementen für $k \in \{1, 2, 4, \ldots, 1024\}$ anstelle einzelner Elemente. Tragen Sie die erreichte Bandbreite gegen die Blockgröße in Bytes auf. Vergleichen Sie die Blockgröße am Übergang mit der Sektorgröße von 32 Byte und mit der Größe einer DRAM-Zeile, die im Kilobytebereich liegt.

<!--en:eed363c8-->
7. **Latenz verbergen.** ● Wenden Sie das Gesetz von Little (:numref:`subsec_hw-gpu`) auf das Verbergen der Speicherlatenz an:
    1. Berechnen Sie, wie viele Bytes bei 250 ns Latenz gleichzeitig unterwegs sein müssen, um die 8 TB/s des HBM der B200 auszuschöpfen. Vergleichen Sie den Wert mit den 250 KB der RTX 4090. Die Bandbreite steigt pro Generation ungefähr um den Faktor $2\times$, während die DRAM-Latenz nahezu konstant bleibt. Was folgt daraus für die Parallelität, die Kernel bieten müssen?
    1. Beim Verfolgen einer Zeigerkette hängt die Adresse jedes Lesezugriffs vom vorherigen ab. Daher kann pro Kette nur eine Anfrage gleichzeitig unterwegs sein. Berechnen Sie für eine CPU mit 90 ns DDR5-Latenz und 64 Byte großen Cachezeilen die Bandbreite einer einzelnen Kette. Wie viele unabhängige Ketten braucht man, um einen Sockel mit 0.5 TB/s auszulasten?

<!--en:8886704a-->
[Diskussion](https://d2l.discourse.group/t/36788)

<!--en:4eea747b-->
Aufbau der Hardware: CPUs, GPUs, Tensor Cores und Speicherhierarchie<br>**Speicherhierarchie · Leitungszahl und Chiprand · Sektorzugriffe · SMs und Tensor Cores · PCIe und NVLink · Zeit für einen Speicherdurchlauf bei der Dekodierung**

<!--en:9faf768d-->
::: {.slide title="Warum eine GPU mit 165 TFLOP/s nur 450 Token/s dekodiert"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:e9ec4f16-->
- **Rechenzeit ($12\,\mu\text{s}$):** Für $2N = 2 \times 10^9\text{ FLOPs}$ pro Token braucht ein GPT mit 1 Mrd. Parametern $12\,\mu\text{s}$ bei $165\text{ TFLOP/s}$.

<!--en:8349ac13-->
- **Übertragungszeit ($2.2\text{ ms}$):** Alle $2\text{ GB}$ an `bf16`-Gewichten mit $0.9\text{ TB/s}$ aus dem GDDR6X-Speicher zu lesen dauert $2.2\text{ ms}$, also fast **$200\times$ so lange**.

<!--en:646fa542-->
- **Durchsatzgrenze ($\sim 450\text{ tok/s}$):** Beim Dekodieren einer einzelnen Sequenz bleiben die Tensor Cores über $99\%$ der Zeit untätig.

<!--en:ceff8f2d-->
::: {.slide title="Speicherhierarchie: Mehr Kapazität kostet Bandbreite"}

<!--en:9a868bb7-->
[Speicherhierarchie]{.kicker}

<!--en:ac361101-->
- **Register und SRAM auf dem Chip ($L_1$/gemeinsam genutzter Speicher, $L_2$):** Sie fassen $\sim 0.1\text{ GB}$, liegen wenige Millimeter von den ALUs entfernt und liefern einige Dutzend $\text{TB/s}$.

<!--en:7c70234a-->
- **GPU-DRAM außerhalb des Chips (GDDR6X / HBM3e) und weitere Stufen:** Die Kapazität wächst um Größenordnungen ($24\text{--}180\text{ GB}$ DRAM $\to$ Hunderte $\text{GB}$ Host-RAM $\to$ $\text{TB}$ NVMe), die Bandbreite sinkt an jeder Grenze um Größenordnungen.

<!--en:8acc2b43-->
::: {.slide title="Bandbreite gleich Leitungszahl mal Bitrate pro Leitung"}

<!--en:4b6b3fdf-->
[Physikalische Bandbreite]{.kicker}

<!--en:2597bfe4-->
- **Bandbreitenregel:** $\text{BW} = N_{\text{wires}} \times r_{\text{wire}}$ (Anzahl der Datenleitungen mal Bitrate pro Leitung).

<!--en:64fb383f-->
- **GDDR6X der RTX 4090 und PCIe 4.0 $\times 16$:** Der Speicher erreicht $384\text{ wires} \times 21\text{ Gb/s} \approx 1.0\text{ TB/s}$, PCIe pro Richtung $16\text{ lanes} \times 16\text{ Gb/s} = 32\text{ GB/s}$. Bei ähnlicher Geschwindigkeit pro Leitung hat PCIe **$24\times$ weniger Leitungen**.

<!--en:7b5ca0d2-->
- **Die Entfernung bestimmt die Leitungszahl:** Mit jedem Schritt weg vom Siliziumchip (HBM-Interposer $\to$ Leiterbahnen auf der Platine $\to$ PCIe-Steckplatz/Kabel) sinken Leitungsdichte und Bandbreite etwa um den Faktor $10\times$.

<!--en:1db0205f-->
::: {.slide title="Latenzen vom Registerzugriff bis zum Python-Aufruf"}

<!--en:e0e9d590-->
[Zugriffslatenz]{.kicker}

<!--en:669a3c23-->
- **Bandbreite und Latenz:** Die Bandbreite bestimmt den kontinuierlichen Durchsatz. Die Latenz $\alpha$ ist die Wartezeit bis zum ersten Byte.

<!--en:e33038c7-->
- **Innerhalb und außerhalb des Chips:** Register- und SRAM-Zugriffe dauern Nanosekunden ($10\text{--}30\text{ ns}$), Zugriffe auf externes GPU-DRAM $\sim 250\text{ ns}$.

<!--en:211bd83a-->
- **Start aus Python ($10\text{--}100\,\mu\text{s}$):** Eine Operation im Eager-Modus aus Python zu starten dauert $40\text{--}400\times$ so lange wie ein DRAM-Lesezugriff. In dieser Zeit kann die RTX 4090 $10\text{ MB}$ übertragen.

<!--en:1236efd3-->
::: {.slide title="Sequenzielles Lesen und zufällige Zugriffe auf 32-Byte-Sektoren"}

<!--en:861ebe13-->
[Zugriffsmuster]{.kicker}

<!--en:5fbe5e09-->
GPU-DRAM erreicht seine Nennbandbreite $\beta$ nur, wenn sich Zugriffe auf aufeinanderfolgende Adressen zusammenfassen lassen. Wir lesen ein $1\text{ GiB}$ großes `float32`-Array mit `x[idx]` aus, einmal mit der identischen Permutation und einmal mit einer Zufallspermutation:

<!--en:12e94181-->
**Warum zufällige Gather-Zugriffe $8\text{--}10\times$ so lange dauern:** Speichertransaktionen holen **32-Byte-Sektoren**. Bei jedem zufälligen 4-Byte-Zugriff bleiben $28$ von $32\text{ B}$ ungenutzt. Zufällige Adressen verfehlen immer wieder offene **DRAM-Zeilen von Kilobytegröße** und die Caches für die Adressübersetzung.

<!--en:3253bb5b-->
::: {.slide title="Blockübertragung und Blockgröße n_1/2 für halbe Bandbreite"}

<!--en:a9e4506f-->
[Übertragungsmodell nach Hockney]{.kicker}

<!--en:4d6b3c77-->
- **Blockgröße für halbe Bandbreite ($n_{1/2} = \alpha\beta$):** Bei dieser Nutzdatenmenge entspricht die Übertragungszeit $n/\beta$ der Anlauflatenz $\alpha$. Die erreichte Bandbreite beträgt $\frac{1}{2}\beta$.

<!--en:4d490071-->
- **Mischen von Daten auf NVMe ($\alpha \approx 70\,\mu\text{s}$, $\beta = 14\text{ GB/s}$):** Es gilt $n_{1/2} \approx \mathbf{1\text{ MB}}$. Zufällige Lesezugriffe auf je $4\text{ KB}$ lassen $>99\%$ der Bandbreite ungenutzt. Deshalb müssen Datensätze in Blöcken von Megabytegröße gespeichert werden.

<!--en:2ca8ab27-->
::: {.slide title="Engpass am Chiprand: Fläche für Recheneinheiten, Umfang für I/O"}

<!--en:88086dce-->
[Geometrie des Siliziumchips]{.kicker}

<!--en:b226c581-->
- **Fläche und Umfang:** Logiktransistoren füllen das Chipinnere ($O(L^2)$), I/O-Kontakte für externe Verbindungen liegen am *Chiprand* ($O(L)$).

<!--en:92b35be2-->
- **HBM auf einem Silizium-Interposer:** Vertikal gestapelte DRAM-Chips neben der GPU und Tausende Interposer-Leitungen im Mikrometerbereich ermöglichen der B200 $8\text{ TB/s}$ (gegenüber $\sim 0.5\text{ TB/s}$ für DDR5-Kanäle einer CPU).

<!--en:627a28cc-->
- **Mehrere Chips in einem Gehäuse:** Zwei Chips von maximaler Belichtungsfeldgröße (je $\sim 858\text{ mm}^2$) verdoppeln den nutzbaren Umfang. So finden acht HBM-Stapel Anschluss.

<!--en:50e9371a-->
::: {.slide title="Rechenleistung wächst von Generation zu Generation schneller als Bandbreite"}

<!--en:da7e4482-->
[Entwicklung der Hardware]{.kicker}

<!--en:afe26494-->
- **Unterschiedliches Wachstum (2017–2024):** Von V100 $\to$ A100 $\to$ H100 $\to$ B200 stieg die 16-Bit-Spitzenleistung $P$ der Tensor Cores um den Faktor $18\times$, die HBM-Bandbreite $\beta$ um den Faktor $8.9\times$.

<!--en:770f9622-->
- **Verdoppelter Knickpunkt ($I^* = P/\beta$):** Die Intensität, ab der die Recheneinheiten ausgelastet sind, stieg von $139\text{ FLOP/B}$ (V100) auf $281\text{--}295\text{ FLOP/B}$ (B200/H100).

<!--en:7ade6fe2-->
- **Folge für die Ausführung:** Eine Operation mit $I = 200\text{ FLOP/B}$ war auf der V100 rechenbegrenzt und ist auf H100 und B200 **bandbreitenbegrenzt**.

<!--en:575c0de0-->
::: {.slide title="Energiebedarf für Datenübertragung und Berechnung"}

<!--en:b01d7338-->
[Leistung und Energie]{.kicker}

<!--en:1149fd5a-->
- **Speicher außerhalb des Chips bestimmt die Leistungsaufnahme:** In einem CMOS-Prozess mit $45\text{ nm}$ (Horowitz, 2014) kostet eine `fp32`-Multiplikation $3.7\text{ pJ}$. Das Lesen von $64\text{ bits}$ aus externem DRAM kostet $1.3\text{--}2.6\text{ nJ}$, etwa so viel wie **500 `fp32`-Multiplikationen**.

<!--en:12144c91-->
- **SRAM-Zugriffe auf dem Chip brauchen $20\text{--}100\times$ weniger Energie** ($10\text{--}100\text{ pJ}$).

<!--en:344e9dd4-->
- **Die elektrische Leistung begrenzt die Bandbreite:** Wer Speicherverkehr einspart (durch Operatorfusion, schmalere Formate oder blockweise Verarbeitung auf dem Chip), spart aus demselben physikalischen Grund Watt wie Millisekunden.

<!--en:b2e6fbc2-->
::: {.slide title="Streaming Multiprocessors und das Verbergen von Latenz"}

<!--en:58dec86e-->
[Ausführungsmodell der GPU]{.kicker}

<!--en:afaf00b8-->
- **Warp-Wechsel ohne Zusatzaufwand:** Jeder der $128$ SMs führt Threads in **Warps** zu je 32 Threads aus. Wartet ein Warp auf einen DRAM-Zugriff von $250\text{ ns}$, kann der SM in jedem Takt zu einem anderen bereitstehenden Warp wechseln.

<!--en:36818245-->
- **Mindestauslastung nach dem Gesetz von Little:** Für dauerhaft $1.0\text{ TB/s}$ müssen $\ge 250\text{ KB}$ gleichzeitig unterwegs sein ($\sim 16$ aktive Warps pro SM mit 4-Byte-Lesezugriffen). Sehr kleine Thread-Grids können die DRAM-Latenz nicht verbergen.

<!--en:6096c028-->
::: {.slide title="Spitzenleistung in FLOP/s aus SMs, Takt und Tensor Cores berechnen"}

<!--en:f9421e0b-->
[Tensor Cores]{.kicker}

<!--en:2697ce8b-->
- **Systolische Multiplikation von Matrixblöcken:** Ein einziger Befehl mit `fp32`-Akkumulation verarbeitet einen ganzen Matrixblock. So verteilen Tensor Cores den Aufwand für das Laden des Befehls und den Registerzugriff auf den ganzen Block.

<!--en:3b6e2834-->
- **Durchsatz je nach Präzision:** `tf32` erreicht $82.6\text{ TFLOP/s}$ ($1\times$), `bf16`/`fp16` verdoppelt den Durchsatz auf $165.2\text{ TFLOP/s}$ ($2\times$) und `fp8` vervierfacht ihn auf $330.4\text{ TFLOP/s}$ ($4\times$).

<!--en:cd2f677f-->
::: {.slide title="Gleitkommaformate: Exponentenbereich und Präzision"}

<!--en:024af3ca-->
[Zahlenformate]{.kicker}

<!--en:c8e7c385-->
- **Warum `bfloat16` beim Training `float16` verdrängt hat:** `fp32`, `tf32` und `bf16` haben alle einen **8-Bit-Exponenten** (dynamischer Bereich $\sim 10^{\pm 38}$). Deshalb braucht `bf16` kein Loss-Scaling.

<!--en:48dc42ec-->
- **Doppelter Vorteil im Roofline-Modell:** Eine Halbierung der Bitbreite (außer bei `tf32`, das im DRAM als 32-Bit-Wert in `fp32` liegt) **halbiert die übertragene Datenmenge** und **verdoppelt die Spitzenleistung $P$ der Tensor Cores**. Formate mit höchstens 8 Bit (`fp8`, `fp4`) brauchen eine Mikroskalierung pro Tensor oder Block (16–32 Elemente).

<!--en:37ffb105-->
::: {.slide title="Technische Daten von GPUs für Rechenzentren und Endkunden"}

<!--en:4c4e1fe3-->
[Beschleuniger im Vergleich]{.kicker}

<!--en:233e6669-->
Der Vergleich zweier Spitzenmodelle für Rechenzentren mit zwei Endkundenkarten zeigt, welche Abstriche die Endkundenkarten bei Speicherkapazität, Bandbreite und Verbindungen zwischen GPUs machen:

<!--en:6fbf18ae-->
| Kenngröße | H100 SXM | B200 | RTX 4090 (bei uns) | RTX 5090 |

<!--en:698f600d-->
| **Speicherkapazität ($C$)** | $80\text{ GB HBM3}$ | $180\text{ GB HBM3e}$ | $24\text{ GB GDDR6X}$ | $32\text{ GB GDDR7}$ |

<!--en:95c2cfb4-->
| **Speicherbandbreite ($\beta$)** | $3.35\text{ TB/s}$ | $8.0\text{ TB/s}$ | $1.01\text{ TB/s}$ | $1.79\text{ TB/s}$ |

<!--en:ae6ba066-->
| **Tensor Cores mit `bfloat16` ($P$)** | $989\text{ TFLOP/s}$ | $2{,}250\text{ TFLOP/s}$ | $165.2\text{ TFLOP/s}$ | $209.5\text{ TFLOP/s}$ |

<!--en:b06ec2af-->
| **Tensor Cores mit `fp8` / `fp4`** | $1{,}979$ / – | $4{,}500$ / $9{,}000$ | $330.4$ / – | $419$ / $1{,}676$ |

<!--en:3e9fbb60-->
| **Knickpunkt ($I^*$, `bf16`)** | $295\text{ FLOP/B}$ | $281\text{ FLOP/B}$ | $164\text{ FLOP/B}$ | $117\text{ FLOP/B}$ |

<!--en:47d2578c-->
| **GPU-Verbindung (eine Richtung)** | NVLink 4 ($450\text{ GB/s}$) | NVLink 5 ($900\text{ GB/s}$) | PCIe 4.0 ($\le 32\text{ GB/s}$) | PCIe 5.0 ($\le 64\text{ GB/s}$) |

<!--en:4b9de433-->
::: {.slide title="Vom Host zur GPU: auslagerbarer und auslagerungsgesperrter Speicher"}

<!--en:59dc1eb4-->
[Datenpfade des Hosts]{.kicker}

<!--en:7fe7216f-->
Die DMA-Kopiereinheiten der GPU können Minibatches nur dann parallel zur Berechnung auf den SMs über PCIe übertragen, wenn der Quellpuffer in **auslagerungsgesperrtem** Host-RAM liegt:

<!--en:98d38aee-->
- **Warum auslagerbarer Speicher langsamer ist:** Das Betriebssystem kann solche Speicherseiten jederzeit auslagern. Deshalb muss der Treiber die Arrays vor dem DMA-Transfer auf der CPU in einen eigenen auslagerungsgesperrten Puffer kopieren.

<!--en:fdbf2ad1-->
- **Faustregel:** Kleine Datensätze im GPU-DRAM belassen. Bei fortlaufender Übertragung die Batches auf dem Host in auslagerungsgesperrtem Speicher anlegen (`pin_memory=True` / `'pinned_host'`) und asynchron kopieren (`non_blocking=True`).

<!--en:017b8b3c-->
::: {.slide title="Verbindungen zwischen GPUs: NVLink-Verbindungsnetze und PCIe über den Host"}

<!--en:6827c9e1-->
[Topologie mehrerer GPUs]{.kicker}

<!--en:f0aa8e1a-->
- **Weder NVLink noch Peer-to-Peer-Zugriff (P2P) über PCIe:** Wie `nvidia-smi topo` zeigt, läuft auf unserer Workstation jedes Byte von GPU zu GPU über eine PCIe-4.0-Verbindung mit $\times 16$ Lanes zum Host-DRAM und über eine zweite zur Ziel-GPU ($\le 23\text{ GB/s}$, fast **$40\times$ langsamer** als NVLink 5 mit $900\text{ GB/s}$).

<!--en:6c670de2-->
- **Folge für unser GPT mit 1 Mrd. Parametern ($4\text{ GB}$ an `fp32`-Gradienten):** Die Übertragung der Gradienten dauert $\ge 0.17\text{ s}$ über PCIe mit Umweg über den Host, gegenüber $< 5\text{ ms}$ über NVLink 5.

<!--en:c434f3fb-->
::: {.slide title="LLM-Inferenz im Roofline-Modell: Prefill und Dekodierung"}

<!--en:2881e5bb-->
[Analyse der Arbeitslast]{.kicker}

<!--en:4cb24938-->
Bei der autoregressiven Inferenz eines Sprachmodells mit $N$ Parametern in `bfloat16` ($2N$ Bytes) unterscheiden wir zwei Bereiche der Roofline:

<!--en:b668aa7a-->
- **Obergrenze bei einer einzelnen Sequenz:** Für jedes neue Token liest das Modell alle $2N$ Bytes der Gewichte sowie den KV-Cache. Bei der Intensität $I \approx 1 \ll I^*$ bestimmt allein die Speicherbandbreite $\beta$ die Tokenrate:
  $$\textrm{tokens/s} \;\lesssim\; \frac{\beta}{\textrm{bytes moved per token}} \;=\; \frac{\beta}{2N + \textrm{KV bytes}}$$

<!--en:445aee07-->
- **GPT mit 1 Mrd. Parametern auf der RTX 4090 ($\beta \approx 0.9\text{ TB/s}$, $2N = 2\text{ GB}$):** Bei kurzem Kontext gilt $\textrm{tokens/s} \lesssim \frac{900\text{ GB/s}}{2\text{ GB}} \approx \mathbf{450\text{ tokens/s}}$.

<!--en:21c5f513-->
::: {.slide title="20 ms für einen Speicherdurchlauf und Dekodierung in Batches"}

<!--en:ffecd950-->
[Kapazität und Zeit für einen Speicherdurchlauf]{.kicker}

<!--en:353a2c75-->
- **Nahezu konstante $\sim 20\text{ ms}$ pro Speicherdurchlauf ($C/\beta$):** Das gesamte GPU-DRAM einmal zu lesen dauert auf RTX 4090, RTX 5090, H100 und B200 jeweils $18\text{--}24\text{ ms}$.

<!--en:ce62d29d-->
- **Obergrenze für eine Sequenz bei voller Karte:** Ein Modell, dessen Gewichte den Kartenspeicher füllen, dekodiert eine einzelne Sequenz mit $\beta/C \approx \mathbf{42\text{--}56\text{ tokens/s}}$.

<!--en:04c98d60-->
- **Dekodierung im Batch ($B$ Sequenzen):** Jeder Lesezugriff auf die Gewichte dient $B$ Token ($I \approx B$). Der Gesamtdurchsatz steigt linear, bis $B \approx I^*$ gilt.

<!--en:9809da66-->
::: {.slide title="Verständnisfragen zu Bandbreite, Zahlenformaten und Dekodierung"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:394a8301-->
Schätzen Sie die folgenden Größen für die RTX 4090 ab ($\beta_{\textrm{achieved}} \approx 0.9\text{ TB/s}$, `bf16`-Spitzenleistung $P = 165\text{ TFLOP/s}$, `fp8`-Spitzenleistung $330\text{ TFLOP/s}$, Knickpunkt $I^* = 164\text{ FLOP/byte}$):

<!--en:d82f2004-->
1. Wir wechseln von `bfloat16` zu `fp8`, und zwar (a) für einen elementweisen Kernel mit $I = 0.25\text{ FLOP/B}$ und (b) für ein großes Matrixprodukt mit $I = 1{,}000\text{ FLOP/B}$. Um welchen Faktor beschleunigt sich die Rechnung jeweils, und warum erreicht keiner der beiden Fälle $4\times$?

<!--en:d62d1ae2-->
2. Ein Modell mit 8 Mrd. Parametern in `bfloat16` ($16\text{ GB}$ an Gewichten, KV-Cache vernachlässigt) läuft auf der RTX 4090. Wie hoch ist der maximale Tokendurchsatz für **1 Sequenz** ($B = 1$) und insgesamt für **64 gleichzeitige Sequenzen** ($B = 64$)?

<!--en:557b98b6-->
**Antworten:**

<!--en:034b037c-->
1. **(a) $2\times$ schneller** (bandbreitenbegrenzt, halbe übertragene Datenmenge). **(b) $2\times$ schneller** (rechenbegrenzt, Verdopplung der Spitzenleistung $P$ der Tensor Cores: $165 \to 330\text{ TFLOP/s}$). Keiner der beiden Fälle erreicht $4\times$, weil jeweils nur eine Grenze wirksam ist.

<!--en:811a5b39-->
2. **1 Sequenz ($B = 1$):** Das Lesen von $16\text{ GB}$ mit $0.9\text{ TB/s}$ dauert $17.8\text{ ms/step}$ $\to$ **$\approx 56\text{ tokens/s}$**. **Batch mit $B = 64$:** Die Dekodierung bleibt bandbreitenbegrenzt ($I \approx 64 < 164$). Jeder Schritt dauert weiterhin $\sim 17.8\text{ ms}$ $\to$ insgesamt **$64 \times 56 \approx 3{,}600\text{ tokens/s}$**.

<!--en:a5e27904-->
::: {.slide title="Zusammenfassung: Physikalische Gesetzmäßigkeiten und typische Größenordnungen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:d1a2ffc6-->
| Prinzip oder Kenngröße | Regel oder Formel | RTX 4090 / Workstation | praktische Konsequenz |

<!--en:e4f0c827-->
| **Bandbreite der Leitungen** | $\beta = \textrm{wires} \times \textrm{rate/wire}$ | $1.0\text{ TB/s}$ beim DRAM gegenüber $23\text{ GB/s}$ bei PCIe | Mit jedem Schritt weg vom Chip sinken Leitungszahl und Bandbreite um den Faktor $\sim 10\text{--}40\times$ |

<!--en:f3d02f2e-->
| **Chiprand und Energie** | Rechenleistung $\propto L^2$, I/O $\propto L$ | $1\text{ DRAM read} \approx 500\text{ FLOPs}$ | Der Knickpunkt ($117\text{--}295\text{ FLOP/B}$) steigt über die Generationen |

<!--en:140ff3fb-->
| **Sektor- und Zeilenzugriff** | $t(n) = \alpha + n/\beta$, 32-Byte-Sektoren | Zufälliger Gather-Zugriff $\sim 10\times$ langsamer | Lesezugriffe zusammenfassen, Daten auf Laufwerken in Blöcken von $\ge 1\text{ MB}$ mischen |

<!--en:4d57db85-->
| **Gesetz von Little (SMs)** | $\textrm{In-flight} = \beta \times \textrm{latency}$ | $\approx 250\text{ KB}$ ($128$ SMs) | Kernel brauchen genügend Warps, um $250\text{ ns}$ DRAM-Latenz zu verbergen |

<!--en:ebf9d672-->
| **Geringere Präzision** | $\textrm{Bytes} \div 2$, Spitzenleistung der Tensor Cores $P \times 2$ | `tf32` $83\text{ TF/s} \to$ `bf16` $165\text{ TF/s}$ | Beschleunigt bandbreitenbegrenzte und rechenbegrenzte Kernel |

<!--en:33127d6d-->
| **Dekodierung und Speicherdurchlauf** | $\textrm{tok/s} \lesssim \beta / (2N + \textrm{KV})$ | $C/\beta \approx 24\text{ ms}$ ($\sim 42\text{ tok/s}$ bei vollem Speicher) | Bei einer einzelnen Sequenz begrenzt der Speicher die Dekodierung. Batching erhöht die Intensität auf $I \approx B$ |
