<!--en:b8374a83-->
# Hardware

<!--en:6865db47-->
Die meisten Leserinnen und Leser führen die Notebooks dieses Buches mit der GPU aus, die ihnen gerade zur Verfügung steht: auf einem Laptop, in einem gehosteten Notebook oder auf einer gemieteten Karte. Früher oder später stellt sich die Frage, ob sich ein eigener Rechner lohnt und welcher es sein soll. Angenommen, Sie möchten ein Sprachmodell mit 8 Milliarden Parametern anhand Ihrer eigenen Dokumente feinabstimmen. Ein Datenblatt enthält Dutzende von Kennzahlen. Selbst die scheinbar schnellste Karte kann für Ihr Modell zu klein sein.

<!--en:0521de38-->
Drei Größen entscheiden über fast jeden solchen Kauf: die Speicherkapazität (engl. memory capacity), die Speicherbandbreite (engl. memory bandwidth) und der Rechendurchsatz (engl. compute throughput). In :numref:`sec_hardware` haben wir gesehen, wie sie das Verhalten eines Rechners bestimmen. Hier geht es um die Kaufentscheidung. Ermitteln Sie, welche Größe Ihre Arbeitslast begrenzt, und wählen Sie die Hardware so, dass sie gerade davon mehr bietet. Die Rechner und Preise im Folgenden entsprechen dem Stand von November 2026. Das Angebot ändert sich, die Methode bleibt.

<!--en:a13c92f0-->
**Die Kapazität entscheidet, was überhaupt läuft.** Beim Training liegen die Gewichte, ihre Gradienten und der Optimiererzustand im Speicher. Zusammen beanspruchen sie ein Mehrfaches der Modellgröße. Distributed Data Parallel (DDP) hält auf jeder GPU eine Kopie. Fully Sharded Data Parallel (FSDP) und der Zero Redundancy Optimizer (ZeRO) verteilen den Zustand auf mehrere GPUs, wie wir in :numref:`sec_training_systems` sehen. Passt ein Modell nicht in den Speicher, läuft es auch auf der schnellsten Karte nicht. **Die Bandbreite bestimmt, wie schnell ein Modell Text erzeugt.** Für jedes neue Token muss das Modell alle Gewichte lesen, die es für dieses Token nutzt, und außerdem den KV-Cache. Dieser enthält die Schlüssel und Werte, die die Attention für alle früheren Token speichert (:numref:`sec_kv-cache` und für den Betrieb im großen Maßstab :numref:`sec_model_serving`). Daraus folgt:

<!--en:c85d2288-->
Nach dem Fine-Tuning und der Quantisierung der Gewichte auf 4 Bit liest das Modell mit 8 Milliarden Parametern bei kurzem Kontext etwa 4.5 GB pro Token. Mit einer Karte, die 900 GB/s überträgt, erzeugt es somit höchstens etwa 200 Token pro Sekunde. Bei der Verarbeitung des Prompts (*Prefill*) und beim Training verarbeitet das Modell viele Token gleichzeitig und nutzt jedes gelesene Gewicht mehrfach. Hier begrenzt also der Rechendurchsatz die Geschwindigkeit. Auf jeder dieser Achsen unterscheiden sich die Rechner um eine Größenordnung, oft in entgegengesetzter Richtung. Welcher Rechner sich am besten eignet, hängt daher von der Arbeitslast ab.

<!--en:561b64dc-->
## Anforderungen der Arbeitslast

<!--en:74734a5c-->
Ausgangspunkt für den Kauf ist die Arbeitslast. Erst dann wählen wir die Karte. Beim Training müssen wir wissen, ob der Speicher auch für den Spitzenbedarf ausreicht (:numref:`fig_tools_memory_fit`):

<!--en:d4d68ab5-->
![Spitzenbedarf an Speicher beim Training, ungefähr maßstabsgetreu für ein Modell mit 7 Mrd. Parametern in BF16 mit Adam: Zu den Parametern kommen Gradienten, Optimiererzustand und Aktivierungen hinzu. Für alles zusammen muss der Speicher des Beschleunigers noch eine Reserve bieten.](../img/tools-memory-fit.svg)

<!--en:efd6cea3-->
Für das vollständige Fine-Tuning eines Modells mit 7 Milliarden Parametern brauchen wir etwa 86 GiB. So viel Speicher hat keine Karte für den Verbrauchermarkt. QLoRA :cite:`Dettmers.Pagnoni.Holtzman.ea.2023` friert die Gewichte ein, speichert sie mit 4 Bit und trainiert kleine zusätzliche Adaptermatrizen. Damit sinkt der Speicherbedarf für den Trainingszustand bei derselben Aufgabe von 78 GiB auf etwa 4 GiB. So reicht eine Karte mit 16 GB für manches Adapter-Fine-Tuning mit 7 Milliarden Parametern. Für die Inferenz wenden wir die Obergrenze der Dekodierrate auf jeden Rechner in diesem Abschnitt an:

<!--en:4dc23cbd-->
Die tatsächliche Geschwindigkeit hängt von den Kerneln ab, doch die Obergrenze sagt die Rangfolge oft richtig voraus. Der DGX Spark und ein Rechner mit Strix Halo haben fast dieselbe Bandbreite und dekodieren ähnlich schnell (:numref:`subsec_hardware-buyers-unified`), obwohl der Spark etwa doppelt so viel kostet und wesentlich mehr Rechenleistung bietet.

<!--en:2df7f230-->
![Diskrete GPUs bieten bei mäßiger Kapazität eine sehr hohe Bandbreite. Bei Rechnern mit gemeinsamem Speicher verhält es sich umgekehrt. Die höchste Bandbreite bietet eine gemietete GPU für Rechenzentren (H100). Keines der Systeme liegt oben rechts, wo sowohl Kapazität als auch Bandbreite hoch sind.](../img/tools-hardware-menu.svg)

<!--en:ec19b250-->
:numref:`fig_tools_hardware_menu` ordnet typische Rechner auf diesen beiden Achsen ein. Sie bilden zwei Gruppen: schnelle diskrete GPUs mit wenig Speicher und langsamere Systeme mit großem gemeinsamem Speicher (engl. unified memory). Die RTX PRO 6000 aus der Workstation-Klasse verbindet die Vorzüge beider Gruppen zu einem hohen Preis.

<!--en:61b85b73-->
## Diskrete GPUs für das Training

<!--en:54a89667-->
Für viele Aufgaben beim Training und Fine-Tuning bietet eine diskrete NVIDIA-GPU hohe Speicherbandbreite und breite Softwareunterstützung. Dazu gehören die Frameworks dieses Buches, FlashAttention, Quantisierungskernel und die CUDA-Software, die auch auf Mietrechnern üblich ist.

<!--en:dbe0cef6-->
### Die RTX 5070 Ti und die gebrauchte RTX 3090

<!--en:af18ceb5-->
Beim ersten Trainingsrechner gibt die Speicherkapazität die Mindestanforderung vor. Die RTX 5070 Ti (16 GB GDDR7 bei 896 GB/s, 300 W) hatte bei der Einführung einen Listenpreis von 749 US-Dollar. Im November 2026 kostete sie im Handel etwa 1100–1300 US-Dollar. Sechzehn Gigabyte reichen für alle Notebooks dieses Buches, die auf einer einzelnen GPU laufen. Selbst die größten brauchen in der Spitze nur rund 13 GiB. Auch das QLoRA-Fine-Tuning von Modellen mit 7–8 Milliarden Parametern ist damit möglich. Eine Karte mit 12 GB schließt deutlich mehr Konfigurationen aus. Die Speicherpreise bestimmen auch die übrige Ausstattung: 64 GB DDR5 kosten etwa 1000–1200 US-Dollar, also ungefähr so viel wie die Karte. Die gebrauchte RTX 3090 (24 GB bei 936 GB/s, etwa 1250–1450 US-Dollar) bietet mehr Speicher bei ähnlicher Bandbreite. Sie nimmt allerdings 350 W auf, hat keine Tensorkerne für FP8 und FP4 und kommt ohne Garantie. Ihr Gebrauchtpreis stieg im Laufe des Jahres 2026.

<!--en:59ecdec3-->
### Die Workstation mit RTX 5090

<!--en:234f8576-->
Wenn 16 oder 24 GB nicht ausreichen, sind 32 GB die nächste Stufe bei Desktop-Rechnern. Die RTX 5090 (32 GB GDDR7 bei 1.79 TB/s, 575 W) hatte einen Listenpreis von 1999 US-Dollar. Im November 2026 war sie knapp und kostete neu etwa 6000–8000 US-Dollar. Sie ist die schnellste Karte für den Verbrauchermarkt. Sie bietet etwa die halbe Bandbreite einer H100, Tensorkerne für FP8 und FP4 sowie genug Speicher für vollständiges Fine-Tuning bis etwa 2 Milliarden Parameter und QLoRA-Fine-Tuning bis etwa 30 Milliarden. NVIDIA gibt für das System eine Netzteilleistung von 1000 W vor. Das Gehäuse muss 575 W Wärme abführen, und der Stromkreis muss das gesamte System versorgen. Ein ausgewogener Rechner (:numref:`fig_tools_workstation`) ergänzt die Karte um eine CPU mit 16 Kernen, 64–128 GB RAM und schnellen NVMe-Speicher. Insgesamt kostet er rund 8500–12 500 US-Dollar.

<!--en:540e2451-->
![Eine ausgewogene Workstation versorgt die GPU mit Daten, speichert Daten und Checkpoints und stellt auch unter Last genügend Strom und Kühlung bereit.](../img/tools-workstation.svg)

<!--en:eccc9f6b-->
In Systemen mit mehreren handelsüblichen GPUs addieren sich die Speicherkapazitäten. Häufig kommen zwei bis vier gebrauchte RTX 3090 zum Einsatz. Datenparallelität funktioniert über PCIe, doch übliche Desktop-Mainboards bieten nur einen Steckplatz mit voller Bandbreite. Wer solche Systeme baut, greift wegen der zusätzlichen PCIe-Lanes auf Threadripper- oder EPYC-Plattformen zurück. Auch zwei Netzteile sind üblich. Seit der 3090 bietet keine Karte für den Verbrauchermarkt mehr NVLink. Die kommunikationsintensive Tensorparallelität skaliert daher schlecht. Wer vor dem Kauf kurzzeitig einen Knoten mit mehreren GPUs mietet (:numref:`sec_cloud_instances`), kann prüfen, ob die Bandbreite zwischen den GPUs die Arbeitslast begrenzt.

<!--en:2fa5a551-->
## Rechner mit gemeinsamem Speicher für die Inferenz

<!--en:fce9a3be-->
![Diskrete GPUs kopieren Daten zwischen System-RAM und VRAM. Systeme mit gemeinsamem Speicher nutzen einen einzigen Adressraum. Kapazität und Bandbreite setzen aber weiterhin getrennte Grenzen.](../img/tools-memory-path.svg)

<!--en:19b5f8b7-->
Bei der zweiten Rechnergruppe steht eine andere Frage im Vordergrund: Wie groß darf ein Modell sein, damit man es zu Hause *ausführen* kann? Die Trainingsgeschwindigkeit ist hier zweitrangig. Diese Systeme geben der GPU direkten Zugriff auf einen großen LPDDR5X-Speicher, wie er in Laptops und Smartphones steckt. Er bietet viel Platz bei einem Bruchteil der Bandbreite von GDDR7 oder HBM (:numref:`fig_tools_memory_path`). Die Obergrenze der Dekodierrate macht den Zielkonflikt deutlich: Große Modelle passen in den Speicher, dekodieren aber langsam. Ein Modell mit einer Mixture of Experts (MoE) nutzt für jedes Token nur wenige seiner Expertennetze. Es liest daher nur seine aktiven Parameter und dekodiert schneller als ein dichtes Modell ähnlicher Gesamtgröße.

<!--en:a22622f7-->
:Rechner mit gemeinsamem Speicher für die lokale Inferenz (Stand: November 2026)

<!--en:ea79714a-->
| Rechner | Speicher | Bandbreite | Preis (ca.) | Anmerkungen |

<!--en:f6e4ab01-->
| Mini-PC mit AMD Strix Halo (Ryzen AI Max+ 395) | 128 GB | 256 GB/s | 3400–4400 US-Dollar | Framework, GMKtec, Minisforum, Beelink und weitere Anbieter. Unter Linux mit ROCm oder Vulkan |

<!--en:9908d388-->
| NVIDIA DGX Spark (GB10) | 64 oder 128 GB | 273 GB/s | 5000 US-Dollar (64 GB, bei OEM-Partnern), 6950 US-Dollar (128 GB) | Blackwell mit nativer CUDA-Unterstützung und schnellem Prefill. Bis zu vier Geräte bilden über ConnectX-7 einen Cluster |

<!--en:b1f0ea9e-->
| Mac Studio M5 Max | bis zu 128 GB | 614 GB/s | ab 2500 US-Dollar | MLX-Ökosystem. Basismodell mit 36 GB bei 460 GB/s |

<!--en:59009e0d-->
| Mac Studio M5 Ultra | 96–512 GB | 1.2 TB/s | ab 5500 US-Dollar | Größter Speicher und höchste Bandbreite in dieser Klasse |

<!--en:d759fe36-->
| MacBook Pro M5 Max | bis zu 128 GB | 614 GB/s | ab 3600 US-Dollar | M5 Max im Laptop. Basismodell mit 36 GB bei 460 GB/s |

<!--en:3d4ba3ed-->
Messungen aus der Community folgen ungefähr der Rangfolge nach Bandbreite. Das MoE-Modell gpt-oss-120b hat 117 Milliarden Parameter, nutzt aber nur etwa 5 Milliarden pro Token. In Benchmarks mit llama.cpp erzeugt es auf einem DGX Spark rund 60 Token/s und auf Strix Halo rund 50. Tests des M5 Ultra mit Ollama und MLX kommen auf etwa 115–120 Token/s. Ein dichtes Modell mit 70 Milliarden Parametern und 4-Bit-Gewichten liest etwa 40 GB pro Token. Schon die Bandbreitengrenze hält es auf dem Spark und auf Strix Halo unter 7 Token/s und auf dem M5 Ultra unter 30 Token/s. Stärker unterscheiden sich die Rechner beim Prefill: In denselben Benchmarks mit llama.cpp verarbeiten die Tensorkerne des Spark einen Prompt mit 2048 Token etwa zweieinhalbmal so schnell wie Strix Halo. Auch die Software spielt eine Rolle (CUDA auf dem Spark, ROCm oder Vulkan auf Strix Halo, MLX auf Apple-Rechnern). Messen Sie deshalb mit den Modellen, die Sie einsetzen möchten.

<!--en:bcf8a85c-->
## GPUs für Workstations

<!--en:784737af-->
Wer zu Hause etwa ein Modell mit 70 Milliarden Parametern feinabstimmen möchte, braucht sowohl Kapazität als auch Bandbreite und landet damit bei Workstation-Karten. Die RTX PRO 6000 Blackwell bietet 96 GB GDDR7 mit denselben 1.79 TB/s wie die 5090. Bei ihrer Einführung 2025 kostete sie 8565 US-Dollar. Im November 2026 bot NVIDIA sie im eigenen Shop für 16 000 US-Dollar an. Die Max-Q-Variante mit 300 W hat denselben Speicher, und vier dieser Karten passen in ein Gehäuse. Zwei Karten bieten 192 GB. Das reicht bei 70 Milliarden Parametern für die Inferenz ohne Quantisierung und für LoRA-Fine-Tuning. Vier Karten bieten 384 GB, genug für MoE-Modelle mit mehreren Hundert Milliarden Parametern bei 4 Bit. Seit der Generation nach Ampere haben Workstation-Karten kein NVLink mehr. Daten laufen zwischen den Karten daher über PCIe 5.0 mit etwa 64 GB/s je Richtung. Das ist fast um den Faktor dreißig langsamer als der Zugriff jeder Karte auf den eigenen Speicher. Datenparallele und pipelineparallele Arbeitslasten kommen damit zurecht. Bei Tensorparallelität sinkt die Geschwindigkeit.

<!--en:5a61beab-->
Bei einem solchen Budget lohnt sich eine Gegenrechnung. Für den Kaufpreis von vier Karten, 64 000 US-Dollar, lässt sich eine H100 zum Marktplatzpreis von 2 US-Dollar pro Stunde etwa 32 000 Stunden lang mieten (:numref:`tab_cloud_prices`). Der Kauf rechnet sich, sobald die Miete für die produktiven Stunden den Kaufpreis erreicht:

<!--en:b6d80708-->
Bei acht produktiven Stunden pro Tag entsprechen 5250 Stunden etwa 1.8 Jahren. Ein Rechner zu Hause steht öfter ungenutzt da, als man beim Kauf erwartet. Stromkosten, Wiederverkaufswert und Geschwindigkeitsunterschiede zwischen den Rechnern verändern die Rechnung (:exref:`ex_hardware-break-even`). Für den Kauf sprechen der Datenschutz, der Zugang ohne Warteschlange und der Wert von Hardware, die man selbst auseinandernehmen kann.

<!--en:bfd3061c-->
## Quellen für aktuelle Zahlen

<!--en:58cb0dc5-->
Die genannten Preise werden sich ändern, und auch neuere Messwerte brauchen eine Gegenprobe. Hinweise zum Rechnerbau und aktuelle Handelspreise finden Sie auf [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) (Sammelthreads zu Hardware) und im [Forum von Level1Techs](https://forum.level1techs.com/) (Bauberichte zu Systemen mit mehreren GPUs und Hinweise zu Strix Halo). [ServeTheHome](https://www.servethehome.com/) veröffentlicht Tests von Strix-Halo-Rechnern, dem DGX Spark und gebrauchten Servern. Vergleichbare Zahlen liefern die Benchmark-Beiträge in den [Diskussionen zu llama.cpp](https://github.com/ggml-org/llama.cpp/discussions). Dort kommt `llama-bench` mit denselben Modelldateien auf vielen Rechnern zum Einsatz. Der zuletzt 2023 aktualisierte [Leitfaden](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) von Tim Dettmers zeigt, wie man Hardware anhand der Arbeitslast auswählt. Aktuelle Empfehlungen für einzelne Komponenten gibt er nicht. Wer nach GPUs sucht, stößt häufig auf maschinell erzeugte Seiten mit geschätzten oder erfundenen Benchmark-Ergebnissen. Verlassen Sie sich auf Datenblätter, namentlich bekannte Tester und Messwerte, die mit der Obergrenze der Dekodierrate vereinbar sind.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:52a27664-->
Die Hardwareauswahl beginnt bei der physikalischen Größe, die die Arbeitslast begrenzt. Die Speicherkapazität entscheidet, was überhaupt läuft. Die Speicherbandbreite bestimmt die Geschwindigkeit der autoregressiven Dekodierung. Mit ihr lässt sich die Tokenrate bis auf einen Faktor zwei vorhersagen. Der Rechendurchsatz bestimmt die Geschwindigkeit von Training und Prefill. Im November 2026 eignet sich eine RTX 5070 Ti mit 16 GB (etwa 1100–1300 US-Dollar) als diskrete GPU für den Einstieg ins Training. Eine RTX 5090 mit 32 GB (etwa 6000–8000 US-Dollar) ist das Herzstück eines High-End-Desktops und stellt hohe Anforderungen an Stromversorgung und Kühlung. Gebrauchte RTX 3090 bieten 24 GB Kapazität, dafür fehlen Garantie und neuere Tensorformate. Systeme mit gemeinsamem Speicher (Strix Halo, DGX Spark und Apple Silicon) bieten umgekehrt viel Kapazität bei geringerer Bandbreite. Damit sind dünn besetzte Mixture-of-Experts-Modelle besonders attraktiv. Rechner mit ähnlicher Bandbreite unterscheiden sich vor allem beim Prefill. Workstation-Karten der Klasse einer RTX PRO 6000 mit 96 GB verbinden die Vorteile beider Gruppen. Vergleichen Sie die fünfstelligen Preise stets mit den Mietkosten für dieselbe Arbeit in der Cloud. Halten Sie zu jedem Preis das Datum fest. Prüfen Sie Messwerte anhand von Datenblättern und Originalmessungen aus der Community. Maschinell erzeugte Seiten in den Suchergebnissen sind dafür ungeeignet.

<!--en:2294243a-->
## Übungen

<!--en:efe6896b-->
1. [code] **Speicherbedarf beim Training.** `training_memory_gib` veranschlagt pro Parameter 2 Byte für Gewichte, 2 Byte für Gradienten und 8 Byte für den Optimiererzustand. Für Aktivierungen und Arbeitspuffer kommen feste Beträge hinzu.
    1. Beim Training mit gemischter Genauigkeit und Adam zählen :citet:`Rajbhandari.Rasley.Ruwase.ea.2020` 16 Byte pro Parameter. Welchen Tensor lassen die Standardwerte von `training_memory_gib` aus? Wie ändert sich die Schätzung für das Modell mit 7 Milliarden Parametern, wenn Sie ihn mitrechnen? Bestimmen Sie für beide Zählweisen das jeweils größte Modell, dessen vollständiges Fine-Tuning auf Karten mit 16, 24, 32 und 96 GB passt. Behandeln Sie die GB-Angabe der Karten als GiB.
    1. LoRA :cite:`Hu.Shen.Wallis.ea.2021` friert die vortrainierten Gewichte ein und trainiert kleine zusätzliche Adaptermatrizen. QLoRA :cite:`Dettmers.Pagnoni.Holtzman.ea.2023` speichert die eingefrorenen Gewichte zusätzlich mit 4 Bit, zuzüglich 0.127 Bit pro Parameter für Quantisierungskonstanten. Erweitern Sie `training_memory_gib` um ein Argument für den Anteil der trainierbaren Parameter. Für einen eingefrorenen Parameter zählen nur die gespeicherten Byte, für einen trainierbaren wie bisher Gewichte, Gradienten und Optimiererzustand. Werten Sie die Funktion für ein Modell mit 8 Milliarden Parametern und 1% trainierbaren Parametern aus. Verwenden Sie die Standardwerte für Aktivierungen und Arbeitspuffer, einmal für LoRA mit eingefrorenen bf16-Gewichten und einmal für QLoRA. Welches Fine-Tuning passt auf eine Karte mit 16 GB? Welcher Term ist in diesem Fall am größten?
    1. Der Speicherbedarf der Aktivierungen wächst mit der Zahl der Token pro Schritt. Beim blockweisen Aktivierungs-Checkpointing (:numref:`subsec_mp-checkpointing`) speichert ein Modell nur die Eingabe jedes Blocks, also $d$ bf16-Werte pro Token. Beim Neuberechnen eines Blocks kommen dessen Aktivierungen hinzu. Vernachlässigen Sie diese in dieser Teilaufgabe. Verwenden Sie für das QLoRA-Fine-Tuning der zweiten Teilaufgabe die Architektur von Llama 3.1 8B :cite:`Grattafiori.Dubey.Jauhri.ea.2024` mit 32 Blöcken der Breite $d = 4096$ und einem Vokabular von 128 256 Token. Für wie viele Token pro Schritt reicht der Speicher, der auf der Karte mit 16 GB für Aktivierungen übrig bleibt?
    1. Die Ausgabeschicht speichert für den Verlust zusätzlich ein bf16-Logit je Vokabulareintrag und Token. Wie viele Token pro Schritt passen in den Speicher, wenn Sie die Logits mitzählen? Vergleichen Sie die Zahl der Byte pro Token für die Logits mit der für die Blockeingaben.
   :exlabel:`ex_hardware-training-memory`

<!--en:d7cc792f-->
1. **Kontextlänge und Dekodierung.** Jeder Dekodierschritt liest außer den Gewichten auch den KV-Cache für alle früheren Token. Deshalb veranschlagt der Code, der `bandwidth_gbs` definiert, für ein Modell mit 8 Milliarden Parametern bei 4 Bit pro Token `active_gb = 4.5` GB. Ein Modell mit $L$ Schichten und $H_\textrm{kv}$ Schlüssel-Wert-Köpfen der Dimension $d_h$ speichert $2 L H_\textrm{kv} d_h$ Werte je Kontexttoken, nämlich einen Schlüssel und einen Wert pro Kopf und Schicht (:numref:`sec_kv-cache`). Llama 3.1 8B :cite:`Grattafiori.Dubey.Jauhri.ea.2024` hat 32 Schichten und 8 Schlüssel-Wert-Köpfe der Dimension 128 und speichert seinen Cache in bf16.
    1. Berechnen Sie die Cachegröße in Byte pro Token. Welcher Kontextlänge entsprechen die 0.5 GB an Cache-Datenverkehr in `active_gb`?
    1. Bestimmen Sie die Obergrenze der Dekodierrate für die RTX 5070 Ti und den M5 Ultra bei Kontextlängen von 4K, 32K und 128K Token. Geben Sie jeweils an, welcher Anteil der pro Token gelesenen Byte auf den Cache entfällt. Bei welchen dieser Kontextlängen passen Gewichte und Cache noch auf die Karte mit 16 GB?
    1. Die Quantisierung der Gewichte von bf16 auf 4 Bit verringert ihren Speicherbedarf um den Faktor 4. Um welchen Faktor steigt dadurch die Obergrenze der Dekodierrate bei 4K Token und um welchen bei 128K Token? Was bedeutet das für die Kaufentscheidung, wenn die Arbeitslast lange Kontexte verwendet?
   :exlabel:`ex_hardware-context-decode`

<!--en:e7e0de89-->
1. **Dekodierung mit Mixture of Experts.** Ein Mixture-of-Experts-Modell (MoE) speichert viele Expertennetze, führt aber für jedes Token nur wenige davon aus. Daher liest jeder Dekodierschritt nur die *aktiven* Parameter. Betrachten Sie ein 30B-A3B-Modell mit 30 Milliarden Parametern, von denen pro Token 3 Milliarden aktiv sind, und ein dichtes Modell mit 30 Milliarden Parametern. Beide verwenden 4-Bit-Gewichte. Vernachlässigen Sie den KV-Cache.
    1. Bestimmen Sie mit :eqref:`eq_hardware-buyers-decode` die Obergrenze der Dekodierrate beider Modelle auf jedem Rechner in :numref:`tab_unified_memory`. Auf welchen Rechnern dekodiert das jeweilige Modell schneller als 20 Token/s, einer gängigen Schwelle für den interaktiven Einsatz?
    1. Ein dichtes Modell mit 3 Milliarden Parametern bei 4 Bit liest pro Token genauso viele Byte wie das MoE-Modell. Was muss man beim Kauf für das MoE-Modell zusätzlich einplanen? Auf welchen Karten in `bandwidth_gbs` läuft das MoE-Modell?
    1. Laut Modellkarte von gpt-oss-120b :cite:`OpenAI.2025` entfallen 114.71 Milliarden Parameter auf die Experten der 36 MoE-Schichten, 0.96 Milliarden auf die Attention und 1.16 Milliarden auf die gleich großen Einbettungs- und Ausgabematrizen mit getrennten Gewichten. Jede Schicht hat 128 Experten, von denen ein Token 4 nutzt. Die Experten sind in MXFP4 mit 4.25 Bit pro Parameter gespeichert, alle übrigen Gewichte in bf16. Berechnen Sie, wie viele Byte ein Dekodierschritt liest. Berücksichtigen Sie, dass zum Nachschlagen der Einbettung eine einzige Zeile genügt. Bestimmen Sie die Obergrenzen der Dekodierrate auf dem DGX Spark und auf Strix Halo. Vergleichen Sie sie mit den in :numref:`subsec_hardware-buyers-unified` angegebenen Raten von llama.cpp.
    1. Die auf dem DGX Spark gemessene Rate übersteigt Ihre Obergrenze. Die Modelldatei von llama.cpp, die der Messung zugrunde liegt, belegt 59.02 GiB für 116.83 Milliarden Parameter. Was verrät ihre Größe über das Format der Gewichte außerhalb der Experten? Berechnen Sie erneut die Zahl der gelesenen Byte pro Dekodierschritt und den Anteil der jeweiligen Obergrenze, den die gemessenen Raten erreichen. Was sagt der Vergleich zwischen dem DGX Spark und Strix Halo darüber aus, wie gut sich die Dekodiergeschwindigkeit anhand der Bandbreite vorhersagen lässt?
   :exlabel:`ex_hardware-moe-decode`

<!--en:6ec7db4d-->
1. [code] **Kostenschwelle mit Betriebskosten.** Der Code zur Berechnung der Kostenschwelle dividiert `purchase` durch `rental_per_productive_hour`. Der Nenner gibt an, wie viel Miete für die Arbeit anfällt, die der eigene Rechner in einer produktiven Stunde erledigt.
    1. Ein eigener Rechner verbraucht in jeder produktiven Stunde Strom und lässt sich später weiterverkaufen. Formulieren Sie die Bedingung für Kostengleichheit bei einem Kaufpreis $C$, einem Wiederverkaufswert $R$, einer Leistungsaufnahme $p$ an der Steckdose unter Last, einem Strompreis $e$ pro kWh und einem Mietpreis $r$ pro produktiver Stunde. Lösen Sie nach der Anzahl produktiver Stunden $H$ auf.
    1. Berechnen Sie $H$ für den Desktop-Rechner mit RTX 5070 Ti aus :numref:`subsec_hardware-buyers-training`. Er kostet 3000 US-Dollar, hat nach drei Jahren einen Wiederverkaufswert von 800 US-Dollar und nimmt an der Steckdose 450 W auf. Vergleichen Sie ihn mit einer RTX 4090 zum Marktplatzpreis aus :numref:`tab_cloud_prices` von 0.35 US-Dollar pro Stunde. Rechnen Sie mit einem Strompreis von 0.20 US-Dollar pro kWh. Nehmen Sie an, dass die 5070 Ti pro Stunde 70% der Arbeit einer 4090 erledigt. Wie vielen Jahren mit acht produktiven Stunden pro Tag entspricht $H$?
    1. Der Code vergleicht einen Rechner mit 5090 mit dem Mietpreis einer GPU der H100-Klasse. Er setzt damit für eine Stunde auf beiden Rechnern dieselbe Arbeitsmenge an. Schätzen Sie anhand von :numref:`tab_gpu_specs`, wie viel mehr Arbeit eine H100 pro Stunde bei rechenbegrenztem Training und bei bandbreitenbegrenzter Dekodierung erledigt. Berechnen Sie $H$ für beide Arbeitslasten erneut. Verwenden Sie für den Rechner mit 5090 aus :numref:`subsec_hardware-buyers-training` die Mitte seiner Preisspanne, ohne Wiederverkauf und mit 900 W an der Steckdose.
   :exlabel:`ex_hardware-break-even`

<!--en:f9e8b1f3-->
1. **Prefill und Dekodierung im Vergleich.** ● Bei einer Anfrage verarbeitet das Modell zunächst den Prompt (Prefill) und erzeugt dann Token für Token die Antwort (Dekodierung). Laut :numref:`subsec_hardware-buyers-unified` dekodiert gpt-oss-120b auf dem DGX Spark mit etwa 60 Token/s und auf Strix Halo mit etwa 50 Token/s. Nehmen Sie an, der Prefill verarbeitet auf dem DGX Spark 2400 Prompt-Token/s und auf Strix Halo 1000 Prompt-Token/s. Jede Antwort umfasst 500 Token.
    1. Berechnen Sie auf beiden Rechnern die Zeit bis zum ersten Token und die Gesamtdauer jeweils für eine Chatanfrage mit einem Prompt von 500 Token und für die Anfrage eines Programmieragenten mit einem Prompt von 30 000 Token. Um welchen Faktor ist der DGX Spark bei der jeweiligen Anfrage schneller? Der Spark mit 128 GB kostet etwa doppelt so viel wie ein Rechner mit Strix Halo (:numref:`tab_unified_memory`). Bei welcher Arbeitslast lässt sich der höhere Preis eher rechtfertigen?
    1. Rechnen Sie für die 5.13 Milliarden aktiven Parameter von gpt-oss-120b :cite:`OpenAI.2025` mit 2 FLOPs pro aktivem Parameter und Token. Vernachlässigen Sie die FLOPs der Attention-Scores. Rechnen Sie beide Prefill-Raten in dauerhaft erzielte FLOP/s um.
    1. Verarbeitet der Prefill den Prompt in Abschnitten von 512 Token, liest er pro Abschnitt jedes Gewicht einmal, das mindestens eines der Token darin nutzt. Nehmen Sie an, jedes Token wählt seine 4 Experten einer Schicht gleichverteilt zufällig aus den 128 Experten aus, unabhängig von den anderen Token. Wie viele verschiedene Experten einer Schicht liest ein Abschnitt im Mittel? Berechnen Sie mit der Zählweise aus Aufgabe :exref:`ex_hardware-moe-decode`c die Zahl der Byte, die ein Abschnitt liest. Bestimmen Sie anhand der Bandbreite die Obergrenze der Prefill-Rate jedes Rechners. Welcher Rechner könnte seine angenommene Prefill-Rate mit Abschnitten von 512 Token erreichen? Was folgt daraus für die Abschnittsgröße bei der Messung auf dem DGX Spark?
    1. Wie ändert sich die Bandbreitengrenze für den Prefill auf dem DGX Spark bei Abschnitten von 2048 Token? Warum braucht ein MoE-Modell längere Abschnitte als ein dichtes Modell mit derselben Anzahl aktiver Parameter, bevor der Prefill rechenbegrenzt ist?

<!--en:128d9e88-->
1. [extended] **Kalibrierung der Obergrenze der Dekodierrate.** ●● Laut Zusammenfassung sagt :eqref:`eq_hardware-buyers-decode` die Erzeugungsrate bis auf einen Faktor zwei voraus. Prüfen Sie diese Aussage auf Hardware, zu der Sie Zugang haben, und ersetzen Sie die Obergrenze durch ein kalibriertes Modell. Eine Laptop-CPU genügt, da llama.cpp auch auf CPUs läuft.
    1. Messen Sie, welche Bandbreite Ihr Rechner beim fortlaufenden Kopieren einer großen Datenmenge aufrechterhält. Vergleichen Sie den Wert mit der Bandbreite laut Datenblatt.
    1. Messen Sie mit `llama-bench` aus llama.cpp die Dekodierrate für mindestens sechs Konfigurationen mit unterschiedlichen Modellgrößen und Gewichtsformaten, etwa dichte Modelle mit 1, 3 und 8 Milliarden Parametern bei 4 und 8 Bit sowie ein MoE-Modell. Berechnen Sie aus Parameteranzahl und Format, wie viele Byte jede Konfiguration pro Token liest. Tragen Sie die gemessenen Raten gegen die Obergrenze auf.
    1. Passen Sie das Modell $1/\textrm{rate} = t_0 + \textrm{bytes}/(\eta \beta)$ mit der Bandbreite $\beta$ aus der ersten Teilaufgabe an die Messwerte an. Schätzen Sie so den Anteil $\eta$ der Bandbreite, den die Dekodierung erreicht, und die feste Zeit $t_0$ pro Token. Welche Konfigurationen weichen am stärksten von der angepassten Kurve ab und warum?
    1. Sagen Sie die Dekodierrate für eine Konfiguration voraus, die Sie bei der Anpassung ausgespart haben. Treffen Sie außerdem eine Vorhersage für eine Konfiguration mit einem Kontext von 16K Token und zählen Sie dabei die Cache-Byte aus Aufgabe :exref:`ex_hardware-context-decode` mit. Messen Sie beide Raten. Vergleichen Sie Ihre Vorhersagefehler mit dem Faktor zwei aus der Zusammenfassung und mit einem veröffentlichten Wert für Ihre Hardware aus den Quellen in :numref:`subsec_hardware-buyers-current`.

<!--en:7c8acad0-->
[Diskussion](https://d2l.discourse.group/t/36844)

<!--en:8b88de19-->
Hardware: Speicherkapazität, Bandbreitengrenzen, gemeinsamer Speicher und Gesamtbetriebskosten<br>**Speicherbedarf beim Training · Bandbreitengrenze der Dekodierung · diskrete GPUs mit GDDR7 · gemeinsamer LPDDR5X-Speicher · Rechnen mit MoE · Kostenschwelle für Miete und Kauf**

<!--en:0d20903d-->
::: {.slide title="Drei Größen entscheiden über jeden Hardwarekauf"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:3ed8c491-->
- **1. Speicherkapazität (GB / GiB):** Bestimmt, **was überhaupt läuft**. Übersteigen der Trainingszustand mit $16N$ oder bei der Inferenz die Gewichte samt KV-Cache den verfügbaren RAM/VRAM, spielt die Zahl der FLOP/s keine Rolle.

<!--en:7af44f31-->
- **2. Speicherbandbreite (GB/s oder TB/s):** Bestimmt, **wie schnell ein Modell Token erzeugt** ($\text{tok/s} \lesssim \beta / B_{\text{active}}$).

<!--en:697327ff-->
- **3. Rechendurchsatz (TFLOP/s):** Bestimmt **die Geschwindigkeit von Training und Prefill**, da beide die Gewichte für Hunderte von Token wiederverwenden.

<!--en:73de1c3a-->
::: {.slide title="Wie die Speicherkapazität das Training begrenzt"}

<!--en:2167a76e-->
[Speicherbudget]{.kicker}

<!--en:d8aa6b42-->
Warum braucht das Training oder das vollständige Fine-Tuning $4\times\text{--}8\times$ so viel Speicher wie das Laden desselben Modells für die Inferenz?

<!--en:7c43d2bf-->
- **Adam-Zustand bei gemischter Genauigkeit ($16\text{ B/param}$):**
  - BF16-Parameter ($2\text{ B}$) + BF16-Gradienten ($2\text{ B}$).
  - FP32-Hauptgewichte ($4\text{ B}$) + zwei FP32-Momente von Adam $m, v$ ($8\text{ B}$).
  - Ohne separate Kopie der Hauptgewichte ($12\text{ B/param}$) belegt allein der Zustand bereits $6\times$ so viel Speicher wie das BF16-Modell.

<!--en:0b2a764d-->
- **Aktivierungen und Arbeitspuffer:** Für die Backpropagation bleiben Aktivierungen aus dem Vorwärtsdurchlauf gespeichert (ihr Umfang wächst mit Batchgröße $\times$ Sequenzlänge). Zusammen mit den Arbeitspuffern von cuBLAS/cuDNN belegen sie mehrere zusätzliche GiB.

<!--en:c34d77f9-->
::: {.slide title="Spitzenbedarf an Trainingsspeicher nach Modellgröße abschätzen"}

<!--en:7c223d15-->
[Vollständiges Fine-Tuning und QLoRA]{.kicker}

<!--en:c7d52f78-->
Unsere Funktion addiert zum Speicherbedarf für den Parameterzustand ($12\text{ B/param}$ ohne Hauptgewichte oder $16\text{ B/param}$ mit Hauptgewichten) den Bedarf für Aktivierungen und Arbeitspuffer:

<!--en:8264dac2-->
- **Vollständiges Fine-Tuning mit 7 Mrd. Parametern ($\approx 86\text{--}112\text{ GiB}$):** Übersteigt die Kapazität jeder handelsüblichen Desktop-GPU ($16\text{--}32\text{ GB}$).

<!--en:2c439d6f-->
- **Wie QLoRA mit 7–8 Mrd. Parametern auf eine Karte mit 16 GB passt:**
  - Die eingefrorenen Basisgewichte liegen in 4-Bit-NormalFloat vor ($\sim 0.52\text{ B/param} \approx 4\text{ GiB}$ bei 8 Mrd. Parametern).
  - Nur die Adapter mit niedrigem Rang werden trainiert. Sie umfassen $\sim 1\%$ der Parameter ($\approx 0.08\text{B} \times 12\text{ B} \approx 1\text{ GiB}$). Damit bleiben $\sim 11\text{ GiB}$ für Aktivierungen und Logits.

<!--en:c9d41ea0-->
::: {.slide title="Bandbreitengrenze der autoregressiven Dekodierung"}

<!--en:f7af5858-->
[Roofline der Inferenz]{.kicker}

<!--en:4beaeab1-->
- **Rechenbeispiel (dichtes Modell mit 8 Mrd. Parametern bei 4 Bit):** Pro Token liest das Modell $B_w = 4.0\text{ GB}$ an Gewichten und $B_{\text{kv}} \approx 0.5\text{ GB}$ aus dem BF16-KV-Cache für 4K Token (insgesamt $4.5\text{ GB/token}$).

<!--en:8eac1c58-->
- **Warum die Recheneinheiten kaum ausgelastet sind:** $16\text{ GFLOPs}$ bei $4.5\text{ GB}$ ergeben eine Rechenintensität von $I \approx 3.5\text{ FLOP/B}$. Das ist um den Faktor $40\times\text{--}85\times$ weniger als am Knickpunkt der GPU-Roofline.

<!--en:11107663-->
::: {.slide title="Hardwarevergleich anhand der Bandbreitengrenze der Dekodierung"}

<!--en:4364a2e8-->
[Rangfolge nach Bandbreite]{.kicker}

<!--en:76672644-->
Mit $\beta / (4.5\text{ GB})$ lässt sich die Erzeugungsrate bis auf einen Faktor zwei vorhersagen: für diskrete Karten für den Verbrauchermarkt, Rechner mit gemeinsamem Speicher und eine H100 im Rechenzentrum.

<!--en:d14e3189-->
- **Die Bandbreite setzt die Obergrenze:**
  - Eine H100 SXM ($3{,}350\text{ GB/s}$) erreicht beim Dekodieren $\le 744\text{ tok/s}$.
  - Eine RTX 5090 oder PRO 6000 ($1{,}792\text{ GB/s}$) erreicht $\le 398\text{ tok/s}$.
  - Beim DGX Spark ($273\text{ GB/s}$) und bei Strix Halo ($256\text{ GB/s}$) liegen die Obergrenzen ähnlich hoch ($\le 57\text{--}61\text{ tok/s}$), obwohl der Spark doppelt so viel kostet und erheblich mehr Rechenleistung bietet.

<!--en:efe33c47-->
::: {.slide title="Wie der KV-Cache bei langem Kontext zum Flaschenhals wird"}

<!--en:1053cc5f-->
[Rechnen mit langem Kontext]{.kicker}

<!--en:17d55b40-->
- **Llama 3.1 8B ($128\text{ KiB/token}$):** Der KV-Cache wächst von $0.5\text{ GB}$ bei $4\text{K}$ Token ($11\%$ des Datenverkehrs) auf $4.0\text{ GB}$ bei $32\text{K}$ Token ($50\%$) und **$16.0\text{ GB}$ bei $128\text{K}$ Token** ($80\%$ des Datenverkehrs, zu viel für eine Karte mit 16 GB).

<!--en:87c1bf1a-->
- **Abnehmender Nutzen bei Quantisierung allein der Gewichte:** Mit 4-Bit-Gewichten dekodiert das Modell $3.6\times$ so schnell bei $4\text{K}$ Token Kontext, aber nur $1.6\times$ so schnell bei $128\text{K}$ Token, sofern der KV-Cache nicht ebenfalls quantisiert ist.

<!--en:e850e799-->
::: {.slide title="Zwei Gruppen von Hardware für den lokalen Einsatz"}

<!--en:acf23fe9-->
[Hardware im Überblick]{.kicker}

<!--en:90a6d62c-->
Im Diagramm mit **Speicherkapazität** auf der horizontalen und **Speicherbandbreite** auf der vertikalen Achse zeichnen sich deutliche Unterschiede zwischen den Architekturen ab:

<!--en:e2215859-->
- **Diskrete GPUs für den Verbrauchermarkt (oben links):** Sehr schneller GDDR6X/GDDR7-Speicher ($896\text{--}1792\text{ GB/s}$) mit mäßiger Kapazität ($16\text{--}32\text{ GB}$). Schnell beim Training und bei kleinen bis mittelgroßen Modellen.

<!--en:5af0f79e-->
- **Systeme mit gemeinsamem Speicher (unten rechts):** Sehr großer LPDDR5X-Speicher ($64\text{--}512\text{ GB}$) mit geringerer Bandbreite ($256\text{--}1200\text{ GB/s}$). Selbst sehr große Modelle laufen lokal, allerdings bei geringerer Tokenrate.

<!--en:a688785d-->
- **Karten für Workstations und Rechenzentren (oben auf der Bandbreitenachse):** Die RTX PRO 6000 ($96\text{ GB}$, $1.79\text{ TB/s}$) und eine gemietete H100 ($80\text{ GB}$, $3.35\text{ TB/s}$) sind bei mittlerer Kapazität am schnellsten. Die Kaufpreise sind fünfstellig. Oben rechts liegt keine Karte.

<!--en:fd446b99-->
::: {.slide title="Diskrete GPUs für den Trainingseinstieg: 5070 Ti und gebrauchte 3090"}

<!--en:4823fc5d-->
[Desktop-Karten mit 16 bis 24 GB]{.kicker}

<!--en:a7389bae-->
- **Neue RTX 5070 Ti (16 GB GDDR7, $896\text{ GB/s}$):** Für alle D2L-Notebooks mit einer GPU ($< 13\text{ GiB}$) und QLoRA mit 7–8 Mrd. Parametern. Native Blackwell-Tensorkerne für **FP8/FP4**, $300\text{ W}$ Leistungsaufnahme und Garantie beim Neukauf.

<!--en:1c8e3ef7-->
- **Gebrauchte RTX 3090 (24 GB GDDR6X, $936\text{ GB/s}$):** Statt FP8/FP4 und Garantie bietet sie **mehr VRAM ($+8\text{ GB}$)** zu einem ähnlichen Handelspreis ($\sim\$1{,}250\text{--}1{,}450$).

<!--en:8dc096e7-->
::: {.slide title="Training auf dem High-End-Desktop: die Workstation mit RTX 5090"}

<!--en:189c7d54-->
[Ausgewogene Systemausstattung]{.kicker}

<!--en:f5392d9f-->
Der Schritt auf **32 GB GDDR7 bei $1.79\text{ TB/s}$** (RTX 5090, Handelspreis $\approx \$6{,}000\text{--}8{,}000$ im November 2026) ermöglicht vollständiges Fine-Tuning mit bis zu $\sim 2\text{B}$ Parametern und QLoRA mit bis zu $\sim 30\text{B}$:

<!--en:2f2789f5-->
- **Kühlung und Stromversorgung:** Allein die GPU gibt **$575\text{ W}$** Wärme ab. NVIDIA sieht ein Netzteil mit **$1000\text{ W}$**, ein gut belüftetes Gehäuse und einen eigenen, ausreichend belastbaren Stromkreis vor.

<!--en:410119ce-->
- **Datenversorgung der GPU (Gesamtpreis $\approx \$8{,}500\text{--}12{,}500$):** Eine CPU mit 16 Kernen, $64\text{--}128\text{ GB}$ DDR5-RAM und schnelle NVMe-SSDs mit PCIe Gen4/5 sorgen dafür, dass weder das Laden von Daten noch das Checkpointing die Berechnung ausbremst.

<!--en:6704ccdd-->
- **Grenzen handelsüblicher Systeme mit mehreren GPUs:** Seit der 3090 hat keine Karte für den Verbrauchermarkt mehr NVLink. Abseits der Threadripper- und EPYC-Plattformen bieten Desktop-Mainboards selten mehrere PCIe-Steckplätze mit jeweils vollen 16 Lanes.

<!--en:b8f5cfba-->
::: {.slide title="Gemeinsamer Speicher: Kapazität auf Kosten der Bandbreite"}

<!--en:b2910723-->
[Lokale Inferenz mit großen Modellen]{.kicker}

<!--en:01d75594-->
Wie kann ein Mini-PC oder Mac Studio ein Sprachmodell mit 70 oder 120 Mrd. Parametern ausführen, das auf eine RTX 5090 nicht passt?

<!--en:1819b934-->
- **Kopieren über PCIe als Flaschenhals vermeiden:** Diskrete GPUs kopieren Tensoren über PCIe zwischen dem DDR5-Hauptspeicher und ihrem kleinen, fest verlöteten GDDR7-Grafikspeicher.

<!--en:24113e35-->
- **Gemeinsamer LPDDR5X-Speicher:** CPU- und GPU-Kerne greifen über einen breiten Bus direkt auf denselben großen LPDDR5X-Speicher zu ($64\text{--}512\text{ GB}$).

<!--en:f705c159-->
- **Der physikalische Zielkonflikt:** Solche Systeme bieten $4\times\text{--}16\times$ so viel Speicherkapazität wie eine Desktop-GPU, bei $15\%\text{--}65\%$ der Speicherbandbreite einer RTX 5090.

<!--en:45184d67-->
::: {.slide title="Systeme mit gemeinsamem Speicher im Vergleich (November 2026)"}

<!--en:50b6ea44-->
[Systeme mit gemeinsamem Speicher]{.kicker}

<!--en:59b11367-->
- **Klasse mit $256\text{--}273\text{ GB/s}$ (128 GB):** AMD Strix Halo ($\sim\$3.4\text{K--}4.4\text{K}$) bietet den günstigsten Rechner mit 128 GB. Der NVIDIA DGX Spark ($\$5\text{K--}7\text{K}$) bietet zusätzlich natives CUDA und einen $2.4\times$ so schnellen Prefill.

<!--en:86065120-->
- **Klasse mit $614\text{--}1{,}200\text{ GB/s}$ (Apple Silicon):** Mac Studio und MacBook Pro mit M5 Max ($128\text{ GB}$, $614\text{ GB/s}$) sowie der M5 Ultra (bis zu $512\text{ GB}$, $1.2\text{ TB/s}$) bieten beim Dekodieren aus gemeinsamem Speicher die doppelte bzw. vierfache Bandbreite.

<!--en:b990f493-->
::: {.slide title="Warum sich Mixture of Experts (MoE) für gemeinsamen Speicher eignet"}

<!--en:5e8101e3-->
[Dekodierung mit dichten und dünn besetzten Modellen]{.kicker}

<!--en:1318e589-->
- **Dichtes Modell mit 70 Mrd. Parametern (4 Bit, $\sim 40\text{ GB}$ pro Token gelesen):** Passt in 128 GB gemeinsamen Speicher. Da es $40\text{ GB}$ pro Token liest, erreicht es auf Strix Halo und Spark nur $< 7\text{ tok/s}$ ($256\text{--}273\text{ GB/s}$).

<!--en:1a0f7bca-->
- **MoE-Modell `gpt-oss-120b` (117 Mrd. Parameter, 5.1 Mrd. pro Token aktiv):** Alle Experten zusammen belegen $\sim 59\text{ GiB}$, doch das Modell liest nur **$\sim 3\text{--}3.5\text{ GB/token}$**. Es dekodiert mit **$50\text{--}60\text{ tok/s}$** auf Strix Halo und Spark und mit **$115\text{--}120\text{ tok/s}$** auf dem M5 Ultra.

<!--en:aaa00cd1-->
::: {.slide title="Rechner mit gleicher Bandbreite unterscheiden sich beim Prefill"}

<!--en:7e7ae4eb-->
[Promptverarbeitung und Dekodierung]{.kicker}

<!--en:bedef404-->
- **Kurzer Chat (Prompt mit 500 Token und Antwort mit 500 Token):** Die Bandbreite beim Dekodieren gibt den Ausschlag. DGX Spark ($273\text{ GB/s}$) und Strix Halo ($256\text{ GB/s}$) brauchen daher ähnlich lange ($8.5\text{ s}$ bzw. $10.5\text{ s}$).

<!--en:076e1300-->
- **Prompt eines Programmieragenten mit 30 000 Token:** Der Prefill ist rechenbegrenzt. Die Blackwell-Tensorkerne des Spark verarbeiten $2{,}400\text{ tok/s}$ gegenüber $1{,}000\text{ tok/s}$ bei Strix Halo. Dadurch sinkt die Zeit bis zum ersten Token (**$30.0\text{ s} \to 12.5\text{ s}$**, um den Faktor $2.4\times$ schneller).

<!--en:71468c60-->
::: {.slide title="Workstation-GPUs: Ab wann lohnt sich der Kauf gegenüber der Miete?"}

<!--en:ffcf5006-->
[Gesamtbetriebskosten]{.kicker}

<!--en:db9875b4-->
Die **RTX PRO 6000 Blackwell** bietet **96 GB GDDR7 bei $1.79\text{ TB/s}$** ($\$16{,}000$, November 2026). Vier Max-Q-Karten mit je 300 W bringen $384\text{ GB}$ in einem Gehäuse unter. Sie kosten zusammen $\$64{,}000$ und sind über PCIe 5.0 mit $64\text{ GB/s}$ verbunden. Wann lohnt sich der Kauf gegenüber der Miete?

<!--en:b7e20bcd-->
- **Einfache Kostenschwelle:** Der Preis von $\$10{,}500$ für eine Workstation mit RTX 5090, geteilt durch den Marktplatzpreis einer H100 von $\$2.00/\text{hr}$, ergibt **$5{,}250$ produktive Stunden** ($\approx 1.8\text{ years}$ bei $8\text{ hr/day}$ an jedem Tag).

<!--en:7c4810de-->
- **Kostenschwelle unter Berücksichtigung von Strom, Wiederverkauf und relativer Geschwindigkeit $s$:**
  $$H = \frac{C_{\text{buy}} - R_{\text{resale}}}{s \cdot r_{\text{cloud}} - p_{\text{kW}} \cdot e_{\text{kWh}}}$$
  Eine H100 erledigt pro Stunde etwa $1.9\times$ (bandbreitenbegrenzte Dekodierung) bis $4.7\times$ (rechenbegrenztes Training) so viel Arbeit. Bei Stromkosten von $0.9\text{ kW} \times \$0.20/\text{kWh}$ rechnet sich der Kauf bei $8\text{ hr/day}$ nach **etwa 4 Jahren** (Dekodierung) bzw. **etwa 15 Jahren** (Training). Für Datenschutz, sofortigen Zugriff und eigene Hardwareexperimente spricht der Kauf, für kurzzeitige Lastspitzen die Miete.

<!--en:761c46b2-->
::: {.slide title="Benchmarks prüfen und maschinell erzeugte Angaben erkennen"}

<!--en:1b250133-->
[Messwerte prüfen]{.kicker}

<!--en:6e8addc1-->
- **Physikalische Plausibilität:** Behauptet ein Blog, eine Karte mit $900\text{ GB/s}$ dekodiere ein dichtes Modell von $35\text{ GB}$ mit $45\text{ tok/s}$ ($1{,}575\text{ GB/s} > 900\text{ GB/s}$), widerspricht das der Bandbreitengrenze.

<!--en:af83594b-->
- **Kalibriertes affines Modell und Primärquellen:** Den festen Zeitbedarf $t_0$ pro Token und die Busauslastung $\eta \in [0.6, 0.85]$ anhand standardisierter Messungen anpassen. Quellen sind Diskussionen zu `llama-bench`, Tests von ServeTheHome und Level1Techs sowie Messprotokolle auf `r/LocalLLaMA`.

<!--en:5ea85b37-->
::: {.slide title="Verständnisfragen: Hardwarebedarf und Roofline-Grenzen"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:a1144914-->
Drei Rechenaufgaben zum Hardwarebedarf:

<!--en:ad4fd1ba-->
1. Llama 3.1 8B soll mit LoRA ($1\%$ trainierbare Parameter, $6\text{ GiB}$ für Aktivierungen, $2\text{ GiB}$ für Arbeitspuffer) auf einer **RTX 5070 Ti mit 16 GB** feinabgestimmt werden. Passt Standard-LoRA mit eingefrorenen BF16-Basisgewichten in den Speicher, oder ist QLoRA mit eingefrorenen 4-Bit-Basisgewichten nötig?

<!--en:6b048553-->
2. Ein dichtes Modell mit 30 Mrd. Parametern und ein **30B-A3B-MoE-Modell** ($30\text{B}$ Parameter insgesamt, $3\text{B}$ pro Token aktiv) sind beide auf 4 Bit quantisiert ($0.5\text{ B/param}$). Welche Obergrenze setzt die Bandbreite der Dekodierrate auf einem Mac Studio M5 Max ($614\text{ GB/s}$), wenn man den KV-Cache vernachlässigt?

<!--en:825e5e16-->
3. Warum läuft das 30B-A3B-MoE-Modell nicht auf einer RTX 5070 Ti mit 16 GB, obwohl es pro Token nur $1.5\text{ GB}$ an Gewichten liest?

<!--en:98b6bc76-->
::: {.slide title="Verständnisfragen: Hardwarebedarf und Roofline-Grenzen (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:b3c730e5-->
1. **Standard-LoRA mit BF16 passt nicht:** Allein die eingefrorenen BF16-Gewichte belegen $8\text{B} \times 2\text{ B} \approx 14.9\text{ GiB}$. Hinzu kommen $1\text{ GiB}$ für den Adapterzustand $+ 8\text{ GiB}$ für Aktivierungen und Arbeitspuffer, insgesamt $\approx 23.9\text{ GiB} > 16\text{ GB}$. **QLoRA passt problemlos:** 4-Bit-Basisgewichte und Adapter belegen zusammen $\approx 3.9\text{ GiB} + 0.9\text{ GiB}$. Mit Aktivierungen und Arbeitspuffern ergibt sich $+ 8\text{ GiB} \approx 12.8\text{ GiB} < 16\text{ GB}$.

<!--en:abdc26da-->
2. **Dichtes Modell mit 30 Mrd. Parametern:** $15\text{ GB/tok}$ gelesen $\implies$ $614 / 15 \approx \mathbf{41\text{ tok/s}}$. **30B-A3B-MoE-Modell:** $1.5\text{ GB/tok}$ gelesen $\implies$ $614 / 1.5 \approx \mathbf{409\text{ tok/s}}$ ($10\times$ so schnell).

<!--en:9896874a-->
3. **Kapazität und Bandbreite:** Alle $30\text{B}$ Parameter ($\approx 15\text{ GB}$, mit KV-Cache und Laufzeitpuffern $> 16\text{ GB}$) müssen im Speicher bleiben, weil für das nächste Token jeder Experte zum Einsatz kommen kann.

<!--en:798707f1-->
::: {.slide title="Zusammenfassung: Hardwareauswahl nach Arbeitslast"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:fcfb914c-->
| Arbeitslast | Maßgebliche physikalische Grenze | Empfohlene Hardwareklasse (Nov. 2026) |

<!--en:84b18e79-->
| **D2L-Notebooks und QLoRA mit 7–8 Mrd. Parametern** | $\ge 16\text{ GB}$ VRAM und CUDA-Software | **RTX 5070 Ti (16 GB)** oder **gebrauchte RTX 3090 (24 GB)** |

<!--en:debe0365-->
| **Training auf dem Desktop und QLoRA mit 30 Mrd. Parametern** | $32\text{ GB}$ VRAM und hohe GDDR7-Bandbreite ($1.79\text{ TB/s}$) | **Workstation mit RTX 5090** (Netzteil mit $1000\text{ W}$, 64–128 GB RAM) |

<!--en:13b6df5b-->
| **Lokale MoE-Inferenz mit 70–120 Mrd. Parametern (Chat)** | $64\text{--}128\text{ GB}$ Speicher bei niedrigem Kaufpreis | **AMD Strix Halo (128 GB)** oder **Mac Studio M5 Max** |

<!--en:2d99ba16-->
| **Lokale MoE-Modelle mit 70–120 Mrd. Parametern und Agenten mit langen Prompts** | $128\text{ GB+}$ Speicher und schneller Prefill bzw. höhere Bandbreite | **NVIDIA DGX Spark (128 GB)** oder **Mac Studio M5 Ultra (bis zu 512 GB)** |

<!--en:e5621b75-->
| **Lokales LoRA oder unquantisierte Modelle mit 70 Mrd. Parametern** | $96\text{--}384\text{ GB}$ bei voller Bandbreite von $1.79\text{ TB/s}$ | **1–4$\times$ RTX PRO 6000 (96 GB)** (zuerst mit den Kosten gemieteter H100-Stunden vergleichen) |

<!--en:2d28f0ee-->
| **Tensorparalleles Training auf mehreren GPUs** | Schnelle Verbindung zwischen den GPUs ($450+\text{ GB/s}$) | **Cloud-Knoten mit NVLink mieten** (Karten für den Verbrauchermarkt und Workstations nutzen PCIe mit $64\text{ GB/s}$) |
