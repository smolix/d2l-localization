<!--en:48166296-->
# Cloud-Computing

<!--en:f354a725-->
Die meisten Leserinnen und Leser führen die Notebooks dieses Buchs auf einem Laptop, auf einer Desktop-GPU oder bei einem Dienst wie Colab aus (:numref:`sec_hosted_notebooks`). Früher oder später stößt ein Projekt bei allen drei Möglichkeiten an Grenzen. Die Vorverarbeitung dauert länger als eine Sitzung beim Anbieter, ein Modell braucht 80 GB Beschleunigerspeicher oder ein Experiment erfordert mehrere GPUs gleichzeitig. Meist mieten wir dann einen Rechner stundenweise. Damit stellen sich zwei Fragen. Welchen Rechner sollen wir mieten, und wie viel kostet uns das fertige Ergebnis? Und wie arbeiten wir auf einem Rechner, der jederzeit verschwinden kann?

<!--en:e15a7879-->
Die Preise ändern sich monatlich. Die folgenden Zahlen sind daher Beispiele mit Stand zum angegebenen Zeitpunkt. Zwei Grundsätze haben länger Bestand. Für die Wahl des Rechners zählen die Kosten pro fertigem Ergebnis mehr als der Stundenpreis. Jede gemietete Instanz ist ersetzbar. Bevor wir sie löschen, kopieren wir alles Wichtige auf dauerhaften Speicher. Für die Notebooks dieses Buchs genügt bescheidene Hardware. Wir führen sie auf Karten vom Typ RTX 4090 mit 24 GB aus. Die meisten brauchen weniger als 8 GB. Auf einer gemieteten Karte mit 24 GB läuft also jedes Notebook für eine einzelne GPU. Nur die Abschnitte zu mehreren GPUs in :numref:`chap_performance` erfordern mehr als eine Karte.

<!--en:0b494136-->
## Der Mietmarkt

<!--en:039d70c2-->
### Drei Anbietergruppen

<!--en:30316478-->
Drei Arten von Unternehmen vermieten GPUs. Sie unterscheiden sich vor allem darin, was sie neben der Hardware anbieten und wer sie betreibt (:numref:`fig_tools_cloud_spectrum`).

<!--en:f33c04bb-->
![Die Cloud-Angebote unterscheiden sich darin, wie viel Integration und Kontrolle der Anbieter übernimmt, wie stark die Preise streuen und wie viel Verantwortung beim Nutzer liegt.](../img/tools-cloud-spectrum.svg)

<!--en:6c780629-->
* **Hyperscaler** wie AWS, Google Cloud und Microsoft Azure bieten GPUs als Teil einer umfassenden Cloud mit Identitätsverwaltung, virtuellen Netzwerken, verwaltetem Speicher und allen zugehörigen Diensten an. Ihre On-Demand-Preise für GPUs sind am höchsten. Für die größten Instanzen muss man häufig ein höheres Kontingent beantragen oder den Vertrieb kontaktieren. Diese Anbieter eignen sich für Projekte, deren Daten bereits in der jeweiligen Cloud liegen oder die regulatorische Vorgaben erfüllen müssen. Sie eignen sich auch für Training als Teil eines größeren verwalteten Systems. Google vermietet außerdem TPUs, auf denen der JAX-Code dieses Buchs läuft. Ein v5e-Chip kostet im November 2026 1.20 US-Dollar pro Stunde.

<!--en:0444c838-->
* **GPU-Spezialisten** wie Lambda, Nebius, Voltage Park und CoreWeave vermieten Beschleuniger mit schnellen Verbindungen untereinander und vorgefertigten Systemabbildern für maschinelles Lernen. Darüber hinaus bieten sie wenig an. Im November 2026 kostet eine H100 bei diesen Anbietern etwa 2–6 US-Dollar pro GPU-Stunde. Bei Lambda, Nebius und Voltage Park lässt sie sich mit einem neuen Konto ohne Kontakt zum Vertrieb starten. Diese Anbietergruppe eignet sich für umfangreiche Trainingsläufe, die keine weiteren Dienste eines Hyperscalers benötigen.

<!--en:821cbf86-->
* **Marktplätze** wie Vast.ai und RunPod bündeln Angebote vieler unabhängiger Betreiber. Dazu zählen GPUs für den Endkundenmarkt, die in den großen Clouds fehlen. Hier finden sich die niedrigsten Stundenpreise. Im November 2026 kostet eine RTX 4090 pro Stunde 0.30–0.50 US-Dollar, eine H100 gibt es ab etwa 1.80 US-Dollar. Die Qualität hängt vom einzelnen Angebot ab. Zuverlässigkeit des Rechners, Geschwindigkeit des Massenspeichers und Netzwerkqualität variieren, und der Code läuft auf einem fremden Rechner. Für Lehrveranstaltungen und öffentliche Daten ist das vertretbar. Für sensible Inhalte kommen nur geprüfte Rechenzentrumsangebote wie die Secure Cloud von RunPod infrage.

<!--en:e7ccf5d8-->
### Preise im November 2026

<!--en:2a678855-->
Die Preise hängen von Region, Verfügbarkeit, Vertragsbindung und Betreiber ab. :numref:`tab_cloud_prices` zeigt die Preisspanne zwischen den Anbietergruppen. Prüfen Sie vor der Buchung die aktuellen Angebote.

<!--en:6a1855b9-->
:On-Demand-Mietpreise pro GPU (November 2026)

<!--en:03395bfe-->
| Anbieter (Gruppe) | Günstige GPU | ≈ US-Dollar/h | 1× H100 80 GB ≈ US-Dollar/h |

<!--en:abed5e1b-->
| Vast.ai (Marktplatz) | RTX 4090 24 GB | 0.30–0.50 | 1.80–2.40 |

<!--en:55559b2c-->
| RunPod (Marktplatz) | RTX 4090 24 GB | 0.34–0.89 | 1.99–3.99 |

<!--en:a5ceef48-->
| Lambda (Spezialist) | A100 40 GB | 1.99 | 3.29–4.29 |

<!--en:59e0af8d-->
| Nebius (Spezialist) | RTX PRO 6000 96 GB | 1.80 | 4.50 |

<!--en:cd8e0177-->
| Voltage Park (Spezialist) | H100 80 GB (einziger angegebener Preis) | 1.99 | 1.99 |

<!--en:364ebbcb-->
| AWS (Hyperscaler) | L4 24 GB (g6.xlarge) | 0.80 | 6.88 (p5) |

<!--en:14e4eafc-->
| Azure (Hyperscaler) | T4 16 GB (NC4as T4 v3) | 0.53 | 6.98 (NC40ads H100 v5, 94 GB) |

<!--en:90ef12ac-->
| Google Cloud (Hyperscaler) | L4 24 GB (g2-standard-4) | 0.71 | ≈ 11 (A3 mit 8 GPUs) |

<!--en:4e94d595-->
Bei RunPod reichen die Preisspannen von Rechnern der Community bis zum Rechenzentrumsangebot. Die H100-Instanz von Azure mit einer einzelnen GPU enthält eine H100 NVL mit 94 GB. Google Cloud bietet die H100 auf Abruf nur in Rechnern mit 8 GPUs an. Der angegebene Preis gilt pro GPU.

<!--en:a6cce338-->
Zwei Einschränkungen sind wichtig. Erstens steigt der Preis eines Knotens mit mehreren GPUs annähernd linear mit ihrer Anzahl. Über die Verbindung zwischen den GPUs sagt das wenig aus. Ein Rechner mit 8×H100 und NVLink unterscheidet sich von acht PCIe-Karten, wie sich bei kommunikationsintensivem Training zeigt (:numref:`sec_training_systems`). Zweitens bedeutet neuere Hardware nicht immer niedrigere Kosten pro Arbeitseinheit. B200-Instanzen kosten im November 2026 etwa 7–14 US-Dollar pro GPU-Stunde. Sie lohnen sich nur, wenn ein Lauf ihren größeren Speicher und ihren FP8/FP4-Durchsatz nutzt.

<!--en:3b626a76-->
### Kosten pro fertigem Ergebnis

<!--en:af6755f3-->
Der Stundenpreis ist der sichtbarste Posten der Gesamtkosten. Hinzu kommen die Zeit für die Einrichtung, Leerlauf während der Fehlersuche, Speicherkosten auch nach dem Lauf, ausgehender Datenverkehr und die Arbeitszeit der Fachkräfte. Eine schnellere, teurere GPU kann daher die Kosten eines *abgeschlossenen* Experiments senken. Mit dem folgenden Modell untersuchen wir, welche von drei GPUs auf einem Marktplatz einen Auftrag am günstigsten erledigt. Auf der langsamsten dauert er acht Stunden. Führen Sie die Rechnung mit Ihren eigenen Annahmen erneut aus.

<!--en:d4185e1b-->
Die Rechnungen belaufen sich auf 5.40 US-Dollar, 8.60 US-Dollar und 9.92 US-Dollar. Nach der Rechnung allein ist also die preiswerte Karte am günstigsten. Mit einer halben Stunde Arbeitszeit steigen die Gesamtkosten auf 35.40 US-Dollar, 38.60 US-Dollar und 39.92 US-Dollar. Relativ gesehen liegen die Beträge nun viel näher beieinander, die Reihenfolge bleibt gleich. Erfordert die langsame Karte eine zusätzliche halbe Stunde Fehlersuche gegenüber der schnellen, steigen ihre Gesamtkosten auf über 65 US-Dollar und die Reihenfolge kehrt sich um. Die Anzahl solcher Durchgänge gehört daher ebenso ins Kostenmodell wie der Stundenpreis. Noch wichtiger ist der Umfang der Rechenaufgabe. Das LoRA-Fine-Tuning eines Modells mit 7 Milliarden Parametern kann auf einer einzelnen GPU für den Endkundenmarkt Stunden dauern. Das Pre-Training eines kleinen Sprachmodells kann Tausende GPU-Stunden beanspruchen. Schätzen Sie den Rechenaufwand ab, bevor Sie kleine Unterschiede im Stundenpreis vergleichen.

<!--en:89cb8cd2-->
Zwei Kostenfaktoren übersieht man leicht:

<!--en:870a1ee4-->
* **Spot-Kapazität und andere unterbrechbare Kapazität** sind oft um die Hälfte oder mehr vergünstigt. Sie eignen sich für Aufträge, die Checkpoints schreiben und sich zuverlässig fortsetzen lassen (:numref:`sec_training_systems`). Für interaktive Sitzungen oder Läufe ohne Checkpoints sind sie ungeeignet. AWS warnt zwei Minuten im Voraus. Google Cloud gewährt standardmäßig höchstens 30 Sekunden zum Herunterfahren. Bei Vast.ai stoppt eine unterbrechbare Instanz samt allen Prozessen, sobald ein höheres Gebot eingeht. Testen Sie die Wiederherstellung vor dem langen Lauf.

<!--en:4031bbf8-->
* **Gebühren für ausgehenden Datenverkehr.** Im November 2026 berechnen AWS und Azure für Datenübertragungen ins Internet nach den ersten 100 GB pro Monat etwa 0.09 US-Dollar pro GB. Wer einen Monat lang täglich einen Checkpoint von 140 GB herunterlädt, zahlt dafür etwa 370 US-Dollar. Das ist mehr als der Rund-um-die-Uhr-Betrieb einer RTX 4090 auf einem Marktplatz (720 Stunden zu 0.44 US-Dollar, also etwa 320 US-Dollar). Halten Sie Daten und Rechenressourcen in derselben Region. Wenn ein Arbeitsablauf große Artefakte überträgt, bevorzugen Sie Anbieter wie RunPod, die keine Gebühren für ausgehenden Datenverkehr erheben.

<!--en:a45e3f3e-->
## Arbeit auf einem gemieteten Rechner

<!--en:181ce5c9-->
### Einrichtung und Prüfung der Instanz

<!--en:444ebe16-->
Eine neue Instanz ist zunächst ein unbekannter Rechner. Die ersten Minuten entscheiden darüber, ob man sich bei einem langen Lauf auf ihn verlassen kann. Starten Sie mit dem aktuellen Systemabbild des Anbieters für Deep Learning. Es enthält bereits die Treiber, CUDA und die Container-Laufzeitumgebung. Bei einem reinen Betriebssystemabbild müssen Sie die Umgebung selbst einrichten und die Einrichtung später reproduzieren. Stellen Sie die Verbindung über den SSH-Tunnel aus :numref:`sec_interactive_development` her. Das funktioniert bei jedem Anbieter gleich. Alternativ können Sie VS Code Remote SSH verwenden. Führen Sie lange Aufträge unter `tmux` aus, damit das Training bei einem Abbruch der SSH-Verbindung weiterläuft. Prüfen Sie zuerst den Rechner:

<!--en:501c4493-->
Für einen kurzen Funktionstest schicken wir einen Batch durch das Modell, schreiben einen Checkpoint und lesen ihn wieder ein. So erkennen wir Probleme, die den Zustandsanzeigen des Anbieters entgehen: fehlerhafte Treiber, volle Laufwerke, schreibgeschützt eingebundene Dateisysteme und Datensätze, deren Übertragung zu langsam ist, um die GPU auszulasten.

<!--en:3be3f75a-->
### Ersetzbare Instanzen, dauerhafte Ergebnisse

<!--en:58d2e1d2-->
Der Anbieter kann eine gemietete Instanz unterbrechen, sie kann abstürzen oder versehentlich gelöscht werden. In allen drei Fällen muss die Arbeit erhalten bleiben.

<!--en:3c76535a-->
![Rechenressourcen bereitstellen, eine sichere Verbindung herstellen, Checkpoints auf dauerhaftem Speicher sichern und die VM löschen. Auf einem Ersatzrechner die Arbeit aus diesem Speicher fortsetzen.](../img/tools-cloud-lifecycle.svg)

<!--en:1d06494e-->
Der Ablauf in :numref:`fig_tools_cloud_lifecycle` trennt die Lebensdauer des Rechners von der Lebensdauer der Arbeit. Wir speichern den Code in Git und klonen das Repository auf die Instanz. Der Auftrag selbst steuert, wann er Daten und Checkpoints auf dauerhaften Speicher kopiert, etwa in einen Objektspeicher wie S3 oder GCS oder auf ein Volume des Anbieters. Beim Verlust der Instanz geht höchstens die Arbeit seit dem letzten Checkpoint verloren. Ein Ersatzrechner setzt den Lauf mit den gespeicherten Daten fort.

<!--en:a4f4c885-->
Löschen Sie nach dem Experiment alles, was Kosten verursacht: die Instanz, ihre Laufwerke und Snapshots, reservierte IP-Adressen und nicht mehr benötigte Buckets. Richten Sie am ersten Tag eine Kostenwarnung ein. Eine ungenutzte GPU kostet pro Stunde genauso viel wie eine ausgelastete. Warnungen treffen erst mit Verzögerung ein. Wenn Zugangsdaten in fremde Hände geraten, begrenzt daher das GPU-Kontingent des Kontos den finanziellen Schaden.

<!--en:ace9a186-->
Wir können einen Mietrechner nun nach den Kosten pro fertigem Ergebnis auswählen und unsere Arbeit vor einem Ausfall des Rechners schützen. Ist eine GPU an den meisten Tagen des Jahres ausgelastet, übersteigen die Mietkosten irgendwann den Kaufpreis. Welche Hardware sich dann zum Kauf eignet, untersucht :numref:`sec_hardware_buyers`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:822c7fbc-->
Die meisten Notebooks dieses Buchs benötigen weniger als 8 GB GPU-Speicher. Jedes Notebook für eine einzelne GPU läuft auf einer Karte mit 24 GB. Hohe Mietkosten fallen daher vor allem bei Rechenaufgaben in der Größenordnung von Sprachmodellen an. Auf dem Markt für Cloud-GPUs bieten Hyperscaler, GPU-Spezialisten und Marktplätze identische Chips zu Preisen an, die sich um ein Mehrfaches unterscheiden. Auch bei der Integration verwalteter Dienste, der Zuverlässigkeit und dem Vertrauen, das der Betrieb voraussetzt, gibt es große Unterschiede. Vergleichen Sie Rechner stets nach den Gesamtkosten pro fertigem Ergebnis. Neben dem Stundenpreis zählen Einrichtung, Fehlersuche, dauerhafter Speicher, ausgehender Datenverkehr und Arbeitszeit. Spot-Kapazität senkt die Rechenkosten für Aufträge, die sich von einem Checkpoint aus fortsetzen lassen. Für interaktive Arbeit oder Läufe ohne Checkpoints ist sie ungeeignet. Wer große Checkpoints wiederholt zwischen Clouds überträgt, zahlt rasch hohe Gebühren für ausgehenden Datenverkehr. Behandeln Sie jede gemietete Instanz als ersetzbar. Starten Sie mit einem vorgefertigten Systemabbild, verbinden Sie sich über einen SSH-Tunnel und arbeiten Sie unter `tmux`. Übertragen Sie Checkpoints fortlaufend auf dauerhaften Speicher und löschen Sie nach Abschluss des Laufs sofort alle kostenpflichtigen Ressourcen.

<!--en:2294243a-->
## Übungen

<!--en:a6b18485-->
1. [code] **Maximaler Speicherbedarf und Batchgröße.** Die meisten Notebooks dieses Buchs benötigen weniger als 8 GB GPU-Speicher. Eine Karte mit 24 GB aus :numref:`tab_cloud_prices` bietet daher Platz für einen größeren Batch. Führen Sie das Fine-Tuning des Encoders aus :numref:`sec_lm_bert` auf einer GPU aus, zum Beispiel in einem gehosteten Notebook (:numref:`sec_hosted_notebooks`). Verwenden Sie einmal die dort angegebene Batchgröße und einmal die Hälfte. Erfassen Sie für beide Läufe die Spitzenwerte aller drei in :numref:`fig_bg_allocator` dargestellten Größen für den GPU-Speicher.
    1. Welche der drei Größen entscheidet, ob der Speicher einer Karte für den Lauf ausreicht, und warum?
    1. Passen Sie eine lineare Funktion an, die den Spitzenwert dieser Größe in Abhängigkeit von der Batchgröße beschreibt. Sagen Sie die maximale Batchgröße für eine Karte mit 24 GB und für eine mit 80 GB vorher. Welchen Teil des Speicherbedarfs beschreibt der Achsenabschnitt?
    1. Testen Sie eine Batchgröße zwischen den beiden bisherigen Werten. Falls Ihnen eine Karte mit 24 GB zur Verfügung steht, können Sie stattdessen die Vorhersage für diese Karte testen. Vergleichen Sie den gemessenen Spitzenwert mit der linearen Vorhersage.
   :exlabel:`ex_cloud-instances-peak-memory`

<!--en:b16ce611-->
1. **Vertrauensgrenze.** Beim Fine-Tuning liest ein Trainingsauftrag die Trainingsdaten mit einem auf der Instanz hinterlegten Zugriffsschlüssel aus einem Objektspeicher. Die Checkpoints schreibt er auf das lokale Laufwerk der Instanz.
    1. Der Datensatz ist öffentlich und der Auftrag läuft auf einem über einen Marktplatz gemieteten Rechner. Was kann der Betreiber während des Laufs außer dem Datensatz lesen oder verändern? Was davon ist schützenswert, obwohl die Daten öffentlich sind? Warum schützt eine Verschlüsselung des Laufwerks der Instanz diese Inhalte nicht vor dem Betreiber?
    1. Nun handelt es sich bei den Daten um Patientenakten eines Krankenhauses, und der Auftrag zieht zu einem Hyperscaler um. Wer kann während des Laufs weiterhin die Daten und Checkpoints lesen? Welche Voraussetzungen müssen neben technischen Schutzmaßnahmen erfüllt sein, bevor die Akten hochgeladen werden dürfen?

<!--en:95619864-->
1. [code] **Arbeitszeit im Kostenmodell.** In `#cloud-instances-cost-model` addiert die Spalte `complete` den Betrag `setup_hours * engineer_per_hour` zur Rechnung.
    1. Erklären Sie, warum dieser Term unabhängig vom Wert von `engineer_per_hour` keinen Einfluss darauf hat, welche GPU am günstigsten ist, solange `setup_hours` für alle drei GPUs gleich ist.
    1. Wer auf Ergebnisse wartet oder den Fortschritt prüft, wendet dafür einen Anteil $f$ von `wall_hours` auf. Rechnen Sie diese Zeit zum Stundensatz `engineer_per_hour` in `complete` ein. Bestimmen Sie den Wert von $f$, bei dem die H100 und die RTX 4090 gleich viel kosten. Wie vielen Minuten Arbeitszeit während des 8-stündigen Laufs auf der RTX 4090 entspricht dieser Wert?
    1. Lassen Sie beide Terme für die Arbeitszeit weg. Welchen Wert für `relative_speed` müsste die H100 erreichen, damit allein die Rechnung so hoch ausfällt wie bei der RTX 4090?
    1. Ermitteln Sie auf einem der Marktplätze aus :numref:`tab_cloud_prices` die heutigen On-Demand-Preise für die drei GPUs. Notieren Sie das Datum und berechnen Sie erneut den Anteil $f$, bei dem die Kosten gleich sind. Ändert er sich um mehr als den Faktor zwei?
   :exlabel:`ex_cloud-instances-engineering-time`

<!--en:47408d3e-->
1. [code] **Skript für den Funktionstest.** Schreiben Sie ein Skript, dessen Prüfungen ein Mietrechner vor einem langen Lauf bestehen muss. Es trainiert ein kleines Modell Ihrer Wahl auf `d2l.FashionMNIST` und bricht beim ersten fehlgeschlagenen Test mit einer klaren Meldung ab.
    1. Prüfen Sie vier Bedingungen. Der auf der Kommandozeile angegebene Beschleuniger ist sichtbar. Im Checkpoint-Verzeichnis ist eine vorgegebene Anzahl von Bytes frei. Ein Trainingsschritt läuft auf dem Beschleuniger. Ein mit `d2l.save_checkpoint` geschriebener und mit `d2l.load_checkpoint` wieder eingelesener Checkpoint stellt jeden Parameter- und Optimierertensor exakt wieder her.
    1. Ergänzen Sie eine fünfte Prüfung. Vergleichen Sie die Anzahl der Batches, die der Dateniterator pro Sekunde liefert, mit der Anzahl der Trainingsschritte, die das Modell pro Sekunde schafft, wenn der Batch bereits auf dem Gerät liegt. Warum müssen Sie beide Raten getrennt messen?
    1. Lösen Sie jeden Fehler auf Ihrem eigenen Rechner aus. Fordern Sie mehr freien Speicherplatz an, als das Laufwerk bietet. Machen Sie das Checkpoint-Verzeichnis schreibgeschützt und bremsen Sie den Iterator mit einem Aufruf von `time.sleep` pro Batch. Führen Sie den Schreibschutztest ohne Root-Rechte aus, da der Root-Benutzer die Verzeichnisberechtigungen umgehen kann. Wann wäre jeder dieser Fehler ohne das Skript bei einem 8-stündigen Lauf aufgefallen, der stündlich einen Checkpoint schreibt? Welcher davon würde den Lauf nie anhalten?
   :exlabel:`ex_cloud-instances-smoke-test`

<!--en:c4992092-->
1. **NVLink oder PCIe.** ● Zwei Anbieter vermieten acht H100 zum selben Preis pro GPU-Stunde. Auf einem Knoten verbindet NVLink die GPUs, auf dem anderen nur PCIe 5.0. Ein Auftrag trainiert mit Datenparallelität einen Transformer mit $N$ Parametern und BF16-Gradienten. In die Berechnung der Kommunikationszeit gehen daher $S = 2N$ Bytes ein (:numref:`sec_multi_gpu`). Jede GPU verarbeitet pro Schritt $b$ Token. Das kostet $6Nb$ FLOPs (:numref:`sec_scaling-laws`). Verwenden Sie für die Busbandbreite $1/\beta_\textrm{link}$ die Nennrate je Richtung, 450 GB/s bei NVLink und 64 GB/s bei PCIe. Beide Knoten erreichen dauerhaft 40% der 989 TFLOPS, die die H100 für dichte BF16-Berechnungen bietet. Nehmen Sie an, dass sich Kommunikation und Berechnung zeitlich nicht überlappen.
    1. Berechnen Sie für $N = 1.3 \times 10^9$ die Schrittzeit auf jedem Knoten bei $b = 16{,}384$ und bei $b = 4{,}096$. Um welchen Anteil muss der PCIe-Preis sinken, damit die Kosten pro Token denen des NVLink-Knotens entsprechen?
    1. Das Verhältnis von Kommunikationszeit zu Rechenzeit hängt nicht von $N$ ab, da beide Zeiten proportional zu dieser Größe wachsen (:numref:`sec_hardware`). Bestimmen Sie für jeden Knoten die Anzahl der Token pro GPU und Schritt, bei der die Kommunikation genauso lange dauert wie die Berechnung.
    1. Bei der Gradientenakkumulation (:numref:`subsec_mp-accumulation`) tauschen die GPUs nur einmal alle $N_{\textrm{accumulation}}$ Mikrobatches Daten aus. Wie ändert sich dadurch der gefundene Schwellenwert, und was ändert sich am Training?

<!--en:5e51df06-->
1. **Neustarts auf Spot-Kapazität.** ● Ein Auftrag benötigt $T$ Stunden Rechenzeit. Bei Spot-Kapazität folgen die Unterbrechungen einem Poisson-Prozess. Die Zeit zwischen zwei Unterbrechungen hat den Mittelwert $M$. Nach jeder Unterbrechung dauert es $R$ Stunden, bis eine Ersatzinstanz gestartet ist. Der Auftrag beginnt dann wieder von vorn. Spot-Kapazität kostet einen Anteil $q$ des On-Demand-Preises. Auch die Zeit für den Neustart wird berechnet. Während eine Ersatzinstanz startet, treten keine Unterbrechungen auf. Die erwartete Laufzeit bis zum Abschluss beträgt

   $$\mathbb{E}[T_{\textrm{wall}}] = \left(e^{T/M} - 1\right)(M + R).$$

   Aufgabe :exref:`ex_hosted-notebooks-checkpoint-interval`b aus :numref:`sec_hosted_notebooks` leitet die entsprechende Formel für den Fall her, dass auch die Wiederherstellung unterbrochen werden kann. Dort wird aus dem Faktor $M + R$ der Faktor $M e^{R/M}$.
    1. Berechnen Sie für $T = 8$ h, $R = 0.25$ h und $q = 0.5$ die erwarteten Spot-Kosten als Anteil der On-Demand-Kosten für $M = 24$, $8$ und $4$ h.
    1. Es gelte $R \ll M$ und $q = 0.5$. Zeigen Sie, dass Spot-Kapazität im Erwartungswert genau dann günstiger ist, wenn $T$ unter einem festen Vielfachen von $M$ liegt. Berechnen Sie dieses Vielfache.
    1. Nun schreibt der Auftrag nach jeweils $\tau = \sqrt{2 \delta
       M}$ Stunden Rechenzeit einen Checkpoint. Das entspricht dem Intervall von :citet:`Young.1974` aus Aufgabe :exref:`ex_hosted-notebooks-checkpoint-interval`a. Jeder Schreibvorgang dauert $\delta$ Stunden. Bei einer Unterbrechung geht nur die Arbeit seit dem letzten Checkpoint verloren. Wenden Sie die Formel für $\mathbb{E}[T_{\textrm{wall}}]$ auf ein Segment der Länge $\tau + \delta$ an. Bestimmen Sie das größte $q$, bei dem Spot-Kapazität im Erwartungswert günstiger ist, für $\delta = 2$ min, $R = 0.25$ h und $M = 24$, $8$, $4$, $2$ und $1$ h.
   :exlabel:`ex_cloud-instances-spot-restart`

<!--en:6f2f5d3a-->
[Diskussion](https://d2l.discourse.group/t/36843)

<!--en:f2f65b97-->
Cloud-Computing: GPU-Anbieter, Kosten pro fertigem Ergebnis und Schutz vor Spot-Unterbrechungen<br>**Hyperscaler, Spezialisten und Marktplätze · Kosten pro Ergebnis · NVLink und PCIe · Neustarts auf Spot-Instanzen · Gebühren für ausgehenden Datenverkehr · ersetzbare VMs**

<!--en:de0a669e-->
::: {.slide title="Wenn lokale Rechner und gehostete Umgebungen an ihre Grenzen stoßen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:734e2594-->
- **Drei Gründe für die Cloud-Miete:** (1) Das Training überschreitet die Sitzungsdauer von 12 Stunden oder das Kontingent des Anbieters. (2) Vollständiges Fine-Tuning oder lange Kontexte erfordern $80\text{--}96\text{ GB}$ HBM. (3) Die Skalierung auf mehrere GPUs erfordert NVLink mit $450\text{--}900\text{ GB/s}$.

<!--en:e804ab1e-->
- **Zwei dauerhafte Grundsätze:** Instanzen nach den **Gesamtkosten pro fertigem Ergebnis** auswählen (nie allein nach dem Stundenpreis). Jede gemietete VM als **ersetzbar** behandeln.

<!--en:d723510d-->
::: {.slide title="Drei Anbietergruppen für Beschleuniger in der Cloud"}

<!--en:e2ad9d75-->
[Der Mietmarkt]{.kicker}

<!--en:dd8c7c00-->
GPU-Vermieter unterscheiden sich weniger in den Chips als in den verwalteten Diensten, der Preisspanne und der Frage, wem man beim Betrieb vertrauen muss:

<!--en:e8ac7f97-->
- **Hyperscaler (AWS, GCP, Azure):** Umfassende Cloud für Unternehmen (IAM, VPCs, Compliance, verwalteter Speicher, GCP-TPUs für $\sim\$1.20/\text{hr}$ bei v5e). Höchste On-Demand-Preise für GPUs, Kontingente genehmigungspflichtig

<!--en:b59ef5fb-->
- **GPU-Spezialisten (Lambda, Nebius, Voltage Park, CoreWeave):** Schnelle Verbindungen zwischen GPUs und vorgefertigte ML-Systemabbilder. Instanzen lassen sich selbst starten ($\sim\$2\text{--}6/\text{hr}$ pro H100).

<!--en:6c93e948-->
- **Marktplätze (Vast.ai, RunPod):** Bündeln Angebote von Rechenzentren und Community-Betreibern, darunter die RTX 4090 für den Endkundenmarkt ($\sim\$0.30\text{--}0.89/\text{hr}$) und die H100 ab $\sim\$1.80/\text{hr}$

<!--en:10c7c72f-->
::: {.slide title="Vertrauensgrenzen, Compliance und Isolation auf dem Host"}

<!--en:601ce421-->
[Sicherheitsarchitektur]{.kicker}

<!--en:7ecdfc55-->
- **Community-Rechner auf Marktplätzen:** Fremde Betreiber haben physischen Zugriff und Root-Rechte auf dem Hypervisor. Festplattenverschlüsselung schützt weder Schlüssel noch Gewichte, die zur Laufzeit im Klartext im RAM oder VRAM liegen müssen.

<!--en:a6d9c34a-->
- **Geprüfte Rechenzentren und Hyperscaler:** Für proprietäre Gewichte geprüfte Angebote und kurzlebige Token mit eng begrenzten Rechten nutzen. Vor dem Hochladen regulierter Daten (HIPAA) eine unterzeichnete Vereinbarung (**BAA**) sowie VPC- und KMS-Kontrollen verlangen.

<!--en:0373268b-->
::: {.slide title="On-Demand-Preise für GPUs (November 2026)"}

<!--en:bfe6f3da-->
[Preisspanne]{.kicker}

<!--en:fb0600b7-->
- **Preisunterschiede um den Faktor $3\times\text{--}5\times$ bei identischen Chips:** Eine H100 mit 80 GB kostet auf Marktplätzen $\$1.80\text{--}3.99/\text{hr}$, bei GPU-Spezialisten $\$1.99\text{--}4.50/\text{hr}$ und bei Hyperscalern $\$6.88\text{--}11.00/\text{hr}$.

<!--en:241a3eaf-->
- **Einstieg mit 24 GB:** Eine RTX 4090 auf einem Marktplatz ($\$0.30\text{--}0.89/\text{hr}$) oder eine L4 bei einem Hyperscaler ($\$0.71\text{--}0.80/\text{hr}$) genügt für Fine-Tuning und Auswertung auf einer GPU bei minimalen Mietkosten.

<!--en:e379b70b-->
::: {.slide title="Warum die Preise für GPU-Verbindungen nicht linear skalieren"}

<!--en:0090f05f-->
[Bandbreiten im Vergleich]{.kicker}

<!--en:62986f88-->
- **NVLink ($450\text{ GB/s}$) und PCIe 5.0 ($64\text{ GB/s}$) bei $8\times\text{H100}$:** Bei dauerhaft $395.6\text{ TFLOP/s}$ dauert die Kommunikation mit NVLink bei $b_{\text{balance}} \approx 513$ Token/GPU genauso lange wie die Berechnung, mit PCIe bei $3{,}606$ Token/GPU ($W = 8$).

<!--en:a0b439f4-->
- **Bei $b = 4{,}096$ Token/GPU:** NVLink verlängert die Schrittzeit nur um $\sim 13\%$, PCIe dagegen **um $88\%$**, sofern Gradientenakkumulation die Kommunikationskosten nicht auf mehrere Schritte verteilt.

<!--en:5101a0ef-->
::: {.slide title="Gesamtkosten pro fertigem Ergebnis und Stundenpreis"}

<!--en:1bd8ba3a-->
[Kosten eines Experiments]{.kicker}

<!--en:448750b8-->
- **Rechnung des Cloud-Anbieters:** Enthält die Laufzeit $T_{\text{wall}}$, die Stunden für die Einrichtung $T_{\text{setup}}$, die Kosten für dauerhaften Speicher $C_{\text{store}}$ (auch bei gestoppter Instanz) und für ausgehenden Datenverkehr $C_{\text{egress}}$

<!--en:99a068aa-->
- **Arbeitszeit ($p_{\text{eng}}$):** Einrichtung, Fehlersuche und der Anteil $f \cdot T_{\text{wall}}$ der Wartezeit, der Aufmerksamkeit erfordert, kosten oft mehr als die GPU-Miete.

<!--en:58f05932-->
::: {.slide title="Modellrechnung für Experimentkosten mit verschiedenen GPUs" layout="tight"}

<!--en:0553b36a-->
[Ausführbares Kostenmodell]{.kicker}

<!--en:0a3ed353-->
Vergleich dreier Marktplatzangebote für einen Auftrag, der auf der 4090 8 Stunden dauert: RTX 4090 ($\$0.40/\text{hr}$), A100 80 GB ($\$1.20/\text{hr}$) und H100 ($\$2.50/\text{hr}$)

<!--en:f724decc-->
- **Allein nach der Rechnung:** Die 4090 gewinnt ($\$5.40$ gegenüber $\$8.60$ und $\$9.92$). Ihr um den Faktor $6.25\times$ niedrigerer Stundenpreis wiegt den Geschwindigkeitsvorteil der H100 um den Faktor $3\times$ auf.

<!--en:ff31fbf0-->
- **Mit 0.5 h Einrichtung zu $\$60/\text{hr}$:** Die Gesamtkosten liegen mit **$\$35.40$, $\$38.60$ und $\$39.92$** relativ nah beieinander. Der feste Betrag erhöht alle drei Summen gleichermaßen. Der Abstand bleibt unter $\$5$.

<!--en:7b5ccff9-->
::: {.slide title="Modellrechnung für Experimentkosten mit verschiedenen GPUs (Code)" layout="tight"}

<!--en:0553b36a-2-->
[Ausführbares Kostenmodell]{.kicker}

<!--en:4cb1c5cb-->
::: {.slide title="Wann eine teurere GPU insgesamt günstiger ist"}

<!--en:304bac5c-->
[Kostenschwelle]{.kicker}

<!--en:7ed72f16-->
- **1. Weitere 30 Minuten Fehlersuche ($\$30+$):** Die Kosten für 8 Stunden auf der RTX 4090 steigen von $\$35.40$ auf über $\$65$. Der Lauf auf der H100 dauert 2.67 Stunden und ist mit $\$39.92$ deutlich günstiger.

<!--en:8d7bb770-->
- **2. Die Schwelle von 7 Minuten Arbeitszeit ($f \approx 1.4\%$):** Die H100 spart $5.33\text{ hours}$ Laufzeit für nur $\$4.52$ mehr auf der Rechnung. Bindet der 8-stündige Lauf während der Wartezeit auch nur **$6.8\text{ minutes}$** Arbeitszeit zu $\$60/\text{hr}$, ist die H100 günstiger.

<!--en:5e0c7e77-->
::: {.slide title="Kosten von Spot-Kapazität und anderen unterbrechbaren Angeboten"}

<!--en:125427aa-->
[Unterbrechbare Rechenkapazität]{.kicker}

<!--en:62cdd198-->
- **Exponentieller Anstieg ohne Checkpoints:** Beginnt ein Lauf von $8\text{ h}$ nach jeder Unterbrechung durch einen Poisson-Prozess von vorn ($M = 4\text{ h}$, $R = 0.25\text{ h}$), erreicht die erwartete Laufzeit $27.2\text{ h}$. Er kostet **$1.70\times$ so viel** wie ein On-Demand-Lauf, selbst bei einem Spot-Rabatt von $50\%$.

<!--en:99e13f47-->
- **Auf Spot-VMs nie Aufträge ohne Checkpoints oder interaktive Sitzungen ausführen**

<!--en:e23b326e-->
::: {.slide title="Wie Checkpointing Spot-Kapazität rentabel macht"}

<!--en:1616a06c-->
[Ausfallsichere Spot-Läufe]{.kicker}

<!--en:2821b1ff-->
Benötigt ein Trainingsskript zum dauerhaften Speichern eines Checkpoints die Zeit $\delta$ und speichert es alle $\tau^* = \sqrt{2\delta M}$ Stunden, geht bei einer Unterbrechung höchstens die Arbeit aus $\tau^*$ Stunden verloren:

<!--en:f9c3bf53-->
- **Erwartungswert der Laufzeit pro Segment:** Für ein Segment der Länge $\tau^* + \delta$ beträgt die erwartete Laufzeit
  $$\mathbb{E}[T_{\text{seg}}] = \left(e^{(\tau^* + \delta)/M} - 1\right)(M + R)$$

<!--en:b419726d-->
- Wegen $(\tau^* + \delta) / M \ll 1$ entfällt der exponentielle Anstieg ($e^x - 1 \approx x + x^2/2$).

<!--en:ee9f3aaf-->
- **Zahlenbeispiel ($T = 8\text{ h}$, $\delta = 2\text{ min}$, $R = 15\text{ min}$):**
  - Selbst bei Unterbrechungen im Abstand von durchschnittlich $M = 4\text{ hours}$ bleibt der Mehraufwand mit einem Checkpoint alle $\tau^* \approx 31\text{ minutes}$ so gering, dass Spot-Instanzen bei jedem Preisverhältnis $q \lesssim 0.80$ günstiger sind als On-Demand-Instanzen.

<!--en:3159c391-->
::: {.slide title="Versteckte Kosten durch ausgehenden Datenverkehr und ungenutzten Speicher"}

<!--en:ec6c7757-->
[Speicher- und Netzwerkkosten]{.kicker}

<!--en:e51e2639-->
- **Hohe Gebühren für ausgehenden Datenverkehr ($\sim\$0.09/\text{GB}$):** Wer einen Checkpoint von $140\text{ GB}$ an 30 Tagen je einmal herunterlädt ($4.2\text{ TB}$), zahlt **$\approx\$370$ allein für den Datenverkehr**. Das übersteigt die Kosten einer RTX 4090, die für den ganzen Monat 24/7 auf einem Marktplatz gemietet wird.

<!--en:6682e559-->
- **Drei Regeln gegen unnötige Kosten:** Rechenressourcen und Objektspeicher in derselben Cloud-Region betreiben, für Übertragungen zwischen Clouds Speicher ohne Gebühren für ausgehenden Datenverkehr nutzen (z. B. R2 oder RunPod) und nicht eingebundene NVMe-Volumes sofort löschen.

<!--en:95fc5039-->
::: {.slide title="Prüfung und Funktionstest in der ersten Minute"}

<!--en:7012bfd8-->
[Inbetriebnahme der Instanz]{.kicker}

<!--en:b73e2bc2-->
- **Start und Prüfung in der Shell:** In `tmux new -s train` beginnen, `nvidia-smi` und `df -h` ausführen. Vor langen Läufen prüfen, ob das Framework die GPU über CUDA erkennt.

<!--en:e6f83c1b-->
- **Automatischer Funktionstest in 60 Sekunden:** (1) Freien GPU-Speicher prüfen. (2) Schreibzugriff auf den Checkpoint-Speicher testen, um schreibgeschützte Laufwerke vor Ablauf von Stunde 1 zu erkennen. (3) 1 Schritt mit Vorwärts- und Rückwärtsdurchlauf ausführen, Checkpoint speichern und bitgenau wieder einlesen. (4) Prüfen: Batches/s des Datenladers $>$ Schritte/s der GPU.

<!--en:e96e2fdf-->
::: {.slide title="Ersetzbare Rechner, dauerhafte Artefakte"}

<!--en:9ff61899-->
[Lebenszyklus einer Instanz]{.kicker}

<!--en:d0c050b9-->
Die Lebensdauer der Forschungsarbeit von der Lebensdauer der einzelnen virtuellen Maschine trennen:

<!--en:bce3864c-->
- **Bereitstellen und klonen:** Von einem Standard-Systemabbild für Deep Learning starten, über einen SSH-Tunnel verbinden und das Repository mit `git clone` klonen

<!--en:2546d4b7-->
- **Checkpoints extern sichern:** Checkpoints und Protokolle regelmäßig in dauerhaften Objektspeicher (S3, GCS, R2) kopieren. Beim Verlust der VM geht höchstens die Arbeit eines Checkpoint-Intervalls verloren.

<!--en:c36f99b1-->
- **Aufräumen und Kosten begrenzen:** VM, temporäre Laufwerke, Snapshots und reservierte IP-Adressen sofort nach Abschluss löschen. Am ersten Tag eine **Kostenwarnung** einrichten und die **GPU-Kontingente des Kontos** eng begrenzen, damit mit einem entwendeten API-Schlüssel niemand über Nacht 100 GPUs starten kann.

<!--en:06f5b97a-->
::: {.slide title="Verständnisfragen zu Cloud-Kosten und Zuverlässigkeit"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:cd25949f-->
Drei Aufgaben zum Betrieb von Cloud-Systemen:

<!--en:73127adc-->
1. Beim Fine-Tuning von BERT mit verschiedenen Batchgrößen folgt der Spitzenwert des aktiv belegten Speichers der Funktion $M(B) = 1.8\text{ GiB} + 0.15\text{ GiB} \cdot B$. Der vom Speicherallokator mit Cache reservierte Speicher ist um $10\%$ größer. Wofür steht der Achsenabschnitt von $1.8\text{ GiB}$? Welche Speicherkennzahl entscheidet, ob einer Karte mit 24 GB während des Laufs der Speicher ausgeht?

<!--en:11def560-->
2. Wann fällt ohne vorherigen Funktionstest bei einem 8-stündigen Trainingslauf mit stündlicher Sicherung auf, dass das Checkpoint-Verzeichnis (`/mnt/ckpt`) schreibgeschützt ist? Wie verhält es sich mit einem langsamen Netzlaufwerk, das 2 Batches/s für eine GPU mit 10 Schritten/s liefert?

<!--en:bb8a30fa-->
3. Welchen Wert für `relative_speed` braucht die H100 ($\$2.50/\text{hr}$) in `#cloud-instances-cost-model`, damit ihre Cloud-Rechnung ohne Arbeitszeit genauso hoch ausfällt wie die der RTX 4090 ($\$0.40/\text{hr}$)?

<!--en:7babdaf6-->
::: {.slide title="Verständnisfragen zu Cloud-Kosten und Zuverlässigkeit (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:7b95968f-->
1. Der **Achsenabschnitt von $1.8\text{ GiB}$** beschreibt den von der Batchgröße unabhängigen Speicherbedarf (Parameter, Gradienten, Optimiererzustände und CUDA-Kontext). Ob der Speicher reicht, entscheidet **der vom Allokator reservierte bzw. maximal belegte Speicher im Verhältnis zum physischen VRAM**.

<!--en:45812d13-->
2. Der Lauf bricht wegen des schreibgeschützten Verzeichnisses beim ersten Speichern **nach 1 Stunde** ab. Damit geht eine Stunde Rechenzeit verloren. Der langsame Datenlader **meldet nie einen Fehler**. Er versorgt die GPU unbemerkt so langsam mit Daten, dass sie nur zu $20\%$ ausgelastet ist und $5\times$ so hohe Kosten anfallen.

<!--en:068a3a0f-->
3. Damit die Rechnung wie bei der 4090 auf $\$5.40$ kommt ($\$3.40$ für GPU-Rechenzeit nach Abzug von $\$2.00$ für Speicher und ausgehenden Datenverkehr), muss gelten: $(8.0 / s + 0.5) \times 2.50 = 3.40$ $\implies$ $8.0 / s = 0.86$ $\implies$ **`relative_speed` $\approx 9.3\times$**.

<!--en:7b4c9b7d-->
::: {.slide title="Zusammenfassung zum Einsatz von Beschleunigern in der Cloud"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:42251f8e-->
| Entscheidung | Naiver Ansatz | Vorgehen in der Praxis |

<!--en:ad38497f-->
| **Anbietergruppe** | Stets das Angebot mit dem niedrigsten Preis in $\$/\text{hr}$ wählen | Nach Vertrauensgrenze wählen: Marktplätze für öffentliche Daten, Spezialisten oder Hyperscaler für private oder regulierte Daten |

<!--en:5bfd6b42-->
| **Knoten mit mehreren GPUs** | $8\times\text{H100}$ mit PCIe zum selben Preis wie mit NVLink mieten | Bandbreite der Verbindung ($450\text{ GB/s}$ mit NVLink gegenüber $64\text{ GB/s}$ mit PCIe) anhand der Tokenzahl $b$ pro Batch und Schritt prüfen |

<!--en:15399a0b-->
| **Kostenvergleich** | GPU-Stundenpreis $p_{\text{gpu}}$ vergleichen | **Kosten pro fertigem Ergebnis** vergleichen, einschließlich Laufzeit, Einrichtung, Fehlersuche, Speicher, ausgehendem Datenverkehr und Arbeitszeit |

<!--en:2db76214-->
| **Spot-VMs und andere unterbrechbare VMs** | Lange Läufe oder interaktive Sitzungen ohne Checkpoints starten | Spot-Instanzen nur mit automatischem Checkpointing ($\tau^* = \sqrt{2\delta M}$) und getesteten Skripten zur Wiederherstellung nutzen |

<!--en:73eb592a-->
| **Übertragung von Daten und Checkpoints** | Checkpoints von 100 GB und mehr wiederholt aus anderen Clouds herunterladen | Rechenressourcen und Speicher in derselben Region betreiben oder Anbieter ohne Gebühren für ausgehenden Datenverkehr nutzen |

<!--en:ca8e9552-->
| **Lebenszyklus der Instanz** | Code und die einzigen Kopien der Checkpoints lokal auf der VM speichern | VMs als ersetzbar behandeln: Git für Code nutzen, beim Start `tmux` öffnen und einen Funktionstest ausführen, Daten mit dem Objektspeicher abgleichen, alle Ressourcen löschen |
