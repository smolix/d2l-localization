<!--en:6e00f32f-->
# Colab und Kaggle

<!--en:eab65bb5-->
Im vorigen Abschnitt haben wir die Notebooks zum Buch auf dem eigenen Rechner eingerichtet. Vielen fehlt dafür ein geeignetes Gerät. Manche haben nur einen Laptop ohne GPU, einen Schulrechner, auf dem sie nichts installieren dürfen, oder ein Tablet. Für sie gibt es [Google Colab](https://colab.research.google.com/) und [Kaggle](https://www.kaggle.com/code). Bei beiden lässt sich ein Notebook im Browser öffnen. Es läuft auf einer virtuellen Maschine, die der Anbieter für die Dauer der Sitzung bereitstellt. Python und die wichtigsten Frameworks sind vorinstalliert. Im Rahmen der Nutzungsgrenzen steht auch eine kostenlose GPU zur Verfügung, sofern eine frei ist.

<!--en:cd3448aa-->
Bei einem geliehenen Rechner stellt sich eine Frage, die sich beim eigenen Rechner erübrigt. Was ist morgen noch da? Angenommen, Sie installieren ein Paket, laden Fashion-MNIST herunter und trainieren eine Stunde lang einen Klassifikator. Der Anbieter hat die Maschine beim Verbindungsaufbau erstellt und löscht sie, sobald die Sitzung nach längerer Inaktivität oder beim Erreichen einer Nutzungsgrenze endet. Am nächsten Morgen finden Sie das Notebook und alle Dateien wieder, die Sie auf einem dauerhaften Speicher gesichert haben. Das Paket, der Datensatz und die lokal gespeicherten Checkpoints sind verschwunden.

<!--en:df108e15-->
Die Lösung besteht darin, die Laufzeitumgebung (engl. runtime) als austauschbar zu behandeln. Ein Notebook hält dazu fest, wie sich seine Umgebung wiederherstellen lässt, und speichert alles, was erhalten bleiben muss. So läuft es auf Colab und Kaggle ebenso wie mit einem lokalen Kernel, selbst wenn der Anbieter sein Softwareabbild aktualisiert. Im Folgenden sehen wir uns an, wie die Startschaltflächen des Buchs zu den beiden Anbietern führen und worin sich deren Angebote unterscheiden. Anschließend schreiben wir Zellen, die die Umgebung auch auf einer neuen Maschine einrichten.

<!--en:5f7e7c77-->
![Eine gehostete Laufzeitumgebung besteht nur vorübergehend. Notebooks, Daten und Checkpoints bleiben nur erhalten, wenn sie ausdrücklich dauerhaft gespeichert werden.](../img/tools-hosted-lifecycle.svg)

<!--en:70177e12-->
:numref:`fig_tools_hosted_lifecycle` zeigt diesen Lebenszyklus. Eine Laufzeitumgebung kann nach längerer Inaktivität anhalten, eine Sitzungsgrenze erreichen oder beim nächsten Start ein anderes Softwareabbild verwenden. Eine Einrichtungszelle (engl. setup cell) erneut auszuführen kostet wenig, wenn das Notebook alles enthält, was zum Wiederherstellen der Umgebung nötig ist. Die Notebooks zum Buch sind so aufgebaut.

<!--en:59e06a93-->
## Start eines Notebooks zum Buch

<!--en:d359e4ab-->
Jede Buchseite mit Code bietet im Menü **Notebook ausführen** drei Aktionen an: **Colab**, **Kaggle** und **Herunterladen**. Das Menü richtet sich nach dem aktiven Framework-Reiter und bietet das zugehörige Notebook für PyTorch, TensorFlow oder JAX an. Gibt es für den Reiter kein gehostetes Notebook, etwa bei MXNet, erscheint *Notebook nicht verfügbar*. Seiten, deren Code ohne Framework auskommt, bieten ein einziges NumPy-Notebook an. Die Notebooks liegen unter festen Pfaden in einem öffentlichen GitHub-Branch. Jedes Notebook nennt den Quellstand, aus dem es erzeugt wurde. Importiert ein Notebook `d2l`, beginnt es mit einer automatisch erzeugten Einrichtungszelle. Diese behält die vorinstallierten Pakete des Anbieters bei, soweit sie kompatibel sind. Andernfalls installiert sie die Versionen, mit denen das Buch getestet wurde. Die `d2l`-Hilfsmodule lädt sie für genau diesen Quellstand von GitHub herunter.

<!--en:f4d0ecee-->
![Von der Quellseite über den automatisch erzeugten Notebook-Branch bis zum Start beim Anbieter gibt es getrennte Schritte. Die Zuordnung von Buchseite zu Notebook bleibt dabei fest.](../img/tools-notebook-pipeline.svg)

<!--en:4214432d-->
Wo Ihre Arbeit gespeichert wird, hängt davon ab, mit welcher der beiden Aktionen Sie das Notebook starten:

<!--en:4dd002f9-->
* **Colab öffnet einen Verweis auf GitHub.** Die URL hat die Form `colab.research.google.com/github/<org>/<repo>/blob/<branch>/<path>.ipynb`. Bei jedem Öffnen ruft Colab das Notebook von GitHub ab. Sie sehen stets die aktuell veröffentlichte Version. Ihre Änderungen schreibt Colab jedoch nicht zurück. Um sie zu behalten, wählen Sie **Datei → Kopie in Drive speichern** oder laden Sie die `.ipynb`-Datei herunter.

<!--en:6be96a56-->
* **Kaggle importiert eine Kopie.** Das Buch übergibt die URL der Notebook-Rohdatei an die Importfunktion von Kaggle (`kaggle.com/kernels/welcome?src=...`). Diese erstellt ein neues privates Notebook in Ihrem Kaggle-Konto. Sie brauchen dafür also ein Konto. Ihre Kopie ist unabhängig vom Buch und übernimmt spätere Änderungen am Buch nicht automatisch. Ihre eigenen Änderungen behalten Sie, indem Sie auf Kaggle eine Version speichern.

<!--en:1f667ff8-->
In beiden Fällen verweist die Startschaltfläche auf ein öffentlich zugängliches Notebook, das Sie vor dem Ausführen lesen können. Gehen Sie bei jeder Schaltfläche „Open in Colab“ im Internet genauso vor. Prüfen Sie das Repository, aus dem das Notebook stammt, und lesen Sie die Einrichtungszelle, bevor Sie fremden Code mit den Berechtigungen Ihres Kontos ausführen.

<!--en:823b811b-->
## Colab

<!--en:7d6f27f1-->
### Ressourcen im kostenlosen Tarif

<!--en:6c892c0d-->
Colab bietet im kostenlosen Tarif GPUs und TPUs an. Laut den [häufig gestellten Fragen](https://research.google.com/colaboratory/faq.html) garantiert Colab weder bestimmte Hardware noch feste Nutzungsgrenzen. Dort heißt es mit Stand November 2026, dass sich die angebotenen Beschleunigertypen mit der Zeit ändern und die Nutzungsgrenzen schwanken. Je nach Verfügbarkeit und bisheriger Nutzung läuft ein Notebook höchstens 12 Stunden. Bei längerer Inaktivität endet die Sitzung. Im kostenlosen Tarif teilt Colab häufig eine T4 mit 16 GB Speicher zu, garantiert sie aber nicht.

<!--en:5cb02741-->
Das Menü zeigt, welche Hardware Sie angefordert haben. Welche Sie erhalten haben, prüfen wir deshalb im Code:

<!--en:e87e1244-->
Die Zelle erkennt Colab am Modul `google.colab`, das Colab beim Start lädt, und Kaggle am Arbeitsverzeichnis. Auskunft über den Beschleuniger geben bei PyTorch `torch.cuda.is_available()` und `torch.cuda.get_device_properties(0)`, bei TensorFlow `tf.config.list_physical_devices("GPU")` und bei JAX `jax.devices()`. Auch wenn ein Beschleuniger gemeldet wird, reicht sein Speicher möglicherweise für Ihre Berechnung nicht aus.

<!--en:c365252c-->
Über den kostenlosen Tarif hinaus verkauft Colab *Recheneinheiten* (engl. compute units), sowohl einzeln über Pay As You Go als auch über die Abonnements Pro und Pro+. Ein Guthaben verbessert den Zugang zu Beschleunigern, garantiert aber keinen bestimmten Typ. Preise und Verbrauchsraten ändern sich. Lesen Sie deshalb die Angaben auf der Kaufseite, bevor Sie die Kosten abschätzen.

<!--en:581c7d5d-->
### Versionen der Laufzeitumgebung und die Kommandozeile

<!--en:c21c85e7-->
Colab aktualisiert sein Softwareabbild regelmäßig. Dadurch kann ein Notebook ausfallen, das im Vormonat noch funktioniert hat. Seit September 2025 lässt sich für ein Notebook unter **Laufzeit → Laufzeittyp ändern** eine *Version der Laufzeitumgebung* mit einem bestimmten Datum festlegen. Mit Stand November 2026 bleiben frühere Versionen nach ihrer Veröffentlichung ein Jahr lang verfügbar. Der Verbindungsaufbau dauert bei ihnen zwei bis drei Minuten. Für die meisten Arbeiten empfiehlt Google die neueste Laufzeitumgebung mit fest vorgegebenen Bibliotheksversionen. Für Notebooks, die viele gemeinsam nutzen, etwa in einem Kurs, empfiehlt Google eine feste Version der Laufzeitumgebung.

<!--en:6dc18545-->
Mit der [Colab CLI](https://github.com/googlecolab/google-colab-cli), die im Juni 2026 für Linux und macOS erschien, lassen sich dieselben Laufzeitumgebungen vom Terminal aus oder über einen Programmieragenten steuern. `colab new` fordert eine Laufzeitumgebung an, `colab exec` führt darauf ein Skript oder Notebook aus und `colab stop` gibt sie wieder frei.

<!--en:aa40fa12-->
### Speichern der Arbeit

<!--en:6d02b4cc-->
Colab speichert Ihre Arbeit in Google Drive. Mit **Kopie in Drive speichern** sichern Sie das Notebook selbst. Datensätze und Checkpoints behalten Sie üblicherweise über mehrere Sitzungen hinweg, indem Sie Drive einbinden (`drive.mount`). Ein von GitHub geöffnetes Notebook schreibt Colab nie automatisch dorthin zurück. Colab kann sich auch mit einer *lokalen Laufzeitumgebung* verbinden. Dann steuert die Browserseite einen Jupyter-Server auf Ihrem eigenen Rechner. Damit verschiebt sich die Sicherheitsgrenze. Das Notebook läuft mit den Berechtigungen Ihres lokalen Jupyter-Prozesses, und Drive lässt sich nicht mehr einbinden. Verbinden Sie nur Notebooks, denen Sie vertrauen.

<!--en:5d7dc26f-->
## Kaggle

<!--en:6e2e675e-->
### Eingaben, Ausgaben und Versionen

<!--en:0b6ef719-->
Im Mittelpunkt von Kaggle stehen Datensätze und Wettbewerbe. Das prägt auch die Notebooks der Plattform. Wie ein Kaggle-Notebook mit Dateien umgeht, hängt von drei Arten von Verzeichnissen ab:

<!--en:8c9991fd-->
* Unter `/kaggle/input/` sind die hinzugefügten Datensätze, Modelle und Ausgaben anderer Notebooks **schreibgeschützt** und in einer bestimmten Version eingebunden. Einen so eingebundenen Datensatz muss das Notebook nicht herunterladen.

<!--en:9151f4c8-->
* `/kaggle/working/` ist Ihr beschreibbares Ausgabeverzeichnis. Mit Stand November 2026 bietet es 20 GB Speicherplatz. Sein Inhalt bleibt nur in *gespeicherten Versionen* des Notebooks erhalten.

<!--en:a9283fc2-->
* Alle anderen Verzeichnisse bieten temporären Speicher, dessen Inhalt mit der Sitzung verschwindet.

<!--en:e23621f2-->
Das zweite Kennzeichen ist die Versionierung. **Save & Run All** führt das gesamte Notebook in einer neuen Sitzung von oben nach unten aus. Kaggle speichert das Ergebnis zusammen mit den Eingaben, dem Code, den Ausgaben und den Protokollen als unveränderliche Version. Der bewährte Ablauf, den Kernel neu zu starten und alle Zellen auszuführen (:numref:`sec_interactive_development`), ist damit in die Plattform eingebaut.

<!--en:45cd8e7d-->
### Kontingente und Hardware

<!--en:2d2f0dd9-->
Mit Stand November 2026 stehen jedem Konto pro Woche 30 GPU-Stunden zur Verfügung, bei entsprechend geringer Nachfrage manchmal mehr. Sie lassen sich mit einer P100 oder zwei T4-GPUs nutzen. Für TPUs gibt es ein eigenes Wochenkontingent. Eine Sitzung läuft auf CPU oder GPU höchstens 12 Stunden und auf einer TPU höchstens 9 Stunden. Die aktuellen Werte für Ihr Konto finden Sie in der Notebook-Oberfläche. Prüfen Sie sie, bevor Sie einen Lauf planen. Kaggle kann eine Bestätigung Ihrer Telefonnummer verlangen, bevor es Beschleuniger oder den Internetzugang freischaltet.

<!--en:4a8e21c3-->
Der Internetzugang lässt sich für jedes Notebook einzeln einstellen und ist bei manchen Wettbewerben gesperrt. Ein Notebook sollte seine Daten deshalb als Eingaben einbinden, statt sie während der Ausführung herunterzuladen. Mit der [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) können Sie vom Terminal aus Notebooks und Datensätze hoch- und herunterladen sowie Notebooks starten. Bewahren Sie das zugehörige Token im Speicher für Zugangsdaten des Anbieters auf, niemals in einer Zelle.

<!--en:8dd89d9f-->
## Portable Notebooks

<!--en:f902eb4f-->
### Wahl des Anbieters

<!--en:07f573ad-->
Die Notebooks zum Buch laufen bei beiden Anbietern. Unterschiede gibt es beim Speicherort Ihrer Dateien und Ergebnisse und bei der Frage, wie gut sich die Nutzungsgrenzen vorhersagen lassen. Einen Überblick gibt :numref:`tab_hosted_notebook_tradeoffs`.

<!--en:fb2f00a4-->
:Notebook-Angebote im Vergleich (Stand November 2026)

<!--en:7bcb76ac-->
| Anforderung | Colab | Kaggle |

<!--en:311b5e65-->
| GitHub-Notebook direkt öffnen | Ja, Abruf bei jedem Öffnen | Eigene Kopie beim Import |

<!--en:2fad06ea-->
| Kostenlose GPU | Meist eine T4, Grenzen unveröffentlicht | P100 oder 2×T4, veröffentlichtes Kontingent von 30 h/Woche |

<!--en:64755f83-->
| Planbares Kontingent | Nein | Ja |

<!--en:22d2a67f-->
| Dauerhaft gespeicherte Dateien | Google Drive | Versionierte Datensätze und Ausgaben |

<!--en:c196088c-->
| Reproduzierbare Läufe | Manueller Neustart und Ausführung aller Zellen, feste Laufzeitversionen | Versionen aus Save & Run All |

<!--en:aef964cf-->
| Unterstützung für Wettbewerbe | Extern | In die Plattform integriert |

<!--en:3ba62188-->
| Kostenpflichtige Erweiterung | Recheneinheiten (Pay As You Go, Pro, Pro+) | Für das veröffentlichte Kontingent nicht nötig |

<!--en:a8882872-->
Wenn Sie den Code eines Buchabschnitts ausführen möchten, ist das direkte Öffnen von GitHub in Colab der kürzeste Weg. Für ein Experiment mit einem bestimmten Datensatz, das Sie teilen oder erneut ausführen möchten, eignen sich die versionierten Ein- und Ausgaben von Kaggle besser.

<!--en:eb270f59-->
### Einrichtungszellen

<!--en:368b59fb-->
Da die Laufzeitumgebung austauschbar ist, muss das Notebook seine Einrichtung selbst übernehmen. Eine gute Einrichtungszelle ist kurz, idempotent und gibt jeden Schritt ausdrücklich vor. Sie legt Versionen fest, installiert nur fehlende Pakete und enthält keine Zugangsdaten.

<!--en:15461d5b-->
Vermeiden Sie es, am Anfang eines Notebooks bedingungslos `pip install --upgrade ...` auszuführen. Damit verwerfen Sie die getestete Umgebung des Anbieters und verlangsamen jeden Start. Dasselbe Notebook kann dann heute bei der Auflösung der Abhängigkeiten andere Pakete auswählen als gestern. Installieren Sie gezielt das fehlende Paket in einer festgelegten Version. Auch bei unverändertem Notebook kann der Anbieter sein Softwareabbild austauschen. Gibt das Notebook einen kurzen Fingerabdruck der Umgebung (engl. environment fingerprint) aus, lassen sich Ergebnisse einordnen und Fehlerberichte besser auswerten:

<!--en:a4385794-->
### Pfade und Zugangsdaten

<!--en:537dadf4-->
Portable Notebooks verzichten auch auf fest einprogrammierte Pfade einzelner Anbieter. Ein kleiner Adapter wählt das Arbeitsverzeichnis, und `pathlib` übernimmt die übrigen Dateioperationen:

<!--en:c293e7fc-->
Beide Anbieter stellen einen Speicher für Zugangsdaten bereit, aus dem ein Notebook seine API-Token lesen sollte. Geben Sie ein Token niemals aus. Speichern Sie es weder in einer Ausgabe noch als Teil einer versionierten Kopie. Ein öffentliches Notebook ist eine ausführbare Veröffentlichung, zu der auch die Ausgaben gehören.

<!--en:37364118-->
Mit diesen Vorgehensweisen bleibt ein Notebook unabhängig von der Maschine, auf der es läuft. Die Sitzung dauert dadurch nicht länger, und die Verfügbarkeit eines Beschleunigers bleibt ungewiss. Wie Sie für lange Läufe und garantierte Hardware eine Maschine mieten, die Sie selbst steuern, zeigt :numref:`sec_cloud_instances`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:dbb4a686-->
Colab und Kaggle stellen kostenlos temporäre Laufzeitumgebungen für Notebooks mit begrenztem Zugang zu GPUs und TPUs bereit. Die Hardware ist nie garantiert. Prüfen Sie deshalb stets im Code, welche Geräte verfügbar sind. Wo Ihre Arbeit gespeichert wird, hängt von der Startschaltfläche im Buch ab. Colab öffnet die aktuelle Version von GitHub und schreibt Änderungen nicht zurück. Eine Kopie in Drive sichert Ihre Änderungen. Kaggle importiert eine unabhängige Momentaufnahme in Ihr Konto. Bei Colab schwanken die Grenzen des kostenlosen Tarifs mit der Nachfrage. Kostenpflichtige Recheneinheiten ermöglichen bevorzugten Zugang. Bei Bedarf lässt sich die Laufzeitversion festlegen und damit das Container-Abbild beibehalten. Kaggle veröffentlicht ein Wochenkontingent und speichert jeden Lauf mit Save & Run All als unveränderliche Version. Eine gehostete Maschine kann bei Inaktivität oder am Ende der zulässigen Sitzungsdauer jederzeit verschwinden. Idempotente Einrichtungszellen, gezieltes Checkpointing auf dauerhaftem Speicher und die Ablage von Zugangsdaten im dafür vorgesehenen Speicher des Anbieters halten ein Notebook auch über längere Zeit bei verschiedenen Anbietern lauffähig.

<!--en:2294243a-->
## Übungen

<!--en:7a49166d-->
1. **Prüfung einer Einrichtungszelle.** Über eine Schaltfläche „Open in Colab“ öffnen Sie ein Notebook, das mit einer vierzeiligen Einrichtungszelle beginnt. Diese führt zuerst `!pip install git+https://github.com/someone/helpers` aus, dann `!curl -fsSL https://example.org/setup.sh | bash`, danach `from google.colab import drive` und schließlich `drive.mount("/content/drive")`.
    1. Geben Sie für jede Zeile an, welchen Code sie ausführt oder welche Ihrer Daten sie für das Notebook lesbar macht. Welchen Teil dieses Codes können Sie prüfen, bevor die Zelle läuft?
    1. Schreiben Sie die ersten beiden Zeilen so um, dass der Code, den sie ausführen, genau dem Stand zum Zeitpunkt Ihrer Prüfung entspricht.
    1. Verbinden Sie dasselbe Notebook nun mit einer lokalen Laufzeitumgebung, sodass der Browser einen Jupyter-Server auf Ihrem eigenen Rechner steuert. Bei welcher Zeile ändern sich die Folgen am stärksten und warum?

<!--en:7a50909d-->
1. [code] **Neustart nach der Installation.** Führen Sie das folgende Experiment in einem gehosteten Notebook oder in einem Jupyter-Kernel einer neuen lokalen virtuellen Umgebung aus. Prüfen Sie zuerst, ob `"tabulate" in sys.modules` den Wert `False` hat. Installieren Sie Version 0.8.10 des kleinen Pakets `tabulate` und importieren Sie es. Installieren Sie in einer späteren Zelle derselben Kernel-Sitzung Version 0.9.0 ohne Neustart.
    1. Geben Sie nach der zweiten Installation `tabulate.__version__` und `importlib.metadata.version("tabulate")` aus. Warum unterscheiden sich die beiden Werte? Welcher beschreibt den Code, der ausgeführt wird?
    1. Die Zelle in diesem Abschnitt, die `missing` berechnet, verwendet `importlib.util.find_spec`. Warum erkennt diese Prüfung die gerade beobachtete Abweichung nicht?
    1. Schreiben Sie eine Funktion `ensure(dist, module, version)`, die `dist==version` nur dann installiert, wenn die installierte Distribution eine andere Version hat. Die Funktion soll zurückgeben, ob der Kernel neu starten muss, weil `module` bereits vor der Installation importiert wurde. Rufen Sie die Funktion in einer Einrichtungszelle zweimal hintereinander auf und bestätigen Sie, dass der zweite Aufruf nichts installiert.
   :exlabel:`ex_hosted-notebooks-restart-after-install`

<!--en:9b6d9414-->
1. [code] **Erweiterung des Fingerabdrucks.** Die Zelle, die `fingerprint` erstellt, erfasst die Python-Version, die NumPy-Version und die Rechnerarchitektur.
    1. Die Zelle soll auch die Version jedes installierten Deep-Learning-Frameworks erfassen, das dieses Buch verwendet. Warum ist es ungünstig, dazu in dieser Zelle jedes Framework zu importieren?
    1. Ergänzen Sie die Zelle um diese Framework-Versionen, den von `nvidia-smi` gemeldeten Namen und die Speichergröße jeder sichtbaren NVIDIA-GPU sowie die Umgebungsvariablen des Anbieters, deren Namen mit `COLAB_` oder `KAGGLE_` beginnen. Die Zelle muss auch in einer reinen CPU-Umgebung laufen und die Framework-Versionen aus den Paketmetadaten lesen. Erfassen Sie bei bereits importierten Frameworks zusätzlich `__version__`. Warum die beiden Werte abweichen können, zeigt :exref:`ex_hosted-notebooks-restart-after-install`. Führen Sie die Zelle bei einem der Anbieter aus. Welche der erfassten Variablen würden erkennen lassen, dass der Anbieter sein Softwareabbild geändert hat?
    1. Ein Notebook mit `pip install --upgrade transformers` in der ersten Zelle lief am Montag noch. Am Freitag scheitert es an einem Importfehler in `transformers`. Betrachten Sie zwei Fälle. Im ersten zeigt der Fingerabdruck am Freitag dasselbe Softwareabbild und dieselbe NumPy-Version wie am Montag. Die Version von `transformers` ist von 4.56.1 auf 4.57.0 gestiegen. Im zweiten Fall hat sich das Softwareabbild geändert, NumPy ist von 2.0.2 auf 2.3.1 gestiegen und `transformers` ist bei 4.56.1 geblieben. Welche Änderung hat den Fehler jeweils verursacht? Welche Änderung am Notebook verhindert in jedem der beiden Fälle, dass der Fehler erneut auftritt?
   :exlabel:`ex_hosted-notebooks-fingerprint-extension`

<!--en:4c633ee7-->
1. [code] **Pfade für dauerhaften und temporären Speicher.** Die Zelle, die `work` setzt, wählt auf Kaggle `/kaggle/working`, auf Colab `/content` und andernorts das aktuelle Verzeichnis.
    1. In welcher der drei Laufzeitumgebungen bleibt ein in `work` gespeicherter Checkpoint nach Sitzungsende erhalten und in welcher Form?
    1. Ersetzen Sie die Zelle durch eine Funktion `locations(root)`, die ein Wörterbuch mit drei Einträgen zurückgibt. Dabei bezeichnet `inputs` das schreibgeschützte Eingabeverzeichnis, `durable` ein Verzeichnis für Checkpoints, das über die Sitzung hinaus erhalten bleibt, und `scratch` Speicherplatz, der mit ihr verschwinden darf. Das Argument `root` steht für das Wurzelverzeichnis des Dateisystems. Ein gehostetes Notebook ruft `locations(Path("/"))` auf. Die Funktion prüft `root / "kaggle" / "working"` anstelle von `/kaggle/working`. Geben Sie für jeden Speicherort, den die Laufzeitumgebung nicht bietet, `None` zurück, etwa für das Eingabeverzeichnis auf Colab. Dort gibt es ein dauerhaftes Verzeichnis nur dann, wenn Google Drive unter `/content/drive` eingebunden ist. Gibt die Funktion sonst `None` zurück, kann der Code zum Schreiben der Checkpoints das Speichern in einem Verzeichnis ablehnen, das mit der Sitzung verschwindet.
    1. Testen Sie die Funktion auf Ihrem eigenen Rechner mit vier temporären Verzeichnissen als `root`. Das erste enthält `kaggle/input` und `kaggle/working`, das zweite nur `content`, das dritte `content/drive` und das vierte ist leer.
   :exlabel:`ex_hosted-notebooks-durable-paths`

<!--en:4a55d993-->
1. **Pre-Training mit kostenlosem Kontingent.** ● BERT-base hat rund 110 Millionen Parameter. :citet:`Devlin.Chang.Lee.ea.2018` trainierten es über 1 000 000 Schritte mit Batches von je 256 Sequenzen vor. Bei 90% der Schritte umfasste jede Sequenz 128 Token, bei den übrigen 512. NVIDIA gibt für die T4 eine Spitzenleistung von 65 TFLOP/s in FP16 an. Nehmen Sie an, dass das Training auf einer T4 dauerhaft 30% dieser Leistung erreicht.
    1. Das Training eines Modells mit $N$ Parametern auf $D$ Token kostet ungefähr $C \approx 6ND$ Gleitkommaoperationen. Auf jeden Parameter und jedes Token entfallen zwei Operationen im Vorwärtsdurchlauf und vier im Rückwärtsdurchlauf (:numref:`sec_scaling-laws`). Schätzen Sie so die Rechenkosten für das Training von BERT-base. Wie viele T4-Stunden sind nötig? Wie vielen Sitzungen zu je 12 Stunden und wie vielen Wochen mit einem Wochenkontingent von 30 Stunden entspricht das?
    1. Welche Kosten des Pre-Trainings von BERT fehlen in $6ND$, und welche setzt die Formel zu hoch an? Betrachten Sie die Attention-Scores bei beiden Sequenzlängen (:numref:`sec_scaling-laws` gibt die Korrektur an), den Kopf für das maskierte Sprachmodell, der nur an 15% der Positionen Vorhersagen trifft, und die Einbettungstabelle. Steigt oder sinkt die Schätzung durch die jeweilige Korrektur und ungefähr um wie viel?
    1. :citet:`hoffmann2022training` ermittelten eine rechenoptimale Aufteilung mit etwa zwanzig Trainingstoken pro Parameter. Welche Modellgröße und welche Tokenzahl wären bei dieser Aufteilung in einer einzigen T4-Sitzung von 12 Stunden möglich? Vergleichen Sie die Anzahl der Token pro Parameter bei BERT-base mit dieser Aufteilung.

<!--en:74ddeed9-->
1. [code] **Checkpoint-Intervall.** ●● Modellieren Sie Unterbrechungen einer gehosteten Sitzung, etwa durch Inaktivität oder weil der Anbieter die Maschine zurückfordert, als Poisson-Prozess mit mittlerer Zeit $M$ zwischen zwei Unterbrechungen. Beim Training wird nach jeweils $\tau$ Sekunden Rechenzeit ein Checkpoint geschrieben. Das Schreiben dauert $\delta$ Sekunden. Nach einer Unterbrechung dauern der erneute Verbindungsaufbau und das Wiederherstellen des jüngsten Checkpoints zusammen $R$ Sekunden. Die seit diesem Checkpoint geleistete Rechenarbeit geht verloren. Auch während der Wiederherstellung sind Unterbrechungen möglich.
    1. Nehmen Sie $\delta \ll \tau \ll M$ an. Zeigen Sie, dass der erwartete Anteil der Laufzeit, der auf das Schreiben von Checkpoints und auf verlorene Rechenarbeit entfällt, in erster Ordnung für folgendes Intervall minimal ist:

       $$\tau^* = \sqrt{2 \delta M},$$

       also für das Intervall von :citet:`Young.1974`.
    1. Ein Segment besteht aus $\tau$ Sekunden Rechenzeit und dem anschließenden Schreiben eines Checkpoints. Zeigen Sie, dass die erwartete Laufzeit für ein vollständiges Segment durch folgenden Ausdruck gegeben ist:

       $$\mathbb{E}[T] = M e^{R/M} \left(e^{(\tau + \delta)/M} - 1\right).$$

       Minimieren Sie $\mathbb{E}[T]/\tau$ numerisch für $\delta/M \in \{0.01, 0.05, 0.1, 0.25, 0.5\}$. Wie weit weicht $\tau^*$ vom exakten Optimum ab? Um wie viel übersteigt die erwartete Laufzeit bei $\tau^*$ das Minimum? Warum unterscheiden sich die beiden Abweichungen so stark in ihrer Größe?
    1. Schreiben Sie eine Monte-Carlo-Simulation des Prozesses und überprüfen Sie die Formel für $\mathbb{E}[T]$ für eine Kombination von $\delta/M$ und $R/M$.
    1. Für eine Kaggle-Sitzung gilt eine feste Grenze von 12 Stunden. Ihr Ende ist in diesem Fall kein zufälliger Zeitpunkt. Wie sollte sich der Zeitplan für Checkpoints ändern, wenn der Zeitpunkt einer Unterbrechung im Voraus feststeht? Welcher Teil der Analyse gilt weiterhin, wenn zusätzlich zufällige Verbindungsabbrüche durch Inaktivität auftreten?
   :exlabel:`ex_hosted-notebooks-checkpoint-interval`

<!--en:44e5ebc0-->
[Diskussion](https://d2l.discourse.group/t/36842)

<!--en:eb6f40db-->
Colab und Kaggle: temporäre Laufzeitumgebungen beim Anbieter, Kontingente und portable Einrichtung<br>**temporäre VMs · Notebook-Start über GitHub · Colab-Recheneinheiten · versionierte Ein- und Ausgaben bei Kaggle · idempotente Einrichtung · Youngs Formel für das Checkpoint-Intervall**

<!--en:272dbfa2-->
::: {.slide title="Geliehene Rechner und ihre kurzlebigen Laufzeitumgebungen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:7f71e984-->
- **GPUs im Browser ohne Installation:** Colab und Kaggle verleihen vorkonfigurierte Container mit kostenlosen GPUs. Damit lassen sich Notebooks auf jedem Laptop ausführen.

<!--en:463aa3fd-->
- **Verluste bei Sitzungsende:** Bei Inaktivität oder nach 12 Stunden **wird die VM gelöscht**. Dabei gehen der Inhalt des lokalen temporären Speichers (`/content`), installierte Pakete und der Inhalt des Kernel-Arbeitsspeichers verloren.

<!--en:74bbc5d0-->
- **Abhilfe:** Jede gehostete VM als austauschbar behandeln, die Umgebung in Zelle 1 wiederherstellen und Checkpoints direkt auf dauerhaften Speicher schreiben.

<!--en:78e86430-->
::: {.slide title="Lebenszyklus einer gehosteten Laufzeitumgebung"}

<!--en:8bcf8712-->
[Denkmodell]{.kicker}

<!--en:cb6e190b-->
Bei einem portablen Arbeitsablauf auf einer gehosteten Maschine bleiben der vorübergehende Ausführungszustand und der dauerhafte Speicher in jeder Phase der Sitzung getrennt:

<!--en:9c6649c7-->
- **1. Anfordern und wiederherstellen:** Eine neue VM startet mit dem aktuellen Container-Abbild des Anbieters. Eine idempotente Einrichtungszelle legt Paketversionen fest und installiert nur fehlende Pakete.

<!--en:69cf6d67-->
- **2. Lokalen Speicher für die Berechnung nutzen:** Die schnelle lokale SSD (`/content` oder ein anderer temporärer Speicherbereich) nimmt während des Trainings Zwischendateien auf.

<!--en:dff2ddb7-->
- **3. Gezielt dauerhaft sichern:** Checkpoints, Metriken und bearbeitete Notebooks mit dauerhaftem Speicher synchronisieren, *bevor* der Anbieter die VM wegen Inaktivität oder für eine andere Aufgabe zurückfordert.

<!--en:3b51e3e8-->
::: {.slide title="Wie D2L mit „Notebook ausführen“ ein Notebook startet"}

<!--en:b90f7152-->
[Infrastruktur des Buchs]{.kicker}

<!--en:490bbe0d-->
Auf jeder D2L-Seite mit Code bietet das Menü **Notebook ausführen** die Optionen **Colab**, **Kaggle** und **Herunterladen** für den aktiven Framework-Reiter:

<!--en:39a158ef-->
- **Automatisch erzeugter Branch `notebooks`:** Beim Erstellen des Buchs werden eigenständige `.ipynb`-Dateien veröffentlicht. Eine automatisch erzeugte Einrichtungszelle prüft die Framework-Versionen und lädt passende `d2l`-Hilfsmodule herunter.

<!--en:c21665aa-->
- **Colab öffnet einen Verweis auf die aktuelle GitHub-Version:** Bei jedem Öffnen lädt Colab das veröffentlichte Notebook neu. Änderungen schreibt es **nicht** zurück. Mit **Datei $\to$ Kopie in Drive speichern** bleiben sie in Drive erhalten.

<!--en:994e7dbf-->
- **Kaggle importiert eine unabhängige Kopie:** Im eigenen Kaggle-Konto entsteht eine private Momentaufnahme, die über die dort gespeicherten Versionen hinweg erhalten bleibt.

<!--en:fe065575-->
::: {.slide title="Sicherheitsprüfung vor dem Notebook-Start über eine Schaltfläche"}

<!--en:7d3bd5b8-->
[Sicherheit der Softwarelieferkette]{.kicker}

<!--en:123d0fc7-->
- **Einrichtungszellen vor dem Start prüfen:** Ohne festgelegten Quellstand kann `pip install git+https://...` oder `curl ... | bash` beliebigen Code aus dem Netz ausführen. `drive.mount` gewährt vollständigen Lese- und Schreibzugriff auf das eigene Google Drive.

<!--en:4c193d8e-->
- **Unveränderliche Commit-Hashes festlegen:** URLs, die auf veränderliche Branches verweisen, durch 40-stellige Git-Commit-Hashes ersetzen und Skripte vor der Ausführung prüfen.

<!--en:85a51d4b-->
- **Nur vertrauenswürdige Notebooks mit einer lokalen Laufzeitumgebung verbinden:** Lokal läuft der Code mit allen Rechten des eigenen Benutzerkontos auf dem eigenen Rechner.

<!--en:2282e6ec-->
::: {.slide title="Google Colab: variable Kontingente und Recheneinheiten"}

<!--en:497b9cc2-->
[Plattform Colab]{.kicker}

<!--en:1b1d4e1c-->
- **Kostenloser Tarif (Nov. 2026):** Meist eine NVIDIA **T4 (16 GB VRAM)** für höchstens 12 Stunden pro Sitzung. Bei Inaktivität trennt Colab die Verbindung rasch. Die Verfügbarkeit schwankt mit der Nachfrage.

<!--en:7c631544-->
- **Kostenpflichtige Tarife (Pay As You Go, Pro, Pro+):** **Recheneinheiten** ermöglichen bevorzugten Zugang zu L4- oder A100-GPUs und die Ausführung im Hintergrund. Die Zuteilung hängt von der Verfügbarkeit ab.

<!--en:12722816-->
::: {.slide title="Prüfung der zugeteilten Ressourcen im Code"}

<!--en:4d016801-->
[Prüfung der Laufzeitumgebung]{.kicker}

<!--en:23b64bd5-->
Das Notebook-Menü zeigt den *angeforderten* Beschleuniger. Welchen der Scheduler *zugeteilt* hat, deshalb stets im Code prüfen:

<!--en:8ee75cf1-->
- **Plattform erkennen:**
  - Colab lädt sein Modul beim Kernel-Start. Deshalb gilt dort `"google.colab" in sys.modules`.
  - Kaggle bindet `/kaggle/working` als beschreibbares Ausgabeverzeichnis ein.

<!--en:4d4652a4-->
- **Beschleuniger prüfen:** Mit `torch.cuda.get_device_properties(0)`, `tf.config.list_physical_devices("GPU")` oder `jax.devices()` vor dem Start einer Berechnung das Gerätemodell und den gesamten VRAM prüfen.

<!--en:f93a9a9b-->
::: {.slide title="Feste Colab-Laufzeitversionen und die Colab CLI"}

<!--en:58a833e1-->
[Stabile Abbilder und Automatisierung]{.kicker}

<!--en:ff00f8eb-->
- **Datierte Laufzeitversionen:** Ein datiertes Container-Abbild unter **Laufzeit $\to$ Laufzeittyp ändern** hält die Versionen von Python, CUDA und Bibliotheken ein Jahr lang fest. Das eignet sich besonders für Kursaufgaben.

<!--en:bb884fea-->
- **Automatisierung im Terminal (Colab CLI):** Mit `colab new`, `colab exec` und `colab stop` steuern lokale Skripte und Programmieragenten gehostete GPUs, auch ohne geöffneten Browser-Reiter.

<!--en:0d2dda41-->
::: {.slide title="Kaggle-Notebooks: Verzeichnisse und Versionierung"}

<!--en:975867cc-->
[Plattform Kaggle]{.kicker}

<!--en:3bbccae8-->
- **Drei Bereiche im Dateisystem:** Unter `/kaggle/input/` stehen Datensätze und Modelle sofort schreibgeschützt bereit. Der Inhalt von `/kaggle/working/` (20 GB) bleibt in gespeicherten Versionen erhalten. Alle anderen Pfade bieten nur temporären Speicher.

<!--en:5712e71b-->
- **Unveränderliche Versionen aus Batchläufen (`Save & Run All`):** Kaggle führt das ganze Notebook von oben nach unten in einem neuen Hintergrundcontainer aus und sichert Code, Protokolle und Ausgaben aus `/kaggle/working/` als Momentaufnahme.

<!--en:226fb1cd-->
::: {.slide title="Kaggle-Kontingente: Wie viel Pre-Training ist kostenlos möglich?"}

<!--en:8a03f687-->
[Abschätzung des Kontingentbedarfs]{.kicker}

<!--en:0179c457-->
- **Warum ein vollständiges Pre-Training den kostenlosen Tarif übersteigt:** Bei $30\%$ der T4-Spitzenleistung von $65\text{ TFLOP/s}$ ($19.5\text{ TFLOP/s}$) benötigt BERT-base $\sim 400\text{ T4 hours}$. Das sind mehr als **33 Sitzungen zu je zwölf Stunden** oder über 13 Wochen mit dem Kaggle-Kontingent von 30 h/Woche.

<!--en:787a777a-->
- **Was in eine 12-Stunden-Sitzung mit einer T4 passt ($8.4 \times 10^{17}\text{ FLOPs}$):** Alle D2L-Kapitelnotebooks, QLoRA-Fine-Tuning für Modelle mit 7–8 Milliarden Parametern oder das Training eines Modells mit $\sim 85\text{M}$ Parametern.

<!--en:7134f784-->
::: {.slide title="Colab und Kaggle im direkten Vergleich"}

<!--en:95e075b0-->
[Wahl des Anbieters]{.kicker}

<!--en:f334ce9a-->
- **Google Colab:** Öffnet die aktuelle Version von GitHub. Änderungen über **Kopie in Drive speichern** sichern. Dateien bleiben mit `drive.mount` erhalten. Für den Zugang zu L4 und A100 verkauft Colab Recheneinheiten.

<!--en:48fdfb5c-->
- **Kaggle Notebooks:** Importiert eine unabhängige Kopie, veröffentlicht ein Kontingent von **30 h/Woche** ($1\times\text{P100}$ oder $2\times\text{T4}$), bindet versionierte Datensätze unter `/kaggle/input` ein und speichert mit **Save & Run All** unveränderliche Versionen.

<!--en:b2cc6a12-->
::: {.slide title="Idempotente Einrichtungszellen"}

<!--en:47c90781-->
[Portable Notebooks]{.kicker}

<!--en:42f3e196-->
Da die gehostete VM austauschbar ist, muss das Notebook seine Umgebung selbst einrichten und dabei die vom Anbieter getestete Umgebung funktionsfähig halten:

<!--en:094d6163-->
- **Vor der Installation prüfen:** Mit `importlib.util.find_spec` (oder `importlib.metadata.version`) ermitteln, was fehlt, und nur diese Pakete in einer **festgelegten Version** installieren.

<!--en:c20bddd4-->
- **Nie bedingungslos `pip install --upgrade` ausführen:** Das überschreibt getestete Wheel-Pakete des Anbieters, verzögert den Start und führt dazu, dass am Freitag andere Abhängigkeiten aufgelöst werden als am Montag.

<!--en:3ad5aafa-->
- **Nach dem Aktualisieren eines importierten Pakets den Kernel neu starten:** Steht ein Modul bereits in `sys.modules`, aktualisiert `pip install` die Metadaten auf der Festplatte (`importlib.metadata.version`). Bis zum Neustart führt der Kernel weiter den alten Code im Arbeitsspeicher aus (`module.__version__`).

<!--en:9776b334-->
::: {.slide title="Erfassung eines Fingerabdrucks der Umgebung"}

<!--en:60ad5100-->
[Änderungen am Softwareabbild erkennen]{.kicker}

<!--en:5ee5af83-->
Versagt ein Notebook in einer gehosteten Laufzeitumgebung, obwohl es letzte Woche noch lief, hilft der protokollierte Fingerabdruck der Umgebung, Codefehler von Änderungen am Softwareabbild des Anbieters zu unterscheiden:

<!--en:cc3cd267-->
- **Angaben im Ausgabeprotokoll:**
  - Python-Version, Prozessor- bzw. Rechnerarchitektur und Versionen der wichtigsten Bibliotheken (`numpy`, Framework-Versionen aus den Metadaten).
  - Modell und VRAM der zugeteilten GPU (`nvidia-smi`) sowie Variablen aus dem Softwareabbild des Anbieters (`COLAB_*`, `KAGGLE_*`).

<!--en:c11ca6a7-->
- **Metadaten ohne Import abfragen:** Mit `importlib.metadata.version("torch")` die Version des installierten Frameworks erfassen. So entfallen mehrere Sekunden Wartezeit und Nebenwirkungen wie die Initialisierung der GPU beim Import ungenutzter Backends.

<!--en:fd24ea34-->
::: {.slide title="Portable Pfade, Zugangsdaten und optimales Checkpointing"}

<!--en:15560f06-->
[Dauerhafte Speicherung und Sicherheit]{.kicker}

<!--en:bae53399-->
- **Pfadwahl und Zugangsdaten:** Ausgaben automatisch nach `/kaggle/working`, `/content/drive` oder `Path.cwd()` schreiben. API-Token im Speicher für Zugangsdaten des Anbieters aufbewahren.

<!--en:cf7ab39f-->
- **Youngs Formel für das Checkpoint-Intervall ($\tau^* = \sqrt{2\delta M}$):** Dauert das Speichern $\delta = 30\text{ s}$ und bricht die Verbindung im Mittel alle $M = 3\text{ h}$ ab, alle $\tau^* \approx 13.4\text{ min}$ einen Checkpoint schreiben.

<!--en:3b6b8e04-->
::: {.slide title="Verständnisfragen: temporäre Laufzeitumgebungen und Portabilität"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:cc705585-->
Drei Situationen aus der Praxis mit gehosteten Notebooks:

<!--en:348705b4-->
1. Ein D2L-Notebook über die Startschaltfläche des Buchs in Colab öffnen, zwei Stunden lang die Übungen bearbeiten und den Browser-Reiter schließen. Wo sind die Änderungen?

<!--en:652606f7-->
2. Ein Colab-Notebook speichert alle 15 Minuten Modell-Checkpoints unter `/content/checkpoints/`, ohne Google Drive einzubinden. Die Sitzung erreicht ihre Grenze von 12 Stunden. Welche Checkpoints bleiben erhalten?

<!--en:e3fe0977-->
3. In einem Kernel lief bereits `import transformers`. Nun führt eine Zelle `!pip install transformers==4.57.0` aus. Warum gibt `transformers.__version__` weiterhin `4.56.1` aus?

<!--en:557b98b6-->
**Lösungen:**

<!--en:6d49edc6-->
1. **Nirgends. Sie sind verloren.** Colab öffnet über einen schreibgeschützten Verweis die Version auf GitHub. Um Änderungen zu behalten, vor Sitzungsende ausdrücklich **Datei $\to$ Kopie in Drive speichern** wählen oder die `.ipynb`-Datei herunterladen.

<!--en:7e4b77df-->
2. **Keine.** `/content` ist temporärer Speicher der VM und wird mit dem Container gelöscht. Vor einem langen Lauf sollte der Code zum Schreiben der Checkpoints prüfen, ob `/content/drive` eingebunden ist, oder die Daten mit einem externen Speicher synchronisieren.

<!--en:dea726bb-->
3. Python speichert importierte Module in `sys.modules` zwischen. Änderungen an Dateien in `site-packages` auf der Festplatte laden die Objekte im Arbeitsspeicher des Kernels nicht neu. Nach Paketänderungen ist ein Neustart des Kernels nötig.

<!--en:673b7436-->
::: {.slide title="Zusammenfassung: portable Notebooks auf gehosteten Maschinen"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:16df24f6-->
- **Änderungen sichern und Hardware prüfen:** Von GitHub in Colab geöffnete Notebooks sofort nach Drive kopieren und die Eigenschaften der zugeteilten GPU im Code prüfen.

<!--en:101fef21-->
- **Idempotente Einrichtung und Protokollierung von Änderungen:** Vor der Installation von Paketen in festgelegten Versionen mit `find_spec` prüfen, welche fehlen. Bei bereits importierten Paketen nach der Installation den Kernel neu starten. Den Fingerabdruck der Umgebung im JSON-Format protokollieren.

<!--en:bfe798b4-->
- **Dauerhafte Checkpoints und Zugangsdaten:** Checkpoints auf das eingebundene Drive-Laufwerk oder nach `/kaggle/working` schreiben und dabei das Intervall $\tau^* = \sqrt{2\delta M}$ einhalten. API-Schlüssel im Speicher für Zugangsdaten des Anbieters aufbewahren.
