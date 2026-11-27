<!--en:cf0ae197-->
# Notebooks

<!--en:4b265849-->
In jedem bisherigen Abschnitt stand der Code neben dem erläuternden Text. Unter jeder Codezelle war ihre Ausgabe zu sehen. Vielleicht wollten Sie beim Lesen eine Zelle ändern und sehen, was passiert, wenn Sie eine kleinere Lernrate, eine breitere Schicht oder nur die Hälfte der Trainingsdaten verwenden. Genau dafür ist das Buch geschrieben. Jeder Abschnitt ist ein Jupyter-Notebook, also ein einzelnes Dokument mit Text, Mathematik, Code und Ergebnissen, das Sie Zelle für Zelle bearbeiten und ausführen können.

<!--en:222e5070-->
Wer Zellen einzeln ausführt, steht vor einer Frage, die sich bei einem Skript nie stellt: Welcher Code hat das Ergebnis auf dem Bildschirm erzeugt? Ein Notebook merkt sich alles, was Sie ausgeführt haben, und zwar in der Reihenfolge der Ausführung. Diese kann von der Reihenfolge im Dokument abweichen. Ein Notebook kann deshalb ein Ergebnis anzeigen, das nach einem Neustart niemand reproduzieren kann, auch der Autor selbst nicht. Starten Sie das Notebook daher neu und führen Sie es von oben aus, bevor Sie einem Ergebnis vertrauen oder es weitergeben.

<!--en:67c5ee3e-->
Im Folgenden erklären wir, wo ein Notebook seinen Zustand speichert und wie Sie ihn überprüfen. Wir richten die Notebooks zum Buch auf dem eigenen Rechner ein und arbeiten mit ihnen in einem Editor und auf einer entfernten GPU. Wenn Sie lieber nichts installieren möchten, zeigt :numref:`sec_hosted_notebooks`, wie Sie dieselben Notebooks kostenlos auf Colab oder Kaggle ausführen. Falls Sie mehr Rechenleistung benötigen, als Ihr Laptop bietet, siehe :numref:`sec_cloud_instances`.

<!--en:223df25a-->
## Das Ausführungsmodell eines Notebooks

<!--en:ca31af2b-->
Ein gewöhnliches Python-Skript läuft jedes Mal von Anfang an. Zeitaufwendige Schritte wie das Laden und Vorverarbeiten eines Datensatzes wiederholen sich daher bei jedem Experiment. In einem Notebook läuft zwischen den Ausführungen ein Python-Prozess weiter (der *Kernel*). Sie laden die Daten einmal und können dann das Modell beliebig oft überarbeiten, während die Daten im Speicher bleiben. Eingebettete Diagramme und Erläuterungen zwischen den Codezellen machen dieses Ausführungsmodell für Lehre und Forschung gleichermaßen nützlich.

<!--en:9b8fb374-->
Der Speicherinhalt überdauert auch den Code, der ihn erzeugt hat. Eine Variable behält ihren Wert, wenn Sie die Zelle mit der Zuweisung bearbeiten oder löschen. Wir müssen daher wissen, wo sich dieser *Zustand* (engl. state) befindet und wie lange er erhalten bleibt.

<!--en:753d0092-->
### Dokument und Kernel

<!--en:8389c217-->
Ein Notebook besteht aus zwei Teilen. Das *Dokument* speichert Zellen und ihre gesicherten Ausgaben. Dieses Dokument lesen Sie, geben es weiter und nehmen es in die Versionsverwaltung auf. Der *Kernel* hält den aktuellen Zustand: importierte Module, Variablen, Zufallszahlengeneratoren, offene Dateien, kompilierte Programme und belegten Beschleunigerspeicher. Im Dokument stehen die Zellen von oben nach unten. Der Kernel kennt nur die Reihenfolge, in der Sie diese Zellen *ausgeführt* haben. Wer die beiden Reihenfolgen verwechselt, kann damit viele Fehler in einem Notebook verursachen.

<!--en:3fa3a42f-->
![Im Dokument stehen die Zellen von oben nach unten. Der Zustand des Kernels ergibt sich aus der Ausführungsreihenfolge. Nach einem Neustart macht die Ausführung aller Zellen mögliche Abweichungen sichtbar: Sie folgt der Reihenfolge im Dokument, der einzigen, die andere beim Lesen nachvollziehen können.](../img/tools-kernel-state.svg)

<!--en:33eb86c8-->
:numref:`fig_tools_kernel_state` zeigt einen typischen Fehler. Beim Bearbeiten haben Sie zuerst eine Zelle ausgeführt, die Daten lädt. Dann haben Sie *darüber* eine Zelle eingefügt, die das Modell definiert. Alles funktionierte, weil `data` im Kernel bereits vorhanden war. Das Dokument beschreibt jetzt einen Ablauf, den es nie gegeben hat. Wer die Zellen in der dargestellten Reihenfolge ausführt, erhält einen `NameError`. So geht es auch Ihnen, wenn Sie morgen den Kernel neu starten.

<!--en:2810c168-->
Die Ausführungszähler (engl. execution counters) nummerieren die Zellenausführungen seit dem Start des Kernels, etwa `In [3]`, `In [1]` und so weiter. Sie geben Hinweise auf den bisherigen Ablauf, bilden aber keinen Abhängigkeitsgraphen. Die beiden folgenden Zellen zeigen eine unproblematische Abhängigkeit in der Reihenfolge des Dokuments: Die zweite Zelle funktioniert nur, weil die erste bereits lief.

<!--en:7e12c6e4-->
Diese Abhängigkeit ist unproblematisch, weil sie der Lesereihenfolge folgt. Problematisch ist dagegen eine Zelle, die eine Zuweisung *weiter unten* oder eine inzwischen gelöschte Zelle voraussetzt. Halten Sie Zellen klein, trennen Sie Funktionsdefinitionen von Experimenten und übergeben Sie Daten explizit, statt globale Variablen in vielen Zellen zu verändern.

<!--en:27209795-->
### Neustart und Ausführung aller Zellen

<!--en:2959c439-->
Der Befehl **Restart Kernel and Run All Cells** verwirft den angesammelten Zustand des Kernels und führt das Dokument in Lesereihenfolge von Grund auf neu aus. Dadurch fallen versteckter Zustand und Abhängigkeiten von der Ausführungsreihenfolge auf, bevor Sie das Notebook weitergeben. Führen Sie den Befehl aus, bevor Sie ein Notebook speichern, weitergeben oder in die Versionsverwaltung aufnehmen. Beim Erstellen des Buchs kommt dieselbe Prüfung zum Einsatz: Jedes Notebook läuft in einem frisch gestarteten Kernel von oben nach unten durch. Aus diesen Durchläufen stammen die Ausgaben auf der Website.

<!--en:7f44237f-->
## Die Notebooks zum Buch lokal ausführen

<!--en:70591487-->
### Einrichtung

<!--en:73dc16fc-->
Numerische Ergebnisse und sogar Programmierschnittstellen ändern sich von Paketversion zu Paketversion. Deshalb legt das Buch die Versionen für seine Umgebungen fest. Das Notebook-Archiv für das jeweilige Framework ist in :numref:`chap_installation` beschrieben. Es enthält die Notebooks, das Paket `d2l` und die `uv`-Lockdateien für eine CPU- und eine GPU-Umgebung. Erstellen Sie im entpackten Verzeichnis einmal die Umgebung und starten Sie darin JupyterLab:

<!--en:20a82c89-->
Unter Linux mit einer NVIDIA-GPU synchronisieren Sie die Umgebung stattdessen mit `pylock.gpu.toml`. Die Option `--no-sync` verhindert, dass uv die durch die Lockdatei festgelegte Umgebung verändert. In einem Klon des [GitHub-Repositorys](https://github.com/smolix/d2l-neu) erstellen Sie die Umgebungen hingegen mit `make`. Wie das geht, zeigt :numref:`sec_developers_guide`.

<!--en:6ffb4182-->
Der im Notebook ausgewählte *Kernel* muss zu der Umgebung gehören, in der die Pakete installiert sind. Ein Jupyter-Server kann mehrere Kernel zur Auswahl anbieten. Sein eigener Python-Prozess bestimmt nicht, welcher Kernel Ihr Notebook ausführt.

<!--en:ff792284-->
### Kurzprüfung der Umgebung

<!--en:1f8f1679-->
Halten Sie vor der ersten Codeänderung mit einer kurzen Prüfung fest, welchen Interpreter und welches Arbeitsverzeichnis Sie verwenden:

<!--en:57f1fb0a-->
Diese eine Zelle deckt zwei häufige Fehler bei der Einrichtung auf: Ein Kernel stammt aus der falschen Umgebung (`sys.executable` zeigt auf einen unerwarteten Pfad), oder das Notebook wurde in einem Verzeichnis geöffnet, von dem aus die relativen Datenpfade ins Leere führen. Die Zelle kürzt das Home-Verzeichnis mit `~` ab und gibt vom Arbeitsverzeichnis nur die letzte Pfadkomponente aus. So verrät die gespeicherte Ausgabe keinen persönlichen Pfad.

<!--en:ef26507e-->
## Arbeiten im Editor

<!--en:53f4fc5a-->
### JupyterLab

<!--en:479463d2-->
[JupyterLab](https://jupyterlab.readthedocs.io/) vereint Dateiverwaltung, Notebook-Editor, Terminals, Texteditor, Debugger-Unterstützung und eine Übersicht der laufenden Kernel. Auch wenn sich die Oberfläche weiterentwickelt, bleiben die grundlegenden Arbeitsschritte gleich:

<!--en:178326f0-->
* Führen Sie eine Zelle mit `Shift+Enter` aus. Verwenden Sie für `uv`, Git und das Prüfen von Dateien einen Terminal-Tab.

<!--en:f0cc71a4-->
* Wählen Sie den Kernel anhand des Umgebungsnamens und prüfen Sie dann `sys.executable`.

<!--en:231e866f-->
* Unterbrechen Sie eine lange Berechnung, bevor Sie den Kernel neu starten. Ein Neustart verwirft den Python-Zustand und gibt normalerweise auch den Beschleunigerspeicher frei, den dieser Prozess belegt.

<!--en:e9399101-->
* Führen Sie **Restart Kernel and Run All Cells** aus, bevor Sie ein Ergebnis für andere speichern.

<!--en:fe509628-->
* Prüfen Sie die Ansicht **Running** und beenden Sie Kernel, die Sie nicht mehr brauchen. Wenn Sie einen Browser-Tab schließen, läuft sein Kernel weiter. Ein verwaister Kernel kann mehrere Gigabyte GPU-Speicher belegen.

<!--en:6be69e8a-->
### VS Code

<!--en:b58c4d03-->
Mit [Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) lassen sich `.ipynb`-Dateien bearbeiten und ausführen. Der Editor bietet eine Kernel-Auswahl, eine Variablenansicht, einen Debugger für einzelne Zellen und einen auf Notebooks zugeschnittenen Versionsvergleich. Öffnen Sie das entpackte Verzeichnis als Ordner. Wählen Sie den Interpreter der oben erstellten Umgebung und legen Sie ihn als Notebook-Kernel fest.

<!--en:e23b263d-->
Wenn aus einem Notebook ein Projekt wird, können Sie in VS Code zu Definitionen springen, Tests ausführen, Code formatieren und Änderungen in Git prüfen. Legen Sie testbare Programmlogik in `.py`-Modulen ab und nutzen Sie das Notebook für Erläuterungen und Versuchsprotokolle. Dieses Repository enthält außerdem eine VS-Code-Erweiterung für die Arbeit am Buch. Damit können Sie zwischen Framework-Ansichten wechseln, Folien in der Vorschau ansehen und Änderungen in die Quelldateien zurückübertragen. Näheres dazu in :numref:`sec_developers_guide`.

<!--en:e6bcea0e-->
### Fehlersuche und Zeitmessung

<!--en:1b50ae66-->
Eine Ausnahme beendet die Ausführung der Zelle. Der Kernel läuft weiter und behält den Traceback und alle Variablen. Daher öffnet `%debug` in der nächsten Zelle einen Debugger genau in dem Aufrufrahmen, in dem die Ausnahme auftrat. Nutzen Sie den Notebook-Debugger, wenn der Kernel ihn unterstützt. Alternativ können Sie einen kleinen fehlschlagenden Aufruf in ein Modul verschieben und im Terminal untersuchen. Vermeiden Sie pauschale `try/except`-Blöcke, die nur „fehlgeschlagen“ ausgeben. Sie verwerfen den Typ und den Ort der Ausnahme, die Sie für die Diagnose brauchen. Für schnelle Messungen bietet IPython eingebaute Magic-Befehle:

<!--en:39fd9c98-->
Bei Zeitmessungen auf Beschleunigern ist Vorsicht geboten. GPU-Operationen laufen asynchron. Eine Zeitmessung auf dem Host erfasst daher möglicherweise nur, wie lange das Anstoßen der Arbeit dauert, und endet schon vor deren Abschluss. Synchronisieren Sie explizit oder verwenden Sie die Benchmark-Werkzeuge des Frameworks. Führen Sie kompilierte Kernel zuerst zum Aufwärmen aus und notieren Sie neben dem Messwert die Formen, die Rechengenauigkeit, das Gerät und die Bibliotheksversionen.

<!--en:ae85488c-->
## Arbeiten auf entfernten Rechnern

<!--en:6d505928-->
Editor, Notebook-Server, Kernel und Beschleuniger können sich auf verschiedenen Rechnern befinden. Häufig läuft JupyterLab auf dem Rechner mit der GPU, während Sie das Notebook auf einem Laptop bearbeiten.

<!--en:2e3a3795-->
![Ein lokaler Editor erreicht den entfernten Jupyter-Server und dessen Kernel über einen SSH-Tunnel. Der Beschleuniger bleibt mit dem entfernten Kernel verbunden.](../img/tools-remote-layers.svg)

<!--en:1502e546-->
Binden Sie den entfernten Server an die Loopback-Adresse und leiten Sie den Port über SSH weiter:

<!--en:522d12fb-->
Öffnen Sie dann lokal die URL `http://127.0.0.1:8888/...` mit dem zugehörigen Token. Binden Sie einen Jupyter-Server ohne Authentifizierung niemals an alle Netzwerkschnittstellen. Wer den Port erreichen kann, kann unter Ihrer Kennung Code ausführen. Der SSH-Tunnel sorgt für Verschlüsselung und Zugriffskontrolle. Das Jupyter-Token bietet zusätzlichen Schutz.

<!--en:3b6fb16e-->
Remote SSH von VS Code installiert auf dem Server eine kleine Komponente. Die Oberfläche des Editors läuft lokal, während sich Terminals, Erweiterungen, Dateien und Kernel auf dem entfernten Rechner befinden. Prüfen Sie vor der Auswahl der Python-Umgebung, ob in der Statusleiste **SSH** steht. Ein lokaler Kernel kann die entfernte GPU nicht nutzen. Rechnen Sie bei beiden Varianten damit, dass Ihnen der entfernte Rechner nur vorübergehend zur Verfügung steht. Nehmen Sie Ihren Code in die Versionsverwaltung auf und kopieren Sie Checkpoints auf einen dauerhaften Speicher, bevor der Rechner neu startet, Ihnen entzogen oder gelöscht wird (:numref:`sec_cloud_instances`).

<!--en:a6cf8eac-->
Prüfen Sie abschließend die Reproduzierbarkeit und den Datenschutz, bevor Sie ein Notebook weitergeben oder in die Versionsverwaltung aufnehmen. Starten Sie den Kernel neu und führen Sie alle Zellen aus. Prüfen Sie, ob die Einrichtung ohne persönliche Pfade oder undokumentierte Downloads auskommt. Entfernen Sie Zugangsdaten und private Daten aus den Ausgaben. Behalten Sie Ausgaben, die etwas erklären oder überprüfen, und löschen Sie überflüssige Fortschrittsprotokolle. Notieren Sie, welche Angaben zur Umgebung und Hardware man braucht, um die Messwerte einzuordnen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:9e9aa887-->
Notebooks verbinden Text, Code und Ergebnisse. Einmal aufwendig erzeugter Zustand bleibt über mehrere Experimente hinweg im Speicher. Es lohnt sich daher, beim Lesen den Code selbst zu verändern. Das gespeicherte Dokument und der laufende Kernel halten den Zustand in unterschiedlicher Reihenfolge fest. Ob beide übereinstimmen, prüfen Sie, indem Sie den Kernel neu starten und alle Zellen von oben nach unten ausführen. Für reproduzierbare lokale Durchläufe starten Sie JupyterLab in der Umgebung, die Sie anhand der `uv`-Lockdateien des Buchs synchronisiert haben. Prüfen Sie den aktiven Kernel mit `sys.executable`. JupyterLab vereint Notebooks, Terminals und Kernel-Verwaltung in einer Arbeitsumgebung im Browser. Wächst das Projekt, bietet VS Code zusätzlich die Navigation zwischen Modulen, Tests, Versionsvergleiche und die Entwicklung auf entfernten Rechnern. Nutzen Sie für die Verbindung zu einem entfernten Beschleuniger einen SSH-Tunnel oder VS Code Remote SSH. So müssen Sie keinen Notebook-Server ohne Authentifizierung im Netzwerk zugänglich machen.

<!--en:2294243a-->
## Übungen

<!--en:de57ff12-->
1. [code] **Ausführung in abweichender Reihenfolge.** Stellen Sie den Fehler aus :numref:`fig_tools_kernel_state` in einem neuen Notebook nach. Führen Sie eine Zelle aus, die `data` definiert. Fügen Sie darüber eine Zelle ein, die `data` verwendet, und führen Sie die neue Zelle aus. Wenn Sie jetzt den Kernel neu starten und alle Zellen ausführen, bricht die Ausführung mit einem `NameError` ab.
    1. Erstellen Sie zwei Zellen, bei denen die Ausführung in abweichender Reihenfolge einen Wert ändert, ohne einen Fehler zu melden. Wählen Sie zum Beispiel eine Lernrate, die eine weiter unten stehende Zelle neu setzt, bevor Sie die Trainingszelle erneut ausführen. Was meldet der Kernel bei diesem Notebook, wenn Sie ihn neu starten und alle Zellen ausführen? Welcher der beiden Fehler ist gefährlicher und warum?
    1. Die Ausführungszähler eines gespeicherten Notebooks lauten $1, 2, \ldots, n$, wenn man seine $n$ Codezellen in Dokumentreihenfolge liest. Welche Aussage lässt sich damit über die Entstehung der Ausgaben garantieren? :citet:`Pimentel.Murta.Braganholo.ea.2019` unterscheiden drei Ursachen für versteckten Zustand: eine erneut ausgeführte Zelle, eine nach der Ausführung bearbeitete Zelle und eine gelöschte Zelle. Welche davon sind bei einem solchen Notebook noch möglich? Berücksichtigen Sie, dass ein Neustart des Kernels den Zähler auf $1$ zurücksetzt.
    1. Ein anderes Notebook zeigt auf seinen drei Codezellen die Zähler $3, 8, 9$. Was verraten die Lücken über Ausführungen, die im Dokument nicht mehr zu sehen sind? Beweisen sie, dass die Ausgaben von verstecktem Zustand abhängen?
   :exlabel:`ex_interactive-development-out-of-order`

<!--en:14459144-->
1. **Grenzen eines Durchlaufs mit frischem Kernel.** Wenn Sie den Kernel neu starten und alle Zellen ausführen, läuft das Dokument in einem frischen Kernel, aber auf demselben Rechner. Betrachten Sie fünf Notebooks, die diese Prüfung auf dem Rechner ihres Autors bestehen: (i) Eine Zelle lädt `features.npy`, falls die Datei existiert. Andernfalls berechnet sie deren Inhalt. Der Berechnungszweig enthält einen Fehler, und `features.npy` fehlt im Repository. (ii) Ein Paket wurde aus einer inzwischen gelöschten Zelle heraus installiert. (iii) Eine Zelle liest `/home/alice/data/train.csv`. (iv) Eine Zelle liest eine Umgebungsvariable, die in der Shell gesetzt ist, aus der JupyterLab gestartet wurde. (v) Ein Absatz nennt eine Genauigkeit, die eine Zelle mit einem Zufallszahlengenerator ohne festen Startwert berechnet.
    1. Beschreiben Sie für jedes Notebook, was passiert, wenn jemand es herunterlädt und ausführt.
    1. Nehmen Sie nun an, das Notebook läuft auf einem frisch eingerichteten Rechner von oben nach unten durch. Es stammt aus einem neuen Klon des Repositorys, und die Umgebung wurde anhand der Lockdatei neu erstellt. Welche der fünf Fehler deckt das auf? Nennen Sie für die übrigen Fehler jeweils eine Prüfung, die sie aufdeckt.

<!--en:c9659f6a-->
1. [code] **Zugriffe auf noch nicht definierte Namen erkennen.** ● Wenn Sie den Kernel neu starten und alle Zellen ausführen, zeigt sich eine Abhängigkeit von einer späteren Zelle erst bei der Ausführung des Notebooks. Eine statische Prüfung liest das Dokument, ohne es auszuführen.
    1. Schreiben Sie eine Funktion, die eine `.ipynb`-Datei liest und jeden Namen mit folgenden Eigenschaften meldet: Eine Codezelle liest ihn, weder eine frühere Zelle noch Python selbst stellt eine Bindung dafür bereit, aber eine spätere Zelle bindet ihn. Wandeln Sie Magic-Befehle wie `%timeit` mit `IPython.core.inputtransformer2.TransformerManager` in Python-Code um, bevor Sie eine Zelle mit `ast` analysieren. Testen Sie die Funktion am Notebook aus Aufgabe :exref:`ex_interactive-development-out-of-order`.
    1. Der Rumpf einer Funktion läuft erst beim Aufruf, nicht schon bei der Definition. Erstellen Sie ein Notebook, bei dem Ihr Prüfprogramm deshalb einen Fehlalarm auslöst, und eines, bei dem es einen tatsächlichen Fehler übersieht. Korrigieren Sie das Prüfprogramm so, dass der Fehlalarm ausbleibt.
    1. Ergänzen Sie zwei Warnungen: Eine Zelle bindet einen Namen erneut, nachdem eine frühere Zelle ihn gelesen hat, oder eine Zelle liest und bindet denselben Namen (wie in `pixels = pixels / 255`). Welche Warnung zielt auf den Fehler aus Aufgabe :exref:`ex_interactive-development-out-of-order`a, welche auf den Fehler bei wiederholter Ausführung einer Zelle, die ihren eigenen Ausgangswert überschreibt? Kann eine der Warnungen harmlose von schädlichen Fällen unterscheiden?
   :exlabel:`ex_interactive-development-use-before-define`

<!--en:ca20f1a9-->
1. [code] **Arbeitsprozesse.** ● Das Python-Modul `multiprocessing` startet Arbeitsprozesse nach einem von drei Verfahren: `fork` kopiert den Elternprozess, `spawn` startet jeden Arbeitsprozess in einem neuen Interpreter, und `forkserver` erzeugt jeden Arbeitsprozess als Kopie eines eigens neu gestarteten Serverprozesses. Bei allen drei Verfahren sendet ein Pool jede Aufgabe einschließlich der aufzurufenden Funktion mit `pickle` an seine Arbeitsprozesse. Datenlader mit Arbeitsprozessen nutzen denselben Mechanismus.
    1. Definieren Sie in einer Notebook-Zelle eine Funktion `square(i)`, die `i * i` zurückgibt. Führen Sie in einer anderen Zelle `p.map(square, range(4))` innerhalb von `with multiprocessing.get_context(method).Pool(2) as p:` aus, und zwar für jedes Verfahren, das Ihr System anbietet. Falls die Ausführung einer Zelle nicht endet, unterbrechen Sie den Kernel. Was passiert bei den einzelnen Verfahren und wo erscheinen die Fehlermeldungen?
    1. Erklären Sie das Verhalten anhand der Informationen, die `pickle` für eine Funktion speichert, und des Moduls, in dem ein Notebook seine Funktionen definiert.
    1. Die Umgebungen zum Buch verwenden Python 3.12. Diese Version startet Arbeitsprozesse unter Linux mit `fork`, unter macOS und Windows mit `spawn`. Seit Python 3.14 ist unter Linux `forkserver` die Voreinstellung. Bauen Sie das Notebook so um, dass der Pool mit jedem Startverfahren funktioniert.

<!--en:44bf17bf-->
1. [code] **Zeitmessung auf asynchronen Beschleunigern.** ● Betrachten Sie das Produkt zweier $8192 \times 8192$-Matrizen in einem 16-Bit-Format auf einer GPU. Nutzen Sie bf16, wenn die GPU es unterstützt, und fp16 auf der T4 und der P100 in den kostenlosen Angeboten von Colab und Kaggle (zur Einrichtung siehe :numref:`sec_hosted_notebooks`). Diese beiden GPUs haben keine bf16-Einheiten. Falls Ihr Rechner keine GPU hat, verwenden Sie eine dieser kostenlosen GPUs.
    1. Berechnen Sie, wie lange die Multiplikation bei dem Durchsatz für dicht besetzte Matrizen dauert, den das Datenblatt Ihrer GPU für dieses Format angibt. Verwenden Sie für eine RTX 4090 den entsprechenden Eintrag in :numref:`tab_gpu_specs`.
    1. Messen Sie die Dauer eines Aufrufs mit `time.perf_counter()`, einmal mit und einmal ohne Gerätesynchronisierung vor dem Ende der Messung. Synchronisieren Sie in beiden Fällen vor Beginn der Messung und lassen Sie den ersten Aufruf unberücksichtigt. Welche Messung passt zu (a)? Welchen Durchsatz würde die andere ergeben?
    1. Messen Sie die Dauer des ersten Aufrufs gesondert und erklären Sie den Unterschied zu späteren Aufrufen.
    1. Messen Sie die Dauer der Multiplikation ohne Synchronisierung mit `%timeit`. Dieser Befehl führt die Multiplikation viele Male hintereinander aus. Wo liegt das Ergebnis im Vergleich zu den beiden Messungen aus (b) und warum?

<!--en:6dcd5249-->
1. **Ein gemeinsam genutzter GPU-Server.** ● Mehrere Benutzer eines entfernten Rechners starten jeweils JupyterLab mit `uv run --no-sync jupyter lab --no-browser --ip 127.0.0.1 --port 8888` und stellen von ihren Laptops aus mit `ssh -N -L 8888:127.0.0.1:8888 myserver` eine Verbindung her.
    1. Wer kann sich mit einem Server verbinden, der auf diesem Rechner an `127.0.0.1` lauscht? Was schützt Ihren Kernel vor diesen Benutzern? Finden Sie heraus, wo Ihr laufender Server sein Token auf der Festplatte speichert und welche Benutzer diese Datei lesen können.
    1. Ihr Server belegt Port 8888, als ein zweiter Benutzer denselben Befehl ausführt. Jupyter probiert dann standardmäßig bis zu 50 weitere Ports mit benachbarten Nummern aus. Wohin führt der Tunnel des zweiten Benutzers und was sieht er im Browser? Wie findet er den Port heraus, an dem sein eigener Server lauscht?
    1. Wenn Sie den Tunnel öffnen, lauscht auf Ihrem Laptop bereits ein von Ihnen genutzter lokaler JupyterLab-Server an Port 8888. Kann `ssh` den lokalen Port nicht binden, gibt es standardmäßig nur eine Warnung aus und läuft weiter. Welchen Server erreicht der Browser unter `http://127.0.0.1:8888`? Die Zelle `#interactive-development-identity` gibt sowohl auf dem Laptop als auch auf dem Server `~` für das Home-Verzeichnis aus. Mit welcher Prüfung unterscheiden Sie die beiden Rechner?
    1. Schreiben Sie eine Anleitung zum Verbindungsaufbau, bei der jeder der beiden Portkonflikte aus (b) und (c) einen erkennbaren Fehler auslöst.

<!--en:3fbc7f91-->
[Diskussion](https://d2l.discourse.group/t/36841)

<!--en:d624b916-->
Notebooks: lokale interaktive Entwicklung, Kernel-Zustand und entfernte GPUs<br>**Dokument und Kernel · abweichende Ausführungsreihenfolge · Neustart und Ausführung aller Zellen · uv-Lockdateien · Fehlersuche und Zeitmessung · SSH-Tunnel**

<!--en:9c6928f9-->
::: {.slide title="Wie interaktive Notebooks eine neue Fehlerquelle schaffen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:4793d92f-->
- **Vorteil des interaktiven Zustands:** Ein mehrere Gigabyte großer Datensatz bleibt im RAM oder VRAM, während Sie Zelle für Zelle Modellschichten und Diagramme überarbeiten.

<!--en:410a8926-->
- **Risiko des fortbestehenden Zustands:** Variablenbindungen überdauern die Zellen, die sie erzeugt haben. Wer eine Zelle bearbeitet oder löscht, beseitigt damit nicht ihre Nebenwirkungen im Speicher.

<!--en:87a3e4df-->
- **Grundregel:** Experimentieren Sie interaktiv Zelle für Zelle. Prüfen Sie das Ergebnis immer, indem Sie das gesamte Dokument in einem frisch gestarteten Kernel in Lesereihenfolge ausführen.

<!--en:3382e274-->
::: {.slide title="Dokument und Kernel: zwei verschiedene Reihenfolgen"}

<!--en:6c9ecd4b-->
[Ausführungsmodell]{.kicker}

<!--en:8fc8b16e-->
Ein Notebook besteht aus zwei Teilen, deren Zustände beim Bearbeiten auseinanderlaufen können:

<!--en:498e7af9-->
- **Dokument (`.ipynb` / `.md`):** Es speichert Text, Codezellen und gesicherte Ausgaben in der Reihenfolge **von oben nach unten**. Dieses Dokument lesen Sie, geben es weiter und nehmen es in Git auf.

<!--en:5ad32b09-->
- **Kernel (laufender Prozess):** Importierte Module, Tensoren, die Zustände der Pseudozufallszahlengeneratoren, offene Dateien und GPU-Speicherbelegungen ergeben sich aus der **bisherigen Ausführungsreihenfolge** (`In [1]`, `In [2]`, $\dots$).

<!--en:0e3a5fcb-->
- **Typischer Reihenfolgefehler:** Wer zuerst einen Datenlader ausführt und danach *darüber* eine Modellzelle einfügt, hat im laufenden Kernel Erfolg, weil `data` bereits existiert. Ein neuer Durchlauf von oben nach unten scheitert mit einem `NameError`.

<!--en:acf17bb4-->
::: {.slide title="Zustand in Dokumentreihenfolge und Ausführungszähler"}

<!--en:bc9989fc-->
[Abhängigkeiten zwischen Zuständen]{.kicker}

<!--en:9cdea087-->
Ein von mehreren Zellen genutzter Zustand ist unproblematisch, wenn jede Zelle nur von Zellen weiter oben im Dokument abhängt:

<!--en:92442e92-->
- **Datenfluss von oben nach unten:** Die erste Zelle setzt den Startwert eines NumPy-Zufallszahlengenerators und weist `samples` einen Wert zu. Die zweite Zelle berechnet `float(samples.mean())`.

<!--en:0bf927a5-->
- **Ausführungszähler (`In [k]`):** Sie geben die laufende Nummer seit dem Start des Kernels an. Damit liefern sie Hinweise auf frühere Ausführungen, aber keinen Abhängigkeitsgraphen.

<!--en:f0d46a29-->
- **Fehler durch den Zellaufbau vermeiden:** Beschränken Sie jede Zelle auf eine Aufgabe. Trennen Sie Definitionen reiner Funktionen von Experimentaufrufen, die den Zustand verändern. Übergeben Sie Tensoren explizit, statt globale Variablen über mehrere Zellen hinweg zu verändern.

<!--en:b49fb1bb-->
::: {.slide title="Drei Ursachen für versteckten Zustand im Kernel"}

<!--en:290c9dfa-->
[Gefahren für die Reproduzierbarkeit]{.kicker}

<!--en:64d4e4e9-->
- **1. Geänderte Reihenfolge oder wiederholte Ausführung:** Wer eine weiter unten stehende Zelle (`lr = 1e-4`) und danach erneut eine Trainingszelle weiter oben ausführt, erhält bei einem neuen Durchlauf von oben nach unten unbemerkt andere Zahlenwerte (`lr = 1e-2`).

<!--en:c61bff30-->
- **2. Überschreiben des eigenen Ausgangswerts:** Wer `pixels = pixels / 255` interaktiv zweimal ausführt, skaliert die Daten mit $1/255^2$. Bei einem frischen Durchlauf ist der Faktor $1/255$.

<!--en:935ce5c2-->
- **3. Nach der Ausführung gelöschte oder bearbeitete Zellen:** Veraltete Variablen bleiben im RAM (beim erneuten Durchlauf droht ein `NameError`), oder der angezeigte Code passt nicht mehr zur gespeicherten Ausgabe.

<!--en:7845cd24-->
::: {.slide title="Die goldene Regel: Kernel neu starten und alle Zellen ausführen"}

<!--en:7433f6af-->
[Überprüfung]{.kicker}

<!--en:7283e7dc-->
- **Was ein Neustart mit anschließender Ausführung aller Zellen aufdeckt:** Der Neustart beendet den Python-Prozess und gibt dessen gesamten RAM und VRAM frei. Beim neuen Durchlauf fallen ein `NameError` beim Zugriff auf undefinierte Namen, doppelte Änderungen am selben Wert und noch nicht ausgeführte Codeänderungen auf.

<!--en:6abf2858-->
- **Was ein frischer lokaler Durchlauf übersieht:** Lokale Cache-Dateien außerhalb der Versionsverwaltung (`features.npy`), fest eingetragene persönliche Pfade (`/home/alice/...`) und Zufallszahlengeneratoren ohne festen Startwert.

<!--en:a73dbafb-->
::: {.slide title="Deterministische lokale Umgebungen mit uv-Lockdateien"}

<!--en:0ddb7167-->
[Lokale Einrichtung]{.kicker}

<!--en:49d4388a-->
- **Exakter Abgleich mit der Lockdatei (`uv pip sync pylock.*.toml`):** Installiert die festgelegten Paketversionen und entfernt zusätzliche Pakete, die beim Import Vorrang haben könnten.

<!--en:7e479eef-->
- **Start in der festgelegten Umgebung (`uv run --no-sync jupyter lab`):** Startet JupyterLab, ohne die Abhängigkeiten erneut aufzulösen oder die festgelegte Umgebung `.venv` zu verändern.

<!--en:4c286fc6-->
- **Server und Kernel:** Prüfen Sie immer, ob der ausgewählte Notebook-Kernel tatsächlich auf `.venv/bin/python` verweist.

<!--en:03c5bcd3-->
::: {.slide title="Kernel und Arbeitsverzeichnis prüfen"}

<!--en:90254c1f-->
[Umgebung prüfen]{.kicker}

<!--en:ae298ce4-->
Prüfen Sie mit einer kurzen Abfrage am Anfang des Notebooks, in welcher Umgebung Sie arbeiten, bevor Sie Experimente ausführen oder Importfehler untersuchen:

<!--en:f54e5487-->
- **Prüfung von `sys.executable`:** Verweist der aktive Notebook-Kernel tatsächlich auf den Python-Interpreter der durch die Lockdatei festgelegten virtuellen Umgebung oder auf das systemweit installierte Python?

<!--en:c73ad559-->
- **Prüfung von `os.getcwd()`:** Stimmt das Arbeitsverzeichnis, sodass relative Pfade zum Datensatz (`../data`) zum Ziel führen?

<!--en:582a306c-->
- **Datenschutz bei der Ausgabe:** Wer den Pfad zum Home-Verzeichnis durch `~` ersetzt und nur das Ergebnis von `os.path.basename` ausgibt, vermeidet persönliche Benutzernamen in versionierten Notebook-Ausgaben.

<!--en:288f3c82-->
::: {.slide title="Effizientes Arbeiten in JupyterLab"}

<!--en:a11c5c4e-->
[Arbeiten im Editor]{.kicker}

<!--en:33c1521c-->
- **Unterbrechen (`I, I`) und neu starten (`0, 0`):** Eine Unterbrechung stoppt eine festgefahrene Schleife. Die Aufruffolge bleibt zur Untersuchung erhalten. Ein Neustart verwirft den Python-Zustand und gibt den VRAM der GPU frei.

<!--en:2dbe9d9a-->
- **Vorsicht vor verwaisten GPU-Kerneln:** Das Schließen eines Browser-Tabs **beendet den zugehörigen Kernel-Prozess nicht**. Ein einziger vergessener Notebook-Tab kann auf unbestimmte Zeit 10–20 GB VRAM belegen.

<!--en:f1559be7-->
- **Laufende Prozesse prüfen:** Beenden Sie ungenutzte Kernel ausdrücklich in der Seitenleiste **Running Terminals and Kernels**.

<!--en:004f82f3-->
::: {.slide title="Vom Notebook zu Modulen in VS Code"}

<!--en:4015b11a-->
[Projektarchitektur]{.kicker}

<!--en:9b035928-->
- **Wiederverwendbare Logik von Erläuterungen trennen:** Verschieben Sie getestete Datenlader, Modellklassen und Verlustfunktionen in `.py`-Module. Nutzen Sie das Notebook für Erläuterungen, Diagramme und Ablationsstudien.

<!--en:475c1f1e-->
- **Warum `.py`-Module bei `DataLoader(num_workers > 0)` helfen:** Unter `spawn` (macOS/Windows) und `forkserver` (Python 3.14+ unter Linux) importieren Arbeitsprozesse Funktionen anhand ihres Modulpfads. Das scheitert bei Funktionen, die das Notebook in `__main__` definiert.

<!--en:839ec956-->
::: {.slide title="Fehlersuche nach einer Ausnahme mit %debug" only="pytorch"}

<!--en:b529ef09-->
[Fehlerdiagnose]{.kicker}

<!--en:0dac5553-->
- **Sofortige Untersuchung nach einem Fehler:** Löst eine Zelle eine Ausnahme aus, öffnet `%debug` in der nächsten Zelle den Debugger `ipdb` direkt im betroffenen Aufrufrahmen. Alle lokalen Tensoren sind noch im Speicher.

<!--en:72f920ad-->
- **Haltepunkte im Editor:** Setzen Sie in JupyterLab oder VS Code Haltepunkte direkt an einzelnen Zeilen einer Zelle. Prüfen Sie so Schritt für Schritt Tensorformen und `NaN`-Werte.

<!--en:e839f228-->
- **Pauschales `try / except` vermeiden:** Wer die Ausnahme abfängt und nur `"failed"` ausgibt, verliert den Aufrufrahmen, den `%debug` braucht.

<!--en:78003298-->
::: {.slide title="Fehlersuche nach einer Ausnahme mit %debug" only="jax"}

<!--en:b529ef09-2-->
[Fehlerdiagnose]{.kicker}

<!--en:0dac5553-2-->
- **Sofortige Untersuchung nach einem Fehler:** Löst eine Zelle eine Ausnahme aus, öffnet `%debug` in der nächsten Zelle den Debugger `ipdb` direkt im betroffenen Aufrufrahmen. Alle lokalen Tensoren sind noch im Speicher.

<!--en:72f920ad-2-->
- **Haltepunkte im Editor:** Setzen Sie in JupyterLab oder VS Code Haltepunkte direkt an einzelnen Zeilen einer Zelle. Prüfen Sie so Schritt für Schritt Tensorformen und `NaN`-Werte.

<!--en:e839f228-2-->
- **Pauschales `try / except` vermeiden:** Wer die Ausnahme abfängt und nur `"failed"` ausgibt, verliert den Aufrufrahmen, den `%debug` braucht.

<!--en:641a220c-->
::: {.slide title="Fehlersuche nach einer Ausnahme mit %debug" only="tensorflow"}

<!--en:b529ef09-3-->
[Fehlerdiagnose]{.kicker}

<!--en:0dac5553-3-->
- **Sofortige Untersuchung nach einem Fehler:** Löst eine Zelle eine Ausnahme aus, öffnet `%debug` in der nächsten Zelle den Debugger `ipdb` direkt im betroffenen Aufrufrahmen. Alle lokalen Tensoren sind noch im Speicher.

<!--en:72f920ad-3-->
- **Haltepunkte im Editor:** Setzen Sie in JupyterLab oder VS Code Haltepunkte direkt an einzelnen Zeilen einer Zelle. Prüfen Sie so Schritt für Schritt Tensorformen und `NaN`-Werte.

<!--en:e839f228-3-->
- **Pauschales `try / except` vermeiden:** Wer die Ausnahme abfängt und nur `"failed"` ausgibt, verliert den Aufrufrahmen, den `%debug` braucht.

<!--en:e613a654-->
::: {.slide title="Fehlersuche nach einer Ausnahme mit %debug" only="mxnet"}

<!--en:b529ef09-4-->
[Fehlerdiagnose]{.kicker}

<!--en:0dac5553-4-->
- **Sofortige Untersuchung nach einem Fehler:** Löst eine Zelle eine Ausnahme aus, öffnet `%debug` in der nächsten Zelle den Debugger `ipdb` direkt im betroffenen Aufrufrahmen. Alle lokalen Tensoren sind noch im Speicher.

<!--en:72f920ad-4-->
- **Haltepunkte im Editor:** Setzen Sie in JupyterLab oder VS Code Haltepunkte direkt an einzelnen Zeilen einer Zelle. Prüfen Sie so Schritt für Schritt Tensorformen und `NaN`-Werte.

<!--en:e839f228-4-->
- **Pauschales `try / except` vermeiden:** Wer die Ausnahme abfängt und nur `"failed"` ausgibt, verliert den Aufrufrahmen, den `%debug` braucht.

<!--en:5223c18c-->
::: {.slide title="Mikrobenchmarks mit %timeit und die Tücken asynchroner GPUs"}

<!--en:dd8ecd62-->
[Leistungsmessung]{.kicker}

<!--en:c465e032-->
Der IPython-Magic-Befehl `%timeit` führt automatisch Aufwärmdurchläufe und mehrere Messschleifen aus. So misst er auch schnelle Operationen auf dem Host zuverlässig:

<!--en:f555b751-->
- **Zeitmessung auf Host und Beschleuniger:** Bei Arrays auf der CPU (`values @ values`) misst `%timeit` die tatsächliche Zeit bis zum Abschluss der Berechnung.

<!--en:f821971a-->
- **Fehlerquelle beim asynchronen GPU-Kernelstart:** Bei CUDA und XLA erhält Python die Kontrolle *sofort* zurück, sobald die Arbeit für den Datenstrom auf dem Gerät eingereiht ist.

<!--en:37669fd2-->
- **Drei Regeln für Zeitmessungen auf der GPU:**
  1. Führen Sie einen Aufwärmdurchlauf aus, um die JIT- oder CUDA-Kompilierung von der Messung auszuschließen.
  2. Synchronisieren Sie explizit (`torch.cuda.synchronize()` oder `.block_until_ready()`), bevor Sie die Zeitmessung beenden.
  3. Notieren Sie bei jedem Benchmark Tensorformen, Datentyp, Gerätenamen und Bibliotheksversionen.

<!--en:db61259f-->
::: {.slide title="Interaktive Entwicklung auf einer entfernten GPU: die beteiligten Prozesse"}

<!--en:5672dacf-->
[Architektur bei Fernzugriff]{.kicker}

<!--en:083c8937-->
Die Oberfläche Ihres Browsers oder Editors, der Jupyter-Server, der Python-Kernel und die GPU können sich auf verschiedenen physischen Rechnern befinden:

<!--en:e57bc481-->
- **Entkoppelte Architektur:** Auf einem leichten Laptop läuft die Browseroberfläche oder der VS-Code-Client. Notebook-Server und Kernel laufen direkt auf einer entfernten Workstation oder einer virtuellen Maschine in der Cloud, an die die GPU angeschlossen ist.

<!--en:b2f65c4c-->
- **Wo der Code läuft:** Tensoren und GPU-Speicher gehören zum **entfernten Kernel-Prozess**. Prüfen Sie bei VS Code Remote SSH immer, ob in der Statusleiste **SSH** steht, bevor Sie den Python-Interpreter auswählen. Ein lokaler Kernel auf dem Laptop hat keinen Zugriff auf die entfernte GPU.

<!--en:ae36ea61-->
::: {.slide title="Sicherer Fernzugriff über SSH-Portweiterleitung"}

<!--en:8effa457-->
[Netzwerksicherheit im Mehrbenutzerbetrieb]{.kicker}

<!--en:c90594e2-->
- **Niemals mit `--ip 0.0.0.0` an alle Schnittstellen binden:** Wer einen offen zugänglichen Jupyter-Port erreicht, kann unter Ihrer Kennung beliebige Shell-Befehle ausführen. Binden Sie den Server immer mit `--ip 127.0.0.1` an die Loopback-Adresse und nutzen Sie einen SSH-Tunnel (`ssh -N -L`).

<!--en:019657af-->
- **Portkonflikte auf Server und Laptop:** Auf einem gemeinsam genutzten Server ist Port `8888` möglicherweise schon von einem Kollegen belegt (das Token passt dann nicht). Auch auf Ihrem Laptop kann ein lokaler Server diesen Port belegen. Nutzen Sie einen eigenen Port und die Option `-o ExitOnForwardFailure=yes`.

<!--en:4cec6655-->
::: {.slide title="Verständnisfragen zu Kernel-Zustand und Zeitmessung bei Fernzugriff"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:21e8ffa4-->
Beantworten Sie diese drei Fragen zur Diagnose:

<!--en:93c420ab-->
1. Die 3 Zellen eines gespeicherten Notebooks zeigen die Ausführungszähler `In [3]`, `In [8]`, `In [9]`. Beweist das, dass die Ausgaben nicht reproduzierbar sind? Würden die Zähler `In [1]`, `In [2]`, `In [3]` die Reproduzierbarkeit garantieren?

<!--en:6b470755-->
2. Sie messen die Dauer einer BF16-Matrizenmultiplikation der Größe $8192 \times 8192$ auf einer RTX 4090 mit `time.perf_counter()` ohne Gerätesynchronisierung und erhalten $25\,\mu\text{s}$. Warum ist das physikalisch unmöglich?

<!--en:16e703aa-->
3. Welche fünf Prüfungen gehören auf Ihre Checkliste, bevor Sie ein fertiges Notebook in Git aufnehmen und weitergeben?

<!--en:557b98b6-->
**Antworten:**

<!--en:01374477-->
1. Lücken (`3, 8, 9`) zeigen, dass Zellen erneut ausgeführt oder gelöscht wurden. Das *warnt* vor möglichem verstecktem Zustand, beweist aber keinen Fehler in den Ausgaben. Umgekehrt gilt für `1, 2, 3`: **Auch diese Folge garantiert keine Reproduzierbarkeit**, wenn jemand eine Zelle *nach* der Ausführung bearbeitet hat. Nur ein **Neustart mit anschließender Ausführung aller Zellen** prüft, ob Code und Ausgabe übereinstimmen.

<!--en:1941631b-->
2. Eine Matrizenmultiplikation der Größe $8192^3$ erfordert $2 \times 8192^3 \approx 1.1 \times 10^{12}\text{ FLOPs}$. Bei $25\,\mu\text{s}$ ergäbe das $44{,}000\text{ TFLOP/s}$ ($\sim 270\times$ so viel wie die BF16-Spitzenleistung der 4090 von $165\text{ TFLOP/s}$). Sie haben den **Zeitaufwand für den CUDA-Kernelstart auf dem Host** gemessen, die GPU-Ausführung selbst blieb unberücksichtigt.

<!--en:fffd59b0-->
3. (i) Kernel neu starten und alle Zellen ausführen, (ii) persönliche Pfade und undokumentierte Downloads ausschließen, (iii) Zugangsdaten und Token aus den Ausgaben entfernen, (iv) ausufernde Fortschrittsprotokolle kürzen, (v) Umgebung und Hardwaredaten festhalten.

<!--en:da08c547-->
::: {.slide title="Zusammenfassung: reproduzierbare interaktive Entwicklung"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:ef880b34-->
- **Reproduzierbarkeit und Umgebung:** Führen Sie immer **Restart Kernel and Run All Cells** aus. Gleichen Sie die Abhängigkeiten mit `uv pip sync` mit der Lockdatei ab und prüfen Sie `sys.executable` in Zelle 1.

<!--en:62e2bb23-->
- **Ressourcen und Zeitmessung:** Beenden Sie verwaiste GPU-Kernel in der JupyterLab-Ansicht **Running** und synchronisieren Sie die GPU-Datenströme, bevor Sie eine Benchmark-Zeitmessung beenden.

<!--en:f90b0e06-->
- **Sicherer Fernzugriff:** Binden Sie Jupyter an `127.0.0.1` und nutzen Sie einen SSH-Tunnel mit `ExitOnForwardFailure=yes` oder VS Code Remote SSH.
