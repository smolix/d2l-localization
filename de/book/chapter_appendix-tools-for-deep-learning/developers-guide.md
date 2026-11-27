<!--en:5079c8f0-->
# Leitfaden zur Mitarbeit

<!--en:0353d411-->
Sie haben inzwischen viele Notebooks dieses Buches ausgeführt. Vielleicht ist Ihnen dabei ein unverständlicher Satz aufgefallen, ein vertauschter Index in einer Gleichung oder eine Zelle, die mit einer neueren Bibliotheksversion nicht mehr läuft. Eine Korrektur soll überall dort erscheinen, wo der Text oder Code vorkommt: auf der Website, im PDF, im Notebook jedes Frameworks und auf der Folie, die dieselbe Zelle zeigt. Wie gelangt eine einzeilige Korrektur an all diese Stellen? Und wie prüfen wir, ob danach noch alles funktioniert?

<!--en:43ed6569-->
Wir pflegen das Buch wie ein Open-Source-Projekt. Jede Seite hat eine einzige Markdown-Datei als Quelle, die unter Versionskontrolle steht. Aus diesen Dateien erzeugt ein Build-System alle weiteren Formate. Eine Änderung ist bereit zur Begutachtung, sobald die betroffenen Notebooks fehlerfrei laufen und sich die Website erzeugen lässt. Wir beschreiben den Build, das Vorgehen mit Git beim Vorschlagen einer Änderung, den optionalen Einsatz von Programmieragenten und die Prüfungen, die jede Änderung bestehen muss. Dieselben Fragen stellen sich in jedem Repository für maschinelles Lernen, dessen Dokumente ausführbar sein müssen.

<!--en:17cc5f72-->
## Das Build-System

<!--en:cce63472-->
### Eine Quelldatei, viele Artefakte

<!--en:10692b44-->
Jeder Abschnitt steht in einer einzigen Markdown-Datei. Für diesen Abschnitt ist das `chapter_appendix-tools-for-deep-learning/developers-guide.md`. Aus diesen Quelldateien erzeugt das Build-System alle weiteren Formate:

<!--en:0f3d901b-->
Daraus ergeben sich drei praktische Konsequenzen:

<!--en:331e41ed-->
* **Bearbeiten Sie niemals erzeugte Dateien.** Der nächste Build überschreibt jede Änderung an einer `.qmd`- oder `.ipynb`-Datei. Bearbeiten Sie die `.md`-Quelldatei und erzeugen Sie die Dateien neu.

<!--en:1d67e654-->
* **Die Framework-Tabs stehen in einer Datei.** Ein Codeblock mit der Markierung `%%tab pytorch` erscheint nur in der PyTorch-Ansicht. Ein Python-Block ohne Markierung erscheint in jedem Notebook, das aus der Datei entsteht. Für jedes Framework, für das mindestens ein Block markiert ist, entsteht ein Notebook. Enthält die Datei keine markierten Blöcke, entsteht nur ein PyTorch-Notebook. Eine Quelldatei liefert somit bis zu vier Notebooks, die alle ausführbar sein müssen. Die Kapitel der Teile „Grundlagen“, „Bildmodelle“, „Mathematik für Deep Learning“ und „Werkzeuge für Deep Learning“ unterstützen alle vier Frameworks. Die Kapitel für Fortgeschrittene verwenden PyTorch und JAX (in :numref:`chap_reinforcement_learning` nur PyTorch), der Teil über Sprachmodelle verwendet PyTorch.

<!--en:acc22337-->
* **Zellen behalten ihre Kennung.** Jeder Python-Block erhält von `tools/add_cell_ids.py` eine dauerhafte Kennung. So können Folien, gespeicherte Ausgaben und Framework-Ansichten auch nach dem Umbenennen oder Verschieben auf dieselbe Zelle verweisen.

<!--en:623cbf02-->
Wegen der Versionskontrolle verwenden wir Markdown als Quellformat. Eine `.ipynb`-Datei enthält Code, Ausgaben, Ausführungszähler und Metadaten des Editors gemeinsam im JSON-Format. Ändert man ein Wort und führt das Notebook erneut aus, ändern sich viele weitere Zeilen. Bei Markdown zeigt der Versionsvergleich nur die Änderungen an Text und Code. Die Ausgaben eines Laufs speichern wir separat. Wir erfassen sie nur dann neu, wenn sich der zugehörige Zellcode, der von ihm verwendete Bibliothekscode oder die Framework-Version ändert.

<!--en:b0b597d1-->
Die Bibliothek `d2l`, die Sie im ganzen Buch importiert haben, entsteht auf dieselbe Weise. Der Befehl `make lib` übernimmt alle mit `#@save` markierten Blöcke in die jeweiligen Framework-Module `d2l/torch.py`, `d2l/tensorflow.py`, `d2l/jax.py` und `d2l/mxnet.py`. Korrigieren Sie eine Funktion dort, wo das Buch sie einführt, erreicht die Korrektur beim nächsten Build alle späteren Kapitel. Deshalb steht am Anfang dieser Dateien der Hinweis, sie nicht direkt zu bearbeiten. Auch die Colab- und Kaggle-Notebooks aus :numref:`sec_hosted_notebooks` entstehen automatisch auf einem eigenen Branch namens `notebooks`. Dieser entsteht jedes Mal neu aus den Quelldateien und wird nie von Hand bearbeitet.

<!--en:0548b902-->
### Aufbau des Repositorys

<!--en:6e9cc751-->
Im Stammverzeichnis des Repositorys finden Sie die Quelldateien, die Einstiegspunkte für den Build und die zugehörige Dokumentation:

<!--en:a4a45156-->
Nach dem Klonen installiert `./bootstrap.sh` zunächst Git LFS. Dort liegen die Bilder der in Git gespeicherten Ausgaben. Das Skript lädt die Bilder herunter und erstellt `.venv-build`, die Umgebung mit Quarto. Anschließend erzeugt `make html` aus den gespeicherten Ausgaben die gesamte Website, auch auf einem Laptop ohne GPU. Die Notebooks auszuführen ist ein eigener Schritt. Das Makefile benötigt GNU Make ab Version 4.3. Unter macOS müssen Sie diese Version gesondert installieren (etwa als `gmake` über MacPorts). Der Befehl `make help` listet die Build-Ziele auf, und `docs/build-system.md` erläutert sie.

<!--en:e0cc17a7-->
## Grundlagen von Git

<!--en:b236ea35-->
Jede Änderung am Buch reichen Sie als Git-Commit auf einem Branch Ihrer eigenen Kopie des Repositorys ein. Wenn Sie Git bereits täglich nutzen, können Sie diesen Abschnitt überspringen. Legen Sie zunächst die Kopie an: Klicken Sie auf der [Repository-Seite](https://github.com/smolix/d2l-neu) auf **Fork**. Klonen Sie dann Ihren Fork und erstellen Sie einen Branch für Ihre Änderung:

<!--en:3ba5a521-->
Bearbeiten Sie die `.md`-Quelldatei. Speichern und veröffentlichen Sie dann die Änderung mit Git:

<!--en:85f8e70e-->
GitHub bietet Ihnen an, einen *Pull-Request* zu öffnen. Damit schlagen Sie vor, Ihren Branch in das Projekt zu übernehmen. Die Projektverantwortlichen können die Änderungen im Versionsvergleich prüfen und kommentieren. (Mit dem [Kommandozeilenwerkzeug `gh`](https://cli.github.com/) erledigt das der Befehl `gh pr create`.) Kleine Pull-Requests, die sich auf eine Sache beschränken, lassen sich leichter prüfen als Änderungen über mehrere Kapitel hinweg.

<!--en:0c45ff19-->
## Programmieragenten

<!--en:16e4fa6a-->
Ein Programmieragent ist ein Sprachmodell, das das Repository liest, Dateien bearbeitet und Befehle ausführt. Er hilft bei klar umrissenen Aufgaben. Er kann etwa eine Funktionsdefinition finden, dieselbe Änderung auf mehrere Framework-Varianten übertragen oder Prüfungen ausführen und Fehlermeldungen auswerten. Der Einsatz ist freiwillig. Änderungen eines Agenten müssen ebenso sorgfältig geprüft werden wie alle anderen Beiträge. Drei Eigenschaften dieses Repositorys erleichtern den Einsatz:

<!--en:26578e40-->
* **Das Build-System ist dokumentiert.** In `CLAUDE.md` und `docs/` steht, wie ein Agent die Artefakte neu erzeugt, ausführt und rendert. Dieselben Anleitungen gelten auch für Menschen, die am Buch mitarbeiten.

<!--en:edfc3c28-->
* **Die Prüfungen sind ausführbar.** Eine Änderung funktioniert, wenn das Notebook in jedem unterstützten Framework von Anfang bis Ende läuft, die Prüfung der in Git gespeicherten Ausgaben erfolgreich ist und sich die Seite rendern lässt. Ein Agent kann die Dateien bearbeiten, neu erzeugen und ausführen, Fehler untersuchen und diese Schritte wiederholen. Wer die Änderung einreicht, prüft sowohl den abschließenden Versionsvergleich als auch die Prüfergebnisse.

<!--en:e2dcd934-->
* **Die mühsamen Schritte folgen festen Regeln.** Einige Aufgaben betreffen mehrere Dateien: die Implementierungen der Frameworks auf demselben Stand halten, für jeden Verweis mit `:numref:` das Ziel prüfen und Abbildungen neu erzeugen. Ein Agent kann diese Aufgaben übernehmen, ein Mensch das Ergebnis prüfen.

<!--en:20b47049-->
Ein guter Auftrag benennt die Änderung und gibt an, wie sie sich prüfen lässt. Zum Beispiel: „Das Konturdiagramm im Abschnitt über Momentum hat falsche Achsenbeschriftungen. Korrigieren Sie die Quelldatei und führen Sie die betroffenen Notebooks erneut aus.“ Achten Sie bei der Prüfung auf zwei mögliche Fehler. Wie jemand, der neu im Projekt ist, bearbeitet ein Agent unter Umständen eine erzeugte Datei, wenn die Dokumentation nicht davor warnt. Unsere Dokumentation enthält diesen Hinweis. Soll ein Agent einen fehlgeschlagenen Test zum Bestehen bringen, schwächt er womöglich den Test oder den Code ab, ohne die Ursache zu beheben. Das Notebook läuft dann wieder, vermittelt aber möglicherweise etwas Falsches.

<!--en:c46fde9e-->
## Einreichen von Änderungen

<!--en:c529b02d-->
Beiträge reichen von einzeiligen Korrekturen bis zu neuem Code. Der Umfang der Prüfungen richtet sich nach der Änderung. Für eine einzeilige Korrektur brauchen Sie nicht einmal eine lokale Arbeitskopie. Öffnen Sie die Quelldatei auf GitHub und klicken Sie zum Bearbeiten auf das Stiftsymbol. GitHub bietet an, einen Fork des Repositorys anzulegen, und führt Sie nach der Bearbeitung vom Commit bis zum Pull-Request. Alles geschieht im Browser.

<!--en:9c99139a-->
Für Änderungen an Code und Inhalt klonen Sie das Repository, legen einen Branch an und bearbeiten die `.md`-Quelldatei. Prüfen Sie die bearbeitete Datei mit `tools/lint_source.py`. Das Werkzeug erkennt unpaarige Tab- und Folienmarkierungen, doppelte Zellkennungen und Gleichungsmarken, die keiner Gleichung zugeordnet würden. Mit `--corpus` prüft es außerdem, ob jeder Querverweis auf eine vorhandene Marke zeigt. Führen Sie dann jedes betroffene Notebook erneut aus, kopieren Sie seine Ausgaben in den versionierten Ausgabespeicher und rendern Sie die Website:

<!--en:da142d4a-->
Wenn Sie Code mit der Markierung `#@save` ändern, erzeugen Sie zuerst die Bibliothek mit `make lib` neu. Führen Sie erst dann ein Notebook aus. Auch die Notebooks anderer Abschnitte, die das geänderte Symbol verwenden, gelten nun als veraltet. Der Befehl `make audit-outputs` listet sie auf. Auf einem Laptop ohne GPU lassen sich GPU-Notebooks nicht erneut ausführen. In diesem Fall gibt `make html` eine Warnung aus und setzt den Build fort. Geben Sie im Pull-Request an, welche Notebooks noch auf einer GPU laufen müssen. Für neuen Text gilt `docs/writing-guide.md`, für neuen Code `docs/code-guide.md`.

<!--en:b2c56cd5-->
Wenn Sie lieber im Notebook als im Markdown-Quelltext arbeiten, verwenden Sie die VS-Code-Erweiterung in `.vscode-extension/`. Die zugehörige README-Datei beschreibt, wie Sie die Erweiterung bauen und installieren. Die Erweiterung öffnet das erzeugte Notebook des aktuellen Abschnitts und wählt den passenden Kernel vorab aus. Sie können zwischen Framework-Ansichten wechseln und die Folien in einer Live-Vorschau ansehen. Mit einem Befehl führen Sie das Notebook erneut aus und erfassen seine Ausgaben. Die Erweiterung schreibt Änderungen im Notebook in die Markdown-Quelldatei zurück. So gehen Ihre Änderungen beim nächsten Build nicht verloren. Der Befehl `make kernels` registriert die verwendeten Jupyter-Kernel.

<!--en:41d3348a-->
Unabhängig vom Vorgehen ist eine Änderung bereit zur Begutachtung, wenn der Code in jedem betroffenen Framework aus einem frischen Anfangszustand fehlerfrei läuft und die Änderung in der Quelldatei steht. Erzeugte Dateien werden nie von Hand bearbeitet.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:8828a6c8-->
Aus einer einzigen Markdown-Quelldatei pro Abschnitt entstehen die Website, PDFs, Notebooks für die einzelnen Frameworks, Folien und das Paket `d2l`. Die Ausgaben der ausgeführten Zellen stehen separat unter Versionskontrolle. Erzeugte Dateien werden nie von Hand bearbeitet. Da Text und Code als reines Markdown vorliegen, zeigt Git Änderungen übersichtlich Zeile für Zeile. So lässt sich auch ein Buch mit mehr als 600 Notebooks gut prüfen und pflegen. Für jeden Beitrag gilt derselbe Ablauf. Legen Sie einen Fork des Repositorys und einen Branch für die jeweilige Änderung an. Bearbeiten Sie die `.md`-Quelldatei, prüfen Sie sie mit dem Linter und führen Sie die betroffenen Notebooks erneut aus. Erfassen Sie die Ausgaben, rendern Sie die Website und öffnen Sie einen kleinen Pull-Request. Weil die Build-Regeln dokumentiert und die Prüfungen ausführbar sind, können Programmieragenten viele Routinearbeiten über mehrere Frameworks hinweg automatisieren. Die Menschen, die den Beitrag einreichen, prüfen die Änderungen und tragen dafür die Verantwortung.

<!--en:2294243a-->
## Übungen

<!--en:aef59e43-->
1. **Framework-Tabs.** Eine Quelldatei enthält genau fünf Codeblöcke in dieser Reihenfolge: (1) einen mit `%%tab pytorch` markierten Block, der `import torch` und `x = torch.arange(6.0)` ausführt, (2) einen mit `%%tab jax` markierten Block, der `import jax.numpy as jnp` und `x = jnp.arange(6.0)` ausführt, (3) den Block `y = x * 2` ohne Markierung, (4) einen mit `%%tab pytorch` markierten Block, der `x[0] = 10.0` ausführt, und (5) den Block `print(y.sum(), x.sum())` ohne Markierung.
    1. Welche Notebooks erzeugt das Build-System aus dieser Datei, welche Blöcke enthält jedes Notebook und was gibt es aus?
    1. Jemand entfernt die Markierung von Block (4), sodass jedes Framework den Block ausführt. Welches Notebook bricht nun mit welchem Fehler ab? Ersetzen Sie Block (4) durch markierte Blöcke, sodass beide Notebooks dieselben Werte ausgeben.
    1. Sie bearbeiten Block (3) im erzeugten PyTorch-Notebook. Die VS-Code-Erweiterung schreibt die Änderung in die Quelldatei zurück. Welche Notebooks ändern sich dadurch? Welche müssen Sie ausführen, bevor Sie die Änderung einreichen?
   :exlabel:`ex_developers-guide-framework-tabs`

<!--en:80709989-->
1. **Veraltete Ausgaben.** Der versionierte Ausgabespeicher enthält die Ausgaben eines Notebooks in `outputs/<fw>/<chapter>/<stem>.json`. Jede Ausgabe steht unter der Kennung der Zelle, die sie erzeugt hat. Zu jeder Ausgabe gehört ein Fingerabdruck: ein Hashwert über den Code dieser Zelle und aller vorhergehenden Codezellen im Notebook. Leerraum am Ende jeder Zeile bleibt dabei unberücksichtigt. Für das gesamte Notebook speichert das System außerdem die installierte Framework-Version ohne lokale Build-Kennung (`torch==2.11.0` statt `torch==2.11.0+cu128`) sowie einen weiteren Hashwert. In diesen gehen die Bibliotheksblöcke aller Symbole `d2l.X` ein, die im Code des Abschnitts vorkommen, sowie die Blöcke, auf die diese Symbole wiederum zurückgreifen. Eine Ausgabe gilt als veraltet, sobald einer dieser Werte nicht mehr zum aktuellen Quelltext oder zur aktuellen Umgebung passt. Die Prüfung mit `tools/audit_outputs.py` läuft bei `make html` vor dem Rendern der Website. Bei einem veralteten Bild gibt sie eine Warnung aus. Bei einer veralteten Textausgabe bricht sie den Build ab, sofern das Notebook nicht mehr GPUs benötigt, als der Rechner besitzt.
    1. Ein Abschnitt enthält zehn Codezellen. Jede seiner Framework-Varianten ruft `d2l.Trainer` auf. Welche Ausgaben gelten nach den folgenden Änderungen in welchen Frameworks als veraltet? (i) In einem Absatz wird ein Tippfehler korrigiert. (ii) In einem Kommentar der dritten Codezelle, die keine Tab-Markierung trägt, wird ein Tippfehler korrigiert. (iii) Leerzeichen am Ende jeder Zeile werden entfernt. (iv) Die fünfte und die sechste Codezelle tauschen die Plätze. (v) In einer `#@save`-Zelle ohne Tab-Markierung in einem anderen Abschnitt ändert sich der Docstring einer `Trainer`-Methode. (vi) PyTorch wird von 2.11.0 auf 2.12.0 aktualisiert. (vii) Für PyTorch 2.11.0 wird die CPU-Version für macOS anstelle der CUDA-Version für Linux installiert.
    1. Welche Markierungen aus (a) betreffen Ausgaben, die sich gar nicht geändert haben können? Welche tatsächliche Änderung einer Ausgabe bleibt in (vii) unentdeckt? Warum nehmen die Regeln beide Fehlerarten in Kauf?
    1. Warum stuft das Build-System eine Ausgabe nicht bereits dann als veraltet ein, wenn ein neuer Lauf etwas anderes ausgibt als der gespeicherte Lauf?
   :exlabel:`ex_developers-guide-stale-outputs`

<!--en:24a21b14-->
1. [code] **Doppelte Definitionen.** Der Befehl `make lib` sammelt die `#@save`-Blöcke aller Abschnitte in der Reihenfolge des Buches. Definieren mehrere Blöcke dieselbe Funktion oder Methode, übernimmt die Bibliothek nur eine der Definitionen. Die Methode `Trainer.prepare_batch` ist in der Zelle `#linear-regression-scratch-training-1` von :numref:`sec_oo-design` definiert. Sie trägt keine Tab-Markierung und läuft in jedem Framework. Die Framework-Varianten der Zelle `#gpus-devices-memory-the-trainer-now-with-devices-1` in :numref:`sec_use_gpu` definieren die Methode erneut, diesmal mit Geräteunterstützung.
    1. Ergänzen Sie je eine Kommentarzeile im Rumpf der Definition ohne Tab-Markierung und in der PyTorch-Variante mit Geräteunterstützung. Führen Sie `make lib` aus und suchen Sie in `d2l/torch.py` nach beiden Kommentaren. Welcher Kommentar erscheint dort, und welche Regel lässt sich daran erkennen?
    1. Welche Definition enthält `d2l.Trainer.prepare_batch` beim Start des PyTorch-Notebooks zu :numref:`sec_oo-design`? Welche Definition gilt nach Ausführung der Zelle `#linear-regression-scratch-training-1`?
    1. Warum übernimmt das Build-System die letzte Definition statt der ersten?
    1. Jemand ändert die PyTorch-Methode `prepare_batch` mit Geräteunterstützung so, dass sie jeden Batch zusätzlich in 16-Bit-Gleitkommazahlen umwandelt. Welche PyTorch-Notebooks können nun andere Ergebnisse ausgeben: diejenigen vor :numref:`sec_use_gpu` in der Reihenfolge des Buches, diejenigen danach oder beide Gruppen?
   :exlabel:`ex_developers-guide-duplicate-definitions`

<!--en:08633609-->
1. [extended] **Veraltete Ausgaben anhand des Datenflusses erkennen.** ●● Der Fingerabdruck aus :exref:`ex_developers-guide-stale-outputs` kennzeichnet ab der ersten bearbeiteten Zelle jede Zelle als veraltet. Eine Datenflussregel markiert nur Zellen, die direkt oder über andere Zellen einen Namen lesen, den die bearbeitete Zelle bindet. Das Prüfwerkzeug aus :exref:`ex_interactive-development-use-before-define` in :numref:`sec_interactive_development` ermittelt bereits für jede Zelle, welche Namen sie bindet und welche sie liest.
    1. Erzeugen Sie die PyTorch-Notebooks mit `make notebooks-pytorch` und ermitteln Sie für jede Codezelle die Namen, die sie bindet und liest. Betrachten Sie nacheinander jede einzelne Zelle als bearbeitet. Bestimmen Sie jeweils, welche Zellen die Datenflussregel als veraltet markiert und welche zur Neuberechnung laufen müssen. Dazu zählen auch alle früheren Zellen, von denen eine veraltete Zelle abhängt. Geben Sie pro Kapitel für beide Regeln den mittleren Anteil der als veraltet markierten Zellen und den mittleren Anteil der erneut auszuführenden Zellen an.
    1. Suchen Sie im Buch nach Zellen, bei denen eine Bearbeitung spätere Ausgaben verändert, ohne einen Namen zu binden, den die späteren Zellen lesen. Das kann etwa über den Zustand eines Zufallszahlengenerators geschehen. Welche weiteren Kanäle dieser Art nutzen die Notebooks des Buches?
    1. :citet:`Macke.Gong.Lee.ea.2021` zeichnen solche Abhängigkeiten beim Ausführen des Notebooks auf, statt sie aus dem Quelltext abzuleiten. Welche Kanäle aus (b) kann ein Werkzeug zur Laufzeitverfolgung beobachten, und welche bleiben ihm verborgen?
    1. Würden Sie die Präfixregel des Build-Systems durch Ihre Regel ersetzen? Begründen Sie Ihre Antwort anhand der Einsparungen aus (a) und der Fälle aus (b), in denen die Regel versagt.
   :exlabel:`ex_developers-guide-dataflow-staleness`

<!--en:86ef2f57-->
1. [code] **Bisektion mit verrauschten Testergebnissen.** ●● Mit einer binären Suche findet `git bisect` den Commit, der einen Fehler eingeführt hat. Dazu lädt Git einen Commit in der Mitte zwischen einem bekanntermaßen fehlerfreien und einem bekanntermaßen fehlerhaften Commit. Ein Test an diesem Commit halbiert den Suchbereich. Ein Test, der ein Modell trainiert, liefert zufallsabhängige Ergebnisse. Angenommen, ein Notebook besteht die Prüfung „Validierungsgenauigkeit mindestens 0.9“ mit Wahrscheinlichkeit $p = 0.9$ bei jedem Commit vor dem ersten fehlerhaften Commit $k$ und mit Wahrscheinlichkeit $q = 0.3$ bei $k$ und jedem späteren Commit. Die Läufe sind voneinander unabhängig. Commit 0 ist nachweislich fehlerfrei, Commit 64 nachweislich fehlerhaft, und $k$ ist auf $\{1, \ldots, 64\}$ gleichverteilt.
    1. Mit welcher Wahrscheinlichkeit findet `git bisect` bei einem Lauf pro Test den Commit $k$?
    1. Wiederholen Sie jeden Test $m$-mal. Stufen Sie einen Commit als fehlerfrei ein, wenn die Anzahl der bestandenen Läufe einen Schwellenwert erreicht. Bestimmen Sie das kleinste $m$ und den zugehörigen Schwellenwert, für die die Wahrscheinlichkeit, einen falschen Commit zu nennen, höchstens 5% beträgt. Wie viele Läufe benötigt die Suche?
    1. Der probabilistische Bisektionsalgorithmus führt eine A-posteriori-Verteilung über die Position des Fehlers. Er testet an ihrem Median und aktualisiert die Verteilung nach jedem Lauf mit dem Satz von Bayes :cite:`Frazier.Henderson.Waeber.2019`. Simulieren Sie den Algorithmus mit einer A-posteriori-Verteilung über $k$ und den durch $p$ und $q$ gegebenen Likelihoods, bis die Verteilung einem Commit die Wahrscheinlichkeit 0.95 zuweist. Warum kann die Suche auf einer diskreten Menge von Commits ins Stocken geraten, wenn sie immer am Median testet? Ändern Sie die Wahl des Test-Commits, um das Problem zu beheben, und vergleichen Sie die mittlere Anzahl der Läufe und die Fehlerrate mit (b).
    1. Fassen Sie einen Lauf als Kanal von $\{\text{good}, \text{bad}\}$ nach $\{\text{pass}, \text{fail}\}$ auf. Berechnen Sie seine Kapazität $C = \max_{p(x)} I(X; Y)$, die größte gegenseitige Information zwischen Eingabe und Ausgabe über alle Eingabeverteilungen (:numref:`sec_mdl-mutual-information`). Jeder Lauf liefert höchstens $C$ Bit Information über $k$, auch bei adaptiver Wahl der Tests. Nach der Fano-Ungleichung erfüllt ein aus den Läufen berechneter Schätzwert $\hat k$ mit Fehlerwahrscheinlichkeit $P_e = P(\hat k \neq k)$ die Ungleichung $H_b(P_e) + P_e \log_2 63 \geq H(k \mid \text{runs})$, wobei $H_b$ die binäre Entropie bezeichnet. Leiten Sie daraus eine untere Schranke für die mittlere Anzahl der Läufe jeder Strategie ab, die $k$ mit einer Wahrscheinlichkeit von mindestens 0.95 findet. Wie nahe kommt (c) dieser Schranke?
   :exlabel:`ex_developers-guide-noisy-bisection`

<!--en:67664e51-->
[Diskussion](https://d2l.discourse.group/t/36848)

<!--en:2c42fbb5-->
Leitfaden zur Mitarbeit: Build aus einer Markdown-Quelle, Framework-Tabs und Arbeiten mit Git<br>**Markdown als verbindliche Quelle · Tabs für mehrere Frameworks · Bibliothekserzeugung mit #@save · Fingerabdrücke für Ausgaben · Prüfung von Quellen und Ausgaben · Pull-Requests mit Git**

<!--en:37930bc9-->
::: {.slide title="Warum D2L Markdown als einzige Quelle verwendet"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:b25ba9e7-->
- **Warum sich reine `.ipynb`-Dateien für Git schlecht eignen:** Verschachteltes JSON vermischt Text, Code, Base64-kodierte PNGs und Ausführungszähler. Schon die Korrektur von 1 Wort mit anschließendem Zelllauf ändert 500 oder mehr Zeilen. So entstehen immer wieder Konflikte beim Zusammenführen.

<!--en:6a5825f5-->
- **Markdown als einzige Quelle (`chapter_*/*.md`):** Bei reinem, gut lesbarem Markdown umfasst der Git-Vergleich für eine solche Korrektur nur 1 Zeile. Die Ausgaben ausgeführter Zellen liegen separat in `outputs/<fw>/*.json`.

<!--en:8dac19b8-->
::: {.slide title="Eine Quelldatei, fünf erzeugte Artefakte"}

<!--en:a0f7545a-->
[Der Build-Ablauf]{.kicker}

<!--en:9b35ef98-->
- **Grundregel:** **Nur** `chapter_*/*.md` bearbeiten (niemals `_notebooks/`, `*.qmd`, `_slides/` oder `d2l/*.py`).

<!--en:2619982e-->
- **Fest vorgegebene Ableitung:** Das Build-System fügt die Ausgaben aus `outputs/<fw>/*.json` ein und erzeugt automatisch die HTML-Website, PDFs, `.ipynb`-Notebooks für jedes Framework, Reveal.js-Foliensätze und das Paket `d2l`.

<!--en:d760e7a8-->
::: {.slide title="Framework-Tabs und Notebook-Erzeugung"}

<!--en:90e563c5-->
[Gleichstand der Framework-Varianten]{.kicker}

<!--en:7de9ba80-->
- **Zellen mit und ohne Markierung (`%%tab <fw>`):** Markierte Zellen erscheinen nur in den angegebenen Frameworks. Python-Zellen **ohne Markierung** laufen in **jedem** Framework-Notebook, das aus der Datei entsteht.

<!--en:a9c32fdd-->
- **Gemeinsame Zellen:** Nach Änderungen an einer Zelle ohne Markierung die Notebooks aller Frameworks der Datei erneut ausführen und ihre Ausgaben erfassen (in den Kapiteln für Fortgeschrittene `pytorch` und `jax`).

<!--en:de0b50db-->
::: {.slide title="Dauerhafte Zellkennungen und Bibliothekserzeugung mit #@save"}

<!--en:84b00c37-->
[Kennung und Wiederverwendung von Code]{.kicker}

<!--en:ff9ba578-->
- **Dauerhafte Zellkennungen (`#sec-slug-k`):** `tools/add_cell_ids.py` vergibt die Kennungen. Auch nach Textänderungen identifizieren sie gespeicherte Ausgaben in `outputs/<fw>/` und dienen auf Folien als Verweisziel (`@cell-id`).

<!--en:b2c2d310-->
- **Bibliothek aus `#@save`-Blöcken (`make lib`):** Der Build übernimmt die Blöcke in Kapitelreihenfolge in `d2l/*.py`. **Die letzte Definition gilt.** So überschreibt der GPU-fähige `Trainer` aus §{sec_use_gpu} den CPU-`Trainer` aus §{sec_oo-design} in `d2l`. Das Notebook zu §{sec_oo-design} führt seine eigene Definition aus.

<!--en:7dc0a9c6-->
::: {.slide title="Aufbau des Repositorys und Einrichtung nach dem Klonen"}

<!--en:ab28e5ea-->
[Entwicklungsumgebung]{.kicker}

<!--en:69c6ace7-->
- **Einmalig `./bootstrap.sh` ausführen:** Das Skript lädt die Diagramme aus Git LFS und erstellt `.venv-build` mit Quarto und den Python-Werkzeugen für den Build.

<!--en:ce36afce-->
- **Lokaler HTML-Build ohne GPU (`gmake html`):** Der Build fügt die in Git gespeicherten Ausgaben aus `outputs/<fw>/*.json` in die `.qmd`-Dateien ein. Auf jedem Laptop erzeugt er in Sekunden die interaktive Website und die Folien (setzt GNU Make $\ge 4.3$ voraus).

<!--en:ff5103fb-->
::: {.slide title="Wie Fingerabdrücke der Ausgaben veraltete Notebooks erkennen lassen" only="pytorch"}

<!--en:e609ded9-->
[Prüfung der Reproduzierbarkeit]{.kicker}

<!--en:5fccbc95-->
- **Drei Kennwerte je JSON-Ausgabedatei:** (1) **Hashwert des Codes bis zur jeweiligen Zelle** (Zellen $1\dots k$, ohne Leerzeichen am Zeilenende), (2) **öffentliche Framework-Version** (ohne lokale Build-Kennung) und (3) **Hashwert der Symbole `d2l.X` samt transitiven Abhängigkeiten**.

<!--en:b82b7da8-->
- **Gezielte Kennzeichnung:** Bei Textänderungen veralten 0 Zellen. Ändert sich Zelle $k$, gelten die Zellen $k\dots n$ als veraltet. Ändert sich eine `#@save`-Hilfsfunktion, betrifft das nur nachgelagerte Notebooks, die sie aufrufen.

<!--en:572a0fca-->
::: {.slide title="Wie Fingerabdrücke der Ausgaben veraltete Notebooks erkennen lassen" only="jax"}

<!--en:e609ded9-2-->
[Prüfung der Reproduzierbarkeit]{.kicker}

<!--en:5fccbc95-2-->
- **Drei Kennwerte je JSON-Ausgabedatei:** (1) **Hashwert des Codes bis zur jeweiligen Zelle** (Zellen $1\dots k$, ohne Leerzeichen am Zeilenende), (2) **öffentliche Framework-Version** (ohne lokale Build-Kennung) und (3) **Hashwert der Symbole `d2l.X` samt transitiven Abhängigkeiten**.

<!--en:b82b7da8-2-->
- **Gezielte Kennzeichnung:** Bei Textänderungen veralten 0 Zellen. Ändert sich Zelle $k$, gelten die Zellen $k\dots n$ als veraltet. Ändert sich eine `#@save`-Hilfsfunktion, betrifft das nur nachgelagerte Notebooks, die sie aufrufen.

<!--en:462fdc4a-->
::: {.slide title="Wie Fingerabdrücke der Ausgaben veraltete Notebooks erkennen lassen" only="tensorflow"}

<!--en:e609ded9-3-->
[Prüfung der Reproduzierbarkeit]{.kicker}

<!--en:5fccbc95-3-->
- **Drei Kennwerte je JSON-Ausgabedatei:** (1) **Hashwert des Codes bis zur jeweiligen Zelle** (Zellen $1\dots k$, ohne Leerzeichen am Zeilenende), (2) **öffentliche Framework-Version** (ohne lokale Build-Kennung) und (3) **Hashwert der Symbole `d2l.X` samt transitiven Abhängigkeiten**.

<!--en:b82b7da8-3-->
- **Gezielte Kennzeichnung:** Bei Textänderungen veralten 0 Zellen. Ändert sich Zelle $k$, gelten die Zellen $k\dots n$ als veraltet. Ändert sich eine `#@save`-Hilfsfunktion, betrifft das nur nachgelagerte Notebooks, die sie aufrufen.

<!--en:4e85d6ce-->
::: {.slide title="Wie Fingerabdrücke der Ausgaben veraltete Notebooks erkennen lassen" only="mxnet"}

<!--en:e609ded9-4-->
[Prüfung der Reproduzierbarkeit]{.kicker}

<!--en:5fccbc95-4-->
- **Drei Kennwerte je JSON-Ausgabedatei:** (1) **Hashwert des Codes bis zur jeweiligen Zelle** (Zellen $1\dots k$, ohne Leerzeichen am Zeilenende), (2) **öffentliche Framework-Version** (ohne lokale Build-Kennung) und (3) **Hashwert der Symbole `d2l.X` samt transitiven Abhängigkeiten**.

<!--en:b82b7da8-4-->
- **Gezielte Kennzeichnung:** Bei Textänderungen veralten 0 Zellen. Ändert sich Zelle $k$, gelten die Zellen $k\dots n$ als veraltet. Ändert sich eine `#@save`-Hilfsfunktion, betrifft das nur nachgelagerte Notebooks, die sie aufrufen.

<!--en:afa5544a-->
::: {.slide title="Warum die Präfixregel veraltete Ausgaben zuverlässiger erkennt als statische Datenflussanalyse"}

<!--en:7fc29b45-->
[Verborgene Seiteneffekte]{.kicker}

<!--en:7d60cd89-->
- **Warum Variablennamen im abstrakten Syntaxbaum nicht ausreichen:** Eine zusätzliche Zufallsziehung in Zelle 3 verschiebt die Pseudozufallsfolge für Zelle 8, auch ohne gemeinsamen Variablennamen. Änderungen an Gewichten oder Einstellungen direkt im Speicher beeinflussen spätere Zellen ebenfalls unbemerkt.

<!--en:78e3b674-->
- **Präfixregel und Prüfung mit `audit_outputs.py`:** Ein frisch gestarteter Kernel muss die Zellen $1\dots k-1$ ohnehin ausführen. Werden $k\dots n$ als veraltet markiert, erfasst die Regel zu 100% alle betroffenen Zellen (Abbruch bei veraltetem Text, Warnung bei veralteten Bildern).

<!--en:31153204-->
::: {.slide title="Beiträge mit Git: Fork, Branch, Bearbeitung und Pull-Request"}

<!--en:78114f58-->
[Eine Änderung vorschlagen]{.kicker}

<!--en:22fc54cd-->
- **Fork und eigener Branch:** Für Textkorrekturen von 1 Zeile das Stiftsymbol auf GitHub verwenden oder den eigenen Fork klonen und pro inhaltlich zusammenhängender Änderung einen Branch anlegen (`git switch -c fix-topic`).

<!--en:25f5b2dd-->
- **Änderungen prüfen und Pull-Request öffnen:** `chapter_*/*.md` bearbeiten, mit `git diff` prüfen, dass keine erzeugten Dateien von Hand geändert wurden, die Änderungen in den eigenen Fork hochladen und `gh pr create` ausführen.

<!--en:10519f6c-->
::: {.slide title="Quelltext prüfen, Notebooks ausführen, Ausgaben erfassen und rendern"}

<!--en:09d04c40-->
[Prüfablauf]{.kicker}

<!--en:aa3d4c70-->
- **1. Quelltextprüfung und Bibliothek (`< 100 ms`):** `python3 tools/lint_source.py --corpus <file>.md` ausführen. Die Bibliothek mit `gmake lib` neu erzeugen, falls sich ein Block mit `#@save` geändert hat.

<!--en:9c5dc9bd-->
- **2. Ausführen, erfassen und prüfen:** Für jeden Framework-Tab `gmake -B _notebooks/<fw>/<ch>/<f>.executed` und `gmake capture-outputs FILES=<ch>/<f>.md` ausführen. Anschließend mit `gmake audit-outputs && gmake html` prüfen.

<!--en:ccaaff58-->
::: {.slide title="Interaktives Bearbeiten mit der D2L-Erweiterung für VS Code"}

<!--en:efe553ed-->
[Integration in den Editor]{.kicker}

<!--en:4b89f284-->
Wer Zellen lieber interaktiv ausführt als Codeblöcke im Markdown-Quelltext zu bearbeiten, findet im Repository eine eigene Erweiterung für VS Code unter `.vscode-extension/`:

<!--en:c20e66a4-->
- **Synchronisierung in beide Richtungen:**
  - Öffnet das erzeugte `.ipynb`-Notebook des aktiven `.md`-Abschnitts und wählt den passenden Framework-Kernel vorab aus (`make kernels`).
  - Schreibt Änderungen an Code oder Text in der Notebook-Ansicht **direkt in die Quelldatei unter `chapter_*/*.md` zurück**. So bleibt keine Änderung allein in einer erzeugten `.ipynb`-Datei zurück.

<!--en:3f77e5d9-->
- **Frameworks und Folien in einer Ansicht:**
  - Direkt zwischen den Tabs für PyTorch, JAX, TensorFlow und MXNet wechseln.
  - Den Reveal.js-Foliensatz des Abschnitts neben dem Notebook ansehen.
  - Mit einem Befehl das ganze Notebook von Anfang bis Ende ausführen und die Ausgaben in `outputs/<fw>/` erfassen.

<!--en:f7c0bef5-->
::: {.slide title="Zusammenarbeit mit Programmieragenten"}

<!--en:7403071f-->
[KI-gestützte Softwareentwicklung]{.kicker}

<!--en:4b4529c1-->
Drei Eigenschaften von `d2l-neu` machen Programmieragenten bei klar umrissenen Entwicklungsaufgaben über mehrere Dateien hinweg besonders wirksam:

<!--en:a2dc8e08-->
- **1. Explizite Build- und Stilregeln:** `CLAUDE.md`, `docs/build-system.md`, `docs/writing-guide.md`, `docs/code-guide.md` und `docs/slide-guide.md` dokumentieren alle Regeln in maschinenlesbarer Form.

<!--en:19cc07c1-->
- **2. Ausführbare Prüfungen:** Ein Agent kann selbstständig `lint_source.py` aufrufen, die Notebooks aller Frameworks ausführen, `audit_outputs.py` starten und die Aufruflisten bei Fehlern untersuchen.

<!--en:1858777e-->
- **3. Regelgeleitete Übertragung zwischen Frameworks:** Der Abgleich einer Änderung zwischen PyTorch- und JAX-Tabs oder die Neuerzeugung von SVG-Abbildungen mit `tools/gen_*_figures.py` lässt sich schnell automatisieren. Menschen können das Ergebnis leicht mit `git diff` prüfen.

<!--en:c2827b2c-->
- **Zwei mögliche Fehler von Agenten, auf die jeder Mensch bei der Prüfung achten muss:**
  1. **Bearbeitung erzeugter Dateien:** Mit `git status` prüfen, ob die Änderungen in `chapter_*/*.md` und den erfassten Ausgaben unter `outputs/` stehen. In `_notebooks/` oder `d2l/*.py` dürfen keine manuellen Korrekturen stehen.
  2. **Reward-Hacking durch Abschwächen von Prüfungen:** Scheitert ein Test oder ein Notebook, lockert ein Agent womöglich einen Schwellenwert, unterdrückt eine Ausnahme oder vereinfacht das Lehrbeispiel, ohne die Ursache zu beheben. Deshalb immer prüfen, wie sich die Änderung auf den Lehrinhalt auswirkt.

<!--en:4f61215a-->
::: {.slide title="Regressionsfehler bei verrauschten Trainingsläufen finden"}

<!--en:9b43603d-->
[Informationstheoretische Bisektion]{.kicker}

<!--en:f8ff4a08-->
- **Warum 1 Lauf pro `git bisect`-Test nicht ausreicht:** Bei $64 = 2^6$ Commits und einem verrauschten Test (Erfolgswahrscheinlichkeit $p = 0.90$ bei fehlerfreien, $q = 0.30$ bei fehlerhaften Commits) verlaufen nur in $\sim 26\%$ der Fälle alle 6 binären Suchschritte richtig.

<!--en:04d9d062-->
- **Bayessche Bisektion und Fano-Schranke:** Die nach jedem Test aktualisierte A-posteriori-Verteilung $P(K = k \mid \text{runs})$ identifiziert den fehlerhaften Commit mit $\ge 95\%$ Sicherheit in $\sim 22\text{--}25$ Läufen. Das kommt der durch die Shannon-Kapazität $C$ gesetzten Grenze nahe.

<!--en:492c98d5-->
::: {.slide title="Verständnisfragen: Tabs, veraltete Ausgaben und Reihenfolge der Bibliotheksdefinitionen"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:d2ba5df3-->
Drei Fragen zum Build-System von D2L:

<!--en:73f55b09-->
1. Eine `.md`-Quelldatei enthält 5 Zellen: (1) `%%tab pytorch` definiert `x`, (2) `%%tab jax` definiert `x`, (3) `y = x * 2` ohne Markierung, (4) `%%tab pytorch` ändert `x`, (5) `print(y.sum())` ohne Markierung. Welche Notebooks ändern sich bei einer Bearbeitung von Zelle (3) und müssen erneut laufen?

<!--en:a1b5094f-->
2. Die lokale PyTorch-Umgebung wechselt von `torch==2.11.0+cu124` unter Linux auf die CPU-Version `torch==2.11.0` unter macOS. Markiert `tools/audit_outputs.py` die in Git gespeicherten PyTorch-Ausgaben als veraltet? Was geschieht bei einem Wechsel auf `torch==2.12.0`?

<!--en:7316657d-->
3. Die Methode `Trainer.prepare_batch` ist mit `#@save` in §{sec_oo-design} definiert (nur CPU). Eine neue Definition mit `#@save` in §{sec_use_gpu} verschiebt Batches auf die GPU. Welche Version steht in `d2l/torch.py`, und welche läuft am Ende des Notebooks zu §{sec_oo-design}?

<!--en:20cb6c50-->
::: {.slide title="Verständnisfragen: Tabs, veraltete Ausgaben und Reihenfolge der Bibliotheksdefinitionen (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:5d0eeb70-->
1. **Sowohl das PyTorch- als auch das JAX-Notebook.** Zelle (3) trägt keine Markierung und gehört daher zu jedem Framework, für das aus der Datei ein Notebook entsteht (`pytorch` und `jax`). Eine Änderung macht die Präfix-Fingerabdrücke der Zellen (3), (4) und (5) im PyTorch-Notebook und der Zellen (3) und (5) im JAX-Notebook ungültig.

<!--en:42bfdb4a-->
2. Beim Wechsel `2.11.0+cu124` $\to$ `2.11.0` (CPU) **gelten die Ausgaben weiterhin als aktuell**. Der Fingerabdruck lässt lokale Build-Kennungen (`+cu124`) weg, damit sich die Website auf CPU-Laptops ohne Fehlalarm erzeugen lässt. Beim Wechsel auf **`2.12.0` gelten alle PyTorch-Ausgaben als veraltet**.

<!--en:f0e1bd0b-->
3. In `d2l/torch.py` steht die **Definition aus §{sec_use_gpu} mit GPU-Unterstützung** (die letzte Definition in Buchreihenfolge gilt). Beim Start von §{sec_oo-design} enthält `d2l.Trainer.prepare_batch` die Version aus §{sec_use_gpu}. Sobald in §{sec_oo-design} die eigene `@d2l.add_to_class`-Zelle läuft, überschreibt sie diese Definition im Speicher. Für den Rest von §{sec_oo-design} gilt die CPU-Version.

<!--en:8855a2f2-->
::: {.slide title="Zusammenfassung: Mitarbeit an Dive into Deep Learning"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:4555b508-->
| Bestandteil des Build-Systems | Funktionsweise | Aufgabe bei der Mitarbeit |

<!--en:2094fec8-->
| **Eine Quelle (`chapter_*/*.md`)** | Erzeugt `.qmd`, `.ipynb`, `_slides/` und `d2l/*.py` | Nur die `.md`-Quelle bearbeiten (auch über die VS-Code-Erweiterung mit Synchronisierung), erzeugte Dateien nie von Hand ändern |

<!--en:b2f16e50-->
| **Tabs für mehrere Frameworks** | `%%tab <fw>` beschränkt Code auf bestimmte Frameworks. Unmarkierte Zellen laufen in allen Frameworks. | Nach Änderungen an einer gemeinsamen Zelle die Notebooks für *jeden* Framework-Tab der Datei erneut ausführen und die Ausgaben erfassen |

<!--en:a026d2e7-->
| **Bibliothek aus `#@save`-Blöcken** | `make lib` sammelt die `#@save`-Blöcke in Buchreihenfolge (die letzte Definition gilt) | Vor dem erneuten Ausführen der Notebooks `make lib` aufrufen, wenn sich ein `#@save`-Block geändert hat |

<!--en:586d2d43-->
| **Fingerabdrücke für Ausgaben nach der Präfixregel** | Erfasst den Hashwert des Zellpräfixes, die Framework-Version `major.minor.patch` und den Hashwert der transitiven Abhängigkeiten einschließlich der Symbole `d2l.X` selbst | `make capture-outputs` ausführen und prüfen, dass `make audit-outputs` keine Zellen mit veralteter Textausgabe meldet |

<!--en:50a4fc51-->
| **Quelltextprüfung (`lint_source.py`)** | Schnelle Prüfung ($<100\text{ ms}$) auf paarige Div- und Tab-Markierungen, gültige Zellkennungen und Querverweise | Vor jedem Commit `python3 tools/lint_source.py --corpus <file>.md` ausführen |

<!--en:39699c07-->
| **Prüfung von Git-Änderungen und Beiträgen von Programmieragenten** | Kleine Pull-Requests mit einem Branch pro Änderung und ausführbaren Prüfungen | Mit `git diff` prüfen, ob Agenten die Quellen bearbeitet haben, ohne erzeugte Dateien zu ändern oder Prüfungen abzuschwächen |
