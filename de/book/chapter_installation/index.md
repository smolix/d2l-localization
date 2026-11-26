<!--en:a04bc782-->
# Installation

<!--en:8f3faf64-->
Am meisten lernen Sie mit diesem Buch, wenn Sie den Code selbst ausführen, verändern und damit experimentieren. Jedes Kapitel steht als ausführbares Jupyter-Notebook zur Verfügung. Damit die Einrichtung auf Ihrem Rechner leichtfällt, bieten wir für jedes unterstützte Framework ein eigenständiges `.zip`-Paket an. Sie finden es oben auf der Seite im Menü **Notebooks**. Die Notebooks enthalten bereits alle gespeicherten Ausgaben und Abbildungen, sodass Sie sie sofort durchsehen können. Zum Paket gehören außerdem der passende Quellcode der Bibliothek `d2l` und [uv](https://docs.astral.sh/uv/)-Lockdateien mit festgelegten Paketversionen für CPU- und GPU-Umgebungen.

<!--en:2e4f21f8-->
Die vorberechneten Ausgaben in den Notebooks können Sie auch ohne Deep-Learning-Bibliotheken lesen. Um den Code auszuführen und zu verändern, brauchen Sie Python, JupyterLab und das Framework Ihrer Wahl. Mit den mitgelieferten `uv`-Lockdateien richten Sie mit wenigen Befehlen eine vollständige, getestete Umgebung ein.

<!--en:c640e612-->
## Wahl des Frameworks

<!--en:67f9fba9-->
Wenn Sie kein bestimmtes Framework bevorzugen, empfehlen wir den Einstieg mit **PyTorch**. Nur für PyTorch gibt es Implementierungen zu jedem Kapitel. Die Buchteile zu den Grundlagen (:numref:`chap_preliminaries` bis :numref:`chap_rnn`), zu Bildmodellen (:numref:`chap_cv`), zur Mathematik für Deep Learning (:numref:`chap_mdl-linear-algebra` bis :numref:`chap_mdl-dynamics`) und zu Werkzeugen für Deep Learning (:numref:`chap_appendix_tools`) sind in allen vier Frameworks umgesetzt (PyTorch, JAX, TensorFlow und MXNet). Der weiterführende Teil (:numref:`chap_optimization` bis :numref:`chap_diffusion`) verwendet PyTorch und JAX. Der Teil zu Sprachmodellen (ab :numref:`chap_lm_pretrained`) verwendet PyTorch.

<!--en:654dccbc-->
Für den Einstieg brauchen Sie keine spezielle Hardware. Die Beispiele der ersten Kapitel laufen problemlos auf der CPU jedes modernen Laptops. Eine GPU lohnt sich später, wenn wir größere konvolutionäre Netze, Transformer und generative Modelle trainieren.

<!--en:1be28c09-->
Die Notebook-Pakete zum Herunterladen entsprechen der Buchausgabe auf dieser Website. Wenn Sie sich die neuesten Quelldateien, Build-Skripte oder Entwicklungsumgebungen ansehen möchten, können Sie das Repository auch direkt von [GitHub](https://github.com/smolix/d2l-neu) klonen:

<!--en:0d8b8cd0-->
Zum Lesen und Ausführen der Notebooks ist das `.zip`-Paket für Ihr Framework jedoch deutlich kleiner und kommt mit wesentlich weniger Build-Abhängigkeiten aus.

<!--en:129dda44-->
## Installation von `uv`

<!--en:15759085-->
Wir verwalten mit [uv](https://docs.astral.sh/uv/getting-started/installation/) sowohl den Python-Interpreter als auch die Paketabhängigkeiten der Notebooks. Sie brauchen daher weder Conda noch eine vorhandene Python-Installation. Unter macOS oder Linux installieren Sie `uv` mit dem eigenständigen Installationsprogramm oder der Paketverwaltung Ihres Systems:

<!--en:966ef21a-->
Führen Sie unter Windows folgenden Befehl in PowerShell aus:

<!--en:42b16734-->
Falls das Installationsprogramm Ihren `PATH` aktualisiert, öffnen Sie ein neues Terminalfenster und prüfen Sie, ob `uv` verfügbar ist:

<!--en:37e592a4-->
Unsere Notebook-Umgebungen verwenden Python 3.12. Falls Python 3.12 auf Ihrem Rechner fehlt, lädt `uv` automatisch einen isolierten Interpreter herunter und verwaltet ihn.

<!--en:b554221e-->
## Notebooks herunterladen und entpacken

<!--en:10e19672-->
Laden Sie oben auf der Seite im Menü **Notebooks** das Archiv für Ihr Framework herunter und entpacken Sie es. Um zum Beispiel das PyTorch-Paket unter macOS oder Linux zu entpacken, führen Sie im Download-Verzeichnis folgende Befehle aus:

<!--en:94faaf8b-->
Unter Windows können Sie das `.zip`-Archiv mit dem Datei-Explorer oder mit PowerShell entpacken. Führen Sie alle weiteren Befehle im entpackten Ordner aus. Er enthält `pyproject.toml`, den Paketquellcode unter `d2l/` und die `uv`-Lockdateien.

<!--en:a6ea99ac-->
## CPU-Umgebung einrichten

<!--en:3f68b305-->
Wir empfehlen, mit der CPU-Umgebung zu beginnen. So können Sie die Einrichtung schnell überprüfen. Erstellen Sie im entpackten Notebook-Verzeichnis eine isolierte virtuelle Umgebung mit Python 3.12 und synchronisieren Sie sie mit den Paketversionen aus der CPU-Lockdatei:

<!--en:0ebc3709-->
Wenn Sie MXNet auf einem Mac mit Apple Silicon verwenden, nutzen Sie zur Synchronisierung die eigene Lockdatei für macOS:

<!--en:ea000e91-->
Für MXNet unter Linux verwendet `pylock.cpu.toml` das Linux-Wheel für `x86-64` aus der [von uns gepflegten Version von MXNet 2.0](https://github.com/smolix/mxnet/releases/tag/mxnet-2.0.0.zombie.1). MXNet-Binärpakete gibt es nur für Linux (`x86-64`) und macOS (Apple Silicon). Verwenden Sie unter Windows PyTorch, JAX oder TensorFlow, oder führen Sie die Notebooks in einer Cloud-Umgebung aus. Für PyTorch, JAX und TensorFlow wählt `pylock.cpu.toml` automatisch die passenden Pakete für Ihr Betriebssystem und Ihre Rechnerarchitektur.

<!--en:4d95a546-->
Starten Sie nach der Synchronisierung JupyterLab:

<!--en:28f23ec3-->
JupyterLab öffnet normalerweise automatisch einen Tab in Ihrem Standardbrowser. Falls das ausbleibt, kopieren Sie die lokale Adresse `http://localhost:...` aus dem Terminal in die Adressleiste Ihres Browsers. Mit `--no-sync` weisen Sie `uv` an, direkt in der CPU- oder GPU-Umgebung zu arbeiten, die Sie gerade mit `uv pip sync` synchronisiert haben.

<!--en:0c4d418d-->
Für den Buchteil zu Sprachmodellen (ab :numref:`chap_lm_pretrained`) enthält das PyTorch-Paket (`d2l-pytorch.zip`) außerdem `pylock.lm-cpu.toml` und `pylock.lm-gpu.toml`. Diese Dateien ergänzen PyTorch um die Hugging-Face-Bibliotheken `transformers` und `accelerate`, den OpenAI-Client sowie Bibliotheken für Retrieval und Datenkuratierung. Installieren Sie diese zusätzlichen Abhängigkeiten in einer eigenen virtuellen Umgebung namens `.venv-lm`, damit die Hauptumgebung `.venv` klein bleibt:

<!--en:ecf475a5-->
## GPU-Umgebung einrichten

<!--en:48c6eb2f-->
Mit den mitgelieferten GPU-Lockdateien können Sie die hardwarebeschleunigten Umgebungen nachbilden, in denen wir den Code für dieses Buch ausgeführt haben. Sie sind für Linux-Rechner (`x86-64`) mit einer NVIDIA-GPU ausgelegt. Prüfen Sie vor der Installation der GPU-Umgebung, ob Ihr NVIDIA-Treiber aktiv ist und erkannt wird:

<!--en:8c2464eb-->
Erstellen Sie im entpackten Notebook-Verzeichnis die virtuelle Umgebung, falls Sie das noch nicht getan haben. Synchronisieren Sie sie anschließend mit der GPU-Lockdatei:

<!--en:a65ba9af-->
Für den Wechsel von der CPU zur GPU brauchen Sie keine eigene virtuelle Umgebung. Der Befehl `uv pip sync pylock.gpu.toml` ersetzt in der vorhandenen Umgebung die CPU-Pakete durch die entsprechenden GPU-Pakete. Analog stellt `VIRTUAL_ENV=.venv-lm uv pip sync pylock.lm-gpu.toml` die Sprachmodell-Umgebung im PyTorch-Paket auf GPU-Betrieb um. Die GPU-Lockdateien legen folgende Softwarepakete für die Hardwarebeschleunigung fest:

<!--en:92d52e20-->
* **PyTorch:** Wheels für CUDA 12.8.

<!--en:17405190-->
* **JAX:** die pip-Distribution für CUDA 12.

<!--en:30a2aa80-->
* **TensorFlow:** die Distribution `and-cuda` mit den mitgelieferten CUDA-Bibliotheken.

<!--en:aab96f43-->
* **MXNet:** die gepflegte Version für CUDA 13.3 von `smolix/mxnet`.

<!--en:47e45ce5-->
Welche GPU-Software Sie brauchen, hängt von Ihrem Betriebssystem, dem Kernel-Treiber und der GPU-Architektur ab. Die Paketversionen in der Lockdatei passen daher möglicherweise nicht zu Ihrem Rechner. Falls `pylock.gpu.toml` mit Ihrer Hardware oder Ihrem Betriebssystem inkompatibel ist, lesen Sie die offizielle Installationsanleitung für [PyTorch](https://pytorch.org/get-started/locally/), [JAX](https://docs.jax.dev/en/latest/installation.html) oder [TensorFlow](https://www.tensorflow.org/install/pip). Für die ersten Kapitel können Sie in jedem Fall die CPU-Lockdatei verwenden. Für die rechenintensiveren Kapitel können Sie später zu einem gehosteten Notebook (:numref:`sec_hosted_notebooks`) oder einer GPU-Instanz in der Cloud (:numref:`sec_cloud_instances`) wechseln.

<!--en:707e075f-->
## Installation überprüfen

<!--en:6f9cdc79-->
Mit einem einzigen Befehl können Sie prüfen, ob Ihr Framework korrekt installiert ist und ob es eine GPU zur Beschleunigung erkennt:

<!--en:a63a46cd-->
Bei einer reinen CPU-Installation gibt die zweite Zeile `False` aus. Ist eine NVIDIA-GPU verfügbar, lautet die Ausgabe `True`.

<!--en:0099091a-->
Bei einer reinen CPU-Installation führt `jax.devices()` eine einzelne CPU auf. Ist eine NVIDIA-GPU verfügbar, listet die Funktion Ihre GPUs auf.

<!--en:1bb1e073-->
Bei einer reinen CPU-Installation liefert `list_physical_devices('GPU')` eine leere Liste `[]`. Ist eine NVIDIA-GPU verfügbar, enthält die Liste Ihre GPUs.

<!--en:f897c06d-->
Bei einer reinen CPU-Installation gibt `num_gpus()` den Wert `0` aus. Ist eine NVIDIA-GPU verfügbar, erscheint die Anzahl der erkannten GPUs.

<!--en:4f53fce8-->
## Standardimporte

<!--en:3befd324-->
Jedes Notebook in diesem Buch beginnt mit derselben Codezelle. Sie importiert das integrierte Python-Modul `math` für skalare Funktionen und Konstanten, NumPy unter dem üblichen Alias `np` für CPU-Arrays, das Deep-Learning-Framework selbst und das passende Backend unserer Begleitbibliothek als `d2l`:

<!--en:e45921e3-->
Hier ist `torch` das Kernpaket von PyTorch. Sein Modul `nn` stellt Schichten neuronaler Netze und Verlustfunktionen als Klassen mit eigenem Zustand bereit. Über `F` (`torch.nn.functional`) stehen die entsprechenden Operationen als zustandslose Funktionen wie `F.relu` und `F.cross_entropy` zur Verfügung.

<!--en:f0b6ed27-->
Hier stellt `jax` Transformationen für Funktionen (etwa `jax.grad` und `jax.jit`) und eine explizite Schlüsselverwaltung für den Pseudozufallszahlengenerator bereit. Über `jnp` greifen wir auf die NumPy-kompatible Array-Schnittstelle mit Hardwarebeschleunigung zu. Schichten und Modelle neuronaler Netze erstellen wir mit dem Flax-Modul `nnx`. Für das Training verwenden wir Optimierer und Lernratenpläne aus `optax`.

<!--en:037884d1-->
Hier importieren wir TensorFlow unter dem üblichen Alias `tf`. Die übergeordnete Schnittstelle für Schichten neuronaler Netze und für Modellabstraktionen liegt in `tf.keras`.

<!--en:2055c50f-->
Da das Array-Modul von MXNet die NumPy-API nachbildet, importieren wir es als `mnp`. Damit bezeichnet `np` eindeutig das Standard-NumPy. Das Modul `npx` ergänzt Funktionen für Deep Learning. Das Modul `autograd` zeichnet Berechnungsgraphen für automatisches Differenzieren auf. Die Module `gluon` und `nn` stellen Schichten für neuronale Netze bereit, `init` liefert Initialisierer für Parameter. Der Aufruf `npx.set_np()` aktiviert die NumPy-kompatible Array-Semantik von MXNet.

<!--en:db60fd12-->
Der letzte Import macht das passende Backend des Begleitpakets zum Buch unter dem einheitlichen Namen `d2l` verfügbar. Definieren und erläutern wir in einem Abschnitt eine wiederverwendbare Hilfsfunktion oder Klasse mit der Kennzeichnung `#@save`, exportieren wir sie nach `d2l`. So können die späteren Kapitel sie ohne erneute Definition aufrufen.

<!--en:86174fac-->
Damit das Buch übersichtlich bleibt, zeigen wir die Codezelle mit den Standardimporten nur hier. In den herunterladbaren und gehosteten Jupyter-Notebooks steht sie am Anfang jeder Datei, sodass jedes Notebook eigenständig läuft. In der Web- und PDF-Ausgabe beginnt dagegen jeder Abschnitt direkt mit dem Code zum jeweiligen Thema. Benötigt ein Notebook eine weitere Bibliothek (etwa Gymnasium in den Kapiteln zum bestärkenden Lernen) oder besondere Compiler-Optionen vor der Initialisierung des Frameworks, zeigen wir die zusätzliche Codezelle ausdrücklich. Für Abbildungen ist kein zusätzlicher Code nötig. Jupyter zeigt sie standardmäßig direkt im Notebook an.

<!--en:068ea902-->
## Daten, Modelle und Speicherplatz

<!--en:0663fe11-->
Das `.zip`-Archiv enthält sämtliche Notebooks, die vorberechneten Ausgaben der Codezellen und alle Abbildungen. Große Trainingsdatensätze und vortrainierte Modellgewichte laden die Notebooks beim ersten Gebrauch automatisch herunter und speichern sie in einem lokalen Cache-Verzeichnis. Sie können die Notebooks und ihre gespeicherten Ausgaben also vollständig offline lesen. Um Notebooks auszuführen, die solche Daten benötigen, brauchen Sie eine Internetverbindung und zusätzlichen Speicherplatz.

<!--en:a9822b41-->
Einige weiterführende Kapitel benötigen mehr Arbeitsspeicher oder Rechenleistung, als ein üblicher Laptop bietet. Beginnen Sie für die Grundlagenkapitel mit der CPU-Umgebung. Wechseln Sie erst zu einer lokalen GPU oder einem Cloud-Rechner, wenn die Berechnungen es erfordern. Falls Sie eigene Hardware für Deep Learning kaufen möchten, finden Sie in :numref:`sec_hardware_buyers` praktische Hinweise für die Auswahl.
