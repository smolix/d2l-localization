<!--en:f44e2bdc-->
# Speichern, Laden und vortrainierte Gewichte

<!--en:825598a2-->
Ein trainiertes neuronales Netz geht überraschend leicht verloren. Es existiert nur im Arbeitsspeicher des Prozesses, der es trainiert hat. Sobald der Prozess endet, verschwinden alle gelernten Gewichte, sofern sie nicht auf der Festplatte gespeichert sind. Das gilt für das reguläre Ende des Skripts ebenso wie für einen Fehler, ein versehentlich herausgezogenes Stromkabel am Server oder den Abbruch einer Cloud-Instanz durch den Anbieter. Bei den kleinen Beispielmodellen der vorigen Kapitel kostet uns der Verlust des Trainingsfortschritts Minuten. Bei heutigen Modellgrößen können Wochen an Rechenzeit verloren gehen, und Hardwareausfälle gehören zum Alltag. Innerhalb von 54 Tagen verzeichnete Meta beim Pre-Training von Llama 3 405B auf bis zu 16 384 GPUs 419 unerwartete Unterbrechungen, also etwa eine alle drei Stunden :cite:`Grattafiori.Dubey.Jauhri.ea.2024`. Ein Lauf dieser Größenordnung kann nur zu Ende kommen, wenn sich der Trainingsstand per Checkpointing zuverlässig sichern und das Training anschließend fortsetzen lässt.

<!--en:f165cb98-->
Ebenso häufig speichern wir Gewichte, um trainierte Modelle mit anderen zu teilen. So können diese das Modell für die Inferenz nutzen oder feinabstimmen, ohne es von Grund auf trainieren zu müssen (:numref:`sec_fine_tuning`). In beiden Fällen stellen sich dieselben praktischen Fragen. Was genau gehört auf die Festplatte? Welche Dateiformate lassen sich sicher laden? Wie prüfen wir, ob die Gewichte zu unserer Architektur passen?

<!--en:d3955fb7-->
*Was* wir speichern müssen, ergibt sich direkt aus :numref:`sec_parameters`. Ein Modell besteht aus Code und Zustand. Der Code legt die Architektur und den Vorwärtsdurchlauf fest. Der *Zustand* (engl. state) umfasst alle benannten Tensoren, die aus dem Training hervorgehen, also Parameter und Puffer. Wir serialisieren nur den Zustand als Zuordnung von Namen zu Tensoren. Um das Modell wiederherzustellen, erzeugen wir im Code eine Instanz der Architektur und laden die Tensoren wieder hinein. Um einen unterbrochenen Trainingslauf fortzusetzen, brauchen wir mehr. Zusätzlich zu den Modellgewichten speichern wir den Optimiererzustand (engl. optimizer state), etwa die gleitenden Momente von Adam, die Zustände der Zufallszahlengeneratoren, den Schrittzähler und die Konfiguration der Architektur. Dadurch ist ein Trainingscheckpoint um ein Mehrfaches größer als eine Gewichtsdatei für die Inferenz.

<!--en:33e33be8-->
In diesem Praxisteil untersuchen und serialisieren wir zunächst den Modellzustand. Dann vergleichen wir unsichere Pickle-Dateien mit dem sicheren Format `safetensors` und erstellen einen atomaren Trainingscheckpoint, mit dem sich das Training bitgenau fortsetzen lässt. Zum Schluss laden wir vortrainierte Gewichte von ResNet-50 in eine geänderte Architektur und prüfen, welche Schlüssel fehlen und welche unerwartet sind.

<!--en:032e08b7-->
## Modellzustand

<!--en:3aff9f9c-->
Ein Modell enthält Dutzende oder Hunderte von Tensoren. Um es zu speichern, brauchen wir deshalb eine Zuordnung von Namen zu Tensoren. Jedes Framework erzeugt diese Zuordnung automatisch, indem es den Modulbaum durchläuft (:numref:`sec_model_construction`). Als Schlüssel für jeden Tensor dient sein Pfad von der Wurzel aus. Wir untersuchen nun den Zustand eines mehrschichtigen Perzeptrons (engl. multilayer perceptron, MLP) mit einer einzigen versteckten Schicht aus 256 versteckten Einheiten.

<!--en:13379995-->
`collect_params()` und `state_dict()` ordnen Pfaden mit Punkten als Trennzeichen (`hidden.weight`, `output.bias`) die Parameter und Puffer zu:

<!--en:28140a0f-->
`net.weights` listet die Variablen mit ihrem jeweiligen `path` im Schichtbaum auf (`mlp/hidden/kernel`). Explizite `name=`-Argumente sorgen dafür, dass die Pfade deterministisch sind:

<!--en:7118c8dd-->
`nnx.state` liefert einen verschachtelten Zustandsbaum, dessen Schlüssel die Attributpfade im Modul sind (`hidden.kernel`, `output.bias`):

<!--en:ab27789a-->
Das Zustandswörterbuch (engl. state dictionary) enthält ausschließlich Namen, Formen und Zahlenwerte. Weder die Klasse `MLP` noch die Logik ihres Vorwärtsdurchlaufs oder ihre Aktivierungsfunktion sind darin enthalten. Zur Wiederherstellung des Modells brauchen wir also Python-Code, der genau dieselben Namen und Formen erzeugt. Schon die Umbenennung eines Attributs von `output` in `head` macht die Zuordnung unbrauchbar, obwohl das Modell dieselbe Funktion berechnet. Wie wir solche Abweichungen erkennen und ihre Ursache finden, zeigen wir später in diesem Abschnitt.

<!--en:6bd7119d-->
## Pickle und safetensors

<!--en:bc7c95d4-->
Solange Checkpoint-Dateien auf der eigenen Festplatte bleiben, genügt fast jedes Format zur Serialisierung. Sobald Sie jedoch vortrainierte Gewichte von einer öffentlichen Plattform herunterladen oder einen fremden Checkpoint übernehmen, stellt sich eine Sicherheitsfrage. Kann das Laden dieser Datei beliebigen Code auf Ihrem Rechner ausführen? Beim Standardformat von Python lautet die Antwort leider ja.

<!--en:9eba8659-->
### Codeausführung beim Laden

<!--en:fe99b532-->
Das eingebaute Python-Modul `pickle` speichert ein Bytecode-Programm zur Rekonstruktion von Python-Objekten statt passiver Daten. Beim Deserialisieren wird das Programm *ausgeführt* und kann jede importierbare Funktion aufrufen. Jede Klasse kann über `__reduce__` festlegen, was Pickle ausführt. Die Methode gibt ein aufrufbares Objekt und dessen Argumente zurück. Unser Beispiel `Tripwire` ruft unten lediglich `print` auf. Ein bösartiger Checkpoint könnte ebenso leicht `os.system` aufrufen.

<!--en:b290ae44-->
NumPy verwendet `pickle`, sobald es Python-Objekte speichert, die keine primitiven Datentypen sind. Beim Laden mit `allow_pickle=True` wird der eingebettete Code sofort ausgeführt:

<!--en:1359154e-->
`torch.save` schreibt ein Pickle-Archiv. Beim Laden mit `weights_only=False` (der Standardeinstellung vor PyTorch 2.6) wird der eingebettete Code sofort ausgeführt:

<!--en:4fdaaf4c-->
Standardmäßig weisen die Ladefunktionen beliebige Python-Objekte zurück:

<!--en:f11bf97f-->
`np.load` weist Pickle-Daten zurück, sofern nicht `allow_pickle=True` gesetzt ist.

<!--en:9fa910b8-->
`torch.load` verwendet seit PyTorch 2.6 standardmäßig `weights_only=True`.

<!--en:8c57456f-->
Eingeschränkte Pickle-Ladefunktionen bieten nur so weit Schutz, wie ihre Positivlisten fehlerfrei sind. In der Vergangenheit ließen sich selbst gehärtete Ladefunktionen durch Sicherheitslücken umgehen. Ein Dateiformat, das ausführbare Anweisungen *gar nicht kodieren kann*, beseitigt diese Angriffsfläche durch seinen Aufbau vollständig.

<!--en:99828c52-->
Der Modus `weights_only` von PyTorch schränkt das Laden auf diese Weise ein.

<!--en:a73e64ff-->
Weder die Checkpoint-Dateien von TensorFlow noch die `.weights.h5`-Dateien von Keras enthalten Pickle-Daten. Ein `.keras`-Archiv für ein vollständiges Modell kann Code in Form der Funktion einer `Lambda`-Schicht enthalten. Keras führt sie nur aus, wenn wir `safe_mode=False` übergeben.

<!--en:c4901f55-->
### Das Format safetensors

<!--en:792ab894-->
Das Format *safetensors* speichert ausschließlich rohe Tensordaten und Metadaten. Code ist ausgeschlossen. Die Binärdatei besteht aus nur drei unmittelbar aufeinanderfolgenden Abschnitten (:numref:`fig_bg_safetensors_layout`). Am Anfang steht eine 8 Byte lange Ganzzahl im Little-Endian-Format, die die Länge des Dateikopfs angibt. Es folgt der JSON-Dateikopf in UTF-8, der jedem Tensornamen den Datentyp, die Form und die Bytepositionen zuordnet. Den Abschluss bilden die lückenlos aneinandergereihten Tensorpuffer. Beim Laden wird kein Code aus der Datei ausgeführt. Die Ladefunktion validiert lediglich den JSON-Dateikopf und prüft, ob alle Bytebereiche innerhalb der Datei liegen. Dass die Bytepositionen schon zu Beginn feststehen, hat einen weiteren technischen Vorteil. Die Datei lässt sich in den Speicher einblenden (`mmap`), und auf einzelne Tensoren können wir direkt zugreifen, ohne die gesamte Datei zu lesen. Weil safetensors schnell und vom Framework unabhängig ist, hat es sich auf dem Hugging Face Hub als Standardformat für die Weitergabe von Gewichten etabliert.

<!--en:7c502263-->
![Die safetensors-Datei des folgenden MLP als Bytefolge. Auf die 8 Byte lange Längenangabe folgt der JSON-Dateikopf mit Datentyp, Form und Bytebereich jedes Tensors. Danach stehen die Rohbytes der vier Tensoren lückenlos in der Reihenfolge ihrer Namen. Die Bytepositionen unter den Daten kennzeichnen die im Dateikopf verzeichneten Bereiche.](../img/bg-safetensors-layout.svg)

<!--en:c4a6ee38-->
Wir speichern den Zustand unseres MLP in `mlp.safetensors`, laden ihn in eine neue Instanz derselben Architektur und prüfen, ob beide Instanzen identische Ausgaben liefern.

<!--en:7357e40b-->
Speichern und erneutes Laden erhalten die Daten bitgenau. Ein weiterer Vorteil ist, dass der Dateikopf als gewöhnlicher JSON-Text an einer festen Byteposition steht. Um den Dateiinhalt zu untersuchen, genügt deshalb die Python-Standardbibliothek:

<!--en:6414a2a1-->
Die Länge jedes Bytebereichs entspricht der Anzahl der Tensorelemente mal 4 Byte (`F32` = float32). Der Bias der versteckten Schicht belegt mit seinen 256 Einträgen die ersten 1024 Byte. Die Gewichtsmatrix derselben Schicht belegt mit ihren $256 \times 20 = 5{,}120$ Einträgen die nächsten 20 480 Byte. Jeder Block schließt unmittelbar an den vorherigen an. Die Daten bestehen ausschließlich aus Tensorwerten und enthalten keinen versteckten Code.

<!--en:ab229a0a-->
Für private, vorläufige Checkpoints auf dem eigenen Rechner eignen sich die Formate des jeweiligen Frameworks ohne Weiteres. Sobald Gewichte eine Vertrauensgrenze überschreiten, etwa bei der Veröffentlichung oder beim Herunterladen aus dem Web, verwenden Sie ein codefreies Format wie safetensors zusammen mit einer kryptografischen Prüfsumme. So können die Empfänger die Integrität prüfen.

<!--en:60bb8bfd-->
## Trainingscheckpoints

<!--en:0f008482-->
Eine Datei mit den Gewichten allein genügt für die Inferenz, aber *nicht*, um das Training fortzusetzen. Der Grund liegt darin, dass die Optimierung vom bisherigen Zustand abhängt. Die meisten modernen Trainingsläufe im Deep Learning verwenden *Adam* (:numref:`sec_adam`). Adam führt für jeden Parameter zwei gleitende Mittelwerte (:numref:`sec_parameters`), nämlich $\mathbf{m}$ für den Gradienten $\mathbf{g}$ und $\mathbf{v}$ für dessen elementweises Quadrat $\mathbf{g} \odot \mathbf{g}$. Jeder Parameter ändert sich entgegen der Richtung seines geglätteten Gradienten. Wo die Gradienten zuletzt groß waren, fällt die Änderung kleiner aus:

<!--en:8efaf481-->
Dabei ist $\eta$ die Lernrate und $\epsilon$ eine kleine Konstante (typischerweise $10^{-8}$). Division und Quadratwurzel wirken elementweise. Die Dächer kennzeichnen die Bias-Korrektur, deren Faktoren die Initialisierung mit null ausgleichen (Herleitung in :numref:`sec_adam`). Entscheidend ist, dass $\mathbf{m}$ und $\mathbf{v}$ den bisherigen Verlauf der Optimierung festhalten. Verwerfen wir beide, verhält sich Adam wie zu Beginn eines neuen Trainingslaufs.

<!--en:aee1de8b-->
Ein Checkpoint, mit dem wir das Training nahtlos fortsetzen können, enthält deshalb fünf Bestandteile (:numref:`fig_bg_checkpoint_contents`): (1) den Modellzustand (Parameter und Puffer), (2) den Optimiererzustand ($\mathbf{m}$ und $\mathbf{v}$), (3) die Zustände der Zufallszahlengeneratoren, damit das Mischen der Minibatches und das Dropout nahtlos weiterlaufen, (4) den Schrittzähler, der die Bias-Korrektur von Adam und einen etwaigen Lernratenplan steuert, und (5) die Modellkonfiguration, die wir zum Wiederaufbau der Architektur brauchen. Die Gradienten lassen wir weg, weil der nächste Schritt sie ohnehin neu berechnet. Von den 16 Byte pro Parameter, die Adam beim Training in fp32 im Speicher hält (:numref:`sec_parameters`), bleiben 12 Byte für Gewicht und zwei Momente. Bei einem Modell mit 8 Milliarden Parametern umfasst der Checkpoint damit 96 GB. Das ist dreimal so viel wie die 32 GB für die fp32-Gewichte und sechsmal so viel wie die 16 GB für die bf16-Inferenzdatei (:numref:`sec_numerics`).

<!--en:616348d0-->
![Die Pfeile zeigen für jeden der fünf Bestandteile einer Checkpoint-Datei, was er beim Fortsetzen wiederherstellt. Der Modellzustand liefert die Gewichte, der Optimiererzustand die beiden gleitenden Mittelwerte von Adam, der Zustand des Zufallszahlengenerators die Datenreihenfolge und das Dropout, der Schrittzähler die Position im Lernratenplan und die Konfiguration die Architektur.](../img/bg-checkpoint-contents.svg)

<!--en:770bd138-->
Checkpointing muss auch *absturzsicher* sein. Überschreiben wir einen bestehenden Checkpoint direkt, können die alte und die neue Version verloren gehen, wenn der Prozess beim Schreiben abbricht. Wir schreiben daher in eine temporäre Datei. Erst wenn sie vollständig ist, ersetzen wir die Zieldatei durch atomares Umbenennen. Wer die Datei liest, sieht so entweder den vollständigen vorigen oder den vollständigen neuen Checkpoint. Das Umbenennen schützt vor einem Prozessabsturz. Für den Schutz vor einem Stromausfall müssen wir zusätzlich `os.fsync` für die Datei und für ihr Verzeichnis aufrufen. Die Einzelheiten erarbeiten Sie in den Übungen.

<!--en:0296bfb1-->
Wir schreiben `.params` (Modell), `.states` (Pickle-Datei des Trainers, nur von einem vertrauenswürdigen Datenträger laden) und `.json` (Schritt, Konfiguration und Zustand des NumPy-Zufallszahlengenerators) zunächst als `.tmp`-Dateien. Dann ersetzen wir mit `os.replace` jede Zieldatei atomar. Für die drei Dateien als Ganzes ist das Ersetzen jedoch nicht atomar. Bei einem Absturz zwischen zwei Umbenennungen treffen neue Parameter auf einen alten Optimiererzustand. In den Übungen erstellen Sie eine Manifestdatei, die diese Lücke schließt.

<!--en:48aed0b9-->
Wir bündeln Modellzustand, Optimiererzustand, Schritt, `asdict(cfg)` und die Zustände der Zufallszahlengeneratoren für CPU, CUDA und `torch.Generator` in einem Wörterbuch. Es enthält nur Tensoren und primitive Datentypen und ist daher mit `weights_only=True` kompatibel. Wir speichern es über eine `.tmp`-Datei und `os.replace`:

<!--en:bc0d1680-->
Um das Fortsetzen des Trainings zu testen, trainieren wir ein kleines MLP für die Regression mit 64 versteckten Einheiten. Wir verwenden Adam ($\eta = 0.05$) und 256 verrauschte Beispiele für einen linearen Zusammenhang in 20 Dimensionen. Minibatches von je 64 Beispielen ziehen wir mit einem eigenen Zufallszahlengenerator mit Startwert 0. So gehört die Reihenfolge der Minibatches zum gespeicherten Zustand. Wie in :numref:`sec_model_construction` hält eine Datenklasse namens `Config` alle Hyperparameter der Architektur und des Trainings fest. Damit stehen sie direkt im Checkpoint.

<!--en:e1660af2-->
Wir trainieren 200 Schritte lang, schreiben einen Checkpoint und setzen denselben Lauf sofort für weitere 5 Schritte fort. So erhalten wir einen ununterbrochenen Referenzlauf mit 205 Schritten. Bei jedem Aufruf von `train` zeichnen wir den Verlust des Minibatches vor jedem Update auf.

<!--en:09806220-->
`tf.train.Checkpoint` bündelt Modell, Optimierer, Schrittvariable und `tf.random.Generator` in nummerierten Checkpoint-Dateien (`run-tf-1`, ...). Dazu gehört eine `.cfg.json`-Datei mit der Konfiguration. TensorFlow führt diese Operationen auf einer GPU aus, sofern eine sichtbar ist. Auf der GPU addiert es die Terme des Gradienten eines Bias in keiner festen Reihenfolge. Zwei Läufe können sich dann in den letzten Bits unterscheiden. Deshalb läuft `train` auf der CPU:

<!--en:6fbdf3c3-->
Wir bündeln die Zustandsbäume von Modell und Optimierer, das Array der Schlüssel für den Pseudozufallszahlengenerator, den Schritt und das Konfigurationswörterbuch in einem Wörterbuch. Dieses speichern wir mit Orbax (`ocp.StandardCheckpointer`) atomar:

<!--en:669a91fe-->
Nun simulieren wir einen Neustart in einem neuen Prozess. Die Funktion `resume` baut Modell und Optimierer anhand der gespeicherten Konfiguration wieder auf und stellt die Gewichte wieder her. Wahlweise stellt sie auch den Optimiererzustand und die Zustände der Zufallszahlengeneratoren wieder her. Dann führt sie die verbleibenden Schritte bis `num_steps` aus. Mit `max_diff` ermitteln wir die größte elementweise Abweichung zwischen den Gewichten zweier Modelle:

<!--en:ee5793df-->
Da `Adam` in Keras die Variablen für seine Momente erst beim ersten Update anlegt, ruft `resume` zunächst `opt.build(model.trainable_variables)` auf und stellt erst dann den Optimiererzustand wieder her.

<!--en:3341a913-->
Wir vergleichen die vollständige Wiederherstellung mit einer Variante, in der wir nur den Zustand des Zufallszahlengenerators weglassen (`restore_rng=False`, als würde das Skript erneut den Startwert `0` setzen). In beiden Fällen messen wir die maximale Abweichung der Gewichte von denen des Modells `net` nach 205 Schritten ohne Unterbrechung:

<!--en:bcc1b07f-->
Mit dem vollständigen Checkpoint erhalten wir exakt die Gewichte des ununterbrochenen Laufs, Bit für Bit. Ohne den Zustand des Zufallszahlengenerators verwendet die Trainingsschleife erneut die allerersten Minibatches, statt die nächsten zu ziehen. Dadurch unterscheiden sich die Gewichte um etwa 0.004 bis 0.02. Bitgenaue Übereinstimmung setzt außerdem deterministische Operationen voraus, die wir in :numref:`sec_repro` untersuchen.

<!--en:94f37c91-->
Weit schwerwiegender ist es, den Zustand des *Optimierers* wegzulassen. Wir setzen das Training nun mit den gespeicherten Gewichten und dem gespeicherten Zustand des Zufallszahlengenerators fort, verwenden aber einen neu initialisierten Adam-Optimierer. Für die ersten fünf Minibatches nach dem Neustart vergleichen wir die Verlustwerte beider Varianten:

<!--en:2c8b9088-->
Mit dem gespeicherten Optimiererzustand läuft das Training nahtlos weiter. Der Verlust bewegt sich um 0.01 oder darunter. Mit einem neu initialisierten Adam-Optimierer steigt der Verlust in einem *einzigen* Schritt auf etwa das Tausendfache. Selbst vier Schritte später liegt er noch mehr als hundertmal so hoch wie zuvor.

<!--en:33d95eb6-->
Warum wirft schon ein Schritt mit neu initialisiertem Adam ein beinahe konvergiertes Modell so weit zurück? Die Antwort liefert :eqref:`eq_saving_adam`. Beginnt Adam mit $\mathbf{m} = \mathbf{0}$ und $\mathbf{v} = \mathbf{0}$, kürzt die Bias-Korrektur in Schritt 1 den Initialisierungsfaktor exakt heraus. Es bleiben $\hat{\mathbf{m}} = \mathbf{g}$ und $\hat{\mathbf{v}} = \mathbf{g} \odot \mathbf{g}$. Für jeden Parameter mit einem Gradienten ungleich null beträgt die Schrittweite daher $\eta\, |g_i| / (|g_i| + \epsilon) \approx \eta = 0.05$. Er ändert sich also um die volle Lernrate, selbst bei einem winzigen Gradienten. Das gespeicherte $\mathbf{v}$ berücksichtigt dagegen auch die viel größeren quadrierten Gradienten aus früheren Trainingsphasen. Dadurch bleibt $\sqrt{\hat{\mathbf{v}}}$ im Verhältnis zu den winzigen Gradienten gegen Ende des Trainings groß, und die Updates bleiben klein. Je näher ein Modell der Konvergenz ist, desto stärker wirft es ein Neustart ohne den bisherigen Adam-Zustand zurück. Gewichte allein reichen nie aus, um das Training fortzusetzen.

<!--en:75916e7f-->
## Vortrainierte Gewichte

<!--en:322ecdc8-->
Nun kommen wir zum zweiten wichtigen Einsatz gespeicherter Zustände. Wir initialisieren ein Modell mit Gewichten, die andere bereits trainiert haben. Ein auf ImageNet (1.2 Millionen Bilder aus 1000 Kategorien) trainiertes Basisnetz (engl. backbone) lernt vielfältige visuelle Repräsentationen, die sich gut auf neue Aufgaben übertragen lassen. Für eine Aufgabe mit 10 Klassen laden wir alle Schichten bis auf den *Ausgabekopf* des Klassifikators. An seine Stelle setzen wir eine neue lineare Schicht mit 10 Ausgaben. Wir zeigen das am Beispiel von ResNet-50 (rund 25 Millionen Parameter, ohne den Ausgabekopf für 1000 Klassen sind es 23.5 Millionen). Konvolutionäre Netze untersuchen wir in :numref:`chap_cnn` und :numref:`chap_modern_cnn`. Zum Laden der Gewichte genügt es, ResNet-50 als Zustandswörterbuch mit einigen Hundert benannten Tensoren zu betrachten.

<!--en:531e0ff2-->
Externe Gewichtsdateien passen selten genau zu einem eigens angepassten Zielmodell. Wer Abweichungen stillschweigend übergeht, übersieht leicht Fehler. Um zu zeigen, wie eine Ladefunktion eine teilweise Übernahme prüft, entfernen wir zwei Gruppen von Tensoren aus dem vortrainierten Zustand von ResNet-50. Den Ausgabekopf für die Klassifikation entfernen wir, weil wir ihn *absichtlich* ersetzen. Zusätzlich entfernen wir die gesamte letzte Stufe des Basisnetzes, um einen beschädigten oder unpassenden Checkpoint zu simulieren. Wir laden die übrigen Tensoren in ein neues ResNet-50 für 10 Klassen und untersuchen zwei Mengen von Schlüsseln. *Fehlende* Schlüssel stehen für Modellparameter, die in der Datei fehlen und ihre Zufallsinitialisierung behalten. *Unerwartete* Schlüssel bezeichnen Dateieinträge, die das Modell nicht verwendet.

<!--en:a3eb0abb-->
`load_state_dict(..., strict=False)` lädt die Tensoren mit passenden Schlüsseln und gibt `(missing_keys, unexpected_keys)` direkt zurück:

<!--en:7be5dd27-->
Wir berechnen die fehlenden und die unerwarteten Schlüssel als Mengendifferenzen zwischen den Parameterpfaden des Modells und den Schlüsseln des Checkpoints:

<!--en:2a347227-->
Dass die Schlüssel für Gewicht und Bias des neuen Ausgabekopfs für 10 Klassen fehlen, ist zu erwarten. Dieser Ausgabekopf beginnt mit einer Zufallsinitialisierung. Die fehlenden Schlüssel aus der letzten Stufe des Basisnetzes sind dagegen ein Warnsignal. Hätten wir blind nur die passenden Tensoren geladen, bestünde ein großer Teil unseres „vortrainierten“ ResNet-50 unbemerkt aus Zufallsrauschen. Hier ist die Menge der unerwarteten Schlüssel leer. Enthält sie Einträge, deutet das meist auf ein umbenanntes Modul hin, dessen Gewichte nie geladen wurden. Prüfen Sie in Produktivcode immer ausdrücklich, ob die fehlenden und unerwarteten Schlüssel *nur* den Schichten entsprechen, die Sie ersetzen oder entfernen wollten. Lösen Sie bei jeder anderen Abweichung einen Fehler aus.

<!--en:49f5938e-->
Nach dieser Prüfung beginnen wir das Transferlernen (engl. transfer learning) üblicherweise mit dem *Einfrieren* des vortrainierten Basisnetzes (:numref:`subsec_freezing`). So trainieren die ersten Updates nur den neuen Ausgabekopf. Anschließend geben wir gezielt tiefere Schichten für das Training frei (:numref:`sec_fine_tuning`).

<!--en:994397f0-->
Bisher fanden alle Berechnungen in diesem Kapitel auf der CPU des Hostrechners statt, und der Zustand lag in dessen Arbeitsspeicher. In :numref:`sec_use_gpu` verlagern wir sowohl den Zustand als auch die Berechnungen auf GPUs.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:da6ca64c-->
Wenn wir ein Modell speichern, sichern wir seinen *Zustand*. Das ist ein Wörterbuch, das den Pfaden im Modulbaum die Parameter- und Puffertensoren zuordnet. Die Architektur selbst steht im Code. Da das Python-Modul `pickle` beim Laden beliebigen Code ausführen kann, sollten wir Gewichte über Vertrauensgrenzen hinweg in einem codefreien Format wie `safetensors` austauschen. Es enthält eine 8 Byte lange Längenangabe, einen JSON-Dateikopf und dicht gepackte Tensorbytes. Absturzsichere Trainingscheckpoints schreiben wir über temporäre Dateien atomar. Sie erfassen Modellzustand, Optimiererzustand, Zustand der Zufallszahlengeneratoren, Schrittzähler und Konfiguration. Mit allen fünf Bestandteilen lässt sich das Training bitgenau fortsetzen. Ohne den Zustand der Zufallszahlengeneratoren ändert sich die Datenfolge. Ohne die Momente von Adam ändert sich jeder Parameter im ersten Schritt nach dem Neustart um die volle Lernrate, und der Verlust steigt um Größenordnungen. Laden Sie vortrainierte Gewichte in eine geänderte Architektur, prüfen Sie immer, ob die Mengen der fehlenden und der unerwarteten Schlüssel Ihren Erwartungen entsprechen. Übergehen Sie Abweichungen nicht stillschweigend.

<!--en:2294243a-->
## Übungen

<!--en:41290a51-->
1. [code] **Der safetensors-Dateikopf.** Die ersten 8 Byte der safetensors-Datei des MLP geben als Ganzzahl im Little-Endian-Format die Länge des Dateikopfs an (:numref:`fig_bg_safetensors_layout`).
    1. Sagen Sie für zwei Fälle voraus, wie sich diese Länge ändert, und messen Sie anschließend nach. Verdoppeln Sie im ersten Fall die Breite der versteckten Schicht und fügen Sie im zweiten Fall eine dritte Schicht hinzu. Erklären Sie, welche Teile des Dateikopfs überhaupt von den Tensorgrößen abhängen.
    1. Lesen Sie den Bias der versteckten Schicht ohne die safetensors-Bibliothek aus der Datei. Verwenden Sie nur die Angaben im Dateikopf und springen Sie an die entsprechende Dateiposition. Dekodieren Sie die Bytes anhand des Datentyps und der Form im Dateikopf. Prüfen Sie, ob das Ergebnis mit dem von `load_file` zurückgegebenen Tensor übereinstimmt.
    1. Erklären Sie, warum sich ein einzelner Tensor aus einem Pickle-basierten Checkpoint nicht auf dieselbe Weise lesen lässt, ohne den gesamten Objektgraphen zu deserialisieren.

<!--en:48b2b6fe-->
2. [code] **Teilweises Laden.** Definieren Sie `MLP2` als Kopie von `MLP`. Fügen Sie vor der Ausgabeschicht eine zweite versteckte Schicht namens `hidden2` mit der Breite 256 ein. Benennen Sie das Attribut `output` in `head` um.
    1. Laden Sie `mlp.safetensors` in `MLP2` mit `load_dict(..., allow_missing=True, ignore_extra=True)`. Berechnen Sie die Mengen der fehlenden und der unerwarteten Schlüssel und erläutern Sie jeden Eintrag.
    1. Benennen Sie die Schlüssel der Datei vor dem Laden so um, dass sie den Namen von `MLP2` entsprechen. Prüfen Sie, ob nur `hidden2` seine neue Initialisierung behält. Sorgen Sie dafür, dass das Laden bei jedem unerwarteten Schlüssel und bei jedem fehlenden Schlüssel außerhalb von `hidden2` einen Fehler auslöst.

    *Nach Matthew Inkawhich, [Modelle speichern und laden](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), Abschnitt „Warmstarting Model Using Parameters from a Different Model“.*

<!--en:5f64d242-->
2. [code] **Teilweises Laden.** Definieren Sie `MLP2` als Kopie von `MLP`. Fügen Sie vor der Ausgabeschicht eine zweite versteckte Schicht namens `hidden2` mit der Breite 256 ein. Benennen Sie das Attribut `output` in `head` um.
    1. Laden Sie `mlp.safetensors` in `MLP2` mit `load_state_dict(..., strict=False)`. Lesen Sie den zurückgegebenen Bericht und erläutern Sie jeden fehlenden und jeden unerwarteten Schlüssel.
    1. Benennen Sie die Schlüssel der Datei vor dem Laden so um, dass sie den Namen von `MLP2` entsprechen. Prüfen Sie, ob nur `hidden2` seine neue Initialisierung behält. Sorgen Sie dafür, dass das Laden bei jedem unerwarteten Schlüssel und bei jedem fehlenden Schlüssel außerhalb von `hidden2` einen Fehler auslöst.

    *Nach Matthew Inkawhich, [Modelle speichern und laden](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), Abschnitt „Warmstarting Model Using Parameters from a Different Model“.*

<!--en:5512be56-->
2. [code] **Teilweises Laden.** Definieren Sie `MLP2` als Kopie von `MLP`. Fügen Sie vor der Ausgabeschicht eine zweite versteckte Schicht namens `hidden2` mit der Breite 256 ein. Nennen Sie die Ausgabeschicht `head` statt `output`.
    1. Laden Sie `mlp.safetensors` in `MLP2`, indem Sie jeder Variablen, deren `path` in der Datei vorkommt, ihren Wert aus der Datei zuweisen. Listen Sie die Variablen auf, die ihre neue Initialisierung behalten, sowie die Dateieinträge, zu denen es keine Variable gibt. Erläutern Sie jeden Fall.
    1. Benennen Sie die Schlüssel der Datei vor dem Zuweisen so um, dass sie den Pfaden von `MLP2` entsprechen. Prüfen Sie, ob nur `hidden2` seine neue Initialisierung behält. Sorgen Sie dafür, dass das Laden bei jedem unerwarteten Schlüssel und bei jedem fehlenden Pfad außerhalb von `hidden2` einen Fehler auslöst.

    *Nach Matthew Inkawhich, [Modelle speichern und laden](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), Abschnitt „Warmstarting Model Using Parameters from a Different Model“.*

<!--en:9e06d3d0-->
2. [code] **Teilweises Laden.** Definieren Sie `MLP2` als Kopie von `MLP`. Fügen Sie vor der Ausgabeschicht eine zweite versteckte Schicht namens `hidden2` mit der Breite 256 ein. Benennen Sie das Attribut `output` in `head` um.
    1. Laden Sie `mlp.safetensors` in `MLP2`, indem Sie die Zustände anhand ihrer Schlüsselmengen zusammenführen, wie beim Laden des beschädigten Zustands von ResNet-50. Geben Sie die Mengen der fehlenden und der unerwarteten Schlüssel an und erläutern Sie jeden Eintrag.
    1. Benennen Sie die Schlüssel der Datei vor dem Zusammenführen so um, dass sie den Namen von `MLP2` entsprechen. Prüfen Sie, ob nur `hidden2` seine neue Initialisierung behält. Sorgen Sie dafür, dass das Zusammenführen bei jedem unerwarteten Schlüssel und bei jedem fehlenden Schlüssel außerhalb von `hidden2` einen Fehler auslöst.

    *Nach Matthew Inkawhich, [Modelle speichern und laden](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), Abschnitt „Warmstarting Model Using Parameters from a Different Model“.*

<!--en:d67c819b-->
3. [code] **Speichern und Laden mit Datentypwechsel.** ● Eine `float16`-Zahl speichert 5 Exponentenbits und 10 Mantissenbits, eine `float32`-Zahl dagegen 8 und 23 (:numref:`sec_numerics`). Wandeln Sie die Parameter eines `MLP` in `float16` um, speichern Sie sie und laden Sie sie anschließend in ein `float32`-Modell der Klasse `MLP`.
    1. Sagen Sie anhand der Mantissenbreite den größten relativen Fehler eines wiederhergestellten Parameters voraus. Messen Sie die maximale absolute Differenz zwischen den wiederhergestellten und den ursprünglichen Werten. Messen Sie für die von null verschiedenen ursprünglichen Parameter außerdem die maximale relative Differenz. Erklären Sie für jeden Parameter, dessen relativer Fehler die Vorhersage übersteigt, wie es dazu kommt.
    1. Vergleichen Sie die Ausgaben des ursprünglichen und des wiederhergestellten `MLP` für `X`. Entscheiden Sie, ob die Abweichungen beim Speichern und Laden für die Inferenz akzeptabel sind.
    1. Adam führt einen gleitenden Mittelwert der quadrierten Gradienten. Berechnen Sie, was `float16` für das Quadrat einer Gradientenkomponente von $10^{-4}$ speichert. Erklären Sie, warum ein Checkpoint zum Fortsetzen des Trainings den Optimiererzustand in `float32` hält.

<!--en:3b585cd8-->
4. [extended] **Absturzsichere Manifestdatei.** ●
    1. Schreiben Sie ein Hilfsprogramm, das Modellzustand, Optimiererzustand und eine JSON-Datei mit Metadaten unter einem neuen Präfix mit der Schrittnummer speichert. Veröffentlichen Sie die drei Dateien gemeinsam, indem Sie eine einzige Manifestdatei an ihren Zielpfad umbenennen. Sie soll für jede Datei den Namen, die Größe und die Prüfsumme enthalten.
    1. Simulieren Sie zwei Fehlerfälle. Im ersten Fall stürzt das Programm ab, nachdem es die drei Dateien geschrieben, aber bevor es die Manifestdatei umbenannt hat. Im zweiten Fall wird eine Datei eines bereits veröffentlichten Checkpoints abgeschnitten. Zeigen Sie, dass das Laden im ersten Fall das Training vom vorigen vollständigen Checkpoint aus fortsetzt und im zweiten Fall den beschädigten Checkpoint zurückweist.
    1. Das Umbenennen ist atomar. Wer die Manifestdatei liest, sieht also entweder die alte oder die neue Fassung. POSIX verlangt keine Garantie dafür, dass die Daten der umbenannten Dateien vor der Umbenennung die Festplatte erreichen. `os.fsync` erzwingt, dass gepufferte Schreibvorgänge für eine Datei oder für die Einträge eines Verzeichnisses auf die Festplatte gelangen. Geben Sie an, wo das Hilfsprogramm `os.fsync` aufrufen muss, damit bei einem Stromausfall entweder der vorige oder der neue Checkpoint intakt bleibt. Begründen Sie jeden Aufruf.

<!--en:d451fd99-->
[Diskussion](https://d2l.discourse.group/t/36757)

<!--en:9571056e-->
Dateiein- und -ausgabe, Checkpoints und Serialisierung<br>**Zustandswörterbücher · Codeausführung bei Pickle und Schutz durch safetensors · atomare Checkpoints aus 5 Teilen · Schlüsselprüfung bei vortrainierten Gewichten**

<!--en:2b23de38-->
::: {.slide title="Warum Checkpoints und sichere Dateiformate unerlässlich sind"}

<!--en:64c35049-->
[Fehlertoleranz und Sicherheit]{.kicker}

<!--en:ce2766da-->
- **Hardwareausfälle gehören zum Alltag:** In 54 Tagen Pre-Training von Llama 3 405B auf $16{,}384$ GPUs verzeichnete Meta **419 unerwartete Unterbrechungen** (etwa 1 Absturz alle 3 Stunden).

<!--en:a84f208e-->
- **Zum Fortsetzen reichen Gewichte allein nicht aus:** Startet Adam nur mit den gespeicherten Gewichten neu, steigt der Verlust schon beim ersten Update um den Faktor $>100\times$.

<!--en:4ce0a5b0-->
- **Beliebige Codeausführung beim Laden:** Ein heruntergeladener `pickle`-Checkpoint aus fremder Quelle kann beim Aufruf von `load` schädliche Shell-Befehle ausführen.

<!--en:d469ca67-->
::: {.slide title="Serialisierung des Zustandsbaums ohne Python-Klassencode"}

<!--en:0abf09b3-->
[Zustand und Architektur]{.kicker}

<!--en:5b00ad35-->
- Das Zustandswörterbuch speichert nur `(path, shape, dtype, raw_bytes)` und enthält keine Informationen über die Klasse `MLP`:

<!--en:1c2d1117-->
::: {.slide title="Sicherheitsrisiko bei Python: Pickle führt beim Laden Code aus"}

<!--en:8f5f7a39-->
[Warum die Weitergabe per Pickle unsicher ist]{.kicker}

<!--en:beb60584-->
- Hinter dem Python-Format `pickle` steht eine stapelbasierte virtuelle Maschine. Sie führt beliebige aufrufbare Objekte aus, die `__reduce__` zurückgibt:

<!--en:144332c5-->
::: {.slide title="Gehärtete Ladefunktionen weisen beliebige Objekte standardmäßig zurück"}

<!--en:191f1e70-->
[Pickle-Ladefunktionen mit Positivliste]{.kicker}

<!--en:c74d4c03-->
- Gehärtete Ladefunktionen weisen beliebige Pickle-Objekte zurück. Bei `safetensors` entfällt die Angriffsfläche des Parsers für Befehlscodes vollständig:

<!--en:b0a0df2f-->
::: {.slide title="safetensors: kein Code, Speicherabbildung mit mmap ohne Kopie"}

<!--en:0c06a5ad-->
[Sicherer Austausch von Gewichten]{.kicker}

<!--en:6bcf0529-->
Eine `safetensors`-Datei enthält passive Metadaten und rohe Tensorbytes in drei flachen Abschnitten:

<!--en:db1d6a0b-->
1. **Vorzeichenlose 8-Byte-Ganzzahl im Little-Endian-Format (`<Q`):** exakte Länge $n$ des JSON-Dateikopfs in Byte.

<!--en:9bd849ac-->
2. **JSON-Dateikopf in UTF-8 ($n$ Byte):** Ordnet jedem Tensorpfad `dtype`, `shape` und die Bytepositionen `[begin, end]` relativ zum Datenabschnitt zu.

<!--en:9ad7e36f-->
3. **Lückenlos gepackte Tensorbytes:** Die Array-Puffer liegen unmittelbar hintereinander. Das ermöglicht die sofortige **Speicherabbildung ohne Kopie (`mmap`)** und den direkten Zugriff auf einzelne Tensoren, ohne die ganze Datei zu lesen.

<!--en:47081d2b-->
::: {.slide title="Speichern und byteweises Prüfen einer safetensors-Datei"}

<!--en:71fd7eb3-->
[safetensors speichern, laden und den Dateikopf prüfen]{.kicker}

<!--en:7a24503e-->
- MLP in `mlp.safetensors` speichern, bitgenaue Wiederherstellung prüfen und Dateikopf mit `struct` und `json` aus der Standardbibliothek auslesen

<!--en:87dcc2cb-->
::: {.slide title="Speichern und Wiederherstellen einer safetensors-Datei"}

<!--en:71fd7eb3-2-->
[safetensors speichern, laden und den Dateikopf prüfen]{.kicker}

<!--en:5ac08213-->
::: {.slide title="Den safetensors-Dateikopf von Hand auslesen"}

<!--en:71fd7eb3-3-->
[safetensors speichern, laden und den Dateikopf prüfen]{.kicker}

<!--en:317f794e-->
::: {.slide title="Warum der Optimiererzustand zum Fortsetzen des Trainings nötig ist"}

<!--en:12634e4f-->
[Zustandsbehaftete Optimierung mit Adam]{.kicker}

<!--en:92397b29-->
- **Warum der Verlust bei neu initialisiertem Adam hochschnellt:** Das Verwerfen von $\mathbf{m}$ und $\mathbf{v}$ beim Fortsetzen ($\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$) erzwingt bei der Bias-Korrektur in Schritt 1 $\hat{\mathbf{m}}_1 = \mathbf{g}$ und $\hat{\mathbf{v}}_1 = \mathbf{g} \odot \mathbf{g}$.

<!--en:90cd26db-->
- **Sprung mit voller Schrittweite:** Jeder Parameter ändert sich um $\Delta w_i = -\eta \, \frac{g_i}{|g_i| + \epsilon} \approx \pm \eta$. Damit verlässt das Modell das Einzugsgebiet des erreichten Minimums.

<!--en:fe9530ee-->
::: {.slide title="Die fünf Bestandteile eines Checkpoints zum Fortsetzen des Trainings"}

<!--en:0b1e8e73-->
[Aufbau eines vollständigen Checkpoints]{.kicker}

<!--en:df68c80d-->
- **Fünf nötige Bestandteile:** Modellzustand (Parameter und Puffer), Optimiererzustand ($\mathbf{m}, \mathbf{v}$), Zustände der Pseudozufallszahlengeneratoren (CPU, GPU, Datenlader), Schrittzähler $t$ und Konfiguration der Architektur.

<!--en:247646e6-->
- **Absturzsicheres atomares Schreiben:** Immer in eine temporäre Datei schreiben (`path + '.tmp'`) und dann `os.replace(tmp, path)` aufrufen. So beschädigt ein Absturz beim Schreiben nie den vorigen Checkpoint.

<!--en:ab17013a-->
::: {.slide title="Speicherbedarf eines Checkpoints für 8 Milliarden Parameter: 96 GB"}

<!--en:f68d42c2-->
[Speicherbedarf eines Checkpoints berechnen]{.kicker}

<!--en:9226523a-->
- **Inferenzdatei (`16 GB` in `bf16`):** Enthält bei $2\text{ B/param}$ nur die Modellgewichte $\mathbf{w}$.

<!--en:46971594-->
- **Checkpoint zum Fortsetzen des Trainings (`96 GB` in `fp32`):** Gewichte $\mathbf{w}$ (`32 GB`) $+$ Adam-Moment $\mathbf{m}$ (`32 GB`) $+$ Adam-Moment $\mathbf{v}$ (`32 GB`) $= 12\text{ B/param}$ ($6\times$ so groß wie die Inferenzdatei, die Gradienten $\mathbf{g}$ werden nicht mitgespeichert).

<!--en:e8b40835-->
::: {.slide title="Hilfsfunktionen für atomares Speichern und Laden implementieren" only="pytorch,mxnet"}

<!--en:b8943190-->
[Absturzsicheres Checkpointing im Code]{.kicker}

<!--en:007b6414-->
- Die Hilfsfunktion `save_checkpoint` bündelt alle fünf Bestandteile und speichert sie mit `os.replace` atomar.

<!--en:72fb02b4-->
::: {.slide title="Hilfsfunktionen für atomares Speichern und Laden im Code" only="pytorch,mxnet"}

<!--en:b8943190-2-->
[Absturzsicheres Checkpointing im Code]{.kicker}

<!--en:8705c017-->
::: {.slide title="Bitgenaues Fortsetzen nach 200 Schritten prüfen"}

<!--en:dbb2c465-->
[Experimentelle Überprüfung]{.kicker}

<!--en:2dec6088-->
- Nach Wiederherstellung aller fünf Bestandteile stimmen die Schritte 201–205 bis aufs letzte Bit überein (Abweichung `0.0`). Ohne den Zustand des Zufallszahlengenerators wiederholen sich frühe Minibatches:

<!--en:06532e50-->
::: {.slide title="Ohne Optimiererzustand geraten fast konvergierte Modelle aus dem Tritt"}

<!--en:0e67dda9-->
[Verlustsprung bei neu initialisiertem Adam]{.kicker}

<!--en:bf8f9e15-->
- Beim Fortsetzen nach Schritt 200 mit gespeicherten Gewichten, gespeichertem Zustand des Zufallszahlengenerators und einem **neu initialisierten Adam-Optimierer** steigt der Verlust schon nach dem ersten Update um etwa drei Größenordnungen:

<!--en:8b6a2f39-->
::: {.slide title="Vortrainierte Gewichte des Basisnetzes für einen neuen Ausgabekopf laden"}

<!--en:38ef4284-->
[Vorbereitung auf das Transferlernen]{.kicker}

<!--en:7cd6e837-->
Um ein auf ImageNet vortrainiertes ResNet-50 ($\sim 25\text{M}$ Parameter, $1{,}000$ Klassen) an einen Datensatz mit 10 Klassen anzupassen, das vortrainierte Basisnetz beibehalten und den abschließenden linearen Ausgabekopf ersetzen (`2048 -> 10`):

<!--en:2f8523ea-->
::: {.slide title="Fehlende und unerwartete Schlüssel beim teilweisen Laden prüfen"}

<!--en:ad5beb54-->
[Verborgene Abweichungen im Checkpoint erkennen]{.kicker}

<!--en:988b4ab8-->
- Beim Laden externer Gewichte in eine geänderte Architektur mit erlaubten Abweichungen immer die fehlenden und unerwarteten Schlüssel prüfen

<!--en:1b2fa6cd-->
::: {.slide title="Den Schlüsselbericht nach teilweisem Laden auswerten"}

<!--en:ad5beb54-2-->
[Verborgene Abweichungen im Checkpoint erkennen]{.kicker}

<!--en:07c815e9-->
::: {.slide title="Verständnisfragen zur Serialisierung und zum Fortsetzen ab einem Checkpoint"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:e8025b20-->
Drei Fragen zum Checkpointing:

<!--en:72bbc623-->
1. Ein MLP ($20 \to 256 \to 10$, `float32`) wird in `mlp.safetensors` gespeichert. Der JSON-Dateikopf sei $n = 296$ Byte lang (die genaue Länge hängt davon ab, welche Tensornamen das Framework schreibt). Wie viele Byte umfasst die gesamte Datei genau?

<!--en:f2885448-->
2. Das Regressionsmodell $20 \to 64 \to 1$ hat $20 \times 64 + 64 + 64 \times 1 + 1 = 1{,}409$ Parameter. Das Training wird bei $\eta = 0.05$ mit einem neu initialisierten Adam-Optimierer ($\mathbf{m}=\mathbf{v}=\mathbf{0}$) fortgesetzt. Jede Gradientenkomponente ist ungleich null. Wie groß ist die euklidische Norm $\|\Delta \mathbf{w}\|_2$ des allerersten Parameterupdates?

<!--en:b7b4e128-->
3. In ResNet-50 wird `self.fc` in `self.classifier` umbenannt und ein ursprünglicher vortrainierter Checkpoint mit `strict=False` geladen. Was steht in `missing_keys` und `unexpected_keys`?

<!--en:557b98b6-->
**Lösungen:**

<!--en:31be5ef7-->
1. **$8 + 296 + 4 \times 7{,}946 = 32{,}088$ Byte:** 8 Byte für die Längenangabe $+$ 296 Byte für den JSON-Dateikopf $+$ $31{,}784$ Byte für die $7{,}946$ Parameter in `float32`.

<!--en:4bd3b3b7-->
2. **$\|\Delta \mathbf{w}\|_2 \approx 0.05 \sqrt{1{,}409} \approx 1.88$:** Jede Koordinate ändert sich um $\pm \eta = \pm 0.05$, gleichgültig, wie klein $\|\mathbf{g}\|_2$ ist.

<!--en:977ea09b-->
3. **`missing_keys` enthält `classifier.weight, classifier.bias` (behalten ihre Zufallsinitialisierung). `unexpected_keys` enthält `fc.weight, fc.bias` (Einträge der Datei, die beim Laden übergangen werden).**

<!--en:a83bda2d-->
::: {.slide title="Zusammenfassung: Speichern, Laden und vortrainierte Gewichte"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:0d7345e7-->
- **Zustandswörterbücher und `safetensors`:** Tensordaten mit dem Pfad im Modulbaum als Schlüssel speichern. Mit `safetensors` (8 Byte Längenangabe + JSON + Rohbytes) lassen sich Gewichte ohne Risiko einer Codeausführung aus der Datei weitergeben und mit `mmap` ohne Kopie einblenden.

<!--en:01cde870-->
- **Atomare Checkpoints aus 5 Teilen:** Modellzustand, Optimierermomente $\mathbf{m}, \mathbf{v}$, Zustände der Pseudozufallszahlengeneratoren, Schritt $t$ und Konfiguration ($12\text{ B/param} = 96\text{ GB}$ bei 8 Mrd. Parametern) über `.tmp` und `os.replace` speichern.

<!--en:89ede936-->
- **Schlüssel vortrainierter Gewichte prüfen:** Sicherstellen, dass `missing_keys` und `unexpected_keys` *nur* die absichtlich ersetzten Schichten betreffen.
