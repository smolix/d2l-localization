<!--en:24d92728-->
# Parameter, Zustand und Speicher

<!--en:7cb5c687-->
Im vorigen Abschnitt haben wir Modelle als Modulbäume organisiert, damit Frameworks jeden Parameter automatisch finden. Bisher konnten wir die Verwaltung dieser Tensoren dem Framework überlassen. Was ein Modell sonst noch speichert und wie viel Speicher das Training braucht, spielte dabei kaum eine Rolle. Bei kleinen Modellen fallen diese Details selten ins Gewicht. Bei großen Modellen bestimmen sie jede technische Entscheidung. Betrachten wir ein Modell mit 8 Milliarden Parametern wie Llama 3 aus :numref:`sec_model_construction`. Seine Gewichte belegen als 32-Bit-Gleitkommazahlen 32 GB. Das *Training* benötigt schon vor dem ersten Beispiel viermal so viel Speicher (128 GB). Zu jedem Parameter gehört ein Gradient. Außerdem benötigt der Adam-Optimierer (:numref:`sec_adam`) zwei weitere Tensoren derselben Größe für die gleitenden Mittelwerte der Gradienten und ihrer Quadrate. Das übersteigt bereits die Kapazität einer GPU mit 80 GB.

<!--en:dbc26eb2-->
Um den Speicherbedarf genau zu berechnen, müssen wir jeden Tensor des Modells und seine Funktion kennen. Auch alltägliche Aufgaben setzen diesen Überblick voraus. Checkpoints müssen jeden persistenten Tensor speichern, den der Vorwärtsdurchlauf benötigt. Beim Übertragen auf die GPU müssen alle diese Tensoren gemeinsam auf das Gerät gelangen. Beim Fine-Tuning müssen wir genau festlegen, welche Tensoren der Optimierer aktualisieren darf.

<!--en:34b9b7d8-->
Frameworks nutzen für all diese Vorgänge die *Registrierung* (:numref:`sec_model_construction`). Jede Schicht trägt ihre Tensoren unter strukturierten Namen in den Modulbaum ein. So können der Optimierer und die Routinen für das Speichern von Checkpoints und den Gerätewechsel denselben Baum durchlaufen. Wir untersuchen diesen Zustandsbaum (engl. state tree) in drei Schritten. Zuerst unterscheiden wir trainierbare *Parameter* von nicht trainierbaren *Puffern* (engl. buffers). Dann berechnen wir den Speicherbedarf beim Training in Byte pro Parameter und zeigen, wie wir Parameter zählen können, ohne Speicher für sie zu reservieren. Zum Schluss betrachten wir zwei Verfahren, die bei der Sprachmodellierung und beim Transferlernen häufig zum Einsatz kommen: die Gewichtsbindung (engl. weight tying) und das Einfrieren von Parametern (engl. parameter freezing).

<!--en:330a3214-->
## Der Zustandsbaum

<!--en:311b5f0f-->
### Zugriff über den Pfad

<!--en:02979e9b-->
Als durchgängiges Beispiel dient uns ein kompaktes Residual-MLP. Seine Eingabeschicht projiziert 20 Merkmale auf 64 versteckte Einheiten. Darauf folgen zwei Residualblöcke der Breite 64 und eine Ausgabeschicht mit 10 Einheiten. Damit das Notebook eigenständig ausführbar ist, definieren wir `ResidualBlock` aus :numref:`sec_model_construction` hier erneut. Jeder Block addiert die Ausgabe seines zweischichtigen *Hauptzweigs* direkt zur Eingabe.

<!--en:3564ac78-->
Die Parameter bilden die Blätter des Modulbaums. Der durch Punkte gegliederte Pfad zu einem Blatt (:numref:`fig_blocks`) entspricht dem Zugriff in Python. So können wir bei der Fehlersuche gezielt eine Schicht untersuchen. Für die Optimierung und das Speichern von Checkpoints brauchen wir dagegen *alle* Blätter auf einmal. Deshalb bietet jedes Framework eine Möglichkeit, den gesamten Baum zu durchlaufen und jeden Parameter mit seinem Pfad abzurufen.

<!--en:bf2c2cc9-->
In Gluon liefert `collect_params()` ein Wörterbuch mit den Pfaden als Schlüsseln. Jeder Wert ist ein `Parameter`. Dieses Objekt fasst das Array `.data()` und seinen Gradienten `.grad()` zusammen. Der Gradient enthält nur Nullen, bis die Backpropagation Werte einträgt.

<!--en:97108842-->
In PyTorch verwenden wir `named_parameters()`. Jeder Parameter ist ein `nn.Parameter`, also ein Tensor, der sich selbst beim Modulsystem als trainierbar registriert. Sein Gradient `.grad` hat den Wert `None`, bis die Backpropagation ihn berechnet.

<!--en:e34e1123-->
In Keras verwenden wir `net.weights`, eine Liste von Variablen, die jeweils ihren Pfad enthalten. Die Variablen speichern ihre Gradienten nicht. Ein `tf.GradientTape` gibt die Gradienten in einer zweiten Liste in derselben Reihenfolge zurück.

<!--en:9bc10387-->
In NNX verwenden wir `nnx.state(net, nnx.Param)`. In der flachen Ansicht erhalten wir jedes Array mit seinem Pfad. Die Parameter speichern ihre Gradienten nicht. `nnx.grad` gibt sie als zweiten Zustandsbaum mit denselben Pfaden zurück.

<!--en:86982b1a-->
Unser Residual-MLP enthält insgesamt 12 Parametertensoren. Wir sehen uns die Gesamtzahl und die vier Tensoren des ersten Residualblocks an:

<!--en:43c38c68-->
Ein Name wie `1.body.0.weight` beschreibt den Pfad vom Kind `1` von `net` (dem ersten Residualblock) über `body` und dessen erste Schicht `0` bis zu `weight` (`net[1].body[0].weight`). In Gluon kennzeichnet `grad_req == 'write'` trainierbare Parameter. In PyTorch stimmen die Schlüssel von `state_dict()` mit den Namen aus `named_parameters()` überein, wenn das Modell keine Puffer hat:

<!--en:cfd24575-->
Ein Pfad wie `sequential_2/residual_block/sequential/dense_1/kernel` führt über die Schichtnamen bis zu `kernel` (`net.layers[1].body.layers[0].kernel`). Standardmäßig enthalten Schichtnamen einen Zähler, der beim Anlegen neuer Schichten weiterläuft. Geben Sie deshalb `name=` explizit an, wenn die Pfade in verschiedenen Skripten übereinstimmen müssen. Ohne Puffer stimmt `net.weights` mit `net.trainable_weights` überein:

<!--en:5b687299-->
Ein Pfadtupel wie `('layers', 1, 'body', 'layers', 0, 'kernel')` entspricht dem Zugriff über `net.layers[1].body.layers[0].kernel`. Gibt es ausschließlich trainierbare Parameter, stimmt `nnx.state(net)` mit `nnx.state(net, nnx.Param)` überein:

<!--en:1e3501e8-->
Alle Hilfsfunktionen des Frameworks durchlaufen denselben Zustandsbaum entlang dieser Pfade. Die Gleichheit oben gilt nur, weil unser Residual-MLP ausschließlich trainierbare Tensoren enthält. Das trifft nicht auf alle Schichten zu.

<!--en:3cef9221-->
### Parameter und Puffer

<!--en:d6a5287e-->
Angenommen, ein Modell standardisiert seine Eingaben anhand von Mittelwert und Standardabweichung jedes Merkmals. Beide Größen berechnen wir einmal aus dem Trainingsdatensatz. Checkpoints müssen diese Statistiken speichern, denn ohne sie wären die Vorhersagen bei der Inferenz falsch. Auch beim Übertragen der Gewichte auf die GPU müssen die Statistiken mitkommen. Der Optimierer darf sie jedoch *niemals* durch Gradientenabstieg aktualisieren, denn sie beschreiben die Daten und sind keine gelernten Gewichte. Gewöhnliche Python-Attribute eignen sich dafür nicht, weil das Framework sie beim Durchlaufen des Modulbaums überspringt (:numref:`sec_model_construction`). Die Batchnormalisierung (:numref:`sec_batch_norm`, :cite:`Ioffe.Szegedy.2015`) nutzt genau diese Art von Zustand. Sie aktualisiert gleitende Mittelwerte und Varianzen bei den Vorwärtsdurchläufen im Training und verwendet sie bei der Inferenz.

<!--en:506c29b7-->
Solche persistenten, nicht trainierbaren Tensoren heißen *Puffer*. In jedem Framework können wir sie so registrieren, dass Checkpoints und Gerätewechsel sie berücksichtigen und der Optimierer sie überspringt. Für jeden Tensor einer Schicht gilt damit eine von drei Regeln. Soll der Optimierer ihn aktualisieren, registrieren wir ihn als *Parameter*. Brauchen wir ihn für Checkpoints und Gerätewechsel, aber ohne Gradienten, registrieren wir ihn als *Puffer*. Wird er nur für einen Vorwärts- und Rückwärtsdurchlauf benötigt, genügt eine gewöhnliche lokale Variable. Dazu gehören etwa die für die Backpropagation zwischengespeicherten Aktivierungen aus :numref:`sec_backprop`. Konstanten der Architektur wie Schichtbreiten bleiben gewöhnliche Python-Attribute. :numref:`tab_bg_tensor_kinds` fasst die Regeln zum Nachschlagen zusammen.

<!--en:b2e86ac2-->
:Die vier Arten von Werten in einer Schicht und ihre Behandlung bei der Optimierung, beim Speichern von Checkpoints und beim Gerätewechsel.

<!--en:60ea31b7-->
| Art | Beispiel | Vom Optimierer aktualisiert | In Checkpoints gespeichert | Mit dem Modell übertragen |

<!--en:4fbfdd09-->
| Parameter | Gewichte, Bias-Terme | ja | ja | ja |

<!--en:0459e847-->
| Puffer | Eingabestatistiken, gleitende Mittelwerte der Batchnormalisierung | nein | ja | ja |

<!--en:47a5e7b8-->
| Gewöhnliches Attribut | Schichtbreiten | nein | nein | nein |

<!--en:cc581807-->
| Lokale Variable | Zwischenaktivierungen | nein | nein | nein |

<!--en:c8a72812-->
Zur Veranschaulichung bauen wir ein Modul namens `Standardizer`. Es normalisiert für jedes Merkmal die Eingabe mit dem vorberechneten Mittelwert und der vorberechneten Standardabweichung. Erst dann wendet es eine lineare Schicht an.

<!--en:406e90ea-->
`gluon.Constant` registriert einen `Parameter` mit `grad_req='null'`.

<!--en:e22e9762-->
`self.register_buffer(name, tensor)` registriert einen nicht trainierbaren Puffer.

<!--en:0341451a-->
`self.add_weight(..., trainable=False)` registriert eine nicht trainierbare Variable.

<!--en:e716533b-->
Mit einer Unterklasse von `nnx.Variable` (hier `class Buffer(nnx.Variable)`) lassen sich Puffer von `nnx.Param` unterscheiden.

<!--en:a1f62349-->
Vergleichen wir den trainierbaren und den nicht trainierbaren Zustand, sehen wir, dass Checkpoints alle vier Tensoren erfassen, der Optimierer dagegen nur Gewicht und Bias der linearen Schicht:

<!--en:6a84e3c8-->
Die Registrierung bestimmt auch, wie Operationen auf dem gesamten Modell wirken. Bei der Umwandlung von Datentypen (:numref:`sec_numerics`) und der Übertragung auf die GPU (:numref:`sec_use_gpu`) durchläuft das Framework denselben Zustandsbaum. Es berücksichtigt Parameter und Puffer gleichermaßen und überspringt gewöhnliche, nicht registrierte Attribute.

<!--en:bb77a0cb-->
## Speicherbedarf beim Training

<!--en:bfae2228-->
### Byte pro Parameter

<!--en:d092542b-->
Wie viel Speicher brauchen wir für das Training unseres Residual-MLP? Betrachten wir nur den Modell- und den Optimiererzustand, so hängt der Speicherbedarf von drei Größen pro Parameter ab: dem Parameter selbst, seinem Gradienten und dem Zustand, den der Optimierer für ihn speichert. Der einfache stochastische Gradientenabstieg mit Minibatches (SGD) speichert keinen eigenen Zustand. Mit 32-Bit-Gleitkommazahlen (fp32, 4 Byte pro Zahl) benötigt er daher 8 Byte pro Parameter. Bei Adam kommen pro Parameter zwei fp32-Tensoren hinzu. Sie enthalten die exponentiellen gleitenden Mittelwerte des Gradienten und seines Quadrats. Da fast alle großen Trainingsläufe Adam oder eine eng verwandte Variante verwenden, sind 16 Byte pro Parameter in der Praxis der Ausgangswert:

<!--en:86093fbf-->
| Trainingszustand | Zahlengenauigkeit | Byte pro Parameter |

<!--en:39040850-->
| Gewichte | fp32 | 4 |

<!--en:70fae526-->
| Gradienten | fp32 | 4 |

<!--en:409651a0-->
| Mittelwert der Gradienten bei Adam | fp32 | 4 |

<!--en:f1717012-->
| Mittelwert der quadrierten Gradienten bei Adam | fp32 | 4 |

<!--en:21a418b7-->
| Insgesamt | | 16 |

<!--en:aaf2653c-->
Diese 16 Byte pro Parameter sind eine feste Untergrenze. Hinzu kommen die *Aktivierungen* aus Zwischenschichten, die der Vorwärtsdurchlauf für die Backpropagation speichert. Ihr Speicherbedarf wächst mit der Batchgröße und der Sequenzlänge. Auch temporäre Arbeitsbereiche der Kernel und die Fragmentierung bei der Speichervergabe erhöhen den Bedarf. All diese Beiträge messen wir in :numref:`sec_use_gpu` direkt auf einer GPU. Für unser Residual-MLP ergibt sich folgende Untergrenze:

<!--en:e83de5d8-->
Wir können die Parameteranzahl von Hand überprüfen. Die Eingabeschicht trägt $20 \times 64 + 64 = 1344$ Parameter bei, jeder Residualblock $2 \times (64 \times 64 + 64) = 8320$ und die Ausgabeschicht weitere $64 \times 10 + 10 = 650$. Insgesamt sind es 18 634 Parameter. Der Optimiererzustand von Adam ist mehr als ein theoretischer Tabellenwert. Für jeden Tensor belegt Adam tatsächlich zusätzlichen Speicher, insgesamt so viel wie zwei vollständige Kopien des Modells:

<!--en:496fc5b7-->
In Gluon verwaltet der `Trainer` diesen Zustand. Seine öffentliche Schnittstelle erlaubt keinen direkten Durchlauf. Bei der ersten Aktualisierung eines Parameters legt die Aktualisierungsroutine zwei mit Nullen gefüllte Arrays in der Form des Gewichts an. Mit `save_states` lässt sich der Zustand jedoch serialisieren. Bis auf einen kleinen Verwaltungsanteil enthält die Datei die beiden gleitenden Mittelwerte von Adam: 8 Byte pro Parameter, also zwei weitere Kopien des Modells.

<!--en:e77e1c59-->
`Adam` reserviert den Speicher für die beiden Momente jedes Parameters beim ersten Aufruf von `step()`.

<!--en:94e00082-->
In Keras reserviert `Adam` den Speicher für die beiden Momente jedes Gewichts beim ersten Aufruf von `apply_gradients`.

<!--en:9772a5b2-->
`nnx.Optimizer` reserviert den Speicher für beide Momente bereits beim Erzeugen des Optimierers, noch vor der ersten Aktualisierung.

<!--en:9d83423d-->
Bei unserem kleinen MLP beträgt der Mindestbedarf lediglich 0.28 MiB. Deshalb spielte der Speicher in früheren Kapiteln kaum eine Rolle. Bei einem Modell mit 8 Milliarden Parametern ergibt dieselbe Rechnung jedoch sofort 128 GB, unabhängig von der Batchgröße. Man könnte erwarten, dass 16-Bit-Training mit *gemischter Genauigkeit* (engl. mixed precision, :numref:`sec_numerics`) diesen statischen Mindestbedarf halbiert. Überraschenderweise bleibt er bestehen. Bei gemischter Genauigkeit liegen die Hauptgewichte und die Adam-Momente in fp32 vor, damit auch sehr kleine Aktualisierungen beim Runden erhalten bleiben :cite:`Micikevicius.Narang.Alben.ea.2018`. Die 16-Bit-Werte für den Vorwärtsdurchlauf entstehen durch Typumwandlung aus den Hauptgewichten.

<!--en:fcc3daab-->
### Parameterzählung ohne Speicherreservierung

<!--en:1d1d7bd4-->
Bisher haben wir nur Parameter gezählt, für deren Tensoren bereits Speicher reserviert war. Bei einem Modell mit 8 Milliarden Parametern ist das die falsche Reihenfolge. Um zu prüfen, ob 32 GB Gewichte in den Arbeitsspeicher passen, sollten wir nicht erst 32 GB reservieren müssen. Zum Zählen der Parameter genügen die Formen und Datentypen der Tensoren. Beides ist durch die Argumente des Konstruktors vollständig bestimmt. Betrachten wir die Ausgabeprojektion von Llama 3 8B. Sie bildet einen 4096-dimensionalen versteckten Zustand auf Logits für 128 256 *Token* des Vokabulars ab. Ihre Gewichtsmatrix enthält $4096 \times 128{,}256 = 525{,}336{,}576$ Einträge (2.1 GB in fp32). Wir erzeugen nun diese Schicht und zählen ihre Parameter, ohne dafür Tensorspeicher zu reservieren.

<!--en:d8fe077c-->
Erzeugen wir eine `Dense`-Schicht mit `in_units`, hält sie die Formen ihrer Parameter sofort fest. Speicher reserviert sie erst beim Aufruf von `initialize()`:

<!--en:0a21df63-->
Erzeugen wir ein Modul innerhalb von `with torch.device('meta'):`, entstehen *Meta-Tensoren*. Sie enthalten Formen und Datentypen, ohne Speicher für die Werte zu reservieren:

<!--en:32b688e3-->
Solange eine `Dense`-Schicht noch nicht aufgebaut ist, legt sie keine Variablen an. Die Form ihrer Gewichtsmatrix `(inputs, units)` ergibt sich direkt aus der Eingabebreite und `head.units`:

<!--en:c7b5488b-->
`nnx.eval_shape` verfolgt den Konstruktoraufruf abstrakt und liefert Blätter vom Typ `jax.ShapeDtypeStruct` mit Formen und Datentypen. Dabei legt es kein einziges Array an:

<!--en:fe93aee7-->
Um ein Modell vom Meta-Gerät auf dem Zielgerät anzulegen, rufen wir `to_empty(device=...)` und dann `load_state_dict` auf. So entfällt die unnötige Zufallsinitialisierung vor dem Laden des Checkpoints:

<!--en:ee1b0f35-->
Untersuchen wir alle Schichten eines Modells mit 8 Milliarden Parametern auf diese Weise, erhalten wir die exakte Parameteranzahl von 8 Milliarden und den Mindestbedarf von 128 GB für das Training, ohne ein einziges Byte Tensorspeicher zu reservieren.

<!--en:a842436e-->
## Gemeinsam genutzte und eingefrorene Parameter

<!--en:e7b27be3-->
### Gebundene Parameter

<!--en:e080ed5f-->
Derselbe Parametertensor kann in einem Netz mehrere Rollen übernehmen. Das klassische Beispiel stammt aus der Sprachmodellierung (:numref:`sec_language-model`). Ein Modell liest Token aus einem Vokabular $V$ und sagt das nächste Token vorher. Die *Einbettung* (engl. embedding) der Eingabe ist eine Nachschlagetabelle der Größe $|V| \times d$. Die Zeile an Position $v^\textrm{th}$ enthält die $d$-dimensionale Darstellung des Tokens $v$. Der Ausgabekopf projiziert einen $d$-dimensionalen versteckten Zustand zurück auf $|V|$ Token-Logits. Dazu dient eine weitere Tabelle der Größe $|V| \times d$ mit genau derselben Geometrie und semantischen Bedeutung. Durch die *Bindung* beider Schichten an dieselbe Gewichtsmatrix entfallen $|V| \times d$ Parameter, und häufig verbessert sich die Generalisierung :cite:`Press.Wolf.2017,Inan.Khosravi.Socher.2017`. Token, die als Eingaben austauschbar sind, sollten auch als Ausgaben ähnliche Scores erhalten. Die Gewichtsbindung erzwingt eine einheitliche Geometrie für die Token. Das kann viel Speicher sparen. Bei GPT-2 small :cite:`Radford.Wu.Child.ea.2019` entfallen auf die gebundene Tabelle der Größe $50257 \times 768$ rund 38.6 Millionen der 124 Millionen Modellparameter, also fast ein Drittel des gesamten Netzes.

<!--en:28cd6960-->
![Eine Gewichtsmatrix W an zwei Aufrufstellen: Die Einbettung liest anhand der Token-ID eine Zeile aus. Der Ausgabekopf multipliziert mit der transponierten Matrix und erhält so die Logits. Bei Gewichtsbindung verweisen beide auf denselben Tensor, sodass sich die Gradienten aus beiden Verwendungen an einer Stelle addieren.](../img/bg-weight-tying.svg)

<!--en:cb8eb939-->
:numref:`fig_bg_weight-tying` zeigt die beiden Aufrufstellen. Wir überprüfen die Funktionsweise an dem kleinen Sprachmodell `ToyLM` mit einem Vokabular von 100 Token und einer versteckten Schicht der Breite $d = 16$.

<!--en:0e2a3b98-->
Gluon speichert die Gewichte von `Dense` in der Form `(outputs, inputs)`. Daher haben `head` und `emb` genau dieselbe Form $(|V|, d)$. Mit `share_parameters` verweisen beide auf dasselbe `Parameter`-Objekt:

<!--en:75eb7faf-->
Durch die Zuweisung `self.head.weight = self.emb.weight` verweisen beide Schichten auf denselben Tensor vom Typ `nn.Parameter`:

<!--en:40bf8159-->
Eine einfache `TiedHead`-Schicht multipliziert die versteckten Zustände mit `tf.transpose(self.emb.embeddings)`, ohne eine zweite Variable anzulegen:

<!--en:b702d691-->
`nnx.Embed.attend(h)` projiziert die versteckten Zustände direkt mit der Transponierten der Einbettungstabelle von `self.emb`:

<!--en:52edd459-->
Zählt die gemeinsam genutzte Matrix einmal oder zweimal? Wir vergleichen das Modell mit Gewichtsbindung mit einer ansonsten identischen Variante ohne Gewichtsbindung.

<!--en:48908716-->
`collect_params()` listet beide Namen auf, die auf dasselbe `Parameter`-Objekt verweisen. Entfernen wir Duplikate anhand von `id(p)`, wie es `Trainer` intern tut, zählt die Tabelle nur einmal:

<!--en:fdb45b9a-->
Das Modell mit Gewichtsbindung enthält $1600 + 272 = 1872$ Parameter (eine Einbettungstabelle und die Schicht im Hauptzweig). In der Variante ohne Bindung kommt eine zweite Tabelle der Größe $100 \times 16$ hinzu, sodass es insgesamt 3472 Parameter sind.

<!--en:e1a08ae6-->
Sowohl `collect_params()` als auch `state_dict()` behalten beide Schlüssel (`emb.weight` und `head.weight`). Diese verweisen auf denselben Speicher:

<!--en:be63bf18-->
Was geschieht bei der Backpropagation, wenn ein Tensor zweimal zum Einsatz kommt? Nach der Kettenregel für mehrere Variablen trägt jede Aufrufstelle einen eigenen Gradiententerm bei. Die Summe beider Terme ergibt den Gradienten des gemeinsam genutzten Tensors. Bezeichnen wir die beiden Tabellen des Modells ohne Gewichtsbindung mit $\mathbf{W}_\textrm{emb}$ und $\mathbf{W}_\textrm{head}$. Initialisieren wir beide mit denselben Werten wie die gebundene Matrix $\mathbf{W}$, so berechnen beide Modelle genau dieselbe Funktion. Es gilt

<!--en:e69cbd33-->
Wir prüfen diese Identität numerisch. Dazu kopieren wir die gebundenen Gewichte in das Modell ohne Bindung und vergleichen die Gradienten für denselben Batch:

<!--en:47020512-->
### Eingefrorene Parameter

<!--en:98d5332c-->
Beim Fine-Tuning eines vortrainierten Netzes für eine neue Aufgabe (:numref:`sec_fine_tuning`) halten wir häufig das vortrainierte Basisnetz fest und trainieren nur einen neuen Ausgabekopf. Das senkt sowohl den Datenbedarf als auch den Speicherbedarf des Optimierers erheblich. Schützen wir Parameter vor Aktualisierungen durch Gradienten, sprechen wir vom *Einfrieren*. Im Folgenden frieren wir alle Schichten eines neuen Residual-MLP bis auf den Ausgabekopf ein. Damit bleiben nur 650 seiner 18 634 Parameter trainierbar.

<!--en:e6889522-->
`finetune[:-1].setattr('grad_req', 'null')` schaltet die Gradientenberechnung im gesamten Teilbaum des Basisnetzes aus:

<!--en:9c44c289-->
Mit `p.requires_grad = False` schließen wir die Parameter des Basisnetzes von der Gradientenakkumulation aus:

<!--en:75c4e510-->
Mit `layer.trainable = False` wechseln die Variablen der Schicht von `trainable_weights` zu `non_trainable_weights`:

<!--en:6c733ce8-->
In NNX wählt ein `trainable`-Filter den Ausgabekopf (`layers[3]`) aus. Wir übergeben den Filter sowohl an `nnx.grad` als auch an `nnx.Optimizer(..., wrt=trainable)`:

<!--en:bcc5d4e7-->
Ein Adam-Schritt sollte jetzt nur den Ausgabekopf aktualisieren und alle früheren Schichten unverändert lassen. Wir prüfen jeden Parameter vor und nach dem Schritt.

<!--en:155034f3-->
Nur das Gewicht und der Bias des Ausgabekopfs haben sich geändert. Der Optimierer hat Speicher für die Adam-Momente nur für diese 650 trainierbaren Parameter reserviert. Beim Einfrieren gibt es zwei leicht zu übersehende Fallstricke. Keiner führt zu einer Fehlermeldung. Erstens gibt das Einfrieren eines Parameters bereits belegten Speicher für den Optimiererzustand *nicht* frei. Haben wir Adam für alle Parameter angelegt und mindestens einen Schritt ausgeführt, belegen die gleitenden Momente weiterhin 8 Byte pro Parameter, auch in nachträglich eingefrorenen Schichten. Bei 18 634 Parametern fällt das kaum ins Gewicht, bei 8 Milliarden dagegen schon, wie die Übungen zeigen.

<!--en:91a3b37f-->
Der zweite Fallstrick ist noch leichter zu übersehen. Einfrieren blockiert Gradienten, doch Teile des Schichtzustands ändern sich auch *ohne* Gradienten. Insbesondere aktualisiert die Batchnormalisierung (:numref:`sec_batch_norm`) ihren gleitenden Mittelwert und ihre gleitende Varianz bei jedem *Vorwärtsdurchlauf* im Trainingsmodus, ganz ohne Beteiligung des Optimierers. Ob das Einfrieren der trainierbaren Gewichte einer BatchNorm-Schicht auch ihre gleitenden Statistiken festsetzt, hängt daher vom Framework ab. Wir leiten einen Batch mit verschobenen Werten durch eine eingefrorene BatchNorm-Schicht im Trainingsmodus. Ohne einen einzigen Optimiererschritt prüfen wir, ob sich ihr gleitender Mittelwert verschiebt.

<!--en:a5d3cf4b-->
Der gleitende Mittelwert hat sich verschoben. Das Einfrieren der Parametergradienten (`grad_req='null'`) und das Umschalten der Schicht in den Auswertungsmodus (`use_global_stats=True`) sind zwei getrennte Vorgänge. Um eine BatchNorm-Schicht beim Fine-Tuning vollständig einzufrieren, sind beide Einstellungen nötig.

<!--en:3635bd34-->
Der gleitende Mittelwert hat sich verschoben. Das Einfrieren der Parametergradienten (`requires_grad=False`) und das Umschalten der Schicht in den Auswertungsmodus (`bn.eval()`) sind zwei getrennte Vorgänge. Um eine BatchNorm-Schicht beim Fine-Tuning vollständig einzufrieren, sind beide Einstellungen nötig.

<!--en:e9f18f51-->
Der gleitende Mittelwert hat sich verschoben. Das Einfrieren der Parametergradienten (mit einem Filter für `nnx.Param`) und das Umschalten der Schicht in den Auswertungsmodus (`use_running_average=True`) sind zwei getrennte Vorgänge. Um eine BatchNorm-Schicht beim Fine-Tuning vollständig einzufrieren, sind beide Einstellungen nötig.

<!--en:b8eb9763-->
Keras behandelt `BatchNormalization` als Sonderfall. Hier erzwingt `bn.trainable = False` auch für die gleitenden Statistiken den Inferenzmodus. Bei anderen zustandsbehafteten Schichten wie Dropout bleiben `trainable` und `training=False` voneinander unabhängig.

<!--en:45cd7ff3-->
Alle Parameter, die wir in diesem Abschnitt untersucht haben, begannen mit Anfangswerten aus dem Framework, gleich ob wir die Parameter gebunden, eingefroren oder gemittelt haben. Im nächsten Abschnitt untersuchen wir, wie diese Anfangswerte zustande kommen und wie wir die Initialisierung im gesamten Modulbaum steuern können.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3f7df26c-->
Der persistente Zustand eines Modells bildet einen Baum aus benannten Tensoren. Die Registrierung bestimmt, welche Operationen des Frameworks den jeweiligen Tensor erfassen. Optimierer aktualisieren *Parameter*. Checkpoints und Gerätewechsel berücksichtigen auch *Puffer*, also persistente Tensoren ohne Gradienten wie etwa gleitende Statistiken. Nicht registrierte Attribute überspringt das Framework ohne Hinweis. Das Training mit Adam in fp32 benötigt bereits ohne Aktivierungen mindestens 16 Byte pro Parameter für Gewichte, Gradienten und zwei Adam-Momente. Bei einem Modell mit 8 Milliarden Parametern sind das 128 GB. Durch die abstrakte Untersuchung der Tensorformen können wir den Speicherbedarf berechnen, ohne Speicher zu reservieren. Bei der Gewichtsbindung übernimmt ein Tensor mehrere Rollen. Er zählt nur einmal und erhält die Summe der Gradienten aller Aufrufstellen. Das Einfrieren unterbindet Aktualisierungen durch Gradienten, gibt aber den Speicher für vorhandenen Optimiererzustand nicht frei. Auch kann die Batchnormalisierung ihre gleitenden Statistiken beim Vorwärtsdurchlauf weiterhin aktualisieren.

<!--en:2294243a-->
## Übungen

<!--en:9935faae-->
1. [code] **Gleitende Statistiken.** Für diese Aufgabe brauchen Sie nur die folgende Aktualisierungsregel der Batchnormalisierungsschicht aus :numref:`sec_batch_norm`. Im Trainingsmodus normalisiert die Schicht mit dem Mittelwert $\hat{\boldsymbol{\mu}}_\mathcal{B}$ und der Varianz des Minibatchs. Bei jedem Trainingsschritt ändert sich der gleitende Mittelwert $\bar{\boldsymbol{\mu}}$ gemäß

   $$\bar{\boldsymbol{\mu}} \leftarrow \rho\, \bar{\boldsymbol{\mu}} + (1 - \rho)\, \hat{\boldsymbol{\mu}}_\mathcal{B}.$$

    1. Eine neu angelegte Schicht beginnt mit $\bar{\boldsymbol{\mu}} = \mathbf{0}$. Sagen Sie $\bar{\boldsymbol{\mu}}$ voraus, nachdem die Schicht im Trainingsmodus $k$ Batches verarbeitet hat. In jedem Minibatch hat jedes Merkmal den Mittelwert 3. Prüfen Sie Ihre Vorhersage für $k = 1, 5, 20$ an einer neuen Schicht mit 4 Merkmalen, ohne einen Optimiererschritt auszuführen. Lesen Sie ab, welchen Wert von $\rho$ die Schicht standardmäßig verwendet.
    1. Angenommen, Sie registrieren $\bar{\boldsymbol{\mu}}$ als trainierbaren Parameter und entfernen die obige Aktualisierungsregel. Welchen Gradienten liefert der Trainingsverlust für $\bar{\boldsymbol{\mu}}$? Wie fallen Trainingsverlust und Genauigkeit bei der Auswertung im Vergleich zum unveränderten Modell aus?
    1. Finden Sie eine Verlustfunktion, die nur von $\bar{\boldsymbol{\mu}}$ abhängt und deren Gradientenschritt mit Lernrate $\eta$ für jeden Minibatch die entfernte Aktualisierung reproduziert. Geben Sie an, welches $\eta$ zu $\rho$ passt.

<!--en:525b1351-->
2. [code] **Speicherbedarf.** Schreiben Sie eine Hilfsfunktion, die für jeden Block auf der obersten Ebene von `net` die Anzahl der Zustandseinträge und den Speicherbedarf in Byte ausgibt. Gehen Sie von einem Training mit Adam in fp32 aus, bei dem alle Parameter trainiert werden.
    1. Welcher Block benötigt den meisten Speicher? Ab welcher Ausgabebreite benötigt die Ausgabeschicht mehr Speicher als ein Residualblock?
    1. Fügen Sie zwischen den beiden Residualblöcken eine Batchnormalisierungsschicht für die 64 versteckten Merkmale ein. Erweitern Sie die Hilfsfunktion so, dass sie den Speicherbedarf der gleitenden Statistiken korrekt berücksichtigt. Geben Sie für jeden Block die Byte pro Eintrag an. Bei welchen Blöcken ist der Anteil am Gesamtspeicher größer als der Anteil an allen Einträgen, bei welchen ist er kleiner? Erklären Sie beide Fälle.

<!--en:31348f9f-->
3. [code] **Gewichtsbindung und Einfrieren.** Betrachten Sie das Modell `ToyLM` mit Gewichtsbindung aus diesem Abschnitt.
    1. Setzen Sie `lm.emb.weight.grad_req = 'null'`. Was hat sich am Ausgabekopf geändert, und wie viele Parameter bleiben trainierbar? Was zeigt das über das Zusammenspiel von Gewichtsbindung und Einfrieren?
    1. Erstellen Sie ausgehend von den Werten von `lm` ein Modell, dessen Einbettung fest bleibt und dessen Ausgabekopf sich anpasst. Überprüfen Sie beide Eigenschaften nach einem Optimiererschritt. Wie viele Parameter speichert das Modell und wie viele trainiert es im Vergleich zu einem vollständig trainierbaren `lm`?

<!--en:17323102-->
3. [code] **Gewichtsbindung und Einfrieren.** Betrachten Sie das Modell `ToyLM` mit Gewichtsbindung aus diesem Abschnitt.
    1. Frieren Sie `lm.emb.weight` ein und lassen Sie `lm.head` unverändert. Wie viele Parameter bleiben trainierbar, und was zeigt das über das Zusammenspiel von Gewichtsbindung und Einfrieren?
    1. Erstellen Sie ausgehend von den Werten von `lm` ein Modell, dessen Einbettung fest bleibt und dessen Ausgabekopf sich anpasst. Überprüfen Sie beide Eigenschaften nach einem Optimiererschritt. Wie viele Parameter speichert das Modell und wie viele trainiert es im Vergleich zu einem vollständig trainierbaren `lm`?

<!--en:8a32b50c-->
3. [code] **Gewichtsbindung und Einfrieren.** Betrachten Sie das Modell `ToyLM` mit Gewichtsbindung aus diesem Abschnitt.
    1. Setzen Sie `lm.emb.trainable = False` und lassen Sie `lm.head` unverändert. Wie viele Parameter bleiben trainierbar? Kann sich der Ausgabekopf noch anpassen? Was zeigt das über das Zusammenspiel von Gewichtsbindung und Einfrieren?
    1. Erstellen Sie ausgehend von den Werten von `lm` ein Modell, dessen Einbettung fest bleibt und dessen Ausgabekopf sich anpasst. Überprüfen Sie beide Eigenschaften nach einem Optimiererschritt. Wie viele Parameter speichert das Modell und wie viele trainiert es im Vergleich zu einem vollständig trainierbaren `lm`?

<!--en:e0ec51bf-->
3. [code] **Gewichtsbindung und Einfrieren.** Betrachten Sie das Modell `ToyLM` mit Gewichtsbindung aus diesem Abschnitt.
    1. Schließen Sie die Einbettungstabelle vom Filter für trainierbare Parameter aus, etwa mit `nnx.All(nnx.Param, nnx.Not(nnx.PathContains('emb')))`. Übergeben Sie den Filter an `nnx.grad` und `nnx.Optimizer`. Wie viele Parameter werden trainiert? Kann sich der Ausgabekopf noch anpassen? Was zeigt das über das Zusammenspiel von Gewichtsbindung und Einfrieren?
    1. Erstellen Sie ausgehend von den Werten von `lm` ein Modell, dessen Einbettung fest bleibt und dessen Ausgabekopf sich anpasst. Überprüfen Sie beide Eigenschaften nach einer Aktualisierung. Wie viele Parameter speichert das Modell und wie viele trainiert es im Vergleich zu einem vollständig trainierbaren `lm`?

<!--en:10da8964-->
4. **Speicherbedarf beim Fine-Tuning.** Ein Sprachmodell hat 8 Milliarden Parameter. Seine Ausgabeschicht ist eine Matrix der Größe $4096 \times 128{,}256$ ohne Gewichtsbindung an die Einbettung. Der gesamte Zustand liegt in fp32 vor, als Optimierer dient Adam.
    1. Wie viele Byte belegen die Gewichte, die Gradienten und die gleitenden Mittelwerte von Adam, wenn alle Parameter trainiert werden? Wie viele GPUs mit 80 GB füllt dieser Zustand allein?
    1. Frieren Sie alles bis auf die Ausgabeschicht ein und legen Sie den Optimierer nur für die Ausgabeschicht an. Wie viele Byte belegt der persistente Zustand nun? Passt er auf eine GPU mit 80 GB?
    1. Angenommen, Sie legen den Optimierer stattdessen für alle Parameter an, führen vor dem Einfrieren einen Schritt aus und behalten den Optimierer danach bei. Wie viele Byte bleiben belegt? Passt der Zustand noch auf die GPU?

<!--en:933ea121-->
[Diskussion](https://d2l.discourse.group/t/36753)

<!--en:05d8b31f-->
Parameter, Zustand und Speicher<br>**Zustandsbaum · trainierbare Parameter und Puffer · mindestens 16 B pro Parameter mit Adam · Gewichtsbindung · Einfrieren**

<!--en:eed1a964-->
::: {.slide title="Warum 8 Mrd. Parameter schon vor einem Batch der Größe 1 insgesamt 128 GB benötigen"}

<!--en:bb0d7688-->
[Speicherbedarf des Systems]{.kicker}

<!--en:131071cd-->
- **Vier Kopien mit voller Genauigkeit ($16\text{ B/param}$):** Adam speichert in `fp32` die Gewichte $\mathbf{w}$ ($4\text{ B}$), die Gradienten $\mathbf{g}$ ($4\text{ B}$), das erste Moment $\mathbf{m}$ ($4\text{ B}$) und das zweite Moment $\mathbf{v}$ ($4\text{ B}$).

<!--en:9050b697-->
- **Statischer Mindestbedarf für 8 Mrd. Parameter:** $8 \times 10^9 \times 16\text{ B} = 128\text{ GB}$. Das übersteigt die Kapazität einer GPU mit `80 GB` schon vor dem Laden des ersten Eingabebatchs.

<!--en:1412900b-->
- **Wozu den Zustandsbaum kennen?** Wer den Speicherbedarf berechnet, Checkpoints speichert oder Schichten einfriert, muss jeden Tensor im Zustandsbaum erfassen.

<!--en:878e63b4-->
::: {.slide title="Durchgängiges Beispiel: ein Residual-MLP mit zwei Blöcken"}

<!--en:3332c652-->
[Aufbau des Zustandsbaums]{.kicker}

<!--en:fc97bf03-->
- Als Beispiel für den Durchlauf dient ein kompaktes Residual-MLP ($20 \to 64 \to \text{ two } 64\text{-wide residual blocks } \to 10$) mit 6 linearen Schichten und 12 Parametertensoren:

<!--en:6b8d577e-->
::: {.slide title="Zugriff auf Parameter über ihren Pfad im Zustandsbaum" except="tensorflow"}

<!--en:6a04f6f8-->
[Baumdurchlauf]{.kicker}

<!--en:a0b8cca7-->
- Ein Iterator über den gesamten Baum liefert jedes Blatt mit seinem Pfad:

<!--en:4b5543d5-->
::: {.slide title="Zugriff auf Parameter über ihren Pfad im Zustandsbaum" only="tensorflow"}

<!--en:6a04f6f8-2-->
[Baumdurchlauf]{.kicker}

<!--en:a0b8cca7-2-->
- Ein Iterator über den gesamten Baum liefert jedes Blatt mit seinem Pfad:

<!--en:0fe480ac-->
::: {.slide title="Drei Arten von Tensoren in einer Schicht"}

<!--en:a7113e9f-->
[Parameter und Puffer]{.kicker}

<!--en:5f95cc59-->
- **Parameter:** Der Optimierer aktualisiert sie, Checkpoints speichern sie, und beim Gerätewechsel gelangen sie auf die GPU ($\mathbf{W}, \mathbf{b}, \mathbf{g}$).

<!--en:352f0bd8-->
- **Registrierte Puffer:** Der Optimierer überspringt sie. Checkpoints speichern sie, Geräte- und Typwechsel erfassen sie mit dem Modell ($\boldsymbol{\mu}, \boldsymbol{\sigma}$ je Merkmal, gleitende Statistiken der Batchnormalisierung).

<!--en:6fcd6476-->
- **Gewöhnliche Attribute:** Checkpoints und Gerätewechsel überspringen sie. Persistente Tensoren niemals als gewöhnliche Python-Attribute speichern.

<!--en:58e03158-->
::: {.slide title="Registrierung nicht trainierbarer Puffer zur Standardisierung"}

<!--en:02fa69c4-->
[Puffer im Code]{.kicker}

<!--en:b898cd1a-->
- Eine `Standardizer`-Schicht registriert die Statistiken $\boldsymbol{\mu}$ und $\boldsymbol{\sigma}$ des Trainingsdatensatzes als nicht trainierbare Puffer.

<!--en:01fa7488-->
::: {.slide title="Eine Standardisierungsschicht mit zwei Puffern"}

<!--en:02fa69c4-2-->
[Puffer im Code]{.kicker}

<!--en:75c2624b-->
::: {.slide title="Checkpoints speichern Puffer, Optimierer überspringen sie"}

<!--en:45954bfc-->
[Filtern des Zustandsbaums]{.kicker}

<!--en:b6a4ccbc-->
- Der Vergleich des gesamten Zustandsbaums mit den trainierbaren Parametern zeigt: `mean` und `std` gelangen auf den Datenträger und auf die GPU. Der Optimierer aktualisiert nur `weight` und `bias`:

<!--en:accf2b96-->
::: {.slide title="Parameteranzahl und Speicherbedarf des Residual-MLP"}

<!--en:692a63b4-->
[Berechnung des Speicherbedarfs]{.kicker}

<!--en:ee8c291c-->
Die Parameteranzahl des Residual-MLP lässt sich rechnerisch überprüfen. Bei `fp32`-Training mit Adam ergeben sich daraus mit 16 Byte pro Parameter folgende Speichergrößen:

<!--en:07c7fb3a-->
| Schichtgruppe | Berechnung der Parameteranzahl ($W + b$) | Parameteranzahl | Zustand bei `fp32`-Training mit Adam ($16\text{ B/param}$) |

<!--en:50543dff-->
| **Eingabeschicht (`0`)** | $20 \times 64 + 64$ | $1{,}344$ | $21{,}504\text{ B}$ |

<!--en:3411905a-->
| **Residualblöcke (`1`, `2`)** | $2 \times [2 \times (64 \times 64 + 64)]$ | $16{,}640$ | $266{,}240\text{ B}$ |

<!--en:efe3eef9-->
| **Ausgabekopf (`3`)** | $64 \times 10 + 10$ | $650$ | $10{,}400\text{ B}$ |

<!--en:b3fdb72e-->
| **Gesamtes Netz** | **12 Parametertensoren** | **$18{,}634$** | **$298{,}144\text{ B}$ ($0.28\text{ MiB}$)** |

<!--en:4415c3c7-->
::: {.slide title="Nachweis des Speicherbedarfs von Adam: zwei vollständige Modellkopien" except="jax"}

<!--en:fb200a8b-->
[Speicher für den Optimiererzustand]{.kicker}

<!--en:b95f62bf-->
- Adam führt für jeden Parameter das erste und das zweite Moment ($\mathbf{m}$ und $\mathbf{v}$) mit. Das ergibt zusätzlich $2 \times 18{,}634 = 37{,}268$ Gleitkommazahlen ($8\text{ B/param}$):

<!--en:815a4e91-->
::: {.slide title="Nachweis des Speicherbedarfs von Adam: zwei vollständige Modellkopien" only="jax"}

<!--en:fb200a8b-2-->
[Speicher für den Optimiererzustand]{.kicker}

<!--en:b95f62bf-2-->
- Adam führt für jeden Parameter das erste und das zweite Moment ($\mathbf{m}$ und $\mathbf{v}$) mit. Das ergibt zusätzlich $2 \times 18{,}634 = 37{,}268$ Gleitkommazahlen ($8\text{ B/param}$):

<!--en:72fd9a33-->
::: {.slide title="Abstrakte Parameterzählung ohne Speicherreservierung im RAM"}

<!--en:73057fa5-->
[Meta-Geräte und Bestimmung der Tensorformen]{.kicker}

<!--en:b2739b17-->
- Frameworks werten Konstruktoren abstrakt allein anhand von Formen und Datentypen aus. So lässt sich der Ausgabekopf von Llama 3 8B mit $525.3\text{M}$ Parametern (`2.1 GB`) prüfen, ohne **ein einziges Byte** zu reservieren:

<!--en:c08dfb44-->
::: {.slide title="Erzeugen eines Modells vom Meta-Gerät auf der Zielhardware" only="pytorch"}

<!--en:b570816f-->
[Laden von Checkpoints ohne unnötige Initialisierung]{.kicker}

<!--en:3c50e318-->
- Das Modell auf `torch.device('meta')` erzeugen und `to_empty(device=...)` vor `load_state_dict` aufrufen. So entfällt die unnötige Zufallsinitialisierung auf der CPU:

<!--en:d92628dd-->
::: {.slide title="Gewichtsbindung: ein Parametertensor in mehreren Rollen"}

<!--en:9b492fef-->
[Gemeinsam genutzte Parameter]{.kicker}

<!--en:46d7d2f9-->
Sprachmodelle bilden die IDs der Eingabetoken auf $d$-dimensionale Vektoren ab. Dazu verwenden sie eine Einbettungstabelle der Größe $|V| \times d$. Die letzten $d$-dimensionalen Vektoren projizieren sie zurück auf $|V|$ Token-Logits. Der Ausgabekopf hat die Größe $|V| \times d$:

<!--en:905abeda-->
- **Gemeinsame semantische Geometrie:** Die Bindung beider Schichten an **ein und denselben** Tensor der Größe $|V| \times d$ erzwingt eine einheitliche Token-Darstellung und spart $|V| \times d$ Parameter.

<!--en:aa609e04-->
- **Große Speicherersparnis:** Bei GPT-2 small ($|V|=50{,}257, d=768$) entfallen auf die gebundene Tabelle $38.6\text{M}$ von $124\text{M}$ Parametern ($31\%$ des gesamten Netzes).

<!--en:63509aeb-->
::: {.slide title="Gewichtsbindung in einem kleinen Sprachmodell"}

<!--en:bd586db3-->
[Gewichtsbindung im Code]{.kicker}

<!--en:b4f21f95-->
- In `ToyLM` ($|V| = 100, d = 16$) verweist die Ausgabeprojektion auf dieselbe Einbettungstabelle wie die Eingabe. Die Tabelle liegt daher nur einmal im Speicher:

<!--en:8c5021cd-->
::: {.slide title="Gebundene Tensoren zählen nur einmal"}

<!--en:9616e62c-->
[Entfernen doppelter Parametereinträge]{.kicker}

<!--en:7645d4f3-->
- Beim Baumdurchlauf zählt gemeinsam genutzter Speicher nur einmal. Daher geht die Einbettungstabelle mit $100 \times 16 = 1{,}600$ Einträgen nur einmal in die Summe ein ($1{,}872$ Parameter mit Bindung, $3{,}472$ ohne):

<!--en:888bb8e1-->
::: {.slide title="Summe der Gradienten aller Aufrufstellen im gebundenen Tensor"}

<!--en:27498137-->
[Kettenregel für mehrere Variablen]{.kicker}

<!--en:0d6bafee-->
::: {.slide title="Einfrieren von Parametern beim Transferlernen"}

<!--en:17c30c21-->
[Gezielte Optimierung]{.kicker}

<!--en:b0151c57-->
Beim Fine-Tuning eines vortrainierten Basisnetzes auf einem kleinen Datensatz die Gewichte des Basisnetzes **einfrieren** und nur den Ausgabekopf trainieren ($650$ von $18{,}634$ Parametern im Residual-MLP):

<!--en:7176889a-->
::: {.slide title="Nach dem Einfrieren lernt nur der Ausgabekopf" layout="tall-output"}

<!--en:17c30c21-2-->
[Gezielte Optimierung]{.kicker}

<!--en:5e42eb37-->
::: {.slide title="Zwei Fallstricke ohne Fehlermeldung beim Einfrieren"}

<!--en:b6d8e59a-->
[Risiken beim Einfrieren]{.kicker}

<!--en:40c60b82-->
- **Fallstrick 1:** Die Adam-Momente ($8\text{ B/param}$) bleiben im Speicher, sofern der Optimierer nicht ausschließlich die weiterhin trainierbaren Parameter umfasst.

<!--en:0f1e1ab1-->
- **Fallstrick 2:** Im Trainingsmodus können BatchNorm-Schichten beim Vorwärtsdurchlauf die Puffer für gleitenden Mittelwert und gleitende Varianz weiterhin aktualisieren. Erst das Umschalten in den Auswertungsmodus verhindert das.

<!--en:788c7409-->
::: {.slide title="Verständnisfragen zu Speicherbedarf und gemeinsam genutzten Parametern"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:a10993aa-->
Die folgenden Fragen beziehen sich auf das Sprachmodell mit 8 Mrd. Parametern (`fp32`, Adam, Ausgabekopf mit $4096 \times 128{,}256 \approx 525\text{M}$ Parametern):

<!--en:b616b0f0-->
1. Angenommen, alle Gewichte des Modells mit 8 Mrd. Parametern sind bis auf den Ausgabekopf mit $525\text{M}$ Parametern eingefroren, und Adam umfasst nur den Ausgabekopf. Wie viel Speicher benötigt der persistente Zustand in `fp32`? Passt er auf eine einzelne GPU mit `80 GB`?

<!--en:11313ecb-->
2. Was ändert sich, wenn Adam anfangs alle 8 Mrd. Parameter umfasst, 1 Schritt in der Aufwärmphase ausführt und das Basisnetz *erst danach* eingefroren wird, ohne den Optimierer neu anzulegen?

<!--en:60d141fa-->
3. Zwei Schichten nutzen dieselbe Gewichtsmatrix $\mathbf{W}$. Wenn $\mathbf{W}$ über die Einbettungsschicht eingefroren wird, kann sich `self.head` dann noch anpassen?

<!--en:557b98b6-->
**Lösungen:**

<!--en:a4f3296c-->
1. **$\approx 38.3\text{ GB}$ (passt auf eine GPU mit 80 GB):** Alle 8 Mrd. Gewichte benötigen zusammen $32\text{ GB}$. Die Gradienten und Adam-Momente für den Ausgabekopf mit $525\text{M}$ Parametern benötigen nur $12 \times 0.525\text{B} = 6.3\text{ GB}$.

<!--en:3779bc16-->
2. **$\approx 98.1\text{ GB}$ (80 GB reichen nicht aus):** Adam behält die $64\text{ GB}$ für die in Schritt 1 angelegten Momente bei ($32\text{ GB weights} + 64\text{ GB moments} + 2.1\text{ GB head grad}$).

<!--en:51587e2b-->
3. **Nein:** `self.emb.weight` und `self.head.weight` verweisen auf ein und denselben Parametertensor. Wird er eingefroren, betrifft das beide Rollen.

<!--en:d2ab6585-->
::: {.slide title="Zusammenfassung: Parameter, Zustand und Speicher"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:a9ff9a0d-->
- **Parameter und Puffer:** Beide gelangen in Checkpoints und auf die GPU. Der Optimierer aktualisiert nur trainierbare Parameter.

<!--en:3d7d486a-->
- **Statischer Speicherbedarf und Zählen auf Meta-Geräten:** Adam benötigt bei `fp32` pro Parameter $16\text{ B/param}$ (`128 GB` für 8 Mrd. Parameter). Mit Meta-Geräten lässt sich der exakte Bedarf ohne physische Speicherreservierung bestimmen.

<!--en:52a1af7b-->
- **Gewichtsbindung und Einfrieren:** Gemeinsam genutzte Tensoren zählen nur einmal und erhalten die Summe der Gradienten aller Aufrufstellen. Beim Einfrieren die Parameter für den Optimierer filtern und BatchNorm in den Auswertungsmodus schalten.
