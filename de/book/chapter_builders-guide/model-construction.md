<!--en:986eb32f-->
# Module und Modellaufbau

<!--en:7d9b893f-->
Das mehrschichtige Perzeptron (engl. multilayer perceptron, MLP) in :numref:`sec_mlp-implementation` hatte nur vier Parameterarrays. Die konnten wir von Hand verwalten. Architekturen aus der Praxis sind weit komplexer. Ein modernes Residualnetz für Computer Vision umfasst Dutzende oder Hunderte aufeinanderfolgender Schichten. Ein Sprachmodell wie Llama 3 8B :cite:`Grattafiori.Dubey.Jauhri.ea.2024` hat 8 Milliarden Parameter in 32 identischen Transformer-Blöcken. Seine Gewichte verteilen sich auf 291 separate Arrays. Jedes einzelne Array muss den Optimierer erreichen, in Checkpoints gespeichert und auf die GPU übertragen werden. Besonders tückisch ist, dass ein vergessenes Array keinen Syntaxfehler auslöst. Es bleibt unbemerkt untrainiert oder ungespeichert.

<!--en:bf206375-->
Deep-Learning-Frameworks lösen dieses Verwaltungsproblem mit einer einzigen rekursiven Abstraktion, dem *Modul*. Ein Modul kann eine einzelne Schicht, eine Gruppe von Schichten oder ein ganzes Modell darstellen. Es verwaltet eigene Parameter und Kindmodule und definiert eine Vorwärtsberechnung, die Eingaben auf Ausgaben abbildet. Da Module weitere Module enthalten können, bildet jede Architektur einen Modulbaum (:numref:`fig_blocks`). Jedes Framework findet beim Durchlaufen dieses Baums sämtliche Parameter, sofern jedes Kindmodul bei seinem übergeordneten Modul korrekt *registriert* ist.

<!--en:7d26260e-->
In diesem Abschnitt arbeiten wir mit praktischen Beispielen. Wenn Sie bereits eigene Module geschrieben haben, können Sie die Grundlagen überfliegen. Lesen Sie aber, wie die Registrierung funktioniert und wann sie unbemerkt fehlschlägt. Dieser Fehler unterläuft einem besonders leicht. Wir zeigen auch, wie wir eigene Vorwärtsberechnungen wie Residualverbindungen (engl. residual connections) formulieren, wie die verzögerte Initialisierung (engl. lazy initialization) Formen zur Laufzeit erschließt und wie wir die Hyperparameter der Architektur in einem kompakten Konfigurationsobjekt zusammenfassen. In späteren Abschnitten durchlaufen wir denselben Baum, um Parameter zu zählen (:numref:`sec_parameters`), Gewichte zu initialisieren (:numref:`sec_init_param`), Checkpoints zu speichern (:numref:`sec_read_write`) und Modelle auf GPUs zu übertragen (:numref:`sec_use_gpu`).

<!--en:3e7fa56c-->
![Modulbaum des residualen MLP aus dem letzten Teil dieses Abschnitts. Jede Kante trägt den Namen, unter dem das übergeordnete Modul sein Kindmodul registriert hat. Die Namen auf dem Pfad von der Wurzel bis zu einem Parameter ergeben dessen Namen, hier `1.body.0.weight`. Die Blöcke 2 bis 4 wiederholen Block 1.](../img/bg-module-names.svg)

<!--en:cda27237-->
## Das Modul als Abstraktion

<!--en:672f0b89-->
### Sequenzielle Container und das Register

<!--en:66d0b2e9-->
Jedes Modell, das wir seit :numref:`sec_oo-design` geschrieben haben, ist ein Modul. `d2l.Module` ist eine Unterklasse der jeweiligen Modulbasisklasse des Frameworks. Am schnellsten verbinden wir mehrere Schichten mit einem *sequenziellen Container*, der seine Kindmodule hintereinanderschaltet. Wie in :numref:`sec_mlp-implementation` bauen wir ein MLP mit einer versteckten Schicht aus 256 ReLU-Einheiten und 10 Ausgängen. Als Eingabe übergeben wir einen Minibatch aus zwei Beispielen mit je 20 Merkmalen.

<!--en:47fb3413-->
In Gluon heißt die Basisklasse `nn.Block`. In `nn.Sequential` fügt `add` Schichten hinzu. Vor dem ersten Aufruf bereitet `initialize()` die Parameter vor (:numref:`sec_lazy_init`).

<!--en:ebd9ce5d-->
In PyTorch heißt die Basisklasse `nn.Module`. `nn.Sequential` verkettet die als Argumente übergebenen Schichten.

<!--en:9a79a46c-->
In Keras heißt die Basisklasse `tf.keras.layers.Layer` (die Trainingsmethoden gehören zu `tf.keras.Model`). `tf.keras.Sequential` verkettet die Schichten einer Liste.

<!--en:4b13f34e-->
In Flax NNX heißt die Basisklasse `nnx.Module`. `nnx.Sequential` verkettet seine Argumente. Die Schichten legen ihre Parameter bereits im Konstruktor mithilfe von `rngs` an.

<!--en:52e05fa0-->
Intern speichert der Container seine Kindmodule in einem geordneten *Register* (engl. registry), das jedem Namen ein Kindmodul zuordnet. Beim Vorwärtsdurchlauf (engl. forward pass) durchläuft er diese Tabelle in Einfügereihenfolge.

<!--en:a044a938-->
Die Ausgabe des Blocks zeigt seine Kindmodule mit ihrem Index als Schlüssel (`0`, `1`).

<!--en:57c6bea6-->
`named_children` liefert Paare der Form `(name, module)` mit den Namen `'0'`, `'1'` und `'2'`.

<!--en:7b983492-->
Keras speichert die Kindmodule mit automatisch erzeugten Namen in `net.layers`.

<!--en:9cca1e43-->
`nnx.state(net, nnx.Param)` listet jeden Parameter mit seinem Pfad durch `layers` auf.

<!--en:f5a977e9-->
Jeder Durchlauf durch den Modulbaum beginnt bei diesem Register. Ob wir Parameter für den Optimierer sammeln, Gewichte auf einen Datenträger serialisieren oder Tensoren auf eine GPU übertragen: Jede dieser Operationen folgt rekursiv den registrierten Kindmodulen bis zu jedem Parameter an den Blättern. Für diese Operationen ist eine unregistrierte Schicht unsichtbar.

<!--en:ada682c2-->
### Unterklassen der Modulklasse

<!--en:89e15731-->
Für lineare Ketten eignet sich ein sequenzieller Container. Viele Architekturen verzweigen sich jedoch, führen Zweige zusammen oder verwenden Komponenten mehrfach. Für alles, was über eine einfache Kette hinausgeht, leiten wir eine Unterklasse von der Modulbasisklasse des Frameworks ab. Wir implementieren zwei Methoden: einen Konstruktor, der Kindmodule erzeugt, und eine Vorwärtsmethode, die den Datenfluss durch diese Module festlegt.

<!--en:46046445-->
In Gluon heißt die Vorwärtsmethode `forward`.

<!--en:967a99bc-->
In PyTorch heißt die Vorwärtsmethode `forward`.

<!--en:f2c88f80-->
In Keras heißt die Vorwärtsmethode `call`.

<!--en:6c62ddea-->
In Flax NNX heißt die Vorwärtsmethode `__call__`.

<!--en:0feda807-->
So schreiben wir dasselbe MLP noch einmal:

<!--en:1edf8555-->
Wir erzeugen eine Instanz von `MLP`. Der Aufruf `net(X)` führt unsere Vorwärtsmethode aus und bildet den $2 \times 20$ großen Eingabebatch `X` auf einen Ausgabetensor der Form `(2, 10)` ab:

<!--en:c1205484-->
In MXNet legt `net.initialize()` zuerst die Parameter an.

<!--en:e0cdcdc2-->
In JAX übergeben wir dem Konstruktor explizit ein Objekt vom Typ `nnx.Rngs`.

<!--en:ca0193bc-->
Dabei greifen im Hintergrund zwei Mechanismen. Erstens speichert eine Zuweisung wie `self.hidden = ...` mehr als ein gewöhnliches Python-Attribut. Das Basismodul fängt die Zuweisung mit `__setattr__` ab, erkennt das Modul am zugewiesenen Wert und registriert es unter dem Attributnamen (hier `hidden` und `out`). Dank dieser automatischen Registrierung erreichen die Parameter beider Schichten den Optimierer. Zweitens brauchen wir keine Rückwärtsmethode zu schreiben. Automatisches Differenzieren (:numref:`sec_autograd`) leitet die Gradienten direkt aus den Operationen des Vorwärtsdurchlaufs ab.

<!--en:1f2852f2-->
Rufen Sie ein Modul immer mit `net(X)` auf.

<!--en:f865e868-->
Ein direkter Aufruf von `forward` überspringt die Hooks, die `__call__` ausführt (:numref:`sec_repro`).

<!--en:4e77c132-->
Ein direkter Aufruf von `call` umgeht `__call__`. Diese Methode legt beim ersten Aufruf die bis dahin zurückgestellten Variablen an (:numref:`sec_lazy_init`).

<!--en:0f3eecde-->
## Modulcontainer

<!--en:4bc16e58-->
### Implementierung eines Containers

<!--en:262ec370-->
Ein sequenzieller Container hat nur zwei Aufgaben: Er registriert jedes Kindmodul beim Hinzufügen und reicht die Eingabe der Reihe nach durch alle Kindmodule. An einer eigenen Minimalversion sehen wir genau, wie die Registrierung funktioniert und wo Fehler auftreten können.

<!--en:de02ee59-->
`register_child` trägt einen Block ins Register ein. Da das Register schwache Referenzen speichert, hält `self._layers` die Blöcke am Leben.

<!--en:582c835a-->
`add_module` registriert jedes Kindmodul unter seinem Index als Zeichenkette. `children()` durchläuft das Register in Einfügereihenfolge.

<!--en:e82d118f-->
In Keras durchsucht `__setattr__` automatisch auch Python-Listen, Tupel und Dictionaries. So erfasst Keras jede Schicht in `self.modules`.

<!--en:f4c97958-->
`nnx.List` fasst eine Sequenz von Modulen zu einem Container zusammen, den der Graph unter `modules` erfasst.

<!--en:09aff31d-->
Unser eigener Container `MySequential` verhält sich wie der eingebaute:

<!--en:daa9bfc9-->
### Schichten in einer gewöhnlichen Liste

<!--en:324701e9-->
Was passiert, wenn wir Schichten in einer gewöhnlichen Python-Liste vom Typ `list` speichern? Ohne Kenntnis der Modulregistrierung liegt das nahe. Der Vorwärtsdurchlauf kann die Liste weiterhin in einer Schleife durchlaufen, doch eine gewöhnliche `list` ist kein Modul. Findet das Framework die darin enthaltenen Schichten trotzdem?

<!--en:47823bc2-->
Der folgende Block speichert seine Schichten in einer gewöhnlichen Liste. `net.initialize()` durchläuft das Register und erreicht sie daher nicht. Damit überhaupt ein Vorwärtsdurchlauf möglich ist, initialisieren wir jede Schicht von Hand:

<!--en:472bc7a3-->
In Keras speichert man eine Folge von Schichten üblicherweise in einer gewöhnlichen Liste, weil Keras beim Durchsuchen der Attribute auch deren Listenelemente prüft:

<!--en:6b853614-->
Der Vorwärtsdurchlauf gelingt, doch `collect_params()` ist leer. Bei einem ganz leeren Parameter-Dictionary bricht `Trainer` sofort ab. Bei einem *teilweise* registrierten Modell bleibt der Fehler beim Training unbemerkt: Die äußeren Schichten sind Attribute, die inneren liegen in einer gewöhnlichen Liste. Wir führen mit einem solchen Modell einen SGD-Schritt aus und prüfen, welche Gewichte sich ändern:

<!--en:07dcffb0-->
Der Vorwärtsdurchlauf gelingt, doch `net.parameters()` ist leer. Der Grund: `__setattr__` behandelt eine gewöhnliche Python-Liste vom Typ `list` als normales Attribut und erkennt darin keinen Modulcontainer. Ein leeres Modell weist `torch.optim.SGD` zurück. Tückischer ist ein *teilweise* registriertes Modell, dessen äußere Schichten Attribute sind, während die inneren in einer gewöhnlichen Liste liegen. Wir führen einen SGD-Schritt aus und prüfen, welche Gewichte sich ändern:

<!--en:67d15464-->
Da Keras auch den Inhalt von Python-Listen prüft, die Attributen zugewiesen sind, erfasst es automatisch alle 7946 Parameter ($20 \cdot 256 + 256 + 256 \cdot 10 + 10$). Einen anderen Strukturfehler verhindert Keras ausdrücklich: eine neue Schicht hinzuzufügen, *nachdem* das Modell bereits aufgebaut ist:

<!--en:f5bd1f24-->
Enthält eine gewöhnliche Python-Liste Module, löst NNX bereits beim Erzeugen des Modells einen `ValueError` aus. Die Fehlermeldung nennt `nnx.List` als Lösung. So kann kein teilweise registriertes Modell unbemerkt entstehen:

<!--en:1e4fd6f0-->
Um zu sehen, welche Schichten sich beim Training tatsächlich ändern, speichern wir Kopien aller vier Gewichtsmatrizen von `PartlyRegisteredMLP` in `before`. Wir führen einen einzigen Rückwärtsdurchlauf für `net(X).sum()` aus, gefolgt von einem SGD-Schritt. Dann vergleichen wir die Gewichte jeder Schicht mit `before`:

<!--en:73876837-->
Nur die äußeren Gewichte haben sich geändert. Die Backpropagation erreichte über `self.blocks` auch `self.first`, aber `Trainer` erhielt die Parameter der Liste nie (allerdings gibt `collect_params()` eine Warnung aus). Wir beheben den Fehler, indem wir jedes Kindmodul mit `register_child` registrieren:

<!--en:eda9f2e5-->
Nur die äußeren Gewichte haben sich geändert, ohne eine einzige Warnung oder Fehlermeldung. Die Backpropagation erreichte über `self.blocks` auch `self.first`, aber der Optimierer bekam die mittleren Schichten nie zu sehen. Ein Container vom Typ `nn.ModuleList` für die Liste (oder `nn.ModuleDict` für Dictionaries) registriert jedes Kindmodul korrekt:

<!--en:548bb7e4-->
Jetzt sind alle 7946 Parameter registriert. In Gluon registrieren Sie Kindmodule durch Zuweisung an ein Attribut, mit `nn.Sequential` oder mit `register_child`.

<!--en:75ce9400-->
Jetzt sind alle 7946 Parameter registriert. Verwenden Sie `nn.Sequential` für einfache Ketten. Verwenden Sie `nn.ModuleList` oder `nn.ModuleDict`, wenn Ihre Methode `forward` die Kindmodule selbst in einer Schleife durchläuft oder über einen Index anspricht.

<!--en:39321cce-->
In Keras müssen alle Schichten in `__init__` oder `build` hinzugefügt werden. Keras erfasst automatisch jede Schicht, die von einem Attribut aus erreichbar ist, auch innerhalb von Listen oder Dictionaries.

<!--en:864a3f69-->
Mit `nnx.List` (oder `nnx.Dict` für Zuordnungen) sind alle 7946 Parameter korrekt registriert.

<!--en:d9a91000-->
## Vorwärtsberechnung

<!--en:5922c9a1-->
Die Vorwärtsmethode eines Moduls ist gewöhnlicher Python-Code. Sie kann beliebige Tensoroperationen ausführen, Schleifen und Bedingungen verwenden und Pfade definieren, die Schichten überspringen. Wir nutzen diese Freiheit für eine Residualverbindung. Anschließend untersuchen wir, welche Einschränkungen gelten, wenn ein Framework den Trainingsschritt kompiliert.

<!--en:eac9140e-->
### Residualverbindungen

<!--en:2a0a81d5-->
Die einfachste weitverbreitete Abweichung von einer linearen Kette ist eine *Residualverbindung*. Sie addiert die Eingabe eines Blocks direkt zu seiner Ausgabe :cite:`He.Zhang.Ren.ea.2016*1`. Bezeichnen wir den *Hauptzweig* des Blocks mit $f$ (hier zwei vollständig verbundene Schichten mit einer ReLU dazwischen), so berechnet der Block

<!--en:6e87a024-->
Für $f(\mathbf{x}) = \mathbf{0}$ ergibt sich die identische Abbildung. Signale können so viele aufeinanderfolgende Blöcke ungehindert durchlaufen. Jeder Block lernt nur eine Verfeinerung $f(\mathbf{x})$, statt die Darstellung von Grund auf neu zu bilden. Die ResNet-Architekturen aus :numref:`sec_resnet` beruhen auf diesem Muster. Kein Framework braucht eine spezielle eingebaute Schicht für Residualverbindungen. In einem eigenen Modul genügt eine Zeile, `X + self.body(X)`:

<!--en:60de603c-->
![Ein Residualblock. Die Eingabe verzweigt sich in den Hauptzweig (lineare Schicht, ReLU, lineare Schicht) und einen Identitätspfad. Die Addition führt beide Zweige wieder zusammen.](../img/bg-residual-block.svg)

<!--en:3c8ec42d-->
:numref:`fig_bg_residual-block` zeigt die Verbindungen. Damit wir $\mathbf{x}$ und $f(\mathbf{x})$ addieren können, müssen ihre Formen übereinstimmen. Ein Residualblock erhält daher nur eine Breite `num_hiddens` für die versteckten Schichten.

<!--en:5a3453b6-->
Keras leitet die Eingabebreite beim Aufbau der Schicht ab. Daher muss der aufrufende Code die passende Eingabe liefern. Hat sie mehr oder weniger als `num_hiddens` Spalten, schlägt die Addition fehl.

<!--en:b3997517-->
Eine Vorwärtsmethode kann auch auf Werte zugreifen, die weder Eingaben noch trainierbare Parameter sind. Eine skalare Konstante lässt sich als gewöhnliches Python-Attribut speichern. Ein Tensor, dessen Werte wir aus Daten fortschreiben (etwa laufende Statistiken der Eingaben), muss dagegen zusammen mit dem Modell in Checkpoints gespeichert und auf andere Geräte übertragen werden. Der Optimierer darf ihn dabei nicht aktualisieren. Wie wir solche nicht trainierbaren *Puffer* (engl. buffers) registrieren, behandeln wir in :numref:`subsec_params_buffers`.

<!--en:df0d8080-->
### Sofortige und kompilierte Ausführung

<!--en:f4971e04-->
Bisher liefen unsere Vorwärtsdurchläufe wie gewöhnlicher Python-Code ab, eine Operation nach der anderen. Um den Zusatzaufwand durch Python zu vermeiden und Operationen zusammenzufassen, können Frameworks die Berechnung einmal aufzeichnen und bei späteren Aufrufen den kompilierten Graphen ausführen. Dieser Geschwindigkeitsgewinn hat seinen Preis: Die kompilierte Ausführung schränkt den datenabhängigen Kontrollfluss in Python innerhalb des Vorwärtsdurchlaufs ein.

<!--en:0f653f04-->
Sowohl `Block` in Gluon als auch `nn.Module` in PyTorch verwenden standardmäßig die *sofortige Ausführung* (engl. eager execution). Python führt eine Anweisung nach der anderen aus. Datenabhängiger Kontrollfluss funktioniert daher wie in NumPy (die Graphkompilierung behandeln wir in :numref:`chap_performance`).

<!--en:fecd9d53-->
Direkte Aufrufe werden sofort ausgeführt. Dagegen kompiliert `d2l.Trainer` die Trainingsschritte mit `tf.function`. Dazu zeichnet es `call` einmal anhand symbolischer Tensoren ohne konkrete Werte auf. Da Keras innerhalb von `Layer.call` kein AutoGraph verwendet, schlägt eine Python-Schleife mit `while` beim Aufzeichnen fehl, wenn ihre Bedingung von Tensorwerten abhängt:

<!--en:dd8fde7b-->
Direkte Aufrufe werden sofort ausgeführt. Dagegen kompiliert `d2l.Trainer` die Trainingsschritte mit `nnx.jit`. Dazu zeichnet es `__call__` einmal anhand abstrakter *Tracer* auf, die Formen und Datentypen, aber keine Werte enthalten. Hängt die Bedingung einer Python-Schleife mit `while` von einem Tracer ab, tritt ein Fehler auf:

<!--en:0c994920-->
Bei sofortiger Ausführung läuft die datenabhängige Schleife. Beim Aufzeichnen lässt sich dagegen weder ein symbolischer Tensor noch ein Tracer in einen Python-Wert vom Typ `bool` umwandeln. Hier ist `tf.while_loop` oder `jax.lax.while_loop` erforderlich. Schleifen über eine statische Modellstruktur, etwa über `self.modules`, lassen sich schon beim Aufzeichnen vollständig entrollen.

<!--en:83ca1c82-->
## Verzögerte Initialisierung

<!--en:e4682de1-->
### Formen aus den Daten

<!--en:042109af-->
Viele Bibliotheken erlauben es, beim Erzeugen einer Schicht nur ihre Ausgabedimension festzulegen. Die Schicht leitet ihre Eingabebreite aus dem ersten Minibatch ab.

<!--en:67665810-->
Vor dem ersten Vorwärtsdurchlauf legt `nn.Dense(256)` Platzhalter für die Parameter an. Die unbekannte Eingabedimension trägt den Wert `-1`:

<!--en:17f1dee5-->
Vor dem ersten Vorwärtsdurchlauf enthält `nn.LazyLinear(256)` einen Parameter vom Typ `UninitializedParameter`, dessen Form noch nicht feststeht:

<!--en:d7e74594-->
Vor dem ersten Aufruf hat `Dense(256)` noch keine Variablen angelegt:

<!--en:0463f247-->
NNX-Schichten verlangen explizite Eingabe- und Ausgabebreiten und legen ihre Parameter sofort in `__init__` an.

<!--en:58162174-->
Wenn wir den ersten Minibatch `X` durch das Netz schicken, ergibt sich die Form jeder Schicht aus der vorherigen. Dabei werden alle Gewichtstensoren angelegt:

<!--en:ca5ee176-->
### Initialisierung nach einem Probelauf

<!--en:c9e15c74-->
Bei Schichten mit verzögerter Initialisierung brauchen wir die Eingabedimensionen nicht von Hand von Schicht zu Schicht weiterzureichen. Gerade bei tiefen Modellen schleichen sich dabei leicht Fehler ein. Dafür stehen die Parameter erst zur Verfügung, wenn Daten durch das Netz fließen. Vor dem ersten Durchlauf können wir weder die Parameteranzahl prüfen noch formabhängige Verfahren wie die Xavier-Initialisierung anwenden (:numref:`subsec_xavier`). Für ein solches Modell müssen wir das Initialisierungsverfahren daher vorab hinterlegen oder nach einem *Probelauf* (engl. dry run) anwenden. Das ist ein einzelner Vorwärtsdurchlauf mit einem Minibatch aus Testwerten, der alle Formen festlegt und dessen Ausgabe wir verwerfen.

<!--en:c69436d6-->
`net.initialize(init.Xavier())` hinterlegt das Initialisierungsverfahren im Voraus und wendet es beim ersten Vorwärtsdurchlauf automatisch an.

<!--en:9ed31f31-->
`d2l.Module.apply_init` führt einen Vorwärtsdurchlauf ohne Erfassung der Gradienten aus, um die noch offenen Formen festzulegen. Anschließend ruft die Methode über `self.apply` eine Initialisierungsfunktion für jedes Modul im Baum auf:

<!--en:bffe870b-->
Keras-Schichten erhalten `kernel_initializer` beim Erzeugen und wenden das Verfahren an, wenn `build` läuft (beim ersten Aufruf mit Daten oder über `net.build((None, 20))`).

<!--en:76883333-->
`nnx.Linear` erhält `kernel_init` und zieht die Zufallswerte für die Gewichte sofort im Konstruktor.

<!--en:6df1d75e-->
`nn.Module.apply(fn)` ruft `fn` für jedes Modul im Baum auf, zuerst für die Kindmodule und dann für das übergeordnete Modul. Auch diese Aufgabe besteht also in einem Baumdurchlauf. Hier verwenden wir `apply_init` mit der gleichverteilten Variante der Xavier-Initialisierung:

<!--en:bb4f3193-->
Wann sollten wir Schichten mit verzögerter Initialisierung verwenden, wann explizite Dimensionen angeben? Für schnelle Prototypen mit unbekannten Datensätzen ist die verzögerte Initialisierung praktisch. Bei Modellen für den Produktiveinsatz möchten wir oft die Parameter zählen, sie programmgesteuert initialisieren oder das Modell ohne Testdaten aus einer Konfigurationsdatei erzeugen. Dafür ist es übersichtlicher, die Eingabe- und Ausgabebreiten explizit anzugeben. So verfahren wir im restlichen Kapitel.

<!--en:7c2e446f-->
## Modelle aus einer Konfiguration

<!--en:0109468f-->
### Konfigurationen und Aufbaufunktionen

<!--en:8d31b8df-->
Können wir ein trainiertes Modell allein aus seinen gespeicherten Gewichtstensoren rekonstruieren? Nicht ganz. Zu einem trainierten Modell gehören die numerischen Arrays und der Code, der sie interpretiert. Bisher haben wir alle Breiten und Tiefen direkt in den Konstruktoraufrufen fest vorgegeben. Eine Gewichtsdatei spiegelt diese Entscheidungen nur indirekt wider: Die Tensorformen verraten die Schichtbreiten. Ob wir ReLU oder tanh verwenden, geht aus den Arrays hingegen nicht hervor, weil diese Wahl keine Parameter erfordert.

<!--en:7b055e8e-->
Hyperparameter der Architektur, die sich von Experiment zu Experiment ändern, gehören daher in ein eigenes Objekt für die *Konfiguration* (kurz *Config*). Wir können es zusammen mit den Metriken protokollieren und neben den Checkpoints speichern. Die Topologie selbst bleibt im Code: Er legt fest, was jeder Block berechnet und wie die Blöcke verbunden sind. Im folgenden Beispiel hält eine Python-Klasse vom Typ `dataclass` vier Architektureigenschaften fest. Die Fabrikfunktion `build` erzeugt aus jeder Konfiguration einen vollständigen Modulbaum:

<!--en:e0efcad6-->
Eine einzige Konfiguration legt eine Instanz der Architektur vollständig fest: eine Eingabeprojektion von `num_inputs` auf `num_hiddens`, gefolgt von `num_blocks` Residualblöcken und einem Ausgabekopf. Beachten Sie, dass `blocks` zunächst eine temporäre Python-Liste ist. Das ist hier unbedenklich, weil das Entpacken in den sequenziellen Container jeden Block sofort registriert. Da jede Schicht sowohl ihre Eingabe- als auch ihre Ausgabebreite angibt, können wir die Parameter sofort und ohne Probelauf untersuchen und zählen.

<!--en:53d792fa-->
Das resultierende Modell bildet einen Minibatch aus 784-dimensionalen Vektoren (etwa zu Vektoren umgeformten Fashion-MNIST-Bildern) auf je 10 Ausgabe-Logits ab:

<!--en:5efeea02-->
### Bestimmung der Parameteranzahl

<!--en:ed901dd6-->
Aus der Konfiguration können wir die genaue Parameteranzahl auch auf dem Papier berechnen, bevor wir Speicher belegen. Die Eingabeprojektion hat $784 \cdot 256 + 256 = 200{,}960$ Parameter. Jeder Residualblock enthält zwei vollständig verbundene Schichten mit zusammen $2 \cdot (256 \cdot 256 + 256) = 131{,}584$ Parametern. Die Ausgabeschicht steuert weitere $256 \cdot 10 + 10 = 2{,}570$ bei. Das ergibt

<!--en:d6c17477-->
Parameter insgesamt. Den größten Anteil haben die Residualblöcke. Um das Modell zu vergrößern, brauchen wir jetzt nur zwei Konfigurationsfelder zu ändern. Der Modellcode bleibt derselbe. Ein Durchlauf durch den Parameterbaum bestätigt unsere Rechnung:

<!--en:00a3ae7f-->
Verdoppeln wir die Breite der versteckten Schichten, vervierfacht sich die Parameteranzahl innerhalb jedes Blocks näherungsweise. Doppelt so viele Blöcke verdoppeln dann die Summe der Blockparameter. Insgesamt enthält die größere Konfiguration daher mehr als sechsmal so viele Parameter. Bei großen Basismodellen in der Praxis gilt dasselbe Prinzip: Unser Sprachmodell mit 8 Milliarden Parametern ist durch eine kompakte Konfiguration (etwa 32 Blöcke, Breite der versteckten Schichten 4096) und den zugehörigen Code zum Aufbau des Modulbaums festgelegt.

<!--en:f818133d-->
Eine Konfiguration speichert die Hyperparameter. Der Implementierungscode liegt separat vor. Um ein Modell aus einem Checkpoint wiederherzustellen, brauchen wir alle drei Bestandteile: die Gewichte, die Konfiguration und den passenden Code mit der Funktion `build` und den Blockdefinitionen. In :numref:`sec_read_write` speichern wir Gewichte und Konfiguration zusammen. Den Code verwalten wir mit der Versionskontrolle.

<!--en:7bc672a3-->
Bisher haben wir Parameter nur aufgespürt und gezählt. In :numref:`sec_parameters` untersuchen wir die Tensoren eines Modells genauer, bestimmen den Speicherbedarf beim Training und sehen, wie sich Gewichte gemeinsam nutzen oder einfrieren lassen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:132156e8-->
Drei Grundgedanken bestimmen den Modellaufbau. Erstens bildet jedes neuronale Netz einen Modulbaum. Jeder Parameter ist über seinen Pfad von der Wurzel aus eindeutig ansprechbar. Zweitens durchlaufen Optimierer, Checkpoint-Verfahren und Operationen zum Gerätewechsel die registrierten Kindmodule. Schichten in einem unregistrierten Container werden daher beim Training und beim Speichern unbemerkt übergangen. Drittens lassen sich Architekturen leicht protokollieren, skalieren und reproduzieren, wenn wir variable Dimensionen in einer schlanken Konfiguration zusammenfassen und diese mit einer Funktion `build` kombinieren. Auf dieser Grundlage können wir in eigenen Vorwärtsmethoden die volle Flexibilität von Python nutzen. Bei kompilierter Ausführung gelten dabei die Einschränkungen der Graphaufzeichnung. Schichten mit verzögerter Initialisierung erleichtern schnelle Experimente, legen die Formen aber erst bei einem Probelauf fest.

<!--en:2294243a-->
## Übungen

<!--en:41107b3c-->
1. [code] **Unregistrierte Module.** Wegen der gewöhnlichen Liste in `PlainListMLP` erfassen `collect_params()`, `initialize()` und `save_parameters` die enthaltenen Schichten nicht. `collect_params()` gibt dazu eine Warnung aus. Sagen Sie voraus, wie sich zwei weitere Operationen auf das Modell auswirken, und überprüfen Sie Ihre Vorhersagen.

    1. Schicken Sie einen Batch durch ein neues `PlainListMLP`. Rufen Sie `net.cast('float64')` auf, prüfen Sie den Datentyp von `net.layers[0].weight.data()` und schicken Sie einen Batch vom Typ `float64` durch das Modell.
    1. Die Methode `load_parameters` ist das Gegenstück zu `save_parameters`. Sie liest eine gespeicherte Parameterdatei in ein Modell ein. Führen Sie mit einem `RegisteredListMLP` einen Vorwärtsdurchlauf aus und speichern Sie das Modell mit `save_parameters`. Laden Sie die Datei in ein neues `PlainListMLP`, indem Sie `load_parameters` aufrufen.
    1. Welche dieser Vorgänge scheitern unbemerkt und welche lösen einen Fehler aus? Erklären Sie beide Fälle anhand derselben fehlenden Registrierung.

<!--en:fc6800b8-->
1. [code] **Unregistrierte Module.** Wegen der gewöhnlichen Liste in `PlainListMLP` erfassen `net.parameters()`, `net.to(device)` und `net.state_dict()` die enthaltenen Schichten nicht. Sagen Sie voraus, wie sich drei weitere Operationen auf das Modell auswirken, und überprüfen Sie Ihre Vorhersagen.

    1. Rufen Sie `net.apply(init_xavier)` auf und prüfen Sie, ob sich das Gewicht von `net.layers[0]` ändert.
    1. Fügen Sie in der Liste `nn.Dropout(0.5)` hinter `nn.ReLU()` ein. Rufen Sie `net.eval()` auf und schicken Sie dieselbe Eingabe zweimal durch das Modell.
    1. Die Methode `load_state_dict` ist das Gegenstück zu `state_dict`. Sie kopiert ein Dictionary gespeicherter Tensoren in ein Modell. Rufen Sie `net.load_state_dict(ModuleListMLP().state_dict())` auf.
    1. Welche dieser Vorgänge scheitern unbemerkt und welche lösen einen Fehler aus? Erklären Sie alle drei Fälle anhand derselben fehlenden Registrierung.

<!--en:895beafc-->
1. [code] **Unregistrierte Module.** Keras erfasst eine Schicht, die der Konstruktor einem Attribut zuweist. Eine `Dense`-Schicht, die erst *innerhalb* von `call` entsteht, erfasst es dagegen nicht. Ein solches Modell läuft ohne Fehlermeldung. Schreiben Sie ein solches Modell, rufen Sie es auf und prüfen Sie anschließend `len(net.trainable_variables)`. Erklären Sie, was ein Optimierer trainieren würde und was zwischen zwei Aufrufen mit den Gewichten der Schicht geschieht.

<!--en:8d744c06-->
1. [code] **Unregistrierte Container.** NNX weist gewöhnliche Python-Listen zurück, die Module enthalten, und nennt `nnx.List` und `nnx.data` als Lösungen.

    1. Schreiben Sie zwei Varianten von `ListMLP`. Speichern Sie die beiden Schichten zunächst in einem `nnx.Dict` unter den Schlüsseln `'hidden'` und `'out'`. Kapseln Sie in der zweiten Variante die gewöhnliche Liste in `nnx.data`. Prüfen Sie, ob in beiden Fällen alle 7946 Parameter in `nnx.state(net, nnx.Param)` erscheinen.
    1. Vergleichen Sie die Parameterpfade beider Varianten mit denen von `ListMLP`. Ein Checkpoint speichert den Zustand als Zuordnung von Pfaden zu Arrays. Mit welcher der beiden Varianten könnten Sie einen Checkpoint von `ListMLP` wiederherstellen und warum?

<!--en:04a352de-->
2. **Verbreiterung eines Residualblocks.** `ResidualBlock` verlangt übereinstimmende Eingabe- und Ausgabebreiten. Angenommen, ein Block bildet die Breite $d$ auf eine größere Breite $d'$ ab. Sein Hauptzweig endet also in einer Schicht der Breite $d'$, und auch die Skip-Verbindung muss die Breite $d'$ liefern. Geben Sie für diese Verbindung eine Konstruktion mit Parametern und eine ohne Parameter an. Bestimmen Sie jeweils die Anzahl der zusätzlichen Parameter und geben Sie an, welche Ausgabekoordinaten der Identitätspfad weiterhin erreicht.

<!--en:e4d31e2a-->
3. [code] **Ein Modul von Grund auf.** ● Schreiben Sie in reinem Python und NumPy, ohne die Basisklasse eines Frameworks zu verwenden, eine Klasse `Parameter`, die ein Array kapselt. Schreiben Sie außerdem eine minimale Klasse `Module`, deren Methode `__setattr__` jedes zugewiesene Objekt vom Typ `Parameter` oder `Module` registriert. Geben Sie `Module` eine Methode `parameters()`, die rekursiv die registrierten Kindmodule durchläuft. Schreiben Sie eine Unterklasse `Linear` von `Module` mit einem Gewicht und einem Bias. Bauen Sie aus zwei Instanzen ein zweischichtiges MLP und prüfen Sie, ob `parameters()` alle vier Arrays zurückgibt. Speichern Sie dann die beiden Schichten in einer gewöhnlichen Liste und prüfen Sie, was `parameters()` zurückgibt. Erweitern Sie `Module` so, dass auch die Listenvariante alle vier Arrays findet.

    *Nach CMU 10-714, [Hausaufgabe 2](https://github.com/dlsyscourse/hw2), Frage 2.*

<!--en:e7aab761-->
4. [code] **Verzögerte Bestimmung der Eingabebreite.** Erstellen Sie einen Container vom Typ `nn.Sequential` mit `nn.Dense(256, activation='relu')` und `nn.Dense(10)`. Rufen Sie `initialize()` auf und schicken Sie zuerst einen Batch mit 20 Merkmalen hindurch, dann einen mit 30 Merkmalen. Erklären Sie das Ergebnis anhand dessen, was beim ersten Vorwärtsdurchlauf in jeder Schicht geschah. Beschreiben Sie anschließend ein Modell, das beide Eingabebreiten akzeptiert und für beide alle Schichten nach der ersten gemeinsam nutzt.

<!--en:f6963bdc-->
4. [code] **Verzögerte Bestimmung der Eingabebreite.** Erstellen Sie das Modell `net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))` und schicken Sie zuerst einen Batch mit 20 Merkmalen hindurch, dann einen mit 30 Merkmalen. Erklären Sie das Ergebnis anhand dessen, was beim ersten Vorwärtsdurchlauf in jeder Schicht geschah. Beschreiben Sie anschließend ein Modell, das beide Eingabebreiten akzeptiert und für beide alle Schichten nach der ersten gemeinsam nutzt.

<!--en:e598fbeb-->
4. [code] **Verzögerte Bestimmung der Eingabebreite.** Erstellen Sie einen Container vom Typ `tf.keras.Sequential` mit `tf.keras.layers.Dense(256, activation='relu')` und `tf.keras.layers.Dense(10)`. Schicken Sie zuerst einen Batch mit 20 Merkmalen hindurch, dann einen mit 30 Merkmalen. Erklären Sie das Ergebnis anhand dessen, was beim ersten Vorwärtsdurchlauf in jeder Schicht geschah. Beschreiben Sie anschließend ein Modell, das beide Eingabebreiten akzeptiert und für beide alle Schichten nach der ersten gemeinsam nutzt.

<!--en:550b8fd7-->
4. [code] **Gemeinsam genutzte Kindmodule.** Wenn wir dasselbe Kindmodul an mehreren Stellen verwenden, wird der Modulbaum aus :numref:`fig_blocks` zu einem Graphen. Bauen Sie ein kleines Netz, das dasselbe Kindmodul an zwei Stellen verwendet. Sagen Sie voraus, ob die Parameter des gemeinsam genutzten Moduls in `nnx.state(net, nnx.Param)` einmal oder zweimal erscheinen, und überprüfen Sie Ihre Vorhersage. Die Funktion `nnx.split(net)` zerlegt ein Modul in eine statische Graphdefinition und seinen Zustand. Aus beidem baut `nnx.merge` das Modul wieder auf. Verweisen beide Stellen nach dem Zerlegen und Wiederaufbau noch auf dasselbe Objekt? Prüfen Sie das mit `is`.

<!--en:aa93c36f-->
[Diskussionen](https://d2l.discourse.group/t/36752)

<!--en:2f7264e4-->
Module und Modellaufbau<br>**Modulbäume · Registrierung von Kindmodulen · Container · Residualblöcke · verzögerte Formbestimmung · Konfigurationen**

<!--en:b7c373ba-->
::: {.slide title="Warum Modelle eine rekursive Parameterverwaltung brauchen"}

<!--en:cd4a7eba-->
[Motivation und Größenordnung]{.kicker}

<!--en:e7fff520-->
- **Hunderte verteilte Gewichtstensoren:** Llama 3 8B enthält 32 aufeinanderfolgende Transformer-Blöcke mit je 9 Parameterarrays und 3 weitere Arrays außerhalb der Blöcke, insgesamt $32 \times 9 + 3 = 291$ verschiedene Tensoren.

<!--en:71cac550-->
- **Jeder Tensor zählt:** Jeden Tensor muss der Optimierer finden. Jeder Tensor muss in Checkpoints serialisiert, initialisiert und in den GPU-Speicher übertragen werden.

<!--en:62645375-->
- **Unbemerkter Fehler:** Fehlt ein Gewichtsarray im Modulregister, läuft der Vorwärtsdurchlauf auf der CPU fehlerfrei. Die Gewichte behalten ihre Anfangswerte.

<!--en:4a2b54c0-->
::: {.slide title="Pfade im Modulbaum eines Modells"}

<!--en:5cb42080-->
[Der Modulbaum]{.kicker}

<!--en:0471553d-->
- **Module auf jeder Ebene:** Jede Schicht, jeder zusammengesetzte Block und das gesamte Netz sind jeweils ein **Modul** mit eigenen Parametern, registrierten Kindmodulen und einem definierten Vorwärtsdurchlauf.

<!--en:d311ca34-->
- **Hierarchische Pfade:** Jede Kante trägt den Attributnamen oder Index, unter dem das übergeordnete Modul sein Kindmodul registriert hat. Der Pfad von der Wurzel benennt jeden Parameter an einem Blatt eindeutig (z. B. Block 1, Schicht 0, Gewicht).

<!--en:cf1e22e5-->
::: {.slide title="Verkettung registrierter Kindmodule im sequenziellen Container"}

<!--en:4e6489ec-->
[Das Modul als Abstraktion]{.kicker}

<!--en:1ea22477-->
Bei einer einfachen Kette von Schichten registriert ein sequenzieller Container jedes Kindmodul unter einem ganzzahligen Index und reicht die Aktivierungen in Einfügereihenfolge weiter:

<!--en:0778c93d-->
::: {.slide title="Registrierung von Attributen in eigenen Modulunterklassen"}

<!--en:20141fff-->
[Ableitung vom Basismodul]{.kicker}

<!--en:0a1bc61c-->
- **Automatische Registrierung mit `__setattr__`:** Die Zuweisung `self.hidden = ...` trägt das Kindmodul ins Register des Moduls ein. Immer `net(X)` aufrufen, damit Hooks und Aufbaumethoden ausgeführt werden.

<!--en:b4337564-->
::: {.slide title="Das Modulregister bei Operationen am gesamten Modell"}

<!--en:3ce9c1fb-->
[Bedeutung der Registrierung]{.kicker}

<!--en:885a4b19-->
- **Fünf rekursive Baumdurchläufe:** Beim Einrichten des Optimierers, beim Checkpointing, beim Gerätewechsel, beim Initialisieren und beim Umschalten von Modi oder Hooks werden die registrierten Kindmodule durchlaufen.

<!--en:58cac2e5-->
- **Folgen einer unregistrierten Schicht:** Ihre Gewichte ändern sich nie, fehlen in Checkpoints und bleiben beim Wechsel auf die GPU zurück. Dropout und BatchNorm bleiben auch bei der Auswertung im Trainingsmodus.

<!--en:7523c68d-->
::: {.slide title="Registrierung und Schleife im sequenziellen Container" except="mxnet"}

<!--en:30c46ea7-->
[Aufbau eines Containers]{.kicker}

<!--en:01861dea-->
- Ein sequenzieller Container registriert lediglich seine Kindmodule in `__init__` und durchläuft sie in `forward`:

<!--en:d7538afc-->
::: {.slide title="Registrierung und Schleife im sequenziellen Container" only="mxnet"}

<!--en:30c46ea7-2-->
[Aufbau eines Containers]{.kicker}

<!--en:e0090283-->
- Ein sequenzieller Container registriert lediglich seine Kindmodule in `add` und durchläuft sie in `forward`.

<!--en:4ad2f89a-->
::: {.slide title="Ein sequenzieller Container im Code" only="mxnet"}

<!--en:30c46ea7-3-->
[Aufbau eines Containers]{.kicker}

<!--en:5c9d74f5-->
::: {.slide title="Für PyTorch unsichtbare Schichten in gewöhnlichen Python-Listen" only="pytorch"}

<!--en:2676f33c-->
[Die Falle bei gewöhnlichen Listen]{.kicker}

<!--en:737ab418-->
- Mit Schichten in einer gewöhnlichen Python-Liste vom Typ `list` (`self.layers = [...]`) läuft `forward`, doch `net.parameters()` bleibt leer (`0` statt `7,946` Parameter):

<!--en:b4dc338f-->
::: {.slide title="Für Gluon unsichtbare Schichten in gewöhnlichen Python-Listen" only="mxnet"}

<!--en:2676f33c-2-->
[Die Falle bei gewöhnlichen Listen]{.kicker}

<!--en:2cc63625-->
- Schichten in einer gewöhnlichen Python-Liste vom Typ `list` bleiben für `net.initialize()` und `net.collect_params()` unsichtbar (`0` statt `7,946` Parameter):

<!--en:5f768681-->
::: {.slide title="Unbemerkt eingefrorene innere Schichten bei teilweiser Registrierung" only="pytorch,mxnet"}

<!--en:a403ba1f-->
[Ein unbemerkter Trainingsfehler]{.kicker}

<!--en:85d9f12b-->
Äußere Schichten als Attribute, mittlere Schichten in einer gewöhnlichen Liste: Der Optimierer akzeptiert das Modell ohne Fehlermeldung.

<!--en:587095de-->
- **Gradienten fließen, Gewichte bleiben eingefroren:** Die Gewichte von `self.first` ändern sich (`True`). Der Optimierer hat `self.blocks` nie erhalten, daher bleiben die Gewichte dieser Schichten unverändert (`False, False`).

<!--en:60277e89-->
::: {.slide title="Prüfung von Listen in Python-Attributen durch Keras" only="tensorflow"}

<!--en:b6fc425e-->
[Containerregistrierung in Keras]{.kicker}

<!--en:b38d91c8-->
- In Keras prüft `__setattr__` automatisch den Inhalt von Python-Listen, Tupeln und Dictionaries, die Modellattributen zugewiesen sind. So erfasst Keras alle $7{,}946$ Parameter:

<!--en:4a6a0e66-->
::: {.slide title="Gewöhnliche Modullisten schon beim Aufbau in Flax NNX unzulässig" only="jax"}

<!--en:a68cc404-->
[Containerregistrierung in NNX]{.kicker}

<!--en:c5a46a49-->
- Flax NNX prüft die Attribute in `__init__` und löst sofort einen `ValueError` aus, wenn eine gewöhnliche Python-Liste ein `nnx.Module` enthält:

<!--en:7de00c86-->
::: {.slide title="Korrekte Registrierung von Listen mit Modulcontainern" only="pytorch"}

<!--en:a299d511-->
[Container für registrierte Sammlungen]{.kicker}

<!--en:f4064094-->
- Dynamische Sammlungen von Schichten in `nn.ModuleList` (oder `nn.ModuleDict`) kapseln, damit der Modulbaum alle $7{,}946$ Parameter erfasst:

<!--en:5c23b3b7-->
::: {.slide title="Korrekte Registrierung von Listen mit register_child" only="mxnet"}

<!--en:a299d511-2-->
[Container für registrierte Sammlungen]{.kicker}

<!--en:93884adb-->
- Jeden Block einer Python-Liste mit `self.register_child(layer)` registrieren, damit alle $7{,}946$ Parameter in `net.collect_params()` erscheinen:

<!--en:8a043e9c-->
::: {.slide title="Keine zusätzlichen Schichten nach dem Modellaufbau in Keras" only="tensorflow"}

<!--en:daeb14b0-->
[Strukturelle Integrität in Keras]{.kicker}

<!--en:7d1f0e4e-->
- Keras sperrt den Modulbaum, sobald die Variablen angelegt sind. Wer danach eine Schicht hinzufügt, löst sofort einen `ValueError` aus:

<!--en:13947ed0-->
::: {.slide title="Korrekte Registrierung von Listen mit nnx.List" only="jax"}

<!--en:a299d511-3-->
[Container für registrierte Sammlungen]{.kicker}

<!--en:7b7d3c1c-->
- Eine Sequenz von Schichten in `nnx.List` (oder `nnx.Dict`) kapseln, damit jedes Kindmodul in `nnx.state(net, nnx.Param)` erfasst wird:

<!--en:248c6e17-->
::: {.slide title="Eigene Vorwärtsdurchläufe mit Residualverbindung"}

<!--en:cebd09f3-->
[Verbindungen jenseits einfacher Ketten]{.kicker}

<!--en:3b4028d2-->
Die Vorwärtsmethode eines Moduls erlaubt beliebige Verzweigungen und Pfade, die Schichten überspringen. Ein Beispiel ist ein **Residualblock**, der seine Eingabe direkt zum Ergebnis einer zweischichtigen Transformation $f(\mathbf{x})$ addiert:

<!--en:59ce64d9-->
- **Identität als Ausgangspunkt:** Für $f(\mathbf{x}) = \mathbf{0}$ wirkt der Block als identische Abbildung $\mathbf{y} = \mathbf{x}$. So können viele aufeinanderfolgende Blöcke schrittweise Verfeinerungen lernen.

<!--en:d90e2711-->
- **Gleiche Form:** Die elementweise Addition verlangt für $\mathbf{x}$ und $f(\mathbf{x})$ exakt dieselbe Merkmalsbreite `num_hiddens`.

<!--en:8fe76e64-->
::: {.slide title="Ein wiederverwendbares ResidualBlock-Modul"}

<!--en:a7eba2f1-->
[Vorwärtsberechnung in Python]{.kicker}

<!--en:bdf52681-->
Eine Skip-Verbindung braucht keine spezielle Operation des Frameworks. Die Vorwärtsmethode gibt `X + self.body(X)` zurück:

<!--en:31234696-->
::: {.slide title="Grenzen der Aufzeichnung bei sofortiger und kompilierter Ausführung" only="tensorflow,jax"}

<!--en:fa83dc38-->
[Grenzen der Graphkompilierung]{.kicker}

<!--en:38f183d1-->
- `d2l.Trainer` kompiliert Trainingsschritte, indem es `forward` einmal anhand symbolischer Tensoren aufzeichnet:

<!--en:ee85e00e-->
::: {.slide title="Sofortige Ausführung und Aufzeichnung des Graphen" only="pytorch,mxnet"}

<!--en:2c5d34c6-->
[Ausführungsmodi der Vorwärtsberechnung]{.kicker}

<!--en:dbc205ea-->
- **Sofortige Ausführung (Standard):** Jede Zeile arbeitet mit konkreten Werten. Datenabhängige Verzweigungen mit `if`, Schleifen mit `while` und die Fehlersuche mit `print` funktionieren direkt.

<!--en:7a92b5da-->
- **Aufzeichnung für die Graphkompilierung:** Statische Schleifen über registrierte Module (`for block in self.blocks:`) lassen sich vollständig entrollen. Datenabhängige Verzweigungen erfordern symbolischen Kontrollfluss.

<!--en:bbf5b9e4-->
::: {.slide title="Bestimmung der Eingabeformen beim ersten Batch" except="jax"}

<!--en:b33592e8-->
[Verzögerte Formbestimmung]{.kicker}

<!--en:4e787cee-->
- Bei verzögerter Initialisierung ist beim Erzeugen der Schicht nur die Ausgabebreite nötig. Die Gewichtsmatrizen entstehen beim ersten Vorwärtsdurchlauf:

<!--en:4ef4c689-->
::: {.slide title="Explizite Formen beim Aufbau in Flax NNX" only="jax"}

<!--en:a85c5454-->
[Sofortiges Anlegen der Parameter]{.kicker}

<!--en:5e425922-->
- Flax-NNX-Schichten verlangen Eingabe- und Ausgabebreiten (`nnx.Linear(20, 256, rngs=rngs)`) und legen die Gewichte sofort in `__init__` an:

<!--en:31e29cc0-->
::: {.slide title="Verzögerte Initialisierung von Modellen nach einem Probelauf" except="jax"}

<!--en:ea9b63f7-->
[Festlegung der Formen vor dem Training]{.kicker}

<!--en:cf6054be-->
- Initialisierungsverfahren mit Varianzskalierung hängen von $n_{\text{in}}$ und $n_{\text{out}}$ ab. Vor einer eigenen Initialisierung brauchen Schichten mit verzögerter Formbestimmung daher einen Vorwärtsdurchlauf mit Testdaten (**Probelauf**):

<!--en:081f8eff-->
::: {.slide title="Trennung von Architekturkonfiguration und Aufbaufunktion"}

<!--en:ee5e5887-->
[Konfigurationen als Datenklassen]{.kicker}

<!--en:f23684f3-->
- Code für den Produktiveinsatz fasst die Hyperparameter in einer serialisierbaren `@dataclass` zusammen und kombiniert sie mit einer Funktion `build`.

<!--en:fdacb501-->
::: {.slide title="Modellaufbau aus einer Konfiguration"}

<!--en:ee5e5887-2-->
[Konfigurationen als Datenklassen]{.kicker}

<!--en:6609d151-->
::: {.slide title="Exakte Parameteranzahl aus der Konfiguration"}

<!--en:1ae01bc5-->
[Rechnung und Kontrolle durch Baumdurchlauf]{.kicker}

<!--en:53383120-->
Die expliziten Breiten in `MLPConfig(num_inputs=784, num_hiddens=256, num_blocks=4, num_outputs=10)` erlauben es, die genaue Parameteranzahl auf dem Papier zu berechnen und am Modulbaum zu prüfen:

<!--en:b34552a3-->
| Modellkomponente | Formel aus den Dimensionen ($W + b$) | Parameteranzahl |

<!--en:3bde9471-->
| **Eingabeprojektion (`0`)** | $784 \times 256 + 256$ | $200{,}960$ |

<!--en:cf8e2b33-->
| **4 Residualblöcke (`1..4`)** | $4 \times 2 \times (256 \times 256 + 256)$ | $526{,}336$ |

<!--en:b25b68da-->
| **Ausgabekopf (`5`)** | $256 \times 10 + 10$ | $2{,}570$ |

<!--en:4353a49c-->
| **Gesamtes Netz** | **10 lineare Schichten (20 Parametertensoren)** | **$729{,}866$** |

<!--en:04f1ab25-->
::: {.slide title="Verständnisfragen zu Modulbäumen und Parameteranzahl"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:3fc4d349-->
Beide Fragen anhand der Architektur von `MLPConfig` beantworten ($d_{\text{in}}=784$, $K$ Residualblöcke der Breite $d$, $d_{\text{out}}=10$):

<!--en:117f79b5-->
1. Angenommen, die 4 Residualblöcke (`d = 256`) liegen in einer gewöhnlichen Python-Liste vom Typ `list` und das Framework durchsucht keine Listen. Die Eingabe- und die Ausgabeschicht sind den Attributen `self.first` und `self.last` zugewiesen. Wie viele Parameter erhält der Optimierer, und was passiert auf einer GPU?

<!--en:87edfa2f-->
2. Angenommen, sowohl `num_hiddens` ($256 \to 512$) als auch `num_blocks` ($4 \to 8$) verdoppeln sich. Warum steigt die Gesamtzahl der Parameter um einen Faktor von mehr als $6\times$ ($729{,}866 \to 4{,}609{,}546$)?

<!--en:557b98b6-->
**Lösungen:**

<!--en:17c6a71a-->
1. **Nur $200{,}960 + 2{,}570 = 203{,}530$ Parameter** erreichen den Optimierer. Die $526{,}336$ Blockparameter bleiben eingefroren. Der Aufruf von `.to(gpu)` verschiebt nur die äußeren Schichten. Der erste Vorwärtsdurchlauf bricht ab, weil die Tensoren auf unterschiedlichen Geräten liegen (CPU und GPU).

<!--en:c4164c9a-->
2. Die Anzahl der Gewichte pro Residualblock wächst **quadratisch** mit der Breite ($2 d^2$), ihre Gesamtzahl linear mit der Tiefe ($K$). So wächst die Zahl der Blockparameter näherungsweise um den Faktor $2^2 \times 2 = 8\times$ ($526{,}336 \to 4{,}202{,}496$).

<!--en:dd3ba2d2-->
::: {.slide title="Zusammenfassung zu Modulen und Modellaufbau"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:d4235c82-->
| Konzept | Mechanismus oder Regel | Praktische Auswirkung |

<!--en:218dbe90-->
| **Modulbaum** | Rekursive Hierarchie aus Schichten und Blöcken mit Parametern an den Blättern | Eindeutiger Parametername aus dem Pfad von der Wurzel zum Blatt, mit Punkten als Trennzeichen |

<!--en:d70f46d6-->
| **Registrierung** | `__setattr__` und registrierte Modulcontainer | Unregistrierte gewöhnliche Listen verbergen Schichten vor Optimierern, Checkpoints und GPUs |

<!--en:d32c19e2-->
| **Eigene Methode `forward`** | Beliebiger Python-Code für Tensoren (z. B. $\mathbf{x} + f(\mathbf{x})$), Aufruf über `net(X)` | In kompilierten Schritten sind Python-Verzweigungen anhand von Tensorwerten unzulässig |

<!--en:f3d500f0-->
| **Schichten mit verzögerter Initialisierung** | $n_{\text{in}}$ aus dem ersten Minibatch ableiten, nach einem Probelauf initialisieren | Praktisch für schnelle Prototypen, für den Produktiveinsatz sind explizite Breiten vorzuziehen |

<!--en:fa2f20b7-->
| **Konfiguration und Aufbaufunktion** | Serialisierbare Hyperparameter in einer `@dataclass` und Fabrikfunktion `build(cfg)` | Zur Rekonstruktion sind Gewichte, Konfiguration und Code der Aufbaufunktion nötig |
