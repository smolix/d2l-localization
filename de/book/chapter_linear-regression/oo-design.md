<!--en:8386c2f6-->
# Objektorientierter Entwurf für die Implementierung

<!--en:0b062db3-->
In :numref:`sec_linear_scratch` haben wir die lineare Regression direkt mit Tensoren implementiert. Für das Training genügte eine eigenständige Funktion `train` mit zwölf Zeilen, zugeschnitten auf ein Modell und einen Datensatz. Als Nächstes befassen wir uns mit mehrschichtigen Perzeptronen, konvolutionären Netzen, rekurrenten Netzen und Transformern. Würden wir in jedem Abschnitt erneut Datenlader (engl. data loader), Epochenschleifen, Hilfsfunktionen zum Mitteln von Metriken und die laufende grafische Darstellung programmieren, gingen die neuen Ideen in all den Wiederholungen unter.

<!--en:5bbb8438-->
Wir wollen uns auf die Modellierung konzentrieren und dennoch jede Zeile unseres Trainingsgerüsts (engl. training harness) nachvollziehen und ändern können. Deshalb gliedern wir die Bibliothek `d2l` in drei objektorientierte Abstraktionen mit klaren Aufgaben (:numref:`fig_oo_design`). Als Vorbild dienen Frameworks wie PyTorch Lightning :cite:`Falcon.ea.2019`.

<!--en:2af4f292-->
1. **`Module` (Modelle):** Enthält die trainierbaren Parameter, definiert den Vorwärtsdurchlauf (engl. forward pass) mit `forward` und berechnet den Verlust mit `loss`. Die Klasse legt `training_step` und `validation_step` fest und gibt über `configure_optimizers` ihren Optimierer zurück.

<!--en:c0ea323f-->
2. **`DataModule` (Daten):** Kapselt das Herunterladen, Vorverarbeiten und Aufteilen des Datensatzes. Über `train_dataloader()` und `val_dataloader()` stellt die Klasse iterierbare Objekte bereit.

<!--en:7565dc6c-->
3. **`Trainer` (Trainingsschleife):** Verbindet ein beliebiges `Module` mit einem beliebigen `DataModule`. In jeder Epoche steuert die Klasse Training und Validierung anhand von Minibatches, mittelt die Metriken über die Batches und zeichnet die Lernkurven fortlaufend auf einem `ProgressBoard`.

<!--en:d9bad7a4-->
![Die drei zentralen Klassen des Trainingsgerüsts von `d2l`: `DataModule` liefert Minibatches für Training und Validierung, `Module` definiert Modell, Verlust und Optimierer, und `Trainer` steuert die Trainings- und Validierungsschleifen.](../img/mdl-linreg-oo-classes.svg)

<!--en:e0ac1db6-->
## Hilfsmittel

<!--en:9605999e-->
Bevor wir die drei zentralen Klassen entwickeln, führen wir zwei kleine Hilfsmittel in Python ein. Damit bleiben die Definitionen im Notebook gut lesbar und in sich geschlossen.

<!--en:1896af79-->
1. **`@add_to_class(Class)`:** Stehen in einem Lehrbuch oder Notebook alle 60 Zeilen einer Klasse in einem einzigen Codeblock, lässt sich jede Methode nur schwer dort erklären, wo sie eingeführt wird. Mit dem Dekorator `@add_to_class(Class)` definieren wir zunächst das Grundgerüst der Klasse und fügen ihr in späteren Codezellen nach und nach Methoden hinzu.

<!--en:96adce25-->
2. **`HyperParameters` (`save_hyperparameters()`):** Zuweisungen wie `self.a = a; self.b = b; ...` am Anfang jeder `__init__`-Methode sind reine Schreibarbeit. Wir leiten die Klasse von `HyperParameters` ab und rufen `self.save_hyperparameters()` auf. Die Methode untersucht den Aufrufrahmen und speichert jedes Argument von `__init__` (außer den in `ignore` aufgeführten) automatisch als Attribut von `self`.

<!--en:e994fee1-->
Probieren wir den Dekorator an einer kleinen Beispielklasse `A` und einer bereits erzeugten Instanz `a` aus.

<!--en:edd11414-->
Nun fügen wir in einer eigenen Zelle eine neue Methode `do` hinzu. Python sucht Methoden erst beim Aufruf in der Klasse. Deshalb kann auch unsere bereits vorhandene Instanz `a` sofort auf `do` zugreifen.

<!--en:54bd4e3f-->
Unser zweites Hilfsmittel erspart uns in Konstruktoren wiederkehrende Zuweisungen wie `self.lr = lr` und `self.batch_size = batch_size`. Da im Deep Learning die meisten Konstruktorargumente Hyperparameter sind, leiten wir unsere Klassen von `d2l.HyperParameters` ab und rufen `self.save_hyperparameters()` auf. Die Methode speichert jedes Konstruktorargument (außer den in `ignore` aufgeführten) automatisch als Attribut von `self`. Dazu untersucht sie mit dem Python-Modul `inspect` den Aufrufrahmen der aufrufenden Funktion. Ihre Implementierung findet sich in :numref:`sec_utils`.

<!--en:a7b9d20c-->
Alle Klassen, die wir im weiteren Verlauf dieses Abschnitts entwickeln, erben von `HyperParameters`.

<!--en:0281e568-->
## Modelle

<!--en:ce374478-->
Die Klasse `Module` ist die Basisklasse aller Modelle, die wir implementieren. Sie erbt sowohl von der Containerklasse des jeweiligen Frameworks für neuronale Netze als auch von `d2l.HyperParameters`. Standardmäßig gilt:

<!--en:7667c9ac-->
- Ein Aufruf von `model(X)` ruft `model.forward(X)` auf.

<!--en:8a22a3a6-->
- `training_step(batch)` übergibt die Merkmale `batch[:-1]` an das Modell, vergleicht die Vorhersagen mit den Labels `batch[-1]` und gibt den mittleren Verlust des Minibatchs zurück. Spätere Modelle können ein Wörterbuch mit Metriken zurückgeben, das unter anderem `'loss'` enthält.

<!--en:e03db1cb-->
- `validation_step(batch)` berechnet denselben Verlust für einen Minibatch von Validierungsdaten. In :numref:`sec_softmax_scratch` überschreibt `Classifier` die Methode `validation_step`, um sowohl `'loss'` als auch `'acc'` zurückzugeben.

<!--en:29615dca-->
- Die Hilfsfunktion `as_metrics` bringt einen einzelnen Skalar oder ein Wörterbuch als Rückgabewert in ein einheitliches Format, ein Wörterbuch mit Metriken. So kann `Trainer` beliebige benannte Größen auf dieselbe Weise protokollieren.

<!--en:9d0d26c2-->
In MXNet dient `nn.Block` als Container.

<!--en:e5df3e7f-->
In PyTorch dient `nn.Module` als Container.

<!--en:81762897-->
In TensorFlow dient `tf.keras.Model` als Container.

<!--en:1ef8561b-->
In JAX dient `nnx.Module` aus Flax als Container.

<!--en:1646ff34-->
Nach unserer Konvention ist jeder Minibatch ein Tupel. Sein letztes Element ist der Labeltensor, alle vorherigen Elemente sind die Modelleingaben. Sowohl `training_step` als auch `validation_step` entpacken mit `*batch[:-1]` die Eingaben für den Aufruf `self(...)`. Sie berechnen den Verlust anhand von `batch[-1]` und *geben das Ergebnis zurück*. Sie schreiben nichts auf den Bildschirm oder ins Protokoll und zeichnen keine Kurven. Wir haben die Modelle bewusst so entworfen, dass sie nichts über den Trainer wissen müssen. So kann jede selbst geschriebene Schleife ein `Module` ohne versteckte Seiteneffekte trainieren oder auswerten.

<!--en:1973420a-->
In diesem Kapitel gibt jeder Schritt einen einzelnen Skalar zurück, den mittleren Verlust des Minibatchs. Ab :numref:`sec_softmax_scratch` kann ein Schritt auch ein Wörterbuch zurückgeben, das jedem Metriknamen den Mittelwert für den Minibatch zuordnet. Es enthält immer `'loss'`, daneben auch Größen wie `'acc'`. Unsere kleine Hilfsfunktion `as_metrics` bringt einen einzelnen skalaren Verlust in die Form `{'loss': out}`. So kann der Trainer beide Fälle einheitlich behandeln.

<!--en:af9c78fb-->
### Lineare Regression als Modul

<!--en:8be254f5-->
Sehen wir uns zunächst an, wie `Module` in der Praxis funktioniert, bevor wir die eingebauten Schichten der Frameworks verwenden. Dazu fassen wir die selbst implementierten Bestandteile der linearen Regression in einer Klasse zusammen: die Parameter $(\mathbf{w}, b)$, den Vorwärtsdurchlauf, den quadratischen Verlust und die Aktualisierungsregel `SGD`. Die Klasse heißt `LinearRegressionScratch(d2l.Module)`:

<!--en:cda42036-->
Als Nächstes fügen wir `forward` und `loss` hinzu. Sie entsprechen `linreg` und `squared_loss` aus :numref:`sec_linear_scratch`, lesen aber `self.w` und `self.b` direkt aus der Modellinstanz.

<!--en:a6686047-->
Nun kapseln wir unsere selbst implementierte Aktualisierungsregel für Minibatch-SGD in der wiederverwendbaren Klasse `SGD(d2l.HyperParameters)`. In `LinearRegressionScratch.configure_optimizers` geben wir eine Instanz dieser Klasse zurück.

<!--en:d89ef5f2-->
Schließlich verbindet `configure_optimizers` das Modell mit unserem Optimierer `SGD`:

<!--en:69e3dc6d-->
Wir übergeben `SGD` die zugrunde liegenden Arrays aus dem Aufruf von `.data()`. In ihren Attributen `.grad` werden bei der Backpropagation die Gradienten gespeichert.

<!--en:864d49c6-->
## Daten

<!--en:44c6f214-->
Die Basisklasse `DataModule` ist das Gegenstück zu `Module` für Datensätze. Jede Unterklasse für einen konkreten Datensatz implementiert `train_dataloader()` und `val_dataloader()`. Meist delegieren beide Methoden ihre Arbeit an `get_dataloader(train=True/False)`.

<!--en:618e6ee5-->
Fassen wir unseren Generator für synthetische Regressionsdaten aus :numref:`sec_synthetic-regression-data` in der Klasse `SyntheticRegressionData(d2l.DataModule)` zusammen. Sie erzeugt vorab `num_train + num_val` Beispiele (standardmäßig 1000 Trainingsbeispiele und 1000 Validierungsbeispiele). So kann `Trainer` nach jeder Epoche sowohl den Trainingsverlust als auch den Validierungsverlust auf den zurückgehaltenen Daten verfolgen.

<!--en:0bf27647-->
### Ein Datenlader für jede Epoche

<!--en:1be1b249-->
In `get_dataloader` brauchen wir ein Objekt, das sich in jeder Epoche einmal durchlaufen lässt. Zu Beginn jeder Epoche soll es die Trainingsdaten neu mischen, die Validierungsdaten dagegen in deterministischer Reihenfolge lesen. Die Methode `get_tensorloader` erzeugt dieses Objekt aus den Tensoren der jeweiligen Teilmenge.

<!--en:906a7b29-->
Wir implementieren das Objekt selbst als Klasse `ArrayLoader`. Darin verwenden wir den Code aus :numref:`sec_linear_scratch`, der die Indizes permutiert und sie dann Slice für Slice verarbeitet.

<!--en:3f5404da-->
`TensorDataset` ordnet die Tensoren entlang ihrer ersten Achse einander zu, der Achse der Beispiele. `DataLoader` zerlegt sie in Minibatches und mischt die Beispiele bei jedem Durchlauf automatisch neu, sofern `shuffle=True` gilt.

<!--en:f3d180ce-->
`from_tensor_slices` führt die jeweils zusammengehörigen Beispiele aus den Tensoren zusammen. `shuffle` zieht Beispiele aus einem Puffer. Für eine gleichmäßige Zufallsmischung der Trainingsdaten umfasst der Puffer die gesamte Teilmenge, für die Validierung genügt ein Puffer der Größe `1`, damit die Reihenfolge erhalten bleibt. `batch` fasst die Beispiele zu Minibatches zusammen.

<!--en:8a44c864-->
`get_tensorloader` wählt die angeforderte Teilmenge per Slice aus und kapselt die Tensoren in einem `ArrayLoader`.

<!--en:0c2236b3-->
`get_tensorloader` wählt die angeforderte Teilmenge per Slice aus und kapselt die Tensoren in einem `ArrayLoader`. Der `DataLoader` von Gluon setzt Minibatches Beispiel für Beispiel zusammen. `ArrayLoader` dagegen liest mit einem einzigen vektorisierten Slice pro Tensor einen ganzen Batch.

<!--en:2fb4a44d-->
Für `SyntheticRegressionData.get_dataloader` genügen jetzt zwei Zeilen. Die Methode wählt die Trainings- oder Validierungsdaten per Slice aus und übergibt sie an `get_tensorloader`.

<!--en:0e6b5e14-->
Erzeugen wir eine Instanz unseres synthetischen Datensatzes. Wiederholen wir damit den Test mit zwei Durchläufen, an dem unser einfacher Generator zuvor gescheitert ist.

<!--en:6b833440-->
Diesmal liefern beide Durchläufe alle $32$ Minibatches, einschließlich des letzten mit den restlichen $8$ Beispielen. Jeder Durchlauf beginnt mit einer neuen zufälligen Reihenfolge. Außerdem können wir mit `len(loader)` die Länge direkt abfragen, ohne den Datensatz zu durchlaufen.

<!--en:4723154f-->
## Training

<!--en:0148ef68-->
### Trainingsschleife und Trainingsverlauf

<!--en:f912df6a-->
Nun entwickeln wir `Trainer`, die Klasse, die ein `Module` mit einem `DataModule` verbindet. Der Aufruf `trainer.fit(model, data)` bereitet die Datenlader vor, bindet das Modell und seinen Optimierer ein und startet `max_epochs` Durchläufe. In jeder Epoche durchläuft `fit_epoch` die Trainingsminibatches, aktualisiert die Parameter und mittelt die Trainingsmetriken. Anschließend folgt der Aufruf von `validate` für die Validierungsminibatches. Der Trainer speichert die Epochenmittelwerte in `self.history` und aktualisiert die Kurven auf dem `ProgressBoard`.

<!--en:7bbf9706-->
Für die Lernratenpläne aus :numref:`sec_scheduler` kann `prepare_optimizer` auch ein optionales Tupel `(optimizer, scheduler)` entpacken, falls `configure_optimizers` ein solches zurückgibt.

<!--en:ff8d32c8-->
`prepare_model` erzeugt zwei Sichten auf das Modell mit denselben Parametern, eine für das Training und eine für die Auswertung. So arbeiten Schichten wie Dropout und Batchnormalisierung jeweils im richtigen Modus. `prepare_optimizer` kapselt die Gradiententransformation von Optax in `nnx.Optimizer` und speichert die Struktur der Transformation zwischen. Bei weiteren Trainingsläufen lassen sich die bereits JIT-kompilierten Kernel wiederverwenden.

<!--en:14782a83-->
Die Methode `fit(model, data)` implementiert dieselbe äußere Schleife wie die Funktion `train` aus :numref:`sec_linear_scratch`. Nach dem Aufruf der drei Vorbereitungsmethoden durchläuft sie `max_epochs` Epochen. In jeder Epoche durchläuft `fit_epoch()` den Trainingsdatenlader und gibt die gemittelten Trainingsmetriken zurück. `validate()` wertet die Daten aus dem Validierungsdatenlader aus, sofern es einen gibt. `record()` protokolliert die Ergebnisse beider Durchläufe und zeichnet die Kurven.

<!--en:31b6a56c-->
Die Methode `fit` stellt die aktuelle Epochennummer über `self.epoch` bereit und gibt selbst nichts zurück. Jede aufgezeichnete Kurve findet sich in `trainer.history`, einem Wörterbuch von Listen. Die Schlüssel setzen sich aus der Teilmenge und dem Metriknamen zusammen (`'train_loss'`, `'val_loss'` und später `'val_acc'`). So liefert `trainer.history['val_loss'][-1]` direkt den letzten Validierungsverlust. Da beim Sortieren `'loss'` zuerst kommt, verwendet die Fortschrittsanzeige bei allen Modellen dieselben Farben für die entsprechenden Kurven.

<!--en:05d75079-->
Die Punkte der beiden Kurven liegen an unterschiedlichen Stellen der Zeitachse. Den Grund zeigt :numref:`fig_oo_epoch_timeline`. Der Trainingsverlust der Epoche $e$ ist ein Mittelwert über alle Minibatch-Schritte dieser Epoche. Jedem Schritt liegen andere Parameter zugrunde. Deshalb zeichnet `fit` den Mittelwert in der Mitte der Epoche bei $e + \frac{1}{2}$ ein. Den Validierungsverlust berechnen wir nur einmal mit den Parametern vom Ende der Epoche und zeichnen ihn bei $e + 1$ ein.

<!--en:632290d1-->
![Eine Epoche auf der Fortschrittsanzeige. Die Verluste der einzelnen Minibatches (grau) sinken, während sich die Parameter verbessern. Ihr Mittelwert ist der Trainingsverlust, eingezeichnet bei $e + \frac{1}{2}$. Der Validierungsverlust wird mit den Parametern vom Ende der Epoche berechnet und bei $e + 1$ eingezeichnet.](../img/mdl-linreg-epoch-timeline.svg)

<!--en:2c64989a-->
### Epochenmittelwerte

<!--en:28577fae-->
Der letzte Minibatch einer Epoche ist oft kleiner als `batch_size` (etwa bei $1{,}000 = 31 \times 32 + 8$). Würden wir die mittleren Verluste aller Batches ungewichtet mitteln, hätten die 8 Beispiele im letzten Batch zusammen dasselbe Gewicht wie die 32 Beispiele in jedem vorherigen Batch. Unsere Hilfsklasse `Averager` gewichtet die Metrik jedes Batchs mit seiner Batchgröße `n` und summiert die Werte. So liefert `avg.mean()` stets den exakten Mittelwert über alle Beispiele der jeweiligen Teilmenge.

<!--en:b55ca022-->
### Trainings- und Validierungsdurchläufe

<!--en:58703ff3-->
Als Nächstes implementieren wir mit `validate` und `fit_epoch` die batchweise Validierung und das batchweise Training für unsere vier Frameworks. Das optionale Gradienten-Clipping mit `gradient_clip_val` ist bereits für die rekurrenten Netze in :numref:`chap_rnn` vorgesehen.

<!--en:b1645821-->
Mit genau diesem Trainer trainieren wir im weiteren Verlauf des Buchs jedes Modell. Sein Code ist daher etwas länger, als es die lineare Regression allein erfordert. Sein algorithmischer Kern bleibt genau der SGD-Zyklus aus :numref:`fig_linreg_sgd_cycle` mit seinen vier Schritten: Vorwärtsdurchlauf, Verlust, Rückwärtsdurchlauf und Parameterupdate. Einige Erweiterungen für spätere Kapitel bleiben hier noch inaktiv.

<!--en:d6deb0fb-->
Die beiden `if`-Blöcke sind für Gradienten-Clipping und Lernratenpläne vorgesehen. Wir rufen `.detach()` auf, bevor wir die Metriken an `Averager` übergeben. So hält die laufende Verlustsumme die Berechnungsgraphen der einzelnen Minibatches nicht im Speicher. Mit `model.train()` und `model.eval()` schalten wir später Schichten wie Dropout und Batchnormalisierung in den jeweiligen Modus.

<!--en:d3592d43-->
Der Aufruf `self.optim.step(1)` übergibt die Batchgröße `1`, da unser Verlust bereits ein Mittelwert über den Minibatch ist. Mit `wait_to_read()` warten wir auf den Verlust des vorherigen Batchs. So eilt Python dem asynchronen Backend von MXNet um höchstens einen Minibatch voraus, was den Speicherbedarf begrenzt.

<!--en:afdeaf52-->
TensorFlow-Operationen von Python aus einzeln und sofort auszuführen, verursacht erheblichen Zusatzaufwand. Deshalb führt die überschriebene Methode `prepare_model` zunächst einen Vorwärtsdurchlauf mit einer Beispieleingabe aus. Dabei legt Keras alle Schichtvariablen an. Die Methode kapselt `train_step` und `val_step` mit `tf.function` und kompiliert sie so zu schnellen Graphen mit zusammengefassten Operationen.

<!--en:28744b54-->
Versehen wir `_trainer_train_step` und `_trainer_validation_step` mit dem Dekorator `@nnx.jit`, kompiliert dieser Vorwärtsdurchlauf, Verlustberechnung, Gradientenberechnung und Optimiererschritt zu einer einzigen XLA-Berechnung. Mit `has_aux=True` kann `nnx.grad` den Verlust `out['loss']` differenzieren und neben den Gradienten das vollständige Wörterbuch der Metriken zurückgeben.

<!--en:7a93eb4e-->
### Die Fortschrittsanzeige

<!--en:66310bc6-->
Während das Notebook läuft, zeichnet `ProgressBoard` die Lernkurven nach jeder Epoche neu. Für Versuchsreihen ohne grafische Ausgabe lässt sich die Anzeige mit `trainer.board.display = False` abschalten.

<!--en:765b63c7-->
## Erneutes Training der linearen Regression

<!--en:cc6c6283-->
Prüfen wir, wie die gerade entwickelten Klassen `Module`, `DataModule` und `Trainer` zusammenspielen. Dazu trainieren wir `LinearRegressionScratch` auf `SyntheticRegressionData` über 10 Epochen und überprüfen die geschätzten Parameter.

<!--en:577e31cd-->
Der Validierungsverlust verläuft genau wie in :numref:`sec_linear_scratch`. Er sinkt pro Epoche ungefähr um den Faktor $7\times$, bis er die Rauschgrenze bei etwa $5 \times 10^{-5}$ erreicht. Die Trainingskurve liegt genau auf der Validierungskurve (bei nur $3$ Parametern und $1{,}000$ Beispielen lernt das Modell das Rauschen der Labels kaum auswendig). Prüfen wir den endgültigen Validierungsverlust und die Schätzfehler der Parameter numerisch.

<!--en:592cd636-->
Trainings- und Validierungsverlust sinken gleichmäßig bis zur Rauschgrenze von $0.00005$. Die geschätzten Parameter stimmen bis auf etwa ein Tausendstel mit $(\mathbf{w}^*, b^*)$ überein. Damit bestätigt sich, dass unser objektorientiertes Trainingsgerüst dieselben Ergebnisse liefert wie die selbst geschriebene, eigenständige Schleife. In späteren Kapiteln erspart es uns wiederkehrenden Code. In :numref:`sec_linear_concise` gehen wir einen Schritt weiter. Wir ersetzen unsere selbst angelegten Tensoren, den selbst implementierten quadratischen Verlust und die Klasse `SGD` durch die Schichten, Verlustfunktionen und Optimierer des jeweiligen Frameworks.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:9fc407f6-->
Damit wir uns in späteren Kapiteln auf die Modellierung konzentrieren können, trennt die Bibliothek `d2l` Modelle (`Module`), Datensätze (`DataModule`) und Optimierungsschleifen (`Trainer`). Dabei helfen `@add_to_class`, `HyperParameters`, `Averager` und `ProgressBoard`, wiederkehrenden Trainingscode zu vermeiden. `DataModule` liefert Datenlader für mehrere Epochen. Sie mischen die Trainingsbeispiele bei jedem Durchlauf neu und sind nach einer Epoche weiterhin verwendbar. `Averager` gewichtet jede Minibatch-Metrik nach der Anzahl der Beispiele. So verzerren unvollständige letzte Batches die Epochenmittelwerte nicht. Die Schnittstelle von `Module` legt die Rückgabewerte von `training_step`, `validation_step` und `configure_optimizers` fest. Ob wir ein Modell wie `LinearRegressionScratch` direkt mit Tensoren oder mit den Schichten eines Frameworks aufbauen, der Aufruf für Training und Validierung ist immer derselbe: `trainer.fit(model, data)`.

<!--en:2294243a-->
## Übungen

<!--en:6a4b1d05-->
1. **Trainingskurven lesen.** Der Trainer zeichnet den mittleren Trainingsverlust jeder Epoche in ihrer Mitte ein und den Validierungsverlust an ihrem Ende.
    1. Zu Beginn des Trainings liegt die Trainingskurve oft über der Validierungskurve, obwohl die Beispiele beider Teilmengen aus derselben Verteilung stammen. Erklären Sie diesen Unterschied anhand der Berechnung der beiden Mittelwerte.
    1. Angenommen, `fit` berechnete den Trainingsverlust wie den Validierungsverlust mit den Parametern vom Ende jeder Epoche. Welcher zusätzliche Aufwand entstünde pro Epoche? Was würde der Abstand zwischen den beiden Kurven dann messen?

<!--en:624bce41-->
1. [code] **Early Stopping.** Bei einer festen Anzahl von Epochen kann das Training zu früh oder zu spät enden. Schreiben Sie eine Unterklasse von `d2l.Trainer`, deren Methode `fit` das Training beendet, sobald der Validierungsverlust über `patience` aufeinanderfolgende Epochen nicht mehr gesunken ist. Treffen Sie die Entscheidung allein anhand von `self.history`. Erklären Sie, warum sich weder die Modellklasse noch die Datensatzklasse ändern muss. Testen Sie Ihren Trainer mit den synthetischen Regressionsdaten und einem großen Wert für `max_epochs`.

<!--en:2d23c93c-->
1. [code] **Das plancksche Strahlungsgesetz.** ● Mit denselben Klassen `Module` und `Trainer` lässt sich jedes Modell mit differenzierbarem Verlust anpassen, auch ein nichtlineares. Bestimmen Sie mit dem [planckschen Strahlungsgesetz](https://en.wikipedia.org/wiki/Planck%27s_law) die Temperatur eines schwarzen Körpers aus Messungen seiner spektralen Strahldichte bei mehreren Wellenlängen. Für die spektrale Strahldichte gilt

    $$B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1},$$

    wobei $\lambda$ die Wellenlänge ist, $T$ die Temperatur, $c$ die Lichtgeschwindigkeit, $h$ das plancksche Wirkungsquantum und $k$ die Boltzmann-Konstante. Erzeugen Sie Messwerte bei bekannter Temperatur $T$. Schreiben Sie dann eine Unterklasse von `d2l.Module`, deren Methode `forward` die Strahldichte $B(\lambda, T)$ mit $T$ als einzigem Parameter berechnet. Passen Sie das Modell mit `d2l.Trainer` an. Wählen Sie vor der Anpassung Einheiten, in denen $B$ und $T$ von der Größenordnung eins sind.

<!--en:5d2b6e53-->
1. **Ein unvollständiger letzter Minibatch.** Bei 1000 Beispielen pro Teilmenge und `batch_size=32` liefert jeder Datenlader am Ende eines Durchlaufs einen unvollständigen Minibatch mit 8 Beispielen. `ArrayLoader` verwirft ihn bei `drop_last=True`. Der Trainingsdatenlader mischt die Beispiele in jeder Epoche neu, der Validierungsdatenlader liest sie in fester Reihenfolge. Geben Sie für jeden Datenlader an, welche Beispiele über viele Epochen hinweg entfallen, wenn er den letzten unvollständigen Minibatch verwirft. Entscheiden Sie, bei welchem Datenlader das vertretbar ist.

<!--en:33781f8f-->
1. **Ein unvollständiger letzter Minibatch.** Bei 1000 Beispielen pro Teilmenge und `batch_size=32` liefert jeder Datenlader am Ende eines Durchlaufs einen unvollständigen Minibatch mit 8 Beispielen. `DataLoader` verwirft ihn bei `drop_last=True`. Der Trainingsdatenlader mischt die Beispiele in jeder Epoche neu, der Validierungsdatenlader liest sie in fester Reihenfolge. Geben Sie für jeden Datenlader an, welche Beispiele über viele Epochen hinweg entfallen, wenn er den letzten unvollständigen Minibatch verwirft. Entscheiden Sie, bei welchem Datenlader das vertretbar ist.

<!--en:cce5d52f-->
1. **Ein unvollständiger letzter Minibatch.** Bei 1000 Beispielen pro Teilmenge und `batch_size=32` liefert jeder Datenlader am Ende eines Durchlaufs einen unvollständigen Minibatch mit 8 Beispielen. `Dataset.batch` verwirft ihn bei `drop_remainder=True`. Der Trainingsdatenlader mischt die Beispiele in jeder Epoche neu, der Validierungsdatenlader liest sie in fester Reihenfolge. Geben Sie für jeden Datenlader an, welche Beispiele über viele Epochen hinweg entfallen, wenn er den letzten unvollständigen Minibatch verwirft. Entscheiden Sie, bei welchem Datenlader das vertretbar ist.

<!--en:4b4306cb-->
1. **Ein unvollständiger letzter Minibatch.** Bei 1000 Beispielen pro Teilmenge und `batch_size=32` liefert jeder Datenlader am Ende eines Durchlaufs einen unvollständigen Minibatch mit 8 Beispielen. `ArrayLoader` verwirft ihn bei `drop_last=True`.
    1. Der Trainingsdatenlader mischt die Beispiele in jeder Epoche neu, der Validierungsdatenlader liest sie in fester Reihenfolge. Geben Sie für jeden Datenlader an, welche Beispiele über viele Epochen hinweg entfallen, wenn er den letzten unvollständigen Minibatch verwirft. Entscheiden Sie, bei welchem Datenlader das vertretbar ist.
    1. Für jede neue Eingabeform wird ein kompilierter Schritt einmal neu kompiliert. Zählen Sie, wie viele Kompilierungen der Trainings- und Validierungsschritte über zehn Epochen anfallen, wenn der unvollständige Minibatch erhalten bleibt. Vergleichen Sie deren Kosten damit, in jeder Epoche 8 Beispiele zu verwerfen.

<!--en:3a961683-->
1. [code] **Datenerzeugung bei Bedarf.** Schreiben Sie eine Unterklasse von `DataModule`, die weder $\mathbf{X}$ noch $\mathbf{y}$ speichert. Ihr Trainingsdatenlader soll die Beispiele jedes Minibatchs erst bei der Anforderung neu aus dem datenerzeugenden Prozess ziehen.
    1. Ein Strom immer neuer Minibatches hat kein natürliches Ende. Der `Trainer` beendet eine Epoche aber erst, wenn ein Durchlauf des Trainingsdatenladers endet. Legen Sie fest, wie viele Minibatches eine Epoche liefert, und beenden Sie jeden Durchlauf nach dieser Anzahl. Sorgen Sie dafür, dass `len` diese Anzahl zurückgibt.
    1. Entscheiden Sie, ob auch der Validierungsdatenlader bei jedem Durchlauf neue Beispiele ziehen soll, und setzen Sie Ihre Entscheidung um. Begründen Sie sie mit dem Zweck des Validierungsverlusts.

<!--en:5dc98cd0-->
[Diskussionen](https://d2l.discourse.group/t/6645)

<!--en:739cd97c-->
[Diskussionen](https://d2l.discourse.group/t/6646)

<!--en:44c2b62f-->
[Diskussionen](https://d2l.discourse.group/t/6647)

<!--en:08edde1e-->
[Diskussionen](https://d2l.discourse.group/t/17974)

<!--en:e5b5c54b-->
Objektorientierter Entwurf für Deep Learning<br>
**Module · DataModule · Trainer · wiederverwendbare Datenlader · nach Batchgröße gewichtete Epochenmittelwerte**

<!--en:24b7cb33-->
::: {.slide title="Warum Modelle, Datensätze und Trainingsschleifen entkoppeln?"}

<!--en:2159915a-->
[Gründe für die Softwarearchitektur]{.kicker}

<!--en:d6119c66-->
- **Getrennte Zuständigkeiten:** Die Aufteilung jedes Experiments in `DataModule`, `Module` und `Trainer` ersetzt $N \times M$ gleichartige Schleifen durch ein wiederverwendbares Trainingsgerüst.

<!--en:95f8802d-->
::: {.slide title="Die drei zentralen Abstraktionen des d2l-Trainingsgerüsts"}

<!--en:3031be70-->
[Aufgaben und Schnittstellen der Klassen]{.kicker}

<!--en:0015a5f9-->
- **`DataModule` (Daten):** Verwaltet die Rohdaten in Arrays oder Dateien sowie die Indizes der Teilmengen und stellt `train_dataloader()` und `val_dataloader()` bereit.

<!--en:e7624856-->
- **`Module` (Modelle):** Enthält trainierbare Parameter und Schichten und definiert `forward(X)`, `loss(y_hat, y)`, `training_step(batch)` und `configure_optimizers()`.

<!--en:bef8dc3f-->
- **`Trainer` (Trainingsschleife):** Verbindet ein beliebiges `Module` mit einem beliebigen `DataModule` über `fit(model, data)` und steuert `fit_epoch()`, `validate()` und `ProgressBoard`.

<!--en:382a5a0c-->
::: {.slide title="Hilfsmittel fürs Notebook: add_to_class und HyperParameters"}

<!--en:bc5c4d5b-->
[Lesbarer Code mit begleitenden Erklärungen]{.kicker}

<!--en:102beca0-->
- **Methoden schrittweise ergänzen, Argumente automatisch binden:** `@add_to_class(Class)` fügt Methoden aus verschiedenen Notebook-Zellen hinzu. `self.save_hyperparameters()` speichert die Argumente von `__init__` automatisch als Attribute von `self`.

<!--en:6c275772-->
::: {.slide title="Die Schnittstelle der Basisklasse Module"}

<!--en:b212bab0-->
[Modellabstraktion]{.kicker}

<!--en:d7992e53-->
- **Vorwärtsdurchlauf und Verlust (`forward`, `loss`):** `forward(X)` ruft standardmäßig `self.net(X)` auf. `loss(y_hat, y)` berechnet den skalaren Wert der Zielfunktion für den Minibatch.

<!--en:33b56be3-->
- **Schrittmethoden (`training_step`, `validation_step`):** Eingaben und Labels mit `*batch[:-1]` und `batch[-1]` entpacken, skalaren Verlust oder Wörterbuch mit Metriken zurückgeben (vereinheitlicht mit `as_metrics`).

<!--en:e5f12b1c-->
- **Unabhängig vom Trainer:** Modelle geben selbst keinen Text aus, zeichnen keine Kurven und ändern den Zustand des Trainers nie direkt.

<!--en:100fd718-->
::: {.slide title="Selbst implementierte lineare Regression als Modul"}

<!--en:7bdb6108-->
[LinearRegressionScratch]{.kicker}

<!--en:24303e99-->
- **Parameter registrieren:** Sind die Tensoren als Modulparameter registriert, findet sie das Framework für die Gradientenberechnung und die Übertragung auf ein anderes Gerät.

<!--en:68c31b57-->
::: {.slide title="Die eigene SGD-Aktualisierungsregel als Optimierer"}

<!--en:5f6c4845-->
[Die Schnittstelle des Optimierers]{.kicker}

<!--en:7e42a01e-->
::: {.slide title="DataModule und wiederverwendbare Datenlader für mehrere Epochen"}

<!--en:02d2af58-->
[Abstraktion der Daten-Pipeline]{.kicker}

<!--en:e65b50a9-->
- **Vorteil eines wiederverwendbaren Datenladers:** Ein Python-Generator mit `yield` ist nach Epoche 1 erschöpft. Der Datenlader aus `get_tensorloader` erzeugt für jede Epoche einen neuen Iterator mit neu gemischter Reihenfolge und unterstützt `len(loader)` in der Zeit $O(1)$.

<!--en:faca8514-->
::: {.slide title="Neumischen in jeder Epoche mit synthetischen Daten prüfen"}

<!--en:d02844cd-->
[Verhalten des Datenladers]{.kicker}

<!--en:a3a581aa-->
Der Aufruf `data.train_dataloader()` liefert einen wiederverwendbaren Datenlader. Bei jedem Durchlauf liefert er $32$ Minibatches, zuletzt die restlichen $8$ Beispiele. Jede Epoche beginnt mit einer neuen zufälligen Permutation:

<!--en:bcb854fa-->
::: {.slide title="Wie ungewichtete Batchmittelwerte den Epochenmittelwert verzerren"}

<!--en:c009d88e-->
[Nach Batchgröße gewichtete Epochenmittelwerte]{.kicker}

<!--en:fdb261a1-->
- **Jedes Beispiel richtig gewichten:** Wegen $1{,}000 = 31 \times 32 + 8$ erhalten die letzten 8 Beispiele beim ungewichteten Mitteln jeweils $4\times$ so viel Gewicht wie ein Beispiel aus einem vollen Batch. `Averager` gewichtet jeden Batch mit seiner Größe $n_b$.

<!--en:edffc034-->
::: {.slide title="Die äußere Trainingsschleife und der Verlauf über die Epochen"}

<!--en:1de2fa9e-->
[Steuerung des Trainingszyklus]{.kicker}

<!--en:b7167d82-->
::: {.slide title="Position der Trainings- und Validierungspunkte"}

<!--en:1de2fa9e-2-->
[Steuerung des Trainingszyklus]{.kicker}

<!--en:302be3c1-->
- **Epochenmitte und Epochenende:** `fit_epoch()` mittelt den Trainingsverlust über die Aktualisierungsschritte und zeichnet ihn bei $x = \text{epoch} + 0.5$ ein. `validate()` wertet die Gewichte vom Epochenende bei $x = \text{epoch} + 1.0$ aus.

<!--en:2f1fd1ab-->
::: {.slide title="Die vier Schritte in fit_epoch und validate" only="pytorch"}

<!--en:8a0d2cde-->
[Ablauf pro Minibatch]{.kicker}

<!--en:6af99427-->
- **Metriken vom Graphen abtrennen:** Der Aufruf von `.detach()` für die protokollierten Metriken gibt den Autograd-Graphen jedes Batchs sofort frei.

<!--en:3d542247-->
::: {.slide title="Die vier Schritte in fit_epoch und validate" only="tensorflow"}

<!--en:8a0d2cde-2-->
[Ablauf pro Minibatch]{.kicker}

<!--en:cb8e9d7a-->
- **Kompilierter Schritt:** `train_step` zeichnet den Vorwärtsdurchlauf mit `tf.GradientTape` auf, berechnet die Gradienten und wendet sie an. `prepare_model` kapselt die Methode mit `tf.function`. So läuft der gesamte Schritt als ein kompilierter Graph ab.

<!--en:631dc0d5-->
- **Schlanke äußere Schleife:** `fit_epoch` übergibt dem kompilierten Schritt nur die Minibatches und reicht jede zurückgegebene Metrik an `Averager` weiter, der sie nach Batchgröße gewichtet.

<!--en:5df5e3b0-->
::: {.slide title="Die vier Schritte in fit_epoch und validate" only="jax"}

<!--en:8a0d2cde-3-->
[Ablauf pro Minibatch]{.kicker}

<!--en:84f6b25e-->
- **Kompilierter Schritt:** `_trainer_train_step` berechnet Verlust und Gradienten mit `nnx.grad(..., has_aux=True)` und aktualisiert den Optimierer. Alles geschieht in einer einzigen mit `@nnx.jit` kompilierten XLA-Berechnung.

<!--en:631dc0d5-2-->
- **Schlanke äußere Schleife:** `fit_epoch` übergibt dem kompilierten Schritt nur die Minibatches und reicht jede zurückgegebene Metrik an `Averager` weiter, der sie nach Batchgröße gewichtet.

<!--en:a9f2a183-->
::: {.slide title="Die vier Schritte in fit_epoch und validate" only="mxnet"}

<!--en:8a0d2cde-4-->
[Ablauf pro Minibatch]{.kicker}

<!--en:38b18b6b-->
- **Vier Schritte pro Minibatch:** `fit_epoch` zeichnet Vorwärtsdurchlauf und Verlust mit `autograd.record()` auf, ruft `backward()` auf und führt `self.optim.step(1)` aus (der Verlust ist bereits ein Mittelwert über den Minibatch).

<!--en:af558fee-->
- **Begrenzter Vorlauf:** Durch Warten auf den Verlust des vorherigen Batchs (`wait_to_read()`) eilt Python dem asynchronen Backend von MXNet höchstens einen Minibatch voraus.

<!--en:fcf6b2de-->
::: {.slide title="Die Schnittstelle von ProgressBoard für laufend aktualisierte Kurven"}

<!--en:6db02ea3-->
[Darstellung des Trainingsfortschritts]{.kicker}

<!--en:32e1a467-->
`ProgressBoard` zeichnet die Trainings- und Validierungskurven laufend an derselben Stelle neu:

<!--en:93e741a1-->
::: {.slide title="ProgressBoard: Kurvenkoordinaten und Betrieb ohne Anzeige"}

<!--en:6db02ea3-2-->
[Darstellung des Trainingsfortschritts]{.kicker}

<!--en:08aecc6d-->
- **Training in der Epochenmitte, Validierung am Epochenende:** Die Punkte für `'train_loss'` bei $x = \text{epoch} + 0.5$ und für `'val_loss'` bei $x = \text{epoch} + 1.0$ markieren für jede Kurve den mittleren Zeitpunkt, zu dem die jeweiligen Parameter ausgewertet wurden.

<!--en:7ed3a992-->
- **Versuchsreihen ohne Anzeige:** `trainer.board.display = False` schaltet die grafische Darstellung ab. `trainer.history` zeichnet die Werte für Versuchsreihen mit unterschiedlichen Hyperparametern weiterhin auf.

<!--en:fa6aa5b5-->
::: {.slide title="Test des Gesamtablaufs mit LinearRegressionScratch"}

<!--en:967cbd68-->
[Vergleich mit der eigenen Implementierung]{.kicker}

<!--en:d9af4fa1-->
Mit `Trainer(max_epochs=10)` und `LinearRegressionScratch(2, lr=0.03)` zeigen sich wieder dieselbe Konvergenz (der Verlust sinkt pro Epoche um den Faktor $7\times$) und Parameterschätzungen mit einem Fehler von höchstens etwa $10^{-3}$:

<!--en:800146c7-->
- **Deckungsgleiche Trainings- und Validierungskurven:** Bei nur $3$ Parametern und $1{,}000$ Beispielen ist Auswendiglernen unmöglich. Der Validierungsverlust folgt dem Trainingsverlust bis hinunter zu $\sigma^2/2 = 5 \times 10^{-5}$.

<!--en:02706483-->
- **Vorbereitet auf die kompakte Implementierung:** Im nächsten Abschnitt kommen eingebaute Schichten, Verlustfunktionen und Optimierer zum Einsatz. `SyntheticRegressionData` und `Trainer` bleiben unverändert.

<!--en:e68e959b-->
::: {.slide title="Verständnisfragen zu Averager, Kurvenpositionen und Klassenentwurf"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:3ae10682-->
Drei Fragen zum objektorientierten Trainingsgerüst:

<!--en:9e3959c9-->
1. **Nach Batchgröße gewichtetes Mittel mit `Averager`:** $1{,}000$ Validierungsbeispiele verteilen sich auf Minibatches der Größe $32$ ($31$ Batches mit je $32$ Beispielen und $1$ Batch mit $8$ Beispielen). Die $31$ vollen Batches haben jeweils den mittleren Verlust $0.010$, der letzte Batch mit $8$ Beispielen den mittleren Verlust $0.050$. Welchen Epochenmittelwert liefert `Averager`? Was ergibt das ungewichtete Mittel der $32$ Batchmittelwerte?

<!--en:fba5f9f5-->
2. **Warum der Trainingsverlust anfangs über dem Validierungsverlust liegt:** Warum liegt der eingezeichnete Trainingsverlust in Epoche 1 oft *höher* als der Validierungsverlust, obwohl die Daten beider Teilmengen aus genau derselben Verteilung stammen?

<!--en:47c1a7fb-->
3. **Ansatzpunkte für Erweiterungen:** Welche Klasse und welche Methode sind jeweils anzupassen, um (a) neben dem Verlust die Klassifikationsgenauigkeit auszugeben, (b) von SGD auf Adam umzusteigen oder (c) Early Stopping einzusetzen, sobald der Validierungsverlust stagniert?

<!--en:557b98b6-->
**Lösungen:**

<!--en:e6257a93-->
1. `Averager` liefert $(992 \times 0.010 + 8 \times 0.050) / 1000 = 0.01032$. Das ungewichtete Mittel der Batchmittelwerte ergibt $(31 \times 0.010 + 1 \times 0.050) / 32 = 0.01125$. Es liegt um $9\%$ zu hoch, weil die $8$ verrauschten Beispiele jeweils etwa $4\times$ so viel Gewicht erhalten wie bei korrekter Gewichtung.

<!--en:f59e697a-->
2. `fit_epoch()` mittelt den Trainingsverlust über die gesamte Epoche, beginnend mit dem hohen Verlust in Schritt 1. Der Mittelwert liegt im Diagramm bei $x = 0.5$. `validate()` wertet das Modell erst aus, *nachdem* alle $32$ Gradientenschritte der Epoche den Verlust bereits gesenkt haben. Dieser Punkt liegt bei $x = 1.0$.

<!--en:4b7f4aae-->
3. (a) `Module.validation_step` überschreiben und `{'loss': ..., 'acc': ...}` zurückgeben. (b) `Module.configure_optimizers` ändern. (c) In einer Unterklasse einmal `Trainer.fit` überschreiben und dort `self.history['val_loss']` prüfen. Dafür sind keine Änderungen an Modell- oder Datensatzklassen nötig.

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und zentrale Erkenntnisse"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:81abff0c-->
| Klasse oder Hilfsmittel | Hauptaufgabe | Vermiedenes Problem | Verwendung im weiteren Buch |

<!--en:40fdfbb0-->
| **`@add_to_class` und `HyperParameters`** | Schrittweises Ergänzen von Methoden, automatische Zuweisung an `self.<arg>` | Keine unübersichtlichen Zellen mit 60 Zeilen und keine wiederholten Zuweisungen im Konstruktor | Verfügbar in allen Klassen auf Basis von `Module`, `DataModule` und `Trainer` |

<!--en:259780c9-->
| **`d2l.Module`** | Enthält Parameter, `forward`, `loss`, `training_step` und Optimierer | Einheitliche Rückgabewerte (`as_metrics`) entkoppeln Modelle von der Protokollierung | Basisklasse für alle neuronalen Netze im Buch |

<!--en:6c57f313-->
| **`d2l.DataModule`** | Kapselt die Aufteilung der Daten und liefert wiederverwendbare Datenlader für mehrere Epochen | Kein unbemerktes Erschöpfen eines nur einmal durchlaufbaren Generators nach Epoche 1 | Basisklasse für synthetische Daten, Tabellendaten, Bilddaten und Sequenzdaten |

<!--en:1ab03b7c-->
| **`d2l.Averager`** | Berechnet $\sum_b n_b \bar{L}_b / \sum_b n_b$ durch Aufsummieren auf dem Gerät | Keine Verzerrung der Epochenmittelwerte durch unvollständige letzte Minibatches ($n_B < |\mathcal{B}|$) | Für alle Aufgaben in `fit_epoch` und `validate` eingesetzt |

<!--en:e5735b51-->
| **`d2l.Trainer`** | Steuert Epochenschleifen und zeichnet Ergebnisse in `history` und auf dem `ProgressBoard` auf | Protokollierte Metriken halten die Gradientengraphen der einzelnen Batches nicht im Speicher | Passt jedes `Module` an jedes `DataModule` mit `trainer.fit(model, data)` an |
