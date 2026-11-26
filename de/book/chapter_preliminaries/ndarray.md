<!--en:cb1cc458-->
# Arbeiten mit Daten

<!--en:5632b2b2-->
Für maschinelles Lernen müssen wir Daten effizient im Speicher ablegen und verarbeiten können. Wir beginnen mit $n$-dimensionalen Arrays, die im Deep Learning *Tensoren* heißen.

<!--en:b52a1394-->
Warum brauchen wir spezielle Datenstrukturen für Arrays? Betrachten wir einen einzelnen *Minibatch* aus $256$ Graustufenfotos mit einer Auflösung von $28 \times 28$ Pixeln. Er enthält $256 \times 28 \times 28 = 200{,}704$ Zahlen. Bilddateien speichern jeden Pixel als ganze Zahl in $\{0, \ldots, 255\}$. Wir skalieren diese Werte auf $[0, 1]$, indem wir sie durch $255$ teilen. In reinem Python würde der Interpreter in drei verschachtelten Schleifen $200{,}704$ einzelne skalare Divisionen ausführen. In einer Tensorbibliothek ruft der einzelne Ausdruck `images / 255` kompilierte C++- oder CUDA-Kernel auf, die alle $200{,}704$ Zahlen parallel auf einer CPU oder GPU dividieren. Im gesamten Deep Learning arbeiten wir mit ganzen Tensoren auf einmal, anstatt einzelne Skalare in Python-Schleifen zu durchlaufen.

<!--en:39c908fa-->
Die Tensorklassen aller modernen Deep-Learning-Frameworks lehnen sich eng an NumPys `ndarray` an :cite:`Harris.Millman.Walt.ea.2020`. Zwei wesentliche Fähigkeiten kommen hinzu:

<!--en:123e4041-->
1. **Hardwarebeschleunigung:** Die Frameworks führen Tensoroperationen auf GPUs und spezialisierten KI-Beschleunigern aus. NumPy läuft auf CPUs.

<!--en:445a944f-->
2. **Automatisches Differenzieren** (engl. automatic differentiation): Die Frameworks zeichnen Berechnungen mit Tensoren auf und berechnen Gradienten automatisch mit Backpropagation (:numref:`sec_autograd`).

<!--en:83fa9602-->
In MXNet heißt die Tensorklasse `ndarray`. Wir importieren das NumPy-kompatible Array-Modul von MXNet als `mnp` und behalten `np` für das reguläre NumPy bei. Außerdem importieren wir `npx` (`numpy_extension`) und aktivieren mit `npx.set_np()` die NumPy-kompatible Tensorsemantik.

<!--en:32babc1e-->
In PyTorch heißt die Tensorklasse `torch.Tensor`. Wir importieren PyTorch (`torch`).

<!--en:c35bf960-->
In TensorFlow heißt die Tensorklasse `tf.Tensor`. Wir importieren TensorFlow unter dem üblichen Kurznamen `tf`.

<!--en:04efb4ec-->
In JAX heißt die Tensorklasse `jax.Array`. Wir importieren `jax` und dessen NumPy-kompatible Array-Schnittstelle `jax.numpy` als `jnp`.

<!--en:7c5d99a4-->
## Tensoren erzeugen

<!--en:f0c15e8f-->
### Form

<!--en:0e43cefb-->
Ein Tensor ordnet Zahlenwerte entlang einer oder mehrerer *Achsen* an. Ein Tensor mit einer Achse ist ein *Vektor*, mit zwei Achsen eine *Matrix* und mit $k > 2$ Achsen ein *Tensor der Ordnung* $k^\textrm{th}$. Die Anzahl seiner Achsen ist also seine *Ordnung* (im Code `ndim`). Unser Minibatch aus Graustufenbildern ist ein Tensor der Ordnung $4^\textrm{th}$ mit der Form (engl. shape) $(256, 1, 28, 28)$. Achse $0$ indiziert das Bild im Batch, Achse $1$ den Farbkanal ($1$ für Graustufen, $3$ für RGB), Achse $2$ die Pixelzeile in vertikaler Richtung und Achse $3$ die Pixelspalte in horizontaler Richtung.

<!--en:c9c043b0-->
Der Aufruf `arange(n)` erzeugt einen eindimensionalen Vektor aus gleichmäßig verteilten Werten von $0$ (einschließlich) bis $n$ (ausschließlich). Neue Tensoren liegen standardmäßig für Berechnungen auf der CPU im Hauptspeicher. Hier geben wir 32-Bit-Gleitkommazahlen als Datentyp an (`float32`):

<!--en:aad7421d-->
In TensorFlow heißt diese Funktion `tf.range`.

<!--en:06cf7339-->
Jeder Wert in einem Tensor ist ein *Element*. Die Gesamtzahl der Elemente in `x` (seine *Größe*) können wir abfragen:

<!--en:0f848b6d-->
In PyTorch liefert `numel()` die Gesamtzahl der Elemente.

<!--en:81342bbc-->
Als Nächstes lesen wir mit `x.shape` die *Form* des Tensors aus. Dieses Tupel gibt die Länge jeder Achse an (für unseren Vektor mit 12 Elementen das 1-Tupel `(12,)`):

<!--en:406bcdec-->
:numref:`fig_ndarray_shapes` zeigt Tensoren der Ordnungen $0$ bis $3$. Eine einzelne Zahl (ein *Skalar*) ist ein Tensor der Ordnung $0^\textrm{th}$ mit der leeren Form `()`. Die Größe eines Tensors ist immer das Produkt der Achsenlängen in seinem Formtupel (für unseren Minibatch aus Bildern also $256 \times 1 \times 28 \times 28 = 200{,}704$).

<!--en:3c284b12-->
![Tensoren mit null, einer, zwei und drei Achsen. Die Form gibt die Länge jeder Achse an. Die Größe ist das Produkt dieser Längen: 1, 4, 12 und 24 Elemente.](../img/ndarray-rank-ladder.svg)

<!--en:3abedce3-->
### Datentypen

<!--en:38c45ac0-->
Jeder Tensor hat einen *Datentyp* (`dtype`), der die binäre Darstellung seiner Elemente festlegt. Deep-Learning-Modelle rechnen meist mit 32-Bit-Gleitkommazahlen (`float32`, $4$ Byte pro Element, Genauigkeit von $\approx 7$ Dezimalstellen) oder in 16-Bit-Formaten (`float16` / `bfloat16`, $2$ Byte pro Element, siehe :numref:`sec_numerics`). Die Pixel eines Rohbilds sind meist als vorzeichenlose 8-Bit-Ganzzahlen gespeichert (`uint8`, $1$ Byte pro Pixel für $\{0, \ldots, 255\}$). Unser Minibatch mit $200{,}704$ Pixeln belegt daher $0.2\textrm{ MB}$ als `uint8` und $0.8\textrm{ MB}$ als `float32`.

<!--en:73e23848-->
MXNet verwendet standardmäßig `float32`. Mit `x.astype('int32')` wandeln wir den Typ um.

<!--en:a05aa3d4-->
`torch.arange` verwendet bei ganzzahligen Argumenten standardmäßig `int64`. Mit `x.to(torch.int32)` wandeln wir den Typ um.

<!--en:d7d18414-->
`tf.range` verwendet bei ganzzahligen Argumenten standardmäßig `int32`. Mit `tf.cast(x, tf.int32)` wandeln wir den Typ um.

<!--en:80509d43-->
`jnp.arange` verwendet bei ganzzahligen Argumenten standardmäßig `int32`. Mit `x.astype(jnp.int32)` wandeln wir den Typ um.

<!--en:a1451368-->
### Anfangswerte

<!--en:a73dab82-->
Die Funktionen `zeros` und `ones` erzeugen Tensoren beliebiger Form, deren Elemente alle den Wert $0$ beziehungsweise $1$ haben (etwa für Bias-Terme oder Puffer):

<!--en:67eb9f20-->
Die Gewichte eines neuronalen Netzes initialisieren wir üblicherweise, indem wir ihre Werte unabhängig aus einer Wahrscheinlichkeitsverteilung ziehen (:numref:`sec_numerical_stability`). Hier ziehen wir die Einträge einer $3 \times 4$-Matrix aus der Standardnormalverteilung (Mittelwert $0$, Standardabweichung $1$):

<!--en:9943c1e2-->
Vom Computer erzeugte Zufallszahlen sind *Pseudozufallszahlen*: Ein deterministischer Algorithmus berechnet sie aus einem ganzzahligen *Zufallsstartwert* (engl. seed). Beginnen wir erneut mit demselben Startwert, erhalten wir genau dieselbe Folge (unten stimmen `a` und `c` überein, `b` weicht davon ab):

<!--en:f3a59237-->
MXNet setzt seinen globalen Generator mit `mnp.random.seed` zurück.

<!--en:7c122650-->
PyTorch setzt seinen globalen Generator mit `torch.manual_seed` zurück.

<!--en:019626e5-->
TensorFlow setzt seinen globalen Generator mit `tf.random.set_seed` zurück.

<!--en:847f739a-->
JAX arbeitet rein funktional und hat keinen globalen Zustand für Zufallszahlen. Jede Zufallsfunktion erhält explizit einen *Schlüssel* für den Pseudozufallsgenerator. Mit `jax.random.split` leiten wir deterministisch neue, voneinander unabhängige Teilschlüssel ab. Unten verwendet `c` erneut `key_a` und liefert daher dasselbe Ergebnis wie `a`.

<!--en:2784729a-->
Kennen wir die Einträge eines kleinen Tensors im Voraus, können wir ihn direkt aus verschachtelten Python-Listen erzeugen. Hier erstellen wir eine $3 \times 4$-Matrix. Die äußere Liste enthält die drei Zeilen, jede innere Liste liefert die Werte für die vier Spalten:

<!--en:d5df7d75-->
Zum Schluss erzeugen wir den synthetischen Minibatch `images`, der uns als durchgehendes Beispiel dient. Er hat die Form `(256, 1, 28, 28)` und den Typ `uint8`. Wir füllen ihn mit zufälligen ganzen Zahlen aus $\{0, \ldots, 255\}$:

<!--en:61162d7f-->
## Umformen und Indizieren

<!--en:78c24351-->
### Umformen

<!--en:09daef49-->
Der Aufruf `reshape` ordnet die Achsen eines Tensors neu zu. Dabei bleiben die Anzahl der Elemente und ihre zeilenweise Reihenfolge im Speicher erhalten (:numref:`fig_ndarray_reshape`). Hier formen wir den Vektor `x` mit 12 Elementen in eine $3 \times 4$-Matrix `X` um:

<!--en:d9ea25c5-->
![Umformen eines Vektors mit 12 Elementen in eine $3 \times 4$-Matrix. Die Elemente behalten ihre Reihenfolge bei und füllen die Matrix Zeile für Zeile.](../img/ndarray-reshape.svg)

<!--en:b9d1a50b-->
Da die Gesamtgröße feststeht, kann das Framework die Länge einer Dimension automatisch bestimmen, wenn wir dafür `-1` angeben. Beispielsweise können wir unseren Bild-Minibatch der Form $(256, 1, 28, 28)$ in eine $(256, 784)$-Matrix mit einem Bild pro Zeile umformen:

<!--en:05c1e3a7-->
In TensorFlow verwenden wir für `reshape` die Funktion `tf.reshape(x, (-1, 4))`.

<!--en:ddcd8c3c-->
Beim Indizieren fügt `None` an der angegebenen Stelle eine neue Achse der Länge $1$ ein. Mit `squeeze` entfernen wir eine Achse der Länge $1$:

<!--en:dae7f07d-->
In PyTorch fügt auch `image.unsqueeze(0)` an der Position `0` eine Achse der Länge eins ein.

<!--en:fef7814a-->
In TensorFlow rufen wir `squeeze` als `tf.squeeze` auf.

<!--en:5f2e805b-->
Beim Umformen gruppieren wir die Elemente nur neu. Da ihre Reihenfolge erhalten bleibt, können wir so keine zwei Achsen vertauschen. Manche Bibliotheken erwarten Minibatches von Bildern mit der Kanalachse an letzter Stelle, also in der Form $(256, 28, 28, 1)$. Ausgehend von unserem Minibatch der Form $(256, 1, 28, 28)$ müssen wir dazu Achse $1$ ans Ende verschieben. Das ist eine *Transposition*: Wir geben die bisherigen Achsen in ihrer neuen Reihenfolge an, hier $(0, 2, 3, 1)$. Bei einer Matrix lassen sich nur Zeilen und Spalten vertauschen. Dafür schreiben wir `X.T`. Vergleichen wir das Ergebnis mit einer Umformung auf dieselbe Form:

<!--en:7d9d078c-->
In PyTorch heißt die Methode zum Umordnen aller Achsen `permute`. Dagegen vertauscht `transpose(i, j)` nur die beiden Achsen `i` und `j`.

<!--en:d3fb23c4-->
In TensorFlow übergeben wir `tf.transpose(x, perm)` die neue Achsenreihenfolge als `perm`. Der Aufruf `tf.transpose(X)` entspricht `X.T`.

<!--en:4cdf3ff5-->
Beide Matrizen haben die Form $4 \times 3$. Nach dem Umformen lesen wir zeilenweise weiterhin $0, 1, 2, \ldots$. Beim Transponieren sind die Zeilen von `X` dagegen zu Spalten geworden. Da die Formen übereinstimmen, löst ein versehentliches Umformen anstelle einer Transposition keinen Fehler aus. Die nächste Schicht erhält die richtigen Zahlen an den falschen Stellen.

<!--en:38f55b42-->
### Indizes und Slices

<!--en:48cffdf7-->
Auf Elemente und Teiltensoren greifen wir mit ganzzahligen Indizes ab $0$ zu (negative Indizes zählen vom Ende rückwärts) oder mit halboffenen Slices der Form `start:stop` (einschließlich `start`, ausschließlich `stop`). Die Angaben für die einzelnen Achsen trennen wir durch Kommas:

<!--en:9df70605-->
Beachten Sie die unterschiedlichen Formen (:numref:`fig_ndarray_index`). Ein ganzzahliger Index (`X[-1]`) entfernt die indizierte Achse und liefert einen Vektor der Form `(4,)`. Ein Slice (`X[1:3]`) erhält die Achse und liefert eine Matrix der Form `(2, 4)`.

<!--en:6db876ee-->
![Indizieren einer $3 \times 4$-Matrix. Der ganzzahlige Index `X[-1]` wählt die letzte Zeile aus und entfernt Achse 0. Übrig bleibt die Form $(4,)$. Der Slice `X[1:3]` wählt die Zeilen 1 und 2 aus und erhält Achse 0. Das Ergebnis hat die Form $(2, 4)$.](../img/ndarray-index-read.svg)

<!--en:2d211507-->
Aus unserem Bild-Minibatch der Form $(256, 1, 28, 28)$ entnimmt `images[0]` das erste Bild mit der Form $(1, 28, 28)$. Der Slice `images[:32]` wählt die ersten $32$ Bilder mit der Form $(32, 1, 28, 28)$ aus. Mit `images[:, 0, 14, 14]` erhalten wir den mittleren Pixel aus allen $256$ Bildern als Vektor der Form $(256,)$:

<!--en:ed556481-->
Wir können einen Tensor auch mit *ganzzahligen Indextensoren* indizieren (*Gather*). Übergeben wir die beiden eindimensionalen ganzzahligen Tensoren `rows = (0, 1, 2)` und `cols = (3, 0, 2)`, so wählen wir die Einträge `(0, 3)`, `(1, 0)` und `(2, 2)` aus:

<!--en:2d74d7e4-->
In TensorFlow wählt `tf.gather(X, cols, batch_dims=1)` den Eintrag `cols[i]` aus jeder Zeile `i` von `X` aus:

<!--en:5937e24b-->
Ebenso liefert ein elementweiser Vergleich eine *boolesche Maske* aus den Werten `True` und `False` (verknüpfbar mit `&`, `|` und `~`). Die Indizierung `X[mask]` fasst alle Einträge, für die `mask` den Wert `True` hat, in einem eindimensionalen Vektor zusammen:

<!--en:dcd81acb-->
Da die Ausgabelänge von `X[mask]` von den Daten abhängt, funktioniert die Indizierung mit booleschen Masken nur bei sofortiger Ausführung (Eager-Modus). Verwenden Sie in Funktionen, die mit `jax.jit` kompiliert werden, stattdessen `jnp.where`.

<!--en:bafd1481-->
Möchten wir die Form $3 \times 4$ von `X` erhalten, verwenden wir `where(mask, X, 0)`. Der Aufruf wählt Werte aus `X`, wo `mask` den Wert `True` hat, und füllt alle übrigen Positionen mit `0`:

<!--en:f0185de3-->
:numref:`fig_ndarray_gather_mask` stellt die drei Auswahlverfahren gegenüber. Sie unterscheiden sich in der Form der Ausgabe. Indextensoren liefern hier einen Eintrag pro Zeile, eine Maske liefert einen Vektor datenabhängiger Länge und `where` liefert einen Tensor mit derselben Form wie `X`.

<!--en:e717b5a8-->
![Auswahl von Einträgen der $3 \times 4$-Matrix `X`. (a) Die Indextensoren `rows = (0, 1, 2)` und `cols = (3, 0, 2)` wählen einen Eintrag pro Zeile. (b) Die Maske `(X > 2) & (X < 8)` extrahiert die fünf Einträge zwischen $2$ und $8$ als Vektor. Dagegen erhält `where` die Form von `X` und schreibt an allen übrigen Stellen $0$.](../img/ndarray-gather-mask.svg)

<!--en:57724fb4-->
### Elemente schreiben

<!--en:b6598b00-->
Ob sich ein Tensor an Ort und Stelle überschreiben lässt, hängt vom Framework ab:

<!--en:50b5bbc2-->
MXNet-Tensoren sind *veränderlich*: `X[i, j] = val` überschreibt den Wert direkt im Speicher. Wir weisen `17` dem Eintrag `X[1, 2]` zu (linearer Index $6$) und betrachten anschließend `X` und `x`:

<!--en:0b9633b1-->
PyTorch-Tensoren sind *veränderlich*: `X[i, j] = val` überschreibt den Wert direkt im Speicher. Außerdem lieferte `reshape` eine *Sicht* (engl. view) auf den ursprünglichen Puffer, ohne ihn zu kopieren. Eine Änderung an `X` ändert deshalb auch `x`. Wir weisen `17` dem Eintrag `X[1, 2]` zu (linearer Index $6$) und betrachten anschließend `X` und `x`:

<!--en:0488a7c8-->
TensorFlow-Tensoren sind *unveränderlich*. Lernbare Parameter liegen in veränderlichen Containern vom Typ `tf.Variable`. Wir aktualisieren sie mit `.assign()`:

<!--en:eebb9866-->
JAX-Arrays sind *unveränderlich*. JAX bietet funktionale Aktualisierungen wie `X.at[1, 2].set(17)`. Sie liefern ein neues Array, während `X` unverändert bleibt:

<!--en:11d7ab24-->
In MXNet bleibt `x[6]` bei `6.0`, weil `reshape` eine Kopie angelegt hat.

<!--en:77417afe-->
In PyTorch hat sich `x[6]` in `17.` geändert, weil `reshape` eine Sicht auf denselben Speicher wie `x` geliefert hat.

<!--en:d835c242-->
Ein Slice auf der linken Seite einer Zuweisung setzt einen ganzen Block auf einmal:

<!--en:b10e1541-->
## Operationen

<!--en:8b8c8f75-->
### Elementweise Operationen

<!--en:6bbaa75f-->
Die üblichen binären Rechenoperatoren (`+`, `-`, `*`, `/`, `**`) übertragen die skalare Arithmetik auf zwei Tensoren `x` und `y` gleicher Form. Jedes Paar einander entsprechender Elemente wird unabhängig verarbeitet:

<!--en:c562e2d2-->
Einstellige mathematische Funktionen wie `exp` arbeiten ebenfalls elementweise. Sie berechnen $e^{x_i}$ für jeden Eintrag von `x`:

<!--en:8ccf5103-->
Dividieren wir unseren Minibatch vom Typ `uint8` mit dem Namen `images` durch `255`, skalieren wir alle $200{,}704$ Pixel auf einmal. Dabei wechselt der Datentyp des Ergebnisses zu `float32`:

<!--en:d729dbe7-->
Auch Vergleichsoperatoren wie `x == y` arbeiten elementweise. Sie liefern einen booleschen Tensor mit dem Wert `True` an allen Positionen, an denen `x` und `y` übereinstimmen (hier nur bei Index $1$, wo beide den Wert `2.0` haben):

<!--en:7a68e3f5-->
Die Summation mit `sum()` fasst alle Einträge eines Tensors zu einem Skalar zusammen ($1 + 2 + 4 + 8 = 15$ für `x`). Da `True` als $1$ und `False` als $0$ zählt, liefert die Summe der booleschen Maske `(x == y).sum()` die Zahl der übereinstimmenden Einträge:

<!--en:b24309b1-->
Verwenden Sie in TensorFlow `tf.reduce_sum` (boolesche Werte zuerst mit `tf.cast` umwandeln).

<!--en:2b732c2e-->
### Broadcasting

<!--en:3fe4c267-->
Häufig verknüpfen wir Tensoren *unterschiedlicher* Form. Beispielsweise addieren wir einen Bias-Vektor `v` mit 4 Elementen zu jeder Zeile einer $3 \times 4$-Matrix `X`:

<!--en:d3423cf8-->
Anstatt Kopien im Speicher anzulegen, wenden Tensorbibliotheken eine zweistufige Regel an, das *Broadcasting*:

<!--en:caf27bbf-->
1. **Formen rechtsbündig ausrichten:** Ergänzen Sie das Formtupel des Tensors niedrigerer Ordnung links um Einträge mit dem Wert $1$, bis beide Tupel gleich lang sind. Zwei einander zugeordnete Dimensionen sind kompatibel, wenn sie gleich sind oder eine von ihnen den Wert $1$ hat.

<!--en:7e123c16-->
2. **Achsen der Länge eins strecken:** Jede Achse der Länge $1$ wird virtuell auf die entsprechende Länge $k > 1$ des anderen Tensors gestreckt, ohne Daten zu kopieren. Anschließend wird die elementweise Operation ausgeführt.

<!--en:8b7c4d65-->
Um zu sehen, wie beide Tensoren gleichzeitig gestreckt werden (:numref:`fig_ndarray_broadcasting`), erzeugen wir einen $3 \times 1$-Spaltenvektor `a` und einen $1 \times 2$-Zeilenvektor `b`:

<!--en:5a2b7ca2-->
![Broadcasting streckt Achsen der Länge 1. Die Spalte `a` der Form $(3, 1)$ und die Zeile `b` der Form $(1, 2)$ werden vor der elementweisen Addition auf die Form $(3, 2)$ ausgedehnt (gestrichelt: virtuelle Kopien).](../img/ndarray-broadcasting.svg)

<!--en:4a389b60-->
Bei der Addition `a + b` wird `a` virtuell auf zwei Spalten und `b` auf drei Zeilen ausgedehnt. So entsteht die $3 \times 2$-Matrix aller paarweisen Summen $a_{i0} + b_{0j}$:

<!--en:695f8011-->
Unterscheiden sich zwei einander zugeordnete Dimensionen und hat keine von beiden den Wert $1$ (etwa bei `(3, 2)` und `(2, 3)`), löst Broadcasting einen Fehler aus:

<!--en:76ab211e-->
Wir geben die letzte Zeile der Fehlermeldung von MXNet aus:

<!--en:72f2a96e-->
Wir geben den Klassennamen der Ausnahme aus:

<!--en:feb6393f-->
Ziehen wir also ein Mittelwertbild der Form $(28, 28)$ von `images` mit der Form $(256, 1, 28, 28)$ ab, gelingt das Broadcasting zu $(1, 1, 28, 28)$. Addieren wir dagegen zu abgeflachten Bildern einen Vektor mit einem Wert pro Bild, müssen wir den Vektor der Form $(256,)$ für die Bilder der Form $(256, 784)$ erst in eine Spalte der Form $(256, 1)$ umformen.

<!--en:e3d748fe-->
Eine Fehlermeldung ist noch der günstige Fall. Nach denselben Regeln dürfen wir Labels der Form $(3,)$ von Vorhersagen der Form $(3, 1)$ abziehen. Wir erhalten eine $3 \times 3$-Matrix aller paarweisen Differenzen, obwohl wir nur drei Fehlerwerte wollten (:numref:`fig_ndarray_broadcast_bug`). Der über diese Matrix gemittelte Verlust ist weiterhin eine einzelne Zahl. So fällt der Fehler erst auf, wenn das Modell nicht lernt.

<!--en:348de320-->
![Broadcasting ohne Fehlermeldung. (a) Vorhersagen `y_hat` der Form $(3, 1)$ minus Labels `y` der Form $(3,)$ ergeben alle neun paarweisen Differenzen. Nur die Diagonale (durchgezogen) enthält die drei gesuchten Fehlerwerte. (b) Formen wir `y` zu $(3, 1)$ um, liefert dieselbe Subtraktion genau diese drei Werte.](../img/ndarray-broadcast-bug.svg)

<!--en:51414c62-->
### Verketten

<!--en:64c43707-->
Mit `concatenate` (oder `cat` / `concat`) hängen wir Tensoren entlang einer *bestehenden* Achse aneinander. Alle anderen Achsen müssen gleich lang sein. Verketten wir zwei $3 \times 4$-Matrizen `X` und `Y` entlang `axis=0`, erhalten wir ihre Zeilen untereinander in der Form `(6, 4)`. Mit `axis=1` stehen ihre Spalten nebeneinander in der Form `(3, 8)`:

<!--en:a7c40ec8-->
Mit `stack` stapeln wir Tensoren gleicher Form entlang einer *neuen* Achse. So fassen wir `X` und `Y` zu einem Tensor der Ordnung $3^\textrm{rd}$ zusammen. Standardmäßig hat er die Form `(2, 3, 4)` (`axis=0`), beim Stapeln entlang der letzten Achse die Form `(3, 4, 2)` (`axis=2`):

<!--en:b21e55fe-->
## Speicher und Konvertierung

<!--en:baec9e9b-->
### Speicher anfordern und Werte direkt überschreiben

<!--en:e06c866f-->
Übliche Ausdrücke wie `Y = Y + X` legen einen neuen Puffer an und binden `Y` an die neue Adresse. Das können wir mit der Python-Funktion `id()` prüfen:

<!--en:8904224b-->
Bei Modellen mit mehreren Gigabyte an Parametern verschwenden unnötig angelegte Puffer Speicherbandbreite. Andere Referenzen zeigen unter Umständen weiter auf veraltete Puffer. Beim Vorwärtsdurchlauf dürfen Sie allerdings keine Aktivierungen überschreiben, die das automatische Differenzieren noch benötigt (siehe :numref:`sec_autograd`).

<!--en:bf156b68-->
In MXNet und PyTorch schreibt die Slice-Zuweisung `Z[:] = X + Y` das Ergebnis in einen bereits angelegten Puffer `Z`. Deshalb bleibt `id(Z)` unverändert:

<!--en:3e69c8f9-->
In TensorFlow aktualisiert `Z.assign(X + Y)` den veränderlichen Puffer vom Typ `tf.Variable` namens `Z` direkt. Daher bleibt `id(Z)` unverändert:

<!--en:50bc6933-->
In JAX sind Arrays unveränderlich. Bei `@jax.jit` kümmert sich der XLA-Compiler automatisch um die Wiederverwendung der Puffer.

<!--en:b07ca996-->
Brauchen wir den bisherigen Wert von `X` nicht mehr, können wir mit der erweiterten Zuweisung `X += Y` verlangen, dass die Summe direkt in `X` geschrieben wird. Mit `id(X)` prüfen wir, ob das Framework dies umsetzen konnte:

<!--en:b2f311d8-->
`id(X) == before` ergibt `True`: Der Wert von `X` wurde direkt überschrieben.

<!--en:e8b95dcd-->
Da `X` ein unveränderlicher Tensor ist, ergibt `id(X) == before` den Wert `False`.

<!--en:8a0a218d-->
### Umwandlung in andere Python-Objekte

<!--en:9a9b9877-->
Für Grafik- oder Datenbibliotheken wandeln wir regelmäßig Framework-Tensoren in NumPy-Arrays um und umgekehrt. Je nach Framework teilen sie sich danach den Speicher oder es entsteht eine Kopie:

<!--en:c84a8f6e-->
`X.asnumpy()` und `mnp.array(A)` legen jeweils eine unabhängige Kopie an. Deshalb bleiben beim folgenden Schreibzugriff sowohl `X[0, 0]` als auch `B[0, 0]` unverändert.

<!--en:9ea05b86-->
Auf der CPU verwenden `X.numpy()` und `torch.from_numpy(A)` **ein und denselben Speicherpuffer**. Eine Änderung an einem der Objekte wirkt sich sofort auf das andere aus. Nach dem folgenden Schreibzugriff ändern sich sowohl `X[0, 0]` als auch `B[0, 0]` auf `100.0`.

<!--en:7c39c1e3-->
`X.numpy()` und `tf.constant(A)` legen jeweils eine unabhängige Kopie an. Deshalb bleiben beim folgenden Schreibzugriff sowohl `X[0, 0]` als auch `B[0, 0]` unverändert.

<!--en:4550f68d-->
`jax.device_get(X)` liefert ein schreibgeschütztes NumPy-Array (auf der CPU unter Umständen ohne Kopie). Mit `jax.device_put(A)` übertragen wir es zurück. Da die NumPy-Sicht schreibgeschützt ist, löst der folgende Schreibzugriff einen `ValueError` aus.

<!--en:7712ee82-->
Ob `A`, `B` und `X` gemeinsamen Speicher nutzen, lässt sich mit einem Schreibzugriff prüfen. Im NumPy-Array setzen wir `A[0, 0] = 100` und sehen uns `X[0, 0]` und `B[0, 0]` an:

<!--en:bd861d87-->
Um einen Tensor mit einem Element (Ordnung $0^\textrm{th}$) in einen Python-Wert vom Typ `float` oder `int` umzuwandeln, etwa für die Protokollierung des skalaren Verlusts, verwenden wir `.item()`, `float()` oder `int()`. Dabei muss die CPU warten, bis noch ausstehende GPU-Kernel fertig sind:

<!--en:d3669e77-->
Um einen Tensor mit einem Element (Ordnung $0^\textrm{th}$) in einen Python-Wert vom Typ `float` oder `int` umzuwandeln, etwa für die Protokollierung des skalaren Verlusts, verwenden wir `.numpy()`, `float()` oder `int()`. Dabei muss die CPU warten, bis noch ausstehende GPU-Kernel fertig sind:

<!--en:dd106620-->
> **Tipp: Programmierschnittstellen erkunden.** Bei einem unbekannten Modul oder einer unbekannten Funktion listet `dir(obj)` die verfügbaren Attribute auf (auch per Tab-Vervollständigung). In Jupyter zeigen `help(fn)` oder `fn?` die Signatur und den Dokumentationstext, `fn??` den Python-Quellcode. Prüfen Sie Ihr Verständnis mit einer kleinen Testeingabe, deren Ausgabe Sie von Hand vorhersagen können.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:dbea8ceb-->
Tensoren speichern Eingaben, Parameter, Aktivierungen und Gradienten als $n$-dimensionale Arrays. Ihre Form und ihren Datentyp beschreiben `shape` und `dtype`. Im Deep Learning führen wir Berechnungen für ganze Tensoren auf einmal aus. So skalieren wir alle $200{,}704$ Pixel eines Minibatches mit einer einzigen kompilierten Operation (`images / 255`). Wir formen Achsen mit `reshape` um, wählen oder ändern Teilmengen mit ganzzahligen Indizes, Slices und booleschen Masken (`where`) und verbinden Tensoren mit `concatenate` und `stack`. Tensoren unterschiedlicher Form verknüpfen wir durch *Broadcasting*: Wir richten die Formen rechtsbündig aus und strecken Achsen der Länge 1. Bei veränderlichen Tensoren verwenden direkte Aktualisierungen (`Z[:] = ...`, `X += Y`) den Speicher erneut. Andernfalls übernehmen veränderliche Container oder der Compiler diese Aufgabe.

<!--en:2294243a-->
## Übungen

<!--en:04c3f832-->
1. [code] **Automatische Achsenlänge beim Umformen.** Ein Minibatch mit $n$ Graustufenbildern der Auflösung $28 \times 28$ liege als Tensor der Form $(n, 1, 28, 28)$ vor. Jedes Bild soll zu einem Zeilenvektor mit $784$ Pixelwerten abgeflacht werden. Überprüfen Sie jede der folgenden Vorhersagen mit Code:
    1. Welche Form ergibt sich, wenn Sie den Minibatch zu `(-1, 784)` beziehungsweise `(256, -1)` umformen? Betrachten Sie zunächst einen vollständigen Minibatch mit $n = 256$ Bildern, dann den letzten, kleineren Minibatch einer Epoche mit $n = 100$ Bildern. Welche der beiden Zielformen funktioniert zuverlässig für *jede* Batchgröße $n$?
    1. Angenommen, die Bilder wurden versehentlich im 3-Kanal-RGB-Format der Form $(n, 3, 28, 28)$ geladen. Was geschieht beim Umformen des Tensors zu `(-1, 784)`? Erkennt `reshape(-1, 784)` die falsche Kanalzahl oder verfälscht der Aufruf unbemerkt die Batchdimension?

<!--en:f62268d4-->
2. [code] **Formen und unbemerkte Fehler beim Broadcasting.** Broadcasting richtet zwei Formtupel rechtsbündig aus.
    1. Erzeugen Sie zwei Tensoren der Ordnung $3^\textrm{rd}$ namens `a` und `b`. Ihre Formen sollen sich in jeder Achse unterscheiden und dennoch Broadcasting erlauben. Bestimmen Sie die Form von `a + b` von Hand und überprüfen Sie Ihre Vorhersage mit Code.
    1. Ein Regressionsmodell liefert für fünf Beispiele eine Spalte mit Vorhersagen `y_hat` der Form $(5, 1)$. Die tatsächlichen Zielwerte `y` liegen als 1D-Vektor der Form $(5,)$ vor. Sagen Sie die Form des Residuums `y_hat - y` voraus. Weisen Sie nun `y` die Werte `[0.0, 1.0, 2.0, 3.0, 4.0]` zu und setzen Sie `y_hat = y.reshape(5, 1)`, sodass jede Vorhersage exakt ihrem wahren Label entspricht. Berechnen Sie den Mittelwert der quadrierten Einträge von `y_hat - y`. Erklären Sie, warum dieser „mittlere quadratische Fehler“ trotz perfekter Vorhersagen von null abweicht. Zeigen Sie, wie Sie `y` (oder `y_hat`) umformen müssen, damit jede Vorhersage nur mit ihrem eigenen Label verglichen wird.

<!--en:dee00484-->
2. [code] **Formen und unbemerkte Fehler beim Broadcasting.** Broadcasting richtet zwei Formtupel rechtsbündig aus.
    1. Erzeugen Sie zwei Tensoren der Ordnung $3^\textrm{rd}$ namens `a` und `b`. Ihre Formen sollen sich in jeder Achse unterscheiden und dennoch Broadcasting erlauben. Bestimmen Sie die Form von `a + b` von Hand und überprüfen Sie Ihre Vorhersage mit Code.
    1. Ein Regressionsmodell liefert für fünf Beispiele eine Spalte mit Vorhersagen `y_hat` der Form $(5, 1)$. Die tatsächlichen Zielwerte `y` liegen als 1D-Vektor der Form $(5,)$ vor. Sagen Sie die Form des Residuums `y_hat - y` voraus. Weisen Sie nun `y` die Werte `[0.0, 1.0, 2.0, 3.0, 4.0]` zu und setzen Sie `y_hat = tf.reshape(y, (5, 1))`, sodass jede Vorhersage exakt ihrem wahren Label entspricht. Berechnen Sie den Mittelwert der quadrierten Einträge von `y_hat - y`. Erklären Sie, warum dieser „mittlere quadratische Fehler“ trotz perfekter Vorhersagen von null abweicht. Zeigen Sie, wie Sie `y` (oder `y_hat`) umformen müssen, damit jede Vorhersage nur mit ihrem eigenen Label verglichen wird.

<!--en:7d1c39d7-->
3. [code] **Arithmetische Grenzen von Datentypen.** Die Rohpixel in `images` liegen als vorzeichenlose 8-Bit-Ganzzahlen (`uint8`) vor. Die Schichten eines neuronalen Netzes rechnen dagegen mit Gleitkommazahlen (`float32`).
    1. Welches Ergebnis erwarten Sie, wenn Sie `100` zu einem Tensoreintrag vom Typ `uint8` mit dem Wert `200` addieren? Prüfen Sie Ihre Vorhersage mit Code. Erklären Sie, warum das Ergebnis beim Überschreiten der Obergrenze an den Anfang des Wertebereichs springt. Zeigen Sie, wie Sie das erste Bild von `images` um $100$ Helligkeitsstufen aufhellen, ohne einen Ganzzahlüberlauf zu verursachen (durch Begrenzen der Werte oder Wechsel des Datentyps).
    1. Eine Gleitkommazahl nach IEEE 754 vom Typ `float32` besitzt eine Genauigkeit von $24$ Bit im Signifikanden (Mantisse). Erklären Sie, warum `float32` jede ganze Zahl von null bis $2^{24} = 16{,}777{,}216$ exakt darstellen kann. Prüfen Sie mit Code, dass die Auswertung von `float32(2**24) + float32(1)` den Zahlenwert unverändert bei $16{,}777{,}216$ lässt. Eine Trainingsschleife zählt die verarbeiteten Beispiele oder Token in einem skalaren Zähler vom Typ `float32` und erhöht ihn jedes Mal um $1$. Nach wie vielen Beispielen zählt er gar nicht mehr weiter? Welchen Datentyp sollten Sie stattdessen verwenden?

<!--en:b7e3d5e9-->
4. [code] **Verknüpfte Operationen im selben Speicher.** ● Berechnen Sie `(A + B) * (-A / 2)` für zwei gleich geformte Tensoren vom Typ `float32` namens `A` und `B`, *ohne Speicher für einen Zwischentensor anzufordern*. Verwenden Sie nur die direkt überschreibenden PyTorch-Methoden (`add_`, `div_`, `neg_`, `mul_`) oder das Schlüsselwortargument `out=`. Prüfen Sie, dass `A.data_ptr()` und `B.data_ptr()` gleich bleiben und das Endergebnis mit dem separat berechneten Ausdruck `(A + B) * (-A / 2)` übereinstimmt. Erklären Sie, warum die Zuweisung `A[:] = A + B` die Anforderung, keinen Zwischenspeicher anzulegen, *nicht* erfüllt.

    *Nach Nicolas P. Rougier, [100 NumPy-Übungen](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md), Aufgabe 35.*

<!--en:82846945-->
4. [code] **Verknüpfte Operationen im selben Speicher.** ● Berechnen Sie `(A + B) * (-A / 2)` für zwei gleich geformte Arrays vom Typ `float32` namens `A` und `B`, *ohne Speicher für ein Zwischenarray anzufordern*. Verwenden Sie nur die direkt überschreibenden erweiterten Zuweisungsoperatoren `+=`, `*=` und `/=`. Prüfen Sie, dass das Endergebnis mit dem separat berechneten Ausdruck `(A + B) * (-A / 2)` übereinstimmt. Erklären Sie, warum die Zuweisung `A[:] = A + B` die Anforderung, keinen Zwischenspeicher anzulegen, *nicht* erfüllt.

    *Nach Nicolas P. Rougier, [100 NumPy-Übungen](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md), Aufgabe 35.*

<!--en:1c47aae5-->
4. [code] **Verknüpfte Operationen im selben Speicher.** ● Gewöhnliche TensorFlow-Tensoren sind unveränderlich. Jeder arithmetische Operator legt daher einen neuen Tensor an. Verpacken Sie zwei gleich geformte Tensoren vom Typ `float32` namens `A` und `B` in Container vom Typ `tf.Variable`. Berechnen Sie `(A + B) * (-A / 2)` so, dass das Endergebnis in einer der beiden Variablen liegt. Verwenden Sie `assign`, `assign_add` und verwandte Variablenmethoden. Führen Sie die Rechenschritte einzeln auf. Welche legen bei sofortiger Ausführung noch einen temporären `Tensor` an? Erklären Sie, warum sich diese Zwischenergebnisse ohne Graphkompilierung mit `@tf.function` bei sofortiger Ausführung nicht vermeiden lassen.

    *Nach Nicolas P. Rougier, [100 NumPy-Übungen](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md), Aufgabe 35.*

<!--en:75c3d141-->
4. [code] **Verknüpfte Operationen im selben Speicher und Fusion durch den Compiler.** ● JAX-Arrays sind unveränderlich. Daher legt jede unmittelbar ausgeführte arithmetische Operation ein neues Array an. Schreiben Sie `(A + B) * (-A / 2)` in eine Python-Funktion `f(A, B)`. Zählen Sie, wie viele Zwischenarrays bei unmittelbarer Auswertung entstehen. Kompilieren Sie dann `f` mit `jax.jit(f)`. Untersuchen Sie mit `jax.jit(f).lower(A, B).compile().as_text()` das in die HLO-Darstellung übersetzte Programm: Wie viele fusionierte Kernel bleiben übrig? Vergleichen Sie die kompilierte Ausführung mit einer NumPy-Implementierung, die `A` direkt überschreibt. Welche Zwischenarrays der sofortigen Ausführung spart `jax.jit` automatisch ein? Wann legt die kompilierte Funktion trotzdem einen Ausgabepuffer an, sofern Sie `donate_argnums` nicht angegeben haben?

    *Nach Nicolas P. Rougier, [100 NumPy-Übungen](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md), Aufgabe 35.*

<!--en:3354eba6-->
5. [code] **Python-Schleifen und Broadcasting im Vergleich.** ● Gegeben seien ein 1D-Vektor `a` mit $200$ Elementen und ein 1D-Vektor `b` mit $300$ Elementen. Gesucht ist die $200 \times 300$-Matrix $D$ der paarweisen absoluten Differenzen mit den Einträgen $d_{ij} = |a_i - b_j|$.
    1. Berechnen Sie $D$ zuerst mit zwei verschachtelten Python-Schleifen vom Typ `for`. Verwenden Sie dann eine einzige Subtraktion mit Broadcasting. Formen Sie dazu `a` zu `(200, 1)` und `b` zu `(1, 300)` um. Prüfen Sie, dass die Matrizen numerisch übereinstimmen. Messen Sie die tatsächlich verstrichene Laufzeit beider Verfahren und geben Sie den Beschleunigungsfaktor an.
    1. Beim Broadcasting müssen die Eingaben `a` und `b` nicht kopiert werden. Die Ausgabematrix $D$ muss dennoch vollständig im Speicher liegen. Wie viele Megabyte belegt $D$ bei Verwendung von `float32`, wenn `a` und `b` je $10{,}000$ Elemente enthalten? Was folgt daraus für den maximalen Speicherbedarf äußerer Operationen mit Broadcasting?

<!--en:04aaf7ed-->
[Diskussion](https://d2l.discourse.group/t/26)

<!--en:217ad141-->
[Diskussion](https://d2l.discourse.group/t/27)

<!--en:effab0b9-->
[Diskussion](https://d2l.discourse.group/t/187)

<!--en:0bdbb840-->
[Diskussion](https://d2l.discourse.group/t/17966)

<!--en:b99fff1d-->
Tensoren als zusammenhängende Speicherbereiche, geometrische Sichten und Grundlage paralleler Berechnungen<br>**Speicherpuffer · Formen und Datentypen · Sichten ohne Kopie · Broadcasting · Speicherverwaltung**

<!--en:778c16ef-->
::: {.slide title="Warum Tensoren? Von Python-Zeigern zu parallelen Kerneln"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:14675fec-->
- **Größenordnung eines Batches:** Ein Batch aus 256 Fashion-MNIST-Bildern ($28 \times 28$) enthält $200{,}704$ Pixel. Moderne Netze aktualisieren Milliarden von Parametern pro Schritt.

<!--en:a173f09f-->
- **Python-Listen (`PyObject*`):** 8-Byte-Zeiger verweisen auf verstreute 24-Byte-Objekte auf dem Heap. Das führt zu Cache-Fehlzugriffen und Mehraufwand durch Schleifen im Interpreter.

<!--en:5d595ca8-->
- **Zusammenhängende Tensorpuffer:** Die Werte (`uint8`, `float32`, `bfloat16`) liegen ohne Objekthüllen unmittelbar hintereinander im RAM/HBM. Ein vektorisierter CPU-/GPU-Kernel verarbeitet sie gemeinsam, mit Unterstützung für Autograd.

<!--en:e0196559-->
::: {.slide title="Das gemeinsame Prinzip: Speicherpuffer und Metadaten"}

<!--en:c5ccffef-->
[Systemsicht]{.kicker}

<!--en:0b79e11b-->
Jeder Tensor verbindet einen **zusammenhängenden 1D-Speicherpuffer** mit einem kleinen **Metadatenkopf** (`shape`, `strides`, `offset`, `dtype`):

<!--en:27b4502e-->
- **Sichten mit Strides in $O(1)$** (Slices wie `X[1:3]`, Transposition, Broadcasting mit $s_r = 0$ sowie zusammenhängendes `reshape` bei Unterstützung für Sichten) legen einen neuen Metadatenkopf für *denselben* Puffer an.

<!--en:bba10d25-->
- **Speicheranforderungen in $O(N)$** (`X + Y`, Gather, Masken, `cat`) schreiben in einen neuen Puffer.

<!--en:505a1df8-->
::: {.slide title="Achsen, Ordnung (Rang) und Form"}

<!--en:d710c13a-->
[Geometrie]{.kicker}

<!--en:649036ed-->
- Die **Ordnung (der Rang)** ist die Anzahl der Achsen. Die **Form** enthält jede Achsenlänge. Die **Größe** ist ihr Produkt:
  - **0D-Skalar `()`** (z. B. Verlust $\mathcal{L}$) $\;\to\;$ **1D-Vektor `(d,)`** (Bias $\mathbf{b}$) $\;\to\;$ **2D-Matrix `(n, d)`** (Gewichte $\mathbf{W}$).
  - **4D-Bildbatch `(N, C, H, W)`:** Unser Fashion-MNIST-Minibatch `(256, 1, 28, 28)` enthält $256 \times 1 \times 28 \times 28 = 200{,}704$ Pixel.

<!--en:0239f042-->
::: {.slide title="Tensoren erzeugen und Metadaten prüfen"}

<!--en:e0590469-->
[Einstieg]{.kicker}

<!--en:78966a08-->
- `arange(n)` legt einen 1D-Tensor mit $n$ aufeinanderfolgenden Werten $[0, 1, \ldots, n-1]$ in einem einzigen zusammenhängenden Puffer an.

<!--en:d5a67abd-->
- Prüfen Sie bei der Fehlersuche in jedem Modell `.shape` (Achsenlängen) und die Elementzahl (`.numel()` oder `.size`, stets gleich $\prod_r d_r$).

<!--en:4bb85d14-->
::: {.slide title="Datentypen bestimmen Genauigkeit, Wertebereich und Speicherbedarf"}

<!--en:f7d5e909-->
[Datentypen]{.kicker}

<!--en:839cbf66-->
- **`float32` (4 B):** Standard für die Hauptkopie der Gewichte, Gradienten und Verlustreduktionen ($4\text{ GB} / 10^9$ Parameter).

<!--en:2b63bbe6-->
- **`bfloat16` (2 B):** Behält den 8-Bit-Exponenten von `float32` bei (**gleicher Dynamikbereich**, vermeidet Überläufe) und halbiert die HBM-Datenübertragung bei Matrixprodukten mit Tensor Cores.

<!--en:95e43866-->
- **`uint8` (1 B) und `int64` (8 B):** Kompakte Speicherung der Rohpixel ($0\ldots 255$) und exakte Klassen- oder Tokenindizes.

<!--en:c2465420-->
::: {.slide title="Deterministische Konstruktion und zufällige Initialisierung der Gewichte"}

<!--en:1580e77b-->
[Initialisierung]{.kicker}

<!--en:991cb1e8-->
- **Konstanten und Literale:** `zeros` und `ones` legen feste Puffer für Bias-Terme, Padding-Masken und Akkumulatoren an. Aus Python-Listen entstehen kleine Nachschlagetabellen.

<!--en:a87d0f07-->
- **Warum Gewichte zufällig sein müssen:** Beginnen alle Neuronen einer Schicht mit identischen Gewichten, berechnen sie dieselbe Aktivierung und erhalten denselben Gradienten. Ziehen aus $\mathcal{N}(0, 1)$ (`randn`) bricht die Symmetrie.

<!--en:b4a8e734-->
::: {.slide title="Reproduzierbarer Zufall: Startwert und Zustand des Generators" except="jax"}

<!--en:0101bb1c-->
[Reproduzierbarkeit]{.kicker}

<!--en:f9f8aa19-->
- Zufall spielt beim Training überall eine Rolle: Initialisierung der Gewichte, Mischen der Minibatches, Dropout-Masken und Datenaugmentierung.

<!--en:f73a242e-->
- Pseudozufallsgeneratoren (PRNGs) sind deterministische Zustandsautomaten. Derselbe **Startwert** liefert für wiederholbare Experimente genau dieselbe Folge von Ziehungen.

<!--en:3122909d-->
- **Globaler Zustand:** Das Framework schaltet den Generator bei jeder Ziehung weiter. Ein Startwert legt daher die gesamte nachfolgende Sequenz fest.

<!--en:4a47b9ac-->
::: {.slide title="Reproduzierbarer Zufall: Startwert und Zustand des Generators" only="jax"}

<!--en:0101bb1c-2-->
[Reproduzierbarkeit]{.kicker}

<!--en:f9f8aa19-2-->
- Zufall spielt beim Training überall eine Rolle: Initialisierung der Gewichte, Mischen der Minibatches, Dropout-Masken und Datenaugmentierung.

<!--en:f73a242e-2-->
- Pseudozufallsgeneratoren (PRNGs) sind deterministische Zustandsautomaten. Derselbe **Startwert** liefert für wiederholbare Experimente genau dieselbe Folge von Ziehungen.

<!--en:0a430843-->
- **Expliziter Zustand:** JAX teilt einen expliziten `PRNGKey` auf, statt einen globalen Generator weiterzuschalten. So bleibt paralleler JIT-Code rein funktional.

<!--en:7bd087ed-->
::: {.slide title="Umformen deutet Achsen in zeilenweiser Reihenfolge neu"}

<!--en:ceb57638-->
[Umformen]{.kicker}

<!--en:6b7323fd-->
- `x.reshape(3, 4)` ersetzt die Form `(12,)` durch `(3, 4)`, ohne die 12 Elemente zu verschieben (**zeilenweise Anordnung**: Spalten-Stride $1$, Zeilen-Stride $4$).

<!--en:3e65fa0e-->
- Bei zusammenhängenden, veränderlichen Tensoren ändert `reshape` nur die Metadaten für Form und Strides in $O(1)$ Zeit. Eine Kopie entsteht nur bei nicht zusammenhängender Anordnung oder wenn das Backend kopiert.

<!--en:bdd8106a-->
::: {.slide title="Automatische Längenbestimmung (-1) und Achsen der Länge eins"}

<!--en:d6fe8036-->
[Umformen in Verarbeitungsketten]{.kicker}

<!--en:325646e8-->
- **Abflachen mit `-1`:** `images.reshape(-1, 784)` ermittelt die Batchgröße $\textrm{size}/784 = 256$ und funktioniert auch mit kleineren letzten Batches einer Epoche.

<!--en:7f97f90a-->
- **Achsen der Länge 1 (`None` / `squeeze`):** `None` fügt eine Achse der Länge 1 für Batches oder Broadcasting ein. `squeeze` entfernt Achsen der Länge 1.

<!--en:f77a7fcb-->
::: {.slide title="Ganzzahlige Indizes senken die Ordnung, Slices erhalten sie"}

<!--en:ba434d5c-->
[Teilmengen auswählen]{.kicker}

<!--en:ed229995-->
- **Ein ganzzahliger Index entfernt eine Achse** (Ordnung sinkt um 1): `X[-1]` liefert die letzte Zeile als 1D-Vektor `(4,)`, `images[0]` ein Bild der Form `(1, 28, 28)`.

<!--en:e04a7045-->
- **Ein Slice (`start:stop`) erhält die Achse** (Ordnung bleibt gleich): `X[1:3]` liefert die Zeilen 1 und 2 als Sicht auf eine Matrix der Form `(2, 4)`. Mit `images[:32]` wählen wir einen Teilbatch aus 32 Bildern.

<!--en:0dfef9e5-->
::: {.slide title="Verständnisfrage: Formen und Sichten ohne Kopie vorhersagen"}

<!--en:929b7671-->
[Verständnisfrage]{.kicker}

<!--en:68661989-->
Gegeben sei `images` mit der Form `(256, 1, 28, 28)`. Welche Ausgabeform und welches Speicherverhalten erwarten Sie bei jedem Ausdruck?

<!--en:ed758382-->
1. `images[5]` im Vergleich zu `images[5:6]`

<!--en:538b3d33-->
3. `images.reshape(256, -1)` im Vergleich zu `images.reshape(-1, 784)` beim letzten Batch einer Epoche mit 96 Bildern der Form `(96, 1, 28, 28)`

<!--en:557b98b6-->
**Antworten:**

<!--en:e6c6443f-->
1. `images[5]` hat die Form **`(1, 28, 28)`** (Achse 0 entfällt). `images[5:6]` hat die Form **`(1, 1, 28, 28)`** (Achse 0 bleibt mit Länge 1 erhalten). Beide liefern Sichten ohne Kopie in $O(1)$.

<!--en:d23bfc02-->
2. **`(64, 1, 14, 14)`**: Wählt 64 Bilder und jede zweite Zeile und Spalte. Die räumlichen Strides verdoppeln sich. Das Ergebnis bleibt ein Slice mit Strides.

<!--en:a0e66437-->
3. `reshape(256, -1)` **liefert fälschlich die Form (256, 294)** ($96 \times 784$ ist durch 256 teilbar). `reshape(-1, 784)` ermittelt zuverlässig **`(96, 784)`**.

<!--en:8232a43c-->
::: {.slide title="Ganzzahlige Indizes und boolesche Masken beim maschinellen Lernen"}

<!--en:cd8888c4-->
[Gezielte Auswahl]{.kicker}

<!--en:e50cf724-->
- **Gather mit ganzzahligen Indizes:** Paart Zeilen- und Spaltenindizes. Damit wählt die Kreuzentropie für jedes Beispiel die Bewertung der tatsächlichen Klasse aus.

<!--en:be526d20-->
- **Maske und `where`:** Eine Maske fasst die Treffer zu einem **1D-Vektor** zusammen. `where(mask, X, 0)` erhält die 2D-Form für ReLU und Attention-Masken.

<!--en:be62489c-->
::: {.slide title="Sichten und Veränderlichkeit: Schreiben über einen umgeformten Tensor" only="pytorch"}

<!--en:d1c95dc0-->
[Sichten und Änderungen]{.kicker}

<!--en:5a15ccae-->
- Da `X = x.reshape(3, 4)` eine **Sicht** auf den Speicherpuffer von `x` ist, schreibt `X[1, 2] = 17` direkt an den Versatz $1 \times 4 + 2 = 6$. Damit ändert sich auch `x[6]` auf `17`.

<!--en:fb7c25e6-->
::: {.slide title="Direkte Zuweisung bei veränderlichen Tensoren" only="mxnet"}

<!--en:d1c95dc0-2-->
[Sichten und Änderungen]{.kicker}

<!--en:a8143282-->
- Indizes links von `=` überschreiben Elemente direkt im Puffer von `X` (`X[1, 2] = 17`, `X[:2, :] = 12`). In MXNet kopiert `reshape` den Puffer, sodass `x` unverändert bleibt.

<!--en:53369b52-->
::: {.slide title="Rein funktional: unveränderliche Arrays und .at[].set() in JAX" only="jax"}

<!--en:d1c95dc0-3-->
[Sichten und Änderungen]{.kicker}

<!--en:41758b1c-->
- JAX-Arrays sind **unveränderlich**. Dadurch kann XLA reine Funktionen gefahrlos verschmelzen und differenzieren. Änderungen verwenden `X.at[1, 2].set(17)` und `X.at[:2, :].set(12)` (in `@jax.jit` kann der Compiler denselben Puffer wiederverwenden).

<!--en:52115b36-->
::: {.slide title="Unveränderliche Tensoren und veränderlicher Zustand in tf.Variable" only="tensorflow"}

<!--en:d1c95dc0-4-->
[Sichten und Änderungen]{.kicker}

<!--en:152e6e79-->
- In TensorFlow sind Werte vom Typ `tf.Tensor` **unveränderlich**. Trainierbare Parameter liegen in veränderlichen Containern vom Typ **`tf.Variable`**, die `.assign(...)` direkt aktualisiert.

<!--en:8ab2a5ef-->
::: {.slide title="Vektorisierte elementweise Operationen und Typanhebung" except="tensorflow"}

<!--en:4823693b-->
[Elementweise Berechnung]{.kicker}

<!--en:7fbbfad7-->
- Skalare Operatoren (`+`, `-`, `*`, `/`, `**`, `exp`) wirken elementweise: $c_i = f(x_i, y_i)$. Der Ausdruck `x * y` bezeichnet das **Hadamard-Produkt** $\mathbf{x} \odot \mathbf{y}$ und keine Matrixmultiplikation.

<!--en:286a156e-->
- **Typanhebung:** Dividieren wir die Pixel vom Typ `uint8` in `images` durch `255`, erhalten wir Werte vom Typ `float32` in $[0, 1]$ ($0.8\text{ MB}$). Damit können wir Gradienten berechnen.

<!--en:74d3ac75-->
::: {.slide title="Vektorisierte elementweise Operationen und Typanhebung" only="tensorflow"}

<!--en:4823693b-2-->
[Elementweise Berechnung]{.kicker}

<!--en:7fbbfad7-2-->
- Skalare Operatoren (`+`, `-`, `*`, `/`, `**`, `exp`) wirken elementweise: $c_i = f(x_i, y_i)$. Der Ausdruck `x * y` bezeichnet das **Hadamard-Produkt** $\mathbf{x} \odot \mathbf{y}$ und keine Matrixmultiplikation.

<!--en:947549d3-->
::: {.slide title="Vektorisierte elementweise Operationen und Typanhebung (Fortsetzung)" only="tensorflow"}

<!--en:4823693b-3-->
[Elementweise Berechnung]{.kicker}

<!--en:286a156e-2-->
- **Typanhebung:** Dividieren wir die Pixel vom Typ `uint8` in `images` durch `255`, erhalten wir Werte vom Typ `float32` in $[0, 1]$ ($0.8\text{ MB}$). Damit können wir Gradienten berechnen.

<!--en:0324c883-->
::: {.slide title="Boolesche Vergleiche und Reduktionen entlang von Achsen"}

<!--en:e2594c05-->
[Vergleiche und Reduktionen]{.kicker}

<!--en:7e7d5d4b-->
- **Elementweise Vergleiche** (`x == y`, `x < y`, `x >= y`) liefern einen booleschen Maskentensor exakt derselben Form.

<!--en:c3244691-->
- **Reduktionen** (`sum`, `mean`, `max`) fassen Werte entlang einer oder aller Achsen zusammen. Die Reduktion über alle Achsen ergibt einen **skalaren 0D-Tensor** der Form `()`.

<!--en:cb3e043e-->
- **Batchgenauigkeit mit einem Aufruf:** Da `True` als `1` und `False` als `0` zählt, ermittelt `(predictions == labels).sum()` die Anzahl korrekter Vorhersagen ohne Schleifen.

<!--en:0d66d82d-->
::: {.slide title="Broadcasting: virtuelle Wiederholung ohne Kopieren"}

<!--en:f09f2998-->
[Broadcasting]{.kicker}

<!--en:59a838df-->
Bei unterschiedlichen Formen richtet **Broadcasting** die Achsen aneinander aus und erweitert Achsen der Länge 1 virtuell in zwei Schritten:

<!--en:ae8e9fd0-->
1. **Formen rechtsbündig ausrichten** und dem Tensor niedrigerer Ordnung links Achsen der Länge `1` hinzufügen: `(3, 4)` und `(4,)` werden als `(3, 4)` und `(1, 4)` ausgerichtet.

<!--en:f9ea115e-->
2. **Abgleichen oder strecken:** Die Längen jeder Achse müssen übereinstimmen oder eine von ihnen muss `1` sein.

<!--en:d681743b-->
**Ohne Kopie:** Beim Strecken einer Achse der Länge 1 auf die Länge $n$ wird ihr **Stride auf `0` gesetzt**. Der Kernel liest dieselbe Speicheradresse $n$-mal, ohne Daten zu kopieren.

<!--en:7233882f-->
::: {.slide title="Wann Broadcasting scheitert und wann Fehler unbemerkt bleiben"}

<!--en:bafd8aa9-->
[Fallstricke beim Broadcasting]{.kicker}

<!--en:501f4648-->
- **Fehlermeldung (`(3, 2)` + `(2, 3)`):** Die letzten Achsen haben verschiedene Längen ($2 \neq 3$), und keine hat die Länge `1`.

<!--en:ce8e1df9-->
- **Unbemerkter Fehler (`(B, 1)` − `(B,)` $\to$ `(B, B)`):** Rechtsbündige Ausrichtung behandelt `y: (B,)` als `(1, B)`. Dadurch liefert `y_hat - y` alle $B^2$ paarweisen Differenzen.

<!--en:3e6228b0-->
::: {.slide title="Tensoren verbinden: Verketten und Stapeln"}

<!--en:0fc2794e-->
[Tensoren kombinieren]{.kicker}

<!--en:fb823954-->
- **`concatenate` / `cat` verkettet entlang einer *bestehenden* Achse** (Ordnung bleibt gleich):
  - `axis=0` stapelt Zeilen: $(3, 4) + (3, 4) \to (6, 4)$ (Batches zusammenführen).
  - `axis=1` verbindet Spalten: $(3, 4) + (3, 4) \to (3, 8)$ (Merkmale zusammenführen).

<!--en:6017a4db-->
- **`stack` erzeugt eine *neue* Achse** (Ordnung $+1$): Zwei Matrizen der Form $(3, 4)$ ergeben beim Stapeln die Form $(2, 3, 4)$ mit `axis=0` oder $(3, 4, 2)$ mit `axis=2`.

<!--en:64e7e7df-->
::: {.slide title="Speicherverwaltung: Namen neu binden oder Puffer überschreiben" except="jax,tensorflow"}

<!--en:19590d36-->
[Speichereffizienz]{.kicker}

<!--en:76991d5c-->
- **Neubindung (`Y = Y + X`):** Das Ergebnis von `Y + X` landet in einem **neuen Puffer**, an den `Y` gebunden wird (`id(Y)` ändert sich, $4\text{ GB}$ je Zwischenwert bei $10^9$ Parametern vom Typ `float32`).

<!--en:f841321d-->
- **Direktes Überschreiben (`Z[:] = X + Y` oder `X += Y`):** Schreibt in den vorhandenen Puffer (`id` bleibt gleich). Für Optimierungsschritte geeignet, nie für Aktivierungen, die der Rückwärtsdurchlauf noch braucht.

<!--en:071fb042-->
::: {.slide title="Speicherverwaltung: Variablen und unveränderliche Tensoren" only="tensorflow"}

<!--en:19590d36-2-->
[Speichereffizienz]{.kicker}

<!--en:d8d48069-->
- **Unveränderlicher `tf.Tensor` (`Y = Y + X` oder `X += Y`):** Legt einen neuen Puffer an und bindet den Python-Namen neu (`id` ändert sich).

<!--en:a24d3272-->
- **Veränderliche `tf.Variable` (`Z.assign(X + Y)`):** Aktualisiert Puffer trainierbarer Parameter direkt, ohne eine zweite Kopie anzulegen.

<!--en:c2b4ad60-->
::: {.slide title="Speicherverwaltung: unveränderliche Arrays und Pufferwiederverwendung in XLA" only="jax"}

<!--en:19590d36-3-->
[Speichereffizienz]{.kicker}

<!--en:b0429661-->
- Bei sofortiger Ausführung in Python legen `Y = Y + X` und `X += Y` beide ein neues JAX-Array an und binden den Namen neu (`id` ändert sich).

<!--en:eb305b4f-->
- Innerhalb von **`@jax.jit`** verschmilzt der XLA-Compiler Operationen und **übernimmt Eingabepuffer oder verwendet sie erneut**, sobald die bisherigen Werte nicht mehr benötigt werden.

<!--en:70aaa0df-->
::: {.slide title="Datenaustausch ohne Kopie mit NumPy und Umwandlung in Python-Skalare" only="pytorch"}

<!--en:401c0022-->
[Austausch mit Python]{.kicker}

<!--en:a0f317d9-->
- **Gemeinsamer CPU-Puffer (`X.numpy()` / `torch.from_numpy(A)`):** Alle nutzen denselben zusammenhängenden RAM-Puffer (**ohne Kopie**). Wer `A` ändert, ändert auch `X` und `B`.

<!--en:d54609f6-->
- **Skalar auslesen (`.item()`, `float()`):** Wandelt einen Tensor mit 1 Element in eine Python-Zahl um (erzwingt eine Synchronisation von GPU und CPU, daher nur in Abständen protokollieren).

<!--en:16897b32-->
::: {.slide title="Datenaustausch mit NumPy und Auslesen von Skalaren" only="mxnet,tensorflow"}

<!--en:401c0022-2-->
[Austausch mit Python]{.kicker}

<!--en:064f5956-->
- **Umwandlung nach NumPy (`X.asnumpy()` / `X.numpy()`):** Kopiert den Puffer zum Plotten oder für pandas oder SciPy in ein unabhängiges NumPy-Array vom Typ `ndarray` (Änderungen an `A` lassen `X` und `B` unverändert).

<!--en:0d9ad65e-->
- **Skalar auslesen (`.item()`, `float()`, `int()`):** Wandelt einen Tensor mit 1 Element in einen Python-Skalar um (wartet auf den Abschluss ausstehender GPU-Kernel).

<!--en:ab9b3cc5-->
::: {.slide title="Schreibgeschützte NumPy-Sichten ohne Kopie und Auslesen von Skalaren in JAX" only="jax"}

<!--en:401c0022-3-->
[Austausch mit Python]{.kicker}

<!--en:5dedeced-->
- **`jax.device_get(X)` / `np.asarray(X)`:** Wandelt ein JAX-Array in ein NumPy-Array vom Typ `ndarray` um. Auf der CPU können beide denselben Speicher nutzen. JAX schützt die NumPy-Sicht daher **vor Schreibzugriffen**, damit das Array unveränderlich bleibt.

<!--en:95f44065-->
- **Skalar auslesen (`.item()`, `float()`, `int()`):** Wandelt ein Array mit 1 Element in eine Python-Zahl um (synchronisiert Host und Gerät).

<!--en:a3bc587f-->
::: {.slide title="Zusammenfassung: Sichten über Metadaten und neue Speicherpuffer"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:821bd817-->
| Operation | Beispiel | Ausgabeform | Speicherverhalten | Haupteinsatz im Deep Learning |

<!--en:d027a570-->
| **Umformen / Abflachen** | `images.reshape(-1, 784)` | `(256, 784)` | **Sicht** ($O(1)$ bei zusammenhängenden Daten) oder Kopie | Räumliche 4D-Tensoren an lineare 2D-Schichten übergeben |

<!--en:e68a7ecb-->
| **Slice / Stride** | `images[:32]`, `X[1:3]` | `(32, 1, 28, 28)` | **Sicht ohne Kopie** ($O(1)$) | Mikrobatches, Bildausschnitte, Unterabtastung mit Stride |

<!--en:52cb6a1e-->
| **Broadcasting** | `X + b` (`(3, 4) + (4,)`) | `(3, 4)` | **Virtueller Stride $= 0$** | Bias pro Kanal, Batchnormalisierung |

<!--en:3a95cfc1-->
| **Gather / Maske** | `X[rows, cols]`, `X[mask]` | `(3,)`, `(k,)` | **Neuer Puffer** (Kopie in $O(k)$) | Log-Wahrscheinlichkeiten der Zielklassen für die Kreuzentropie, Filterung |

<!--en:e4936183-->
| **Verketten / Stapeln** | `cat([X, Y], 0)`, `stack` | `(6, 4)`, `(2, 3, 4)` | **Neuer Puffer** (Kopie in $O(N)$) | Merkmale verbinden (`axis=1`), Minibatches zusammenstellen |

<!--en:32b5a32e-->
| **Direkte Aktualisierung** | `X += Y`, `Z[:] = X + Y` | Unverändert | **Puffer wird wiederverwendet** | Parameterschritte des Optimierers ($W \leftarrow W - \eta \nabla \mathcal{L}$) |
