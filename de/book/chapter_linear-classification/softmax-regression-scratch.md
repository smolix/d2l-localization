<!--en:38c9d41c-->
# Implementierung der Softmax-Regression

<!--en:263d3f8d-->
In :numref:`chap_regression` haben wir die lineare Regression auf zwei Arten implementiert, einmal von Grund auf und einmal in kompakter Form. Nun gehen wir für die Softmax-Regression auf Fashion-MNIST ebenso vor. Ausgehend von Tensoren implementieren wir die zeilenweise Softmax-Funktion, den vektorisierten Kreuzentropieverlust (engl. cross-entropy loss) und die Parameter von `SoftmaxRegressionScratch` ($\mathbf{W} \in \mathbb{R}^{784 \times 10}, \mathbf{b} \in \mathbb{R}^{10}$). Dazu definieren wir die wiederverwendbare Basisklasse `Classifier(d2l.Module)`. Wir trainieren das Modell bis zu einer Validierungsgenauigkeit (engl. validation accuracy) von etwa $83\%$. An seiner $10 \times 10$-Konfusionsmatrix (engl. confusion matrix) und seinen $28 \times 28$-Gewichtsschablonen sehen wir, was das lineare Modell gelernt hat. In :numref:`sec_softmax_concise` ersetzen wir die manuell verwalteten Tensoren durch eine eingebaute lineare Schicht. Statt die Wahrscheinlichkeiten nach unten zu begrenzen, berechnen wir den Kreuzentropieverlust mit der verschobenen *Log-Sum-Exp*-Formel direkt aus den unnormalisierten Logits. Damit vermeiden wir sowohl Gleitkommaüberläufe (engl. overflow) als auch Gleitkommaunterläufe (engl. underflow).

<!--en:bbc94b68-->
## Die Softmax-Funktion

<!--en:a55f96a7-->
### Zeilenweise Normalisierung

<!--en:22fbe1e5-->
Für einen Minibatch mit $n$ Beispielen bilden die Logits eine $n \times q$-Matrix $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ (:numref:`subsec_softmax_vectorization`). Die Softmax-Funktion normalisiert jede Zeile unabhängig von den anderen:

<!--en:02183d31-->
Im Code wenden wir die Exponentialfunktion auf `O` an und summieren über die Spalten (`axis=1`). Mit `keepdims=True` behalten die Zeilensummen die Form `(n, 1)`. Anschließend dividieren wir mittels Broadcasting:

<!--en:5578b840-->
Prüfen wir `softmax` anhand zweier Zeilen: Die erste enthält die Logits $(1.0, 2.2, 0.3)$ für Katze, Huhn und Hund aus :numref:`sec_softmax`, die zweite drei Nullen.

<!--en:2156f2af-->
### Überlauf

<!--en:eefbb4e0-->
Warum sollten Sie diese drei Zeilen für `softmax` nie in einem Produktivsystem einsetzen? Im 32-Bit-Gleitkommaformat (`float32`) liegt die größte darstellbare Zahl bei etwa $3.4 \times 10^{38}$. Daher läuft $\exp(o)$ über und ergibt `inf`, sobald $o > 88.7$ gilt. Übergeben wir `softmax` die Logits $(1000, 0, 0)$, so ergibt $\infty / \infty$ den Wert `NaN`, obwohl die tatsächliche Wahrscheinlichkeitsverteilung bis auf Maschinengenauigkeit $(1, 0, 0)$ beträgt.

<!--en:ce8434c9-->
Wie wir in :numref:`sec_softmax` bewiesen haben, ändert sich das Ergebnis der Softmax-Funktion nicht, wenn wir alle Logits einer Zeile um dieselbe Konstante verschieben. Ziehen wir vor dem Exponenzieren das Zeilenmaximum $o_{\max} = \max_k o_k$ ab, so ist jeder Exponent $o_j - o_{\max}$ nun $\leq 0$, der größte sogar genau $0$. Damit ist ein Überlauf ausgeschlossen. Die eingebaute Softmax-Funktion jedes Frameworks verwendet intern diese Verschiebung:

<!--en:6114b4ad-->
Beim folgenden Training unserer eigenen Implementierung auf Fashion-MNIST liegen die Pixelwerte in $[0, 1]$ und die Gewichte bleiben klein. Daher läuft `softmax` beim Training nicht über. Bei stark negativen Logits schützt die Verschiebung allein jedoch nicht vor einem *Unterlauf* (in float32 gilt $\exp(-200) = 0.0$, sodass $\log 0$ den Wert $-\infty$ ergibt). In :numref:`sec_softmax_concise` beseitigen wir beide Gefahren, indem wir die Softmax-Funktion und den Logarithmus im logarithmischen Raum zusammenfassen.

<!--en:0860e30f-->
## Verlust und Genauigkeit

<!--en:1fa8011d-->
### Zwei Verwendungen der Logits

<!--en:72aa17fc-->
Der Gradientenabstieg (engl. gradient descent) minimiert den glatten Kreuzentropieverlust. Wir beurteilen einen Klassifikator hingegen nach seiner *Genauigkeit*: Bei welchem Anteil der Beispiele stimmt die Klasse mit dem höchsten Score mit dem wahren Label $y^{(i)}$ überein?

<!--en:c09debcf-->
Beide Metriken ergeben sich aus demselben Logit-Vektor (:numref:`fig_mdl-clf-loss-accuracy`). Bei der Validierung müssen wir beide verfolgen: Ordnet das Modell bei einigen Fehlentscheidungen der falschen Klasse eine immer höhere Wahrscheinlichkeit zu, kann der Kreuzentropieverlust auf den Validierungsdaten steigen, selbst wenn die Validierungsgenauigkeit weiter zunimmt.

<!--en:0a1e3bf9-->
![Beim Training (oben) führt der Weg von den Logits über die Softmax-Funktion zum glatten Kreuzentropieverlust. Bei der Auswertung (unten) liefert der Vergleich von $\operatorname{argmax}_j o_j$ mit dem Label $y$ die diskrete Genauigkeit.](../img/mdl-clf-loss-accuracy.svg)

<!--en:396262d2-->
### Die Klasse `Classifier`

<!--en:be52e686-->
Wir bündeln die Validierung anhand beider Metriken in `Classifier(d2l.Module)`, der Basisklasse für alle Klassifikationsmodelle in diesem Buch. Zunächst definieren wir `validation_step` so, dass die Methode für jeden Validierungsminibatch sowohl `'loss'` als auch `'acc'` zurückgibt:

<!--en:43e91e35-->
Als Nächstes ergänzen wir die Methode `accuracy(self, Y_hat, Y, averaged=True)` in `Classifier`. Sie formt `Y_hat` so um, dass die Klassen-Scores jedes Beispiels in einer Zeile stehen. Mit `argmax` entlang `axis=1` bestimmt sie den Index der vorhergesagten Klasse und wandelt ihn in den Typ `Y.dtype` um. Aus dem Vergleich entsteht für jedes Beispiel ein float32-Wert. Die Methode gibt entweder den Mittelwert oder die gesamte Gleichheitsmaske `preds == Y` zurück:

<!--en:a6334316-->
Schließlich verwenden wir den stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD) mit Minibatches und der Lernrate `self.lr` als Standard für die Methode `configure_optimizers` von `d2l.Module`. So erbt jeder Klassifikator einen unmittelbar einsetzbaren Optimierer:

<!--en:12b218dc-->
## Das Modell und der Verlust

<!--en:c720f4d1-->
### Parameter und Vorwärtsdurchlauf

<!--en:d97dc202-->
Formen wir jedes $28 \times 28$-Bild aus Fashion-MNIST zu einem Vektor um, erhalten wir $d = 784$ Eingabemerkmale für $q = 10$ Klassen. Daher legt `SoftmaxRegressionScratch` die Gewichtsmatrix $\mathbf{W} \in \mathbb{R}^{784 \times 10}$ mit Anfangswerten aus $\mathcal{N}(0, 0.01^2)$ und den Bias-Vektor $\mathbf{b} \in \mathbb{R}^{10}$ mit dem Anfangswert $0$ an, insgesamt also $7{,}850$ Parameter. Das liefert eine Plausibilitätsprüfung: Anfangs liegen die Gewichte und damit die Logits nahe null. Das untrainierte Modell sagt deshalb nahezu gleich große Klassenwahrscheinlichkeiten vorher ($\hat{y}_j \approx 0.1$). Sein anfänglicher Kreuzentropieverlust liegt bei etwa $-\log(0.1) = \log 10 \approx 2.30$:

<!--en:b15da458-->
In `forward(self, X)` formen wir jedes Bild des Minibatchs zu einem Zeilenvektor der Länge `self.num_inputs` ($784$) um. Dazu verwenden wir `d2l.reshape(X, (-1, self.num_inputs))`. Auf die affinen Logits $\mathbf{X}\mathbf{W} + \mathbf{b}$ wenden wir `softmax` an:

<!--en:1b7fdcec-->
### Der Kreuzentropieverlust

<!--en:85097c8c-->
Aus :numref:`subsec_softmax-regression-loss-func` kennen wir die Beziehung $l(\mathbf{y}, \hat{\mathbf{y}}) = -\log \hat{y}_y$. Statt One-Hot-Matrizen anzulegen und innere Produkte zu berechnen, wählen wir die Wahrscheinlichkeit der wahren Klasse $\hat{y}^{(i)}_{y^{(i)}}$ für jede Zeile $i$ in einem einzigen vektorisierten Indizierungsschritt aus:

<!--en:a7aa456e-->
TensorFlow liest diese Werte mit `tf.gather(y_hat, y, batch_dims=1)` aus.

<!--en:04c0a6ba-->
So erhalten wir 0.1 für das erste Beispiel und 0.5 für das zweite. In `cross_entropy` fassen wir diese Schritte zusammen und mitteln die negativen Logarithmen über den Minibatch. Vor dem Logarithmieren begrenzen wir die Wahrscheinlichkeit zur Sicherheit nach unten auf $10^{-12}$. So ergibt der Logarithmus einer durch Unterlauf entstandenen Null nicht $-\infty$, und der Verlust pro Beispiel beträgt höchstens $27.6$. Sobald die Begrenzung greift, wird allerdings auch der Gradient null. Später in diesem Abschnitt (:numref:`sec_softmax_concise`) ersetzen wir diesen Behelf durch eine kombinierte Verlustfunktion, die direkt mit Logits rechnet.

<!--en:7f00ae20-->
Das ergibt den Mittelwert von $-\log 0.1 \approx 2.30$ und $-\log 0.5 \approx 0.69$, also etwa 1.50. Fügen wir diese Funktion dem Modell als Methode `loss` hinzu:

<!--en:e1f67b84-->
## Training und Auswertung

<!--en:cd477beb-->
### Der Trainingslauf

<!--en:d339a89c-->
Trainieren wir `SoftmaxRegressionScratch` auf Fashion-MNIST über $10$ Epochen mit der Minibatchgröße $256$ und der Lernrate $\eta = 0.1$:

<!--en:a99437e8-->
Damit wir ein trainiertes Modell jederzeit auf dem gesamten Validierungs- oder Trainingsdatensatz auswerten können, fügen wir die Hilfsmethode `evaluate(self, model, data, train=False)` zu `d2l.Trainer` hinzu. Sie weist `self.val_dataloader` den Datenlader für die Validierung oder das Training zu und ruft `self.validate()` auf:

<!--en:aa8e4c77-->
Über alle 10 000 Validierungsbilder gemittelt stimmen diese Metriken mit den zuletzt in `trainer.history['val_loss'][-1]` und `trainer.history['val_acc'][-1]` protokollierten Werten überein.

<!--en:8220a90c-->
Unser Klassifikator erkennt etwa $83\%$ der Validierungsbilder richtig. Mit `train=True` werten wir dieselben Gewichte auf allen $60{,}000$ Trainingsbildern aus:

<!--en:4e93e02e-->
Nach 10 Epochen pendelt sich die Validierungsgenauigkeit bei etwa $83\%$ ein (mit Schwankungen von $\pm 1\%$ je nach Zufallsstartwert). Die Trainingsgenauigkeit liegt bei etwa $84\%$. Die geringe Generalisierungslücke (engl. generalization gap) von $1\%$ spiegelt die begrenzte Kapazität eines linearen Klassifikators wider: Mit $7{,}850$ Parametern kann er sich $60{,}000$ Bilder nicht einprägen (:numref:`chap_classification_generalization`).

<!--en:0aa5c03e-->
### Falsch klassifizierte Bilder

<!--en:214bf662-->
Um zu sehen, *welche* Fehler das Modell macht, laden wir den ersten Validierungsminibatch mit $256$ Bildern und zählen die Fehlklassifikationen:

<!--en:9a1203fc-->
Zeigen wir die ersten acht falsch klassifizierten Bilder mit dem wahren Label über der Vorhersage des Modells (`true → pred`):

<!--en:5d960914-->
### Die Konfusionsmatrix

<!--en:1db75c21-->
Um die Fehler über alle $10{,}000$ Validierungsbilder hinweg aufzuschlüsseln, zählen wir die Einträge der $10 \times 10$-*Konfusionsmatrix* $\mathbf{N} \in \mathbb{R}^{q \times q}$. Jede Zeile steht für eine vorhergesagte Klasse $\hat{c}^{(m)} = \operatorname{argmax}_j o^{(m)}_j$, jede Spalte für eine wahre Klasse $y^{(m)}$:

<!--en:509f2b17-->
(Ordnen wir die wahren Klassen den Spalten und die vorhergesagten Klassen den Zeilen zu, lassen sich die spaltenweise normalisierten bedingten Wahrscheinlichkeiten im Produkt $\boldsymbol{\mu} = \mathbf{C}\mathbf{p}$ verwenden. Das nutzen wir bei der Labelverteilungsverschiebung in :numref:`sec_environment-and-distribution-shift`.) Wir stellen die vorhergesagten und die wahren Klassen eines Minibatchs durch One-Hot-Matrizen $\mathbf{P}, \mathbf{Y} \in \{0, 1\}^{n \times q}$ dar. Ihr Produkt $\mathbf{P}^\top\mathbf{Y}$ liefert alle $100$ Häufigkeiten mit einer einzigen Matrizenmultiplikation pro Minibatch:

<!--en:5771443b-->
Damit haben wir alle 10 000 Validierungsbilder erfasst. Da die absoluten Anzahlen von der Klassenhäufigkeit abhängen, teilen wir jede Spalte von $\mathbf{N}$ durch ihre Summe. So erhalten wir mit $C_{ij} = N_{ij} / \sum_{i'} N_{i'j}$ einen empirischen Schätzwert für die bedingte Wahrscheinlichkeit $P(\hat{c} = i \mid y = j)$. Jede Spalte zeigt, welchen Klassen das Modell die Beispiele der wahren Klasse $j$ zuordnet. Auf der Diagonalen stehen die Erkennungsraten der einzelnen Klassen. Stellen wir $\mathbf{C}$ als Heatmap mit `d2l.show_heatmaps` dar.

<!--en:21aabb93-->
Geben wir auch die fünf größten Einträge außerhalb der Diagonalen aus. Sie zeigen die häufigsten Verwechslungen:

<!--en:c5b79671-->
Die Konfusionsmatrix bestätigt, was die Betrachtung der Bilder in :numref:`sec_fashion_mnist` erwarten ließ: Hosen, Taschen und Schuhe erkennt das Modell in $\sim 90\%$ der Fälle oder häufiger. Die fünf häufigsten Verwechslungen betreffen dagegen ausschließlich die vier ähnlich geformten Oberteile (T-Shirt, Pullover, Mantel und Hemd). Hemden erkennt das Modell nur in etwa der Hälfte der Fälle oder seltener, weil ihre Silhouette zwischen denen der drei anderen Oberteile liegt.

<!--en:69a4e73d-->
### Die erlernten Schablonen

<!--en:8aad19ed-->
Jedes Logit $o_j = \mathbf{w}_j^\top\mathbf{x} + b_j$ ergibt sich aus dem inneren Produkt des zum Vektor umgeformten Bildes $\mathbf{x}$ mit Spalte $j$ von $\mathbf{W}$ und dem additiven Bias. Formen wir jede Spalte $\mathbf{w}_j$ mit ihren 784 Elementen wieder auf $28 \times 28$ um, sehen wir den gesamten Klassifikator als zehn bildliche *Schablonen* (engl. templates). Rot steht für positive, Blau für negative Gewichte:

<!--en:dc7bfec3-->
Jede Schablone zeigt schemenhaft die Silhouette ihres Kleidungsstücks. Der blaue Streifen zwischen den Hosenbeinen trägt mit negativen Gewichten dazu bei, dass Kleider einen niedrigeren Score für die Klasse Hose erhalten. Die vier Oberteile haben am Rumpf ähnliche Formen und unterscheiden sich hauptsächlich an Kragen und Ärmeln. Eine lineare Schicht kann pro Klasse nur ein einziges festes, globales Skalarprodukt bilden. Feine lokale Merkmale wie Kragen oder Knöpfe fallen daher gegenüber den ähnlichen Gesamtumrissen zu wenig ins Gewicht. Um über $\sim 83\%$ hinauszukommen, benötigen wir nichtlineare versteckte Schichten (:numref:`chap_perceptrons`) und räumliche Faltungen (:numref:`chap_cnn`).

<!--en:d52bb9d8-->
## Kompakte Implementierung

<!--en:7116af25-->
Sehen wir uns nun an, wie Deep-Learning-Frameworks die Softmax-Regression vereinfachen. Dabei beseitigen sie die Überlauf- und Unterlaufprobleme unserer naiven Funktion `softmax` und der Wahrscheinlichkeitsbegrenzung in `cross_entropy`.

<!--en:bd68577a-->
### Definition des Modells

<!--en:4f999f64-->
In `SoftmaxRegression(d2l.Classifier)` formen wir zunächst jedes Bild zu einem Vektor um. Eine einzige eingebaute vollständig verbundene Schicht mit $10$ Ausgaben berechnet dann die Logits $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$. Wie in unserer eigenen Implementierung ziehen wir die Anfangsgewichte aus $\mathcal{N}(0, 0.01^2)$ und setzen die Bias-Terme auf null. Entscheidend ist, dass **`forward` die unnormalisierten Logits $\mathbf{O}$ direkt zurückgibt, ohne die Softmax-Funktion anzuwenden**:

<!--en:6e3ed5e8-->
### Ein stabiler Kreuzentropieverlust

<!--en:cf70f535-->
Warum endet `SoftmaxRegression.forward` bei den Logits, statt Wahrscheinlichkeiten zurückzugeben? Erinnern wir uns an die zwei Gefahren der Gleitkommarechnung, wenn wir zuerst $\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})$ und dann $-\log \hat{y}_y$ in `float32` berechnen:

<!--en:3063a6b6-->
1. **Überlauf:** $\exp(o)$ läuft über und ergibt `inf`, sobald $o > 88.7$ gilt. Ziehen wir $o_{\max} = \max_k o_k$ von jedem Logit ab, ist ein Überlauf ausgeschlossen, denn jeder verschobene Exponent $o_k - o_{\max}$ ist $\leq 0$.

<!--en:313d4476-->
2. **Unterlauf:** Auch nach der Verschiebung um $o_{\max}$ bleibt ein Problem: Liegt das Logit einer Klasse mehr als $104$ unter $o_{\max}$ (wie bei der zweiten Klasse, die im Code das Label 1 trägt, in $\mathbf{o} = (0, -200, 0)$), so ergibt $\exp(o_y - o_{\max})$ in float32 durch Unterlauf exakt `0.0`. Dann liefert $\log(0.0)$ den Wert `-inf`. Die Untergrenze `1e-12` unserer eigenen Implementierung begrenzt den Verlust in diesem Fall auf $27.6$ und **setzt den Gradienten auf null**. Aus einer mit hoher Sicherheit getroffenen Fehlentscheidung kann das Modell dann nicht lernen.

<!--en:1dbe5d34-->
Wir vermeiden sowohl Überlauf als auch Unterlauf, indem wir *Wahrscheinlichkeiten gar nicht erst als Zwischenergebnis berechnen*. Mit dem *Log-Sum-Exp-Trick* fassen wir den Logarithmus und die verschobene Softmax-Funktion algebraisch zusammen. So wird aus der Division eine Subtraktion:

<!--en:01c1bb48-->
Jeder verschobene Exponent $o_k - o_{\max}$ ist $\leq 0$ und mindestens einer ist gleich $0$. Daher liegt die Summe im Logarithmus im Intervall $[1, q]$ und ihr Logarithmus im Intervall $[0, \log q]$. Weder ein Überlauf noch $\log 0$ können auftreten. Automatisches Differenzieren (engl. automatic differentiation) liefert den endlichen Gradienten $\mathrm{softmax}(\mathbf{o}) - \mathbf{y}$.

<!--en:30b339cb-->
Jedes Framework stellt diese kombinierte Log-Sum-Exp-Berechnung für die Kreuzentropie bereit. Wir verwenden sie einmalig als `d2l.Classifier.loss`. So erben sie alle Klassifikatoren im weiteren Verlauf des Buches:

<!--en:bb9ae1d6-->
Testen wir `SoftmaxRegression.loss` mit den extremen Logits `O = [[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]` und den wahren Labels `y = [0, 1]`. Die kombinierte Log-Sum-Exp-Implementierung liefert für die einzelnen Beispiele die exakten Verluste `[0., 1000.]` ohne Überlauf oder Unterlauf:

<!--en:10ed2efe-->
(Achten Sie auf einen häufigen Fehler: `Classifier.loss` erwartet unnormalisierte Logits und berechnet intern den Logarithmus der Softmax-Funktion. Steht versehentlich auch am Ende von `forward` eine Softmax-Funktion, so wenden wir sie *zweimal* an. Der Trainingsverlust kann dann eine positive Untergrenze nicht unterschreiten. Siehe dazu die Aufgabe :exref:`ex_softmax-double-softmax`.)

<!--en:70d292d9-->
### Training

<!--en:bb20c1d8-->
Das Training von `SoftmaxRegression` mit `d2l.Trainer` benötigt vier Codezeilen. Wir erreichen wieder eine Validierungsgenauigkeit von $\sim 83\%$, diesmal bei stabiler numerischer Berechnung:

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:76b3a150-->
Bei der eigenen Implementierung der Softmax-Regression haben wir die wiederverwendbare Basisklasse `Classifier` eingeführt. Beim Training optimieren wir die glatte Kreuzentropie, bei der Validierung bestimmen wir mit $\operatorname{argmax}$ die diskrete Genauigkeit. Auf Fashion-MNIST erreichen wir in zehn Epochen eine Validierungsgenauigkeit von etwa $83\%$. Summieren wir die One-Hot-Produkte $\mathbf{P}^\top\mathbf{Y}$, erhalten wir die $10 \times 10$-Konfusionsmatrix. Sie zeigt, dass das lineare Modell vor allem die vier Oberteile verwechselt, deren erlernte $28 \times 28$-Gewichtsschablonen sich stark überlappen. Berechnen wir `softmax` und `log` getrennt, kommt es bei großen positiven Logits zum Überlauf. Bei stark negativen Logits führt ein Unterlauf zu $\log 0$. Daher gibt unsere kompakte Implementierung in `forward` die unnormalisierten Logits zurück und berechnet die Kreuzentropie mit der verschobenen Log-Sum-Exp-Formel $o_{\max} + \log\sum_k \exp(o_k - o_{\max}) - o_y$. So bleiben die Werte für alle Größenordnungen der Logits endlich und die Gradienten erhalten.

<!--en:2294243a-->
## Übungen

<!--en:324d790f-->
1. [code] **Kreuzentropie aus One-Hot-Labels.** Die Funktion `cross_entropy` aus diesem Abschnitt wählt die Wahrscheinlichkeit der wahren Klasse durch Indizierung aus. Implementieren Sie eine Alternative, die der Definition $-\sum_j y_j \log \hat{y}_j$ unmittelbar folgt. Wandeln Sie dazu `y` in eine One-Hot-Matrix um und bilden Sie zeilenweise innere Produkte.
    1. Verwenden Sie diese Funktion als `loss` von `SoftmaxRegressionScratch` und prüfen Sie, ob das Training unverändert verläuft.
    1. Messen Sie die Laufzeit beider Varianten für zufällige Wahrscheinlichkeitsmatrizen mit $256$ Zeilen und $q \in \{10, 10^3, 10^5\}$ Spalten. Geben Sie an, wie der von jeder Variante belegte Speicher mit $q$ wächst.
    1. Beschreiben Sie einen Fall, in dem die Berechnung mit inneren Produkten anwendbar ist, die Indizierung jedoch nicht.

<!--en:ea83e057-->
1. **Große Vokabulare.** Eine Softmax-Regression soll aus einem Merkmalsvektor der Dimension $d = 1024$ das nächste Wort vorhersagen. Das Vokabular umfasst $V = 10^6$ Wörter, jeder Minibatch enthält $n = 256$ Beispiele. Listen Sie die Operationen in `forward` und `cross_entropy` auf, deren Aufwand mit $V$ wächst. Berechnen Sie die Anzahl der Multiplikations-Additions-Operationen des Matrixprodukts, den float32-Speicherbedarf von `W` und den float32-Speicherbedarf der $n \times V$-Wahrscheinlichkeitsmatrix. Vergleichen Sie jede Zahl mit dem entsprechenden Wert für das Fashion-MNIST-Modell aus diesem Abschnitt.

<!--en:58054d8a-->
1. [code] **Schablonen und Klassenmittelwerte.** Die Schablonen zeigen, was der Minibatch-SGD aus den Pixeln der Trainingsbilder gelernt hat.
    1. Trainieren Sie `SoftmaxRegressionScratch` erneut mit `std=0`, sodass jedes Gewicht bei 0 beginnt und alle Klassen mit demselben Logit starten. Erklären Sie, warum das Training dennoch funktioniert. Vergleichen Sie Genauigkeit und Schablonen mit den Ergebnissen im Text.
    1. Zeigen Sie für Anfangsgewichte von null, dass nach einem SGD-Schritt die Spalte $j$ von $\mathbf{W}$ proportional zu $\sum_i \big(y^{(i)}_j - \tfrac{1}{10}\big)\, \mathbf{x}^{(i)}$ ist. Die Summe läuft über den ersten Minibatch: Von der Summe der Bilder der Klasse $j$ wird ein Zehntel der Summe aller Bilder abgezogen.
    1. Berechnen Sie für jede Klasse das mittlere Bild über den Trainingsdatensatz. Vergleichen Sie es mit der erlernten Schablone der Klasse, sowohl visuell als auch anhand der Korrelation der Pixelwerte. Wo stimmen die beiden überein und wo weicht die Schablone vom Mittelwert ab? Erklären Sie, warum Schablonen, die nur den jeweiligen Klassenmittelwert wiedergeben, bei Oberteilen zu noch mehr Verwechslungen führen würden.

<!--en:ab0c8640-->
1. [code] **Doppelte Softmax-Anwendung.** Die Methode `loss` aus diesem Abschnitt erwartet Logits. Angenommen, `forward` endet bei einem Modell versehentlich mit der Softmax-Funktion. Die Verlustfunktion erhält dann statt der Logits einen Wahrscheinlichkeitsvektor $\mathbf{p}$ und berechnet $-\log \mathrm{softmax}(\mathbf{p})_y$.
    1. Bestimmen Sie als Funktion der Klassenanzahl den kleinsten Verlust über alle Wahrscheinlichkeitsvektoren $\mathbf{p}$ und Labels $y$ bei $q$ Klassen. Berechnen Sie ihn für $q = 10$.
    1. Verwenden Sie die in diesem Abschnitt selbst implementierte Funktion `softmax`, die nicht zum Paket `d2l` gehört. Leiten Sie von `SoftmaxRegression` eine Klasse ab, in der `forward` diese Funktion `softmax` auf die Logits anwendet. Trainieren Sie das Modell mit `lr=0.1`, `d2l.FashionMNIST(batch_size=256)` und `d2l.Trainer(max_epochs=10)`. Vergleichen Sie Trainingsverlust und Validierungsgenauigkeit mit den Werten von `SoftmaxRegression`. Setzen Sie den abschließenden Trainingsverlust zum Ergebnis des vorigen Schritts in Beziehung.
   :exlabel:`ex_softmax-double-softmax`

<!--en:13ec7d45-->
1. [code] **Stabile Softplus-Funktion.** Für zwei Klassen mit den Logits $(x, 0)$ lautet der erste Term des Verlusts $\log(1 + e^x)$. Das ist die *Softplus-Funktion* von $x$.
    1. Plotten Sie $\log(1 + e^x)$ und $\max(x, 0)$ auf dem Intervall $[-5, 5]$. Zeigen Sie, dass die erste Funktion die zweite um höchstens $\log 2$ übersteigt und diese Differenz bei $x = 0$ erreicht. Erklären Sie in einem Satz, warum Log-Sum-Exp als glattes Maximum bezeichnet wird.
    1. Zeigen Sie, dass $\log(1 + e^x) = \max(x, 0) + \log(1 + e^{-|x|})$ gilt.
    1. Berechnen Sie in einfacher Genauigkeit den naiven Ausdruck `log(1 + exp(x))`, den umgeformten Ausdruck und eine Variante des umgeformten Ausdrucks mit `log1p` anstelle von `log(1 + ...)`. Verwenden Sie dafür $x \in \{-100, -20, -5, 0, 5, 20, 100\}$. Vergleichen Sie jeden Wert mit einem Referenzwert in doppelter Genauigkeit. Geben Sie für jede Variante an, wo ein Überlauf auftritt und wo sie jegliche relative Genauigkeit verliert.
    1. Berechnen Sie die Ableitung des naiven Ausdrucks an der Stelle $x = -20$ durch automatisches Differenzieren. Erklären Sie, warum die Ableitung genau ist, obwohl der Funktionswert ungenau ist. Was bedeutet dies für das Training und für die Angabe des Verlusts bei Beispielen, die das Modell mit hoher Sicherheit klassifiziert?

<!--en:8da8096a-->
1. [code] **Der Gradient bleibt erhalten.** Für die Logits $(0, -200, 0)$ mit dem wahren Label 1 ergibt die verschobene Softmax-Funktion für Klasse 1 in float32 durch Unterlauf die Wahrscheinlichkeit $0$. In unserer eigenen Verlustfunktion `cross_entropy(softmax(O - O_max), y)` greift daher die Untergrenze `1e-12` für die Wahrscheinlichkeit. Der Gradient bezüglich der Logits ist null. Übergeben Sie dieselben Logits und dasselbe Label der kombinierten Methode `loss` von `SoftmaxRegression`. Berechnen Sie den Gradienten bezüglich der Logits durch automatisches Differenzieren. Vergleichen Sie ihn mit $\hat{\mathbf{y}} - \mathbf{y}$ aus :numref:`sec_softmax`. Erklären Sie, warum die kombinierte Verlustfunktion das Logit der Klasse 1 verändert, während es bei der Verlustfunktion mit Wahrscheinlichkeitsbegrenzung unverändert bleibt.

<!--en:9877c042-->
1. **Zahlenformate.** Formate mit halber Genauigkeit speichern eine Zahl in 16 Bit. FP16 hat 5 Exponentenbits und 10 Mantissenbits, BFLOAT16 hat 8 und 7, FP32 hat 8 und 23. Bestimmen Sie für jedes Format das größte Logit $o$, bei dem $\exp(o)$ ohne Überlauf darstellbar ist. Erklären Sie dann, warum sich ein Netz, dessen Logits den Wert 20 erreichen, in BFLOAT16 trainieren lässt, in FP16 hingegen nur dann, wenn die Softmax-Funktion in FP32 berechnet wird. Betrachten Sie schließlich die verschobene Summe aus :eqref:`eq_softmax-concise-lse-loss` für $q = 10^5$ Klassen. Jeder Summand ist höchstens 1 und die Summation erfolgt in FP16. Was geht schief, sobald die Teilsumme 2048 übersteigt, und welche Rolle spielt dabei die 10-Bit-Mantisse?

<!--en:ab8a3a37-->
[Diskussion](https://d2l.discourse.group/t/52)

<!--en:f5f4e50e-->
[Diskussion](https://d2l.discourse.group/t/53)

<!--en:98b2688b-->
[Diskussion](https://d2l.discourse.group/t/260)

<!--en:c431dd80-->
[Diskussion](https://d2l.discourse.group/t/17983)

<!--en:cc4cdb12-->
Implementierung der Softmax-Regression: von Grund auf bis zu Log-Sum-Exp<br>**Zeilenweise Softmax-Funktion · Überlauf und Unterlauf in float32 · Basisklasse `Classifier` · Konfusionsmatrix · Schablonen · kombinierter Log-Sum-Exp-Verlust**

<!--en:9dcd72b1-->
::: {.slide title="Warum naive Softmax-Funktion und Kreuzentropie in float32 scheitern"}

<!--en:b1a22a93-->
[Numerische und implementierungstechnische Motivation]{.kicker}

<!--en:39b1f531-->
Die Softmax-Regression auf Fashion-MNIST ($d = 784$, $q = 10$, $7{,}850$ Parameter) stellt zwei unmittelbare Anforderungen an die Implementierung:

<!--en:70c80e09-->
- **Überlauf und Unterlauf in float32:** In `float32` entsteht durch Überlauf der Wert `inf` oberhalb von $\exp(88.7) \approx 3.4 \times 10^{38}$ ($\infty / \infty = \texttt{NaN}$), durch Unterlauf der Wert $0.0$ unterhalb von $\exp(-104)$ ($-\log 0 = +\infty$).

<!--en:359bbba0-->
- **Zwei Metriken für ein Modell:** Jeder Klassifikator optimiert bei der Backpropagation die glatte Kreuzentropie und protokolliert die diskrete $\operatorname{argmax}$-Genauigkeit über unterschiedlich große Validierungsminibatches.

<!--en:3233af30-->
::: {.slide title="Zeilenweise Softmax-Funktion mit Broadcasting"}

<!--en:c4146cde-->
[Eigene Implementierung der Softmax-Funktion]{.kicker}

<!--en:482d3673-->
Für einen Minibatch mit Logits $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b} \in \mathbb{R}^{n \times q}$ normalisiert die Softmax-Funktion jede Zeile unabhängig von den anderen:

<!--en:84cf35a8-->
- **Reduzierte Achse beibehalten (`keepdims=True`):** Die Form `(n, 1)` bleibt erhalten. Bei der Division verteilt Broadcasting jede Zeilensumme über alle $q$ Spalten.

<!--en:74590981-->
::: {.slide title="Zeilenweise Softmax-Funktion an Testwerten prüfen"}

<!--en:a7f9a231-->
[Plausibilitätsprüfung]{.kicker}

<!--en:6b6684c4-->
Ein Test von `softmax` mit den Logits $(1.0, 2.2, 0.3)$ für Katze, Huhn und Hund aus §{sec_softmax} und einer Zeile mit drei Nullen bestätigt die zeilenweise Normalisierung:

<!--en:d5a1f466-->
- **Erwartete Ausgaben:** Die erste Zeile liefert wieder $(0.21, 0.69, 0.10)$, die zweite die Gleichverteilung $(1/3, 1/3, 1/3)$. Jede Zeilensumme beträgt $1.0$.

<!--en:a3293581-->
::: {.slide title="Subtraktion des Zeilenmaximums verhindert Überlauf"}

<!--en:eda64bb1-->
[Schutz vor Überlauf]{.kicker}

<!--en:7429e605-->
Nach Subtraktion des Zeilenmaximums $o_{\max} = \max_k o_k$ gilt für jedes verschobene Logit $o_j - o_{\max} \le 0$. Das größte ist $0$:

<!--en:f013f89e-->
- **Warum die Verschiebung für den Verlust nicht genügt:** Die Subtraktion von $o_{\max}$ verhindert den Überlauf von `exp`. Gilt jedoch $o_y - o_{\max} < -104$, ergibt $\exp(o_y - o_{\max})$ durch Unterlauf weiterhin $0.0$ und damit $-\log \hat{y}_y = +\infty$.

<!--en:dbc6b1c7-->
::: {.slide title="Ein Logit-Vektor, zwei Rechenwege für Verlust und Genauigkeit"}

<!--en:23393bdc-->
[Metriken für Training und Auswertung]{.kicker}

<!--en:5e2cafd2-->
Training und Auswertung verwenden dieselbe Logit-Matrix $\mathbf{O} \in \mathbb{R}^{n \times q}$, gehen dann aber verschiedene Wege:

<!--en:5aec2d08-->
- **Training (oben):** Die Softmax-Funktion liefert aus den Logits die Wahrscheinlichkeiten für den glatten, differenzierbaren Kreuzentropieverlust $l(\mathbf{y}, \hat{\mathbf{y}})$.

<!--en:28dc4eb9-->
- **Auswertung (unten):** Aus $\operatorname{argmax}_j o_j$ ergibt sich direkt die diskrete **Genauigkeit**:
  $$\mathrm{accuracy} = \frac{1}{n}\sum_{i=1}^n \mathbf{1}\!\Big(\operatorname*{argmax}_j o^{(i)}_j = y^{(i)}\Big)$$

<!--en:e04b0cb6-->
- **Warum beide verfolgen?** Ist sich das Modell bei einigen schwierigen Validierungsbeispielen zu sicher, kann der Validierungsverlust steigen, selbst wenn die Validierungsgenauigkeit weiter zunimmt.

<!--en:7787bcde-->
::: {.slide title="Basisklasse für Klassifikatoren und vektorisierte Berechnung der Genauigkeit"}

<!--en:87f9e309-->
[Wiederverwendbare objektorientierte Abstraktion]{.kicker}

<!--en:90d6a99f-->
Die Validierung mit beiden Metriken gehört zu `Classifier(d2l.Module)`, der Basisklasse aller Klassifikationsnetze in diesem Buch:

<!--en:e291e856-->
- **Typsicherer Vergleich:** `accuracy` fasst die Batchachsen zur Form `(-1, q)` zusammen, wandelt `argmax(axis=1)` in `Y.dtype` um und mittelt den Indikatortensor.

<!--en:04438baf-->
::: {.slide title="Minibatch-SGD als Standardoptimierer für Module"}

<!--en:455c6b5d-->
[Konfiguration des Optimierers]{.kicker}

<!--en:f0d3ca40-->
Fast alle Basismodelle verwenden stochastischen Gradientenabstieg mit Minibatches und der Lernrate `self.lr`. Deshalb gehört die Standardmethode `configure_optimizers` direkt zu `d2l.Module`:

<!--en:ac38f672-->
- **Automatische Parametererkennung:** Der Optimierer des Frameworks erfasst automatisch alle registrierten lernbaren Parameter des Modells.

<!--en:45abb3b5-->
::: {.slide title="Parameter registrieren und den Vorwärtsdurchlauf selbst implementieren"}

<!--en:03e89cbc-->
[7850 lernbare Parameter]{.kicker}

<!--en:a4372d3e-->
Ein $28 \times 28$-Bild aus Fashion-MNIST ergibt als Vektor $d = 784$ Merkmale für $q = 10$ Kleidungsklassen ($\mathbf{W} \in \mathbb{R}^{784 \times 10}$ aus $\mathcal{N}(0, 0.01^2)$, $\mathbf{b} = \mathbf{0} \in \mathbb{R}^{10}$):

<!--en:e7d3b354-->
::: {.slide title="Formen der Parameter und Prüfung des Anfangsverlusts"}

<!--en:03e89cbc-2-->
[7850 lernbare Parameter]{.kicker}

<!--en:a31b3273-->
- **Parameteranzahl:** $784 \times 10$ Gewichte und $10$ Bias-Terme ergeben $7{,}850$ lernbare Parameter.

<!--en:e7d97693-->
- **Plausibilitätsprüfung für den Anfangsverlust:** Bei $\mathbf{W} \approx \mathbf{0}$ gilt $\hat{y}_j \approx 0.1$ und die anfängliche Kreuzentropie liegt bei etwa $-\log(0.1) = \log 10 \approx 2.30\text{ nats}$.

<!--en:76e49703-->
::: {.slide title="Vektorisierte Kreuzentropie durch ganzzahlige Indizierung"}

<!--en:2d6dbcc5-->
[Indizierung in O(n) gegenüber One-Hot-Produkten in O(nq)]{.kicker}

<!--en:83076378-->
Eine One-Hot-Matrix der Form $n \times q$ erübrigt sich: Ganzzahlige Indizierung liefert in $O(n)$ Schritten den Wert $\hat{Y}_{i, y_i}$ jeder Zeile:

<!--en:e79e8936-->
::: {.slide title="Indizierung und One-Hot-Produkte im Vergleich, Grenzen der Wahrscheinlichkeitsbegrenzung"}

<!--en:2d6dbcc5-2-->
[Indizierung in O(n) gegenüber One-Hot-Produkten in O(nq)]{.kicker}

<!--en:6b840ec3-->
- **Linearer Aufwand:** Die Indizierung greift auf $n$ Einträge zu. Ein One-Hot-Produkt multipliziert alle $nq$ Einträge, von denen fast alle null sind.

<!--en:bc400a05-->
- **Warum die Untergrenze `1e-12` nur ein Behelf ist:** Wo die Begrenzung $\max(\hat{y}_y, 10^{-12})$ greift, ist die Ableitung **null**. Gerade bei den größten Fehlern verschwindet der Gradient.

<!--en:9a1a67b2-->
::: {.slide title="Training der eigenen Softmax-Regression"}

<!--en:f3f57318-->
[10 Epochen Minibatch-SGD]{.kicker}

<!--en:6f53a8f1-->
Das Training von `SoftmaxRegressionScratch` läuft über $10$ Epochen auf Fashion-MNIST mit der Minibatchgröße $n = 256$ und der Lernrate $\eta = 0.1$:

<!--en:c3d95f43-->
- **Schnelle Konvergenz:** Die Validierungsgenauigkeit übersteigt in der ersten Epoche $75\%$ und liegt stabil bei etwa $83\%$, sobald Epoche $10$ erreicht ist. Der Kreuzentropieverlust auf den Validierungsdaten fällt von $\approx 2.30$ auf $\approx 0.50$, der Trainingsverlust auf $\approx 0.46$.

<!--en:199f3c27-->
- **Auswertung auf vollständigen Teildatensätzen:** Über alle $10{,}000$ Validierungsbilder und $60{,}000$ Trainingsbilder ergibt sich eine geringe Generalisierungslücke von $\sim 1\%$:

<!--en:d04d36ab-->
::: {.slide title="Einzelne falsch klassifizierte Kleidungsstücke"}

<!--en:637372a8-->
[Qualitative Fehleranalyse]{.kicker}

<!--en:76a05556-->
Im ersten Validierungsminibatch mit $256$ Bildern klassifiziert das Modell etwa $35$ bis $40$ Kleidungsstücke falsch. Bei den ersten acht Fehlern (`true → predicted`) fällt ein Muster auf:

<!--en:10c68a4f-->
- **Ähnliche Silhouetten:** Bei fast jedem Fehler verwechselt das Modell zwei Oberteile (Hemd, Pullover, Mantel, T-Shirt) oder zwei Schuhe, deren Pixel ähnliche Umrisse bilden.

<!--en:060a03f8-->
::: {.slide title="Die 10 × 10-Konfusionsmatrix durch Aufsummieren bilden"}

<!--en:01e1b510-->
[Quantitative Fehleranalyse]{.kicker}

<!--en:856e7760-->
Mit den One-Hot-Matrizen $\mathbf{P}, \mathbf{Y} \in \{0, 1\}^{n \times 10}$ lassen sich alle Vorhersagen gemäß $\mathbf{N} \mathrel{+}= \mathbf{P}^\top\mathbf{Y}$ zählen:

<!--en:97ef660b-->
- **Spalten = wahre Klasse:** Die spaltenweise Normalisierung ergibt $C_{ij} = P(\hat{c} = i \mid y = j)$.

<!--en:8aaea7bf-->
::: {.slide title="Die spaltenweise normalisierte Konfusionsmatrix"}

<!--en:f2094fcd-->
[Wo das lineare Modell Schwierigkeiten hat]{.kicker}

<!--en:d93253ab-->
Die Darstellung von $C_{ij} = N_{ij} / \sum_{i'} N_{i'j}$ als $10 \times 10$-Heatmap zeigt, welchen Klassen das Modell die Bilder jeder wahren Kleidungskategorie zuordnet:

<!--en:6136ada9-->
- **Klassen mit hoher Genauigkeit ($\ge 90\%$):** Hosen (`1`), Taschen (`8`), Sandalen (`5`), Turnschuhe (`7`) und Stiefeletten (`9`) haben markante Silhouetten und helle Diagonaleinträge.

<!--en:23041e01-->
- **Schwierige Oberteile:** Die fünf größten Einträge außerhalb der Diagonalen betreffen ausschließlich Verwechslungen zwischen **T-Shirt (`0`), Pullover (`2`), Mantel (`4`) und Hemd (`6`)**.

<!--en:49eb8999-->
- **Hemden erkennt das Modell nur in etwa der Hälfte der Fälle oder seltener:** Ihr kantiger Rumpf ähnelt dem aller drei anderen Oberteile.

<!--en:a60ea772-->
::: {.slide title="Die erlernten Spalten von W als Schablonen"}

<!--en:221e8fef-->
[Was ein linearer Klassifikator lernt]{.kicker}

<!--en:4595b023-->
Jedes Logit $o_j = \mathbf{w}_j^\top\mathbf{x} + b_j$ beruht auf dem inneren Produkt von Bild $\mathbf{x}$ und Spalte $j$ von $\mathbf{W}$. Jede Spalte $\mathbf{w}_j$ enthält 784 Elemente. In der Form $28 \times 28$ zeigen sich die zehn räumlichen **Schablonen** (Rot $> 0$, Blau $< 0$):

<!--en:57df574b-->
**Globaler Schablonenvergleich:** Der blaue Streifen zwischen den Hosenbeinen senkt mit seinen negativen Gewichten den Score für Kleider in der Klasse Hose. Die roten Rumpfflächen der vier Oberteile sind fast identisch. Eine lineare Schicht bildet pro Klasse nur *ein globales Skalarprodukt*. Details an Kragen oder Knöpfen fallen gegenüber den ähnlichen Gesamtumrissen zu wenig ins Gewicht.

<!--en:d92976b8-->
::: {.slide title="Kompaktes Modell mit unnormalisierten Logits als Ausgabe"}

<!--en:3dfbbc6e-->
[Kompakte Implementierung mit dem Framework]{.kicker}

<!--en:853462e7-->
In `SoftmaxRegression(d2l.Classifier)` ersetzen eine Schicht zum Umformen der Bilder in Vektoren und eine einzige eingebaute lineare Schicht die manuelle Parameterverwaltung:

<!--en:0e575c7a-->
::: {.slide title="Warum forward bei den Logits endet"}

<!--en:3dfbbc6e-2-->
[Kompakte Implementierung mit dem Framework]{.kicker}

<!--en:0d245fc2-->
- **Entscheidend ist, dass `forward` die unnormalisierten Logits $\mathbf{O}$ zurückgibt:** Sowohl `Classifier.accuracy` ($\operatorname{argmax}_j O_{ij}$) als auch die kombinierte Kreuzentropiefunktion arbeiten direkt mit unnormalisierten Logits. Eine Softmax-Funktion in `forward` würde die Softmax-Funktion **zweimal** anwenden.

<!--en:6991c6d2-->
::: {.slide title="Der Log-Sum-Exp-Trick verbindet Softmax-Funktion und Logarithmus"}

<!--en:5060ac47-->
[Überlauf und Unterlauf vermeiden]{.kicker}

<!--en:f90b62be-->
$\mathrm{softmax}(\mathbf{o})_y$ und $-\log(\cdot)$ im logarithmischen Raum algebraisch zusammenfassen und die Logits um $o_{\max} = \max_k o_k$ verschieben:

<!--en:d622d255-->
- **Kein Überlauf:** Für jeden Term gilt $o_k - o_{\max} \le 0$ und damit $\exp(o_k - o_{\max}) \in (0, 1]$.

<!--en:88730011-->
- **Kein Unterlauf mit $\log(0)$:** Für $k = \operatorname{argmax} o_k$ gilt $\exp(0) = 1$. Daher liegt die Summe in $[1, q]$ und ihr Logarithmus in $[0, \log q]$. Der exakte Gradient lautet $\mathrm{softmax}(\mathbf{o}) - \mathbf{y}$.

<!--en:5df1dacc-->
::: {.slide title="Die kombinierte Kreuzentropiefunktion in Classifier"}

<!--en:ea226035-->
[Verlustfunktionen des Frameworks für Logits]{.kicker}

<!--en:720ee151-->
Die kombinierte Log-Sum-Exp-Kreuzentropie des Frameworks als `d2l.Classifier.loss` einsetzen und mit den extremen Logits $(1000, 0, 0)$ testen:

<!--en:d08f62de-->
::: {.slide title="Kompakter Trainingslauf auf Fashion-MNIST"}

<!--en:fa641af1-->
[Prüfung des gesamten Ablaufs]{.kicker}

<!--en:6e151cf8-->
Das Training von `SoftmaxRegression` mit `d2l.Trainer` benötigt vier Codezeilen und erreicht wieder eine Validierungsgenauigkeit von $\sim 83\%$. Die Berechnung ist für Logits aller Größenordnungen numerisch stabil:

<!--en:dd51eff9-->
- **Mathematisch gleicher Trainingsverlauf:** Da die Anfangsgewichte aus $\mathcal{N}(0, 0.01^2)$ stammen und die Bias-Terme bei $0$ beginnen, folgt die Optimierung des kompakten Modells derselben Dynamik wie bei `SoftmaxRegressionScratch`.

<!--en:86e02b8b-->
- **Grundlage für die Kapitel §{num:chap_perceptrons}–§{num:chap_rnn}:** Die einzelne lineare Schicht in `self.net` lässt sich durch mehrschichtige Perzeptrons oder konvolutionäre Netze ersetzen, ohne `Classifier`, `loss` oder `Trainer` zu ändern.

<!--en:736e8bb6-->
::: {.slide title="Verständnisfragen zu Unterlauf, doppelter Softmax-Anwendung und Skalierung"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:b57533e0-->
Numerische Stabilität und typische Implementierungsfehler anhand dreier Fragen prüfen:

<!--en:6c7c51ec-->
1. **Stark negative Logits:** Was liefert unsere eigene Funktion `softmax` ohne Verschiebung für $\mathbf{o} = (-1000, -1000, -1000)$ in `float32` und was liefert die verschobene Formel?

<!--en:fd9b1fd9-->
2. **Doppelte Softmax-Anwendung:** Angenommen, jemand fügt versehentlich `softmax` am Ende von `forward` ein und übergibt die Ausgabe der Kreuzentropieverlustfunktion des Frameworks. Was ist bei $q = 10$ Klassen der kleinstmögliche Verlust, selbst wenn das Modell einen deterministischen One-Hot-Vektor $\mathbf{p} = (1, 0, \ldots, 0)$ vorhersagt?

<!--en:bb7ab0bf-->
3. **Skalierung mit dem Vokabular:** Ein Sprachmodell hat eine versteckte Dimension von $d = 1024$, eine Batchgröße von $n = 256$ und ein Vokabular von $q = 10^6$ Token. Wie viel `float32`-Speicher belegen die Gewichtsmatrix der Ausgabeschicht $\mathbf{W} \in \mathbb{R}^{d \times q}$ und die Logit-Matrix $\mathbf{O} \in \mathbb{R}^{n \times q}$?

<!--en:4520b9d0-->
**Lösungen:** **(1)** Ohne Verschiebung gilt $\exp(-1000) = 0.0$ an allen drei Positionen. Das ergibt $0.0 / 0.0 = \texttt{NaN}$. Nach Verschiebung um $o_{\max} = -1000$ gilt $\mathrm{softmax}(0, 0, 0) = (1/3, 1/3, 1/3)$. **(2)** Das Einsetzen von $\mathbf{p} = (1, 0, \ldots, 0)$ in $-\log\mathrm{softmax}(\mathbf{p})_1$ ergibt $-\log(e^1 / (e^1 + 9e^0)) = \log(e + 9) - 1 \approx 1.46\text{ nats}$. Der Trainingsverlust kann $1.46$ nicht unterschreiten. **(3)** $\mathbf{W}$ belegt $1024 \times 10^6 \times 4\text{ B} \approx 4.10\text{ GB}$, ein einzelner Batch von Logits $\mathbf{O}$ belegt $256 \times 10^6 \times 4\text{ B} \approx 1.02\text{ GB}$.

<!--en:22841c4a-->
::: {.slide title="Zusammenfassung der eigenen und der kompakten Softmax-Regression"}

<!--en:27de615d-->
[Zusammenfassung des Abschnitts]{.kicker}

<!--en:2b5426fe-->
| Komponente | Eigene Implementierung | Kompakte Implementierung mit dem Framework |

<!--en:9a3d2473-->
| **Parameter** | Manuell angelegte Parameter $\mathbf{W} \in \mathbb{R}^{784\times 10},\; \mathbf{b} \in \mathbb{R}^{10}$ (insgesamt $7{,}850$ Parameter) | Eingebaute lineare Schicht mit $10$ Ausgaben |

<!--en:1539197a-->
| **Ausgabe des Vorwärtsdurchlaufs** | Wahrscheinlichkeitsmatrix $\hat{\mathbf{Y}} = \mathrm{softmax}(\mathbf{X}\mathbf{W}+\mathbf{b})$ | Unnormalisierte Logit-Matrix $\mathbf{O} = \mathbf{X}\mathbf{W}+\mathbf{b}$ (ohne Softmax-Funktion) |

<!--en:bcf416da-->
| **Kreuzentropieverlust** | $-\log \max(\hat{Y}_{i, y_i}, 10^{-12})$ (Indizierung mit Untergrenze) | Kombinierte Log-Sum-Exp-Formel: $o_{\max} + \log\sum_k \exp(o_k - o_{\max}) - o_y$ |

<!--en:6ab12091-->
| **Numerische Stabilität** | Überlauf für $o > 88.7$. Bei Unterlauf setzt die Begrenzung den Gradienten auf null. | Schutz vor Überlauf und Unterlauf, exakter Gradient $\mathrm{softmax}(\mathbf{o}) - \mathbf{y}$ |

<!--en:ad434712-->
| **Basisklasse `Classifier`** | Protokolliert `'loss'` und `'acc'` bei der Validierung, verwendet SGD als Standardoptimierer | Alle Klassifikatoren im Buch erben dieses Verhalten unverändert. |

<!--en:cfa7ed9a-->
| **Diagnose ($83\%$ Validierungsgenauigkeit)** | $10\times 10$-Konfusionsmatrix $\mathbf{N} = \sum \mathbf{P}^\top\mathbf{Y}$ und $28\times 28$-Schablonen | Verwechslungen vor allem zwischen den $4$ Oberteilen mit ähnlichen Umrissen |
