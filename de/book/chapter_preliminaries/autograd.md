<!--en:321a9f76-->
# Automatisches Differenzieren

<!--en:c6fa9d57-->
In :numref:`sec_calculus` konnten wir den Gradienten $\nabla_{\mathbf{w}} L = \mathbf{X}^\top (2\mathbf{r}) = (2, 4)$ unseres Geradenmodells mit zwei Parametern mühelos von Hand herleiten. Bei Dutzenden von Schichten und Milliarden von Parametern wäre das aussichtslos. Moderne Deep-Learning-Frameworks erledigen dies durch *automatisches Differenzieren* (engl. automatic differentiation, kurz *Autograd*). Während ein Programm einen skalaren Verlust auswertet, zeichnet das Framework die elementaren Operationen in einem *Berechnungsgraphen* (engl. computational graph) auf. Anschließend durchläuft es den Graphen rückwärts (*Backpropagation*) und wendet dabei die Kettenregel an. Ob ein Modell zwei Parameter hat oder eine Milliarde, ein einziger Rückwärtsdurchlauf liefert den exakten Gradienten nach *allen* Parametern. Der Rechenaufwand dafür beträgt nur ein kleines konstantes Vielfaches des Aufwands für den Vorwärtsdurchlauf.

<!--en:cf8ffdd8-->
## Berechnung von Gradienten

<!--en:9a7f0043-->
### Der Vorwärtsdurchlauf

<!--en:e1a73137-->
Zuerst setzen wir das Geradenmodell $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ am Punkt $\mathbf{w} = (1, 0)$ aus :numref:`sec_calculus` mit Tensoren des Frameworks um. Dabei schreiben wir nur den Vorwärtsdurchlauf:

<!--en:24a843c5-->
Operationen aufzuzeichnen und Zwischenaktivierungen zu speichern kostet Speicherplatz. Das Framework verfolgt daher nur Operationen, die von ausdrücklich für die Gradientenberechnung markierten Parametern abhängen.

<!--en:85210098-->
Mit `w.attach_grad()` markieren wir `w` für die Gradientenberechnung und reservieren seinen mit Nullen initialisierten Gradientenpuffer `w.grad`:

<!--en:4291d98a-->
Der Aufruf `w.requires_grad_(True)` markiert `w` direkt am vorhandenen Tensor. PyTorch verfolgt daraufhin Operationen mit `w` und speichert den resultierenden Gradienten in `w.grad`:

<!--en:c7aea1f1-->
Indem wir `w` in `tf.Variable` kapseln, kennzeichnen wir den Tensor als trainierbaren Parameter. TensorFlow zeichnet Operationen mit ihm automatisch auf, sobald eine Gradientenaufzeichnung läuft:

<!--en:117f3d1f-->
Wir werten nun `L = loss(w)` in einem Block mit `with autograd.record():` aus. So zeichnet MXNet die Operationen auf, die `w` mit `L` verbinden:

<!--en:5a5f91b6-->
Wir werten nun `L = loss(w)` aus. PyTorch zeichnet die Folge der Operationen von `w` bis `L` automatisch auf:

<!--en:9c74c5e4-->
Wir werten nun `L = loss(w)` in einem Block mit `with tf.GradientTape() as t:` aus. So zeichnet TensorFlow die Operationen von `w` bis `L` auf dem Band `t` auf:

<!--en:26ef09db-->
JAX arbeitet rein funktional. Arrays haben keine veränderlichen Gradientenpuffer. Werten wir `loss(w)` für sich aus, führt JAX gewöhnliche Arrayoperationen aus und gibt `1.0` zurück. Erst wenn wir die Funktion unten differenzieren, zeichnet `jax.grad(loss)` die Ausführung von `loss` nach:

<!--en:5bee0d8c-->
Beachten Sie `grad_fn=<SumBackward0>` an `L`. Dieser Eintrag hält die letzte Operation im Berechnungsgraphen fest.

<!--en:1dbba45a-->
Die aufgezeichnete Folge bildet den *Berechnungsgraphen* in :numref:`fig_autograd_graph`. Er zerlegt die Verlustberechnung in vier elementare Operationen: $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$, $\mathbf{r} = \hat{\mathbf{y}} - \mathbf{y}$, $\mathbf{s} = \mathbf{r} \odot \mathbf{r}$ und $L = \sum_i s_i$.

<!--en:77e38cac-->
![Berechnungsgraph von $L = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ bei $\mathbf{w} = (1, 0)$. Je ein Knoten steht für $\mathbf{w}$ und für jeden daraus berechneten Wert. Die Konstanten $\mathbf{X}$ und $\mathbf{y}$ sind nicht eingezeichnet. Der Ausdruck $\mathbf{r} \odot \mathbf{r}$ bezeichnet das elementweise Quadrat. Jeder Knoten zeigt seinen Wert (schwarz) und den Gradienten von $L$ nach diesem Wert (rot). Der Vorwärtsdurchlauf (blau) verläuft von links nach rechts. Der Rückwärtsdurchlauf (rot) beginnt mit $1$ bei $L$ und wendet von rechts nach links die lokale Regel jeder Operation an.](../img/autograd-comp-graph.svg)

<!--en:e2e02f72-->
### Der Rückwärtsdurchlauf

<!--en:cab06b1c-->
Lösen wir den Rückwärtsdurchlauf aus, werden die Empfindlichkeiten ausgehend von $\partial L/\partial L = 1$ am Ausgang bis zu $\mathbf{w}$ zurückgereicht. Dazu verwenden wir an jedem Knoten die lokale Ableitungsregel:

<!--en:687852ac-->
So erhalten wir wieder $(2, 4)$:

<!--en:1da3ec35-->
`L.backward()` führt den Rückwärtsdurchlauf aus und speichert den Gradienten in `w.grad`.

<!--en:d81950c1-->
`t.gradient(L, w)` berechnet den Gradienten von `L` nach `w` anhand des Bands `t`.

<!--en:2e8dce24-->
`jax.grad(loss)(w)` wertet die Gradientenfunktion bei `w` aus.

<!--en:dfdf2445-->
Die lokalen Regeln für den Rückwärtsdurchlauf benötigen Zwischenergebnisse des Vorwärtsdurchlaufs (etwa $\mathbf{r}$ und $\mathbf{X}$). Deshalb verbraucht das Training erheblich mehr Speicher als die Inferenz (:numref:`fig_autograd_memory`).

<!--en:67689383-->
![Speicherbelegung beim Vorwärtsdurchlauf durch vier Schichten mit Ausgaben $\mathbf{h}_1, \ldots, \mathbf{h}_4$. (a) Bei aktivierter Gradientenverfolgung bleibt jede Ausgabe erhalten, weil der Rückwärtsdurchlauf (rot) sie benötigt. (b) Ohne Gradientenverfolgung, etwa bei der Inferenz, lässt sich jede Ausgabe freigeben, sobald die nächste Schicht sie verwendet hat.](../img/autograd-activation-memory.svg)

<!--en:5bafdc1d-->
### Wiederholte Rückwärtsdurchläufe

<!--en:2fe9e585-->
Was passiert, wenn wir den Gradienten am selben Parametervektor $\mathbf{w}$ ein zweites Mal auswerten?

<!--en:3a89e956-->
Der Aufruf `jax.grad(loss)(w)` wertet eine reine Funktion aus. Sie liefert bei jedem Aufruf ein neues Array und speichert keinen Zustand an `w`. Wiederholte Aufrufe ergeben daher ohne Seiteneffekte stets `(2, 4)`. Um Gradienten über mehrere Mikrobatches zu akkumulieren, addieren wir die zurückgegebenen Arrays explizit.

<!--en:93959838-->
MXNet überschreibt `w.grad` standardmäßig bei jedem neuen Rückwärtsdurchlauf und liefert erneut `(2, 4)`.

<!--en:8515fec6-->
In PyTorch nimmt `w.grad` den Wert `(4, 8)` an. PyTorch *addiert* standardmäßig jeden neuen Gradienten zu `w.grad`. So lassen sich Gradienten über mehrere Mikrobatches oder mehrere Verlustterme bequem aufsummieren. In einer gewöhnlichen Trainingsschleife leeren Sie den Puffer vor dem nächsten Schritt stets mit `w.grad.zero_()`.

<!--en:1a4fd689-->
`t.gradient(L, w)` gibt einen neuen Tensor zurück, ohne einen Zustand an `w` zu speichern. Das Ergebnis ist wieder `(2, 4)`. Um Gradienten über mehrere Mikrobatches zu akkumulieren, addieren wir die zurückgegebenen Tensoren explizit.

<!--en:ea810815-->
### Vektorwertige Funktionen

<!--en:685fbefd-->
Eine Operation erzeuge einen Vektor mit $m$ Ausgaben $\mathbf{u} \in \mathbb{R}^m$ aus $n$ Eingaben $\mathbf{x} \in \mathbb{R}^n$ und habe die $m \times n$-Jacobi-Matrix $\mathbf{J}$. Initialisieren wir den Rückwärtsdurchlauf mit einem Gewichtsvektor $\mathbf{v} \in \mathbb{R}^m$, berechnet er das transponierte *Vektor-Jacobi-Produkt* (engl. vector–Jacobian product) $\mathbf{J}^\top \mathbf{v} = \nabla_{\mathbf{x}}(\mathbf{v}^\top \mathbf{u})$, ohne $\mathbf{J}$ explizit aufzubauen. Übergeben wir $\mathbf{v} = \mathbf{1}$ für die quadratischen Fehler der einzelnen Beispiele $(r_1^2, r_2^2, r_3^2)$, differenzieren wir ihre Summe:

<!--en:012c6d66-->
`losses.backward(v)` nimmt $\mathbf{v}$ entgegen (ohne Angabe gilt $\mathbf{1}$).

<!--en:6202f257-->
Beim Aufruf von `backward()` auf einem nichtskalaren Tensor müssen wir $\mathbf{v}$ explizit mit `gradient=torch.ones(3)` übergeben (gleichwertig mit `losses.sum().backward()`).

<!--en:c8b5621d-->
`t.gradient` nimmt $\mathbf{v}$ über `output_gradients` entgegen (ohne Angabe gilt $\mathbf{1}$).

<!--en:56f63dc5-->
`jax.vjp(losses, w)` gibt den Wert aus dem Vorwärtsdurchlauf und eine Funktion zurück, die $\mathbf{v}$ auf $(\mathbf{J}^\top\mathbf{v},)$ abbildet.

<!--en:f5d36c16-->
## Steuerung des Berechnungsgraphen

<!--en:5054076c-->
### Abtrennen von Werten

<!--en:a12480a2-->
Beim *Abtrennen* (engl. detaching) eines Tensors entsteht eine Sicht oder Kopie mit denselben Zahlenwerten. Die Verbindung zu seinen Vorgängern im Berechnungsgraphen entfällt, sodass die Backpropagation ihn als Konstante behandelt. Betrachten wir $u = x^2$ und $z = u x$ für $\mathbf{x} = (0, 1, 2, 3)$ (:numref:`fig_autograd_detach`). Bleibt $u$ verbunden, fließt der Gradient entlang beider Pfade und ergibt $\partial z/\partial x = 3x^2$. Trennen wir $u$ als $c$ ab, behandelt die Ableitung von $z = c x$ den Faktor $c$ als Konstante und liefert $\partial z/\partial x = c = x^2$. Dies ist entscheidend, wenn ein Netz seinen eigenen Bootstrap-Zielwert berechnet, wie beim tiefen Q-Learning (:numref:`sec_dqn`).

<!--en:1fe60a00-->
![Das Produkt $z = ux$ mit $u = x^2$, (a) verbunden und (b) abgetrennt. Blaue Pfeile bezeichnen Operationen im Vorwärtsdurchlauf. Rote Pfeile zeigen den Rückwärtsdurchlauf und tragen die lokalen Ableitungen, mit denen er multipliziert. Das Abtrennen von $u$ erzeugt eine Konstante $c$ mit demselben Wert, ohne Verbindung zu $x$.](../img/autograd-detach.svg)

<!--en:d370327f-->
`u.detach()` liefert einen Tensor mit denselben Werten wie `u`, aber ohne dessen Berechnungsgeschichte.

<!--en:eded9e78-->
`tf.stop_gradient(u)` reicht `u` im Vorwärtsdurchlauf weiter und blockiert die Gradienten im Rückwärtsdurchlauf.

<!--en:6537b7d4-->
`jax.lax.stop_gradient(u)` blockiert die Gradienten innerhalb einer differenzierten Funktion.

<!--en:7d9d00fd-->
Berechnen wir dagegen `z = u * x` direkt, ohne `u` abzutrennen, fließt die Backpropagation durch beide Faktoren von $z = x^3$ und liefert $\partial z / \partial x = 3x^2 = (0, 3, 12, 27)$:

<!--en:0b77f7b8-->
### Abschalten der Gradientenverfolgung

<!--en:44c4d108-->
Bei der Validierung, der Inferenz oder der Parameteraktualisierung $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$ schalten wir die Gradientenverfolgung ab. So gelangen diese Operationen nicht in den Berechnungsgraphen des Verlusts:

<!--en:c6ef6e53-->
Code außerhalb von `autograd.record()` läuft ohne Aufzeichnung. Mit `w[:] = ...` aktualisieren wir `w` direkt und erhalten `w.grad`.

<!--en:3e99d3fa-->
`with torch.no_grad():` schaltet die Verfolgung ab. Das ist nötig, wenn wir einen Blatttensor mit `requires_grad=True` direkt verändern.

<!--en:8f5847ed-->
Code außerhalb von `tf.GradientTape` läuft ohne Aufzeichnung. Mit `w.assign_sub(...)` aktualisieren wir `w` direkt.

<!--en:fd254edc-->
Außerhalb von `jax.grad` zeichnet JAX Arrayoperationen nicht auf.

<!--en:9fa8f97b-->
Wie in :numref:`sec_calculus` verschiebt ein Schritt mit $\eta = 0.1$ den Parametervektor $\mathbf{w}$ von $(1, 0)$ nach $(0.8, -0.4)$ und senkt den Verlust von $1$ auf $0.4$.

<!--en:cde7bea8-->
## Differenzieren von Programmen

<!--en:08cc83c2-->
### Kontrollfluss

<!--en:6a4c84b4-->
Autograd zeichnet die Folge der elementaren Operationen auf, die für eine Eingabe *tatsächlich ausgeführt wurden*. So differenziert es auch Python-Funktionen mit datenabhängigen `while`-Schleifen und `if`/`else`-Verzweigungen direkt:

<!--en:874ee0de-->
Für jeden von null verschiedenen Skalar $a$ (bei $a = 0$ würde die `while`-Schleife nie enden) führt `f(a)` eine feste Folge von Multiplikationen mit $2$ und gegebenenfalls mit $100$ aus. Daher ist `f(a) = k * a` stückweise linear mit lokaler Steigung $k = f(a)/a$ (:numref:`fig_autograd_dynamic`). Für $a = 0.3$ sind $12$ Verdopplungen nötig, bis $0.3 \times 2^{12} = 1228.8 \ge 1000$ erreicht ist. Danach läuft der positive Zweig ($k = 2^{12} = 4096$). Für $a = -0.75$ führen $11$ Verdopplungen zu $-1536$, und es läuft der `else`-Zweig ($k = 100 \times 2^{11} = 204{,}800$). Autograd liefert beide Steigungen:

<!--en:a609f90f-->
![Die von `f` aufgezeichneten Graphen für zwei Eingaben. Für $a = 0.3$ verdoppelt `f` den Wert von $a$ insgesamt $12$-mal, und der erste Zweig läuft. Für $a = -0.75$ verdoppelt die Funktion den Wert von $a$ insgesamt $11$-mal, und der zweite Zweig läuft. Jeder Graph ist linear in $a$ und hat seine eigene Steigung.](../img/autograd-dynamic.svg)

<!--en:d983e5f5-->
Autograd differenziert den *tatsächlich ausgeführten Rechenweg*. An einer Schwelle, an der sich die Zahl der Schleifendurchläufe sprunghaft ändert, ist die mathematische Funktion $f(a)$ unstetig. Autograd liefert dennoch die Steigung des ausgeführten Zweigs.

<!--en:8dcc7309-->
## Vorwärts- und Rückwärtsmodus

<!--en:ec4a3859-->
Für eine Abbildung $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$ mit der $m \times n$-Jacobi-Matrix $\mathbf{J}$ lässt sich die Kettenregel in beiden Richtungen anwenden (:numref:`fig_autograd_fwd_rev`, siehe :citet:`Wengert.1964,Linnainmaa.1970` und :numref:`sec_mdl-matrix-calculus-autodiff`):

<!--en:2a33f32d-->
* **Rückwärtsmodus (Vektor-Jacobi-Produkt, VJP):** Ein Rückwärtsdurchlauf mit dem Startvektor $\mathbf{v} \in \mathbb{R}^m$ an den Ausgaben berechnet $\mathbf{J}^\top \mathbf{v}$. Mit dem $j^\textrm{th}$ Standardbasisvektor $\mathbf{v} = \mathbf{e}_j$ erhält man die Zeile $j$ von $\mathbf{J}$ ($m$ Durchläufe für die gesamte Jacobi-Matrix).

<!--en:66ded8ad-->
* **Vorwärtsmodus (Jacobi-Vektor-Produkt, JVP):** Ein Vorwärtsdurchlauf mit dem Tangentenvektor $\mathbf{d} \in \mathbb{R}^n$ an den Eingaben führt die Ableitungen zusammen mit den Werten vorwärts und berechnet $\mathbf{J}\mathbf{d}$. Der Startvektor $\mathbf{d} = \mathbf{e}_i$ liefert die Spalte $i$ von $\mathbf{J}$ ($n$ Durchläufe für die gesamte Jacobi-Matrix, mit konstantem Speicherfaktor $2\times$).

<!--en:ffb242db-->
![Zwei Möglichkeiten, die $m \times n$-Jacobi-Matrix $\mathbf{J}$ einer Funktion mit $n$ Eingaben und $m$ Ausgaben zu füllen: (a) Vorwärtsmodus mit insgesamt $n$ Durchläufen, (b) Rückwärtsmodus mit insgesamt $m$ Durchläufen.](../img/autograd-fwd-vs-rev.svg)

<!--en:0a96d980-->
MXNet implementiert automatisches Differenzieren ausschließlich im Rückwärtsmodus.

<!--en:2511c0e9-->
PyTorch stellt `torch.func.vjp`, `torch.func.jvp`, `torch.func.jacrev` und `torch.func.jacfwd` bereit.

<!--en:34a5499e-->
TensorFlow stellt `GradientTape.jacobian` (Rückwärtsmodus) und `tf.autodiff.ForwardAccumulator` (Vorwärtsmodus) bereit.

<!--en:911ee726-->
JAX stellt `jax.vjp`, `jax.jvp`, `jax.jacrev` und `jax.jacfwd` bereit.

<!--en:1621c5cb-->
Eine skalare Verlustfunktion hat $m = 1$ Ausgabe und $n$ Parameter. Ihre Jacobi-Matrix besteht also aus einer einzigen Zeile $\nabla_{\mathbf{w}} L^\top$. Der Rückwärtsmodus (*Backpropagation*) berechnet alle $n$ Komponenten des Gradienten in *einem Durchlauf*. Der Vorwärtsmodus bräuchte $n$ Durchläufe. Das erklärt, warum zum Training tiefer Netze durchweg der Rückwärtsmodus zum Einsatz kommt.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:5af680dd-->
Automatisches Differenzieren zeichnet die elementaren Operationen des Vorwärtsdurchlaufs in einem Berechnungsgraphen auf. Die lokalen Ableitungsregeln liefern in umgekehrter Reihenfolge exakte Gradienten. Für unser Geradenmodell erhalten wir so $\nabla_{\mathbf{w}} L = (2, 4)$, ohne Code für die Ableitung zu schreiben. Trennen wir ein Zwischenergebnis ab, verliert es die Verbindung zu seinen Vorgängern, und die Backpropagation behandelt es als Konstante. Schalten wir die Gradientenverfolgung ab, bleiben Parameteraktualisierungen und Inferenzauswertungen außerhalb des Graphen. Aktivierungen aus dem Vorwärtsdurchlauf müssen dann nicht zwischengespeichert werden. Da der Graph die für eine Eingabe tatsächlich ausgeführten Operationen erfasst, differenziert Autograd auch durch Schleifen und Verzweigungen in Python. Ein Durchlauf im Vorwärtsmodus berechnet ein Jacobi-Vektor-Produkt $\mathbf{J}\mathbf{d}$ (eine Spalte pro Standardbasisvektor als Startwert). Ein Durchlauf im Rückwärtsmodus berechnet ein Vektor-Jacobi-Produkt $\mathbf{J}^\top\mathbf{v}$ (eine Zeile pro Standardbasisvektor). Für einen skalaren Verlust ($m = 1$) liefert daher ein einziger Rückwärtsdurchlauf den Gradienten nach allen $n$ Parametern.

<!--en:2294243a-->
## Übungen

<!--en:46543801-->
1. [code] **Abgetrennte Zwischenergebnisse.** Es gelte `u = x * x` und `c = u.detach()`. Betrachten Sie `z = c * c`, `z = u * c` und `z = u * u`. Berechnen Sie für jedes `z` die Werte `u`, `c` und `z` neu innerhalb eines `autograd.record()`-Blocks und rufen Sie `z.backward()` auf. Sagen Sie vor der Ausführung für jeden Aufruf das Ergebnis einschließlich `x.grad` vorher. Prüfen Sie Ihre Vorhersagen und erklären Sie das Verhältnis der Gradienten für `z = u * c` und `z = u * u`.

<!--en:5ee70aef-->
2. [code] **Manuelle Parameteraktualisierung.** Führen Sie den Gradientenschritt dieses Abschnitts mit `w = w - 0.1 * w.grad` statt mit `w[:] = w - 0.1 * w.grad` aus und untersuchen Sie `w.grad`. Erklären Sie, was die neue Bindung für den von `attach_grad` reservierten Puffer bedeutet. Sagen Sie vorher, was bei einem zweiten aufgezeichneten Vorwärtsdurchlauf mit anschließendem `backward` passiert, und prüfen Sie Ihre Vorhersage.

<!--en:ba97f06b-->
1. [code] **Abgetrennte Zwischenergebnisse.** Es gelte `u = x * x` und `c = u.detach()`. Betrachten Sie `z = c * c`, `z = u * c` und `z = u * u`. Berechnen Sie für jedes `z` die Werte `u` und `c` neu, setzen Sie `x.grad` zurück und rufen Sie `backward` auf der Summe der Einträge von `z` auf. Sagen Sie vor der Ausführung für jeden Aufruf das Ergebnis einschließlich `x.grad` vorher. Prüfen Sie Ihre Vorhersagen und erklären Sie das Verhältnis der Gradienten für `z = u * c` und `z = u * u`.

<!--en:70f8818c-->
2. [code] **Manuelle Parameteraktualisierung.** Führen Sie den Gradientenschritt dieses Abschnitts mit `w = w - 0.1 * w.grad` außerhalb von `torch.no_grad()` aus. Untersuchen Sie `w.requires_grad`, `w.grad_fn`, `w.is_leaf` und `w.grad`. Erklären Sie, wie die Aktualisierung den Graphen verändert hat. Sagen Sie vorher, was ein zweiter Vorwärts- und Rückwärtsdurchlauf in `w.grad` hinterlässt, und prüfen Sie Ihre Vorhersage.

<!--en:145bfcd7-->
1. [code] **Abgetrennte Zwischenergebnisse.** Berechnen Sie `u = x * x` und `c = tf.stop_gradient(u)` innerhalb einer dauerhaften Gradientenaufzeichnung mit `tf.GradientTape` namens `t`. Sagen Sie vor der Ausführung das Ergebnis von `t.gradient(z, x)` für `z = c * c`, `z = u * c` und `z = u * u` vorher, die alle innerhalb von `t` berechnet werden. Prüfen Sie Ihre Vorhersagen und erklären Sie das Verhältnis der Ergebnisse für `z = u * c` und `z = u * u`.

<!--en:93468bd7-->
2. [code] **Manuelle Parameteraktualisierung.** Führen Sie den Gradientenschritt dieses Abschnitts mit `w = w - 0.1 * g` aus, wobei `g` der von der Aufzeichnung gelieferte Gradient ist. Ersetzen Sie damit den Aufruf von `assign_sub` und untersuchen Sie `type(w)`. Sagen Sie vorher, was `gradient` für einen zweiten Vorwärtsdurchlauf mit neuer Aufzeichnung zurückgibt. Prüfen Sie Ihre Vorhersage und erklären Sie, warum eine Trainingsschleife Variablen mit `assign_sub` aktualisiert.

<!--en:d8691d1d-->
1. [code] **Abgetrennte Zwischenergebnisse.** Schreiben Sie drei Funktionen von `x`, die jeweils `u = x * x` und `c = jax.lax.stop_gradient(u)` berechnen. Die erste soll `c * c` zurückgeben, die zweite `u * c` und die dritte `u * u`. Sagen Sie vor der Ausführung für jedes Ergebnis den Gradienten der Summe seiner Einträge nach `x` vorher. Prüfen Sie Ihre Vorhersagen mit `jax.grad` und erklären Sie das Verhältnis der Ergebnisse für `u * c` und `u * u`.

<!--en:8f79c6d8-->
2. [code] **Manuelle Parameteraktualisierung.** Es gelte `w = jnp.arange(4.0)` und `g = lambda x: (w * x).sum()`. Die Funktion liest `w` aus dem umgebenden Gültigkeitsbereich. Sagen Sie das Ergebnis von `jax.grad(g)(x)` vorher, prüfen Sie es und geben Sie an, nach welchem Array `jax.grad` differenziert. Schreiben Sie dann `g` so um, dass `jax.grad` den Gradienten von `(w * x).sum()` nach `w` zurückgibt. Aktualisieren Sie damit einmal `w` mit Lernrate 0.1. Erklären Sie, wie eine JAX-Trainingsschleife demnach die Parameter an die Verlustfunktion übergeben muss.

<!--en:48f06460-->
3. [code] **Von der skalaren zur vektorwertigen Eingabe.** Ersetzen Sie die skalare Eingabe `a` der Funktion `f` aus :numref:`subsec_autograd_control_flow` durch einen Zufallsvektor der Länge 5. Differenzieren Sie die Summe der Einträge von `fa = f(a)` nach `a`.
    1. Drücken Sie `f(a)` durch `a` aus. Geben Sie die Jacobi-Matrix von `f` an der Stelle `a` und den Gradienten der Summe der Einträge von `fa` an.
    1. Berechnen Sie diesen Gradienten durch automatisches Differenzieren und prüfen Sie, ob er noch elementweise mit `fa / a` übereinstimmt.
    1. Beschreiben Sie die Menge der Eingaben `a`, an denen `f` springt. Welcher Teil dieser Menge hat bei einer skalaren Eingabe keine Entsprechung?

<!--en:8a45abd7-->
1. [code] **Programme und Funktionen.** Automatisches Differenzieren leitet die Operationen ab, die ein Programm ausführt. Verschiedene Programme können dabei dieselbe mathematische Funktion berechnen.
    1. Die Funktionen $|x|$ und $\sqrt{x^2}$ sind gleich. Berechnen Sie beide Ableitungen bei $x = 0$ durch automatisches Differenzieren. Verwenden Sie die Betrags- und die Quadratwurzelfunktion des Frameworks. Erklären Sie beide Ergebnisse anhand der lokalen Ableitungen der beteiligten Operationen. Keine der Funktionen ist bei $0$ differenzierbar. Welches der beiden Ergebnisse schadet beim Training mehr und warum?
    1. Schreiben Sie ein Programm, das $x^2$ für jede Eingabe $x$ zurückgibt, dessen automatische Ableitung bei $x = 0$ aber $1$ statt $0$ beträgt. Erklären Sie, warum automatisches Differenzieren diesen Unterschied nicht erkennen kann.

<!--en:7827b2eb-->
1. **Aufwand für zweite Ableitungen.** ● Das Zählargument in :numref:`subsec_autograd_fwd_rev` ergibt für den Gradienten eines skalaren Verlusts mit $n$ Parametern einen Aufwand von einem Rückwärtsdurchlauf. Schätzen Sie mit demselben Argument den Aufwand für die gesamte Hesse-Matrix (die $n \times n$-Matrix der zweiten Ableitungen) und für ein einzelnes Hesse-Matrix-Vektor-Produkt. Verwenden Sie als Einheit den Aufwand einer Funktionsauswertung. Erklären Sie den Unterschied.

<!--en:a0062c88-->
[Diskussion](https://d2l.discourse.group/t/34)

<!--en:c6310535-->
[Diskussion](https://d2l.discourse.group/t/35)

<!--en:aace7e50-->
[Diskussion](https://d2l.discourse.group/t/200)

<!--en:e6d9297e-->
[Diskussion](https://d2l.discourse.group/t/17970)

<!--en:882c7fed-->
Automatisches Differenzieren: Berechnungsgraphen, Rückwärtsmodus und dynamische Programme<br>**Vorwärts- und Rückwärtsmodus · Berechnungsgraphen · Vektor-Jacobi-Produkte · Abtrennen und Speicherverwaltung · dynamischer Kontrollfluss**

<!--en:e9d67459-->
::: {.slide title="Drei Wege, Code zu differenzieren: Warum Deep Learning Autograd braucht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:402fd5d8-->
- **Symbolische Algebra:** Ausdrücke wachsen exponentiell. Das Verfahren scheitert an Schleifen und Verzweigungen in Python.

<!--en:33486ca2-->
- **Numerische endliche Differenzen:** $O(n)$ Vorwärtsauswertungen ($10^9$ pro Schritt). Auslöschung bei der Subtraktion kostet die Hälfte der Stellen von `float32`.

<!--en:4a5e8037-->
- **Autograd im Rückwärtsmodus:** Exakte lokale Regeln für Vektor-Jacobi-Produkte (VJPs) liefern entlang der aufgezeichneten Operationen $\nabla_{\mathbf{w}} L \in \mathbb{R}^n$ in **einem Rückwärtsdurchlauf**.

<!--en:0932e548-->
::: {.slide title="Vorwärts- und Rückwärtsmodus bei n ≫ m"}

<!--en:3141b480-->
[Jacobi-Matrizen und Durchläufe]{.kicker}

<!--en:78720556-->
Für $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ mit der $m \times n$-Jacobi-Matrix $\mathbf{J}$ ($J_{ji} = \partial f_j / \partial x_i$) gilt:

<!--en:b0152040-->
- **Vorwärtsmodus (JVP, $\mathbf{d} \mapsto \mathbf{J}\mathbf{d}$):** Start mit Eingabetangente $\dot{\mathbf{x}} = \mathbf{e}_i$, dann vorwärts **eine Spalte von $\mathbf{J}$** pro Durchlauf berechnen ($n$ Durchläufe).

<!--en:5769e42a-->
- **Rückwärtsmodus (VJP, $\mathbf{v} \mapsto \mathbf{J}^\top \mathbf{v}$):** Start mit adjungierter Ausgabegröße $\bar{\mathbf{y}} = \mathbf{e}_j$, dann rückwärts **eine Zeile von $\mathbf{J}$** pro Durchlauf berechnen ($m$ Durchläufe).

<!--en:2d5d550d-->
- **Warum Backpropagation skaliert:** Für einen skalaren Verlust gilt $m = 1 \ll n$. Der Rückwärtsmodus berechnet alle $n$ Komponenten von $\nabla_{\mathbf{w}} L$ in **einem Rückwärtsdurchlauf**.

<!--en:99ec0250-->
::: {.slide title="Der Vorwärtsdurchlauf: Von der Verlustfunktion zum Code"}

<!--en:43bf2c56-->
[Aufzeichnung des Vorwärtsdurchlaufs]{.kicker}

<!--en:f01bea4b-->
- In Deep-Learning-Frameworks genügt es, **den Vorwärtsdurchlauf** zu programmieren. Der Modellcode braucht keine von Hand hergeleitete Ableitungsformel.

<!--en:7363c44c-->
- **Durchgängiges Beispiel (§{sec_calculus}):** Lineares Modell $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ mit Entwurfsmatrix $\mathbf{X} \in \mathbb{R}^{3 \times 2}$ und Zielwerten $\mathbf{y} = [1, 1, 0]^\top$, ausgewertet bei $\mathbf{w} = [1, 0]^\top$.

<!--en:8b4f3a81-->
- Jede elementare Operation in `loss(w)` (Matrix-Vektor-Produkt, Subtraktion, elementweises Quadrieren und Summieren) hat ihre eigene lokale VJP-Regel für den Rückwärtsdurchlauf.

<!--en:87558343-->
::: {.slide title="Gradientenverfolgung aktivieren und Vorwärtsdurchlauf aufzeichnen" except="jax"}

<!--en:f39e2496-->
[Aufbau des Graphen]{.kicker}

<!--en:41f766b6-->
- **Explizite Auswahl spart GPU-Speicher:** Frameworks zeichnen nur DAG-Knoten für Parameter auf, die ausdrücklich für die Gradientenverfolgung markiert sind.

<!--en:ef715e25-->
::: {.slide title="Funktionales Differenzieren mit jax.grad: Reine Funktionen ohne veränderlichen Zustand" only="jax"}

<!--en:f39e2496-2-->
[Aufbau des Graphen]{.kicker}

<!--en:d8b1a04f-->
- **Reine Funktionstransformation:** JAX-Arrays haben keinen `.grad`-Zustand. `jax.grad(loss)` verwandelt `loss(w)` in eine reine Funktion mit Rückgabewert $\nabla_{\mathbf{w}} L(\mathbf{w}) = [2, 4]^\top$.

<!--en:953fe067-->
::: {.slide title="Im Berechnungsgraphen: Vorwärtswerte und adjungierte Größen im Rückwärtsdurchlauf"}

<!--en:71b62cf0-->
[Funktionsweise der Backpropagation]{.kicker}

<!--en:905933a9-->
- **Vorwärtsdurchlauf (blau, links $\to$ rechts):** Zwischenaktivierungen bei $\mathbf{w} = (1, 0)$ berechnen und zwischenspeichern:
  - $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} = (1,1,1)$ $\to$ $\mathbf{r} = \hat{\mathbf{y}} - \mathbf{y} = (0,0,1)$
  - $\mathbf{s} = \mathbf{r}^{\odot 2} = (0,0,1)$ $\to$ $L = \sum_i s_i = 1$

<!--en:dcc5557b-->
- **Rückwärtsdurchlauf (rot, rechts $\to$ links):** Mit $\bar{L} = 1$ starten und die lokale VJP-Regel jedes Knotens anwenden:
  - $\bar{L}=1$ $\to$ $\bar{\mathbf{s}}=(1,1,1)$ $\to$ $\bar{\mathbf{r}} = 2\mathbf{r} \odot \bar{\mathbf{s}} = (0,0,2)$
  - $\bar{\hat{\mathbf{y}}} = \bar{\mathbf{r}} = (0,0,2)$ $\to$ $\bar{\mathbf{w}} = \mathbf{X}^\top \bar{\hat{\mathbf{y}}} = (2,4)$

<!--en:93808dc4-->
::: {.slide title="Rückwärtsdurchlauf und Prüfung der analytischen Gradienten"}

<!--en:fe05ce94-->
[Rückwärtsdurchlauf]{.kicker}

<!--en:a5b82d6f-->
- Der Rückwärtsdurchlauf besucht die Knoten in umgekehrter topologischer Reihenfolge und liefert den exakten analytischen Gradienten $\nabla_{\mathbf{w}} L = \mathbf{X}^\top(2\mathbf{r}) = [2, 4]^\top$.

<!--en:e67a5f1a-->
- **Prüfung von Hand bei $\mathbf{w} = [1, 0]^\top$:**
  - **Residuen:** $\mathbf{r} = \mathbf{X}\mathbf{w} - \mathbf{y} = [1, 1, 1]^\top - [1, 1, 0]^\top = [0, 0, 1]^\top$.
  - **VJP im Rückwärtsdurchlauf:** $\mathbf{X}^\top(2\mathbf{r}) = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ 0 \\ 2 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$.

<!--en:71a68ca1-->
::: {.slide title="Gradientenakkumulation: Warum PyTorch standardmäßig zu .grad addiert" only="pytorch"}

<!--en:8dbd01f1-->
[Puffer mit Zustand]{.kicker}

<!--en:7e78258e-->
- **Standardmäßige Akkumulation (`w.grad += g`):** Gradienten über Mikrobatches summieren, wenn ein ganzer Batch den GPU-Speicher übersteigt.

<!--en:c723cd28-->
- **Typischer Fehler:** Fehlt `w.grad.zero_()` vor dem nächsten Schritt, addiert PyTorch zum alten Gradienten (`(2, 4) + (2, 4) = (4, 8)`).

<!--en:ecb7d932-->
::: {.slide title="Ein zweiter Rückwärtsdurchlauf liefert einen neuen Gradienten" only="tensorflow,mxnet"}

<!--en:7d59fdbd-->
[Gradienten ohne eigenen Zustand]{.kicker}

<!--en:7d72253e-->
- **Standardmäßig keine Akkumulation:** Jeder Rückwärtsdurchlauf liefert allein den Gradienten des aktuellen Verlusts. Ein zweiter Durchlauf ergibt erneut $(2, 4)$ statt $(4, 8)$.

<!--en:32f3c2d5-->
- **Akkumulation über Mikrobatches** erfordert daher explizites Summieren der zurückgegebenen Gradienten.

<!--en:8431ea91-->
::: {.slide title="Gradientenakkumulation und rein funktionale Gradienten in JAX" only="jax"}

<!--en:7d59fdbd-2-->
[Gradienten ohne eigenen Zustand]{.kicker}

<!--en:5375bb6c-->
- **Kein gespeicherter Gradientenzustand:** `jax.grad(loss)(w)` liefert bei jedem Aufruf ein neues Array. Es gibt keinen zurückzusetzenden Puffer und keinen alten Gradienten, der den nächsten Schritt verfälschen könnte.

<!--en:ea5671d2-->
::: {.slide title="Rückwärtsdurchlauf bei nichtskalaren Ausgaben: Vektor-Jacobi-Produkte (VJPs)"}

<!--en:a9ce307a-->
[Vektorwertige Ausgaben]{.kicker}

<!--en:705dd6ac-->
- Beim Differenzieren von $\mathbf{u}(\mathbf{w}) \in \mathbb{R}^m$ entsteht keine explizite $m \times n$-Jacobi-Matrix. Ein Rückwärtsdurchlauf mit Startvektor $\mathbf{v} \in \mathbb{R}^m$ berechnet $\mathbf{J}^\top \mathbf{v} = \nabla_{\mathbf{w}}(\mathbf{v}^\top \mathbf{u})$ ($\mathbf{v} = \mathbf{1}$ summiert die Gradienten der einzelnen Beispiele).

<!--en:ca293c24-->
::: {.slide title="Teilgraphen mit detach und stop_gradient abtrennen"}

<!--en:a5ae4216-->
[Eingriffe in den Graphen]{.kicker}

<!--en:bbceca6c-->
- Für $u = x^2$ und $z = u \cdot x = x^3$ gilt:
  - **Verbunden (a):** Gradienten fließen durch beide Pfade $\implies \frac{\partial z}{\partial x} = 2x^2 + x^2 = 3x^2$.
  - **Abgetrennt (b):** `c = u.detach()` (`stop_gradient(u)`) fixiert $u$ als Konstante $x^2 \implies \frac{\partial z}{\partial x}\big|_c = c = x^2$.

<!--en:2587f097-->
- **Anwendungen im maschinellen Lernen:** DQN-Zielnetze, Lehrerzweige von BYOL/SimSiam und Aktualisierungen des GAN-Diskriminators.

<!--en:e4577fd5-->
::: {.slide title="Abtrennen im Code: Gradienten mit Verbindung (3x²) und ohne (x²)"}

<!--en:af80e978-->
[Abtrennen in der Praxis]{.kicker}

<!--en:45e0b126-->
- **Mit `detach()` / `stop_gradient()` (`z = c * x`):**
  Bei $\mathbf{x} = [0, 1, 2, 3]^\top$ wirkt $c = \mathbf{x}^2 = [0, 1, 4, 9]^\top$ als Konstante und ergibt $\nabla_{\mathbf{x}} \sum_i z_i = \mathbf{x}^2 = [0, 1, 4, 9]^\top$:

<!--en:18959810-->
- **Ohne `detach()` (`z = u * x`):**
  Beide Pfade von $\mathbf{x}$ nach $\mathbf{z}$ tragen Gradienten bei und ergeben die vollständige Ableitung der kubischen Funktion $\nabla_{\mathbf{x}} \sum_i z_i = 3\mathbf{x}^2 = [0, 3, 12, 27]^\top$:

<!--en:ca039c3c-->
::: {.slide title="Speicher für Aktivierungen und Abschalten der Gradientenverfolgung"}

<!--en:7d907b49-->
[Speicher und Inferenz]{.kicker}

<!--en:eeba108d-->
- **Beim Training bleiben $O(L)$ Aktivierungen im Speicher:** VJPs benötigen Zwischenergebnisse des Vorwärtsdurchlaufs ($\mathbf{r}$ für $2\mathbf{r}$, $\mathbf{X}$ für $\mathbf{X}^\top$). Deshalb bleiben die Aktivierungen jeder Schicht bis zum Rückwärtsdurchlauf im HBM.

<!--en:506fd4e5-->
- **Inferenz ohne Gradientenverfolgung:** Jeder Schichtpuffer wird sofort freigegeben (Speicherbedarf $O(1)$ bezüglich der Tiefe). SGD-Schritte werden nicht aufgezeichnet.

<!--en:978f2682-->
::: {.slide title="Dynamischer Kontrollfluss: Differenzieren durch Schleifen und Verzweigungen in Python"}

<!--en:90325ab6-->
[Differenzierbare Programme]{.kicker}

<!--en:47ac8bfd-->
- **Graphaufbau zur Laufzeit:** Jede Vorwärtsauswertung zeichnet genau die Folge elementarer Tensoroperationen auf, die für die konkrete Eingabe *tatsächlich ausgeführt wurden*.

<!--en:3d6dc927-->
- Ablauf von `f(a)`:
  1. `b` in einer datenabhängigen `while`-Schleife verdoppeln, bis $\|\mathbf{b}\|_2 \ge 1000$ gilt (insgesamt $k$ Verdopplungen).
  2. Anhand von `b.sum() > 0` verzweigen und entweder `b` oder `100 * b` zurückgeben.

<!--en:a29efbab-->
- Für jede skalare Eingabe $a \neq 0$ besteht der ausgeführte Pfad aus einer Kette von Multiplikationen $f(a) = m \cdot 2^k \cdot a$ mit $m \in \{1, 100\}$.

<!--en:43a07bc3-->
::: {.slide title="Jeder Rechenweg liefert einen eigenen Graphen und dessen exakte Steigung"}

<!--en:5f712abd-->
[Ausgeführter Rechenweg und Funktion]{.kicker}

<!--en:745298a1-->
- Da $f(a) = k \cdot a$ entlang jedes ausgeführten Pfades stückweise linear ist, entspricht die Ableitung genau der Steigung $k = f(a) / a$:
  - $a = 0.3 > 0$: $12$ Verdopplungen (`if`-Zweig) $\implies k = 2^{12} = 4096$.
  - $a = -0.75 < 0$: $11$ Verdopplungen (`else`-Zweig) $\implies k = 100 \cdot 2^{11} = 204{,}800$.

<!--en:65b26457-->
- **Einschränkung:** Autograd differenziert den *ausgeführten Rechenweg*. An Sprungstellen, an denen sich die Anzahl der Schleifendurchläufe ändert, liefert es die Steigung des ausgeführten Zweigs.

<!--en:a215e57b-->
::: {.slide title="Verständnisfragen zu Aktivierungsspeicher, Abtrennen und Akkumulationsfehlern" only="pytorch"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:63b829c1-->
Welches Verhalten und welche Gradienten ergeben sich in den einzelnen Fällen?

<!--en:fc1ad2fb-->
1. **Zurücksetzen vergessen:** Für 3 Mikrobatches läuft jeweils `L.backward()` ohne Aufruf von `w.grad.zero_()`. Was enthält `w.grad`?

<!--en:4d0ad23c-->
2. **Abgetrennte Zweige:** Es gelte `u = x * x` und `c = u.detach()` (oder `stop_gradient(u)`). Wie lautet $\nabla_{\mathbf{x}} \sum_i z_i$ für (a) `z = u * u`, (b) `z = u * c` und (c) `z = c * c`?

<!--en:173df157-->
3. **Vorwärts- und Rückwärtsdurchläufe:** Ein physikalischer Simulator bildet $n = 3$ Parameter einer Flügelform auf $m = 10^6$ Druckwerte eines Rechengitters ab. Welcher Modus des automatischen Differenzierens berechnet die gesamte $10^6 \times 3$-Jacobi-Matrix schneller? Wie viele Durchläufe benötigt er?

<!--en:557b98b6-->
**Lösungen:**

<!--en:fac9efca-->
1. **Die Summe aller 3 Mikrobatchgradienten** ($\sum_{b=1}^3 \nabla_{\mathbf{w}} L_b$). Für die Gradientenakkumulation über Mikrobatches ist das beabsichtigt. Hat der Optimierer nach den Mikrobatches 1 und 2 schon einen Schritt gemacht, ist es ein Fehler.

<!--en:1c7a1c21-->
2. (a) `z = u * u = x^4` $\implies$ **$4\mathbf{x}^3$**. (b) `z = u * c = c * x^2` $\implies$ **$2c\mathbf{x} = 2\mathbf{x}^3$** (die Hälfte von $4\mathbf{x}^3$, da ein Faktor festgehalten wird). (c) Von `z = c * c` führt kein aktiver Pfad zu `x` $\implies$ **$0$** (oder kein `grad_fn`).

<!--en:299ddd4c-->
3. **Der Vorwärtsmodus (JVP)** braucht nur **$n = 3$ Durchläufe** (einen pro Spalte). Der Rückwärtsmodus braucht $m = 10^6$ Durchläufe (einen pro Zeile).

<!--en:3f5bca23-->
::: {.slide title="Verständnisfragen zu Aktivierungsspeicher, Abtrennen und Akkumulationsfehlern" except="pytorch"}

<!--en:929b7671-2-->
[Wissensprüfung]{.kicker}

<!--en:63b829c1-2-->
Welches Verhalten und welche Gradienten ergeben sich in den einzelnen Fällen?

<!--en:8f5912e4-->
1. **Mikrobatches:** Der Rückwärtsdurchlauf läuft nacheinander auf 3 Mikrobatches. Was enthält der letzte Gradient, und wie lassen sich alle drei akkumulieren?

<!--en:4d0ad23c-2-->
2. **Abgetrennte Zweige:** Es gelte `u = x * x` und `c = u.detach()` (oder `stop_gradient(u)`). Wie lautet $\nabla_{\mathbf{x}} \sum_i z_i$ für (a) `z = u * u`, (b) `z = u * c` und (c) `z = c * c`?

<!--en:173df157-2-->
3. **Vorwärts- und Rückwärtsdurchläufe:** Ein physikalischer Simulator bildet $n = 3$ Parameter einer Flügelform auf $m = 10^6$ Druckwerte eines Rechengitters ab. Welcher Modus des automatischen Differenzierens berechnet die gesamte $10^6 \times 3$-Jacobi-Matrix schneller? Wie viele Durchläufe benötigt er?

<!--en:557b98b6-2-->
**Lösungen:**

<!--en:b0a55fb5-->
1. **Nur den Gradienten des dritten Mikrobatchs** $\nabla_{\mathbf{w}} L_3$. Frühere Ergebnisse bleiben unverändert. Alle drei zurückgegebenen Gradienten müssen explizit addiert werden.

<!--en:1c7a1c21-2-->
2. (a) `z = u * u = x^4` $\implies$ **$4\mathbf{x}^3$**. (b) `z = u * c = c * x^2` $\implies$ **$2c\mathbf{x} = 2\mathbf{x}^3$** (die Hälfte von $4\mathbf{x}^3$, da ein Faktor festgehalten wird). (c) Von `z = c * c` führt kein aktiver Pfad zu `x` $\implies$ **$0$** (oder kein `grad_fn`).

<!--en:299ddd4c-2-->
3. **Der Vorwärtsmodus (JVP)** braucht nur **$n = 3$ Durchläufe** (einen pro Spalte). Der Rückwärtsmodus braucht $m = 10^6$ Durchläufe (einen pro Zeile).

<!--en:50e899fb-->
::: {.slide title="Überblick: Autograd-Operationen, Speicherbedarf und Anwendungen" only="pytorch"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:1d2b114b-->
| Operation / Begriff | API | Graph und Speicher | Anwendung im Deep Learning |

<!--en:5116f977-->
| **Aktivieren / Aufzeichnen** | `w.requires_grad_(True)` | Speichert Vorwärtsaktivierungen im GPU-Speicher für VJPs im Rückwärtsdurchlauf | Übliches Training neuronaler Netze ($O(1)$ Rückwärtsdurchlauf) |

<!--en:10f45677-->
| **Rückwärtsdurchlauf (VJP)** | `L.backward()` oder `u.backward(v)` | Gibt Aktivierungspuffer nach Berechnung von $\mathbf{J}^\top \mathbf{v}$ frei | Backpropagation eines skalaren Verlusts ($m = 1$) oder eines nachgelagerten Vektors $\mathbf{v}$ |

<!--en:78a1bfc1-->
| **Zurücksetzen / Akkumulieren** | `w.grad.zero_()` (addiert standardmäßig mit `+=`) | Legt fest, ob der neue Gradient zum letzten addiert wird oder ihn ersetzt | Gradienten über Mikrobatches akkumulieren oder neuen Optimierungsschritt beginnen |

<!--en:9e0f8049-->
| **Teilgraph abtrennen** | `u.detach()` | Konstante Sicht ohne Kopie, trennt die Rückwärtskante | Zielnetze beim Reinforcement Learning, Lehrerzweig von BYOL/SimSiam, GAN-Schritte |

<!--en:666da2dd-->
| **Inferenz ohne Gradienten** | `with torch.no_grad():` | Kein Graphaufbau und keine Zwischenspeicherung von Aktivierungen ($3\text{--}10\times$ weniger GPU-Speicher) | Validierungsschleifen, Inferenz im Einsatz, Parameteraktualisierungen |

<!--en:9c45af02-->
::: {.slide title="Überblick: Autograd-Operationen, Speicherbedarf und Anwendungen" only="tensorflow"}

<!--en:c5a222f8-2-->
[Das Wichtigste]{.kicker}

<!--en:1d2b114b-2-->
| Operation / Begriff | API | Graph und Speicher | Anwendung im Deep Learning |

<!--en:dd331d1c-->
| **Aktivieren / Aufzeichnen** | `tf.Variable` + `tf.GradientTape()` | Speichert Vorwärtsaktivierungen im GPU-Speicher für VJPs im Rückwärtsdurchlauf | Übliches Training neuronaler Netze ($O(1)$ Rückwärtsdurchlauf) |

<!--en:99837903-->
| **Rückwärtsdurchlauf (VJP)** | `t.gradient(L, w)` oder `t.gradient(u, w, output_gradients=v)` | Gibt Aktivierungspuffer nach Berechnung von $\mathbf{J}^\top \mathbf{v}$ frei | Backpropagation eines skalaren Verlusts ($m = 1$) oder eines nachgelagerten Vektors $\mathbf{v}$ |

<!--en:ddaaeaf3-->
| **Zurücksetzen / Akkumulieren** | Keine: neuer Tensor pro Aufruf | Legt fest, ob der neue Gradient zum letzten addiert wird oder ihn ersetzt | Gradienten über Mikrobatches akkumulieren oder neuen Optimierungsschritt beginnen |

<!--en:e2ffcd7f-->
| **Teilgraph abtrennen** | `tf.stop_gradient(u)` | Konstante Sicht ohne Kopie, trennt die Rückwärtskante | Zielnetze beim Reinforcement Learning, Lehrerzweig von BYOL/SimSiam, GAN-Schritte |

<!--en:523fca08-->
| **Inferenz ohne Gradienten** | Außerhalb von `GradientTape` | Kein Graphaufbau und keine Zwischenspeicherung von Aktivierungen ($3\text{--}10\times$ weniger GPU-Speicher) | Validierungsschleifen, Inferenz im Einsatz, Parameteraktualisierungen |

<!--en:fc0d418a-->
::: {.slide title="Überblick: Autograd-Operationen, Speicherbedarf und Anwendungen" only="jax"}

<!--en:c5a222f8-3-->
[Das Wichtigste]{.kicker}

<!--en:1d2b114b-3-->
| Operation / Begriff | API | Graph und Speicher | Anwendung im Deep Learning |

<!--en:ee184a57-->
| **Aktivieren / Aufzeichnen** | `jax.grad(f)` | Speichert Vorwärtsaktivierungen im GPU-Speicher für VJPs im Rückwärtsdurchlauf | Übliches Training neuronaler Netze ($O(1)$ Rückwärtsdurchlauf) |

<!--en:295715f7-->
| **Rückwärtsdurchlauf (VJP)** | `jax.grad(loss)(w)` oder `jax.vjp(f, w)` | Gibt Aktivierungspuffer nach Berechnung von $\mathbf{J}^\top \mathbf{v}$ frei | Backpropagation eines skalaren Verlusts ($m = 1$) oder eines nachgelagerten Vektors $\mathbf{v}$ |

<!--en:06df609b-->
| **Zurücksetzen / Akkumulieren** | Keine: neues Array pro Aufruf | Legt fest, ob der neue Gradient zum letzten addiert wird oder ihn ersetzt | Gradienten über Mikrobatches akkumulieren oder neuen Optimierungsschritt beginnen |

<!--en:7ae527b9-->
| **Teilgraph abtrennen** | `jax.lax.stop_gradient(u)` | Konstante Sicht ohne Kopie, trennt die Rückwärtskante | Zielnetze beim Reinforcement Learning, Lehrerzweig von BYOL/SimSiam, GAN-Schritte |

<!--en:cfb2ff22-->
| **Inferenz ohne Gradienten** | Außerhalb von `jax.grad` | Kein Graphaufbau und keine Zwischenspeicherung von Aktivierungen ($3\text{--}10\times$ weniger GPU-Speicher) | Validierungsschleifen, Inferenz im Einsatz, Parameteraktualisierungen |

<!--en:0768ef12-->
::: {.slide title="Überblick: Autograd-Operationen, Speicherbedarf und Anwendungen" only="mxnet"}

<!--en:c5a222f8-4-->
[Das Wichtigste]{.kicker}

<!--en:1d2b114b-4-->
| Operation / Begriff | API | Graph und Speicher | Anwendung im Deep Learning |

<!--en:4b2fa311-->
| **Aktivieren / Aufzeichnen** | `w.attach_grad()` + `autograd.record()` | Speichert Vorwärtsaktivierungen im GPU-Speicher für VJPs im Rückwärtsdurchlauf | Übliches Training neuronaler Netze ($O(1)$ Rückwärtsdurchlauf) |

<!--en:10f45677-2-->
| **Rückwärtsdurchlauf (VJP)** | `L.backward()` oder `u.backward(v)` | Gibt Aktivierungspuffer nach Berechnung von $\mathbf{J}^\top \mathbf{v}$ frei | Backpropagation eines skalaren Verlusts ($m = 1$) oder eines nachgelagerten Vektors $\mathbf{v}$ |

<!--en:242c8227-->
| **Zurücksetzen / Akkumulieren** | Keine: überschreibt standardmäßig `w.grad` | Legt fest, ob der neue Gradient zum letzten addiert wird oder ihn ersetzt | Gradienten über Mikrobatches akkumulieren oder neuen Optimierungsschritt beginnen |

<!--en:9e0f8049-2-->
| **Teilgraph abtrennen** | `u.detach()` | Konstante Sicht ohne Kopie, trennt die Rückwärtskante | Zielnetze beim Reinforcement Learning, Lehrerzweig von BYOL/SimSiam, GAN-Schritte |

<!--en:0bff2f9c-->
| **Inferenz ohne Gradienten** | Außerhalb von `autograd.record()` | Kein Graphaufbau und keine Zwischenspeicherung von Aktivierungen ($3\text{--}10\times$ weniger GPU-Speicher) | Validierungsschleifen, Inferenz im Einsatz, Parameteraktualisierungen |
