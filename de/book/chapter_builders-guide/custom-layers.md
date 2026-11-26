<!--en:6c8090c7-->
# Eigene Schichten und Funktionen

<!--en:113b57c0-->
Die Bibliothek eines Deep-Learning-Frameworks bietet die Standardschichten, die sich in der Praxis bereits etabliert haben, etwa lineare Projektionen, Aktivierungsfunktionen und Dropout. Die Forschung ist jedoch schneller als der Veröffentlichungszyklus der Bibliotheken. Bis eine neu vorgeschlagene Schicht in einer offiziellen Bibliothek erscheint, können Jahre vergehen. Ein Beispiel ist *RMSNorm*, eine Normalisierungsschicht, die jeden Aktivierungsvektor auf das quadratische Mittel (engl. root mean square, RMS) eins skaliert. Sie wurde 2019 vorgestellt :cite:`Zhang.Sennrich.2019`, fand aber erst Jahre später Eingang in die Standardbibliotheken für Schichten. Bis dahin musste jedes Projekt, das RMSNorm nutzen wollte, die Schicht selbst implementieren oder eine Version von Drittanbietern importieren. Damit sich eine eigene Schicht (engl. custom layer) wie eine eingebaute verhält, muss sie sich nahtlos ins Framework einfügen. Der Optimierer muss ihre Parameter finden. Ihr Zustand muss sich in Checkpoints serialisieren und auf GPUs übertragen lassen. Außerdem muss sie sich in sequenziellen und eigenen Containern mit eingebauten Schichten kombinieren lassen.

<!--en:9beb757b-->
Manchmal liegt die Schwierigkeit in der Ableitung einer Vorwärtsoperation. Um ein Modell mit 8 Milliarden Parametern (:numref:`sec_model_construction`) in knappem Speicher unterzubringen, quantisiert man die Gewichte häufig auf 4-Bit-Ganzzahlen. Dadurch sinkt der Speicherbedarf auf nur 4 GB, ein Viertel des Bedarfs bei 16-Bit-Gleitkommazahlen. Für die Quantisierung eines fertigen Modells brauchen wir keine Gradienten. Verwenden wir jedoch beim *Training* quantisierte Gewichte im Vorwärtsdurchlauf (engl. forward pass), brauchen wir Gradienten. Hier steht uns die Rundung im Weg. Ihre Ableitung ist fast überall null. Jedes im Vorwärtsdurchlauf quantisierte Gewicht erhält daher den Gradienten null und lernt nicht mehr.

<!--en:108a220e-->
In diesem Abschnitt entwickeln wir beide Arten von Erweiterungen. Bei der Optimierung, beim Checkpointing und beim Gerätewechsel durchläuft das Framework jeweils den registrierten Zustand im Modulbaum (:numref:`sec_model_construction`). Damit eine eigene Schicht automatisch an all diesen Operationen teilnimmt, genügt es, vom Basismodul zu erben und unsere Tensoren zu registrieren. Wir betrachten nacheinander vier Beispiele. Auf eine zustandslose Schicht folgen `RMSNorm` mit einem gelernten Parameter, eine Schicht für zufällige Merkmale (engl. random features) mit festen, nicht trainierbaren Puffern und ein eigener Rückwärtsdurchlauf (engl. backward pass). Dieser *Straight-Through-Schätzer* (engl. straight-through estimator) leitet Gradienten durch die Rundung hindurch.

<!--en:fce2e222-->
## Schichten ohne Parameter

<!--en:3d5a6aa1-->
Wir beginnen mit der einfachsten Form einer eigenen Schicht. Sie speichert keinerlei Zustand. `CenteredLayer` zieht vom Eingabetensor dessen globalen Mittelwert ab. Dazu leiten wir die Schicht vom Basismodul des Frameworks ab und implementieren nur den Vorwärtsdurchlauf.

<!--en:4ea1c8fb-->
Der Vektor `[1, 2, 3, 4, 5]` hat den Mittelwert 3. Daher verschiebt `CenteredLayer` seine Werte so, dass sie von $-2$ bis $2$ reichen:

<!--en:5939ba1c-->
Da `CenteredLayer` dieselbe Modulschnittstelle hat wie eingebaute Schichten, können wir sie in einem sequenziellen Container direkt hinter einer linearen Schicht einsetzen, um deren Ausgaben zu zentrieren. Der Mittelwert sollte dann null sein. Bleibt bei der Gleitkommarechnung ein kleiner Rest von etwa $10^{-8}$ oder weniger, liegt das am üblichen Rundungsfehler.

<!--en:aa9687f2-->
Die benachbarte Schicht `Dense` hat Parameter. Deshalb müssen wir vor dem ersten Aufruf des Containers `initialize()` ausführen.

<!--en:23fc7cf7-->
NNX erzeugt die Parameter von `Linear` im Konstruktor. Deshalb geben wir dort sowohl die Eingabe- als auch die Ausgabebreite an.

<!--en:102c9ddf-->
## Schichten mit Parametern

<!--en:22f2a493-->
In :numref:`sec_numerical_stability` haben wir gesehen, wie sich die Größenordnung der Aktivierungen mit zunehmender Tiefe je nach Schichtgewichten verschieben kann. Eine *Normalisierungsschicht* beseitigt diese Abhängigkeit. Sie bringt jeden Aktivierungsvektor auf eine vorgegebene Größe, die das Netz anschließend über einen gelernten Skalenparameter anpassen kann. Ab :numref:`sec_batch_norm` untersuchen wir diese Familie genauer. Hier implementieren wir ihren einfachsten Vertreter, *RMSNorm*.

<!--en:bb93ce31-->
### RMSNorm

<!--en:96e0926e-->
Ein Eingabevektor $\mathbf{x} \in \mathbb{R}^d$ habe das quadratische Mittel $\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$. RMSNorm teilt $\mathbf{x}$ durch sein quadratisches Mittel und multipliziert das Ergebnis elementweise mit einem Vektor gelernter *Verstärkungsfaktoren* (engl. gain) $\mathbf{g} \in \mathbb{R}^d$:

<!--en:7cb19041-->
Eine kleine Konstante $\epsilon$ (etwa $10^{-6}$) verhindert eine Division durch null. Initialisieren wir $\mathbf{g}$ mit $\mathbf{1}$, liegt das quadratische Mittel der Ausgabe unabhängig von der Skalierung der Eingabe stets nahe bei 1. Für $\mathbf{x} = (3, 4)$ ist das mittlere Quadrat beispielsweise $(9 + 16)/2 = 12.5$ und das quadratische Mittel $3.536$. Das ergibt $(0.849, 1.131)$. Skalieren wir die Eingabe auf $(300, 400)$, erhalten wir bei der angegebenen Genauigkeit dieselbe Ausgabe. Geometrisch betrachtet (:numref:`fig_bg_rmsnorm`) projiziert RMSNorm den Vektor $\mathbf{x}$ unter Beibehaltung seiner Richtung auf die Sphäre mit Radius $\sqrt{d}$. Anschließend skaliert $\mathbf{g}$ jede Koordinatenachse einzeln.

<!--en:a452fff6-->
![RMSNorm in zwei Dimensionen. Links bewegt sich bei Verstärkungsfaktor 1 jede Eingabe (blau) auf ihrem Strahl vom Ursprung zum Kreis mit Radius $\sqrt{2}$ (orange). Zwei Eingaben auf demselben Strahl mit den Längen 0.6 und 2.9 landen im selben Punkt. Rechts verformt der Verstärkungsvektor $\mathbf{g} = (1.5, 0.5)$ den Kreis zu einer Ellipse mit den Halbachsen $1.5\sqrt{2}$ und $0.5\sqrt{2}$.](../img/bg-rmsnorm.svg)

<!--en:9133da9f-->
RMSNorm vereinfacht die *Schichtnormalisierung* (engl. layer normalization) :cite:`Ba.Kiros.Hinton.2016`. Diese zieht zusätzlich für jedes Beispiel den Mittelwert ab (wie eine zeilenweise angewandte `CenteredLayer`) und addiert nach der Skalierung einen gelernten Bias. Zhang und Sennrich zeigten, dass der Verzicht auf die Zentrierung am Mittelwert Rechenaufwand spart und die Modellgüte nur selten beeinträchtigt. Deshalb verwenden viele moderne Sprachmodelle durchgängig RMSNorm, darunter Llama 3 8B :cite:`Grattafiori.Dubey.Jauhri.ea.2024` (:numref:`sec_model_construction`).

<!--en:b43cd0c3-->
Da wir $\mathbf{g}$ trainieren müssen, registriert die Schicht den Vektor als Parameter (:numref:`sec_parameters`). Der übrige Code setzt direkt die Formel um.

<!--en:6f6bb63d-->
In Gluon registrieren wir einen `gluon.Parameter`, indem wir ihn einem Attribut zuweisen. Bei der Deklaration geben wir Namen, Form und Initialisierer an. `initialize()` erzeugt das zugehörige Array. Im Vorwärtsdurchlauf ruft `data(X.device)` die Kopie auf dem Gerät der Eingabe ab.

<!--en:d0bec796-->
In PyTorch registrieren wir einen Tensor, indem wir ihn in `nn.Parameter` verpacken und einem Attribut zuweisen.

<!--en:853b1c21-->
In Keras erzeugt und registriert `add_weight` eine Variable. Wir rufen die Funktion in der Methode `build` auf. Diese wird beim ersten Aufruf ausgeführt, sobald die Form der Eingabe feststeht. Deshalb müssen wir die Breite unserer Schicht nicht angeben.

<!--en:c9e19d33-->
In NNX wird ein Array zum Parameter des Moduls, wenn wir es in `nnx.Param` verpacken und einem Attribut zuweisen.

<!--en:5245170c-->
Wir testen `RMSNorm` mit $(3, 4)$ und mit dem um den Faktor 100 skalierten Vektor $(300, 400)$. Beide Zeilen sollten auf $(0.849, 1.131)$ abgebildet werden:

<!--en:4e246375-->
### Registrierung

<!--en:d2329f3e-->
Warum implementieren wir RMSNorm als Unterklasse eines Moduls? Warum rufen wir keine gewöhnliche Funktion mit einem separaten Verstärkungsvektor auf? Durch die Registrierung erhält unsere Schicht automatisch alle vier Fähigkeiten eingebauter Schichten. Optimierer finden ihre Parameter, Container können sie mit anderen Schichten kombinieren, ihr Zustand lässt sich in Checkpoints serialisieren und auf andere Geräte übertragen (:numref:`sec_model_construction`, :numref:`sec_parameters`). Wir prüfen das Zusammenspiel, indem wir `RMSNorm` zwischen zwei lineare Schichten setzen. Nach einem Gradientenschritt weichen die Verstärkungsfaktoren von eins ab. Wir kopieren den gesamten Zustand in ein neu initialisiertes Modell. Beide Modelle sollten nun identische Ausgaben liefern (in :numref:`sec_read_write` erweitern wir das Verfahren auf Checkpoints auf der Festplatte).

<!--en:f755d2ea-->
Tatsächlich ändern sich die anfangs auf eins gesetzten Verstärkungsfaktoren, und die Kopie liefert bitgenau dieselben Ausgaben wie das trainierte Modell. Dazu brauchen wir in `RMSNorm` über die Registrierung von `self.gain` hinaus keinen zusätzlichen Code.

<!--en:f0a7369f-->
### Vergleich mit den Implementierungen der Frameworks

<!--en:6765297f-->
Die bestandenen Integrationstests zeigen, dass unsere Schicht korrekt mit dem Framework zusammenarbeitet. Ob die Berechnung selbst stimmt, bleibt offen. Bietet die Bibliothek eine entsprechende Schicht an, können wir die Ausgaben beider Implementierungen mit nichttrivialen Gewichten vergleichen. So erhalten wir unmittelbar eine numerische Plausibilitätsprüfung.

<!--en:8e4bad43-->
Gluon bietet `nn.LayerNorm`, aber keine eingebaute RMSNorm-Schicht. Deshalb verwenden wir unsere eigene Implementierung `RMSNorm`.

<!--en:f68182cb-->
Wir setzen dieselben nichttrivialen Verstärkungsfaktoren (`linspace(0.5, 1.5, 8)`) in unserer eigenen Schicht `RMSNorm` und in der eingebauten Schicht `nn.RMSNorm` und prüfen, ob ihre Ausgaben übereinstimmen:

<!--en:29091cf2-->
Wir setzen dieselben nichttrivialen Verstärkungsfaktoren (`linspace(0.5, 1.5, 8)`) in unserer eigenen Schicht `RMSNorm` und in der eingebauten Schicht `tf.keras.layers.RMSNormalization` und prüfen, ob ihre Ausgaben übereinstimmen:

<!--en:25f896ce-->
Wir setzen dieselben nichttrivialen Verstärkungsfaktoren (`linspace(0.5, 1.5, 8)`) in unserer eigenen Schicht `RMSNorm` und in der eingebauten Schicht `nnx.RMSNorm` und prüfen, ob ihre Ausgaben übereinstimmen:

<!--en:ae3304f5-->
Erfüllt eine eingebaute Schicht Ihre Anforderungen, verwenden Sie diese im Produktivbetrieb. Das erleichtert die Wartung, und zusammengefasste Kernel beschleunigen die Berechnung. Eigene Schichten lohnen sich für neue Operationen oder Varianten, die in der Bibliothek fehlen.

<!--en:27ce4ae6-->
## Schichten mit festem Zustand

<!--en:03ff3ec5-->
Manche Schichten benötigen Arrays, die beim Aufbau des Modells einmal entstehen und während seiner gesamten Lebensdauer unverändert bleiben müssen. Der Gradientenabstieg lernt sie nicht, und die Schicht berechnet sie auch nicht bei jedem Aufruf neu. Ein klassisches Beispiel ist eine Schicht für *zufällige Merkmale* :cite:`Rahimi.Recht.2007`. Sie projiziert einen Minibatch $\mathbf{X}$ mit einer festen Matrix $\mathbf{W} \in \mathbb{R}^{d \times D}$ aus standardnormalverteilten Einträgen. Dann verschiebt sie das Ergebnis um zufällige Phasen $\mathbf{b} \in \mathbb{R}^D$ aus der Gleichverteilung auf $[0, 2\pi]$ und wendet elementweise den Kosinus an:

<!--en:4a230b3d-->
Jede Zeile von $\mathbf{Z}$ liefert $D$ feste nichtlineare Merkmale. Bemerkenswerterweise genügt eine einzige trainierbare lineare Schicht über diesen Merkmalen, um vielfältige nichtlineare Funktionen zu approximieren, selbst wenn sich $\mathbf{W}$ und $\mathbf{b}$ nie ändern.

<!--en:a8ff0aeb-->
Wie die fortlaufend aktualisierten Statistiken in `Standardizer` (:numref:`subsec_params_buffers`) müssen $\mathbf{W}$ und $\mathbf{b}$ bei der Optimierung unverändert bleiben, aber zusammen mit dem Modell in Checkpoints gespeichert und auf andere Geräte übertragen werden. Deshalb registrieren wir sie als Puffer:

<!--en:a2a5cc02-->
Mit `RandomFeatures` approximieren wir nun $y = \sin(3x)$ an 100 gleichmäßig verteilten Punkten im Intervall $[-2, 2]$. Dazu geben wir 500 zufällige Merkmale an eine einzige lineare Schicht weiter, die wir mit Adam trainieren. Da $\mathbf{W}$ und $\mathbf{b}$ als Puffer registriert sind, aktualisiert der Optimierer nur die lineare Ausgabeschicht, obwohl wir ihm das ganze Modell übergeben:

<!--en:09f4acb5-->
Die gelernte Linearkombination folgt $\sin(3x)$ so genau, dass der mittlere quadratische Fehler (engl. mean squared error, MSE) unter 0.01 fällt. Eine konstante Vorhersage von null ergibt dagegen 0.5.

<!--en:1ccef5ce-->
Was passiert, wenn ein Checkpoint nur die trainierte lineare Ausgabeschicht speichert? Ein neu erzeugtes Modell zieht ganz andere Werte für $\mathbf{W}$ und $\mathbf{b}$. Die lineare Schicht erhält dadurch Merkmale, die sie beim Training nie gesehen hat. Wir vergleichen die Ergebnisse, wenn wir nur die lineare Ausgabeschicht oder den gesamten Zustand einschließlich der Puffer wiederherstellen:

<!--en:6fd9c038-->
Laden wir nur die Gewichte der linearen Schicht, steigt der MSE um mehr als zwei Größenordnungen. Damit ist die Vorhersage nicht besser als eine konstante Null. Mit dem vollständigen Zustand einschließlich der Puffer erhalten wir dagegen exakt die trainierte Funktion zurück.

<!--en:6ac608ef-->
Denselben Puffermechanismus verwenden wir in späteren Kapiteln für nicht trainierbare Tabellen und für fortlaufend aktualisierte Statistiken, etwa für Mittelwert und Varianz bei der Batchnormalisierung (:numref:`sec_batch_norm`).

<!--en:3018e5db-->
## Funktionen mit eigenen Gradienten

<!--en:ef8a0dc2-->
Bisher haben wir den Vorwärtsdurchlauf implementiert und den Rückwärtsdurchlauf dem automatischen Differenzieren (engl. automatic differentiation) überlassen. Autograd leitet jede einzelne Operation exakt ab. Genau das steht uns gelegentlich im Weg. Betrachten wir erneut die 4-Bit-Quantisierung vom Anfang des Abschnitts. Die Funktion, die im Vorwärtsdurchlauf die Gewichte auf ganze Zahlen rundet, ist stückweise konstant. Ihre Ableitung ist also fast überall null. Damit erhält jedes Gewicht, dessen Wert einen Rundungsschritt durchläuft, den Gradienten null:

<!--en:c7d93d64-->
TensorFlow stuft das Runden als nicht differenzierbar ein. Ohne das Argument `unconnected_gradients` gibt das Band daher `None` zurück. Mit dem Argument fordern wir die Nullen an, die sich beim Ableiten ergeben.

<!--en:3dc0468a-->
Mathematisch ist der Gradient null korrekt. In der Praxis bringt er das Lernen zum Stillstand. Der *Straight-Through-Schätzer* (STE) :cite:`Bengio.Leonard.Courville.2013` umgeht das Problem, indem er die beiden Richtungen entkoppelt. Der Vorwärtsdurchlauf rundet weiterhin auf ganze Zahlen. Der Rückwärtsdurchlauf behandelt das Runden wie die Identitätsfunktion und reicht den eingehenden Gradienten unverändert weiter (Ableitung 1). Das ist bewusst *nicht* die tatsächliche Ableitung. In der Praxis funktioniert die Näherung gut, weil die Treppenfunktion $\textrm{round}(x)$ um höchstens $1/2$ von der Identität abweicht (:numref:`fig_bg_ste`). Autograd kann aus der Vorwärtsfunktion keine abweichende Ersatzableitung herleiten. Deshalb geben wir die Regel für den Rückwärtsdurchlauf ausdrücklich an.

<!--en:e5146107-->
![Der Straight-Through-Schätzer. Im Vorwärtsdurchlauf liegt die Treppenfunktion round(x) nahe an der Identität (gestrichelt). Im Rückwärtsdurchlauf ist die tatsächliche Ableitung der Rundung fast überall null. Der Schätzer verwendet stattdessen die Ableitung der Identität, also 1.](../img/bg-ste.svg)

<!--en:2ee77b5e-->
In Gluon leiten wir eine Unterklasse von `autograd.Function` ab und definieren für jede Richtung eine Methode:

<!--en:c98f8612-->
In PyTorch leiten wir eine Unterklasse von `torch.autograd.Function` ab und definieren für jede Richtung eine statische Methode:

<!--en:49b5cb73-->
In TensorFlow verwenden wir den Dekorator `@tf.custom_gradient`. Die dekorierte Funktion gibt ihre Ausgabe zusammen mit ihrer Regel für den Rückwärtsdurchlauf zurück:

<!--en:454a60d8-->
In JAX verwenden wir `jax.custom_vjp`, um einer gewöhnlichen Funktion eine selbst geschriebene Regel für den Rückwärtsdurchlauf zuzuordnen. Diese berechnet ein Vektor-Jacobi-Produkt:

<!--en:2b660eb1-->
Erzeugen Sie bei jedem Aufruf eine neue Instanz mit `RoundSTE()`. Eine Instanz von `Function` lässt sich nur einmal verwenden. Benötigt `backward` Tensoren aus dem Vorwärtsdurchlauf, speichern Sie diese mit `self.save_for_backward`.

<!--en:43efd269-->
Verwenden Sie stets `RoundSTE.apply(X)`. Ein direkter Aufruf von `forward` umgeht die Registrierung bei Autograd. Benötigt `backward` Tensoren aus dem Vorwärtsdurchlauf, speichern Sie diese mit `ctx.save_for_backward`.

<!--en:2b442889-->
Die mit `@tf.custom_gradient` dekorierte Funktion gibt `(output, grad_fn)` zurück. Die innere Funktion `grad(upstream)` kann als Closure auf alle Tensoren des Vorwärtsdurchlaufs im umgebenden Gültigkeitsbereich zugreifen.

<!--en:68796c57-->
`round_ste_fwd` gibt `(output, residuals)` zurück (als zweiten Eintrag `None`, wenn keine Werte aus dem Vorwärtsdurchlauf benötigt werden). `round_ste_bwd(residuals, grad_output)` gibt passend zu den Eingabeargumenten das einelementige Tupel `(grad_output,)` zurück.

<!--en:03a6e364-->
Um unsere eigene Rückwärtsregel zu prüfen, multiplizieren wir die gerundeten Gewichte elementweise mit $(1, 2, 3)$ und summieren die Produkte. Der Straight-Through-Schätzer sollte unmittelbar $(1, 2, 3)$ als Gradienten bezüglich $\mathbf{w}$ liefern:

<!--en:0f66e764-->
Der Gradient gelangt unverändert zurück zu $\mathbf{w}$, als wäre die Rundung die Identität. Der Vorwärtsdurchlauf rechnet weiterhin mit ganzzahlig gerundeten Gewichten. Kann der Gradientenabstieg mit dieser Ersatzableitung tatsächlich lernen? Wir trainieren ein lineares Modell mit acht Gewichten, die wir im Vorwärtsdurchlauf auf ganze Zahlen runden. Die synthetischen Daten erzeugen wir mit Gewichten aus der Gleichverteilung auf $[-7, 7]$ (dem symmetrischen Teil des vorzeichenbehafteten 4-Bit-Bereichs $[-8, 7]$). Ausgehend von $\mathbf{w} = \mathbf{0}$ vergleichen wir gewöhnliches Runden mit dem Straight-Through-Schätzer. Zum Vergleich betrachten wir auch den Anfangsverlust und den Verlust bei gerundeten wahren Gewichten. Letzterer dient als Referenzwert, denn das separate Runden jedes Gewichts liefert keine optimale Anpassung mit ganzzahligen Gewichten:

<!--en:14b58c45-->
Beim gewöhnlichen Runden ist jeder Gradient null, und der Verlust bleibt auf seinem Anfangswert. Mit dem Straight-Through-Schätzer sinkt der Verlust von einigen Dutzend oder Hundert auf 1 oder 2. Damit liegt er nur um einen kleinen Faktor über dem Verlust bei gerundeten wahren Gewichten. Mit einem bewusst nur näherungsweise bestimmten Ersatzgradienten lässt sich also ein quantisiertes Modell trainieren.

<!--en:61fd37c6-->
Die Identität ist nur eine mögliche Ersatzfunktion. Ein vollständiger 4-Bit-Quantisierer begrenzt seine Eingaben zusätzlich auf $[-8, 7]$. In der Praxis reichen Implementierungen Gradienten daher oft nur innerhalb des darstellbaren Bereichs weiter und setzen sie außerhalb, wo Sättigung eintritt, auf null (siehe Übungen). Welche Ersatzfunktion sich eignet, ist eine Modellierungsentscheidung und hängt von der zu approximierenden Operation ab.

<!--en:c96fb68c-->
Wenn die Ersatzfunktion die Identität *ist*, genügt eine einzige Codezeile. Eine eigene Autograd-Funktion erübrigt sich dann. Diese Variante ist im Produktivcode am häufigsten anzutreffen. Wir verwenden einen Operator für den *Gradientenstopp* (engl. stop-gradient). Im Vorwärtsdurchlauf wirkt er als Identität, im Rückwärtsdurchlauf hat er die Ableitung null. Damit ergibt $X + \textrm{stop}(\textrm{round}(X) - X)$ im Vorwärtsdurchlauf den Wert $\textrm{round}(X)$. Im Rückwärtsdurchlauf trägt nur das erste $X$ zur Ableitung bei. Für andere Ersatzfunktionen, etwa bei Quantisierern mit Wertebegrenzung, brauchen wir weiterhin eine eigene Rückwärtsregel. Beim einfachen Straight-Through-Runden liefern beide Definitionen exakt dieselben Werte und Gradienten:

<!--en:ce7c8011-->
In MXNet bewirkt `npx.stop_gradient` den Gradientenstopp.

<!--en:1f97b86f-->
In PyTorch übernimmt die Methode `detach()` den Gradientenstopp. Sie gibt einen Tensor mit denselben Werten zurück, den Autograd als Konstante behandelt.

<!--en:388b195a-->
In TensorFlow bewirkt `tf.stop_gradient` den Gradientenstopp.

<!--en:6b775ebd-->
In JAX bewirkt `jax.lax.stop_gradient` den Gradientenstopp.

<!--en:78a559be-->
Eine selbst geschriebene Rückwärtsregel kann fehlerhaft sein, ohne dass dies im Vorwärtsdurchlauf auffällt. Der übliche Test ist die *Gradientenprüfung* (engl. gradient check). Wir verändern jede Eingabekomponente geringfügig und messen, wie stark sich die Ausgabe ändert. Die mit dieser endlichen Differenz bestimmte Steigung vergleichen wir mit dem Gradienten, den die Regel liefert. Wir prüfen eine glatte eingebaute Funktion und unsere Straight-Through-Regel:

<!--en:c9a34505-->
Die eingebaute Funktion besteht die Prüfung. Der Straight-Through-Schätzer fällt erwartungsgemäß durch. Im Vorwärtsdurchlauf haben wir eine Treppenfunktion. Auf jeder Stufe ergibt die endliche Differenz die Steigung null, während unsere Rückwärtsregel die Steigung eins liefert. Eine Gradientenprüfung beantwortet also die Frage, ob eine Regel die Ableitung ihres Vorwärtsdurchlaufs berechnet. Ob die Regel nützlich ist, lässt sich so nicht beurteilen. Prüfen Sie auf diese Weise jede Regel, die exakt sein soll, etwa die Wertebegrenzung in den Übungen.

<!--en:f0969e75-->
Mit einem eigenen Gradienten überschreiben wir die Ableitung eines Ausdrucks, der aus vorhandenen Tensoroperationen besteht. Eigene GPU-Kernel von Grund auf zu schreiben, ist ein gesondertes Thema der Systemprogrammierung und geht über diesen Abschnitt hinaus. In :numref:`chap_performance` untersuchen wir, wie sich mit den vorhandenen Grundoperationen der Frameworks die höchstmögliche Geschwindigkeit erzielen lässt.

<!--en:3ba34a95-->
Bei Schichten wie `RMSNorm` bleibt eine wichtige Frage offen. Wie sicher ist ihre Gleitkommarechnung? Wenn wir Aktivierungen quadrieren und die Quadrate summieren, kommt es bei 16-Bit-Formaten leicht zu einem Überlauf. In :numref:`sec_numerics` untersuchen wir Zahlenformate und das Training mit gemischter Genauigkeit genauer.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:732f25fb-->
Eine eigene Schicht ist eine Unterklasse des Basismoduls des Frameworks, die einen Vorwärtsdurchlauf implementiert. Sobald wir ihre Tensoren registrieren, kümmert sich das Framework automatisch um Optimierung, Einbindung in Container, Checkpointing und Gerätezuordnung. Gelernte Gewichte registrieren wir als Parameter, etwa die Verstärkungsfaktoren von `RMSNorm`. Feste, nicht trainierbare Tensoren wie die Zufallsprojektionen in `RandomFeatures` registrieren wir als Puffer. So enthalten Checkpoints auch diese Tensoren, während der Optimierer nur die Parameter sieht. Liefert die tatsächliche Ableitung einer Operation wie der Rundung kein brauchbares Lernsignal, ermöglicht eine eigene Rückwärtsregel das Training. Ein Beispiel ist der Straight-Through-Schätzer mit seiner Kurzform über den Gradientenstopp. Erfüllt eine eingebaute Schicht Ihre Anforderungen, bevorzugen Sie diese. Das erleichtert die Wartung und nutzt den Geschwindigkeitsvorteil zusammengefasster Kernel.

<!--en:2294243a-->
## Übungen

<!--en:f425b8b5-->
1. [code] **Skaleninvarianz von RMSNorm.** ● Setzen Sie $\epsilon = 0$.
    1. Zeigen Sie, dass $\textrm{RMSNorm}(c\mathbf{x}) = \textrm{RMSNorm}(\mathbf{x})$ für jedes $c > 0$ gilt.
    1. Differenzieren Sie beide Seiten nach $c$ an der Stelle $c = 1$. Folgern Sie, dass die Jacobi-Matrix $\mathbf{J}$ von RMSNorm an der Stelle $\mathbf{x}$ die Gleichung $\mathbf{J}\mathbf{x} = \mathbf{0}$ erfüllt.
    1. Folgern Sie, dass der Gradient jedes Verlusts bezüglich der Eingabe $\mathbf{x}$ von RMSNorm orthogonal zu $\mathbf{x}$ ist. Prüfen Sie dies durch automatisches Differenzieren für ein zufälliges $\mathbf{x}$ und eine zufällige lineare Verlustfunktion. Wie verändert ein kleiner Gradientenschritt die Länge von $\mathbf{x}$?

<!--en:5f38661c-->
2. [code] **Bias für RMSNorm.** Fügen Sie einen optionalen gelernten Bias $\mathbf{b}$ zu `RMSNorm` hinzu. Er verschiebt die Ausgabe nach der Verstärkung und stellt damit einen Teil der Schichtnormalisierung wieder her. Prüfen Sie, ob die mit `save_parameters` geschriebene Datei für ein Modell mit dieser Schicht um den erwarteten Eintrag wächst. Setzen Sie die Schicht dann direkt vor eine lineare Schicht mit Gewicht $\mathbf{W}$ und Bias $\mathbf{c}$. Zeigen Sie, dass sich $\mathbf{b}$ in $\mathbf{c}$ einbeziehen lässt, ohne die Ausgaben zu verändern. Wie lautet der zusammengefasste Bias?

<!--en:6813322e-->
2. [code] **Bias für RMSNorm.** Fügen Sie einen optionalen gelernten Bias $\mathbf{b}$ zu `RMSNorm` hinzu. Er verschiebt die Ausgabe nach der Verstärkung und stellt damit einen Teil der Schichtnormalisierung wieder her. Prüfen Sie, ob das Zustandswörterbuch eines Modells mit dieser Schicht um den erwarteten Eintrag wächst. Setzen Sie die Schicht dann direkt vor eine lineare Schicht mit Gewicht $\mathbf{W}$ und Bias $\mathbf{c}$. Zeigen Sie, dass sich $\mathbf{b}$ in $\mathbf{c}$ einbeziehen lässt, ohne die Ausgaben zu verändern. Wie lautet der zusammengefasste Bias?

<!--en:463044e2-->
2. [code] **Bias für RMSNorm.** Fügen Sie einen optionalen gelernten Bias $\mathbf{b}$ zu `RMSNorm` hinzu. Er verschiebt die Ausgabe nach der Verstärkung und stellt damit einen Teil der Schichtnormalisierung wieder her. Prüfen Sie, ob `get_weights` für ein Modell mit dieser Schicht den erwarteten zusätzlichen Eintrag liefert. Setzen Sie die Schicht dann direkt vor eine lineare Schicht mit Gewicht $\mathbf{W}$ und Bias $\mathbf{c}$. Zeigen Sie, dass sich $\mathbf{b}$ in $\mathbf{c}$ einbeziehen lässt, ohne die Ausgaben zu verändern. Wie lautet der zusammengefasste Bias?

<!--en:b70bb4ba-->
2. [code] **Bias für RMSNorm.** Fügen Sie einen optionalen gelernten Bias $\mathbf{b}$ zu `RMSNorm` hinzu. Er verschiebt die Ausgabe nach der Verstärkung und stellt damit einen Teil der Schichtnormalisierung wieder her. Prüfen Sie, ob der Parameterzustand eines Modells mit dieser Schicht um den erwarteten Eintrag wächst. Setzen Sie die Schicht dann direkt vor eine lineare Schicht mit Gewicht $\mathbf{W}$ und Bias $\mathbf{c}$. Zeigen Sie, dass sich $\mathbf{b}$ in $\mathbf{c}$ einbeziehen lässt, ohne die Ausgaben zu verändern. Wie lautet der zusammengefasste Bias?

<!--en:b0c1b606-->
3. [code] **Approximation eines Kernels durch zufällige Merkmale.** Der Vektor $\mathbf{w}$ habe unabhängige standardnormalverteilte Einträge, und $b$ sei auf $[0, 2\pi]$ gleichverteilt.
    1. Zeigen Sie mithilfe von $\cos\alpha\cos\beta = \frac{1}{2}(\cos(\alpha - \beta) + \cos(\alpha + \beta))$ und $\mathbb{E}[\cos(\mathbf{w}^\top\mathbf{u})] = \exp(-\|\mathbf{u}\|^2/2)$, dass die Merkmale $\mathbf{z}(\mathbf{x})$ von `RandomFeatures` die Gleichung $\mathbb{E}[\mathbf{z}(\mathbf{x})^\top \mathbf{z}(\mathbf{x}')] = \exp(-\|\mathbf{x} - \mathbf{x}'\|^2/2)$ für jede Anzahl $D$ von Merkmalen erfüllen.
    1. Prüfen Sie dies numerisch für zwei Eingaben mit Abstand 1 und für $D = 10$, 100, 1000 und 10 000. Mitteln Sie den Fehler für jedes $D$ über mehrere Schichten. Wie nimmt der Fehler mit wachsendem $D$ ab?
    1. Wiederholen Sie die Approximation von $\sin(3x)$ aus diesem Abschnitt mit $D = 5$ und $D = 50$ Merkmalen. Verwenden Sie anschließend anstelle der zufälligen Merkmale ein lineares Modell, dessen einzige Eingabe $x$ ist. Welchen Fehler erreichen die Modelle jeweils und warum?

    *Nach :cite:`Rahimi.Recht.2007`.*

<!--en:9b142d6a-->
4. [code] **Wertebegrenzung mit eigenem Gradienten.** Implementieren Sie eine Unterklasse von `autograd.Function`. Der Vorwärtsdurchlauf berechnet `mnp.clip(X, lo, hi)`. Der Rückwärtsdurchlauf gibt den Gradienten nur dort weiter, wo die Eingabe strikt innerhalb des zulässigen Intervalls lag. Speichern Sie in `forward` den Tensor `X` mit `save_for_backward`. Vergleichen Sie die Gradienten mit denen der eingebauten Operation `mnp.clip` für Eingaben innerhalb und außerhalb des Intervalls sowie genau auf dem Rand. Welche Konvention verwendet die eingebaute Operation am Rand?

    *Nach dem PyTorch-Tutorial [PyTorch anhand von Beispielen lernen](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html) und dem JAX-Notebook [Eigene Ableitungsregeln](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).*

<!--en:3d848a36-->
4. [code] **Wertebegrenzung mit eigenem Gradienten.** Implementieren Sie eine Unterklasse von `autograd.Function`. Der Vorwärtsdurchlauf berechnet `X.clamp(lo, hi)`. Der Rückwärtsdurchlauf gibt den Gradienten nur dort weiter, wo die Eingabe strikt innerhalb des zulässigen Intervalls lag. Vergleichen Sie die Gradienten mit denen der eingebauten Operation `torch.clamp` für Eingaben innerhalb und außerhalb des Intervalls sowie genau auf dem Rand. Welche Konvention verwendet die eingebaute Operation am Rand?

    *Nach dem PyTorch-Tutorial [PyTorch anhand von Beispielen lernen](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html) und dem JAX-Notebook [Eigene Ableitungsregeln](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).*

<!--en:1c134af0-->
4. [code] **Wertebegrenzung mit eigenem Gradienten.** Implementieren Sie eine mit `@tf.custom_gradient` dekorierte Funktion. Der Vorwärtsdurchlauf berechnet `tf.clip_by_value(X, lo,
   hi)`. Der Rückwärtsdurchlauf gibt den Gradienten nur dort weiter, wo die Eingabe strikt innerhalb des zulässigen Intervalls lag. Die Closure für den Gradienten muss `X` aus dem umgebenden Gültigkeitsbereich übernehmen. Vergleichen Sie die Gradienten mit denen der eingebauten Operation `tf.clip_by_value` für Eingaben innerhalb und außerhalb des Intervalls sowie genau auf dem Rand. Welche Konvention verwendet die eingebaute Operation am Rand?

    *Nach dem PyTorch-Tutorial [PyTorch anhand von Beispielen lernen](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html) und dem JAX-Notebook [Eigene Ableitungsregeln](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).*

<!--en:a3f294e2-->
4. [code] **Wertebegrenzung mit eigenem Gradienten.** Implementieren Sie eine Funktion mit `custom_vjp`. Der Vorwärtsdurchlauf berechnet `jnp.clip(X, lo, hi)`. Der Rückwärtsdurchlauf gibt den Gradienten nur dort weiter, wo die Eingabe strikt innerhalb des zulässigen Intervalls lag. Die Vorwärtsregel muss `X` als Residuum speichern. Vergleichen Sie die Gradienten mit denen der eingebauten Operation `jnp.clip` für Eingaben innerhalb und außerhalb des Intervalls sowie genau auf dem Rand. Welche Konvention verwendet die eingebaute Operation am Rand?

    *Nach dem PyTorch-Tutorial [PyTorch anhand von Beispielen lernen](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html) und dem JAX-Notebook [Eigene Ableitungsregeln](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).*

<!--en:37513a18-->
[Diskussionen](https://d2l.discourse.group/t/36755)

<!--en:fe4d8bc5-->
Eigene Schichten und Funktionen<br>**Zustandslose Module · RMSNorm · RandomFeatures mit festen Puffern · eigener Rückwärtsdurchlauf und Straight-Through-Schätzer**

<!--en:ee5a19f5-->
::: {.slide title="Warum Bibliotheken in der Praxis erweitert werden müssen"}

<!--en:7a8cd49a-->
[Über die eingebauten Grundoperationen hinaus]{.kicker}

<!--en:15df0653-->
- **Verzögerte Aufnahme in Bibliotheken:** *RMSNorm* erschien 2019 und kam in Llama und Gopher zum Einsatz. Einige große Bibliotheken nahmen die Schicht erst Jahre später auf.

<!--en:defb959a-->
- **Freie Kombinierbarkeit:** Eigene Schichten müssen Optimierern ihre lernbaren Gewichte zugänglich machen, Puffer in Checkpoints serialisieren und automatisch das Gerät wechseln können.

<!--en:9ef66ea9-->
- **Nicht differenzierbare Vorwärtsoperationen:** Bei der 4-Bit-Quantisierung von Gewichten hat $\text{round}(w)$ fast überall die Ableitung null. Der Rückwärtsdurchlauf braucht daher eine eigene Ersatzableitung.

<!--en:44b0ab39-->
::: {.slide title="Zustandslose Schichten als Unterklassen des Basismoduls"}

<!--en:f65941c9-->
[Schichten ohne Parameter]{.kicker}

<!--en:8160ffcd-->
- Eine Schicht ohne persistenten Zustand (`CenteredLayer`) muss nur vom Basismodul erben und ihren Vorwärtsdurchlauf definieren:

<!--en:be4ebf48-->
::: {.slide title="Zustandslose Schichten in sequenziellen Containern"}

<!--en:8d1d4d16-->
[Integration in Container]{.kicker}

<!--en:f8be3352-->
- Da `CenteredLayer` die Modulschnittstelle erbt, passt sie direkt in eine sequenzielle Verarbeitungskette (`|output.mean()| ≲ 10⁻⁸`):

<!--en:a7e87a17-->
::: {.slide title="RMSNorm: Schichtnormalisierung mit dem quadratischen Mittel"}

<!--en:968fb60f-->
[Eigene Schichten mit Parametern]{.kicker}

<!--en:4ce1f4b3-->
- **Skaleninvarianz:** Sowohl $(3, 4)$ als auch $(300, 400)$ werden auf $(0.849, 1.131)$ abgebildet, wenn $\mathbf{g} = \mathbf{1}$ gilt.

<!--en:db13f357-->
- **Vereinfachte Schichtnormalisierung:** An allen 65 Normalisierungsstellen in Llama 3 8B entfallen die Zentrierung am Mittelwert und die Verschiebung um den Bias.

<!--en:3adbd33c-->
::: {.slide title="RMSNorm in zwei Dimensionen"}

<!--en:3a4e4bd7-->
[Richtungserhaltung und Achsenskalierung]{.kicker}

<!--en:eb7a5f86-->
- **Projektion auf eine Sphäre:** Die Division durch $\text{RMS}(\mathbf{x}) = \|\mathbf{x}\|_2 / \sqrt{d}$ bildet jeden Strahl vom Ursprung auf einen Punkt der Sphäre mit Radius $\sqrt{d}$ ab (links). Anschließend verformt $\mathbf{g}$ die Sphäre zu einem achsenparallelen Ellipsoid (rechts).

<!--en:2b4c5e4e-->
::: {.slide title="RMSNorm mit registriertem Verstärkungsparameter"}

<!--en:8bdb981e-->
[RMSNorm im Code]{.kicker}

<!--en:5c1ba3cc-->
::: {.slide title="Kombinierbarkeit durch Registrierung"}

<!--en:2c265d8f-->
[Optimierung und Serialisierung prüfen]{.kicker}

<!--en:67b158ea-->
- Nach einem SGD-Schritt weicht `gain` vom Anfangswert eins ab. Eine Kopie mit dem gespeicherten Zustand liefert bitgenau dieselben Ausgaben:

<!--en:a0bd0008-->
::: {.slide title="Vergleich einer eigenen Schicht mit der eingebauten Version" except="mxnet"}

<!--en:ff2af144-->
[Numerische Plausibilitätsprüfung]{.kicker}

<!--en:e44bd121-->
- Eigene Schichten mit nichttrivialen Gewichten (`linspace(0.5, 1.5, 8)`) gegen die eingebaute Version des Frameworks testen:

<!--en:6e946c52-->
::: {.slide title="Prüfung eigener Schichten ohne eingebaute Vergleichsversion" only="mxnet"}

<!--en:ff2af144-2-->
[Numerische Plausibilitätsprüfung]{.kicker}

<!--en:c8f365f4-->
- **Drei Prüfungen für jede neue Schicht:**
  1. **Von Hand nachrechnen:** Skaleninvarianz für $(3, 4)$ und $(300, 400) \mapsto (0.849, 1.131)$ bei einem Verstärkungsfaktor von eins prüfen.
  2. **Gradientenfluss:** Prüfen, ob 1 Optimierungsschritt jeden registrierten Parameter gegenüber seinem Anfangswert verändert.
  3. **Serialisierung:** Zustand speichern, in eine neue Kopie laden und prüfen, ob die Ausgaben exakt übereinstimmen.

<!--en:f534a3d6-->
::: {.slide title="Schichten mit festem Zustand: Zufällige Fourier-Merkmale"}

<!--en:6b0c034d-->
[Nicht trainierbare Puffer in eigenen Schichten]{.kicker}

<!--en:2c2d5101-->
- **Kernel-Approximation (Rahimi und Recht, 2007):** Die Projektion mit festen $\mathbf{W} \sim \mathcal{N}(0, 1)$ und $\mathbf{b} \sim \mathcal{U}(0, 2\pi)$ approximiert über Kosinusmerkmale den gaußschen RBF-Kernel. Damit kann eine lineare Ausgabeschicht nichtlineare Zielgrößen approximieren.

<!--en:f4949bc0-->
- **Warum $\mathbf{W}, \mathbf{b}$ als Puffer registriert sein müssen:** Der Optimierer darf sie nie verändern, Checkpoints müssen sie aber speichern.

<!--en:4d52d0dc-->
::: {.slide title="Implementierung von RandomFeatures mit registrierten Puffern"}

<!--en:02fa69c4-->
[Puffer im Code]{.kicker}

<!--en:47021609-->
`W` und `b` als nicht trainierbare Puffer registrieren:

<!--en:b54a8aa1-->
::: {.slide title="Approximation von sin(3x) mit 500 festen zufälligen Merkmalen"}

<!--en:c4cbf17a-->
[Lineare Regression mit nichtlinearen Merkmalen]{.kicker}

<!--en:74c99878-->
Eine einzige trainierbare lineare Schicht (`500 -> 1`) hinter `RandomFeatures(1, 500)` approximiert $y = \sin(3x)$ mit einem MSE unter $0.01$. Adam lässt die Puffer für die Zufallsprojektion unverändert:

<!--en:1845c3c9-->
::: {.slide title="Fehler beim Speichern: Die lineare Ausgabeschicht allein genügt nicht"}

<!--en:95028879-->
[Warum Checkpoints die Puffer enthalten müssen]{.kicker}

<!--en:7a0d7179-->
- Wird nur die lineare Ausgabeschicht in ein neues Modell mit einer `RandomFeatures`-Schicht und anderen Zufallswerten $\mathbf{W}', \mathbf{b}'$ geladen, ist der Fehler mehr als $100\times$ so groß. Der vollständige Zustand stellt die trainierte Funktion wieder her:

<!--en:6d5ffbee-->
::: {.slide title="Grenzen der exakten Ableitung bei Quantisierung und Rundung"}

<!--en:88299513-->
[Lernstillstand bei Gradient null]{.kicker}

<!--en:78452938-->
- Da $\text{round}(w)$ eine Treppenfunktion ist, ist ihre Ableitung außer an den Halbzahlen überall null. Dadurch bleiben Gewichte vor einem Rundungsschritt unverändert:

<!--en:d1a5765a-->
::: {.slide title="Der Straight-Through-Schätzer (STE)"}

<!--en:d4f8e439-->
[Rückwärtsdurchläufe mit Ersatzableitungen]{.kicker}

<!--en:12e71b1b-->
Der **Straight-Through-Schätzer** (Bengio et al., 2013) entkoppelt Vorwärts- und Rückwärtsdurchlauf:

<!--en:1634f248-->
- **Vorwärtsdurchlauf (diskret):** Wertet die quantisierte Treppenfunktion $y = \text{round}(x)$ exakt aus.

<!--en:badd18b1-->
- **Rückwärtsdurchlauf (Identität als Ersatz):** Die Ableitung der Identität ($\frac{\partial y}{\partial x} \gets 1$) ersetzt die Nullableitung. Der eingehende Gradient bleibt unverändert:
  $$\nabla_{\mathbf{x}} L \gets \nabla_{\mathbf{y}} L$$

<!--en:c968979c-->
::: {.slide title="Definition einer eigenen Rückwärtsfunktion"}

<!--en:c2f221d6-->
[Eigene Grundoperationen für Autograd]{.kicker}

<!--en:5dfdc939-->
Weicht die Rückwärtsregel von der Ableitung des Vorwärtsdurchlaufs ab, beide Richtungen explizit definieren:

<!--en:a7e71c16-->
::: {.slide title="Training quantisierter Gewichte mit dem STE"}

<!--en:e50ed8dc-->
[Optimierung unter Berücksichtigung der Quantisierung]{.kicker}

<!--en:3513c4f8-->
Ein lineares Modell mit 8 Gewichten rundet diese bei jedem Vorwärtsdurchlauf auf ganze Zahlen (`w_true` gleichverteilt auf $[-7, 7]$). Beim Training senkt der STE den Verlust auf einen Wert nahe dem Verlust bei gerundeten wahren Gewichten. Bei gewöhnlicher Rundung bleibt er auf dem Wert aus Schritt 0:

<!--en:1b9fa3bd-->
- **Gradientenstopp in einer Zeile:** Dient die Identität als Ersatz, ergibt `X + stop_gradient(round(X) - X)` vorwärts den Wert `round(X)`. Rückwärts läuft die Ableitung über `X`. Eine eigene Klasse entfällt.

<!--en:d9fa4700-->
::: {.slide title="Gradientenstopp mit der Identität als Ersatzfunktion"}

<!--en:6adb7e79-->
[STE in einer Codezeile]{.kicker}

<!--en:0976dd65-->
- Prüfen, ob `X + stop_gradient(round(X) - X)` dieselben Ausgaben im Vorwärtsdurchlauf und dieselben Gradienten im Rückwärtsdurchlauf liefert wie die eigene Funktion `RoundSTE`:

<!--en:a257c5e5-->
::: {.slide title="Verständnisfragen zu eigenen Schichten und Ersatzgradienten"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:f060d63d-->
Drei Fragen zum Modellentwurf beantworten:

<!--en:028570e3-->
1. Llama 3 8B hat 32 Blöcke mit der verborgenen Breite $d = 4096$. Jeder Block enthält zwei `RMSNorm`-Schichten. Eine abschließende `RMSNorm` liegt vor der Ausgabeschicht. Wie viele Parameter entfallen insgesamt auf die Verstärkungsfaktoren von `RMSNorm`? Welchen Anteil machen sie an den 8 Mrd. Parametern des Modells aus?

<!--en:4baf77c1-->
2. Welche zwei Probleme treten beim Speichern eines Checkpoints und beim Verschieben des Modells auf eine GPU auf, wenn die Zufallsmatrix `W` in `RandomFeatures` als gewöhnliches Attribut gespeichert und nicht als Puffer registriert ist?

<!--en:1165786c-->
3. Beim Straight-Through-Runden mit `round_ste`: Welchen Gradienten hat $L(\mathbf{w}) = \sum_i (\text{round\_ste}(w_i))^2$ an der Stelle $\mathbf{w} = (0.9, 1.4, 2.6)$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:8917d87d-->
1. **$65 \times 4096 = 266{,}240$ Parameter ($\approx 0.0033\%$ von 8 Mrd.):** Ihr Speicherbedarf ist gering, doch für die Stabilisierung jedes Residualzweigs sind sie entscheidend.

<!--en:3c243253-->
2. **Unvollständiger Checkpoint und GPU-Absturz:** `W` fehlt im gespeicherten Zustand. Neu geladene Modelle ziehen daher eine unpassende Basis. Beim Wechsel des Modells auf die GPU bleibt die Matrix auf der CPU.

<!--en:334a130d-->
3. **$\nabla_{\mathbf{w}} L = 2\,\text{round}(\mathbf{w}) = (2, 2, 6)$:** In die äußere Quadratfunktion gehen die im Vorwärtsdurchlauf gerundeten Werte $(1, 1, 3)$ ein. Der STE ersetzt $\text{round}'(w_i) = 0$ durch $1$.

<!--en:e2ea0c8f-->
::: {.slide title="Zusammenfassung: Eigene Schichten und Funktionen"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:78ece916-->
- **Zustandslose und parametrisierte Schichten:** Von `Module` erben (`CenteredLayer`) und lernbare Gewichte als `Parameter` registrieren (`RMSNorm`).

<!--en:e65eab19-->
- **Puffer für festen Zustand:** Nicht trainierbare Tensoren als Puffer registrieren (`RandomFeatures`), damit sie in Checkpoints gespeichert und auf GPUs übertragen werden.

<!--en:c8590350-->
- **Ersatzgradienten:** Eigene Vektor-Jacobi-Produkte (`RoundSTE`) oder den Einzeiler `X + stop_gradient(f(X) - X)` verwenden, um Modelle mit diskreten Operationen zu trainieren.
