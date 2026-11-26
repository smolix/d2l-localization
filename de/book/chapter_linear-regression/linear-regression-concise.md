<!--en:216c2dac-->
# Kompakte Implementierung der linearen Regression

<!--en:3cb13709-->
In :numref:`sec_linear_scratch` haben wir die lineare Regression allein mit Tensoren implementiert. Wir haben Speicher für die Parameter reserviert, den affinen Vorwärtsdurchlauf (engl. forward pass) und den quadratischen Verlust (engl. squared loss) berechnet, im Rückwärtsmodus differenziert, einen Generator für Minibatches geschrieben und den Aktualisierungsschritt des stochastischen Gradientenabstiegs (engl. stochastic gradient descent, SGD) umgesetzt. In :numref:`sec_oo-design` haben wir diese Komponenten in wiederverwendbaren Basisklassen zusammengefasst (`d2l.Module`, `d2l.DataModule` und `d2l.Trainer`). Wir haben überprüft, dass das Training von `LinearRegressionScratch` mit `trainer.fit(model, data)` unsere Benchmark-Ergebnisse reproduziert.

<!--en:491c2121-->
Für ein lineares Modell mit 3 Parametern war es leicht, die Parametertensoren und Aktualisierungsregeln selbst zu implementieren und zu testen. Moderne Architekturen wie ResNet-50 (:numref:`sec_resnet`) oder GPT (:numref:`sec_gpt`) verwalten jedoch Millionen bis Milliarden von Parametern in Dutzenden von Schichten. Beim Training kommen adaptive Optimierer zum Einsatz, die ihren Zustand über mehrere Schritte hinweg speichern. Deshalb bieten Deep-Learning-Frameworks wie PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, JAX :cite:`Frostig.Johnson.Leary.2018` (mit Flax und Optax), TensorFlow :cite:`Abadi.Barham.Chen.ea.2016` und MXNet :cite:`Chen.Li.Li.ea.2015` bereits an die Hardware angepasste Schichten, numerisch getestete Verlustfunktionen (engl. loss functions) und Standardoptimierer. In diesem Abschnitt sehen wir, wie wenig Code wir für die lineare Regression brauchen, wenn wir diese Bausteine in `d2l.Module` und `d2l.Trainer` einsetzen.

<!--en:4d13c0fd-->
## Die vollständig verbundene Schicht

<!--en:64f7f409-->
In unserer kompakten Implementierung kapselt die Unterklasse `LinearRegression(d2l.Module)` eine einzige eingebaute vollständig verbundene Schicht (engl. fully connected layer) mit einer Ausgabeeinheit. Die Anfangswerte ihrer Gewichte ziehen wir aus $\mathcal{N}(0, 0.01^2)$, den Bias setzen wir auf null.

<!--en:a8c98ebc-->
In PyTorch benötigt `nn.LazyLinear(1)` beim Erzeugen nur die Ausgabedimension. Die Anzahl der Eingabemerkmale $d$ bestimmt die Schicht automatisch beim ersten Vorwärtsdurchlauf.

<!--en:e7a2cb3c-->
Auch `nn.Dense(1)` in Gluon bestimmt die Eingabedimension erst, wenn die Schicht den ersten Minibatch verarbeitet.

<!--en:39b2baee-->
Ebenso bestimmt `layers.Dense(1)` in Keras die Breite der Eingabe erst beim ersten Aufruf.

<!--en:510d8f4e-->
Schichten in Flax NNX reservieren den Speicher für ihre Parameter bereits beim Erzeugen. Deshalb erwartet `nnx.Linear` sowohl die Eingabedimension (`num_inputs`, hier `2`) als auch die Ausgabedimension (`1`).

<!--en:0877220c-->
Die Klasse `LinearRegression` braucht `forward` nicht einmal zu definieren. Sie erbt die Methode `Module.forward` aus :numref:`sec_oo-design`, die den Aufruf direkt an `self.net(X)` weitergibt.

<!--en:6f5e4337-->
## Die Verlustfunktion

<!--en:94bd9db2-->
Statt $(\hat{y} - y)^2 / 2$ selbst zu implementieren, greifen wir in `loss` auf die Framework-Funktion für den mittleren quadratischen Fehler (engl. mean squared error, MSE) zurück. In der üblichen Definition des mittleren quadratischen Fehlers fehlt der Faktor $\frac{1}{2}$ aus unserer eigenen Verlustfunktion. Der Verlust ist daher doppelt so groß und pendelt sich bei $\sigma^2 = 10^{-4}$ statt bei $\sigma^2 / 2$ ein.

<!--en:0d66f2a1-->
In MXNet enthält `gluon.loss.L2Loss` bereits den Faktor $\frac{1}{2}$. Deshalb multiplizieren wir das Ergebnis mit $2$ und erhalten den üblichen mittleren quadratischen Fehler.

<!--en:8aa0af9a-->
## Der Optimierer

<!--en:73e27925-->
Entsprechend gibt `configure_optimizers` den eingebauten SGD-Optimierer des Frameworks für Minibatches zurück. Er arbeitet mit den registrierten Parametern des Moduls und der Lernrate (engl. learning rate) `self.lr`:

<!--en:ae027837-->
## Training mit Framework-Komponenten

<!--en:a5c1ad1c-->
Nachdem wir `LinearRegression` definiert haben, brauchen wir für das Training nur wenige Schritte. Wir erzeugen das Datenmodul und das Modell mit der Lernrate $\eta = 0.015$. Gegenüber unserer ursprünglichen Lernrate $\eta = 0.03$ ist das die Hälfte, weil sich der MSE-Gradient verdoppelt hat. Anschließend erzeugen wir den Trainer mit `d2l.Trainer(max_epochs=10)` und rufen `trainer.fit(model, data)` auf:

<!--en:0e49b824-->
Wie bei unserer eigenen Implementierung weicht der Validierungsverlust (engl. validation loss) am Ende um höchstens etwa $3\%$ von der theoretischen Rauschgrenze $\sigma^2 = 10^{-4}$ ab. Ergänzen wir eine kleine Hilfsmethode `get_w_b`. Damit lesen wir den gelernten Gewichtstensor und den Bias aus der Framework-Schicht aus und vergleichen sie mit $(\mathbf{w}^*, b^*)$:

<!--en:89073002-->
Wieder betragen die Fehler in $\hat{\mathbf{w}}$ und $\hat{b}$ höchstens etwa ein Tausendstel, wie beim Training mit unserer eigenen Implementierung. Unabhängig davon, wer die Schicht implementiert hat, überwiegen dieselben beiden Fehlerquellen: das Labelrauschen in den $1{,}000$ Beispielen und der verbleibende Abstand zum Minimum nach $10$ Epochen.

<!--en:632d52e1-->
Bei allen Überprüfungen in den letzten drei Abschnitten kannten wir die wahren Parameter. Bei realen Datensätzen haben wir diesen Vorteil nie. In :numref:`sec_generalization_basics` untersuchen wir, wie wir Modelle ausschließlich anhand zurückgehaltener Daten (engl. held-out data) beurteilen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:5350d0e0-->
Mit den abstrakten Schnittstellen des Frameworks genügt für die lineare Regression eine einzige vollständig verbundene Schicht in einer Unterklasse von `d2l.Module`. Für `loss` nutzt die Klasse den eingebauten mittleren quadratischen Fehler, für `configure_optimizers` den eingebauten SGD-Optimierer. Beim üblichen mittleren quadratischen Fehler entfällt der Faktor $\frac{1}{2}$ unserer selbst implementierten Zielfunktion. Deshalb verdoppeln sich sowohl der Verlust, der sich bei $\sigma^2 = 10^{-4}$ einpendelt, als auch sein Gradient. Halbieren wir die Lernrate auf $\eta = 0.015$, erhalten wir dieselben Parameterupdates wie mit unserer eigenen Implementierung. Da `LinearRegression` die Schnittstelle von `d2l.Module` erfüllt, kann `d2l.Trainer` das Modell ohne jede Änderung an der Datenpipeline oder der Trainingsschleife anpassen.

<!--en:2294243a-->
## Übungen

<!--en:762c1537-->
1. [code] **Eine Schicht ohne Bias.** Implementieren Sie `LinearRegression` mit einer Schicht ohne Bias und trainieren Sie das Modell wie in diesem Abschnitt. In jedem der Frameworks hat die vollständig verbundene Schicht ein Argument, mit dem Sie den Bias abschalten können.
    1. Sagen Sie vor dem Training voraus, bei welchem Wert sich der Validierungsverlust einpendelt. Verwenden Sie dazu die Formel für den erwarteten quadratischen Fehler aus :numref:`sec_linear_scratch` und berücksichtigen Sie den Faktor $2$ im mittleren quadratischen Fehler. Überprüfen Sie dann Ihre Vorhersage.
    1. Sagen Sie voraus, gegen welche Werte die Gewichte konvergieren und wie weit sie am Ende von $\mathbf{w}^*$ entfernt sind. Da die Merkmale den Mittelwert null haben, wirkt der fehlende Bias auf die Gewichte wie Labelrauschen mit der Standardabweichung $b^* = 4.2$. Er beeinflusst sowohl den Schätzfehler aus :numref:`subsec_synthetic-regression-data-recovery` als auch die Größe der Kugel um das Minimum beim Minibatch-SGD (:numref:`sec_linear_regression`). Vergleichen Sie Ihre Vorhersage mit den Fehlern der trainierten Gewichte.
   :exlabel:`ex_linear-regression-concise-no-bias`

<!--en:ca87da4a-->
1. [code] **Mehrere Ausgaben.** Erhöhen Sie die Ausgabebreite der Schicht in `LinearRegression` von 1 auf 2. Erzeugen Sie `data` und hängen Sie an `data.y` eine zweite Labelspalte $y_2 = \mathbf{x}^\top \mathbf{w}_2^* + b_2^* + \epsilon_2$ an. Berechnen Sie diese aus `data.X` mit selbst gewählten Parametern und mit Rauschen derselben Stärke.
    1. Zeigen Sie, dass der über beide Ausgaben summierte quadratische Verlust zu zwei unabhängigen linearen Regressionen führt, einer pro Ausgabe. Folgern Sie daraus, dass die für eine Ausgabe gelernten Parameter unabhängig von den Labels der anderen sind.
    1. Die Verlustfunktionen der Frameworks mitteln über alle Einträge der $|\mathcal{B}| \times 2$-Fehlermatrix eines Minibatchs $\mathcal{B}$, also auch über die Ausgaben jedes Beispiels. Geben Sie an, um welchen Faktor der Gradient für die Parameter jeder Ausgabe gegenüber dem Modell mit einer Ausgabe schrumpft. Welche Lernrate reproduziert den Trainingslauf mit einer Ausgabe? Überprüfen Sie dies, indem Sie beide Modelle mit denselben Anfangswerten auf denselben Minibatches trainieren.

<!--en:7929096a-->
1. [code] **Merkmalsskala und Schrittweite.** Multiplizieren Sie die erste Spalte von `data.X` nach dem Erzeugen von `data` mit 100. Lassen Sie `data.y` unverändert, sodass das Gewicht dieses Merkmals im datenerzeugenden Modell nun $0.02$ beträgt. Erinnern Sie sich an die Übung :exref:`ex_linear-regression-scratch-stability`: Der Gradientenabstieg divergiert, sobald $\eta$ den Wert $2/\mu_{\max}$ übersteigt. Dabei ist $\mu_{\max}$ der größte Eigenwert der Hesse-Matrix $\mathbf{H}$ der Verlustfunktion.
    1. Berechnen Sie $\mathbf{H}$ für `LinearRegression.loss` und schätzen Sie die Eigenwerte vor und nach der Skalierung.
    1. Sagen Sie für beide Fälle die größte Lernrate voraus, bei der das Training noch nicht divergiert. Überprüfen Sie beide Vorhersagen durch Trainingsläufe mit Lernraten in der Nähe der jeweiligen Grenze.
    1. Wählen Sie eine Lernrate knapp unter der neuen Grenze. Sagen Sie voraus, wie viele Updates das zweite Gewicht benötigt, um seinen Fehler um den Faktor 100 zu verringern. Vergleichen Sie Ihre Vorhersage mit einer Messung. Standardisieren Sie anschließend das umskalierte Merkmal wie in :numref:`sec_pandas` und zeigen Sie, dass `lr=0.015` wieder funktioniert.

<!--en:532140ec-->
1. [code] **Optimierungsfehler und Schätzfehler im Vergleich.** ● Trainieren Sie `LinearRegression` mit `lr=0.015` und `max_epochs=10` auf `SyntheticRegressionData`. Setzen Sie `num_train` nacheinander auf $5, 50, 500, 5{,}000$ und verwenden Sie `batch_size=min(32, num_train)`. Vergleichen Sie für jeden Lauf den Abstand $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2$ der trainierten Gewichte von den datenerzeugenden Gewichten $\mathbf{w}^*$ in `data.w` mit dem entsprechenden Abstand für die Lösung nach der Methode der kleinsten Quadrate. Berechnen Sie diese Lösung auf denselben Trainingsdaten. Gehen Sie dabei vor wie im Unterabschnitt zur Rekonstruktion der Parameter (:numref:`subsec_synthetic-regression-data-recovery`).
    1. Zählen Sie die Parameterupdates in jedem Lauf und erklären Sie für jeden Wert von `num_train` das Verhältnis der beiden Fehler.
    1. Bestimmen Sie, nach wie vielen Epochen der Fehler im Lauf mit `num_train=5` höchstens noch doppelt so groß ist wie der Fehler der Lösung nach der Methode der kleinsten Quadrate.

<!--en:0b6a5df3-->
[Diskussion](https://d2l.discourse.group/t/44)

<!--en:7b39ab9d-->
[Diskussion](https://d2l.discourse.group/t/45)

<!--en:08464b8d-->
[Diskussion](https://d2l.discourse.group/t/204)

<!--en:f95ef81b-->
[Diskussion](https://d2l.discourse.group/t/17977)

<!--en:cbdf3032-->
Kompakte Implementierung der linearen Regression<br>
**Framework-Schichten · Formbestimmung beim ersten Aufruf · Skalierung des MSE-Verlusts · eingebaute Optimierer · Wiederverwendung des Trainers**

<!--en:94d766df-->
::: {.slide title="Warum von der eigenen Implementierung zur Framework-API wechseln?"}

<!--en:7b7950ad-->
[Übergang zu tiefen Netzen]{.kicker}

<!--en:63979fb0-->
- **An die Hardware angepasste Schichten:** Eine einzige vollständig verbundene Framework-Schicht ersetzt das manuelle Anlegen von `w, b` und die Berechnung `matmul(X, w) + b`.

<!--en:58ce7236-->
- **Getestete Verlustfunktionen und Optimierer:** Die eigene Implementierung von `squared_loss` und die Schleifen über die Parameter durch den eingebauten MSE und `SGD` ersetzen

<!--en:ec52a503-->
- **Keine Änderungen am Trainer:** Dieselben Klassen `SyntheticRegressionData` und `d2l.Trainer` aus dem Abschnitt zum objektorientierten Entwurf kommen wieder zum Einsatz.

<!--en:e139fa15-->
::: {.slide title="Kompaktes Modell mit einer einzigen vollständig verbundenen Schicht" except="jax"}

<!--en:352e813a-->
[Schnittstelle der Framework-Schicht]{.kicker}

<!--en:eb94e965-->
- **Geerbtes `forward` und Formbestimmung beim ersten Aufruf:** `d2l.Module.forward(X)` leitet den Aufruf direkt an `self.net(X)` weiter. Die vollständig verbundene Schicht bestimmt die Eingabedimension $d = 2$ automatisch anhand des ersten Minibatchs.

<!--en:8eac08be-->
::: {.slide title="Kompaktes Modell mit einer einzigen vollständig verbundenen Schicht" only="jax"}

<!--en:352e813a-2-->
[Schnittstelle der Framework-Schicht]{.kicker}

<!--en:ea7a10ee-->
- **Geerbtes `forward` und Formbestimmung beim Erzeugen:** `d2l.Module.forward(X)` leitet den Aufruf direkt an `self.net(X)` weiter. Da `nnx.Linear` seine Parameter bereits beim Erzeugen anlegt, ist die Eingabedimension $d = 2$ explizit anzugeben.

<!--en:1b8d75bb-->
::: {.slide title="Anfangswerte der Parameter: kleine normalverteilte Gewichte, Bias null" only="pytorch"}

<!--en:eb1bfb15-->
[Initialisierung von Gewichten und Bias]{.kicker}

<!--en:ba3aa3e2-->
- **Verzögerte Initialisierung (`LazyLinear`):** Mit einem Probeaufruf `model(data.X[:1])` die Form der Parameter festlegen, dann `normal_(0, 0.01)` auf das Gewicht und `zeros_` auf den Bias anwenden

<!--en:e2c9a444-->
::: {.slide title="Anfangswerte der Parameter: kleine normalverteilte Gewichte, Bias null" only="tensorflow"}

<!--en:eb1bfb15-2-->
[Initialisierung von Gewichten und Bias]{.kicker}

<!--en:003ab0e4-->
- **Verzögerte Initialisierung (`Dense`):** Beim Erzeugen `RandomNormal(stddev=0.01)` angeben. Die Schicht legt die Gewichte beim ersten Minibatch an. Der Anfangswert des Bias ist null.

<!--en:3c42279e-->
::: {.slide title="Anfangswerte der Parameter: kleine normalverteilte Gewichte, Bias null" only="mxnet"}

<!--en:eb1bfb15-3-->
[Initialisierung von Gewichten und Bias]{.kicker}

<!--en:f11542b9-->
- **Verzögerte Initialisierung (`Dense`):** Beim Erzeugen `init.Normal(0.01)` angeben. Die Schicht legt die Gewichte beim ersten Minibatch an. Der Anfangswert des Bias ist null.

<!--en:58574123-->
::: {.slide title="Anfangswerte der Parameter: kleine normalverteilte Gewichte, Bias null" only="jax"}

<!--en:eb1bfb15-4-->
[Initialisierung von Gewichten und Bias]{.kicker}

<!--en:b35203da-->
- **Sofortige Initialisierung (`nnx.Linear`):** Die Schicht legt die Arrays bereits beim Erzeugen mit `num_inputs=2`, `kernel_init=normal(0.01)` und `rngs` an. Der Anfangswert des Bias ist null.

<!--en:3ea06a6a-->
::: {.slide title="Mittlerer quadratischer Fehler im Framework und Halbierung der Lernrate"}

<!--en:1f879c30-->
[Skalierung des Verlusts und Kopplung der Hyperparameter]{.kicker}

<!--en:131b0402-->
- **Warum $\eta$ von $0.03$ auf $0.015$ halbieren?** Beim üblichen MSE des Frameworks entfällt der Faktor $\frac{1}{2}$. Dadurch verdoppeln sich der Verlust (Rauschgrenze $\sigma^2 = 10^{-4}$) und sein Gradient ($2\mathbf{g}$). Mit der halbierten Lernrate $\eta = 0.015$ ergibt sich derselbe Schritt wie bei der eigenen Implementierung: $-\eta(2\mathbf{g}) = -0.03\mathbf{g}$.

<!--en:bae01c36-->
::: {.slide title="Konfiguration des eingebauten Optimierers"}

<!--en:291636bc-->
[Abstraktion des Optimierers]{.kicker}

<!--en:02937ffb-->
- **Automatische Ermittlung der Parameter:** Das Framework durchläuft den Modulbaum und erfasst alle registrierten Parameter. Der Optimierer braucht daher keine von Hand erstellte Tensorliste.

<!--en:aa09aa3b-->
::: {.slide title="Training und Rekonstruktion der Parameter mit Framework-Komponenten"}

<!--en:ad59a9d8-->
[Die kompakte Implementierung im Einsatz]{.kicker}

<!--en:63965467-->
Das Training von `LinearRegression(lr=0.015)` mit `d2l.Trainer(max_epochs=10)` reproduziert die Konvergenz (Fehler pro Epoche $7\times$ kleiner) und rekonstruiert die Parameter mit einem Fehler der Größenordnung $O(10^{-4})$:

<!--en:32f65357-->
- **Rauschgrenze bei $\sigma^2 = 10^{-4}$:** Beim MSE des Frameworks entfällt der Faktor $\frac{1}{2}$. Der Verlust pendelt sich daher bei $\sigma^2 = 10^{-4}$ ein. Die Quadratwurzel $\sqrt{10^{-4}} = 0.01 = \sigma$ ergibt das quadratische Mittel der Vorhersagefehler in den Einheiten der Labels.

<!--en:e16d0b29-->
::: {.slide title="Auslesen und Überprüfen der gelernten Schichtparameter"}

<!--en:7d61e8cc-->
[Gewichte der Framework-Schicht]{.kicker}

<!--en:5f696322-->
- **Übereinstimmung mit der Ground Truth:** Gewichte und Bias weichen von $(\mathbf{w}^*, b^*)$ nur um wenige Vielfache von $\sigma / \sqrt{n} \approx 3 \times 10^{-4}$ ab.

<!--en:482f8a71-->
::: {.slide title="Drei Abstraktionsebenen: eigene Funktionen → eigene Klassen → kompakte Implementierung"}

<!--en:8ec5c9f8-->
[Aufbau des Kapitels]{.kicker}

<!--en:431354f0-->
- **Einzelne Funktionen:** Einfache Tensoren `w, b`, selbst implementierte Funktionen `linreg` und `squared_loss`, ein manueller Rückwärtsdurchlauf und ein Generator `minibatches` für einen Durchlauf

<!--en:34d4069e-->
- **Eigene objektorientierte Implementierung (`LinearRegressionScratch`):** Kapselt einfache Tensoren und den selbst implementierten Optimierer `SGD(lr=0.03)` in `d2l.Module`. Das Training nutzt `SyntheticRegressionData` und `d2l.Trainer`.

<!--en:8e0b237b-->
- **Kompakte Implementierung (`LinearRegression`):** Nutzt vollständig verbundene Framework-Schichten, den eingebauten MSE und den eingebauten Optimierer `SGD(lr=0.015)`. `DataModule` und `Trainer` bleiben unverändert.

<!--en:f0188e78-->
::: {.slide title="Verständnisfragen: Skalierung des Verlusts, fehlender Bias und mehrere Ausgaben"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:8a6cd163-->
Drei Fragen zu den Konventionen der Frameworks für Schichten und Verlustfunktionen:

<!--en:8873dcbc-->
1. **Warum der Verlust der kompakten Implementierung ein doppelt so hohes Plateau erreicht:** Warum pendelt er sich bei etwa $10^{-4}$ ein, der Verlust der eigenen Implementierung aber bei etwa $5 \times 10^{-5}$, obwohl beide denselben Generator für synthetische Daten nutzen?

<!--en:3086bcab-->
2. **Ohne Bias (`bias=False`):** Welchen Wert erreicht der Validierungs-MSE auf dem Plateau, wenn in `LinearRegression` für diesen Datensatz der Bias deaktiviert ist ($b^* = 4.2$, Merkmale mit Mittelwert null: $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$)?

<!--en:4fd74cfa-->
3. **Skalierung des MSE bei zwei Ausgaben:** Liefert `LinearRegression` nun $2$ Vorhersagen statt $1$, mittelt der MSE des Frameworks über alle $2|\mathcal{B}|$ Einträge der Residuenmatrix. Wie wirkt sich das auf den Gradienten jeder Ausgabe und die äquivalente Lernrate aus?

<!--en:557b98b6-->
**Lösungen:**

<!--en:468759fe-->
1. Das Framework berechnet den MSE als $\frac{1}{|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ ohne den Faktor $\frac{1}{2}$ der eigenen Implementierung. Bei den wahren Parametern beträgt sein Erwartungswert daher $\mathbb{E}[\epsilon^2] = \sigma^2 = (0.01)^2 = 10^{-4}$ statt $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$.

<!--en:11312eaf-->
2. Wegen $\mathbb{E}[\mathbf{x}] = \mathbf{0}$ konvergieren die Gewichte weiterhin gegen Werte nahe $\mathbf{w}^* = (2, -3.4)^\top$. Der im Modell fehlende Achsenabschnitt $b^* = 4.2$ bleibt in jedem Residuum erhalten. Dadurch steigt die Rauschgrenze des MSE auf $(b^*)^2 + \sigma^2 = 4.2^2 + 10^{-4} \approx 17.64$.

<!--en:83067e78-->
3. Die Division durch $2|\mathcal{B}|$ statt durch $|\mathcal{B}|$ halbiert den Gradienten bezüglich der Parameter jeder Ausgabe. Eine Verdopplung der Lernrate auf $\eta = 0.03$ ergibt wieder die Schrittweite des Modells mit einer Ausgabe.

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und Kernaussagen"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:f6e8f2c0-->
| Komponente | Eigene Implementierung | Kompakte Framework-Implementierung | Gemeinsamkeiten und Unterschiede |

<!--en:a6b92315-->
| **Modell und Parameter** | `LinearRegressionScratch` (`self.w`, `self.b`) | `LinearRegression` (vollständig verbundene Framework-Schicht) | Erbt `Module.forward(X) -> self.net(X)` |

<!--en:ddcf88cc-->
| **Verlustfunktion** | $\frac{1}{2|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ | Eingebauter MSE | Ohne den Faktor $1/2 \implies$ Verlust und Gradient jeweils $2\times$ so groß |

<!--en:8b6518eb-->
| **Optimierer und Schrittweite** | Eigener `SGD` mit $\eta = 0.03$ | Eingebauter `SGD` mit $\eta = 0.015$ | Halbierung von $\eta$ ergibt exakt dasselbe Parameterupdate $-\eta\nabla L$ |

<!--en:91431fa0-->
| **Rauschgrenze und Parameterrekonstruktion** | Rauschgrenze $\sigma^2/2 = 5 \times 10^{-5}$ | Rauschgrenze $\sigma^2 = 10^{-4}$ ($\sqrt{\text{MSE}} = \sigma = 0.01$) | Beide rekonstruieren $(\mathbf{w}^*, b^*)$ mit einem Fehler von höchstens $\sim 3\sigma/\sqrt{n} \approx 10^{-3}$ |
