<!--en:135a254d-->
# Softmax-Regression

<!--en:b3568888-->
Mit der Regression beantworten wir Fragen wie *Wie viel?* oder *Wie viele?* Beim überwachten Lernen lautet die Frage jedoch oft *Welche Kategorie?* Ein Spamfilter sortiert E-Mails in zwei Klassen, ein klinisches Diagnosesystem wählt unter möglichen Diagnosen aus, und ein Sprachmodell sagt das nächste Token aus einem Vokabular von mehreren Zehntausend Wörtern vorher.

<!--en:d5763599-->
Angenommen, ein Onlinehändler möchte Produktfotos in zehn Modekategorien einteilen: T-Shirt, Hose, Pullover, Kleid, Mantel, Sandale, Hemd, Sneaker, Tasche und Stiefelette. Warum nummerieren wir die Kategorien nicht von $1$ bis $10$ und wenden direkt die lineare Regression an? Bei ungeordneten Kategorien dienen die ganzen Zahlen nur als beliebig gewählte Namen. Als Mengenangaben haben sie keinen Sinn. Eine Vorhersage von $5.5$ würde ein Foto auf halbem Weg zwischen Mantel und Sandale einordnen. Der quadratische Fehler würde die Verwechslung eines T-Shirts ($1$) mit einer Stiefelette ($10$) $81$-mal so stark bestrafen wie die Verwechslung mit einer Hose ($2$). Selbst wenn wir am Ende eine *harte Zuordnung* (engl. hard assignment) brauchen, also eine einzige Kategorie wählen müssen, wünschen wir uns fast immer ein Modell für *weiche Zuordnungen* (engl. soft assignments). Es liefert für jede mögliche Klasse eine kalibrierte Wahrscheinlichkeit. So können wir bei späteren Entscheidungen die Konfidenz gegen die Kosten eines Fehlers abwägen.

<!--en:398e0bfc-->
Die *Softmax-Regression* (auch *multinomiale logistische Regression*) löst beide Probleme. Sie berechnet für jede Klasse einen affinen Score und bildet die Scores mit der Softmax-Funktion auf eine gültige Wahrscheinlichkeitsverteilung ab. Folgen wir dem Maximum-Likelihood-Prinzip aus :numref:`sec_linear_regression` und minimieren die negative Log-Likelihood der beobachteten Kategorien, so ergibt sich der *Kreuzentropieverlust* (engl. cross-entropy loss).

<!--en:91e0cf09-->
## Das Modell

<!--en:6a6c23aa-->
### Klassenlabel

<!--en:cd94455b-->
Beginnen wir mit der Bildklassifikation für $2 \times 2$-Graustufenbilder. Wir haben vier skalare Pixelmerkmale $x_1, x_2, x_3, x_4$ und drei Kategorien: Katze ($1$), Huhn ($2$) und Hund ($3$). Allgemein ist bei $q$ Klassen das Label eines Beispiels eine ganze Zahl $y \in \{1, \ldots, q\}$.

<!--en:5d33f1c5-->
Um ungeordnete Kategorien ohne künstliche Zahlenordnung darzustellen, verwendet man in der Statistik die *One-Hot-Kodierung*. Sie stellt jede Kategorie als Vektor $\mathbf{y} \in \{0, 1\}^q$ dar. An der Stelle der wahren Klasse steht eine $1$ ($y_y = 1$), an allen anderen Stellen eine $0$. In unserem Beispiel mit drei Klassen steht $(1, 0, 0)$ für eine Katze, $(0, 1, 0)$ für ein Huhn und $(0, 0, 1)$ für einen Hund. Je zwei verschiedene Klassen haben nun denselben euklidischen Abstand $\sqrt{2}$. Außerdem ist $\mathbf{y}$ selbst eine gültige Wahrscheinlichkeitsverteilung. Diese entartete Verteilung legt ihre gesamte Masse auf die wahre Klasse. Wir können sie direkt mit der vom Modell vorhergesagten Verteilung $\hat{\mathbf{y}}$ vergleichen. (In Gleichungen verwenden wir den bei $1$ beginnenden Klassenindex $y$ und den One-Hot-Vektor $\mathbf{y}$. Im Python-Code speichern wir direkt den bei $0$ beginnenden ganzzahligen Index.)

<!--en:b17f8902-->
### Lineare Scores

<!--en:9ef9269c-->
Um alle $q$ Klassen gleichzeitig zu bewerten, ordnet die Softmax-Regression jeder Klasse $j \in \{1, \ldots, q\}$ eine eigene affine Funktion der Eingabemerkmale zu. Mit $w_{kj}$ bezeichnen wir das Gewicht, das Merkmal $k$ mit Klasse $j$ verbindet, und mit $b_j$ den Bias der Klasse $j$. Unser Bildklassifikator für $2 \times 2$-Bilder berechnet drei unnormierte Scores (*Logits*):

<!--en:4f63acd4-->
Als neuronales Netz betrachtet (:numref:`fig_softmaxreg`) ist die Softmax-Regression ein einschichtiges, vollständig verbundenes Netz mit $d$ Eingängen und $q$ Ausgabeeinheiten ($d \cdot q$ Gewichte und $q$ Bias-Terme).

<!--en:027d9525-->
![Softmax-Regression als einschichtiges, vollständig verbundenes neuronales Netz. Jede Eingabe $x_k$ ist über eine Verbindung mit jedem Ausgabe-Logit $o_j$ verknüpft. Die Verbindung hat das Gewicht $w_{kj}$.](../img/softmaxreg.svg)

<!--en:c266ab95-->
Wir fassen die Gewichte in einer Matrix $\mathbf{W} \in \mathbb{R}^{d \times q}$ zusammen (eine Zeile je Merkmal, eine Spalte $\mathbf{w}_j \in \mathbb{R}^d$ je Klasse) und die Bias-Terme in $\mathbf{b} \in \mathbb{R}^q$. Für den Logit-Vektor zur Eingabe $\mathbf{x} \in \mathbb{R}^d$ ergibt sich die kompakte Schreibweise

<!--en:15a1670a-->
mit $o_j = \mathbf{w}_j^\top \mathbf{x} + b_j$. Für die $28 \times 28$-Bilder von Kleidungsstücken aus Fashion-MNIST, die wir in :numref:`sec_fashion_mnist` klassifizieren, gilt $d = 784$ und $q = 10$. Das ergibt $7{,}840$ Gewichte und $10$ Bias-Terme.

<!--en:d9d9a684-->
### Die Softmax-Funktion

<!--en:ea21d0bc-->
Angenommen, unsere lineare Schicht gibt für Katze, Huhn und Hund die Logits $\mathbf{o} = (1.0, 2.2, 0.3)$ aus. Das Huhn hat den höchsten Score. Die Logits selbst sind jedoch keine Wahrscheinlichkeiten. Ihre Summe beträgt $3.5$ statt $1$, und ein affiner Score kann negativ werden oder $1$ überschreiten. Wie lässt sich ein beliebiger reeller Vektor $\mathbf{o} \in \mathbb{R}^q$ in eine gültige Wahrscheinlichkeitsverteilung $\hat{\mathbf{y}}$ umwandeln, also in $q$ positive Zahlen mit Summe $1$, ohne die Reihenfolge der Scores zu ändern? Die *Softmax-Funktion* exponentiert jeden Logit und dividiert die so erhaltenen Werte durch ihre Summe:

<!--en:7b347046-->
Für $\mathbf{o} = (1.0, 2.2, 0.3)$ ergeben sich die Exponentialwerte $(2.72, 9.03, 1.35)$ mit der Summe $13.09$. Daraus folgt $\hat{\mathbf{y}} = (0.21, 0.69, 0.10)$. Das Modell sagt also für das Huhn eine Wahrscheinlichkeit von $69\%$ vorher. Da $\exp$ streng monoton wächst, erhält die Softmax-Funktion die Reihenfolge der Logits. Die wahrscheinlichste Klasse ist daher immer die mit dem größten Logit:

<!--en:80340f80-->
Brauchen wir bei der Inferenz nur die wahrscheinlichste Kategorie, können wir auf die Exponentialfunktion verzichten und direkt $\operatorname{argmax}_j o_j$ bestimmen.

<!--en:f3bed626-->
Warum verwenden wir die Exponentialfunktion und keine andere positive, monoton wachsende Funktion? Die Exponentialfunktion verwandelt Summen in Produkte. Dadurch erhält die Softmax-Funktion drei grundlegende Eigenschaften:

<!--en:b6f95ccb-->
1. **Verschiebungsinvarianz (nur die Differenzen zwischen Logits zählen):** Addieren wir eine beliebige skalare Konstante $c$ zu jedem Eintrag von $\mathbf{o}$, so erhalten Zähler und Nenner in :eqref:`eq_softmax_y_and_o` denselben Faktor $\exp(c)$. Die Wahrscheinlichkeiten bleiben also gleich: $\mathrm{softmax}(\mathbf{o} + c\mathbf{1}) = \mathrm{softmax}(\mathbf{o})$. Gleichbedeutend damit ist die Aussage, dass die Differenz zweier beliebiger Logits dem Logarithmus des Verhältnisses ihrer vorhergesagten Wahrscheinlichkeiten entspricht:
   $$o_i - o_j = \log \frac{\hat{y}_i}{\hat{y}_j}.$$
   In unserem Beispiel gilt $o_2 - o_1 = 1.2$. Damit ist ein Huhn $e^{1.2} \approx 3.3$-mal so wahrscheinlich wie eine Katze ($0.69 / 0.21$). Die Verschiebungsinvarianz schützt auch vor einem wichtigen numerischen Problem (:numref:`sec_softmax_scratch`). Ziehen wir vor dem Exponentieren $o_{\max} = \max_k o_k$ ab, verhindern wir einen Gleitkommaüberlauf, ohne das Ergebnis zu verändern.

<!--en:929d9656-->
2. **Rückführung auf die logistische Regression für $q = 2$:** Bei zwei Klassen hängen die Wahrscheinlichkeiten allein von der Differenz $o = o_1 - o_2$ ab:
   $$\hat{y}_1 = \frac{\exp(o_1)}{\exp(o_1) + \exp(o_2)} = \frac{1}{1 + \exp(-o)} = \sigma(o),$$
   :eqlabel:`eq_softmax_to_sigmoid`
   Dabei ist $\sigma(o)$ die *logistische Sigmoidfunktion*. Die Softmax-Regression mit zwei Klassen ist also identisch mit der klassischen *logistischen Regression*. Der Ausdruck $o = \log(\hat{y}_1 / \hat{y}_2)$ bezeichnet die Log-Odds. Hier liegt der historische Ursprung des Wortes *Logit*.

<!--en:6a5d56b2-->
3. **Einfache Gradienten bei Maximum-Likelihood:** Wie wir weiter unten zeigen, wirken die Exponentialfunktion in der Softmax-Funktion und der Logarithmus in der Log-Likelihood so zusammen, dass sich der Gradient des Verlusts auf das Residuum $\hat{\mathbf{y}} - \mathbf{y}$ reduziert.

<!--en:5573628b-->
### Minibatches

<!--en:1b95bf4d-->
Um die Berechnung für einen Minibatch mit $n$ Beispielen und je $d$ Merkmalen zu vektorisieren, fassen wir die Eingaben in $\mathbf{X} \in \mathbb{R}^{n \times d}$, die Gewichte in $\mathbf{W} \in \mathbb{R}^{d \times q}$ und die Bias-Terme in $\mathbf{b} \in \mathbb{R}^{1 \times q}$ zusammen:

<!--en:ddba6205-->
Dabei wird $\mathbf{b}$ durch Broadcasting auf alle $n$ Zeilen von $\mathbf{O} \in \mathbb{R}^{n \times q}$ angewandt. Die Funktion $\mathrm{softmax}$ wirkt *zeilenweise*: Wir exponentieren die Einträge von $\mathbf{O}$ und dividieren jede Zeile durch ihre eigene Zeilensumme.

<!--en:0b8f65ec-->
### Entscheidungsregionen

<!--en:a53de328-->
Da die Softmax-Funktion die Reihenfolge der affinen Logits $o_j = \mathbf{w}_j^\top\mathbf{x} + b_j$ erhält, zerlegt die Softmax-Regression den Eingaberaum $\mathbb{R}^d$ in konvexe Polyeder mit ebenen Seitenflächen (:numref:`fig_mdl-clf-decision-regions`). Klasse $j$ gewinnt überall dort, wo $o_j \geq o_k$ für alle $k \neq j$ gilt. Diese Region ist der Schnitt von $q - 1$ Halbräumen $(\mathbf{w}_j - \mathbf{w}_k)^\top\mathbf{x} + (b_j - b_k) \geq 0$. Benachbarte Klassen treffen auf der Hyperebene aufeinander, auf der ihre beiden Logits gleich sind. Wie bei der linearen Regression sind die Grenzen zwischen den Klassen stets *eben*, also Hyperebenen. Gekrümmte oder nichtkonvexe Entscheidungsregionen erfordern versteckte Schichten (:numref:`chap_perceptrons`).

<!--en:c4d2f670-->
![Entscheidungsregionen der Softmax-Regression in $\mathbb{R}^2$. Links: Bei drei Klassen zerfällt die Ebene in drei konvexe Regionen. Sie treffen entlang der Geraden aufeinander, auf denen jeweils zwei Logits gleich sind. Rechts: Bei zwei Klassen sind die Linien konstanter Wahrscheinlichkeit $\hat{y}_1 = \sigma(o)$ parallele Geraden senkrecht zu $\mathbf{w}_1 - \mathbf{w}_2$. Die Entscheidungsgrenze liegt bei $\hat{y}_1 = 1/2$.](../img/mdl-clf-decision-regions.svg)

<!--en:35de5731-->
Unsere zentrale Modellannahme lautet: Jedes Beispiel gehört zu *genau einer* Klasse aus einer *ungeordneten* Menge. Können für dieselbe Eingabe mehrere Label gleichzeitig gelten (*Mehrlabelklassifikation*, engl. multi-label classification), verwendet man für jedes Label eine unabhängige binäre Sigmoidfunktion anstelle einer gemeinsamen Softmax-Funktion mit konkurrierenden Klassen :cite:`Tsoumakas.Katakis.2007`. Haben die Klassen eine natürliche Rangfolge (*ordinale Regression*), nutzen spezielle Schwellenwertmodelle diese Ordnung aus :cite:`Moon.Smola.Chang.ea.2010`.

<!--en:f85d63ca-->
## Der Kreuzentropieverlust

<!--en:73a5f16e-->
### Maximum-Likelihood

<!--en:bec5e510-->
Wie messen wir die Abweichung zwischen der vorhergesagten Wahrscheinlichkeitsverteilung $\hat{\mathbf{y}}$ und dem beobachteten Label $y$? Nach dem Maximum-Likelihood-Prinzip aus :numref:`sec_linear_regression` interpretieren wir $\hat{y}_j = P(y = j \mid \mathbf{x})$ als bedingte Wahrscheinlichkeit der Klasse $j$. Mit dem One-Hot-Vektor $\mathbf{y} \in \{0, 1\}^q$ lässt sich die Wahrscheinlichkeit für die beobachtete Klasse $y$ schreiben als

<!--en:b1575305-->
Für $n$ unabhängige Trainingsbeispiele ist die negative Log-Likelihood die Summe der Verluste der einzelnen Beispiele:

<!--en:dc041fe5-->
Diese Zielfunktion heißt *Kreuzentropieverlust*. Betrachten wir ihr Verhalten:

<!--en:c4f1123c-->
- Wegen $\hat{y}_y \in (0, 1)$ gilt stets $l(\mathbf{y}, \hat{\mathbf{y}}) > 0$. Der Verlust nähert sich $0$ für $\hat{y}_y \to 1$ und wächst gegen $+\infty$ für $\hat{y}_y \to 0$. Fehler mit hoher Konfidenz wiegen also besonders schwer.

<!--en:57d1ba10-->
- Für unser Bild mit den Logits $(1.0, 2.2, 0.3)$ und den Wahrscheinlichkeiten $(0.21, 0.69, 0.10)$ beträgt der Verlust bei wahrem Label Huhn ($y = 2$) nur $-\log 0.69 \approx 0.37$ Nat. Zeigt das Bild hingegen einen Hund ($y = 3$), steigt der Verlust auf das Sechsfache, nämlich $-\log 0.10 \approx 2.30$ Nat.

<!--en:699278a4-->
### Der Gradient

<!--en:1502b2dc-->
Leiten wir den Kreuzentropieverlust nach den Logits $\mathbf{o}$ ab. Setzen wir :eqref:`eq_softmax_y_and_o` in :eqref:`eq_l_cross_entropy` ein und nutzen $\sum_{j=1}^q y_j = 1$, so können wir den Verlust direkt als Funktion von $\mathbf{o}$ schreiben:

<!--en:7e4d9d6c-->
Der erste Term $\log \sum_{k=1}^q \exp(o_k)$ ist die *Log-Sum-Exp-Funktion*, eine glatte, konvexe obere Schranke für $\max_k o_k$. Sie liegt höchstens um $\log q$ über $\max_k o_k$ (Übung :exref:`ex_softmax-regression-realsoftmax`). Die Ableitung nach $o_j$ ergibt

<!--en:416e2b45-->
Wie bei der linearen Regression mit quadratischem Verlust ist **der Gradient nach den natürlichen Parametern $\mathbf{o}$ gleich der Differenz aus Vorhersage und Beobachtung, also $\hat{\mathbf{y}} - \mathbf{y}$**. Diese Eigenschaft haben alle verallgemeinerten linearen Modelle der Exponentialfamilie mit natürlicher Parametrisierung (siehe :numref:`subsec_mdl-glm`). Für unser Bild mit dem Huhn ($\mathbf{y} = (0, 1, 0)$) gilt $\nabla_{\mathbf{o}}\, l = (0.21, -0.31, 0.10)$. Ein Schritt des Gradientenabstiegs erhöht also den Logit für das Huhn. Die Logits für Katze und Hund sinken jeweils genau proportional zu der Wahrscheinlichkeitsmasse, die der jeweiligen Klasse irrtümlich zugewiesen wurde. Wenden wir die Kettenregel auf $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ an, so ergibt sich als Minibatch-Gradient nach $\mathbf{W}$ der Ausdruck $\frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{Y}} - \mathbf{Y})$. Der Verlust ist global *konvex* in $(\mathbf{W}, \mathbf{b})$. Seine Hesse-Matrix bezüglich $\mathbf{o}$ ist die Kovarianzmatrix einer kategorialen Ziehung aus $\hat{\mathbf{y}}$ (siehe Übung :exref:`ex_softmax-regression-hessian`).

<!--en:b7ba2bb7-->
| | Lineare Regression | Softmax-Regression |

<!--en:cff595fd-->
| **Zielgröße** | stetig, $y \in \mathbb{R}$ | Klasse $y \in \{1, \ldots, q\}$, One-Hot-Vektor $\mathbf{y}$ |

<!--en:7393411a-->
| **Vorhersage** | $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$ | $\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{W}^\top \mathbf{x} + \mathbf{b})$ |

<!--en:43f0cefc-->
| **Probabilistisches Modell** | Gauß-Verteilung $\mathcal{N}(\hat{y}, \sigma^2)$ | Kategorialverteilung $\mathrm{Cat}(\hat{\mathbf{y}})$ |

<!--en:8b4b51c2-->
| **Verlust (NLL)** | $\frac{1}{2}(\hat{y} - y)^2$ | $-\sum_{j=1}^q y_j \log \hat{y}_j = -\log \hat{y}_y$ |

<!--en:18bf6173-->
| **Gradient** | $\hat{y} - y$ (nach $\hat{y}$) | $\hat{\mathbf{y}} - \mathbf{y}$ (nach $\mathbf{o}$) |

<!--en:f00586c1-->
### Entropie und Kreuzentropie

<!--en:d6790b02-->
Die Bezeichnung *Kreuzentropie* (engl. cross-entropy) stammt aus der Informationstheorie von Claude Shannon (:numref:`sec_mdl-information_theory`). Tritt ein Ergebnis mit Wahrscheinlichkeit $p$ ein, so ist sein *Überraschungswert* (engl. surprisal), auch Informationsgehalt genannt, nach Shannon gleich $-\log p$. Bei natürlichen Logarithmen messen wir ihn in *Nat*, bei $\log_2$ in *Bit*. Dabei gilt $1\textrm{ nat} = 1/\log 2 \approx 1.44\textrm{ bits}$. Für eine wahre Verteilung $\mathbf{p} = (p_1, \ldots, p_q)$ über $q$ Klassen ist der erwartete Überraschungswert ihre *Entropie*:

<!--en:a40ac07f-->
Sie reicht von $0$ (wenn ein Ergebnis sicher eintritt) bis $\log q$ (wenn alle $q$ Ergebnisse gleich wahrscheinlich sind). Angenommen, wir kodieren Ergebnisse aus $\mathbf{p}$ mit einem Code, der für eine Modellverteilung $\hat{\mathbf{p}}$ optimiert ist. Dann ist die mittlere Codelänge (der erwartete Überraschungswert) die *Kreuzentropie*:

<!--en:b3653eaa-->
Nach der *Gibbs-Ungleichung* gilt $H(\mathbf{p}, \hat{\mathbf{p}}) \geq H(\mathbf{p})$, mit Gleichheit genau dann, wenn $\hat{\mathbf{p}} = \mathbf{p}$ gilt. Die Differenz ist die Kullback-Leibler-Divergenz $D_{\mathrm{KL}}(\mathbf{p} \,\|\, \hat{\mathbf{p}}) = H(\mathbf{p}, \hat{\mathbf{p}}) - H(\mathbf{p}) \geq 0$. Vergleichen wir $H(\mathbf{p}, \hat{\mathbf{p}})$ mit :eqref:`eq_l_cross_entropy`, so sehen wir: Der Verlust für ein einzelnes Beispiel $l(\mathbf{y}, \hat{\mathbf{y}}) = H(\mathbf{y}, \hat{\mathbf{y}})$ ist genau die Kreuzentropie zwischen der empirischen Labelverteilung $\mathbf{y}$ und der vorhergesagten Verteilung $\hat{\mathbf{y}}$. Die Gibbs-Ungleichung sagt sogar etwas über Eingaben aus, die von sich aus mehrdeutig sind. Denken wir an ein unscharfes Foto, das mit Wahrscheinlichkeit $60\%$ ein Hemd und mit Wahrscheinlichkeit $40\%$ ein T-Shirt zeigt. Der erwartete Kreuzentropieverlust erreicht sein Minimum genau dann, wenn das Modell die wahren, kalibrierten Wahrscheinlichkeiten $(0.6, 0.4)$ angibt. Eine übermäßig sichere Vorhersage erhöht den Verlust.

<!--en:59a25acf-->
## Genauigkeit und Konfidenz

<!--en:0fc2b5f0-->
### Die Fehlerrate

<!--en:a41b981a-->
Wir bewerten Klassifikatoren anhand ihrer *Genauigkeit* (engl. accuracy) oder ihrer *Fehlerrate* $= 1 - \textrm{accuracy}$. Für das Training können wir den Null-Eins-Fehler jedoch nicht direkt ableiten. Als *Margin* eines Beispiels bezeichnen wir den Logit seiner wahren Klasse abzüglich des höchsten Logits der übrigen Klassen:

<!--en:4897186e-->
Ob der Klassifikator einen Fehler macht ($m < 0$) oder eine richtige Vorhersage trifft ($m > 0$), beschreibt eine unstetige Stufenfunktion von $m$ (:numref:`fig_mdl-clf-margin-losses`). Ihre Ableitung ist überall dort null, wo sie existiert. Damit fehlt dem Gradientenabstieg jedes Lernsignal. Die Kreuzentropie ersetzt die flache Stufe durch eine glatte, streng monoton fallende, konvexe Ersatzfunktion ($\log(1 + e^{-m})$ für $q = 2$). In Bit gemessen, also nach Division durch $\log 2$, ist sie eine obere Schranke für den Null-Eins-Fehler. Für jedes Beispiel liefert sie einen von null verschiedenen Gradienten.

<!--en:80eca7e9-->
![Null-Eins-Fehler bei der Klassifikation und Kreuzentropieverlust für zwei Klassen (in Bit, $\log_2(1 + e^{-m})$) als Funktionen der Margin $m = o_y - \max_{j \neq y} o_j$.](../img/mdl-clf-margin-losses.svg)

<!--en:0ae5391e-->
### Kalibrierung und Temperatur

<!--en:10acf908-->
Die glatte Kurve in :numref:`fig_mdl-clf-margin-losses` hat eine bemerkenswerte Eigenschaft: Die Kreuzentropie sinkt auch dann noch, wenn bereits $m > 0$ gilt und die Klassifikation stimmt. Ein Trainingsdatensatz heißt *linear separierbar*, wenn es Parameter $(\mathbf{W}, \mathbf{b})$ gibt, die jedes Trainingsbeispiel korrekt klassifizieren ($m^{(i)} > 0$ für alle $i$). Multiplizieren wir dann $(\mathbf{W}, \mathbf{b})$ mit einem beliebigen Faktor $c > 1$, wächst jede Margin um den Faktor $c$. So lässt sich der Trainingsverlust beliebig nahe an null bringen (Übung :exref:`ex_softmax-regression-separable`). Alle vorhergesagten Wahrscheinlichkeiten nähern sich $0$ oder $1$, ohne dass sich die Entscheidungsgrenze verschiebt.

<!--en:6c19b57a-->
Allgemeiner können wir alle Logits durch eine positive *Softmax-Temperatur* $T > 0$ dividieren, bevor wir die Softmax-Funktion anwenden:

<!--en:26c6de6a-->
Für $T \to 0$ konzentriert sich die Verteilung auf den One-Hot-Vektor der Klasse mit dem größten Logit. Sie nähert sich dagegen der Gleichverteilung $(1/q, \ldots, 1/q)$, wenn $T \to \infty$ gilt (:numref:`fig_mdl-clf-temperature`). Die Reihenfolge der Logits und damit die Genauigkeit des Klassifikators bleiben gleich. Ob die vorhergesagten Wahrscheinlichkeiten eines trainierten Modells mit der tatsächlich beobachteten Genauigkeit übereinstimmen (*Kalibrierung*, engl. calibration), prüfen wir in :numref:`sec_classification` empirisch anhand zurückgehaltener Daten.

<!--en:fca0e31e-->
![Softmax-Wahrscheinlichkeiten für die Logits $\mathbf{o} = (1.0, 2.2, 0.3)$ bei drei Temperaturen: $T = 0.25$ (stark konzentriert), $T = 1$ (herkömmliche Softmax-Funktion) und $T = 4$ (nahezu gleichverteilt).](../img/mdl-clf-temperature.svg)

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3462b89d-->
Die Softmax-Regression sagt ungeordnete diskrete Kategorien vorher. Sie berechnet für jede Klasse einen affinen Logit, $\mathbf{o} = \mathbf{W}^\top\mathbf{x} + \mathbf{b}$, und normiert die Logits zu Wahrscheinlichkeiten $\hat{y}_j = \exp(o_j)/\sum_k \exp(o_k)$. Dadurch zerfällt der Eingaberaum in konvexe polyedrische Entscheidungsregionen, die durch Hyperebenen begrenzt sind. Die Softmax-Funktion hängt nur von den paarweisen Logit-Differenzen $o_i - o_j = \log(\hat{y}_i/\hat{y}_j)$ ab. Daher ist sie invariant gegenüber einer gemeinsamen Verschiebung aller Einträge von $\mathbf{o}$. Für $q = 2$ ergibt sich die binäre logistische Regression. Maximieren wir die Likelihood der Kategorialverteilung für die One-Hot-Zielvektoren $\mathbf{y}$, erhalten wir den konvexen Kreuzentropieverlust $l(\mathbf{y}, \hat{\mathbf{y}}) = -\log \hat{y}_y = \log\sum_k \exp(o_k) - o_y$. In Bit gemessen ist er eine glatte obere Schranke für den Null-Eins-Fehler der Klassifikation. Sein Gradient nach den Logits vereinfacht sich zum Residuum $\hat{\mathbf{y}} - \mathbf{y}$. Dividieren wir die Logits durch eine Temperatur $T > 0$, lässt sich die Konfidenz der Vorhersage zwischen der deterministischen Wahl des Maximums und einer Gleichverteilung einstellen. Klassenreihenfolge und Genauigkeit bleiben dabei gleich.

<!--en:2294243a-->
## Übungen

<!--en:82caaca3-->
1. **Softmax-Funktion und Sigmoidfunktion.** Nach :eqref:`eq_softmax_to_sigmoid` ergibt sich die logistische Regression aus der Softmax-Regression für zwei Klassen, wenn man einen redundanten Logit entfernt. Sie erhalten von einem Kollegen ein angepasstes logistisches Modell $P(y = 1 \mid x) = \sigma(\hat\beta_0 + \hat\beta_1 x)$ mit $\hat\beta_0 = 2$ und $\hat\beta_1 = -1$.
    1. Charakterisieren Sie alle Koeffizientenpaare $(w_1, b_1)$ und $(w_2, b_2)$ eines Softmax-Modells mit zwei Klassen, deren Vorhersagen für jedes $x$ mit denen des logistischen Modells übereinstimmen. Bestimmen Sie das Paar mit $o_2 \equiv 0$.
    1. Zeigen Sie, dass beim Gradientenabstieg für den Kreuzentropieverlust des Softmax-Modells mit zwei Klassen sowohl $w_1 + w_2$ als auch $b_1 + b_2$ konstant bleiben. Angenommen, der Gradientenabstieg beginnt bei $w_1 = w_2 = b_1 = b_2 = 0$ und konvergiert auf den Trainingsdaten Ihres Kollegen gegen Koeffizienten, die das logistische Modell reproduzieren. Bestimmen Sie, welches Paar er erreicht.

    *Nach James, Witten, Hastie und Tibshirani, [An Introduction to Statistical Learning](https://www.statlearning.com/), Übung 4.12.*

<!--en:727f8afa-->
1. **Hesse-Matrix beim Softmax-Modell.** Schreiben Sie wie in :numref:`subsec_softmax_and_derivatives` den Kreuzentropieverlust als Funktion der Logits, $l(\mathbf{y}, \mathbf{o}) = g(\mathbf{o}) - \mathbf{y}^\top \mathbf{o}$ mit $g(\mathbf{o}) = \log \sum_k \exp(o_k)$.
    1. Berechnen Sie die Hesse-Matrix $\partial^2_{\mathbf{o}}\, l(\mathbf{y}, \mathbf{o})$ und zeigen Sie, dass sie unabhängig von $\mathbf{y}$ ist.
    1. Es sei $\mathbf{e}_Y$ der One-Hot-Vektor einer Klasse $Y$, die aus $\mathrm{softmax}(\mathbf{o})$ gezogen wird. Berechnen Sie die Kovarianzmatrix von $\mathbf{e}_Y$ und zeigen Sie, dass sie gleich der Hesse-Matrix aus dem vorigen Schritt ist.
    1. Folgern Sie, dass $g$ konvex ist und dass der Kreuzentropieverlust konvex in den Logits ist.
   :exlabel:`ex_softmax-regression-hessian`

<!--en:9bc244ce-->
1. **Bradley-Terry-Modell.** Das [Bradley-Terry-Modell](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) :cite:`Bradley.Terry.1952` erfasst Präferenzen mithilfe eines logistischen Modells. Eine Person wählt zwischen einem Apfel und einer Orange. Die Wahrscheinlichkeiten für die beiden Optionen ergeben sich aus der Softmax-Funktion der Scores $o_{\textrm{apple}}$ und $o_{\textrm{orange}}$. Fügen Sie nun die Standardoption hinzu, keines von beiden zu wählen. Ihr Score sei fest vorgegeben: $o_{\textrm{none}} = 0$.
    1. Untersuchen Sie, ob sich die Wahlwahrscheinlichkeiten ändern, wenn Sie zu $o_{\textrm{apple}}$ und $o_{\textrm{orange}}$ dieselbe Konstante addieren, zunächst ohne und dann mit der Standardoption. Folgern Sie, welche Scores sich in jedem der Modelle aus den beobachteten Entscheidungen bestimmen lassen.
    1. Vergleichen Sie das Verhältnis $P(\textrm{apple}) / P(\textrm{orange})$ mit und ohne Standardoption. Beschreiben Sie eine Entscheidungssituation, in der sich dieses Verhältnis bei realen Personen ändern würde.

<!--en:7e67f4b0-->
1. **Log-Sum-Exp.** Die Funktion $\mathrm{lse}(a, b) = \log(\exp(a) + \exp(b))$ ist die Log-Sum-Exp-Funktion aus :numref:`subsec_softmax_and_derivatives` für zwei Argumente. Ihr Gradient ist somit die Softmax-Funktion an der Stelle $(a, b)$.
    1. Beweisen Sie
       $$\max(a, b) < \mathrm{lse}(a, b) \leq \max(a, b) + \log 2,$$
       wobei rechts genau dann Gleichheit gilt, wenn $a = b$ ist. Zeigen Sie, dass der Abstand zu $\max(a, b)$ beliebig klein werden kann. Hinweis: Setzen Sie ohne Beschränkung der Allgemeinheit $b = 0$ und $a \geq 0$.
    1. Beweisen Sie für jedes $\lambda > 0$ die Ungleichung
       $$\max(a, b) < \lambda^{-1} \mathrm{lse}(\lambda a, \lambda b) \leq \max(a, b) + \lambda^{-1} \log 2,$$
       und folgern Sie daraus
       $\lambda^{-1} \mathrm{lse}(\lambda a, \lambda b) \to \max(a, b)$
       für $\lambda \to \infty$.
    1. Konstruieren Sie analog ein glattes Minimum.
    1. Erweitern Sie die Definition und beide Schranken auf $q$ Zahlen. Zeigen Sie, dass der Abstand zu $\max_i a_i$ höchstens $\log q$ beträgt.
   :exlabel:`ex_softmax-regression-realsoftmax`

<!--en:d05ef142-->
1. **Temperatur.** Es sei $P = \mathrm{softmax}(\mathbf{o})$. Bei der Temperatur $T$ sei die Verteilung durch $\mathrm{softmax}(\mathbf{o}/T)$ gegeben. Wir wählen eine weitere Verteilung $Q$ mit $Q(i) \propto P(i)^\alpha$ für $\alpha > 0$.
    1. Welcher Wert von $\alpha$ entspricht einer Verdopplung der Temperatur? Welcher einer Halbierung?
    1. Angenommen, genau $s$ der $q$ Logits erreichen den größten Wert. Bestimmen Sie den Grenzwert von $\mathrm{softmax}(\mathbf{o}/T)$ für $T \to 0$.
    1. Es sei $\bar{o} = \frac{1}{q}\sum_j o_j$ der Mittelwert der Logits. Zeigen Sie, dass für große $T$ gilt:
       $$\mathrm{softmax}(\mathbf{o}/T)_i = \frac{1}{q} + \frac{o_i - \bar{o}}{qT} + O(T^{-2}).$$
    1. Bei der Temperaturskalierung passen wir einen einzigen Wert $T$ an. Dazu minimieren wir die Kreuzentropie auf einem zurückgehaltenen Datensatz und halten die Logits des Klassifikators fest. Zeigen Sie, dass diese Kreuzentropie eine konvexe Funktion von $\beta = 1/T$ ist und sich die Anpassung somit auf ein eindimensionales konvexes Problem reduziert. Hinweis: Nutzen Sie die Konvexität von $g$ aus Übung :exref:`ex_softmax-regression-hessian`.
   :exlabel:`ex_softmax-regression-temperature`

<!--en:16f1b8ea-->
1. **Separierbare Daten.** ● Angenommen, es gibt Parameter $(\mathbf{W}, \mathbf{b})$, mit denen der Klassifikator jedes Trainingsbeispiel eindeutig und korrekt klassifiziert: Der Logit der wahren Klasse ist bei jedem Beispiel strikt größer als alle anderen Logits. Betrachten Sie die skalierten Parameter $(c\mathbf{W}, c\mathbf{b})$ für $c > 0$.
    1. Zeigen Sie, dass der Kreuzentropieverlust jedes Trainingsbeispiels mit wachsendem $c$ streng monoton fällt und gegen $0$ geht, wenn $c \to \infty$ gilt.
    1. Folgern Sie, dass der Trainingsverlust keinen Minimierer hat und dass jede Folge von Parametern, deren Trainingsverlust gegen $0$ geht, in der Norm unbeschränkt ist.
    1. Ergänzen Sie den Trainingsverlust um den Strafterm für den Gewichtszerfall aus :numref:`sec_weight_decay`. Wenden Sie ihn sowohl auf die Gewichte als auch auf die Bias-Terme an: $\frac{\lambda}{2}(\|\mathbf{W}\|^2 + \|\mathbf{b}\|^2)$ mit $\lambda > 0$. Zeigen Sie, dass der Verlust mit Strafterm einen Minimierer besitzt.
   :exlabel:`ex_softmax-regression-separable`

<!--en:8b1b1974-->
1. **Quadratischer Fehler der Wahrscheinlichkeiten.** Anstelle der Kreuzentropie könnten wir den quadratischen Verlust $\frac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|^2$ verwenden, um die vorhergesagten Wahrscheinlichkeiten an das One-Hot-Label anzupassen.
    1. Zeigen Sie, dass für die Ableitungen der Softmax-Funktion $\partial \hat{y}_k / \partial o_j = \hat{y}_k(\mathbf{1}(j = k) - \hat{y}_j)$ gilt. Berechnen Sie damit den Gradienten des quadratischen Verlusts nach den Logits.
    1. Schreiben Sie beide Verluste für zwei Klassen als Funktionen der Margin $m$ und berechnen Sie ihre Ableitungen nach $m$. Zeigen Sie, dass für $m \to -\infty$, also bei einem Fehler mit hoher Konfidenz, die Ableitung des quadratischen Verlusts gegen 0 geht, die der Kreuzentropie dagegen gegen $-1$.
    1. Erläutern Sie die Bedeutung dieses Unterschieds für ein Modell, das zu Beginn des Trainings falsche Vorhersagen mit hoher Konfidenz trifft.

<!--en:737c4e99-->
1. [code] **Gumbel-Max-Trick.** ● Angenommen, wir addieren zu jedem Logit unabhängiges Rauschen und wählen als Label die Klasse mit dem höchsten verrauschten Score, $y = \operatorname*{argmax}_i \, (o_i + \gamma_i)$. Gaußsches Rauschen ergibt das Probit-Modell, dessen Klassenwahrscheinlichkeiten sich nicht in geschlossener Form angeben lassen. Eine andere Rauschverteilung liefert genau die Softmax-Wahrscheinlichkeiten. Es seien $\gamma_1, \ldots, \gamma_q$ unabhängige, identisch verteilte Ziehungen aus der [Standard-Gumbel-Verteilung](https://en.wikipedia.org/wiki/Gumbel_distribution) mit der Verteilungsfunktion $P(\gamma \leq z) = \exp(-\exp(-z))$.
    1. Zeigen Sie, dass $P(o_k + \gamma_k \leq z) = \exp(-\exp(o_k)\exp(-z))$ gilt.
    1. Zeigen Sie, dass $P\left(\operatorname{argmax}_i \, (o_i + \gamma_i) = k\right) = \mathrm{softmax}(\mathbf{o})_k$ gilt. Hinweis: Bedingen Sie auf den Wert von $o_k + \gamma_k$ und verwenden Sie die Substitution $u = \exp(-z)$.
    1. Folgern Sie, dass die Skalierung des Rauschens gemäß $\gamma_i \mapsto T\gamma_i$ Ziehungen aus $\mathrm{softmax}(\mathbf{o}/T)$ liefert.
    1. Prüfen Sie Ihr Ergebnis numerisch: Ziehen Sie für $\mathbf{o} = (1.0, 2.2, 0.3)$ und $T \in \{0.25, 1, 4\}$ jeweils $10^5$ Stichproben von $\operatorname{argmax}_i (o_i + T\gamma_i)$. Bestätigen Sie, dass die relativen Häufigkeiten der Label bis auf den Monte-Carlo-Fehler mit $\mathrm{softmax}(\mathbf{o}/T)$ übereinstimmen.
   :exlabel:`ex_softmax-regression-gumbel-max-trick`

<!--en:50084103-->
[Diskussion](https://d2l.discourse.group/t/46)

<!--en:36f0b13e-->
Softmax-Regression: lineare Modelle für Klassenwahrscheinlichkeiten<br>**One-Hot-Kodierung · affine Logits · Softmax-Funktion · Verschiebungsinvarianz · Kreuzentropie · Gradient $\hat{\mathbf{y}} - \mathbf{y}$ · Softmax-Temperatur**

<!--en:51e06648-->
::: {.slide title="Warum skalare Regression bei ungeordneten Kategorien scheitert"}

<!--en:d152545d-->
[Motivation und Ursache des Problems]{.kicker}

<!--en:030ddbb0-->
Schon beim Versuch, Kleidungsfotos mit einer stetigen skalaren Regression in $q = 10$ ungeordnete Kategorien einzuteilen, treten Probleme auf:

<!--en:1e6a103a-->
- **Willkürliche Abstände und Reihenfolge:** Die Vorhersage $\hat{y} = 5.5$ ordnet ein Foto auf halbem Weg zwischen Mantel ($5$) und Sandale ($6$) ein und erzwingt eine künstliche Zahlenordnung.

<!--en:1f91e0b4-->
- **Verzerrte Strafen beim quadratischen Fehler:** Mit $\frac{1}{2}(\hat{y} - y)^2$ kostet die Verwechslung eines T-Shirts ($1$) mit einer Stiefelette ($10$) $(10 - 1)^2 = 81\times$ so viel wie die Verwechslung mit einer Hose ($2$).

<!--en:f8880c0c-->
- **Bedarf an kalibrierten Verteilungen:** Nachfolgende Entscheidungen erfordern einen vollständigen Wahrscheinlichkeitsvektor $\hat{\mathbf{y}} \in \Delta^q$ über alle $q$ möglichen Klassen.

<!--en:cf1d8eda-->
::: {.slide title="One-Hot-Kodierung stellt Kategorien als Verteilungen dar"}

<!--en:6aeaf080-->
[Darstellung der Zielgröße]{.kicker}

<!--en:7633bef6-->
Gegeben sei ein $2 \times 2$-Graustufenbild mit vier Pixelmerkmalen $(x_1, x_2, x_3, x_4)$ und $q = 3$ Klassen: Katze ($1$), Huhn ($2$) und Hund ($3$).

<!--en:8ff99205-->
- **One-Hot-Vektor für das Label:** Klasse $y \in \{1, \ldots, q\}$ erhält den Vektor $\mathbf{y} \in \{0, 1\}^q$ mit $y_y = 1$ und $0$ an allen anderen Stellen: Katze $(1, 0, 0)$, Huhn $(0, 1, 0)$, Hund $(0, 0, 1)$.

<!--en:4014c6b7-->
- **Symmetrische Simplexecken:** Je zwei verschiedene Kategorien haben genau den euklidischen Abstand $\|\mathbf{e}_i - \mathbf{e}_j\|_2 = \sqrt{2}$. So entsteht keine künstliche Ordnung.

<!--en:4dc9e57d-->
- **Speicherbedarf:** In Formeln steht der One-Hot-Vektor $\mathbf{y} \in \{0, 1\}^q$. Der Code speichert den ganzzahligen Index $y \in \{0, \ldots, q-1\}$ mit Speicherbedarf $O(1)$.

<!--en:28932788-->
::: {.slide title="Ein affiner Score (Logit) je möglicher Klasse"}

<!--en:21279914-->
[Die lineare Schicht]{.kicker}

<!--en:d750184d-->
Die Softmax-Regression ordnet jeder Klasse $j \in \{1, \ldots, q\}$ einen eigenen Gewichtsvektor $\mathbf{w}_j \in \mathbb{R}^d$ und einen Bias $b_j \in \mathbb{R}$ zu:

<!--en:a0a2f725-->
- **Unnormierte Klassenscores (Logits):** Für die Eingabe $\mathbf{x} \in \mathbb{R}^d$ erhält Klasse $j$ den affinen Score $o_j = \mathbf{w}_j^\top\mathbf{x} + b_j$.

<!--en:db0b14f8-->
- **Matrixform:** Fasst man die Spalten $\mathbf{w}_j$ in $\mathbf{W} \in \mathbb{R}^{d \times q}$ und die Bias-Terme in $\mathbf{b} \in \mathbb{R}^q$ zusammen, ergibt sich:
  $$\mathbf{o} = \mathbf{W}^\top \mathbf{x} + \mathbf{b}$$

<!--en:45cff44f-->
- **Parameteranzahl:** Für ein $2\times 2$-Bild ($d=4, q=3$) genügen $4\times 3 + 3 = 15$ Parameter, für ein $28\times 28$-Bild aus Fashion-MNIST ($d=784, q=10$) sind es $7{,}840 + 10 = 7{,}850$ Parameter.

<!--en:051da6b3-->
::: {.slide title="Die Softmax-Funktion bildet Logits auf Wahrscheinlichkeiten ab"}

<!--en:2c199c46-->
[Normierung der Wahrscheinlichkeiten]{.kicker}

<!--en:3cedb8df-->
Die **Softmax-Funktion** exponentiert jeden Logit und dividiert die erhaltenen Werte durch ihre Summe:

<!--en:0b0aa5ec-->
- **Rechenbeispiel:** Exponentieren der Logits $\mathbf{o} = (1.0, 2.2, 0.3)$ ergibt $(2.72, 9.03, 1.35)$ (Summe $13.09$). Daraus folgt $\hat{\mathbf{y}} = (0.21, 0.69, 0.10)$.

<!--en:16ee7dbc-->
- **Punkt im Wahrscheinlichkeitssimplex und Erhalt der Reihenfolge:** Für alle Klassen gilt $\hat{y}_j \in (0, 1)$, außerdem $\sum_j \hat{y}_j = 1$ und $\operatorname{argmax}_j \hat{y}_j = \operatorname{argmax}_j o_j$.

<!--en:e1392bd8-->
::: {.slide title="Verschiebungsinvarianz und logarithmierte Wahrscheinlichkeitsverhältnisse"}

<!--en:c0b4c506-->
[Eigenschaften der Softmax-Funktion]{.kicker}

<!--en:172408ed-->
Die Exponentialfunktion macht aus einer additiven Verschiebung einen gemeinsamen Faktor, der sich herauskürzt:

<!--en:0af42c52-->
- **Logit-Differenzen sind logarithmierte Wahrscheinlichkeitsverhältnisse:** Es gilt $o_i - o_j = \log(\hat{y}_i / \hat{y}_j)$. Für Huhn ($2.2$) gegenüber Katze ($1.0$) ergibt sich damit das Wahrscheinlichkeitsverhältnis $e^{1.2} \approx 3.3\times$.

<!--en:830d2ff2-->
- **Schutz vor Überlauf:** Nach Abzug von $c = \max_k o_k$ ist jeder Exponent $\le 0$. Das verhindert einen Überlauf in `float32` ($o > 88.7$).

<!--en:fc9c5e25-->
::: {.slide title="Bei zwei Klassen ergibt die Softmax-Funktion die logistische Regression"}

<!--en:59cb1be9-->
[Zusammenhang mit der binären Klassifikation]{.kicker}

<!--en:295d155f-->
Für $q = 2$ folgt aus der Verschiebungsinvarianz, dass die beiden Wahrscheinlichkeiten allein von der skalaren Differenz $o = o_1 - o_2$ abhängen:

<!--en:4cd53c4e-->
- **Äquivalenz zur logistischen Sigmoidfunktion:** Es gilt $\sigma(o) = (1 + e^{-o})^{-1}$ und $\hat{y}_2 = 1 - \hat{y}_1 = \sigma(-o)$.

<!--en:5e011d26-->
- **Redundante Parametrisierung:** Die Parameter $(\mathbf{w}_1, b_1), (\mathbf{w}_2, b_2)$ lassen sich zu einem einzigen Differenzvektor $(\mathbf{w}_1 - \mathbf{w}_2, b_1 - b_2)$ mit den Log-Odds $o = \log(\hat{y}_1 / \hat{y}_2)$ zusammenfassen.

<!--en:b9f2713f-->
::: {.slide title="Vektorisierung der Softmax-Funktion für einen Minibatch"}

<!--en:aea699a2-->
[Matrixform für den gesamten Minibatch]{.kicker}

<!--en:05d61d2a-->
Der Code verarbeitet einen Minibatch aus $n$ Beispielen mit je $d$ Merkmalen in einer einzigen Matrixmultiplikation (GEMM) und wendet anschließend die Softmax-Funktion zeilenweise an:

<!--en:3db3eec2-->
- **Tensorformen:** $\mathbf{X} \in \mathbb{R}^{n \times d}$ mal $\mathbf{W} \in \mathbb{R}^{d \times q}$ plus $\mathbf{b} \in \mathbb{R}^{1 \times q}$ per Broadcasting ergibt $\mathbf{O} \in \mathbb{R}^{n \times q}$.

<!--en:40a5e5f4-->
- **Zeilenweise Normierung:** Die exponentierten Werte jeder Zeile $i$ durch ihre eigene Normierungssumme $\sum_{k=1}^q \exp(O_{ik})$ dividieren (`axis=1, keepdims=True`).

<!--en:eca08435-->
::: {.slide title="Geometrie der Entscheidungsregionen der Softmax-Regression"}

<!--en:a376ac07-->
[Lineare Entscheidungsgrenzen]{.kicker}

<!--en:2387ae60-->
Da die Softmax-Funktion die Reihenfolge der Logits erhält, gewinnt Klasse $j$ überall dort, wo $o_j \ge o_k$ für alle $k \neq j$ gilt:

<!--en:9a9bdfdf-->
- **Schnitt von Halbräumen:** Ausgeschrieben lautet die Bedingung $o_j \ge o_k$ gerade $(\mathbf{w}_j - \mathbf{w}_k)^\top\mathbf{x} + (b_j - b_k) \ge 0$. Jede Entscheidungsregion ist also ein **konvexes Polyeder** mit ebenen Seitenflächen, das aus dem Schnitt von $q - 1$ Halbräumen entsteht.

<!--en:fec20a2a-->
- **Hyperebene zwischen zwei Klassen:** Die benachbarten Klassen $j$ und $k$ grenzen entlang der Hyperebene aneinander, auf der ihre Logits gleich sind ($o_j = o_k$).

<!--en:cf1fd0c7-->
- **Voraussetzung:** Konkurrierende Softmax-Ausgaben setzen *einander ausschließende, ungeordnete* Klassen voraus. Bei der Mehrlabelklassifikation verwendet man je Label eine unabhängige binäre Sigmoidfunktion.

<!--en:215f86e4-->
::: {.slide title="Maximum-Likelihood führt zum Kreuzentropieverlust"}

<!--en:8661b186-->
[Probabilistische Herleitung des Verlusts]{.kicker}

<!--en:979f3d60-->
Die negative Log-Likelihood für die Beobachtung der wahren Klasse $y$ (One-Hot-Vektor $\mathbf{y} \in \{0, 1\}^q$) unter dem Modell $\hat{y}_j = P(y = j \mid \mathbf{x})$ ist der **Kreuzentropieverlust**:

<!--en:f5b7b7f0-->
- **Verhalten an den Rändern:** Es gilt $l \to 0$ für $\hat{y}_y \to 1$. Der Verlust wächst gegen $+\infty$ für $\hat{y}_y \to 0$ und bestraft damit Fehler mit hoher Konfidenz besonders stark.

<!--en:e0363296-->
- **Rechenbeispiel für $(0.21, 0.69, 0.10)$:** Ist das Tier ein Huhn ($y = 2$), beträgt der Verlust $-\log 0.69 \approx 0.37\text{ nats}$. Ist es ein Hund ($y = 3$), steigt der Verlust auf das Sechsfache ($-\log 0.10 \approx 2.30\text{ nats}$).

<!--en:53e48b49-->
::: {.slide title="Warum der quadratische Fehler bei Softmax-Wahrscheinlichkeiten versagt"}

<!--en:cf6aff1d-->
[Optimierungsproblem: verschwindende Gradienten bei groben Fehlern]{.kicker}

<!--en:c06df28a-->
Bei der Minimierung des quadratischen Fehlers $\frac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ der Ausgaben der Softmax-Funktion stockt die Optimierung gerade bei Fehlentscheidungen mit hoher Konfidenz:

<!--en:430d20bf-->
- **Sättigung der Jacobi-Matrix der Softmax-Funktion:** Die Ableitung $\partial \hat{y}_k / \partial o_j = \hat{y}_k(\mathbf{1}_{j=k} - \hat{y}_j)$ enthält den Faktor $\hat{y}_k$. Geht die Wahrscheinlichkeit der wahren Klasse gegen null ($\hat{y}_y \to 0$), verschwinden ihre Ableitungen nach den Logits.

<!--en:0fe7cb75-->
- **Logarithmus und Exponentialfunktion in der Kreuzentropie:** Für die Margin $m = o_y - o_{\text{other}} \ll 0$ bei zwei Klassen geht der Gradient des quadratischen Verlusts gegen null: $-2\sigma(m)(1-\sigma(m))^2 \to 0$. Der Gradient der Kreuzentropie $-(1-\sigma(m)) \to -1$ wirkt dem Fehler dagegen mit dem vollen Betrag eins entgegen.

<!--en:80ede376-->
::: {.slide title="Log-Sum-Exp-Darstellung und Residuum als Gradient"}

<!--en:b583b4e5-->
[Exakte Herleitung des Gradienten]{.kicker}

<!--en:e005eb53-->
Mit der **Log-Sum-Exp-Funktion** lässt sich die Kreuzentropie direkt durch die Logits $\mathbf{o}$ ausdrücken:

<!--en:7c84373d-->
Die Ableitungen nach den einzelnen Logits $o_j$ ergeben zusammen genau den Residuenvektor:

<!--en:9da05788-->
- **Vorhersage minus Beobachtung:** Für $\hat{\mathbf{y}} = (0.21, 0.69, 0.10)$ und das wahre Label Huhn $\mathbf{y} = (0, 1, 0)$ ergibt sich $\nabla_{\mathbf{o}}\, l = (0.21, -0.31, 0.10)$. Ein Abstiegsschritt erhöht den Logit für das Huhn und senkt die übrigen Logits proportional zu ihren Wahrscheinlichkeiten.

<!--en:6e109b2e-->
::: {.slide title="Gradienten nach den Gewichten und globale Konvexität"}

<!--en:65b1925a-->
[Kettenregel für die Modellparameter]{.kicker}

<!--en:d907fa48-->
Backpropagation durch $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ liefert die Gradienten nach den Parametern für den Minibatch:

<!--en:690f32bf-->
- **Anpassung der Schablonen:** Für ein einzelnes Beispiel $\mathbf{x}$ zieht $-\nabla_{\mathbf{w}_j} l = (y_j - \hat{y}_j)\mathbf{x}$ die Schablone der wahren Klasse $\mathbf{w}_y$ in Richtung $\mathbf{x}$ und schiebt die konkurrierenden Schablonen $\mathbf{w}_{j \neq y}$ davon weg.

<!--en:6533be40-->
- **Globale Konvexität:** Die Hesse-Matrix nach den Logits $\nabla^2_{\mathbf{o}}\, l = \mathrm{diag}(\hat{\mathbf{y}}) - \hat{\mathbf{y}}\hat{\mathbf{y}}^\top \succeq \mathbf{0}$ ist die Kovarianzmatrix einer Kategorialverteilung. Damit ist $L(\mathbf{W}, \mathbf{b})$ global konvex.

<!--en:07b3daa1-->
::: {.slide title="Informationstheorie: Überraschungswert, Entropie und KL-Divergenz"}

<!--en:aada7e2a-->
[Woher die Kreuzentropie ihren Namen hat]{.kicker}

<!--en:88616b71-->
In Shannons Informationstheorie hat ein Ereignis der Wahrscheinlichkeit $p$ den **Überraschungswert** $-\log p$:

<!--en:fa1a23c8-->
- **Zerlegung in Entropie und KL-Divergenz:**
  $$H(\mathbf{p}, \hat{\mathbf{p}}) = -\sum_{j=1}^q p_j \log \hat{p}_j = H(\mathbf{p}) + D_{\mathrm{KL}}(\mathbf{p} \,\|\, \hat{\mathbf{p}})$$

<!--en:6f00544a-->
- **Gibbs-Ungleichung ($D_{\mathrm{KL}} \ge 0$):** Wegen $D_{\mathrm{KL}}(\mathbf{p} \,\|\, \hat{\mathbf{p}}) = 0 \iff \hat{\mathbf{p}} = \mathbf{p}$ ist die erwartete Kreuzentropie genau dann minimal, wenn die vorhergesagten Wahrscheinlichkeiten mit der wahren bedingten Verteilung übereinstimmen: $\hat{\mathbf{p}} = \mathbf{p}$ (strikt propere Bewertungsregel).

<!--en:6dd172eb-->
::: {.slide title="Kreuzentropie als glatte Ersatzfunktion für den Null-Eins-Fehler"}

<!--en:4912f6ba-->
[Genauigkeit und differenzierbare Zielfunktion beim Training]{.kicker}

<!--en:72cb2216-->
Die **Margin** eines Beispiels bei der Klassifikation ist der Logit der wahren Klasse abzüglich des höchsten konkurrierenden Logits, also $m = o_y - \max_{j \neq y} o_j$:

<!--en:31c11dd5-->
- **Fehlende Differenzierbarkeit des Null-Eins-Fehlers:** Für $m < 0$ irrt der Klassifikator, für $m > 0$ liegt er richtig. Überall, wo die Ableitung dieser Stufenfunktion existiert, hat sie den Wert $0$. Damit liefert der Gradient kein Lernsignal.

<!--en:011d2bcb-->
- **Konvexe obere Schranke:** Der Kreuzentropieverlust für zwei Klassen in Bit, $\log_2(1 + e^{-m})$, ist eine glatte, streng monoton fallende, konvexe obere Schranke für den $0\text{--}1$-Fehler.

<!--en:950c63bb-->
- **Fehlerschranke:** Jedes falsch klassifizierte Beispiel ($m \le 0$) verursacht einen Kreuzentropieverlust von mindestens $\log 2 \approx 0.693\text{ nats}$ ($1\text{ bit}$).

<!--en:e9b79c77-->
::: {.slide title="Separierbare Daten, unbeschränkte Gewichte und Temperatur"}

<!--en:e803cd61-->
[Konfidenz und Entscheidungsgrenzen]{.kicker}

<!--en:00b8df06-->
Da die Kreuzentropie auch für $m > 0$ weiter sinkt, treibt ihre Minimierung die vorhergesagten Wahrscheinlichkeiten in Richtung $0$ und $1$, selbst wenn bereits alle Entscheidungen richtig sind:

<!--en:48caf8af-->
- **Unbeschränkt wachsende Gewichte bei linear separierbaren Daten:** Klassifiziert $(\mathbf{W}, \mathbf{b})$ jeden Trainingspunkt richtig ($m^{(i)} > 0$), so geht bei der Skalierung $(c\mathbf{W}, c\mathbf{b})$ mit $c \to \infty$ der Verlust gegen $0$. Die Entscheidungsgrenze bleibt dabei gleich. Mit Gewichtszerfall ($\frac{\lambda}{2}\|\mathbf{W}\|_F^2$) gibt es wieder einen endlichen Minimierer.

<!--en:2ab1d826-->
- **Temperaturskalierung:** Vor Anwendung der Softmax-Funktion alle Logits durch $T > 0$ dividieren:
  $$\hat{\mathbf{y}}(T) = \mathrm{softmax}(\mathbf{o} / T)$$
  Für $T \to 0$ konzentriert sich die Verteilung auf das Maximum, für $T \to \infty$ nähert sie sich der Gleichverteilung $(1/q, \ldots, 1/q)$. Dabei bleiben $\operatorname{argmax}_j o_j$ und die Genauigkeit gleich.

<!--en:a7ef2c68-->
::: {.slide title="Verständnisfragen zu Verschiebungsinvarianz und Kreuzentropie"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:289f582b-->
Die folgenden drei Fragen lassen sich mit den Eigenschaften der Softmax-Funktion und der Kreuzentropie beantworten:

<!--en:f8c9cf28-->
1. **Verschiebungsinvarianz:** Es gilt $\mathrm{softmax}(2, 0, 0) \approx (0.79, 0.11, 0.11)$, da $e^2 \approx 7.39$ ist. Was ergibt $\mathrm{softmax}(102, 100, 100)$, und um welchen Faktor übersteigt die Wahrscheinlichkeit von Klasse $1$ die von Klasse $2$?

<!--en:99d4cc00-->
2. **Kalibrierte Vorhersagen (Gibbs-Ungleichung):** Mehrdeutige Bilder zeigen in $60\%$ der Fälle Hemden und in $40\%$ der Fälle T-Shirts. Welche konstante Vorhersage erzielt die geringere erwartete Kreuzentropie: $(0.6, 0.4)$ oder $(0.9, 0.1)$?

<!--en:bd9181db-->
3. **Fehlerschranke aus dem Ersatzverlust:** Ein Klassifikator erreicht über alle Trainingsbilder einen mittleren Kreuzentropieverlust von $0.10\text{ nats}$. Wie viele der $60{,}000$ Trainingsbilder kann er höchstens falsch klassifizieren?

<!--en:19bc6c25-->
**Lösungen:** **(1)** Nach Abzug von $100$ von jedem Logit ergibt sich $\mathrm{softmax}(2, 0, 0) \approx (0.79, 0.11, 0.11)$ mit dem Wahrscheinlichkeitsverhältnis $e^{102-100} = e^2 \approx 7.39$. **(2)** Die Vorhersage $(0.6, 0.4)$ erzielt $H(\mathbf{p}) = -0.6\log 0.6 - 0.4\log 0.4 \approx 0.67\text{ nats}$. Das ist günstiger als $(0.9, 0.1)$ mit $-0.6\log 0.9 - 0.4\log 0.1 \approx 0.98\text{ nats}$. **(3)** Der Gesamtverlust beträgt $60{,}000 \times 0.10 = 6{,}000\text{ nats}$. Jeder Fehler ($m \le 0$) kostet mindestens $\log 2 \approx 0.693\text{ nats}$. Damit sind höchstens $6{,}000 / 0.693 \approx 8{,}656$ Fehler möglich ($14.4\%$).

<!--en:a25aa511-->
::: {.slide title="Zusammenfassung der Softmax-Regression"}

<!--en:27de615d-->
[Abschnittsüberblick]{.kicker}

<!--en:4a627f54-->
| Begriff | Mathematische Form | Eigenschaft oder praktische Bedeutung |

<!--en:3096746b-->
| **One-Hot-Zielvektor** | $\mathbf{y} \in \{0, 1\}^q,\; y_y = 1,\; \sum_j y_j = 1$ | Gleicher Abstand $\sqrt{2}$ zwischen je zwei Klassen, gültige Kategorialverteilung |

<!--en:e01156b6-->
| **Affine Logits für einen Minibatch** | $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b} \in \mathbb{R}^{n \times q}$ | $d \cdot q + q$ Parameter ($7{,}850$ bei Fashion-MNIST), Entscheidungsgrenzen sind Hyperebenen |

<!--en:d4656ee0-->
| **Softmax-Funktion** | $\hat{y}_j = \exp(o_j) / \sum_{k=1}^q \exp(o_k)$ | Erhält die Reihenfolge ($\operatorname{argmax} \hat{y}_j = \operatorname{argmax} o_j$), verschiebungsinvariant ($\mathrm{softmax}(\mathbf{o}+c\mathbf{1}) = \mathrm{softmax}(\mathbf{o})$) |

<!--en:7550af2d-->
| **Log-Odds und Sonderfall $q=2$** | $o_i - o_j = \log(\hat{y}_i / \hat{y}_j),\; \hat{y}_1 = \sigma(o_1 - o_2)$ | Logit-Differenzen sind logarithmierte Wahrscheinlichkeitsverhältnisse. Bei zwei Klassen ergibt sich die binäre logistische Regression. |

<!--en:65d6bb54-->
| **Kreuzentropieverlust** | $l(\mathbf{y}, \hat{\mathbf{y}}) = -\log \hat{y}_y = \log\sum_k \exp(o_k) - o_y$ | Negative Log-Likelihood, konvex in $(\mathbf{W}, \mathbf{b})$, in Bit eine obere Schranke für den $0\text{--}1$-Fehler ($\ge \log 2$ Nat bei Fehlentscheidungen) |

<!--en:299a326f-->
| **Gradienten nach Logits und Gewichten** | $\nabla_{\mathbf{o}} l = \hat{\mathbf{y}} - \mathbf{y},\; \nabla_{\mathbf{W}} L = \frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{Y}}-\mathbf{Y})$ | Vorhersage minus Zielwert, bei Fehlern mit hoher Konfidenz anders als beim quadratischen Fehler kein verschwindender Logit-Gradient |

<!--en:76b48bf1-->
| **Temperaturskalierung** | $\hat{\mathbf{y}}(T) = \mathrm{softmax}(\mathbf{o} / T),\; T > 0$ | Steuert die Konfidenz ($T \to 0$: Maximum, $T \to \infty$: Gleichverteilung) bei gleichbleibender Klassenvorhersage |
