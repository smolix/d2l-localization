<!--en:7ea9f141-->
# Generalisierung

<!--en:b5997d01-->
Die beiden Studentinnen Ellie und Irene bereiten sich mit alten Klausuren auf ihre Abschlussprüfung in Statistik vor. Ellie lernt jede Frage samt der genauen Zahlenantwort auswendig. Bei den Übungsklausuren erreicht sie so 100%. Irene versucht, die Konzepte und Formeln zu verstehen. Wegen gelegentlicher Rechenfehler erreicht sie bei den Übungsaufgaben 92%. Als in der Abschlussprüfung ganz neue Aufgaben gestellt werden, kommt Irene auf 90%, während Ellie kaum zurechtkommt.

<!--en:9313bc46-->
Bei Modellen des maschinellen Lernens zeigt sich derselbe Unterschied. Unser Ziel reicht über gute Ergebnisse auf dem endlichen *Trainingsdatensatz* (engl. training set) hinaus. Wir wollen genaue Vorhersagen für bisher unbekannte Beispiele aus der Grundgesamtheit treffen. Diese Eigenschaft heißt *Generalisierung*. In unserem früheren Experiment zur linearen Regression mit synthetischen Daten haben wir 3 Parameter an 1000 Beispiele angepasst. Auswendiglernen war damit unmöglich, und der Trainingsverlust folgte dem Validierungsverlust eng. Ist eine Modellklasse jedoch flexibel genug, um sich einzelnen Datenpunkten anzupassen, kann ein Trainingsfehler von null die Vorhersagen für neue Daten *verschlechtern*. In diesem Abschnitt definieren wir Trainingsfehler und Generalisierungsfehler. Anhand von Polynomkurven veranschaulichen wir *Unteranpassung* (engl. underfitting) und *Überanpassung* (engl. overfitting). Wir erklären die *Bias-Varianz-Zerlegung* und führen *Validierungsdatensätze* (engl. validation sets) und die *$K$-fache Kreuzvalidierung* (engl. K-fold cross-validation) als Verfahren zur fundierten Modellauswahl ein.

<!--en:cd72bf41-->
## Trainingsfehler und Generalisierungsfehler

<!--en:2c48fcfa-->
Beim überwachten Lernen nehmen wir an, dass sowohl die Trainingsbeispiele als auch die später im Einsatz auftretenden Testbeispiele unabhängig aus derselben Wahrscheinlichkeitsverteilung $P(\mathbf{x}, y)$ stammen. Die Beispiele sind nach dieser Annahme *unabhängig und identisch verteilt* (engl. independent and identically distributed, i.i.d.).

<!--en:1c5db665-->
Gegeben sei ein endlicher Trainingsdatensatz $\mathcal{D} = \{(\mathbf{x}^{(1)}, y^{(1)}), \ldots, (\mathbf{x}^{(n)}, y^{(n)})\}$ mit $n$ Beispielen, die wir unabhängig und identisch verteilt aus $P$ ziehen. Der *Trainingsfehler* (auch *empirisches Risiko*) eines Prädiktors $f$ für die Verlustfunktion $l$ ist der Mittelwert des Verlusts über die Stichprobe $\mathcal{D}$:

<!--en:4bb91793-->
Der *Generalisierungsfehler* (auch *wahres Risiko* oder *Populationsrisiko*) ist dagegen der erwartete Verlust für ein neues Beispiel $(\mathbf{x}, y)$ aus $P$:

<!--en:c430f55b-->
Wir suchen eine Funktion $f$, die das wahre Risiko $R[f]$ minimiert. Wir haben jedoch keinen direkten Zugriff auf die Verteilung $P$, sondern beobachten nur die endliche Stichprobe $\mathcal{D}$. Unser Trainingsalgorithmus minimiert daher $R_{\textrm{emp}}[f]$ (*empirische Risikominimierung*).

<!--en:fc0cd05c-->
Warum kann $R_{\textrm{emp}}[\hat{f}]$ das wahre Risiko $R[\hat{f}]$ so stark unterschätzen? Angenommen, wir legen eine *feste* Funktion $f$ fest, bevor wir $\mathcal{D}$ betrachten. Dann wäre $R_{\textrm{emp}}[f]$ ein erwartungstreuer Schätzer für $R[f]$. Nach dem Gesetz der großen Zahlen nähert er sich mit wachsendem Stichprobenumfang dem wahren Risiko. Bei endlicher Verlustvarianz haben seine Schwankungen die Größenordnung $1/\sqrt{n}$. Unser trainiertes Modell $\hat{f}$ steht jedoch *nicht* unabhängig von $\mathcal{D}$ fest. Wir haben es aus einer ganzen Modellklasse $\mathcal{F}$ so ausgewählt, dass es den Fehler auf $\mathcal{D}$ minimiert. Ist $\mathcal{F}$ flexibel genug, um zufällige Muster im Rauschen von $\mathcal{D}$ zu erfassen, fällt der Trainingsfehler $R_{\textrm{emp}}[\hat{f}]$ gegenüber dem wahren Risiko $R[\hat{f}]$ systematisch zu niedrig aus (siehe :exref:`ex_generalization-optimism`). Um $R[\hat{f}]$ unverzerrt zu messen, müssen wir $\hat{f}$ auf einem unabhängigen, zurückgehaltenen *Testdatensatz* (engl. test set) oder *Validierungsdatensatz* $\mathcal{D}'$ auswerten, der beim Anpassen der Parameter keine Rolle spielte.

<!--en:4a3f835e-->
## Unteranpassung und Überanpassung

<!--en:a1bcfabb-->
Am Vergleich von Trainingsfehler und Validierungsfehler erkennen wir zwei entgegengesetzte Probleme:

<!--en:cd9d45aa-->
- **Unteranpassung (hohe Verzerrung):** Das Modell ist zu starr, um den tatsächlichen Trend in den Daten zu erfassen. Trainingsfehler und Validierungsfehler bleiben daher *beide* hoch und liegen dicht beieinander.

<!--en:634b0b44-->
- **Überanpassung (hohe Varianz):** Gemessen an der Größe des Datensatzes ist das Modell so flexibel, dass es neben dem Signal auch das zufällige Rauschen in den Trainingsbeispielen erfasst. Der Trainingsfehler wird sehr klein, während der Validierungsfehler deutlich höher steigt. Dadurch entsteht eine große *Generalisierungslücke* (engl. generalization gap) $R[\hat{f}] - R_{\textrm{emp}}[\hat{f}]$.

<!--en:086d9497-->
### Kurvenanpassung mit Polynomen

<!--en:b34799c2-->
Schon bei der linearen Regression können wir Unteranpassung, eine ausgewogene Anpassung und extreme Überanpassung beobachten. Nehmen wir an, wir ziehen eine skalare Eingabe $x$ gleichverteilt aus $[-1, 1]$. Das wahre Label ergibt sich aus einer glatten Kurve mit additivem gaußschem Rauschen:

<!--en:91d838d4-->
mit $\sigma = 0.1$. Für den quadratischen Fehler $(\hat{y} - y)^2$ liegt die unvermeidbare Rauschgrenze (engl. noise floor) in der Grundgesamtheit bei $\sigma^2 = 0.01$. Selbst die wahre Funktion $f^*(x)$ hat einen erwarteten quadratischen Fehler von $0.01$, weil sich das Rauschen $\epsilon$ nicht vorhersagen lässt.

<!--en:c2c23688-->
Um eine gekrümmte Kurve mit linearer Regression anzupassen, bilden wir die skalare Eingabe $x$ auf einen Vektor von *Polynommerkmalen* (engl. polynomial features) ab. Seine Einträge sind Monome: $(1, x, x^2, \ldots, x^k)^\top \in \mathbb{R}^{k+1}$. Ein Polynom vom Grad $k$

<!--en:c5d98834-->
ist nichtlinear in $x$, aber **linear in den Gewichten $\mathbf{w} = (w_0, \ldots, w_k)^\top$**. Um ein beliebiges Polynom vom Grad $k$ an $n$ Trainingspunkte anzupassen, bilden wir die Vandermonde-Matrix als Entwurfsmatrix $\mathbf{X} \in \mathbb{R}^{n \times (k+1)}$ mit den Einträgen $X_{ij} = (x^{(i)})^j$. Dann rufen wir `lstsq` auf, wie in :numref:`subsec_linear-regression-analytic-solution`. Erzeugen wir $n = 20$ Trainingspunkte und $1{,}000$ Testpunkte. Damit lässt sich der Testfehler genau schätzen. Vergleichen wir nun drei Polynomgrade: $k = 1$ (eine Gerade), $k = 3$ (eine kubische Kurve) und $k = 19$ (ein Polynom vom Grad 19 mit $20$ Parametern für $20$ Punkte):

<!--en:9b3ff767-->
Im folgenden Code erzeugt `features(x, d)` die Entwurfsmatrix $[1, x, \ldots, x^d]$. Die Funktion `fit_degree` bestimmt die Gewichte exakt nach der Methode der kleinsten Quadrate, und `mse` berechnet den mittleren quadratischen Fehler (engl. mean squared error, MSE). Den Faktor $\frac{1}{2}$ lassen wir weg, sodass die Rauschgrenze bei $\sigma^2 = 0.01$ liegt:

<!--en:9695510b-->
Zeichnen wir die angepassten Kurven vom Grad $1$, $3$ und $19$ zusammen mit den $20$ verrauschten Trainingspunkten und der wahren kubischen Kurve:

<!--en:1d9b90b0-->
Die drei Teilbilder zeigen die beiden klassischen Probleme des überwachten Lernens:

<!--en:d83a0a22-->
- **Grad $k = 1$ (Unteranpassung):** Eine Gerade kann der S-förmigen kubischen Kurve nicht folgen. Sie verfehlt Trainingspunkte und Testpunkte gleichermaßen.

<!--en:9091c11c-->
- **Grad $k = 3$ (passende Modellklasse):** Ein kubisches Polynom hat genau die Flexibilität von $f^*$. Die angepasste Kurve ist von der wahren Kurve kaum zu unterscheiden.

<!--en:7c44acf5-->
- **Grad $k = 19$ (Überanpassung):** Bei $k + 1 = 20$ Parametern und $n = 20$ verschiedenen Trainingspunkten hat das Polynom genügend Freiheitsgrade, um jeden einzelnen Trainingspunkt zu interpolieren (siehe :exref:`ex_generalization-exact-polynomial`). Zwischen den Datenpunkten schwingt die Kurve jedoch so stark, dass der Testfehler um Größenordnungen steigt.

<!--en:7dafdefa-->
Was geschieht bei Überanpassung mit den erlernten Gewichten $w_j$ des Polynoms? Geben wir für jeden der drei Grade den Trainingsfehler, den Testfehler und den größten Gewichtsbetrag $\max_j |\hat{w}_j|$ aus:

<!--en:b0f77b78-->
Die genauen Zahlen schwanken bei jeder Ziehung leicht, doch der Unterschied ist groß. Bei Grad $1$ sind beide Fehler hoch. Der mittlere quadratische Testfehler liegt bei etwa $2$, also bei etwa $200\times$ der Rauschgrenze $\sigma^2 = 0.01$. Bei Grad $3$ sinken Trainingsfehler und Testfehler bis zur Rauschgrenze $\sigma^2 = 0.01$. Das betragsgrößte Gewicht liegt nahe am wahren Leitkoeffizienten $5.6$. Bei Grad $19$ fällt der Trainingsfehler weit unter die Rauschgrenze, praktisch auf null. Der Testfehler wächst dagegen um viele Größenordnungen. Die Gewichte zeigen, warum: Bei Grad $19$ hat das Polynom sehr große positive und negative Koeffizienten. Ihre Beiträge heben sich an den $20$ Trainingseingaben fast genau auf, während die Werte andernorts in die Höhe schießen.

<!--en:50796478-->
Betrachten wir alle Grade von $1$ bis $19$ auf einer logarithmischen Skala, so sehen wir den gesamten Übergang:

<!--en:331518a3-->
Mit wachsendem $k$ sinkt der Trainingsfehler monoton. Jedes zusätzliche Monom $x^{k+1}$ vergrößert den Unterraum der Funktionen, die der Methode der kleinsten Quadrate zur Verfügung stehen. So fällt der Trainingsfehler weit unter $\sigma^2$, wenn $k = 19$ erreicht ist. Der Testfehler beschreibt dagegen eine klassische **U-förmige Kurve**. Mit abnehmender Unteranpassung fällt er steil bis zu seinem Minimum bei $k = 3$ oder einem benachbarten Grad. Dann steigt er um Größenordnungen, weil sich die zusätzlichen Potenzen von $x$ dem Rauschen in den 20 Trainingspunkten anpassen. Die Potenzen $x^{17}$ und $x^{19}$ verlaufen auf $[-1, 1]$ fast identisch. Bei hohen Graden gleicht das Polynom ein sehr großes positives Vielfaches der einen Potenz durch ein sehr großes negatives Vielfaches der anderen aus. So trifft es die 20 verrauschten Punkte. Dieser Instabilität begegnen wir in :numref:`sec_weight_decay` gezielt mit Gewichtszerfall (engl. weight decay).

<!--en:78261cde-->
### Verzerrung und Varianz

<!--en:ab554fd4-->
Warum ergibt sich für den Generalisierungsfehler in Abhängigkeit von der Modellkomplexität eine U-förmige Kurve? Zerlegen wir den erwarteten quadratischen Fehler eines Modells $\hat{f}_{\mathcal{D}}$, das wir auf einem zufälligen Datensatz $\mathcal{D}$ trainieren. Am Testpunkt $x$ gelte $y = f^*(x) + \epsilon$, wobei $\mathbb{E}[\epsilon] = 0$ und $\mathrm{Var}(\epsilon) = \sigma^2$ gelten und $\epsilon$ unabhängig von $\mathcal{D}$ ist. Mit $\bar{f}(x) = \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)]$ bezeichnen wir den mittleren Prädiktor über alle möglichen Trainingsdatensätze der Größe $n$. Im Ausdruck für den erwarteten Fehler addieren und subtrahieren wir $f^*(x)$ und $\bar{f}(x)$. Damit ändert sich der Wert von $\mathbb{E}_{\mathcal{D}, \epsilon}[(\hat{f}_{\mathcal{D}}(x) - y)^2]$ nicht. Da die gemischten Terme verschwinden, ergibt sich die bekannte *Bias-Varianz-Zerlegung*:

<!--en:35f2b920-->
Der erwartete Testfehler jedes Modells ist die Summe dreier nichtnegativer Terme:

<!--en:79ddc690-->
1. **Quadrierte Verzerrung $(\bar{f}(x) - f^*(x))^2$:** Sie misst, wie weit das *im Mittel* angepasste Modell von der wahren Funktion $f^*(x)$ entfernt liegt, selbst wenn wir über unendlich viele Trainingsdatensätze mitteln. Starre Modelle wie eine Gerade ($k = 1$) für eine kubische Kurve haben eine hohe Verzerrung.

<!--en:2280d8c0-->
2. **Varianz $\mathbb{E}_{\mathcal{D}}[(\hat{f}_{\mathcal{D}}(x) - \bar{f}(x))^2]$:** Sie misst, wie stark das angepasste Modell von einer zufälligen Trainingsstichprobe $\mathcal{D}$ zur nächsten schwankt. Zu flexible Modelle ($k = 12$) ändern sich bei jeder neuen Ziehung von 20 Punkten stark und haben daher eine sehr hohe Varianz.

<!--en:0310920f-->
3. **Irreduzibles Rauschen $\sigma^2$:** Das ist die Varianz des Labels $y$ um $f^*(x)$. Kein Prädiktor kann diese Grenze unterschreiten.

<!--en:37273f6a-->
In unserem Polynomexperiment können wir alle drei Terme direkt messen. Halten wir die $20$ Trainingseingaben fest und ziehen wir nur das Rauschen ihrer Labels neu. Die Gewichte der Kleinste-Quadrate-Lösung hängen linear von $\mathbf{y}$ ab, und das Rauschen hat den Mittelwert null. Daher entspricht die mittlere angepasste Funktion $\bar{f}$ der Anpassung an die rauschfreien Labels $f^*(x^{(i)})$. Daraus ergibt sich die quadrierte Verzerrung exakt. Um die Varianz an jedem Testpunkt zu schätzen, passen wir das Modell für $200$ unabhängige Ziehungen des Trainingsrauschens neu an:

<!--en:763db1e2-->
Die Kurven zeigen, wie die U-Form des Testfehlers entsteht. Die quadrierte Verzerrung ist bei den Graden $1$ und $2$ groß und fällt bei Grad $3$ auf null. Erst diese Modellfamilie enthält die wahre kubische Kurve. Die Varianz hingegen wächst mit jedem zusätzlichen Grad. Anfangs liegt sie etwa bei $\sigma^2(k+1)/n$, also bei der Rauschvarianz mal der Anzahl der Parameter pro Beispiel (siehe :exref:`ex_generalization-optimism`). Schließlich übersteigt sie die Rauschgrenze um Größenordnungen. Bei Unteranpassung dominiert die *Verzerrung* den Fehler, bei Überanpassung die *Varianz*. Den idealisierten Zusammenhang aus :numref:`fig_capacity_vs_error` finden wir in der klassischen Statistik und im maschinellen Lernen immer wieder. Mit wachsender Komplexität sinkt der Trainingsfehler stetig. Der Testfehler erreicht sein Minimum bei mittlerer Komplexität und fällt nie unter $\sigma^2$.

<!--en:4dd70eaf-->
![Trainingsfehler und Testfehler in Abhängigkeit von der Modellkomplexität. Links vom Optimum ist das Modell unterangepasst, beide Fehler sind hoch. Rechts davon ist es überangepasst, der Trainingsfehler fällt unter den Rauschpegel und die Generalisierungslücke wächst.](../img/mdl-gen-capacity-error.svg)

<!--en:c3e400e7-->
### Modellkomplexität

<!--en:affdfaae-->
Wie :numref:`fig_capacity_vs_error` schematisch zeigt, sinkt die Verzerrung mit zunehmender Modellkomplexität (Kapazität), während die Varianz steigt. Das Modell generalisiert dort am besten, wo die Summe aus quadrierter Verzerrung und Varianz minimal ist. Bei klassischen parametrischen Modellen bestimmen die Anzahl der Parameter $p$ und die Norm des Gewichtsvektors $\|\mathbf{w}\|$ die Komplexität. Beim Deep Learning (:numref:`sec_generalization_deep`) sehen wir, wie Regularisierung, die Wahl der Architektur und die Größe des Datensatzes den Kurvenverlauf verändern.

<!--en:26c55598-->
### Größe des Datensatzes

<!--en:496a1fa8-->
Neben der Begrenzung der Modellkomplexität gibt es einen unmittelbaren Weg, die Varianz zu verringern: Wir erheben *mehr Trainingsdaten*. Passen wir bei festem Polynomgrad $k$ das Modell an $n = 1{,}000$ statt an $n = 20$ Punkte an, mittelt sich das zufällige Rauschen heraus. Selbst die Koeffizienten der hohen Potenzen lassen sich dann genau bestimmen. Die Schätzvarianz nimmt dabei gemäß $(k + 1)\sigma^2 / n$ ab. Beobachten wir, was bei den Graden $1$, $3$ und $19$ geschieht, wenn wir einen neuen Trainingsdatensatz von $n = 20$ auf $n = 1{,}000$ Beispiele vergrößern:

<!--en:1c35303b-->
Der Unterschied ist deutlich. Ein größerer Datensatz hilft dem überangepassten Polynom vom Grad 19 erheblich. Schon bei $n = 100$ sinkt sein Testfehler um Größenordnungen, und bei $n = 1{,}000$ stimmt die angepasste Kurve mit der wahren kubischen Kurve überein. Die unterangepasste Gerade ändert sich hingegen kaum. Ihr Testfehler bleibt bei etwa $2$. Mehr Daten verringern die *Varianz*, können aber die *Verzerrung* nicht beheben. Kein Modell ist an sich „zu komplex“. Es ist nur *im Verhältnis zum Signal und zu den Daten, die seine Anpassung einschränken*, zu komplex. Dieser Zusammenhang mit der Datenmenge ist ein wesentlicher Grund für den Aufschwung des Deep Learning, als große digitale Datensätze allgemein verfügbar wurden. In :numref:`sec_generalization_deep` sehen wir allerdings, dass die klassische Kurve bei tiefen Netzen in wichtigen Punkten anders verläuft.

<!--en:74c55b18-->
## Modellauswahl

<!--en:c9b2f4f2-->
### Validierungsdatensätze

<!--en:1cc150eb-->
In unserem Experiment mit synthetischen Daten kannten wir $f^*(x)$ und konnten nach Belieben 1000 Testpunkte erzeugen. Wie wählen wir bei einer realen Aufgabe Hyperparameter wie den Polynomgrad $k$, die Lernrate $\eta$ oder die Stärke $\lambda$ des Gewichtszerfalls?

<!--en:07709ba8-->
Wir dürfen Hyperparameter nie durch Minimierung des Trainingsfehlers auswählen. Der Trainingsfehler bevorzugt stets das komplexeste Modell, das am ehesten auswendig lernt ($k = 19$). Wir sollten die Hyperparameter auch nicht wiederholt anhand des endgültigen *Testdatensatzes* abstimmen. Sonst beeinflusst das Rauschen in den Testdaten unsere Entscheidungen (:numref:`chap_classification_generalization`). Deshalb teilen wir die verfügbaren Daten in drei disjunkte Mengen auf:

<!--en:f7043caa-->
1. **Trainingsdatensatz:** Für jede infrage kommende Einstellung der Hyperparameter passen wir die Modellparameter $(\mathbf{w}, b)$ an diese Daten an.

<!--en:5236b1b2-->
2. **Validierungsdatensatz:** Hier vergleichen wir die Einstellungen der Hyperparameter und wählen die beste aus (*Modellauswahl*).

<!--en:e184c8d0-->
3. **Testdatensatz:** Diesen Datensatz halten wir unter Verschluss und werten ihn ganz am Ende *einmal* aus. So erhalten wir eine erwartungstreue Schätzung des Generalisierungsfehlers des ausgewählten Modells.

<!--en:aeef62b2-->
### Kreuzvalidierung

<!--en:37bff4b9-->
Bei einem kleinen Datensatz, etwa $n = 20$ oder $n = 100$ Krankenakten, hat eine feste Aufteilung mit 20% Validierungsdaten einen Nachteil: Es bleiben zu wenige Beispiele für das Training und zu wenige für eine verlässliche Validierung. Bei der *$K$-fachen Kreuzvalidierung* (:numref:`fig_kfold_cv`) erfüllt jedes Beispiel beide Aufgaben. Wir teilen den Datensatz zufällig in $K$ gleich große, disjunkte *Gruppen* (engl. folds) $\mathcal{D}_1, \ldots, \mathcal{D}_K$ auf, meist mit $K = 5$ oder $K = 10$. In Runde $i \in \{1, \ldots, K\}$ trainieren wir das Modell auf den $K - 1$ Gruppen in $\mathcal{D} \setminus \mathcal{D}_i$ von Grund auf. Den Validierungsverlust messen wir auf der zurückgehaltenen Gruppe $\mathcal{D}_i$. Aus den $K$ Validierungsverlusten bilden wir den Mittelwert als Ergebnis der $K$-fachen Kreuzvalidierung. Nach der Wahl des besten Hyperparameters trainieren wir das Modell erneut auf allen $n$ Beispielen. Im Grenzfall $K = n$ enthält jede Gruppe genau einen Datenpunkt. Dieses Verfahren heißt *Leave-one-out-Kreuzvalidierung*. Bei der linearen Methode der kleinsten Quadrate lassen sich mit einer bemerkenswerten algebraischen Identität alle $n$ Leave-one-out-Residuen aus einer *einzigen* Anpassung berechnen. Wir müssen das Modell also nicht $n$-mal neu trainieren (siehe :exref:`ex_generalization-leave-one-out-shortcut`).

<!--en:c6002132-->
![Bei der $K$-fachen Kreuzvalidierung (hier $K = 5$) zerfällt der Datensatz in $K$ Gruppen. In jeder Runde dient eine Gruppe zur Validierung (orange), die übrigen $K - 1$ Gruppen dienen zum Training.](../img/mdl-mlp-kfold.svg)

<!--en:61f923f5-->
Reale Datensätze sind häufig bereits sortiert. Deshalb müssen wir die Gruppen aus einer *zufälligen* Permutation bilden. Bei nur vier Punkten pro Gruppe hängt das Ergebnis einer einzelnen Aufteilung stark vom Zufall ab. Daher erzeugt `kfold_splits` im folgenden Code zehn unabhängige zufällige Aufteilungen in $K = 5$ Gruppen (*wiederholte* Kreuzvalidierung). Die Funktion `cv_error` mittelt den Validierungsfehler der Polynomanpassung über alle Gruppen aller Aufteilungen:

<!--en:81726eba-->
Wählen wir den Polynomgrad ausschließlich anhand der $20$ Trainingsbeispiele. Für einen fairen Vergleich bewerten wir jeden Grad auf denselben Gruppen. Prüfen wir dann, wie nahe unsere Wahl dem tatsächlichen Optimum auf dem Testdatensatz kommt, der bei der Kreuzvalidierung unberührt bleibt:

<!--en:74e4b441-->
Zeichnen wir für alle 19 Grade die Kurve der 5-fachen Kreuzvalidierung und die Testfehlerkurve in dasselbe Diagramm:

<!--en:77649078-->
Die wiederholte 5-fache Kreuzvalidierung findet zuverlässig das Minimum der Testfehlerkurve, ohne den Testdatensatz heranzuziehen. Meist wählt sie Grad $3$, gelegentlich einen benachbarten Grad mit fast identischem Testfehler. Nahe dem Minimum unterscheiden sich die Kurven nur um einen kleinen Faktor. Die Kurve der Kreuzvalidierung schwankt etwas stärker, da wir über $20$ statt über $1{,}000$ Punkte mitteln. Bei hohen Graden steigen beide Kurven um Größenordnungen. Der Fehler der Kreuzvalidierung kann sogar früher steigen als der Testfehler. In jeder Runde trainieren wir nur auf $16$ statt auf $20$ Punkten. Daher interpoliert bereits Grad $15$ alle $16$ Trainingspunkte. Validierungspunkte nahe den Rändern von $[-1, 1]$ liegen dann außerhalb des Anpassungsbereichs, wo die Beträge hochgradiger Polynome stark wachsen. Bei einer einzelnen Ziehung von $20$ Punkten kann jedoch jede der beiden Kurven zuerst ansteigen.

<!--en:6198c14f-->
Wie wählen wir $K$? In jeder Runde trainieren wir nur auf dem Anteil $(K - 1) / K$ des Datensatzes. Das angepasste Modell schneidet daher etwas schlechter ab als ein Modell, das wir auf allen $n$ Beispielen trainieren. Die $K$-fache Kreuzvalidierung überschätzt den Fehler also leicht (die genaue Verzerrung berechnen wir in :exref:`ex_generalization-fold-cross-validation`). Mit $K = n$ (*Leave-one-out*-Kreuzvalidierung) verschwindet diese Verzerrung fast vollständig. Im Allgemeinen sind dafür aber $n$ getrennte Anpassungen pro Kandidat nötig. Bei einem instabilen Lernverfahren wie einem Entscheidungsbaum kann ein einziges weggelassenes Beispiel das ganze Modell verändern. Seine $n$ fast identischen Trainingsdatensätze können dann eine stärker schwankende Schätzung liefern als eine gröbere Aufteilung. Bei hohen Kosten pro Anpassung ist deshalb $K = 5$ oder $K = 10$ üblich :cite:`Kohavi.1995`. Die Methode der kleinsten Quadrate ist kostengünstig und stabil. Eine einzige Anpassung liefert alle $n$ Leave-one-out-Residuen, und die angepasste Funktion ändert sich mit den Daten stetig.

<!--en:8ae8eea5-->
Messen wir die Verzerrung und die zufälligen Schwankungen für unser kubisches Polynom. Auf jedem von 500 neuen Trainingsdatensätzen mit je 20 Punkten passen wir das Polynom an alle 20 Punkte an und halten seinen Testfehler fest. Zusätzlich schätzen wir diesen Fehler ausschließlich anhand der 20 Punkte mit drei Verfahren: Leave-one-out ($K = 20$, ein Punkt pro Gruppe), 5-fache Kreuzvalidierung mit einer zufälligen Aufteilung und 5-fache Kreuzvalidierung mit zehn Aufteilungen:

<!--en:cab2d4c8-->
Im Mittel liegt die Leave-one-out-Schätzung nahe am Testfehler. Die Abweichung beträgt höchstens etwa ein Zehntel seines Werts. Beide Schätzwerte der 5-fachen Kreuzvalidierung liegen über der Leave-one-out-Schätzung. Diese Verzerrung nach oben entsteht, weil in jeder Runde nur 16 statt 19 Punkte zum Training dienen. Von den drei Schätzverfahren hat Leave-one-out über die 500 Ziehungen auch die kleinste Standardabweichung. Die verbreitete Annahme, seine Schätzungen schwankten stärker, trifft hier also nicht zu. Zehn Wiederholungen der 5-fachen Aufteilung verringern die Streuung, doch sie bleibt größer als bei Leave-one-out. Ein großer Teil der Streuung hängt davon ab, welche 20 Punkte wir zufällig ziehen, und weniger davon, wie wir sie aufteilen. Bei der Methode der kleinsten Quadrate ist Leave-one-out somit kostengünstig und zugleich das genaueste der drei Verfahren. Bei den Netzen in späteren Kapiteln dauert eine Anpassung Minuten oder Stunden. Dort bleibt eine gröbere Aufteilung die praktikable Wahl.

<!--en:0e4d0f3f-->
Die Wahl eines diskreten Polynomgrads war hier möglich, weil unsere Merkmale eine natürliche Rangfolge $1, x, x^2, \ldots, x^k$ bilden. Was tun wir bei einem Datensatz mit Hunderten ungeordneter Merkmale, wenn sich keine Reihenfolge für das Entfernen einzelner Spalten anbietet? In :numref:`sec_weight_decay` ersetzen wir die diskrete Merkmalsauswahl durch einen kontinuierlichen Strafterm für die Norm der Gewichte.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3be107a9-->
Beim überwachten Lernen minimieren wir das empirische Risiko $R_{\textrm{emp}}[f]$ auf einer endlichen Trainingsstichprobe. Unser Ziel ist ein geringes Populationsrisiko $R[f] = \mathbb{E}[l(f(\mathbf{x}), y)]$ auf bisher unbekannten Daten. Die Optimierung wählt $\hat{f}$ gezielt so, dass das Modell zum Trainingsdatensatz passt. Ist die Modellklasse flexibel genug, um Labelrauschen zu erfassen, fällt $R_{\textrm{emp}}[\hat{f}]$ deshalb systematisch zu niedrig aus. Der erwartete quadratische Testfehler zerfällt in die quadrierte Verzerrung, die Schätzvarianz und das irreduzible Rauschen $\sigma^2$. Starre Modelle sind unterangepasst und haben eine hohe Verzerrung. Zu flexible Modelle sind bei kleinen Stichproben überangepasst und haben eine hohe Varianz. Mehr Trainingsdaten verringern die Varianz proportional zu $1/n$, ohne die Verzerrung zu mindern. Der Trainingsfehler bevorzugt stets das Auswendiglernen, und wer den endgültigen Testdatensatz wiederholt heranzieht, verfälscht die Bewertung. Deshalb müssen wir die Hyperparameter anhand eines zurückgehaltenen Validierungsdatensatzes oder durch $K$-fache Kreuzvalidierung wählen.

<!--en:2294243a-->
## Übungen

<!--en:f38db785-->
1. **Wenn die i.i.d.-Annahme verletzt ist.** Nennen Sie drei Beispiele, in denen die Beobachtungen voneinander abhängen und es daher unangebracht ist, sie als unabhängig und identisch verteilt zu behandeln. Geben Sie jeweils an, ob eine zufällige Aufteilung in Trainingsdaten und Validierungsdaten den Generalisierungsfehler über- oder unterschätzt. Beschreiben Sie eine Aufteilung, die das Problem vermeidet.

<!--en:84574531-->
1. **Exakte Polynomregression.** Zeigen Sie: Sind die $n$ Trainingseingaben $x^{(1)}, \ldots, x^{(n)}$ paarweise verschieden, so passt ein Polynom vom Grad $n-1$ exakt zu den $n$ Labels. Erklären Sie, wo Sie die Verschiedenheit verwenden. Welches Ergebnis liefert die Methode der kleinsten Quadrate, wenn zwei Eingaben übereinstimmen, ihre Labels aber verschieden sind?
   :exlabel:`ex_generalization-exact-polynomial`

<!--en:81da4550-->
1. [code] **Eine bessere Basis für Grad 19.** Die hohen Potenzen $x^{17}$ und $x^{19}$ unterscheiden sich überall auf $[-1, 1]$ um weniger als 0.05. Die Spalten von `features(x, 19)` sind daher nahezu parallel. Die Chebyshev-Polynome $T_0, \ldots, T_{19}$ spannen denselben Funktionenraum auf, bleiben auf dem Intervall aber zwischen $-1$ und $1$.
    1. Treffen Sie zunächst eine Vorhersage: Wir passen ein Polynom vom Grad 19 in der Chebyshev-Basis an die 20 Trainingspunkte an. Welche der drei Größen Trainingsfehler, Testfehler und größter Gewichtsbetrag ändern sich und um wie viel?
    1. Prüfen Sie Ihre Vorhersage in mehreren Durchläufen. Ersetzen Sie für die Trainings- und Testeingaben `features(x, 19)` durch `np.polynomial.chebyshev.chebvander(x, 19)`. Vergleichen Sie die Konditionszahlen der beiden Entwurfsmatrizen (`np.linalg.cond`), die Trainings- und Testfehler sowie die jeweils größten Gewichtsbeträge.
    1. Erklären Sie, warum beide Anpassungen bei exakter Arithmetik dasselbe Polynom ergeben. Welche Ihrer Zahlen unterscheiden sich bei der Rechnung mit float64 und warum? Hilft die bessere Basis dem Polynom vom Grad 19 bei 20 Punkten? Was sagt Ihre Antwort über die Größe der Gewichte als Maß für die Komplexität aus?
   :exlabel:`ex_generalization-chebyshev-basis`

<!--en:dd082c3f-->
1. **Warum der Trainingsfehler zu niedrig ausfällt.** ● Betrachten Sie die Methode der kleinsten Quadrate mit einer festen $n \times p$-Entwurfsmatrix $\mathbf{X}$ von vollem Spaltenrang und den Labels $\mathbf{y} = \mathbf{X}\mathbf{w}^* + \boldsymbol{\epsilon}$. Die Rauschterme sind unabhängig und haben den Mittelwert null und die Varianz $\sigma^2$. Die angepassten Vorhersagen lauten $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ mit $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$.
    1. Zeigen Sie für den erwarteten Trainingsfehler die Gleichung $\mathbb{E}[\|\mathbf{y} - \hat{\mathbf{y}}\|^2]/n = \sigma^2 (n - p)/n$. Hinweis: Es gilt $\mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\boldsymbol{\epsilon}$, und die Spur von $\mathbf{I} - \mathbf{H}$ beträgt $n - p$.
    1. Ziehen Sie für dieselben Eingaben neue Labels $\mathbf{y}' = \mathbf{X}\mathbf{w}^* + \boldsymbol{\epsilon}'$. Zeigen Sie, dass der erwartete Fehler der Vorhersagen $\hat{\mathbf{y}}$ auf diesen Labels $\sigma^2 (n + p)/n$ beträgt. Der Trainingsfehler unterschätzt ihn also um $2\sigma^2 p/n$. Zeigen Sie, dass $\sigma^2 p/n$ die über die $n$ Eingaben gemittelte Varianz der Vorhersagen ist.
    1. Werten Sie beide Ausdrücke für die kubische Anpassung aus :numref:`subsec_polynomial-curve-fitting` aus ($n = 20$, $p = 4$, $\sigma = 0.1$). Vergleichen Sie Ihre Ergebnisse mit den Trainings- und Testfehlern für Grad 3 aus mehreren Durchläufen des Experiments. Erklären Sie die Unterschiede: Wie stark streut der Trainingsfehler eines einzelnen Durchlaufs um seinen Erwartungswert? Warum gilt die Formel aus (b) an den Trainingseingaben exakt, an den Testeingaben aber nicht?
   :exlabel:`ex_generalization-optimism`

<!--en:ba8184ec-->
1. **$K$-fache Kreuzvalidierung.**
    1. Nehmen Sie an, ein auf $m$ Beispielen trainiertes Modell habe den erwarteten Generalisierungsfehler $R(m) = R_\infty + c/m$ mit $c > 0$. In jeder Runde dienen $(K-1)n/K$ Beispiele zum Training. Berechnen Sie die Verzerrung der Schätzung aus der $K$-fachen Kreuzvalidierung gegenüber $R(n)$, dem Fehler des auf allen $n$ Beispielen trainierten Modells. Geben Sie die Verzerrung als Bruchteil von $R(n) - R_\infty$ an und berechnen Sie diesen Bruchteil für $K = 2$, $5$, $10$ und $n$.
    1. Nach der Wahl eines Hyperparameters durch Kreuzvalidierung trainieren wir das endgültige Modell erneut auf allen $n$ Beispielen. Erklären Sie, warum dieser Schritt schwieriger wird, wenn der beste Wert des Hyperparameters von der Größe des Trainingsdatensatzes abhängt. Geben Sie ein Beispiel an.
   :exlabel:`ex_generalization-fold-cross-validation`

<!--en:47c9e25c-->
1. **Abkürzung für Leave-one-out.** ● Die Leave-one-out-Kreuzvalidierung erfordert im Allgemeinen $n$ Modellanpassungen. Betrachten Sie die Methode der kleinsten Quadrate mit der Entwurfsmatrix $\mathbf{X}$, deren Zeilen $\mathbf{x}^{(i)\top}$ sind. Sei $\mathbf{A} = \mathbf{X}^\top\mathbf{X}$. Nehmen Sie an, diese Matrix bleibe invertierbar, wenn man eine beliebige einzelne Zeile von $\mathbf{X}$ entfernt. Bezeichnen Sie mit $\hat{y}^{(i)}$ die Vorhersage an der Stelle $\mathbf{x}^{(i)}$ nach Anpassung an alle $n$ Beispiele. Mit $\hat{y}^{(-i)}$ bezeichnen Sie die Vorhersage an der Stelle $\mathbf{x}^{(i)}$, wenn bei der Anpassung Beispiel $i$ entfällt. Sei ferner $h_i = \mathbf{x}^{(i)\top}\mathbf{A}^{-1}\mathbf{x}^{(i)}$. Verwenden Sie die Sherman-Morrison-Identität (:numref:`sec_mdl-linear-maps`)

   $$(\mathbf{A} - \mathbf{u}\mathbf{u}^\top)^{-1} = \mathbf{A}^{-1} + \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{u}^\top\mathbf{A}^{-1}}{1 - \mathbf{u}^\top\mathbf{A}^{-1}\mathbf{u}}$$

   mit $\mathbf{u} = \mathbf{x}^{(i)}$, um folgende Gleichung zu zeigen:

   $$y^{(i)} - \hat{y}^{(-i)} = \frac{y^{(i)} - \hat{y}^{(i)}}{1 - h_i}.$$

   Geben Sie an, welche Kosten die Leave-one-out-Kreuzvalidierung bei der Methode der kleinsten Quadrate im Vergleich zu einer einzelnen Anpassung verursacht.
   :exlabel:`ex_generalization-leave-one-out-shortcut`

<!--en:28d26529-->
[Diskussionen](https://d2l.discourse.group/t/96)

<!--en:23a46fe2-->
[Diskussionen](https://d2l.discourse.group/t/97)

<!--en:95269eaa-->
[Diskussionen](https://d2l.discourse.group/t/234)

<!--en:05accb8d-->
[Diskussionen](https://d2l.discourse.group/t/17978)

<!--en:f767027c-->
Generalisierung, Verzerrung, Varianz und Modellauswahl<br>
**Empirisches Risiko und Populationsrisiko · Unteranpassung und Überanpassung · Bias-Varianz-Zerlegung · Kreuzvalidierung**

<!--en:a80eaca0-->
::: {.slide title="Auswendiglernen von Beispielen und Erkennen von Mustern in der Grundgesamtheit"}

<!--en:a3151f50-->
[Die zentrale Frage des überwachten Lernens]{.kicker}

<!--en:c5ee794c-->
Werden 20 Polynomgewichte an 20 verrauschte Punkte einer kubischen Kurve angepasst, sinkt der Trainingsverlust auf null. Bei neuen Eingaben versagt das Modell jedoch völlig:

<!--en:56b52286-->
- **Grad 1 (links, Unteranpassung):** Die Gerade ist zu starr für die Krümmung und verfehlt sowohl die Punkte als auch die wahre Kurve.

<!--en:6e1b0671-->
- **Grad 3 (Mitte, ausgewogen):** Das Polynom gehört zur passenden kubischen Familie und folgt der gestrichelten wahren Kurve.

<!--en:e4f6d8c4-->
- **Grad 19 (rechts, Überanpassung):** Mit $20$ Parametern trifft die Kurve alle $20$ verrauschten Punkte und schwingt dazwischen stark.

<!--en:08a2e3c6-->
::: {.slide title="Empirisches Risiko (Trainingsfehler) und Populationsrisiko"}

<!--en:0db3be39-->
[Statistische Grundlagen]{.kicker}

<!--en:ecd5e67a-->
- **Empirisches Risiko ($R_{\textrm{emp}}[f]$):** Mittlerer Verlust auf den $n$ Trainingsbeispielen in $\mathcal{D}$. Das Training minimiert ihn.

<!--en:0a0d1324-->
- **Populationsrisiko ($R[f]$):** Erwarteter Verlust auf neuen Beispielen aus $P(\mathbf{x}, y)$. Er bestimmt die Genauigkeit im Einsatz.

<!--en:0bef1c71-->
- **Generalisierungslücke ($R[\hat{f}] - R_{\textrm{emp}}[\hat{f}]$):** Sie misst, wie viel schlechter $\hat{f}$ auf neuen Daten abschneidet als auf $\mathcal{D}$.

<!--en:2873800f-->
::: {.slide title="Warum der Trainingsfehler systematisch zu niedrig ausfällt"}

<!--en:8b8cb0a6-->
[Der Optimismus der empirischen Risikominimierung]{.kicker}

<!--en:06a07257-->
- **Selektionsbias bei der Optimierung:** Die Wahl von $\hat{f}$ minimiert den Verlust *auf $\mathcal{D}$*. Deshalb nimmt jeder angepasste Parameter $\sigma^2/n$ des Trainingsrauschens auf.

<!--en:b05015f8-->
- **Exakte Interpolation bei $p = n = 20$:** Selbst bei reinem Rauschen verschwindet der erwartete mittlere quadratische Trainingsfehler ($\sigma^2(1 - 20/20) = 0$).

<!--en:55125c69-->
::: {.slide title="Polynomregression durch lineare Anpassung mit Monomen"}

<!--en:fc98be7a-->
[Kontrollierter Vergleich der Modellkapazität]{.kicker}

<!--en:96923815-->
- **Linear in den Gewichten $\mathbf{w}$:** Die Abbildung von $x \in [-1, 1]$ auf $[1, x, x^2, \ldots, x^d]$ macht die Anpassung eines Polynoms vom Grad $d$ zu einem linearen Problem der kleinsten Quadrate $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ mit $n = 20$ Punkten aus $f^*(x) = 5 + 1.2x - 3.4x^2 + 5.6x^3$ ($\sigma = 0.1$).

<!--en:af1359d9-->
::: {.slide title="Grad 1, 3 und 19: Fehler und stark wachsende Gewichte"}

<!--en:d089375c-->
[Quantitative Diagnose]{.kicker}

<!--en:8606e48a-->
- **Warum die Gewichte bei Grad 19 so groß werden:** Die hohen Potenzen $x^{17}$ und $x^{19}$ sind auf $[-1, 1]$ nahezu kollinear. Bei der Methode der kleinsten Quadrate gleichen sich daher sehr große Koeffizienten mit wechselndem Vorzeichen ($|\hat{w}_j| > 10^6$) in ihrer Wirkung aus, damit die Kurve alle 20 Punkte trifft.

<!--en:557b6dcf-->
::: {.slide title="Die U-förmige Testfehlerkurve für die Polynomgrade 1 bis 19"}

<!--en:9cdfe102-->
[Verlauf von Trainingsfehler und Generalisierungsfehler]{.kicker}

<!--en:4ea7dcf9-->
Steigt der Polynomgrad $d$ von $1$ auf $19$, sinkt der Trainingsfehler monoton. Der Testfehler beschreibt eine U-förmige Kurve:

<!--en:3fa20c0f-->
- **Monotoner Trainingsfehler:** Die Modellklasse vom Grad $d+1$ enthält die Klasse vom Grad $d$ als echte Teilmenge ($w_{d+1} = 0$ setzen). Zusätzliche Monome können den Trainingsfehler der Methode der kleinsten Quadrate daher *nie* erhöhen.

<!--en:d07b767c-->
- **Minimaler Testfehler ($d = 3$):** Der mittlere quadratische Testfehler fällt von $d = 1$ bis $d = 3$ um zwei Größenordnungen. Danach steigt er stetig, weil sich zusätzliche Parameter dem Rauschen anpassen.

<!--en:c839f5b2-->
- **Diagnoseregel:** Ein hoher Trainingsfehler deutet auf **Unteranpassung** hin, ein niedriger Trainingsfehler bei großer Generalisierungslücke auf **Überanpassung**.

<!--en:4a8bec10-->
::: {.slide title="Bias-Varianz-Zerlegung des erwarteten Testfehlers"}

<!--en:e0f68c0c-->
[Theoretische Zerlegung]{.kicker}

<!--en:d4eefabf-->
- **Quadrierte Verzerrung:** Systematischer, auch bei unendlich vielen Daten verbleibender Approximationsfehler der Modellfamilie (dominiert bei starren Modellen mit $d = 1, 2$)

<!--en:1ba9fa50-->
- **Varianz und Rauschen $\sigma^2$:** Zur irreduziblen Rauschgrenze kommt die Empfindlichkeit von $\hat{f}_{\mathcal{D}}$ gegenüber zufälligem Rauschen in der Trainingsstichprobe hinzu. Bei flexiblen Modellen ($d \gg 3$) dominiert diese Varianz den Fehler oberhalb von $\sigma^2$.

<!--en:0a3b1f63-->
::: {.slide title="Empirische quadrierte Verzerrung und Varianz nach Polynomgrad"}

<!--en:e27703c7-->
[Messung der einzelnen Fehleranteile]{.kicker}

<!--en:6990fe85-->
Das Rauschen der 20 Trainingspunkte bei festen Eingaben $200$-mal neu ziehen. Für jeden Grad ergibt sich $\textrm{bias}^2$ aus der mittleren Anpassung an die rauschfreien Labels und $\textrm{variance}$ aus der Streuung der 200 Anpassungen:

<!--en:33ffb342-->
- **Quadrierte Verzerrung fällt bei $d = 3$ auf null:** Bei $d = 1, 2$ ist sie sehr groß, da Geraden und Parabeln die kubische Kurve nicht wiedergeben können. Für alle $d \geq 3$ ist sie null, weil jede Familie mit $d \geq 3$ die Funktion $f^*(x)$ enthält.

<!--en:d9dd2c86-->
- **Die Varianz steigt monoton:** Bei niedrigen Graden wächst sie gemäß $\approx \sigma^2 (d + 1) / n$. In der Nähe von $d = 19$ wird die Vandermonde-Matrix nahezu singulär, und die Varianz steigt sprunghaft.

<!--en:e9f91d15-->
- **Optimaler Kompromiss:** Die Summe $\textrm{bias}^2 + \textrm{variance} + \sigma^2$ erreicht ihr Minimum bei $d = 3$.

<!--en:6d170059-->
::: {.slide title="Modellkomplexität und klassische Kapazitätskurve"}

<!--en:7c95bcd5-->
[Zielkonflikt zwischen Verzerrung und Varianz]{.kicker}

<!--en:c22b41aa-->
Bei klassischen parametrischen Modellfamilien verschiebt sich der Fehler mit steigender Kapazität von der Verzerrung hin zur Varianz:

<!--en:4c00590a-->
- **Links vom Optimum (Unteranpassung):** Die Modellkapazität ist zu gering. Die hohe Verzerrung hält Trainingsrisiko und Populationsrisiko hoch.

<!--en:7f907292-->
- **Rechts vom Optimum (Überanpassung):** Die Kapazität übersteigt den Informationsgehalt von $n$ Beispielen. Der Trainingsfehler fällt unter die Rauschgrenze $\sigma^2$, während die Varianz das Testrisiko erhöht.

<!--en:624a6044-->
- **Gewichtsnorm als kontinuierliches Kapazitätsmaß:** Selbst bei Grad 19 können Funktionen mit kleiner Koeffizientennorm $\|\mathbf{w}\|_2$ nicht stark schwingen. Darauf beruht der Gewichtszerfall im nächsten Abschnitt.

<!--en:4096a239-->
::: {.slide title="Mehr Daten verringern die Varianz, nie die Verzerrung"}

<!--en:9c5b6a25-->
[Stichprobenkomplexität]{.kicker}

<!--en:12a23c24-->
- **Die Varianz fällt gemäß $O(p / n)$:** Wächst $n$ von $20$ auf $1{,}000$, sinkt der mittlere quadratische Testfehler bei Grad 19 bis zur Rauschgrenze $\sigma^2 = 0.01$. Bei der unterangepassten Geraden vom Grad 1 bleibt er nahe $2$.

<!--en:eb53ea68-->
::: {.slide title="Fundierte Modellauswahl durch Dreiteilung der Daten"}

<!--en:87009fb6-->
[Datenleckage aus dem Testdatensatz vermeiden]{.kicker}

<!--en:debdfbc8-->
- **Nie anhand des Trainingsfehlers auswählen:** Der Trainingsfehler bevorzugt stets das komplexeste Modell, das am ehesten auswendig lernt ($d = 19$).

<!--en:12ce1a69-->
- **Nie anhand des Testdatensatzes abstimmen:** Wer ihn wiederholt auswertet, passt die Hyperparameter an das Rauschen im Testdatensatz an.

<!--en:0b6b670c-->
- **Dreiteilung:** $(\mathbf{w}, b)$ auf dem **Trainingsdatensatz** anpassen, Hyperparameter $(d, \eta, \lambda)$ auf dem **Validierungsdatensatz** auswählen und das beste Modell **einmal** auf dem **Testdatensatz** auswerten.

<!--en:ff85cd75-->
::: {.slide title="K-fache Kreuzvalidierung bei wenigen Daten"}

<!--en:2d8e0a6f-->
[Validierung bei kleinem Stichprobenumfang]{.kicker}

<!--en:ac7601b1-->
Bei wenigen Daten ($n = 20$) gehen durch einen separaten Validierungsdatensatz wertvolle Trainingsbeispiele verloren. Die **$K$-fache Kreuzvalidierung** teilt die $n$ Beispiele in $K$ gleich große Gruppen:

<!--en:30ab67d3-->
- **Validierungsgruppe wechseln:** In Runde $k \in \{1, \ldots, K\}$ auf $K - 1$ Gruppen ($\frac{K-1}{K}n$ Beispielen) von Grund auf trainieren und auf der zurückgehaltenen Gruppe $k$ validieren. Anschließend die $K$ Validierungsergebnisse mitteln.

<!--en:2512e619-->
- **Bestes Modell erneut anpassen:** Nach der Wahl des besten Hyperparameters $d^*$ das endgültige Modell auf allen $n$ Beispielen neu trainieren.

<!--en:c6067596-->
- **Wahl von $K$ ($5$ oder $10$ gegenüber $K = n$):** Je kleiner $K$, desto weniger Trainingsbeispiele pro Runde (leichte Verzerrung nach oben). Leave-one-out ($K = n$) ist nahezu erwartungstreu, erfordert aber im Allgemeinen $n$ Modellanpassungen (eine bei der Methode der kleinsten Quadrate).

<!--en:f204b35d-->
::: {.slide title="Wahl des Polynomgrads durch 5-fache Kreuzvalidierung"}

<!--en:9cdd0844-->
[Kreuzvalidierung im Code]{.kicker}

<!--en:9844a1b9-->
Allein anhand der $n = 20$ Trainingsbeispiele ($K = 5$, also $16$ Trainingspunkte und $4$ Validierungspunkte pro Runde) findet die 5-fache Kreuzvalidierung das Minimum der Testfehlerkurve, ohne den Testdatensatz zu verwenden:

<!--en:8eebeba5-->
- **Genaue Modellauswahl:** Die Kreuzvalidierung wählt zuverlässig Grad $3$ oder einen benachbarten Grad, dessen mittlerer quadratischer Testfehler nur um einen winzigen Bruchteil vom Optimum abweicht.

<!--en:044aed6b-->
- **Früherer Anstieg des Kreuzvalidierungsfehlers bei hohen Graden:** Jede Runde nutzt $16$ statt $20$ Trainingspunkte. Schon Grad $15$ interpoliert alle Trainingspunkte der Runde und liefert an zurückgehaltenen Randpunkten extreme Werte.

<!--en:9c510e18-->
::: {.slide title="Verständnisfragen zu Fehlerdifferenz, Stichprobenumfang und Kosten der Kreuzvalidierung"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:68409f50-->
Die Grundsätze der Generalisierung und der Modellauswahl auf drei Aufgaben anwenden:

<!--en:4cf64398-->
1. **Statistische Signifikanz einer Lücke:** Ein Modell erreicht einen Trainingsverlust von $0.020$ und einen Validierungsverlust von $0.030$ auf $n' = 400$ zurückgehaltenen Beispielen. Die Standardabweichung des Verlusts pro Beispiel beträgt $\sigma_l = 0.04$. Kann die Lücke von $0.010$ durch eine zufällige Schwankung im Validierungsdatensatz entstehen?

<!--en:4ae045cc-->
2. **Wann lohnen sich mehr Daten?** Modell A hat einen Trainingsfehler von $1.9$ und einen Validierungsfehler von $2.0$. Bei Modell B liegen sie bei $10^{-14}$ und $10^4$. Bei welchem Modell verbessern $10\times$ so viele Trainingsbeispiele das Ergebnis auf den Validierungsdaten?

<!--en:a221614a-->
3. **Abkürzung für Leave-one-out:** Die direkte Leave-one-out-Kreuzvalidierung für $19$ Polynomgrade auf $n = 20$ Punkten erfordert $19 \times 20 = 380$ Anpassungen. Wie viele Anpassungen sind mit der Identität $y^{(i)} - \hat{y}^{(-i)} = (y^{(i)} - \hat{y}^{(i)}) / (1 - H_{ii})$ und $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ nötig?

<!--en:557b98b6-->
**Lösungen:**

<!--en:c66f09e5-->
1. Der Standardfehler beträgt $\sigma_l / \sqrt{n'} = 0.04 / 20 = 0.002$. Eine Lücke von $0.010$ entspricht damit $5$ Standardfehlern und weist auf einen tatsächlichen Unterschied hin.

<!--en:a866d94e-->
2. **Modell B** ist überangepasst und hat eine hohe Varianz, die durch mehr Daten stark sinkt. Modell A ist unterangepasst (hohe Verzerrung). Sein Fehler bleibt auch mit mehr Daten nahe $2.0$. Es braucht eine flexiblere Modellklasse.

<!--en:cef60fdd-->
3. Nur **$19$ Anpassungen** (eine pro Grad). Die Residuen einer Anpassung und die Diagonaleinträge $H_{ii}$ liefern die exakten Leave-one-out-Residuen mit $O(n)$ Rechenoperationen.

<!--en:ba43a2dd-->
::: {.slide title="Zusammenfassung und zentrale Ergebnisse"}

<!--en:d018528e-->
[Zusammenfassung]{.kicker}

<!--en:e2056880-->
| Begriff oder Phänomen | Mathematischer Ausdruck | Skalierung oder Verhalten | Praktische Maßnahme oder Abhilfe |

<!--en:5bb08370-->
| **Empirisches und wahres Risiko** | $R_{\textrm{emp}}[f] = \frac{1}{n}\sum_i l_i$ gegenüber $R[f] = \mathbb{E}[l]$ | $\mathbb{E}[R_{\textrm{emp}}] = \sigma^2(1 - p/n) < \mathbb{E}[R]$ | Modelle nie anhand des Trainingsfehlers beurteilen oder auswählen |

<!--en:918aa186-->
| **Unteranpassung (hohe Verzerrung)** | $\textrm{bias}^2(x) = (\bar{f}(x) - f^*(x))^2 \gg \sigma^2$ | Trainingsfehler und Validierungsfehler hoch ($d=1$) | Modellkapazität erhöhen oder Merkmale hinzufügen (mehr Daten helfen nicht) |

<!--en:ccb0de51-->
| **Überanpassung (hohe Varianz)** | $\operatorname{Var}(\hat{f}_{\mathcal{D}}(x)) \approx \sigma^2 p / n \gg \sigma^2$ | Trainingsfehler $\to 0$, Validierungsfehler steigt rasant ($d=19$) | Trainingsdaten ergänzen ($n \uparrow$), $p$ verringern oder Gewichtszerfall anwenden (§{sec_weight_decay}) |

<!--en:d7bdba68-->
| **Bias-Varianz-Identität** | $\mathbb{E}[(y - \hat{f}(x))^2] = \textrm{bias}^2 + \operatorname{Var} + \sigma^2$ | Minimum des U-förmigen Testrisikos bei mittlerem Grad $d^*$ | Näherungsvermögen auf Stichprobenumfang $n$ abstimmen |

<!--en:4ad79e5a-->
| **Validierung und $K$-fache Kreuzvalidierung** | Mittlerer Verlust über $K$ zurückgehaltene Gruppen $\mathcal{D}_k$ | Training auf $\frac{K-1}{K}n$ Punkten pro Runde ($K=5$ oder $10$) | Alle Hyperparameter durch Validierung oder Kreuzvalidierung wählen, Testdatensatz nur einmal auswerten |
