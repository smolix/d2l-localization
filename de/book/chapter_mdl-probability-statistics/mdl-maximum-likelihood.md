<!--en:3644fdb1-->
# Maximum-Likelihood

<!--en:e92cc6a3-->
Angenommen, wir werfen eine Münze dreizehnmal und erhalten neunmal Kopf und viermal Zahl. Als Wahrscheinlichkeit für Kopf würden wir wohl alle $9/13\approx 0.69$ angeben. Doch zwei Fragen drängen sich auf. Warum gerade dieser Wert, und wie sehr können wir ihm vertrauen? Könnte die Münze doch fair sein?

<!--en:c68c9c18-->
Ein Prinzip beantwortet beide Fragen. Wir wählen die Parameter, unter denen die beobachteten Daten am wahrscheinlichsten sind. Dieses *Maximum-Likelihood*-Prinzip liegt fast jedem Trainingsverlust (engl. training loss) in diesem Buch zugrunde. Für die Münze liefert es $9/13$. Bei einem neuronalen Netz ergibt sich aus einer Annahme über das Rauschen der Zielwerte der Verlust, den wir minimieren. So erhalten wir sowohl den quadratischen Fehler (engl. squared error) als auch die Kreuzentropie (engl. cross-entropy), die wir in früheren Kapiteln wegen ihrer guten Ergebnisse gewählt haben. Bemerkenswerterweise verrät dieselbe Verlustfunktion auch, wie genau die Daten die Parameter festlegen. Je stärker die Verlustfunktion um ihr Minimum gekrümmt ist, desto kürzer ist der Fehlerbalken. Bei unserer Münze verrät uns die Krümmung, ob dreizehn Würfe ausreichen, um eine faire Münze auszuschließen.

<!--en:cc540d0a-->
In :numref:`sec_mdl-statistics` haben wir den Fehler von Schätzern mit geschlossener Formel bestimmt, etwa beim Stichprobenmittelwert oder beim Stichprobenanteil. Hier definiert ein Optimierungsproblem den Schätzer, und sein Fehlerbalken ergibt sich unmittelbar aus der Verlustfunktion. Zunächst leiten wir die üblichen Verlustfunktionen früherer Kapitel aus dem Maximum-Likelihood-Prinzip ab. Dann wenden wir uns der zentralen Frage zu. Wie genau ist der Maximum-Likelihood-Schätzer? Das wichtigste Ergebnis sollten wir uns merken. Mit wachsender Stichprobe nähert sich die Verteilung des Schätzfehlers einer Normalverteilung, deren Varianz die Krümmung der Verlustfunktion bestimmt. An drei aufeinanderfolgenden Würfen mit Kopf sehen wir schließlich, was bei wenigen Daten schiefgeht und wie aus einer Annahme vor der Beobachtung der Daten ein Regularisierungsterm entsteht. Modelle, bei denen ein Teil jeder Beobachtung verborgen bleibt, behandeln wir in :numref:`sec_mdl-latent-em-elbo`.

<!--en:41bbec26-->
## Das Maximum-Likelihood-Prinzip

<!--en:00296956-->
Ein probabilistisches Modell weist jedem Datensatz, den es erzeugen könnte, eine Wahrscheinlichkeit zu. Sobald uns die Daten vorliegen, kehren wir die Betrachtung um. Wir halten die beobachteten Daten fest und fragen, wie wahrscheinlich sie bei den jeweils infrage kommenden Parameterwerten sind.

<!--en:08a25fd0-->
Es bezeichne $\boldsymbol{\theta}$ die Parameter eines Modells und $X$ den beobachteten Datensatz. Die **Likelihood** ist die Wahrscheinlichkeit der beobachteten Daten (bei stetigen Daten ihre Dichte), aufgefasst als Funktion der Parameter, $P(X\mid\boldsymbol{\theta})$. Der **Maximum-Likelihood-Schätzer** (engl. maximum likelihood estimate, MLE) liefert die Parameterwerte, die diese Funktion maximieren,

<!--en:87107ceb-->
Unter allen möglichen Parameterwerten wählt der MLE diejenigen, unter denen unsere tatsächliche Beobachtung am wahrscheinlichsten war. Dabei behandeln wir $\boldsymbol{\theta}$ als unbekannte Konstante. Nur die Daten sind zufällig, und nur Datensätzen weisen wir Wahrscheinlichkeiten zu. Diese Unterscheidung wird wichtig, wenn wir in :numref:`subsec_mdl-map` A-priori-Verteilungen (engl. priors) einführen.

<!--en:9de53f69-->
### Die Münze

<!--en:72271f94-->
Eine Münze, bei der Kopf und Zahl unterschiedlich wahrscheinlich sein können, ist das einfachste Modell, an dem wir das Prinzip vollständig durchrechnen können. Ihr einziger Parameter ist die Wahrscheinlichkeit $\theta$ für Kopf. In :numref:`sec_mdl-distributions` haben wir sie $p$ genannt. Da wir sie nun als Unbekannte schätzen wollen, schreiben wir $\theta$. Weil die Würfe unabhängig sind, multiplizieren sich ihre Wahrscheinlichkeiten. Eine bestimmte Folge mit $n_H$-mal Kopf und $n_T$-mal Zahl hat also die Wahrscheinlichkeit

<!--en:b9162c58-->
Als Funktion von $\theta$ ist dieser Ausdruck die Likelihood. Bei unseren dreizehn Würfen HHHTHTTHHHHHT mit $n_H=9$-mal Kopf und $n_T=4$-mal Zahl beträgt sie $\theta^9(1-\theta)^4$. Eine faire Münze weist dieser Folge die Wahrscheinlichkeit $2^{-13}\approx 1.2\times10^{-4}$ zu. Bei einer Münze mit $\theta=9/13$ beträgt sie $3.3\times10^{-4}$, also etwa das $2.7$-Fache.

<!--en:e8a8035e-->
Um das Maximum zu finden, nehmen wir zunächst den Logarithmus. Da er streng monoton wächst, bleibt die Lage des Maximums erhalten. Aus dem Produkt wird eine Summe, die sich leicht ableiten lässt. Setzen wir die Ableitung gleich null, erhalten wir

<!--en:f845be7a-->
An den Randpunkten $\theta=0$ und $\theta=1$ hat jede Folge, die sowohl Kopf als auch Zahl enthält, die Wahrscheinlichkeit null. Das Maximum muss daher im Inneren $(0,1)$ liegen, und dieser stationäre Punkt ist tatsächlich der globale Maximierer. Dieselbe Rechnung für die allgemeine Likelihood :eqref:`eq_mdl-coin-like` ergibt

<!--en:99261961-->
Der MLE für die Kopfwahrscheinlichkeit ist also die beobachtete relative Häufigkeit von Kopf, wie wir intuitiv erwarten. Sehen wir uns die Ableitung genauer an. Kodieren wir jeden Wurf als $x\in\{0,1\}$ (mit $x=1$ für Kopf und $x=0$ für Zahl), so ist die Ableitung eine Summe mit einem Term pro Wurf, $x/\theta-(1-x)/(1-\theta)$. Jeder Term ist der Gradient der Log-Wahrscheinlichkeit einer einzelnen Beobachtung. Er spielt eine zentrale Rolle, wenn wir die Genauigkeit von $\hat\theta$ bestimmen.

<!--en:cc6aa2b7-->
Angenommen, dieselbe relative Häufigkeit ergäbe sich aus zehnmal so vielen Würfen, also $90$-mal Kopf bei $130$ Würfen. Die Punktschätzung bleibt gleich, da sie nur vom Verhältnis abhängt. Die Daten grenzen den Wert nun aber genauer ein. Wir stellen beide Likelihood-Kurven dar und teilen jede durch ihr Maximum, damit sie dieselbe vertikale Skala haben.

<!--en:725f24b2-->
Beide Kurven haben ihr Maximum bei $9/13\approx 0.69$, doch die zweite ist nur etwa ein Drittel so breit. Später in diesem Abschnitt leiten wir aus der Breite einen präzisen Fehlerbalken ab.

<!--en:621112da-->
### Die negative Log-Likelihood

<!--en:90c51331-->
Für die Münze haben wir eine geschlossene Lösung erhalten. Bei einem Modell mit Millionen von Parametern fehlt sie. Wir müssen daher numerisch optimieren. In der Gleitkommaarithmetik ist die Likelihood in ihrer ursprünglichen Form jedoch unbrauchbar. Bei $n$ unabhängigen Datenpunkten ist sie ein Produkt aus $n$ Wahrscheinlichkeiten,

<!--en:c899d086-->
die jeweils in $[0,1]$ liegen. Eine Milliarde Faktoren nahe $1/2$ ergeben ein Produkt von etwa $(1/2)^{10^9}$. In jedem Gleitkommaformat führt das zum Unterlauf auf null, lange bevor wir einen Gradienten berechnen können.

<!--en:2f3b9197-->
Der Logarithmus macht aus dem Produkt eine Summe, die problemlos im darstellbaren Bereich bleibt:

<!--en:422baa99-->
Die Likelihood zu maximieren ist daher gleichbedeutend damit, die *Log-Likelihood* $\log P(X\mid\boldsymbol{\theta})$ zu maximieren. Da Optimierungsverfahren üblicherweise Verluste minimieren, wechseln wir das Vorzeichen und definieren die **negative Log-Likelihood** (engl. negative log-likelihood, NLL):

<!--en:bd443aad-->
Die NLL enthält einen Summanden pro Beobachtung. Er misst, wie überraschend der jeweilige Datenpunkt für das Modell ist. Für die Münze lautet sie $\ell(\theta) = -\big(n_H\log\theta + n_T\log(1-\theta)\big)$. Wir können sie mit Gradientenabstieg minimieren, als gäbe es keine geschlossene Lösung. Im Folgenden wenden wir den Gradientenabstieg auf fast neun Millionen Würfe an und vergleichen das Ergebnis mit $n_H/(n_H+n_T)$.

<!--en:f3aa628c-->
Der Gradientenabstieg liefert $0.9713$. Der Wert stimmt in allen ausgegebenen Stellen mit der beobachteten relativen Häufigkeit überein. Die Lernrate $10^{-9}$ erscheint winzig, doch die Summation des Verlusts erzwingt sie. Der Gradient des über fast neun Millionen Würfe summierten Verlusts enthält einen Faktor $n$. Entsprechend sinkt die größte stabile Schrittweite (:numref:`subsec_mdl-quadratic-model`). Deshalb mitteln wir in diesem Buch den Verlust eines Minibatchs über die Beispiele.

<!--en:12d31a22-->
Der Logarithmus gibt auch dem Gradienten eine einfache additive Form,

<!--en:1514f5ea-->
mit einem unabhängigen Term pro Beispiel. Würden wir die Produktregel direkt auf die Likelihood anwenden, ergäben sich $n$ Produkte mit je $n-1$ Faktoren. Dank der additiven Form liefert ein zufällig gezogener Minibatch eine erwartungstreue Schätzung von $\nabla\ell$. Das ermöglicht den stochastischen Gradientenabstieg (engl. stochastic gradient descent, SGD).

<!--en:9dab9615-->
## Verlustfunktionen als negative Log-Likelihoods

<!--en:c6d5823b-->
Bei der Münze haben wir die NLL direkt aus einem probabilistischen Modell hergeleitet. In früheren Kapiteln haben wir Trainingsverluste wie den quadratischen Fehler und die Kreuzentropie unmittelbar angegeben und eingesetzt, weil sie sich bewährt haben. Wie sich herausstellt, sind die meisten dieser üblichen Verluste negative Log-Likelihoods. Mit der Verlustfunktion wählen wir also zugleich ein probabilistisches Modell für die Daten.

<!--en:c902b856-->
### Kreuzentropie und Entropie als Untergrenze

<!--en:0b93d8d9-->
Der Trainingsverlust eines Klassifikators ist eine mittlere NLL. Die Trainingskurve flacht häufig bei einem positiven Wert ab, bevor sie null erreicht. Was bedeutet dieser Wert, und wie klein kann er überhaupt werden? Dazu schreiben wir die mittlere NLL als Summe über die verschiedenen Ergebnisse. Die *empirische Verteilung* $\hat p_{\textrm{data}}$ der Stichprobe ordnet jedem Ergebnis $x$, das $n_x$-mal unter den $n$ Beobachtungen auftritt, die Wahrscheinlichkeit $\hat p_{\textrm{data}}(x) = n_x/n$ zu, also seine beobachtete relative Häufigkeit. Für unsere Münzwürfe gilt $\hat p_{\textrm{data}}(\textrm{H}) = 9/13$ und $\hat p_{\textrm{data}}(\textrm{T}) = 4/13$.

<!--en:aae716fc-->
Wir fassen nun alle Beobachtungen mit $x_i = x$ zusammen. Aus der Summe über Datenpunkte wird eine Summe über *verschiedene Ergebnisse*. Bei der Münze liefert neunmal Kopf neunmal denselben Term $\log p_{\boldsymbol{\theta}}(\textrm{H})$. Kopf geht also mit dem Gewicht $9/13$ in den Mittelwert ein und Zahl mit dem Gewicht $4/13$. Allgemein gilt

<!--en:5b13c9f0-->
Die rechte Seite ist die *Kreuzentropie* der empirischen Verteilung bezüglich des Modells, $\textrm{CE}(p,q)=-\sum_x p(x)\log q(x)$. Die Skalierung mit der positiven Konstanten $1/n$ ändert den Minimierer nicht. Bei diskreten Ergebnissen ist Maximum-Likelihood daher gleichbedeutend mit der Minimierung der Kreuzentropie. Diese eine Umgruppierung erklärt den Namen des Klassifikationsverlusts.

<!--en:c1ecf05e-->
Wie klein kann diese Kreuzentropie werden? Sie zerfällt in einen Teil, den die Daten festlegen, und einen Teil, den das Modell bestimmt. Aus :numref:`subsec_mdl-max-entropy-derivation` kennen wir die Entropie einer Verteilung $p$ auf einer endlichen Menge. Sie beträgt $H(p) = -\sum_x p(x)\log p(x)$ und misst die mittlere Überraschung. Die Kullback-Leibler-Divergenz (KL-Divergenz) $D_{\textrm{KL}}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$ misst die mittlere zusätzliche Überraschung, wenn wir $q$ verwenden, obwohl $p$ gilt (beide Größen behandeln wir ausführlich in :numref:`sec_mdl-information_theory`). Wir verwenden durchgehend natürliche Logarithmen und messen alle drei Größen daher in *Nat*. Addieren und subtrahieren wir $\sum_x p(x)\log p(x)$, so erhalten wir

<!--en:cb90934e-->
Die Entropie hängt allein von $p$ ab. Nur die KL-Divergenz hängt vom Modell $q$ ab. Nach der Gibbs-Ungleichung (:numref:`subsec_mdl-max-entropy-derivation`) ist die KL-Divergenz nie negativ und verschwindet nur dann, wenn $q=p$ gilt.

<!--en:1f720c5e-->
Setzen wir $p=\hat p_{\textrm{data}}$ und $q=p_{\boldsymbol{\theta}}$, sehen wir, dass der Trainingsverlust nie unter die Entropie der beobachteten relativen Häufigkeiten fallen kann. Er erreicht diese Untergrenze nur, wenn die Modellfamilie diese Häufigkeiten exakt wiedergeben kann. Das Training kann nur die KL-Divergenz von der empirischen Verteilung zum Modell verringern. Bei unserer Münze liegt die Untergrenze bei $H(9/13) = -\tfrac{9}{13}\log\tfrac{9}{13}-\tfrac{4}{13}\log\tfrac{4}{13} =
0.617$ Nat. Für eine faire Münze beträgt die Kreuzentropie $\log 2 = 0.693$ Nat, die KL-Divergenz also $0.076$ Nat. Bei einem Modell mit $\theta=0.9$ beträgt der Abstand zur Untergrenze $0.164$ Nat. :numref:`fig_mdl-mle-kl` zeigt die gesamte Kurve.

<!--en:6969a66f-->
![Kreuzentropie pro Wurf für die Münzdaten ($9$-mal Kopf und $4$-mal Zahl) als Funktion der Kopfwahrscheinlichkeit $\theta$ im Modell. Der Verlust fällt nie unter die Entropie der beobachteten relativen Häufigkeiten von $0.617$ Nat (gestrichelte Linie). Beim MLE $9/13$ erreicht er diese Untergrenze. Bei jedem anderen $\theta$ liegt er um die KL-Divergenz von der empirischen Verteilung zum Modell darüber (orange): $0.076$ Nat bei $\theta=1/2$ und $0.164$ Nat bei $\theta=0.9$.](../img/mdl-prob-mle-kl.svg)

<!--en:1fcb2650-->
Dieselbe Identität gilt für einen Klassifikator. Er modelliert $p_{\boldsymbol{\theta}}(y\mid\mathbf x)$, und seine mittlere NLL ist der Mittelwert von $-\log p_{\boldsymbol{\theta}}(y_i\mid\mathbf x_i)$ über den Trainingsdatensatz. Gruppieren wir die Beispiele nach ihrer Eingabe, erhalten wir für jede verschiedene Eingabe wieder die Zerlegung der Kreuzentropie. Die empirischen Labelhäufigkeiten für diese Eingabe übernehmen dabei die Rolle von $\hat p_{\textrm{data}}$ (Übung :exref:`ex_mdl-maximum-likelihood-cross-entropy-floor`). Bei One-Hot-kodierten Labels und der Softmax-Funktion als Ausgabe entspricht diese mittlere NLL genau dem Softmax-Kreuzentropieverlust aus :numref:`sec_softmax`. In einem typischen Datensatz kommt jede Trainingseingabe nur einmal vor. Die empirische Labelverteilung ist dann für jede Eingabe auf ein Label konzentriert (One-Hot), und die Untergrenze des Trainingsverlusts ist null, unabhängig davon, wie verrauscht die Labels sind. Die in der Praxis unvermeidbare Untergrenze zeigt sich auf zurückgehaltenen Daten. Dort kann der erwartete Verlust nicht unter die bedingte Entropie der Labels bei gegebenen Eingaben fallen (:numref:`subsec_mdl-irreducible-loss`).

<!--en:60cc2b99-->
Ein Sprachmodell ist derselbe Klassifikator mit sehr vielen möglichen Labels. Die Eingabe ist der bisherige Text, das Label ist das nächste Token, und das Training minimiert die mittlere NLL pro Token. Meist gibt man den Exponentialwert dieser Größe an, die *Perplexität* (:numref:`subsec_perplexity`). Sie gibt an, wie viele gleich wahrscheinliche Token dieselbe mittlere Überraschung hervorrufen würden. Auf zurückgehaltenen Daten liegt die Untergrenze hier weit über null. Für die meisten Sätze gibt es viele sinnvolle Fortsetzungen. Daher ist die bedingte Entropie des nächsten Tokens groß. Auch das größte Modell kann seinen Verlust auf zurückgehaltenen Daten im Erwartungswert nicht darunter senken.

<!--en:b83d2ebc-->
### Dichten

<!--en:238fc373-->
Für stetige Daten erfordert die Definition der Likelihood einen weiteren Schritt. Die Wahrscheinlichkeit, genau eine bestimmte reelle Zahl zu beobachten, ist null. Wörtlich genommen wäre die Likelihood also $0$ für jedes $\boldsymbol{\theta}$ und könnte verschiedene Parameterwerte nicht unterscheiden. Wir verlangen daher für jede Beobachtung nur Übereinstimmung innerhalb einer kleinen Toleranz $\epsilon$. Wie wir gleich sehen, fällt $\epsilon$ bei der Optimierung weg.

<!--en:d5142451-->
Betrachten wir unabhängige, identisch verteilte Beobachtungen $x_1,\ldots,x_n$ mit Dichte $p(x\mid\boldsymbol{\theta})$. Die Wahrscheinlichkeit, dass jedes $X_i$ in ein Intervall der Breite $\epsilon$ um $x_i$ fällt, beträgt bis zur ersten Ordnung in $\epsilon$

<!--en:674e2ec1-->
Nehmen wir den negativen Logarithmus, erhalten wir

<!--en:7ccc9105-->
Die Toleranz $\epsilon$ geht nur in die additive Konstante $-n\log\epsilon$ ein, die unabhängig von $\boldsymbol{\theta}$ ist. Ob wir vier genaue Stellen verlangen oder vierhundert, ändert nur diese Konstante. Der Minimierer bleibt gleich. Daher lassen wir die Konstante weg und minimieren wie im diskreten Fall $-\sum_i \log p(x_i\mid\boldsymbol{\theta})$. Ein Dichtewert ist selbst keine Wahrscheinlichkeit. Erst das Produkt $\epsilon\, p(x_i\mid\boldsymbol{\theta})$ nähert eine Wahrscheinlichkeit an. Daher hat $p$ die inverse Einheit von $x$ und kann größer als $1$ sein.

<!--en:f9e9c160-->
Die Identität für die Kreuzentropie in der Grundgesamtheit gilt auch bei stetigen Daten. Unsere Zielfunktion bleibt die über die Stichprobe gemittelte negative Log-Dichte. Ihr Erwartungswert unter der wahren Dichte $p_\star$, die die Daten erzeugt, beträgt

<!--en:dd145cb3-->
Das folgt durch dieselbe Addition und Subtraktion wie zuvor, wobei Integrale an die Stelle der Summen treten. Da die *differentielle Entropie* $h(p_\star)$ unabhängig von $\boldsymbol{\theta}$ ist, minimiert Maximum-Likelihood auch hier in der Grundgesamtheit eine KL-Divergenz. Die Identität für endliche Stichproben aus dem diskreten Fall hat kein stetiges Gegenstück, denn die empirische Verteilung von $n$ verschiedenen reellen Zahlen besitzt keine Dichte.

<!--en:665fce96-->
### Gaußsches Rauschen und quadratischer Fehler

<!--en:8e29cb4f-->
Der übliche Regressionsverlust ergibt sich aus demselben Prinzip, sobald wir ein Modell für das Rauschen der Zielwerte angeben. In :numref:`subsec_normal_distribution_and_squared_loss` haben wir diesen Zusammenhang für die lineare Regression hergeleitet. Die Herleitung setzt jedoch an keiner Stelle Linearität voraus. Angenommen, jeder Zielwert ist die Summe aus der Modellvorhersage und gaußschem Rauschen mit *fester* Varianz, also $y_i \sim \mathcal{N}(\hat y_i, \sigma^2)$ mit $\hat y_i = f_{\boldsymbol{\theta}}(\mathbf{x}_i)$. Summieren wir den negativen Logarithmus der Gauß-Dichte $(2\pi\sigma^2)^{-1/2}\exp\!\big(-(y-\hat y)^2/(2\sigma^2)\big)$ über $n$ unabhängige Beispiele, erhalten wir

<!--en:e23d34d0-->
Da die Varianz $\sigma^2$ fest ist, hängt der zweite Term nicht von $\boldsymbol{\theta}$ ab. Der Vorfaktor $1/(2\sigma^2)$ ist eine positive Konstante. Diese NLL zu minimieren ist daher gleichbedeutend damit, $\sum_i (y_i-\hat y_i)^2$ zu minimieren. Regression mit quadratischem Fehler entspricht also Maximum-Likelihood unter der Annahme gaußverteilter Residuen. Auf dieselbe Weise lassen sich viele vertraute Verluste als NLL einer bedingten Verteilung $p(y\mid\mathbf x)$ verstehen, darunter die vier in der folgenden Tabelle. Jede Zeile ergibt sich, indem wir den negativen Logarithmus der Wahrscheinlichkeit oder Dichte nehmen und die von $\boldsymbol{\theta}$ unabhängigen Konstanten weglassen.

<!--en:6c9dfe37-->
| Rauschmodell $p(y\mid\mathbf x)$ | Negative Log-Likelihood $-\log p(y\mid\mathbf x)$ | Verlust im Deep Learning |

<!--en:e8053d92-->
| Normalverteilung mit fester Varianz $\sigma^2$ | $\tfrac{1}{2\sigma^2}(y-\hat y)^2 + \textrm{const}$ | mittlerer quadratischer Fehler (MSE) |

<!--en:7165db82-->
| Bernoulli-Verteilung | $-\,y\log\hat p - (1-y)\log(1-\hat p)$ | binäre Kreuzentropie (BCE) |

<!--en:3b1b3acf-->
| Kategoriale Verteilung (One-Hot-kodiertes $y$) | $-\sum_k y_k \log \hat p_k$ | Softmax-Kreuzentropie |

<!--en:a5eb476b-->
| Laplace-Verteilung mit festem Skalenparameter $s$ | $\tfrac{1}{s}\lvert y-\hat y\rvert + \textrm{const}$ | mittlerer absoluter Fehler (MAE) |

<!--en:1f419ee2-->
Bei all diesen Verlustfunktionen wählen wir mit der Funktion zugleich ein Rauschmodell. Es gibt aber auch nützliche Verluste, die sich so nicht herleiten lassen. Dem Hinge-Verlust der Support-Vektor-Maschine fehlt eine Normierungskonstante, die unabhängig von der Modellausgabe ist. Daher entspricht er selbst bis auf eine additive Konstante keiner NLL einer bedingten Verteilung (Übung :exref:`ex_mdl-maximum-likelihood-losses-not-likelihoods`).

<!--en:94c9ee22-->
Sobald die Daten einen Ausreißer enthalten, macht sich die Wahl des Rauschmodells bemerkbar. Wir passen eine einzige Konstante $c$ an fünf Sensormesswerte $\{1,2,3,4,100\}$ an. Der letzte Messwert stammt von einem defekten Sensor. Bei gaußschem Rauschen minimieren wir $\tfrac12\sum_i(y_i-c)^2$. Die Ableitung $-\sum_i(y_i-c)$ verschwindet beim Stichprobenmittelwert $c=22$. Bei Laplace-Rauschen minimieren wir $\sum_i\lvert y_i-c\rvert$, eine Summe V-förmiger Kurven. Ihre Steigung bei $c$ ist die Anzahl der Messwerte unterhalb von $c$ abzüglich der Anzahl oberhalb. Links vom Median ist die Steigung negativ, rechts davon positiv. Das Minimum liegt also beim Median $c=3$. Hätte der fünfte Messwert $5$ betragen, hätten beide Rauschmodelle $3$ geliefert. Wie :numref:`fig_mdl-mle-gauss-laplace` zeigt, zieht der fehlerhafte Messwert die Schätzung unter gaußschem Rauschen bis auf $22$. Die Schätzung unter Laplace-Rauschen bleibt gleich.

<!--en:e811a738-->
![Anpassung einer Konstanten $c$ an die Messwerte $1, 2, 3, 4, 100$. (a) Negative Log-Likelihood pro Beispiel als Funktion des Residuums, quadratisch bei gaußschem Rauschen und V-förmig bei Laplace-Rauschen. (b) Der summierte quadratische Verlust bildet eine Parabel. Ihr Minimum beim Mittelwert $22$ liegt durch den Ausreißer weit in dessen Richtung. (c) Der summierte absolute Verlust hat bei jedem Messwert einen Knick (schwarze Markierungen) und erreicht ihr Minimum beim Median $3$. Jeder Punkt zieht unabhängig von seiner Entfernung mit einer Steigung vom Betrag eins an $c$.](../img/mdl-prob-mle-gauss-laplace.svg)

<!--en:1d135791-->
## Genauigkeit des Maximum-Likelihood-Schätzers

<!--en:85d0ff97-->
Kommen wir auf unsere Ausgangsfrage zurück. Wie sicher können wir uns bei $9/13$ sein? Für einen Stichprobenanteil haben wir in :numref:`sec_mdl-statistics` den Standardfehler $\sqrt{\hat\theta(1-\hat\theta)/n}$ angegeben. Er schätzt die Standardabweichung von $\hat\theta$ über wiederholte Stichproben hinweg. In einem allgemeinen Modell ist der MLE jedoch kein einfacher Stichprobenmittelwert. Wir brauchen daher ein allgemeines Verfahren, um den Fehlerbalken direkt aus der Verlustfunktion abzulesen. Dazu genügen zwei Größen, der Gradient pro Beispiel, den wir von der Münze kennen, und die Krümmung der NLL.

<!--en:c3704833-->
### Score und Fisher-Information

<!--en:17c5aa23-->
Der **Score** einer einzelnen Beobachtung ist der Gradient ihrer Log-Wahrscheinlichkeit bezüglich der Parameter,

<!--en:8a3dd1d6-->
Damit übertragen wir den Gradienten eines einzelnen Münzwurfs auf beliebige probabilistische Modelle. Die Gleichung $\sum_i s(x_i;\hat{\boldsymbol{\theta}}) = \mathbf 0$ ist die Optimalitätsbedingung erster Ordnung, mit der wir den MLE bestimmen.

<!--en:007dc6c1-->
Der Score hat eine bemerkenswerte Eigenschaft. Am wahren Parameter ist sein Erwartungswert null. Das sehen wir, wenn wir die Normierungsbedingung $\int p(x\mid\boldsymbol{\theta})\,dx = 1$ unter dem Integralzeichen ableiten:

<!--en:2943a4e2-->
Im dritten Schritt haben wir die Identität für die logarithmische Ableitung $\nabla_{\boldsymbol{\theta}}\, p = p\,\nabla_{\boldsymbol{\theta}} \log p = p\,s$ verwendet. Mitteln wir über Daten aus dem Modell, so verschwindet der Gradient der Log-Likelihood an dem Parameterwert, der die Daten erzeugt hat. Weil der Score den Erwartungswert null hat, stimmt seine Kovarianzmatrix mit seinem zweiten Moment überein. Diese Kovarianz ist die **Fisher-Information** :cite:`Fisher.1925`,

<!--en:9e397d6f-->
Die mittlere Gleichheit folgt aus dem verschwindenden Erwartungswert. Die letzte Gleichheit ist die *Informationsidentität*. Die Varianz des Scores ist gleich der erwarteten Krümmung der NLL. Das folgt, wenn wir $\mathbb{E}_{\boldsymbol{\theta}}[s]=\mathbf 0$ erneut ableiten und die Ableitung wieder unter das Integralzeichen ziehen. Nach der Produktregel zerfällt die Ableitung in zwei Terme,

<!--en:f4f453b5-->
Setzen wir im zweiten Integral $\nabla_{\boldsymbol{\theta}}\, p = p\,s$ ein, so wird es zu $\mathbb{E}_{\boldsymbol{\theta}}[\,s\,s^\top]$. Damit bleibt

<!--en:fe6e04fd-->
Durch Umstellen erhalten wir unmittelbar die letzte Gleichheit in :eqref:`eq_mdl-fisher`. Anschaulich grenzt eine stark gekrümmte NLL den Parameter eng ein. Bei einer flachen NLL bleibt viel Spielraum. Die Fisher-Information misst die Krümmung der NLL einer einzelnen Beobachtung, gemittelt über Daten aus dem Modell.

<!--en:f0f4b148-->
In beiden Herleitungen haben wir die Ableitung ohne weitere Begründung unter das Integralzeichen gezogen. Die folgenden Ergebnisse setzen einige *Regularitätsbedingungen* voraus, die wir hier zusammenstellen:

<!--en:8a183e75-->
- Die Dichte ist in $\boldsymbol{\theta}$ zweimal stetig differenzierbar.

<!--en:83e9392f-->
- Ihr Träger, die Menge aller $x$ mit $p(x\mid\boldsymbol{\theta})>0$, hängt nicht von $\boldsymbol{\theta}$ ab.

<!--en:969da502-->
- Ableitung und Integral lassen sich vertauschen (das Verfahren aus :numref:`sec_mdl-expectation-gradients`). Das rechtfertigt jeweils die zweite Gleichheit in den beiden Herleitungen. Bei einem veränderlichen Träger würde dieser Schritt scheitern.

<!--en:92987ba2-->
- Der wahre Parameter liegt im Inneren des Parameterraums.

<!--en:2aede1c9-->
- Die Fisher-Information ist endlich und nicht singulär.

<!--en:a26379ff-->
Die Übungen :exref:`ex_mdl-maximum-likelihood-relu-symmetries` und :exref:`ex_mdl-maximum-likelihood-moving-support` zeigen konkret, welche Schritte scheitern, wenn die Fisher-Information singulär ist oder sich der Träger mit dem Parameter ändert.

<!--en:c9a42c40-->
Berechnen wir $I(\theta)$ für drei zunehmend reichhaltige Modelle. Beginnen wir mit der Münze. Ein einzelner Wurf $x\in\{0,1\}$ hat die Log-Likelihood $\log p(x\mid\theta) = x\log\theta + (1-x)\log(1-\theta)$. Sein Score und die zweite Ableitung der Log-Likelihood sind also

<!--en:f2ddbe7f-->
Bilden wir $-\mathbb{E}[\,\cdot\,]$ und verwenden $\mathbb{E}[x]=\theta$ liefert die Fisher-Information pro Wurf und ihren durch $n$ geteilten Kehrwert,

<!--en:09263b4b-->
Die Information pro Wurf ist bei $\theta = 1/2$ am kleinsten. Bei einer fairen Münze lässt sich die Kopfwahrscheinlichkeit also am schwersten eingrenzen, weil jeder Wurf am wenigsten über sie verrät.

<!--en:672e7d8f-->
Ein Modell mit zwei Parametern veranschaulicht die Matrixform. Für eine Normalverteilung mit Mittelwert $\mu$ und Standardabweichung $\sigma$ beträgt die NLL einer Beobachtung bis auf eine Konstante $\log\sigma + (x-\mu)^2/(2\sigma^2)$. Ihre zweite Ableitung nach $\mu$ ist $1/\sigma^2$. Die zweite Ableitung nach $\sigma$ ist $-1/\sigma^2+3(x-\mu)^2/\sigma^4$ und hat den Erwartungswert $2/\sigma^2$. Die gemischte Ableitung $2(x-\mu)/\sigma^3$ hat den Erwartungswert null. Daher verschwinden die Einträge außerhalb der Diagonalen, und es gilt $I(\mu,\sigma)=\operatorname{diag}(1/\sigma^2,\,2/\sigma^2)$. In Richtung $\sigma$ ist die NLL doppelt so stark gekrümmt wie in Richtung $\mu$. Die Daten legen also die Skala genauer fest als den Mittelwert. :numref:`fig_mdl-prob-mle-fisher` zeigt die schüsselförmige Verlustfläche für $30$ Beobachtungen.

<!--en:e968da41-->
![Negative Log-Likelihood der Parameter $(\mu,\sigma)$ einer Normalverteilung für $n=30$ Beobachtungen aus $\mathcal{N}(1, 1.5^2)$. Der Punkt markiert den MLE. Die gestrichelte Ellipse stellt $I(\hat\mu,\hat\sigma)^{-1}/n$ dar. Sie folgt eng der innersten Höhenlinie, auf der die NLL um ein Halb über ihrem Minimum liegt, und ist in Richtung $\mu$ breiter als in Richtung $\sigma$.](../img/mdl-prob-mle-fisher.svg)

<!--en:0d3907fc-->
Sowohl bei der Münze als auch beim Mittelwert der Normalverteilung ist der MLE ein Stichprobenmittelwert. Seine Streuung hätten wir also ohne Fisher-Information bestimmen können. Was die Theorie darüber hinaus leistet, zeigt ein Modell, dessen MLE kein Mittelwert ist. Das *Cauchy-Lagemodell* hat die Dichte

<!--en:941293b1-->
eine symmetrische Glockenkurve mit Zentrum $\theta$. Ihre Verteilungsenden fallen so langsam ab, dass ihr Erwartungswert nicht existiert, wie wir in :numref:`sec_mdl-random_variables` gesehen haben. Schreiben wir $u = x-\theta$, so ist der Score $s = 2u/(1+u^2)$ und seine Ableitung $\partial_\theta s = -2(1-u^2)/(1+u^2)^2$. Mit einer trigonometrischen Substitution können wir die Fisher-Information direkt ablesen. Setzen wir $u=\tan\varphi$. Wegen $d\varphi = du/(1+u^2)$ ist der Winkel $\varphi$ einer standardmäßig Cauchy-verteilten Zufallsvariablen gleichverteilt auf $(-\pi/2,\pi/2)$, und es gilt

<!--en:740bd074-->
Jede Cauchy-Beobachtung liefert also eine halbe Informationseinheit über den Lageparameter, ebenso viel wie eine normalverteilte Beobachtung mit $\sigma^2=2$. Der Betrag des Scores überschreitet nie $1$. Bei weit entfernten Punkten geht er sogar gegen null. Deshalb verschieben selbst extreme Ausreißer den MLE kaum.

<!--en:a4df8732-->
Auch ohne geschlossene Lösung lassen sich beide Formen von $I$ durch automatisches Differenzieren direkt berechnen. Die folgende Codezelle zieht $10{,}000$ Cauchy-verteilte Beobachtungen bei $\theta^\star=0$. Sie gibt den mittleren Score, die Varianz des Scores und die mittlere Krümmung mit umgekehrtem Vorzeichen aus. Aus der Darstellung mit gleichverteiltem Winkel ergeben sich für diese drei Schätzungen Monte-Carlo-Standardfehler von etwa $0.007$, $0.004$ und $0.008$.

<!--en:7aad83b8-->
Das automatische Differenzieren in MXNet kann die für diesen Score benötigte zweite Ableitung nicht berechnen. Daher nähert die Codezelle die Krümmung durch einen zentralen Differenzenquotienten aus zwei automatisch berechneten Scores bei $\theta\pm10^{-3}$ an.

<!--en:e8ddb529-->
`torch.func.grad` differenziert die Log-Dichte einer Beobachtung, und `vmap` wendet die Funktion auf alle $10{,}000$ Beobachtungen an. Zweimaliges Anwenden von `grad` liefert die Krümmung.

<!--en:da2443cf-->
Jede Beobachtung erhält eine eigene Kopie von $\theta$. Der Gradient der summierten Log-Dichte bezüglich des so gebildeten Vektors ist daher der Vektor der einzelnen Scores. Eine zweite Aufzeichnung der Rechenschritte ermöglicht es, die Scores erneut abzuleiten.

<!--en:732cb94b-->
`jax.grad` differenziert die Log-Dichte einer Beobachtung, und `vmap` wendet die Funktion auf alle $10{,}000$ Beobachtungen an. Zweimaliges Anwenden von `grad` liefert die Krümmung.

<!--en:c431135e-->
Die drei Schätzwerte aus der Stichprobe liegen jeweils höchstens zwei bis drei Standardfehler von ihren theoretischen Werten $0$, $0.5$ und $0.5$ entfernt.

<!--en:e527287c-->
### Der Fehlerbalken für die Münze

<!--en:05279c70-->
Die Fisher-Information liefert den Fehlerbalken für $9/13$, nach dem wir zu Beginn des Abschnitts gefragt haben. Im nächsten Unterabschnitt zeigen wir, dass die Varianz des MLE für große $n$ näherungsweise $1/\big(nI(\theta)\big)$ beträgt. Nehmen wir dieses Ergebnis vorweg, können wir es sofort anwenden. Für die Münze gilt $1/(nI)=\theta(1-\theta)/n$ nach :eqref:`eq_mdl-coin-fisher`. Setzen wir $\hat\theta$ für den unbekannten Wert $\theta$ ein, erhalten wir den geschätzten Standardfehler

<!--en:549ebac3-->
Damit gilt $\hat\theta = 0.69 \pm 0.13$. Der Wert für eine faire Münze liegt deutlich innerhalb des näherungsweisen $95\%$-Konfidenzintervalls $\hat\theta\pm1.96\,\widehat{\textrm{se}}=[0.44,\,0.94]$, denn dieses enthält $1/2$. Dreizehn Würfe reichen also nicht aus, um diese Münze von einer fairen zu unterscheiden. Um $\theta$ mit einer Genauigkeit von $\pm0.05$ bei einem Konfidenzniveau von $95\%$ zu bestimmen, wären etwa $1.96^2\,\hat\theta(1-\hat\theta)/0.05^2 = 1.96^2\cdot 0.213/0.05^2\approx 330$ Würfe nötig.

<!--en:d1204c35-->
Entscheidend ist, dass genau dieselbe Zahl direkt aus der Verlustfunktion folgt. Für die zweite Ableitung der NLL der Münze an der Stelle $\hat\theta$ gilt

<!--en:b1166d15-->
und $1/\sqrt{61.0} = 0.128$. Für die Münze ist die Übereinstimmung exakt. Wegen $n_H=n\hat\theta$ und $n_T=n(1-\hat\theta)$ ist die beobachtete Krümmung gleich $n/\big(\hat\theta(1-\hat\theta)\big) = nI(\hat\theta)$. Für jedes Modell, das unsere Regularitätsbedingungen erfüllt, gilt dieser Zusammenhang bei großem $n$ näherungsweise. Nach dem Anpassen des Modells können wir die Krümmung der NLL an ihrem Minimum berechnen (ihre Hesse-Matrix). Bei einem Parameter liefert der Kehrwert der Quadratwurzel der Krümmung den Fehlerbalken. Bei einem Parametervektor sind die einzelnen Fehlerbalken die Quadratwurzeln der Diagonaleinträge der inversen Hesse-Matrix. Tiefe neuronale Netze verletzen diese Regularitätsbedingungen. Welche Teile des Verfahrens dort noch gelten, erfahren wir in :numref:`subsec_mdl-mle-limits`. Die Krümmung erklärt auch die Likelihood-Kurven vom Anfang des Abschnitts. Zehnmal so viele Daten verzehnfachen die Krümmung und verkleinern die Breite der Kurve um den Faktor $\sqrt{10}\approx 3.2$.

<!--en:2e0c278a-->
### Konsistenz und asymptotische Normalität

<!--en:4c58fd2b-->
Warum beträgt die Varianz des MLE näherungsweise $1/(nI)$? Zunächst müssen wir zeigen, dass sich der MLE überhaupt dem richtigen Wert nähert. Anschließend müssen wir seine Schwankungen um diesen Wert beschreiben. Dazu verwenden wir drei Werkzeuge aus früheren Abschnitten. Aus :numref:`sec_mdl-statistics` übernehmen wir die Konsistenz ($\hat{\boldsymbol{\theta}}\xrightarrow{P}\boldsymbol{\theta}^\star$) und das schwache Gesetz der großen Zahlen. Aus :numref:`sec_mdl-distributions` verwenden wir den zentralen Grenzwertsatz. Die Summe von $n$ unabhängigen, identisch verteilten Termen mit Erwartungswert null und Varianz $v$ konvergiert nach Division durch $\sqrt n$ in Verteilung ($\xrightarrow{d}$) gegen $\mathcal N(0,v)$.

<!--en:98d30528-->
Die Identität für die Kreuzentropie aus :numref:`subsec_mdl-nll-crossentropy` erklärt, warum sich der MLE dem wahren Parameter nähert. Angenommen, die Daten stammen aus dem Modell mit wahrem Parameter $\boldsymbol{\theta}^\star$. Die mittlere NLL ist ein Mittelwert unabhängiger, identisch verteilter Terme. Nach dem Gesetz der großen Zahlen konvergiert sie daher gegen die NLL der Grundgesamtheit,

<!--en:cdb956b6-->
wobei für stetige Dichten die differentielle Entropie $h$ an die Stelle von $H$ tritt (:numref:`subsec_mdl-mle-densities`). Nach der Gibbs-Ungleichung (:numref:`subsec_mdl-max-entropy-derivation`) erreicht die Zielfunktion ihr Minimum dort, wo die KL-Divergenz verschwindet, also bei $\boldsymbol{\theta}=\boldsymbol{\theta}^\star$. Dieser Minimierer ist eindeutig, wenn das Modell **identifizierbar** ist, wenn also verschiedene Parametervektoren stets verschiedene Verteilungen festlegen. Neuronale Netze sind nicht identifizierbar. Vertauschen wir die versteckten Einheiten einer Schicht samt ihren ein- und ausgehenden Gewichten, ändert sich der Parametervektor, während das Netz dieselbe Funktion berechnet. Bei solchen Modellen bezieht sich die Konsistenz auf die angepasste Verteilung oder auf die Äquivalenzklasse der Parameter, die sie erzeugen. Ein einzelner Parametervektor ist damit nicht festgelegt.

<!--en:e038f6fc-->
Wir geben die Voraussetzungen für Konsistenz ohne Beweis an :cite:`Wasserman.2013`. Der MLE ist konsistent, wenn vier Bedingungen erfüllt sind. Erstens stammen die Daten unabhängig und identisch verteilt aus $p_{\boldsymbol{\theta}^\star}$. Zweitens ist das Modell identifizierbar. Drittens ist das Optimum in der Grundgesamtheit klar getrennt. Kein $\boldsymbol{\theta}$ weit entfernt von $\boldsymbol{\theta}^\star$ liefert also einen Funktionswert, der dem Optimum beliebig nahe kommt (bei kompakter Parametermenge und stetiger Zielfunktion genügt Eindeutigkeit). Viertens konvergiert die mittlere NLL gleichmäßig in $\boldsymbol{\theta}$ gegen die NLL der Grundgesamtheit. Gerade die Gleichmäßigkeit übersieht man leicht. Das Gesetz der großen Zahlen garantiert die Konvergenz für jedes feste $\boldsymbol{\theta}$. Punktweise Konvergenz allein erzwingt jedoch keine Konvergenz der Minimierer. Bei der Münze erübrigt sich dieser Satz. Aus dem Gesetz der großen Zahlen folgt direkt $\hat\theta = n_H/n\to\theta^\star$.

<!--en:1c4a0656-->
Die Konsistenz sagt uns, wohin $\hat{\boldsymbol{\theta}}$ konvergiert. Für einen Fehlerbalken brauchen wir die Größenordnung und die Form der Verteilung seiner Schwankungen um $\boldsymbol{\theta}^\star$. Für einen skalaren Parameter $\theta$ genügen drei kurze Schritte. Der entscheidende Gedanke ist, die Optimalitätsbedingung erster Ordnung in eine Taylorreihe zu entwickeln. Eine Entwicklung der NLL selbst erübrigt sich. Wir schreiben $S_n(\theta)=\sum_i s(x_i;\theta)$ für den gesamten Score.

<!--en:e173c53d-->
**I.** Nach Definition löst der MLE die Gleichung $S_n(\hat\theta)=0$. Entwickeln wir $S_n(\hat\theta)$ bis zur ersten Ordnung um $\theta^\star$, erhalten wir

<!--en:02743cae-->
**II.** Wir lösen nach dem Schätzfehler auf und multiplizieren mit $\sqrt n$. So erhalten wir einen Quotienten aus zwei normierten Summen,

<!--en:74f6a75c-->
**III.** Im Zähler steht eine Summe von $n$ unabhängigen, identisch verteilten Scores mit Erwartungswert null und Varianz $I(\theta^\star)$, geteilt durch $\sqrt n$. Nach dem zentralen Grenzwertsatz konvergiert sie in Verteilung gegen $\mathcal N\big(0, I(\theta^\star)\big)$. Der Nenner ist der Mittelwert von $n$ unabhängigen, identisch verteilten Termen $-\partial_\theta s(x_i;\theta^\star)$. Ihr Erwartungswert ist nach der Informationsidentität $I(\theta^\star)$. Nach dem Gesetz der großen Zahlen konvergiert der Nenner also gegen $I(\theta^\star)$. Teilen wir eine normalverteilte Zufallsvariable mit Varianz $I$ durch die Konstante $I$, erhalten wir eine normalverteilte Zufallsvariable mit Varianz $I/I^2=1/I$.

<!--en:194d1a4d-->
Hier greifen die beiden Formen der Fisher-Information ineinander. In den Zähler geht die Varianz des Scores ein, in den Nenner die erwartete Krümmung. Die Informationsidentität setzt beide mit derselben Zahl $I$ gleich. Deshalb ist die Endformel so einfach. Für Vektoren führt dieselbe Rechnung mit Matrizen auf die Kovarianz $I^{-1}II^{-1}=I^{-1}$. Ein strenger Beweis muss auch das Restglied der Taylorentwicklung beschränken. Dazu dient die integrierbare Schranke für die Krümmung in der folgenden Proposition.

<!--en:89a847ed-->
**Proposition (asymptotische Normalität).** *Es gelten die Regularitätsbedingungen und die obigen Bedingungen für Konsistenz. Gilt zudem $\lVert\nabla^2_{\boldsymbol{\theta}}\log p(x\mid\boldsymbol{\theta})\rVert\le M(x)$ mit $\mathbb{E}_{\boldsymbol{\theta}^\star}[M(x)]<\infty$ für alle $\boldsymbol{\theta}$ in der Nähe von $\boldsymbol{\theta}^\star$, so folgt*

<!--en:29e86e25-->
:cite:`Wasserman.2013`. Über die drei Schritte hinaus führen wir den Beweis hier nicht aus. Dies ist jedoch das zentrale Ergebnis des Abschnitts. Auf ihm beruhen der Fehlerbalken, die Effizienz des MLE und die Aussagen zu den Grenzen der Theorie. Anschaulich ist der Schätzfehler des MLE näherungsweise normalverteilt und schrumpft mit der Rate $1/\sqrt n$. Die inverse Fisher-Information bestimmt seine Kovarianz. Bei endlichem $n$ verwenden wir die Näherung $\textrm{Cov}(\hat{\boldsymbol{\theta}})\approx I(\boldsymbol{\theta}^\star)^{-1}/n$. Daraus haben wir den Fehlerbalken für die Münze bestimmt. Um den Fehlerbalken zu halbieren, benötigen wir also viermal so viele Daten. Ist das Modell fehlspezifiziert, stimmen die Ausdrücke für $I$ aus Varianz und Krümmung nicht mehr überein. Die Kovarianz ergibt sich dann aus der „Sandwich“-Formel in :numref:`subsec_mdl-mle-limits`.

<!--en:dea29a37-->
Um die asymptotische Normalität an einem anspruchsvolleren Beispiel zu prüfen, brauchen wir ein Modell, dessen MLE kein Stichprobenmittelwert ist. Dafür eignet sich das Cauchy-Lagemodell. Seine Likelihood-Gleichung $\sum_i 2u_i/(1+u_i^2)=0$ hat keine geschlossene Lösung. Wir lösen sie numerisch mit dem **Fisher-Scoring**. Das ist das Newton-Verfahren, bei dem wir für die beobachtete Krümmung die erwartete Krümmung $nI=n/2$ einsetzen:

<!--en:114265e4-->
Die beobachtete Krümmung durch $nI$ zu ersetzen hat einen konkreten Vorteil. Die erwartete Krümmung ist immer positiv. Die beobachtete Krümmung einer Cauchy-NLL kann abseits des Minimums negativ sein, sodass ein gewöhnlicher Newton-Schritt die NLL erhöht. Wir beginnen jeden Lauf beim Stichprobenmedian, der bereits nahe an der Lösung liegt. Für $4{,}000$ Datensätze mit je $n=100$ Beobachtungen bei $\theta^\star=0$ passen wir das Modell mit jeweils zehn Fisher-Scoring-Schritten an. Ist die asymptotische Näherung schon bei $n=100$ genau, sollte das Histogramm von $\sqrt n\,\hat\theta$ mit $\mathcal N(0, 1/I)=\mathcal N(0,2)$ übereinstimmen.

<!--en:d9e232f8-->
Das Histogramm folgt der vorhergesagten Kurve in Lage, Breite und Höhe. Die Standardabweichung von $\sqrt n\,\hat\theta$ liegt in den verschiedenen Läufen zwischen etwa $1.40$ und $1.47$, nahe am asymptotischen Grenzwert $\sqrt2\approx 1.41$. Bei $n=100$ liegt die Varianz noch einige Prozent über ihrem Grenzwert.

<!--en:b312e6ed-->
### Die Cramér-Rao-Schranke

<!--en:97cd0695-->
Ist eine Varianz von $1/(nI)$ gut, oder könnte ein geschickter gewählter Schätzer sie unterbieten? Tatsächlich liefert die Fisher-Information auch eine Untergrenze, die kein im Mittel richtiger Schätzer unterschreiten kann. Aus :numref:`sec_mdl-statistics` wissen wir, dass ein Schätzer $\tilde\theta$ *erwartungstreu* (engl. unbiased) heißt, wenn $\mathbb{E}_\theta[\tilde\theta]=\theta$ für jedes $\theta$ gilt.

<!--en:c89ef2c6-->
**Proposition (Cramér-Rao-Schranke).** *Unter den Regularitätsbedingungen gilt für jeden erwartungstreuen Schätzer $\tilde\theta$ eines skalaren Parameters aus $n$ unabhängigen, identisch verteilten Beobachtungen*

<!--en:ba5ad86f-->
:cite:`Cramer.1946,Radhakrishna-Rao.1945`. Mit anderen Worten, kein erwartungstreuer Schätzer kann präziser sein, als es die Krümmung der Likelihood zulässt. Der Beweis braucht nur drei Zeilen. Sein Grundgedanke ist, die Kovarianz zwischen Schätzer und Score zu betrachten.

<!--en:80c84bc8-->
**Beweis.** Sei $S_n=\partial_\theta\log P(X\mid\theta)$ der Score des gesamten Datensatzes, also die Summe von $n$ unabhängigen Scores einzelner Beispiele. Sein Erwartungswert ist null und seine Varianz $nI(\theta)$. Wegen der Erwartungstreue gilt $\int\tilde\theta(X)\,P(X\mid\theta)\,dX=\theta$. Leiten wir beide Seiten unter dem Integralzeichen ab und setzen $\partial_\theta P = P\,S_n$ ein, erhalten wir

<!--en:ae413624-->
wobei die letzte Gleichheit aus $\mathbb{E}[S_n]=0$ folgt. Mit der Cauchy-Schwarz-Ungleichung für Kovarianzen erhalten wir $1\le\textrm{Var}(\tilde\theta)\textrm{Var}(S_n)=\textrm{Var}(\tilde\theta)\,nI(\theta)$.
$\blacksquare$

<!--en:ea09bd10-->
Die Erwartungstreue geht über die „$1$“ auf der linken Seite ein. Den festen Träger brauchen wir beim Ableiten unter dem Integralzeichen. In Übung :exref:`ex_mdl-maximum-likelihood-moving-support` finden wir einen erwartungstreuen Schätzer, der die Schranke unterschreitet, wenn sich der Träger mit $\theta$ ändert. Für einen Parametervektor verallgemeinert sich die Schranke zu $\textrm{Cov}(\tilde{\boldsymbol{\theta}}) \succeq I(\boldsymbol{\theta})^{-1}/n$ bezüglich der durch positiv semidefinite Matrizen definierten Ordnung.

<!--en:f529d604-->
Die asymptotische Varianz :eqref:`eq_mdl-asymptotic-normality` des MLE erreicht diese Untergrenze genau. Der MLE ist daher *asymptotisch effizient*. Im Grenzfall großer Stichproben erreicht kein erwartungstreuer Schätzer eine kleinere Varianz. Bei der Münze erreichen wir die Schranke sogar für jedes endliche $n$, denn $n_H/n$ ist ein erwartungstreuer Stichprobenmittelwert mit Varianz $\theta(1-\theta)/n = 1/(nI)$. Im Allgemeinen erreicht der MLE die Schranke jedoch erst für $n\to\infty$. Bei endlichem $n$ kann er verzerrt sein. Dann gilt die Schranke für ihn nicht, wie wir gleich am Beispiel der Varianz einer Normalverteilung sehen.

<!--en:aaf76e30-->
Welche Rolle spielt die asymptotische Effizienz in der Praxis? Beim Cauchy-Lagemodell bietet sich der Stichprobenmedian als Alternative an, während der Stichprobenmittelwert die naive Wahl wäre. Aus Symmetriegründen sind der MLE und der Median erwartungstreu. Für das $n$-Fache der Varianz beträgt die Cramér-Rao-Schranke $1/I=2$. Die asymptotische Theorie des Medians sagt dagegen $\pi^2/4\approx 2.47$ voraus. Überlegen Sie vor dem Lesen der Ausgabe, wie sich der Stichprobenmittelwert verhalten wird. Die Codezelle verwendet wieder unsere $4{,}000$ Datensätze.

<!--en:ae7d8aa4-->
Beim MLE liegt die skalierte Varianz zwischen etwa $1.9$ und $2.2$, also nahe an der Untergrenze $2$. Beim Median liegt sie nahe bei $2.5$. Für denselben Fehlerbalken benötigt der MLE etwa $20\%$ weniger Beobachtungen. Der Stichprobenmittelwert zeigt dagegen eine enorme empirische Varianz, die von Lauf zu Lauf stark schwankt. Der Mittelwert von $n$ unabhängigen Cauchy-verteilten Zufallsvariablen ist selbst wieder Cauchy-verteilt. Mitteln bringt hier also keinerlei Vorteil.

<!--en:865e4dff-->
### Grenzen der Theorie

<!--en:5fa48a4b-->
Die obigen Garantien gelten unter den angegebenen Regularitätsbedingungen und im Grenzfall $n\to\infty$. In den ersten vier der folgenden Fälle ist jeweils eine dieser Voraussetzungen verletzt. In den Aufgaben untersuchen wir sie im Einzelnen. Der fünfte Fall, ein fehlspezifiziertes Modell, ist im praktischen Einsatz von Deep Learning die Regel. Daher behandeln wir ihn hier.

<!--en:bf3ab6c1-->
- *Verzerrung bei endlicher Stichprobe.* Der MLE für die Varianz einer Normalverteilung teilt die Summe der quadrierten Residuen durch $n$. Damit unterschätzt er $\sigma^2$ im Mittel um den Faktor $(n-1)/n$ (:numref:`sec_mdl-statistics`). Der MLE ist im Grenzfall optimal, aber nicht bei jedem $n$ erwartungstreu.

<!--en:484090d7-->
- *Der Rand.* Liegt ein wahrer Parameter nahe am Rand seines Wertebereichs, braucht die Normalapproximation weit mehr Daten, um genau zu werden. Ein Beispiel ist eine Münze mit Kopfwahrscheinlichkeit $0.95$. Entscheidend ist die erwartete Anzahl des seltenen Ergebnisses. Der Stichprobenumfang $n$ allein reicht nicht aus. Am Rand selbst existiert der MLE unter Umständen gar nicht (Übung :exref:`ex_mdl-maximum-likelihood-small-samples-boundary`).

<!--en:5f8cb019-->
- *Ein vom Parameter abhängiger Träger.* Für die Gleichverteilung auf $[0,\theta]$ ist der MLE $\max_i x_i$. Sein Fehler schrumpft mit der Rate $1/n$ anstelle von $1/\sqrt n$. Eine erwartungstreue Umskalierung unterschreitet die Cramér-Rao-Schranke (Übung :exref:`ex_mdl-maximum-likelihood-moving-support`).

<!--en:40f2c419-->
- *Singuläre Information.* Aufgrund der Skalierungssymmetrie eines ReLU-Netzes hat die Fisher-Matrix $I$ einen von null verschiedenen Vektor im Kern. Daher existiert $I^{-1}$ nicht (Übung :exref:`ex_mdl-maximum-likelihood-relu-symmetries`).

<!--en:f8a74e93-->
*Fehlspezifikation.* Stimmt kein Mitglied $p_{\boldsymbol{\theta}}$ der Modellfamilie mit der wahren, datenerzeugenden Verteilung überein, konvergiert der MLE gegen den Parameter, der die KL-Divergenz von der wahren Verteilung zum Modell minimiert :cite:`White.1982`. Der erwartete Score verschwindet an diesem Grenzwert weiterhin. Doch die Varianz des Scores, $B$, stimmt im Allgemeinen nicht mehr mit der erwarteten Krümmung der NLL, $A$, überein. Schritt III unserer Herleitung liefert dann die Varianz $B/A^2$ anstelle von $1/I$. In Matrixform ergibt sich die *Sandwich*-Kovarianz $\mathbf A^{-1}\mathbf B\mathbf A^{-1}$ (Übung :exref:`ex_mdl-maximum-likelihood-sandwich`). Passen wir zum Beispiel ein Poisson-Modell mit Rate $\lambda$ an überdisperse Zähldaten mit Mittelwert $m$ und Varianz $m+m^2/\alpha$ an. Da der Poisson-MLE der Stichprobenmittelwert ist, schätzt er weiterhin $m$ und hat die tatsächliche Varianz $(m+m^2/\alpha)/n$. Die Fisher-Information $I=1/\lambda$ des Poisson-Modells sagt jedoch die Varianz $m/n$ voraus. Sie ist um den Faktor $1+m/\alpha$ zu klein. Dieser Faktor beträgt $3$ für $m=4$ und $\alpha=2$.

<!--en:c9fa697c-->
Ein tiefes neuronales Netz ist sowohl fehlspezifiziert als auch nicht identifizierbar. Daher können wir einzelnen Gewichten keine klassischen Fehlerbalken zuordnen. Die Theorie erlaubt weiterhin Aussagen über die angepasste Vorhersageverteilung, die sich unter Parametersymmetrien nicht ändert. Die Fisher-Matrix begegnet uns später in diesem Buch zweimal wieder. Sie beschreibt die lokale Krümmung (Hesse-Matrix) der KL-Divergenz zwischen benachbarten Verteilungen einer parametrischen Familie (:numref:`sec_mdl-divergences-distances`). Außerdem dient sie bei der Optimierung mit dem natürlichen Gradienten als Vorkonditionierer (:numref:`sec_mdl-adaptive-stochastic-methods`).

<!--en:faa31af0-->
## MAP-Schätzung und Regularisierung

<!--en:5097e874-->
Dreizehn Würfe haben uns eine plausible Schätzung geliefert. Bei drei Würfen kann Maximum-Likelihood völlig versagen. Angenommen, wir erhalten dreimal Kopf (HHH). Der MLE ist $\hat\theta=3/3=1$ und weist Zahl die Wahrscheinlichkeit null zu. Schon die erste Zahl, die wir später beobachten, führt also zu einer unendlichen NLL. Dasselbe Problem hat jeder Klassifikator auf der Grundlage von Häufigkeiten, sobald er auf eine Kombination aus Merkmal und Klasse trifft, die beim Training nie vorkam. Wer schon einmal eine Münze in der Hand hatte, glaubt kaum, dass dreimal Kopf beweist, dass Zahl unmöglich ist. Wir brauchen ein Verfahren, mit dem wir schon vor der ersten Beobachtung ausdrücken können, dass Münzen nur selten immer auf dieselbe Seite fallen.

<!--en:6115d930-->
Eine Wahrscheinlichkeitsverteilung über die Parameter erfasst solches Vorwissen. Bei Maximum-Likelihood war dafür kein Platz, weil wir nur Datensätzen Wahrscheinlichkeiten zugewiesen haben. Führen wir eine **A-priori-Verteilung** $p(\boldsymbol{\theta})$ ein, so verknüpft der Satz von Bayes sie mit der Likelihood. Wir erhalten eine Verteilung über die Parameter bei gegebenen Beobachtungen:

<!--en:3699ef1e-->
Auf der linken Seite steht die **A-posteriori-Verteilung** (engl. posterior). Sie beschreibt unseren Kenntnisstand nach Beobachtung der Daten. Der Nenner ist die **Evidenz**, also die über alle Parameterwerte gemittelte marginale Wahrscheinlichkeit der Daten. Entscheidend ist, dass sie nicht von $\boldsymbol{\theta}$ abhängt. Die **Maximum-a-posteriori-Schätzung** (MAP) ist der Modus der A-posteriori-Verteilung, also die Stelle ihres Maximums. Da die Evidenz bezüglich $\boldsymbol{\theta}$ konstant ist, erhalten wir durch Maximieren der A-posteriori-Dichte und Übergang zu negativen Logarithmen

<!--en:9ad20b90-->
Die MAP-Schätzung minimiert also die vertraute NLL zuzüglich eines Strafterms, des negativen Logarithmus der A-priori-Dichte. Weist die A-priori-Verteilung extremen Parameterwerten wenig Wahrscheinlichkeit zu, bestraft sie diese Werte und wirkt genau wie ein Regularisierungsterm. Für HHH verwenden wir eine A-priori-Verteilung, die einem zusätzlichen fiktiven Kopf und einer fiktiven Zahl entspricht (die unten behandelte $\mathrm{Beta}(2,2)$-Verteilung). Das ergibt $\hat\theta_{\textrm{MAP}}=(3+1)/(3+2)=4/5$. Eine spätere Zahl kostet dann $-\log(1/5)=1.6$ Nat. Die Überraschung ist nun endlich.

<!--en:2f0247a8-->
### Gauß-Prior und Gewichtszerfall

<!--en:6cc7c691-->
Der Gewichtszerfall (engl. weight decay), die übliche Regularisierung im Deep Learning, ist der MAP-Strafterm eines Gauß-Priors. Das gilt für den Gewichtszerfall in Form eines $L_2$-Strafterms. Die entkoppelte Variante für Adam (:numref:`sec_adamw`) entspricht keinem exakten MAP-Strafterm mehr. In :numref:`sec_weight_decay` haben wir den Zusammenhang für die lineare Regression hergeleitet. Hier formulieren wir ihn mit der Notation dieses Abschnitts für beliebige Modelle mit gaußschem Rauschen der Zielwerte. Wir nehmen a priori kleine, voneinander unabhängige Parameter an, $\theta_j \sim \mathcal{N}(0,\tau^2)$. Dabei beschreibt $\tau$ ihre erwartete typische Größenordnung. Die A-priori-Dichte ist proportional zu $\exp\!\big(-\lVert\boldsymbol{\theta}\rVert_2^2/(2\tau^2)\big)$. Ihr negativer Logarithmus ist daher bis auf eine Konstante gleich $\lVert\boldsymbol{\theta}\rVert_2^2/(2\tau^2)$. Zusammen mit der gaußschen NLL :eqref:`eq_mdl-gaussian-nll` ergibt sich für die MAP-Zielfunktion :eqref:`eq_mdl-map`

<!--en:b9013ff1-->
Multiplizieren wir alle Terme mit der positiven Konstanten $\sigma^2/n$, bleibt der Minimierer gleich, und wir erhalten die Zielfunktion mit Gewichtszerfall aus :numref:`sec_weight_decay`,

<!--en:7828932e-->
Wie in diesem Buch üblich können wir den Bias $b$ vom Strafterm ausnehmen. Das entspricht einer flachen A-priori-Verteilung für $b$. Das Gewicht $\lambda$ des Strafterms im gemittelten Verlust nimmt proportional zu $1/n$ ab. Das Vorwissen zählt also am meisten bei wenigen Daten. Mit wachsender Stichprobe verliert es an Einfluss. In :numref:`subsec_mdl-weight-decay-duality` deuten wir denselben Strafterm gleichwertig als Nebenbedingung für die Norm der Gewichte.

<!--en:08ed1622-->
Ersetzen wir den Gauß-Prior durch eine *Laplace*-Verteilung $p(\boldsymbol{\theta})\propto e^{-\lVert\boldsymbol{\theta}\rVert_1/s}$ mit Skalenparameter $s$, so entsteht der $L_1$-Strafterm $\lVert\boldsymbol{\theta}\rVert_1/s$. Sein scharfer Knick bei null begünstigt exakt verschwindende Einträge und damit dünn besetzte Parametervektoren. :numref:`fig_mdl-prob-map-prior` veranschaulicht das Zusammenspiel in einer Dimension. Die NLL zieht die Schätzung zum MLE, der negative Logarithmus der A-priori-Dichte zum A-priori-Mittelwert. Die MAP-Schätzung liegt dazwischen.

<!--en:0a294a35-->
![Maximum a posteriori wägt Daten und Vorwissen gegeneinander ab. Die negative Log-Likelihood (blau) erreicht beim MLE ihr Minimum. Die Addition des negativen Logarithmus einer um ihren A-priori-Mittelwert zentrierten Gauß-Dichte (orange, gestrichelt) ergibt die MAP-Zielfunktion (grün). Ihr Minimum verschiebt sich vom MLE zum A-priori-Mittelwert hin. Eine engere A-priori-Verteilung (kleineres $\tau$) zieht stärker. Für $\tau\to\infty$ wird die A-priori-Verteilung flach, und die MAP-Schätzung geht in den MLE über.](../img/mdl-prob-map-prior.svg)

<!--en:11276f71-->
An derselben Abbildung lassen sich beide Grenzfälle ablesen. Wird die A-priori-Verteilung flach ($\tau\to\infty$ oder $s\to\infty$), verschwindet der Strafterm. Wächst der Stichprobenumfang $n$, so vertieft sich die Mulde der NLL proportional zu $n$, während die A-priori-Verteilung gleich bleibt. In beiden Grenzfällen konvergiert die MAP-Schätzung wieder gegen den MLE.

<!--en:4deca5c6-->
### Beta-A-priori-Verteilung und Pseudohäufigkeiten

<!--en:9dfaa1a8-->
Bei der Münze liegt der Parameter $\theta$ in $[0,1]$. Daher bietet sich die Familie der Beta-Verteilungen als A-priori-Verteilung an. Eine $\mathrm{Beta}(a,b)$-A-priori-Verteilung mit $a,b>0$ hat eine zu $\theta^{a-1}(1-\theta)^{b-1}$ proportionale Dichte. Multiplizieren wir sie mit der Likelihood der Münze :eqref:`eq_mdl-coin-like`, addieren sich die beobachteten Anzahlen zu den Exponenten. Die A-posteriori-Verteilung gehört also wieder zur Beta-Familie. Das ist die konjugierte Aktualisierung aus :numref:`sec_mdl-exponential-family`:

<!--en:c4ec064a-->
Diese A-posteriori-Dichte hat genau die Form einer Münz-Likelihood mit $n_H+a-1$-mal Kopf und $n_T+b-1$-mal Zahl. Sind beide Exponenten positiv, maximieren wir sie genauso wie die Likelihood beim MLE der Münze. Es gilt „Exponent für Kopf geteilt durch die Summe der Exponenten“:

<!--en:652cccfe-->
Beim A-posteriori-Modus wirkt die A-priori-Verteilung also so, als ergänzten wir die Stichprobe um $a-1$-mal Kopf und $b-1$-mal Zahl, bevor wir die relative Häufigkeit berechnen. Das entspricht der Konvention in :numref:`sec_mdl-exponential-family`. Die Parameter $a$ und $b$ sind die *Pseudohäufigkeiten* (engl. pseudo-counts) für den A-posteriori-Mittelwert, $a-1$ und $b-1$ diejenigen für den A-posteriori-Modus. Eine $\mathrm{Beta}(2,2)$-A-priori-Verteilung ergänzt für den Modus einen fiktiven Wurf je Ergebnis. Damit verschiebt sich unser Wert $9/13\approx 0.692$ auf $(9+1)/(13+2) = 2/3$, und für HHH ergibt sich $4/5$. Eine starke $\mathrm{Beta}(30,30)$-A-priori-Verteilung hält die MAP-Schätzung nahe bei $1/2$, bis die beobachteten Würfe mehr Gewicht haben als die $58$ fiktiven.

<!--en:371e2b3f-->
Beide Kenngrößen treffen in einer klassischen Formel zusammen. Bei der flachen A-priori-Verteilung $\mathrm{Beta}(1,1)$ ergibt sich der A-posteriori-Mittelwert aus der Nachfolgeregel von Laplace $(n_H+1)/(n+2)$ :cite:`Laplace.1814`. Bei Klassifikatoren auf Grundlage von Häufigkeiten ist sie als Add-eins-Glättung bekannt. Dasselbe Verhältnis ist die MAP-Schätzung unter $\mathrm{Beta}(2,2)$. Unter der flachen $\mathrm{Beta}(1,1)$-A-priori-Verteilung liefert MAP dagegen den gewöhnlichen MLE. Die Add-eins-Glättung entspricht bei flacher A-priori-Verteilung also dem A-posteriori-Mittelwert. Die MAP-Schätzung ist in diesem Fall ein anderer Wert.

<!--en:02094966-->
### A-posteriori-Modus und A-posteriori-Verteilung

<!--en:78e21cf6-->
Der A-posteriori-Modus hat eine leicht zu übersehende Schwäche. Er hängt davon ab, wie wir das Modell parametrisieren. Bei einer nichtlinearen Variablentransformation kommt in der Dichte ein Jacobi-Faktor hinzu. Er kann ihr Maximum verschieben, obwohl die Wahrscheinlichkeiten aller entsprechenden Bereiche gleich bleiben. Eine $\mathrm{Beta}(1,1)$-A-priori-Verteilung ist flach in $\theta$. Drücken wir genau dasselbe Vorwissen durch die Log-Odds $\eta=\log\frac{\theta}{1-\theta}$ aus, ist die Dichte nicht mehr flach. Die MAP-Schätzung in $\eta$ liefert nach Rücktransformation $10/15$ statt $9/13$ (Übung :exref:`ex_mdl-maximum-likelihood-map-log-odds`). Bei der Likelihood tritt kein Jacobi-Faktor auf, denn ihre Werte sind Datenwahrscheinlichkeiten, die unmittelbar den Punkten im Parameterraum zugeordnet sind. Der MLE ist daher **äquivariant**. Ist $\hat\theta$ der MLE für $\theta$, dann ist $g(\hat\theta)$ der MLE für $g(\theta)$ bei jeder eineindeutigen Transformation $g$ :cite:`Wasserman.2013`. So ist der MLE für die Log-Odds der Münze gleich $\log\frac{9/13}{4/13}=\log\frac94$. Deshalb definieren wir Maximum-Likelihood ohne A-priori-Verteilung. Eine Definition als „MAP mit flacher A-priori-Verteilung“ würde bei einem Wechsel der Parametrisierung ein anderes Ergebnis liefern. Auch Äquivarianz hat ihren Preis. Wie dieselbe Aufgabe zeigt, kann kein Schätzer bei jeder Reparametrisierung zugleich äquivariant und erwartungstreu sein.

<!--en:271a8c86-->
Eine noch grundlegendere Einschränkung besteht darin, dass die MAP-Schätzung unser Wissen ebenso wie der MLE auf einen einzigen Parametervektor reduziert. Die vollständige A-posteriori-Verteilung $p(\boldsymbol{\theta}\mid X)$ erfasst, wie unsicher die Parameter nach Beobachtung der Daten noch sind. Mitteln wir für eine Vorhersage über diese Verteilung, $\int p(x\mid\boldsymbol{\theta})\,p(\boldsymbol{\theta}\mid X)\,d\boldsymbol{\theta}$, berücksichtigen wir die Parameterunsicherheit. Setzen wir nur den Modus ein, geht sie verloren. Da solche Integrale selten geschlossene Lösungen besitzen, befassen wir uns in :numref:`sec_mdl-bayesian-computation` mit ihrer Näherung.

<!--en:039d9c59-->
Zunächst stoßen wir auf ein anderes Hindernis. Bisher haben wir in diesem Abschnitt vorausgesetzt, dass wir die Wahrscheinlichkeit jeder Beobachtung direkt als Funktion der Parameter angeben können. Stammt jeder Datenpunkt aus einer von mehreren versteckten Quellen, ohne dass wir die jeweilige Quelle beobachten, wird diese Wahrscheinlichkeit zu einer Summe über die möglichen Quellen. Der Logarithmus einer Summe lässt sich nicht mehr in einfache Terme zerlegen. In :numref:`sec_mdl-latent-em-elbo` sehen wir, was wir in diesem Fall optimieren.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:72ca6440-->
Maximum-Likelihood wählt die Parameter, unter denen die beobachteten Daten am wahrscheinlichsten sind. In der Praxis minimieren wir dazu die negative Log-Likelihood, also die Summe der Überraschungen der einzelnen Beispiele. Die meisten üblichen Trainingsverluste ergeben sich auf diese Weise aus der Wahl des Rauschmodells. Gaußsches Rauschen führt zum quadratischen Fehler, Laplace-Rauschen zum absoluten Fehler und diskrete Ergebnisse zur Kreuzentropie. Bei diskreten Ergebnissen kann der mittlere Verlust nie unter die Entropie der empirischen Häufigkeitsverteilung fallen. Die Krümmung derselben Verlustfunktion gibt an, wie präzise die Schätzung ist. Unter Regularitätsbedingungen ist der Fehler des MLE näherungsweise normalverteilt mit Varianz $1/(nI)$. Die Fisher-Information $I$ ist sowohl die Varianz des Scores als auch die erwartete Krümmung der NLL. Damit erreicht der MLE asymptotisch die Cramér-Rao-Untergrenze für erwartungstreue Schätzer. Bei fehlspezifizierten oder nicht identifizierbaren Modellen versagen diese klassischen Garantien. Auf tiefe neuronale Netze trifft beides zu. Führen wir schließlich eine A-priori-Verteilung ein, minimieren wir bei der MAP-Schätzung die Summe aus NLL und Regularisierungsterm. Ein Gauß-Prior ergibt Gewichtszerfall, eine Beta-A-priori-Verteilung ergänzt Pseudohäufigkeiten. Die MAP-Schätzung reduziert die A-posteriori-Verteilung allerdings weiterhin auf einen einzigen Punkt.

<!--en:2294243a-->
## Übungen

<!--en:7607c3bb-->
1. **Untergrenze der Kreuzentropie.** Bei diskreten Ergebnissen ist die mittlere NLL gleich $H(\hat p_{\textrm{data}})+D_{\textrm{KL}}(\hat p_{\textrm{data}}\,\|\,p_{\boldsymbol{\theta}})$ (siehe :numref:`subsec_mdl-nll-crossentropy`) und kann daher nicht unter die empirische Entropie fallen.
    1. Berechnen Sie diese Untergrenze in Nat für die Münzwurffolge mit $9$-mal Kopf und $4$-mal Zahl. Berechnen Sie auch den Abstand zur Untergrenze, also die KL-Divergenz, für das Modell einer fairen Münze mit $\theta=1/2$.
    1. Ein Klassifikator modelliert $p(y\mid\mathbf x)$ und kann für jede verschiedene Eingabe unabhängig jede beliebige Labelverteilung darstellen. Zeigen Sie, dass seine kleinste erreichbare mittlere NLL auf einem Trainingsdatensatz die empirische bedingte Entropie $\sum_{\mathbf x}\hat p_{\textrm{data}}(\mathbf x)\,H\big(\hat p_{\textrm{data}}(\cdot\mid\mathbf x)\big)$ ist. Berechnen Sie sie für eine binäre Aufgabe, bei der jede von $100$ verschiedenen Eingaben $10$-mal vorkommt und genau einmal das falsche Label trägt.
    1. Nun kommt jede Trainingseingabe wie in einem typischen Bilddatensatz nur einmal vor. Wie groß ist die Untergrenze aus (b)? Warum sagt ein Trainingsverlust nahe null dann nichts darüber aus, wie stark die Labels verrauscht sind? Für welche messbare Größe ist die bedingte Entropie der Grundgesamtheit $H(Y\mid X)$ eine Untergrenze?
   :exlabel:`ex_mdl-maximum-likelihood-cross-entropy-floor`

<!--en:d1599121-->
1. **Erlernen der Rauschvarianz.** Die gaußsche NLL :eqref:`eq_mdl-gaussian-nll` hält die Varianz $\sigma^2$ fest. Hier lernen wir die Varianz mit.
    1. Führen Sie eine gemeinsame Varianz $\sigma^2$ als weiteren Parameter neben $\boldsymbol{\theta}$ ein. Minimieren Sie die NLL über $\sigma^2$ bei festem $\boldsymbol{\theta}$ und zeigen Sie, dass die verbleibende Zielfunktion bis auf eine Konstante $\tfrac n2\log\big(\tfrac1n\sum_i(y_i-\hat y_i)^2\big)$ ist. Liefert die Minimierung über $\boldsymbol{\theta}$ weiterhin die Lösung der Methode der kleinsten Quadrate?
    1. Ein Netz gibt für jede Eingabe einen Mittelwert $\hat y(\mathbf x)$ und eine Varianz $\hat\sigma^2(\mathbf x)>0$ aus. Bis auf eine additive Konstante beträgt seine NLL auf dem Trainingsdatensatz

       $$\sum_{i=1}^n\Big[\frac{(y_i-\hat y(\mathbf x_i))^2}{2\hat\sigma^2(\mathbf x_i)}+\frac12\log\hat\sigma^2(\mathbf x_i)\Big].$$

       Zeigen Sie, dass diese Zielfunktion nach unten unbeschränkt ist, wenn das Netz einen Trainingspunkt exakt treffen und zugleich dessen Varianz verkleinern kann. Welche Eigenschaft der Likelihood eines gaußschen Mischmodells in :numref:`sec_mdl-latent-em-elbo` hat dieselbe Ursache?
    1. Berechnen Sie den Gradienten des Terms von Beispiel $i$ bezüglich $\hat y(\mathbf x_i)$. Angenommen, die vorhergesagte Varianz folgt dem quadrierten Residuum, also $\hat\sigma^2(\mathbf x_i)\approx r_i^2$ mit $r_i=y_i-\hat y(\mathbf x_i)$. Wie hängt der Betrag des Gradienten dann von $|r_i|$ ab? Wie verhält sich das im Vergleich zum quadratischen Fehler? Wie wirkt sich diese Gewichtung in Bereichen auf die Anpassung des Mittelwerts aus, in denen die Anpassung derzeit schlecht ist :cite:`Seitzer.Tavakoli.Antic.ea.2022`?
   :exlabel:`ex_mdl-maximum-likelihood-learned-noise-variance`

<!--en:d5526387-->
1. **Verluste ohne Likelihood-Entsprechung.** Sei $\phi(y,o)$ eine Verlustfunktion für ein Label $y$ und eine Modellausgabe $o$. Weiter sei $Z(o)=\sum_y e^{-\phi(y,o)}$. Hängt $Z(o)$ nicht von $o$ ab, dann ist $e^{-\phi(y,o)}/Z$ eine bedingte Verteilung von $y$ bei gegebenem $o$. Ihre NLL ist bis auf eine Konstante gleich $\phi$. Die Minimierung von $\phi$ entspricht also Maximum-Likelihood.
    1. Zeigen Sie für Labels $y\in\{-1,+1\}$, dass für den logistischen Verlust $\phi(y,o)=\log(1+e^{-yo})$ die Gleichung $Z(o)=1$ gilt. Welche Wahrscheinlichkeit weist das Modell $y=+1$ zu?
    1. Berechnen Sie $Z(o)$ für den Hinge-Verlust $\phi(y,o)=\max(0,1-yo)$ der Support-Vektor-Maschine, getrennt für $|o|\le1$ und $|o|>1$. Folgern Sie, dass der Hinge-Verlust selbst bis auf eine Konstante nicht die NLL einer bedingten Verteilung von $y$ bei gegebenem $o$ sein kann. Welchen Term müssen Sie zum Hinge-Verlust addieren, um die NLL von $e^{-\phi(y,o)}/Z(o)$ zu erhalten? Wie verhält er sich bei wachsendem $|o|$?
    1. Betrachten Sie für einen reellen Zielwert $y$ und eine Vorhersage $\hat y$ eine beliebige Verlustfunktion des Residuums, $\phi(y,\hat y)=\rho(y-\hat y)$ mit $\int e^{-\rho(r)}\,dr<\infty$, etwa den Huber-Verlust. Zeigen Sie, dass $Z(\hat y)=\int e^{-\rho(y-\hat y)}\,dy$ unabhängig von $\hat y$ ist. Warum ist die Bedingung dieser Aufgabe für solche Regressionsverluste automatisch erfüllt, für Klassifikationsverluste aber einschränkend?
   :exlabel:`ex_mdl-maximum-likelihood-losses-not-likelihoods`

<!--en:d6d0aec3-->
1. [code] **Randfälle bei separierbaren Daten und seltenen Ergebnissen.** Maximum-Likelihood bereitet Probleme, wenn der beste Parameterwert am Rand seines Wertebereichs liegt.
    1. Passen Sie an zwei Punkte mit Labels, $x=-1$ mit $y=0$ und $x=+1$ mit $y=1$, eine logistische Regression $P(y=1\mid x)=\operatorname{sigmoid}(wx)$ ohne Bias an. Zeigen Sie, dass die NLL $2\log(1+e^{-w})$ beträgt, für jedes $w$ streng fällt und gegen $0$ strebt. Zeigen Sie auch, dass kein endliches $w$ diesen Grenzwert erreicht. Welche Münzwurffolge aus dem Text weist dasselbe Problem auf? Was sagt $w\to\infty$ für einen neuen Punkt bei $x=0.1$ vorher?
    1. Ergänzen Sie die A-priori-Verteilung $w\sim\mathcal N(0,\tau^2)$. Zeigen Sie, dass die MAP-Schätzung die Gleichung $2\operatorname{sigmoid}(-w)=w/\tau^2$ löst, und lösen Sie diese numerisch für $\tau=1$. Verallgemeinern Sie das Ergebnis. Zeigen Sie, dass die logistische NLL für jeden linear separierbaren Datensatz das Infimum $0$ und keinen Minimierer hat. Zeigen Sie auch, dass ein Gauß-Prior wieder eine endliche MAP-Schätzung liefert.
    1. Bei der Münze hat $\hat\theta=n_H/n$ die Schiefe $(1-2\theta)/\sqrt{n\theta(1-\theta)}$. Berechnen Sie diese für $\theta=0.7$ und für $\theta=0.95$ bei $n=400$. Bestimmen Sie das $n$, bei dem $\theta=0.95$ dieselbe Schiefe erreicht wie $\theta=0.7$ bei $n=400$. Drücken Sie für $\theta$ nahe $1$ die Schiefe durch die erwartete Anzahl der Würfe mit Zahl $n(1-\theta)$ aus. Warum entscheidet gerade diese Anzahl und nicht $n$ allein darüber, wie gut die Normalapproximation ist?
    1. Simulieren Sie $20{,}000$ Datensätze mit je $n=400$ Würfen (`rng.binomial(n, theta, 20000) / n`). Zeichnen Sie für $\sqrt n(\hat\theta-\theta)$ bei $\theta=0.7$ und $\theta=0.95$ jeweils ein Histogramm und legen Sie die Kurve von $\mathcal N\big(0,\theta(1-\theta)\big)$ darüber. Beschreiben Sie, wie das Histogramm für $\theta=0.95$ von der Kurve abweicht.
   :exlabel:`ex_mdl-maximum-likelihood-small-samples-boundary`

<!--en:01364979-->
1. **MAP-Schätzung für Log-Odds.** Parametrisieren Sie die Münze durch ihre Log-Odds $\eta=\log\frac{\theta}{1-\theta}$, sodass $\theta=\operatorname{sigmoid}(\eta)$ und $\frac{d\theta}{d\eta}=\theta(1-\theta)$ gelten. Behalten Sie die gleichförmige A-priori-Verteilung $\mathrm{Beta}(1,1)$ für $\theta$ und die Folge mit $9$-mal Kopf und $4$-mal Zahl bei.
    1. Bestimmen Sie mit der Transformationsformel für Dichten $p_\eta(\eta)=p_\theta(\theta)\,\lvert d\theta/d\eta\rvert$ aus :numref:`sec_mdl-joint-distributions`, welche A-priori-Dichte für $\eta$ sich aus diesem Vorwissen ergibt. Zeigen Sie, dass es die logistische Dichte $e^{-\eta}/(1+e^{-\eta})^2$ ist, also die Dichte von $\log\frac{U}{1-U}$ für ein auf dem Einheitsintervall gleichverteiltes $U$. Erklären Sie das Ergebnis ohne Rechnung mithilfe der Inversionsmethode.
    1. Bestimmen Sie die MAP-Schätzung für $\eta$ und transformieren Sie sie zurück auf $\theta$. Vergleichen Sie sie mit der MAP-Schätzung in der $\theta$-Parametrisierung, mit dem MLE $\log\frac94$ aus :numref:`subsec_mdl-map-vs-posterior` und mit der Nachfolgeregel von Laplace.
    1. Wiederholen Sie (b) für eine allgemeine A-priori-Verteilung $\mathrm{Beta}(a,b)$ und die Anzahlen $n_H,n_T$. Welche Kenngröße der A-posteriori-Verteilung über $\theta$ liefert die MAP-Schätzung für die Log-Odds nach Rücktransformation?
    1. Der MLE ist äquivariant, also ist der MLE für $\theta^2$ gleich $\hat\theta^2$. Zeigen Sie, dass seine Verzerrung $\mathbb{E}[\hat\theta^2]-\theta^2=\textrm{Var}(\hat\theta)=\theta(1-\theta)/n$ beträgt, und berechnen Sie sie für $\theta=9/13$ und $n=13$. Folgern Sie, dass kein Schätzer unter allen glatten eineindeutigen Reparametrisierungen äquivariant und zugleich für alle erwartungstreu sein kann.
   :exlabel:`ex_mdl-maximum-likelihood-map-log-odds`

<!--en:444bc986-->
1. **Symmetrien eines ReLU-Netzes.** ● Betrachten Sie das Netz $f_{\boldsymbol{\theta}}(x)=\sum_{j=1}^h a_j\,\mathrm{ReLU}(w_jx)$ mit skalarer Eingabe $x$ und Parametern $\boldsymbol{\theta}=(a_1,w_1,\ldots,a_h,w_h)$ sowie das Regressionsmodell $y\sim\mathcal N(f_{\boldsymbol{\theta}}(x),\sigma^2)$ mit festem $\sigma^2$. Die Definition der Identifizierbarkeit finden Sie in :numref:`subsec_mdl-mle-estimator-theory`.
    1. Ersetzen Sie $(a_j,w_j)$ durch $(c\,a_j,\,w_j/c)$ für ein beliebiges $c>0$. Zeigen Sie, dass $f_{\boldsymbol{\theta}}$ dabei unverändert bleibt. Wie viele verschiedene Parametervektoren mit derselben Funktion erhalten Sie durch Permutation der versteckten Einheiten, wenn keine zwei Einheiten übereinstimmen?
    1. Zeigen Sie anhand von :eqref:`eq_mdl-fisher`, dass die Fisher-Information dieses Modells gleich $I(\boldsymbol{\theta})=\mathbb{E}_x\big[\nabla_{\boldsymbol{\theta}}f_{\boldsymbol{\theta}}(x)\,\nabla_{\boldsymbol{\theta}}f_{\boldsymbol{\theta}}(x)^\top\big]/\sigma^2$ ist. Leiten Sie die Umskalierung aus (a) nach $c$ an der Stelle $c=1$ ab. Bestimmen Sie damit einen von null verschiedenen Vektor $\mathbf v$ mit $I(\boldsymbol{\theta})\mathbf v=\mathbf 0$. Welche Voraussetzung für :eqref:`eq_mdl-asymptotic-normality` ist verletzt?
    1. Der Diagonaleintrag der Hesse-Matrix des mittleren quadratischen Fehlers $\tfrac1n\sum_i(y_i-f_{\boldsymbol{\theta}}(x_i))^2$ bezüglich $a_j$ beträgt $\tfrac2n\sum_i\mathrm{ReLU}(w_jx_i)^2$. Wie ändert sich dieser Eintrag bei der Umskalierung aus (a)? Berechnen Sie den Eintrag für $w_j$ und erklären Sie, warum er bei einer ReLU-Einheit (abseits der Knickstellen $w_jx_i=0$) keinen Term mit einem Residuum enthält. Wie ändert er sich bei der Umskalierung? Was folgt daraus für Maße der Flachheit eines Minimums, die auf Eigenwerten der Hesse-Matrix beruhen :cite:`Dinh.Pascanu.Bengio.ea.2017`?
   :exlabel:`ex_mdl-maximum-likelihood-relu-symmetries`

<!--en:4923ca9c-->
1. **Ein veränderlicher Träger.** ● Die Zufallsvariablen $x_1,\ldots,x_n$ seien unabhängig und auf $[0,\theta]$ gleichverteilt, mit unbekanntem $\theta>0$. Dieses Modell verletzt eine der Regularitätsbedingungen aus :numref:`subsec_mdl-mle-estimator-theory`.
    1. Geben Sie die Likelihood als Funktion von $\theta$ an und zeigen Sie, dass der MLE $\hat\theta=\max_i x_i$ ist. Warum finden Sie ihn nicht, indem Sie eine Ableitung gleich null setzen?
    1. Zeigen Sie, dass $P\big(n(\theta-\hat\theta)>t\big)=\big(1-\tfrac{t}{n\theta}\big)^n$ für $0\le t\le n\theta$ gilt, und bestimmen Sie den Grenzwert für $n\to\infty$. Mit welcher Rate in $n$ schrumpft der Fehler? Ist der Grenzwert normalverteilt?
    1. Auf dem Träger ist der Score $s(x;\theta)=-1/\theta$. Berechnen Sie $\mathbb{E}_\theta[s]$. Welcher Schritt in der Herleitung des verschwindenden Erwartungswerts des Scores scheitert (vergleichen Sie Übung :exref:`ex_mdl-integral-calculus-moving-support` in :numref:`sec_mdl-expectation-gradients`)?
    1. Der erwartungstreue Schätzer $\tfrac{n+1}{n}\hat\theta$ hat die Varianz $\theta^2/(n(n+2))$. Vergleichen Sie sie mit dem Wert $1/(n\,\mathbb{E}_\theta[s^2])$, den :eqref:`eq_mdl-cramer-rao` als Untergrenze liefern würde. Welcher Schritt im Beweis der Schranke scheitert hier?
   :exlabel:`ex_mdl-maximum-likelihood-moving-support`

<!--en:f136f1ae-->
1. [extended] **Fehlspezifikation und Sandwich-Kovarianz.** ●● Leiten Sie die asymptotische Normalität heuristisch für den Fall her, dass die beiden Formen der Fisher-Information verschieden sind. Erzeugt kein Mitglied der Modellfamilie die Daten, konvergiert der MLE gegen den pseudowahren Parameter $\boldsymbol{\theta}^\dagger$, der die KL-Divergenz in der Grundgesamtheit minimiert (:numref:`subsec_mdl-mle-limits`). Seine Kovarianz beträgt dann im Allgemeinen nicht mehr $I^{-1}/n$ :cite:`White.1982`. Schreiben Sie $\mathbf A=-\mathbb{E}\big[\nabla^2_{\boldsymbol{\theta}}\log p(x\mid\boldsymbol{\theta}^\dagger)\big]$ und $\mathbf B=\mathbb{E}\big[s(x;\boldsymbol{\theta}^\dagger)\,s(x;\boldsymbol{\theta}^\dagger)^\top\big]$, wobei sich die Erwartungswerte auf die datenerzeugende Verteilung beziehen.
    1. Entwickeln Sie die Bedingung erster Ordnung $\sum_i s(x_i;\hat{\boldsymbol{\theta}})=\mathbf 0$ bis zur ersten Ordnung um $\boldsymbol{\theta}^\dagger$. Wenden Sie den zentralen Grenzwertsatz auf $\tfrac{1}{\sqrt n}\sum_i s(x_i;\boldsymbol{\theta}^\dagger)$ an und begründen Sie damit

       $$\sqrt n\,\big(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta}^\dagger\big)\xrightarrow{d}\mathcal N\big(\mathbf 0,\ \mathbf A^{-1}\mathbf B\mathbf A^{-1}\big).$$

       Warum gilt $\mathbb{E}\big[s(x;\boldsymbol{\theta}^\dagger)\big]=\mathbf 0$, obwohl das Modell falsch ist? Zeigen Sie, dass sich bei korrektem Modell die Kovarianz aus :eqref:`eq_mdl-asymptotic-normality` ergibt.
    1. Passen Sie ein Poisson-Modell mit Rate $\lambda$ an Zähldaten aus einer negativen Binomialverteilung mit Mittelwert $m$ und Formparameter $\alpha$ an. Ihre Varianz beträgt $m+m^2/\alpha$. Bestimmen Sie $\hat\lambda$, $\lambda^\dagger$, $\mathbf A$ und $\mathbf B$. Um welchen Faktor unterschätzt die inverse Fisher-Information die Varianz von $\hat\lambda$?
    1. Passen Sie die Poisson-Regression $\log\lambda(x)=w_0+w_1x$ mit einem Merkmal über das Newton-Verfahren an. Für dieses Modell lautet der Schritt $\mathbf w\leftarrow\mathbf w+(\mathbf Z^\top\operatorname{diag}(\lambda_i)\mathbf Z)^{-1}\mathbf Z^\top(\mathbf y-\boldsymbol\lambda)$, mit den Zeilen $(1, x_i)$ in $\mathbf Z$ und $\lambda_i=\lambda(x_i)$. Eine Beobachtung ist nun ein Paar $(x_i,y_i)$, und $\log p(y_i\mid x_i,\mathbf w)$ ersetzt $\log p(x\mid\boldsymbol{\theta})$ in $\mathbf A$ und $\mathbf B$. Schätzen Sie beide Matrizen durch Stichprobenmittelwerte beim angepassten Wert $\hat{\mathbf w}$. Erzeugen Sie $2{,}000$ Datensätze mit je $n=500$ Paaren mit $x\sim\mathcal N(0,1)$, $\mathbf w=(1,0.5)$ und negativ binomialverteilten Zählwerten mit $\alpha=2$. Ziehen Sie dazu eine Rate aus der Gamma-Verteilung mit Formparameter $\alpha$ und Mittelwert $\lambda(x)$. Ziehen Sie anschließend einen Poisson-verteilten Zählwert mit dieser Rate. Vergleichen Sie drei Standardfehler für $\hat w_1$. Bestimmen Sie einen aus der inversen Fisher-Information, einen aus der Sandwich-Kovarianz und einen aus der Streuung von $\hat w_1$ über die Datensätze hinweg. Welcher Anteil der nominellen $95\%$-Intervalle der ersten beiden Verfahren enthält jeweils $w_1=0.5$? Wiederholen Sie die Rechnung mit $\alpha=10^6$.
    1. Übung :exref:`ex_mdl-distributions-poisson-regression`c in :numref:`sec_mdl-exponential-family` zeigt, dass die Poisson-Anpassung auch bei überdispersen Zähldaten $\mathbf w$ wiedergewinnt. Welcher Teil des Modells müsste falsch sein, damit $\mathbf w^\dagger$ von $\mathbf w$ abweicht? Welchen Parameter deckt das Sandwich-Intervall aus (c) dann ab?
   :exlabel:`ex_mdl-maximum-likelihood-sandwich`

<!--en:229f8652-->
[Diskussion](https://d2l.discourse.group/t/416)

<!--en:2b46c1c0-->
[Diskussion](https://d2l.discourse.group/t/1096)

<!--en:8f59a3a0-->
[Diskussion](https://d2l.discourse.group/t/1097)

<!--en:8f59a3a0-2-->
[Diskussion](https://d2l.discourse.group/t/1097)

<!--en:3d2f153d-->
Maximum-Likelihood, Fisher-Information und MAP-Schätzung<br>**Likelihood und NLL · KL-Untergrenze · Rauschmodelle · Score und Fisher-Information · Cramér-Rao-Schranke · MAP und A-priori-Verteilungen**

<!--en:819eec5a-->
::: {.slide title="Neunmal Kopf bei dreizehn Würfen: Warum 9/13?"}

<!--en:25d9aa8a-->
[Motivation und Leitfrage]{.kicker}

<!--en:dd32f200-->
- **Welches Prinzip liefert $\hat{\theta} = 9/13 \approx 0.692$?** Warum ist die empirische relative Häufigkeit der richtige Schätzer? Wie definiert dasselbe Prinzip Verluste für Netze mit $10^9$ Parametern?

<!--en:d46913ac-->
- **Wie genau legen $13$ Würfe den Wert von $\theta$ fest?** Aus der Krümmung ergibt sich der Standardfehler $1/\sqrt{nI(\hat{\theta})} \approx 0.128$. Das $95\%$-Intervall $[0.44, 0.94]$ schließt eine faire Münze ($\theta = 0.5$) noch ein.

<!--en:a4f1dd57-->
- **Was, wenn $3$ Würfe $\mathrm{HHH}$ ergeben?** A-priori-Verteilungen (MAP) regularisieren die Schätzung bei kleinen Stichproben. Damit bleibt Zahl möglich, anders als beim MLE ($\hat{\theta} = 1$).

<!--en:76db6728-->
::: {.slide title="Das Maximum-Likelihood-Prinzip"}

<!--en:08cb5ebf-->
[Die Likelihood-Funktion]{.kicker}

<!--en:6a2951d3-->
- **Wahrscheinlichkeit und Likelihood:** Bei festem $\boldsymbol{\theta}$ und variablem $X$ ergibt sich eine normierte Verteilung über Datensätze. Bei fester Beobachtung $X$ und variablem $\boldsymbol{\theta}$ ergibt sich die **Likelihood** (ihr Integral ist im Allgemeinen ungleich $1$, wenn über $\boldsymbol{\theta}$ integriert wird).

<!--en:514f665e-->
- **Frequentistische Sicht:** $\boldsymbol{\theta}$ ist eine feste, unbekannte Konstante. Der MLE wählt den Parametervektor, unter dem der beobachtete Datensatz $X$ am wahrscheinlichsten war.

<!--en:62572eb6-->
::: {.slide title="Rechenbeispiel: Likelihood einer Münze"}

<!--en:b7088e6c-->
[Maximum-Likelihood im Bernoulli-Modell]{.kicker}

<!--en:8ba05e61-->
- **Bei unabhängigen Würfen multiplizieren sich die Wahrscheinlichkeiten:** Für $n_H$-mal Kopf und $n_T$-mal Zahl ist die Likelihood $P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}$.

<!--en:26625088-->
- **Der Logarithmus erhält die Lage des Maximums:** Ableiten von $\log P(X \mid \theta) = n_H \log \theta + n_T \log(1-\theta)$ ergibt
  $$\frac{n_H}{\theta} - \frac{n_T}{1-\theta} = 0 \implies \hat{\theta} = \frac{n_H}{n_H + n_T}$$

<!--en:2ea82a51-->
- **Größere Stichprobe, stärkere Krümmung:** Beim Übergang von $9/13$ zu $90/130$ bleibt das Maximum bei $\hat{\theta} = 9/13 \approx 0.69$. Die Likelihood-Kurve wird um den Faktor $\sqrt{10} \approx 3.2\times$ schmaler.

<!--en:c0ee66eb-->
::: {.slide title="Gleitkommaunterlauf und NLL"}

<!--en:78345ec6-->
[Numerische Stabilität]{.kicker}

<!--en:73d58e61-->
- **Unterlauf bei Produkten:** Das Produkt aus $10^9$ Faktoren nahe $1/2$ ergibt $(1/2)^{10^9} = \texttt{0.0}$ in `float32` und `float64`. Damit geht die gesamte Information über den Gradienten verloren.

<!--en:e3b71d02-->
- **Additive NLL und SGD:** Mit $-\log$ wird aus Produkten eine gut konditionierte Summe. Ihr Gradient lässt sich durch Mittelwerte über Minibatches erwartungstreu schätzen.

<!--en:1593669b-->
::: {.slide title="Minimierung der NLL mit Gradientenabstieg"}

<!--en:7233da4b-->
[Numerische Optimierung]{.kicker}

<!--en:adf19569-->
Auch bei bekannter geschlossener Lösung lässt sich die NLL der Münze $\ell(\theta) = -\bigl(n_H \log \theta + n_T \log(1-\theta)\bigr)$ für $n_H = 8{,}675{,}309$-mal Kopf und $n_T = 256{,}245$-mal Zahl numerisch minimieren:

<!--en:d9b90987-->
**Fallstrick bei der Implementierung (summierter oder gemittelter Verlust):** Da der Verlust $\ell(\theta)$ über $n \approx 8.9 \times 10^6$ Würfe *summiert* wird, enthält der Gradient den Faktor $n$. Das erzwingt die winzige Lernrate `lr = 1e-9`. Nach Division durch $n$ bleibt bei der Minimierung der *mittleren* NLL $\frac{1}{n}\ell(\boldsymbol{\theta})$ die stabile Schrittweite unabhängig von der Größe des Batchs oder Datensatzes.

<!--en:27157b8f-->
::: {.slide title="Kreuzentropie und empirische Entropie als Untergrenze"}

<!--en:aeb9f79b-->
[Informationstheoretische Sicht]{.kicker}

<!--en:062b578f-->
- **Warum Trainingskurven oberhalb von null abflachen:** Die Daten legen die empirische Entropie $H(\hat{p}_{\mathrm{data}})$ fest ($0.617$ Nat bei einem Kopfanteil von $9/13$).

<!--en:410a2cc3-->
- **MLE als Projektion mit der Vorwärts-KL-Divergenz:** Wer die mittlere NLL minimiert, bringt $D_{\mathrm{KL}}(\hat{p}_{\mathrm{data}} \,\|\, p_{\boldsymbol{\theta}}) \ge 0$ so nahe an null heran, wie die Modellfamilie es erlaubt.

<!--en:07c976a5-->
::: {.slide title="MLE durch Minimierung der Vorwärts-KL-Divergenz"}

<!--en:3f7996e5-->
[Geometrie der Kreuzentropie]{.kicker}

<!--en:a63971ae-->
- **Unvermeidbare Untergrenze:** Bei $9$-mal Kopf und $4$-mal Zahl beträgt die empirische Entropie $H(9/13) = 0.617$ Nat. Nur bei $\hat{\theta} = 9/13$ erreicht die NLL diese Untergrenze.

<!--en:6b2fa11d-->
- **KL-Divergenz bei falsch gewählten Parametern:** Bei einer fairen Münze ($\theta = 0.5$) beträgt die NLL $\log 2 = 0.693$ Nat (KL-Divergenz $0.076$ Nat). Bei $\theta = 0.9$ beträgt die KL-Divergenz $0.164$ Nat.

<!--en:61b705b3-->
- **Untergrenzen bei der Klassifikation:** Kommt jede Trainingseingabe $\mathbf{x}_i$ nur einmal vor, ist ihre empirische Labelverteilung One-Hot-kodiert ($H = 0$). Überparametrisierte Netze können die Trainings-NLL auf $0$ senken. Für die erwartete NLL auf unabhängigen Testdaten bleibt $H(Y \mid X)$ eine Untergrenze.

<!--en:3992e64f-->
::: {.slide title="Stetige Daten und differentielle Entropie"}

<!--en:c401684a-->
[Likelihood bei stetigen Daten]{.kicker}

<!--en:6d4e4879-->
Für stetige Beobachtungen $x_1, \dots, x_n$ aus einer Dichte $p(x \mid \boldsymbol{\theta})$ ist die Wahrscheinlichkeit, eine bestimmte reelle Zahl exakt zu treffen, null. Es genügt daher Übereinstimmung innerhalb einer kleinen Toleranz $\epsilon > 0$:

<!--en:7808da11-->
- **Die Messtoleranz fällt weg:** Die additive Konstante $-n\log\epsilon$ hängt nicht von $\boldsymbol{\theta}$ ab. Die Minimierung der negativen Log-Dichte $-\sum_i \log p(x_i \mid \boldsymbol{\theta})$ ist daher unabhängig von $\epsilon$.

<!--en:7096be43-->
- **Dichten können größer als $1$ sein (und die NLL negativ):** Nur $\epsilon\,p(x \mid \boldsymbol{\theta})$ ist eine dimensionslose Wahrscheinlichkeit. Für eine schmale Dichte mit kleiner Varianz $\sigma^2 \ll 1$ können daher nahe ihrem Maximum $p(x \mid \boldsymbol{\theta}) > 1$ und $\log p > 0$ gelten.

<!--en:f6a9b791-->
- **KL-Zielfunktion in der Grundgesamtheit:** Unter der wahren Dichte $p_\star$ gilt in der Grundgesamtheit $\mathbb{E}_{x \sim p_\star}[-\log p(x \mid \boldsymbol{\theta})] = h(p_\star) + D_{\mathrm{KL}}(p_\star \,\|\, p_{\boldsymbol{\theta}})$. Dabei ist $h(p_\star) = -\int p_\star(x)\log p_\star(x)\,dx$ die differentielle Entropie.

<!--en:4f0f1566-->
::: {.slide title="Das Rauschmodell bestimmt den Trainingsverlust"}

<!--en:584eb32f-->
[Probabilistische Verlustfunktionen]{.kicker}

<!--en:7b9f58ee-->
Folgen die Zielwerte dem Modell $y_i = f_{\boldsymbol{\theta}}(\mathbf{x}_i) + \varepsilon_i$ mit gaußschem Rauschen $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, ergibt sich $-\log \prod_i \mathcal{N}(y_i; \hat{y}_i, \sigma^2) = \frac{1}{2\sigma^2}\sum_i (y_i - \hat{y}_i)^2 + \frac{n}{2}\log(2\pi\sigma^2)$. Jedem der üblichen Trainingsverluste entspricht eine bedingte Rauschverteilung $p(y \mid \mathbf{x})$:

<!--en:c3c72eab-->
| Rauschmodell $p(y \mid \mathbf{x})$ | NLL pro Beispiel $-\log p(y \mid \mathbf{x})$ | Verlust im Deep Learning |

<!--en:2ba27c3f-->
| **Normalverteilung** (festes $\sigma^2$) | $\tfrac{1}{2\sigma^2}(y - \hat{y})^2 + \tfrac{1}{2}\log(2\pi\sigma^2)$ | Mittlerer quadratischer Fehler (MSE) |

<!--en:c77be15a-->
| **Bernoulli-Verteilung** ($y \in \{0, 1\}$) | $-y\log \hat{p} - (1-y)\log(1-\hat{p})$ | Binäre Kreuzentropie (BCE) |

<!--en:530135ec-->
| **Kategoriale Verteilung** (One-Hot-kodiertes $\mathbf{y}$) | $-\sum_{k=1}^K y_k \log \hat{p}_k$ | Softmax-Kreuzentropie |

<!--en:3c9cda8a-->
| **Laplace-Verteilung** (fester Skalenparameter $s$) | $\tfrac{1}{s}\lvert y - \hat{y}\rvert + \log(2s)$ | Mittlerer absoluter Fehler (MAE / $L_1$) |

<!--en:fdd22e54-->
Nicht jeder Verlust ist eine negative Log-Likelihood. Beim Hinge-Verlust der SVM $\max(0, 1 - y\hat{y})$ fehlt eine endliche Normierungskonstante $Z(\hat{y}) = \sum_{y \in \{\pm 1\}} e^{-\phi(y,\hat{y})}$, die unabhängig von der Modellausgabe $\hat{y}$ ist.

<!--en:01416f6e-->
::: {.slide title="Gaußsches Rauschen und Laplace-Rauschen bei Ausreißern"}

<!--en:8e715395-->
[Robustheit und Verhalten in den Verteilungsenden]{.kicker}

<!--en:a63edbfa-->
- **Sensormesswerte $\{1, 2, 3, 4, 100\}$:** Gesucht ist eine einzige Konstante $c$. Der fünfte Messwert ist durch einen Messfehler ein Ausreißer.

<!--en:fa00789e-->
- **NLL bei gaußschem Rauschen (quadratisch):** Die Minimierung von $\frac{1}{2}\sum_i (y_i - c)^2$ ergibt $\sum_i (c - y_i) = 0$. Weil der Gradient linear mit dem Residuum wächst, liegt der Schätzwert beim **Stichprobenmittelwert** $c = 22$.

<!--en:dbc12220-->
- **NLL bei Laplace-Rauschen (stückweise linear):** Bei der Minimierung von $\sum_i \lvert y_i - c\rvert$ trägt jeder Punkt unabhängig von seiner Entfernung eine Steigung vom Betrag eins bei ($\pm 1$). Der Schätzwert liegt daher beim **Stichprobenmedian** $c = 3$.

<!--en:95f926bf-->
::: {.slide title="Die Score-Funktion hat den Erwartungswert null"}

<!--en:1bcd4dc3-->
[Optimalität erster Ordnung]{.kicker}

<!--en:95b15fd5-->
- **Optimalität erster Ordnung:** Der MLE $\hat{\boldsymbol{\theta}}$ löst $\sum_{i=1}^n s(x_i; \hat{\boldsymbol{\theta}}) = \mathbf{0}$.

<!--en:764abaa9-->
- **Warum der mittlere Score bei $\boldsymbol{\theta}^\star$ verschwindet:** Mit $\nabla_{\boldsymbol{\theta}} p = p\,\nabla_{\boldsymbol{\theta}} \log p$ ergibt das Ableiten von $\int p(x \mid \boldsymbol{\theta})\,dx = 1$ unter dem Integralzeichen $\mathbb{E}_{\boldsymbol{\theta}}[s(x; \boldsymbol{\theta})] = \nabla_{\boldsymbol{\theta}} 1 = \mathbf{0}$.

<!--en:7cbc2e5d-->
::: {.slide title="Fisher-Information und Krümmungsidentität"}

<!--en:5dac73e9-->
[Geometrie zweiter Ordnung]{.kicker}

<!--en:e49f0bf3-->
- **Informationsidentität:** Erneutes Ableiten von $\int s(x;\boldsymbol{\theta})p(x\mid\boldsymbol{\theta})\,dx = \mathbf{0}$ mit der Produktregel ergibt $\mathbb{E}[\nabla^2 \log p] + \mathbb{E}[s\,s^\top] = \mathbf{0}$.

<!--en:35dbd72d-->
- **Dualität:** Die Fisher-Information ist zugleich die **Varianz des Gradienten pro Beispiel** und die **erwartete Krümmung (Hesse-Matrix) der NLL**.

<!--en:ec050c55-->
::: {.slide title="Fisher-Information für drei Modelle"}

<!--en:2988cf19-->
[Fisher-Informationsmatrizen an Beispielen]{.kicker}

<!--en:493ea609-->
- **Bernoulli-Münze:** Es gelten $s(x;\theta) = \frac{x}{\theta} - \frac{1-x}{1-\theta}$ und $-s'(x;\theta) = \frac{x}{\theta^2} + \frac{1-x}{(1-\theta)^2}$. Der Erwartungswert mit $\mathbb{E}[x] = \theta$ liefert
  $$I(\theta) = \frac{1}{\theta(1-\theta)}$$

<!--en:22f3b576-->
- **Normalverteilung $\mathcal{N}(\mu, \sigma^2)$:** Es gilt $I(\mu, \sigma) = \operatorname{diag}\bigl(\frac{1}{\sigma^2},\, \frac{2}{\sigma^2}\bigr)$. Die NLL ist in Richtung $\sigma$ doppelt so stark gekrümmt wie in Richtung $\mu$.

<!--en:5bef0488-->
- **Lageparameter der Cauchy-Verteilung:** Für $p(x\mid\theta) = \frac{1}{\pi(1+(x-\theta)^2)}$ liefert die Substitution $x-\theta = \tan\varphi$ die Ausdrücke $s = \sin 2\varphi$ sowie $I(\theta) = \mathbb{E}[\sin^2 2\varphi] = \frac{1}{2}$.

<!--en:e7a80930-->
::: {.slide title="Prüfung der Informationsidentität durch automatisches Differenzieren"}

<!--en:641f1d8b-->
[Automatisches Differenzieren]{.kicker}

<!--en:fa75d2ec-->
Ist eine geschlossene Integration mühsam, liefert automatisches Differenzieren den Score und seine Ableitung pro Beispiel. So lassen sich beide Formen der Fisher-Information direkt auswerten, hier anhand von $10{,}000$ Cauchy-Beobachtungen bei $\theta^\star = 0$:

<!--en:baef6490-->
Die Verteilungsenden der Cauchy-Verteilung fallen so langsam ab, dass weder Erwartungswert noch Varianz existieren. Ihr Score $s(x;\theta) = \frac{2(x-\theta)}{1+(x-\theta)^2}$ liegt jedoch im beschränkten Intervall $[-1, 1]$. Daher ist die Fisher-Information $I = 0.5$ endlich, und extreme Ausreißer erhalten automatisch weniger Gewicht.

<!--en:f1768641-->
::: {.slide title="Fehlerbalken aus der Krümmung der NLL"}

<!--en:1b605a05-->
[Unsicherheit in der Praxis]{.kicker}

<!--en:7d5641f9-->
- **Regel für die beobachtete Krümmung:** An der Stelle $\hat{\theta} = 9/13$ gilt $\ell''(\hat{\theta}) = 61.0 \implies \widehat{\mathrm{se}} = 1/\sqrt{61.0} = 0.128$.

<!--en:c9bd95c6-->
- **Eine faire Münze bleibt möglich:** Das $95\%$-Konfidenzintervall $[0.44, 0.94]$ enthält $\theta = 0.5$. Für eine Halbbreite von $\pm 0.05$ sind $n \approx 330$ Würfe nötig.

<!--en:0de79a1e-->
::: {.slide title="Konsistenz des Maximum-Likelihood-Schätzers"}

<!--en:82b1d939-->
[Konvergenz bei großen Stichproben]{.kicker}

<!--en:50bedfea-->
Vor der Analyse der Schwankungen um $\boldsymbol{\theta}^\star$ ist zu zeigen, dass $\hat{\boldsymbol{\theta}}_n \xrightarrow{P} \boldsymbol{\theta}^\star$ für $n \to \infty$ gilt. Nach dem Gesetz der großen Zahlen konvergiert die empirisch gemittelte NLL gegen die Kreuzentropie in der Grundgesamtheit:

<!--en:5c798d68-->
- **Vier Bedingungen für Konsistenz:** (1) unabhängige, identisch verteilte Daten aus $p_{\boldsymbol{\theta}^\star}$, (2) **Identifizierbarkeit** ($p_{\boldsymbol{\theta}} = p_{\boldsymbol{\theta}^\star} \implies \boldsymbol{\theta} = \boldsymbol{\theta}^\star$), (3) ein klar getrenntes Minimum in der Grundgesamtheit und (4) **gleichmäßige Konvergenz** von $\frac{1}{n}\ell(\boldsymbol{\theta})$ über $\boldsymbol{\theta}$.

<!--en:c2688de7-->
- **Warum punktweise Konvergenz nicht genügt:** Das Gesetz der großen Zahlen gilt bei jedem festen $\boldsymbol{\theta}$. Um zu garantieren, dass der *Minimierer* der zufälligen Kurve gegen den Minimierer der Grenzkurve konvergiert, ist gleichmäßige Konvergenz erforderlich.

<!--en:5d79dd2f-->
- **Fehlende Identifizierbarkeit in tiefen Netzen:** Vertauschen versteckter Einheiten oder Umskalieren von ReLU-Schichten ändert $\boldsymbol{\theta}$, während die Vorhersageverteilung $p_{\boldsymbol{\theta}}(y \mid \mathbf{x})$ gleich bleibt.

<!--en:05a3ec98-->
::: {.slide title="Asymptotische Normalität des MLE"}

<!--en:817b730c-->
[Herleitung in drei Schritten]{.kicker}

<!--en:70781d40-->
- **1. Score bei $\boldsymbol{\theta}^\star$ linearisieren:** $0 = S_n(\hat{\theta}) \approx S_n(\theta^\star) + S_n'(\theta^\star)(\hat{\theta} - \theta^\star)$.

<!--en:b26529b7-->
- **2. Zentraler Grenzwertsatz im Zähler, Gesetz der großen Zahlen im Nenner:** Ersterer liefert $S_n(\theta^\star)/\sqrt{n} \xrightarrow{d} \mathcal{N}(0, I(\theta^\star))$, Letzteres $-S_n'(\theta^\star)/n \xrightarrow{P} I(\theta^\star)$. Daraus folgt die Varianz $I / I^2 = 1/I$.

<!--en:e22043fe-->
::: {.slide title="Fisher-Scoring beim Cauchy-Lagemodell"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:031c5195-->
- **Warum das gewöhnliche Newton-Verfahren scheitert:** Abseits des Minimums wird die beobachtete zweite Ableitung $\frac{2(1-u^2)}{(1+u^2)^2}$ der Cauchy-NLL für $\lvert u\rvert > 1$ negativ. Newton-Schritte führen dann bergauf.

<!--en:ad5f0517-->
- **Fisher-Scoring:** Der Ersatz der beobachteten durch die erwartete Krümmung $nI = n/2 > 0$ liefert die stabile Aktualisierung $\theta \leftarrow \theta + \frac{2}{n}S_n(\theta)$.

<!--en:42dc726c-->
- **Gute Übereinstimmung bei $n = 100$:** Über $4{,}000$ Datensätze folgt das Histogramm von $\sqrt{n}(\hat{\theta} - \theta^\star)$ der Kurve $\mathcal{N}(0, 1/I) = \mathcal{N}(0, 2)$ (über mehrere Läufe $\mathrm{sd} \approx 1.40$ bis $1.47$, Grenzwert $\sqrt{2} \approx 1.41$).

<!--en:a07ce1bd-->
::: {.slide title="Cramér-Rao-Schranke und Effizienz"}

<!--en:f9549d60-->
[Optimalität des MLE]{.kicker}

<!--en:f1aa0b99-->
Kann ein erwartungstreuer Schätzer ($\mathbb{E}_\theta[\tilde{\theta}] = \theta$ für alle $\theta$) eine kleinere Varianz als $\frac{1}{nI(\theta)}$ erreichen? Ableiten von $\int \tilde{\theta}(X)\,P(X \mid \theta)\,dX = \theta$ unter dem Integralzeichen mit $\partial_\theta P = P\,S_n$ ergibt:

<!--en:62f80413-->
Quadrieren beider Seiten liefert die **Cramér-Rao-Schranke** $\operatorname{Var}(\tilde{\theta}) \ge \frac{1}{nI(\theta)}$. Damit ist der MLE **asymptotisch effizient**. Der Vergleich dreier Schätzer für den Cauchy-Lageparameter anhand der $4{,}000$ Datensätze bestätigt den Unterschied:

<!--en:6cd224c6-->
::: {.slide title="Grenzen der klassischen Theorie des MLE"}

<!--en:9424be2e-->
[Statistische Fallstricke und Fehlspezifikation]{.kicker}

<!--en:f9943472-->
Jede Regularitätsbedingung für die asymptotische Normalität und die Cramér-Rao-Schranke kann in der Praxis verletzt sein:

<!--en:4efaa4d9-->
- **Verzerrung bei endlichen Stichproben und Randwerte:** Der MLE für die Varianz einer Normalverteilung teilt durch $n$ ($\mathbb{E}[\hat{\sigma}^2] = \frac{n-1}{n}\sigma^2$). Bei linear separierbaren Daten für die logistische Regression oder bei der Münzwurffolge $\mathrm{HHH}$ liegt das Infimum der NLL bei $\pm\infty$ bzw. am Rand.

<!--en:adc36530-->
- **Parameterabhängiger Träger:** Für $\mathrm{Uniform}(0, \theta)$ konvergiert $\hat{\theta} = \max_i x_i$ mit Rate $1/n$ statt $1/\sqrt{n}$. Der Schätzer $\frac{n+1}{n}\hat{\theta}$ erreicht die Varianz $O(1/n^2)$ und unterschreitet die formale Cramér-Rao-Schranke. Am beweglichen Rand $x = \theta$ scheitert das Ableiten unter dem Integralzeichen.

<!--en:8a4aa2f4-->
- **Singuläre Fisher-Matrix in ReLU-Netzen:** Die Umskalierung $(a_j, w_j) \mapsto (c a_j, w_j/c)$ lässt $a_j \mathrm{ReLU}(w_j x)$ unverändert. Dadurch hat $I(\boldsymbol{\theta})$ Eigenwerte null, sodass $I(\boldsymbol{\theta})^{-1}$ nicht existiert.

<!--en:eb28f169-->
- **Fehlspezifikation und Sandwich-Kovarianz:** Gilt für die wahre Verteilung $p_\star \notin \{p_{\boldsymbol{\theta}}\}$, konvergiert der MLE gegen die KL-Projektion $\boldsymbol{\theta}^\dagger$. Die Score-Varianz $\mathbf{B} = \mathbb{E}[s\,s^\top]$ stimmt im Allgemeinen nicht mehr mit der erwarteten Krümmung $\mathbf{A} = -\mathbb{E}[\nabla^2 \log p]$ überein. Es ergibt sich die **Sandwich-Kovarianz nach Huber und White** $\sqrt{n}(\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}^\dagger) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathbf{A}^{-1}\mathbf{B}\mathbf{A}^{-1})$.

<!--en:411ad2b6-->
::: {.slide title="Kleine Stichproben und MAP-Schätzung"}

<!--en:ce677c5b-->
[A-priori-Verteilungen zur Regularisierung]{.kicker}

<!--en:25920106-->
- **MLE bei fehlendem Ergebnis:** Bei $3$-mal Kopf ($\mathrm{HHH}$) gilt $\hat{\theta}_{\mathrm{MLE}} = 3/3 = 1$. Der erste spätere Wurf mit Zahl erhält damit eine unendliche NLL ($-\log 0 = \infty$).

<!--en:772c6878-->
- **MAP-Regularisierung:** Eine $\mathrm{Beta}(2,2)$-A-priori-Verteilung ergänzt $1$ fiktiven Wurf mit Kopf und $1$ mit Zahl. Damit verschiebt sich $\hat{\theta}_{\mathrm{MAP}}$ auf $4/5 = 0.80$ ($1.61$ Nat bei einer späteren Zahl).

<!--en:d94b19cb-->
::: {.slide title="Gauß- und Laplace-Verteilungen als Ursprung von L2- und L1-Straftermen"}

<!--en:0be53040-->
[Gewichtszerfall und dünne Besetzung]{.kicker}

<!--en:083e7f88-->
- **Gauß-Prior $\to$ $L_2$-Gewichtszerfall:** Eine isotrope A-priori-Verteilung $\theta_j \sim \mathcal{N}(0, \tau^2)$ ergänzt den Term $-\log p(\boldsymbol{\theta}) = \frac{1}{2\tau^2}\|\boldsymbol{\theta}\|_2^2 + \mathrm{const}$. Mit gaußschem Rauschen der Zielwerte $\mathcal{N}(\hat{y}_i, \sigma^2)$ und Skalierung um $\sigma^2 / n$ ergibt sich
  $$\frac{1}{n}\sum_{i=1}^n \frac{1}{2}(y_i - \hat{y}_i)^2 + \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2, \quad \lambda = \frac{\sigma^2}{n\tau^2}$$

<!--en:3fd7ff2f-->
- **Laplace-A-priori-Verteilung $\to$ $L_1$ und dünne Besetzung:** $p(\boldsymbol{\theta}) \propto e^{-\|\boldsymbol{\theta}\|_1 / s}$ ergänzt den Term $\frac{1}{s}\|\boldsymbol{\theta}\|_1$. Sein Knick bei $0$ setzt kleine Gewichte exakt auf null.

<!--en:ed0b06e8-->
- **Asymptotisch verschwindender Einfluss:** Für $n \to \infty$ oder $\tau \to \infty$ gilt $\lambda \to 0$, und die MAP-Schätzung nähert sich wieder dem MLE.

<!--en:d960c3de-->
::: {.slide title="Beta-A-priori-Verteilungen, Pseudohäufigkeiten und Reparametrisierung"}

<!--en:3f4ee11a-->
[Konjugierte A-priori-Verteilungen und Äquivarianz]{.kicker}

<!--en:d4ff8d2c-->
Das Produkt der Likelihood der Münze $\theta^{n_H}(1-\theta)^{n_T}$ und der konjugierten $\mathrm{Beta}(a,b)$-A-priori-Dichte $\propto \theta^{a-1}(1-\theta)^{b-1}$ ergibt die $\mathrm{Beta}(n_H + a,\, n_T + b)$-A-posteriori-Verteilung $\propto \theta^{n_H+a-1}(1-\theta)^{n_T+b-1}$:

<!--en:4ea2cc29-->
- **Add-eins-Glättung als A-posteriori-Mittelwert:** Bei der flachen A-priori-Verteilung $\mathrm{Beta}(1,1)$ entspricht der A-posteriori-Mittelwert der Nachfolgeregel von Laplace $\frac{n_H+1}{n+2}$. Die MAP-Schätzung entspricht hier dem gewöhnlichen MLE $\frac{n_H}{n}$. Erst mit $\mathrm{Beta}(2,2)$ ergibt sich als MAP-Schätzung $\frac{n_H+1}{n+2}$.

<!--en:51a8ff31-->
- **Die MAP-Schätzung hängt von der Parametrisierung ab:** Bei einer Variablentransformation $\eta = g(\theta)$ erhält die A-posteriori-Dichte den Jacobi-Faktor $\lvert d\theta / d\eta\rvert$. Aus einer flachen $\mathrm{Beta}(1,1)$-A-priori-Verteilung in $\theta$ wird die logistische Dichte $e^{-\eta}/(1+e^{-\eta})^2$ für die Log-Odds $\eta = \log\frac{\theta}{1-\theta}$. Damit verschiebt sich die MAP-Schätzung für $9/13$ auf $10/15 = 2/3$.

<!--en:422904c8-->
- **Der MLE ist äquivariant:** Da die Likelihood keinen Jacobi-Faktor enthält, ist der MLE von $g(\theta)$ stets $g(\hat{\theta}_{\mathrm{MLE}})$.

<!--en:219d94cd-->
::: {.slide title="Verständnisfragen zu Maximum-Likelihood, Fisher-Information und MAP"}

<!--en:929b7671-->
[Verständnisfragen]{.kicker}

<!--en:a02997fc-->
1. **Stichprobenumfang aus der Fisher-Information:** Eine Münze zeigt $9$-mal Kopf bei $13$ Würfen ($\hat{\theta} \approx 0.69$). Wie viele Würfe braucht man, um das $95\%$-Konfidenzintervall $\hat{\theta} \pm 1.96/\sqrt{nI(\hat{\theta})}$ auf eine Halbbreite von $\pm 0.05$ zu verkleinern?

<!--en:4bfd5378-->
2. **Drei Schätzer nach $\mathrm{HHH}$:** Wie lauten bei $3$-mal Kopf in $3$ Würfen der MLE, die MAP-Schätzung unter $\mathrm{Beta}(2,2)$ und der A-posteriori-Mittelwert unter $\mathrm{Beta}(1,1)$?

<!--en:8cc4f300-->
3. **Fisher-Information bei Fehlspezifikation:** Ein Poisson-Modell ($\operatorname{Var}(X) = \lambda$) wird an überdisperse Zähldaten mit wahrem Mittelwert $m = 4$ und Varianz $m + m^2/\alpha = 12$ ($\alpha = 2$) angepasst. Um welchen Faktor unterschätzt $(nI)^{-1}$ die Größe $\operatorname{Var}(\hat{\lambda}_{\mathrm{MLE}})$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:a0a2d9b2-->
1. Aus $1.96\sqrt{0.69(0.31)/n} = 0.05$ folgen $n = \frac{1.96^2 \cdot 0.2139}{0.05^2} \approx 329$ Würfe. Nötig sind also etwa $25\times$ so viele Daten, um den Fehlerbalken um den Faktor $5\times$ zu verkleinern.

<!--en:e6e2f15a-->
2. $\hat{\theta}_{\mathrm{MLE}} = 1$, $\hat{\theta}_{\mathrm{MAP}} = \frac{3+2-1}{3+2+2-2} = \frac{4}{5}$ und $\mathbb{E}[\theta \mid X] = \frac{3+1}{3+1+1} = \frac{4}{5}$ (Nachfolgeregel von Laplace).

<!--en:d96ad901-->
3. Der Poisson-MLE ist $\bar{X}$ mit tatsächlicher Varianz $12/n$. Die Fisher-Information $I = 1/\lambda$ des Poisson-Modells sagt jedoch $4/n$ voraus. Sie unterschätzt die tatsächliche Varianz um den Faktor $1 + m/\alpha = 3$ (die Sandwich-Varianz $B/A^2$ korrigiert das).

<!--en:1d771dd2-->
::: {.slide title="Zusammenfassung: Maximum-Likelihood, Fisher-Information und MAP"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:f932efc8-->
| Konzept | Mathematische Form | Bedeutung für Statistik und Deep Learning |

<!--en:a9205731-->
| **Negative Log-Likelihood** | $\ell(\boldsymbol{\theta}) = -\sum_{i=1}^n \log p(x_i \mid \boldsymbol{\theta})$ | Verhindert Unterlauf und macht aus Produkten additive Minibatch-Verluste |

<!--en:9993fdb3-->
| **Zerlegung in Entropie und KL-Divergenz** | $\frac{1}{n}\ell(\boldsymbol{\theta}) = H(\hat{p}_{\mathrm{data}}) + D_{\mathrm{KL}}(\hat{p}_{\mathrm{data}} \,\|\, p_{\boldsymbol{\theta}})$ | Der MLE minimiert die Vorwärts-KL-Divergenz oberhalb der durch die Datenentropie gegebenen Untergrenze |

<!--en:94f674c7-->
| **Rauschen $\to$ Trainingsverlust** | $\mathcal{N}(\hat{y}, \sigma^2) \to \mathrm{MSE}$ und $\mathrm{Laplace}(\hat{y}, s) \to \mathrm{MAE}$ | Die Verlustfunktion legt eine Annahme über die Verteilungsenden der Residuen fest |

<!--en:700ebf4b-->
| **Score und Fisher-Information** | $I(\boldsymbol{\theta}) = \operatorname{Var}[s] = -\mathbb{E}[\nabla^2 \log p(x \mid \boldsymbol{\theta})]$ | Verknüpfen Gradientenschwankungen mit der erwarteten Krümmung der Verlustfunktion |

<!--en:dc4aaa96-->
| **Asymptotische Normalität und Cramér-Rao-Schranke** | $\sqrt{n}(\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}^\star) \xrightarrow{d} \mathcal{N}\bigl(\mathbf{0}, I(\boldsymbol{\theta}^\star)^{-1}\bigr)$ | Der MLE erreicht die Cramér-Rao-Untergrenze für die Varianz bei $n \to \infty$ |

<!--en:21953d12-->
| **MAP-Schätzung** | $\operatorname{argmin}_{\boldsymbol{\theta}} [\ell(\boldsymbol{\theta}) - \log p(\boldsymbol{\theta})]$ | Gauß-Prior ergibt $L_2$-Gewichtszerfall ($\lambda = \frac{\sigma^2}{n\tau^2}$), Beta-Verteilung ergänzt Pseudohäufigkeiten |
