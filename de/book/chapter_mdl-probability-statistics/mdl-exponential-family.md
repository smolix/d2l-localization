<!--en:8bc79012-->
# Die Exponentialfamilie

<!--en:2b826c43-->
Lineare Regression, logistische Regression und Softmax-Regression minimieren jeweils eine andere Verlustfunktion (engl. loss function). Leiten wir diese Funktionen nach den Modellgewichten ab, erhalten wir dennoch in allen drei Fällen dieselbe Trainingsregel. Wir multiplizieren die Eingabe mit der Differenz aus Vorhersage und Beobachtung. Wenn drei scheinbar unabhängige Herleitungen zur selben Formel führen, liegt ihnen meist dieselbe Struktur zugrunde. Auch die Übersicht in :numref:`sec_mdl-distributions` legt diese Vermutung nahe. Dort schien jede Verteilung einen eigenen algebraischen Kniff zu verlangen, um Mittelwert und Varianz zu bestimmen.

<!--en:87ff74cf-->
Betrachten wir die Münze, die uns durch dieses Kapitel begleitet. Bei 13 Würfen fiel 9-mal Kopf. Die naheliegende Schätzung für die Wahrscheinlichkeit von Kopf ist $9/13\approx0.69$. Passen wir eine logistische Regression mit einem Bias als einzigem Parameter an dieselben 13 Würfe an, gelangt sie auf einem Umweg zu $9/13$. Ihre letzte Schicht berechnet die Log-Odds $\log(9/4)\approx0.81$, und eine Sigmoidfunktion bildet diesen Wert wieder auf $9/13$ ab. Warum nehmen wir den Umweg über die Log-Odds, wenn wir direkt mit der Wahrscheinlichkeit arbeiten könnten?

<!--en:eabb1d2a-->
Eine Idee erklärt beide Beobachtungen. Fast jede Verteilung in unserer Übersicht lässt sich in derselben kanonischen Form schreiben. Die Parameter gehen nur über die Exponentialfunktion einer Linearkombination weniger Statistiken der Daten ein. Verteilungen mit dieser Struktur bilden die *Exponentialfamilie*. Bei einer Münze sind die Log-Odds genau der Parameter im Exponenten. Allein aus dieser Form folgen vier bemerkenswerte Eigenschaften. Mittelwert und Varianz sind die ersten beiden Ableitungen derselben Funktion. Die Maximum-Likelihood-Schätzung wird zu einem konvexen Optimierungsproblem, dessen Gradient stets die Differenz aus Vorhersage und Beobachtung ist. Modellieren wir Klassenlabel, ergibt sich die Softmax-Funktion von selbst. Eine A-priori-Verteilung (engl. prior) derselben Form aktualisieren wir durch Addition von Anzahlen.

<!--en:07fb0f7f-->
Für Deep Learning ist die zweite Eigenschaft das zentrale Ergebnis. Die erste liefert den Beweis dafür (:numref:`subsec_mdl-moment-matching` und :numref:`subsec_mdl-glm`). Zusammen zeigen sie uns, welche Verlustfunktion zu einer neuen Art von Label passt, etwa zu einer Anzahl. Auch erkennen wir, wann die Maximum-Likelihood-Schätzung überhaupt keine endliche Lösung hat, etwa wenn bei jedem Münzwurf Kopf fällt. Bei der ersten Lektüre können Sie zwei Teile dieses Abschnitts überspringen. Die Herleitung über die maximale Entropie (:numref:`subsec_mdl-max-entropy-derivation`) erklärt den Ursprung der Exponentialform. Die alternativen Herleitungen der Softmax-Funktion gehen von Entscheidungen unter Rauschen und von der statistischen Physik aus.

<!--en:c1b046d8-->
## Eine Form für viele Verteilungen

<!--en:2db131ca-->
Beginnen wir mit der einfachsten Verteilung unserer Übersicht, der eines einzelnen Münzwurfs. Wir formen sie so lange um, bis die Log-Odds von selbst hervortreten. Diese kurze Rechnung zeigt uns, wie die allgemeine Form aussehen muss.

<!--en:3b52c6d0-->
### Die Bernoulli-Verteilung in Exponentialform

<!--en:b576acb9-->
Bei einer Münze mit Kopfwahrscheinlichkeit $p$ fällt Kopf ($x=1$) mit Wahrscheinlichkeit $p$ und Zahl ($x=0$) mit Wahrscheinlichkeit $1-p$. Im vorigen Abschnitt haben wir beide Fälle in einem Ausdruck zusammengefasst, $p^x(1-p)^{1-x}$. Wie hängen das Ergebnis $x$ und der Parameter $p$ zusammen? Wir logarithmieren den Ausdruck und fassen die Terme zusammen, die $x$ enthalten:

<!--en:b91082d3-->
Die beiden Rollen sind nun klar getrennt. Das Ergebnis geht nur als Produkt von $x$ und den Log-Odds $\log\frac{p}{1-p}$ ein. Der zweite Term hängt allein von $p$ ab. Bezeichnen wir die Log-Odds mit $\eta=\log\frac{p}{1-p}$ und drücken wir den zweiten Term durch $\eta$ aus. Auflösen nach $p$ ergibt $p=e^\eta/(1+e^\eta)$, also $1-p=1/(1+e^\eta)$ und $\log(1-p)=-\log(1+e^\eta)$. Wenden wir auf beide Seiten die Exponentialfunktion an, erhalten wir

<!--en:1ef41930-->
Die Wahrscheinlichkeit des Münzwurfergebnisses hat nun folgende Form. Im Exponenten steht eine lineare Funktion von $x$ abzüglich eines Terms, der von $x$ unabhängig ist. Dieser zweite Term dient der Normierung. Die beiden unnormierten Gewichte $e^{0}=1$ und $e^\eta$ ergeben in der Summe $1+e^\eta$.

<!--en:f9561e8e-->
Drei Bestandteile dieses Ausdrucks begegnen uns im ganzen Abschnitt wieder. Die *Statistik* $T(x)=x$ erfasst, worauf es beim Ergebnis ankommt. Mit dem *natürlichen Parameter* $\eta$ multiplizieren wir diese Statistik. Bei der Bernoulli-Verteilung sind das die Log-Odds, auch *Logit* genannt. Die *logarithmische Zustandssumme* (engl. log-partition function) $A(\eta)=\log(1+e^\eta)$ ist der Logarithmus der Normierungskonstante. Vielleicht kennen Sie diese Funktion als *Softplus*. Kehren wir unsere frühere Substitution um, erhalten wir

<!--en:0cc4b161-->
Die Sigmoidfunktion der logistischen Regression bildet also den natürlichen Parameter der Münze wieder auf ihren Mittelwert ab. Sie ergibt sich von selbst.

<!--en:4e89b38b-->
In unserem Münzbeispiel entspricht der empirische Kopfanteil $9/13$ dem natürlichen Parameter

<!--en:42a0731c-->
und die Sigmoidfunktion liefert wieder den Ausgangswert: $\operatorname{sigmoid}(\log\frac94)=\frac{9/4}{1+9/4}=\frac{9}{13}$. In Kürze sehen wir, warum $\hat\eta$ der Maximum-Likelihood-Wert der Log-Odds ist. Auch sehen wir, was die Krümmung des Verlusts in der Umgebung von $\hat\eta$ über die Unsicherheit dieser Schätzung aussagt.

<!--en:536cf90c-->
### Definition und Beispiele

<!--en:86e7a195-->
Unsere Umformung der Bernoulli-Verteilung hatte drei Bestandteile. Wir brauchten eine Statistik der Daten, einen Parameter, mit dem wir sie multiplizieren, und eine Normierungskonstante. Verallgemeinern wir dies auf mehrere Statistiken und lassen für jedes Ergebnis ein festes, parameterfreies Gewicht zu. So erhalten wir eine gemeinsame Form für die meisten Verteilungen unserer Übersicht.

<!--en:18de9a23-->
Eine Verteilungsfamilie heißt *Exponentialfamilie*, wenn sich ihre Wahrscheinlichkeitsfunktion oder Dichte schreiben lässt als

<!--en:df4eb014-->
In dieser Form sind Daten und Parameter allein über das innere Produkt $\boldsymbol\eta^\top T(\mathbf x)$ miteinander verknüpft. Das *Basismaß* (engl. base measure) $h(\mathbf x)\ge0$ weist jedem Ergebnis unabhängig von den Parametern ein festes Gewicht zu. Der Bereich, in dem $h>0$ gilt, bestimmt den Träger der Verteilung. Die *natürlichen Parameter* $\boldsymbol\eta\in\mathbb R^d$ lassen sich anpassen. Die *suffizienten Statistiken* (engl. sufficient statistics) $T(\mathbf x)\in\mathbb R^d$ sind die Merkmale der Daten, die wir mit diesen Parametern multiplizieren. Die *logarithmische Zustandssumme* ist der Logarithmus der Normierungskonstante,

<!--en:f677e5c1-->
mit einem Integral für stetige Dichten und einer Summe für diskrete $\mathbf x$. Anders gesagt ist $A$ genau die Konstante, für die das Integral oder die Summe über :eqref:`eq_mdl-exp_family` eins ergibt. Zulässig sind die natürlichen Parameter, für die $A(\boldsymbol\eta)$ endlich ist. Die Bernoulli-Verteilung ist der Spezialfall mit $h(x)=1$, $T(x)=x$ und $A(\eta)=\log(1+e^\eta)$.

<!--en:47709c82-->
Warum heißen die Merkmale $T(\mathbf x)$ *suffizient*? Weil sie alles enthalten, was uns die Daten über $\boldsymbol\eta$ verraten können. Für $n$ unabhängige Beobachtungen zerfällt die gemeinsame Likelihood in Faktoren:

<!--en:9bd2ca60-->
Die Parameter hängen mit den Daten nur über die Summe $\sum_i T(\mathbf x_i)$ zusammen. Stimmen zwei Datensätze in dieser Summe überein, unterscheiden sich ihre Likelihoods nur durch einen konstanten Faktor, der von $\boldsymbol\eta$ unabhängig ist. Sie liefern also dieselben Parameterschätzungen. Bei der Münze zählt die Summe, wie oft Kopf fällt. Für jede Kopfwahrscheinlichkeit sind alle Umordnungen der 13 Würfe gleich wahrscheinlich. Wissen wir, dass 9-mal Kopf fiel, verrät uns die Reihenfolge nichts weiter über die Münze.

<!--en:146d02a0-->
Durch Vergleich mit :eqref:`eq_mdl-exp_family` erkennen wir weitere Mitglieder der Familie. Bei einer Poisson-Verteilung mit Rate $\lambda$ ist die Wahrscheinlichkeit für $k$ Ereignisse gleich $\frac{\lambda^k e^{-\lambda}}{k!}=\frac{1}{k!}\exp(k\log\lambda-\lambda)$. Wir lesen also $h(k)=1/k!$, $T(k)=k$, $\eta=\log\lambda$ und $A(\eta)=e^\eta=\lambda$ ab. Eine Gauß-Verteilung mit unbekanntem Mittelwert und unbekannter Varianz erfordert zwei Statistiken. Wir multiplizieren das Quadrat im Exponenten aus und fassen die Terme mit $x$ und $x^2$ zusammen. Das ergibt

<!--en:ff94da24-->
also $T(x)=(x,x^2)^\top$ und $h(x)=1/\sqrt{2\pi}$. Der zweite natürliche Parameter muss strikt negativ sein. Für $\eta_2<0$ bewirkt der Faktor $e^{\eta_2 x^2}$, dass die Dichte der Gauß-Verteilung an den Rändern rasch abfällt. Für $\eta_2\ge0$ hingegen divergiert das Normierungsintegral für $A$. :numref:`tab_mdl-exp-family-members` fasst diese und weitere verbreitete Mitglieder zusammen.

<!--en:cfaffde1-->
:Mitglieder der Exponentialfamilie. In jeder Zeile hat die Wahrscheinlichkeit oder Dichte die Form $h(\mathbf x)\exp(\boldsymbol\eta^\top T(\mathbf x)-A(\boldsymbol\eta))$. Die Binomialverteilung setzt eine feste Anzahl $n$ an Versuchen voraus, die Laplace-Verteilung einen festen Lageparameter $\mu$.

<!--en:a55d4b51-->
| Verteilung | $h(\mathbf x)$ | $T(\mathbf x)$ | $\boldsymbol\eta$ | $A(\boldsymbol\eta)$ |

<!--en:8db143d3-->
| Bernoulli($p$) | $1$ | $x$ | $\log\frac{p}{1-p}$ | $\log(1+e^\eta)$ |

<!--en:dc56212e-->
| Binomial($n,p$) | $\binom nx$ | $x$ | $\log\frac{p}{1-p}$ | $n\log(1+e^\eta)$ |

<!--en:21b7fc9d-->
| Poisson($\lambda$) | $1/k!$ | $k$ | $\log\lambda$ | $e^\eta$ |

<!--en:8c9ad817-->
| Kategorialverteilung, One-Hot-Vektor $\mathbf y$ | $1$ | $\mathbf y$ | Logits $\mathbf z$ | $\log\sum_k e^{z_k}$ |

<!--en:bbbfc8e0-->
| Exponential($\lambda$) | $1$ für $x\ge0$ | $x$ | $-\lambda$ | $-\log(-\eta)$ |

<!--en:0ee9af8a-->
| Laplace($\mu,b$) | $1$ | $\lvert x-\mu\rvert$ | $-1/b$ | $\log(-2/\eta)$ |

<!--en:9ec2eb22-->
In der Zeile zur Kategorialverteilung drücken wir die Klassenwahrscheinlichkeiten durch die Logits aus. So entsteht die Softmax-Funktion, deren Ursprung wir in :numref:`subsec_mdl-softmax-origins` untersuchen. Zwei weitere Mitglieder finden wir in höheren Dimensionen. Die multivariate Gauß-Verteilung hat die suffizienten Statistiken $\mathbf x$ und $\mathbf x\mathbf x^\top$ und die natürlichen Parameter $\boldsymbol\Sigma^{-1}\boldsymbol\mu$ und $-\frac12\boldsymbol\Sigma^{-1}$. Hinzu kommt die Multinomialverteilung mit fester Anzahl von Ziehungen. Die Aufteilung auf $h$, $\boldsymbol\eta$ und $T$ ist nicht eindeutig. Eine Redundanz in der Zeile zur Kategorialverteilung spielt später eine Rolle. Die $K$ Einträge eines One-Hot-Vektors ergeben in der Summe immer eins. Daher ist eines der $K$ Logits redundant (:numref:`subsec_mdl-moment-matching`).

<!--en:28675ccd-->
### Verteilungen außerhalb der Familie

<!--en:a3f2c92c-->
Nicht jede Verteilung hat die Form :eqref:`eq_mdl-exp_family`. An zwei klassischen Ausnahmen sehen wir, wie unterschiedlich die Gründe dafür sein können.

<!--en:3051d62e-->
Betrachten wir zunächst die Gleichverteilung $U(0,\theta)$. Ihre Dichte ist $1/\theta$ auf $[0,\theta]$ und null außerhalb dieses Bereichs. Hier ändert sich der Träger selbst mit dem Parameter $\theta$. Jede Exponentialfamilie hat hingegen einen festen Träger, die Menge aller Punkte mit $h>0$, denn der Exponentialfaktor wird nie null. Ein veränderlicher Träger wirkt sich grundlegend auf die Schätzung aus. Angenommen, wir beobachten drei Ziehungen mit den Werten $0.1$, $0.2$ und $0.9$. Die Likelihood beträgt $\theta^{-3}$ für $\theta\ge0.9$ und $0$ für $\theta<0.9$. Ein kleineres $\theta$ hätte den Wert $0.9$ nie hervorbringen können. Der Maximum-Likelihood-Schätzer ist also das Stichprobenmaximum $\hat\theta=\max_i x_i=0.9$.

<!--en:01d0b203-->
Innerhalb einer Exponentialfamilie setzt die Maximum-Likelihood-Schätzung den Mittelwert der suffizienten Statistik unter dem Modell stets mit ihrem Stichprobenmittelwert gleich. Das beweisen wir in Kürze. Wenden wir dieses Verfahren hier an, setzen wir den Mittelwert $\theta/2$ gleich $\bar x=0.4$ und erhalten $\theta=0.8$. Bei diesem Wert wäre die Beobachtung $0.9$ unmöglich. Die Likelihood der Gleichverteilung hängt von den Daten über ihr Maximum ab und nicht über eine Summe. Daher greifen die Methoden dieses Abschnitts hier nicht. Aufgabe :exref:`ex_mdl-maximum-likelihood-moving-support` in :numref:`sec_mdl-maximum_likelihood` vergleicht das Verhalten der beiden Schätzer bei wachsendem Stichprobenumfang.

<!--en:5d7dce8a-->
Verteilungen mit schweren Rändern können aus einem zweiten Grund außerhalb der Familie liegen. Für die *Student-$t$-Verteilung* gehen wir von einer Gauß-Verteilung aus, deren Präzision $1/\sigma^2$ selbst zufällig ist. Mitteln wir über diese Präzision, fallen die Ränder nur noch polynomiell ab statt so rasch wie $e^{-x^2}$ (Herleitung der Dichte in Aufgabe :exref:`ex_mdl-distributions-student-t`). Bei Skalenparameter eins und $\nu$ Freiheitsgraden ist die Likelihood von $n$ Beobachtungen ein Produkt aus Faktoren der Form $\bigl(1+(x_i-\mu)^2/\nu\bigr)^{-(\nu+1)/2}$. Es lässt sich durch keine feste Anzahl von Summen über die Daten darstellen. Nach einem klassischen Satz von Pitman, Koopman und Darmois besitzen unter den regulären Familien mit festem Träger tatsächlich nur Exponentialfamilien eine suffiziente Statistik fester Dimension :cite:`Darmois.1935,Koopman.1936,Pitman.1936`.

<!--en:4faaf7be-->
Das hat eine praktische Folge. Wie wir noch zeigen, ist die negative Log-Likelihood für jede Exponentialfamilie konvex. Außerhalb dieser Familie kann die Konvexität verloren gehen. Bilden die Daten zwei deutlich getrennte Gruppen, so hat die negative Log-Likelihood einer Student-$t$-Verteilung als Funktion des Lageparameters $\mu$ in der Nähe jeder Gruppe ein eigenes lokales Minimum.

<!--en:d5404ca4-->
## Maximale Entropie

<!--en:af6bb889-->
Bisher haben wir die Exponentialform durch Umformen vertrauter Verteilungen gewonnen. Doch woher kommt diese Form überhaupt? Angenommen, wir kennen von einer Zufallsgröße nur die Mittelwerte einiger ihrer Statistiken. Welche Verteilung sollten wir wählen? Erstaunlicherweise gehört die Verteilung, die außer diesen Mittelwerten nichts voraussetzt, immer zur Exponentialfamilie.

<!--en:7663b62d-->
### Entropie und ein gezinkter Würfel

<!--en:a73fb297-->
Was bedeutet es, „außer den Mittelwerten nichts vorauszusetzen“? Dazu brauchen wir ein Maß dafür, wie breit eine Verteilung ihre Masse streut, wie wenig sie sich also festlegt. Für eine Verteilung $p=(p_1,\dots,p_K)$ auf einer endlichen Menge ist die *Entropie* definiert als

<!--en:1e129d82-->
gemessen in Nat, mit der Konvention $0\log0=0$. Jeder Summand gewichtet die Überraschung $-\log p_k$ über das Ergebnis $k$ mit dessen Wahrscheinlichkeit. Die Entropie ist also die mittlere Überraschung. Bei einem fairen sechsseitigen Würfel ist die Überraschung für jede Augenzahl gleich $\log6$. Das ergibt eine Entropie von $\log6\approx1.79$ Nat. Ein Würfel, der immer eine Sechs zeigt, birgt keinerlei Überraschung und hat die Entropie $0$. Unter allen Verteilungen auf $K$ Ergebnissen erreicht die Gleichverteilung die größtmögliche Entropie $\log K$. Das beweisen wir gleich.

<!--en:e3ee0141-->
E. T. Jaynes veranschaulichte das Problem mit einer konkreten Aufgabe :cite:`Jaynes.1957`. Angenommen, Sie erfahren über einen sechsseitigen Würfel nur, dass er im Mittel $4.5$ statt $3.5$ zeigt. Viele Verteilungen auf $\{1,\dots,6\}$ haben den Mittelwert $4.5$. Eine davon weist die Wahrscheinlichkeit $1/2$ der Augenzahl $3$ zu und $1/2$ der Augenzahl $6$. Damit behaupten wir allerdings, dass $1$, $2$, $4$ und $5$ nie auftreten können. Dafür gibt uns die Angabe zum Würfel keinerlei Anhaltspunkt. Die Entropie dieser Verteilung beträgt nur $\log2\approx0.69$.

<!--en:5f173b7d-->
Jaynes’ *Prinzip der maximalen Entropie* wählt unter allen Verteilungen mit den bekannten Mittelwerten diejenige mit der größten Entropie. Damit legen wir uns nur so weit fest, wie unser Wissen es verlangt. Nehmen wir das Ergebnis vor dem Beweis vorweg. Das Maximum erreicht genau eine Verteilung. Sie entsteht durch *exponentielle Kippung* (engl. exponential tilt) der Gleichverteilung,

<!--en:115b6a04-->
wobei wir $\eta$ so wählen, dass der Mittelwert $4.5$ beträgt. Die auf Dreien und Sechsen konzentrierte Verteilung büßt $\log6-\log2\approx1.10$ der $1.79$ Nat des fairen Würfels ein, um diesen Mittelwert zu erreichen. Schätzen Sie vor dem Ausführen des folgenden Codes, wie viel Entropie bei der exponentiellen Kippung verloren geht. Da der Mittelwert der gekippten Verteilung monoton mit $\eta$ wächst, finden wir den gesuchten Wert von $\eta$ durch Bisektion.

<!--en:9396c2f1-->
Die Wahrscheinlichkeiten der Augenzahlen bilden eine geometrische Folge. Jede Augenzahl ist $e^{0.371}\approx1.45$-mal so wahrscheinlich wie die nächstkleinere. Die Entropie beträgt $1.61$ Nat, also nur $0.18$ Nat weniger als beim fairen Würfel. Setzen wir $\eta=0$, erhalten wir wieder den fairen Würfel. :numref:`fig_mdl-prob-maxent-die` vergleicht die exponentiellen Kippungen für drei vorgegebene Mittelwerte.

<!--en:729f115b-->
![Verteilungen maximaler Entropie für einen Würfel mit vorgegebener mittlerer Augenzahl $3.5$, $4.5$ oder $5.5$ (schwarze Dreiecke). Jede entsteht durch exponentielle Kippung $p_k\propto e^{\eta k}$ der Gleichverteilung. Mit wachsendem Sollmittelwert steigt der Kippungsparameter $\eta$ und sinkt die Entropie $H$.](../img/mdl-prob-maxent-die.svg)

<!--en:6460058c-->
Diese gekippte Verteilung hat bereits die Form :eqref:`eq_mdl-exp_family` einer Exponentialfamilie, mit $h\equiv1$ für die sechs Augenzahlen, der Statistik $T(k)=k$ und dem natürlichen Parameter $\eta$. Wir haben den Mittelwert einer Statistik festgelegt und die Entropie maximiert. So sind wir direkt zur Exponentialfunktion eines Vielfachen dieser Statistik gelangt. Das liegt nicht an einer Besonderheit des Würfels.

<!--en:b412bcee-->
### Die Proposition über die maximale Entropie

<!--en:3b915aec-->
Für den allgemeinen Satz müssen wir den Entropiebegriff etwas erweitern, denn auf manchen Ergebnisräumen sind natürliche Grundgewichte vorgegeben. Für eine Verteilung $q$ und ein Basismaß $h$ ist die *Entropie relativ zu* $h$ gegeben durch

<!--en:8b1b8fd5-->
für diskrete Ergebnisse. Bei stetigen Dichten tritt ein Integral an die Stelle der Summe. Mit $h\equiv1$ erhalten wir die oben definierte gewöhnliche Entropie.

<!--en:cd9c22a7-->
Im Beweis vergleichen wir jede mögliche Verteilung $q$ direkt mit einem Mitglied $p$ der Exponentialfamilie. Gegeben seien zwei Verteilungen $p$ und $q$ auf demselben Raum. Es gelte $p(\mathbf x)>0$ überall dort, wo $q(\mathbf x)>0$ gilt. Die *Kullback-Leibler-Divergenz* von $q$ bezüglich $p$ ist durch $D_{\textrm{KL}}(q\|p)=\sum q\log(q/p)$ gegeben. Die Summe läuft über die Ergebnisse mit $q>0$. Für Dichten verwenden wir das entsprechende Integral :cite:`Cover.Thomas.1999`. :numref:`sec_mdl-information_theory` behandelt die Entropie und diese Divergenz ausführlich. Hier brauchen wir nur eine elementare Tatsache, die sich mit einer einzigen Tangente beweisen lässt. Da der Logarithmus konkav ist, gilt $\log u\le u-1$ für alle $u>0$, mit Gleichheit nur bei $u=1$ (:numref:`fig_mdl-prob-gibbs-tangent`).

<!--en:5256800a-->
![Die Tangentenschranke $\log u\le u-1$. Der konkave Logarithmus (blau) liegt unter seiner Tangente an der Stelle $u=1$ (orange). Das gilt für alle $u>0$, und nur am Berührpunkt stimmen beide überein.](../img/mdl-prob-gibbs-tangent.svg)

<!--en:d3831096-->
**Lemma (Gibbs-Ungleichung).** *Die Kullback-Leibler-Divergenz ist nie negativ,*

<!--en:3ad9ecc7-->
*wobei Gleichheit genau dann gilt, wenn $q=p$.*

<!--en:006d5240-->
**Beweis.** Wir wenden die Tangentenschranke $\log u \le u-1$ mit $u=p(\mathbf x)/q(\mathbf x)$ an und summieren über alle Ergebnisse mit $q(\mathbf x)>0$:

<!--en:1831a734-->
Gleichheit in der ersten Ungleichung setzt $p=q$ überall dort voraus, wo $q>0$ gilt. Daraus folgt $\sum_{q>0}p=1$. Für $p$ bleibt also keine Wahrscheinlichkeitsmasse außerhalb des Trägers von $q$ übrig. Damit gilt überall $p=q$. $\blacksquare$

<!--en:3094e2ff-->
**Proposition (maximale Entropie).** *Sei $p(\mathbf x)=h(\mathbf x)\exp(\boldsymbol\eta^\top T(\mathbf x)-A(\boldsymbol\eta))$ ein Mitglied einer Exponentialfamilie und $q$ eine beliebige Verteilung (im stetigen Fall mit Dichte). Sie verschwinde überall dort, wo $h$ verschwindet, und der Mittelwert ihrer Statistik, $\mathbb{E}_q[T(\mathbf x)]$, sei endlich. Dieser stimme mit dem Mittelwert unter $p$ überein: $\mathbb{E}_q[T(\mathbf x)]=\mathbb{E}_p[T(\mathbf x)]$. Dann gilt $H_h[q]\le H_h[p]$, mit Gleichheit genau dann, wenn $q=p$.*

<!--en:34e90e8d-->
**Beweis.** Wir zerlegen den Logarithmus in $H_h[q]$, indem wir $p$ einfügen: $\log\frac qh=\log\frac qp+\log\frac ph$. Für den ersten Term ergibt die Gibbs-Ungleichung $-D_{\textrm{KL}}(q\|p)\le0$. Sie ist anwendbar, denn $p>0$ gilt überall dort, wo $h>0$ gilt. Der zweite Term ist linear in der suffizienten Statistik, $\log\frac ph=\boldsymbol\eta^\top T-A(\boldsymbol\eta)$. Daher geht $q$ nur über den Mittelwert von $T$ ein:

<!--en:f1011b40-->
Die mittlere Gleichheit folgt aus der Nebenbedingung für den Mittelwert. Für die letzte Gleichheit wiederholen wir dieselbe Rechnung mit $p$ anstelle von $q$. Zusammen ergeben die beiden Terme $H_h[q]=H_h[p]-D_{\textrm{KL}}(q\|p)\le H_h[p]$, mit Gleichheit nur für $q=p$. $\blacksquare$

<!--en:998eab3a-->
Entscheidend ist der zweite Schritt. Da $\log(p/h)$ linear in $T$ ist, hängt der gemischte Term von $q$ nur über $\mathbb{E}_q[T]$ ab. Diesen Wert legt die Nebenbedingung fest. Anders gesagt: *Erfüllt ein Mitglied der Exponentialfamilie die Mittelwertbedingungen, so hat es als einzige aller Verteilungen, die diese Bedingungen erfüllen, die größte Entropie*. Dafür brauchen wir keine Lagrange-Multiplikatoren. Der Beweis liefert sogar ein globales Maximum und nicht nur einen stationären Punkt.

<!--en:f0f1a02a-->
Beim gezinkten Würfel gehört die gekippte Verteilung mit $\eta\approx0.371$ zur Familie mit $h\equiv1$ und $T(k)=k$. Sie hat den Mittelwert $4.5$ und erreicht daher als einzige das Maximum. Drei bekannte Verteilungen ergeben sich unmittelbar als Folgerungen:

<!--en:3798277c-->
* *Keine Nebenbedingung* ($T\equiv0$): Jedes $q$ ist zulässig, und $p\propto h$ erreicht das Maximum, sofern die Summe über $h$ endlich ist. Auf einer endlichen Menge mit $h\equiv1$ ist dies die Gleichverteilung mit Entropie $\log K$. Der faire Würfel entspricht dem Fall $K=6$.

<!--en:374039aa-->
* *Mittelwert und Varianz auf $\mathbb R$*: Mit $T(x)=(x,x^2)$ und konstantem $h$ erreicht die Gauß-Verteilung das Maximum. Da ein konstantes $h$ die Entropie $H_h$ nur um eine feste Konstante verschiebt, hat die Gauß-Verteilung unter allen Dichten mit vorgegebenem Mittelwert und vorgegebener Varianz die größte Entropie.

<!--en:cedb4cf7-->
* *Mittelwert für Zähldaten* $\{0,1,2,\dots\}$ mit $T(k)=k$: Wählen wir $h(k)=1/k!$, so erreicht die Entropie $H_h$ bei vorgegebenem Mittelwert $m$ ihr Maximum für die Poisson-Verteilung mit Rate $\lambda=m$. Wählen wir hingegen $h\equiv1$, ergibt sich die geometrische Verteilung $p_k=(1-r)r^k$ mit $r=m/(1+m)$. Das Basismaß entscheidet, welche Verteilung entsteht. Dies untersuchen Sie in Aufgabe :exref:`ex_mdl-distributions-max-entropy-integers`.

<!--en:4477359b-->
Die Voraussetzung, dass ein Mitglied der Familie die Nebenbedingungen erfüllt, ist wesentlich. Angenommen, wir geben nur den Mittelwert $m$ auf ganz $\mathbb R$ vor, mit $h\equiv1$. Die in Betracht kommende Dichte $e^{\eta x}$ ist auf $\mathbb R$ für kein $\eta$ integrierbar. Es gibt also kein solches Mitglied. Es gibt sogar überhaupt keine Verteilung maximaler Entropie. Die Gauß-Verteilung $\mathcal N(m,s^2)$ hat den Mittelwert $m$ und die Entropie $\frac12\log(2\pi e s^2)$. Diese wächst für $s\to\infty$ über alle Grenzen.

<!--en:a2e4d3cd-->
## Die logarithmische Zustandssumme

<!--en:6a878855-->
In der Übersicht in :numref:`sec_mdl-distributions` schien jede Berechnung von Mittelwert und Varianz einen eigenen Kunstgriff zu verlangen. Bei der Bernoulli-Verteilung nutzten wir $X^2=X$, bei der Binomialverteilung eine Summe unabhängiger Münzwürfe, bei der Poisson-Verteilung eine Indexverschiebung in der Reihe und bei der Gauß-Verteilung partielle Integration. Innerhalb der Exponentialfamilie genügt für alle diese Rechnungen eine einzige Operation. Wir leiten die logarithmische Zustandssumme $A$ ab. Die erste Ableitung liefert den Mittelwert, die zweite die Varianz. Dieselbe Funktion bestimmt auch die Maximum-Likelihood-Schätzung.

<!--en:b49a1e74-->
### Die Momenteneigenschaft

<!--en:2e519c27-->
Erinnern wir uns: $A(\boldsymbol\eta)$ ist der Logarithmus eines Integrals oder einer Summe von Exponentialtermen $\exp(\boldsymbol\eta^\top T(\mathbf x))$. Beim Ableiten dieses Logarithmus tritt der Faktor $T(\mathbf x)$ vor die Exponentialfunktion, gewichtet mit der Wahrscheinlichkeit von $\mathbf x$. Das ergibt gerade den Erwartungswert von $T$.

<!--en:7bd98881-->
**Proposition (Mittelwert der suffizienten Statistiken).** *Für eine Verteilung :eqref:`eq_mdl-exp_family` aus einer Exponentialfamilie gilt*

<!--en:d509ff8f-->
**Beweis.** Wir differenzieren :eqref:`eq_mdl-log_partition` unter dem Integralzeichen. Dies ist im Inneren der Menge der natürlichen Parameter zulässig, für die $A(\boldsymbol\eta)<\infty$ gilt :cite:`Folland.1999`. Nach der Hölder-Ungleichung ist diese Menge konvex. Von nun an wählen wir $\boldsymbol\eta$ in ihrem Inneren. Schreiben wir $Z(\boldsymbol\eta)=e^{A(\boldsymbol\eta)}=\int h(\mathbf x)e^{\boldsymbol\eta^\top T(\mathbf x)}d\mathbf x$, so liefert die Kettenregel $\nabla A = (\nabla Z)/Z$ mit

<!--en:88381bad-->
Durch Division durch $Z=e^{A}$ erhalten wir unter dem Integral den Faktor $e^{-A}$. Damit steht dort die vollständige Dichte :eqref:`eq_mdl-exp_family`:

<!--en:86caf876-->
Leiten wir $\nabla A=(\nabla Z)/Z$ ein zweites Mal ab und vertauschen wieder Integration und Differentiation, erhalten wir die Kovarianzmatrix der suffizienten Statistiken:

<!--en:7d832733-->
Kurz gesagt liefert die Steigung von $A$ den Mittelwert von $T$ und ihre Krümmung die Kovarianz von $T$. Jede Kovarianzmatrix ist positiv semidefinit ($\mathbf v^\top\textrm{Cov}(T)\mathbf v=\textrm{Var}(\mathbf v^\top T)\ge0$ für alle $\mathbf v$, siehe :numref:`subsec_mdl-psd`). Daher ist die logarithmische Zustandssumme $A$ immer konvex.

<!--en:81ac625a-->
Prüfen wir dies an zwei vertrauten Beispielen. Für die Münze liefert $A(\eta)=\log(1+e^\eta)$ den Bernoulli-Mittelwert $A'(\eta)=\operatorname{sigmoid}(\eta)=p$ und die Varianz $A''(\eta)=\operatorname{sigmoid}(\eta)\bigl(1-\operatorname{sigmoid}(\eta)\bigr)=p(1-p)$. Bei $\hat\eta=\log\frac94$ ergeben sich $9/13\approx0.692$ und $\frac{9}{13}\cdot\frac{4}{13}=\frac{36}{169}\approx0.213$. Für die Poisson-Verteilung ist $A(\eta)=e^\eta$ seine eigene Ableitung. Bei $\lambda=4$ ($\eta=\log4$) haben also $A'$ und $A''$ beide den Wert $4$. Da die Momente Ableitungen sind, können wir sie mit automatischem Differenzieren direkt für jede Funktion $A$ berechnen, die wir in Code ausdrücken können. Prüfen wir beide Fälle.

<!--en:1682f1da-->
Beide Ableitungen stimmen in allen vier ausgegebenen Stellen mit dem analytischen Mittelwert und der analytischen Varianz überein. In der Übersicht brauchten wir eine Potenzreihe, um die bekannte Gleichheit von Mittelwert und Varianz der Poisson-Verteilung zu zeigen. Jetzt genügt eine Zeile: $A=A'=A''$ für $A(\eta)=e^\eta$.

<!--en:3d6bc368-->
### Maximum-Likelihood-Schätzung durch Momentenabgleich

<!--en:ef636d4f-->
Dank der Momenteneigenschaft lässt sich die Maximum-Likelihood-Schätzung durch eine einfache Bedingung an die Erwartungswerte beschreiben. Das angepasste Modell muss die Stichprobenmittelwerte der suffizienten Statistiken reproduzieren. Für unabhängige Beobachtungen $\mathbf x_1,\dots,\mathbf x_n$ nehmen wir den negativen Logarithmus der als Produkt geschriebenen Likelihood aus :eqref:`eq_mdl-exp_family` und erhalten

<!--en:bc1d398e-->
Die Daten gehen hier nur über die Summe $\sum_{i=1}^n T(\mathbf x_i)$ ein. Da $A$ konvex und der zweite Term linear in $\boldsymbol\eta$ ist, ist die negative Log-Likelihood konvex in $\boldsymbol\eta$. Wir setzen ihren Gradienten gleich null und wenden die Momenteneigenschaft :eqref:`eq_mdl-exp_family_moment` an. Das ergibt

<!--en:a9492f94-->
Die Maximum-Likelihood-Schätzung setzt also den Erwartungswert der suffizienten Statistik unter dem Modell gleich ihrem beobachteten Stichprobenmittelwert. Dieses Prinzip heißt *Momentenabgleich* (engl. moment matching). Da die Zielfunktion konvex ist, nimmt sie bei jeder Lösung dieser Gleichung ihr globales Minimum an.

<!--en:e8072e76-->
Was ergibt sich für unsere Münze mit 9-mal Kopf in 13 Würfen? Ihre negative Log-Likelihood ist $13A(\eta)-9\eta=13\log(1+e^\eta)-9\eta$. Die Ableitung $13\operatorname{sigmoid}(\eta)-9$ verschwindet genau dort, wo $\operatorname{sigmoid}(\hat\eta)=9/13$ gilt. Das ergibt $\hat\eta=\log\frac94\approx0.811$. An diesem Minimum beträgt der Verlust

<!--en:80f9bd29-->
also das 13-Fache der Entropie einer Münze mit Kopfwahrscheinlichkeit $9/13$, nämlich $-\frac9{13}\log\frac9{13}-\frac4{13}\log\frac4{13}\approx0.617$. Im Optimum ist der mittlere Verlust pro Wurf gleich der Entropie der angepassten Münzverteilung.

<!--en:87945294-->
Diese Identität gilt für jede Exponentialfamilie mit $h\equiv1$. Die mittlere negative Log-Likelihood bei $\hat{\boldsymbol\eta}$ beträgt $A(\hat{\boldsymbol\eta})-\hat{\boldsymbol\eta}^\top\bar T$, und die Entropie der angepassten Verteilung ist $A(\hat{\boldsymbol\eta})-\hat{\boldsymbol\eta}^\top\nabla A(\hat{\boldsymbol\eta})$. Wegen $\nabla A(\hat{\boldsymbol\eta})=\bar T$ stimmen beide Werte überein.

<!--en:f459688d-->
Die zweite Ableitung des Verlusts am Minimum beträgt $13A''(\hat\eta)=13\cdot\frac{36}{169}=\frac{36}{13}\approx2.77$. Bei großer Krümmung steigt der Verlust mit dem Abstand von $\hat\eta$ steil an. Die Log-Odds sind dann eng eingegrenzt. Dreizehn Würfe lassen noch beträchtlichen Spielraum. Eine Abweichung um $1/\sqrt{2.77}\approx0.60$ von $\hat\eta\approx0.81$ erhöht den Verlust nur um etwa ein halbes Nat. In :numref:`sec_mdl-maximum_likelihood` erkennen wir diese Krümmung als Fisher-Information wieder und leiten daraus einen Fehlerbalken ab. :numref:`fig_mdl-prob-log-partition` zeigt die Tangente an $A$ bei $\hat\eta$ und den Verlauf der negativen Log-Likelihood.

<!--en:1b322e5c-->
![Links die logarithmische Zustandssumme $A(\eta)=\log(1+e^\eta)$ der Bernoulli-Verteilung mit ihren Asymptoten und ihrer Tangente bei $\hat\eta=\log(9/4)$. Die Steigung der Tangente ist der Mittelwert $9/13$. Rechts die negative Log-Likelihood $13A(\eta)-x\eta$ für $x=9$ Kopfwürfe unter 13 Würfen mit Minimum bei $\hat\eta$ und für $x=13$ Kopfwürfe. Im zweiten Fall fällt sie für jedes $\eta$ und erreicht ihr Minimum an keiner endlichen Stelle.](../img/mdl-prob-log-partition.svg)

<!--en:d41a3653-->
Dabei ist eine Einschränkung wichtig. Die Gleichung für den Momentenabgleich $\nabla A(\hat{\boldsymbol\eta})=\bar T$ hat nur dann eine Lösung, wenn ein Mitglied der Familie den Mittelwert $\bar T$ tatsächlich erreichen kann. Beim Bernoulli-Münzwurf sind die erreichbaren Mittelwerte $\operatorname{sigmoid}(\eta)$ für reelle $\eta$. Sie bilden das offene Intervall $(0,1)$. Fällt bei allen 13 Würfen Kopf, liegt $\bar T=1$ auf dem Rand und ist nicht erreichbar. Die negative Log-Likelihood lautet dann $13A(\eta)-13\eta=13\log(1+e^{-\eta})$. Sie fällt für jedes $\eta$ monoton und hat den Wert $9.01$ bei $\eta=0$, $0.087$ bei $\eta=5$ und $0.0006$ bei $\eta=10$. Sie nähert sich $0$, ohne diesen Wert je zu erreichen. Es gibt also keine endliche Minimalstelle. Die angepasste Wahrscheinlichkeit nähert sich $1$ für $\hat\eta\to\infty$.

<!--en:109e606c-->
Bei der logistischen Regression tritt dasselbe Problem auf, wenn eine Hyperebene die beiden Klassen im Trainingsdatensatz trennt. Mit zunehmender Skalierung der Gewichte sinkt der Verlust immer weiter und treibt die Gewichte ins Unendliche. Eine A-priori-Verteilung für die Parameter, wie wir sie am Ende dieses Abschnitts einführen, oder ein Gewichtszerfall liefert wieder ein endliches Optimum. Die Einzelheiten untersuchen Sie in Aufgabe :exref:`ex_mdl-maximum-likelihood-small-samples-boundary` in :numref:`sec_mdl-maximum_likelihood`.

<!--en:73696a36-->
Existiert eine Minimalstelle, so ist sie eindeutig, wenn die negative Log-Likelihood strikt konvex ist. Dies gilt, wenn $\textrm{Cov}(T)$ positiv definit ist. Das bedeutet, dass keine nichttriviale Linearkombination der suffizienten Statistiken auf dem gesamten Träger konstant ist. Eine solche Parametrisierung heißt *minimal*. Die Parametrisierung der Kategorialverteilung durch alle $K$ Logits ist nicht minimal. Die $K$ One-Hot-Koordinaten summieren sich stets zu eins. Addieren wir zu jedem Logit dieselbe Konstante, bleiben alle Wahrscheinlichkeiten gleich. Daher ist die negative Log-Likelihood in Richtung $\mathbf 1$ konstant. Ihre Minimalstellen bilden eine Gerade. Bibliotheken für Deep Learning verwenden dennoch alle $K$ Logits. Sie nutzen die Verschiebungsinvarianz, um vor dem Exponenzieren das größte Logit abzuziehen und so die numerische Stabilität zu verbessern (:numref:`sec_softmax_concise`).

<!--en:88369871-->
### Verallgemeinerte lineare Modelle

<!--en:ea490941-->
Beim überwachten Lernen sagen Modelle für ein Label $y$ die bedingte Verteilung bei gegebener Eingabe $\mathbf x$ vorher. Ein *verallgemeinertes lineares Modell* (engl. generalized linear model, GLM) wählt diese Verteilung aus einer Exponentialfamilie :eqref:`eq_mdl-exp_family`. Die Eingabe geht über eine lineare Abbildung $\mathbf W^\top\mathbf x$ ein, wobei ein konstantes Merkmal einen etwaigen Bias aufnimmt. Am einfachsten setzen wir diese lineare Abbildung direkt mit dem natürlichen Parameter gleich, $\boldsymbol\eta = \mathbf W^\top\mathbf x$. Diese Zuordnung heißt *kanonische Linkfunktion* (engl. canonical link). Sei $\boldsymbol\mu(\mathbf x) = \nabla A(\boldsymbol\eta) = \mathbb{E}[T(y)\mid\mathbf x]$ der vom Modell vorhergesagte Mittelwert der suffizienten Statistik. Für ein einzelnes Beispiel lautet die negative Log-Likelihood dann $A(\boldsymbol\eta) - \boldsymbol\eta^\top T(y) - \log h(y)$. Nach der Momenteneigenschaft :eqref:`eq_mdl-exp_family_moment` ist ihr Gradient bezüglich $\boldsymbol\eta$ gleich $\boldsymbol\mu(\mathbf x) - T(y)$. Wenden wir die Kettenregel auf $\boldsymbol\eta = \mathbf W^\top\mathbf x$ an, erhalten wir unmittelbar

<!--en:e085ef9d-->
Unabhängig von der Exponentialfamilie ist der Gradient immer das Produkt aus Eingabe und Vorhersagefehler. Dieser Fehler ist der vorhergesagte Mittelwert abzüglich der beobachteten Statistik. Mit der Momenteneigenschaft lässt sich die Frage vom Anfang des Abschnitts in zwei Zeilen beantworten. Außerdem ist der Verlust konvex in $\mathbf W$. Wir verketten nämlich die konvexe Funktion $A$ mit einer linearen Funktion von $\mathbf W$ und addieren einen linearen Term. Diese einfache Gradientenformel gilt speziell für die kanonische Linkfunktion. Das Probit-Modell in :numref:`subsec_mdl-softmax-origins` setzt denselben linearen Score in die Verteilungsfunktion einer Gauß-Verteilung ein. Sein Gradient enthält daher einen zusätzlichen Faktor, der von der Vorhersage abhängt. Tauschen wir die Exponentialfamilie aus, erhalten wir vier bekannte Modelle:

<!--en:8675ed42-->
| Label $y$ | Familie | $T(y)$ | Mittelwert $\boldsymbol\mu(\mathbf x)$ | Modell |

<!--en:7c7f0de8-->
| reelle Zahl | Gauß-Verteilung, Varianz 1 | $y$ | $\eta$ | lineare Regression |

<!--en:01e50342-->
| ja oder nein | Bernoulli-Verteilung | $y \in \{0, 1\}$ | $\operatorname{sigmoid}(\eta)$ | logistische Regression |

<!--en:a24cb1ac-->
| eine von $K$ Klassen | Kategorialverteilung | One-Hot-Vektor $\mathbf y$ | $\mathrm{softmax}(\boldsymbol\eta)$ | Softmax-Regression |

<!--en:a99e5965-->
| Anzahl | Poisson-Verteilung | $y$ | $e^{\eta}$ | Poisson-Regression |

<!--en:6e359d6e-->
Betrachten wir die Regel für eine weniger vertraute Art von Label in einem Poisson-Modell für Zähldaten. Sagt das Modell für eine Eingabe den Mittelwert $e^\eta=3$ vorher und beträgt die beobachtete Anzahl $y=5$, so ist der Gradient des Verlusts bezüglich $\eta$ gleich $3-5=-2$. Ein Schritt des Gradientenabstiegs erhöht also $\eta$. Damit steigt der vorhergesagte Mittelwert in Richtung $5$.

<!--en:03db5103-->
In der ersten Zeile setzen wir die Varianz der Gauß-Verteilung auf $1$ fest. Damit geht der quadratische Term $e^{-y^2/2}$ in das Basismaß $h(y)$ ein. Die negative Log-Likelihood ergibt bis auf eine additive Konstante den quadratischen Verlust aus :numref:`sec_linear_regression`. Ein tiefes neuronales Netz behält diese letzte Schicht bei: eine lineare Abbildung in die natürlichen Parameter, gefolgt von der negativen Log-Likelihood der Familie. An die Stelle der ursprünglichen Eingabe $\mathbf x$ treten gelernte Repräsentationen. Der Gesamtverlust ist zwar nicht mehr konvex in den Gewichten des Netzes. Sein Gradient bezüglich der natürlichen Parameter in der Ausgabeschicht ist aber weiterhin die Differenz aus Vorhersage und Beobachtung :cite:`Bishop.2006`.

<!--en:33ee911f-->
## Ursprung der Softmax-Funktion

<!--en:5b50a897-->
Im Haupttext wandeln Klassifikatoren einen Vektor aus $K$ Logits $\mathbf z$ mit der *Softmax-Funktion* in Klassenwahrscheinlichkeiten um,

<!--en:5393518e-->
Diese Abbildung hat sich bereits in unseren Tabellen zur Exponentialfamilie aus der Zeile für die Kategorialverteilung ergeben. Ihre Vorzüge liegen auf der Hand. Jede Wahrscheinlichkeit wächst mit ihrem eigenen Logit, und alle Wahrscheinlichkeiten sind positiv und summieren sich zu eins. Doch viele andere Abbildungen haben dieselben Eigenschaften. Normieren wir beispielsweise $\log(1+e^{z_k})$ anstelle von $e^{z_k}$, erhalten wir ebenfalls positive Wahrscheinlichkeiten mit derselben Rangfolge wie die Logits. Die Softmax-Funktion zeichnet sich dadurch aus, dass Differenzen von Logits den Log-Odds entsprechen: $z_i-z_j=\log(p_i/p_j)$. Drei unabhängige Argumente aus drei verschiedenen Gebieten führen zu genau dieser Formel. Eines geht von einem Modell für menschliche Entscheidungen unter Rauschen aus, eines von der Boltzmann-Verteilung der statistischen Physik und eines von der Exponentialfamilie. Das dritte Argument ist das kürzeste. Nur dieses brauchen wir für den Rest des Abschnitts.

<!--en:a1f7832c-->
### Entscheidungen unter Rauschen und Gumbel-Rauschen

<!--en:a7bad269-->
Das erste Argument modelliert, wie diskrete Label überhaupt entstehen. Stellen wir uns vor, jemand wählt beim Einkauf ein Produkt aus oder vergibt ein Label für ein Bild. Dabei erhält jede Option $k$ einen zugrunde liegenden Score $z_k$, doch gelegentlich kommt es zu Fehlern. Eine naheliegende Modellierung addiert zu jedem Score unabhängiges Rauschen $\epsilon_k$ und wählt die Option mit dem höchsten verrauschten Score:

<!--en:2e74d1c1-->
In diesem Modell ist die Wahrscheinlichkeit der Klasse $k$ gleich der Wahrscheinlichkeit, dass die Option $k$ trotz des Rauschens den höchsten Score erzielt. Ihre Formel hängt also von der Verteilung des Rauschens ab. Ein erster Versuch ist meist Gauß-Rauschen. Bei zwei Optionen und $\epsilon_k\sim\mathcal N(0,s^2)$ gewinnt Option 1 genau dann, wenn $\epsilon_2-\epsilon_1<z_1-z_2$ gilt. Die Differenz $\epsilon_2-\epsilon_1$ ist selbst gaußverteilt mit Varianz $2s^2$. Damit beträgt die Gewinnwahrscheinlichkeit

<!--en:adcf4f10-->
wobei $\Phi$ die Verteilungsfunktion der Standardnormalverteilung bezeichnet (und $\phi$ ihre Dichte). Dies ist das *Probit-Modell*. Die Wahlwahrscheinlichkeit hängt nur von der Differenz der Scores ab, gemessen in Einheiten der Rauschskala.

<!--en:27834be4-->
Bei mehr als zwei Optionen besitzt das Probit-Modell leider keine geschlossene Form mehr. Die Wahrscheinlichkeit, dass Option $k$ alle $K-1$ anderen übertrifft, ist dann das Integral über ein Produkt von $K-1$ Gauß-Verteilungsfunktionen. Dafür gibt es keine elementare Formel. Wir müssten für jedes Trainingsbeispiel bei jedem Schritt sowohl dieses Integral als auch seinen Gradienten numerisch berechnen.

<!--en:55934ac7-->
Ersetzen wir das Gauß-Rauschen durch Gumbel-Rauschen, entfällt das Integral vollständig. Die standardisierte *Gumbel-Verteilung* hat die Verteilungsfunktion $G(x)=\exp(-e^{-x})$. In der Extremwerttheorie tritt sie als Grenzverteilung eines Maximums auf. Dazu ziehen wir viele unabhängige Werte aus einer Verteilung mit exponentiell abfallendem Rand, etwa einer Gauß- oder Exponentialverteilung, und verschieben und skalieren ihr Maximum geeignet :cite:`Gumbel.1954`. :citet:`McFadden.1974` verwendete sie zur Modellierung wirtschaftlicher Entscheidungen.

<!--en:cabb008d-->
**Proposition (Gumbel-Rauschen liefert die Softmax-Funktion).** *Sind $\epsilon_1,\dots,\epsilon_K$ unabhängige standardisierte Gumbel-Zufallsvariablen, so folgt das Argmax der verrauschten Scores der Softmax-Verteilung,*

<!--en:a2153339-->
**Beweis.** Sei $S=\sum_j e^{z_j}$. Jeder verrauschte Score $z_j+\epsilon_j$ hat die Verteilungsfunktion $G(t-z_j)=\exp(-e^{z_j}e^{-t})$. Angenommen, Option $k$ erzielt den Score $t=z_k+\epsilon_k$. Dass alle übrigen $K-1$ Optionen unter $t$ bleiben, hat wegen der Unabhängigkeit die Wahrscheinlichkeit $\prod_{j\ne k}\exp(-e^{z_j}e^{-t})=\exp\bigl(-(S-e^{z_k})e^{-t}\bigr)$. Wir multiplizieren mit der Dichte $e^{z_k}e^{-t}\exp(-e^{z_k}e^{-t})$ von $t=z_k+\epsilon_k$, integrieren über $t$ und setzen $v=e^{-t}$. Das ergibt

<!--en:b1ca4a39-->
Die Rechnung geht auf, weil das Produkt von Gumbel-Verteilungsfunktionen wieder eine Gumbel-Verteilungsfunktion ist. Für das Maximum der Scores aller $K$ Optionen gilt $P\bigl(\max_j(z_j+\epsilon_j)\le t\bigr)=\exp(-Se^{-t})=G(t-\log S)$. Der höchste Score ist also eine standardisierte Gumbel-Zufallsvariable, verschoben um den Log-Sum-Exp-Wert $\log\sum_j e^{z_j}$. Aus dieser Sicht ist Log-Sum-Exp ein glattes Maximum der Logits. Sein Gradient, die Softmax-Funktion, gibt für jede Option die Wahrscheinlichkeit an, mit der sie dieses Maximum erreicht. Aufgabe :exref:`ex_mdl-distributions-exponential-races` liefert einen zweiten Beweis derselben Identität anhand konkurrierender exponentiell verteilter Wartezeiten.

<!--en:892b68b7-->
Umgekehrt gelesen liefert die Proposition auch ein Verfahren zum Ziehen von Zufallswerten. Um eine Kategorie aus $\mathrm{softmax}(\mathbf z)$ zu ziehen, addieren wir zu jedem Logit unabhängiges standardisiertes Gumbel-Rauschen und bestimmen das Argmax. Dieses verbreitete Verfahren heißt *Gumbel-Max-Trick*.

<!--en:9a99b6f2-->
Im Rest von :numref:`subsec_mdl-softmax-origins` folgen wir der Schreibweise des Haupttexts. Das Symbol $T$ bezeichnet nun die Temperatur und keine suffiziente Statistik mehr. Da $\operatorname{argmax}_k(z_k+T\epsilon_k)=\operatorname{argmax}_k(z_k/T+\epsilon_k)$ für jedes $T>0$ gilt, liefert die Skalierung des Gumbel-Rauschens mit $T$ Ziehungen aus $\mathrm{softmax}(\mathbf z/T)$. Stärker verrauschte Entscheidungen entsprechen höheren Temperaturen. Prüfen wir den Gumbel-Max-Trick direkt. Wir simulieren $10^5$ Entscheidungen zwischen vier Logits bei $T=1$ und $T=2$ und vergleichen die empirischen Gewinnhäufigkeiten mit den exakten Softmax-Wahrscheinlichkeiten.

<!--en:25c0fed9-->
Die empirischen Häufigkeiten stimmen bis auf wenige Tausendstel mit $\mathrm{softmax}(\mathbf z/T)$ überein. Das entspricht den Monte-Carlo-Schwankungen bei $10^5$ Ziehungen. Eine Verdopplung der Rauschskala gleicht die Gewinnhäufigkeiten genauso aneinander an wie eine Halbierung der Logits.

<!--en:2962e934-->
Bei zwei Optionen folgt die Differenz zweier unabhängiger standardisierter Gumbel-Zufallsvariablen der logistischen Verteilung mit Verteilungsfunktion $\operatorname{sigmoid}(x)=1/(1+e^{-x})$. Dieses *Logit-Modell* liefert also $P(y=1)=\operatorname{sigmoid}(z_1-z_2)$. Die Sigmoidfunktion nähert eine umskalierte Gauß-Verteilungsfunktion gut an ($|\operatorname{sigmoid}(1.7x)-\Phi(x)|<0.01$ für alle $x$). Deshalb liefern angepasste Probit- und Logit-Modelle für gewöhnliche Beispiele nahezu identische Wahrscheinlichkeiten. An den Rändern unterscheiden sie sich jedoch. Bei einem Ausreißer, dessen Label der Vorhersage mit hoher Sicherheit widerspricht, wächst der Logit-Verlust nur linear mit der Differenz der Scores, der Probit-Verlust hingegen quadratisch. Ausreißer können die Probit-Schätzung daher wesentlich stärker beeinflussen (Aufgabe :exref:`ex_mdl-distributions-probit-logit`).

<!--en:73c2844b-->
### Temperatur und Boltzmann-Verteilung

<!--en:59ec1010-->
Das zweite Argument stammt aus der statistischen Physik. Dort beschreibt genau dasselbe Exponentialgesetz die Energiezustände. Ein physikalisches System im thermischen Gleichgewicht bei Temperatur $T$, etwa ein Molekül in einem Gas, befindet sich im Zustand $k$ mit Energie $E_k$ mit der Wahrscheinlichkeit

<!--en:5852bb1f-->
wobei $k_{\mathrm B}$ die Boltzmann-Konstante ist. Zustände niedriger Energie sind wahrscheinlicher. Die Temperatur $T$ bestimmt, wie stark sich die Wahrscheinlichkeit auf sie konzentriert. Boltzmann entdeckte diese Verteilung für Gasmoleküle, und :citet:`Gibbs.1902` baute die statistische Mechanik darauf auf. Messen wir die Temperatur in Energieeinheiten (setzen wir also $k_{\mathrm B}=1$) und identifizieren die Energien mit den negativen Logits ($E_k=-z_k$), so wird das Boltzmann-Gesetz zu $\mathrm{softmax}(\mathbf z/T)$. In dieser Sprache ist ein Logit eine negative Energie. Die wahrscheinlichste Klasse ist der Zustand niedrigster Energie.

<!--en:e52f6dd0-->
Unsere Proposition über die maximale Entropie liefert dieselbe Formel ganz ohne Physik. In der Physik geben wir nur die mittlere Energie vor. Bei der Klassifikation fixiert die entsprechende Nebenbedingung das mittlere Logit $\sum_k p_k z_k$. Wir nehmen die $K$ Klassen als Ergebnisse mit $h\equiv1$ und das Logit $z_k$ als Statistik. Betrachten wir alle Verteilungen auf den $K$ Klassen mit einem vorgegebenen mittleren Logit, das strikt zwischen dem kleinsten und dem größten Eintrag von $\mathbf z$ liegt. Unter diesen hat $p_k\propto e^{\eta z_k}$ die maximale Entropie. Übersteigt der vorgegebene Mittelwert das ungewichtete Mittel der Logits, so gilt $\eta>0$. Setzen wir $T=1/\eta$, erhalten wir wieder $\mathrm{softmax}(\mathbf z/T)$. Sinkt die Temperatur $T$ von $\infty$ auf $0$, so steigt das mittlere Logit stetig vom Mittelwert der Logits bis zu ihrem Maximum. Für die Logits $\mathbf z=(2,1,0.1,-1)$ in unserer Simulation ergibt die gewöhnliche Softmax-Funktion ($T=1$) ein mittleres Logit von $1.49$. Es liegt zwischen dem ungewichteten Mittelwert $0.525$ und dem Maximum $2$.

<!--en:2eb0108f-->
### Die Kategorialverteilung als Exponentialfamilie

<!--en:8b88ca6f-->
Das dritte Argument kennen wir bereits vom Anfang dieses Abschnitts. Kodieren wir das Klassenlabel als One-Hot-Vektor $\mathbf y$, so lautet die Kategorialverteilung $P(\mathbf y)=\exp\bigl(\mathbf z^\top\mathbf y-A(\mathbf z)\bigr)$ mit logarithmischer Zustandssumme $A(\mathbf z)=\log\sum_k e^{z_k}$. Sie bildet eine Exponentialfamilie, deren natürliche Parameter die Logits sind. Nach der Momenteneigenschaft beträgt ihr Mittelwert $\nabla A(\mathbf z)=\mathrm{softmax}(\mathbf z)=\mathbb{E}[\mathbf y]$. Ihre negative Log-Likelihood ist damit konvex in $\mathbf z$, und nach :eqref:`eq_mdl-glm-gradient` ist ihr Gradient gleich $\mathrm{softmax}(\mathbf z)-\mathbf y$.

<!--en:85ae35e7-->
Alle drei Argumente führen zur selben Funktion, doch jedes erklärt einen anderen Aspekt. Entscheidungen unter Rauschen liefern ein exaktes Verfahren zum Ziehen von Zufallswerten. Die Boltzmann-Verteilung erklärt die Temperaturskalierung. Die Exponentialfamilie liefert eine konvexe Verlustfunktion mit einem leicht zu berechnenden Gradienten.

<!--en:878705db-->
## Konjugierte A-priori-Verteilungen

<!--en:7099f5dc-->
Die Maximum-Likelihood-Schätzung fasst 9-mal Kopf in 13 Würfen in einer einzigen Zahl zusammen, $9/13$. Eine Punktschätzung allein sagt uns nichts darüber, wie sehr wir ihr vertrauen können. Bei 9-mal Kopf in 13 Würfen ergibt sich genau derselbe Anteil wie bei 900-mal Kopf in 1300 Würfen. Beschreiben wir unser Wissen durch eine Wahrscheinlichkeitsverteilung über die Kopfwahrscheinlichkeit der Münze, erfassen wir die Unsicherheit direkt. Idealerweise können wir diese Verteilung nach jedem neuen Wurf leicht aktualisieren, ohne von vorn zu beginnen.

<!--en:2451d40c-->
Von nun an behandeln wir die Kopfwahrscheinlichkeit selbst als unbekannte Zufallsgröße. Wie in den folgenden Abschnitten bezeichnen wir sie mit $\theta$. Eine Verteilung über $\theta$ vor der Beobachtung der Daten heißt *A-priori-Verteilung*. Nach der Bayes-Regel multiplizieren wir die A-priori-Verteilung mit der Likelihood und normieren das Ergebnis, $p(\theta\mid x)\propto p(\theta)\,P(x\mid\theta)$. So erhalten wir die *A-posteriori-Verteilung* (engl. posterior). Eine A-priori-Verteilung heißt zu einer Likelihood *konjugiert*, wenn die A-posteriori-Verteilung derselben Familie angehört wie die A-priori-Verteilung. Beobachten wir Daten, ändern sich dann nur die Parameter.

<!--en:f8b6497e-->
### Die Beta-Verteilung als A-priori-Verteilung für eine Münze

<!--en:9fe2b8c2-->
Eine A-priori-Dichte für die Kopfwahrscheinlichkeit einer Münze muss ihren Träger in $[0,1]$ haben. Als Funktion von $\theta$ ist die Likelihood für $x$-mal Kopf in $n$ Würfen gleich $\theta^x(1-\theta)^{n-x}$. Hat die A-priori-Dichte dieselbe algebraische Form, genügt beim Multiplizieren die Addition der Exponenten. Für ihre Normierungskonstante brauchen wir die Gammafunktion $\Gamma(z)=\int_0^\infty t^{z-1}e^{-t}\,dt$ aus :numref:`sec_mdl-distributions`, die über $\Gamma(z+1)=z\,\Gamma(z)$ und $\Gamma(n+1)=n!$ die Fakultät verallgemeinert. Die *Beta-Verteilung* mit Parametern $a,b>0$ hat die Dichte

<!--en:633edf0f-->
mit der Normierungskonstante $B(a,b)=\int_0^1\theta^{a-1}(1-\theta)^{b-1}d\theta$ gleich $\Gamma(a)\Gamma(b)/\Gamma(a+b)$. Aus dieser Konstante lässt sich der Mittelwert leicht bestimmen. Multiplizieren wir die Dichte mit $\theta$, erhöht sich $a$ um eins:

<!--en:0c227b31-->
Hier haben wir die Gammafaktoren mithilfe von $\Gamma(a+1)=a\,\Gamma(a)$ und $\Gamma(a+b+1)=(a+b)\Gamma(a+b)$ gekürzt. Zweimal angewendet liefert derselbe Kunstgriff die Varianz. Bezeichnen wir den Mittelwert mit $m=a/(a+b)$, so gilt

<!--en:73a35dad-->
Mit wachsendem $a+b$ konzentriert sich die Dichte stärker um ihren Mittelwert. Für $a=b=1$ erhalten wir die konstante Dichte der Gleichverteilung auf $[0,1]$.

<!--en:828727da-->
Angenommen, wir beobachten $x$-mal Kopf in $n$ Würfen. Multiplizieren wir die A-priori-Dichte mit der Likelihood, addieren sich die Exponenten direkt,

<!--en:9d60fd08-->
und nach der Normierung erhalten wir wieder eine Beta-Dichte. Ihre Parameter erhöhen sich um die beobachteten Anzahlen:

<!--en:800066a4-->
Diese Aktualisierung lässt sich mithilfe von *Pseudoanzahlen* (engl. pseudo-counts) deuten. Die Parameter $a$ und $b$ zählen im Voraus virtuelle Würfe mit Kopf und Zahl. Beobachten wir $x$-mal Kopf und $n-x$-mal Zahl, addieren wir die entsprechenden Anzahlen. Der A-posteriori-Mittelwert,

<!--en:c43bdeb0-->
ist der Anteil der Kopfwürfe einschließlich dieser Pseudoanzahlen. Er liegt zwischen dem A-priori-Mittelwert $a/(a+b)$ und dem Maximum-Likelihood-Schätzwert $x/n$. Er konvergiert gegen $x/n$, wenn $n$ wächst.

<!--en:6718f61f-->
Ein feiner Unterschied ist wichtig. Wie vielen Würfen die A-priori-Verteilung entspricht, hängt von der betrachteten Kenngröße ab. Sind beide A-posteriori-Parameter größer als $1$, so beträgt der A-posteriori-Modus $(a+x-1)/(a+b+n-2)$. Zum Modus trägt die A-priori-Verteilung also $a-1$-mal Kopf und $b-1$-mal Zahl bei. Es gilt folgende Konvention: *$a$ und $b$ sind die Pseudoanzahlen für den A-posteriori-Mittelwert, $a-1$ und $b-1$ die für den A-posteriori-Modus*. Bei der gleichförmigen A-priori-Verteilung $\mathrm{Beta}(1,1)$ geht in den Mittelwert je ein virtueller Wurf mit Kopf und Zahl ein, in den Modus keiner.

<!--en:0e08e91e-->
Kehren wir zu unserer Münze mit gleichförmiger A-priori-Verteilung zurück. Bei 9-mal Kopf in 13 Würfen wird aus $\mathrm{Beta}(1,1)$ die Verteilung $\mathrm{Beta}(10,5)$. Ihr A-posteriori-Mittelwert beträgt $10/15=2/3$. Die klassische Formel dafür heißt *Laplacesche Sukzessionsregel*, $(x+1)/(n+2)$ :cite:`Laplace.1814`. Der Modus $9/13$ stimmt mit dem Maximum-Likelihood-Schätzwert überein, und die Standardabweichung beträgt $\sqrt{(2/3)(1/3)/16}\approx0.118$.

<!--en:4eab466e-->
Beobachten wir hingegen 90-mal Kopf in 130 Würfen, konzentriert sich die A-posteriori-Verteilung stärker. Wir erhalten $\mathrm{Beta}(91,41)$ mit Mittelwert $0.689$ und Standardabweichung $0.040$. Bei der zehnfachen Anzahl von Würfen sinkt die Standardabweichung um den Faktor $2.9\approx\sqrt{133/16}$. Die A-posteriori-Unsicherheit nimmt mit der Rate $1/\sqrt n$ ab. Dieselbe Skalierung finden wir in :numref:`sec_mdl-statistics` für Standardfehler wieder.

<!--en:d17a6c8c-->
Welche Wahrscheinlichkeit sollten wir dafür ansetzen, dass der nächste Wurf Kopf ergibt? Bei gegebenem $\theta$ beträgt sie $\theta$. Mitteln wir über unsere A-posteriori-Verteilung, erhalten wir $P(\text{heads}\mid x)=\int_0^1\theta\,p(\theta\mid x)\,d\theta=\mathbb{E}[\theta\mid x]$. Für unsere Münze beträgt diese Vorhersagewahrscheinlichkeit $2/3$. Die Laplace-Regel liefert also direkt die Wahrscheinlichkeit für Kopf beim nächsten Wurf.

<!--en:9b62746d-->
Dank der Konjugiertheit mussten wir das Normierungsintegral der Bayes-Regel nicht berechnen. Was geschieht bei einer nichtkonjugierten A-priori-Verteilung? In einer Dimension können wir das Produkt aus A-priori-Dichte und Likelihood auf einem feinen Gitter von $\theta$-Werten auswerten und über die Summe normieren. Wählen wir für den natürlichen Parameter eine Gauß-Verteilung als A-priori-Verteilung, $\eta\sim\mathcal N(0,1)$. Eine solche Verteilung verwendet auch :numref:`sec_mdl-bayesian-computation` für die Gewichte der logistischen Regression. Sie drückt die Annahme aus, dass die Münze wahrscheinlich annähernd fair ist. Zur Bernoulli-Likelihood ist sie nicht konjugiert. Ein Variablenwechsel liefert als Dichte bezüglich $\theta$ den Ausdruck $\phi(\eta)/\bigl(\theta(1-\theta)\bigr)$ mit der Standardnormaldichte $\phi$. Dieser Ausdruck ist keine Beta-Dichte. Berechnen wir die A-posteriori-Verteilungen für beide A-priori-Verteilungen auf demselben Gitter und vergleichen wir ihre Kenngrößen.

<!--en:4c3d0be0-->
Für die gleichförmige A-priori-Verteilung liefert das Gitter die exakten Kenngrößen von $\mathrm{Beta}(10,5)$: Mittelwert $0.667$, Standardabweichung $0.118$ und Modus $9/13\approx0.692$. Eine Gauß-Verteilung als A-priori-Verteilung für die Log-Odds verschiebt die A-posteriori-Verteilung in Richtung einer fairen Münze. Der Mittelwert sinkt auf $0.644$, der Modus auf $0.665$ und die Standardabweichung geringfügig auf $0.111$. Mit einem Gitter lässt sich die nichtkonjugierte A-priori-Verteilung genauso leicht behandeln wie die konjugierte. Das gelingt hier nur, weil $\theta$ ein Skalar ist. :numref:`sec_mdl-bayesian-computation` entwickelt Verfahren, die sich auch für hochdimensionale Parametervektoren eignen. :numref:`fig_mdl-prob-beta-posterior` zeigt, wie sich die A-posteriori-Verteilung der Beta-Familie mit zunehmender Anzahl von Würfen konzentriert.

<!--en:cddbce70-->
![Bayessche Aktualisierung der Kopfwahrscheinlichkeit $\theta$. Links die gleichförmige A-priori-Verteilung $\mathrm{Beta}(1,1)$. In der Mitte die A-posteriori-Verteilung $\mathrm{Beta}(10,5)$ nach 9-mal Kopf in 13 Würfen einer Münze mit wahrer Kopfwahrscheinlichkeit $\theta^\ast=0.7$ (gestrichelt). Der Mittelwert beträgt $2/3$, die Standardabweichung $0.12$. Rechts die A-posteriori-Verteilung $\mathrm{Beta}(91,41)$ nach 130 Würfen mit Standardabweichung $0.04$. Bei jeder Aktualisierung addieren sich die beobachteten Kopf- und Zahlwürfe zu den Pseudoanzahlen. Die Breite der A-posteriori-Verteilung nimmt wie $1/\sqrt{n}$ ab.](../img/mdl-prob-beta-posterior.svg)

<!--en:3809838a-->
### Konjugiertheit für jede Exponentialfamilie

<!--en:82c4ff89-->
Auf den ersten Blick wirkt die Aktualisierung der Beta-Verteilung wie ein glücklicher algebraischer Zufall. Tatsächlich ist sie Ausdruck einer allgemeinen Eigenschaft. Jede Exponentialfamilie besitzt eine konjugierte A-priori-Verteilung, und zur Aktualisierung genügt es immer, die suffizienten Statistiken zu addieren.

<!--en:5e3c2112-->
**Proposition (konjugierte A-priori-Verteilung).** *Gegeben seien eine Likelihood :eqref:`eq_mdl-exp_family` und die folgende A-priori-Verteilung für ihre natürlichen Parameter:*

<!--en:1dd6cab9-->
*Dabei seien $\boldsymbol\nu\in\mathbb R^d$ und $\kappa\in\mathbb R$ so gewählt, dass sich die Dichte normieren lässt. Nach $n$ unabhängigen Beobachtungen $\mathbf x_1,\dots,\mathbf x_n$ hat die A-posteriori-Verteilung genau dieselbe Form. Dabei wird $\boldsymbol\nu$ durch $\boldsymbol\nu+\sum_i T(\mathbf x_i)$ und $\kappa$ durch $\kappa+n$ ersetzt.*

<!--en:8873285a-->
**Beweis.** Lassen wir die von $\boldsymbol\eta$ unabhängigen Faktoren weg, so trägt die Likelihood $\exp\bigl(\boldsymbol\eta^\top\sum_i T(\mathbf x_i)-nA(\boldsymbol\eta)\bigr)$ bei. Multiplizieren wir sie mit der A-priori-Dichte, können wir die Exponenten in einer Zeile zusammenfassen:

<!--en:0c68e38d-->
Durch Normieren dieses Produkts erhalten wir die A-posteriori-Verteilung. Sie gehört also derselben Familie an und hat die aktualisierten Parameter $\boldsymbol\nu+\sum_i T(\mathbf x_i)$ und $\kappa+n$. $\blacksquare$

<!--en:7092ed76-->
Die Parameter $\boldsymbol\nu$ und $\kappa$ der A-priori-Verteilung heißen *Hyperparameter* und wirken wie *Pseudodaten*. Dabei steht $\boldsymbol\nu$ für eine vorab angesetzte Summe suffizienter Statistiken und $\kappa$ für einen vorab angesetzten Stichprobenumfang :cite:`Bishop.2006`. Beim Bernoulli-Münzwurf ist diese A-priori-Verteilung für das Logit $\eta=\log\frac{\theta}{1-\theta}$ formuliert. Der Variablenwechsel von $\eta$ zu $\theta$ überführt sie in $\mathrm{Beta}(\nu,\kappa-\nu)$. Die allgemeine Aktualisierung $\nu\to\nu+x$, $\kappa\to\kappa+n$ ergibt dann die Aktualisierung der Beta-Verteilung $a\to a+x$, $b\to b+n-x$. In Aufgabe :exref:`ex_mdl-distributions-conjugate-recipe` führen Sie diesen Variablenwechsel einschließlich der Jacobi-Determinante aus.

<!--en:043d51e3-->
### Gamma- und Poisson-Verteilung, Dirichlet- und Multinomialverteilung

<!--en:7b679fb7-->
Zwei weitere verbreitete konjugierte Paare folgen demselben Muster. Das eine beschreibt Ereignisraten, das andere Wahrscheinlichkeiten für Kategorien wie Wörter oder Klassen. Als Funktion der Rate $\lambda$ ist die Likelihood von $n$ poissonverteilten Anzahlen $x_1,\dots,x_n$ proportional zu $\lambda^{\sum_i x_i}e^{-n\lambda}$. Die zugehörige konjugierte A-priori-Verteilung ist die *Gamma-Verteilung* mit Formparameter $a>0$ und Rate $b>0$. Ihre Dichte ist $\mathrm{Gamma}(\lambda\mid a,b)=\frac{b^a}{\Gamma(a)}\lambda^{a-1}e^{-b\lambda}$ für $\lambda>0$ (mit Mittelwert $a/b$). Das Produkt aus A-priori-Dichte und Likelihood liefert nach Normierung die A-posteriori-Verteilung $\mathrm{Gamma}\bigl(a+\sum_i x_i,\ b+n\bigr)$. Die A-priori-Verteilung steht also für $a$ virtuelle Ereignisse in $b$ virtuellen Zeitabschnitten. In Aufgabe :exref:`ex_mdl-distributions-conjugate-recipe` leiten Sie dieses Paar direkt aus der allgemeinen Proposition her.

<!--en:ef9a8d64-->
Um die Beta-Verteilung als A-priori-Verteilung von zwei Ergebnissen auf $K$ Kategorien zu verallgemeinern, verwenden wir die *Dirichlet-Verteilung*. Sie hat eine Dichte über Wahrscheinlichkeitsvektoren $\mathbf p$ mit $p_k\ge0$ und $\sum_k p_k=1$. Für die Konzentrationsparameter $a_k>0$ lautet ihre Dichte

<!--en:fa121c8f-->
mit Mittelwert $\mathbb{E}[p_k]=a_k/\sum_j a_j$. Für $K=2$ ergibt sich die Beta-Verteilung. Beobachten wir die Kategoriehäufigkeiten $x_1,\dots,x_K$, so ist die Likelihood proportional zu $\prod_k p_k^{x_k}$. Die A-posteriori-Verteilung lautet daher $\mathrm{Dir}(\mathbf a+\mathbf x)$. Für jede Kategorie addieren wir die beobachtete Anzahl zu ihrer Pseudoanzahl.

<!--en:6281e28b-->
Warum ist das in der Praxis wichtig? Angenommen, wir beobachten für ein Vokabular aus drei Wörtern die Anzahlen $(3,0,1)$. Die Maximum-Likelihood-Schätzung $(0.75,0,0.25)$ weist dem unbeobachteten zweiten Wort die Wahrscheinlichkeit null zu. Damit erschiene jedes künftige Dokument, das dieses Wort enthält, als unmöglich. Wählen wir eine gleichförmige A-priori-Verteilung $\mathrm{Dir}(1,1,1)$ für $\mathbf p$, erhalten wir die A-posteriori-Verteilung $\mathrm{Dir}(4,1,2)$ mit Mittelwert $(4/7,1/7,2/7)\approx(0.571,0.143,0.286)$. Die sogenannte *Add-One-Glättung* addiert vor dem Normieren zu jeder Anzahl eins. Sie berechnet also den A-posteriori-Mittelwert unter dieser Dirichlet-Verteilung. Der Naive-Bayes-Klassifikator in :numref:`sec_mdl-naive_bayes` verwendet den Fall mit zwei Klassen.

<!--en:a3d30cc9-->
A-posteriori-Verteilungen in geschlossener Form sind die Ausnahme. Außerhalb konjugierter Paare erfordert die bayessche Inferenz numerische Näherungen (:numref:`sec_mdl-bayesian-computation`). Zunächst brauchen wir jedoch eine Antwort auf eine grundlegendere Frage zu Punktschätzungen. Aus 13 Würfen haben wir $9/13$ erhalten, eine einzelne Zahl ohne Fehlerbalken. Wie weit könnte $9/13$ von der wahren Kopfwahrscheinlichkeit entfernt sein? Unterscheidet sich eine Münze mit 11-mal Kopf in 13 Würfen wirklich von einer mit 9-mal Kopf? :numref:`sec_mdl-statistics` greift diese Fragen auf.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:9552448f-->
Fast alle Verteilungen unserer Übersicht haben die Form $h(\mathbf x)\exp(\boldsymbol\eta^\top T(\mathbf x)-A(\boldsymbol\eta))$ einer Exponentialfamilie. Daten und Parameter sind nur über das innere Produkt $\boldsymbol\eta^\top T(\mathbf x)$ verknüpft. Genau diese Form ergibt sich, wenn wir die Entropie bei festen Mittelwerten der Statistiken maximieren. Die Ableitungen der logarithmischen Zustandssumme $A$ liefern Momente. Dabei ist $\nabla A$ der Mittelwert von $T$ und $\nabla^2 A$ die zugehörige Kovarianz. Damit folgt unmittelbar, dass $A$ konvex ist. Die Maximum-Likelihood-Schätzung ist also ein konvexes Optimierungsproblem. Sie gleicht die Erwartungswerte der suffizienten Statistiken unter dem Modell mit ihren Stichprobenmittelwerten ab und hat eine endliche Lösung, sofern diese Mittelwerte erreichbar sind. Eine Exponentialfamilie mit ihrer kanonischen Linkfunktion liefert ein verallgemeinertes lineares Modell. Sein Gradient ist immer das Produkt aus Eingabe und Differenz zwischen Vorhersage und Beobachtung. Das verbindet lineare, logistische, Softmax- und Poisson-Regression und erklärt, warum sich nichtkanonische Linkfunktionen wie beim Probit-Modell anders verhalten. Die Softmax-Funktion bildet die natürlichen Parameter der Kategorialverteilung auf deren Mittelwert ab. Sie ergibt sich auch aus Gumbel-Rauschen und aus der Boltzmann-Verteilung. Schließlich besitzt jede Exponentialfamilie eine konjugierte A-priori-Verteilung, die wir durch Addition der suffizienten Statistiken aktualisieren. Bei einer Münze liefert $\mathrm{Beta}(a,b)$ die Pseudoanzahlen $a$ und $b$ für den A-posteriori-Mittelwert sowie $a-1$ und $b-1$ für den Modus.

<!--en:2294243a-->
## Übungen

<!--en:a1640575-->
1. **Konkurrierende exponentielle Wartezeiten.** Seien $X_1,\dots,X_m$ unabhängig mit $X_i\sim\mathrm{Exp}(\lambda_i)$, etwa die Wartezeiten bis zum nächsten Ereignis in $m$ unabhängigen Ereignisströmen.
    1. Zeigen Sie, dass $\min_i X_i\sim\mathrm{Exp}\bigl(\sum_i\lambda_i\bigr)$ gilt.
    1. Zeigen Sie, dass der $j$-te Strom mit Wahrscheinlichkeit $P\bigl(X_j=\min_i X_i\bigr)=\lambda_j/\sum_i\lambda_i$ das erste Ereignis liefert.
    1. Seien $V_1,\dots,V_K$ unabhängige $\mathrm{Exp}(1)$-Zufallsvariablen und sei $\mathbf z\in\mathbb R^K$ ein Vektor von Logits. Zeigen Sie mit (b), dass $\operatorname{argmax}_k\,(z_k-\log V_k)$ gemäß den Softmax-Wahrscheinlichkeiten :eqref:`eq_mdl-exp_family_softmax` verteilt ist. Da $-\log V_k$ der standardisierten Gumbel-Verteilung folgt, liefert dies einen zweiten Beweis dafür, dass Gumbel-Rauschen die Softmax-Funktion ergibt.
   :exlabel:`ex_mdl-distributions-exponential-races`

<!--en:7dcf7bc5-->
1. **Maximale Entropie auf den ganzen Zahlen.** Betrachten Sie Verteilungen $(p_0,p_1,p_2,\dots)$ auf $\{0,1,2,\dots\}$ mit festem Mittelwert $m>0$.
    1. Maximieren Sie die Shannon-Entropie $H[p]=-\sum_k p_k\log p_k$ unter den Nebenbedingungen $\sum_k p_k=1$ und $\sum_k k\,p_k=m$ mit Lagrange-Multiplikatoren (:numref:`subsec_mdl-lagrange-multipliers`). Zeigen Sie, dass der stationäre Punkt eine geometrische Verteilung $p_k=(1-r)\,r^k$ ist, und drücken Sie $r$ durch $m$ aus.
    1. Wiederholen Sie die Herleitung für die Entropie relativ zu $h(k)=1/k!$, also $H_h[p]=-\sum_k p_k\log\bigl(p_k/h(k)\bigr)$, und zeigen Sie, dass der stationäre Punkt $\mathrm{Poisson}(m)$ ist.
    1. Berechnen Sie bei $m=4$ die Shannon-Entropie beider Verteilungen numerisch. Welche ist größer und warum muss das so sein? Vergleichen Sie $P(X\ge15)$ für beide Verteilungen. Welche Annahme über große Anzahlen drückt die Wahl des Basismaßes aus?
    1. Ein stationärer Punkt einer Lagrange-Funktion muss kein Maximum sein. Beweisen Sie mit der Proposition über die maximale Entropie, dass die beiden Verteilungen aus (a) und (b) jeweils das globale Maximum liefern. Stellen Sie dazu jede als Mitglied einer Exponentialfamilie dar und zeigen Sie, dass ihr Mittelwert $m$ beträgt.
   :exlabel:`ex_mdl-distributions-max-entropy-integers`

<!--en:e793a596-->
1. **Momente aus der logarithmischen Zustandssumme.** Mit der Momenteneigenschaft :eqref:`eq_mdl-exp_family_moment` und ihrer Form zweiter Ordnung $\nabla^2A=\textrm{Cov}(T)$ lassen sich Momente durch Ableiten berechnen.
    1. Für die Poisson-Verteilung gilt $A(\eta)=e^\eta$. Leiten Sie Mittelwert und Varianz aus $A$ her. Welche Eigenschaft der Funktion $A$ erklärt, dass beide gleich sind?
    1. Die natürlichen Parameter der Gauß-Verteilung sind $\eta_1=\mu/\sigma^2$ und $\eta_2=-1/(2\sigma^2)$. Zeigen Sie, dass $A=-\eta_1^2/(4\eta_2)-\frac12\log(-2\eta_2)$ gilt, und bestimmen Sie durch Ableiten $\mathbb{E}[x]$, $\mathbb{E}[x^2]$, $\textrm{Var}(x)$ und $\textrm{Cov}(x,x^2)$.
    1. Die Laplace-Verteilung mit festem Lageparameter $\mu$ und Skalenparameter $b$ hat die suffiziente Statistik $T(x)=|x-\mu|$. Schreiben Sie die Verteilung in der Form :eqref:`eq_mdl-exp_family` und berechnen Sie mit $A$ die Größen $\mathbb{E}\,|x-\mu|$ und $\textrm{Var}|x-\mu|$. Prüfen Sie das Ergebnis anhand der Varianz $2b^2$ der Laplace-Verteilung.
    1. Stellen Sie für $n$ gaußverteilte Beobachtungen die beiden Gleichungen $\nabla A(\hat{\boldsymbol\eta})=\bar T$ für den Momentenabgleich auf und lösen Sie diese nach $\hat\mu$ und $\hat\sigma^2$ auf. Was geschieht für $n=1$? Erläutern Sie den Zusammenhang mit der Bedingung, dass $\bar T$ ein erreichbarer Mittelwert sein muss.
   :exlabel:`ex_mdl-distributions-log-partition-moments`

<!--en:b67b2ede-->
1. [code] **Probit und Logit.** Für zwei Klassen mit Score-Differenz $m=z_1-z_2$ liefert das Logit-Modell $P(y=1)=\operatorname{sigmoid}(m)$. Das Probit-Modell mit Rauschskala $s$ liefert $P(y=1)=\Phi\bigl(m/(\sqrt2\,s)\bigr)$.
    1. Zeigen Sie, dass die Differenz zweier unabhängiger standardisierter Gumbel-Zufallsvariablen die Verteilungsfunktion $\operatorname{sigmoid}(x)$ hat. Die standardisierte Gumbel-Verteilung hat die Varianz $\pi^2/6$. Bestimmen Sie den Skalenparameter $c$, für den die Verteilung mit Verteilungsfunktion $\operatorname{sigmoid}(cx)$ wie die Standardnormalverteilung die Varianz 1 hat.
    1. Bestimmen Sie numerisch den Skalenparameter $c$, der $\max_x|\operatorname{sigmoid}(cx)-\Phi(x)|$ minimiert, sowie den Wert dieses Maximums. Warum ist dieser Skalenparameter kleiner als der aus (a)?
    1. Ein Beispiel der Klasse 2 liegt mit $m>0$ auf der falschen Seite der Entscheidungsgrenze. Zeigen Sie, dass seine negative Log-Likelihood $-\log\operatorname{sigmoid}(-m)$ im Logit-Modell zwischen $m$ und $m+e^{-m}$ liegt und ihre Ableitung nach $m$ kleiner als 1 ist. Zeigen Sie mit den Schranken $\frac{x}{1+x^2}\,\phi(x)\le1-\Phi(x)\le\frac{\phi(x)}{x}$ für $x>0$, dass seine negative Log-Likelihood $-\log\Phi\bigl(-m/(\sqrt2\,s)\bigr)$ im Probit-Modell wie $m^2/(4s^2)$ wächst und ihre Ableitung linear in $m$ wächst.
    1. Ziehen Sie 100 Eingaben $x\sim\mathcal N(1,1)$ für Klasse 1 und 100 Eingaben $x\sim\mathcal N(-1,1)$ für Klasse 2 und setzen Sie $m=wx+b$. Schätzen Sie $(w,b)$ für beide Modelle mit der Maximum-Likelihood-Methode. Fügen Sie dann drei Beispiele bei $x=6$ mit dem Label der Klasse 2 hinzu und wiederholen Sie die Schätzung. Berechnen Sie für jedes Modell das Verhältnis der Steigung $w$ mit den falsch beschrifteten Beispielen zur Steigung ohne diese Beispiele. Erklären Sie den Unterschied zwischen den Modellen anhand von (c). Berechnen Sie $\log\Phi$ mit einem numerisch stabilen Verfahren wie `scipy.special.log_ndtr`.
   :exlabel:`ex_mdl-distributions-probit-logit`

<!--en:51ae1622-->
1. **Minimale Kategorialfamilie.** ● Kodieren Sie ein kategoriales Ergebnis als One-Hot-Vektor $\mathbf x\in\{0,1\}^K$ mit Klassenwahrscheinlichkeiten $\mathbf p=\mathrm{softmax}(\mathbf z)$. Schreiben Sie die Verteilung als Exponentialfamilie mit $T(\mathbf x)=\mathbf x$, natürlichen Parametern $\mathbf z$ und $A(\mathbf z)=\log\sum_ke^{z_k}$.
    1. Berechnen Sie $\nabla^2 A(\mathbf z)$ und zeigen Sie, dass das Ergebnis die Kovarianz $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ des One-Hot-Vektors $\mathbf x$ ist. Jeder One-Hot-Vektor erfüllt $\mathbf x\mathbf x^\top=\operatorname{diag}(\mathbf x)$. Zeigen Sie, dass $\nabla^2A(\mathbf z)\,\mathbf 1=\mathbf 0$ gilt und dass der Gradient der negativen Log-Likelihood eines Datensatzes orthogonal zu $\mathbf 1$ ist, und zwar für jedes $\mathbf z$. Ein Gradientenabstieg mit kleiner Schrittweite beginnt bei $\mathbf z_0$ und aktualisiert die Logits. Jede Klasse kommt in den Daten vor. Welche Funktion von $\mathbf z$ bleibt beim Gradientenabstieg konstant? Welcher Minimalstelle nähert er sich daher?
    1. Setzen Sie $z_K=0$. Damit erhalten Sie eine minimale Darstellung mit $\boldsymbol\eta=(z_1,\dots,z_{K-1})$ und $T(\mathbf x)=(x_1,\dots,x_{K-1})$. Zeigen Sie, dass $\nabla^2A(\boldsymbol\eta)$ positiv definit ist, wenn für jede Klasse $p_k>0$ gilt. Lösen Sie bei gegebenen Klassenhäufigkeiten $n_1,\dots,n_K$ die Bedingung $\nabla A(\hat{\boldsymbol\eta})=\bar T$ für den Momentenabgleich nach dem Maximum-Likelihood-Schätzer $\hat{\boldsymbol\eta}$ auf. Was geschieht, wenn für eine Klasse $n_k=0$ gilt?
   :exlabel:`ex_mdl-distributions-minimal-categorical`

<!--en:194c095c-->
1. [code] **Poisson-Regression.** ● Ein verallgemeinertes lineares Modell für Zähldaten drückt den natürlichen Parameter einer Poisson-Verteilung als affine Funktion der Merkmale aus: $y_i\mid\mathbf x_i\sim\mathrm{Poisson}(e^{\eta_i})$ mit $\eta_i=\mathbf w^\top\mathbf x_i+b$.
    1. Schreiben Sie den Verlust mithilfe der negativen Log-Likelihood :eqref:`eq_mdl-exp_family_nll` als Funktion von $(\mathbf w,b)$ und berechnen Sie den Gradienten und die Hesse-Matrix. Zeigen Sie, dass der Verlust konvex ist. Welche Gleichungen müssen die angepassten Mittelwerte $\hat m_i=e^{\hat\eta_i}$ erfüllen, wenn der Gradient verschwindet?
    1. Erzeugen Sie $n=1000$ Beispiele mit $\mathbf x_i\sim\mathcal N(\mathbf 0,\mathbf I_2)$, $\mathbf w=(-0.3,1.0)$ und $b=0.5$. Schätzen Sie $\mathbf u=(\mathbf w,b)$ mit dem Newton-Verfahren, beginnend bei null. Wiederholen Sie dazu den Schritt $\mathbf u\leftarrow\mathbf u-\bigl(\nabla^2L(\mathbf u)\bigr)^{-1}\nabla L(\mathbf u)$ für den Verlust $L$. Prüfen Sie die Gleichungen aus (a) an der Lösung.
    1. Multiplizieren Sie jede Rate mit einem unabhängigen Faktor $g_i\sim\mathrm{Gamma}(1,1)$ mit Mittelwert $1$. So weisen die Anzahlen Überdispersion auf, während ihre bedingten Mittelwerte weiterhin $e^{\eta_i}$ betragen. Passen Sie das Poisson-Modell an $200$ solche Datensätze und an $200$ Datensätze ohne diesen Faktor an. Vergleichen Sie Mittelwert und Standardabweichung der Schätzwerte über die Datensätze hinweg. Erklären Sie das Verhalten des Mittelwerts mit den Gleichungen aus (a).
   :exlabel:`ex_mdl-distributions-poisson-regression`

<!--en:a682c82f-->
1. **Konjugierte A-priori-Verteilungen aus natürlichen Parametern.** ● Für einen skalaren natürlichen Parameter $\eta$ lautet die konjugierte A-priori-Verteilung :eqref:`eq_mdl-exp_family_conjugate` in expliziter Form $p(\eta)\propto\exp\bigl(\nu\,\eta-\kappa\,A(\eta)\bigr)$. Nach $n$ Beobachtungen aktualisieren wir sie durch $\nu\to\nu+\sum_i T(x_i)$ und $\kappa\to\kappa+n$.
    1. Für die Poisson-Verteilung gilt $\eta=\log\lambda$ und $A(\eta)=e^\eta$. Wechseln Sie von $\eta$ zu $\lambda$ und berücksichtigen Sie die Jacobi-Determinante. Zeigen Sie, dass die A-priori-Verteilung für $\lambda$ eine Gamma-Verteilung ist. Bestimmen Sie Formparameter und Rate. Prüfen Sie, dass die Aktualisierung die A-posteriori-Verteilung der Gamma-Familie aus :numref:`subsec_mdl-other-conjugate-pairs` ergibt.
    1. Für die Bernoulli-Verteilung gilt $\eta=\log\frac{\theta}{1-\theta}$ und $A(\eta)=\log(1+e^\eta)$. Zeigen Sie, dass die A-priori-Verteilung für $\theta$ die Form $\mathrm{Beta}(\nu,\kappa-\nu)$ hat. Bestimmen Sie, für welche Werte von $(\nu,\kappa)$ sie normierbar ist.
    1. Wählen Sie die konstante A-priori-Dichte für das Logit, $\nu=\kappa=0$. Welche Dichte induziert sie auf $\theta$? Ist diese normierbar? Wann ist die A-posteriori-Verteilung nach $x$-mal Kopf in $n$ Würfen normierbar und welchen Mittelwert hat sie? Vergleichen Sie ihn mit dem Mittelwert der A-posteriori-Verteilung :eqref:`eq_mdl-beta_posterior` bei gleichförmiger A-priori-Verteilung $\mathrm{Beta}(1,1)$ auf $\theta$. Was sagt der Vergleich über den Ausdruck „konstante A-priori-Dichte“ aus?
   :exlabel:`ex_mdl-distributions-conjugate-recipe`

<!--en:2a77e68d-->
[Diskussion](https://d2l.discourse.group/t/36862)

<!--en:c3ce7f3d-->
Die Exponentialfamilie: suffiziente Statistiken, Kumulanten, maximale Entropie und Konjugiertheit<br>**Kanonische Form $h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$ · maximale Entropie · $\nabla A = \mathbb{E}[T]$ und $\nabla^2 A = \mathrm{Cov}(T)$ · GLMs · Ursprung der Softmax-Funktion · konjugierte A-priori-Verteilungen**

<!--en:75ed21f2-->
::: {.slide title="Vier Regressionen, vier Verlustfunktionen, ein Gradient"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:07628c5b-->
- **Eine gemeinsame Trainingsregel:** Lineare, logistische, Softmax- und Poisson-Regression haben dieselbe Gradientenformel, weil ihre Zielverteilungen zur **Exponentialfamilie** gehören.

<!--en:0a86fbf5-->
- **Vier Ergebnisse der kanonischen Form:** Kumulanten aus Ableitungen ($\nabla A = \mathbb{E}[T]$, $\nabla^2 A = \mathrm{Cov}(T)$), konvexe Maximum-Likelihood-Schätzung durch Momentenabgleich, maximale Entropie und konjugierte A-priori-Verteilungen mit additiver Aktualisierung.

<!--en:e5f2fc89-->
::: {.slide title="Die Bernoulli-Verteilung in Exponentialform"}

<!--en:734253f8-->
[Wie Log-Odds und Softplus entstehen]{.kicker}

<!--en:803e3459-->
- **Natürlicher Parameter (Logit):** $\eta = \log\frac{p}{1-p} \in \mathbb{R}$. Die logistische Sigmoidfunktion $p = \mathrm{sigmoid}(\eta) = (1 + e^{-\eta})^{-1}$ kehrt die Abbildung um.

<!--en:4d7d6788-->
- **Logarithmische Zustandssumme (Softplus):** $A(\eta) = -\log(1-p) = \log(1 + e^\eta)$ normiert $P(X = x) = \exp(\eta x - A(\eta))$.

<!--en:c7a6e133-->
::: {.slide title="Die kanonische Form der Exponentialfamilie"}

<!--en:a89b98d4-->
[Allgemeine Definition und suffiziente Statistiken]{.kicker}

<!--en:63a52229-->
- **Vier Bestandteile:** parameterfreies Basismaß $h(\mathbf{x}) \ge 0$, suffiziente Statistik $T(\mathbf{x}) \in \mathbb{R}^d$, natürlicher Parameter $\boldsymbol{\eta} \in \mathbb{R}^d$ und logarithmische Zustandssumme $A(\boldsymbol{\eta}) = \log \int h(\mathbf{x})e^{\boldsymbol{\eta}^\top T(\mathbf{x})}\,d\mathbf{x}$ zur Normierung.

<!--en:36fd2bb8-->
- **Warum $T(\mathbf{x})$ suffizient ist:** Bei $n$ unabhängigen, identisch verteilten Ziehungen hängt $\boldsymbol{\eta}$ mit dem Datensatz nur über die Summe $\sum_{i=1}^n T(\mathbf{x}_i)$ fester Dimension zusammen.

<!--en:b9c0432b-->
::: {.slide title="Mitglieder der Exponentialfamilie"}

<!--en:fed34223-->
[Eine Form für diskrete und stetige Verteilungen]{.kicker}

<!--en:cf9c1f3a-->
Jede Zeile stellt $p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$ dar:

<!--en:44cc99cc-->
| Verteilung | Basismaß $h(\mathbf{x})$ | Suffiziente Statistik $T(\mathbf{x})$ | Natürlicher Parameter $\boldsymbol{\eta}$ | Logarithmische Zustandssumme $A(\boldsymbol{\eta})$ |

<!--en:6e12c38f-->
| **$\mathrm{Binomial}(n, p)$** ($n$ fest) | $\binom{n}{x}$ | $x \in \{0, \dots, n\}$ | $\log\frac{p}{1-p}$ | $n\log(1 + e^\eta)$ |

<!--en:d6b14db7-->
| **$\mathrm{Categorical}(\mathbf{p})$** | $1$ | One-Hot-Vektor $\mathbf{y} \in \{0, 1\}^K$ | Logits $\mathbf{z} \in \mathbb{R}^K$ | $\log \sum_{k=1}^K e^{z_k}$ |

<!--en:2ae19f18-->
| **$\mathrm{Laplace}(\mu, b)$** ($\mu$ fest) | $1$ | $\|x - \mu\|$ | $-1/b < 0$ | $\log(-2/\eta)$ |

<!--en:5bf456a9-->
::: {.slide title="Welche Verteilungen liegen außerhalb der Exponentialfamilie?"}

<!--en:dbc1f862-->
[Veränderlicher Träger und fehlende Konvexität bei schweren Rändern]{.kicker}

<!--en:de238aa4-->
- **Veränderlicher Träger ($\mathrm{Unif}(0, \theta)$):** Der Träger $[0, \theta]$ hängt von $\theta$ ab. Der Maximum-Likelihood-Schätzer ist daher das Stichprobenmaximum $\hat{\theta}_{\mathrm{MLE}} = \max_i x_i$. Ein Abgleich der Mittelwerte liefert ihn hier nicht.

<!--en:bdb460e7-->
- **Algebraisch abfallende schwere Ränder (Student-$t_\nu(\mu)$-Verteilung):** Das Produkt $\prod_{i=1}^n \bigl(1 + (x_i-\mu)^2/\nu\bigr)^{-(\nu+1)/2}$ besitzt keine suffiziente Statistik fester Dimension (Satz von Pitman, Koopman und Darmois). Seine negative Log-Likelihood ist **nicht konvex** und kann mehrere lokale Minima haben.

<!--en:aaa0ffca-->
::: {.slide title="Jaynes’ gezinkter Würfel und die maximale Entropie"}

<!--en:656a1723-->
[Woher die Exponentialform kommt]{.kicker}

<!--en:8c2e5588-->
- Die **Shannon-Entropie** misst die mittlere Überraschung (in Nat):
  $$H[p] = -\sum_{k=1}^K p_k \log p_k \quad (\text{fair die: } H = \log 6 \approx 1.792)$$

<!--en:aab796fa-->
- **Jaynes’ Aufgabe:** Ein 6-seitiger Würfel zeigt im Mittel $\mathbb{E}[k] = 4.5$ statt $3.5$. Die Masse $1/2$ auf jeder Augenzahl in $\{3, 6\}$ ergibt den Mittelwert $4.5$. Das setzt voraus, dass die Augenzahlen $\{1, 2, 4, 5\}$ unmöglich sind ($H = \log 2 \approx 0.693$).

<!--en:eb855354-->
- **Prinzip der maximalen Entropie:** Die Verteilung mit $\mathbb{E}[k] = 4.5$, die am wenigsten voraussetzt, entsteht durch **exponentielle Kippung**:
  $$p_k = \frac{e^{\eta k}}{\sum_{j=1}^6 e^{\eta j}}, \quad \eta \approx 0.371 \quad (H \approx 1.614)$$

<!--en:97aff252-->
::: {.slide title="Exponentielle Kippung für maximale Entropie durch Bisektion"}

<!--en:5edd62b5-->
[Monotonie des Mittelwerts der Statistik]{.kicker}

<!--en:bbd439f7-->
Die mittlere Augenzahl $\sum_k k\,p_k(\eta)$ wächst streng monoton mit $\eta$ (Ableitung $\mathrm{Var}_\eta(k) > 0$). Daher liefern 50 Bisektionsschritte $\eta \approx 0.371$:

<!--en:a21bdc60-->
**Kleinster Entropieverlust:** Die exponentielle Kippung kostet nur $\log 6 - 1.614 \approx 0.18\text{ nats}$ an Entropie, um den Mittelwert von $3.5$ auf $4.5$ zu verschieben. Die Zweipunktverteilung auf den beiden Augenzahlen verliert dabei $1.10\text{ nats}$ (Träger $\{3, 6\}$).

<!--en:ff20300f-->
::: {.slide title="Gibbs-Ungleichung: Die KL-Divergenz ist nichtnegativ"}

<!--en:95148b1d-->
[Eine Tangente beweist globale Optimalität]{.kicker}

<!--en:ad550695-->
- **Beweis mit der Tangente ($\log u \le u - 1$):** Einsetzen von $u = p(\mathbf{x})/q(\mathbf{x})$ ergibt $-D_{\mathrm{KL}}(q \| p) \le \sum_{q>0} q(\mathbf{x})\bigl(\frac{p(\mathbf{x})}{q(\mathbf{x})} - 1\bigr) = \sum_{q>0} p(\mathbf{x}) - 1 \le 0$.

<!--en:fe5c60d1-->
- **Eindeutigkeit:** Gleichheit $\log u = u - 1$ gilt nur bei $u = 1$. Daraus folgt überall $p(\mathbf{x}) = q(\mathbf{x})$.

<!--en:1d938266-->
::: {.slide title="Die Proposition über die maximale Entropie und ihre Korollare"}

<!--en:0a73a7e9-->
[Warum Verteilungen maximaler Entropie bei vorgegebenen Momenten zur Exponentialfamilie gehören]{.kicker}

<!--en:b1997ea0-->
Die Entropie relativ zum Basismaß $h(\mathbf{x})$ ist definiert als $H_h[q] = -\sum_{\mathbf{x}} q(\mathbf{x})\log\frac{q(\mathbf{x})}{h(\mathbf{x})}$.

<!--en:12a9ed25-->
**Proposition (maximale Entropie):** Sei $p(\mathbf{x}) = h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$ und sei $q$ eine *beliebige* Verteilung mit denselben Mittelwerten der Statistiken, $\mathbb{E}_q[T(\mathbf{x})] = \mathbb{E}_p[T(\mathbf{x})]$. Dann gilt:

<!--en:fda14fd8-->
- **Beweis in zwei Zeilen:** $\log\frac{q}{h} = \log\frac{q}{p} + \log\frac{p}{h}$ zerlegen. Die erste Summe ist $-D_{\mathrm{KL}}(q \| p) \le 0$. Da $\log\frac{p}{h} = \boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})$ linear in $T$ ist, hängt die zweite Summe von $q$ nur über den vorgegebenen Mittelwert ab: $-\boldsymbol{\eta}^\top\mathbb{E}_q[T] + A(\boldsymbol{\eta}) = -\boldsymbol{\eta}^\top\mathbb{E}_p[T] + A(\boldsymbol{\eta}) = H_h[p]$. $\blacksquare$

<!--en:c635a798-->
- **Bekannte Folgerungen:**
  - **Keine Nebenbedingung ($T \equiv 0$, $h \equiv 1$ auf $K$ Ergebnissen):** Gleichverteilung ($H = \log K$).
  - **Fester Mittelwert $\mu$ und feste Varianz $\sigma^2$ auf $\mathbb{R}$ ($T(x) = (x, x^2)$, $h \equiv 1$):** Gauß-Verteilung $\mathcal{N}(\mu, \sigma^2)$.
  - **Fester Mittelwert $m$ auf $\{0, 1, 2, \dots\}$ ($T(k) = k$):** $\mathrm{Poisson}(m)$ für $h(k) = 1/k!$ oder $\mathrm{Geometric}$ für $h(k) = 1$.

<!--en:4e96a6de-->
::: {.slide title="Die Momenteneigenschaft: Ableitungen von A(η)"}

<!--en:41fb24d5-->
[Kumulanten und Konvexität]{.kicker}

<!--en:c1912d00-->
- **Momente durch Ableiten:** Die erste Ableitung von $A(\boldsymbol{\eta}) = \log \int h(\mathbf{x})e^{\boldsymbol{\eta}^\top T(\mathbf{x})}\,d\mathbf{x}$ liefert den Mittelwert von $T(\mathbf{x})$, die zweite die zugehörige Kovarianzmatrix.

<!--en:c930c4ce-->
- **Konvexität in allen Fällen:** Jede Kovarianzmatrix ist positiv semidefinit. Daher sind $A(\boldsymbol{\eta})$ und die negative Log-Likelihood **immer konvex** in $\boldsymbol{\eta}$.

<!--en:3b5cd594-->
::: {.slide title="Momente durch automatisches Differenzieren berechnen"}

<!--en:53c763aa-->
[Analytische Prüfung und Vergleich mit Autograd]{.kicker}

<!--en:cd4e1500-->
- **Bernoulli ($A(\eta) = \log(1 + e^\eta)$):** $A'(\eta) = \mathrm{sigmoid}(\eta) = p$ und $A''(\eta) = \mathrm{sigmoid}(\eta)(1 - \mathrm{sigmoid}(\eta)) = p(1 - p)$.

<!--en:77a2b94a-->
- **Poisson ($A(\eta) = e^\eta$ mit $\eta = \log\lambda$):** Da $e^\eta$ seine eigene Ableitung ist, gilt $A'(\eta) = A''(\eta) = e^\eta = \lambda$. Das erklärt in einer Zeile, warum Mittelwert und Varianz der Poisson-Verteilung übereinstimmen.

<!--en:1e4d1412-->
::: {.slide title="Momente durch automatisches Differenzieren berechnen (Fortsetzung)"}

<!--en:53c763aa-2-->
[Analytische Prüfung und Vergleich mit Autograd]{.kicker}

<!--en:61ce9ed5-->
::: {.slide title="Maximum-Likelihood-Schätzung als konvexer Momentenabgleich"}

<!--en:27e361ad-->
[Globale Optimierung der Log-Likelihood]{.kicker}

<!--en:db81be44-->
- Für $n$ unabhängige, identisch verteilte Beobachtungen $\mathbf{x}_1, \dots, \mathbf{x}_n$ ist die negative Log-Likelihood (NLL) konvex in $\boldsymbol{\eta}$:
  $$\mathcal{L}(\boldsymbol{\eta}) = n\,A(\boldsymbol{\eta}) - \boldsymbol{\eta}^\top \sum_{i=1}^n T(\mathbf{x}_i) - \sum_{i=1}^n \log h(\mathbf{x}_i)$$

<!--en:c7596c75-->
- **Optimalitätsbedingung für den Momentenabgleich ($\nabla_{\boldsymbol{\eta}} \mathcal{L} = \mathbf{0}$):**
  $$\nabla A(\hat{\boldsymbol{\eta}}) = \bar{T} = \frac{1}{n}\sum_{i=1}^n T(\mathbf{x}_i)$$

<!--en:05776037-->
- **Münze (Kopfanteil $9/13$):** $\mathrm{sigmoid}(\hat{\eta}) = \frac{9}{13} \implies \hat{\eta} = \log\frac{9}{4} \approx 0.811$. Bei $\hat{\eta}$ entspricht die mittlere NLL der Entropie der angepassten Münzverteilung ($0.617\text{ nats/flip}$).

<!--en:01bf27d6-->
::: {.slide title="Divergenz am Rand und redundante Logits"}

<!--en:51edc476-->
[Wann der Momentenabgleich keine endliche oder keine eindeutige Lösung hat]{.kicker}

<!--en:9b86decc-->
- **Unerreichbare Mittelwerte am Rand (Kopfanteil $13/13$):** Der Wertebereich in $\mathrm{sigmoid}(\eta) \in (0, 1)$ ist offen. Für $\bar{T} = 1$ hat der Momentenabgleich daher keine endliche Lösung ($\hat{\eta} \to +\infty$, wie bei linear trennbaren Daten in der logistischen Regression ohne Gewichtszerfall).

<!--en:cd60726f-->
- **Nichtminimale Softmax-Logits:** Die One-Hot-Koordinaten summieren sich zu $\mathbf{1}^\top \mathbf{y} = 1$. Eine Verschiebung $\mathbf{z} \to \mathbf{z} + c\mathbf{1}$ lässt daher die Wahrscheinlichkeiten unverändert. Das nutzt `logsumexp` ($c = \max_k z_k$), um einen Überlauf zu verhindern.

<!--en:a9728f43-->
::: {.slide title="Verallgemeinerte lineare Modelle und kanonische Linkfunktionen"}

<!--en:62f2d42e-->
[Warum der Gradient jedes Standardverlusts gleich Eingabe × (Vorhersage − Zielwert) ist]{.kicker}

<!--en:f87ef7f3-->
- **Kanonische Linkfunktion ($\boldsymbol{\eta} = \mathbf{W}^\top\mathbf{x}$):** Die Kettenregel für $\nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \boldsymbol{\mu}(\mathbf{x})$ liefert für jede Verlustfunktion einer Exponentialfamilie das Produkt aus Eingabe und Vorhersagefehler.

<!--en:7c5982ff-->
- **Vier GLM-Ausgabeschichten:** Identität $\eta$ (lineare Regression), $\mathrm{sigmoid}(\eta)$ (logistische Regression), $\mathrm{softmax}(\boldsymbol{\eta})$ (mehrere Klassen) und $e^\eta$ (Poisson-Regression für Zähldaten).

<!--en:a589a4e3-->
::: {.slide title="Ursprung der Softmax-Funktion: verrauschter Nutzen und Gumbel-Max"}

<!--en:5ecb751d-->
[Diskrete Wahlmodelle und exakte Ziehungen aus der Kategorialverteilung]{.kicker}

<!--en:c0eb245f-->
Angenommen, die Entscheidung fällt auf die Klasse mit dem höchsten verrauschten Score $y = \operatorname{argmax}_{k \in \{1,\dots,K\}}(z_k + \epsilon_k)$:

<!--en:6ea75f09-->
- **Warum Gauß-Rauschen ($\epsilon_k \sim \mathcal{N}(0, s^2)$) für $K > 2$ Schwierigkeiten macht:** Das **Probit-Modell** $P(y=1) = \Phi\!\left(\frac{z_1 - z_2}{\sqrt{2}s}\right)$ erfordert ein $(K-1)$-dimensionales numerisches Integral ohne geschlossene Form, sobald $K \ge 3$ gilt.

<!--en:76138d81-->
- **Proposition (Gumbel-Rauschen liefert die Softmax-Funktion):** Sind $\epsilon_1, \dots, \epsilon_K$ unabhängige, identisch verteilte **Gumbel**-Standardvariablen mit Verteilungsfunktion $G(t) = \exp(-e^{-t})$, so hat $z_j + \epsilon_j$ die Verteilungsfunktion $\exp(-e^{z_j}e^{-t})$. Das Integral über den höchsten Score $t = z_k + \epsilon_k$ lässt sich mit $S = \sum_{j=1}^K e^{z_j}$ und $v = e^{-t}$ analytisch auswerten:
  $$P\Bigl(\operatorname*{argmax}_j (z_j + \epsilon_j) = k\Bigr) = \int_{-\infty}^{\infty} e^{z_k}e^{-t}\exp(-Se^{-t})\,dt = e^{z_k}\int_0^{\infty} e^{-Sv}\,dv = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

<!--en:1c51e145-->
- **Log-Sum-Exp als glattes Maximum:** Für den höchsten Score selbst gilt $\max_j(z_j + \epsilon_j) \sim \mathrm{Gumbel}(\log \sum_j e^{z_j})$. Der Gradient des Log-Sum-Exp-Werts bezüglich $\mathbf{z}$ ist der von der Softmax-Funktion gelieferte Vektor.

<!--en:8e1ea261-->
::: {.slide title="Der Gumbel-Max-Trick und die Temperaturskalierung"}

<!--en:57e1ad1c-->
[Drei Sichtweisen auf die Softmax-Funktion]{.kicker}

<!--en:269a405a-->
- **1. Gumbel-Max-Verfahren:** Da $\operatorname{argmax}_k(z_k + T\epsilon_k) = \operatorname{argmax}_k(z_k/T + \epsilon_k)$ gilt, genügt es, Gumbel-Rauschen mit Skalenparameter $T > 0$ zu addieren und das Argmax zu bestimmen. So erhalten wir exakte Ziehungen aus $\mathrm{softmax}(\mathbf{z}/T)$.

<!--en:7e204b39-->
- **2. Statistische Physik (Boltzmann-Gesetz):** Bei Temperatur $T$ und Zustandsenergie $E_k = -z_k$ gilt im thermischen Gleichgewicht $p_k \propto e^{-E_k/T} = e^{z_k/T}$. Das ist auch die Verteilung maximaler Entropie bei festem Logit-Erwartungswert $\sum_k p_k z_k$.

<!--en:9a5c56e1-->
- **3. Verhalten von Probit und Logit an den Rändern:** Zwar gilt $|\mathrm{sigmoid}(1.7x) - \Phi(x)| < 0.01$, doch bei einem Ausreißer mit falschem Label ($m \gg 0$) wächst der Logit-Verlust linear ($\sim m$) und der Probit-Verlust quadratisch ($\sim m^2/4s^2$).

<!--en:12909654-->
::: {.slide title="Der Gumbel-Max-Trick und die Temperaturskalierung (Fortsetzung)"}

<!--en:57e1ad1c-2-->
[Drei Sichtweisen auf die Softmax-Funktion]{.kicker}

<!--en:8d5ec2cc-->
Prüfung des Gumbel-Max-Verfahrens für $\mathbf{z} = (2, 1, 0.1, -1)$ mit $10^5$ Versuchen bei $T = 1$ und $T = 2$:

<!--en:2e0f165e-->
::: {.slide title="Bayessche Aktualisierung und die konjugierte Beta-Verteilung"}

<!--en:b1539098-->
[Von Punktschätzungen zu A-posteriori-Verteilungen]{.kicker}

<!--en:17bd9033-->
- **Additive Pseudoanzahlen:** Die A-priori-Parameter $(a, b)$ zählen virtuelle Würfe mit Kopf und Zahl. Sie addieren sich direkt zu den beobachteten Anzahlen $(x, n-x)$.

<!--en:8399a6e0-->
- **Mittelwert und Modus:** Der A-posteriori-Mittelwert $\frac{a+x}{a+b+n}$ verwendet die Pseudoanzahlen $(a, b)$ (die Laplace-Regel $\frac{x+1}{n+2} = \frac{2}{3}$ ergibt sich für $\mathrm{Beta}(1,1)$). Der Modus $\frac{a+x-1}{a+b+n-2}$ verwendet $(a-1, b-1)$.

<!--en:c0f2cdd9-->
::: {.slide title="Konzentration der A-posteriori-Verteilung und Gitterrechnung bei fehlender Konjugiertheit"}

<!--en:fdd53a3b-->
[Analytische Aktualisierung der Beta-Verteilung und eindimensionale numerische Integration]{.kicker}

<!--en:d506b15b-->
- **Konzentration mit Rate $1/\sqrt{n}$:** Ausgehend von $\mathrm{Beta}(1, 1)$ ergibt ein Kopfanteil von $9/13$ die Verteilung $\mathrm{Beta}(10, 5)$ ($\mathrm{sd} \approx 0.118$). Bei einem Kopfanteil von $90/130$ ergibt sich $\mathrm{Beta}(91, 41)$ ($\mathrm{sd} \approx 0.040$, um den Faktor $\sqrt{133/16} \approx 2.9\times$ kleiner).

<!--en:4d084765-->
- **Nichtkonjugierte A-priori-Verteilung ($\eta \sim \mathcal{N}(0, 1)$ für die Log-Odds):** Nach dem Variablenwechsel ist $p(\theta) = \frac{\phi(\eta)}{\theta(1-\theta)}$ keine Beta-Dichte. Auf einem feinen eindimensionalen Gitter lässt sich $p(\theta)P(x\mid\theta)$ direkt normieren:

<!--en:7801c83b-->
::: {.slide title="Allgemeine Konjugiertheit und Dirichlet-Glättung"}

<!--en:2fbb70db-->
[A-priori-Verteilungen für beliebige Exponentialfamilien]{.kicker}

<!--en:988f871a-->
**Proposition (allgemeine konjugierte A-priori-Verteilung):** Zu jeder Likelihood $p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$ einer Exponentialfamilie gibt es eine konjugierte A-priori-Verteilung. Ihre Parameter sind die Summe der Pseudostatistiken $\boldsymbol{\nu} \in \mathbb{R}^d$ und der Umfang der Pseudostichprobe $\kappa > 0$:

<!--en:8906cc36-->
| Konjugiertes Paar | A-priori-Dichte | Beobachtete Daten ($n$ Ziehungen) | A-posteriori-Aktualisierung | Anwendung im maschinellen Lernen |

<!--en:7f97e97a-->
| **Beta–Binomial** | $\mathrm{Beta}(\theta \mid a, b)$ | $x$-mal Kopf, $n - x$-mal Zahl | $\mathrm{Beta}(a + x,\ b + n - x)$ | Genauigkeit binärer Klassifikation und Klickraten in A/B-Tests |

<!--en:fcf0ad5f-->
| **Gamma–Poisson** | $\mathrm{Gamma}(\lambda \mid a, b)$ | Gesamtanzahl $\sum_{i=1}^n x_i$ | $\mathrm{Gamma}(a + \sum_i x_i,\ b + n)$ | Ankunftsrate schätzen ($a$ Ereignisse in $b$ Zeitfenstern) |

<!--en:3aae0783-->
| **Dirichlet–Multinomial** | $\mathrm{Dir}(\mathbf{p} \mid \mathbf{a}) \propto \prod_{k=1}^K p_k^{a_k - 1}$ | Vektor der Klassenhäufigkeiten $\mathbf{x}$ | $\mathrm{Dir}(\mathbf{a} + \mathbf{x})$ | **Add-One-Glättung:** Anzahlen $(3,0,1)$ mit $\mathrm{Dir}(\mathbf{1}) \to$, Mittelwert $(4/7, 1/7, 2/7) \ne 0$ |

<!--en:19d0e625-->
::: {.slide title="Verständnisfragen zu Kumulanten, GLMs und Konjugiertheit"}

<!--en:929b7671-->
[Zur Selbstkontrolle]{.kicker}

<!--en:cad67d0d-->
Drei Aufgaben zur Exponentialfamilie:

<!--en:f7aba96d-->
1. **Momente aus der logarithmischen Zustandssumme:** Die Exponentialverteilung $\mathrm{Exp}(\lambda)$ hat $T(x) = x$, den natürlichen Parameter $\eta = -\lambda < 0$ und $A(\eta) = -\log(-\eta)$. $A'(\eta)$ und $A''(\eta)$ berechnen, um $\mathbb{E}[X]$ und $\mathrm{Var}(X)$ zu erhalten.

<!--en:dc5fd1c0-->
2. **Gradientenschritt im Poisson-GLM:** Eine Poisson-Regression mit kanonischer Linkfunktion ($\hat{\lambda} = e^\eta$ mit $\eta = \mathbf{w}^\top\mathbf{x}$) sagt $\hat{\lambda} = 3$ für die Eingabe $\mathbf{x}$ vorher. Die beobachtete Anzahl ist $y = 5$. Wie lautet der Gradient der negativen Log-Likelihood bezüglich $\eta$ und $\mathbf{w}$?

<!--en:834924b9-->
3. **Add-One-Glättung mit der Dirichlet-Verteilung:** Für ein Vokabular aus 4 Wörtern seien die Anzahlen $\mathbf{x} = (5, 0, 1, 0)$ beobachtet ($n = 6$). Wie lautet der Maximum-Likelihood-Schätzer des Wahrscheinlichkeitsvektors? Wie lautet der A-posteriori-Mittelwert bei gleichförmiger A-priori-Verteilung $\mathrm{Dir}(1, 1, 1, 1)$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:a54dffd4-->
1. $A'(\eta) = -\frac{1}{\eta} = \mathbf{\frac{1}{\lambda} = \mathbb{E}[X]}$ und $A''(\eta) = \frac{1}{\eta^2} = \mathbf{\frac{1}{\lambda^2} = \mathrm{Var}(X)}$.

<!--en:3dcfc7a4-->
2. Nach $\nabla_\eta \mathcal{L} = \mu - T(y)$ ist der Gradient bezüglich $\eta$ gleich **$3 - 5 = -2$** und bezüglich $\mathbf{w}$ gleich **$-2\mathbf{x}$**. Beim Gradientenabstieg steigt also $\eta$, und $\hat{\lambda}$ nähert sich $5$.

<!--en:a3bdbeb2-->
3. $\hat{\mathbf{p}}_{\mathrm{MLE}} = (5/6, 0, 1/6, 0)$ weist unbeobachteten Wörtern die Wahrscheinlichkeit null zu. Für $\mathrm{Dir}(1,1,1,1)$ ergibt sich die A-posteriori-Verteilung $\mathrm{Dir}(6, 1, 2, 1)$ mit A-posteriori-Mittelwert **$(0.6, 0.1, 0.2, 0.1)$**.

<!--en:4f655948-->
::: {.slide title="Zusammenfassung: Die Exponentialfamilie"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:0b8311d5-->
| Eigenschaft | Mathematische Aussage | Ergebnis für Münze oder Vergleichsbeispiel | Bedeutung für Deep Learning |

<!--en:8b294cd7-->
| **Kanonische Form** | $p(\mathbf{x}\mid\boldsymbol{\eta}) = h(\mathbf{x})\exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$ | Bernoulli: $\eta = \log\frac{p}{1-p}$, $A(\eta) = \log(1+e^\eta)$ | Vereint Bernoulli-, Kategorial-, Gauß- und Poisson-Verteilung |

<!--en:316cb7d6-->
| **Suffiziente Statistiken** | $\prod_{i=1}^n p(\mathbf{x}_i\mid\boldsymbol{\eta}) \propto \exp(\boldsymbol{\eta}^\top \sum_i T(\mathbf{x}_i) - nA(\boldsymbol{\eta}))$ | Gesamtzahl der Kopfwürfe $\sum_i x_i = 9$ enthält alle Information | Verdichtet beliebige Datensätze zu Summen fester Dimension |

<!--en:695d4855-->
| **Maximale Entropie** | $H_h[q] = H_h[p] - D_{\mathrm{KL}}(q \| p) \le H_h[p]$ für $\mathbb{E}_q[T]=\mathbb{E}_p[T]$ | Gezinkter Würfel ($\mathbb{E}=4.5$): Kippung $p_k \propto e^{0.371k}$ | Erklärt den Ursprung von Gauß-, Poisson- und Boltzmann-Verteilung |

<!--en:388932f6-->
| **Kumulanten aus Ableitungen** | $\nabla A(\boldsymbol{\eta}) = \mathbb{E}[T(\mathbf{x})]$, $\nabla^2 A(\boldsymbol{\eta}) = \mathrm{Cov}(T(\mathbf{x})) \succeq \mathbf{0}$ | Poisson $A(\eta) = e^\eta \implies \text{mean} = \text{var} = \lambda$ | Zeigt die globale Konvexität von $A(\boldsymbol{\eta})$ und der negativen Log-Likelihood |

<!--en:ad1b747e-->
| **Momentenabgleich und GLMs** | $\nabla A(\hat{\boldsymbol{\eta}}) = \bar{T}$, $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{x}(\boldsymbol{\mu}(\mathbf{x}) - T(y))^\top$ | $\mathrm{sigmoid}(\hat{\eta}) = 9/13$, Divergenz bei $\bar{T} \in \{0, 1\}$ | Gemeinsame Gradientenformel für die Ausgabeschicht bei verschiedenen Verlustfunktionen |

<!--en:6e7283ad-->
| **Softmax-Funktion und Konjugiertheit** | $\operatorname{argmax}_k(z_k + T\epsilon_k) \sim \mathrm{softmax}(\mathbf{z}/T)$, $\boldsymbol{\nu} \to \boldsymbol{\nu}+\sum_i T_i$ | $\mathrm{Beta}(1,1) \to \mathrm{Beta}(10,5)$, Mittelwert $2/3$ | Gumbel-Max-Verfahren, Temperatur und Laplace-Glättung |
