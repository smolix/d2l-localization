<!--en:3f1a15d6-->
# Erwartungswerte und ihre Gradienten

<!--en:d285a975-->
Fast jeder Verlust, den wir bisher minimiert haben, ist ein Mittelwert. Wir mitteln über die Beispiele eines Minibatchs, über zufällige Dropout-Masken oder über Rauschen, das wir einer Eingabe hinzufügen. Solange die Quelle des Zufalls nicht von den Modellparametern abhängt, lässt sich ein solcher Mittelwert leicht differenzieren. Seine Ableitung ist der Mittelwert der Ableitungen. Der Minibatch-Gradientenabstieg berechnet sie daher direkt. Schwierig wird es, wenn die Parameter die Wahrscheinlichkeitsverteilung selbst bestimmen. Ein Variationsautoencoder zieht seinen latenten Code aus einer Verteilung, deren Mittelwert und Varianz sein Encoder liefert. Ein Policy-Gradient-Agent (:numref:`sec_policygradient`) zieht Aktionen gemäß genau der Strategie (engl. policy), die er verbessern soll. In beiden Fällen ist die Zielfunktion ein Erwartungswert (engl. expectation) über Samples, deren Verteilung sich mit den Parametern ändert. Ein einmal gezogenes Sample ist jedoch nur eine Zahl. An ihr lässt sich nicht mehr ablesen, wie sie vom Parameter abhängt.

<!--en:0e647f2b-->
Betrachten wir das einfachste Problem dieser Art. Sei $X\sim\mathcal N(\theta,1)$ eine normalverteilte Zufallsvariable mit Mittelwert $\theta$ und Varianz eins. Wir betrachten den erwarteten quadratischen Verlust

<!--en:70bed0aa-->
Die geschlossene Form ergibt sich unmittelbar, wenn wir $X=\theta+Z$ mit standardnormalverteiltem $Z$ schreiben. Dann gilt $\mathbb{E}[(\theta+Z)^2]=\theta^2+2\theta\,\mathbb{E}[Z]+\mathbb{E}[Z^2]$, wobei aus Symmetriegründen $\mathbb{E}[Z]=0$ gilt und außerdem $\mathbb{E}[Z^2]=1$ (:numref:`subsec_mdl-integration-by-parts`). An der Stelle $\theta=1.5$ beträgt die exakte Ableitung $\ell'(1.5)=3$. In der Praxis kann ein Modell zwar Samples von $X$ ziehen, aber $\ell(\theta)$ nicht in geschlossener Form auswerten. Wie können wir dann $\ell'(\theta)$ allein aus Samples schätzen?

<!--en:b1d3ee51-->
Die Analysis gibt zwei grundverschiedene Antworten. Beide Verfahren schreiben den Erwartungswert als Integral (:numref:`sec_mdl-integral_calculus`) und ziehen die Ableitung unter das Integralzeichen. Danach unterscheiden sie sich. Der *Score-Funktionsschätzer* (engl. score-function estimator) differenziert die Wahrscheinlichkeitsdichte und lässt die Samples unverändert. Der *pfadweise* Schätzer (engl. pathwise estimator) differenziert die Samples und lässt die Dichte des Rauschens unverändert. Keines der Verfahren ist dem anderen in allen Fällen überlegen. Ihr Vergleich (:numref:`subsec_mdl-stochastic-gradient-estimators`) steht im Mittelpunkt dieses Abschnitts. Zunächst wiederholen wir kurz, wie wir Erwartungswerte als Integrale schreiben. Dann klären wir, wann wir unter dem Integralzeichen differenzieren dürfen, denn gerade hier können sich schwer erkennbare Fehler einschleichen.

<!--en:9338789b-->
## Erwartungswerte als Integrale

<!--en:3b603a20-->
Bevor wir differenzieren, legen wir unsere Schreibweise für Dichten und Erwartungswerte fest. Das Kapitel zur Wahrscheinlichkeitsrechnung führt beide Begriffe ausführlich ein (:numref:`sec_mdl-random_variables`). Hier formulieren wir sie mithilfe der Integrale aus dem vorigen Abschnitt. Eine **Wahrscheinlichkeitsdichte** auf einem Gebiet $\mathcal X$ ist eine nichtnegative Funktion, deren Integral über das gesamte Gebiet $1$ ergibt:

<!--en:373ce6f4-->
Die Wahrscheinlichkeit, dass eine Zufallsvariable $X$ mit Dichte $p$ in ein Intervall $[a, b]$ fällt, ist die Fläche unter $p$ über diesem Intervall. Es gilt $P(a\le X\le b)=\int_a^b p(x)\,dx$ (:numref:`sec_mdl-random_variables`). Eine Dichte gibt die Wahrscheinlichkeit pro Längeneinheit an. Ihr Wert kann daher größer als $1$ sein. Beispielsweise hat die Dichte der Gleichverteilung auf $[0,\tfrac12]$ überall auf ihrem Träger den Wert $2$. Als durchgängiges Beispiel dient uns die **Standardnormalverteilung** $\mathcal N(0,1)$. Ihre Dichte $p(z)=e^{-z^2/2}/\sqrt{2\pi}$ ist aufgrund des Gauß-Integrals aus :numref:`sec_mdl-integral_calculus` normiert.

<!--en:d0ccdf79-->
Der **Erwartungswert** ist der gewichtete Mittelwert von $X$ oder einer Funktion $g(X)$. Dabei gewichten wir jedes Ergebnis mit seiner Wahrscheinlichkeitsdichte:

<!--en:8826562a-->
Der Erwartungswert $\mathbb{E}[g(X)]$ ist wohldefiniert, wenn das Integral absolut konvergiert, also $\int_{\mathcal X}|g(x)|\,p(x)\,dx<\infty$ gilt. Verteilungen mit schweren Rändern können diese Bedingung verletzen. Beispielsweise besitzt die Cauchy-Verteilung mit der Dichte aus Aufgabe :exref:`ex_mdl-integral-calculus-heavy-tails` keinen endlichen Mittelwert.

<!--en:8228c5e8-->
Im Deep Learning lassen sich Erwartungswerte selten in geschlossener Form angeben. Oft integrieren wir über Tausende von Dimensionen, sodass eine numerische Quadratur auf einem Gitter ausscheidet. Stattdessen ziehen wir unabhängig voneinander die Samples $x_1,\ldots,x_n$ aus $p$ und bilden den Stichprobenmittelwert $\frac1n\sum_{i=1}^n g(x_i)$. Dies ist die *Monte-Carlo*-Schätzung. Hat ein einzelnes Sample $g(X)$ die Standardabweichung $\sigma$, so beträgt die Wurzel des mittleren quadratischen Fehlers des Monte-Carlo-Mittelwerts $\sigma/\sqrt n$, unabhängig von der Dimension (:numref:`subsec_mdl-monte-carlo`). Beide Gradientenschätzer in diesem Abschnitt sind solche Monte-Carlo-Mittelwerte. Da beide denselben wahren Gradienten erwartungstreu schätzen, kommt es für den Fehler nur darauf an, wie stark ihre Werte bei einer einzelnen Ziehung streuen. Die Standardabweichung $\sigma$ bei einer Ziehung ist daher das natürliche Vergleichsmaß.

<!--en:e20f990a-->
## Differenzieren unter dem Integralzeichen

<!--en:1477f52a-->
### Vertauschen von Ableitung und Integral

<!--en:b5602bf7-->
Der Verlust $\ell(\theta)=\mathbb{E}[X^2]$ aus unserem Beispiel ist ein Integral, das von einem Parameter $\theta$ abhängt. Allgemein kann ein Parameter ein Integral an zwei Stellen beeinflussen. Er kann die Integrationsgrenzen bestimmen, für die der Hauptsatz der Differential- und Integralrechnung (:numref:`sec_mdl-integral_calculus`) gilt, oder in den Integranden selbst eingehen.

<!--en:56d252b2-->
Sind die Integrationsgrenzen fest und hängt nur der Integrand von $\theta$ ab, so ziehen wir die Ableitung unter das Integralzeichen:

<!--en:fdc5a885-->
Geometrisch bedeutet das: Die Änderungsrate der Gesamtfläche ist das Integral über die punktweise Änderungsrate der Höhe. Dieses *Vertauschen* von Ableitung und Integral liegt beiden Gradientenschätzern zugrunde. Auf den ersten Blick erscheint die Gleichung unproblematisch. Eine Ableitung ist ein Grenzwert von Differenzenquotienten, ein Integral ein Grenzwert von Summen, und endliche Summen und Differenzen lassen sich stets vertauschen. Zwei Grenzwerte dürfen wir jedoch nur unter gewissen Regularitätsbedingungen vertauschen. Auf einem beschränkten Intervall für $x$ genügt es, dass $\partial_\theta f$ in beiden Variablen gemeinsam stetig ist. Auf einem unbeschränkten Gebiet brauchen wir zusätzlich eine integrierbare Funktion von $x$, die $|\partial_\theta f(x,\theta')|$ für alle $\theta'$ in der Nähe von $\theta$ nach oben beschränkt. Das folgt unmittelbar aus dem Satz von der dominierten Konvergenz, den wir ohne Beweis aus :cite:`Folland.1999` übernehmen. Für unser Beispiel $f(x,\theta)=x^2p_\theta(x)$ mit einer Normaldichte gibt es immer eine solche Majorante, denn an den Rändern fällt die Gauß-Dichte schneller ab, als jedes Polynom wächst.

<!--en:a40c20bc-->
### Leibniz-Regel

<!--en:e93e67ef-->
Ändern sich mit $\theta$ auch die Integrationsgrenzen, liefert das bloße Vertauschen ein unvollständiges und damit falsches Ergebnis. Die Leibniz-Regel ergänzt die fehlenden Randterme. Die Änderung im Inneren und die von den wandernden Grenzen überstrichenen Flächen addieren sich (:numref:`fig_mdl-leibniz-boundary`).

<!--en:fe9a187b-->
**Proposition (Leibniz-Regel).** *Die Funktionen $f$ und $\partial_\theta f$ seien stetig auf einem Gebiet, das $\{(x,\theta):x\in[a(\theta),b(\theta)]\}$ für alle $\theta$ in einem offenen Intervall enthält. Ferner seien $a$ und $b$ differenzierbar. Dann gilt*

<!--en:169bec48-->
**Beweis.** Wir definieren $H(\alpha,\beta,\theta)=\int_\alpha^\beta f(x,\theta)\,dx$. Nach dem Hauptsatz der Differential- und Integralrechnung gelten $\partial_\beta H=f(\beta,\theta)$ und $\partial_\alpha H=-f(\alpha,\theta)$. Das Vertauschen auf dem beschränkten Intervall $[\alpha,\beta]$ ergibt $\partial_\theta H=\int_\alpha^\beta\partial_\theta f\,dx$. Da alle drei partiellen Ableitungen stetig sind, können wir die mehrdimensionale Kettenregel (:numref:`sec_mdl-chain-rule-graphs`) auf die Verkettung $\theta\mapsto H(a(\theta),b(\theta),\theta)$ anwenden. Sie addiert die drei Terme mit den Gewichten $a'(\theta)$, $b'(\theta)$ und $1$. $\blacksquare$

<!--en:c19a50dc-->
![Die Leibniz-Regel für $f(x,\theta)=x/\theta$ auf $[0,\theta]$ am Beispiel der Gleichverteilung. Wächst $\theta$ um $d\theta$, verschiebt sich die Kurve des Integranden nach $f(\cdot,\theta+d\theta)$ (gestrichelt). Hier sinkt sie, die schraffierte Fläche geht also verloren. Gleichzeitig wandert die rechte Grenze nach außen. Dadurch kommt ein Streifen der Höhe $f(b,\theta)$ und der Breite $b'\,d\theta$ hinzu.](../img/mdl-cal-leibniz-boundary.svg)

<!--en:247352ce-->
Wer den Randterm weglässt, riskiert sogar ein falsches Vorzeichen der Ableitung. Sei $X$ auf $[0,\theta]$ gleichverteilt, mit der Dichte $1/\theta$ auf diesem Intervall. Der Mittelwert ist $\mathbb{E}[X]=\int_0^\theta\frac x\theta\,dx=\frac\theta2$, seine Ableitung beträgt $+\tfrac12$. Differenzieren wir nur den Integranden, erhalten wir $\int_0^\theta-\frac{x}{\theta^2}\,dx=-\tfrac12$, also das falsche Vorzeichen. Addieren wir den Beitrag $f(\theta,\theta)\,b'(\theta)=\frac\theta\theta\cdot1=1$ der oberen Grenze, erhalten wir die richtige Ableitung $-\tfrac12+1=+\tfrac12$. Hängt der Träger einer Verteilung vom Parameter ab, ist das bloße Vertauschen unzulässig (in Aufgabe :exref:`ex_mdl-integral-calculus-moving-support` berechnen Sie $\mathbb{E}[X^2]$).

<!--en:32f61b8d-->
### Log-Integral-Exp

<!--en:0c269f13-->
Durch Vertauschen erhalten wir auch das kontinuierliche Gegenstück zur Log-Sum-Exp-Funktion. Wenden wir die Kettenregel und :eqref:`eq_mdl-diff-under-integral` auf $\log\int e^{\theta x}p(x)\,dx$ an, so ergibt sich

<!--en:e600e67b-->
Die rechte Seite ist der Mittelwert von $x$ unter der exponentiell *umgewichteten* Dichte, die proportional zu $e^{\theta x}p(x)$ ist. Bei der Standardnormaldichte zeigt die quadratische Ergänzung $\theta z-\tfrac12z^2=\tfrac12\theta^2-\tfrac12(z-\theta)^2$, dass die umgewichtete Dichte zu $\mathcal N(\theta,1)$ gehört. Außerdem gilt $\log\mathbb{E}[e^{\theta Z}]=\theta^2/2$. Die Ableitung dieses Ausdrucks ist $\theta$, also tatsächlich der Mittelwert der umgewichteten Verteilung. So wie der Gradient von Log-Sum-Exp der Softmax-Vektor ist, ist der Gradient von Log-Integral-Exp ein Erwartungswert. Dieser Identität begegnen wir in Gestalt der Ableitung $A'(\eta)$ der logarithmischen Zustandssumme in :numref:`sec_mdl-exponential-family` wieder.

<!--en:1b3e03ee-->
## Gradienten mit der Score-Funktion und pfadweise Gradienten

<!--en:94ff0dca-->
Vertauschen wir bei einem erwarteten Verlust Ableitung und Integral, erhalten wir das Integral $\int g(x)\,\partial_\theta p_\theta(x)\,dx$. Dieses Integral können wir mit Monte-Carlo-Sampling nicht direkt schätzen. Denn $\partial_\theta p_\theta$ ist keine Wahrscheinlichkeitsdichte: Die Funktion nimmt positive und negative Werte an und hat das Integral null. Aus ihr können wir daher keine Samples ziehen. Mit zwei einfachen Identitäten lässt sich das Integral wieder als Erwartungswert schreiben, den wir aus Samples schätzen können.

<!--en:19b0f589-->
### Der Score-Funktionsschätzer

<!--en:1dc30004-->
Sei $p_{\boldsymbol\theta}$ eine durch $\boldsymbol\theta$ parametrisierte Dichte. Wenden wir die Kettenregel auf $\log p_{\boldsymbol\theta}(x)$ an, erhalten wir die **Identität der logarithmischen Ableitung**:

<!--en:56acc4b3-->
In Worten: Der Gradient einer Dichte ist die Dichte selbst multipliziert mit dem Gradienten ihres Logarithmus. Diesen Vektor bezeichnen wir als **Score** $\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}$. Ist das Vertauschen zulässig und ändert sich der Träger von $p_{\boldsymbol\theta}$ nicht mit $\boldsymbol\theta$, so hat der Score den Erwartungswert null: $\mathbb{E}[\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(X)]=\int\nabla_{\boldsymbol\theta}p_{\boldsymbol\theta}\,dx=\nabla_{\boldsymbol\theta}\int p_{\boldsymbol\theta}\,dx=\nabla_{\boldsymbol\theta}1=\mathbf 0$. Die Dichte der Gleichverteilung auf $[0,\theta]$ zeigt, warum ein fester Träger entscheidend ist. Ihr Score ist $\partial_\theta\log(1/\theta)=-1/\theta$ und hat den Erwartungswert $-1/\theta\ne0$, denn beim Differenzieren von $\int_0^\theta p_\theta(x)\,dx=1$ kommt der Randterm $p_\theta(\theta)=1/\theta$ hinzu. Dass der Score den Mittelwert null hat, ist auch der Ausgangspunkt für die Fisher-Information (:numref:`sec_mdl-maximum_likelihood`).

<!--en:07bc2d7f-->
Multiplizieren wir :eqref:`eq_mdl-log-derivative` mit einer Funktion $g(x)$, die nicht von $\boldsymbol\theta$ abhängt, und integrieren wir anschließend, erhalten wir die **Identität für die Score-Funktion**:

<!--en:4a3fd967-->
Damit ist der Gradient eines erwarteten Verlusts wieder ein Erwartungswert unter $p_{\boldsymbol\theta}$. Der Verlust $g(x)$ jedes Samples erhält ein Gewicht, das angibt, wie stark $\boldsymbol\theta$ die Log-Wahrscheinlichkeit dieses Samples erhöht. Für die Monte-Carlo-Schätzung ziehen wir Samples $x_i\sim p_{\boldsymbol\theta}$ und mitteln die Werte $g(x_i)\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(x_i)$. Im Reinforcement Learning ist dies der REINFORCE-Schätzer der Policy-Gradient-Verfahren (:numref:`sec_policygradient`) :cite:`Williams.1992`. (Hängt $g$ auch direkt von $\boldsymbol\theta$ ab, kommt nach der Produktregel der Term $\mathbb{E}[\nabla_{\boldsymbol\theta}g(X,\boldsymbol\theta)]$ hinzu.)

<!--en:20003492-->
Da der Score den Mittelwert null hat, können wir jede konstante **Baseline** $b$ (unabhängig von $x$) von $g(X)$ abziehen, ohne den Schätzer zu verzerren (:numref:`sec_baselines`). Es gilt $\mathbb{E}[(g(X)-b)\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(X)]=\mathbb{E}[g(X)\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(X)]$. Eine geeignete Wahl von $b$ kann die Varianz der Schätzung erheblich senken.

<!--en:973453ba-->
### Der pfadweise Schätzer

<!--en:bccdef04-->
Mit der zweiten Identität machen wir wieder sichtbar, wie das Sample von $\boldsymbol\theta$ abhängt. Häufig lässt sich ein Sample aus $p_{\boldsymbol\theta}$ durch eine deterministische Transformation $x=T_{\boldsymbol\theta}(\mathbf z)$ erzeugen. Diese wirkt auf eine Hilfsvariable $\mathbf z\sim q$, die das Rauschen beschreibt und deren Verteilung nicht von $\boldsymbol\theta$ abhängt. Eine normalverteilte Zufallsvariable mit Verteilung $\mathcal N(\mu,\sigma^2)$ lässt sich beispielsweise als $\mu+\sigma z$ mit $z\sim\mathcal N(0,1)$ schreiben. Schreiben wir den Erwartungswert als Integral über $\mathbf z$, kommt $\boldsymbol\theta$ in der Dichte nicht mehr vor. Durch Vertauschen ergibt sich die **pfadweise Identität**:

<!--en:aeefe2e5-->
Dieser Schätzer differenziert durch das Sample hindurch. Beim automatischen Differenzieren verfolgen wir den Pfad von $\boldsymbol\theta$ über $T_{\boldsymbol\theta}$ bis in $g$. Dieses Verfahren heißt *Reparametrisierungstrick* (engl. reparameterization trick). Variationsautoencoder nutzen es für die Backpropagation durch normalverteilte latente Variablen :cite:`Kingma.Welling.2014,rezende2014stochastic`. Der Schätzer selbst ist älter. In der Simulationsliteratur war er schon lange als pfadweise Ableitung und als Ableitung durch infinitesimale Störung bekannt :cite:`Mohamed.Rosca.Figurnov.ea.2020`. Voraussetzung ist, dass $T_{\boldsymbol\theta}$ nach $\boldsymbol\theta$ differenzierbar ist und $g$ entlang des Pfades differenzierbar ist (Lipschitz-Stetigkeit von $g$ genügt, wenn $X$ eine stetige Dichte hat). Außerdem brauchen wir für das Vertauschen eine integrierbare Majorante. Ein Randterm tritt hier nie auf, selbst wenn sich der Träger von $X$ mit $\boldsymbol\theta$ ändert. Denn wir integrieren über $\mathbf z$, dessen Dichte $q$ fest bleibt.

<!--en:a7c17d2f-->
Die Differenzierbarkeit von $g$ ist eine wesentliche Voraussetzung. Betrachten wir die Sprungfunktion $g(x)=\mathbf{1}(x>0)$. Sie hat den Wert $1$ für positive $x$ und sonst den Wert $0$. Dabei sei $X=\theta+z$. Der wahre Erwartungswert $\mathbb{E}[g(X)]=P(z>-\theta)$ hat die Ableitung $\varphi(\theta)=e^{-\theta^2/2}/\sqrt{2\pi}$. Ihr Wert beträgt etwa $0.13$ an der Stelle $\theta=1.5$. Der pfadweise Beitrag $\partial_\theta\mathbf{1}(\theta+z>0)$ ist jedoch gleich $0$ für jedes $z$ außer im Punkt $z=-\theta$, einem Ereignis mit Wahrscheinlichkeit null. Der pfadweise Schätzer liefert also $0$ mit Wahrscheinlichkeit $1$. Der Beitrag $\mathbf{1}(x>0)\,z$ des Score-Funktionsschätzers hat dagegen den Erwartungswert $\int_{-\theta}^\infty z\,\varphi(z)\,dz=\bigl[-\varphi(z)\bigr]_{-\theta}^\infty=\varphi(\theta)$ und liefert damit im Mittel die exakte Ableitung. Da der Score-Funktionsschätzer weder durch $g$ noch durch die Ziehung selbst differenziert, eignet er sich auch für unstetige Verluste und diskrete Verteilungen. Beispiele sind Bernoulli-verteilte oder kategoriale Entscheidungen, bei denen kein differenzierbarer Pfad von $\boldsymbol\theta$ zum Sample existiert.

<!--en:083e36f9-->
### Vergleich der beiden Schätzer

<!--en:1ad6927b-->
Wir haben nun zwei erwartungstreue Schätzer derselben Ableitung. Unser Beispiel $\ell(\theta)=\mathbb{E}[X^2]$ mit $X=\theta+z$ ist so einfach, dass wir beide Varianzen von Hand berechnen können. Zum pfadweisen Schätzer trägt eine einzelne Ziehung $\partial_\theta(\theta+z)^2 = 2X$ bei, mit der Varianz $\textrm{Var}(2X)=4$. Für den Score-Funktionsschätzer nutzen wir den Score der Normalverteilungen mit variablem Lageparameter, $\partial_\theta\log p_\theta(X)=X-\theta=z$. Ein Sample trägt also $X^2z$ bei, beziehungsweise $(X^2-b)z$ bei Verwendung einer Baseline $b$. Um die Varianz von $X^2z$ zu berechnen, verwenden wir die Momente der Standardnormalverteilung $\mathbb{E}[z^2]=1$, $\mathbb{E}[z^4]=3$ und $\mathbb{E}[z^6]=15$. Sie folgen alle aus der Stein-Identität $\mathbb{E}[g'(Z)]=\mathbb{E}[Z\,g(Z)]$ (:numref:`subsec_mdl-integration-by-parts`). Aufgrund der Symmetrie fallen bei der Berechnung des zweiten Moments alle ungeraden Potenzen von $z$ weg:

<!--en:e07e4c80-->
Ziehen wir das Quadrat des Mittelwerts $(2\theta)^2$ ab, bleibt die Varianz $\theta^4+14\theta^2+15$. Die konstante Baseline mit der kleinsten Varianz ist $b^\star=\theta^2+3$. Sie senkt die Varianz auf $8\theta^2+6$ (Aufgabe :exref:`ex_mdl-integral-calculus-quartic-estimators`). Bei $\theta=1.5$ ergeben sich für eine einzelne Ziehung die Standardabweichungen $2$ für den pfadweisen Schätzer, $7.18$ für den Score-Funktionsschätzer ohne Baseline und $4.90$ für den Score-Funktionsschätzer mit optimaler Baseline. Die Baseline hilft also deutlich. Bei diesem Verlust kann sie den Abstand jedoch nicht ganz ausgleichen, denn es gilt $8\theta^2+6>4$ für alle $\theta$.

<!--en:ebeef0a1-->
Prüfen wir alle drei analytischen Vorhersagen in einer einzigen Monte-Carlo-Simulation. Da die pfadweise Ableitung hier $2x_i$ beträgt, brauchen wir nicht einmal automatisches Differenzieren. Wenn unsere Rechnung stimmt, sollten die drei empirischen Standardabweichungen nahe bei $2$, $7.18$ und $4.90$ liegen. Jeder Stichprobenmittelwert sollte höchstens einige Standardfehler $\mathrm{sd}/\sqrt n$ von der wahren Ableitung $3$ entfernt sein.

<!--en:e2d6216f-->
In typischen Durchläufen weichen alle drei Mittelwerte nur wenige Standardfehler von $3$ ab. Die empirischen Standardabweichungen stimmen bis auf wenige Prozent mit den analytischen Vorhersagen überein. Bei gleicher Anzahl von Samples ist die pfadweise Schätzung etwa $3.6$-mal so genau wie die Schätzung mit der Score-Funktion ohne Baseline. Anders ausgedrückt braucht der Score-Funktionsschätzer ohne Baseline für dieselbe Genauigkeit rund dreizehnmal so viele Samples.

<!--en:725435cb-->
Der Schluss, der pfadweise Schätzer habe immer die kleinere Varianz, wäre jedoch falsch. Betrachten wir einen oszillierenden Verlust $g(x)=\sin(\omega x)$. Die wahre Ableitung seines Erwartungswerts ist $\omega\cos(\omega\theta)e^{-\omega^2/2}$. Der pfadweise Beitrag $\omega\cos(\omega X)$ skaliert mit der Frequenz $\omega$ und hat eine Standardabweichung nahe $\omega/\sqrt2$. Der Beitrag $\sin(\omega X)\,z$ des Score-Funktionsschätzers hat dagegen höchstens die Standardabweichung $\sqrt{\mathbb{E}[z^2]}=1$, denn es gilt $|\sin|\le1$. Bei einem rasch oszillierenden Integranden kann die Ableitung sehr groß sein, obwohl seine Funktionswerte beschränkt bleiben. Vergleichen wir die beiden Schätzer bei $\omega=1$ und $\omega=3$ anhand der bereits gezogenen Samples.

<!--en:c418b70d-->
Schon bei $\omega=1$ streut der Score-Funktionsschätzer weniger (etwa $0.5$ gegenüber $0.7$). Bei $\omega=3$ ist seine Standardabweichung um den Faktor drei kleiner (etwa $0.7$ gegenüber $2.1$). Beide Stichprobenmittelwerte stimmen bis auf wenige Standardfehler mit den exakten Ableitungen überein.

<!--en:c9dee437-->
Je nach Problem ist der eine oder der andere Schätzer im Vorteil. Der pfadweise Schätzer differenziert $g$, seine Varianz wächst also bei großen Werten von $g'$. Der Score-Funktionsschätzer multipliziert $g$ mit dem Score, seine Varianz wächst also bei großen Werten von $g$ selbst. In beiden Fällen bestimmt die Standardabweichung $\sigma$ bei einer Ziehung, wie viele Samples für eine vorgegebene Genauigkeit nötig sind. Dabei hängt $\sigma$ auch von der Verteilung ab, aus der wir ziehen. Wählen wir die Vorschlagsverteilung so, dass $\sigma$ klein wird, gelangen wir unmittelbar zum Importance-Sampling, das wir in :numref:`sec_mdl-bayesian-computation` behandeln.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:d111d520-->
Bestimmen die Parameter eines Modells die Verteilung seiner Samples, müssen wir den Gradienten des erwarteten Verlusts selbst als Erwartungswert schreiben. Erst dann können wir ihn mit Monte-Carlo-Sampling schätzen. Beide Verfahren ziehen dazu die Ableitung unter das Integralzeichen. Das setzt eine integrierbare Majorante voraus. Ändern sich die Integrationsgrenzen mit dem Parameter, kommen die Randterme der Leibniz-Regel hinzu. Die Identität für die Score-Funktion differenziert die Log-Dichte. Sie gilt für beliebige Verlustfunktionen $g$ und diskrete Zufallsvariablen. Die Beiträge der einzelnen Samples skalieren jedoch mit der Größe von $g$ und können ohne gute Baseline stark streuen. Bei der pfadweisen Identität differenzieren wir durch ein reparametrisiertes Sample hindurch. Das setzt ein differenzierbares $g$ voraus und führt oft, aber nicht immer, zu einer wesentlich kleineren Varianz. Bei beiden Schätzern beträgt die Wurzel des mittleren quadratischen Fehlers $\sigma/\sqrt n$ (:numref:`subsec_mdl-monte-carlo`). Welcher Schätzer sich für ein gegebenes Problem besser eignet, hängt daher von der Standardabweichung $\sigma$ bei einer Ziehung ab.

<!--en:2294243a-->
## Übungen

<!--en:dff04ff2-->
1. **Ein beweglicher Träger.** Sei $X$ auf $[0,\theta]$ gleichverteilt mit $\theta>0$. Dann gelten $\ell(\theta)=\mathbb{E}[X^2]=\int_0^\theta \tfrac{x^2}{\theta}\,dx=\tfrac{\theta^2}{3}$ und $\ell'(\theta)=\tfrac{2\theta}{3}$.
    1. Differenzieren Sie $\ell$ mit der Leibniz-Regel :eqref:`eq_mdl-leibniz`. Welchen Teil des Ergebnisses liefert das bloße Vertauschen gemäß :eqref:`eq_mdl-diff-under-integral`, und welcher Teil fehlt?
    1. Auf dem Träger gilt $\log p_\theta(x)=-\log\theta$. Berechnen Sie den Erwartungswert des Score-Funktionsschätzers $X^2\,\partial_\theta\log p_\theta(X)$ aus :eqref:`eq_mdl-score-function-gradient`. Sei nun $g$ stetig auf $[0,\infty)$. Zeigen Sie, dass der Mittelwert von $g(X)\,\partial_\theta\log p_\theta(X)$ genau dem Ergebnis entspricht, das Sie durch bloßes Vertauschen gemäß :eqref:`eq_mdl-diff-under-integral` bei $\mathbb{E}[g(X)]=\int_0^\theta g(x)\,p_\theta(x)\,dx$ erhalten. Wie groß ist seine Verzerrung als Schätzer für $\frac{d}{d\theta}\mathbb{E}[g(X)]$? Drücken Sie diese durch $g(\theta)$ und $\theta$ aus.
    1. Schreiben Sie $X=\theta U$ mit $U$ gleichverteilt auf $[0,1]$ und berechnen Sie den Erwartungswert des pfadweisen Schätzers aus :eqref:`eq_mdl-pathwise-gradient`. Warum benötigt er keinen Randterm, obwohl sich der Träger von $X$ mit $\theta$ ändert?
   :exlabel:`ex_mdl-integral-calculus-moving-support`

<!--en:65cfcb7c-->
1. [code] **Varianz der Schätzer für eine Funktion vierten Grades.** ● Sei $X\sim\mathcal N(\theta,1)$. Schreiben Sie $X=\theta+z$ und bezeichnen Sie den Score mit $s(X)=\partial_\theta\log p_\theta(X)=z$.
    1. Sei $g(X)=X^4$, sodass $\frac{d}{d\theta}\mathbb{E}[X^4]=4\theta^3+12\theta$ gilt. Leiten Sie den pfadweisen Schätzer aus :eqref:`eq_mdl-pathwise-gradient` und den Score-Funktionsschätzer aus :eqref:`eq_mdl-score-function-gradient` jeweils für eine einzelne Ziehung her. Zeigen Sie mit der Identität aus Aufgabe :exref:`ex_mdl-integral-calculus-gaussian-normalizers` (b), dass beide den Mittelwert $4\theta^3+12\theta$ haben.
    1. Zeigen Sie für eine konstante Baseline $b$, dass die Varianz von $(g(X)-b)\,s(X)$ für $b^\star=\mathbb{E}[g(X)s(X)^2]/\mathbb{E}[s(X)^2]$ minimal ist. Zeigen Sie für $g(X)=X^2$, dass $b^\star=\theta^2+3$ gilt. Leiten Sie mit $\mathbb{E}[z^2]=1$, $\mathbb{E}[z^4]=3$ und $\mathbb{E}[z^6]=15$ die drei im Text angegebenen Varianzen $4$, $\theta^4+14\theta^2+15$ und $8\theta^2+6$ her.
    1. Berechnen Sie $b^\star$ für $g(X)=X^4$. Schätzen Sie mit jeweils $10^6$ Samples bei $\theta\in\{0,1,2\}$ die Standardabweichungen des pfadweisen Schätzers und des Score-Funktionsschätzers ohne und mit Baseline. Für welches $\theta$ bringt die Baseline den größten Gewinn? Schließt sie jemals die Lücke zum pfadweisen Schätzer? Erklären Sie die Rangfolge anhand der Größen von $g'$ und $g$.
    1. Betrachten Sie $g(x)=\mathbf{1}(x>0)$ bei $\theta=1.5$. Wie im Text gezeigt, ergibt der pfadweise Schätzer für fast jedes Sample den Wert $0$. Der Score-Funktionsschätzer $\mathbf{1}(x>0)\,z$ hat dagegen den korrekten Mittelwert $\varphi(\theta)=0.1295$. Prüfen Sie beide Aussagen durch Simulation.
   :exlabel:`ex_mdl-integral-calculus-quartic-estimators`

<!--en:912fec6b-->
1. [extended] **Gradienten durch einen Münzwurf hindurch.** ●● Ist $x$ eine diskrete Entscheidung, gibt es keinen differenzierbaren Pfad von $\theta$ zum Sample und damit keinen pfadweisen Schätzer. Stetige Relaxationen wie Gumbel-Softmax :cite:`Jang.Gu.Poole.2017,Maddison.Mnih.Teh.2017` liefern einen differenzierbaren Ersatz. Straight-Through-Schätzer :cite:`Bengio.Leonard.Courville.2013` verwenden im Vorwärtsdurchlauf das diskrete Sample und im Rückwärtsdurchlauf den Ersatz. Beide Verfahren sind für die ursprüngliche Zielfunktion verzerrt. In dieser Aufgabe messen Sie diese Verzerrung. :citet:`Tucker.Mnih.Maddison.ea.2017` veranschaulichen sie an einem einfachen Beispiel. Für eine einzelne Bernoulli-Variable $b\in\{0,1\}$ gilt $P(b=1)=\sigma(\theta)$ mit Logit $\theta$. Die Zielfunktion ist $\ell(\theta)=\mathbb{E}[(b-t)^2]$ mit Zielwert $t=0.45$. Sei $U$ auf $(0,1)$ gleichverteilt, und sei $\zeta=\log U-\log(1-U)$. Für eine Temperatur $\tau>0$ setzen wir $z=\sigma\bigl((\theta+\zeta)/\tau\bigr)$. Das ist die Gumbel-Softmax-Relaxation für zwei Klassen.
    1. Berechnen Sie $\ell(\theta)$ und $\ell'(\theta)$ in geschlossener Form. Welche Wahrscheinlichkeit $\sigma(\theta)$ minimiert $\ell$? Lässt sie sich mit einem endlichen $\theta$ erreichen?
    1. Zeigen Sie, dass $b=\mathbf{1}(z>\tfrac12)$ die Bedingung $P(b=1)=\sigma(\theta)$ für jedes $\tau$ erfüllt.
    1. Implementieren Sie drei Schätzer für $\ell'(\theta)$ auf der Grundlage jeweils einer Ziehung: REINFORCE :eqref:`eq_mdl-score-function-gradient` mit der Baseline $\ell(\theta)$, den relaxierten Schätzer $\frac{d}{d\theta}(z-t)^2$ und den Straight-Through-Schätzer $2(b-t)\frac{dz}{d\theta}$. Letzterer verwendet das diskrete Sample im Verlust und das relaxierte Sample in der Ableitung. Stellen Sie Mittelwert und Standardabweichung jedes Schätzers dem exakten Wert $\ell'(\theta)$ für $\theta\in\{-2,0,2\}$ und $\tau\in\{1,0.5,0.1\}$ in einer Tabelle gegenüber. Wie hängt die Verzerrung der einzelnen Schätzer von $\theta$ und $\tau$ ab?
    1. Führen Sie einen stochastischen Gradientenabstieg für $\theta$ ab dem Startwert $\theta=0$ mit Schrittweite $1$ über $5000$ Schritte aus. Verwenden Sie REINFORCE sowie für jedes $\tau$ den relaxierten Schätzer und den Straight-Through-Schätzer. Ersetzen Sie die Baseline von REINFORCE durch einen exponentiellen gleitenden Mittelwert der bisherigen Verluste. Stellen Sie den Verlauf von $\ell(\theta)$ für jeden Durchlauf grafisch dar. Berechnen Sie die relaxierte Zielfunktion $R_\tau(\theta)=\mathbb{E}[(z-t)^2]$ durch Quadratur über $U$. Vergleichen Sie ihren Minimierer mit den Werten von $\theta$, bei denen sich die Durchläufe mit dem relaxierten Schätzer einpendeln. Wie ändern sich Verzerrung und Varianz des relaxierten Schätzers gegeneinander, wenn Sie $\tau$ verringern?
   :exlabel:`ex_mdl-integral-calculus-coin-flip`

<!--en:bc62d015-->
[Diskussion](https://d2l.discourse.group/t/36878)

<!--en:bc41bda6-->
Erwartungswerte und ihre Gradienten<br>**Differenzieren unter dem Integralzeichen · Leibniz-Regel · Score-Funktion (REINFORCE) · pfadweise Reparametrisierung · Vergleich der Varianzen**

<!--en:7bf6c776-->
::: {.slide title="Wenn Parameter die Verteilung der Samples bestimmen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:ddd904f1-->
- **Sampling mit parametrisierter Verteilung:** VAEs ziehen latente Codes $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\theta}}, \operatorname{diag}(\boldsymbol{\sigma}_{\boldsymbol{\theta}}^2))$, und beim Reinforcement Learning bestimmen Strategien die Verteilung der gezogenen Aktionen $a \sim \pi_{\boldsymbol{\theta}}(a \mid s)$.

<!--en:d5c84a15-->
- **Unterbrochener Berechnungsgraph:** Ein direkt gezogenes Sample $x_i \sim p_{\boldsymbol{\theta}}$ ist nur eine Zahl. Kein Berechnungsgraph verbindet sie mit $\boldsymbol{\theta}$.

<!--en:c2bd7d0a-->
- **Durchgängiges Vergleichsbeispiel:** Für $X \sim \mathcal{N}(\theta, 1)$ und $\ell(\theta) = \mathbb{E}[X^2] = \theta^2 + 1$ beträgt die wahre Steigung an der Stelle $\theta = 1.5$ genau $\ell'(1.5) = 2\theta = 3$.

<!--en:9d932056-->
::: {.slide title="Erwartungswerte als Integrale und Monte-Carlo-Fehler"}

<!--en:c645f0a9-->
[Grundlagen]{.kicker}

<!--en:57ff7198-->
- **Erwartungswert als Integral:** $\mathbb{E}_{X \sim p}[g(X)] = \int_{\mathcal{X}} g(x)p(x)\,dx$ lässt sich durch $\hat{\mu}_n = \frac{1}{n}\sum_{i=1}^n g(x_i)$ annähern.

<!--en:e33e26cd-->
- **Dimensionsunabhängiger RMSE ($\sigma/\sqrt{n}$):** Die Wurzel des mittleren quadratischen Fehlers skaliert mit $\sigma/\sqrt{n}$, unabhängig von der Dimension $d$.

<!--en:b0ca989b-->
- **Vergleichsmaß:** Bei jedem erwartungstreuen Gradientenschätzer bestimmt die Standardabweichung $\sigma$ bei einer Ziehung die Stichprobeneffizienz.

<!--en:7ecd8cca-->
::: {.slide title="Differenzieren unter dem Integralzeichen durch Vertauschen"}

<!--en:cd018cc7-->
[Vertauschen von Grenzwerten]{.kicker}

<!--en:cf13888c-->
- **Vertauschen bei festen Grenzen:** Auf festem Träger $\mathcal{X}$ gilt $\frac{d}{d\theta}\int_{\mathcal{X}} f(x, \theta)\,dx = \int_{\mathcal{X}} \frac{\partial f}{\partial\theta}(x, \theta)\,dx$.

<!--en:2a64fffd-->
- **Regularitätsbedingungen:** Gemeinsame Stetigkeit von $\partial_\theta f$ auf dem beschränkten Intervall $[a, b]$. Eine integrierbare Majorante gemäß $|\partial_\theta f(x, \theta')| \le M(x)$ ist zusätzlich auf dem unbeschränkten Gebiet $\mathbb{R}$ erforderlich (für Gauß-Dichten erfüllt).

<!--en:a816b28d-->
::: {.slide title="Log-Integral-Exp und exponentielle Umgewichtung in Exponentialfamilien"}

<!--en:995312ec-->
[Kontinuierliche Softmax-Funktion]{.kicker}

<!--en:30a12eb6-->
- **Ableitung der logarithmischen Zustandssumme:** Differenzieren von $\log\int e^{\theta x}p(x)\,dx$ durch Vertauschen ergibt den Erwartungswert $\mathbb{E}_{X \sim \tilde{p}_\theta}[X]$ unter der umgewichteten Dichte $\tilde{p}_\theta(x) \propto e^{\theta x}p(x)$.

<!--en:e84b7451-->
- **Probe mit der Standardnormalverteilung:** Für $Z \sim \mathcal{N}(0, 1)$ hat $\log\mathbb{E}[e^{\theta Z}] = \tfrac{1}{2}\theta^2$ die Ableitung $\theta$. Sie stimmt mit dem Mittelwert von $\tilde{p}_\theta = \mathcal{N}(\theta, 1)$ überein.

<!--en:d1c13cdb-->
::: {.slide title="Die Leibniz-Regel bei beweglichen Integrationsgrenzen"}

<!--en:9f17d24d-->
[Beweglicher Träger]{.kicker}

<!--en:8e640428-->
- **Beiträge aus dem Inneren und vom Rand:** Ändern sich $a(\theta)$ und $b(\theta)$ mit $\theta$, ändert sich die Gesamtfläche aus zwei Gründen. Der Graph des Integranden verschiebt sich vertikal ($\int_a^b \partial_\theta f\,dx$), und die Endpunkte wandern horizontal ($f(b,\theta)b' - f(a,\theta)a'$).

<!--en:1ebebf63-->
- **Beweis mit der mehrdimensionalen Kettenregel (§{sec_mdl-chain-rule-graphs}):** Sei $H(\alpha, \beta, \theta) = \int_\alpha^\beta f(x, \theta)\,dx$. Nach dem Hauptsatz der Differential- und Integralrechnung gelten $\partial_\beta H = f(\beta, \theta)$ und $\partial_\alpha H = -f(\alpha, \theta)$. Vertauschen auf dem beschränkten Intervall liefert $\partial_\theta H = \int_\alpha^\beta \partial_\theta f\,dx$. Bei der Auswertung von $\frac{d}{d\theta}H(a(\theta), b(\theta), \theta)$ addieren sich die Produkte entlang aller drei Pfade.

<!--en:dc8fa72c-->
::: {.slide title="Warum das Weglassen von Randtermen das Vorzeichen umkehren kann"}

<!--en:66bdb56c-->
[Fehlerquelle bei beweglichem Träger]{.kicker}

<!--en:d003adf3-->
- **Mittelwert der Gleichverteilung auf $(0, \theta)$:** Die exakte Ableitung von $\mathbb{E}[X] = \int_0^\theta \frac{x}{\theta}\,dx = \frac{\theta}{2}$ beträgt $+\frac{1}{2}$.

<!--en:be336091-->
- **Falsches Vorzeichen beim bloßen Vertauschen:** Differenzieren nur im Inneren ergibt $\int_0^\theta -\frac{x}{\theta^2}\,dx = -\frac{1}{2}$.

<!--en:31bec4da-->
- **Randterm der Leibniz-Regel:** Der zusätzliche Beitrag der oberen Grenze $f(\theta, \theta)b'(\theta) = 1 \cdot 1 = +1$ liefert wieder $-\frac{1}{2} + 1 = +\frac{1}{2}$.

<!--en:fd48abe7-->
::: {.slide title="Warum sich aus der Ableitung einer Dichte keine Samples ziehen lassen"}

<!--en:c0dd27ee-->
[Ein Hindernis für Monte Carlo]{.kicker}

<!--en:ba7c2eb4-->
- **$\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}}(x)$ ist keine Dichte:** In $\int g(x)\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}}(x)\,dx$ nimmt $\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}}$ auch negative Werte an und hat das Integral $\mathbf{0}$.

<!--en:053f5925-->
- **Zwei Wege zurück zum Erwartungswert:**
  1. *Score-Funktion:* Mit $p_{\boldsymbol{\theta}}(x)$ multiplizieren und durch denselben Ausdruck dividieren ($\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}$).
  2. *Pfadweise:* Die Variablen durch $X = T_{\boldsymbol{\theta}}(\mathbf{z})$ mit $\mathbf{z} \sim q(\mathbf{z})$ ersetzen.

<!--en:7e2b12f1-->
::: {.slide title="Score-Funktion und Trick mit der logarithmischen Ableitung"}

<!--en:72fb7940-->
[Identität für die Score-Funktion]{.kicker}

<!--en:7a4b0e9f-->
- **REINFORCE-Identität:** Mit $\nabla_{\boldsymbol{\theta}}p_{\boldsymbol{\theta}} = p_{\boldsymbol{\theta}}\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}$ ergibt sich $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)]$.

<!--en:9189fde2-->
- **Score mit Mittelwert null:** Auf festem Träger gilt $\mathbb{E}[\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)] = \mathbf{0}$ (für $\mathcal{N}(\theta, 1)$ gilt $s_\theta(X) = X - \theta = Z$).

<!--en:45f7cbbd-->
- **Anwendungsbereich:** Die Ableitung von $g(x)$ wird nicht benötigt. Das Verfahren eignet sich daher für diskrete Aktionen und Belohnungen aus einer Blackbox.

<!--en:b6c73342-->
::: {.slide title="Varianzreduktion durch Baselines als Kontrollvariablen"}

<!--en:f3d2f84c-->
[Subtraktion einer Baseline]{.kicker}

<!--en:18cf286f-->
- **Erwartungstreue bei Subtraktion einer Baseline:** Wegen $\mathbb{E}[s_\theta(X)] = 0$ bleibt der Mittelwert beim Ersetzen von $g(X)$ durch $g(X) - b$ für jede Konstante $b$ unverändert.

<!--en:25e62144-->
- **Optimale konstante Baseline:** $b^\star = \frac{\mathbb{E}[g(X)s_\theta(X)^2]}{\mathbb{E}[s_\theta(X)^2]}$ ($= \theta^2 + 3$ für $g(X) = X^2$) senkt die Varianz von $O(\theta^4)$ auf $8\theta^2 + 6 = O(\theta^2)$.

<!--en:137c921f-->
::: {.slide title="Der pfadweise Schätzer durch Reparametrisierung"}

<!--en:73f03067-->
[Differenzieren durch Samples hindurch]{.kicker}

<!--en:a38b4a2f-->
- **Reparametrisierung ($X = T_{\boldsymbol{\theta}}(\mathbf{z})$, $\mathbf{z} \sim q$):** Der Variablenwechsel zu parameterfreiem Rauschen $\mathbf{z}$ ergibt $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{\mathbf{z} \sim q}[\nabla_{\boldsymbol{\theta}}g(T_{\boldsymbol{\theta}}(\mathbf{z}))]$.

<!--en:cb4c414d-->
- **Kettenregel direkt für $g$:** Für $\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\mathbf{z}$ differenziert die Backpropagation direkt durch $g(\boldsymbol{\mu} + \boldsymbol{\sigma}\odot\mathbf{z})$ hindurch und liefert eine Schätzung mit geringer Varianz.

<!--en:43e28602-->
::: {.slide title="Wann der pfadweise Schätzer versagt: Sprungfunktionen und diskrete Entscheidungen"}

<!--en:ff8d5f64-->
[Grenzen der Differenzierbarkeit]{.kicker}

<!--en:acea94ab-->
- **Sprungförmiger Verlust $g(x) = \mathbf{1}(x > 0)$ bei $\theta = 1.5$:** Die wahre Ableitung von $\Phi(\theta)$ beträgt $\varphi(1.5) \approx 0.1295$.

<!--en:be87e9a8-->
- **Der pfadweise Schätzer liefert fast sicher $0$:** Die Sprungfunktion $\theta \mapsto \mathbf{1}(\theta + z > 0)$ verläuft überall waagerecht bis auf die Sprungstelle $z = -\theta$, die mit Wahrscheinlichkeit null getroffen wird.

<!--en:9012c708-->
- **Die Score-Funktion liefert den richtigen Wert:** $\mathbb{E}[\mathbf{1}(\theta + Z > 0)Z] = \varphi(\theta) \approx 0.1295$.

<!--en:e220ec5d-->
::: {.slide title="Analytischer Vergleich der Varianzen am Beispiel E[X²]"}

<!--en:e8a496ed-->
[Exakte Berechnung der Momente]{.kicker}

<!--en:4f532f01-->
- **Exakte Varianzen für $\ell(\theta) = \mathbb{E}[X^2]$ bei $\theta = 1.5$:**
  - *Pfadweise ($2X$):* $\operatorname{Var} = 4 \implies \sigma = 2.000$.
  - *Score mit Baseline ($b^\star = \theta^2+3$):* $\operatorname{Var} = 8\theta^2+6 \implies \sigma = 4.899$.
  - *REINFORCE ohne Baseline ($X^2 Z$):* $\operatorname{Var} = \theta^4+14\theta^2+15 \implies \sigma = 7.181$ ($12.9\times$ höhere Varianz).

<!--en:86fef3fb-->
::: {.slide title="Monte-Carlo-Vergleich von pfadweisem Schätzer und REINFORCE"}

<!--en:a55ebaca-->
[Empirischer Vergleich]{.kicker}

<!--en:026d42ef-->
Eine Simulation mit $n = 200{,}000$ Samples bei $\theta = 1.5$ bestätigt die Erwartungstreue aller drei Schätzer für $\ell'(1.5) = 3.0000$. Auch die theoretischen Standardabweichungen $2.000$, $7.181$ und $4.899$ zeigen sich in den Ergebnissen:

<!--en:038e4738-->
::: {.slide title="Wann der Score-Funktionsschätzer bei rasch oszillierendem Verlust besser abschneidet"}

<!--en:a0923fa5-->
[Hohe Frequenzen]{.kicker}

<!--en:bdacecc5-->
Hat der pfadweise Schätzer immer eine kleinere Varianz als der Score-Funktionsschätzer? **Nein.** Für den oszillierenden Verlust $g(x) = \sin(\omega x)$ gilt $\frac{d}{d\theta}\mathbb{E}[\sin(\omega X)] = \omega\cos(\omega\theta)e^{-\omega^2/2}$:

<!--en:164bbae1-->
- **Die pfadweise Varianz skaliert mit $|g'|^2 = O(\omega^2)$:** Eine Ziehung liefert $\omega\cos(\omega X)$. Die Standardabweichung wächst linear mit der Frequenz und beträgt $\approx \omega/\sqrt{2}$.

<!--en:6e3c9384-->
- **$|g|^2 \le 1$ beschränkt die Varianz des Score-Funktionsschätzers:** Eine Ziehung liefert $\sin(\omega X)\,Z$. Wegen $|\sin(\omega X)| \le 1$ beträgt das zweite Moment höchstens $\mathbb{E}[Z^2] = 1$, unabhängig davon, wie groß $\omega$ wird.

<!--en:2376a307-->
::: {.slide title="Wahl eines Gradientenschätzers"}

<!--en:5d233c39-->
[Eigenschaften im Vergleich]{.kicker}

<!--en:0498280d-->
| Eigenschaft oder Voraussetzung | Score-Funktion (REINFORCE) | Pfadweise (Reparametrisierung) | Relaxation oder Straight-Through (z. B. Gumbel-Softmax) |

<!--en:e17bf90c-->
| **Grundformel** | $g(X)\,\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)$ | $\nabla_{\boldsymbol{\theta}}g(T_{\boldsymbol{\theta}}(\mathbf{z}))$ $= \mathbf{J}_{T}^\top \nabla_x g$ | Glatte Ersatzfunktion $z_\tau = \sigma((\theta+\zeta)/\tau)$ differenzieren |

<!--en:f363d4a4-->
| **Muss $g$ differenzierbar sein?** | **Nein** (auch für Sprungfunktionen und Belohnungen aus einer Blackbox) | **Ja** ($g$ muss differenzierbar bzw. Lipschitz-stetig sein) | **Ja** (der Ersatzverlust muss differenzierbar sein) |

<!--en:7d373e79-->
| **Für diskretes $X$ geeignet?** | **Ja** (Bernoulli-verteilte und kategoriale Aktionen) | **Nein** (kein stetiger Pfad $T_{\boldsymbol{\theta}}(\mathbf{z})$) | **Ja** (nähert die diskrete Wahl bei Temperatur $\tau$ an) |

<!--en:3c989dd4-->
| **Verzerrung** | **Erwartungstreu** (bei festem Träger) | **Erwartungstreu** (für glattes $g$) | **Verzerrt** (bei $\tau \to 0$ sinkt die Verzerrung, die Varianz wächst unbegrenzt) |

<!--en:30aea8c3-->
| **Varianz bei einer Ziehung** | Skaliert mit $\|g(X)\|^2$ (benötigt Baseline $b$) | Skaliert mit $\|\nabla_x g(X)\|^2$ (klein für glattes $g$) | Hängt von der Temperatur $\tau$ ab (Abwägung zwischen Verzerrung und Varianz) |

<!--en:62658bfe-->
::: {.slide title="Verständnisfragen zum Differenzieren von Erwartungswerten"}

<!--en:b6d9a92f-->
[Wiederholung und Übung]{.kicker}

<!--en:02d1b095-->
Drei Fragen zur Leibniz-Regel, zur Score-Funktion und zum pfadweisen Schätzer bearbeiten:

<!--en:14ca7ad6-->
1. **Verzerrung bei beweglichem Träger:** Für $X \sim \mathrm{Uniform}(0, \theta)$ mit Dichte $p_\theta(x) = 1/\theta$ auf $[0, \theta]$ ist der Score $\partial_\theta \log(1/\theta) = -1/\theta$. Was liefert der Score-Funktionsschätzer $X^2 \partial_\theta \log p_\theta(X)$ im Mittel für $\ell(\theta) = \mathbb{E}[X^2] = \theta^2/3$, und warum verfehlt er $\ell'(\theta) = 2\theta/3$?

<!--en:fae67e1c-->
2. **Reparametrisierung der Gleichverteilung auf $(0, \theta)$:** Wie lässt sich $X \sim \mathrm{Uniform}(0, \theta)$ mithilfe von $U \sim \mathrm{Uniform}(0, 1)$ so reparametrisieren, dass der pfadweise Schätzer die Ableitung $\ell'(\theta) = 2\theta/3$ ohne Randterm erwartungstreu schätzt?

<!--en:5999e77b-->
3. **Varianzvergleich in hohen Dimensionen:** Es gelte $g(\mathbf{x}) = \|\mathbf{x}\|_2^2$ für $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\theta}, \mathbf{I}_d)$. Wie skalieren die Größen von $g(\mathbf{X})$ (bei REINFORCE) und $\nabla_{\mathbf{x}} g(\mathbf{X}) = 2\mathbf{X}$ (beim pfadweisen Schätzer) mit der Dimension $d$?

<!--en:b908ffa6-->
::: {.slide title="Verständnisfragen zum Differenzieren von Erwartungswerten (Fortsetzung)"}

<!--en:b6d9a92f-2-->
[Wiederholung und Übung]{.kicker}

<!--en:557b98b6-->
**Antworten:**

<!--en:9f92ab87-->
1. Der Erwartungswert beträgt $\mathbb{E}[X^2(-1/\theta)] = -\frac{1}{\theta}\frac{\theta^2}{3} = -\frac{\theta}{3}$. Das entspricht dem bloßen Vertauschen im Inneren, $\int_0^\theta x^2 \partial_\theta(1/\theta)\,dx$. Dabei fehlt der Term $f(\theta, \theta)b'(\theta) = \theta^2(1/\theta)(1) = +\theta$ für die obere Grenze aus der Leibniz-Regel ($-\frac{\theta}{3} + \theta = +\frac{2\theta}{3}$).

<!--en:06a74c08-->
2. Mit $X = T_\theta(U) = \theta U$ ist der Träger $U \in [0, 1]$ fest. Der pfadweise Schätzer ist $\partial_\theta(\theta U)^2 = 2\theta U^2$ und hat den Erwartungswert $2\theta\mathbb{E}[U^2] = 2\theta(\tfrac{1}{3}) = \frac{2\theta}{3}$.

<!--en:0bc27b7f-->
3. Der skalare Verlust $g(\mathbf{X}) = \sum_{j=1}^d X_j^2 = O(d)$ multipliziert alle $d$ Koordinaten des Score-Vektors. Daher wächst die Varianz von REINFORCE pro Koordinate wie $O(d^2)$ (mit Baseline wie $O(d)$). Die pfadweise Ableitung für die Koordinate $k$ beträgt dagegen $2X_k$ und hat eine Varianz der Ordnung $O(1)$, unabhängig von $d$.

<!--en:f477f248-->
::: {.slide title="Zusammenfassung der Regeln für Gradienten von Erwartungswerten"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:b61503d9-->
| Regel oder Schätzer | Mathematische Identität | Voraussetzungen und Verhalten der Varianz |

<!--en:ed8ed366-->
| **Vertauschen bei festen Grenzen** | $\frac{d}{d\theta}\int f(x,\theta)\,dx = \int \frac{\partial f}{\partial \theta}(x,\theta)\,dx$ | Feste Grenzen, stetige partielle Ableitung $\partial_\theta f$ und integrierbare Majorante |

<!--en:03bb6f7f-->
| **Leibniz-Regel mit Randtermen** | $\frac{d}{d\theta}\int_{a(\theta)}^{b(\theta)} f\,dx = \int_a^b \partial_\theta f\,dx + f(b,\theta)b' - f(a,\theta)a'$ | Erforderlich, wenn Integrationsgrenzen oder Träger von $\theta$ abhängen |

<!--en:165e5227-->
| **Log-Integral-Exp** | $\frac{d}{d\theta}\log\int e^{\theta x}p(x)\,dx = \mathbb{E}_{\tilde{p}_\theta}[X]$ ($\tilde{p}_\theta \propto e^{\theta x}p$) | Kontinuierliches Gegenstück zu $\nabla\operatorname{LSE}(\mathbf{z}) = \operatorname{softmax}(\mathbf{z})$ |

<!--en:e8962950-->
| **Score-Funktion (REINFORCE)** | $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{p_{\boldsymbol{\theta}}}\bigl[(g(X)-b)\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)\bigr]$ | Für unstetiges $g$ und diskretes $X$ geeignet. Die Varianz skaliert mit $\|g-b\|^2$ |

<!--en:0fd2db34-->
| **Optimale Baseline $b^\star$** | $b^\star = \mathbb{E}[g(X)s_\theta(X)^2] / \mathbb{E}[s_\theta(X)^2]$ | Erwartungstreu, da auf festem Träger $\mathbb{E}[\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(X)] = \mathbf{0}$ gilt |

<!--en:763f7804-->
| **Pfadweise (reparametrisiert)** | $\nabla_{\boldsymbol{\theta}}\mathbb{E}_{p_{\boldsymbol{\theta}}}[g(X)] = \mathbb{E}_{\mathbf{z}\sim q}\bigl[\nabla_{\boldsymbol{\theta}}g(T_{\boldsymbol{\theta}}(\mathbf{z}))\bigr]$ | Erfordert glattes $g$ und stetiges $X = T_{\boldsymbol{\theta}}(\mathbf{z})$. Die Varianz skaliert mit $\|\nabla g\|^2$ |
