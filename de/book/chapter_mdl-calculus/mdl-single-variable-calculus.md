<!--en:29e87049-->
# Differentialrechnung in einer Variablen

<!--en:0e6811d7-->
Alle Modelle im Grundlagenteil haben wir mit Gradientenabstieg (engl. gradient descent) trainiert. Bei jedem Schritt bestimmt der Optimierer den Verlust und seine Steigung an den aktuellen Gewichten, verschiebt die Gewichte ein kleines Stück bergab und wiederholt den Vorgang. Da er den Verlust nur an der aktuellen Stelle auswertet, beruht jeder Schritt auf einer lokalen Vorhersage. Wir nehmen an, dass die Steigung an dieser Stelle den Verlust auch ein kleines Stück weiter noch beschreibt. Wie genau ist diese Vorhersage, und wie groß darf der Schritt sein, damit wir uns auf sie verlassen können?

<!--en:18aef9f4-->
Schon ein einziges Gewicht zeigt, wie die Antwort aussieht. Betrachten wir ein Trainingsbeispiel, ein skalares Gewicht $w$ und den logistischen Verlust $\ell(w)=\log(1+e^{-w})$. Bei $w=0$ beträgt der Verlust $0.693147$ und seine Steigung $-\tfrac12$. Verschieben wir das Gewicht auf $w=0.1$, so sagt die Gerade durch den Kurvenpunkt bei $w=0$ mit dieser Steigung einen Verlust von $0.693147-0.05=0.643147$ voraus. Tatsächlich beträgt der Verlust $0.644397$. Die lineare Vorhersage weicht also nur um $0.00125$ ab. Bemerkenswert ist, wie sich dieser kleine Unterschied zusammensetzt. Auf die angegebenen Stellen genau beträgt er $\tfrac12\cdot\tfrac14\cdot0.1^2$. Dabei misst der Faktor $\tfrac14$, wie stark sich der Graph von $\ell$ bei $w=0$ krümmt. Die Steigung liefert die Vorhersage. Die Krümmung bestimmt, wie schnell deren Fehler mit der Schrittweite wächst. Sobald wir die Ableitungsregeln kennen, leiten wir beide Konstanten her.

<!--en:388165f1-->
In diesem Abschnitt gewinnen wir aus der ersten Ableitung ein lokales Geradenmodell. Mit der zweiten Ableitung erweitern wir es zu einer Parabel, die auch die Krümmung erfasst. Dazu dienen drei Ergebnisse. Der Mittelwertsatz (engl. mean value theorem) verknüpft die Steigung an einzelnen Stellen mit der Änderung über ein Intervall. Das Lagrange-Restglied (engl. Lagrange remainder) liefert Fehlerschranken für die Gerade und die Parabel. Das Abstiegslemma (engl. descent lemma) liefert aus einer Schranke für die Krümmung eine garantiert sichere Schrittweite für den Gradientenabstieg. Wir betrachten außerdem das Newton-Verfahren und den Knick von $\mathrm{ReLU}$. Dort gibt es keine Tangente, doch die Backpropagation muss trotzdem eine Zahl zurückgeben. Hier untersuchen wir diese Ideen für eine Variable. Die zugehörige Optimierungstheorie für mehrere Variablen behandeln wir in :numref:`chap_mdl-optimization`.

<!--en:983f1702-->
In :numref:`sec_calculus` haben wir die Ableitung eingeführt, die Ableitungsregeln zusammengestellt und das automatische Differenzieren (engl. automatic differentiation) als praktisches Werkzeug eingesetzt. Hier wiederholen wir die Regeln kurz und konzentrieren uns auf die Beweise, ihre geometrische Deutung und die Fehlerschranken. Wenn Ihnen die Ableitungsregeln bereits vertraut sind, können Sie direkt bei :numref:`sec_mdl-mvt` weiterlesen. Eine Variable ist ein natürlicher Ausgangspunkt, denn jede Verlustfunktion (engl. loss function) mehrerer Variablen enthält diesen Fall. Halten wir alle Gewichte von $\ell(w_1,\ldots,w_n)$ bis auf $w_i$ fest, so erhalten wir einen eindimensionalen Schnitt. Solche Schnitte untersucht die Differentialrechnung in einer Variablen.

<!--en:2a77d3bc-->
Als durchgängiges Beispiel dient uns in diesem Abschnitt die *Softplus*-Funktion $s(x)=\log(1+e^{x})$. Sie ist der eindimensionale Schnitt $\log(e^0+e^x)$ der Log-Sum-Exp-Funktion, die wir in diesem Kapitel immer wieder verwenden. Für den logistischen Verlust von oben gilt damit $\ell(w)=s(-w)$.

<!--en:b08e5552-->
## Die Ableitung

<!--en:5e2b1fe9-->
### Der Differenzenquotient

<!--en:af2dbbe1-->
Vergrößern wir den Graphen einer glatten Funktion $f$ in der Nähe einer Stelle $x$ hinreichend stark, so lässt sich die Kurve nicht mehr von einer Geraden unterscheiden (:numref:`fig_mdl-zoom-sequence`). Die Steigung dieser Geraden ist die Ableitung.

<!--en:5a188680-->
![Die Kurve $\sin(x^x)$ nahe $x_0=2$ in Ausschnitten mit den Halbbreiten $0.6$, $0.06$ und $0.006$. Je kleiner der Ausschnitt wird, desto weniger lässt sich die Kurve (durchgezogen) von ihrer Tangente (gestrichelt) unterscheiden.](../img/mdl-cal-zoom-sequence.svg)

<!--en:05e03581-->
Um diese Steigung algebraisch zu erfassen, gehen wir wie in :numref:`sec_calculus` von der *Sekante* durch $(x, f(x))$ und $(x+\epsilon, f(x+\epsilon))$ aus. Ihre Steigung ist der *Differenzenquotient* $\bigl(f(x+\epsilon)-f(x)\bigr)/\epsilon$. Geht $\epsilon$ gegen null, so dreht sich die Sekante auf die Tangente zu (:numref:`fig_mdl-secant-to-tangent`).

<!--en:19e73bf2-->
![Der Differenzenquotient ist die Steigung der Sekante durch $(x, f(x))$ und $(x+\epsilon, f(x+\epsilon))$. Für $\epsilon \to 0$ nähert sich diese Steigung der Tangentensteigung $f'(x)$.](../img/mdl-cal-secant-to-tangent.svg)

<!--en:35157219-->
Wie klein muss $\epsilon$ sein, damit die Sekantensteigung die Tangentensteigung zuverlässig annähert? Dafür gibt es keine allgemeine Schwelle. Sie hängt ganz von der Funktion ab. Betrachten wir das Polynom $f(x) = x^2 + 1701(x-4)^3$. Seine tatsächliche Steigung bei $x=4$ beträgt $8$. Beobachten wir, wie langsam sich die Sekantensteigung für abnehmendes $\epsilon$ diesem Wert nähert.

<!--en:46ce501a-->
Bei $\epsilon=0.1$ beträgt die Sekantensteigung $25.11$, mehr als das Dreifache des Grenzwerts. Die ausgegebenen Werte liegen erst dann nahe bei $8$, wenn $\epsilon$ auf $0.001$ gesunken ist. Ausmultiplizieren liefert die Erklärung. Es gilt $f(4+\epsilon)-f(4)=8\epsilon+\epsilon^2+1701\epsilon^3$. Der Differenzenquotient beträgt also $8+\epsilon+1701\epsilon^2$. Der Beitrag des kubischen Terms beträgt somit $17$ für $\epsilon=0.1$, aber nur $1.7\times10^{-7}$ für $\epsilon=10^{-5}$. Der große Koeffizient verlangsamt die Annäherung an den Grenzwert, ohne ihn zu ändern. Automatisches Differenzieren umgeht das Problem vollständig. Es benötigt kein $\epsilon$, denn es wendet die Ableitungsregeln direkt an (:numref:`sec_mdl-matrix-calculus-autodiff`).

<!--en:6ce719b2-->
### Ableitung und lineare Approximation

<!--en:93c5ea4e-->
Die **Ableitung** von $f$ an der Stelle $x$ ist der Grenzwert der Sekantensteigungen,

<!--en:24672212-->
sofern dieser Grenzwert existiert. In diesem Fall heißt $f$ an der Stelle $x$ *differenzierbar*. Für das kubische Polynom von oben gilt $f'(4)=8$. Wir schreiben $f'$ oder $\frac{df}{dx}$. Für die Ableitung eines längeren Ausdrucks verwenden wir $\frac{d}{dx}[\ldots]$. Der Grenzwert kann auch fehlen, wie bei $|x|$ an der Stelle $0$. Auf solche Knicke kommen wir in :numref:`sec_mdl-tangent-fails` zurück.

<!--en:314fa8f1-->
Formen wir diese Definition um. Sei $r(\epsilon)=f(x+\epsilon)-f(x)-\epsilon f'(x)$ der Fehler der Vorhersage mit der Geraden. Aus dem Grenzwert :eqref:`eq_mdl-der_def` folgt $r(\epsilon)/\epsilon\to0$. Wir schreiben $o(\epsilon)$ für jede Größe $r(\epsilon)$, für die $r(\epsilon)/\epsilon\to0$ bei $\epsilon\to0$ gilt. Damit erhalten wir die **lineare Approximation**

<!--en:37f67a74-->
In Worten ändert ein kleiner Schritt $\epsilon$ die Ausgabe um $\epsilon f'(x)$, bis auf einen Fehler, der schneller gegen null geht als $\epsilon$. Auch die Umkehrung gilt. Gilt $f(x+\epsilon)=f(x)+a\epsilon+o(\epsilon)$ für eine Konstante $a$, so dividieren wir durch $\epsilon$ und bilden den Grenzwert für $\epsilon\to0$. Daraus folgt, dass $f$ bei $x$ differenzierbar ist und $f'(x)=a$ gilt. Die Entwicklung :eqref:`eq_mdl-small_change` und der Grenzwert :eqref:`eq_mdl-der_def` enthalten also dieselbe Information. Auf diese Äquivalenz stützen sich unsere Beweise der Summen- und der Kettenregel.

<!--en:c510820e-->
Als Funktion von $x+\epsilon$ betrachtet ist die Gerade $f(x)+\epsilon f'(x)$ die *Tangente* an den Graphen bei $x$. Sie ist die Grenzlage der sich drehenden Sekanten. Mit dieser Geraden haben wir den logistischen Verlust bei $w=0.1$ vorhergesagt. Die Abweichung von $0.00125$ war ihr Restglied der Ordnung $o(\epsilon)$. Für sich genommen sagt $o(\epsilon)$ wenig aus. Ein Fehler der Ordnung $o(\epsilon)$ kann nur geringfügig schneller gegen null gehen als $\epsilon$. Hier nahm die Abweichung jedoch wie $\epsilon^2$ ab. Wie wir in :numref:`subsec_mdl-taylor-series` sehen, gilt diese quadratische Fehlerabschätzung für jede zweimal differenzierbare Funktion.

<!--en:253cc5ce-->
## Berechnung von Ableitungen

<!--en:4ab56f4d-->
In der Praxis berechnen wir Ableitungen nicht über Grenzwerte. Wie in :numref:`sec_calculus` kombinieren wir eine kurze Tabelle elementarer Ableitungen mit Regeln für Summen, Produkte und Verkettungen. Hier beweisen wir jede Regel direkt mit der linearen Approximation. Von den Tabelleneinträgen leiten wir die Formeln für die Exponentialfunktion und den Logarithmus sowie die Potenzregel her. Die Ableitungen von Sinus und Kosinus setzen wir aus der Einführung in die Analysis voraus. Dasselbe gilt für die Existenz einer Basis, deren Exponentialfunktion die Steigung $1$ an der Stelle $0$ hat.

<!--en:24de15c9-->
### Häufig verwendete Ableitungen

<!--en:d440fdad-->
Die folgenden Formeln gelten überall dort, wo ihre rechten Seiten definiert sind. Die Potenzregel gilt für reelles $n$ bei $x>0$ und für ganzzahliges $n$ überall dort, wo $x^n$ definiert ist. Die Regel für den Logarithmus gilt für $x>0$.

<!--en:b1b3a2e8-->
* **Ableitung einer Konstanten.** $\frac{d}{dx}c = 0$.

<!--en:fa15ae50-->
* **Ableitung linearer Funktionen.** $\frac{d}{dx}(ax) = a$.

<!--en:e4eb0ccb-->
* **Potenzregel.** $\frac{d}{dx}x^n = nx^{n-1}$.

<!--en:6755a5da-->
* **Ableitung der Exponentialfunktion.** $\frac{d}{dx}e^x = e^x$.

<!--en:c10bac87-->
* **Ableitung des Logarithmus.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

<!--en:0e7c9f1f-->
* **Ableitung des Sinus.** $\frac{d}{dx}\sin(x) = \cos(x)$.

<!--en:f37cb09c-->
* **Ableitung des Kosinus.** $\frac{d}{dx}\cos(x) = -\sin(x)$.

<!--en:fb8b2403-->
Die Formel für die Exponentialfunktion beruht auf einer kurzen Beobachtung, die zugleich die Besonderheit der Basis $e$ erklärt. Für jedes $a>0$ lässt sich der Differenzenquotient von $a^x$ in zwei Faktoren zerlegen,

<!--en:b70945e6-->
wobei der zweite Faktor der Differenzenquotient an der Stelle $0$ ist. Ist also $a^x$ bei $0$ differenzierbar mit Steigung $c_a$, so gilt $\frac{d}{dx}a^x=c_a\,a^x$ für jedes $x$. Die Steigung einer Exponentialfunktion ist somit stets proportional zu ihrem Funktionswert. Zu jeder Basis gehört eine einzige Proportionalitätskonstante. Numerisch erhalten wir bei $\epsilon=10^{-6}$ den Wert $(2^\epsilon-1)/\epsilon=0.69315$ für die Basis $2$ und $(10^\epsilon-1)/\epsilon=2.30259$ für die Basis $10$. Die Zahl $e$ ist als die einzige Basis definiert, deren Konstante $1$ beträgt. Daraus folgt unmittelbar $\frac{d}{dx}e^x=e^x$. Sobald wir die Ableitung des Logarithmus kennen, liefert die Kettenregel $c_a$ für jede andere Basis.

<!--en:41076900-->
### Rechenregeln für Ableitungen

<!--en:cb555c1f-->
Aus den Funktionen der Tabelle lassen sich durch Summen, Produkte, Quotienten und Verkettungen Verlustfunktionen bilden. Seien $g$ und $h$ an der Stelle $x$ differenzierbar. Für die Kettenregel setzen wir voraus, dass $h$ bei $x$ und $g$ bei $h(x)$ differenzierbar ist. Dann gelten die folgenden Regeln.

<!--en:33d9b5f5-->
* **Summenregel.** $\frac{d}{dx}\bigl(g(x) + h(x)\bigr) = g'(x) + h'(x)$.

<!--en:891a5a0e-->
* **Produktregel.** $\frac{d}{dx}\bigl(g(x)\, h(x)\bigr) = g(x)h'(x) + g'(x)h(x)$.

<!--en:2c86d27a-->
* **Quotientenregel.** $\frac{d}{dx}\frac{g(x)}{h(x)} = \frac{g'(x)h(x) - g(x)h'(x)}{h(x)^2}$ für $h(x)\neq 0$.

<!--en:e1d50dcb-->
* **Kettenregel.** $\frac{d}{dx}g(h(x)) = g'(h(x))\, h'(x)$.

<!--en:5906e947-->
Summen- und Produktregel sind leicht zu zeigen. Nur die Kettenregel erfordert besondere Sorgfalt, denn der naheliegende Ansatz dividiert durch $h(x+\epsilon)-h(x)$. Dieser Ausdruck kann selbst für $\epsilon\neq 0$ null sein. Alle Beweise beruhen auf einer grundlegenden Tatsache. **Eine an der Stelle $x$ differenzierbare Funktion ist an der Stelle $x$ stetig.** Denn $g(x+\epsilon)-g(x)=\epsilon\cdot\frac{g(x+\epsilon)-g(x)}{\epsilon}$ strebt gegen $0\cdot g'(x)=0$ für $\epsilon\to 0$.

<!--en:042af5eb-->
**Summenregel.** Addieren wir die linearen Approximationen für $g$ und $h$, erhalten wir

<!--en:515deedf-->
denn wenn zwei Größen schneller gegen null gehen als $\epsilon$, geht auch ihre Summe schneller gegen null als $\epsilon$. Nach der Umkehrung der linearen Approximation ist der Koeffizient von $\epsilon$ die Ableitung. Änderungen erster Ordnung addieren sich also. $\blacksquare$

<!--en:88994f30-->
**Produktregel.** Addieren und subtrahieren wir $g(x+\epsilon)h(x)$ im Zähler, so zerfällt der Differenzenquotient des Produkts in zwei bekannte Quotienten:

<!--en:3770ad18-->
Für $\epsilon\to0$ folgt aus der Stetigkeit $g(x+\epsilon)\to g(x)$. Also konvergiert die rechte Seite gegen $g(x)h'(x)+h(x)g'(x)$. $\blacksquare$

<!--en:8806c50a-->
**Kettenregel.** Die Division durch $h(x+\epsilon)-h(x)$ lässt sich vermeiden, indem wir die Änderung der inneren Funktion direkt in die lineare Approximation der äußeren Funktion einsetzen. Da $g$ bei $h(x)$ differenzierbar ist, gilt für den normierten Fehler Folgendes. Definieren wir $\rho(t)=\frac{g(h(x)+t)-g(h(x))}{t}-g'(h(x))$ für $t\neq0$ und ergänzen $\rho(0)=0$, so geht dieser Fehler gegen $0$ für $t\to0$. Für jedes $t$, auch für $t=0$, können wir daher schreiben:

<!--en:544fba74-->
Setzen wir nun $t=h(x+\epsilon)-h(x)=\epsilon h'(x)+o(\epsilon)$. Wegen der Stetigkeit von $h$ gilt $t\to0$ für $\epsilon\to0$ und damit auch $\rho(t)\to0$. Setzen wir dieses $t$ ein, erhalten wir

<!--en:1dac4534-->
denn alle Terme außer $g'(h(x))\,\epsilon h'(x)$ gehen schneller gegen null als $\epsilon$. Die Umkehrung der linearen Approximation schließt den Beweis ab. $\blacksquare$

<!--en:2a9c5998-->
Bei der Verkettung multiplizieren sich die Änderungsraten. Wenden wir die Kettenregel Schicht für Schicht auf ein neuronales Netz an, entsteht ein Produkt lokaler Steigungen. Genau dieses Produkt wertet die Backpropagation effizient aus. Die Quotientenregel folgt unmittelbar aus der Produkt- und der Kettenregel für $g\cdot h^{-1}$.

<!--en:7ccbafce-->
### Umkehrfunktionen

<!--en:656b577f-->
Mit der Kettenregel lässt sich auch die Ableitung einer Funktion bestimmen, die eine andere rückgängig macht. Eine Funktion $g$ mit $g(f(x))=x$ für alle $x$ in der Umgebung einer Stelle heißt dort *Umkehrfunktion* von $f$. Nehmen wir an, eine solche Funktion $g$ existiert und ist bei $f(x)$ differenzierbar. Nach dem Satz über die Umkehrfunktion :cite:`Rudin.1976`, den wir ohne Beweis verwenden, gilt beides, wenn $f'$ in einer Umgebung von $x$ stetig und von null verschieden ist. Leiten wir beide Seiten von $g(f(x))=x$ mit der Kettenregel ab, so erhalten wir

<!--en:65f326dc-->
Die Steigung der Umkehrfunktion ist also der Kehrwert der Steigung von $f$ an der zugehörigen Stelle. Geometrisch entsteht der Graph von $g$ aus dem Graphen von $f$ durch Spiegelung an der Winkelhalbierenden. Bei einer Geraden vertauscht diese Spiegelung Zähler und Nenner des Steigungsdreiecks.

<!--en:ad47d45a-->
Da der Logarithmus die Umkehrfunktion von $e^x$ ist, liefert diese Regel $\log'(y)=1/e^{\log y}=1/y$, den fünften Eintrag unserer Tabelle. Die Bedingung $f'(x)\neq0$ ist unverzichtbar. Für $f(x)=x^3$ gilt $f'(0)=0$, sodass die Formel bei $y=0$ versagt. Tatsächlich ist die Umkehrfunktion $x^{1/3}$ bei $0$ nicht differenzierbar. Der Differenzenquotient $\epsilon^{1/3}/\epsilon=\epsilon^{-2/3}$ wächst unbegrenzt, und der Graph hat dort eine senkrechte Tangente.

<!--en:3179a34a-->
Mit dem Logarithmus liefert die Kettenregel auch die beiden noch ausstehenden Formeln der Tabelle. Schreiben wir $a^x=e^{x\log a}$, so erhalten wir $\frac{d}{dx}a^x=(\log a)\,a^x$ und kennen damit die Konstante $c_a=\log a$. Tatsächlich sind $\log 2=0.69315$ und $\log 10=2.30259$ die Konstanten, die wir oben numerisch bestimmt haben. Entsprechend schreiben wir $x^n=e^{n\log x}$ für $x>0$ und erhalten $\frac{d}{dx}x^n=e^{n\log x}\cdot\frac{n}{x}=nx^{n-1}$ für jedes reelle $n$.

<!--en:09a05b26-->
### Softplus, Sigmoidfunktion und Logit

<!--en:00dfccbc-->
Wenden wir diese Regeln auf unser durchgängiges Beispiel $s(x)=\log(1+e^x)$ an. Die Kettenregel mit der äußeren Funktion $\log$ und der inneren Funktion $1+e^x$ ergibt

<!--en:0763d7d4-->
Die Ableitung der Softplus-Funktion ist also die *Sigmoidfunktion* $\sigma$. Wenden wir die Kettenregel erneut auf $\sigma(x)=(1+e^{-x})^{-1}$ an, diesmal mit der äußeren Funktion $u^{-1}$ und der inneren Funktion $1+e^{-x}$ mit Steigung $-e^{-x}$, so erhalten wir

<!--en:7d261c14-->
denn es gilt $1-\sigma(x)=e^{-x}/(1+e^{-x})$. Da ein Produkt $p(1-p)$ bei $p=\tfrac12$ maximal ist, hat die Sigmoidfunktion am Ursprung die größte Steigung. Dort gilt $\sigma(0)=\tfrac12$ und $\sigma'(0)=\tfrac14$. Die Umkehrfunktion von $\sigma$ ist die *Logit*-Funktion $\mathrm{logit}(p)=\log\frac{p}{1-p}$. Die Ableitungsregel für Umkehrfunktionen liefert unmittelbar $\mathrm{logit}'(p)=1/\sigma'(\mathrm{logit}\,p)=\frac{1}{p(1-p)}$. Bei $p=\tfrac12$ ergibt das $4=1/\tfrac14$, den Kehrwert der Steigung der Sigmoidfunktion.

<!--en:2e93abc8-->
Nun können wir die beiden Konstanten aus dem einleitenden Beispiel für den logistischen Verlust überprüfen. Wegen $\ell(w)=s(-w)$ liefert die Kettenregel $\ell'(w)=-\sigma(-w)$, also $\ell'(0)=-\tfrac12$. Erneutes Ableiten ergibt $\ell''(w)=\sigma(-w)\bigl(1-\sigma(-w)\bigr)$ und damit $\ell''(0)=\tfrac14$, die Krümmung am Ursprung. Prüfen wir numerisch, ob die Abweichung zwischen $\ell(\epsilon)$ und der linearen Approximation $\ell(0)-\tfrac12\epsilon$ tatsächlich wie $\epsilon^2$ mit dem führenden Koeffizienten $\tfrac12\ell''(0)=\tfrac18$ abnimmt.

<!--en:e2bbc221-->
Tatsächlich sinkt die Abweichung auf ein Hundertstel, wenn wir $\epsilon$ durch zehn teilen. Der Quotient $\text{gap}/\epsilon^2$ nähert sich $0.125=\tfrac12\ell''(0)$. Der Fehler der linearen Approximation ist also von zweiter Ordnung, und die Krümmung bestimmt seinen führenden Koeffizienten. In der ersten Zeile steht allerdings $0.1249$ statt $0.1250$. Die Abweichung beträgt nicht ganz $\epsilon^2/8$. In :numref:`subsec_mdl-taylor-series` bestimmen wir den Term höherer Ordnung, der diesen kleinen Unterschied erklärt.

<!--en:71a2fedb-->
Bisher bezog sich jede Rechnung auf eine infinitesimal kleine Umgebung einer einzigen Stelle. Was die Steigung über das Verhalten einer Funktion auf einem ganzen Intervall aussagt, ist eine andere Frage.

<!--en:72edb73a-->
## Der Mittelwertsatz

<!--en:c50150f1-->
Bei der Optimierung stützen wir uns regelmäßig auf zwei anschauliche Vorstellungen. Eine positive Steigung bedeutet, dass die Funktion wächst, und eine Stelle mit Steigung null kommt als Minimum infrage. Der Gradientenabstieg beruht auf beiden Vorstellungen. In beiden Fällen schließen wir aus der Ableitung an einzelnen Stellen auf das Verhalten von $f$ über ein Intervall. Keine der beiden Aussagen folgt unmittelbar aus der Definition der Ableitung. Wir beweisen zuerst die zweite, einfachere Aussage.

<!--en:66f1e951-->
**Lemma (Fermat).** *Hat $f$ im Inneren eines Intervalls an der Stelle $\xi$ ein lokales Maximum oder ein lokales Minimum und ist die Funktion bei $\xi$ differenzierbar, so gilt $f'(\xi)=0$.*

<!--en:1101632f-->
**Beweis.** An einem lokalen Maximum gilt $f(\xi + \epsilon) - f(\xi) \le 0$ für alle hinreichend kleinen $\epsilon$ beiderlei Vorzeichens. Der Differenzenquotient ist daher $\le 0$ für $\epsilon > 0$ und $\ge 0$ für $\epsilon < 0$. Da $f$ bei $\xi$ differenzierbar ist, müssen beide einseitigen Grenzwerte gleich $f'(\xi)$ sein. Die einzige Zahl, die zugleich $\le 0$ und $\ge 0$ ist, ist null. Für ein lokales Minimum gilt derselbe Beweis mit umgekehrten Ungleichheitszeichen. $\blacksquare$

<!--en:d2e325ce-->
Für die erste Vorstellung brauchen wir ein stärkeres Werkzeug, den Mittelwertsatz. Jede Fehlerschranke im weiteren Verlauf dieses Abschnitts erhalten wir, indem wir diesen Satz auf eine geeignet gewählte Hilfsfunktion anwenden. Geometrisch (:numref:`fig_mdl-mvt`) besagt er, dass die Sehne zwischen den Endpunkten des Graphen über $[a,b]$ parallel zur Tangente an einer Stelle im Inneren ist.

<!--en:987cf67a-->
![Auf dem Intervall von $a$ bis $b$ ist die Steigung der Sehne zwischen den Endpunkten gleich der mittleren Änderungsrate. Nach dem Mittelwertsatz gibt es eine Stelle $\xi$ im Inneren, an der die Tangente dieselbe Steigung hat und damit parallel zur Sehne verläuft.](../img/mdl-cal-mvt.svg)

<!--en:960c50e5-->
Bei einem einfachen Beispiel wie $f(x)=x^2$ auf $[0,2]$ hat die Sehne die Steigung $(4-0)/2=2$. Die Tangentensteigung $f'(\xi)=2\xi$ stimmt am Mittelpunkt $\xi=1$ mit ihr überein. Der Satz verlangt tatsächlich Differenzierbarkeit an jeder Stelle im Inneren. Für die Betragsfunktion $|x|$ auf $[-1,1]$ gilt $f(-1)=f(1)$. Ihre Sehne ist also waagerecht, doch an keiner Stelle im Inneren beträgt die Steigung $0$. Links ist die Steigung $-1$, rechts $+1$, und bei $0$ ist sie undefiniert.

<!--en:a559f814-->
**Proposition (Mittelwertsatz).** *Ist $f$ auf $[a,b]$ stetig und auf $(a,b)$ differenzierbar, so gibt es eine Stelle $\xi \in (a,b)$ mit*

<!--en:4bbc1d15-->
**Beweis.** Betrachten wir zunächst den Fall $f(a) = f(b)$, also den Satz von Rolle. Wir verwenden ein grundlegendes Ergebnis der reellen Analysis, den **Satz vom Maximum und Minimum**. Eine stetige Funktion auf einem abgeschlossenen, beschränkten Intervall nimmt sowohl ein Maximum als auch ein Minimum an :cite:`Rudin.1976`. Liegen beide Extremwerte an den Endpunkten, so ist $f$ konstant und es gilt $f' \equiv 0$ im gesamten Inneren. Andernfalls liegt mindestens ein Extremum an einer Stelle $\xi$ im Inneren. Dort gilt nach dem Lemma von Fermat $f'(\xi)=0$.

<!--en:c2455f66-->
Im allgemeinen Fall ziehen wir die Sekantengerade ab, um beide Endpunkte auf dieselbe Höhe zu bringen. Definieren wir dazu die geneigte Funktion $\tilde f(x) = f(x) - \frac{f(b)-f(a)}{b-a}(x - a)$. Wegen $\tilde f(a) = \tilde f(b) = f(a)$ liefert der Satz von Rolle eine Stelle $\xi$ im Inneren mit $\tilde f'(\xi) = 0$. Umstellen ergibt unmittelbar :eqref:`eq_mdl-mvt`. $\blacksquare$

<!--en:4ccf1fe5-->
Die mittlere Änderungsrate über ein Intervall tritt also an mindestens einer Stelle im Inneren als momentane Änderungsrate auf. Zwei Folgerungen begegnen uns in diesem Kapitel immer wieder:

<!--en:69e0c45d-->
1. *Gilt $f'>0$ auf einem Intervall, so wächst $f$ dort streng monoton.* Für beliebige $x_1<x_2$ im Intervall folgt aus :eqref:`eq_mdl-mvt` die Beziehung $f(x_2)-f(x_1)=f'(\xi)(x_2-x_1)>0$ für ein $\xi$ zwischen diesen Stellen. Ebenso folgt aus $f'<0$, dass $f$ streng monoton fällt, und aus $f'=0$, dass $f$ konstant ist.

<!--en:9db76619-->
2. *Gilt $f'<0$ auf $(x_0-\delta,x_0)$ und $f'>0$ auf $(x_0,x_0+\delta)$ für ein $\delta>0$, und ist $f$ bei $x_0$ stetig, so ist $x_0$ ein striktes lokales Minimum.* Wenden wir die erste Folgerung auf jede Seite an, sehen wir, dass $f$ von links bis $x_0$ fällt und nach rechts wieder steigt.

<!--en:c6869431-->
Für unser durchgängiges Beispiel gilt überall $s'=\sigma>0$. Die Softplus-Funktion wächst also auf ganz $\mathbb R$ streng monoton.

<!--en:695cb818-->
## Gradientenabstieg in einer Variablen

<!--en:6d3aae13-->
### Der Schritt beim Gradientenabstieg

<!--en:614f0fa7-->
Die lineare Approximation sagt voraus, wie sich $f$ bei einem beliebigen kleinen Schritt $\epsilon$ ändert. Es gilt $f(x+\epsilon)\approx f(x)+\epsilon f'(x)$. Um $f$ zu verkleinern, wählen wir $\epsilon$ mit dem zur Steigung entgegengesetzten Vorzeichen. Setzen wir $\epsilon=-\eta f'(x)$ mit einer positiven *Schrittweite* (engl. step size) $\eta>0$, so sagt das lineare Modell folgende Änderung voraus:

<!--en:077e7236-->
Diese Änderung ist für $f'(x)\neq0$ strikt negativ. Unabhängig vom Vorzeichen der Steigung senkt ein Schritt entgegen der Steigung den Wert des linearen Modells um einen Betrag proportional zum Quadrat der Steigung (:numref:`fig_mdl-gd-step`).

<!--en:701cb6a7-->
![Ein Schritt des Gradientenabstiegs auf einer eindimensionalen, schüsselförmigen Kurve. Bei $x$ hat die Tangente die Steigung $f'(x)$. Der Schritt $-\eta f'(x)$ führt entgegen der Steigung zu $x - \eta f'(x)$, einem tieferen Punkt der Kurve. In erster Ordnung beträgt die Abnahme $\eta f'(x)^2$.](../img/mdl-cal-gd-step.svg)

<!--en:14aca6f3-->
Wiederholen wir diese Aktualisierung, erhalten wir den eindimensionalen **Gradientenabstieg**:

<!--en:f2cd3b8b-->
Jeder Schritt führt entgegen der lokalen Steigung. An einem *stationären Punkt* mit $f'(x_t)=0$ kommt die Iteration zum Stillstand. In einem neuronalen Netz hat jede Koordinate der Gewichtsaktualisierung $w_i\leftarrow w_i-\eta\,\partial\ell/\partial w_i$ diese Form. Alle Koordinaten ändern sich gleichzeitig. Beginnen wir bei unserem logistischen Verlust an der Stelle $w=0$, beträgt die Steigung $\ell'(0)=-\tfrac12$. Ein Schritt mit Schrittweite $\eta$ verschiebt das Gewicht also um $\epsilon=\eta/2$ und lässt eine Abnahme des Verlusts um $\eta/4$ erwarten.

<!--en:cd600299-->
Hier zeigt sich eine wesentliche Grenze des linearen Modells. Die vorhergesagte Änderung $-\eta f'(x)^2$ ist für jedes noch so große $\eta > 0$ negativ. Tatsächlich kann ein zu großer Schritt über das Minimum hinausschießen und an einer höheren Stelle enden, als er begonnen hat. Die Tangente kann uns davor nicht warnen, denn sie enthält keine Information über die Krümmung. Schon die einfachste Parabel $f(x)=x^2$ zeigt, wann die lineare Vorhersage versagt.

<!--en:623fecc3-->
### Der quadratische Fall

<!--en:78333258-->
Für $f(x)=x^2$ ist die Steigung $f'(x)=2x$. Damit vereinfacht sich die Aktualisierung beim Gradientenabstieg zu

<!--en:46b38a21-->
Jeder Schritt multipliziert den aktuellen Iterationswert mit demselben Faktor $1-2\eta$. Das ergibt die geschlossene Formel $x_t=(1-2\eta)^t x_0$. Die Iterationswerte konvergieren daher genau dann gegen das Minimum $x = 0$, wenn $|1 - 2\eta| < 1$ gilt, also $0 < \eta < 1$. Innerhalb dieses Bereichs nähert sich die Iteration bei kleinem $\eta$ langsam und monoton der Stelle $0$. Bei $\eta = \tfrac12$ erreicht sie das Minimum in einem Schritt, bei $\tfrac12 < \eta < 1$ überschießt sie es und nähert sich ihm in gedämpften Schwingungen. Genau an der Grenze $\eta = 1$ springen die Werte unbegrenzt zwischen $+x_0$ und $-x_0$ hin und her. Für $\eta > 1$ wachsen ihre Beträge unbegrenzt.

<!--en:86cb8db4-->
Führen wir von $x_0=1$ aus jeweils zehn Schritte mit fünf verschiedenen Schrittweiten aus, einer aus jedem Bereich. So können wir die Verläufe direkt vergleichen.

<!--en:e6f7804e-->
Die ausgegebenen Werte stimmen mit den Potenzen $(1-2\eta)^{10}$ aus der geschlossenen Formel überein. Mit der vorsichtigen Schrittweite $\eta=0.05$ ist der Wert erst auf $0.349$ gesunken, mit $\eta=0.9$ im Zickzack auf $0.107$. Bei $\eta=1$ ist er wieder bei $1$ und bei $\eta=1.1$ bereits auf $6.19$ angewachsen. Die Stabilitätsgrenze $\eta=1$ entspricht $2/f''$, denn es gilt $f''\equiv2$ für $x^2$. Wie wir in :numref:`subsec_mdl-descent-lemma` sehen, ist das kein Zufall. Für jede glatte Funktion liefert jede obere Schranke $L$ für die Krümmung die Schwelle $2/L$ für sichere Schrittweiten.

<!--en:e257ac00-->
## Krümmung und Taylor-Reihen

<!--en:ac0ef235-->
Wie weit können wir der linearen Approximation vertrauen, und wie groß darf die Schrittweite sein? Beides hängt davon ab, wie schnell sich die Steigung selbst ändert.

<!--en:85cfe629-->
### Die zweite Ableitung

<!--en:64e5738d-->
Da $f'$ selbst eine Funktion von $x$ ist, können wir sie erneut ableiten. Das Ergebnis ist die *zweite Ableitung* $f''=\frac{d^2f}{dx^2}=\frac{d}{dx}\bigl(\frac{df}{dx}\bigr)$. Sie misst die Änderungsrate der Steigung. Differenzieren wir insgesamt $n$-mal, erhalten wir die $n$-te Ableitung,

<!--en:43706672-->
Geometrisch bedeutet $f''>0$, dass die Steigung zunimmt. Der Graph ist wie eine Schüssel nach oben gekrümmt. Bei $f''<0$ nimmt die Steigung ab, und der Graph ist wie eine Kuppel nach unten gekrümmt. An einem stationären Punkt unterscheidet dieses Vorzeichen ein lokales Minimum von einem lokalen Maximum.

<!--en:f6955fbe-->
**Proposition (Kriterium der zweiten Ableitung).** *Es gelte $f'(x_0)=0$, und $f''(x_0)$ existiere. Ist $f''(x_0)>0$, so ist $x_0$ ein striktes lokales Minimum. Ist $f''(x_0)<0$, so ist die Stelle ein striktes lokales Maximum.*

<!--en:181a40a8-->
**Beweis.** Wegen $f'(x_0)=0$ vereinfacht sich die Definition von $f''(x_0)$ zu $f''(x_0)=\lim_{\epsilon\to0}f'(x_0+\epsilon)/\epsilon$. Ist dieser Grenzwert positiv, so muss der Quotient $f'(x_0+\epsilon)/\epsilon$ für alle hinreichend kleinen $\epsilon\neq0$ positiv sein. Daher hat $f'(x_0+\epsilon)$ dasselbe Vorzeichen wie $\epsilon$. Unmittelbar links von $x_0$ ist die Ableitung also negativ, unmittelbar rechts davon positiv. Nach der zweiten Folgerung aus dem Mittelwertsatz ist $x_0$ ein striktes lokales Minimum. Der Fall $f''(x_0)<0$ folgt auf dieselbe Weise mit umgekehrten Vorzeichen. $\blacksquare$

<!--en:a3a17861-->
Der entscheidende Schritt im Beweis war der Schluss vom lokalen Vorzeichenwechsel von $f'$ auf ein Minimum über ein Intervall. Dazu brauchen wir den Mittelwertsatz. Das Kriterium macht nur eine Aussage über eine Umgebung von $x_0$. Für den gesamten Definitionsbereich liefert es keine Garantie. Bei mehreren Variablen übernehmen die Vorzeichen der Eigenwerte der Matrix der zweiten Ableitungen dieselbe Rolle.

<!--en:62c75d6e-->
![Kriterium der zweiten Ableitung. Zeilen: $f''(0)>0$ (Tal, Minimum), $f''(0)<0$ (Kuppe, Maximum) und $f''(0)=0$. Im letzten Fall gilt für $x^3$, $x^4$ und $-x^4$ gleichermaßen $f'(0)=f''(0)=0$, doch die Funktionen haben bei $0$ einen Wendepunkt, ein Minimum beziehungsweise ein Maximum. Spalten: $f''$, $f'$, $f$.](../img/mdl-cal-second-derivative-1d.svg)

<!--en:e12571e4-->
Gilt an einem stationären Punkt $f''(x_0)=0$, so erlaubt das Kriterium keine Entscheidung (:numref:`fig_mdl-second-derivative-test`, untere Reihe). Für $x^3$, $x^4$ und $-x^4$ gilt gleichermaßen $f'(0)=f''(0)=0$. Dennoch ist $0$ bei der ersten Funktion ein Wendepunkt, bei der zweiten ein Minimum und bei der dritten ein Maximum. Bei solchen flachen stationären Punkten können höhere Ableitungen den Ausschlag geben.

<!--en:be472720-->
### Das Taylor-Polynom

<!--en:dc8cff66-->
Die Tangente stimmt an einer Entwicklungsstelle mit Wert und Steigung von $f$ überein. Bezeichnen wir diese Stelle mit $x_0$. Um auch die Krümmung zu erfassen, passen wir eine Parabel an. Eine allgemeine quadratische Funktion $q(x)=ax^2+bx+c$ hat die Ableitungen

<!--en:817a7b36-->
Ihre Krümmung ist also die Konstante $2a$. Der Abgleich mit der Krümmung von $f$ bei $x_0$ legt $a=f''(x_0)/2$ fest. Der Abgleich von Wert und Steigung bestimmt $b$ und $c$. Für $f(x) = \sin(x)$ mit $\sin'=\cos$ und $\sin''=-\sin$ lautet die bei $x_0$ bestangepasste Parabel

<!--en:9c1bdaba-->
Da diese Parabel der Krümmung der Sinuskurve folgt, bleibt sie über ein merklich größeres Intervall genau als die Tangente (:numref:`fig_mdl-best-parabola` zeigt eine Entwicklungsstelle mit $\sin x_0\neq0$, an der die Parabel also eine von null verschiedene Krümmung hat).

<!--en:d2daa1d7-->
![Die Sinuskurve mit ihren beiden lokalen Modellen bei $x_0$. Die Tangente stimmt in Wert und Steigung mit der Kurve überein. Die bestangepasste Parabel stimmt auch in der Krümmung mit ihr überein und bleibt über ein größeres Intervall genau.](../img/mdl-cal-best-parabola.svg)

<!--en:e8260767-->
Für unser durchgängiges Beispiel an der Stelle $x_0=0$ liefern die Werte $s(0)=\log2$, $s'(0)=\tfrac12$ und $s''(0)=\tfrac14$ die bestangepasste Parabel $\log2+\tfrac{x}{2}+\tfrac{x^2}{8}$. Ihr quadratischer Koeffizient $\tfrac18$ ist genau die Konstante, gegen die $\text{gap}/\epsilon^2$ beim logistischen Verlust $\ell(w)=s(-w)$ konvergiert ist.

<!--en:810c233e-->
Diese Idee lässt sich auf jeden Grad $n$ übertragen. Das **Taylor-Polynom** vom Grad $n$ der Funktion $f$ an der Stelle $x_0$ lautet

<!--en:461077b1-->
Es ist das eindeutige Polynom vom Grad höchstens $n$, dessen Wert und erste $n$ Ableitungen mit denen von $f$ bei $x_0$ übereinstimmen. Die Fakultäten treten auf, weil der Term $(x-x_0)^i$ nach genau $i$ Ableitungen bei $x_0$ den Wert $i!$ ergibt, während alle anderen Potenzterme verschwinden. In dieser Schreibweise ist $P_1$ die Tangente und $P_2$ die bestangepasste Parabel. Für $e^x$ bei $x_0=0$ hat jede Ableitung den Wert $1$. Wir erhalten also $P_n(x)=\sum_{i=0}^n x^i/i!$. Mit jedem weiteren Term folgt das Polynom der Exponentialkurve ein Stück weiter vom Ursprung weg. Um zu wissen, wie weit wir $P_n$ vertrauen können und wie schnell der Fehler darüber hinaus wächst, brauchen wir eine explizite Formel für das Restglied.

<!--en:3a895ef2-->
### Das Lagrange-Restglied

<!--en:d8f641d1-->
Der Mittelwertsatz liefert eine exakte Formel für den Fehler von $P_n$. Darin tritt nur eine unbekannte Stelle im Inneren des Intervalls auf. Dieses Ergebnis steht im Mittelpunkt des Abschnitts. Sowohl das Abstiegslemma als auch die quadratische Konvergenzrate des Newton-Verfahrens folgen aus dem Fall $n=1$.

<!--en:a737bd70-->
**Proposition (Lagrange-Restglied).** *Sei $f$ auf einem offenen Intervall $(n+1)$-mal differenzierbar. Enthält das Intervall die Stellen $x_0$ und $x$ mit $x\neq x_0$, so gilt*

<!--en:c8c93c52-->
*für ein $\xi$ strikt zwischen $x_0$ und $x$. Im Fall $n = 0$ erhalten wir wieder den Mittelwertsatz $f(x) = f(x_0) + f'(\xi)(x - x_0)$.*

<!--en:0dbffda6-->
**Beweis.** Der Kunstgriff besteht darin, die passende Hilfsfunktion zu wählen. Danach müssen wir nur noch ableiten. Halten wir $x$ fest und betrachten die Entwicklungsstelle $t$ als Variable. Für $t$ zwischen $x_0$ und $x$ definieren wir

<!--en:9606649a-->
Dies ist der Fehler an der Stelle $x$ für das Taylor-Polynom mit Entwicklungsstelle $t$, abzüglich eines Terms von der erwarteten Form des Fehlers. Wir wählen die Konstante $M$ so, dass $G(x_0)=0$ gilt. Der Rest des Beweises hat drei Schritte.

<!--en:96a69a50-->
Erstens verschwindet $G$ an beiden Endpunkten. Unsere Wahl von $M$ sichert $G(x_0)=0$. Außerdem gilt $G(x)=0$, denn jede positive Potenz von $(x-t)$ verschwindet bei $t=x$, und der Term mit $k=0$ lässt $f(x)-f(x)=0$ übrig. Nach dem Satz von Rolle gibt es daher eine Stelle $\xi$ strikt zwischen $x_0$ und $x$ mit $G'(\xi)=0$.

<!--en:b09aec8f-->
Zweitens leiten wir $G(t)$ nach $t$ ab. Nach der Produktregel trägt jeder Summand mit $k\geq1$ den Ausdruck

<!--en:664954c3-->
zu $G'(t)$ bei, während der Term mit $k=0$ den Beitrag $-f'(t)$ liefert. Benachbarte Terme heben sich paarweise auf (Teleskopsumme). Übrig bleiben nur die höchste Ableitung und die Ableitung des Terms mit $M$:

<!--en:b18a25d9-->
Drittens werten wir den Ausdruck bei $t=\xi$ aus. Da $\xi$ strikt zwischen $x_0$ und $x$ liegt, gilt $(x-\xi)^n\neq0$. Aus $G'(\xi)=0$ folgt also $M=f^{(n+1)}(\xi)$. Setzen wir das wieder in $G(x_0)=0$ ein, so ergibt sich :eqref:`eq_mdl-lagrange`. $\blacksquare$

<!--en:0c4020b7-->
Das Restglied hat dieselbe Form wie der nächste Term des Taylor-Polynoms, doch wir werten die Ableitung der Ordnung $(n+1)$ an einer unbekannten Stelle $\xi$ im Inneren statt bei $x_0$ aus. Bleibt $f^{(n+1)}$ in der Nähe von $x_0$ beschränkt, so ist der Fehler also von der Ordnung $|x-x_0|^{n+1}$. Jede weitere übereinstimmende Ableitung erhöht die Genauigkeitsordnung um eins.

<!--en:2854117f-->
Prüfen wir dies an unserem durchgängigen Beispiel. Für die Softplus-Funktion verschwindet $s'''=\sigma(1-\sigma)(1-2\sigma)$ bei $0$, und es gilt $s''''(0)=-\tfrac18$. Daraus ergibt sich das Polynom vierten Grades

<!--en:9b5ef4e3-->
Bei $x=0.5$ liefert es $0.974072$ gegenüber dem tatsächlichen Wert $s(0.5)=0.974077$. Der Fehler beträgt also nur $5\times10^{-6}$. Er ist sogar noch kleiner, als eine Abschätzung der Ordnung $x^5$ erwarten lässt. Denn $s(x)-\tfrac{x}{2}=\log\bigl(2\cosh\tfrac{x}{2}\bigr)$ ist eine gerade Funktion. Alle höheren Terme ungerader Ordnung verschwinden, und der nächste von null verschiedene Term hat die Ordnung sechs. Der Term vierter Ordnung erklärt auch, warum unser Skript oben $0.1249$ statt $0.1250$ bei $\epsilon=0.1$ ausgegeben hat. Die Ausgabe entsprach $\tfrac18-\tfrac{\epsilon^2}{192}=0.12495$.

<!--en:76b8aaa6-->
Das Verhalten der Ordnung $h^{n+1}$ lässt sich direkt im Code prüfen. Verhält sich der maximale Fehler von $P_n$ im Fenster $|x|\le h$ wie $h^{n+1}$, dann sollte eine Halbierung von $h$ den maximalen Fehler durch $2^{n+1}$ teilen. Für die Tangente wäre das der Faktor $4$, für die Parabel $8$ und für das kubische Polynom $16$. Prüfen wir diese Vorhersage für $e^x$ bei $x_0=0$.

<!--en:be6252a3-->
Die gemessenen Verhältnisse liegen tatsächlich nahe bei $4$, $8$ und $16$. Im doppeltlogarithmischen Diagramm erscheinen die drei Fehlerkurven als Geraden mit den Steigungen $2$, $3$ und $4$, passend zu $n+1$. Bei größeren Fenstern liegen die Verhältnisse etwas über den theoretischen Werten, weil der Faktor $e^{\xi}$ im Restglied dort größer ist.

<!--en:7aaf925c-->
Das Lagrange-Restglied zeigt auch, wann die unendliche Taylor-Reihe die Funktion selbst darstellt. Für $e^x$ ist jede Ableitung gleich $e^x$. Daher liefert :eqref:`eq_mdl-lagrange` um die Stelle $x_0=0$ die Abschätzung

<!--en:5449b527-->
Für jedes feste $x$ geht die rechte Seite bei $n\to\infty$ gegen null, weil Fakultäten schließlich schneller wachsen als Potenzen. Somit gilt

<!--en:bb1df70a-->
auf ganz $\mathbb R$. Allerdings muss eine beliebig oft differenzierbare Funktion nicht mit ihrer Taylor-Reihe übereinstimmen. Bei $f(x) = e^{-1/x^2}$ (ergänzt durch $f(0) = 0$) verschwindet jede Ableitung bei $0$. Die Taylor-Reihe am Ursprung ist also identisch null und stimmt mit $f$ nur bei $x=0$ überein (:numref:`fig_mdl-smooth-not-analytic`). Für das Weitere brauchen wir zum Glück keine unendlichen Reihen. Wir arbeiten ausschließlich mit endlichen Taylor-Polynomen $P_n$, für deren Fehler :eqref:`eq_mdl-lagrange` direkt eine Schranke liefert.

<!--en:4f2faf4a-->
![Bei der glatten Funktion $f(x) = e^{-1/x^2}$ sind am Ursprung alle Ableitungen null. Ihre Taylor-Reihe um den Ursprung ist daher identisch null. Die Reihe konvergiert überall, stimmt aber nur an einer Stelle mit $f$ überein, nämlich bei $x = 0$. Eine glatte Funktion muss also nicht analytisch sein.](../img/mdl-cal-smooth-not-analytic.svg)

<!--en:66f81886-->
### Das Abstiegslemma

<!--en:e4631a26-->
Garantiert eine Schranke für die Krümmung bei jeder glatten Funktion eine Abnahme durch Gradientenabstieg, wie wir sie für $x^2$ gesehen haben? Das Lagrange-Restglied für $n=1$ bejaht diese Frage. Von allen Ergebnissen dieses Abschnitts wird dieses in der Optimierungstheorie am häufigsten verwendet.

<!--en:814e3ddb-->
**Proposition (Abstiegslemma).** *Sei $f$ auf einem offenen Intervall $I$ zweimal differenzierbar. Das Intervall enthalte $x$ und $x-\eta f'(x)$, und es gelte $|f''|\le L$ auf $I$. Dann gilt*

<!--en:32e8fa39-->
*Insbesondere nimmt $f$ für $0 < \eta < 2/L$ und $f'(x) \neq 0$ strikt ab. Die garantierte Abnahme ist bei $\eta = 1/L$ maximal und beträgt dort $f'(x)^2/(2L)$.*

<!--en:990c587d-->
**Beweis.** Für einen beliebigen Schritt $u$ liefert das Lagrange-Restglied mit $n=1$ die Abschätzung $f(x+u)=f(x)+f'(x)u+\tfrac12f''(\xi)u^2\le f(x)+f'(x)u+\tfrac L2u^2$. Setzen wir $u=-\eta f'(x)$, so ergibt sich auf der rechten Seite $f(x)-\eta(1-L\eta/2)f'(x)^2$. Der Faktor $1-L\eta/2$ ist für $\eta<2/L$ positiv, und die quadratische Funktion $\eta(1-L\eta/2)$ nimmt ihr Maximum bei $\eta=1/L$ an. $\blacksquare$

<!--en:851c6b1b-->
Geometrisch ersetzt das Lemma die unbekannte lokale Krümmung $f''(\xi)$ durch ihre obere Schranke $L$. Die resultierende quadratische Funktion $f(x)+f'(x)u+\tfrac L2u^2$ beschreibt eine Parabel, die $f$ bei $x$ berührt und auf ganz $I$ eine obere Schranke bildet. Der Schritt mit $\eta=1/L$ führt direkt zum Minimierer dieser Parabel (:numref:`fig_mdl-descent-lemma`). Diese Schranke ist ein zentrales Werkzeug für die Optimierung glatter Funktionen :cite:`Nesterov.2018`. Sie setzt nicht einmal eine zweite Ableitung voraus. Es genügt, dass $f'$ *$L$-Lipschitz-stetig* ist. Das bedeutet, dass sich die Steigungen an zwei beliebigen Stellen höchstens um das $L$-Fache ihres Abstands unterscheiden, also $|f'(u)-f'(v)|\le L|u-v|$. In dieser Form verwendet man das Lemma für die Optimierung in mehreren Variablen (:numref:`sec_mdl-gradient-based-optimization`). In Aufgabe :exref:`ex_mdl-single-variable-calculus-logistic-smoothness` beweisen Sie es für eine Variable.

<!--en:129d3cee-->
![Für $|f''|\le L$ ist $f(x) + f'(x)u + \frac{L}{2}u^2$ eine quadratische obere Schranke als Funktion des Schritts $u$. An der Entwicklungsstelle stimmt sie mit $f$ überein. Der Gradientenschritt mit $\eta = 1/L$ minimiert diese Schranke und garantiert eine Abnahme um mindestens $\frac{1}{2L}f'(x)^2$. Hier gilt $f(x) = \sin(2x)$ mit $L = 4$, und die tatsächliche Abnahme übertrifft die garantierte.](../img/mdl-cal-descent-lemma.svg)

<!--en:e076e487-->
Prüfen wir das Lemma an zwei bekannten Funktionen. Für $x^2$ gilt $L=2$. Damit liefert $2/L=1$ genau die Stabilitätsgrenze, die wir aus der geschlossenen Formel erhalten haben. Für den logistischen Verlust gilt $\ell''(w)=\sigma(-w)\bigl(1-\sigma(-w)\bigr)\le\tfrac14$ mit Gleichheit bei $w=0$. Damit ist $L=\tfrac14$, und jede positive Schrittweite $\eta<8$ senkt $\ell$ garantiert von jedem Startgewicht aus. Beginnen wir bei $w=0$ mit $\eta=1$, so gelangen wir zu $w_1=0.5$. Die Schranke garantiert einen Verlust von höchstens $0.693147-(1-\tfrac18)\cdot\tfrac14=0.4744$. Tatsächlich beträgt er $\ell(0.5)=0.4741$. Die Schranke ist hier fast scharf, weil die Krümmung über diesen kurzen Schritt nahe an ihrem Maximum $L$ bleibt. Für den größeren Schritt $\eta=4=1/L$ garantiert die Schranke $0.1931$, während der Verlust tatsächlich bis auf $\ell(2)=0.1269$ sinkt. Mit zunehmendem Abstand vom Ursprung nimmt die Krümmung ab. Die Kurve fällt daher stärker, als die obere Schranke für den ungünstigsten Fall erwarten lässt.

<!--en:af725eeb-->
### Das Newton-Verfahren

<!--en:9995d277-->
Das quadratische Modell des Abstiegslemmas verwendet die globale obere Schranke $L$ für die Krümmung. Verwenden wir stattdessen die tatsächliche lokale Krümmung $f''(x_t)$ am aktuellen Iterationspunkt, erreichen wir bei jeder nach oben geöffneten quadratischen Funktion den Minimierer in einem einzigen Schritt. Das lokale Modell bei $x_t$ ist das Taylor-Polynom zweiter Ordnung $P_2$,

<!--en:8ed9abca-->
Für $f''(x_t)>0$ ist diese Parabel nach oben geöffnet und besitzt einen eindeutigen Minimierer. Setzen wir $q'(x) = f'(x_t) + f''(x_t)(x - x_t) = 0$ und lösen nach $x$ auf, so erhalten wir die Aktualisierungsregel des **Newton-Verfahrens**:

<!--en:7326a227-->
Dies ist ein Gradientenschritt mit der adaptiven lokalen Schrittweite $1/f''(x_t)$. Auf jeder nach oben geöffneten Parabel erreicht er den exakten Minimierer in einem Schritt. :numref:`fig_mdl-newton-step` zeigt einen solchen Schritt für $f(x)=x^4/4-x$ mit dem Minimierer $x^\star=1$. Bei $x_t=2$ gilt $f(2)=2$, $f'(2)=7$ und $f''(2)=12$. Der Minimierer der Parabel liegt also bei $x_{t+1}=2-7/12=1.4167$.

<!--en:4d0e4fda-->
![Ein Newton-Schritt für $f(x)=x^4/4-x$ von $x_t=2$ aus. Die Parabel $q$ stimmt in Wert, Steigung und Krümmung mit $f$ bei $x_t$ überein. Ihr Minimierer ist der nächste Iterationspunkt $x_{t+1}=x_t-f'(x_t)/f''(x_t)=2-7/12$, der näher am Minimierer $x^\star=1$ liegt.](../img/mdl-cal-newton-step.svg)

<!--en:d72c84d8-->
Untersuchen wir, wie schnell das Newton-Verfahren für $x^4/4-x$ vom Startwert $x_0=2$ aus konvergiert. Testen wir außerdem eine zweite Funktion $x\arctan x-\tfrac12\log(1+x^2)$ mit den Startwerten $x_0=1.4$ und $x_0=1.3$. Dieses zweite Beispiel analysieren wir gleich nach der Betrachtung der Zahlen.

<!--en:118f3835-->
Für $x^4/4-x$ sinken die Fehler rasch über $1.0$, $0.42$, $0.11$, $1.1\times10^{-2}$ und $1.1\times10^{-4}$ bis auf $1.2\times10^{-8}$. Nahe dem Minimum quadriert jeder Schritt den vorherigen Fehler näherungsweise und verdoppelt etwa die Zahl der richtigen Stellen. Beim Gradientenabstieg für $x^2$ schrumpfte der Fehler dagegen in jedem Schritt um denselben Faktor $|1-2\eta|$. Jede Iteration brachte nur eine feste Zahl richtiger Stellen hinzu.

<!--en:21994308-->
Das Lagrange-Restglied erklärt die quadratische Fehlerabnahme. Sei $x^\star$ ein stationärer Punkt und $e_t=x_t-x^\star$ der Fehler im Schritt $t$. Wenden wir :eqref:`eq_mdl-lagrange` mit $n=1$ auf $f'$ um die Entwicklungsstelle $x_t$ an und werten bei $x^\star$ aus, so erhalten wir $0=f'(x_t)-f''(x_t)\,e_t+\tfrac12f'''(\xi_t)\,e_t^2$. Dividieren wir durch $f''(x_t)\neq0$ und setzen die Newton-Aktualisierung :eqref:`eq_mdl-newton` ein, so bleibt

<!--en:40852cc7-->
für ein $\xi_t$ zwischen $x_t$ und $x^\star$, sofern $f$ dreimal differenzierbar ist. Für $x^4/4-x$ konvergiert der Vorfaktor gegen $f'''(1)/(2f''(1))=6/(2\cdot3)=1$. Die ausgegebenen Fehler liefern tatsächlich $e_{t+1}/e_t^2\approx0.42$, $0.64$, $0.87$ und $0.99$. Den allgemeinen Satz für mehrere Variablen behandeln wir in :numref:`sec_mdl-curvature-methods`. Er garantiert quadratische Konvergenz von jedem Startpunkt hinreichend nahe bei einem Minimierer mit positiver Krümmung.

<!--en:bdfb2ead-->
Die Einschränkung „hinreichend nahe“ ist selbst bei einer streng konvexen Funktion wesentlich. Betrachten wir $f(x)=x\arctan x-\tfrac12\log(1+x^2)$. Es gilt $f'(x)=\arctan x$ und $f''(x)=1/(1+x^2)>0$. Die Funktion ist daher streng konvex und hat ein eindeutiges globales Minimum bei $0$. Vom Startwert $x_0=1.4$ aus gibt die zweite Schleife oben $-1.41, 1.45, -1.55, 1.85, -2.89, 8.71$ aus. Das Newton-Verfahren divergiert mit rasch wachsenden Ausschlägen. Vom etwas näher gelegenen Startwert $x_0=1.3$ aus konvergiert es dagegen problemlos ($-1.16$, $0.86$, $-0.37$, $0.03$, …). Die Grenze zwischen Konvergenz und Divergenz liegt in der Nähe von $x_0\approx1.39$.

<!--en:bc982c8c-->
Die lokale Parabel erklärt den Fehlschlag. Weit vom Ursprung wird die Krümmung $1/(1+x^2)$ sehr klein. Die approximierende Parabel $q$ ist daher fast flach und hat ihren Minimierer weit jenseits von $x^\star$. Üblicherweise verkürzt man den Schritt um einen Faktor, der eine tatsächliche Abnahme von $f$ sicherstellt. Dieses Verfahren heißt *Dämpfung* (engl. damping). Gedämpfte Newton-Schritte und die Kosten ihrer Berechnung in hohen Dimensionen untersuchen wir in :numref:`subsec_mdl-why-not-newton`.

<!--en:355856d9-->
## Nicht differenzierbare Stellen

<!--en:78ec9c2c-->
Jedes Netz mit $\mathrm{ReLU}$ ist voller Knicke. So besitzt $\mathrm{ReLU}(x)=\max(0,x)$ bei $0$ keine Ableitung. Trotzdem gibt die Backpropagation dort eine Zahl zurück. Welche Zahl wählt sie, und in welchem Sinne ist diese Zahl eine gültige Steigung?

<!--en:65a5eb8f-->
### Einseitige Ableitungen

<!--en:c941889c-->
Am Differenzenquotienten :eqref:`eq_mdl-der_def` erkennen wir das Problem sofort. Für $f(x) = |x|$ bei $x = 0$ gilt

<!--en:66f0f691-->
Auf jeder Seite von $0$ ist der Quotient konstant, doch die beiden Konstanten unterscheiden sich. Von rechts betrachtet steigt der Graph mit Steigung $+1$, von links betrachtet fällt er mit Steigung $-1$. Diese Grenzwerte heißen **einseitige Ableitungen**, $f'_+(0) = +1$ und $f'_-(0) = -1$. Die gewöhnliche Ableitung existiert nur, wenn beide übereinstimmen. Dieselbe Rechnung für $\mathrm{ReLU}$ bei $0$ liefert $f'_-(0) = 0$ auf der flachen linken Seite und $f'_+(0) = 1$ auf der rechten. An diesen Knicken gibt es keine Tangentensteigung, doch ein System zum automatischen Differenzieren muss trotzdem eine Zahl zurückgeben. Prüfen wir, welche Zahl es tatsächlich liefert.

<!--en:be3cabe4-->
Für $\mathrm{ReLU}'(0)$ liefert Autograd die linksseitige Steigung $0$. Für $\frac{d}{dx}|x|$ bei $0$ liefert es einen durch Konvention festgelegten Wert.

<!--en:f5dcc217-->
Hier liefert es $0$ gemäß der Konvention $\operatorname{sign}(0)=0$.

<!--en:58bdab3b-->
JAX liefert $1$, die rechtsseitige Steigung.

<!--en:d09bfb56-->
### Subgradienten und Optimalität

<!--en:be926880-->
An einem Knick gibt es keine eindeutige Tangente. Es kann aber eine ganze Schar von Geraden geben, die den Graphen am Knick berühren und nirgends über ihm liegen. Bei einer *konvexen* Funktion, deren Sehnen alle auf oder über dem Graphen liegen (:numref:`sec_mdl-convexity`), übernehmen diese Stützgeraden die Rolle der Tangente. Eine Zahl $g$ heißt **Subgradient** einer konvexen Funktion $f$ an der Stelle $x$, wenn

<!--en:cda96e2a-->
Die Gerade durch $(x, f(x))$ mit Steigung $g$ liegt also überall auf oder unter dem Graphen von $f$. Die Menge aller Subgradienten bei $x$ heißt **Subdifferential** $\partial f(x)$ :cite:`Rockafellar.1970`. An jeder Stelle, an der $f$ differenzierbar ist, enthält $\partial f(x)$ nur die Zahl $f'(x)$. An einem Knick umfasst es das gesamte abgeschlossene Intervall zwischen den einseitigen Ableitungen. Dies entspricht dem Fächer von Stützgeraden in :numref:`fig_mdl-relu-corner`:

<!--en:134d1b8e-->
Jeder oben von Autograd zurückgegebene Wert liegt im zugehörigen Intervall.

<!--en:8049d886-->
![Zwei Knicke und ihre Subdifferentiale. Die Stützgeraden durch den Ursprung bilden bei $|x|$ (links) und $\mathrm{ReLU}$ (rechts) jeweils einen Fächer. Ihre Steigungen durchlaufen die Intervalle $\partial|x|(0)$ beziehungsweise $\partial\,\mathrm{ReLU}(0)$. Außerhalb des jeweiligen Knicks ist die Tangente eindeutig, und das Subdifferential enthält nur die gewöhnliche Ableitung.](../img/mdl-cal-relu-corner.svg)

<!--en:24482ece-->
Unser durchgängiges Beispiel veranschaulicht, woher dieses Intervall kommt. Die zugespitzte Softplus-Funktion $\mathrm{softplus}_\beta(x)=\tfrac1\beta s(\beta x)$ konvergiert gegen $\mathrm{ReLU}(x)$ für $\beta\to\infty$. Ihre Ableitung ist $\sigma(\beta x)$ (:numref:`fig_mdl-softplus-relu`). Bei $x=\pm0.1$ betragen die Steigungen $0.40$ und $0.60$ für $\beta=4$. Mit der schärferen Annäherung betragen sie $0.17$ und $0.83$ für $\beta=16$. Das Subdifferential $[0,1]$ ist genau die Menge der Grenzwerte, denen sich die Steigungen dieser glatten Approximationen nähern können, wenn gleichzeitig $\beta\to\infty$ und $x\to0$ gilt.

<!--en:340448b2-->
![Glättung des Knicks. Links: $\mathrm{softplus}_\beta(x)=\frac1\beta\log(1+e^{\beta x})$ nähert sich $\mathrm{ReLU}$ mit wachsendem $\beta$. Rechts: Die Steigung $\sigma(\beta x)$ geht in eine immer steilere Stufe von $0$ nach $1$ über. Nahe $x=0$ füllen die Steigungen der glatten Näherungen das Intervall $[0,1]$ aus, das Subdifferential von $\mathrm{ReLU}$ am Knick.](../img/mdl-cal-softplus-relu.svg)

<!--en:ad5bda0b-->
Subgradienten liefern auch ein einfaches Optimalitätskriterium. Setzen wir $g=0$ in :eqref:`eq_mdl-subgrad` ein, so erhalten wir $f(y)\ge f(x)$ für alle $y$. Eine Stelle $x$ ist also genau dann ein globaler Minimierer einer konvexen Funktion $f$, wenn

<!--en:165cda7d-->
Beide Richtungen der Äquivalenz folgen unmittelbar aus dieser einen Zeile. Für $|x|$ weist die Zugehörigkeit $0 \in [-1,1]$ den Ursprung als globales Minimum aus, obwohl dort keine Ableitung existiert. Ohne Konvexität ist $f'(x)=0$ bei Differenzierbarkeit dagegen nur eine notwendige Bedingung (Lemma von Fermat). Für $x^3$ gilt $f'(0)=0$, obwohl die Funktion überhaupt kein Minimum hat.

<!--en:918a5c38-->
In :numref:`sec_mdl-convexity` erweitern wir Subgradienten auf mehrere Variablen. Schon hier ist ein Unterschied zum gewöhnlichen Gradientenschritt wichtig. Bei einem Subgradientenschritt ist selbst für kleine Schrittweiten kein Abstieg garantiert. Genau am Minimum von $|x|$ ist die Zahl $g = \tfrac12$ ein gültiger Subgradient. Ein Schritt zu $x=-\eta/2$ vergrößert jedoch $|x|$. Subgradientenverfahren erreichen Konvergenz daher mit allmählich abnehmenden Schrittweiten, ohne bei jedem Schritt einen Fortschritt zu garantieren :cite:`Nesterov.2018`.

<!--en:ccdc6f82-->
### Knicke in einem Netz

<!--en:b48aa6cb-->
Jeder Knick in einem neuronalen Netz entsteht in einem einfachen konvexen Baustein wie $|x|$, $\mathrm{ReLU}$ oder dem Hinge-Verlust. Am Knick wählt Autograd ein festes Element aus dem Subdifferential dieses Bausteins (zum Beispiel $\mathrm{ReLU}'(0)=0$) und reicht es mit der Kettenregel durch die umgebende Verkettung weiter. Verketten wir gültige Subgradienten der einzelnen Bausteine, kann jedoch eine Zahl entstehen, die gar kein Subgradient der zusammengesetzten Funktion ist. Das einfachste Gegenbeispiel ist eine etwas umständliche Darstellung der Identitätsfunktion:

<!--en:cc1d3d46-->
Da überall $\varphi(x)=x$ gilt, ist die einzig richtige Steigung bei $0$ gleich $1$. Mechanisches Anwenden der Kettenregel liefert jedoch $\varphi'(x) = \mathrm{ReLU}'(x) + \mathrm{ReLU}'(-x)$. Das ergibt überall $1$ außer bei $x=0$. Dort liefert die Konvention $\mathrm{ReLU}'(0)=0$ den Wert $0 + 0 = 0$. Prüfen wir, ob automatisches Differenzieren tatsächlich dieses Ergebnis liefert.

<!--en:f156c889-->
Wie vorhergesagt meldet Autograd für die Identitätsfunktion die Steigung $0$.

<!--en:83908fe4-->
Dieser Fehler setzt voraus, dass eine Präaktivierung exakt $0$ trifft. Ziehen wir Daten und Anfangsgewichte aus stetigen Verteilungen, geschieht dies bei exakter reeller Arithmetik in jedem festen Schritt mit Wahrscheinlichkeit null. In der Gleitkommaarithmetik mit endlicher Genauigkeit treten exakte Nullen jedoch auf. :citet:`Bertoin.Bolte.Gerchinovitz.ea.2021` haben an einem kleinen vollständig verbundenen Netz gemessen, wie häufig das geschieht. Wir verwenden die Zahlen aus dem Erratum der Autoren von 2023 (der überarbeiteten arXiv-Fassung des Artikels). Es korrigiert einen Fehler in den veröffentlichten Experimenten und kehrt einige der ursprünglichen Schlussfolgerungen um. Backpropagation mit $\mathrm{ReLU}'(0)=0$ und mit $\mathrm{ReLU}'(0)=1$ lieferte bei 16-Bit-Genauigkeit für jede Initialisierung auf mindestens einem Minibatch unterschiedliche Gradienten. Bei 32-Bit-Genauigkeit war das für knapp die Hälfte aller Initialisierungen der Fall, bei 64-Bit-Genauigkeit für keine.

<!--en:6a637d3a-->
Wie sich diese Unterschiede auf das Training auswirken, ist eine andere Frage. Die beruhigende Antwort lautet: so gut wie gar nicht. Bei 32-Bit-Arithmetik sind nur wenige von einer Million Minibatches betroffen. Selbst bei diesen unterscheiden sich die beiden Gradienten relativ nur um etwa $10^{-3}$. In den korrigierten Experimenten liefert jede Wahl von $\mathrm{ReLU}'(0)$ in $[0,1]$ praktisch denselben Trainingsverlust und dieselbe Testgenauigkeit. Nur Extremwerte wie $\mathrm{ReLU}'(0)=10^4$ beeinträchtigen das Training, und Batchnormalisierung und Adam mildern selbst deren Auswirkungen. Die Konvention führt also tatsächlich zu kleinen numerischen Unterschieden zwischen den Läufen, die jedoch harmlos sind. Aufgabe :exref:`ex_mdl-single-variable-calculus-chained-subgradients` untersucht dieses Verhalten in kleinerem Maßstab.

<!--en:bb2fdc59-->
Bisher haben wir nur ein einzelnes skalares Gewicht betrachtet. In einem neuronalen Netz ändern sich Millionen von Gewichten gemeinsam. Welche gemeinsame Richtung den Verlust senkt und was Krümmung in vielen Dimensionen bedeutet, untersuchen wir in :numref:`sec_mdl-multivariable_calculus`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:85f1e1aa-->
Die Ableitung $f'(x)$ ist die Steigung der linearen Approximation $f(x+\epsilon)=f(x)+\epsilon f'(x)+o(\epsilon)$. Aus dieser Approximation haben wir die Summen-, Produkt- und Kettenregel bewiesen. Damit können wir jede Verkettung von Funktionen aus einer kurzen Tabelle elementarer Funktionen ableiten. Für die Softplus-Funktion ergibt sich beispielsweise $s'=\sigma$. Um von Steigungen an einzelnen Stellen zu Aussagen über ganze Intervalle zu gelangen, verwenden wir den Mittelwertsatz und seine Erweiterung auf höhere Ordnungen, das Lagrange-Restglied. Es begrenzt den Fehler des Taylor-Polynoms vom Grad $n$ durch ein Vielfaches von $|x-x_0|^{n+1}$, sofern $f^{(n+1)}$ beschränkt ist.

<!--en:967cdbf6-->
Für die Ordnung $n=1$ liefert das Lagrange-Restglied das Abstiegslemma. Gilt $|f''|\le L$ auf dem Intervall des Schritts und ist $f'(x)\neq0$, so senkt jeder Gradientenschritt mit $0<\eta<2/L$ den Wert von $f$ strikt. Ersetzen wir die globale Schranke $L$ durch die lokale Krümmung $f''(x_t)$, erhalten wir den Newton-Schritt $-f'/f''$. Nahe einem Minimierer mit $f''>0$ nimmt der Fehler damit quadratisch ab. Weit davon entfernt kann das Verfahren selbst bei einer streng konvexen Funktion divergieren. An nicht differenzierbaren Knicken einer konvexen Funktion tritt das Subdifferential $\partial f(x)$ an die Stelle der Ableitung, und $0\in\partial f(x)$ kennzeichnet globale Minima. Autograd wählt an jedem Knick ein Element aus. Die Verkettung dieser Werte kann in der Gleitkommaarithmetik einen Wert liefern, der kein echter Subgradient ist. Die Auswirkung auf das Training ist jedoch vernachlässigbar.

<!--en:2294243a-->
## Übungen

<!--en:f54af9d8-->
1. **Glattheit des logistischen Verlusts.** Das Abstiegslemma :eqref:`eq_mdl-descent` setzt eine Schranke $L$ für die Krümmung voraus.
    1. Zeigen Sie mit dem Mittelwertsatz :eqref:`eq_mdl-mvt`, dass aus $|f''| \le L$ auf einem Intervall folgt, dass $f'$ dort $L$-Lipschitz-stetig ist. Zeigen Sie umgekehrt anhand der Definition :eqref:`eq_mdl-der_def`, dass eine zweimal differenzierbare Funktion $f$ mit $L$-Lipschitz-stetiger Ableitung $f'$ die Bedingung $|f''| \le L$ erfüllt.
    1. Die logistische Regression mit einem Gewicht $w$, Eingaben $a_i \in \mathbb{R}$ und Labels $y_i \in \{-1, 1\}$ minimiert

       $$\ell(w) = \frac{1}{n}\sum_{i=1}^{n} \log\bigl(1 + e^{-y_i a_i w}\bigr).$$

       Drücken Sie $\ell''$ durch $\sigma$ aus. Bestimmen Sie das kleinste $L$, für das $\ell'$ die Lipschitz-Konstante $L$ auf ganz $\mathbb{R}$ besitzt. Bestimmen Sie die Schrittweiten, für die :eqref:`eq_mdl-descent` von jedem nichtstationären Startpunkt aus eine Abnahme garantiert.
    1. Beweisen Sie das Abstiegslemma unter der schwächeren Annahme, dass $f$ differenzierbar und $f'$ auf dem Intervall des Schritts $L$-Lipschitz-stetig ist. Für $u=-\eta f'(x)$ sei $\psi(t)=f(x+tu)-f(x)-tf'(x)u-\tfrac L2t^2u^2$. Zeigen Sie, dass $\psi(0)=0$ und $\psi'(t)=u\,\bigl(f'(x+tu)-f'(x)\bigr)-Ltu^2\le0$ für $t\in[0,1]$ gilt, und folgern Sie mit dem Mittelwertsatz, dass $\psi(1)\le0$ gilt. In dieser Form verwendet die Optimierung das Lemma für Funktionen mehrerer Variablen.
   :exlabel:`ex_mdl-single-variable-calculus-logistic-smoothness`

<!--en:3385ce11-->
1. **Extremwerte der Entropie.** Die binäre Entropie einer Münze, die mit Wahrscheinlichkeit $p$ Kopf zeigt, beträgt

   $$H(p) = -p\log p - (1-p)\log(1-p), \qquad 0 < p < 1.$$

    1. Bestimmen Sie das Minimum von $f(x) = x\log x$ auf $x \ge 0$, wobei $f(0) = 0$ der Grenzwert ist. Warum liegt das Minimum nicht am Rand $x = 0$?
    1. Zeigen Sie, dass $H'(p) = -\log\frac{p}{1-p}$ gilt, also der negative Logit von $p$ die Ableitung ist. Bestimmen Sie den stationären Punkt von $H$ und klassifizieren Sie ihn mit dem Kriterium der zweiten Ableitung.
    1. Bestimmen Sie den Grenzwert von $H'(p)$ für $p \to 0$. Parametrisieren Sie die Münze dann durch ihren Logit $z$, sodass $p = \sigma(z)$ gilt, und berechnen Sie $\frac{d}{dz}H(\sigma(z))$. In welcher der beiden Parametrisierungen ist die Steigung von $H$ auf dem gesamten Definitionsbereich Lipschitz-stetig, sodass :eqref:`eq_mdl-descent` für den Gradientenabstieg auf $-H$ eine gemeinsame Schrittweite für alle Startpunkte garantiert?

<!--en:213fb566-->
1. **Newton-Verfahren an entarteten Minima.** Für $p > 1$ ist die Funktion $f_p(x) = |x|^p/p$ konvex und hat ihr eindeutiges Minimum bei $x = 0$.
    1. Zeigen Sie, dass $f_p$ überall differenzierbar ist, auch bei $0$, und dass $f_p'(x) = \operatorname{sign}(x)\,|x|^{p-1}$ gilt. Berechnen Sie $f_p''(x)$ für $x \neq 0$. Für welche $p$ existiert $f_p''(0)$, und welchen Wert hat diese Ableitung?
    1. Zeigen Sie, dass das Newton-Verfahren :eqref:`eq_mdl-newton` jedes $x_t \neq 0$ auf $x_{t+1} = \frac{p-2}{p-1}\,x_t$ abbildet. Für welche $p$ konvergiert die Iteration? Wie schnell konvergiert sie für $p = 4$?
    1. Welche Voraussetzung der lokalen Konvergenzaussage in :numref:`subsec_mdl-newton` ist für $p = 4/3$ verletzt, und welche für $p = 4$?
    1. Für $p = 4/3$ verkürzt die gedämpfte Aktualisierung $x_{t+1} = x_t - \alpha f_p'(x_t)/f_p''(x_t)$ den Newton-Schritt um einen Faktor $\alpha > 0$. Für welche $\alpha$ konvergiert die Iteration von jedem $x_0$ aus, und mit welcher Rate?

    *Nach MIT 18.01SC, [Aufgaben zu Sitzung 33](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/resources/mit18_01scf10_ex33prb/), „Cube Root of x“.*
   :exlabel:`ex_mdl-single-variable-calculus-newton-degenerate`

<!--en:2906fcd7-->
1. [code] **Minimierer an Knicken.** Eine Support-Vektor-Maschine mit einem Gewicht $x$, Gewichtszerfall $\lambda > 0$ und einem einzigen Trainingsbeispiel, bei dem das Produkt aus Label und Eingabe $1$ ergibt, minimiert

   $$F(x) = \max(0, 1 - x) + \frac{\lambda}{2}x^2.$$

    1. Berechnen Sie das Subdifferential des Hinge-Verlusts $\max(0, 1-x)$ an jeder Stelle $x$ anhand von :eqref:`eq_mdl-subgrad`.
    1. Setzen Sie voraus, dass beim Addieren einer differenzierbaren konvexen Funktion $q$ zu einer konvexen Funktion $h$ zu jedem Subgradienten der Wert $q'$ hinzukommt, also $\partial(q + h)(x) = q'(x) + \partial h(x)$ gilt. Bestimmen Sie mit der Optimalitätsbedingung :eqref:`eq_mdl-subopt` den Minimierer von $F$ für jedes $\lambda > 0$. Für welche $\lambda$ liegt er am Knick?
    1. Setzen Sie $\lambda = 1/2$ und führen Sie das Subgradientenverfahren $x_{t+1} = x_t - \eta_t g_t$ vom Startwert $x_0 = 0$ aus für $10^4$ Schritte aus. Dabei ist $g_t$ der Wert, den Autograd für $F'(x_t)$ liefert, wenn Sie den Hinge-Verlust als $\mathrm{ReLU}(1-x)$ schreiben. Vergleichen Sie die konstanten Schrittweiten $\eta_t = 0.1$ und $\eta_t = 0.01$ mit den abnehmenden Schrittweiten $\eta_t = 0.5/\sqrt{t+1}$. Wie hängt der Abstand zum Minimierer am Ende von der Wahl der Schrittweitenfolge ab?
    1. Der Knick $\{x = 1\}$ ist ein einzelner Punkt. Eine Stichprobe aus einer Verteilung mit Dichte trifft ihn mit Wahrscheinlichkeit null. Was bedeuten (b) und (c) für das Argument in :numref:`subsec_mdl-nonsmooth-training`, dass das Training selten auf einen Knick trifft?

<!--en:b3beeda4-->
1. [code] **Verkettete Subgradienten.** Autograd liefert $\mathrm{ReLU}'(0) = 0$ und reicht diesen Wert wie in :numref:`subsec_mdl-nonsmooth-training` durch jeden Ausdruck weiter.
    1. Seien $f_1$ und $f_2$ konvex, und es gelte $g_1 \in \partial f_1(x)$, $g_2 \in \partial f_2(x)$ und $c_1, c_2 \ge 0$. Zeigen Sie mit :eqref:`eq_mdl-subgrad`, dass $c_1 g_1 + c_2 g_2 \in \partial (c_1 f_1 + c_2 f_2)(x)$ gilt. Warum lässt sich dieses Argument nicht auf $\mathrm{ReLU}(x) - \mathrm{ReLU}(-x)$ anwenden?
    1. Zeigen Sie, dass für jedes reelle $c$ gilt:

       $$k_c(x) = \mathrm{ReLU}(x) + \mathrm{ReLU}(-x) + c\,\bigl(x - \mathrm{ReLU}(x) + \mathrm{ReLU}(-x)\bigr) = |x|.$$

       Sagen Sie vorher, welchen Wert Autograd für $k_c'(0)$ liefert, und überprüfen Sie Ihre Vorhersage für $c = 0$, $1$ und $5$. Für welche $c$ ist der zurückgegebene Wert ein Subgradient von $|x|$ bei $0$?
    1. Führen Sie den Gradientenabstieg mit der von Autograd gelieferten Ableitung für $\varphi(x) = \mathrm{ReLU}(x) - \mathrm{ReLU}(-x)$ über $10$ Schritte vom Startwert $x_0 = 0.5$ aus durch, einmal mit $\eta = 0.125$ und einmal mit $\eta = 0.1$. Was geschieht in jedem Lauf? Warum verhalten sich die beiden Schrittweiten unterschiedlich? Was bedeutet der Unterschied für das Argument mit der Wahrscheinlichkeit null aus :numref:`subsec_mdl-nonsmooth-training` in der Gleitkommaarithmetik?
   :exlabel:`ex_mdl-single-variable-calculus-chained-subgradients`

<!--en:9eb40e47-->
1. **Abstieg auf einer Funktion vierten Grades.** ● Das Minimum von $f(x) = x^4/4$ ist flach, denn $f''(0) = 0$. Der Gradientenabstieg :eqref:`eq_mdl-gd-loop` für $f$ lautet $x_{t+1} = x_t - \eta x_t^3$.
    1. Bestimmen Sie mithilfe von Aufgabe :exref:`ex_mdl-single-variable-calculus-logistic-smoothness`a das kleinste $L$, für das $f'$ die Lipschitz-Konstante $L$ auf $[-R, R]$ besitzt. Welche Startpunkte $x_0$ erfasst das Abstiegslemma :eqref:`eq_mdl-descent` für ein gegebenes $\eta$? Warum erfasst kein festes $\eta$ alle Startpunkte?
    1. Zeigen Sie, dass die Iterationswerte gegen $0$ konvergieren, wenn $\eta x_0^2 < 2$ gilt, und divergieren, wenn $\eta x_0^2 > 2$ gilt. Was geschieht bei $\eta x_0^2 = 2$? Vergleichen Sie diese Schwelle mit den in (a) erfassten Startpunkten.
    1. Zeigen Sie, dass $1/x_{t+1}^2 = 1/x_t^2 + 2\eta + O(x_t^2)$ gilt, und folgern Sie daraus $x_t^2 \approx 1/(x_0^{-2} + 2\eta t)$. Prüfen Sie die Vorhersage für $\eta = 0.1$ und $x_0 = 1$ bei $t = 10$, $100$ und $1000$. Vergleichen Sie diese Konvergenzrate mit der des Gradientenabstiegs für $x^2$ und mit der des Newton-Verfahrens für $x^4/4$ aus Aufgabe :exref:`ex_mdl-single-variable-calculus-newton-degenerate`b.
   :exlabel:`ex_mdl-single-variable-calculus-quartic-descent`

<!--en:f8c7e56a-->
1. [code] **Zentrale Differenzen.** ● Die Vorwärtsdifferenz $D_+f(x)=\frac{f(x+\epsilon)-f(x)}{\epsilon}$ und die zentrale Differenz $D_0f(x)=\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}$ approximieren beide $f'(x)$.
    1. Zeigen Sie mit dem Lagrange-Restglied :eqref:`eq_mdl-lagrange`, dass $D_+f(x)-f'(x)=\tfrac12f''(\xi)\,\epsilon$ und $D_0f(x)-f'(x)=\tfrac16f'''(\xi')\,\epsilon^2$ für geeignete $\xi$ und $\xi'$ in der Nähe von $x$ gelten. Dabei sei $f$ dreimal stetig differenzierbar (wegen der Stetigkeit von $f'''$ lassen sich zwei Zwischenstellen der Restglieder zu einer zusammenfassen). Entwickeln Sie $f(x\pm\epsilon)$ bis zur benötigten Ordnung und subtrahieren Sie.
    1. Zeigen Sie für $f(x)=x^2+1701(x-4)^3$ bei $x=4$, dass die beiden Fehler $\epsilon+1701\epsilon^2$ und $1701\epsilon^2$ betragen. Bei $\epsilon=0.1$ sind das $17.11$ und $17.01$, also fast gleiche Werte. Die zentrale Differenz erhöht die Fehlerordnung. Das bringt erst einen Vorteil, wenn $\epsilon$ klein ist. Bei welchem $\epsilon$ beträgt der Fehler der zentralen Differenz ein Hundertstel des Fehlers der Vorwärtsdifferenz?

   In der Gleitkommaarithmetik setzt die Rundung dem sinnvoll nutzbaren $\epsilon$ eine untere Grenze. Die Aufgabe zur Gradientenprüfung in :numref:`sec_mdl-chain-rule-graphs` wägt den Rundungsfehler gegen den Abbruchfehler aus (a) ab.
   :exlabel:`ex_mdl-single-variable-calculus-central-differences`

<!--en:149f6a72-->
1. [code] **Gradientenabstieg jenseits von $2/L$.** ●● Für $f(x) = x^2$ divergiert der Gradientenabstieg bei jeder Schrittweite $\eta > 2/L$ von jedem Startpunkt außer dem Minimum aus. Bei anderen Funktionen kann er jenseits dieser Schwelle in eine Schwingung übergehen :cite:`Chen.Bruna.2023`, eine eindimensionale Variante des Verhaltens an der Stabilitätsgrenze :cite:`Cohen.Kaur.Li.ea.2021`. Sei $x^\star$ ein lokaler Minimierer einer viermal stetig differenzierbaren Funktion $f$ mit $f''(x^\star) > 0$. Wir schreiben $T(x) = x - \eta f'(x)$ für einen Schritt von :eqref:`eq_mdl-gd-loop`.
    1. Zeigen Sie mit :eqref:`eq_mdl-mvt`, dass die Iterationswerte bei einem Start hinreichend nahe an $x^\star$ gegen diese Stelle konvergieren, wenn $\eta < 2/f''(x^\star)$ gilt. Zeigen Sie für $\eta > 2/f''(x^\star)$, dass es eine Umgebung von $x^\star$ gibt, in der auf jeden Iterationspunkt außer $x^\star$ selbst ein Punkt mit größerem Abstand zu $x^\star$ folgt.
    1. Ein Zyklus der Periode zwei ist ein Paar $a \neq b$ mit $T(a) = b$ und $T(b) = a$. Zeigen Sie, dass es für jeden solchen Zyklus eine Stelle $\xi$ strikt zwischen $a$ und $b$ mit $f''(\xi) = 2/\eta$ gibt.
    1. Sei $\eta = (1 + \delta)\,2/f''(x^\star)$ für ein kleines $\delta > 0$. Entwickeln Sie $T(T(x))$ um $x^\star$ bis zur dritten Ordnung. Zeigen Sie, dass in der Nähe von $x^\star$ ein Zyklus der Periode zwei existiert und benachbarte Iterationswerte anzieht, wenn

       $$3 f'''(x^\star)^2 - f''(x^\star)\, f''''(x^\star) > 0.$$

       Leiten Sie die Halbbreite $|a - b|/2$ des Zyklus in führender Ordnung in $\delta$ her.
    1. Führen Sie den Gradientenabstieg auf dem Doppelmuldenpotential $f(x) = \tfrac14 (x^2 - 1)^2$ vom Startpunkt $x_0 = 0.5$ aus für jeweils $5000$ Schritte pro Schrittweite $\eta \in [0.9, 1.5]$ aus. Tragen Sie die letzten $100$ Iterationswerte gegen $\eta$ auf. Vergleichen Sie die Halbbreite des Zyklus nahe $\eta = 1$ mit (c). Vergleichen Sie den Bereich von $\eta$ mit stabilem Zyklus der Periode zwei mit dem Bereich $1 < \eta < \sqrt{4.5} - 1$, in dem :citet:`Chen.Bruna.2023` die Konvergenz gegen den Zyklus für fast jeden Startwert $0 < x_0 < 1$ beweisen.
    1. Sagen Sie anhand von (c) vorher, wie sich der Gradientenabstieg mit $\eta = 2.02$ für $\log\cosh x$ und $x^2/2 + x^4/4$ verhält. Für beide Funktionen gilt $x^\star = 0$ und $f''(x^\star) = 1$. Prüfen Sie Ihre Vorhersagen, indem Sie in beiden Fällen bei $x_0 = 0.01$ starten und $500$ Schritte ausführen.

<!--en:ae90a31b-->
[Diskussion](https://d2l.discourse.group/t/412)

<!--en:9f4b9bd3-->
[Diskussion](https://d2l.discourse.group/t/1088)

<!--en:415a90a4-->
[Diskussion](https://d2l.discourse.group/t/1089)

<!--en:415a90a4-2-->
[Diskussion](https://d2l.discourse.group/t/1089)

<!--en:3787f670-->
Differentialrechnung in einer Variablen<br>**lineare Approximation · Ableitungsregeln · Mittelwertsatz · Taylor-Restglied · Abstiegslemma · Newton-Verfahren · Subgradienten**

<!--en:878a47c1-->
::: {.slide title="Lokale Vorhersage mit der Steigung und Fehler durch Krümmung"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:2fdd81d5-->
- **Logistischer Verlust in einer Dimension:** $\ell(w) = \log(1 + e^{-w}) = s(-w)$ hat bei $w = 0$ den Wert $\ell(0) = \log 2 \approx 0.693147$ und die Steigung $\ell'(0) = -\tfrac{1}{2}$.

<!--en:f5a905ad-->
- **Vorhersage erster Ordnung:** Für den Schritt zu $w = 0.1$ sagt die Tangente $0.693147 - 0.05 = 0.643147$ voraus.

<!--en:64689552-->
- **Abweichung zweiter Ordnung durch Krümmung:** Der tatsächliche Verlust $\ell(0.1) \approx 0.644397$ weicht um $0.001250 = \tfrac{1}{2}\ell''(0)(0.1)^2$ ab ($\ell''(0) = 1/4$).

<!--en:4accdc2e-->
::: {.slide title="Unter Vergrößerung nähern sich glatte Kurven ihrer Tangente"}

<!--en:9914b045-->
[Geometrische Anschauung]{.kicker}

<!--en:db1414f0-->
- **Lokale Linearität:** Je stärker die Vergrößerung des Graphen einer glatten Funktion $f(x)$ um eine Stelle $x_0$, desto flacher erscheint die Kurve, bis sie von einer Geraden kaum zu unterscheiden ist.

<!--en:5c406c92-->
- **Kleinerer Bildausschnitt:** Für $f(x) = \sin(x^x)$ nahe $x_0 = 2$ schrumpft die Halbbreite jeweils auf ein Zehntel, von $0.6$ auf $0.06$ und dann auf $0.006$. Der Graph (durchgezogen) deckt sich zusehends mit seiner Tangente (gestrichelt).

<!--en:bd04062e-->
- **Grundfrage der Optimierung:** Die Ableitung ist die Steigung dieser Grenzgeraden. Die zweite Ableitung misst, wie schnell die Kurve bei wachsendem Ausschnitt von der Geraden abweicht.

<!--en:56712c14-->
::: {.slide title="Vom Differenzenquotienten der Sekante zur Tangente"}

<!--en:a8bb29fe-->
[Die Ableitung]{.kicker}

<!--en:3f749b9b-->
- **Sekantensteigung (Differenzenquotient):** Die Gerade durch $(x, f(x))$ und $(x+\epsilon, f(x+\epsilon))$ hat die Steigung
  $$\frac{f(x+\epsilon) - f(x)}{\epsilon}$$

<!--en:5ae918a8-->
- **Grenzübergang:** Für $\epsilon \to 0$ dreht sich die Sekante in die Tangente. Der Grenzwert ihrer Steigung definiert die **Ableitung** $f'(x) = \frac{df}{dx}(x)$.

<!--en:06a32e4d-->
- **Existenz des Grenzwerts:** Dann heißt $f$ an der Stelle $x$ *differenzierbar*. Unterscheiden sich links- und rechtsseitiger Grenzwert der Sekantensteigung (wie bei $|x|$ an der Stelle $x = 0$), gibt es keine Tangente.

<!--en:dbc0a3e5-->
::: {.slide title="Die schwierige Wahl der Schrittweite bei endlichen Differenzen"}

<!--en:a6dc8563-->
[Numerische Fallstricke]{.kicker}

<!--en:61f07584-->
Wie klein muss $\epsilon$ sein, damit eine endliche Differenz $(f(x+\epsilon)-f(x))/\epsilon$ den Wert von $f'(x)$ genau schätzt? Eine allgemeine Schwelle gibt es nicht, denn bei mäßigen Schrittweiten können Terme höherer Ordnung überwiegen:

<!--en:9e03d052-->
- **Ein Polynom mit starker Krümmung:** Betrachtet wird $f(x) = x^2 + 1701(x-4)^3$ bei $x = 4$ mit der tatsächlichen Ableitung $f'(4) = 2(4) = 8$.

<!--en:b598fc10-->
- **Exakter Differenzenquotient:** Ausmultiplizieren von $f(4+\epsilon) - f(4) = 8\epsilon + \epsilon^2 + 1701\epsilon^3$ liefert die Sekantensteigung $8 + \epsilon + 1701\epsilon^2$.

<!--en:79ecf355-->
- **Verzögerte Konvergenz:** Bei $\epsilon = 0.1$ trägt der kubische Term $17.01$ bei. Das ergibt $25.11$ (mehr als $3\times$ die tatsächliche Steigung). Automatisches Differenzieren kommt ohne die Wahl von $\epsilon$ aus, da es die exakten Ableitungsregeln anwendet.

<!--en:a075bd49-->
::: {.slide title="Lineare Approximation und Restglied in Klein-o-Notation"}

<!--en:d7ff38ac-->
[Modell erster Ordnung]{.kicker}

<!--en:84bed6ed-->
- **Äquivalenz zur linearen Approximation:** $f$ ist bei $x$ genau dann mit Ableitung $f'(x)$ differenzierbar, wenn $f(x + \epsilon) = f(x) + \epsilon f'(x) + o(\epsilon)$ mit $\lim_{\epsilon \to 0} r(\epsilon)/\epsilon = 0$ gilt.

<!--en:dfbcc6ff-->
- **Beweis:** Division von $r(\epsilon) = f(x+\epsilon) - f(x) - \epsilon a$ durch $\epsilon$ ergibt $r(\epsilon)/\epsilon \to 0 \iff (f(x+\epsilon)-f(x))/\epsilon \to a = f'(x)$.

<!--en:ca911025-->
- **Stetigkeit:** $f(x+\epsilon) - f(x) = \epsilon f'(x) + o(\epsilon) \to 0$.

<!--en:de0f7cad-->
::: {.slide title="Elementare Ableitungen und die Besonderheit der Basis e"}

<!--en:e1f0e5a2-->
[Ableitungsregeln]{.kicker}

<!--en:fe3a86b1-->
- **Elementare Funktionen mit bekannten Ableitungen:** Automatisches Differenzieren zerlegt jeden Verlust in Konstanten, Potenzen ($nx^{n-1}$), Exponentialfunktionen ($e^x$), Logarithmen ($1/x$) und trigonometrische Funktionen.

<!--en:9afb68f7-->
- **Besonderheit der Basis $e$:** Die Faktorisierung von $(a^{x+\epsilon}-a^x)/\epsilon = a^x(a^\epsilon-1)/\epsilon \to (\log a)a^x$ zeigt, dass $e$ die einzige Basis mit Steigung eins bei $x = 0$ ist.

<!--en:9f72001b-->
::: {.slide title="Lokale Beweise für Summen-, Produkt- und Kettenregel"}

<!--en:ec9e5f24-->
[Rechenregeln für Ableitungen]{.kicker}

<!--en:38416a1c-->
- **Summenregel $(g+h)' = g' + h'$:** Beim Addieren linearer Modelle ergeben zwei Restglieder der Ordnung $o(\epsilon)$ wieder ein Restglied der Ordnung $o(\epsilon)$.

<!--en:5e9e1d6c-->
- **Produktregel $(gh)' = gh' + g'h$:** Bei der Multiplikation linearer Modelle verschwindet der gemischte Term $\epsilon^2 g'h' = o(\epsilon)$ in erster Ordnung.

<!--en:e3076e2b-->
- **Kettenregel $(g \circ h)'(x) = g'(h(x))h'(x)$:** Bei der Verkettung multiplizieren sich die lokalen Steigungen: $\Delta g \approx g'(h(x))\Delta h \approx \epsilon g'(h(x))h'(x)$.

<!--en:e51900fc-->
::: {.slide title="Umkehrfunktionen, Softplus, Sigmoidfunktion und Logit"}

<!--en:1e178678-->
[Durchgängiges Beispiel]{.kicker}

<!--en:c203bec8-->
- **Softplus $\to$ Sigmoidfunktion:** Ableiten von $s(x) = \log(1 + e^x)$ ergibt $s'(x) = \frac{e^x}{1+e^x} = \frac{1}{1+e^{-x}} = \sigma(x)$.

<!--en:d0be923e-->
- **Steigung der Sigmoidfunktion und Krümmung der Softplus-Funktion:** $\sigma'(x) = \sigma(x)(1 - \sigma(x)) \le \tfrac{1}{4}$ (Maximum bei $x = 0$).

<!--en:2631420d-->
- **Logit als Umkehrfunktion ($\sigma^{-1}(p) = \log\frac{p}{1-p}$):** Aus $g'(y) = 1/f'(g(y))$ folgt $\mathrm{logit}'(p) = \frac{1}{p(1-p)}$ ($= 4$ bei $p = 1/2$).

<!--en:9ff52901-->
::: {.slide title="Numerischer Nachweis der quadratischen Fehlerabnahme"}

<!--en:7484a6ca-->
[Numerische Überprüfung]{.kicker}

<!--en:c72ef711-->
Für den logistischen Verlust $\ell(w) = s(-w) = \log(1+e^{-w})$ liefert die Kettenregel $\ell'(0) = -\sigma(0) = -\tfrac{1}{2}$ und die Krümmung $\ell''(0) = \sigma(0)(1-\sigma(0)) = \tfrac{1}{4}$. Sinkt $\epsilon$ auf ein Zehntel, so sinkt die Abweichung vom linearen Modell auf ein Hundertstel:

<!--en:f96fca0b-->
Der Quotient $\text{gap}/\epsilon^2$ konvergiert gegen $\tfrac{1}{2}\ell''(0) = \tfrac{1}{8} = 0.1250$. Die kleine Abweichung $0.1249$ bei $\epsilon = 0.1$ entsteht durch den Taylor-Term vierter Ordnung $-\epsilon^4/192$.

<!--en:0b82647a-->
::: {.slide title="Lemma von Fermat und Mittelwertsatz"}

<!--en:28b4e129-->
[Von Punkten zu Intervallen]{.kicker}

<!--en:09901f36-->
Um aus punktweisen Ableitungen auf das Verhalten einer Funktion im gesamten Intervall $[a, b]$ zu schließen, nutzt die Differentialrechnung das Lemma von Fermat und den Mittelwertsatz:

<!--en:3997496a-->
- **Lemma von Fermat:** Hat $f$ im Inneren bei $\xi$ ein lokales Extremum und ist dort differenzierbar, so haben links- und rechtsseitige Differenzenquotienten entgegengesetzte Vorzeichen oder sind null. Daraus folgt $f'(\xi) = 0$.

<!--en:59f447b2-->
- **Mittelwertsatz:** Ist $f$ auf $[a,b]$ stetig und auf $(a,b)$ differenzierbar, lässt sich die Sekantengerade abziehen und der Satz von Rolle anwenden. Er liefert eine Stelle $\xi \in (a,b)$ im Inneren, an der die Tangente parallel zur Sehne ist:
  $$f'(\xi) = \frac{f(b) - f(a)}{b - a}$$

<!--en:bcea6079-->
- **Differenzierbarkeit im gesamten Inneren ist wesentlich:** Auf $[-1,1]$ hat $|x|$ eine waagerechte Sehne ($|-1|=|1|$), doch an keiner Stelle ist die Steigung $0$.

<!--en:698c4c59-->
::: {.slide title="Folgerungen aus dem Mittelwertsatz und Gradientenabstieg in einer Dimension"}

<!--en:106df48d-->
[Optimierung erster Ordnung]{.kicker}

<!--en:dfc583bd-->
- **Aussagen über Intervalle aus dem Mittelwertsatz:** Wegen $f(x_2) - f(x_1) = f'(\xi)(x_2 - x_1)$ folgt aus $f' > 0$ auf einem Intervall, dass $f$ streng monoton wächst. Aus $f' < 0$ folgt streng monotoner Abfall. Wechselt $f'$ an einer Stelle von $-$ zu $+$, so liegt dort, bei $x_0$, ein striktes lokales Minimum.

<!--en:6efd2909-->
- **Gradientenabstieg in einer Dimension:** Ein Schritt entgegen der Steigung um $\epsilon = -\eta f'(x_t)$ mit Lernrate $\eta > 0$ ergibt:
  $$x_{t+1} = x_t - \eta\, f'(x_t)$$

<!--en:1ce7d571-->
- **Abnahme erster Ordnung und Überschießen:** Das lineare Modell sagt $f(x - \eta f'(x)) - f(x) \approx -\eta f'(x)^2 < 0$ voraus. Da die Gerade die Krümmung außer Acht lässt, führt ein zu großes $\eta$ über die Talsohle hinaus.

<!--en:a4371cea-->
::: {.slide title="Fünf Bereiche der Schrittweite auf einer Parabel"}

<!--en:1e95c4dc-->
[Stabilitätsgrenze]{.kicker}

<!--en:4fc2dcc6-->
Für $f(x) = x^2$ mit $f'(x) = 2x$ und Krümmung $f''(x) \equiv 2$ gilt beim Gradientenabstieg die exakte Rekursion $x_{t+1} = (1 - 2\eta)x_t = (1 - 2\eta)^{t+1}x_0$. Sie konvergiert genau dann, wenn $|1 - 2\eta| < 1$ gilt ($0 < \eta < 1 = 2/f''$):

<!--en:02cd92ed-->
- **$0 < \eta < \tfrac{1}{2}$ (langsam und monoton):** Bei $\eta = 0.05$ sinken die Iterationswerte langsam auf einer Seite der Parabel ($x_{10} = 0.349$).

<!--en:fc0c0d59-->
- **$\eta = \tfrac{1}{2} = 1/f''$ (ein exakter Schritt):** Das Minimum $x_1 = 0$ ist sofort erreicht.

<!--en:83fea6e5-->
- **$\tfrac{1}{2} < \eta < 1$ (gedämpfte Schwingung):** Bei $\eta = 0.9$ springen die Iterationswerte über die Talsohle, und ihre Beträge nehmen ab ($x_{10} = 0.107$).

<!--en:8a4cd993-->
- **$\eta = 1 = 2/f''$ gegenüber $\eta > 1$:** Bei $\eta = 1$ springen die Iterationswerte unbegrenzt zwischen den Werten $\pm x_0$ hin und her. Bei $\eta = 1.1$ wachsen ihre Beträge unbegrenzt ($x_{10} = +6.19$).

<!--en:fb24896d-->
::: {.slide title="Krümmung und Kriterium der zweiten Ableitung"}

<!--en:5dac73e9-->
[Geometrie zweiter Ordnung]{.kicker}

<!--en:fddd8c7b-->
- **Zweite Ableitung $f'' = \frac{d^2 f}{dx^2}$:** Misst die Änderungsrate der Steigung $f'$.

<!--en:6d99243a-->
- **Striktes lokales Minimum ($f'(x_0)=0, f''(x_0)>0$):** Wegen $\lim_{\epsilon \to 0} f'(x_0+\epsilon)/\epsilon > 0$ ist die Steigung $f'$ unmittelbar links von $x_0$ negativ und rechts davon positiv. Nach dem Mittelwertsatz fällt $f$ bis $x_0$ und steigt danach wieder (obere Reihe: Tal).

<!--en:49723cd0-->
- **Striktes lokales Maximum ($f'(x_0)=0, f''(x_0)<0$):** Die Steigung $f'$ wechselt von $+$ zu $-$ (mittlere Reihe: Kuppe).

<!--en:ede60566-->
- **Keine Entscheidung im flachen Fall ($f''(x_0)=0$):** Bei $x_0 = 0$ haben $x^3$ (Wendepunkt), $x^4$ (Minimum) und $-x^4$ (Maximum) alle $f'(0) = f''(0) = 0$ (untere Reihe).

<!--en:7f381acd-->
::: {.slide title="Lokale Modelle höherer Ordnung mit Taylor-Polynomen"}

<!--en:ae4c2bd3-->
[Taylor-Entwicklung]{.kicker}

<!--en:1d818021-->
- **Übereinstimmung von $n$ Ableitungen bei $x_0$:** Das eindeutige Polynom vom Grad höchstens $n$, dessen Wert und erste $n$ Ableitungen mit denen von $f$ bei $x_0$ übereinstimmen, ist das **Taylor-Polynom**:
  $$P_n(x) = \sum_{i=0}^{n} \frac{f^{(i)}(x_0)}{i!}(x - x_0)^i$$

<!--en:154621db-->
- **Tangente ($P_1$) und bestangepasste Parabel ($P_2$):** Für $\sin x$ folgt $P_2(x) = \sin x_0 + \cos x_0(x-x_0) - \tfrac{1}{2}\sin x_0(x-x_0)^2$ der Krümmung über einen viel größeren Ausschnitt als $P_1$.

<!--en:778b4bd4-->
- **Softplus bei $x_0 = 0$:** Wegen $s'''(0) = 0$ und $s^{(4)}(0) = -\tfrac{1}{8}$ gilt $P_4(x) = \log 2 + \tfrac{x}{2} + \tfrac{x^2}{8} - \tfrac{x^4}{192}$. Damit stimmt die Näherung für $s(0.5)$ auf $5 \times 10^{-6}$ genau.

<!--en:56c1305b-->
::: {.slide title="Lagrange-Restglied und exakter Fehler aus dem Satz von Rolle"}

<!--en:58d604fe-->
[Fehlerschranken]{.kicker}

<!--en:df3257e9-->
- **Exaktes Lagrange-Restglied:** Für die $(n+1)$-mal differenzierbare Funktion $f$ gilt $f(x) = P_n(x) + \frac{f^{(n+1)}(\xi)}{(n+1)!}(x - x_0)^{n+1}$ für ein $\xi$ strikt zwischen $x_0$ und $x$.

<!--en:ca3d749c-->
- **Beweis durch Teleskopieren:** Beim Variieren der Entwicklungsstelle $t$ in $G(t)$ und Anwenden des Satzes von Rolle ($G'(\xi) = 0$) heben sich nach der Produktregel benachbarte Terme auf.

<!--en:9d0498fa-->
- **Wichtige Spezialfälle:** $n = 0$ liefert den Mittelwertsatz, $n = 1$ ergibt $f(x_0) + f'(x_0)u + \tfrac{1}{2}f''(\xi)u^2$.

<!--en:d5f338b8-->
::: {.slide title="Fehlerordnung des Taylor-Restglieds im doppeltlogarithmischen Diagramm"}

<!--en:078a4810-->
[Genauigkeitsordnung]{.kicker}

<!--en:875989cc-->
Ist $|f^{(n+1)}|$ nahe $x_0$ beschränkt, so hat der maximale Fehler von $P_n$ auf $|x - x_0| \le h$ die Ordnung $O(h^{n+1})$. Eine Halbierung der Fensterbreite $h$ teilt den Fehler also durch $2^{n+1}$ ($4$ für $P_1$, $8$ für $P_2$ und $16$ für $P_3$):

<!--en:b33fd3b7-->
- **Test für $e^x$ um $x_0 = 0$:** Hier gilt $P_n(x) = \sum_{k=0}^n x^k/k!$ und $|e^x - P_n(x)| = \frac{e^\xi |x|^{n+1}}{(n+1)!}$.

<!--en:ba7a84bb-->
- **Fehlerverhältnisse bei sukzessiver Halbierung:** Für die Fenster $h \in \{0.4, 0.2, 0.1, 0.05\}$ nähern sich die Verhältnisse aufeinanderfolgender Fehler den Werten $2^{n+1} = 4, 8, 16$.

<!--en:8067869d-->
- **Steigungen im doppeltlogarithmischen Diagramm:** Bei doppeltlogarithmischen Achsen beschreibt $\log(\text{error}) \approx (n+1)\log h + c$ Geraden mit den Steigungen $2$, $3$ und $4$.

<!--en:1e1bb12e-->
::: {.slide title="Analytische und glatte Funktionen: Wann stellt die Reihe die Funktion dar?"}

<!--en:f57e5ef5-->
[Grenzen der Analytizität]{.kicker}

<!--en:449271c6-->
- **Konvergenz der unendlichen Reihe gegen $f$:** Für $e^x$ gilt $|R_n(x)| \le \frac{e^{|x|}|x|^{n+1}}{(n+1)!}$ $\to$ $0$ bei $n \to \infty$ für jedes $x \in \mathbb{R}$, weil Fakultäten schneller wachsen als Potenzen. Das beweist $e^x = \sum_{n=0}^\infty x^n/n!$.

<!--en:0c75fe6a-->
- **Glatt ($C^\infty$) bedeutet nicht analytisch:** Die Funktion $f(x) = e^{-1/x^2}$ mit flacher Talsohle (mit $f(0) = 0$) ist überall beliebig oft differenzierbar, und dennoch gilt $f^{(n)}(0) = 0$ für jedes $n \ge 0$.

<!--en:50400f0f-->
- **Endliche $P_n$ in der Optimierung:** Die Taylor-Reihe bei $0$ ist identisch $0$ und stimmt mit $f$ nur bei $x = 0$ überein. Daher verwenden alle Optimierungsbeweise endliche Polynome $P_1$ oder $P_2$ mit dem exakten Lagrange-Restglied $R_n$.

<!--en:50ab973e-->
::: {.slide title="Abstiegslemma und sichere Schrittweiten"}

<!--en:5f0f884a-->
[Abstiegsgarantie]{.kicker}

<!--en:6bddbf95-->
Das Lagrange-Restglied für $n = 1$ liefert aus einer Schranke $L$ für die Krümmung ($|f''| \le L$ oder ein $L$-Lipschitz-stetiger Gradient mit $|f'(u)-f'(v)| \le L|u-v|$) eine garantiert sichere Schrittweite für den Gradientenabstieg:

<!--en:5eaf58c5-->
- **Quadratische obere Schranke:** Wegen $f''(\xi) \le L$ gilt für jeden Schritt $u$ die Schranke $f(x+u) \le f(x) + f'(x)u + \tfrac{L}{2}u^2$.

<!--en:b30ad441-->
- **Einsetzen von $u = -\eta f'(x)$:**
  $$f\bigl(x - \eta f'(x)\bigr) \le f(x) - \eta\Bigl(1 - \frac{L\eta}{2}\Bigr)f'(x)^2$$

<!--en:e691d3e9-->
- **Sicherer Bereich und optimale Schrittweite:** Der Verlust nimmt für $0 < \eta < 2/L$ und $f'(x) \ne 0$ strikt ab. Bei $\eta = 1/L$ ist die garantierte Abnahme maximal (Scheitelpunkt der begrenzenden Parabel). Sie beträgt dann mindestens $f'(x)^2/(2L)$.

<!--en:2380a225-->
- **Prüfung am logistischen Verlust:** Wegen $\ell''(w) \le \tfrac{1}{4} = L$ ist jede positive Schrittweite mit $\eta < 8$ global sicher und $\eta = 1/L = 4$ optimal.

<!--en:8dabf973-->
::: {.slide title="Newton-Verfahren: Sprung zum Minimum der Parabel"}

<!--en:f20fdef3-->
[Optimierung zweiter Ordnung]{.kicker}

<!--en:794223b4-->
- **Lokale Krümmung $f''(x_t)$ anstelle der globalen Schranke $L$:** Lokale Taylor-Parabel $q(x) = f(x_t) + f'(x_t)(x-x_t) + \tfrac{1}{2}f''(x_t)(x-x_t)^2$ anpassen.

<!--en:e9803d8e-->
- **Minimierung von $q(x)$ für $f''(x_t) > 0$:** Aus $q'(x) = f'(x_t) + f''(x_t)(x-x_t) = 0$ folgt die **Newton-Aktualisierung**:
  $$x_{t+1} = x_t - \frac{f'(x_t)}{f''(x_t)}$$

<!--en:48fabfc6-->
- **Quadratische Konvergenz nahe $x^\star$:** Die Entwicklung von $0 = f'(x^\star)$ um $x_t$ bis zur ersten Ordnung mit Lagrange-Restglied liefert die Fehlerrekursion $e_{t+1} = \frac{f'''(\xi_t)}{2f''(x_t)}e_t^2$. Der Fehler nimmt in jedem Schritt quadratisch ab.

<!--en:5ca578e5-->
::: {.slide title="Quadratische Konvergenz in der Nähe, mögliche Divergenz in der Ferne"}

<!--en:09c20191-->
[Einzugsgebiet]{.kicker}

<!--en:ad81c0e9-->
Nahe einem Minimierer mit $f''(x^\star) > 0$ verdoppelt das Newton-Verfahren die Zahl der richtigen Stellen pro Schritt. Weit entfernt von $x^\star$ kann eine geringe lokale Krümmung zu einer fast flachen Parabel führen. Der nächste Iterationspunkt liegt dann weit außerhalb des Einzugsgebiets:

<!--en:eb3b5d8a-->
Für die streng konvexe Funktion $f(x) = x\arctan x - \tfrac{1}{2}\log(1+x^2)$ mit $f'(x)=\arctan x$ und $f''(x)=1/(1+x^2)$ konvergiert das Newton-Verfahren vom Startwert $x_0 = 1.3$ aus. Vom Startwert $x_0 = 1.4$ aus divergiert es mit großen Ausschlägen ($|x_0| \gtrsim 1.39$), sofern der Schritt nicht **gedämpft** wird.

<!--en:5b2be2af-->
::: {.slide title="Nicht differenzierbare Knicke und Subdifferentiale"}

<!--en:5be85ade-->
[Nichtglatte Analysis]{.kicker}

<!--en:850c7493-->
- **Verschiedene einseitige Steigungen:** Bei $x = 0$ hat $|x|$ die Steigungen $f'_-(0) = -1$ und $f'_+(0) = +1$, während für $\mathrm{ReLU}(x) = \max(0, x)$ die Steigungen $f'_-(0) = 0$ und $f'_+(0) = +1$ gelten.

<!--en:2ae52d58-->
- **Subgradienten konvexer Funktionen:** Eine Steigung $g$ ist ein **Subgradient** der konvexen Funktion $f$ bei $x$, wenn ihre Gerade den Graphen für alle $y$ von unten stützt:
  $$f(y) \ge f(x) + g\,(y - x)$$

<!--en:b82f3a28-->
- **Das Subdifferential $\partial f(x)$:** An einem Knick bilden die Steigungen der Stützgeraden das Intervall $[f'_-(x), f'_+(x)]$. Daher gilt $\partial |x|(0) = [-1, 1]$ und $\partial\,\mathrm{ReLU}(0) = [0, 1]$.

<!--en:090f45e8-->
- **Optimalität bei Konvexität:** Eine Stelle $x^\star$ ist genau dann ein globales Minimum der konvexen Funktion $f$, wenn $0 \in \partial f(x^\star)$ gilt.

<!--en:a573186c-->
::: {.slide title="Das Subdifferential als Intervall durch Glättung der ReLU-Funktion"}

<!--en:e8c42aee-->
[Grenzwerte der Steigungen]{.kicker}

<!--en:ce1f4a5d-->
- **Zugespitzte Softplus-Approximation:** Die glatte Funktion $\mathrm{softplus}_\beta(x) = \frac{1}{\beta}\log(1 + e^{\beta x})$ konvergiert gleichmäßig gegen $\mathrm{ReLU}(x)$ für $\beta \to \infty$.

<!--en:e90169fc-->
- **Steigungen füllen $[0, 1]$ nahe dem Knick:** Die Ableitung $\sigma(\beta x)$ geht in einem Fenster um den Ursprung glatt von $0$ zu $1$ über. Das Fenster hat die Breite $O(1/\beta)$.

<!--en:f1699632-->
- **Konvention der Frameworks bei $x = 0$:** Genau am Knick $x = 0$ liefern Frameworks nach Konvention einen festen Wert aus $\partial f(0)$:

<!--en:d674449a-->
::: {.slide title="Verkettete Subgradienten in der Gleitkommaarithmetik"}

<!--en:e757f247-->
[Automatisches Differenzieren an Knicken]{.kicker}

<!--en:5b545aba-->
Die Verkettung fest gewählter Subgradienten in einem Rechengraphen kann eine Zahl liefern, die kein gültiger Subgradient der zusammengesetzten Funktion ist:

<!--en:b319c840-->
- **Die Identitätsfunktion mit zwei Knicken dargestellt:** Für $\varphi(x) = \mathrm{ReLU}(x) - \mathrm{ReLU}(-x) = x$ ist die tatsächliche Ableitung überall $1$. Bei $x = 0$ liefert die mechanische Verkettung mit $\mathrm{ReLU}'(0) = 0$ jedoch $0 + 0 = 0$:

<!--en:3adad43a-->
**Warum das Training mit SGD unbeeinträchtigt bleibt (Bertoin et al., 2021, mit den Korrekturen im Erratum von 2023):** Die Abweichung setzt voraus, dass eine Präaktivierung exakt $0.0$ trifft. Bei 32-Bit-Arithmetik mit `float32` verändert eine andere Wahl von $\mathrm{ReLU}'(0)$ in $[0, 1]$ die Gradienten nur bei wenigen Minibatches pro Million um $\sim 10^{-3}$. Trainings- und Testgenauigkeit bleiben gleich.

<!--en:4f9b1c6d-->
::: {.slide title="Verständnisfragen zur Differentialrechnung in einer Variablen"}

<!--en:b6d9a92f-->
[Wiederholung und Übung]{.kicker}

<!--en:8ea0808f-->
Drei Fragen mithilfe des Taylor-Restglieds, des Abstiegslemmas und der Definition des Subdifferentials lösen:

<!--en:64d55e76-->
1. **Fehlerordnung der Taylor-Approximation:** Um welchen Faktor schrumpft die maximale Fehlerschranke des kubischen Taylor-Polynoms $P_3(x)$ auf $|x - x_0| \le h$, wenn $h$ durch $3$ geteilt wird und die glatte Funktion eine beschränkte vierte Ableitung hat?

<!--en:db97c316-->
2. **Sichere Lernrate:** Für einen Verlust $\ell(w)$ gelte überall $|\ell''(w)| \le L = 10$. Für welche Schrittweiten $\eta$ garantiert das Abstiegslemma bei nicht verschwindender Steigung eine Abnahme von $\ell$, und welches $\eta$ maximiert die garantierte Abnahme in einem Schritt?

<!--en:d5a5683c-->
3. **Optimalität mit Subgradienten:** Für $F(x) = |x| + \frac{1}{2}(x - 1)^2$ anhand von $\partial F(0) = \partial|x|(0) + (0 - 1)$ prüfen, ob $x = 0$ der globale Minimierer ist.

<!--en:557b98b6-->
**Lösungen:**

<!--en:472ef86f-->
1. Nach dem Lagrange-Restglied gilt $|R_3(x)| \le \frac{\max|f^{(4)}|}{4!}h^4$. Division von $h$ durch $3$ verkleinert die Fehlerschranke um den Faktor $3^4 = 81$.

<!--en:a747b200-->
2. Jede Schrittweite $0 < \eta < 2/L = 0.2$ garantiert bei Steigung ungleich null eine Abnahme von $\ell$. Für $\eta^\star = 1/L = 0.1$ ist die garantierte Abnahme $\ell'(w)^2 / (2L) = \ell'(w)^2 / 20$ maximal.

<!--en:8bc001c2-->
3. Bei $x = 0$ gilt $\partial F(0) = [-1, 1] - 1 = [-2, 0]$. Wegen $0 \in [-2, 0]$ ist die Optimalitätsbedingung $0 \in \partial F(0)$ erfüllt, und $x = 0$ ist der eindeutige globale Minimierer.

<!--en:4f4afb25-->
::: {.slide title="Zusammenfassung der Differentialrechnung in einer Variablen"}

<!--en:c5a222f8-->
[Zentrale Ergebnisse]{.kicker}

<!--en:9554c097-->
| Begriff oder Satz | Mathematische Formulierung | Bedeutung für Deep Learning und Optimierung |

<!--en:f7a7c904-->
| **Lineare Approximation** | $f(x+\epsilon) = f(x) + \epsilon f'(x) + o(\epsilon)$ | Lokales Modell der Steigung, Grundlage der Beweise für Summen-, Produkt- und Kettenregel |

<!--en:c5d14e95-->
| **Softplus und Sigmoidfunktion** | $s(x)=\log(1+e^x)$, $s'=\sigma$, $s''=\sigma(1-\sigma) \le \tfrac{1}{4}$ | Glattes Gegenstück zu $\mathrm{ReLU}$, Krümmung des binären logistischen Verlusts |

<!--en:4ada72eb-->
| **Mittelwertsatz** | $f'(\xi) = \frac{f(b)-f(a)}{b-a}$ für ein $\xi \in (a,b)$ | Verknüpft punktweise Ableitungen mit Monotonie auf Intervallen |

<!--en:ee9929c2-->
| **Taylor-Polynom und Lagrange-Restglied $R_n$** | $f(x) = \sum_{i=0}^n \frac{f^{(i)}(x_0)}{i!}(x-x_0)^i + \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$ | Beschreibt den Abbruchfehler der Ordnung $O(h^{n+1})$ bei lokalen Modellen vom Grad $n$ |

<!--en:270489d6-->
| **Abstiegslemma** | $f(x-\eta f'(x)) \le f(x) - \eta(1 - \frac{L\eta}{2})f'(x)^2$ | Garantiert sichere Lernraten $0 < \eta < 2/L$ und die optimale Wahl $\eta = 1/L$ |

<!--en:1157889c-->
| **Newton-Verfahren** | $x_{t+1} = x_t - f'(x_t)/f''(x_t)$, $e_{t+1} \approx \frac{f'''(x^\star)}{2f''(x^\star)}e_t^2$ | Minimiert die lokale Parabel $P_2$, lokal quadratische Konvergenz |

<!--en:3cde1152-->
| **Subdifferential** | $\partial f(x) = \{g : f(y) \ge f(x) + g(y-x)\ \forall y\}$ | Erfasst Knicke von $\mathrm{ReLU}$ und $\|x\|$, globales Minimum genau dann, wenn $0 \in \partial f(x)$ |
