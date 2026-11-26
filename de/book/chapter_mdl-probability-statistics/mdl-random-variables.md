<!--en:59be8661-->
# Zufallsvariablen

<!--en:8312be12-->
In :numref:`sec_prob` haben wir Wahrscheinlichkeiten durch Abzählen bestimmt. Ein Würfel hat sechs Seiten, ein Label bezeichnet eine von zehn Klassen, und die Wahrscheinlichkeit eines Ereignisses ist die Summe der Wahrscheinlichkeiten der zugehörigen Ergebnisse. Die meisten Zahlen, mit denen ein neuronales Netz arbeitet, verhalten sich anders. Die Intensität eines Pixels, ein bei der Initialisierung gezogenes Gewicht oder das gaußsche Rauschen, das ein Diffusionsmodell einem Bild hinzufügt, können jeden Wert in einem Intervall annehmen. Das sind zu viele Werte, um sie einzeln aufzuzählen. Welche Wahrscheinlichkeit hat dann ein einzelner Wert?

<!--en:30c68742-->
Stellen wir uns vor, wir werfen einen Dartpfeil so ungezielt auf eine runde Scheibe, dass er überall auf der Scheibe mit gleicher Wahrscheinlichkeit landet. Er trifft genau auf halbem Weg von der Mitte zum Rand. Wie wahrscheinlich war das? Dazu würden wir den Abstand runden, etwa auf einen Bereich zwischen dem $0.45$- und dem $0.55$-Fachen des Radius. In diesem Bereich landet ungefähr jeder zehnte Pfeil. Verlangen wir eine weitere Stelle Genauigkeit, sinkt die Wahrscheinlichkeit auf ein Zehntel, bei der nächsten Stelle erneut auf ein Zehntel. Bei unendlich vielen Stellen ist die Wahrscheinlichkeit null. Trotzdem ist der Pfeil irgendwo gelandet.

<!--en:a104fe7b-->
Das Rätsel löst sich, wenn wir danach fragen, wie schnell die Wahrscheinlichkeit in der Nähe eines Punktes anwächst. Verkleinern wir die Intervallbreite auf ein Zehntel, sinkt auch die Wahrscheinlichkeit auf ein Zehntel. Das Verhältnis von Wahrscheinlichkeit zu Breite bleibt also konstant. Diese Wahrscheinlichkeit pro Längeneinheit ist die *Dichte*. Mit ihrer Hilfe lassen sich alle Werkzeuge der diskreten Wahrscheinlichkeitstheorie auf stetig veränderliche Größen übertragen. An die Stelle von Summen treten Integrale.

<!--en:80bd6322-->
Einige Überlegungen aus :numref:`sec_prob` greifen wir wieder auf, erweitern sie auf stetige Variablen und führen die Beweise. Mittelwert und Varianz haben wir dort bereits motiviert. Wir gehen daher gleich zu ihren stetigen Formen und Rechenregeln über. Außerdem behandeln wir die Dichte selbst, eine allgemeine Methode zum Ziehen von Samples mithilfe der Verteilungsfunktion und die Schranken von Markov und Chebyshev für die Wahrscheinlichkeit großer Abweichungen. Wir lernen auch eine glockenförmige Dichte kennen, deren Verteilungsränder so schwer sind, dass beide Schranken versagen. Zwei Ergebnisse sollten Sie sich besonders einprägen: die Beziehung zwischen Dichte und Verteilungsfunktion :eqref:`eq_mdl-cdf_deriv` und die Markov-Ungleichung :eqref:`eq_mdl-markov`. Aus ihr folgen alle weiteren Schranken für große Abweichungen in diesem Kapitel. Der Dartpfeil begleitet uns durch den ganzen Abschnitt. Wir betrachten nur eine Zahl, seinen Abstand von der Mitte. Seine beiden Koordinaten und die Transformation von Dichten behandeln wir in :numref:`sec_mdl-joint-distributions`.

<!--en:dda2e5ce-->
## Dichten

<!--en:57cc73a4-->
Wir entwickeln den Begriff der Dichte in drei Schritten. Zuerst betrachten wir einen einzelnen Punkt und bilden einen Grenzwert für immer schmalere Intervalle. Dann betrachten wir ein ganzes Intervall und die zugehörige Fläche unter einer Kurve. Schließlich untersuchen wir die Funktion, die diese Fläche aufsummiert. Sie liefert uns zugleich ein praktisches Verfahren, um Zufallsstichproben zu ziehen.

<!--en:6498ae3d-->
### Von Intervallen zur Dichte

<!--en:103b29bd-->
Um „Wahrscheinlichkeit pro Längeneinheit“ präzise zu fassen, bestimmen wir zunächst die Wahrscheinlichkeit, mit der der Abstand des Pfeils in einen vorgegebenen Bereich fällt. Sei $R$ der Abstand des Pfeils von der Mitte einer Scheibe mit Radius $1$. Der Abstand von der Mitte beträgt höchstens $r$, wenn der Pfeil die Kreisscheibe mit Radius $r$ trifft. Da der Treffer auf der Zielscheibe gleichverteilt ist, entspricht diese Wahrscheinlichkeit dem Flächenanteil der kleineren Kreisscheibe:

<!--en:508cb5eb-->
Innerhalb des halben Radius liegt nur ein Viertel der Wahrscheinlichkeit, denn dort liegt nur ein Viertel der Fläche. Ein Intervall der Breite $\epsilon$ um $0.5$ hat daher die Wahrscheinlichkeit $(0.5+\epsilon/2)^2-(0.5-\epsilon/2)^2=\epsilon$. Das ergibt die Zahlen aus unserem Eingangsbeispiel:

<!--en:9d99c0bd-->
| Intervall | Breite $\epsilon$ | Wahrscheinlichkeit | Wahrscheinlichkeit $/\ \epsilon$ |

<!--en:7a713eb4-->
Verkleinern wir die Breite auf ein Zehntel, sinkt auch die Wahrscheinlichkeit auf ein Zehntel. Ihr Verhältnis bleibt gleich. Dem Grenzwert dieses Verhältnisses geben wir nun einen Namen. Die *Wahrscheinlichkeitsdichte* (engl. probability density) einer Zufallsvariablen (engl. random variable) $X$ an der Stelle $x$ ist die Wahrscheinlichkeit pro Längeneinheit in der Nähe von $x$,

<!--en:016cbcc1-->
sofern dieser Grenzwert existiert. Gleichbedeutend damit ist die folgende Näherung: Die Wahrscheinlichkeit eines kurzen Intervalls ist seine Länge mal der Dichte,

<!--en:f8774122-->
wobei der Fehler schneller gegen null geht als $\epsilon$. Damit ist die Dichte punktweise beschrieben. Das genügt uns vorerst. Die formale Definition über Intervalle folgt im nächsten Unterabschnitt. Für jeden Abstand $r$ in $(0,1)$ hat beim Dartwurf das Intervall $[r,r+\epsilon)$ die Wahrscheinlichkeit $(r+\epsilon)^2-r^2=2r\epsilon+\epsilon^2$. Teilen wir durch die Breite $\epsilon$, erhalten wir $2r+\epsilon$. Im Grenzfall $\epsilon\to 0$ folgt

<!--en:d89b4852-->
Die Dichte wächst von der Mitte zum Rand linear, genau wie der Umfang eines Kreises mit Radius $r$. Weiter außen steht mehr Fläche zur Verfügung. Bei $r=0.5$ ist die Dichte gleich $1$. Das entspricht dem Verhältnis in der Tabelle oben.

<!--en:728a9fed-->
Eine Dichte ist keine Wahrscheinlichkeit. Bei $r=0.9$ beträgt die Dichte für den Dartwurf $1.8$. Eine auf $[0,0.1]$ gleichverteilte Variable hat dort die Dichte $10$, denn die gesamte Wahrscheinlichkeit $1$ verteilt sich auf eine Länge von $0.1$. Eine Dichte kann jeden nichtnegativen Wert annehmen. Sie kann allerdings den Wert $10$ nur auf einer Menge mit einer Gesamtlänge kleiner als $1/10$ überschreiten, sonst wäre die Wahrscheinlichkeit dieser Menge größer als $1$. Stellen Sie sich die Dichte als Rate vor, ähnlich einer Geschwindigkeit. Die Geschwindigkeit eines Autos gibt an, wie schnell die zurückgelegte Strecke wächst. Die Strecke selbst lässt sich daran nicht ablesen. Ebenso gibt die Dichte $1.8$ bei $r=0.9$ an, wie schnell die Wahrscheinlichkeit wächst, wenn $r$ über $0.9$ hinaus zunimmt. Sie gibt nicht die Wahrscheinlichkeit des einzelnen Punktes $0.9$ an.

<!--en:accf86e3-->
Da ein einzelner Punkt die Länge null hat, trägt er auch die Wahrscheinlichkeit null. Für jedes $\epsilon>0$ liegt der Punkt $x$ in $[x,x+\epsilon)$. Somit beträgt $P(X=x)$ höchstens etwa $\epsilon\,p(x)$. Dieser Ausdruck verschwindet, wenn $\epsilon$ gegen null geht. Für jede Zufallsvariable mit Dichte gilt also

<!--en:e2863b97-->
Damit löst sich unser anfängliches Rätsel: Der Pfeil landet mit Wahrscheinlichkeit null im Abstand $0.5$, genau wie in jedem anderen exakt vorgegebenen Abstand. Trotzdem muss er irgendwo landen. In der stetigen Wahrscheinlichkeitstheorie ist ein Ereignis mit Wahrscheinlichkeit null nicht unmöglich.

<!--en:e96f051d-->
Einen einzelnen Punkt mit positiver Wahrscheinlichkeit nennen wir ein *Atom*. Eine Variable mit Dichte hat keine Atome. Eine diskrete Variable wie die Augenzahl beim Würfeln hat dagegen ausschließlich Atome. Bei manchen Variablen kommt beides vor. Bei standardnormalverteilter Eingabe nimmt die Ausgabe einer ReLU-Einheit beispielsweise den Wert $0$ mit Wahrscheinlichkeit $\tfrac12$ an (ein Atom). Die andere Hälfte der Wahrscheinlichkeit verteilt sich mit einer Dichte über $(0,\infty)$. Die Einzelheiten untersuchen Sie in Aufgabe :exref:`ex_mdl-random-variables-rectified-gaussian`.

<!--en:7de9bcf5-->
Für die Tabelle oben haben wir eine exakte Flächenformel verwendet. Eine Simulation kommt ohne diese Formel aus. Um gleichverteilte Treffer auf der Kreisscheibe zu erzeugen, kann der Rechner die Pfeile gleichverteilt auf das umschließende Quadrat $[-1,1]^2$ werfen. Wir behalten nur die Treffer im Kreis. Damit sind die Treffer weiterhin gleichmäßig über die Scheibe verteilt. Im Folgenden bestimmen wir für jedes unserer drei Intervalle den Anteil der Pfeile, die hineinfallen, und teilen ihn durch die Intervallbreite.

<!--en:e4425d73-->
Wie erwartet sinkt bei jeder Verkleinerung auf ein Zehntel auch der Anteil auf ein Zehntel, während das Verhältnis nahe bei $1=p(0.5)$ bleibt. Die Tabelle mit den exakten Werten lässt allerdings nicht erkennen, wie stark das Ergebnis bei einem sehr schmalen Intervall in der Praxis rauscht. Von den vier Millionen Pfeilen treffen rund $3.1$ Millionen die Scheibe, und davon fallen nur etwa $3{,}000$ in das Intervall der Breite $0.001$. Daher schwankt das letzte Verhältnis von Lauf zu Lauf um einige Prozent.

<!--en:3400f33b-->
### Wahrscheinlichkeit als Fläche

<!--en:f94348e3-->
Bisher haben wir die Dichte aus den Wahrscheinlichkeiten winziger Intervalle abgelesen. Kehren wir die Frage um: Wie bestimmen wir bei gegebener Dichte die Wahrscheinlichkeit eines ganzen Intervalls? Zerlegen wir die Zahlengerade in Intervalle $[\epsilon i,\epsilon(i+1))$ der Breite $\epsilon$. Nach :eqref:`eq_mdl-pdf_def` hat jedes Intervall ungefähr die Wahrscheinlichkeit $\epsilon\,p(\epsilon i)$. Da die Intervalle disjunkt sind, addieren sich ihre Wahrscheinlichkeiten. Summieren wir die Produkte aus Breite und Höhe, erhalten wir eine Riemann-Summe. Sie nähert sich der Fläche unter der Kurve, also dem Integral (:numref:`sec_mdl-integral_calculus`), wenn $\epsilon$ gegen null geht. Da $X$ einen reellen Wert annehmen muss, haben alle Intervalle zusammen die Wahrscheinlichkeit eins:

<!--en:7b9ae1b7-->
Summieren wir nur über die Intervalle zwischen $a$ und $b$, liegt eine Vermutung nahe: Die Wahrscheinlichkeit eines Intervalls ist die Fläche unter der Dichte über diesem Intervall. Streng genommen ist das noch kein Beweis. Es gibt etwa $1/\epsilon$ Intervalle. Die einzelnen Fehler sind zwar klein im Vergleich zu $\epsilon$, doch ihre Summe ist nur dann vernachlässigbar, wenn die Fehler gleichmäßig in $x$ schrumpfen. Als sichere Grundlage wählen wir deshalb diese Flächenformel als formale Definition.

<!--en:24c47a5d-->
**Definition.** Eine Zufallsvariable $X$ *hat die Dichte* $p$, wenn $p\ge0$ gilt und für alle $a<b$ die folgende Gleichung erfüllt ist:

<!--en:90a800ba-->
Die Grenzübergänge $a\to-\infty$ und $b\to\infty$ bestätigen, dass die Gesamtfläche unter jeder Dichte gleich $1$ ist. Tatsächlich genügen diese beiden Bedingungen: Eine nichtnegative Funktion ($p\ge 0$) mit Gesamtfläche eins ist eine gültige Dichte. :numref:`fig_mdl-prob-pdf-area` veranschaulicht die Definition anhand einer Dichte mit zwei Gipfeln. Auch beim Dartwurf stimmt die Rechnung: Für $0\le a<b\le 1$ ergibt das Integral $\int_a^b 2r\,dr=b^2-a^2$ wieder die aus der Fläche berechnete Wahrscheinlichkeit $P(a<R\le b)$. Damit ist $2r$ tatsächlich die Dichte des Abstands. Insbesondere beträgt die Wahrscheinlichkeit eines Treffers in der äußeren Hälfte des Radius $\int_{0.5}^1 2r\,dr=1-0.25=0.75$. Ganze drei Viertel der Scheibe liegen also weiter als einen halben Radius von der Mitte entfernt.

<!--en:de85b065-->
![Eine Wahrscheinlichkeitsdichte $p(x)$. Die Gesamtfläche unter der Kurve beträgt $1$. Die Wahrscheinlichkeit, dass $X$ in das Intervall von $a$ bis $b$ fällt, entspricht der schattierten Fläche darüber, also $\int_a^b p(x)\,dx$. Die Dichte selbst ist keine Wahrscheinlichkeit (sie kann größer als $1$ sein). Erst das Integral über ein Intervall ergibt dessen Wahrscheinlichkeit.](../img/mdl-prob-pdf-area.svg)

<!--en:b8e5444e-->
### Die Verteilungsfunktion

<!--en:079a7c09-->
Oft ist eine Funktion praktisch, deren Werte selbst Wahrscheinlichkeiten sind und die diskrete wie stetige Variablen erfasst. Die *Verteilungsfunktion* (engl. cumulative distribution function, CDF) von $X$ gibt die Wahrscheinlichkeit an, dass $X$ höchstens den Wert $x$ annimmt. Hat $X$ eine Dichte, so ist das die Fläche unter der Dichte links von $x$:

<!--en:d09f4650-->
Mit anderen Worten: $F$ durchläuft die Zahlengerade von links nach rechts und summiert dabei die Wahrscheinlichkeit auf. Die Funktion beginnt bei $0$ für $x\to-\infty$ und steigt bis $1$ für $x\to\infty$. Sie kann nie abnehmen, denn die nichtnegative Dichte $p$ liefert ausschließlich nichtnegative Flächenbeiträge. Hat $X$ eine Dichte, so ist $F$ stetig. Für den Dartwurf haben wir die Verteilungsfunktion gleich zu Beginn berechnet: $F(r)=0$ für $r<0$, $F(r)=r^2$ für $0\le r\le 1$ und $F(r)=1$ für $r>1$. Von dieser Funktion sind wir ausgegangen und haben daraus direkt die Dichte $2r$ gewonnen.

<!--en:26458abf-->
Im Unterschied zur Dichte ist die Verteilungsfunktion für jede Zufallsvariable definiert. Für eine faire Münze mit den Ergebnissen $0$ und $1$ gilt

<!--en:73d6689d-->
Das ist eine *Treppenfunktion*, die an jedem Atom um dessen Wahrscheinlichkeit nach oben springt. Da $F(x)=P(X\le x)$ auch ein Atom an der Stelle $x$ selbst einschließt, nimmt die Funktion an jeder Sprungstelle den oberen Wert an. Allgemein ist jede Verteilungsfunktion *rechtsstetig*, das heißt, es gilt $\lim_{y\downarrow x}F(y)=F(x)$. Eine gemischte Variable vereint beide Verhaltensweisen: Die Verteilungsfunktion der zuvor erwähnten ReLU-Ausgabe hat den Wert $0$ für $x<0$, springt auf $\tfrac12$ bei $x=0$ und steigt dann glatt gegen $1$. Eine einzige Funktion erfasst also diskrete, stetige und gemischte Zufallsvariablen. :numref:`fig_mdl-prob-cdf-types` zeigt für jede Art ein Beispiel.

<!--en:76d489a4-->
![Drei Verteilungsfunktionen. (a) Abstand des Dartpfeils von der Mitte: $F(r)=r^2$ wächst stetig von $0$ auf $1$. (b) Faire Münze mit den Ergebnissen $0$ und $1$: Die Treppenfunktion springt an jedem Atom um $\tfrac12$ und nimmt dort den oberen Wert an (ausgefüllte Punkte). (c) ReLU einer standardnormalverteilten Variablen: $F$ ist gleich $0$ für $x<0$, springt auf $\tfrac12$ bei $x=0$ und folgt von dort der Verteilungsfunktion $\Phi(x)$ der Standardnormalverteilung.](../img/mdl-prob-cdf-types.svg)

<!--en:c2867aa6-->
Existiert eine Dichte, enthalten Verteilungsfunktion und Dichte dieselbe Information. Integrieren wir $p$, erhalten wir $F$. Leiten wir $F$ ab, gewinnen wir $p$ zurück. Damit kehren wir zum Grenzwert für immer schmalere Intervalle zurück und können ihn nun als Satz begründen.

<!--en:8c227fd5-->
**Proposition (Dichte als Ableitung der Verteilungsfunktion).** *Hat $X$ eine Dichte $p$, die an der Stelle $x$ stetig ist, so ist $F$ an der Stelle $x$ differenzierbar und es gilt*

<!--en:3fa09c75-->
*Insbesondere konvergiert die Wahrscheinlichkeit pro Längeneinheit $P(x\le X<x+\epsilon)/\epsilon$ gegen $p(x)$ für $\epsilon\downarrow0$.*

<!--en:33f7c6ac-->
**Beweis.** Nach :eqref:`eq_mdl-cdf_def` gibt $F$ die bis zur jeweiligen Stelle aufgelaufene Fläche unter $p$ an. Nach dem Hauptsatz der Differential- und Integralrechnung ist die Ableitung dieser Flächenfunktion an jeder Stetigkeitsstelle des Integranden gleich dem Integranden (:numref:`sec_mdl-integral_calculus`). Für die zweite Aussage betrachten wir den einzelnen Punkt $x$. Seine Wahrscheinlichkeit beträgt höchstens $\int_{x-\delta}^{x}p(t)\,dt$ für jedes $\delta>0$, und diese Integrale gehen mit $\delta$ gegen null. Daher spielt es keine Rolle, ob wir einen Endpunkt einschließen. Es gilt also $P(x\le X<x+\epsilon)=F(x+\epsilon)-F(x)$, und das Verhältnis ist ein Differenzenquotient von $F$. $\blacksquare$

<!--en:e20d9020-->
Beim Dartwurf gilt $F'(r)=2r$ auf $(0,1)$. Das ergibt wieder die Dichte, die wir über die schmalen Intervalle bestimmt haben. :numref:`fig_mdl-prob-pdf-cdf` zeigt beide Funktionen nebeneinander: Die schattierte Fläche $\int_a^b p$ links entspricht dem Höhenunterschied $F(b)-F(a)$ rechts.

<!--en:d589375f-->
![Links die Dichte $p$ mit der schattierten Fläche über dem Intervall von $a$ bis $b$. Rechts die zugehörige Verteilungsfunktion $F(x)=\int_{-\infty}^x p$, die von $0$ auf $1$ steigt. Der Höhenunterschied $F(b)-F(a)$ entspricht der schattierten Fläche links. Die Dichte ist die Steigung der Verteilungsfunktion $\left(F'=p\right)$, die Verteilungsfunktion die aufsummierte Fläche unter der Dichte.](../img/mdl-prob-pdf-cdf.svg)

<!--en:046ae7f7-->
### Quantile und Inversionsmethode

<!--en:183dd68d-->
Mit dem Rechner lassen sich leicht Zufallszahlen erzeugen, die zwischen $0$ und $1$ gleichverteilt sind. Für die Simulation des Dartwurfs oder einer anderen stetigen Verteilung brauchen wir ein Verfahren, das diese gleichverteilten Zufallszahlen in Samples aus unserer Zielverteilung umwandelt.

<!--en:c6f80010-->
Ausgangspunkt ist die einfachste aller Dichten. Eine Zufallsvariable $U$ heißt *gleichverteilt auf $[0,1]$*, wenn ihre Dichte gleich $1$ auf $[0,1]$ und sonst gleich $0$ ist. Jedes Teilintervall von $[0,1]$ hat dann eine Wahrscheinlichkeit gleich seiner Länge, und für die Verteilungsfunktion gilt $F_U(u)=u$ für $0\le u\le 1$.

<!--en:20552630-->
Die Verteilungsfunktion ordnet einem Wert $x$ ein Wahrscheinlichkeitsniveau $u$ zu. Fragen wir umgekehrt, unterhalb welchen Werts ein vorgegebener Anteil $u$ der Wahrscheinlichkeit liegt, erhalten wir die *Quantilfunktion*. Ist $F$ auf dem Intervall $I$, in dem $0<F<1$ gilt, stetig und streng monoton steigend, so bildet die Funktion $I$ bijektiv auf $(0,1)$ ab. Ihre Umkehrfunktion $F^{-1}:(0,1)\to I$ ist die Quantilfunktion. Ihre Werte bei $u=0.25, 0.5, 0.75$ heißen *Quartile*. Das mittlere Quartil $F^{-1}(0.5)$ heißt *Median*. Beim Dartwurf liefert die Umkehrung von $F(r)=r^2$ die Funktion $F^{-1}(u)=\sqrt{u}$ und damit die Quartile $0.5$, $0.707$ und $0.866$. Der Median des Abstands liegt mit $0.707$ deutlich jenseits des halben Radius, denn drei Viertel der Scheibenfläche liegen außerhalb des Radius $0.5$.

<!--en:e3299ad4-->
Bemerkenswerterweise liefert die Quantilfunktion auch ein universelles Verfahren zum Ziehen von Samples.

<!--en:5b0bf111-->
**Proposition (Inversionsmethode).** *Sei $F$ eine stetige Verteilungsfunktion, die auf dem Intervall $I$ mit $0<F<1$ streng monoton steigt, und sei $U$ gleichverteilt auf $[0,1]$. Dann hat $X=F^{-1}(U)$ die Verteilungsfunktion $F$.*

<!--en:a501ce81-->
**Beweis.** Mit Wahrscheinlichkeit eins liegt $U$ in $(0,1)$. Somit ist $X$ definiert und liegt in $I$. Für $x$ in $I$ stimmen die Ereignisse $F^{-1}(U)\le x$ und $U\le F(x)$ überein, da $F$ auf $I$ streng monoton steigt. Da $U$ auf $[0,1]$ gleichverteilt ist, hat das zweite Ereignis eine Wahrscheinlichkeit gleich der Länge von $[0,F(x)]$:

<!--en:a93a3515-->
Für $x$ links von $I$ gilt $F(x)=0$ und stets $X>x$. Für $x$ rechts von $I$ gilt $F(x)=1$ und stets $X\le x$. $\blacksquare$

<!--en:2cea06af-->
Die Voraussetzung lässt zu, dass $F$ außerhalb eines Intervalls konstant ist, wie beim Dartwurf ($F=0$ für $r<0$ und $F=1$ für $r>1$). Ersetzen wir $F^{-1}$ durch die *verallgemeinerte Inverse* $F^{-}(u)=\min\{x:F(x)\ge u\}$, entfällt die Voraussetzung ganz. Dann lassen sich auch Samples aus Verteilungen mit treppenförmiger Verteilungsfunktion ziehen.

<!--en:4b30b511-->
Beim Dartwurf liefert die Inversionsmethode $R=\sqrt{U}$. Die Abkürzung $R=U$ liegt nahe. Warum sie scheitert, lässt sich ohne eine einzige Zeile Code erkennen: Mit $R=U$ landet die Hälfte der Pfeile im Bereich $r<0.5$, obwohl dort nur ein Viertel landen sollte. :numref:`fig_mdl-prob-inverse-transform` veranschaulicht die Konstruktion an unserer Dichte mit zwei Gipfeln. Wo die Dichte hoch ist, steigt die Verteilungsfunktion steil an. Gleichmäßig verteilte Niveaus auf der senkrechten Achse ergeben dort dicht beieinanderliegende Samples auf der waagerechten Achse.

<!--en:75343d2a-->
![Inversionsmethode. Die gleichverteilten Niveaus $U$ auf der senkrechten Achse (links) führen über die Verteilungsfunktion $F$ zu den Werten $x=F^{-1}(U)$ auf der waagerechten Achse. Wo $F$ steil ansteigt, also die Dichte $p=F'$ groß ist, ergeben gleichmäßig verteilte Niveaus dicht beieinanderliegende Samples. Das Histogramm der so gewonnenen Punkte bildet daher $p$ nach.](../img/mdl-prob-inverse-transform.svg)

<!--en:cb474090-->
Für den Abstand des Dartpfeils haben wir nun zwei ganz unterschiedliche Verfahren: Wir können Pfeile auf das Quadrat werfen und nur die Treffer auf der Scheibe behalten, oder wir berechnen direkt $\sqrt{U}$. Die folgende Codezelle vergleicht beide Verfahren mit der naiven Abkürzung $R=U$.

<!--en:5fc29ee2-->
Die beiden korrekten Verfahren stimmen überein. Bei beiden liegt ein Viertel der Abstände unter $0.5$, und der mittlere Abstand liegt nahe bei $0.667$. Diesen Wert leiten wir im nächsten Unterabschnitt her. Bei der naiven Abkürzung $U$ liegt die Hälfte der Samples unter $0.5$, denn sie verteilt die Abstände gleichmäßig über $[0,1]$. Dabei bleibt unberücksichtigt, dass die Scheibe am Rand viel mehr Fläche hat als nahe der Mitte.

<!--en:bab057e7-->
## Mittelwert und Varianz

<!--en:c06c4294-->
Eine Dichte beschreibt eine Verteilung vollständig. Oft sagen uns aber schon zwei Kennzahlen das Wesentliche: Wo liegt die Mitte der Verteilung, und wie weit streuen die Werte? In :numref:`sec_prob` haben wir beide Kennzahlen für diskrete Variablen eingeführt und ihre Verwendung begründet. Nun übertragen wir sie auf Dichten und beweisen die Rechenregeln, die wir verwenden.

<!--en:a7abf451-->
### Der Mittelwert

<!--en:16abd423-->
Der Mittelwert mittelt die Werte von $X$ und gewichtet jeden Wert mit seiner Wahrscheinlichkeit. Hat die Variable eine Dichte, zerlegen wir die Zahlengerade wie zuvor in Intervalle der Breite $\epsilon$. Der Beitrag des Intervalls bei $\epsilon i$ ist sein Wert $\epsilon
i$ mal seiner Wahrscheinlichkeit, die ungefähr $\epsilon\,p(\epsilon i)$ beträgt. Die Summe dieser Beiträge konvergiert gegen ein Integral. Der *Mittelwert* oder *Erwartungswert* von $X$ ist daher

<!--en:0cacd263-->
Wie im diskreten Fall ist der Mittelwert ein Schwerpunkt: Ein dünner Stab mit Massendichte $p$ entlang seiner Länge ist im Gleichgewicht, wenn wir ihn bei $\mu_X$ unterstützen. Damit diese Definition sinnvoll ist, verlangen wir $\sum_i|x_i|\,p_i<\infty$ im diskreten Fall und $\int|x|\,p(x)\,dx<\infty$ bei einer Dichte. Ist diese Bedingung verletzt, existiert der Mittelwert nicht. Eine solche Dichte lernen wir in Kürze kennen. Mit denselben Wahrscheinlichkeiten als Gewichten können wir jede Funktion $g(X)$ direkt mitteln:

<!--en:4a3aceff-->
Auch hier muss die Summe oder das Integral über $|g|$ mit den entsprechenden Wahrscheinlichkeitsgewichten endlich sein. Das ist praktisch: Wir gewichten die Werte von $g$ mit den Wahrscheinlichkeiten von $X$, ohne zuvor die Verteilung der transformierten Variablen $g(X)$ bestimmen zu müssen. Diese Regel heißt *Transformationssatz für Erwartungswerte* (engl. law of the unconscious statistician). Der englische Name spielt darauf an, dass man die Regel so selbstverständlich anwendet, dass man sie leicht für die Definition von $\mathbb{E}[g(X)]$ hält. Tatsächlich ist sie ein Satz. Beachten Sie, dass $\mathbb{E}[g(X)]$ im Allgemeinen von $g(\mathbb{E}[X])$ abweicht. Ist $g$ konvex, gibt die Jensen-Ungleichung (:numref:`subsec_mdl-jensen`) an, welche der beiden Größen größer ist.

<!--en:9cc3fe99-->
Für unseren Dartwurf gilt

<!--en:89da6252-->
Der mittlere Abstand beträgt $0.667$ statt $0.5$, denn nahe am Rand hat die Scheibe mehr Fläche als nahe der Mitte. Der Mittelwert liegt außerdem knapp unter dem Median $0.707$. Die vergleichsweise wenigen Treffer nahe der Mitte ziehen den Durchschnitt stärker nach unten, als sie den mittleren Wert in der geordneten Reihe verschieben.

<!--en:c55e32f7-->
Die am häufigsten genutzte Eigenschaft des Erwartungswerts ist seine Linearität. In :numref:`sec_prob` haben wir sie bereits verwendet. Jetzt beweisen wir sie.

<!--en:1d547a72-->
**Proposition (Linearität des Erwartungswerts).** *Für Zufallsvariablen $X,Y$, deren Mittelwerte existieren, und Konstanten $c,d$ gilt*

<!--en:5db89b9c-->
**Beweis.** Beide Identitäten folgen aus der Linearität der Summe oder des Integrals, das den Mittelwert definiert. Für die erste Identität erhalten wir $\mathbb{E}[cX+d]=\sum_i(cx_i+d)p_i = c\sum_i x_i p_i + d\sum_i p_i =
c\,\mathbb{E}[X]+d$, wobei wir $\sum_i p_i=1$ verwenden. Für die zweite bezeichne $p_{ij}$ die Wahrscheinlichkeit, dass $(X,Y)=(x_i,y_j)$ gilt. Das ist die gemeinsame Verteilung des Paars wie in :numref:`sec_prob`. Zerlegen wir die Summe, erkennen wir die Randwahrscheinlichkeiten $\sum_j p_{ij}=P(X=x_i)$ und $\sum_i p_{ij}=P(Y=y_j)$ und erhalten

<!--en:fdc7f154-->
Für Dichten verläuft die Rechnung genauso, nur integrieren wir über die gemeinsame Dichte. Sie ist das stetige Gegenstück zu $p_{ij}$ und wird in :numref:`sec_mdl-joint-distributions` definiert. $\blacksquare$

<!--en:1c6c0f78-->
Entscheidend ist, dass die zweite Identität keinerlei Unabhängigkeit voraussetzt: Der Mittelwert einer Summe ist die Summe der Mittelwerte, wie stark $X$ und $Y$ auch voneinander abhängen. Immer wenn wir den erwarteten Verlust als Mittelwert über einen Datensatz berechnen, nutzen wir diese Eigenschaft.

<!--en:8c574509-->
### Varianz und Standardabweichung

<!--en:ef9bf0db-->
Der Mittelwert beschreibt die Lage einer Verteilung. Wie eng die Werte um ihn liegen, geht daraus nicht hervor. Wie in :numref:`sec_prob` messen wir die Streuung anhand der mittleren quadratischen Abweichung vom Mittelwert. Sie heißt *Varianz*:

<!--en:9ee94c83-->
Sie ist definiert, wenn $\mathbb{E}[X^2]<\infty$ gilt. Multiplizieren wir das Quadrat aus und nutzen die Linearität :eqref:`eq_mdl-exp_linear`, wobei wir $\mu_X$ als Konstante behandeln, erhalten wir eine meist leichter berechenbare Form: den Mittelwert des Quadrats minus dem Quadrat des Mittelwerts,

<!--en:272c24ec-->
Für eine Variable mit Dichte ergeben sich Mittelwert und Varianz also aus den beiden Integralen

<!--en:f1c89a4e-->
Für den Dartwurf gilt $\mathbb{E}[R^2]=\int_0^1 r^2\cdot 2r\,dr=\tfrac12$ und somit $\textrm{Var}(R)=\tfrac12-\tfrac49=\tfrac1{18}\approx 0.056$. Für die gleichverteilte Variable $U$ auf $[0,1]$ erhalten wir $\mu_U=\int_0^1 u\,du=\tfrac12$ und $\textrm{Var}(U)=\tfrac13-\tfrac14=\tfrac1{12}$.

<!--en:3587c835-->
Einheiten umzurechnen oder Daten zu zentrieren sind Beispiele für affine Transformationen. Bei der affinen Abbildung $cX+d$ entfällt die Verschiebung $d$ in der Varianz, während der Skalierungsfaktor $c$ quadratisch eingeht. Aus der Linearität folgt nämlich $\mu_{cX+d}=c\mu_X+d$. Die Abweichung vom Mittelwert beträgt also $(cX+d)-\mu_{cX+d}=c(X-\mu_X)$. Quadrieren wir diesen Ausdruck und bilden den Erwartungswert, erhalten wir

<!--en:c21c3ea8-->
Zwei weitere Eigenschaften sind wichtig. Erstens ist die Varianz nie negativ. Sie ist genau dann null, wenn $X$ mit Wahrscheinlichkeit eins konstant ist. Denn der Erwartungswert der nichtnegativen Größe $(X-\mu_X)^2$ kann nur verschwinden, wenn sie mit Wahrscheinlichkeit eins null ist. Zweitens addieren sich die Varianzen unabhängiger Zufallsvariablen (:numref:`sec_prob`). Das beweisen wir in :numref:`subsec_mdl-covariance`. Dort leiten wir auch den Kovarianzterm her, der bei abhängigen Variablen hinzukommt.

<!--en:c8a2d823-->
Da die Varianz die Abweichungen quadriert, hat sie eine quadrierte Einheit. Messen wir etwa einen Gewinn in Dollar, so hat seine Varianz die Einheit Dollar zum Quadrat. Die Quadratwurzel liefert wieder die ursprüngliche Einheit. Sie heißt *Standardabweichung*:

<!--en:b7e50e42-->
Bei affinen Abbildungen skaliert sie gemäß $\sigma_{cX+d}=|c|\,\sigma_X$ linear mit dem Betrag des Skalierungsfaktors. Für unseren Dartwurf gilt $\sigma_R=\sqrt{1/18}\approx 0.236$, gemessen in Einheiten des Scheibenradius.

<!--en:c4214046-->
## Verteilungsränder

<!--en:a6c8665f-->
Die Standardabweichung liefert einen natürlichen Maßstab für typische Schwankungen um den Mittelwert. Wie viel Wahrscheinlichkeit kann nun weit draußen an den Verteilungsrändern liegen, wenn wir nur den Mittelwert oder nur Mittelwert und Varianz kennen? Zwei klassische Ungleichungen beantworten diese Frage für alle Verteilungen, die diese Voraussetzungen erfüllen. Beide Schranken sind recht grob. Überraschenderweise lassen sie sich ohne weitere Annahmen dennoch nicht verbessern, wie wir an einer Verteilung auf nur drei Punkten sehen. Anschließend betrachten wir eine glockenförmige Dichte, die weder einen Mittelwert noch eine Varianz besitzt. An ihr wird deutlich, was die beiden Voraussetzungen ausschließen.

<!--en:659653f3-->
### Die Ungleichungen von Markov und Chebyshev

<!--en:c8117ae8-->
Die erste Schranke benötigt nur den Mittelwert und gilt für jede nichtnegative Zufallsvariable mit endlichem Mittelwert. Ihr Beweis passt in eine Zeile. Dennoch erhalten wir fast jede Schranke dieses Kapitels für die Wahrscheinlichkeit großer Abweichungen, indem wir sie auf eine passend gewählte Funktion der Variablen anwenden.

<!--en:e0cf3c31-->
**Proposition (Markov-Ungleichung).** *Hat $X\ge 0$ einen Mittelwert und gilt $t>0$, so folgt*

<!--en:c8e723be-->
**Beweis.** Sei $\mathbf{1}_{X\ge t}$ die Indikatorvariable mit dem Wert $1$ für $X\ge t$ und dem Wert $0$ sonst. Dann gilt $X \ge t\,\mathbf{1}_{X\ge t}$ für jedes Ergebnis: Für $X\ge t$ gilt auf der rechten Seite die Abschätzung $t\le X$, andernfalls gilt $0\le X$. Bilden wir auf beiden Seiten den Erwartungswert und verwenden $\mathbb{E}[\mathbf{1}_{X\ge t}]=P(X\ge t)$, folgt $\mathbb{E}[X]\ge t\,P(X\ge t)$. $\blacksquare$

<!--en:593c4649-->
Geometrisch betrachtet liegt auf $[0,\infty)$ die folgende Stufenfunktion unterhalb der Geraden oder auf ihr: Sie hat den Wert $0$ unterhalb von $t$ und den Wert $t$ ab $t$. Die Gerade ist $y=x$ (:numref:`fig_mdl-prob-markov-step`). Daher ist der mittlere Wert der Stufenfunktion $t\,P(X\ge t)$ höchstens so groß wie der Mittelwert von $X$. Setzen wir $t$ gleich dem Zehnfachen des Mittelwerts, ergibt sich eine Faustregel: In jeder nichtnegativen Grundgesamtheit kann höchstens ein Zehntel der Werte das Zehnfache des Durchschnitts übersteigen. Das gilt für Einkommen, Dateigrößen und Gradientennormen gleichermaßen.

<!--en:5aebaa75-->
![Geometrie der Markov-Ungleichung. Für $x\ge0$ liegt die Stufenfunktion $t\,\mathbf{1}\{x\geq t\}$ (orange) auf oder unter der Geraden $y=x$ (blau). Ihr Mittelwert $t\,P(X\geq t)$ kann daher $\mathbb{E}[X]$ nicht übersteigen. Die schattierte Fläche zeigt den Abstand zwischen den Funktionen, den die Schranke unberücksichtigt lässt.](../img/mdl-prob-markov-step.svg)

<!--en:a2875b30-->
Da die Markov-Ungleichung nur den Mittelwert nutzt, muss sie grob sein. Wenden wir sie auf die quadratische Abweichung $(X-\mu_X)^2$ an, deren Mittelwert die Varianz ist, erhalten wir eine Schranke für beide Verteilungsränder zugleich. Dieses Ergebnis geht auf Bienaymé und Chebyshev zurück :cite:`Chebyshev.1867`.

<!--en:b340c2ab-->
**Proposition (Chebyshev-Ungleichung).** *Für jede Zufallsvariable $X$ mit endlicher, von null verschiedener Varianz und jedes $\alpha>0$ gilt*

<!--en:522deabc-->
**Beweis.** Die Variable $Z=(X-\mu_X)^2$ ist nichtnegativ und hat den Mittelwert $\textrm{Var}(X)=\sigma_X^2$. Das Ereignis $|X-\mu_X|\ge\alpha\sigma_X$ stimmt mit dem Ereignis $Z\ge\alpha^2\sigma_X^2$ überein. Wegen $\sigma_X>0$ ist die Schwelle $t=\alpha^2\sigma_X^2$ positiv. Aus der Markov-Ungleichung :eqref:`eq_mdl-markov` folgt daher $P(|X-\mu_X|\ge\alpha\sigma_X) \le \sigma_X^2/(\alpha^2\sigma_X^2) = 1/\alpha^2$. $\blacksquare$

<!--en:a52c13e5-->
Insbesondere können bei jeder solchen Verteilung höchstens $1\%$ der Wahrscheinlichkeit auf Werte entfallen, die mindestens zehn Standardabweichungen vom Mittelwert entfernt liegen. Führen wir denselben Beweis mit einer absoluten Schwelle $\epsilon>0$ anstelle von $\alpha\sigma_X$, erhalten wir eine gleichwertige Form. Sie verlangt nur eine endliche Varianz und erfasst auch den entarteten Fall $\sigma_X=0$:

<!--en:d1130cdf-->
Für den Dartwurf gilt $\mu_R=0.667$ und $\sigma_R=0.236$. Zwei Standardabweichungen unter dem Mittelwert liegt der Abstand $0.195$. Zwei Standardabweichungen darüber lägen wir schon außerhalb der Scheibe. Daher gilt $P(|R-\mu_R|\ge 2\sigma_R)=P(R\le 0.195)=0.195^2\approx 0.038$. Das liegt weit unter der Chebyshev-Schranke von $\tfrac14$.

<!--en:cf583159-->
Beim Dartwurf ist $\tfrac14$ eine sehr grobe Schranke. Lässt sich dennoch eine bessere Schranke finden, die für jede Verteilung gilt? Ohne weitere Annahmen ist das unmöglich. Zum Beweis genügt eine Verteilung auf drei Punkten. Die Variable $X$ nehme die Werte $a-2$ und $a+2$ mit Wahrscheinlichkeit $q$ je Wert an und den Wert $a$ mit der verbleibenden Wahrscheinlichkeit $1-2q$, wobei $0<q\le\tfrac12$ gilt. Wir nennen dies das *Beispiel mit drei Atomen*. Aus der Symmetrie folgt der Mittelwert $a$. Die Varianz beträgt $\textrm{Var}(X)=\mathbb{E}[(X-a)^2]=4q+4q=8q$, also ist $\sigma_X=2\sqrt{2q}$. Wählen wir $\alpha=2$. Wenn $q$ sinkt, sinkt auch die Schwelle $2\sigma_X=4\sqrt{2q}$, während die beiden äußeren Atome im festen Abstand $2$ von $a$ bleiben. :numref:`fig_mdl-prob-chebyshev` zeigt die drei möglichen Fälle. Für $q>\tfrac18$ liegt die Schwelle über $2$. Keines der äußeren Atome trägt dann zum Ereignis bei, die Wahrscheinlichkeit ist $0$. Bei $q=\tfrac18$ liegt die Schwelle genau bei $2$. Damit gilt $\sigma_X=1$, und die Werte $a\pm 2$ liegen zwei Standardabweichungen vom Mittelwert entfernt. Beide äußeren Atome zählen jetzt mit. Ihre Gesamtwahrscheinlichkeit beträgt $2q=\tfrac14$ und erreicht die Chebyshev-Schranke genau. Für $q<\tfrac18$ liegen beide äußeren Atome weiterhin jenseits der Schwelle, doch ihre gemeinsame Masse $2q$ liegt unter $\tfrac14$.

<!--en:91842d93-->
![Chebyshev-Schranke im Beispiel mit drei Atomen. Senkrechte Balken zeigen die Massen $q$, $1-2q$ und $q$ an den Stellen $a-2$, $a$ und $a+2$. Für den Schwellenabstand $\alpha\sigma_X=4\sqrt{2q}$ bei $\alpha=2$ reicht die waagerechte Klammer von $a-\alpha\sigma_X$ bis $a+\alpha\sigma_X$. Die drei Teilbilder zeigen die Fälle $q>1/8$, $q=1/8$ (die Schranke wird genau erreicht) und $q<1/8$. Orangefarbene Atome liegen auf oder jenseits der Schwelle und zählen zum Ereignis $|X-a|\ge\alpha\sigma_X$.](../img/mdl-prob-chebyshev.svg)

<!--en:0d2d9b5d-->
Dieselbe Konstruktion funktioniert für jedes $\alpha\ge 1$. Wählen wir $q=1/(2\alpha^2)$, also einen Wert von höchstens $\tfrac12$, erhalten wir $\sigma_X=2\sqrt{2q}=2/\alpha$. Damit trifft die Schwelle $\alpha\sigma_X=2$ genau die äußeren Atome, und es gilt $P(|X-a|\ge\alpha\sigma_X)=2q=1/\alpha^2$. Die Schranke wird erreicht. Keine Konstante kleiner als $1/\alpha^2$ kann also für alle Verteilungen gelten. (Für $\alpha<1$ übersteigt die Schranke $1$ und ist damit aussagelos.)

<!--en:f94bedc0-->
Die Chebyshev-Ungleichung erfasst also den ungünstigsten Fall unter allen Verteilungen mit gegebener Varianz. Für die üblichen Verteilungen ist sie recht grob. Betrachten wir die Gauß-Verteilung mit Mittelwert $\mu$ und Standardabweichung $\sigma$, also die Glockenkurve mit Dichte $e^{-(x-\mu)^2/(2\sigma^2)}/(\sigma\sqrt{2\pi})$, die wir in :numref:`subsec_mdl-gaussian-clt` ausführlich untersuchen. Jenseits von zwei beziehungsweise drei Standardabweichungen liegt weit weniger Wahrscheinlichkeit, als die Chebyshev-Schranke zulässt:

<!--en:6e1398d3-->
| Abstand vom Mittelwert | Chebyshev-Schranke | Gauß-Verteilung |

<!--en:a6897de4-->
| $2\sigma$ oder mehr | $25\%$ | $4.6\%$ |

<!--en:7ab0f230-->
| $3\sigma$ oder mehr | $11\%$ | $0.27\%$ |

<!--en:b267edee-->
Im ungünstigsten Fall konzentriert sich die gesamte Masse der Verteilungsränder auf einen einzigen Abstand, wie in unserem Beispiel mit drei Atomen. Bei typischen Verteilungen verteilt sich die Masse auf mehrere Abstände und nimmt nach außen rasch ab.

<!--en:4f273728-->
### Schwere Verteilungsränder

<!--en:697471b1-->
Die Chebyshev-Ungleichung setzt eine endliche Varianz voraus. Schon der Mittelwert erfordert $\int|x|\,p(x)\,dx<\infty$. Überraschenderweise kann selbst eine glatte, harmlos aussehende Glockenkurve beide Bedingungen verletzen. Die *Cauchy-Verteilung* hat die Dichte

<!--en:77b55ecc-->
Das ist eine gültige Dichte, denn es gilt $\int_{-\infty}^\infty\frac{dx}{1+x^2}=[\arctan
x]_{-\infty}^{\infty}=\pi$. Während die Ränder einer Gauß-Dichte wie $e^{-x^2/2}$ steil abfallen, fallen die Ränder der Cauchy-Dichte nur algebraisch ab, wie $1/(\pi x^2)$. Betrachten wir zuerst das zweite Moment. Integration über $[-L,L]$ ergibt

<!--en:593d6236-->
Dieser Ausdruck nimmt die Werte $5.43$, $62.7$ und $636$ für $L=10$, $100$ und $1000$ an. Er wächst unbeschränkt, denn der Integrand $x^2/(\pi(1+x^2))$ nähert sich der von null verschiedenen Konstanten $1/\pi$. Somit gilt $\mathbb{E}[X^2]=\infty$, und die Cauchy-Verteilung hat keine endliche Varianz.

<!--en:8e29a61c-->
Beim Mittelwert liegt der Grund etwas tiefer. Da der Integrand $x\,p(x)$ ungerade ist, könnte man aus der Symmetrie auf den Mittelwert $0$ schließen. Tatsächlich gilt $\int_{-L}^{L}x\,p(x)\,dx=0$ für jedes $L$. Die Definition des Erwartungswerts verlangt jedoch absolute Konvergenz, und der Ausdruck

<!--en:87e97c99-->
wächst für $L\to\infty$ unbeschränkt. Der Mittelwert der Cauchy-Verteilung existiert also nicht. Ohne absolute Konvergenz hängt der Integralwert davon ab, wie die Integrationsgrenzen gegen unendlich streben. Über das asymmetrische Intervall $[-L,2L]$ erhalten wir beispielsweise

<!--en:b3a9ebff-->
Was geschieht, wenn wir unabhängige Ziehungen aus einer Cauchy-Verteilung mitteln, obwohl es keinen Mittelwert gibt, dem sich der Durchschnitt nähern könnte? Man könnte immerhin hoffen, dass der Durchschnitt aus $1000$ Ziehungen näher bei $0$ liegt als eine einzelne Ziehung. Außerhalb von $[-1,1]$ liegt die einzelne Ziehung mit Wahrscheinlichkeit $P(|X|>1)=1-\tfrac2\pi\arctan 1=\tfrac12$. Die folgende Codezelle zeichnet den fortlaufenden Mittelwert von $10^5$ Cauchy-Ziehungen neben dem von $10^5$ Ziehungen aus der Standardnormalverteilung. Dann zählt sie, wie viele von $1000$ getrennt berechneten Mittelwerten (jeweils aus $1000$ Cauchy-Ziehungen) noch immer weiter als $1$ von $0$ entfernt liegen.

<!--en:b8b55e4d-->
Bei der Gauß-Verteilung nähert sich der fortlaufende Mittelwert mit immer kleineren Schwankungen dem Wert $0$. Bei der Cauchy-Verteilung kommt er nie zur Ruhe. Jedes Mal, wenn eine einzelne Ziehung einen extremen Wert liefert, springt der Mittelwert abrupt. Ein Sprung der Größenordnung eins kann bei jedem $n$ auftreten. Das Mitteln bringt keinerlei Verbesserung. Etwa die Hälfte der Mittelwerte aus $1000$ Ziehungen liegt weiterhin mehr als $1$ von $0$ entfernt. Das ist derselbe Anteil wie bei einer einzelnen Ziehung. Tatsächlich hat der Mittelwert von $n$ unabhängigen standard-Cauchy-verteilten Variablen dieselbe Verteilung wie eine einzelne standard-Cauchy-verteilte Variable. Das geben wir hier ohne Beweis an.

<!--en:a8fb5974-->
Eine Verteilung hat *schwere Verteilungsränder* (engl. heavy tails), wenn ihre Randwahrscheinlichkeiten langsamer abfallen als jede Exponentialfunktion $e^{-\lambda t}$ mit $\lambda>0$. Das klassische Beispiel sind Ränder, die einem Potenzgesetz folgen: Fällt $P(|X|>t)$ wie $t^{-\alpha}$ ab, ist jedes Moment $\mathbb{E}[|X|^k]$ mit $k\ge\alpha$ unendlich. Die Cauchy-Verteilung entspricht dem Fall $\alpha=1$. Schwere Verteilungsränder bedeuten jedoch nicht zwangsläufig unendliche Momente. Die Lognormalverteilung, die wir in :numref:`subsec_mdl-change-of-variables` herleiten, hat schwere Verteilungsränder, aber ausschließlich endliche Momente. Im Deep Learning spielen schwere Verteilungsränder eine Rolle, weil Stichprobenmittelwerte dann langsam oder gar nicht konvergieren. So wurde für das Rauschen von Minibatch-Gradienten eine Verteilung mit schweren Rändern vorgeschlagen :cite:`Simsekli.Sagun.Gurbuzbalaban.2019`. Diese Hypothese können Sie in Aufgabe :exref:`ex_mdl-random-variables-heavy-tail-averages` prüfen.

<!--en:ff82de90-->
Bisher haben wir jeweils eine einzelne Zahl betrachtet. Unser Dartpfeil landet jedoch in einer Ebene, und sein Treffer hat zwei Koordinaten. Wie sich zeigt, sind diese Koordinaten unkorreliert, hängen aber stark voneinander ab. Um das zu verstehen, brauchen wir Dichten auf der Ebene. Damit befassen wir uns in :numref:`sec_mdl-joint-distributions`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:91e569fc-->
Bei einer stetigen Zufallsvariablen hat jeder einzelne Wert die Wahrscheinlichkeit null. Wir beschreiben ihre Verteilung hier durch eine Dichte, die Wahrscheinlichkeit pro Längeneinheit. Das Integral der Dichte über ein Intervall ergibt dessen Wahrscheinlichkeit. Die Verteilungsfunktion summiert die Fläche von links nach rechts auf, wobei $F'=p$ gilt. Ihre Umkehrfunktion wandelt gleichverteiltes Rauschen in Samples aus der Verteilung um. Die Ungleichungen von Markov und Chebyshev liefern allein aus dem Mittelwert einer nichtnegativen Zufallsvariablen oder aus der Varianz Schranken für die Wahrscheinlichkeit großer Abweichungen. Die Chebyshev-Schranke ist im ungünstigsten Fall scharf, bei typischen Verteilungen aber grob. Bei schweren Verteilungsrändern können wie bei der Cauchy-Verteilung sowohl Mittelwert als auch Varianz fehlen. Bei der Cauchy-Verteilung verringert sich die Streuung durch Mitteln der Ziehungen überhaupt nicht. Merken Sie sich vor allem die Markov-Ungleichung: Aus ihr folgen alle Schranken für große Abweichungen in :numref:`sec_mdl-concentration-generalization`.

<!--en:2294243a-->
## Übungen

<!--en:798da330-->
1. **Verteilungsrand mit Potenzgesetz.** Sei $X$ eine Zufallsvariable mit der Dichte $p(x) = 1/x^2$ für $x \ge 1$ und $p(x) = 0$ sonst.
    1. Prüfen Sie, ob $p$ eine Dichte ist, und berechnen Sie $P(X > 2)$ sowie die Verteilungsfunktion $F(x)$ für alle reellen $x$.
    1. Bestimmen Sie die Quantilfunktion $F^{-1}(u)$ für $u \in (0, 1)$, den Median und die beiden anderen Quartile. Geben Sie die Formel $X = F^{-1}(U)$ aus :numref:`fig_mdl-prob-inverse-transform` für die Inversionsmethode bei dieser Verteilung an.
    1. Zeigen Sie, dass $\mathbb{E}[X] = \infty$ gilt. Begrenzen wir $X$ auf einen Höchstwert $c > 1$, wie beim Gradienten-Clipping für eine Gradientennorm, erhalten wir $\min(X, c)$. Berechnen Sie $\mathbb{E}[\min(X, c)]$ und beschreiben Sie, wie dieser Erwartungswert mit $c$ wächst.
    1. Leiten Sie für die Cauchy-Dichte $p_C(x) = 1/\bigl(\pi(1+x^2)\bigr)$ die im Text angegebenen Werte von $\int_{-L}^{L} x\,p_C(x)\,dx$ und $\int_{-L}^{2L} x\,p_C(x)\,dx$ sowie deren Grenzwerte für $L \to \infty$ her. Was sagen die beiden Grenzwerte über den Mittelwert der Cauchy-Verteilung aus? Warum kann der Mittelwert der nichtnegativen Variablen $X$ aus Teil (a) nicht in dieser Weise mehrdeutig sein?
   :exlabel:`ex_mdl-random-variables-power-law-tail`

<!--en:fed4817c-->
1. **Gleichgerichtete Gauß-Verteilung.** Sei $X \sim \mathcal{N}(0, 1)$ standardnormalverteilt mit Dichte $p_X(x) = \tfrac{1}{\sqrt{2\pi}}e^{-x^2/2}$ und Verteilungsfunktion $\Phi(x) = \int_{-\infty}^x p_X(t)\,dt$, und sei $Y = \max(0, X)$. Dies ist die Ausgabe einer ReLU-Einheit mit standardnormalverteilter Präaktivierung, ein gängiges Modell für eine Einheit bei der Initialisierung.
    1. Drücken Sie die Verteilungsfunktion $F_Y$ durch $\Phi$ aus und skizzieren Sie sie. Wo hat $Y$ ein Atom und wie groß ist dessen Masse? Ist $Y$ diskret, hat die Variable eine Dichte oder ist ihre Verteilung gemischt?
    1. Erklären Sie, warum $Y$ keine Dichte hat. Für eine streng monotone, stetig differenzierbare Transformation $Y = g(X)$ liefert die Regel zur Variablentransformation aus :numref:`subsec_mdl-change-of-variables` die Dichte $p_Y(y) = p_X(g^{-1}(y))\,|(g^{-1})'(y)|$. Welche Voraussetzung ist bei der ReLU verletzt?
    1. Begrenzen wir die Werte auf $[-c, c]$ mit einer Schwelle $c > 0$, wie bei der Hard-Tanh-Aktivierung für $c = 1$, erhalten wir $C = \min(\max(X, -c), c)$. Geben Sie die Verteilungsfunktion von $C$ an und bestimmen Sie die Schwelle $c$, bei der die Atome von $C$ zusammen $10\%$ der Wahrscheinlichkeit tragen. Die Gleichung hat keine geschlossene Lösung. Verwenden Sie Tabellenwerte von $\Phi$ oder `math.erf` mit $\Phi(z) = \tfrac12\bigl(1 + \operatorname{erf}(z/\sqrt{2})\bigr)$.
    1. Zeigen Sie, dass $\mathbb{E}[Y^2] = \tfrac12$ und $\mathbb{E}[Y] = 1/\sqrt{2\pi}$ gilt. Hinweis: Schreiben Sie $X^2 = Y^2 + \min(0, X)^2$ und nutzen Sie die Symmetrie von $X$. Die He-Initialisierung aus :numref:`subsec_he_init` multipliziert die Varianz der Gewichte mit $2$. Bei welchem der beiden Momente gleicht dieser Faktor die Änderung aus?
   :exlabel:`ex_mdl-random-variables-rectified-gaussian`

<!--en:b9f4c76a-->
1. [code] **Logistische Verteilung und Kalibrierung.** Die logistische Verteilung hat die Verteilungsfunktion $F(x) = \operatorname{sigmoid}(x) = 1/(1 + e^{-x})$.
    1. Bestimmen Sie ihre Dichte mithilfe von :eqref:`eq_mdl-cdf_deriv` und ermitteln Sie ihre Quantilfunktion. Wie lauten ihre Quartile?
    1. Die Zufallsvariable $X$ habe eine stetige, streng monoton steigende Verteilungsfunktion $F$. Zeigen Sie, dass $F(X)$ auf $[0, 1]$ gleichverteilt ist. Dies ist die Umkehrung der Inversionsmethode.
    1. Ein probabilistisches Regressionsmodell sagt für jede Eingabe $i$ eine logistische Verteilung der Zielgröße $y_i$ vorher. Ihre Lage $\mu_i$ ist korrekt, ihr Skalenparameter ist $s$. Die vorhergesagte Verteilungsfunktion lautet also $\operatorname{sigmoid}\bigl((y - \mu_i)/s\bigr)$. Die tatsächlichen Fehler $y_i - \mu_i$ folgen der logistischen Standardverteilung. Ziehen Sie $10^4$ Fehler mit der Quantilfunktion aus (a), berechnen Sie $u_i = \operatorname{sigmoid}\bigl((y_i - \mu_i)/s\bigr)$ für $s \in \{0.5, 1, 2\}$ und zeichnen Sie die drei Histogramme der $u_i$. Welche Form zeigt an, dass die vorhergesagten Verteilungen zu schmal sind, und welche, dass sie zu breit sind? Solche Histogramme sind ein gängiges Mittel, um die Kalibrierung probabilistischer Vorhersagen zu prüfen :cite:`Gneiting.Balabdaoui.Raftery.2007`.
   :exlabel:`ex_mdl-random-variables-logistic-calibration`

<!--en:a640d4b3-->
1. **Einseitige Chebyshev-Ungleichung.** ● Die Chebyshev-Ungleichung :eqref:`eq_mdl-chebyshev` erfasst beide Verteilungsränder zugleich. Die Cantelli-Ungleichung liefert eine Schranke für einen einzelnen Rand: Für $X$ mit Mittelwert $\mu$ und Varianz $\sigma^2$ sowie für jedes $t > 0$ gilt

   $$P(X - \mu \ge t) \le \frac{\sigma^2}{\sigma^2 + t^2}.$$

    1. Wenden Sie für jedes $u \ge 0$ die Markov-Ungleichung :eqref:`eq_mdl-markov` auf $(X - \mu + u)^2$ an und zeigen Sie so, dass $P(X - \mu \ge t) \le (\sigma^2 + u^2)/(t + u)^2$ gilt. Wählen Sie $u$ so, dass sich die Cantelli-Ungleichung ergibt.
    1. Geben Sie die Schranke für $t = \alpha\sigma$ an und vergleichen Sie sie mit :eqref:`eq_mdl-chebyshev` bei $\alpha = 1$ und $\alpha = 2$.
    1. Bestimmen Sie zu gegebenen $\mu$, $\sigma$ und $t$ eine Verteilung auf zwei Punkten mit Mittelwert $\mu$ und Varianz $\sigma^2$, die die Schranke erreicht.
   :exlabel:`ex_mdl-random-variables-one-sided-chebyshev`

<!--en:31eae73b-->
1. [extended] **Mittelwerte bei schweren Verteilungsrändern.** ●●
   :citet:`Simsekli.Sagun.Gurbuzbalaban.2019` argumentieren, dass das Gradientenrauschen von SGD in tiefen Netzen schwere Verteilungsränder hat. Eine $\alpha$-stabile Verteilung, deren Varianz für $\alpha < 2$ unendlich ist, beschreibe es besser als eine Gauß-Verteilung. In diesem Projekt untersuchen Sie empirisch, wie sich ein Monte-Carlo-Mittelwert unter diesen Bedingungen verhält. Anschließend prüfen Sie die zugrunde liegende Annahme an einem kleinen Modell. Die Pareto-Dichte $p(x) = \alpha x^{-\alpha-1}$ auf $x \ge 1$ mit $\alpha > 0$ umfasst die Dichte aus Aufgabe :exref:`ex_mdl-random-variables-power-law-tail` als Spezialfall $\alpha = 1$.
    1. Verallgemeinern Sie die Teile (b) und (c) von Aufgabe :exref:`ex_mdl-random-variables-power-law-tail` auf diese Familie: Zeigen Sie, dass $\mathbb{E}[X^k] < \infty$ genau dann gilt, wenn $k < \alpha$ ist. Berechnen Sie den Mittelwert $\mu$ für $\alpha > 1$ und geben Sie eine Formel zum Ziehen von Samples mit der Inversionsmethode an.
    1. Sei $1 < \alpha < 2$, sodass der Mittelwert endlich ist, die Varianz jedoch nicht. Zeigen Sie, dass der größte von $n$ unabhängig gezogenen Werten die Schwelle $t$ mit einer Wahrscheinlichkeit nahe $n t^{-\alpha}$ überschreitet, wenn $n t^{-\alpha}$ klein ist. Bestimmen Sie die Größenordnung $t_n$, die der größte Wert typischerweise erreicht. Begründen Sie heuristisch, warum der Fehler $|\bar X_n - \mu|$ des Stichprobenmittelwerts die Größenordnung $n^{1/\alpha - 1}$ statt $n^{-1/2}$ hat. Der verallgemeinerte zentrale Grenzwertsatz, auf den sich Simsekli et al. stützen, präzisiert dieses Argument.
    1. Schätzen Sie für $\alpha \in \{1.5, 2.5, 3\}$ und $n \in \{10, 10^2, \ldots, 10^5\}$ den Median von $|\bar X_n - \mu|$ über $400$ Wiederholungen. Passen Sie eine Gerade an den Logarithmus des Medianfehlers als Funktion von $\log n$ an und bestimmen Sie ihre Steigung. Vergleichen Sie die Steigung bei $\alpha = 1.5$ mit der Rate $n^{1/\alpha - 1}$ aus (b). Vergleichen Sie die Steigungen bei $\alpha = 2.5$ und $\alpha = 3$ mit der Rate $n^{-1/2}$, die bei endlicher Varianz gilt.
    1. Bilden Sie bei jeder Wiederholung das Intervall $\bar X_n \pm 1.96\,\hat\sigma_n/\sqrt{n}$, wobei $\hat\sigma_n$ die Stichprobenstandardabweichung bezeichnet. Bei Mittelwerten aus gaußverteilten Ziehungen enthält dieses Intervall $\mu$ mit einer Wahrscheinlichkeit von etwa $95\%$. Halten Sie fest, wie oft es $\mu$ enthält. Erklären Sie die beobachtete Überdeckung bei $\alpha = 1.5$ und $n = 10^5$ unter Berücksichtigung der Tatsache, dass $\hat\sigma_n$ in jedem Lauf endlich ist. Was folgt daraus für Fehlerbalken, die aus Stichproben mit schweren Verteilungsrändern berechnet werden, etwa aus Minibatch-Gradienten?
    1. Der Hill-Schätzer :cite:`Hill.1975` schätzt den Randindex aus den $k$ größten von $m$ positiven Ziehungen $X_{(1)} \ge X_{(2)} \ge \cdots \ge X_{(m)}$ mit der Formel

       $$\hat\alpha_k = \Bigl(\frac1k \sum_{i=1}^k \log\frac{X_{(i)}}{X_{(k+1)}}\Bigr)^{-1}.$$

       Prüfen Sie den Schätzer an $10^5$ Pareto-Ziehungen für $\alpha \in \{1.5, 3\}$ und $k \in \{10^2, 10^3, 10^4\}$. Trainieren Sie anschließend die Softmax-Regression aus :numref:`sec_softmax_concise`. Berechnen Sie bei den trainierten Parametern die Differenz zwischen dem Minibatch-Gradienten und dem Gradienten über den gesamten Datensatz. Sammeln Sie die Koordinaten dieser Differenz über $1000$ Minibatches, jeweils für die Batchgrößen $32$ und $256$. Tragen Sie für beide Batchgrößen den aus den Absolutbeträgen geschätzten Wert $\hat\alpha_k$ gegen $k$ auf. Liegt der Schätzwert unter $2$? Wie stark hängt Ihre Schlussfolgerung von $k$ ab?
   :exlabel:`ex_mdl-random-variables-heavy-tail-averages`

<!--en:6f6c53fb-->
[Diskussion](https://d2l.discourse.group/t/415)

<!--en:de036e57-->
[Diskussion](https://d2l.discourse.group/t/1094)

<!--en:ca58de08-->
[Diskussion](https://d2l.discourse.group/t/1095)

<!--en:ca58de08-2-->
[Diskussion](https://d2l.discourse.group/t/1095)

<!--en:386e2003-->
Zufallsvariablen: Dichten, Verteilungsfunktionen, Momente und Schranken für große Abweichungen<br>**Wahrscheinlichkeitsdichte $p(x)$ · Verteilungsfunktion $F'(x) = p(x)$ · Inversionsmethode · Erwartungswert und Varianz · Markov und Chebyshev · schwere Verteilungsränder**

<!--en:29590a0b-->
::: {.slide title="Ein Dartpfeil landet im Abstand 0.5: Wie wahrscheinlich war das?"}

<!--en:d07bc3d4-->
[Das Paradox stetiger Zufallsvariablen]{.kicker}

<!--en:20710cfc-->
- **Stetige Ergebnisse:** Bei einem gleichverteilten Treffer auf der Einheitskreisscheibe hat der Abstand $R \in [0, 1]$ von der Mitte die Verteilungsfunktion $P(R \le r) = r^2$, den jeweiligen Flächenanteil.

<!--en:6f4a8802-->
- **Immer schmalere Intervalle um $r = 0.5$:** Sinkt die Intervallbreite $\epsilon$ in den Schritten $0.1 \to 0.01 \to 0.001$, sinkt die Wahrscheinlichkeit $P(R \in \text{bin})$ im selben Verhältnis. Der Quotient $P/\epsilon = 1$ bleibt konstant.

<!--en:1a03e21c-->
- **Wahrscheinlichkeit null $\ne$ Unmöglichkeit:** Für jeden einzelnen Punkt gilt $P(R = r) = 0$, denn ein Punkt hat die Breite null. Dennoch wächst die Wahrscheinlichkeit lokal mit endlicher Rate.

<!--en:618c581d-->
::: {.slide title="Wahrscheinlichkeitsdichte als Wahrscheinlichkeit pro Längeneinheit"}

<!--en:ca85bb87-->
[Lokale Zuwachsrate]{.kicker}

<!--en:72115f0b-->
- **Dichte beim Dartwurf:** Division von $P(r \le R < r+\epsilon) = (r+\epsilon)^2 - r^2 = 2r\epsilon + \epsilon^2$ durch $\epsilon$ ergibt im Grenzwert $p(r) = 2r$ auf $[0, 1]$.

<!--en:a76ecdd4-->
- **Geometrische Bedeutung:** Die Dichte wächst linear mit $r$, denn der Kreisumfang $2\pi r$ nimmt zum Rand hin zu.

<!--en:e0b74497-->
- **Zuwachsrate der Wahrscheinlichkeit:** Anders als eine Wahrscheinlichkeit kann eine Dichte größer als $1$ sein (z. B. $p(0.9) = 1.8$ oder $p(x) = 10$ auf $[0, 0.1]$).

<!--en:aa1f2679-->
::: {.slide title="Wahrscheinlichkeit als Fläche unter der Dichtekurve"}

<!--en:43a05416-->
[Formale Definition über das Integral]{.kicker}

<!--en:5c37f10c-->
- Die Zerlegung von $(a, b]$ in Intervalle der Breite $\epsilon$ und die Addition ihrer näherungsweisen Wahrscheinlichkeiten ergeben die Riemann-Summe $\sum_i \epsilon\,p(\epsilon i)$. Sie konvergiert gegen das bestimmte Integral:
  $$P(X \in (a, b]) = \int_a^b p(x)\,dx$$

<!--en:e4382b80-->
- **Zwei Bedingungen kennzeichnen eine gültige Dichte:**
  1. **Nichtnegativität:** $p(x) \ge 0$ für alle $x \in \mathbb{R}$.
  2. **Gesamtfläche eins:** $\int_{-\infty}^{\infty} p(x)\,dx = 1$.

<!--en:ed85b3f9-->
- **Äußerer Ring der Dartscheibe:** $\int_{0.5}^1 2r\,dr = [r^2]_{0.5}^1 = 0.75$. Ganze drei Viertel der Scheibe liegen jenseits des halben Radius.

<!--en:41f4b958-->
::: {.slide title="Simulation des Dartwurfs: Von Intervallen zu Verhältnissen"}

<!--en:7213e337-->
[Überprüfung mit Monte Carlo]{.kicker}

<!--en:f6835ce7-->
Gleichverteilte Treffer auf der Einheitskreisscheibe ohne Flächenformel erzeugen: Gleichverteilte Punkte im umschließenden Quadrat $[-1, 1]^2$ ziehen, nur Punkte mit $x^2 + y^2 \le 1$ behalten und die relativen Häufigkeiten in den Intervallen um $r = 0.5$ durch deren Breite $\epsilon$ teilen:

<!--en:edb95232-->
**Schwankungen bei endlicher Stichprobe:** Sinkt $\epsilon$ auf ein Zehntel, sinkt auch die Trefferzahl auf ein Zehntel. Nur $\sim 3{,}100$ von $3.14 \times 10^6$ Pfeilen treffen das Intervall mit $\epsilon = 0.001$. Dadurch schwankt das empirische Verhältnis um einige Prozent um $p(0.5) = 1$.

<!--en:fface902-->
::: {.slide title="Atome, Dichten und gemischte Zufallsvariablen im Deep Learning"}

<!--en:b62ed923-->
[Diskrete, stetige und gemischte Verteilungen]{.kicker}

<!--en:fc154e13-->
- **Atome und Dichten:** Ein Punkt $x$ mit $P(X = x) > 0$ heißt **Atom** (etwa bei diskreten Klassenlabels $Y \in \{1,\dots,K\}$). Stetig verteilte Gewichte $W_{ij} \sim \mathcal{N}(0, \sigma^2)$ haben keine Atome.

<!--en:0bf19e72-->
- **Gemischte Verteilungen (gleichgerichtete Gauß-Verteilung bei ReLU):** Für $X \sim \mathcal{N}(0, 1)$ bildet $Y = \max(0, X)$ alle negativen Eingaben auf ein einziges **Atom mit $P(Y = 0) = 1/2$** ab.

<!--en:e6d57d82-->
- **Positive Halbachse:** Die übrigen $50\%$ der Wahrscheinlichkeit verteilen sich stetig über $(0, \infty)$ mit der Gauß-Dichte $\frac{1}{\sqrt{2\pi}}e^{-y^2/2}$.

<!--en:e002d1e8-->
::: {.slide title="Die Verteilungsfunktion (CDF)"}

<!--en:11cf4499-->
[Einheitliche Darstellung diskreter, stetiger und gemischter Zufallsvariablen]{.kicker}

<!--en:4f4fca8b-->
- **Allgemeine Regeln:** Jede Verteilungsfunktion ist monoton wachsend und rechtsstetig ($\lim_{y \downarrow x} F(y) = F(x)$). Sie verläuft von $F(-\infty) = 0$ bis $F(+\infty) = 1$.

<!--en:c502767a-->
- **Typische Kurven:** Stetige Dichten ergeben glatte Kurven ($F(r) = r^2$), diskrete Atome Treppen. Bei gemischten ReLU-Ausgaben treten Sprünge und glatte Anstiege gemeinsam auf.

<!--en:9f22c2ab-->
::: {.slide title="Die Dichte als Ableitung der Verteilungsfunktion"}

<!--en:6233b93e-->
[Hauptsatz der Differential- und Integralrechnung]{.kicker}

<!--en:2f199925-->
- **Proposition:** Hat $X$ eine Dichte $p$, die an der Stelle $x$ stetig ist, so ist $F$ an der Stelle $x$ differenzierbar und es gilt
  $$F'(x) = p(x)$$

<!--en:8061002b-->
- **Beweisidee:** Da $F(x) = \int_{-\infty}^x p(t)\,dt$ die aufgelaufene Fläche angibt, liefert der Hauptsatz der Differential- und Integralrechnung $F'(x) = p(x)$. Einzelne Punkte tragen keine Masse, also gilt
  $$\frac{P(x \le X < x + \epsilon)}{\epsilon} = \frac{F(x + \epsilon) - F(x)}{\epsilon} \xrightarrow{\epsilon \downarrow 0} F'(x)$$

<!--en:32b91030-->
- **Dualität von Fläche und Steigung:** Die schattierte Fläche $\int_a^b p(x)\,dx$ links entspricht dem Höhenunterschied $F(b) - F(a)$ rechts.

<!--en:7aea7e7f-->
::: {.slide title="Quantile, Quartile und Inversionsmethode"}

<!--en:c835ea75-->
[Von gleichverteiltem Rauschen zu Samples aus der Zielverteilung]{.kicker}

<!--en:f4964554-->
- **Quantilfunktion $F^{-1}(u)$:** Die Verteilungsfunktion $F$ ordnet einem Wert $x$ ein Wahrscheinlichkeitsniveau $u \in (0, 1)$ zu. Ihre Umkehrfunktion $F^{-1}(u)$ liefert die Schwelle, unterhalb derer der Anteil $u$ der Wahrscheinlichkeit liegt.
  - Dartwurf ($F(r) = r^2 \implies F^{-1}(u) = \sqrt{u}$): Die Quartile sind $0.50$, **Median $\sqrt{0.5} \approx 0.707$** und $0.866$.

<!--en:086537d9-->
- **Proposition (Inversionsmethode):** Gilt $U \sim \mathrm{Unif}(0, 1)$, so hat $X = F^{-1}(U)$ die Verteilungsfunktion $F$:
  $$P\bigl(F^{-1}(U) \le x\bigr) = P\bigl(U \le F(x)\bigr) = F(x)$$

<!--en:d886314f-->
- Wo $p = F'$ groß ist, steigt $F$ steil an. Gleichmäßig verteilte Niveaus auf der senkrechten Achse ergeben dort dicht beieinanderliegende Samples auf der waagerechten Achse.

<!--en:91607942-->
::: {.slide title="Warum die Abkürzung R = U auf der Kreisscheibe scheitert"}

<!--en:da5fba20-->
[Fehler beim Sampling und ihre Überprüfung]{.kicker}

<!--en:70c6bb89-->
Vergleich von $10^6$ Ziehungen $R = \sqrt{U}$ nach der Inversionsmethode mit zweidimensionalen Dartwürfen nach der Verwerfungsmethode und mit der naiven Abkürzung $R = U$:

<!--en:d8ec70e6-->
**Geometrische Verzerrung durch $R = U$:** Mit $R = U$ sind die Abstände auf $[0, 1]$ gleichverteilt. Damit landen $50\%$ der Pfeile im Bereich $r < 0.5$ statt der korrekten $25\%$. Der mittlere Abstand wird unterschätzt ($0.500$ gegenüber $0.667$).

<!--en:a62f8cb1-->
::: {.slide title="Der Erwartungswert als Schwerpunkt einer Verteilung"}

<!--en:b9dd6c54-->
[Erstes Moment und Transformationssatz für Erwartungswerte]{.kicker}

<!--en:015272e4-->
- **Mittelwert und Median beim Dartwurf:** $\mathbb{E}[R] = \int_0^1 r(2r)\,dr = \frac{2}{3} \approx 0.667$ liegt etwas unter dem Median $\sqrt{0.5} \approx 0.707$, denn der lange linke Verteilungsrand zieht den Schwerpunkt nach innen.

<!--en:93729bed-->
- **Transformationssatz und Existenz:** Der Mittelwert jeder Funktion $g(X)$ ergibt sich direkt durch Integration gegen $p(x)$, ohne $p_{g(X)}$ herzuleiten, sofern $\int |g(x)|\,p(x)\,dx < \infty$ gilt.

<!--en:ce7fc577-->
::: {.slide title="Linearität des Erwartungswerts gilt auch ohne Unabhängigkeit"}

<!--en:f337c0f6-->
[Rechenregeln für den Mittelwert]{.kicker}

<!--en:617a9e42-->
**Proposition (Linearität des Erwartungswerts):** Für beliebige Zufallsvariablen $X, Y$ mit endlichen Mittelwerten und Konstanten $c, d \in \mathbb{R}$ gilt

<!--en:f89e5627-->
- **Beweis der Additivität:** Den Ausdruck $x_i + y_j$ mit der gemeinsamen Verteilung $p_{ij} = P(X=x_i, Y=y_j)$ gewichten und in beiden Summen getrennt die Randverteilungen bilden:
  $$\sum_{i,j} (x_i + y_j)p_{ij} = \sum_i x_i \underbrace{\sum_j p_{ij}}_{p_X(x_i)} + \sum_j y_j \underbrace{\sum_i p_{ij}}_{p_Y(y_j)}$$

<!--en:c1415628-->
- **Bedeutung für Deep Learning:**
  - $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ gilt **auch bei starker Abhängigkeit zwischen $X$ und $Y$**.
  - Der erwartete mittlere Verlust über einen Minibatch mit korrelierten Beispielen oder eine Sequenz korrelierter Token ist stets der Mittelwert der einzelnen erwarteten Verluste.

<!--en:46148c86-->
::: {.slide title="Varianz und Standardabweichung"}

<!--en:8269df49-->
[Zweites zentrales Moment]{.kicker}

<!--en:700c48d8-->
- **Affine Transformation:** Eine Verschiebung um $d$ ändert die Streuung nicht. Bei Skalierung mit $c$ geht der Faktor quadratisch in die Varianz ein ($\mathrm{Var}(cX + d) = c^2\,\mathrm{Var}(X)$ und $\sigma_{cX + d} = |c|\,\sigma_X$).

<!--en:c086bfda-->
- **Rechenbeispiele:** Für den Abstand $R$ beim Dartwurf gilt $\mathbb{E}[R^2] = 1/2 \implies \mathrm{Var}(R) = 1/2 - 4/9 = 1/18$ ($\sigma_R \approx 0.236$), für die Gleichverteilung $U(0, 1)$ gilt $\mathrm{Var}(U) = 1/3 - 1/4 = 1/12$.

<!--en:644c7896-->
::: {.slide title="Markov-Ungleichung: Schranken für große Abweichungen allein aus dem Mittelwert"}

<!--en:4fdee0af-->
[Ausgangspunkt aller Konzentrationsschranken]{.kicker}

<!--en:51a57a8b-->
- **Beweis in einer Zeile:** Da $X \ge t\,\mathbf{1}_{X \ge t}$ punktweise für jedes Ergebnis $X \ge 0$ gilt, folgt durch Bildung des Erwartungswerts $\mathbb{E}[X] \ge t\,P(X \ge t)$.

<!--en:51b99297-->
- **Faustregel und Folgerungen:** In jeder nichtnegativen Grundgesamtheit können höchstens $10\%$ der Werte größer sein als $10\times$ der Mittelwert. Die Markov-Ungleichung liefert für $(X-\mu)^2$ die Chebyshev-Ungleichung, mit $e^{\lambda X}$ die Schranken von Chernoff und Hoeffding.

<!--en:3455cb52-->
::: {.slide title="Die Chebyshev-Ungleichung ist schon für drei Atome scharf"}

<!--en:8a5d2a6b-->
[Zweiseitige Schranke aus der Varianz]{.kicker}

<!--en:996deb64-->
- **Proposition (Chebyshev-Ungleichung):** Für jede Zufallsvariable $X$ mit endlicher Varianz $\sigma_X^2 > 0$ und jedes $\alpha > 0$ gilt
  $$P\bigl(|X - \mu_X| \ge \alpha\sigma_X\bigr) \le \frac{1}{\alpha^2}$$

<!--en:425ccea2-->
- **Beweis:** Die Markov-Ungleichung auf die nichtnegative Variable $Z = (X - \mu_X)^2$ mit Mittelwert $\sigma_X^2$ und Schwelle $t = \alpha^2\sigma_X^2$ anwenden.

<!--en:9b9cf5a2-->
- **Warum sich $1/\alpha^2$ nicht verbessern lässt:** Eine Verteilung mit drei Atomen bei $a - 2, a, a + 2$ und Wahrscheinlichkeiten $q, 1-2q, q$ hat die Standardabweichung $\sigma_X = 2\sqrt{2q}$. Bei $q = \frac{1}{2\alpha^2}$ liegen die äußeren Atome genau im Abstand $\alpha\sigma_X = 2$. Damit gilt Gleichheit: $P(|X - a| \ge \alpha\sigma_X) = 2q = 1/\alpha^2$.

<!--en:e5bdbc43-->
::: {.slide title="Verteilungsränder im ungünstigsten und im typischen Fall"}

<!--en:9428249a-->
[Chebyshev-Schranke und glatte Verteilungen]{.kicker}

<!--en:14c0f8d1-->
Die Chebyshev-Ungleichung muss auch den ungünstigsten Fall abdecken, in dem die gesamte Masse der Verteilungsränder genau an den Schwellen $\pm \alpha\sigma$ liegt. Bei typischen glatten Verteilungen fallen die Randwahrscheinlichkeiten viel schneller ab:

<!--en:bb5bbe6d-->
| Abstand vom Mittelwert $|X - \mu|$ | Chebyshev-Schranke ($1/\alpha^2$) | Abstand beim Dartwurf $R$ ($p(r)=2r$) | Gauß-Verteilung $\mathcal{N}(\mu, \sigma^2)$ |

<!--en:cf42b50a-->
| $\ge 3\sigma$ ($\alpha = 3$) | $11.1\%$ | $0.0\%$ (außerhalb von $[0,1]$) | $0.27\%$ |

<!--en:91d272de-->
**Nutzen verteilungsfreier Schranken:** Für eine einzelne gaußverteilte Zufallsvariable ist $1/\alpha^2$ eine grobe Schranke. Auf einen Stichprobenmittelwert $\bar{X}_n$ mit Varianz $\sigma^2/n$ angewandt, genügt sie jedoch, um das schwache Gesetz der großen Zahlen für *jede* Verteilung mit endlicher Varianz zu beweisen.

<!--en:de1f58d9-->
::: {.slide title="Schwere Verteilungsränder und die Cauchy-Verteilung"}

<!--en:c273b1d7-->
[Wenn Momente nicht existieren]{.kicker}

<!--en:221a93f8-->
- **Unendliches zweites Moment ($\mathbb{E}[X^2] = \infty$):** Da die Dichte am Rand algebraisch wie $1/(\pi x^2)$ abfällt, gilt $\int_{-L}^L x^2 p(x)\,dx = \frac{2}{\pi}(L - \arctan L) \to \infty$.

<!--en:91dcc427-->
- **Undefinierter Mittelwert ($\mathbb{E}[|X|] = \infty$):** Wegen $\int_{-L}^L |x|\,p(x)\,dx = \frac{1}{\pi}\log(1+L^2) \to \infty$ liefern asymmetrische Integrationsgrenzen $[-L, 2L]$ einen anderen Grenzwert: $\int x\,p(x)\,dx \to \frac{\log 2}{\pi} \approx 0.221$.

<!--en:d1d0b38f-->
::: {.slide title="Wenn Mitteln versagt: Ziehungen aus der Gauß- und der Cauchy-Verteilung"}

<!--en:f53a25b4-->
[Wenn das Gesetz der großen Zahlen nicht greift]{.kicker}

<!--en:e545dc7a-->
- **Abrupte Sprünge:** Wächst $n$ bis $10^5$, nähert sich der fortlaufende Mittelwert bei der Gauß-Verteilung dem Wert $0$ mit der Rate $1/\sqrt{n}$. Bei der Cauchy-Verteilung löst eine einzige extreme Ziehung plötzliche Sprünge der Größenordnung $O(1)$ aus.

<!--en:f8cf5308-->
- **Skaleninvarianz der Cauchy-Mittelwerte:** Der Mittelwert von $n$ unabhängigen Ziehungen aus der Standard-Cauchy-Verteilung hat *genau dieselbe Verteilung* wie eine einzelne solche Ziehung ($P(|\bar{X}_{1000}| > 1) = 0.50$, identisch mit $P(|X_1| > 1) = 0.50$).

<!--en:f6f22062-->
- **Folge für Deep Learning:** Hat das Gradientenrauschen bei Minibatch-SGD schwere Verteilungsränder, die einem Potenzgesetz folgen ($P(|X| > t) \sim t^{-\alpha}$ mit $\alpha < 2$), versagt die naive Mittelung und Gradienten-Clipping ist erforderlich.

<!--en:d3b10bd0-->
::: {.slide title="Verständnisfragen zu Dichten, Quantilen und Schranken für große Abweichungen"}

<!--en:929b7671-->
[Verständnisprüfung]{.kicker}

<!--en:294413a6-->
Die eigene Intuition anhand von drei grundlegenden Rechnungen aus diesem Abschnitt prüfen:

<!--en:f8ca750e-->
1. **Sampling aus einem Potenzgesetz und Clipping:** Es gelte $p(x) = 1/x^2$ für $x \ge 1$ (sonst $0$). Wie lauten die Verteilungsfunktion $F(x)$, die Formel $X = F^{-1}(U)$ für die Inversionsmethode und der Erwartungswert $\mathbb{E}[X]$?

<!--en:95f93e65-->
2. **Varianz bei affiner Transformation und Chebyshev-Schranke:** Wenn $\mathrm{Var}(X) = 2$ gilt, wie groß ist dann $\mathrm{Var}(3X - 7)$? Wie groß kann bei einer Zufallsvariablen mit $\sigma_X > 0$ die Wahrscheinlichkeit für $|X - \mu_X| \ge 4\sigma_X$ höchstens sein?

<!--en:1e52ac85-->
3. **He-Initialisierung und gleichgerichtete Gauß-Verteilung:** Für die ReLU-Ausgabe $Y = \max(0, X)$ mit $X \sim \mathcal{N}(0, 1)$ mithilfe der Symmetrie ($X^2 = Y^2 + \min(0, X)^2$) den Wert von $\mathbb{E}[Y^2]$ bestimmen. Warum multipliziert die He-Initialisierung die Varianz der Gewichte mit $2$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:caf5621d-->
1. $F(x) = \int_1^x t^{-2}\,dt = 1 - 1/x$ für $x \ge 1$. Auflösen von $1 - 1/x = U$ liefert die Ziehung **$X = 1/(1 - U)$** (Median $= 2$). Für den Mittelwert gilt $\int_1^\infty x(1/x^2)\,dx = [\log x]_1^\infty = \boldsymbol{\infty}$. Bei Begrenzung auf höchstens $c$ ergibt sich der endliche Mittelwert $1 + \log c$.

<!--en:656b584f-->
2. $\mathrm{Var}(3X - 7) = 3^2 \mathrm{Var}(X) = \mathbf{18}$ (die Verschiebung entfällt, der Skalierungsfaktor geht quadratisch ein). Nach der Chebyshev-Ungleichung gilt $P(|X - \mu_X| \ge 4\sigma_X) \le 1/4^2 = \mathbf{1/16 = 6.25\%}$.

<!--en:2f250402-->
3. Es gilt $\mathbb{E}[X^2] = 1$, und $Y^2$ hat dieselbe Verteilung wie $\min(0, X)^2$. Wegen $Y \cdot \min(0, X) = 0$ folgt $2\mathbb{E}[Y^2] = 1 \implies \boldsymbol{\mathbb{E}[Y^2] = 1/2}$. Die He-Initialisierung multipliziert die Varianz der Gewichte mit $2$, damit das zweite Moment der Aktivierungen von Schicht zu Schicht wieder eins beträgt.

<!--en:a0938641-->
::: {.slide title="Zusammenfassung: Zufallsvariablen, Momente und Verteilungsränder"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:b4bb6742-->
| Begriff | Mathematische Definition oder Regel | Dartwurf ($R$ auf der Einheitskreisscheibe) | Bedeutung für Deep Learning |

<!--en:4714054b-->
| **Wahrscheinlichkeitsdichte $p(x)$** | $P(a < X \le b) = \int_a^b p(x)\,dx$, $p \ge 0$, $\int p = 1$ | $p(r) = 2r$ auf $[0, 1]$ ($P(R = r) = 0$) | Stetige Repräsentationen, Gewichte und Rauschen |

<!--en:591d6529-->
| **Verteilungsfunktion und Ableitung** | $F(x) = P(X \le x) = \int_{-\infty}^x p(t)\,dt$, $F'(x) = p(x)$ | $F(r) = r^2$, $P(R > 0.5) = 1 - 0.5^2 = 0.75$ | Einheitliche Beschreibung stetiger, diskreter und gemischter Verteilungen (ReLU) |

<!--en:a8bfa667-->
| **Inversionsmethode** | $X = F^{-1}(U)$ hat für $U \sim \mathrm{Unif}(0, 1)$ die Verteilungsfunktion $F$ | $R = \sqrt{U}$ (Median $F^{-1}(0.5) \approx 0.707$) | Sampling aus beliebigen eindimensionalen Verteilungen und Kalibrierung |

<!--en:511f91d9-->
| **Erwartungswert und Linearität** | $\mathbb{E}[g(X)] = \int g(x)p(x)\,dx$; $\mathbb{E}[X+Y] = \mathbb{E}[X]+\mathbb{E}[Y]$ | $\mathbb{E}[R] = \int_0^1 2r^2\,dr = 2/3 \approx 0.667$ | Erwartungstreue Schätzung durch empirisches Risiko und SGD-Gradienten |

<!--en:579e8d8c-->
| **Varianz und affine Skalierung** | $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mu^2$; $\mathrm{Var}(cX+d) = c^2\mathrm{Var}(X)$ | $\mathbb{E}[R^2] = 1/2 \implies \mathrm{Var}(R) = 1/18$ | Steuerung der Größenordnung von Aktivierungen und Schichtnormalisierung |

<!--en:2a4eacc5-->
| **Markov- und Chebyshev-Schranke** | $P(X \ge t) \le \mathbb{E}[X]/t$; $P(\|X-\mu\| \ge \alpha\sigma) \le 1/\alpha^2$ | Exakte Wahrscheinlichkeit ab $2\sigma$ im Vergleich zur Schranke: $3.8\% \le 25\%$ | Verteilungsfreie Schranken für große Abweichungen und Konvergenzgarantien |

<!--en:82b9d9ab-->
| **Schwere Verteilungsränder (Cauchy)** | Algebraischer Abfall $p(x) \sim x^{-(\alpha+1)}$ ($\mathbb{E}[\|X\|^k] = \infty, k \ge \alpha$) | Cauchy ($\alpha=1$): weder Mittelwert noch Varianz, $\bar{X}_n \stackrel{d}{=} X_1$ | Gradienten-Clipping bei SGD-Rauschen mit schweren Verteilungsrändern |
