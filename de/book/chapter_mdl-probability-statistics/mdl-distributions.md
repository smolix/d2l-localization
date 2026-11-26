<!--en:347eb5bd-->
# Verteilungen

<!--en:9c57a222-->
In :numref:`sec_mdl-random_variables` und :numref:`sec_mdl-joint-distributions` haben wir eine gemeinsame Sprache für Zufallsgrößen entwickelt. Zu ihr gehören Wahrscheinlichkeitsfunktionen (engl. probability mass function, PMF), Dichten, Erwartungswerte und Varianzen. Jedes Vorhersagemodell im Deep Learning verwendet diese Sprache, ob ausdrücklich oder implizit. Ein binärer Klassifikator gibt eine einzige Zahl aus, die Wahrscheinlichkeit dafür, dass das Label 1 ist. Ein Sprachmodell gibt für jedes Token seines Vokabulars eine Wahrscheinlichkeit aus. Zusammen müssen diese Wahrscheinlichkeiten eins ergeben. Ein Regressionsnetz sagt eine reelle Zahl vorher. Beim Training minimiert es den quadratischen Fehler. In jedem dieser Fälle wählt das Modell implizit eine Wahrscheinlichkeitsverteilung für seine Labels. Diese Wahl legt fest, welche Verlustfunktion (engl. loss function) es minimiert. Um welche Verteilungen handelt es sich, woher kommen sie und wie berechnen wir ihre Mittelwerte und Varianzen?

<!--en:24211856-->
Erstaunlicherweise genügt ein einziger Münzwurf, um fast alle diese Fragen zu beantworten. Angenommen, bei 13 Würfen fällt 9-mal Kopf. Wie ungewöhnlich ist dieses Ergebnis, wenn die Münze fair ist? Dazu brauchen wir die Verteilung der Anzahl der Kopfwürfe bei wiederholtem Werfen. Genau dieselbe Verteilung beschreibt, wie viele Testbeispiele ein Klassifikator richtig einordnet. Wie wir gleich sehen, beträgt die Wahrscheinlichkeit für genau 9 Kopfwürfe etwa 9%.

<!--en:c4e90364-->
Aus diesem einen Münzwurf lässt sich eine ganze Reihe grundlegender Verteilungen herleiten. Zählen wir bei einer festen Zahl von Würfen, wie oft Kopf fällt, erhalten wir die Binomialverteilung. Viele Würfe, bei denen Kopf jeweils nur selten fällt, führen zur Poisson-Verteilung. Die Wartezeit bis zum nächsten seltenen Ereignis liefert die Exponentialverteilung. Addieren wir die Ergebnisse vieler gewöhnlicher Münzwürfe und skalieren die Summe neu, erhalten wir im Grenzwert die Gauß-Verteilung. Da jede Verteilung aus einfacheren Bausteinen besteht, können wir ihren Mittelwert und ihre Varianz mit einer kurzen Rechnung direkt aus der Konstruktion ablesen.

<!--en:2ce2354e-->
Die Wahl der richtigen Verteilung hat praktische Folgen. Ist etwa die Varianz eines Modells für Zähldaten zu klein, fallen alle daraus berechneten Fehlerbalken irreführend schmal aus. Drei Ergebnisse haben weit über diesen Abschnitt hinaus Bedeutung: der Poisson-Grenzwert mit den zugehörigen Wartezeiten, der zentrale Grenzwertsatz (engl. central limit theorem) und die Formel für die bedingte Verteilung gaußscher Zufallsvektoren. Die diskrete Gleichverteilung, die Gamma- und die Laplace-Verteilung können Sie beim ersten Lesen überfliegen.

<!--en:da4f12a7-->
:numref:`fig_mdl-prob-family-tree` zeigt, wie die Verteilungen dieses Abschnitts auseinander hervorgehen.

<!--en:3f030837-->
![Alle Verteilungen in diesem Abschnitt gehen auf einen einzigen Münzwurf zurück. Durchgezogene Pfeile stehen für die Konstruktion einer neuen Verteilung über eine Summe, eine Wartezeit, ein zufälliges Vorzeichen, einen Vektor von Koordinaten oder gleiche Wahrscheinlichkeiten. Gestrichelte Pfeile bezeichnen einen Grenzübergang bei wachsender Zahl von Versuchen oder Punkten. Poisson- und Exponentialverteilung beschreiben dieselbe Ereignisfolge: Wir zählen die Ereignisse in Zeitfenstern oder messen die Zeit zwischen ihnen.](../img/mdl-prob-family-tree.svg)

<!--en:3ff3f687-->
## Diskrete Verteilungen

<!--en:c3ea68ea-->
Eine diskrete Zufallsvariable nimmt Werte in einer abzählbaren Menge wie $\{0,1\}$ oder $\{0,1,2,\dots\}$ an. Ihre Wahrscheinlichkeitsfunktion $P(X=k)\ge0$ mit $\sum_k P(X=k)=1$ legt die Verteilung fest. Ihr Mittelwert beträgt $\mathbb{E}[X]=\sum_k k\,P(X=k)$, ihre Varianz $\textrm{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2$. :numref:`fig_mdl-prob-discrete-pmfs` zeigt die diskreten Verteilungen dieses Teils anhand der Beispiele, die uns im Folgenden begleiten.

<!--en:9b48a888-->
![Vier Wahrscheinlichkeitsfunktionen zu unseren Beispielen. Bernoulli$(9/13)$: ein einzelner Münzwurf mit der beobachteten relativen Häufigkeit von Kopf als Erfolgswahrscheinlichkeit. Binomial$(13,\frac{1}{2})$: Anzahl der Kopfwürfe bei 13 fairen Würfen. Der orangefarbene Stab markiert $P(X=9)=715/8192\approx0.087$. Poisson$(4)$: Serveranfragen pro Minute bei einer mittleren Rate von vier. Die Logits $(2, 1, 0.1, -1)$ ergeben eine Kategorialverteilung über vier Klassen, zum Vergleich dient die Gleichverteilung mit Wahrscheinlichkeit $1/4$. Binomial- und Poisson-Verteilung erreichen ihr Maximum nahe ihrem Mittelwert ($6.5$ bzw. $4$) und verwenden dieselbe vertikale Skala. Die beiden Diagramme mit weniger möglichen Ergebnissen teilen sich eine andere Skala.](../img/mdl-prob-discrete-pmfs.svg)

<!--en:0c1a3cab-->
### Bernoulli-Verteilung

<!--en:ed5d7b19-->
Die einfachste Zufallsgröße hat nur zwei mögliche Ergebnisse. Eine Münze zeigt Kopf oder Zahl, jemand klickt auf eine Anzeige oder ignoriert sie, ein Klassifikator ordnet ein Beispiel richtig oder falsch ein. Eine einzige Zahl beschreibt diesen Versuch vollständig: die Wahrscheinlichkeit $p$ des Ergebnisses, das wir mit $1$ bezeichnen.

<!--en:c658080b-->
Eine *Bernoulli-Zufallsvariable* $X\sim\mathrm{Bernoulli}(p)$ nimmt den Wert $1$ mit Wahrscheinlichkeit $p$ und den Wert $0$ mit Wahrscheinlichkeit $1-p$ an. Dabei gilt $0\le p\le1$. Beide Fälle lassen sich in einer Formel zusammenfassen:

<!--en:7ae0cbce-->
Der Exponent wirkt wie ein Schalter. Für $x=1$ ergibt die Formel $p$, für $x=0$ ergibt sie $1-p$.

<!--en:26c8eb14-->
Die Momente lassen sich wie in :numref:`sec_prob` leicht berechnen. Der Mittelwert ist $\mathbb{E}[X]=1\cdot p + 0\cdot(1-p)=p$. Da $X$ nur die Werte $0$ und $1$ annimmt, gilt $X^2=X$ und damit auch $\mathbb{E}[X^2]=p$. Die Varianz beträgt folglich $\textrm{Var}(X)=p-p^2=p(1-p)$, das Produkt der beiden Ergebniswahrscheinlichkeiten. Sie verschwindet, wenn das Ergebnis feststeht ($p=0$ oder $p=1$), und erreicht ihr Maximum von $\tfrac14$ bei einer fairen Münze ($p=\tfrac12$).

<!--en:ea17f015-->
Im Deep Learning gibt ein binärer Klassifikator einen Schätzwert $\hat p$ für die Wahrscheinlichkeit aus, dass das Label $y$ gleich $1$ ist. Nach :eqref:`eq_mdl-bernoulli_pmf` weist er dem beobachteten Label die negative Log-Wahrscheinlichkeit $-\log P(y)=-y\log\hat p-(1-y)\log(1-\hat p)$ zu. Dieser Verlust ist die *binäre Kreuzentropie*. Trainieren wir einen Klassifikator mit binärer Kreuzentropie, passen wir also für jedes Label eine Bernoulli-Verteilung an.

<!--en:4ae3db6c-->
Im Folgenden unterscheiden wir zwei Münzen. Unsere Ausgangsfrage betrifft eine faire Münze mit $p=\tfrac12$. Bei der anderen Münze entspricht die Wahrscheinlichkeit für Kopf der beobachteten relativen Häufigkeit, also $p=9/13\approx0.69$. Ein einzelner Wurf hat dann die Varianz $\tfrac{9}{13}\cdot\tfrac{4}{13}\approx0.21$.

<!--en:1a7a39c2-->
### Binomialverteilung

<!--en:203c61d6-->
Kehren wir zur Ausgangsfrage zurück. Wie ist die Anzahl der Kopfwürfe bei 13 fairen Würfen verteilt, und wie wahrscheinlich sind genau 9 Kopfwürfe?

<!--en:328d2b8d-->
Beginnen wir mit dem kleinsten nichttrivialen Fall, zwei Würfen. Mit T für Zahl und H für Kopf lauten die vier möglichen Ergebnisse TT, HT, TH und HH. Damit gilt

<!--en:deeb9016-->
Der Faktor $2$ in $P(X=1)$ zählt die beiden möglichen Reihenfolgen für einmal Kopf und einmal Zahl. Nach der binomischen Formel summieren sich die drei Wahrscheinlichkeiten zu $(p+1-p)^2=1$.

<!--en:6f945cdb-->
Allgemein ist die Anzahl der Kopfwürfe bei $n$ Würfen die Summe von $n$ Indikatorvariablen, einer pro Wurf. Sind die Würfe $X_1,\dots,X_n$ unabhängig und folgt jeder der Verteilung $\mathrm{Bernoulli}(p)$ mit demselben $p$, dann folgt ihre Summe der *Binomialverteilung*:

<!--en:cf7207ee-->
Es gibt $\binom nk$ Möglichkeiten, die $k$ unter den $n$ Würfen auszuwählen, bei denen Kopf fällt. Wegen der Unabhängigkeit hat jede konkrete Reihenfolge die Wahrscheinlichkeit $p^k(1-p)^{n-k}$. Multiplizieren wir die Anzahl der Reihenfolgen mit der Wahrscheinlichkeit einer einzelnen Reihenfolge, erhalten wir die Wahrscheinlichkeitsfunktion der Binomialverteilung:

<!--en:f6b93e74-->
Für $n=2$ ergeben sich wieder die drei Wahrscheinlichkeiten von oben. Bei unserer fairen Münze können 9 Kopfwürfe unter 13 Würfen in $\binom{13}{9}=715$ verschiedenen Reihenfolgen auftreten, jeweils mit Wahrscheinlichkeit $2^{-13}=1/8192$. Folglich gilt

<!--en:4b907eb7-->
Die Wahrscheinlichkeit für 9 *oder mehr* Kopfwürfe beträgt $(715+286+78+13+1)/8192$, also $1093/8192\approx0.133$. Ob eine Überschreitungswahrscheinlichkeit von $13.3\%$ ausreicht, um daran zu zweifeln, dass die Münze fair ist, ist eine statistische Frage. Wir stellen sie vorerst zurück.

<!--en:c6670502-->
**Mittelwert und Varianz.** Es wäre mühsam, $\sum_k k\,P(X=k)$ direkt aus der Wahrscheinlichkeitsfunktion zu berechnen. Mit der Summendarstellung :eqref:`eq_mdl-binomial_sum` erhalten wir beide Momente in jeweils einer Zeile. Der Erwartungswert ist für beliebige Zufallsvariablen linear, unabhängig davon, ob sie voneinander abhängen. Daher gilt

<!--en:fac42b00-->
Für die Varianz nutzen wir ein Ergebnis aus :numref:`sec_mdl-joint-distributions`. Unabhängige Variablen haben Kovarianz null. Die Varianz einer Summe *unabhängiger* Variablen ist daher die Summe ihrer Varianzen:

<!--en:13817f08-->
Für $n$ unabhängige Würfe erhalten wir also das $n$-Fache des Mittelwerts und das $n$-Fache der Varianz eines einzelnen Wurfs. Bei 13 fairen Würfen beträgt der Mittelwert $6.5$, die Varianz $3.25$ und die Standardabweichung $\sqrt{3.25}\approx1.80$. Mit 9 Kopfwürfen liegen wir $(9-6.5)/1.80\approx1.4$ Standardabweichungen über dem Mittelwert.

<!--en:ecfa16d2-->
Ein leicht zu übersehender Unterschied liegt in den Voraussetzungen der beiden Formeln. Der Mittelwert $np$ gilt auch für abhängige Würfe. Hat Wurf $i$ eine eigene Erfolgswahrscheinlichkeit $p_i$, ergibt sich der Mittelwert $\sum_i p_i$. Die Varianzformel dagegen setzt Unabhängigkeit voraus. Bestünden unsere 13 „Würfe“ aus 13 Kopien desselben fairen Wurfs, könnte $X$ nur die Werte $0$ oder $13$ annehmen. Die Varianz wäre $13^2\cdot\tfrac14=42.25$, also dreizehnmal so groß wie $np(1-p)$.

<!--en:69f88e04-->
**Quotient aufeinanderfolgender Wahrscheinlichkeiten.** Bei der direkten Auswertung von :eqref:`eq_mdl-binomial_pmf` treten Fakultäten auf, die für große $n$ schnell zu einem Überlauf führen. Wir können die gesamte Wahrscheinlichkeitsfunktion ohne eine einzige Fakultät berechnen. Im Quotienten aufeinanderfolgender Wahrscheinlichkeiten kürzen sich alle Fakultäten heraus:

<!--en:c71fa8df-->
Der Quotient lässt sich kombinatorisch deuten. Gehen wir von $k$ Kopfwürfen aus, können wir einen beliebigen der $n-k$ übrigen Würfe von Zahl auf Kopf drehen. Jede so entstehende Menge von $k+1$ Kopfwürfen erreichen wir auf $k+1$ Arten. Der Faktor $p/(1-p)$ ersetzt die Wahrscheinlichkeit für Zahl durch die für Kopf. Beginnen wir bei $P(X=0)=(1-p)^n$ und multiplizieren wiederholt mit dem jeweiligen Quotienten, erhalten wir die gesamte Wahrscheinlichkeitsfunktion. Der Quotient ist größer als $1$, solange $k+1<(n+1)p$ gilt, und fällt nach Überschreiten der Schwelle unter $1$. Die Wahrscheinlichkeitsfunktion steigt also bis zu einem Maximum nahe dem Mittelwert an und fällt dann wieder ab. Am Übergang können zwei benachbarte Werte gleich groß sein.

<!--en:f6012660-->
Für die Rekursion genügen drei Zeilen Python. Wir fassen sie in einer wiederverwendbaren Funktion zusammen, um später sowohl den Poisson-Grenzwert als auch die gaußsche Glockenkurve mit ihr zu vergleichen.

<!--en:96648dab-->
Wie erwartet liefert die Rekursion über die Quotienten wieder $715/8192\approx0.0873$ und die Überschreitungswahrscheinlichkeit $0.1334$.

<!--en:151ade6a-->
### Poisson-Verteilung

<!--en:86ed125c-->
Viele Zählgrößen in der Praxis haben keine natürliche Obergrenze $n$. Angenommen, bei einem Webserver gehen durchschnittlich $\lambda=4$ Anfragen pro Minute ein, ohne feste Obergrenze. Wie wahrscheinlich ist eine Minute ganz ohne Anfrage oder ein plötzlicher Ansturm mit 10 oder mehr Anfragen?

<!--en:aea28694-->
Um den Zusammenhang mit den Münzwürfen zu sehen, teilen wir die Minute gedanklich in $n$ kurze Intervalle. Unser Modell trifft zwei vereinfachende Annahmen. Jedes Intervall ist so kurz, dass nie zwei Anfragen hineinfallen. Außerdem bilden die Intervalle unabhängige Versuche mit derselben Ankunftswahrscheinlichkeit $4/n$. Dann folgt die Gesamtzahl pro Minute der Verteilung $\mathrm{Binomial}(n,4/n)$. Ihr Mittelwert beträgt $n\cdot\tfrac4n=4$ für jedes $n$, und ihre Varianz $4(1-\tfrac4n)$ nähert sich stetig dem Wert $4$. Die Wahrscheinlichkeit für eine Minute ohne Anfrage ist $(1-\tfrac4n)^n$. Sie beträgt $0.0115$ bei $n=20$, $0.0169$ bei $n=100$ und $0.0182$ bei $n=1000$ und konvergiert gegen $e^{-4}\approx0.0183$. Je feiner die Unterteilung, desto weniger ändern sich die Wahrscheinlichkeiten. Das deutet auf eine einfache Grenzverteilung hin.

<!--en:74adbe0e-->
**Herleitung des Grenzwerts.** Betrachten wir $\mathrm{Binomial}(n,\lambda/n)$ bei einer festen Anzahl $k$ für $n\to\infty$. Formen wir die Wahrscheinlichkeitsfunktion der Binomialverteilung um, erhalten wir vier Faktoren. Drei davon haben elementare Grenzwerte:

<!--en:2231b7a7-->
Der erste Faktor ist ein Produkt aus $k$ Brüchen, die jeweils gegen $1$ streben. Der dritte liefert den klassischen Grenzwert $(1-\lambda/n)^n\to e^{-\lambda}$, der die Exponentialfunktion definiert. Der letzte strebt gegen $1$, weil $k$ fest bleibt. Dasselbe Argument gilt für jede Folge von Erfolgswahrscheinlichkeiten $p_n$ mit $np_n\to\lambda$, denn auch dann gilt $(1-p_n)^n\to e^{-\lambda}$. Dieses Ergebnis heißt *Gesetz der seltenen Ereignisse*. Die verbleibenden Faktoren ergeben die Wahrscheinlichkeitsfunktion der *Poisson-Verteilung* mit Rate $\lambda>0$:

<!--en:c042841d-->
Wir schreiben $X\sim\mathrm{Poisson}(\lambda)$. Die Poisson-Verteilung ergibt sich, wenn wir bei vielen unabhängigen Gelegenheiten jeweils ein seltenes Ereignis zählen und der Erwartungswert der Gesamtzahl $\lambda$ fest bleibt. Aus der Taylorreihe $\sum_{k\ge0}\lambda^k/k!=e^{\lambda}$ folgt direkt, dass sich die Wahrscheinlichkeiten zu eins summieren.

<!--en:6e361b0a-->
**Mittelwert und Varianz.** Beide Momente lassen sich direkt aus der Wahrscheinlichkeitsfunktion ablesen. In der Summe für $\mathbb{E}[X]$ verschwindet der Term für $k=0$. Der Faktor $k$ kürzt sich gegen den entsprechenden Faktor in $k!$. Verschieben wir den Summationsindex auf $j=k-1$, bleibt die Gesamtwahrscheinlichkeit einer poissonverteilten Zufallsvariablen:

<!--en:41a10812-->
Dasselbe Vorgehen für $k(k-1)$ ergibt $\mathbb{E}[X(X-1)]=\lambda^2$. Daraus folgt die Varianz:

<!--en:fb21a712-->
Ein Kennzeichen der Poisson-Verteilung ist die Gleichheit von Mittelwert und Varianz. Für unseren Webserver gilt $P(X=0)=e^{-4}\approx0.018$, $P(X=4)=\tfrac{4^4}{4!}e^{-4}\approx0.195$ und $P(X\ge10)\approx0.008$. Etwa einmal pro Stunde bleibt eine Minute ohne Anfrage, und etwa alle zwei Stunden treffen zehn oder mehr Anfragen in einer Minute ein. An der Gleichheit von Mittelwert und Varianz lässt sich das Modell auch prüfen. Übersteigt die Varianz der über viele Minuten beobachteten Anzahlen deren Mittelwert um ein Mehrfaches, ist mindestens eine Modellannahme verletzt. Vielleicht schwankt das Anfrageaufkommen im Laufe der Stunde, oder Anfragen lösen Kaskaden weiterer Anfragen aus. Die Fehlerbalken des Poisson-Modells fallen dann gefährlich schmal aus. In Aufgabe :exref:`ex_mdl-distributions-gamma-poisson-mixture` untersuchen wir die übliche Abhilfe: Wir modellieren die Poisson-Rate selbst als Zufallsvariable.

<!--en:4798a581-->
**Der Poisson-Prozess.** Übertragen wir die Unterteilung mit Rate $\lambda$ pro Zeiteinheit auf die gesamte Zeitachse, erhalten wir ein Modell für Ereignisse in kontinuierlicher Zeit. Jedes Fenster der Länge $t$ umfasst $nt$ Teilintervalle. Seine Ereignisanzahl folgt also der Verteilung $\mathrm{Binomial}(nt,\lambda/n)$, die gegen $\mathrm{Poisson}(\lambda t)$ konvergiert. Zeitfenster ohne Überlappung bestehen aus disjunkten Teilintervallen. Ihre Ereignisanzahlen sind deshalb unabhängig. Diese Ereignisfolge in kontinuierlicher Zeit heißt *Poisson-Prozess* mit Rate $\lambda$. Im nächsten Teil sehen wir, dass die Wartezeiten zwischen seinen Ereignissen der Exponentialverteilung folgen.

<!--en:c30b6d84-->
Wie bei der Binomialverteilung hat der Quotient aufeinanderfolgender Wahrscheinlichkeiten der Poisson-Verteilung eine einfache Form, $P(X=k+1)/P(X=k)=\lambda/(k+1)$. Berechnen wir damit die Wahrscheinlichkeiten für unseren Server und prüfen wir die Momente anhand von $100{,}000$ simulierten Minuten.

<!--en:17082123-->
Bei den simulierten Anzahlen liegen Mittelwert und Varianz beide nahe bei $4$.

<!--en:a2115e8a-->
**Von der Binomial- zur Poisson-Verteilung.** Wie schnell konvergiert $\mathrm{Binomial}(n,4/n)$ gegen $\mathrm{Poisson}(4)$? Zeichnen wir die beiden Wahrscheinlichkeitsfunktionen übereinander und messen wir ihre Abweichung mit dem *Totalvariationsabstand* (engl. total variation distance) $\tfrac12\sum_k|P_{\mathrm{Bin}}(k)-P_{\mathrm{Poi}}(k)|$. Er entspricht der größten Differenz zwischen den Wahrscheinlichkeiten, die beide Verteilungen demselben Ereignis zuweisen.

<!--en:05ec967c-->
Bei $n=5$ sind im Binomialmodell höchstens fünf Anfragen möglich und das Maximum liegt deutlich zu hoch. Bei $n=100$ lassen sich die beiden Wahrscheinlichkeitsfunktionen mit bloßem Auge nicht mehr unterscheiden. Der Totalvariationsabstand sinkt von $0.056$ bei $n=20$ auf $0.010$ bei $n=100$ und $0.001$ bei $n=1000$. Er nimmt also etwa proportional zu $1/n$ ab.

<!--en:eac65fb9-->
### Kategorial- und Multinomialverteilung

<!--en:9d7d123a-->
Ein Klassifikator mit $K$ Klassen oder ein Sprachmodell, das das nächste Token aus einem Vokabular vorhersagt, wählt unter mehr als zwei Ergebnissen. Die Kategorialverteilung entspricht einer Münze mit $K$ Seiten.

<!--en:b7e518eb-->
Eine *kategoriale* Zufallsvariable nimmt Werte in $\{1,\dots,K\}$ an. Der Vektor $\mathbf p=(p_1,\dots,p_K)$ gibt die zugehörigen Wahrscheinlichkeiten an, wobei $p_k\ge0$ und $\sum_k p_k=1$ gelten:

<!--en:d36501a7-->
Für $K=2$ erhalten wir die Bernoulli-Verteilung. In neuronalen Netzen sagen wir Wahrscheinlichkeiten selten direkt vorher. Das Netz gibt reelle Scores $z_1,\dots,z_K$ ohne Nebenbedingungen aus, die sogenannten *Logits*. Mit der *Softmax-Funktion* wandelt es sie in eine Kategorialverteilung um:

<!--en:4eca3589-->
Durch das Exponenzieren wird jeder Eintrag positiv. Nach der Division durch die Summe addieren sich die Einträge zu eins. Beispielsweise ergeben die Logits $\mathbf z=(2,1,0.1,-1)$ die Wahrscheinlichkeiten $\mathbf p\approx(0.638,0.235,0.095,0.032)$. Schreiben wir die wahre Klasse als One-Hot-Vektor $\mathbf y$, erhält das beobachtete Label die negative Log-Wahrscheinlichkeit $-\log p_y=-\sum_k y_k\log p_k$. Das ist der Verlust der *Softmax-Kreuzentropie* aus :numref:`subsec_softmax-regression-loss-func`. So wie die binäre Kreuzentropie für jedes Label eine Bernoulli-Verteilung anpasst, passt die Softmax-Kreuzentropie eine Kategorialverteilung an.

<!--en:ce7e4082-->
**Die diskrete Gleichverteilung.** Die einfachste Kategorialverteilung weist jedem Ergebnis dieselbe Wahrscheinlichkeit zu. Auf $\{1,\dots,n\}$ mit $p_k=1/n$ ist dies die *diskrete Gleichverteilung* $\mathrm{Unif}\{1,\dots,n\}$. Beispiele sind ein fairer Würfel oder ein gleichverteilt gezogener Index. Mittelwert und Varianz folgen aus den bekannten Summen $\sum_{i=1}^n i=\tfrac{n(n+1)}2$ und $\sum_{i=1}^n i^2=\tfrac{n(n+1)(2n+1)}6$:

<!--en:bbd3a0a9-->
Der Mittelwert liegt genau in der Mitte. Die Varianz wächst wie $n^2$, weil die Ergebnisse einen Bereich der Länge $n-1$ abdecken. Für einen fairen sechsseitigen Würfel beträgt der Mittelwert $3.5$ und die Varianz $35/12\approx2.92$.

<!--en:cbd92dc5-->
**Die Multinomialverteilung.** Wiederholen wir einen Bernoulli-Versuch $n$-mal unabhängig, erhalten wir die Binomialverteilung. Ziehen wir entsprechend $n$-mal unabhängig aus einer Kategorialverteilung und zählen, wie oft jede Klasse auftritt, erhalten wir die *Multinomialverteilung*. Für den Vektor der Anzahlen $\mathbf x=(x_1,\dots,x_K)$ mit der Nebenbedingung $\sum_k x_k=n$ gilt die Wahrscheinlichkeitsfunktion

<!--en:c66d664d-->
Wie bei der Binomialverteilung zählt der vorangestellte Multinomialkoeffizient die Reihenfolgen von $n$ Ziehungen, die zu den Anzahlen $\mathbf x$ führen. Das Produkt gibt die Wahrscheinlichkeit einer einzelnen Reihenfolge an. Für $K=2$ vereinfacht sich der Koeffizient zu $\binom{n}{x_1}$ und wir erhalten wieder die Binomialverteilung ($13!/(9!\,4!)=715$). Betrachten wir eine Klasse $k$ für sich. Jede Ziehung fällt entweder in die Klasse $k$ oder außerhalb. Daher gilt $x_k\sim\mathrm{Binomial}(n,p_k)$ mit $\mathbb{E}[x_k]=np_k$ und $\textrm{Var}(x_k)=np_k(1-p_k)$. Über alle Klassen hinweg hat der Vektor der Anzahlen den Mittelwert $n\mathbf p$ und die Kovarianzmatrix $n\bigl(\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top\bigr)$, wie wir in Aufgabe :exref:`ex_mdl-distributions-multinomial-covariance` herleiten. Alle Einträge außerhalb der Diagonalen sind negativ. Da die Gesamtzahl der Ziehungen $n$ feststeht, gehen zusätzliche Treffer in einer Klasse zulasten der anderen. Für $n=1$ ist der Vektor der Anzahlen die One-Hot-Kodierung einer einzelnen kategorialen Ziehung.

<!--en:f155a755-->
Berechnen wir die Softmax-Wahrscheinlichkeiten für unsere Beispiel-Logits. Ziehen wir dann $10{,}000$-mal aus der resultierenden Kategorialverteilung, um die relativen Häufigkeiten zu überprüfen.

<!--en:e54bf8b7-->
Die relativen Häufigkeiten stimmen bis auf etwa einen Prozentpunkt mit $\mathbf p$ überein.

<!--en:3369f693-->
## Stetige Verteilungen

<!--en:6b46beb6-->
Eine stetige Zufallsvariable beschreiben wir durch eine Wahrscheinlichkeitsdichte $p(x)\ge0$ mit $\int p(x)\,dx=1$. Wahrscheinlichkeiten entsprechen Flächen unter der Dichtekurve. Erwartungswerte und Varianzen berechnen wir wie in :numref:`sec_mdl-random_variables` durch Integration. Wir beginnen mit der Exponentialverteilung, weil sie sich unmittelbar aus dem Poisson-Prozess ergibt.

<!--en:339bd65f-->
### Exponentialverteilung

<!--en:bddaf459-->
Kehren wir zu unserem Webserver zurück. Die Poisson-Verteilung gibt an, wie viele Anfragen in einer Minute eintreffen. Wer eine Zeitüberschreitung für denselben Datenverkehr konfigurieren möchte, stellt eine eng verwandte Frage: Wie lange dauert es bis zur nächsten Anfrage?

<!--en:141a480d-->
Erstaunlicherweise haben beide Fragen dieselbe Antwort. Länger als eine Minute auf die nächste Anfrage zu warten, ist genau dasselbe Ereignis wie eine Minute ohne Anfrage. Also gilt $P(\text{wait}>1)=P(N=0)=e^{-4}\approx0.018$, die bereits berechnete Poisson-Wahrscheinlichkeit. :numref:`fig_mdl-prob-poisson-timeline` zeigt, wie beide Verteilungen dieselbe Ereignisfolge beschreiben.

<!--en:d04011d2-->
![Zwei Sichtweisen auf dieselbe Folge von durchschnittlich vier Anfragen pro Minute. (a) Ankunftszeiten über drei Minuten: Die Minutenfenster enthalten $N = 4$, $2$ und $5$ Anfragen. Die ersten drei Wartezeiten sind $T_1$, $T_2$ und $T_3$. (b) Die Anzahl der Anfragen pro Minute folgt der Verteilung Poisson$(4)$. Eine Minute ohne Anfrage hat also die Wahrscheinlichkeit $e^{-4}\approx0.018$. (c) Die Wahrscheinlichkeit für eine Wartezeit von mehr als $t$ Minuten beträgt $e^{-4t}$. Auf einer logarithmischen Skala ergibt sich eine Gerade. Länger als eine Minute zu warten und eine Minute ohne Anfrage zu beobachten ist dasselbe Ereignis.](../img/mdl-prob-poisson-timeline.svg)

<!--en:7944805a-->
Dasselbe Argument gilt für jede Zeitspanne $t\ge0$. In einem Poisson-Prozess mit Rate $\lambda$ übersteigt die Wartezeit $T$ vom Zeitpunkt $0$ bis zum ersten Ereignis genau dann die Dauer $t$, wenn das Intervall $[0,t]$ kein Ereignis enthält. Da die Anzahl in $[0,t]$ der Verteilung $\mathrm{Poisson}(\lambda t)$ folgt, erhalten wir

<!--en:8d6f9c1f-->
Die Verteilungsfunktion ist also $F(t)=1-e^{-\lambda t}$. Ihre Ableitung ergibt die Dichte der *Exponentialverteilung* mit Rate $\lambda>0$:

<!--en:38bd3834-->
Die Dichte misst also, wie schnell die Wahrscheinlichkeit abnimmt, dass ein Zeitfenster leer bleibt. Nach jeder Ankunft bilden die folgenden Zeitintervalle eines Poisson-Prozesses wieder unabhängige Versuche. Die Wartezeit hat daher nach jedem Ereignis dieselbe Verteilung. Die Abstände zwischen aufeinanderfolgenden Ankünften sind unabhängige Zufallsvariablen mit Verteilung $\mathrm{Exp}(\lambda)$.

<!--en:7aca24db-->
**Mittelwert und Varianz.** Zweimalige partielle Integration liefert die Momente. Mit $\int_0^\infty u\,e^{-u}\,du=[-ue^{-u}]_0^\infty+\int_0^\infty e^{-u}\,du=1$ und $\int_0^\infty u^2e^{-u}\,du=[-u^2e^{-u}]_0^\infty+2\int_0^\infty u\,e^{-u}\,du=2$ ergibt die Substitution $u=\lambda t$

<!--en:633c9d9a-->
Wie zu erwarten, bedeutet eine höhere Ankunftsrate kürzere Wartezeiten. Bei vier Anfragen pro Minute warten wir im Mittel $\tfrac14$ Minute, also 15 Sekunden. Der Median $m$ erfüllt $F(m)=\tfrac12$. Daraus folgen $m=\log2/\lambda\approx10.4$ Sekunden: Die Hälfte aller Wartezeiten liegt unter etwa 10 Sekunden, obwohl der Mittelwert 15 Sekunden beträgt. Die Dichte hat ihr Maximum bei $t=0$ und läuft nach rechts langsam aus. Dadurch liegt der Mittelwert deutlich über dem Median.

<!--en:d12c6d2a-->
**Gedächtnislosigkeit** (engl. memorylessness). Angenommen, Sie warten bereits $s$ Minuten auf eine Anfrage. Mit welcher Wahrscheinlichkeit müssen Sie mindestens $t$ weitere Minuten warten?

<!--en:aa7da7dc-->
Die bisherige Wartezeit sagt nichts über die verbleibende Wartezeit aus. Unter allen stetigen Verteilungen auf $(0,\infty)$ ist nur die Exponentialverteilung *gedächtnislos*. Das beweisen wir in Aufgabe :exref:`ex_mdl-distributions-memoryless-converse`. An dieser Eigenschaft erkennt man auch die Grenzen des Modells. Verschleißt der Glühfaden einer Lampe im Betrieb, steigt mit ihrem Alter die Wahrscheinlichkeit eines Ausfalls in der nächsten Stunde. Ihre Lebensdauer ist also nicht exponentialverteilt.

<!--en:fe468cfb-->
**Die Gammaverteilung.** Betrachten wir nun die Zeit $T_k$ bis zur Ankunft der $k$-ten Anfrage. Da $T_k$ die Summe von $k$ unabhängigen exponentialverteilten Wartezeiten ist, folgen aus der Linearität des Erwartungswerts und der Unabhängigkeit unmittelbar der Mittelwert $k/\lambda$ und die Varianz $k/\lambda^2$. Auch die Dichte ergibt sich direkt aus dem Poisson-Prozess. Die $k$-te Anfrage trifft genau dann nach dem Zeitpunkt $t$ ein, wenn höchstens $k-1$ Ereignisse in $[0,t]$ fallen. Somit gilt

<!--en:f9766818-->
Leiten wir den $j$-ten Summanden nach $t$ ab, ergibt die Produktregel $-\lambda e^{-\lambda t}\frac{(\lambda t)^j}{j!}+\lambda e^{-\lambda t}\frac{(\lambda t)^{j-1}}{(j-1)!}$. Der zweite Teil hebt den ersten Teil des vorhergehenden Summanden auf. In dieser Teleskopsumme bleibt nur der erste Teil des letzten Summanden ($j=k-1$) übrig:

<!--en:0826f835-->
Für $k=1$ erhalten wir wieder die Exponentialdichte. Um diese Familie auf nicht ganzzahlige $k$ zu erweitern, ersetzen wir $(k-1)!$ durch die *Gammafunktion* $\Gamma(z)=\int_0^\infty u^{z-1}e^{-u}\,du$ für $z>0$. Partielle Integration liefert $\Gamma(z+1)=z\,\Gamma(z)$. Wegen $\Gamma(1)=1$ gilt daher $\Gamma(n+1)=n!$ für jede nichtnegative ganze Zahl $n$. So erhalten wir die *Gammaverteilung* mit Formparameter $\alpha>0$ und Rate $\beta>0$:

<!--en:e67e4696-->
Die Substitution $u=\beta t$ führt das Integral von $p(t)$ auf $\Gamma(\alpha)/\Gamma(\alpha)=1$ zurück. Dieselbe Substitution ergibt $\mathbb{E}[T]=\Gamma(\alpha+1)/(\beta\,\Gamma(\alpha))=\alpha/\beta$ und $\textrm{Var}(T)=\alpha/\beta^2$. Für $\alpha=k$ und $\beta=\lambda$ sind dies die Momente der Wartezeit bis zum $k$-ten Ereignis. Bei unserem Webserver trifft die vierte Anfrage nach durchschnittlich $1$ Minute ein. Die Standardabweichung beträgt $0.5$ Minuten und es gilt $P(T_4>1)=P(N(1)\le3)\approx0.43$. In 43% der Minuten treffen also weniger als vier Anfragen ein, obwohl der Durchschnitt bei vier liegt.

<!--en:03edf881-->
**Sampling.** Wie wir aus :numref:`sec_mdl-random_variables` wissen, verwandelt die Inversionsmethode (engl. inverse-transform method) eine gleichverteilte Zufallszahl $U$ in eine Ziehung aus einer beliebigen stetigen Verteilung mit invertierbarer Verteilungsfunktion $F$. Dazu berechnen wir $X=F^{-1}(U)$. Für die Exponentialverteilung gilt $F^{-1}(q)=-\log(1-q)/\lambda$ und damit $T=-\log(1-U)/\lambda$. Obwohl $1-U$ und $U$ dieselbe Verteilung haben, ist die Schreibweise $1-U$ in der Praxis wichtig. Zufallszahlengeneratoren liefern Werte in $[0,1)$. Dort kann $U=0$ auftreten, und $-\log U$ würde einen Überlauf ins Unendliche verursachen.

<!--en:e6984394-->
Wenn der Zusammenhang zwischen Poisson-Prozess und Exponentialverteilung stimmt, muss die Konstruktion auch rückwärts funktionieren. Wir erzeugen unabhängige exponentialverteilte Zeitabstände, summieren sie zu Ankunftszeiten auf und sollten wieder poissonverteilte Anzahlen pro Minute erhalten. Prüfen wir das mit $60{,}000$ Zeitabständen bei Rate $4$ und betrachten wir die ersten $10{,}000$ Minuten.

<!--en:996cf679-->
Mittelwert und Varianz der simulierten Anzahlen liegen tatsächlich nahe bei $4$. Sowohl der Anteil der Minuten ohne Anfrage als auch der Anteil der Wartezeiten über einer Minute liegen nahe bei $e^{-4}$. Der Anteil der Minuten mit höchstens drei Anfragen stimmt mit der Vorhersage der Gammaverteilung $P(T_4>1)\approx0.43$ überein. Aus exponentialverteilten Zeitabständen entstehen also poissonverteilte Anzahlen.

<!--en:e561bed1-->
### Gleichverteilung

<!--en:6293004f-->
Alle Verfahren zum Ziehen von Zufallszahlen in diesem Abschnitt haben denselben Ausgangspunkt: eine gleichverteilt aus $[0,1)$ gezogene Zahl. Verallgemeinern wir diese Verteilung auf ein beliebiges endliches Intervall, erhalten wir die stetige Gleichverteilung.

<!--en:33e14e21-->
Die *Gleichverteilung* $U(a,b)$ auf einem Intervall $[a,b]$ mit $a<b$ hat eine konstante Dichte. Ihr Wert ergibt sich aus der Bedingung, dass das Integral eins sein muss:

<!--en:caa618a5-->
Gleich lange Teilintervalle haben dieselbe Wahrscheinlichkeit, nämlich ihre Länge geteilt durch $b-a$. Beide Momente folgen durch direkte Integration. Für die Varianz vereinfachen wir die Rechnung, indem wir den Mittelpunkt des Intervalls in den Ursprung verschieben. Das Intervall reicht dann von $-c$ bis $c$ mit $c=\tfrac{b-a}2$:

<!--en:c84e2989-->
Insbesondere hat $U(0,1)$ den Mittelwert $\tfrac12$ und die Varianz $\tfrac1{12}$, während $U(-\sqrt3,\sqrt3)$ die Varianz $(2\sqrt3)^2/12=1$ hat. Diese Formel ist der stetige Grenzwert der Varianz der diskreten Gleichverteilung: $n$ gleich wahrscheinliche Punkte im Abstand $h$ haben die Varianz $h^2(n^2-1)/12$. Sie konvergiert gegen $(b-a)^2/12$, wenn die Punkte das Intervall $[a,b]$ mit $nh=b-a$ ausfüllen. Mit der Inversionsmethode lässt sich $U(0,1)$ in jede stetige Verteilung überführen, wie wir es gerade für die Exponentialverteilung getan haben. Die Dichte der Gleichverteilung ist flach, hat scharfe Kanten und fällt außerhalb des Intervalls sofort auf null. Sie sieht also ganz anders aus als eine Gauß-Dichte. Damit eignet sie sich besonders gut, um weiter unten zu beobachten, wie aus Summen gleichverteilter Zufallsvariablen nach dem zentralen Grenzwertsatz eine Glockenkurve entsteht.

<!--en:a7504b76-->
### Gauß-Verteilung

<!--en:3cc7b822-->
Mit dem Quotienten aufeinanderfolgender Wahrscheinlichkeiten können wir jede Binomialwahrscheinlichkeit sehr genau berechnen. Wie die Verteilung insgesamt aussieht, verrät er uns noch nicht. Gibt es einen einfachen geschlossenen Ausdruck für ihre Form in der Nähe des Maximums? Erstaunlicherweise gibt es ihn, und darin kommen nur Mittelwert und Varianz des Münzwurfs vor. Das führt uns zur wichtigsten Verteilung in der Wahrscheinlichkeitsrechnung und im maschinellen Lernen.

<!--en:9cbe033c-->
**Der logarithmierte Quotient nahe dem Mittelwert.** Setzen wir $q=1-p$ und beschreiben wir die Anzahl durch ihre Abweichung $x$ vom Mittelwert, also $k=np+x$. Setzen wir diesen Ausdruck in den Quotienten aufeinanderfolgender Binomialwahrscheinlichkeiten ein, erhalten wir

<!--en:ccc7fa91-->
Liegt $x$ in der Größenordnung der Standardabweichung $\sqrt{npq}$ und ist $n$ groß, dann sind sowohl $x/(nq)$ als auch $(x+1)/(np)$ klein. Logarithmieren wir und verwenden die lineare Näherung $\log(1+\epsilon)\approx\epsilon$, so ergibt sich

<!--en:573db108-->
Dabei haben wir die Brüche mithilfe von $\frac1{nq}+\frac1{np}=\frac{p+q}{npq}=\frac{1}{npq}$ zusammengefasst. In führender Ordnung hängt die Änderung beim Schritt von $\log P(X=k)$ zu $\log P(X=k+1)$ linear vom Abstand $x$ zum Mittelwert ab, mit Steigung $-1/(npq)$. Bei welcher Funktion ändern sich die Zuwächse linear? Bei einer Parabel. Tatsächlich ändert sich die Parabel $-x^2/(2npq)$ um

<!--en:1dcb8fb5-->
wenn $x$ um $1$ wächst. Dies entspricht dem Zuwachs von $\log P$ bis auf eine kleine Konstante $(p-q)/(2npq)$. Bei einer fairen Münze verschwindet die Konstante ganz. Im allgemeinen Fall summiert sie sich nur zu $O(1/\sqrt n)$ über die $\sqrt{npq}$ Schritte in der Nähe des Maximums. Wir schließen daraus, dass

<!--en:5c45b19f-->
für eine Konstante $C$ und jede Anzahl $k=np+x$ im Umkreis weniger Standardabweichungen vom Mittelwert gilt. Dabei muss $np$ keine ganze Zahl sein (bei unseren 13 Würfen gilt $np=6.5$). Exponenzieren wir beide Seiten, erkennen wir nahe dem Maximum eine Glockenkurve der Form $e^{C}e^{-x^2/(2npq)}$. Sie ist um den Mittelwert zentriert, ihre Breite bestimmt die Standardabweichung $\sqrt{npq}$. Da wir die Restterme höherer Ordnung ohne Abschätzung weggelassen haben, ist dies eine Beweisskizze. Das präzise Ergebnis ist der *Satz von de Moivre und Laplace*, die älteste Form des zentralen Grenzwertsatzes.

<!--en:9ec62950-->
**Die Normierungskonstante.** Um die Konstante $C$ zu bestimmen, nutzen wir aus, dass sich die Wahrscheinlichkeiten zu eins summieren müssen. Zwischen benachbarten ganzzahligen Anzahlen liegt der Abstand $1$. Die Summe der Glockenkurve über alle $k$ lässt sich daher gut durch das Integral $e^C\int_{-\infty}^\infty e^{-x^2/(2npq)}\,dx$ annähern. Damit bleibt $I=\int_{-\infty}^\infty e^{-x^2/2}\,dx$ zu berechnen. Das sieht zunächst aussichtslos aus, denn $e^{-x^2/2}$ besitzt keine elementare Stammfunktion. Der klassische Trick besteht darin, stattdessen $I^2$ zu berechnen. Wir schreiben das Quadrat als Doppelintegral über die Ebene und wechseln zu Polarkoordinaten $x=r\cos\theta$, $y=r\sin\theta$ mit Flächenelement $dx\,dy=r\,dr\,d\theta$. Der Integrand lässt sich nun elementar integrieren:

<!--en:2e3a9f10-->
Damit gilt $I=\sqrt{2\pi}$. Der zusätzliche Faktor $r$ aus der Jacobi-Determinante der Polarkoordinaten sorgt dafür, dass $r\,e^{-r^2/2}$ die elementare Stammfunktion $-e^{-r^2/2}$ hat. Für $e^{-x^2/2}$ allein gibt es keine solche Stammfunktion. Die Variablentransformation $x=\sqrt{npq}\,u$ liefert dann $\int_{-\infty}^\infty e^{-x^2/(2npq)}\,dx=\sqrt{2\pi npq}$. Damit steht $e^C=1/\sqrt{2\pi npq}$ fest und wir erhalten

<!--en:54780370-->
Prüfen wir das an unseren 13 fairen Würfen. Hier gilt $npq=3.25$, und $k=9$ entspricht $x=2.5$. Die Glockenkurve liefert $e^{-2.5^2/6.5}/\sqrt{2\pi\cdot3.25}\approx0.0846$. Das liegt erstaunlich nahe am exakten Wert $0.0873$, selbst für so kleines $n$. Im Folgenden messen wir, wie schnell die Näherung mit wachsendem $n$ genauer wird.

<!--en:115dce97-->
Bei $n=13$ und $k=9$ liegt die Glockenkurve nur etwa 3% zu niedrig. Ihr relativer Fehler am Maximum fällt von $1.7\%$ bei $n=13$ auf $0.25\%$ bei $n=100$ und $0.03\%$ bei $n=1000$. Er nimmt also ungefähr wie $1/n$ ab.

<!--en:495b5557-->
**Die Standardnormalverteilung.** Der quadratische Exponent legt eine neue Skala nahe. Die standardisierte Anzahl $Y^{(n)}=(X-np)/\sqrt{npq}$ hat den Mittelwert $0$ und die Varianz $1$ für jedes $n$. Ihre möglichen Werte haben den Abstand $1/\sqrt{npq}$. In diesen Einheiten lautet unsere Näherung $P(X=k)\approx\phi(y)/\sqrt{npq}$ an der Stelle $y=(k-np)/\sqrt{npq}$. Es handelt sich also um eine glatte Dichte $\phi(y)$, multipliziert mit dem Gitterabstand, wobei

<!--en:bc197bfd-->
Dies ist die Dichte der *Standardnormalverteilung* $\mathcal N(0,1)$, der Grenzverteilung der standardisierten Binomialverteilung. Wegen $I=\sqrt{2\pi}$ ist ihr Integral eins. Ihr Mittelwert beträgt aufgrund der Symmetrie um den Ursprung $0$. Für die Varianz liefert partielle Integration mit $u=x$ und $dv=x\,e^{-x^2/2}\,dx$

<!--en:c94327ba-->
Die Standardnormalverteilung hat also die Varianz $1$. Verschieben und skalieren wir eine standardnormalverteilte Variable $Y$ mit Lageparameter $\mu$ und Skalenparameter $\sigma>0$, erhalten wir $X=\mu+\sigma Y$. Nach der Formel zur Variablentransformation aus :numref:`sec_mdl-joint-distributions` lautet ihre Dichte $\phi((x-\mu)/\sigma)/\sigma$. Damit ist die allgemeine *Gauß-Verteilung* (oder Normalverteilung) definiert:

<!--en:0b1941d5-->
Wegen $X=\mu+\sigma Y$ ist ihr Mittelwert $\mu$ und ihre Varianz $\sigma^2$. Die beiden Parameter geben also direkt Mittelwert und Varianz an, wobei $\sigma$ die Breite der Glockenkurve bestimmt.

<!--en:d5dab0ec-->
**Der zentrale Grenzwertsatz.** Blicken wir auf die Herleitung zurück, so haben außer Mittelwert und Varianz erstaunlich wenige Eigenschaften der Münzwürfe die endgültige Kurvenform bestimmt. Tatsächlich benötigt der allgemeine Satz keine weiteren Angaben zur Verteilung. Seien $X_1,X_2,\dots$ unabhängige und identisch verteilte Zufallsvariablen mit Mittelwert $\mu$ und endlicher Varianz $0<\sigma^2<\infty$, und sei $S_n=\sum_{i=1}^n X_i$. Dann gilt für jede reelle Zahl $y$

<!--en:38d99fba-->
In Worten: Die standardisierte Summe vieler unabhängiger Ziehungen aus *jeder beliebigen* Verteilung mit endlicher, positiver Varianz konvergiert in Verteilung gegen die Standardnormalverteilung. Für Bernoulli-Summanden ist dies der Satz von de Moivre und Laplace. Einen allgemeinen Beweis gibt :cite:`Wasserman.2013`. Im Grenzwert spielen von den Eigenschaften der einzelnen Summanden nur Mittelwert und Varianz eine Rolle. Deshalb sehen Größen, in denen viele kleine unabhängige Einflüsse zusammenkommen, so häufig gaußverteilt aus. Beispiele sind Sensor- und Messrauschen.

<!--en:f342f020-->
**Summen gleichverteilter Zufallsvariablen.** Besonders anschaulich wird der zentrale Grenzwertsatz, wenn wir ihn in Aktion sehen. Beginnen wir mit $U(0,1)$. Die Dichte ist flach, hat scharfe Kanten und ist außerhalb des Intervalls null. Addieren wir $n$ unabhängige Ziehungen, standardisieren wir die Summe mit dem Mittelwert $n/2$ und der Standardabweichung $\sqrt{n/12}$ und vergleichen wir die Histogramme mit $\phi$.

<!--en:c7f9cecc-->
Bei $n=1$ sehen wir das flache Plateau der Gleichverteilung, bei $n=2$ ein spitzes Dreieck. Schon bei $n=8$ folgt das Histogramm eng der gaußschen Glockenkurve. Bei $n=32$ lassen sich die beiden Kurven in der Auflösung der Abbildung nicht mehr unterscheiden.

<!--en:90e00c2f-->
Auf die Voraussetzung einer endlichen Varianz können wir allerdings nicht verzichten. Wie wir in :numref:`sec_mdl-random_variables` gesehen haben, stabilisieren sich Mittelwerte von Cauchy-verteilten Ziehungen nie, unabhängig von der Anzahl der Ziehungen. Dagegen folgt der Mittelwert von $n$ unabhängigen standardnormalverteilten Größen der Verteilung $\mathcal N(0,1/n)$ und konzentriert sich eng um null.

<!--en:1d98deb6-->
**Gaußsches Rauschen und quadratischer Fehler.** Sagt ein Regressionsnetz $\hat y$ vorher und modelliert es das beobachtete Label als $y\sim\mathcal N(\hat y,\sigma^2)$, dann weist es $y$ die negative Log-Wahrscheinlichkeit $-\log p(y)=(y-\hat y)^2/(2\sigma^2)+\tfrac12\log(2\pi\sigma^2)$ zu. Bis auf einen positiven Skalenfaktor und eine additive Konstante ist das der quadratische Fehler als Verlust (:numref:`subsec_normal_distribution_and_squared_loss`).

<!--en:6c4a1f43-->
### Laplace-Verteilung

<!--en:84a4d796-->
Da der quadratische Fehler quadratisch wächst, wiegt ein Residuum von $10$ hundertmal so schwer wie eines von $1$. Schon wenige Ausreißer können die Anpassung daher stark verfälschen. Der absolute Fehler wächst als Verlust nur linear. Ihm entspricht eine Rauschverteilung, die extremen Werten mehr Wahrscheinlichkeit zuweist (:numref:`subsec_linreg-loss-menu`).

<!--en:ec329098-->
Zur Konstruktion wählen wir ein zufälliges Vorzeichen $S$ mit $P(S=1)=P(S=-1)=\tfrac12$ und eine davon unabhängige exponentialverteilte Variable $E$ mit Mittelwert $b>0$ (Rate $1/b$). Wir setzen $X=\mu+SE$. Für $S=1$ liegt $X$ oberhalb von $\mu$ und es gilt $X-\mu=E$. Für $x>\mu$ beträgt die Dichte daher $\tfrac12\cdot\tfrac1b e^{-(x-\mu)/b}$. Aus Symmetriegründen gilt unterhalb von $\mu$ derselbe Ausdruck, wenn wir $x-\mu$ durch $|x-\mu|$ ersetzen. Damit erhalten wir die *Laplace-Verteilung*:

<!--en:740a4a1a-->
Jede Hälfte der Verteilung trägt die halbe Masse einer Exponentialverteilung. Die Dichte ist symmetrisch um $\mu$, hat dort eine ausgeprägte Spitze und fällt auf beiden Seiten exponentiell ab.

<!--en:ccc86d24-->
**Mittelwert und Varianz.** Beide Momente folgen direkt aus $X=\mu+SE$. Wegen der Unabhängigkeit gilt $\mathbb{E}[SE]=\mathbb{E}[S]\,\mathbb{E}[E]=0$ und damit $\mathbb{E}[X]=\mu$. Da stets $S^2=1$ gilt, ist die Varianz gleich dem zweiten Moment der Exponentialverteilung, also $2/\lambda^2$ mit $\lambda=1/b$:

<!--en:80fc47d2-->
**Laplace-Rauschen und absoluter Fehler.** Modelliert ein Regressionsnetz das Label als $y\sim\mathrm{Laplace}(\hat y,b)$, ergibt sich die negative Log-Wahrscheinlichkeit $-\log p(y)=|y-\hat y|/b+\log 2b$. Bis auf Skalierung und eine Konstante ist dies der absolute Fehler. Als A-priori-Verteilung der Modellgewichte liefert die Laplace-Dichte den $\ell_1$-Strafterm $\|\mathbf w\|_1$ des Lasso :cite:`Tibshirani.1996`. Passen wir einen Lageparameter $\mu$ an Daten $x_1,\dots,x_m$ an, minimieren wir durch Maximieren der Laplace-Likelihood den Ausdruck $\sum_i|x_i-\mu|$. Das Minimum liegt beim Stichprobenmedian. Ein einziger verfälschter Datenpunkt kann den Stichprobenmittelwert beliebig weit verschieben, den Median aber höchstens um einen Rang.

<!--en:7a17b1b5-->
**Randbereiche der Verteilungen.** :numref:`fig_mdl-prob-continuous-pdfs` vergleicht die Dichten der Gleich-, Gauß- und Laplace-Verteilung bei gleicher Varianz. Für $\sigma^2=2b^2$ beträgt die Wahrscheinlichkeit, dass eine Laplace-verteilte Variable mehr als drei Standardabweichungen von ihrem Zentrum entfernt liegt, $P(|X-\mu|>3\sigma)=e^{-3\sigma/b}=e^{-3\sqrt2}\approx0.014$. Bei der Gauß-Verteilung sind es $0.0027$. Die Laplace-Verteilung weist diesem Bereich also etwa fünfmal so viel Wahrscheinlichkeit zu.

<!--en:129b7ccf-->
![Drei Dichten mit Varianz eins: die Gleichverteilung auf $[-\sqrt{3},\sqrt{3}]$, die Standardnormalverteilung und die Laplace-Verteilung mit $b=1/\sqrt{2}$. (a) Dichten auf linearer Skala. (b) Dieselben Dichten auf logarithmischer Skala. Die Gauß-Dichte ergibt eine Parabel, die Laplace-Dichte zwei Geraden und die Dichte der Gleichverteilung ein flaches Plateau mit Grenzen bei $\pm\sqrt{3}$. Jenseits von drei Standardabweichungen liegt bei der Laplace-Verteilung etwa fünfmal so viel Wahrscheinlichkeit wie bei der Gauß-Verteilung.](../img/mdl-prob-continuous-pdfs.svg)

<!--en:708bcbc5-->
Überprüfen wir diesen Vergleich direkt. Dazu ziehen wir nach unserer Konstruktion aus der Laplace-Verteilung (zufälliges Vorzeichen mal exponentialverteilte Größe) und zählen, wie oft der Abstand zum Mittelwert drei Standardabweichungen übersteigt.

<!--en:09370717-->
Beide Stichprobenvarianzen liegen nahe bei $2$. Die Anteile jenseits von drei Standardabweichungen liegen bei etwa $0.014$ und $0.003$.

<!--en:bfa39425-->
## Die multivariate Gauß-Verteilung

<!--en:33e12480-->
Zufallsgrößen sind im Deep Learning selten Skalare. Bildrauschen, latente Darstellungen in einem generativen Modell und Vorhersagen für einen Batch von Eingaben sind Vektoren, deren Koordinaten voneinander abhängen können. Unter den Verteilungen für korrelierte Vektoren zeichnet sich die multivariate Gauß-Verteilung dadurch aus, dass sie bei fast allen für uns wichtigen Operationen analytisch handhabbar bleibt. Als durchgängiges Beispiel betrachten wir zwei Koordinaten mit Varianz $2$ und gemeinsamer Kovarianz $1$:

<!--en:b03c987a-->
### Niveaumengen und Eigenvektoren

<!--en:8bfba234-->
Die *multivariate Gauß-Verteilung* $\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ auf $\mathbb R^d$ hat als Parameter einen Mittelwertvektor $\boldsymbol\mu$ und eine symmetrische, positiv definite Kovarianzmatrix $\boldsymbol\Sigma$. Ihre Dichte ist

<!--en:407dc20e-->
Dabei bezeichnet $|\boldsymbol\Sigma|$ die Determinante von $\boldsymbol\Sigma$. Für $d=1$ ergibt sich wieder :eqref:`eq_mdl-gaussian_pdf`. Die Dichte hängt von $\mathbf x$ nur über die quadratische Form im Exponenten ab. Auf jeder Niveaumenge dieser Form ist also auch die Dichte konstant. Welche Gestalt haben diese Niveaumengen?

<!--en:38b5c1ec-->
**Unser Beispiel von Hand.** Setzen wir $\boldsymbol\mu=\mathbf 0$ und drehen wir die Koordinatenachsen um $45^\circ$. So erhalten wir die Summen- und Differenzkoordinaten $u=(x_1+x_2)/\sqrt2$ und $w=(x_1-x_2)/\sqrt2$. Ihre Varianzen und ihre Kovarianz folgen direkt aus den Einträgen von $\boldsymbol\Sigma$:

<!--en:0f5e8c9f-->
In den gedrehten Koordinaten ist die Kovarianzmatrix diagonal. Die neuen Achsen folgen den Eigenvektoren $\mathbf v_1=(1,1)/\sqrt2$ und $\mathbf v_2=(1,-1)/\sqrt2$ von $\boldsymbol\Sigma$ mit den Eigenwerten $\gamma_1=3$ und $\gamma_2=1$ (tatsächlich gilt $\boldsymbol\Sigma\mathbf v_1=(3,3)/\sqrt2=3\mathbf v_1$).

<!--en:be61d7e7-->
**Der allgemeine Fall.** Nach dem Spektralsatz besitzt jede reelle symmetrische Matrix eine Orthonormalbasis aus Eigenvektoren. Es gilt also $\boldsymbol\Sigma=\mathbf V\operatorname{diag}(\gamma_1,\dots,\gamma_d)\mathbf V^\top$ mit $\mathbf V^\top\mathbf V=\mathbf I$. Wegen der positiven Definitheit erfüllt jeder Eigenwert $\gamma_i>0$. In den Eigenkoordinaten $\mathbf u=\mathbf V^\top(\mathbf x-\boldsymbol\mu)$ wird die quadratische Form zu einer gewichteten Quadratsumme:

<!--en:1b62ed1d-->
Die Niveaumenge, auf der diese Summe gleich $r^2$ ist, bildet ein Ellipsoid. Seine Achsen verlaufen in Richtung der Eigenvektoren $\mathbf v_i$ und seine Halbachsen haben die Längen $r\sqrt{\gamma_i}$. Die Niveaumengen jeder multivariaten Gauß-Dichte sind somit Ellipsoide. Ihre Hauptachsen folgen den Eigenvektoren von $\boldsymbol\Sigma$, ihre Längen sind proportional zu den Standardabweichungen $\sqrt{\gamma_i}$. In unserem Beispiel ergeben sich um $45^\circ$ geneigte Ellipsen mit Achsenverhältnis $\sqrt3:1$ (:numref:`fig_mdl-prob-mvn-contours`, links). Wegen $|\boldsymbol\Sigma|=\prod_i\gamma_i$ zerfällt die Dichte in den Koordinaten $\mathbf u$ zudem in ein Produkt von $d$ eindimensionalen Gauß-Dichten. Entlang der Eigenvektoren sind die Koordinaten also unabhängig und haben die Varianzen $\gamma_i$. Im Spezialfall $\boldsymbol\Sigma=\sigma^2\mathbf I$ sind die Niveaumengen Kugeloberflächen und schon die ursprünglichen Koordinaten sind unabhängig.

<!--en:a32775dc-->
![Unsere bivariate Gauß-Verteilung mit Kovarianzmatrix $\boldsymbol{\Sigma}$ und den Zeilen $(2, 1)$ und $(1, 2)$. (a) 400 Ziehungen und drei Niveaulinien der Dichte, auf denen die quadratische Form im Exponenten die Werte $1$, $4$ und $9$ annimmt. Die Ellipsenachsen folgen den Eigenvektoren von $\boldsymbol{\Sigma}$. Ihre Halbachsenlängen sind proportional zu $\sqrt{\gamma_1} = \sqrt{3}$ und $\sqrt{\gamma_2} = 1$. Die Pfeile in Richtung $\mathbf{v}_1$ und $\mathbf{v}_2$ reichen bis zur mittleren Niveaulinie. (b) Die Beobachtung $x_2 = 2$ liefert einen Schnitt durch die Dichte entlang der gestrichelten Linie. Daraus ergibt sich die bedingte Dichte von $\mathcal{N}(1, \frac{3}{2})$ für $x_1$. Die bedingten Mittelwerte aller Schnitte liegen auf der Geraden $x_1 = x_2/2$, die bedingte Varianz bleibt konstant bei $3/2$.](../img/mdl-prob-mvn-contours.svg)

<!--en:42d5e5ab-->
In hohen Dimensionen kann die aus wenigen Dimensionen vertraute Vorstellung einer Glocke, die sich um den Mittelwert konzentriert, täuschen. Ein standardnormalverteilter Zufallsvektor in $\mathbb R^d$ hat fast immer eine Norm nahe $\sqrt d$, und zwei unabhängige Ziehungen stehen nahezu senkrecht aufeinander. Beide Aussagen beweisen wir in :numref:`sec_mdl-concentration-generalization`.

<!--en:1913697b-->
### Lineare Abbildungen und Sampling

<!--en:8502b00e-->
In der Praxis erzeugen wir häufig neue gaußsche Zufallsvektoren, indem wir die Koordinaten vorhandener Vektoren mischen oder die Vektoren auf einen Unterraum projizieren. Beides sind lineare Abbildungen. Gauß-Verteilungen sind unter anderem deshalb so praktisch, weil sie unter linearen Abbildungen gaußsch bleiben. Wir beweisen das zuerst für invertierbare quadratische Matrizen. Für das Sampling reicht dieser Fall aus. Projektionen mit rechteckigen Matrizen behandeln wir im nächsten Unterabschnitt, sobald wir gezeigt haben, dass jeder Teilvektor eines gaußschen Zufallsvektors selbst gaußverteilt ist.

<!--en:6e87cd8e-->
**Proposition (invertierbare lineare Abbildungen).** Sei $\mathbf x\sim\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ in $\mathbb R^d$, sei $\mathbf A$ eine invertierbare $d\times d$-Matrix und sei $\mathbf b\in\mathbb R^d$. Dann gilt

<!--en:b647e172-->
**Beweis.** Wegen der Linearität des Erwartungswerts gelten die angegebenen Formeln für Mittelwert und Kovarianz für jede Verteilung mit endlichen zweiten Momenten. Insbesondere ist $\mathbb{E}[(\mathbf A\mathbf x-\mathbf A\boldsymbol\mu)(\mathbf A\mathbf x-\mathbf A\boldsymbol\mu)^\top]$ gleich $\mathbf A\,\mathbb{E}[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top]\,\mathbf A^\top
=\mathbf A\boldsymbol\Sigma\mathbf A^\top$. Zu zeigen bleibt, dass die transformierte Dichte eine Gauß-Dichte ist. Nach der Formel zur Variablentransformation aus :numref:`sec_mdl-joint-distributions` hat $\mathbf y=\mathbf A\mathbf x+\mathbf b$ die Dichte $p_{\mathbf y}(\mathbf y)=p_{\mathbf x}(\mathbf A^{-1}(\mathbf y-\mathbf b))/|\det\mathbf A|$. Setzen wir $\mathbf c=\mathbf y-\mathbf A\boldsymbol\mu-\mathbf b$, lässt sich der Exponent als $-\tfrac12\mathbf c^\top\mathbf A^{-\top}\boldsymbol\Sigma^{-1}\mathbf A^{-1}\mathbf c
=-\tfrac12\mathbf c^\top(\mathbf A\boldsymbol\Sigma\mathbf A^\top)^{-1}\mathbf c$ schreiben. Für den Normierungsfaktor gilt $|\det\mathbf A|\,|\boldsymbol\Sigma|^{1/2}=|\mathbf A\boldsymbol\Sigma\mathbf A^\top|^{1/2}$. Also ist $p_{\mathbf y}$ die Gauß-Dichte mit dem behaupteten Mittelwert und der behaupteten Kovarianz. $\blacksquare$

<!--en:15c0f788-->
Invertierbarkeit ist hier notwendig. Für $\mathbf A=\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$ sind beide Ausgabekoordinaten stets identisch, und $\mathbf A\boldsymbol\Sigma\mathbf A^\top=\left(\begin{smallmatrix}6&6\\6&6\end{smallmatrix}\right)$ ist singulär. Der resultierende Zufallsvektor liegt vollständig auf der Geraden $y_1=y_2$ und besitzt keine zweidimensionale Dichte in der Ebene.

<!--en:886ec9f6-->
**Sampling mit einem Matrixfaktor.** Wie ziehen wir aus $\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$, wenn unser Zufallszahlengenerator nur unabhängige standardnormalverteilte Zahlen liefert? Fassen wir $d$ unabhängige Ziehungen zu $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ zusammen. Die gemeinsame Dichte ist das Produkt von $d$ Kopien von $\phi$. Für jede quadratische Matrix $\mathbf L$ mit $\mathbf L\mathbf L^\top=\boldsymbol\Sigma$ ist $\mathbf L$ invertierbar, da $\boldsymbol\Sigma$ positiv definit ist. Unsere Proposition garantiert somit, dass $\mathbf x=\boldsymbol\mu+\mathbf L\mathbf z$ gaußverteilt ist, mit Mittelwert $\boldsymbol\mu$ und Kovarianz $\mathbf L\,\mathbf I\,\mathbf L^\top=\boldsymbol\Sigma$.

<!--en:5a35df12-->
Die Multiplikation mit einem Matrixfaktor verwandelt also unabhängiges in korreliertes Rauschen. Bei einer festen Ziehung ist $\mathbf x$ eine glatte, differenzierbare Funktion von $\boldsymbol\mu$ und $\mathbf L$. Halten wir die Ziehung $\mathbf z$ fest, erhalten wir mit derselben Konstruktion den *Reparametrisierungstrick*, mit dem Variationsautoencoder Gradienten durch stochastische Schichten zurückpropagieren (:numref:`subsec_mdl-stochastic-gradient-estimators`). Am schnellsten lässt sich der *Cholesky-Faktor* berechnen. Jede symmetrische, positiv definite Matrix hat eine eindeutige Zerlegung $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$, wobei $\mathbf L$ eine untere Dreiecksmatrix mit positiven Diagonaleinträgen ist. Gleichen wir in unserem Beispiel die Einträge von $\mathbf L\mathbf L^\top$ nacheinander mit denen von $\boldsymbol\Sigma$ ab, erhalten wir $L_{11}=\sqrt2$, dann $L_{21}=\Sigma_{21}/L_{11}=1/\sqrt2$ und schließlich $L_{22}=\sqrt{\Sigma_{22}-L_{21}^2}=\sqrt{3/2}$. Tatsächlich gilt $L_{21}^2+L_{22}^2=\tfrac12+\tfrac32=2$.

<!--en:dd2f97a5-->
Jeder andere quadratische Faktor mit demselben Produkt aus Faktor und Transponierter eignet sich ebenso gut, etwa der Faktor $\mathbf V\operatorname{diag}(\sqrt{\gamma_1},\dots,\sqrt{\gamma_d})$ aus der Eigenwertzerlegung. Wie Aufgabe :exref:`ex_mdl-distributions-two-factors` zeigt, unterscheiden sich zwei solche Faktoren derselben Kovarianzmatrix $\boldsymbol\Sigma$ nur um eine orthogonale Matrix. Bibliotheken verwenden unterschiedliche Verfahren zum Sampling. Viele nutzen den Cholesky-Faktor. Die NumPy-Funktion `multivariate_normal` verwendet standardmäßig einen Faktor aus der Singulärwertzerlegung.

<!--en:a358c28c-->
Implementieren wir das Cholesky-Verfahren selbst, um zu sehen, wie es die Ziehungen erzeugt. Überprüfen wir sowohl die Stichprobenkovarianzmatrix als auch ihre Eigenwerte.

<!--en:d232ac58-->
Der Cholesky-Faktor stimmt mit unserer Rechnung von Hand überein. Die empirische Kovarianzmatrix weicht von $\boldsymbol\Sigma$ nur um etwa $0.01$ ab, und ihre Eigenwerte liegen nahe bei $1$ und $3$.

<!--en:7ebbadcc-->
### Konditionierung

<!--en:b71a714d-->
Häufig beobachten wir einige Koordinaten eines gaußschen Zufallsvektors und möchten auf die übrigen schließen. Wenn wir in unserem Beispiel $x_2=2$ beobachten, welchen Wert erwarten wir dann für $x_1$? Zerlegen wir den Vektor in zwei Blöcke $\mathbf x=(\mathbf x_1,\mathbf x_2)$ und unterteilen wir Mittelwert und Kovarianzmatrix entsprechend:

<!--en:3cb47ec4-->
Die Konditionierung einer Gauß-Verteilung beruht auf einer geometrischen Idee. Wir ziehen von $\mathbf x_1$ die lineare Vorhersage für $\mathbf x_1$ auf Grundlage von $\mathbf x_2$ ab. Übrig bleibt ein Residuum, das keine Information über $\mathbf x_2$ enthält.

<!--en:3d445012-->
**Proposition (Residuum und Randverteilung).** Sei $\mathbf B=\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}$. Dann ist das Residuum $\mathbf r=\mathbf x_1-\mathbf B\mathbf x_2$ unabhängig von $\mathbf x_2$, und für die Randverteilung des zweiten Blocks gilt $\mathbf x_2\sim\mathcal N(\boldsymbol\mu_2,\boldsymbol\Sigma_{22})$.

<!--en:ff0701ca-->
**Beweis.** Als Hauptdiagonalblock einer positiv definiten Matrix ist $\boldsymbol\Sigma_{22}$ selbst positiv definit und damit invertierbar. Das Residuum ist nach Konstruktion unkorreliert mit $\mathbf x_2$, denn $\textrm{Cov}(\mathbf r,\mathbf x_2)=\boldsymbol\Sigma_{12}-\mathbf B\boldsymbol\Sigma_{22}=\mathbf 0$. Die lineare Abbildung $\mathbf x\mapsto(\mathbf r,\mathbf x_2)$ entspricht der quadratischen Blockmatrix $\left(\begin{smallmatrix}\mathbf I&-\mathbf B\\\mathbf 0&\mathbf I\end{smallmatrix}\right)$. Deren Determinante ist $1$. Nach unserer Proposition über invertierbare lineare Abbildungen ist $(\mathbf r,\mathbf x_2)$ daher gemeinsam gaußverteilt. Die Kovarianzblöcke außerhalb der Diagonalen verschwinden ($\textrm{Cov}(\mathbf r,\mathbf x_2)=\mathbf 0$), die Kovarianzmatrix ist also blockdiagonal. Damit ist auch ihre Inverse blockdiagonal. Die quadratische Form im Exponenten zerfällt in eine Funktion von $\mathbf r$ plus eine Funktion von $\mathbf x_2$, und die Determinante ist das Produkt der beiden Blockdeterminanten. Die gemeinsame Dichte zerfällt folglich in eine Gauß-Dichte in $\mathbf r$ mal eine Gauß-Dichte in $\mathbf x_2$. Die beiden Blöcke sind somit unabhängig, und $\mathbf x_2$ hat den Mittelwert $\boldsymbol\mu_2$ und die Kovarianzmatrix $\boldsymbol\Sigma_{22}$. $\blacksquare$

<!--en:076fa17e-->
Der Beweis beruht auf der Wahl von $\mathbf B$, durch die das Residuum unkorreliert mit $\mathbf x_2$ ist. Die zweite Aussage zeigt zugleich, dass die Randverteilung jedes Teilvektors eines gaußschen Zufallsvektors selbst eine Gauß-Verteilung ist. Für den ersten Block vertauschen wir die Reihenfolge der Blöcke, eine invertierbare lineare Abbildung. Damit können wir nun auch den zurückgestellten Fall rechteckiger Matrizen behandeln.

<!--en:03dcaa03-->
**Folgerung (breite Matrizen).** Ist $\mathbf A$ eine $m\times d$-Matrix mit vollem Zeilenrang $m\le d$ und gilt $\mathbf b\in\mathbb R^m$, so folgt $\mathbf A\mathbf x+\mathbf b\sim\mathcal N(\mathbf A\boldsymbol\mu+\mathbf b,\mathbf A\boldsymbol\Sigma\mathbf A^\top)$. Wir können nämlich $\mathbf A$ um $d-m$ Zeilen zu einer invertierbaren $d\times d$-Matrix ergänzen. Das Bild von $\mathbf x$ ist gaußverteilt, und $\mathbf A\mathbf x+\mathbf b$ ist der erste Block dieses affin transformierten Vektors.

<!--en:3707674d-->
Damit können wir auf $\mathbf x_2$ konditionieren. Es gilt $\mathbf x_1=\mathbf r+\mathbf B\mathbf x_2$, und $\mathbf r$ ist unabhängig von $\mathbf x_2$. Fixieren wir $\mathbf x_2$, verschiebt sich die gaußsche Zufallsvariable $\mathbf r$ lediglich um den bekannten Vektor $\mathbf B\mathbf x_2$. Das Residuum hat den Mittelwert $\boldsymbol\mu_1-\mathbf B\boldsymbol\mu_2$ und die Kovarianzmatrix $\boldsymbol\Sigma_{11}-\mathbf B\boldsymbol\Sigma_{21}-\boldsymbol\Sigma_{12}\mathbf B^\top+\mathbf B\boldsymbol\Sigma_{22}\mathbf B^\top$, also $\boldsymbol\Sigma_{11}-\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}$. Folglich ist $\mathbf x_1\mid\mathbf x_2$ gaußverteilt mit den Parametern

<!--en:d0445f9f-->
Zwei Eigenschaften von :eqref:`eq_mdl-mvn_conditional` fallen auf. Der bedingte Mittelwert ist eine affine Funktion des beobachteten Blocks $\mathbf x_2$, und die bedingte Kovarianz hängt überhaupt nicht vom beobachteten Wert ab. In unserem Beispiel ist $\mathbf B=\tfrac12$. Für die Beobachtung $x_2=2$ erhalten wir daher den bedingten Mittelwert $\tfrac12\cdot2=1$ und die bedingte Varianz $2-1\cdot\tfrac12\cdot1=\tfrac32$. Es gilt also $x_1\mid x_2=2\sim\mathcal N(1,\tfrac32)$, passend zum horizontalen Schnitt bei $x_2=2$ in :numref:`fig_mdl-prob-mvn-contours` (rechts).

<!--en:9e0349ad-->
Die Matrix $\mathbf B$ ist die Matrix der linearen Regression zur Vorhersage von $\mathbf x_1$ aus $\mathbf x_2$. Subtrahieren wir das entsprechende Vielfache von $\mathbf x_2$, bleibt ein unkorreliertes Residuum. Bei der multivariaten Gauß-Verteilung *ist* der bedingte Erwartungswert die lineare Regression, die wir direkt an der Kovarianzmatrix ablesen. Die bedingte Kovarianzmatrix heißt in der linearen Algebra *Schur-Komplement* von $\boldsymbol\Sigma_{22}$ (:numref:`subsec_mdl-block-matrices`). Sie ist nie größer als $\boldsymbol\Sigma_{11}$, denn der subtrahierte Term $\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}$ ist positiv semidefinit. Beobachten wir $x_2$, sinkt unsere Unsicherheit über $x_1$ von der Varianz $2$ auf $\tfrac32$. Enthält $\mathbf x_2$ beobachtete Werte einer unbekannten Funktion und $\mathbf x_1$ ihre Werte an neuen Testeingaben, erhalten wir mit dieser Formel die Gaußprozess-Regression (:numref:`chap_gp`).

<!--en:84a0008c-->
Hier liegt ein Einwand nahe: Aus Kovarianz null folgt im Allgemeinen keine Unabhängigkeit. Das stimmt. Unser Beweis nutzt eine stärkere Voraussetzung, nämlich dass $(\mathbf r,\mathbf x_2)$ *gemeinsam* gaußverteilt ist. Ohne diese Voraussetzung gilt die Schlussfolgerung nicht. Sei beispielsweise $Z\sim\mathcal N(0,1)$ und sei $S$ ein unabhängiges, gleich wahrscheinliches Zufallsvorzeichen. Dann hat auch $SZ$ die Randverteilung $\mathcal N(0,1)$ und es gilt $\textrm{Cov}(SZ,Z)=\mathbb{E}[S]\,\mathbb{E}[Z^2]=0$. Dennoch ist $|SZ|=|Z|$, die beiden Variablen hängen also stark voneinander ab.

<!--en:11eb9907-->
Wir können die bedingte Gauß-Verteilung direkt anhand unserer Cholesky-Ziehungen überprüfen, ganz ohne Matrixformel. Behalten wir nur Ziehungen, deren $x_2$ höchstens $0.05$ von $2$ entfernt ist. Untersuchen wir die Verteilung ihrer $x_1$-Werte und prüfen wir das Residuum $r=x_1-\tfrac12x_2$.

<!--en:a57c03c8-->
Wie vorhergesagt liegt die Kovarianz des Residuums nahe bei $0$ (Kovarianz mit $x_2$), und seine Varianz liegt nahe bei $\tfrac32$. Rund viertausend Ziehungen fallen in den schmalen Streifen. Ihre $x_1$-Werte haben einen Mittelwert nahe $1$ und eine Varianz nahe $\tfrac32$, wie es $\mathcal N(1,\tfrac32)$ vorhersagt.

<!--en:e7515d36-->
Zwei Fragen bleiben offen. Erstens haben fast alle hier behandelten Verteilungen dieselbe algebraische Grundform. In :numref:`sec_mdl-exponential-family` nutzen wir das, um ihre Verluste, Gradienten und Aktualisierungen mit konjugierten Verteilungen einheitlich zu beschreiben. Zweitens haben wir bei der Berechnung von $715/8192$ zu Beginn vorausgesetzt, dass die Münze fair ist. Wie wir aus 13 Würfen auf eine unbekannte Wahrscheinlichkeit für Kopf schließen und unsere Unsicherheit darüber quantifizieren, behandelt :numref:`sec_mdl-statistics`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ddb20b16-->
Die folgende Tabelle fasst die Verteilungen dieses Abschnitts in der Reihenfolge ihrer Einführung zusammen. Bei der Kategorial- und der Multinomialverteilung stellen wir jede Ziehung als One-Hot-Vektor dar. Das entspricht der Kodierung von Klassenlabels beim Kreuzentropieverlust.

<!--en:e75a183b-->
| Verteilung | Wahrscheinlichkeitsfunktion oder Dichte | Mittelwert | Varianz | Modellannahme |

<!--en:6fb87587-->
| $\mathrm{Bernoulli}(p)$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | ein Ja-Nein-Ergebnis |

<!--en:7a7423dc-->
| $\mathrm{Binomial}(n,p)$ | $\binom nk p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | $n$ feste, unabhängige Versuche mit demselben $p$ |

<!--en:c7062a16-->
| $\mathrm{Poisson}(\lambda)$ | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ | viele unabhängige Gelegenheiten für ein seltenes Ereignis, Mittelwert gleich Varianz |

<!--en:1ab0176b-->
| $\mathrm{Categorical}(\mathbf p)$ | $p_k$ | $\mathbf p$ | $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ | eines von $K$ Ergebnissen |

<!--en:908eb515-->
| $\mathrm{Multinomial}(n,\mathbf p)$ | :eqref:`eq_mdl-multinomial_pmf` | $n\mathbf p$ | $n\bigl(\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top\bigr)$ | feste Anzahl $n$ unabhängiger Ziehungen |

<!--en:32999ceb-->
| $\mathrm{Unif}\{1,\dots,n\}$ | $1/n$ | $(n+1)/2$ | $(n^2-1)/12$ | gleich wahrscheinliche Werte |

<!--en:75b73820-->
| $\mathrm{Exp}(\lambda)$ | $\lambda e^{-\lambda t}$, $t\ge0$ | $1/\lambda$ | $1/\lambda^2$ | konstante Ereignisrate (keine Alterung) |

<!--en:31131b41-->
| $\mathrm{Gamma}(\alpha,\beta)$ | $\beta^\alpha t^{\alpha-1}e^{-\beta t}/\Gamma(\alpha)$ | $\alpha/\beta$ | $\alpha/\beta^2$ | Zeitpunkt des $\alpha$-ten Ereignisses (ganzzahliges $\alpha$) |

<!--en:0f1b0873-->
| $U(a,b)$ | $1/(b-a)$ auf $[a,b]$ | $(a+b)/2$ | $(b-a)^2/12$ | gleiche Wahrscheinlichkeit für gleich lange Intervalle |

<!--en:46d836a1-->
| $\mathcal N(\mu,\sigma^2)$ | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ | Summe vieler kleiner, unabhängiger Einflüsse mit endlicher Varianz |

<!--en:bb63a554-->
| $\mathrm{Laplace}(\mu,b)$ | $\frac{1}{2b}e^{-\lvert x-\mu\rvert/b}$ | $\mu$ | $2b^2$ | symmetrische Dichte mit exponentiellem Abfall auf beiden Seiten |

<!--en:b3814baf-->
| $\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ | :eqref:`eq_mdl-mvn_pdf` | $\boldsymbol\mu$ | $\boldsymbol\Sigma$ | gemeinsam gaußverteilte Koordinaten |

<!--en:02c88fab-->
Weitere wichtige Verteilungen begegnen uns an anderen Stellen dieses Kapitels: die Cauchy-Verteilung in :numref:`sec_mdl-random_variables`, die Lognormalverteilung in :numref:`sec_mdl-joint-distributions`, die Beta- und die Dirichlet-Verteilung in :numref:`sec_mdl-exponential-family` sowie die negative Binomialverteilung, die geometrische Verteilung und die Student-$t$-Verteilung in den Aufgaben :exref:`ex_mdl-distributions-gamma-poisson-mixture`, :exref:`ex_mdl-distributions-memoryless-converse` und :exref:`ex_mdl-distributions-student-t`.

<!--en:a38d50f7-->
Eine kleine Familie von Verteilungen genügt, um die Labels und das Rauschen der meisten Systeme des maschinellen Lernens zu modellieren. Die negative Log-Wahrscheinlichkeit jeder dieser Verteilungen liefert einen gängigen Trainingsverlust. Alle gehen aus einem einzigen Münzwurf hervor, über Summen, Grenzwerte, Wartezeiten, Zufallsvorzeichen oder korrelierte Vektoren. Aus jeder Konstruktion ergeben sich in wenigen Zeilen der Mittelwert, die Varianz und ein Verfahren zum Sampling. Der zentrale Grenzwertsatz erklärt, warum die Gauß-Verteilung auftritt, wenn sich viele kleine, unabhängige Einflüsse mit endlicher Varianz addieren. Multivariate Gauß-Verteilungen bleiben zudem unter linearen Transformationen und Konditionierung gaußverteilt. Die Konditionierung eines Blocks auf einen anderen führt direkt zur linearen Regression.

<!--en:2294243a-->
## Übungen

<!--en:8c32e9de-->
1. **Gamma-Poisson-Mischung.** Die Poisson-Rate sei selbst zufällig, $\lambda\sim\mathrm{Gamma}(\alpha,\beta)$ mit der Dichte :eqref:`eq_mdl-gamma_pdf`, und es gelte $X\mid\lambda\sim\mathrm{Poisson}(\lambda)$.
    1. Verwenden Sie den Satz vom totalen Erwartungswert $\mathbb{E}[X]=\mathbb{E}\bigl[\mathbb{E}[X\mid\lambda]\bigr]$ und den Satz von der totalen Varianz

       $$
       \textrm{Var}(X)=\mathbb{E}\bigl[\textrm{Var}(X\mid\lambda)\bigr]+\textrm{Var}\bigl(\mathbb{E}[X\mid\lambda]\bigr)
       $$

       zur Berechnung von $\mathbb{E}[X]$ und $\textrm{Var}(X)$. Drücken Sie die Varianz durch den Mittelwert $m=\alpha/\beta$ und den Formparameter $\alpha$ aus. Bei welchem Grenzübergang für $\alpha$ und $\beta$ ergibt sich wieder die Poisson-Verteilung?
    1. Zeigen Sie, dass $X$ die Wahrscheinlichkeitsfunktion der *negativen Binomialverteilung* besitzt,

       $$P(X=k)=\frac{\Gamma(k+\alpha)}{k!\,\Gamma(\alpha)}\Bigl(\frac{\beta}{1+\beta}\Bigr)^{\alpha}\Bigl(\frac{1}{1+\beta}\Bigr)^{k},\qquad k=0,1,2,\dots,$$

       indem Sie $\lambda$ mithilfe der Normierungskonstante der Gammaverteilung ausintegrieren.
    1. Ein Datensatz mit Zähldaten hat den Stichprobenmittelwert $2$ und die Stichprobenvarianz $6$. Wählen Sie $\alpha$ und $\beta$ so, dass das Modell beide Werte wiedergibt. Welche bekannte Verteilung ergibt sich?
   :exlabel:`ex_mdl-distributions-gamma-poisson-mixture`

<!--en:d754c2e4-->
1. **Umkehrung der Gedächtnislosigkeit.** ● Die Exponentialverteilung ist gedächtnislos. In dieser Aufgabe beweisen Sie die Umkehrung.
    1. Sei $X>0$ eine stetige Zufallsvariable, deren Überlebensfunktion $G(t)=P(X>t)$ die Gleichung $G(s+t)=G(s)\,G(t)$ für alle $s,t\ge0$ erfüllt. Zeigen Sie, dass $G(t)=e^{-\lambda t}$ für ein $\lambda>0$ gilt. Gehen Sie von $G(1)$ aus, erweitern Sie die Formel auf rationale $t$ und nutzen Sie dann, dass $G$ monoton fallend ist. Warum kann $G(1)$ weder $0$ noch $1$ sein?
    1. Die Zufallsvariable $X$ nehme Werte in $\{1,2,\dots\}$ an und erfülle $P(X>s+t\mid X>s)=P(X>t)$ für alle ganzen Zahlen $s,t\ge0$. Zeigen Sie, dass $X$ geometrisch verteilt ist, also $P(X>t)=(1-p)^t$ für ein $p\in(0,1]$ gilt.
   :exlabel:`ex_mdl-distributions-memoryless-converse`

<!--en:df3f37e2-->
1. [extended] **Gehäuftes Auftreten von Wörtern.** ●● :citet:`Church.Gale.1995` stellten fest, dass die Anzahl der Vorkommen eines Wortes zwischen Dokumenten stärker schwankt, als ein Poisson-Modell zulässt. Poisson-Mischungen wie die negative Binomialverteilung passen besser zu den Daten. Vollziehen Sie diesen Befund im Kleinen anhand von *The Time Machine* (:numref:`sec_text-sequence`) nach. Laden Sie den Text mit `d2l.download(d2l.DATA_URL + 'timemachine.txt')` herunter und behalten Sie wie in `d2l.TimeMachine._preprocess` nur Buchstaben bei. Zerlegen Sie den Text in Wörter und unterteilen Sie die Wortfolge in aufeinanderfolgende Abschnitte von je $L=500$ Wörtern, die als Dokumente dienen.
    1. Zählen Sie für jedes der Wörter „the“, „and“, „of“, „time“, „traveller“, „weena“ und „morlocks“ die Vorkommen in jedem Abschnitt. Berechnen Sie für diese Anzahlen das Verhältnis der Stichprobenvarianz zum Stichprobenmittelwert. Welches Verhältnis sagt ein Poisson-Modell vorher? Welche Wörter weichen am stärksten davon ab?
    1. Passen Sie mit der Maximum-Likelihood-Methode für jedes Wort ein Poisson-Modell und ein negatives Binomialmodell (Aufgabe :exref:`ex_mdl-distributions-gamma-poisson-mixture`b) an die Anzahlen an. Vergleichen Sie die maximierten Log-Likelihoods. Parametrisieren Sie die negative Binomialverteilung durch ihren Mittelwert $m$ und ihren Formparameter $\alpha$. Der Maximum-Likelihood-Schätzwert für $m$ ist der Stichprobenmittelwert. Daher benötigt nur $\alpha$ eine eindimensionale Suche, beispielsweise auf einem logarithmischen Gitter.
    1. Bestimmen Sie für jedes Wort den empirischen Anteil der Abschnitte, in denen es fehlt, sowie die empirische *Adaptation* $P(X\ge2\mid X\ge1)$. Vergleichen Sie beide Werte mit den Vorhersagen beider angepassten Modelle. Was misst die Adaptation und warum unterschätzt ein Poisson-Modell sie bei einer Bezeichnung wie „traveller“?
    1. Wiederholen Sie (a) für Abschnittslängen $L\in\{100,250,500,2000\}$. Nehmen Sie an, dass die Rate eines Wortes in einem Abschnitt das $L$-Fache einer Rate pro Wort ist, die aus einer Gammaverteilung mit festem Formparameter gezogen wird. Sagen Sie mithilfe von Aufgabe :exref:`ex_mdl-distributions-gamma-poisson-mixture`a vorher, wie das Verhältnis von Varianz zu Mittelwert mit $L$ wächst. Vergleichen Sie die Vorhersage mit Ihren Messungen.
   :exlabel:`ex_mdl-distributions-word-burstiness`

<!--en:b7d74877-->
1. **Die Student-$t$-Verteilung als Skalenmischung mit Gammaverteilung.** ● Mischen wir Gauß-Verteilungen mit gammaverteilter Präzision (inverser Varianz), fällt die Dichte nach außen langsamer ab als bei jeder Gauß-Verteilung. Es ergibt sich die Student-$t$-Verteilung, die :numref:`sec_mdl-statistics` für den Mittelwert einer kleinen Stichprobe verwendet. Sei $\tau\sim\mathrm{Gamma}(\nu/2,\nu/2)$ mit der Dichte :eqref:`eq_mdl-gamma_pdf` (Formparameter und Rate jeweils $\nu/2$ für ein $\nu>0$), und sei $X\mid\tau\sim\mathcal N(\mu,1/\tau)$.
    1. Integrieren Sie $\tau$ aus und zeigen Sie damit, dass $X$ die folgende Dichte hat:

       $$p(x)=\frac{\Gamma\bigl(\frac{\nu+1}{2}\bigr)}{\sqrt{\pi\nu}\,\Gamma\bigl(\frac{\nu}{2}\bigr)}\Bigl(1+\frac{(x-\mu)^2}{\nu}\Bigr)^{-(\nu+1)/2},$$

       Dies ist die *Student-$t$-Verteilung* mit $\nu$ Freiheitsgraden. *Hinweis:* Es gilt $\int_0^\infty\tau^{a-1}e^{-c\tau}\,d\tau=\Gamma(a)/c^a$ für $a,c>0$.
    1. Zeigen Sie, dass $p(x)$ wie $|x|^{-(\nu+1)}$ abfällt und daher $\mathbb{E}|X-\mu|^k$ nur für $k<\nu$ endlich ist. Welche Verteilung aus :numref:`sec_mdl-random_variables` ergibt sich für $\nu=1$?
    1. Zeigen Sie, dass $\textrm{Var}(X)=\nu/(\nu-2)$ für $\nu>2$ gilt. Verwenden Sie den Satz von der totalen Varianz,

       $$
       \textrm{Var}(X)=\mathbb{E}\bigl[\textrm{Var}(X\mid\tau)\bigr]+\textrm{Var}\bigl(\mathbb{E}[X\mid\tau]\bigr).
       $$

    1. [code] Ziehen Sie $10^5$ Werte für $\nu=3$ und $\mu=0$ nach dem Mischverfahren (zuerst $\tau$, dann $X$). Vergleichen Sie den Anteil mit $|X|>3$ mit dem Wert $0.0027$ der Gauß-Verteilung. Erzeugen Sie dann $10^4$ Stichprobenmittelwerte aus jeweils 20 Ziehungen und vergleichen Sie deren Standardabweichung mit $\sqrt{3/20}$.
   :exlabel:`ex_mdl-distributions-student-t`

<!--en:7be2d800-->
1. **Zwei Faktoren derselben Kovarianzmatrix.** ● Sei $\boldsymbol\Sigma=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$.
    1. Berechnen Sie von Hand den Cholesky-Faktor $\mathbf L$ und den Faktor $\mathbf M=\mathbf V\operatorname{diag}(\sqrt{\gamma_1},\sqrt{\gamma_2})$ aus den im Text bestimmten Eigenvektoren und Eigenwerten. Überprüfen Sie $\mathbf L\mathbf L^\top=\mathbf M\mathbf M^\top=\boldsymbol\Sigma$.
    1. Zeigen Sie, dass für je zwei invertierbare Faktoren mit $\mathbf M_1\mathbf M_1^\top=\mathbf M_2\mathbf M_2^\top$ die Beziehung $\mathbf M_2=\mathbf M_1\mathbf Q$ mit einer orthogonalen Matrix $\mathbf Q$ gilt (*Hinweis:* Setzen Sie $\mathbf Q=\mathbf M_1^{-1}\mathbf M_2$). Wie ist $\mathbf Q\mathbf z$ für $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$ verteilt? Folgern Sie, dass $\boldsymbol\mu+\mathbf M_1\mathbf z$ und $\boldsymbol\mu+\mathbf M_2\mathbf z$ dieselbe Verteilung haben.
    1. *Whitening.* Zeigen Sie, dass $\mathbf z=\mathbf L^{-1}(\mathbf x-\boldsymbol\mu)$ die Kovarianzmatrix $\mathbf I$ besitzt, wenn $\mathbf x\sim\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ gilt. Erklären Sie, warum die Berechnung von $\mathbf z$ für ein einzelnes $\mathbf x$ etwa $d^2$ Operationen erfordert, sobald $\mathbf L$ bekannt ist. Dagegen sind etwa $d^3/3$ Operationen nötig, um $\mathbf L$ zu berechnen.
    1. [code] Ziehen Sie mit jedem Faktor $10{,}000$ Punkte und geben Sie beide empirischen Kovarianzmatrizen aus. Wenden Sie auf die Cholesky-Ziehungen Whitening an, indem Sie ein lineares Gleichungssystem mit Dreiecksmatrix lösen. Geben Sie die Kovarianzmatrix des Ergebnisses aus.
   :exlabel:`ex_mdl-distributions-two-factors`

<!--en:a7502904-->
1. **Überlagerung, Ausdünnung und Reihenfolge der Ankünfte.** ● Zwei unabhängige Quellen senden Anfragen an einen Server. Es handelt sich um Poisson-Prozesse mit den Raten $\lambda_1$ und $\lambda_2$.
    1. Zeigen Sie, dass der zusammengeführte Anfragestrom ein Poisson-Prozess mit Rate $\lambda_1+\lambda_2$ ist. Verwenden Sie die Unterteilung in Zeitintervalle oder zeigen Sie mit dem binomischen Lehrsatz, dass die Summe unabhängiger Anzahlen mit den Verteilungen $\mathrm{Poisson}(\lambda_1)$ und $\mathrm{Poisson}(\lambda_2)$ der Verteilung $\mathrm{Poisson}(\lambda_1+\lambda_2)$ folgt.
    1. Jede Anfrage eines Prozesses mit Rate $\lambda$ wird unabhängig mit Wahrscheinlichkeit $q$ protokolliert. Zeigen Sie, dass die protokollierten Anfragen einen Poisson-Prozess mit Rate $q\lambda$ bilden. Konditionieren Sie dazu auf die Anzahl der Anfragen und mitteln Sie die Binomialwahrscheinlichkeiten über die Poisson-Verteilung. Wie wahrscheinlich ist es bei $\lambda=4$ pro Minute und $q=\tfrac14$, dass in einer Minute keine Anfrage protokolliert wird?
    1. Zeigen Sie unter der Bedingung, dass genau $n$ Ereignisse in $[0,t]$ liegen, dass ihre Zeitpunkte dieselbe Verteilung haben wie $n$ unabhängige, anschließend sortierte Ziehungen aus $U(0,t)$. Verwenden Sie die Unterteilung in Zeitintervalle: Jedes Teilintervall enthält mit derselben Wahrscheinlichkeit ein Ereignis.
    1. [code] Simulieren Sie einen Prozess mit Rate $4$ über $1{,}000$ Minuten mithilfe exponentialverteilter Zeitabstände. Dünnen Sie ihn mit $q=\tfrac14$ aus und prüfen Sie, ob Mittelwert und Varianz der verbleibenden Anzahlen pro Minute nahe bei $1$ liegen. Sammeln Sie für Minuten mit genau drei Ereignissen deren Zeitpunkte relativ zum Minutenbeginn und prüfen Sie, ob das Histogramm flach ist.
   :exlabel:`ex_mdl-distributions-poisson-process`

<!--en:d9980e62-->
1. **Kovarianzmatrix der Multinomialverteilung.** ● Schreiben Sie den Vektor der Anzahlen als $\mathbf x=\sum_{i=1}^n\mathbf e_{Y_i}$, also als Summe von $n$ unabhängigen One-Hot-Vektoren mit $P(Y_i=k)=p_k$.
    1. Zeigen Sie, dass $\mathbb{E}[\mathbf e_Y]=\mathbf p$ und $\textrm{Cov}(\mathbf e_Y)=\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ gelten, und folgern Sie daraus $\mathbb{E}[\mathbf x]=n\mathbf p$ und $\textrm{Cov}(\mathbf x)=n\bigl(\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top\bigr)$. Lesen Sie $\textrm{Var}(x_k)=np_k(1-p_k)$ und $\textrm{Cov}(x_j,x_k)=-np_jp_k$ für $j\ne k$ ab. Warum ist die Kovarianz für jedes Paar verschiedener Klassen negativ?
    1. Zeigen Sie $\textrm{Cov}(\mathbf x)\mathbf 1=\mathbf 0$ und erklären Sie das Ergebnis mit der Nebenbedingung $\sum_kx_k=n$.
    1. Berechnen Sie für 12 Würfe mit einem fairen Würfel $\textrm{Var}(x_k)$, $\textrm{Cov}(x_j,x_k)$ und die Korrelation zwischen zwei verschiedenen Anzahlen.
    1. Zeigen Sie, dass $x_k$ für sich genommen der Verteilung $\mathrm{Binomial}(n,p_k)$ folgt, indem Sie die übrigen $K-1$ Klassen zu einer einzigen zusammenfassen.
   :exlabel:`ex_mdl-distributions-multinomial-covariance`

<!--en:8086baee-->
[Diskussion](https://d2l.discourse.group/t/417)

<!--en:b338af67-->
[Diskussion](https://d2l.discourse.group/t/1098)

<!--en:be82cb5b-->
[Diskussion](https://d2l.discourse.group/t/1099)

<!--en:be82cb5b-2-->
[Diskussion](https://d2l.discourse.group/t/1099)

<!--en:2f695721-->
Häufige Wahrscheinlichkeitsverteilungen: vom Münzwurf zur multivariaten Gauß-Verteilung<br>**Bernoulli- und Binomialverteilung · Poisson- und Exponentialverteilung · Kategorialverteilung und Softmax-Funktion · Gauß-Verteilung und zentraler Grenzwertsatz · Laplace-Verteilung · multivariate Gauß-Verteilung und Konditionierung**

<!--en:9b8b90e5-->
::: {.slide title="Die Wahrscheinlichkeitsverteilung hinter dem Trainingsverlust"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:be0e89da-->
- **Verlust = negative Log-Likelihood:** Jedes Netz zur Vorhersage wählt implizit eine Verteilung $p(y \mid \hat{y})$ für seine Zielgrößen und minimiert $-\log p(y \mid \hat{y})$.

<!--en:a8f5d9d7-->
- **Vier typische Paare:** Bernoulli $\to$ binäre Kreuzentropie, Kategorialverteilung $\to$ Mehrklassen-Kreuzentropie, Gauß $\to$ quadratischer Fehler ($L_2$), Laplace $\to$ absoluter Fehler ($L_1$).

<!--en:a83a377c-->
- **Über die Punktvorhersage hinaus:** Eine falsch angesetzte Varianz (z. B. im Poisson-Modell für gehäuft auftretende Ereignisse) kann zu gefährlich engen Konfidenzintervallen führen.

<!--en:9ffb2dc0-->
::: {.slide title="Ein Münzwurf als Ursprung aller Verteilungen"}

<!--en:fd45406e-->
[Wie die Verteilungen auseinander hervorgehen]{.kicker}

<!--en:a7d556f8-->
- Alle Verteilungen dieses Abschnitts lassen sich aus einem einzigen Münzwurf mit Verteilung $\mathrm{Bernoulli}(p)$ konstruieren:
  - **Endliche Summe von $n$ Würfen:** $\mathrm{Binomial}(n, p)$.
  - **Grenzwert für seltene Ereignisse ($n \to \infty, np \to \lambda$):** $\mathrm{Poisson}(\lambda)$.
  - **Wartezeit im Poisson-Prozess:** $\mathrm{Exp}(\lambda)$ ($1$. Ereignis) und $\mathrm{Gamma}(k, \lambda)$ ($k$. Ereignis).
  - **Zufälliges Vorzeichen $\times$ exponentialverteilte Größe:** $\mathrm{Laplace}(\mu, b)$.
  - **Standardisierte Summe ($n \to \infty$):** $\mathcal{N}(\mu, \sigma^2)$ (zentraler Grenzwertsatz) und korrelierter Vektor $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.

<!--en:cba8f502-->
::: {.slide title="Vier diskrete Verteilungen anhand unserer Beispiele"}

<!--en:b03fd427-->
[Diskrete Wahrscheinlichkeitsfunktionen]{.kicker}

<!--en:1a809f7e-->
- Eine diskrete Wahrscheinlichkeitsfunktion ordnet abzählbar vielen Ergebnissen die Massen $P(X = k) \ge 0$ mit $\sum_k P(X = k) = 1$ zu:
  1. **$\mathrm{Bernoulli}(9/13)$:** Ein Münzwurf mit den beobachteten relativen Häufigkeiten ($0.692$ gegenüber $0.308$).
  2. **$\mathrm{Binomial}(13, 1/2)$:** Anzahl der Kopfwürfe bei $13$ fairen Würfen, Maxima bei $6$ und $7$ ($\mu = 6.5$).
  3. **$\mathrm{Poisson}(4)$:** Webserveranfragen pro Minute mit Rate $\lambda = 4$, nach oben unbeschränkt auf $\{0, 1, 2, \dots\}$.
  4. **$\mathrm{Categorical}(\mathbf{p})$:** Softmax-Verteilung über vier Klassen zu den Logits $\mathbf{z} = (2, 1, 0.1, -1)$.

<!--en:cae9762b-->
::: {.slide title="Die Bernoulli-Verteilung für einen binären Versuch"}

<!--en:de672da8-->
[Ein einzelnes Ergebnis, 0 oder 1]{.kicker}

<!--en:8c815701-->
- **Momente aus $X^2 = X$:** $\mathbb{E}[X] = p$ und $\mathrm{Var}(X) = \mathbb{E}[X^2] - p^2 = p(1 - p)$.

<!--en:a3f3d9ce-->
- **Varianzparabel:** $\mathrm{Var}(X) = p(1-p)$ verschwindet für $p \in \{0, 1\}$ und erreicht bei einer fairen Münze ihr Maximum **$1/4 = 0.25$** ($p = 1/2$), gegenüber $0.213$ bei $p = 9/13$.

<!--en:2abed7fc-->
::: {.slide title="Die Binomialverteilung zählt Kopfwürfe bei n Würfen"}

<!--en:bfc2813e-->
[Summe unabhängiger Bernoulli-Versuche]{.kicker}

<!--en:6c7af69d-->
- **Kombinatorik:** $\binom{n}{k}$ Reihenfolgen von $k$ Kopfwürfen und $n-k$ Zahlwürfen, jede mit Wahrscheinlichkeit $p^k(1-p)^{n-k}$.

<!--en:10f63352-->
- **Momente aus $X = \sum_{i=1}^n X_i$:** $\mathbb{E}[X] = np$ und wegen der Unabhängigkeit $\mathrm{Var}(X) = np(1 - p)$. Wären alle $13$ Würfe Kopien eines einzigen fairen Wurfs, stiege die Varianz um den Faktor $13\times$ von $3.25$ auf $42.25$.

<!--en:ea1cb172-->
::: {.slide title="Binomialwahrscheinlichkeiten über Quotienten ohne Fakultäten berechnen"}

<!--en:bca767b1-->
[Numerische Stabilität und Unimodalität]{.kicker}

<!--en:ff27ced0-->
Die Berechnung von $n!$ führt bei 64-Bit-Gleitkommazahlen für $n > 170$ zum Überlauf. Im Quotienten aufeinanderfolgender Wahrscheinlichkeiten kürzen sich alle Fakultäten heraus:

<!--en:f88bc5b9-->
- **Kombinatorik und Maximum:** Bei $k$ Kopfwürfen kann jeder der $n - k$ Zahlwürfe zu Kopf werden. Jedes Ergebnis entsteht auf $k + 1$ Arten. Der Quotient übersteigt $1$ genau dann, wenn $k + 1 < (n + 1)p$ gilt. Die Binomialverteilung steigt also bis zu einem Maximum nahe $np$ an.

<!--en:5b7de97d-->
::: {.slide title="Binomialwahrscheinlichkeiten über Quotienten ohne Fakultäten berechnen (Fortsetzung)"}

<!--en:bca767b1-2-->
[Numerische Stabilität und Unimodalität]{.kicker}

<!--en:11435805-->
::: {.slide title="Das Gesetz der seltenen Ereignisse: von Binomial zu Poisson"}

<!--en:3d389bf4-->
[Unbeschränkte Ereignisanzahl in kontinuierlicher Zeit]{.kicker}

<!--en:695c476e-->
- **Unterteilung der Zeit:** Wir teilen eine Minute in $n$ kurze Intervalle mit Ankunftswahrscheinlichkeit $p_n = \lambda/n$. Für $n \to \infty$ gilt $(1 - \lambda/n)^n \to e^{-\lambda}$ und $\binom{n}{k}n^{-k} \to 1/k!$.

<!--en:8b94319b-->
- **Mittelwert gleich Varianz:** Für $X \sim \mathrm{Poisson}(\lambda)$ gilt $\mathbb{E}[X] = \mathrm{Var}(X) = \lambda$.

<!--en:cb872ee5-->
::: {.slide title="Konvergenz gegen die Poisson-Verteilung und Überdispersion"}

<!--en:557274fa-->
[Konvergenzgeschwindigkeit und Modelldiagnose]{.kicker}

<!--en:c4d45977-->
- **Konvergenzgeschwindigkeit:** Der Totalvariationsabstand $\frac{1}{2}\sum_k |P_{\mathrm{Bin}}(k) - P_{\mathrm{Poi}}(k)|$ fällt wie $O(1/n)$ ($0.056$ bei $n=20$ $\to$ $0.010$ bei $n=100$ $\to$ $0.001$ bei $n=1000$).

<!--en:781a7b13-->
- **Serveranfragen bei $\lambda = 4\text{ req/min}$:**
  - Minute ohne Anfrage: $P(X = 0) = e^{-4} \approx 0.0183$.
  - Anzahl am Maximum: $P(X = 4) = \frac{4^4}{4!}e^{-4} \approx 0.195$.
  - Anfrageschub ($\ge 10$): $P(X \ge 10) \approx 0.008$.

<!--en:45b7e73b-->
- **Überdispersion erkennen:** Übersteigt die empirische Varianz der Anzahlen den Mittelwert deutlich ($\mathrm{Var} \gg \mathbb{E}$), schwankt die Rate oder Ereignisse häufen sich. Hier eignet sich eine Gamma-Poisson-Mischung, also eine negative Binomialverteilung.

<!--en:3e8061e0-->
::: {.slide title="Kategorialverteilung, Softmax-Funktion und Multinomialverteilung"}

<!--en:3b9f66dc-->
[Mehrklassen-Labels und Tokenanzahlen]{.kicker}

<!--en:475605e7-->
Ein Klassifikator mit $K$ Klassen oder ein Sprachmodell verallgemeinert die Bernoulli-Münze auf $K$ Ergebnisse:

<!--en:a13991bc-->
- **Kategorialverteilung und Softmax ($n = 1$ Ziehung):** Die Softmax-Funktion bildet unbeschränkte Logits $\mathbf{z} \in \mathbb{R}^K$ auf Wahrscheinlichkeiten $P(X = k) = p_k$ ab:
  $$p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \qquad -\log P(\mathbf{y}) = -\sum_{k=1}^K y_k \log p_k$$

<!--en:7b1f2f5f-->
- **Diskrete Gleichverteilung ($\mathrm{Unif}\{1, \dots, n\}$):** Für $p_k = 1/n$ sind Mittelwert und Varianz $\frac{n+1}{2}$ bzw. $\frac{n^2 - 1}{12}$ (beim Würfel $3.5$ und $\frac{35}{12}$).

<!--en:9ac16832-->
- **Multinomialverteilung ($n$ unabhängige Ziehungen):** Der Vektor der Anzahlen $\mathbf{x} = (x_1, \dots, x_K)$ mit $\sum_k x_k = n$ hat die Wahrscheinlichkeitsfunktion $\frac{n!}{x_1!\cdots x_K!}\prod_{k=1}^K p_k^{x_k}$, den Mittelwert $n\mathbf{p}$ und die Kovarianzmatrix $n(\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top)$ (Einträge außerhalb der Diagonalen: $-np_j p_k < 0$).

<!--en:b6cea9aa-->
Wir werten die Softmax-Funktion für die Logits $\mathbf{z} = (2, 1, 0.1, -1)$ aus und überprüfen die relativen Häufigkeiten anhand von $10{,}000$ Ziehungen:

<!--en:fa812af6-->
::: {.slide title="Der Poisson-Prozess und exponentialverteilte Wartezeiten"}

<!--en:750d1371-->
[Ereignisanzahlen in Zeitfenstern und Zeitabstände zwischen Ereignissen]{.kicker}

<!--en:a9f0bcf8-->
- In einem **Poisson-Prozess** in kontinuierlicher Zeit mit Rate $\lambda$ hat die Ereignisanzahl in jedem Fenster der Länge $t$ die Verteilung $N(t) \sim \mathrm{Poisson}(\lambda t)$.

<!--en:5d6b5cb8-->
- **Die grundlegende Dualität:** Die Wartezeit $T$ bis zur ersten Anfrage übersteigt $t$ *genau dann*, wenn in $[0, t]$ keine Anfrage eingeht:
  $$P(T > t) = P\bigl(N(t) = 0\bigr) = e^{-\lambda t}, \quad t \ge 0$$

<!--en:bfa4e218-->
- Ableiten von $F(t) = 1 - e^{-\lambda t}$ liefert die **Exponentialverteilung** $T \sim \mathrm{Exp}(\lambda)$:
  $$p(t) = \lambda e^{-\lambda t} \quad (t \ge 0), \qquad \mathbb{E}[T] = \frac{1}{\lambda}, \qquad \mathrm{Var}(T) = \frac{1}{\lambda^2}$$

<!--en:d5e3c356-->
- Bei $\lambda = 4\text{/min}$ beträgt die mittlere Wartezeit $15\text{ s}$, der Median der Wartezeit $F^{-1}(0.5) = \frac{\log 2}{\lambda}$ beträgt $10.4\text{ s}$.

<!--en:1ac8de18-->
::: {.slide title="Gedächtnislosigkeit, gammaverteilte Wartezeiten und Sampling"}

<!--en:cf2d394f-->
[Überprüfung der Ereignisfolge in kontinuierlicher Zeit]{.kicker}

<!--en:456df775-->
- **Gedächtnislosigkeit:** Die bisherige Wartezeit $s$ verrät nichts über die verbleibende Wartezeit $t$:
  $$P(T > s + t \mid T > s) = \frac{P(T > s + t)}{P(T > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(T > t)$$

<!--en:16ff41d5-->
- **Zeitpunkt der $k$-ten Ankunft ($T_k \sim \mathrm{Gamma}(k, \lambda)$):** Wegen $T_k > t \iff N(t) \le k - 1$ ergibt die Ableitung von $\sum_{j=0}^{k-1} e^{-\lambda t}\frac{(\lambda t)^j}{j!}$ eine Teleskopsumme. Übrig bleibt die **Gammadichte** $p(t) = \frac{\beta^\alpha t^{\alpha-1}e^{-\beta t}}{\Gamma(\alpha)}$ mit Mittelwert $\frac{\alpha}{\beta}$ und Varianz $\frac{\alpha}{\beta^2}$.

<!--en:b7421a70-->
- **Sampling mit der Inversionsmethode ($T = -\log(1 - U)/\lambda$):** Mit $1 - U$ anstelle von $U \in [0, 1)$ vermeiden wir $-\log(0) = \infty$:

<!--en:19f004aa-->
::: {.slide title="Die stetige Gleichverteilung"}

<!--en:27d66424-->
[Konstante Dichte und Momente]{.kicker}

<!--en:25a1fdc5-->
- **Varianz durch Zentrieren:** Verschieben wir $[a, b]$ auf $[-c, c]$ mit $c = (b-a)/2$, ergibt sich $\mathrm{Var}(X) = \int_{-c}^c \frac{y^2}{2c}\,dy = \frac{c^2}{3} = \frac{(b-a)^2}{12}$.

<!--en:45814418-->
- **Referenzwerte:** $U(0, 1)$ hat die Varianz $1/12$. Für $U(-\sqrt{3}, \sqrt{3})$ gelten der Mittelwert $0$ und die Einheitsvarianz $1$.

<!--en:29240674-->
::: {.slide title="Vom Quotienten der Binomialwahrscheinlichkeiten zur gaußschen Parabel"}

<!--en:6de538bf-->
[Herleitung nach de Moivre und Laplace]{.kicker}

<!--en:74154de4-->
Welche Form hat $\mathrm{Binomial}(n, p)$ nahe dem Mittelwert $np$? Setzen wir $q = 1 - p$ und beschreiben wir die Anzahl $k = np + x$ durch ihre Abweichung $x$ vom Mittelwert. Logarithmieren wir nun den Quotienten aufeinanderfolgender Wahrscheinlichkeiten mit der Näherung $\log(1 + \epsilon) \approx \epsilon$:

<!--en:35623e71-->
- **Linear veränderliche Steigung $\implies$ parabelförmige Log-Wahrscheinlichkeit:** Bei jedem Einheitsschritt in $x$ ändert sich $\log P$ um etwa $-\frac{x}{npq}$. Summieren wir die Schritte, erhalten wir näherungsweise die Parabel $\log P(X = np + x) \approx C - \frac{x^2}{2npq}$ oder $P(X = np + x) \approx \frac{1}{\sqrt{2\pi npq}}e^{-x^2/(2npq)}$:

<!--en:a23c295f-->
::: {.slide title="Normierung, Momente und quadratischer Verlust bei der Gauß-Verteilung"}

<!--en:3f337f25-->
[Integration in Polarkoordinaten]{.kicker}

<!--en:8e47c401-->
Zur Normierung der Standardnormaldichte $\phi(y) = \frac{1}{\sqrt{2\pi}}e^{-y^2/2}$ berechnen wir $I = \int_{-\infty}^\infty e^{-x^2/2}\,dx$. Sein Quadrat ist ein Doppelintegral. Darin wechseln wir zu Polarkoordinaten $(x, y) = (r\cos\theta, r\sin\theta)$ mit $dx\,dy = r\,dr\,d\theta$:

<!--en:44758ccc-->
- **Einheitsvarianz durch partielle Integration ($u = x, dv = xe^{-x^2/2}dx$):**
  $$\int_{-\infty}^{\infty} x^2 \phi(x)\,dx = \frac{1}{\sqrt{2\pi}}\left(\bigl[-xe^{-x^2/2}\bigr]_{-\infty}^{\infty} + \int_{-\infty}^{\infty} e^{-x^2/2}\,dx\right) = 0 + 1 = 1$$

<!--en:35df7c38-->
- **Allgemeine Gauß-Verteilung $X = \mu + \sigma Y \sim \mathcal{N}(\mu, \sigma^2)$:** Variablentransformation ergibt $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$. Die negative Log-Likelihood $-\log p(y \mid \hat{y}) = \frac{(y - \hat{y})^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)$ entspricht dem **quadratischen Fehler als Verlust**.

<!--en:2b65336f-->
::: {.slide title="Der zentrale Grenzwertsatz in Aktion"}

<!--en:bc50bf4c-->
[Universelle Konvergenz standardisierter Summen]{.kicker}

<!--en:fe29a608-->
- **Zentraler Grenzwertsatz:** Für *jede* Folge unabhängiger, identisch verteilter Zufallsvariablen $X_1, X_2, \dots$ mit Mittelwert $\mu$ und endlicher Varianz $0 < \sigma^2 < \infty$ konvergiert die Summe $S_n = \sum_{i=1}^n X_i$ nach Standardisierung in Verteilung gegen $\mathcal{N}(0, 1)$:
  $$\lim_{n \to \infty} P\!\left(\frac{S_n - n\mu}{\sigma\sqrt{n}} \le y\right) = \Phi(y) = \int_{-\infty}^y \phi(u)\,du$$

<!--en:cd32a8f3-->
- **Von der Gleich- zur Gauß-Verteilung:** Die Dichte einer einzelnen Ziehung aus $U(0, 1)$ ($n=1$) ist ein Rechteck, bei $n=2$ ergibt sich ein Dreieck. Bei $n=8$ und $n=32$ ist die standardisierte Summe optisch kaum noch von $\phi(y)$ zu unterscheiden.

<!--en:5fb95017-->
- **Warum die Varianz endlich sein muss:** Mittelwerte von Cauchy-verteilten Ziehungen ($\sigma^2 = \infty$) konzentrieren sich auch bei vielen Ziehungen nicht.

<!--en:8d930244-->
::: {.slide title="Laplace-Verteilung, L1-Verlust und Wahrscheinlichkeit extremer Werte"}

<!--en:c946b0c7-->
[Zufälliges Vorzeichen mal exponentialverteilte Größe]{.kicker}

<!--en:4cce817e-->
- **Konstruktion:** Multiplizieren wir ein unabhängiges, gleich wahrscheinliches Vorzeichen $S \in \{\pm 1\}$ mit einer exponentialverteilten Größe $E \sim \mathrm{Exp}(1/b)$ und verschieben das Ergebnis um $\mu$, erhalten wir $X = \mu + SE \sim \mathrm{Laplace}(\mu, b)$:
  $$p(x) = \frac{1}{2b}\exp\!\left(-\frac{|x - \mu|}{b}\right)$$

<!--en:ec09f75f-->
- **Momente aus $X = \mu + SE$:** Wegen $S^2 = 1$ und $\mathbb{E}[S] = 0$ gilt $\mathbb{E}[X] = \mu$ und $\mathrm{Var}(X) = \mathbb{E}[E^2] = 2b^2$.

<!--en:f5c419c5-->
- **Rolle im Deep Learning:**
  - Die negative Log-Dichte liefert den **$L_1$-Verlust (MAE)** $|y - \hat{y}|/b$. Sein Minimum liegt beim ausreißerrobusten Stichprobenmedian.
  - Als A-priori-Verteilung für Gewichte liefert sie die **$\ell_1$-Regularisierung des Lasso** $\|\mathbf{w}\|_1$.

<!--en:9c7f896b-->
::: {.slide title="Extreme Werte bei Laplace- und Gauß-Verteilung in der Simulation"}

<!--en:a7c77eb4-->
[Fünfmal so viel Wahrscheinlichkeit jenseits von 3σ]{.kicker}

<!--en:bfbc0379-->
Bei gleicher Varianz $\sigma^2 = 2b^2$ fällt die Laplace-Dichte wie $e^{-|x|/\sigma \cdot \sqrt{2}}$ ab, die Gauß-Dichte wie $e^{-x^2/(2\sigma^2)}$. Für die Laplace-Verteilung liegen $P(|X - \mu| > 3\sigma) = e^{-3\sqrt{2}} \approx 1.43\%$ jenseits von $3\sigma$, gegenüber $0.27\%$ bei der Gauß-Verteilung (Faktor $5.3\times$):

<!--en:9164cba8-->
**Warum der quadratische Verlust empfindlich auf Ausreißer reagiert:** Bei einem Ausreißer von $10\sigma$ beträgt der Strafterm der negativen Gauß-Log-Likelihood $\frac{1}{2}(10)^2 = 50$. Bei der Laplace-Verteilung beträgt er $10\sqrt{2} \approx 14.1$ mit beschränktem, stückweise konstantem Gradienten $\pm 1/b$.

<!--en:44c3b3bf-->
::: {.slide title="Die multivariate Gauß-Verteilung und ihre Ellipsoide"}

<!--en:ecbd6f93-->
[Geometrie korrelierter Zufallsvektoren]{.kicker}

<!--en:be38eb99-->
- Ein Vektor $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ in $\mathbb{R}^d$ mit symmetrischer, positiv definiter Kovarianzmatrix $\boldsymbol{\Sigma} \succ \mathbf{0}$ hat die Dichte:
  $$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

<!--en:b106d2f5-->
- **Spektralzerlegung:** Schreiben wir $\boldsymbol{\Sigma} = \mathbf{V}\mathrm{diag}(\gamma_1, \dots, \gamma_d)\mathbf{V}^\top$ und drehen das Koordinatensystem auf $\mathbf{u} = \mathbf{V}^\top(\mathbf{x} - \boldsymbol{\mu})$, zerfällt die quadratische Form im Exponenten in Beiträge unabhängiger eindimensionaler Gauß-Verteilungen:
  $$(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) = \sum_{i=1}^d \frac{u_i^2}{\gamma_i}$$

<!--en:32141a56-->
- **Unser Beispiel $\boldsymbol{\Sigma} = \begin{pmatrix}2&1\\1&2\end{pmatrix}$:** Die Eigenvektoren $\mathbf{v}_{1,2} = \frac{1}{\sqrt{2}}(1, \pm 1)^\top$ haben die Eigenwerte $\gamma_1 = 3, \gamma_2 = 1$ (um $45^\circ$ gedrehte Ellipse mit Halbachsen $\propto \sqrt{3} : 1$).

<!--en:a6967033-->
::: {.slide title="Lineare Abbildungen und Cholesky-Reparametrisierung"}

<!--en:35e909a0-->
[Abgeschlossenheit unter affinen Abbildungen und Sampling korrelierter Vektoren]{.kicker}

<!--en:8194fac0-->
**Proposition (Abgeschlossenheit unter affinen Abbildungen):** Ist $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ in $\mathbb{R}^d$ und hat $\mathbf{A} \in \mathbb{R}^{m \times d}$ vollen Zeilenrang $m \le d$, dann gilt

<!--en:b9c84239-->
- **Sampling mit der Cholesky-Zerlegung ($\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$):** Wir ziehen standardnormalverteiltes Rauschen $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ und setzen $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$. Dabei ist $\mathbf{L}$ eine untere Dreiecksmatrix ($\mathbf{L}\mathbf{I}\mathbf{L}^\top = \boldsymbol{\Sigma}$). Für $\boldsymbol{\Sigma} = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ gilt $\mathbf{L} = \begin{pmatrix}\sqrt{2} & 0 \\ 1/\sqrt{2} & \sqrt{3/2}\end{pmatrix}$.

<!--en:b91e9ff2-->
- **Der Reparametrisierungstrick für Variationsautoencoder:** Da $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$ nach $\boldsymbol{\mu}$ und $\mathbf{L}$ bei festem $\mathbf{z}$ differenzierbar ist, lassen sich Gradienten durch stochastische gaußsche Schichten zurückpropagieren.

<!--en:5ee5b552-->
::: {.slide title="Konditionierung einer Gauß-Verteilung als lineare Regression"}

<!--en:119817a3-->
[Orthogonales Residuum und Schur-Komplement]{.kicker}

<!--en:c90005d8-->
Wir zerlegen $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2)$ mit Mittelwert $(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)$ und Kovarianzmatrix in Blockform $\boldsymbol{\Sigma} = \begin{pmatrix}\boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22}\end{pmatrix}$. Die Regressionsmatrix $\mathbf{B} = \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}$ wählen wir so, dass das Residuum $\mathbf{r} = \mathbf{x}_1 - \mathbf{B}\mathbf{x}_2$ unkorreliert mit $\mathbf{x}_2$ ist:

<!--en:c7847108-->
- Wegen $\mathbf{x}_1 = \mathbf{r} + \mathbf{B}\mathbf{x}_2$ und $\mathbf{r} \perp \mathbf{x}_2$ verschiebt die Konditionierung auf $\mathbf{x}_2$ das Residuum $\mathbf{r}$ lediglich um $\mathbf{B}\mathbf{x}_2$. Wir erhalten $\mathbf{x}_1 \mid \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1\mid 2}, \boldsymbol{\Sigma}_{1\mid 2})$ mit
  $$\boldsymbol{\mu}_{1\mid 2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2), \qquad \boldsymbol{\Sigma}_{1\mid 2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

<!--en:672eaee2-->
- **Unser Beispiel ($x_2 = 2$, $B = 1/2$):** $x_1 \mid x_2 = 2 \sim \mathcal{N}\!\left(0 + \frac{1}{2}(2),\ 2 - \frac{1^2}{2}\right) = \mathcal{N}\!\left(1,\ \frac{3}{2}\right)$:

<!--en:2af70ca3-->
::: {.slide title="Verständnisfragen zu Poisson-Wartezeiten, zentralem Grenzwertsatz und Konditionierung"}

<!--en:929b7671-->
[Überprüfen Sie Ihr Verständnis]{.kicker}

<!--en:10d327ba-->
Überprüfen Sie Ihre Intuition anhand dreier Rechnungen zu den behandelten Verteilungen:

<!--en:7d8a7109-->
1. **Poisson-verteilte Anzahlen und exponentialverteilte Wartezeiten:** Anfragen treffen nach einem Poisson-Prozess mit Rate $\lambda = 4$ pro Minute ein. Wie wahrscheinlich ist eine Minute ohne Anfrage ($0$ Anfragen)? Wie wahrscheinlich ist eine Wartezeit von mehr als $30$ Sekunden ($t = 0.5\text{ min}$) bis zur nächsten Anfrage?

<!--en:77c42351-->
2. **Konditionierung einer bivariaten Gauß-Verteilung:** Sei $(X_1, X_2)$ gemeinsam gaußverteilt mit Mittelwert $\mathbf{0}$, Einheitsvarianzen $\Sigma_{11} = \Sigma_{22} = 1$ und Korrelation $\rho \in (-1, 1)$. Wie lautet die bedingte Verteilung $X_1 \mid X_2 = x_2$?

<!--en:cb72c487-->
3. **Warum die gemeinsame Gauß-Verteilung wichtig ist:** Gilt $Z \sim \mathcal{N}(0, 1)$ und ist $S \in \{\pm 1\}$ ein gleich wahrscheinliches Zufallsvorzeichen, dann haben $Z$ und $Y = SZ$ jeweils die Randverteilung $\mathcal{N}(0, 1)$ mit $\mathrm{Cov}(Y, Z) = 0$. Warum folgt $Y \mid Z = 2$ nicht der Verteilung $\mathcal{N}(0, 1)$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:276a51a7-->
1. $P(N(1) = 0) = \mathbf{e^{-4} \approx 0.0183}$. Eine Wartezeit von mehr als $t = 0.5\text{ min}$ entspricht dem Ereignis $N(0.5) = 0$ mit Wahrscheinlichkeit $e^{-4(0.5)} = \mathbf{e^{-2} \approx 0.1353}$.

<!--en:396a46e8-->
2. Hier gilt $B = \Sigma_{12}\Sigma_{22}^{-1} = \rho$ und damit **$X_1 \mid X_2 = x_2 \sim \mathcal{N}(\rho x_2,\ 1 - \rho^2)$**. Die bedingte Varianz $1 - \rho^2$ hängt nie vom beobachteten Wert $x_2$ ab.

<!--en:2ab5d2ab-->
3. $(Y, Z)$ ist **nicht gemeinsam gaußverteilt**. Die gesamte Masse liegt auf den beiden Geraden $y = \pm z$. Unter der Bedingung $Z = 2$ kann $Y$ nur die Werte $+2$ oder $-2$ annehmen, jeweils mit Wahrscheinlichkeit $1/2$.

<!--en:67b33aaf-->
::: {.slide title="Zusammenfassung: diskrete Wahrscheinlichkeitsverteilungen"}

<!--en:ef6e3502-->
[Das Wichtigste (Teil I)]{.kicker}

<!--en:943ea4fc-->
| Verteilung | Wahrscheinlichkeitsfunktion $P(X = k)$ | Mittelwert $\mathbb{E}[X]$ | Varianz $\mathrm{Var}(X)$ | Anwendung im Deep Learning und in der Modellierung |

<!--en:b9c61cb7-->
| **$\mathrm{Bernoulli}(p)$** | $p^x(1-p)^{1-x}$, $x \in \{0,1\}$ | $p$ | $p(1-p)$ | Binäre Klassifikation und binäre Kreuzentropie |

<!--en:48830141-->
| **$\mathrm{Binomial}(n, p)$** | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Richtige Vorhersagen auf einem Testdatensatz mit $n$ Beispielen |

<!--en:336d67b6-->
| **$\mathrm{Poisson}(\lambda)$** | $\lambda^k e^{-\lambda}/k!$, $k \ge 0$ | $\lambda$ | $\lambda$ | Anzahl seltener Ereignisse und Regression für Zähldaten |

<!--en:fd11187a-->
| **$\mathrm{Categorical}(\mathbf{p})$** | $p_k = e^{z_k}/\sum_j e^{z_j}$ | $\mathbf{p}$ (One-Hot) | $\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ | Softmax-Funktion für mehrere Klassen und Tokenvorhersage in großen Sprachmodellen |

<!--en:6ef7d960-->
| **$\mathrm{Multinomial}(n, \mathbf{p})$** | $\frac{n!}{x_1!\cdots x_K!}\prod_{k=1}^K p_k^{x_k}$ | $n\mathbf{p}$ | $n(\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top)$ | Bag-of-Words-Histogramme und Anzahlen pro Klasse |

<!--en:6786601f-->
| **$\mathrm{Unif}\{1, \dots, n\}$** | $1/n$ | $\frac{n+1}{2}$ | $\frac{n^2 - 1}{12}$ | Zufällige Indexauswahl für Minibatches und zufälliges Mischen |

<!--en:57f100db-->
::: {.slide title="Zusammenfassung: stetige Wahrscheinlichkeitsverteilungen"}

<!--en:5327d7c2-->
[Das Wichtigste (Teil II)]{.kicker}

<!--en:3025ad6c-->
| Verteilung | Dichte $p(x)$ | Mittelwert $\mathbb{E}[X]$ | Varianz $\mathrm{Var}(X)$ | Anwendung im Deep Learning und in der Modellierung |

<!--en:2e59a659-->
| **$\mathrm{Exp}(\lambda)$** | $\lambda e^{-\lambda t}$ ($t \ge 0$) | $1/\lambda$ | $1/\lambda^2$ | Gedächtnislose Wartezeiten zwischen Ereignissen |

<!--en:812c730e-->
| **$\mathrm{Gamma}(\alpha, \beta)$** | $\frac{\beta^\alpha t^{\alpha-1}e^{-\beta t}}{\Gamma(\alpha)}$ ($t > 0$) | $\alpha/\beta$ | $\alpha/\beta^2$ | Summe von $\alpha$ Wartezeiten, konjugierte A-priori-Verteilung für Raten |

<!--en:3a1d254b-->
| **$\mathrm{Unif}(a, b)$** | $\frac{1}{b-a}$ auf $[a, b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Ausgangsverteilung für das Sampling mit der Inversionsmethode |

<!--en:64f79db8-->
| **$\mathcal{N}(\mu, \sigma^2)$** | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ | Grenzverteilung im zentralen Grenzwertsatz, quadratischer Fehler als Verlust und Initialisierung der Gewichte |

<!--en:b5df4e2b-->
| **$\mathrm{Laplace}(\mu, b)$** | $\frac{1}{2b}e^{-\|x-\mu\|/b}$ | $\mu$ | $2b^2$ | Robuster $L_1$-Verlust (MAE) und Sparsität durch $\ell_1$-Regularisierung beim Lasso |

<!--en:b93e8bfa-->
| **$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$** | $(2\pi)^{-d/2}\|\boldsymbol{\Sigma}\|^{-1/2}e^{-\frac{1}{2}\|\mathbf{x}-\boldsymbol{\mu}\|_{\boldsymbol{\Sigma}^{-1}}^2}$ | $\boldsymbol{\mu}$ | $\boldsymbol{\Sigma}$ | Cholesky-Reparametrisierung in Variationsautoencodern und Konditionierung von Gauß-Prozessen |
