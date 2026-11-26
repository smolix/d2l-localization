<!--en:56381601-->
# Gemeinsame Verteilungen und Transformationen

<!--en:2474da82-->
In :numref:`sec_mdl-random_variables` haben wir den Auftreffpunkt eines Dartpfeils auf einer Scheibe mit Radius $1$ durch eine einzige Zahl beschrieben. Sein Abstand $R$ vom Mittelpunkt hat die Dichte $2r$. Der Auftreffpunkt hat jedoch sowohl eine horizontale Koordinate $X$ als auch eine vertikale Koordinate $Y$. Die meisten Größen, die wir im maschinellen Lernen modellieren, treten in solchen Gruppen auf. Ein Bild besteht aus Tausenden von Pixeln, deren Werte voneinander abhängen. Beim überwachten Lernen gehört zu jeder Eingabe ein Zielwert. Die Dichte jeder einzelnen Variablen zu kennen reicht nicht aus, denn die einzelnen Dichten sagen nichts darüber aus, wie sich die Variablen gemeinsam verändern.

<!--en:15bbcff8-->
Zwei Fragen leiten uns durch diesen Abschnitt. Erstens: Was verrät eine Variable über eine andere? Trifft der Pfeil die Scheibe weit oben bei $Y=0.9$, kann $X$ höchstens $0.44$ von der Mittellinie entfernt sein. Die Beobachtung von $Y$ verrät uns also etwas über $X$. Überraschenderweise ist unsere beste Vorhersage für $X$ durch einen einzelnen Wert dennoch $0$, unabhängig vom Wert von $Y$. Auch die Kovarianz der beiden Koordinaten ist null. Abhängigkeit, Vorhersagbarkeit und Korrelation sind drei verschiedene Begriffe. Um sie auseinanderzuhalten, brauchen wir die gemeinsame Dichte des Paars. Zweitens: Was geschieht mit einer Dichte, wenn wir eine Zufallsvariable transformieren? Das passiert etwa, wenn eine Schicht eines neuronalen Netzes ihre Eingabe verarbeitet oder ein Stichprobenverfahren gleichverteiltes Rauschen umformt. Wie wir bereits gesehen haben, lässt sich der Abstand des Pfeils als $R=\sqrt U$ aus einer gleichverteilten Variablen $U$ ziehen. Dabei verwandelt die Quadratwurzel eine konstante Dichte in die lineare Dichte $2r$. Eine Transformation bewirkt also mehr, als die Achse einer Dichte neu zu beschriften.

<!--en:b591a8fe-->
Drei Ergebnisse dieses Abschnitts spielen im weiteren Verlauf des Buches eine besonders wichtige Rolle: der Satz von der totalen Varianz :eqref:`eq_mdl-tower`, der bedingte Erwartungswert als bester Prädiktor für den quadratischen Fehler :eqref:`eq_mdl-best_predictor` und die Formel für die Variablentransformation :eqref:`eq_mdl-cov_density`. Die Kovarianzmatrix und die geometrische Deutung der Korrelation als Kosinus können Sie beim ersten Lesen überspringen.

<!--en:e6724ecb-->
## Gemeinsame und bedingte Dichten

<!--en:a253ea56-->
Eine einzige Funktion mit zwei Argumenten enthält alle Informationen über ein Paar stetiger Zufallsvariablen. Definieren wir diese Funktion und leiten wir daraus drei wesentliche Größen ab: die Verteilung einer einzelnen Koordinate, ihre Verteilung nach Beobachtung der anderen Koordinate und ihren Mittelwert bei gegebener anderer Koordinate. Letzteren schätzt ein Regressionsmodell.

<!--en:1debf484-->
### Gemeinsame Dichten und Randdichten

<!--en:fcbb61a4-->
Eine eindimensionale Dichte misst auf kurzen Intervallen die Wahrscheinlichkeit pro Längeneinheit. Entsprechend misst eine zweidimensionale Dichte auf kleinen Quadraten die Wahrscheinlichkeit pro Flächeneinheit. Die *gemeinsame Dichte* (engl. joint density) $p_{X,Y}$ eines Paars $(X,Y)$ liefert näherungsweise die Wahrscheinlichkeit, in ein kleines Quadrat der Seitenlänge $\epsilon$ bei $(x,y)$ zu fallen. Sie ist das Produkt aus Fläche und Dichte,

<!--en:cd5b1892-->
Wie in einer Dimension zerlegen wir ein Gebiet $\mathcal D$ der Ebene in solche Quadrate. So gelangen wir zu der Eigenschaft, die uns als formale Definition dient: $p_{X,Y}\ge0$ und $P((X,Y)\in\mathcal{D})=\iint_{\mathcal{D}} p_{X,Y}(x,y)\,dx\,dy$. Über der gesamten Ebene beträgt das Volumen unter der gemeinsamen Dichte also $1$. Für $n$ Variablen gilt dieselbe Definition mit $n$-dimensionalen Quadern und $n$-fachen Integralen.

<!--en:ef023486-->
Auf unserer Dartscheibe sind gleich große Teilflächen der Einheitskreisscheibe gleich wahrscheinlich. Die Scheibe hat die Fläche $\pi$, also beträgt die gemeinsame Dichte im Inneren $p_{X,Y}(x,y)=1/\pi$ und außerhalb $0$. Obwohl die Dichte auf der Scheibe konstant ist, lässt sie sich nicht als Produkt einer Funktion von $x$ und einer Funktion von $y$ schreiben. Der kreisförmige Rand verknüpft die beiden Koordinaten.

<!--en:00669453-->
Oft kennen wir eine gemeinsame Dichte, interessieren uns aber nur für die Verteilung einer einzelnen Koordinate, ihre *Randverteilung* (engl. marginal distribution). Auf einen schmalen vertikalen Streifen $X\in[x,x+\epsilon]$ entfällt ungefähr die Wahrscheinlichkeit $\epsilon\,p_X(x)$. Zerlegen wir den Streifen entlang der $y$-Achse in kleine Quadrate und addieren deren Wahrscheinlichkeiten, so erhalten wir

<!--en:7811ed48-->
Wir kürzen einen Faktor $\epsilon$ und erkennen in der verbleibenden Summe eine Riemann-Summe in $y$. Damit erhalten wir

<!--en:07fa231a-->
In Worten erhalten wir die Randdichte an der Stelle $x$, indem wir die gemeinsame Dichte entlang des vertikalen Streifens bei $x$ integrieren, wie :numref:`fig_mdl-prob-marginal` zeigt. Die Variable, die wir nicht benötigen, wird *ausintegriert*. Auf der Dartscheibe reicht der vertikale Streifen bei $x$ von $-\sqrt{1-x^2}$ bis $\sqrt{1-x^2}$. Das ergibt

<!--en:04bde322-->
also eine Dichte in Form eines skalierten Halbkreises. Zur Kontrolle: $\int_{-1}^1\sqrt{1-x^2}\,dx$ ist die Fläche der halben Einheitskreisscheibe, also $\pi/2$. Das Integral von $p_X$ ergibt daher tatsächlich $1$. Obwohl die gemeinsame Dichte konstant ist, ist die Randdichte gekrümmt. Ein vertikaler Streifen durch die Mitte der Scheibe ist länger als ein Streifen nahe dem Rand.

<!--en:73138f70-->
![Die Höhenlinien zeigen eine stetige gemeinsame Dichte $p_{X,Y}(x,y)$. Der vertikale Streifen bei $x$ ist hervorgehoben. Das Integral der gemeinsamen Dichte in diesem Streifen über alle $y$ ergibt die Höhe der Randdichte $p_X(x)$, die unten entlang der $\mathit{x}$-Achse aufgetragen ist. Durch Verschieben des Streifens über alle Werte von $x$ ergibt sich die gesamte Randdichte.](../img/mdl-prob-marginal.svg)

<!--en:7a29a033-->
### Bedingte Dichten und Unabhängigkeit

<!--en:72fa3869-->
Angenommen, wir beobachten $Y=y$. Wie ändert sich dadurch die Dichte von $X$? Für diskrete Variablen lautet die Antwort $P(X=x\mid Y=y)=P(X=x,Y=y)/P(Y=y)$. Bei stetigen Dichten führt diese Formel jedoch auf null geteilt durch null, denn $P(Y=y)=0$. Hier hilft derselbe Grenzübergang wie bei der Definition der Dichte. Wir bedingen auf einen schmalen horizontalen Streifen $Y\in[y,y+\epsilon]$ mit positiver Wahrscheinlichkeit und lassen seine Breite gegen null gehen. Für den Zähler verwenden wir unsere Näherung mit Quadraten, für den Nenner die Regel für eine Variable. Das ergibt

<!--en:91106c9f-->
Innerhalb des Streifens ist die Wahrscheinlichkeit eines kurzen $x$-Intervalls gleich seiner Länge $\epsilon$ mal $p_{X,Y}(x,y)/p_Y(y)$. Dieser Quotient ist die Wahrscheinlichkeit pro Längeneinheit von $X$ unter der Bedingung $Y=y$. Wir definieren daher die *bedingte Dichte* (engl. conditional density) durch

<!--en:87c8fc54-->
Für jedes feste $y$ mit $p_Y(y)>0$ ist dies eine gültige Dichte in $x$. Sie ist nichtnegativ und ihr Integral ergibt nach der Formel für die Randdichte :eqref:`eq_mdl-marginal` den Wert $p_Y(y)/p_Y(y)=1$. Geometrisch schneiden wir die gemeinsame Dichte entlang der horizontalen Geraden $Y=y$ und normieren die Fläche unter dem Schnittprofil auf eins. Auf der Dartscheibe schneidet die horizontale Gerade $Y=y$ die Sehne $[-\sqrt{1-y^2},\sqrt{1-y^2}]$ aus. Somit gilt

<!--en:2d9b9396-->
Unter der Bedingung $Y=y$ ist die horizontale Position $X$ auf der Sehne in Höhe $y$ gleichverteilt. Die Sehne wird kürzer, wenn $|y|$ wächst. Die Beobachtung von $Y$ verändert also die Verteilung von $X$.

<!--en:edaa42ce-->
Durch Umstellen von :eqref:`eq_mdl-cond_density` erhalten wir die *Kettenregel* (engl. chain rule) $p_{X,Y}(x,y)=p_{X\mid Y}(x\mid y)\,p_Y(y)$. Schreiben wir die gemeinsame Dichte auf beide Arten, $p_{X\mid Y}\,p_Y=p_{Y\mid X}\,p_X$, und lösen nach $p_{X\mid Y}$ auf, so ergibt sich der *Satz von Bayes für Dichten*,

<!--en:123add04-->
Wie der Satz von Bayes für diskrete Variablen in :numref:`sec_prob` verwandelt diese Identität ein Modell für die Erzeugung von $Y$ aus $X$ in eine Verteilung von $X$ nach Beobachtung von $Y$.

<!--en:ba1fbb35-->
Im Sonderfall, dass uns die Beobachtung von $Y$ nichts Neues über $X$ verrät, gilt $p_{X\mid Y}(x\mid y)=p_X(x)$ für alle Werte mit $p_Y(y)>0$. Wir nennen die Variablen dann *unabhängig* (engl. independent) und schreiben $X\perp Y$. Nach der Kettenregel ist Unabhängigkeit gleichbedeutend damit, dass die gemeinsame Dichte faktorisiert, also $p_{X,Y}(x,y)=p_X(x)\,p_Y(y)$ gilt (bis auf Mengen der Wahrscheinlichkeit null). Für die eine Richtung multiplizieren wir mit $p_Y(y)$, für die andere dividieren wir durch denselben Faktor.

<!--en:f9d096af-->
Anschaulich bedeutet Unabhängigkeit, dass die bedingte Dichte von $X$ als Funktion von $x$ auf jeder Höhe $y$ dieselbe Lage und Form hat. Betrachten wir die gemeinsame Dichte $p_{X,Y}(x,y)=4xy$ auf dem Einheitsquadrat $[0,1]^2$. Ihre Randdichte ist $p_Y(y)=\int_0^1 4xy\,dx=2y$. Daraus folgt $p_{X\mid Y}(x\mid y)=4xy/(2y)=2x$ unabhängig von $y$, also $X\perp Y$. Beim Dartpfeil verhält es sich anders. Zwar ist die bedingte Verteilung auf jeder Höhe $y$ eine Gleichverteilung, doch die Länge der zugehörigen Sehne hängt von $y$ ab.

<!--en:7ed1524b-->
Wie :numref:`fig_mdl-prob-conditional-slice` zeigt, reicht die gleiche Form für Unabhängigkeit nicht aus. Bei einem korrelierten Paar haben die bedingten Dichten auf drei Höhen dieselbe Form, aber ihre Mittelpunkte sind gegeneinander verschoben. Das Paar ist also abhängig. Nur wenn die gemeinsame Dichte faktorisiert, fallen alle drei Schnittprofile zusammen.

<!--en:58117802-->
![Bedingte Dichten als Schnittprofile. Links eine gemeinsame Dichte mit drei horizontalen Schnitten bei $y_0=-1,0,1$. Rechts ergibt jedes auf die Fläche eins normierte Schnittprofil die bedingte Dichte $p(x\mid y_0)$. Oben ein korreliertes Paar. Die drei bedingten Dichten haben dieselbe Form, aber verschobene Mittelpunkte. Die bedingte Dichte hängt also von $y_0$ ab und die Variablen sind abhängig. Unten ein Paar mit faktorisierbarer gemeinsamer Dichte. Die drei bedingten Dichten fallen zusammen, wie es der Unabhängigkeit entspricht.](../img/mdl-prob-conditional-slice.svg)

<!--en:3f2fe5b1-->
Unabhängigkeit liefert auch eine einfache Formel für die Dichte einer Summe. Gilt $X\perp Y$, so hat das Paar $(X,X+Y)$ die gemeinsame Dichte $p_X(x)\,p_Y(s-x)$ an der Stelle $(x,s)$. Integrieren wir $x$ gemäß :eqref:`eq_mdl-marginal` aus, bleibt die *Faltung* (engl. convolution) $p_{X+Y}(s)=\int p_X(x)\,p_Y(s-x)\,dx$.

<!--en:10cd18d3-->
### Bedingter Erwartungswert

<!--en:582e3abf-->
Ein Regressionsmodell sagt aus einer Eingabe einen Zielwert vorher. Wie im vorigen Unterabschnitt bezeichnet $X$ den Zielwert und $Y$ die beobachtete Eingabe (im Haupttext ist es umgekehrt). Beim Training mit quadratischem Verlust suchen wir eine Funktion $h$, die $\mathbb{E}[(X-h(Y))^2]$ möglichst klein macht. Welche Funktion ist die beste? In jedem schmalen Streifen $Y\approx y$ muss das Modell für eine ganze Bandbreite möglicher Zielwerte eine einzige Zahl vorhersagen. Für jede Zufallsvariable $X$ ist die Konstante $c$ mit dem kleinsten mittleren quadratischen Abstand ihr Mittelwert $\mu_X$. Entwickeln wir nämlich das Quadrat um $\mu_X$, ergibt sich $\mathbb{E}[(X-c)^2]=\textrm{Var}(X)+(\mu_X-c)^2$. Es liegt also nahe, in jedem Streifen den Mittelwert von $X$ vorherzusagen. Entsprechend definieren wir den *bedingten Erwartungswert* (engl. conditional expectation) von $X$ unter der Bedingung $Y=y$ als Mittelwert der bedingten Verteilung,

<!--en:5a4ecc3d-->
Er ist der Mittelwert von $X$ über alle Ergebnisse mit $Y=y$. Variieren wir $y$, so beschreibt $\mathbb{E}[X\mid Y{=}y]$ eine Funktion von $y$. Werten wir sie an der Zufallsvariablen $Y$ aus, erhalten wir eine Zufallsvariable $\mathbb{E}[X\mid Y]$, die allein von $Y$ abhängt. Auf dieselbe Weise liefert die Mittelung innerhalb eines Streifens die Formel $\mathbb{E}[g(X,Y)\mid Y{=}y]=\int g(x,y)\,p_{X\mid Y}(x\mid y)\,dx$ für jede Funktion $g$. Da ein Faktor $h(Y)$ innerhalb jedes Streifens konstant ist, können wir ihn aus dem bedingten Erwartungswert herausziehen:

<!--en:2859c454-->
Entsprechend beschreibt die *bedingte Varianz* (engl. conditional variance) die Streuung von $X$ innerhalb eines Streifens. Für sie gilt $\textrm{Var}(X\mid Y)=\mathbb{E}[X^2\mid Y]-(\mathbb{E}[X\mid Y])^2$. Das ist die Rechenformel für die Varianz aus :numref:`sec_mdl-random_variables`, angewandt auf jeden einzelnen Streifen.

<!--en:f7bad574-->
Anschaulich sollte der Mittelwert der Streifenmittelwerte wieder den Gesamtmittelwert ergeben. Ebenso sollte sich die Gesamtvarianz in die Streuung innerhalb der Streifen und die Streuung zwischen ihnen zerlegen lassen. Beides trifft zu. Die zweite Aussage erweist sich als besonders nützlich.

<!--en:9035992c-->
**Satz (Satz vom totalen Erwartungswert und Satz von der totalen Varianz).** *Die Zufallsvariablen $X,Y$ besitzen eine gemeinsame Dichte und $X$ habe endliche Varianz (für diskrete Paare gelten dieselben Identitäten mit Summen anstelle der Integrale). Dann gilt*

<!--en:5037c2b1-->
**Beweis.** Für die erste Identität, die sogenannte *Turmeigenschaft* (engl. tower property), schreiben wir den inneren Erwartungswert als Integral. Mit der Kettenregel $p_{X\mid Y}(x\mid y)\,p_Y(y)=p_{X,Y}(x,y)$ setzen wir die Streifen wieder zur gemeinsamen Dichte zusammen:

<!--en:957b0169-->
Werte von $y$ mit $p_Y(y)=0$, für die die bedingte Dichte undefiniert ist, tragen zu keiner der beiden Seiten bei. Denn $Y$ nimmt solche Werte mit Wahrscheinlichkeit null an. Dieselbe Rechnung mit $g(x,y)$ anstelle von $x$ ergibt $\mathbb{E}\bigl[\mathbb{E}[g(X,Y)\mid Y]\bigr]=\mathbb{E}[g(X,Y)]$. Für die zweite Identität schreiben wir beide Terme auf der rechten Seite aus. Mit der Turmeigenschaft für $X^2$ erhalten wir

<!--en:a167c9e7-->
Wenden wir $\textrm{Var}(Z)=\mathbb{E}[Z^2]-(\mathbb{E}[Z])^2$ (:numref:`sec_mdl-random_variables`) auf die Zufallsvariable $\mathbb{E}[X\mid Y]$ an, deren Mittelwert nach der Turmeigenschaft $\mathbb{E}[X]$ ist, so ergibt sich

<!--en:486712b7-->
Addieren wir die beiden Zeilen, hebt sich der mittlere Term $\mathbb{E}[(\mathbb{E}[X\mid Y])^2]$ weg. Übrig bleibt $\mathbb{E}[X^2]-\mathbb{E}[X]^2=\textrm{Var}(X)$. $\blacksquare$

<!--en:8c199b27-->
Die zweite Identität, der Satz von der totalen Varianz, heißt im Englischen auch *Eve’s law*, nach der Folge E, V, V, E auf der rechten Seite. Sie zerlegt die Gesamtvarianz von $X$ in zwei Anteile. Der erste ist die mittlere Varianz innerhalb der Streifen. Sie bleibt auch dann bestehen, wenn wir $Y$ kennen. Der zweite ist die Varianz der Streifenmittelwerte, also der Anteil, den $Y$ erklärt. In der Statistik dient diese Zerlegung unter dem Namen Varianzanalyse dazu, Streuung in Daten auf Gruppenunterschiede zurückzuführen. Dieselbe Zerlegung beantwortet auch unsere Frage zur Regression.

<!--en:1ef63c3b-->
**Satz (der bedingte Erwartungswert ist der beste Prädiktor für den quadratischen Fehler).** *Die Zufallsvariable $X$ habe endliche Varianz und $h$ sei eine beliebige Funktion mit $\mathbb{E}[h(Y)^2]<\infty$. Dann gilt*

<!--en:5a6743c0-->
**Beweis.** Schreiben wir $m(Y)=\mathbb{E}[X\mid Y]$ und zerlegen wir den Vorhersagefehler in zwei Teile, $X-h(Y)=(X-m(Y))+(m(Y)-h(Y))$. Beim Quadrieren entstehen zwei Quadrate und ein gemischter Term. Nur Letzterer erfordert besondere Aufmerksamkeit. Nach der Turmeigenschaft ist der Erwartungswert des ersten Quadrats $\mathbb{E}[\textrm{Var}(X\mid Y)]$. Der gemischte Term $2(X-m(Y))(m(Y)-h(Y))$ hat den bedingten Erwartungswert null. Der Faktor $m(Y)-h(Y)$ hängt allein von $Y$ ab und lässt sich aus dem Streifenmittelwert herausziehen. Es bleibt $\mathbb{E}[X-m(Y)\mid Y]=m(Y)-m(Y)=0$. Nach der Turmeigenschaft ist damit auch der unbedingte Erwartungswert null. $\blacksquare$

<!--en:90709fae-->
Betrachten wir die beiden Terme rechts. Der erste ist unabhängig von der Wahl von $h$. Der zweite ist nichtnegativ und verschwindet für $h(Y)=\mathbb{E}[X\mid Y]$. Der bedingte Erwartungswert ist also der optimale Prädiktor für den quadratischen Fehler. Kein Prädiktor erreicht einen kleineren erwarteten quadratischen Fehler als die mittlere Varianz innerhalb der Streifen. Ein mit quadratischem Verlust trainiertes Netz schätzt somit $\mathbb{E}[\textrm{target}\mid\textrm{input}]$. Genau so interpretiert :numref:`sec_linear_regression` das dortige Modell, in der Schreibweise $\mathbb{E}[y\mid\mathbf x]$.

<!--en:07ca21b7-->
Zwei Beispiele veranschaulichen diese Zerlegung. Versuchen wir zunächst, bei unserem Dartpfeil $X$ aus $Y$ vorherzusagen. Unter der Bedingung $Y=y$ ist die horizontale Koordinate $X$ auf einer Sehne gleichverteilt, die symmetrisch um $0$ liegt. Also gilt $\mathbb{E}[X\mid Y=y]=0$. Eine gleichverteilte Variable auf einem Intervall der halben Breite $w$ hat die Varianz $w^2/3$. Das ergibt $\textrm{Var}(X\mid Y=y)=(1-y^2)/3$. Aus Symmetriegründen gilt $\mathbb{E}[X^2]=\mathbb{E}[Y^2]=\tfrac12\mathbb{E}[R^2]=\tfrac14$. Der Satz von der totalen Varianz liefert daher

<!--en:13234ce1-->
Die gesamte Varianz entfällt auf den unerklärten Anteil. Kennen wir $Y$, so verengt sich der Wertebereich von $X$, ohne dass sich sein Mittelpunkt verschiebt. Die beste Vorhersage von $X$ aus $Y$ für den quadratischen Fehler ist deshalb die Konstante $0$, obwohl $X$ und $Y$ abhängig sind.

<!--en:0dc9e2c4-->
Für das zweite Beispiel ziehen wir ein binäres Label $Z$. Es ist gleich $1$ mit Wahrscheinlichkeit $0.2$ und gleich $0$ mit Wahrscheinlichkeit $0.8$. Für $Z=1$ ziehen wir $X$ aus der Normalverteilung $\mathcal N(3,1)$ mit Mittelwert $3$ und Varianz $1$. Für $Z=0$ ziehen wir $X$ aus $\mathcal N(-1,1)$. Diese Mischung ergibt die zweigipflige Dichte aus :numref:`fig_mdl-prob-pdf-area`. Ihr Gesamtmittelwert ist $0.2\cdot3+0.8\cdot(-1)=-0.2$. Innerhalb jeder Komponente beträgt die Varianz $1$, also gilt $\mathbb{E}[\textrm{Var}(X\mid Z)]=1$. Je nach Komponente nimmt $\mathbb{E}[X\mid
Z]$ die Werte $3$ und $-1$ an. Deren Varianz beträgt $0.2\cdot0.8\cdot(3-(-1))^2=2.56$. Die Gesamtvarianz ist daher $3.56$ und das Label $Z$ erklärt davon den Anteil $2.56/3.56=72\%$. In einer Stichprobe mit einer Million Ziehungen liegen die empirischen Varianzanteile innerhalb der Komponenten und zwischen ihnen nahe bei $1$ und $2.56$. Was erwarten Sie, bevor Sie den Code ausführen? Stimmt ihre Summe nur ungefähr oder genau mit der Stichprobenvarianz überein?

<!--en:a52a0cbd-->
Genau. Beide Anteile liegen nahe an ihrem theoretischen Wert ($1$ und $2.56$), mit zufallsbedingten Schwankungen in der dritten Nachkommastelle. Dennoch entspricht ihre Summe bei jedem Durchlauf bis auf die Rundung der letzten ausgegebenen Ziffer der Stichprobenvarianz. Der Satz von der totalen Varianz gilt exakt für die Stichprobe selbst. Die empirischen Gruppenanteile und Gruppenvarianzen übernehmen dabei die Rolle der Wahrscheinlichkeiten und bedingten Varianzen.

<!--en:ff696632-->
## Kovarianz und Korrelation

<!--en:2a06d896-->
Der bedingte Erwartungswert beschreibt, wie sich der Mittelwert von $X$ mit $Y$ verändert. Er ist also eine ganze Funktion. Oft möchten wir den Zusammenhang mit einer einzigen Zahl beschreiben: Nehmen $X$ und $Y$ tendenziell gemeinsam große Werte an, oder ist die eine Variable meist groß, wenn die andere klein ist? Diese Zahl ist die Kovarianz. Sie hat zwei Einschränkungen, die wir im Folgenden untersuchen. Zum einen trägt sie die Einheiten beider Variablen. Die Korrelation beseitigt diese Einheiten. Zum anderen erfasst sie nur lineare Trends. Daran ändert auch eine andere Skalierung nichts.

<!--en:a98ec867-->
### Kovarianz

<!--en:726f2b3e-->
In :numref:`sec_prob` haben wir die *Kovarianz* als Erwartungswert des Produkts der Abweichungen beider Variablen von ihren jeweiligen Mittelwerten definiert. Für diskrete Variablen $X,Y$, die das Wertepaar $(x_i,y_j)$ mit Wahrscheinlichkeit $p_{ij}$ annehmen, gilt

<!--en:84c48a11-->
wobei die zweite Form wie bei der Rechenformel für die Varianz durch Ausmultiplizieren entsteht. Insbesondere gilt $\textrm{Cov}(X,X)=\textrm{Var}(X)$. In diesem Abschnitt setzen wir voraus, dass beide Varianzen endlich sind. Wegen $|xy|\le(x^2+y^2)/2$ existiert dann der Erwartungswert von $XY$.

<!--en:7c52a5eb-->
Als konkretes Beispiel betrachten wir $X\in\{1,3\}$ und $Y\in\{-1,3\}$ mit den gemeinsamen Wahrscheinlichkeiten

<!--en:2b28a8d8-->
für einen Parameter $\theta\in[0,1]$. Wir nennen dies das *Vierfelderbeispiel*. Jede Variable nimmt ihre beiden Werte mit Wahrscheinlichkeit $\tfrac12$ an, also gilt $\mu_X=2$ und $\mu_Y=1$. Der Erwartungswert des Produkts ist

<!--en:bc3285a4-->
und damit gilt $\textrm{Cov}(X,Y)=4\theta-2$. Die Kovarianz beträgt $+2$ bei $\theta=1$, $-2$ bei $\theta=0$ und $0$ bei $\theta=\tfrac12$.

<!--en:eb468ba1-->
Für stetige Variablen mit gemeinsamer Dichte wird die Summe zu einem Doppelintegral,

<!--en:2c8bb2ea-->
das das Produkt der Abweichungen über die gemeinsame Dichte mittelt. Beim Dartwurf haben beide Koordinaten den Mittelwert $0$. Außerdem gilt $\textrm{Cov}(X,Y)=\mathbb{E}[XY]=\frac1\pi\iint_{\textrm{disk}}xy\,dx\,dy=0$, denn der Integrand ist ungerade in $x$ und die Scheibe ist symmetrisch. Dennoch sind $X$ und $Y$ abhängig, wie die kürzer werdenden Sehnen gezeigt haben. Die Kovarianz erfasst nur lineare Trends. Die kreisförmige Abhängigkeit beim Dartwurf weist keinen solchen Trend auf. Aus Kovarianz null folgt also keine Unabhängigkeit (:numref:`sec_prob` zeigt ein diskretes Beispiel mit $Y=X^2$). Beide Eigenschaften der Koordinaten auf der Kreisscheibe sind in :numref:`fig_mdl-prob-dart-three-questions` zu sehen.

<!--en:28d5f140-->
![Die Koordinaten des Dartpfeils sind abhängig, aber im Mittel nicht vorhersagbar und unkorreliert. (a) Die Sehne in Höhe $Y=0.9$ beschränkt $X$ auf $|X|\le0.44$. Nahe der Mitte ist die Sehne viel länger. Somit hängt $X$ von $Y$ ab. Der Mittelpunkt jeder Sehne liegt bei $x=0$ (grüne Punkte), also gilt $\mathbb{E}[X\mid Y]=0$. (b) Das Produkt $xy$ ist in zwei Quadranten positiv und in den anderen beiden negativ. Alle vier haben dieselbe Fläche, daher gilt $\mathrm{Cov}(X,Y)=0$.](../img/mdl-prob-dart-three-questions.svg)

<!--en:e772bdfc-->
Die Kovarianz trägt außerdem die Einheiten beider Variablen. In :numref:`fig_mdl-prob-covariance` zeigt die untere Reihe genau dieselben Ziehungen wie die obere, nur geben wir $Y$ in Cent statt in Dollar an. Jede Stichprobenkovarianz wächst um den Faktor $100$, obwohl alle Punktwolken ihre Form behalten. Ohne die Einheiten ist nur das Vorzeichen einer Kovarianz aussagekräftig. Der Korrelationskoeffizient, den wir weiter unten einführen, ist gänzlich einheitenlos.

<!--en:816b821d-->
![Obere Reihe: Punktwolken aus $(X,Y)$-Paaren mit negativer, verschwindender und positiver Kovarianz (von links nach rechts), $Y$ in Dollar. Bei negativer Kovarianz fällt die Wolke nach rechts ab, bei Kovarianz null bleibt sie rund und bei positiver Kovarianz steigt sie nach rechts an. Untere Reihe: dieselben Ziehungen mit $Y$ in Cent. Jede Stichprobenkovarianz wächst um den Faktor $100$, während die Korrelation $\rho$ (jeweils oben in der Ecke) gleich bleibt. Der Betrag der Kovarianz trägt Einheiten, nur ihr Vorzeichen ist skalenunabhängig.](../img/mdl-prob-covariance.svg)

<!--en:750af7e0-->
Die Kovarianz ist in jedem Argument linear. Addieren wir eine Konstante, verschieben sich die Variable und ihr Mittelwert um denselben Betrag. Die Verschiebung fällt aus der Abweichung heraus, während sich multiplikative Faktoren ausklammern lassen: $\textrm{Cov}(cX+d,Y)=c\,\textrm{Cov}(X,Y)$. Ebenso ist die Abweichung einer Summe die Summe der Abweichungen. Aus der Linearität des Erwartungswerts folgt daher $\textrm{Cov}(X+Z,Y)=\textrm{Cov}(X,Y)+\textrm{Cov}(Z,Y)$. Für das zweite Argument gelten dieselben Regeln. Unabhängige Variablen haben außerdem stets Kovarianz null. Faktorisiert die gemeinsame Dichte, so zerfällt das Doppelintegral für $\mathbb{E}[XY]$ in das Produkt $\mathbb{E}[X]\,\mathbb{E}[Y]$.

<!--en:313a8bc3-->
Mit der Kovarianz können wir auch die vollständige Formel für die Varianz einer Summe angeben. Setzen wir $\tilde X=X-\mu_X$ und $\tilde Y=Y-\mu_Y$. Wegen der Linearität hat $X+Y$ den Mittelwert $\mu_X+\mu_Y$ und damit die Abweichung $\tilde X+\tilde Y$. Ausmultiplizieren gemäß $\mathbb{E}[(\tilde X+\tilde Y)^2]=\mathbb{E}[\tilde X^2]+2\,\mathbb{E}[\tilde X\tilde Y]+\mathbb{E}[\tilde Y^2]$ liefert

<!--en:e96298d3-->
Sind $X$ und $Y$ unabhängig, verschwindet die Kovarianz und die Varianzen addieren sich.

<!--en:b629c7a5-->
Die wichtigste Folgerung aus dieser Regel betrifft den Mittelwert vieler Ziehungen. Multiplizieren wir das Quadrat einer Summe von $n$ Abweichungen auf dieselbe Weise aus, erhalten wir $\textrm{Var}(\sum_i X_i)=\sum_i\sum_j\textrm{Cov}(X_i,X_j)$. Die Variablen $X_1,\ldots,X_n$ seien unabhängig und haben alle dieselbe Varianz $\sigma^2$. Alle gemischten Terme verschwinden, also hat die Summe die Varianz $n\sigma^2$. Skalieren wir die Summe mit $c=1/n$, teilt sich ihre Varianz durch $n^2$. Der Mittelwert $\bar X_n=\frac1n\sum_{i=1}^n X_i$ hat daher die Varianz

<!--en:863fd8b9-->
Mitteln wir $n$ unabhängige Ziehungen, so teilt sich die Varianz durch $n$ und die Standardabweichung durch $\sqrt n$. Um den typischen Fehler zu halbieren, brauchen wir viermal so viele Ziehungen. Diese Skalierung mit $\sigma/\sqrt n$ liegt jeder *Monte-Carlo-Schätzung* zugrunde, bei der ein Mittelwert simulierter Samples einen Erwartungswert annähert (:numref:`subsec_mdl-monte-carlo`). Beim Dartwurf hat der mittlere Abstand aus $n$ unabhängigen Würfen die Varianz $1/(18n)$. Es genügen also $556$ Würfe, um seine Standardabweichung auf $0.01$ zu senken. Die Rate $\sigma/\sqrt n$ setzt eine endliche Varianz voraus. Das *Gesetz der großen Zahlen* (engl. law of large numbers) selbst besagt, dass $\bar X_n$ gegen den Erwartungswert konvergiert. Dafür genügt ein endlicher Erwartungswert (:numref:`sec_mdl-statistics`). Die Cauchy-Verteilung besitzt weder einen endlichen Erwartungswert noch eine endliche Varianz. Der Mittelwert ihrer Ziehungen stabilisiert sich nicht, egal wie groß $n$ wird.

<!--en:41986c8e-->
### Korrelation

<!--en:e20db842-->
Ob die Kovarianz $4\theta-2$ in unserem Vierfelderbeispiel groß ist, hängt von den Einheiten von $X$ und $Y$ ab. Um die Einheiten herauszukürzen, dividieren wir durch das Produkt der beiden Standardabweichungen, die genau diese Einheiten tragen. Der *Korrelationskoeffizient* ist

<!--en:ede3a592-->
Wie in :numref:`sec_prob` angegeben, liegt $\rho$ stets zwischen $-1$ und $1$. Beweisen wir diese Schranke und bestimmen wir, wann genau Gleichheit gilt.

<!--en:f64523fb-->
**Satz (Schranke für die Korrelation).** *Für alle $X,Y$ mit endlichen, von null verschiedenen Varianzen gilt $-1\le\rho(X,Y)\le 1$. Gleichheit gilt genau dann, wenn mit Wahrscheinlichkeit eins $Y=cX+d$ für Konstanten $c\ne0$ und $d$ gilt. Dabei ist $\rho=1$ für $c>0$ und $\rho=-1$ für $c<0$.*

<!--en:a6e1aa68-->
**Beweis.** Eine Varianz ist nie negativ, also gilt $\textrm{Var}(tX+Y)\ge 0$ für jedes reelle $t$. Mit der Regel für die Varianz einer Summe und der Regel für affine Transformationen erhalten wir ein quadratisches Polynom in $t$,

<!--en:b48674cd-->
Ein quadratisches Polynom mit positivem Leitkoeffizienten, das nie unter null fällt, hat höchstens eine reelle Nullstelle. Seine Diskriminante ist also nichtpositiv:

<!--en:537f55d7-->
Division beider Seiten durch $\sigma_X^2\sigma_Y^2$ ergibt $\rho^2\le 1$. Für $\rho^2=1$ verschwindet die Diskriminante und das Polynom hat eine doppelte Nullstelle $t^\star$ mit $\textrm{Var}(t^\star X+Y)=0$. Eine Zufallsvariable mit Varianz null ist mit Wahrscheinlichkeit eins konstant. Daher gilt $Y=-t^\star X+d$ mit Wahrscheinlichkeit eins, und es gilt $t^\star\ne0$, weil $\textrm{Var}(Y)>0$. Gilt umgekehrt $Y=cX+d$ mit $c\ne0$, so folgen $\textrm{Cov}(X,Y)=c\,\textrm{Var}(X)$ und $\sigma_Y=|c|\sigma_X$, also $\rho=c/|c|=\pm1$. $\blacksquare$

<!--en:527cc035-->
Die Korrelation misst, wie eng $X$ und $Y$ einer Geraden folgen, und erfasst das Vorzeichen ihrer Steigung. Der Betrag der Steigung spielt dabei keine Rolle: Für $Y=0.5X$ und $Y=2X$ gilt jeweils $\rho=1$. Im Vierfelderbeispiel gilt $\sigma_X=1$ und $\sigma_Y=2$, also $\rho=(4\theta-2)/2=2\theta-1$. Die Korrelation steigt von $-1$ auf $1$, wenn $\theta$ von $0$ auf $1$ wächst. Wie :numref:`fig_mdl-prob-correlation` zeigt, entgehen $\rho$ zwei Dinge: Zwei Punktwolken können trotz gleicher Korrelation verschiedene Steigungen haben, und eine Punktwolke auf einer Parabel kann die Korrelation null haben.

<!--en:e110444f-->
![Die Korrelation misst die Stärke eines linearen Zusammenhangs und das Vorzeichen seiner Steigung. Der Betrag der Steigung kann sich unterscheiden: (a) und (b) haben dasselbe $\rho\approx0.9$ bei Steigungen von $0.5$ und $2$. Auch ein nichtlinearer Zusammenhang kann der Korrelation entgehen: In (c) ist $Y$ eine Funktion von $X$ und es gilt $\rho\approx0$.](../img/mdl-prob-correlation.svg)

<!--en:0a865223-->
Der obige Beweis ist zugleich der klassische Beweis der Cauchy-Schwarz-Ungleichung $|\mathbf u^\top\mathbf v|\le\|\mathbf u\|\,\|\mathbf v\|$. Das ist kein Zufall, denn die Korrelation ist tatsächlich ein Kosinus. Auf dem Raum der Zufallsvariablen mit Mittelwert null und endlicher Varianz bildet der Erwartungswert des Produkts ein inneres Produkt. Dabei fassen wir zwei Variablen als identisch auf, wenn sie mit Wahrscheinlichkeit eins übereinstimmen. Für die zentrierten Variablen $\tilde X$ und $\tilde Y$ ist dieses innere Produkt $\langle \tilde X,\tilde Y\rangle=\mathbb{E}[\tilde X\tilde Y]=\textrm{Cov}(X,Y)$. Die zugehörige Länge ist die Standardabweichung $\|\tilde X\|=\sqrt{\mathbb{E}[\tilde X^2]}=\sigma_X$ und die Korrelation ist der Kosinus $\langle \tilde X,\tilde Y\rangle/(\|\tilde X\|\,\|\tilde Y\|)$. Geometrisch sind unkorrelierte Variablen orthogonal, während $\rho=\pm1$ bedeutet, dass sie parallel sind.

<!--en:886db974-->
### Die Kovarianzmatrix

<!--en:b0501049-->
Betrachten wir $n$ Variablen zugleich, gibt es für jedes Paar eine Kovarianz. Diese Kovarianzen lassen sich übersichtlich in einer Matrix anordnen. Für einen Zufallsvektor $\mathbf X=(X_1,\ldots,X_n)^\top$ mit Mittelwert $\boldsymbol\mu$ ist die *Kovarianzmatrix* der Erwartungswert des äußeren Produkts des Abweichungsvektors mit sich selbst,

<!--en:36a0bb8b-->
Auf ihrer Diagonalen stehen die einzelnen Varianzen, außerhalb die paarweisen Kovarianzen. Wegen $\textrm{Cov}(X_i,X_j)=\textrm{Cov}(X_j,X_i)$ ist die Matrix symmetrisch. Sie ist außerdem positiv semidefinit. Für jeden festen Vektor $\mathbf a$ hat die skalare Projektion $\mathbf a^\top\mathbf X=\sum_i a_iX_i$ die Abweichung $\mathbf a^\top(\mathbf X-\boldsymbol\mu)$. Somit gilt

<!--en:3eb12e1b-->
Die quadratische Form $\mathbf a^\top\boldsymbol\Sigma\,\mathbf a$ ist also die Varianz der Daten nach Projektion auf die Richtung $\mathbf a$, und eine Varianz kann nie negativ sein. Für den Dartwurf gilt $\boldsymbol\Sigma=\operatorname{diag}(\tfrac14,\tfrac14)$. Seine Koordinaten sind unkorreliert und streuen gleich stark. Später begegnen uns die Eigenvektoren von $\boldsymbol\Sigma$ als Richtungen der Hauptkomponentenanalyse (:numref:`sec_mdl-svd-low-rank`) und als Achsen der elliptischen Höhenlinien der multivariaten Normalverteilung (:numref:`sec_mdl-distributions`).

<!--en:e6e593f6-->
## Transformationen von Zufallsvariablen

<!--en:9f747979-->
Im maschinellen Lernen transformieren wir ständig Zufallsvariablen. Eine Aktivierungsfunktion wirkt auf eine Präaktivierung, ein Stichprobenverfahren transformiert gleichverteiltes Rauschen und ein normalisierender Fluss überführt eine einfache Normalverteilung in eine komplexe Datenverteilung. Wenn $X$ eine bekannte Dichte hat und wir $Y=g(X)$ setzen, welche Dichte hat dann $Y$?

<!--en:f2fd984e-->
### Die eindimensionale Transformationsformel

<!--en:70961cd6-->
Die naheliegende Vermutung $p_Y(y)=p_X(g^{-1}(y))$ ist falsch. Unser Dartwurf zeigt bereits, warum. Den Abstand können wir als $R=\sqrt U$ erzeugen, wobei $U$ auf $[0,1]$ gleichverteilt ist. Diese Abbildung verwandelt die konstante Dichte von $U$ jedoch in die linear ansteigende Dichte $2r$. Bei einer Transformation bleibt die Wahrscheinlichkeitsmasse erhalten, die Höhe der Dichte kann sich dagegen ändern. Ein schmales Intervall der Breite $\Delta x$ um $x$ trägt ungefähr die Wahrscheinlichkeit $p_X(x)\,\Delta x$. Die Abbildung $g$ überführt es in ein Intervall mit ungefähr der Breite $\Delta y=|g'(x)|\,\Delta x$ um $y=g(x)$, das dieselbe Wahrscheinlichkeit tragen muss:

<!--en:2737084e-->
Wo $g$ die Zahlengerade streckt, verteilt sich dieselbe Wahrscheinlichkeit über ein breiteres Intervall. Die Dichte muss also um den Streckfaktor $|g'(x)|$ sinken. Wo $g$ die Gerade staucht, muss die Dichte steigen. Die beiden flächengleichen Streifen sind in :numref:`fig_mdl-prob-change-of-variables` für $Y=e^X$ dargestellt.

<!--en:179dff8f-->
![Variablentransformation. Ein Intervall der Breite $\Delta x$ unter $p_X$ (unten) geht durch $g$ in ein Intervall der Breite $\Delta y=g'(x)\,\Delta x$ unter $p_Y$ (links) über. Die beiden schattierten Flächen entsprechen derselben Wahrscheinlichkeit. Wo $g$ streckt ($\Delta y>\Delta x$), muss die Dichte $p_Y$ daher um den Faktor $|g'(x)|$ kleiner sein. Hier gilt $X\sim\mathcal N(0,1)$ und $Y=e^X$.](../img/mdl-prob-change-of-variables.svg)

<!--en:08601335-->
**Satz (Variablentransformation in einer Dimension).** *Die Zufallsvariable $X$ habe eine Dichte $p_X$, die auf einem offenen Intervall $I$ mit $P(X\in I)=1$ stetig ist. Die Abbildung $g:I\to\mathbb{R}$ sei stetig differenzierbar mit $g'\ne0$ auf $I$, sodass $g$ streng monoton und injektiv ist. Dann hat $Y=g(X)$ die Dichte*

<!--en:c5c32fc8-->
*auf dem Intervall $g(I)$ und die Dichte $0$ außerhalb.*

<!--en:bdb7ea7b-->
**Beweis.** Zunächst sei $g$ wachsend. Dann gilt $g(X)\le y$ genau dann, wenn $X\le g^{-1}(y)$. Für die Verteilungsfunktionen folgt $F_Y(y)=F_X(g^{-1}(y))$ für $y\in g(I)$. Differenzieren mit der Kettenregel und $F_X' = p_X$ (:numref:`sec_mdl-random_variables`) ergibt $p_Y(y)=p_X(g^{-1}(y))\,\frac{dg^{-1}}{dy}(y)$. Ist $g$ hingegen fallend, so gilt $g(X)\le y$ genau dann, wenn $X\ge g^{-1}(y)$, also $F_Y(y)=1-F_X(g^{-1}(y))$. Beim Ableiten ergibt sich ein Minuszeichen, das das negative Vorzeichen von $dg^{-1}/dy$ aufhebt. Der Betrag deckt beide Fälle ab. In beiden Fällen ist $F_Y$ auf $g(I)$ stetig differenzierbar. Da dieses Intervall die gesamte Wahrscheinlichkeit trägt, ist die Ableitung nach dem Hauptsatz der Differential- und Integralrechnung eine Dichte von $Y$. $\blacksquare$

<!--en:93dc33ff-->
In Worten erhalten wir die neue Dichte bei $y$, indem wir die alte Dichte am Urbild von $y$ auswerten und mit dem lokalen Skalierungsfaktor der Umkehrabbildung multiplizieren. Wegen $dg^{-1}/dy=1/g'(x)$ an der Stelle $x=g^{-1}(y)$ ist das genau der Faktor $1/|g'(x)|$ aus unserer Betrachtung der Streckung.

<!--en:8eecd991-->
Die Inversionsmethode ist eine Variablentransformation. Die Transformationsformel liefert daher direkt aus der Ziehvorschrift die Dichte des Abstands beim Dartwurf. Sei $X=U$ gleichverteilt auf $I=(0,1)$ und sei $g(u)=\sqrt u$. Dann gilt $g^{-1}(r)=r^2$ und $|dg^{-1}/dr|=2r$. Aus :eqref:`eq_mdl-cov_density_1d` folgt $p_R(r)=1\cdot 2r$, die Dichte des Abstands.

<!--en:0c493b73-->
Ist $g$ nicht injektiv, so können mehrere Eingabewerte denselben Ausgabewert $y$ haben. Ihre Wahrscheinlichkeitsbeiträge addieren sich:

<!--en:a4578f88-->
Dies gilt für Werte $y$ mit endlich vielen Urbildern, an denen jeweils $g'(x)\ne0$ gilt. Für $Y=X^2$ und $y>0$ gilt an beiden Urbildern $\pm\sqrt y$ jeweils $|g'|=2\sqrt y$, also $p_Y(y)=[p_X(\sqrt y)+p_X(-\sqrt y)]/(2\sqrt y)$. Die Voraussetzung $g'\ne0$ ist wesentlich. Für die ReLU-Aktivierung gilt $g'=0$ auf der gesamten negativen Halbachse $(-\infty,0)$. Die gesamte dortige Wahrscheinlichkeitsmasse fällt auf den einen Punkt $0$. So entsteht ein Atom und die Verteilung lässt sich nicht allein durch eine Dichte beschreiben (siehe :exref:`ex_mdl-random-variables-rectified-gaussian`).

<!--en:de8b3d37-->
Als Beispiel betrachten wir $X\sim\mathcal N(0,1)$ mit der Standardnormaldichte $p_X(x)=e^{-x^2/2}/\sqrt{2\pi}$ und setzen $Y=e^X$. Hier gilt $g^{-1}(y)=\log y$ und $|dg^{-1}/dy|=1/y$ für $y>0$. Die Variablentransformation liefert also die Dichte der *Lognormalverteilung*,

<!--en:d034beba-->
Der Vorfaktor $1/y$ ist der Korrekturfaktor der Variablentransformation. Setzen wir die Ableitung von $\log p_Y(y)=-\log y-\tfrac12(\log y)^2+\textrm{const}$ gleich null, erhalten wir $\log y=-1$. Die tatsächliche Dichte erreicht ihr Maximum also bei $y=e^{-1}\approx0.368$. Die naiv vermutete Funktion $p_X(\log y)$ hat ihr Maximum hingegen bei $y=1$. Sie ist nicht einmal eine gültige Dichte. Substituieren wir $y=e^x$, so ergibt sich für die Gesamtfläche $\int p_X(x)\,e^x\,dx=\mathbb{E}[e^X]=e^{1/2}\approx1.65$ (quadratische Ergänzung liefert $x-x^2/2=\tfrac12-\tfrac12(x-1)^2$).

<!--en:c2636c59-->
Dieselbe Rechnung liefert alle Momente $\mathbb{E}[Y^k]=\mathbb{E}[e^{kX}]=e^{k^2/2}$, die ausnahmslos endlich sind. Dennoch klingt die Wahrscheinlichkeit $P(Y>t)=P(X>\log t)$ langsamer ab als jede in $t$ exponentiell fallende Funktion. Die Lognormalverteilung hat also einen schweren rechten Verteilungsrand, besitzt aber im Gegensatz zur Cauchy-Verteilung endliche Momente jeder Ordnung.

<!--en:1320db2a-->
Zeichnen wir ein Histogramm von $10^5$ Samples von $e^X$ zusammen mit den beiden möglichen Kurven. Wir wissen bereits, dass die Fläche unter der naiv vermuteten Kurve um $0.65$ zu groß ist. Wo vermuten Sie diesen Überschuss, bevor Sie die Abbildung betrachten?

<!--en:329a728b-->
Bis auf zufallsbedingte Schwankungen folgt das Histogramm der Dichte der Lognormalverteilung eng. Sein Maximum liegt deutlich links von $y=1$, nahe $e^{-1}$. Die naiv vermutete Kurve hat ihr Maximum an der falschen Stelle. Da sie der tatsächlichen Dichte multipliziert mit $y$ entspricht, liegt sie für $y<1$ unter und für $y>1$ über den Daten. Zum rechten Rand hin wächst der Faktor immer weiter.

<!--en:79b80949-->
### Die multivariate Transformationsformel

<!--en:81674f5d-->
In mehreren Dimensionen streckt eine Transformation Flächen und Volumina. An die Stelle des Längenfaktors tritt daher eine Determinante. Betrachten wir zunächst den einfachsten Fall, eine lineare Abbildung $\mathbf Y=\mathbf A\mathbf X$ mit $\mathbf A=\operatorname{diag}(2,3)$. Sie verwandelt ein Einheitsquadrat in ein $2\times3$-Rechteck der Fläche $6=\det\mathbf A$. Dieses Rechteck muss dieselbe Wahrscheinlichkeit tragen wie das ursprüngliche Quadrat. Die Dichte von $\mathbf Y$ muss daher ein Sechstel der Dichte von $\mathbf X$ am entsprechenden Punkt betragen: $p_{\mathbf Y}(\mathbf y)=p_{\mathbf X}(\mathbf A^{-1}\mathbf y)/6$.

<!--en:38bbf485-->
Bei einer nichtlinearen Abbildung variiert die lokale Streckung von Punkt zu Punkt. Der Transformationssatz für Mehrfachintegrale berücksichtigt dies (:numref:`sec_mdl-integral_calculus`). Sei $g$ eine stetig differenzierbare Bijektion zwischen zwei Gebieten des $\mathbb{R}^n$ mit stetig differenzierbarer Umkehrabbildung. Mit $J_{g^{-1}}$ bezeichnen wir die Jacobi-Matrix der partiellen Ableitungen der Umkehrabbildung. Lokal skaliert $g^{-1}$ das Volumen mit dem Faktor $|\det J_{g^{-1}}|$. Die Substitution $\mathbf x=g^{-1}(\mathbf y)$ in einem Integral über das Urbild eines Gebiets $\mathcal D$ liefert daher

<!--en:01b39b8d-->
Sei nun $\mathbf Y=g(\mathbf X)$. Das Ereignis $\mathbf Y\in\mathcal D$ stimmt mit dem Ereignis $\mathbf X\in g^{-1}(\mathcal D)$ überein. Wenden wir den Satz mit $f=p_{\mathbf X}$ an, erhalten wir

<!--en:922bb96c-->
Der Integrand auf der rechten Seite ist somit die Dichte von $\mathbf Y$:

<!--en:72f6d9cf-->
Die zweite Zeile nutzt $J_{g^{-1}}(\mathbf y)=J_g(\mathbf x)^{-1}$. Bei der Transformation eines Samples durch $g$ verringert sich seine Log-Dichte um den Logarithmus des lokalen Volumenfaktors. Wegen $\log|\det(J_2 J_1)| = \log|\det J_1| + \log|\det J_2|$ ziehen wir bei einer Verkettung von Abbildungen die Summe der logarithmierten Determinantenbeträge ab. Genau so berechnen *normalisierende Flüsse* (engl. normalizing flows) exakte Likelihoods. Sie verketten invertierbare Schichten, deren logarithmierte Determinantenbeträge sich mit geringem Aufwand berechnen lassen :cite:`rezende2015variational`. In :numref:`sec_mdl-continuous-normalizing-flows` konstruieren wir einen solchen Fluss, dessen Schichten den Integrationsschritten einer Differentialgleichung entsprechen. In :exref:`ex_mdl-random-variables-affine-coupling` implementieren Sie selbst eine Kopplungsschicht.

<!--en:7014b062-->
Ein letztes Beispiel liefert unser Dartwurf. Beschreiben wir den Auftreffpunkt in Polarkoordinaten durch den Abstand $R$ und den Winkel $\Theta$ gemäß $(x,y)=(r\cos\theta,r\sin\theta)$. Die Jacobi-Determinante dieser Abbildung von $(r,\theta)$ auf $(x,y)$ ist $\cos\theta\cdot r\cos\theta-(-r\sin\theta)\sin\theta=r$. Die multivariate Transformationsformel ergibt daher

<!--en:592f35e2-->
Die gemeinsame Dichte zerfällt in zwei Faktoren. Der Winkel $\Theta$ ist auf $[0,2\pi)$ gleichverteilt. Für den Abstand $R$ ergibt sich erneut die Dichte $2r$, diesmal direkt aus einer Jacobi-Determinante. Die beiden Variablen sind unabhängig. In kartesischen Koordinaten waren $X$ und $Y$ beim Dartwurf abhängig. Im passenden Koordinatensystem beschreiben zwei unabhängige Variablen denselben Wurf.

<!--en:94f299eb-->
Eine zentrale Frage zu Mittelwerten bleibt in diesem Abschnitt offen. Wir wissen nun, dass der Mittelwert von $n$ unabhängigen Ziehungen die Varianz $\sigma^2/n$ hat (:eqref:`eq_mdl-var_mean`). Seine Verteilung kennen wir noch nicht. Die Chebyshev-Ungleichung (:numref:`sec_mdl-random_variables`) liefert aus der Varianz die obere Schranke $\sigma^2/(n\epsilon^2)$ für die Wahrscheinlichkeit, dass der Mittelwert um mindestens $\epsilon$ vom Erwartungswert abweicht. Für beschränkte oder normalverteilte Ziehungen ist diese Schranke jedoch viel zu pessimistisch. Die tatsächliche Fehlerwahrscheinlichkeit fällt exponentiell in $n$. Welcher Verteilung sich der Mittelwert nähert, bestimmen wir in :numref:`subsec_mdl-gaussian-clt`. Die exponentielle Abklingrate leiten wir in :numref:`sec_mdl-concentration-generalization` her. Zuvor brauchen wir jedoch mehr Beispiele für Verteilungen als nur den Dartwurf.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:7fb01995-->
Randdichten erhalten wir, indem wir nicht benötigte Variablen ausintegrieren. Für bedingte Dichten schneiden wir die gemeinsame Dichte entlang eines schmalen Streifens und normieren das Schnittprofil. Der bedingte Erwartungswert ist der optimale Prädiktor für den quadratischen Fehler. Der Satz von der totalen Varianz zerlegt die Varianz einer Variablen in einen Anteil, den die beobachtete Variable erklärt, und einen unerklärten Anteil. Kovarianz und Korrelation messen lineare Zusammenhänge und erfassen Abhängigkeiten nur teilweise. Mitteln wir $n$ unabhängige Ziehungen mit derselben endlichen Varianz, sinkt die Varianz auf $\sigma^2/n$. Bei einer glatten invertierbaren Abbildung teilen wir die Dichte durch den lokalen Volumenfaktor, den Betrag der Jacobi-Determinante. Der Dartwurf verbindet all diese Ideen. Seine kartesischen Koordinaten sind abhängig und dennoch unkorreliert. In Polarkoordinaten beschreiben ihn zwei unabhängige Variablen.

<!--en:2294243a-->
## Übungen

<!--en:6d404002-->
1. **Unkorreliert, aber abhängig.** Whitening, etwa mit den Hauptkomponenten aus :numref:`sec_mdl-svd-low-rank`, erzeugt unkorrelierte Koordinaten. Das folgende Paar hat normalverteilte Koordinaten und Kovarianz null. Es gelte $X \sim \mathcal{N}(0, 1)$, und $S$ sei ein zufälliges Vorzeichen, unabhängig von $X$, mit $P(S = 1) = P(S = -1) = \tfrac12$. Setzen Sie $Y = SX$.
    1. Zeigen Sie anhand der Verteilungsfunktion, dass $Y \sim \mathcal{N}(0, 1)$ gilt, und weisen Sie nach, dass $\textrm{Cov}(X, Y) = 0$ gilt.
    1. Vergleichen Sie $P(|X| \le 1, |Y| \le 1)$ mit $P(|X| \le 1)\,P(|Y| \le 1)$. Sind $X$ und $Y$ unabhängig?
    1. Zeigen Sie, dass die Verteilung von $X + Y$ bei $0$ ein Atom hat, sodass $X + Y$ nicht normalverteilt ist. Mit Wahrscheinlichkeit eins liegt der Punkt $(X, Y)$ auf einer der beiden Geraden $y = \pm x$. Ein Doppelintegral über eine Gerade ist null, da diese den Flächeninhalt null hat. Folgern Sie, dass das Paar $(X, Y)$ keine gemeinsame Dichte besitzt, obwohl jede Koordinate eine Dichte hat.
   :exlabel:`ex_mdl-random-variables-uncorrelated-dependent`

<!--en:75527cf1-->
1. **Mittelung korrelierter Schätzwerte.** Für unabhängige Ziehungen haben wir $\textrm{Var}(\bar X_n) = \sigma^2/n$ hergeleitet. Mittelwerte abhängiger Schätzwerte verhalten sich anders.
    1. Die Variablen $Y_1, \ldots, Y_n$ haben jeweils die Varianz $\sigma^2$ und für jedes Paar gilt dieselbe Korrelation $\rho$. Das könnte etwa auf die Vorhersagen von $n$ Mitgliedern eines Ensembles zutreffen, die auf denselben Daten trainiert wurden. Zeigen Sie, dass

       $$\textrm{Var}\Bigl(\frac1n\sum_{i=1}^n Y_i\Bigr) = \sigma^2\Bigl(\rho + \frac{1-\rho}{n}\Bigr).$$

       Was bewirkt die Mittelung für $n \to \infty$?
    1. Nutzen Sie $\mathbf{a}^\top \boldsymbol\Sigma\, \mathbf{a} \ge 0$ für die Kovarianzmatrix $\boldsymbol\Sigma$ von $Y_1, \ldots, Y_n$, um $\rho \ge -1/(n-1)$ zu zeigen. Geben Sie Variablen an, die diese Schranke erreichen.
   :exlabel:`ex_mdl-random-variables-using-textrmvar-x-y`

<!--en:82ac0693-->
1. [code] **Sigmoidfunktion einer normalverteilten Variablen.** Sei $X \sim \mathcal{N}(0, s^2)$ die Präaktivierung einer Sigmoideinheit und sei $Y = \operatorname{sigmoid}(X)$. Die Umkehrfunktion ist $\operatorname{logit}(y) = \log\bigl(y/(1-y)\bigr)$.
    1. Bestimmen Sie mit :eqref:`eq_mdl-cov_density_1d` die Dichte $p_Y$ auf $(0, 1)$.
    1. Für welche $s$ ist $y = \tfrac12$ ein lokales Minimum von $p_Y$? Die Rechnung vereinfacht sich, wenn Sie $\log p_Y$ als Funktion von $z = \operatorname{logit}(y)$ schreiben.
    1. Ändern Sie die Zelle `#random-variables-change-of-variables` so, dass Sie ein Histogramm von $Y$ mit $p_Y$ für $s \in \{1, \sqrt{2}, 3\}$ vergleichen können. Was folgt aus dem Histogramm bei $s = 3$ für die Ableitung $\operatorname{sigmoid}'(X) = Y(1 - Y)$, über die Backpropagation den Gradienten dieser Einheiten weitergibt?
   :exlabel:`ex_mdl-random-variables-sigmoid-gaussian`

<!--en:bcd94d31-->
1. **Erklärte Varianz.** ● Ein Regressionsmodell $h$ sagt einen Zielwert $X$ aus einer Eingabe $Y$ vorher. Der Anteil der Varianz, den es erklärt, ist $R^2 = 1 - \mathbb{E}\bigl[(X - h(Y))^2\bigr]/\textrm{Var}(X)$.
    1. Zeigen Sie mit :eqref:`eq_mdl-best_predictor` und dem Satz von der totalen Varianz :eqref:`eq_mdl-tower`, dass für jeden Prädiktor $R^2 \le \textrm{Var}\bigl(\mathbb{E}[X \mid Y]\bigr)/\textrm{Var}(X)$ gilt. Berechnen Sie diese Obergrenze für die Mischung aus zwei Komponenten aus dem Text, wenn Sie $X$ aus dem Label $Z$ vorhersagen, und für den Dartwurf, wenn Sie $X$ aus $Y$ vorhersagen.
    1. Bei gegebener Eingabe nimmt der Zielwert die Werte $-1$ oder $+1$ mit Wahrscheinlichkeit $\tfrac12$ je Wert an. Welche Vorhersage minimiert den erwarteten quadratischen Fehler und wie oft nimmt der Zielwert diesen Wert an? Welche Vorhersagen minimieren den erwarteten absoluten Fehler $\mathbb{E}\bigl[|X - h(Y)|\bigr]$?
   :exlabel:`ex_mdl-random-variables-best-predictor`

<!--en:c4dd321f-->
1. [code] **Affine Kopplungsschicht und Box-Muller-Methode.** ● Normalisierende Flüsse verketten invertierbare Abbildungen, für die sich die logarithmierten Determinantenbeträge in :eqref:`eq_mdl-cov_density` mit geringem Aufwand berechnen lassen. Die affine Kopplungsschicht von RealNVP :cite:`dinh2017density` bildet $\mathbf{x} \in
   \mathbb{R}^2$ wie folgt ab:

   $$y_1 = x_1, \qquad y_2 = x_2\, e^{s(x_1)} + t(x_1),$$

   Dabei sind $s$ und $t$ beliebige Funktionen, etwa kleine neuronale Netze.
    1. Geben Sie die Umkehrabbildung und die Jacobi-Matrix $J_g(\mathbf{x})$ an. Zeigen Sie, dass $\log|\det J_g(\mathbf{x})| = s(x_1)$ gilt, wie kompliziert $s$ und $t$ auch sein mögen.
    1. Seien $X_1$ und $X_2$ unabhängige standardnormalverteilte Variablen. Setzen Sie $s(x_1) = \tanh(x_1)$ und $t(x_1) = x_1^2/2$. Geben Sie mithilfe von :eqref:`eq_mdl-cov_density` einen geschlossenen Ausdruck für $\log p_{\mathbf{Y}}(\mathbf{y})$ an. Werten Sie mit NumPy die Dichte $p_{\mathbf{Y}}$ auf einem Gitter über $[-6, 6] \times [-12, 24]$ mit Gitterweite $0.02$ aus und prüfen Sie, ob ihre Riemann-Summe nahe bei $1$ liegt. Vergleichen Sie den logarithmierten Determinantenwert an einigen zufälligen Punkten mit dem Logarithmus der Determinante einer Jacobi-Matrix, die Sie durch finite Differenzen approximieren.
    1. Die erste Koordinate durchläuft die Schicht unverändert. Welche Verteilung hat $Y_1$ unabhängig von der Wahl von $s$ und $t$? Wie muss eine zweite Kopplungsschicht angeordnet sein, damit die Verkettung beide Randverteilungen verändern kann? Welchen logarithmierten Determinantenwert hat die Verkettung?
    1. Flüsse können auch gleichverteiltes Rauschen in normalverteilte Variablen überführen. Seien $U_1, U_2$ unabhängig und gleichverteilt auf $(0, 1)$. Setzen Sie $R = \sqrt{-2\log U_1}$, $\Theta = 2\pi U_2$ und $(Z_1, Z_2) = (R\cos\Theta, R\sin\Theta)$. Bestimmen Sie mit der eindimensionalen Transformationsformel die Dichte von $R$ und anschließend die gemeinsame Dichte von $(R, \Theta)$. Wechseln Sie dann mithilfe der Jacobi-Determinante für Polarkoordinaten aus dem Text zu kartesischen Koordinaten und zeigen Sie, dass $Z_1$ und $Z_2$ unabhängige standardnormalverteilte Variablen sind. Dies ist die Box-Muller-Methode. Prüfen Sie mit $10^5$ Samples, ob die Stichprobenkovarianzmatrix von $(Z_1, Z_2)$ nahe an der Einheitsmatrix liegt.
   :exlabel:`ex_mdl-random-variables-affine-coupling`

<!--en:b526d670-->
[Diskussionen](https://d2l.discourse.group/t/36880)

<!--en:15cc7cd3-->
Gemeinsame Verteilungen und Transformationen: Bedingen, Kovarianz und Jacobi-Matrizen<br>**Gemeinsame Dichten und Randdichten · bedingte Dichten als Schnittprofile · Satz von der totalen Varianz · optimaler MSE-Prädiktor · Kovarianz und Korrelation · Variablentransformation mit der Jacobi-Determinante**

<!--en:b0049432-->
::: {.slide title="Abhängigkeit, Vorhersagbarkeit und Korrelation"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:4338afb3-->
- **Abhängig ($p_{X,Y} \ne p_X p_Y$):** Auf der Einheitskreisscheibe folgt aus der Beobachtung $Y = 0.9$ die Schranke $|X| \le \sqrt{1 - 0.9^2} \approx 0.44$.

<!--en:fcc63a7c-->
- **Im Mittel nicht vorhersagbar ($\mathbb{E}[X \mid Y = y] = 0$):** Jede horizontale Sehne liegt symmetrisch um $0$. Kenntnis von $Y$ verengt den Wertebereich von $X$, ohne seinen Mittelpunkt zu verschieben.

<!--en:3d6cf199-->
- **Unkorreliert ($\mathrm{Cov}(X, Y) = 0$):** Aus Unabhängigkeit folgt $\mathbb{E}[X \mid Y] = \mathbb{E}[X]$ und daraus wiederum $\mathrm{Cov}(X, Y) = 0$. **Keine der beiden Umkehrungen gilt.**

<!--en:4e7f515e-->
::: {.slide title="Gemeinsame Dichten als Wahrscheinlichkeit pro Flächeneinheit"}

<!--en:4bdfe6fd-->
[Multivariate Dichten]{.kicker}

<!--en:4e19c66b-->
- **Gemeinsame Dichte beim Dartwurf:** Konstant gleich $p_{X,Y}(x, y) = 1/\pi$ innerhalb der Einheitskreisscheibe $x^2 + y^2 \le 1$ (Fläche $\pi$) und gleich $0$ außerhalb.

<!--en:6ad82b1b-->
- **Warum konstant $\ne$ faktorisierbar:** Zwar ist $1/\pi$ innerhalb der Scheibe konstant, doch die Indikatorfunktion $\mathbf{1}_{x^2 + y^2 \le 1}$ der Kreisscheibe lässt sich nicht in der Form $a(x)\,b(y)$ faktorisieren.

<!--en:ce9d4ca6-->
::: {.slide title="Randdichten durch Ausintegrieren von Variablen"}

<!--en:ba44c9fe-->
[Summenregel für stetige Dichten]{.kicker}

<!--en:b927876b-->
- Für die Dichte von $X$ allein die Wahrscheinlichkeiten der kleinen Quadrate, jeweils $\epsilon^2 p_{X,Y}(x, \epsilon i)$, entlang des vertikalen Streifens bei $x$ addieren und $\epsilon$ kürzen:
  $$p_X(x) = \int_{-\infty}^{\infty} p_{X,Y}(x, y)\,dy$$

<!--en:b2ccc0a9-->
- **Randdichte beim Dartwurf als skalierter Halbkreis:** An der horizontalen Position $x \in [-1, 1]$ reicht der vertikale Streifen von $y = -\sqrt{1-x^2}$ bis $+\sqrt{1-x^2}$:
  $$p_X(x) = \int_{-\sqrt{1-x^2}}^{\sqrt{1-x^2}} \frac{1}{\pi}\,dy = \frac{2}{\pi}\sqrt{1 - x^2}$$

<!--en:e232389e-->
- Eine konstante gemeinsame Dichte ergibt eine gekrümmte Randdichte, weil Sehnen durch den Mittelpunkt länger sind als Sehnen nahe dem Rand.

<!--en:dffbbf46-->
::: {.slide title="Bedingte Dichten durch Schneiden und Normieren"}

<!--en:fa630499-->
[Bedingen auf Ereignisse mit Wahrscheinlichkeit null]{.kicker}

<!--en:98d4d46c-->
- Wegen $P(Y = y) = 0$ dividiert die diskrete Formel $P(X, Y)/P(Y)$ null durch null. Bedingen auf einen schmalen Streifen $Y \in [y, y+\epsilon]$ liefert im Grenzübergang $\epsilon \downarrow 0$ die **bedingte Dichte**:
  $$p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x, y)}{p_Y(y)} \quad \text{for } p_Y(y) > 0$$

<!--en:78b7673a-->
- **Geometrische Deutung:** Die Dichtefläche $p_{X,Y}$ entlang der horizontalen Geraden $Y = y$ schneiden und durch die Fläche $p_Y(y)$ unter dem Schnittprofil teilen. Damit ergibt das Integral $1$.

<!--en:fe1228dc-->
- **Bedingte Dichte beim Dartwurf:** Unter der Bedingung $Y = y$ ist $X$ auf der Sehne $[-\sqrt{1-y^2}, \sqrt{1-y^2}]$ mit Dichte $\frac{1}{2\sqrt{1-y^2}}$ gleichverteilt.

<!--en:4856a900-->
::: {.slide title="Kettenregel, Satz von Bayes und Unabhängigkeit"}

<!--en:8597ed3e-->
[Faktorisierung und Summe unabhängiger Variablen]{.kicker}

<!--en:6610dcf4-->
- **Unabhängigkeit ($X \perp Y$):** Gilt genau dann, wenn $p_{X,Y}(x, y) = p_X(x)\,p_Y(y)$ (z. B. für $4xy = (2x)(2y)$ auf $[0, 1]^2$, anders als für $1/\pi$ auf der Kreisscheibe).

<!--en:54e15681-->
- **Dichte einer Summe durch Faltung:** Für $X \perp Y$ liefert das Ausintegrieren von $(X, X+Y)$ die Dichte $p_{X+Y}(s) = \int_{-\infty}^\infty p_X(x)\,p_Y(s - x)\,dx$.

<!--en:bbc37509-->
::: {.slide title="Bedingter Erwartungswert und Turmeigenschaft"}

<!--en:1f4270a2-->
[Mittelung innerhalb der Streifen]{.kicker}

<!--en:e491b297-->
Der **bedingte Erwartungswert** $\mathbb{E}[X \mid Y = y]$ ist der Schwerpunkt der bedingten Dichte auf dem Schnitt $Y = y$:

<!--en:1e05aa85-->
- **Bekannte Faktoren herausziehen:** Im Streifen $Y = y$ verhält sich jede Funktion $h(Y)$ wie eine Konstante:
  $$\mathbb{E}\bigl[h(Y)\,g(X, Y) \mid Y\bigr] = h(Y)\,\mathbb{E}\bigl[g(X, Y) \mid Y\bigr]$$

<!--en:458ec80f-->
- **Satz vom totalen Erwartungswert (Turmeigenschaft):** Der Mittelwert der Streifenmittelwerte ergibt den Gesamtmittelwert:
  $$\mathbb{E}\bigl[\mathbb{E}[X \mid Y]\bigr] = \int \left(\int x\,p_{X \mid Y}(x \mid y)\,dx\right) p_Y(y)\,dy = \iint x\,p_{X,Y}(x, y)\,dx\,dy = \mathbb{E}[X]$$

<!--en:e9c045fd-->
::: {.slide title="Satz von der totalen Varianz (Eve’s law)"}

<!--en:453d5f49-->
[Varianz innerhalb der Streifen und zwischen den Streifen]{.kicker}

<!--en:0b61eb6a-->
- **Unerklärter Anteil (innerhalb):** $\mathbb{E}[\mathrm{Var}(X \mid Y)] = \mathbb{E}[X^2] - \mathbb{E}[(\mathbb{E}[X \mid Y])^2]$ misst die durchschnittliche Streuung, die innerhalb eines Schnitts $Y = y$ verbleibt.

<!--en:08cb816c-->
- **Erklärter Anteil (zwischen den Streifen):** $\mathrm{Var}(\mathbb{E}[X \mid Y]) = \mathbb{E}[(\mathbb{E}[X \mid Y])^2] - (\mathbb{E}[X])^2$ misst, wie stark die Mittelpunkte der Schnittprofile mit $Y$ variieren.

<!--en:1c3ad360-->
::: {.slide title="Der bedingte Erwartungswert minimiert den quadratischen Fehler"}

<!--en:4c5f8b1e-->
[Theoretische Grundlage der Regression]{.kicker}

<!--en:97614d92-->
- **Orthogonale Zerlegung:** Bei der Zerlegung $X - h(Y) = (X - m(Y)) + (m(Y) - h(Y))$ mit $m(Y) = \mathbb{E}[X \mid Y]$ verschwindet der gemischte Term im Mittel in jedem Streifen $Y = y$.

<!--en:93be25d7-->
- **Eindeutiger MSE-Minimierer und Obergrenze für $R^2$:** Mit $h(Y) = \mathbb{E}[X \mid Y]$ entfällt der zweite Term. Der erklärte Anteil ist beschränkt durch $R^2 \le \frac{\mathrm{Var}(\mathbb{E}[X \mid Y])}{\mathrm{Var}(X)}$, bezogen auf $\mathrm{Var}(X)$.

<!--en:9c4674c5-->
::: {.slide title="Der Satz von der totalen Varianz bei Dartwurf und Gauß-Mischung"}

<!--en:45d13783-->
[Analytische und empirische Varianzzerlegung]{.kicker}

<!--en:5516c273-->
- **Dartwurf auf die Scheibe ($0\%$ erklärt):** Jede Sehne auf Höhe $y$ ist symmetrisch ($\mathbb{E}[X \mid Y=y] = 0$). Die Gleichverteilung darauf hat die Varianz $(1 - y^2)/3$. Wegen $\mathbb{E}[Y^2] = \frac{1}{2}\mathbb{E}[R^2] = \frac{1}{4}$ gilt:
  $$\mathrm{Var}(X) = \frac{1 - 1/4}{3} + \mathrm{Var}(0) = \frac{1}{4} + 0$$

<!--en:c4671428-->
- **Mischung aus zwei Komponenten ($72\%$ erklärt):** Das Label $Z \sim \mathrm{Bernoulli}(0.2)$ bestimmt die Wahl zwischen $\mathcal{N}(3, 1)$ und $\mathcal{N}(-1, 1)$. Die Varianz innerhalb der Gruppen beträgt $1$, die zwischen den Gruppen $0.2(0.8)(3 - (-1))^2 = 2.56$ (insgesamt $3.56$).

<!--en:3ea5825c-->
Eine Simulation mit $10^6$ Ziehungen aus der Mischung bestätigt, dass der Satz von der totalen Varianz für die Stichprobe selbst *exakt* gilt:

<!--en:ca632152-->
::: {.slide title="Kovarianz als Maß für den linearen Zusammenhang"}

<!--en:e1f22bb2-->
[Bivariates Moment zweiter Ordnung]{.kicker}

<!--en:65852c40-->
- Die **Kovarianz** ist der Mittelwert des Produkts der Abweichungen von den beiden Mittelwerten:
  $$\sigma_{XY} = \mathrm{Cov}(X, Y) = \mathbb{E}\bigl[(X - \mu_X)(Y - \mu_Y)\bigr] = \mathbb{E}[XY] - \mu_X\mu_Y$$

<!--en:fa987398-->
- **Bilinearität und Verschiebungsinvarianz:**
  - $\mathrm{Cov}(X, X) = \mathrm{Var}(X)$
  - $\mathrm{Cov}(cX + d, Y) = c\,\mathrm{Cov}(X, Y)$
  - $\mathrm{Cov}(X + Z, Y) = \mathrm{Cov}(X, Y) + \mathrm{Cov}(Z, Y)$

<!--en:58f9b227-->
- **Abhängigkeit von der Skala:** Die Umrechnung von $Y$ von Dollar in Cent vergrößert $\mathrm{Cov}(X, Y)$ um den Faktor $100$, ohne die Form der Punktwolke zu verändern.

<!--en:834f2d65-->
::: {.slide title="Varianz einer Summe und die Monte-Carlo-Rate 1/√n"}

<!--en:53d6e699-->
[Warum Mittelung Rauschen reduziert]{.kicker}

<!--en:75c2caf3-->
- **Unabhängige Ziehungen ($\mathrm{Cov} = 0$):** $\mathrm{Var}(\bar{X}_n) = \sigma^2/n \implies \sigma_{\bar{X}_n} = \sigma/\sqrt{n}$ (für den halben Fehler sind $4\times$ so viele Samples nötig).

<!--en:03b7789d-->
- **Korreliertes Ensemble ($\mathrm{Corr} = \rho$):** $\mathrm{Var}(\bar{Y}_n) = \sigma^2\bigl(\rho + \frac{1-\rho}{n}\bigr) \to \rho\sigma^2$. Die Korrelation setzt der Varianz eine untere Grenze.

<!--en:ed0e8d0b-->
::: {.slide title="Korrelationskoeffizient und geometrische Deutung als Kosinus"}

<!--en:40a17cbd-->
[Skalenunabhängiger linearer Zusammenhang]{.kicker}

<!--en:b755e695-->
- Division durch beide Standardabweichungen kürzt die physikalischen Einheiten heraus und liefert den **Korrelationskoeffizienten**:
  $$\rho(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

<!--en:1baac071-->
- **Beweis von $|\rho| \le 1$ (Cauchy-Schwarz):** Für jedes $t \in \mathbb{R}$ gilt
  $$\mathrm{Var}(tX + Y) = \sigma_X^2 t^2 + 2\,\mathrm{Cov}(X, Y)t + \sigma_Y^2 \ge 0$$
  Ein nichtnegatives quadratisches Polynom hat die Diskriminante $\Delta \le 0$. Daraus folgt $\mathrm{Cov}(X,Y)^2 \le \sigma_X^2\sigma_Y^2$, wobei $\rho = \pm 1 \iff Y = cX + d$ gilt.

<!--en:00325cd2-->
- **Deutung im Hilbertraum:** Für das innere Produkt $\langle \tilde{X}, \tilde{Y} \rangle = \mathbb{E}[\tilde{X}\tilde{Y}]$ ist die Korrelation genau $\cos\angle(\tilde{X}, \tilde{Y})$.

<!--en:9b59911a-->
::: {.slide title="Warum unkorrelierte Variablen abhängig sein können"}

<!--en:2ee40c19-->
[Drei klassische Gegenbeispiele]{.kicker}

<!--en:c41a5a9e-->
- **Warum $\mathrm{Cov}(X, Y) = 0$ schwächer ist als $X \perp Y$:** Kovarianz null schließt nur eine *lineare* Steigung aus. Bei symmetrischer nichtlinearer Kopplung gilt weiterhin $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$.

<!--en:2b5376b4-->
- **Drei Gegenbeispiele:** Kreisscheibe ($x^2+y^2 \le 1$), Parabel ($Y = X^2$) und normalverteilte Variable mit zufälligem Vorzeichen ($Y = SX$ mit $|Y| = |X|$).

<!--en:e3bdfde7-->
- **Folge für das Repräsentationslernen:** Lineares PCA-Whitening ($\boldsymbol{\Sigma} = \mathbf{I}$) beseitigt lineare Korrelationen. Unabhängigkeit der Merkmale ist dabei nur garantiert, wenn sie *gemeinsam* normalverteilt sind.

<!--en:3f7d28dc-->
::: {.slide title="Kovarianzmatrix und Varianz in einer Richtung"}

<!--en:54563a61-->
[Zweite Momente von Zufallsvektoren]{.kicker}

<!--en:ce4fdf05-->
Für einen Zufallsvektor $\mathbf{X} = (X_1, \dots, X_n)^\top$ mit Mittelwertvektor $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$ enthält die $n \times n$-**Kovarianzmatrix** die Varianzen auf der Diagonalen und die paarweisen Kovarianzen außerhalb der Diagonalen:

<!--en:d68df033-->
- **Symmetrie:** $\boldsymbol{\Sigma}^\top = \boldsymbol{\Sigma}$, denn $\mathrm{Cov}(X_i, X_j) = \mathrm{Cov}(X_j, X_i)$.

<!--en:4af6ccf7-->
- **Positive Semidefinitheit ($\boldsymbol{\Sigma} \succeq \mathbf{0}$):** Die Projektion von $\mathbf{X}$ auf einen festen Richtungsvektor $\mathbf{a} \in \mathbb{R}^n$ ergibt eine skalare Zufallsvariable $\mathbf{a}^\top \mathbf{X}$. Ihre Varianz ist die quadratische Form
  $$\mathrm{Var}(\mathbf{a}^\top \mathbf{X}) = \mathbb{E}\bigl[\mathbf{a}^\top (\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top \mathbf{a}\bigr] = \mathbf{a}^\top \boldsymbol{\Sigma}\mathbf{a} \ge 0$$

<!--en:e1903941-->
- **Spektrale Deutung:** Die Eigenvektoren von $\boldsymbol{\Sigma}$ zeigen in die Hauptachsenrichtungen der Varianz (PCA). Für den symmetrischen Dartwurf auf die Einheitskreisscheibe gilt $\boldsymbol{\Sigma} = \mathrm{diag}(1/4, 1/4)$.

<!--en:db475f56-->
::: {.slide title="Eindimensionale Variablentransformation und Erhaltung der Wahrscheinlichkeit"}

<!--en:f55ac4bd-->
[Wie Transformationen eine Dichte verändern]{.kicker}

<!--en:01f8d489-->
- Wenn $X$ die Dichte $p_X$ hat und $Y = g(X)$ für eine glatte, streng monotone Abbildung $g$ gilt, warum ist dann $p_Y(y) \ne p_X(g^{-1}(y))$?

<!--en:048de1b2-->
- **Die Wahrscheinlichkeitsmasse bleibt erhalten, die Höhe der Dichte kann sich ändern:** Ein Intervall der Breite $\Delta x$ trägt die Wahrscheinlichkeit $p_X(x)\Delta x$ und wird auf die Breite $\Delta y \approx |g'(x)|\Delta x$ gestreckt:
  $$p_Y(y)\,\Delta y \approx p_X(x)\,\Delta x \implies p_Y(y) = \frac{p_X(x)}{|g'(x)|}$$

<!--en:f12fc474-->
- **Eindimensionale Transformationsformel:** Auf dem Bildintervall $g(I)$ gilt
  $$p_Y(y) = p_X\bigl(g^{-1}(y)\bigr)\left|\frac{dg^{-1}}{dy}(y)\right|$$

<!--en:999ea26a-->
::: {.slide title="Beispiel: Lognormalverteilung und naives Einsetzen"}

<!--en:98f6229e-->
[Warum der Jacobi-Faktor 1/y das Maximum verschiebt]{.kicker}

<!--en:42b21652-->
- Sei $X \sim \mathcal{N}(0, 1)$ und $Y = e^X > 0$. Hier gilt $g^{-1}(y) = \log y$ und $\left|\frac{dg^{-1}}{dy}\right| = \frac{1}{y}$. Daraus folgt die Dichte der **Lognormalverteilung**:
  $$p_Y(y) = \frac{1}{y\sqrt{2\pi}}\exp\!\left(-\frac{(\log y)^2}{2}\right), \quad y > 0$$

<!--en:1cc2b201-->
- **Verschiebung des Maximums:** Ableiten von $\log p_Y(y) = -\log y - \frac{1}{2}(\log y)^2 + C$ liefert den Modus **$y = e^{-1} \approx 0.368$**.

<!--en:6c8d3b92-->
- **Fehler beim naiven Ansatz $p_X(\log y)$:** Ohne $1/y$ liegt das Maximum bei $y = 1$ und die Gesamtfläche beträgt $\int p_X(x)e^x\,dx = \mathbb{E}[e^X] = e^{1/2} \approx 1.65 \ne 1$.

<!--en:4921b61c-->
::: {.slide title="Multivariate Variablentransformation und normalisierende Flüsse"}

<!--en:bf9e1d7d-->
[Jacobi-Determinanten in tiefen generativen Modellen]{.kicker}

<!--en:93b71244-->
- **Lokale Volumenstreckung:** Eine invertierbare Abbildung $\mathbf{y} = g(\mathbf{x})$ skaliert das Volumen mit $|\det J_g(\mathbf{x})|$. Bei $L$ verketteten Flussschichten verringert sich die Log-Dichte um $\sum_{\ell=1}^L \log|\det J_{g_\ell}|$.

<!--en:6aec98ee-->
- **Affine Kopplung mit Dreiecksstruktur (RealNVP):** Für $y_1 = x_1$ und $y_2 = x_2 e^{s(x_1)} + t(x_1)$ ist $J_g$ eine untere Dreiecksmatrix. Der Rechenaufwand für die logarithmierte Determinante beträgt $O(1)$, ihr Wert ist $s(x_1)$.

<!--en:33f16e0f-->
::: {.slide title="Koordinatenwahl beim Dartwurf: Polarkoordinaten"}

<!--en:0ba0afbc-->
[Auflösung der Abhängigkeit mit der Jacobi-Determinante]{.kicker}

<!--en:cbf6bb36-->
Die kartesische Position $(x, y)$ des Dartpfeils durch die Polarkoordinaten $(r, \theta)$ ausdrücken, mit $(x, y) = (r\cos\theta, r\sin\theta)$:

<!--en:7fa24fdb-->
- **Multivariate Variablentransformation** von $(x, y)$ (mit $p_{X,Y} = 1/\pi$) nach $(r, \theta) \in [0, 1] \times [0, 2\pi)$:
  $$p_{R,\Theta}(r, \theta) = p_{X,Y}(r\cos\theta, r\sin\theta)\,\bigl|\det J_g(r, \theta)\bigr| = \frac{1}{\pi}\cdot r = \underbrace{2r}_{p_R(r)} \cdot \underbrace{\frac{1}{2\pi}}_{p_\Theta(\theta)}$$

<!--en:41aa91ae-->
- **Faktorisierung in unabhängige Koordinaten:** In kartesischen Koordinaten koppelt der Kreisrand $X$ und $Y$. In Polarkoordinaten faktorisiert die gemeinsame Dichte, also gilt $R \perp \Theta$ mit $p_R(r) = 2r$ und $\Theta \sim \mathrm{Unif}[0, 2\pi)$.

<!--en:962322f8-->
::: {.slide title="Verständnisfragen zu bedingten Verteilungen, Varianz und Jacobi-Determinanten"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:6486c975-->
Drei Aufgaben zu gemeinsamen Verteilungen und zur Transformation von Dichten:

<!--en:7fa350b5-->
1. **Obergrenze für die erklärte Varianz ($R^2$):** Welchen Höchstwert von $R^2 = 1 - \frac{\mathbb{E}[(X - h(Y))^2]}{\mathrm{Var}(X)}$ kann ein Modell $h(Y)$ bei der Vorhersage der Dartkoordinate $X$ aus $Y$ erreichen? Den Satz von der totalen Varianz und die Fehlerzerlegung für den optimalen Prädiktor nutzen.

<!--en:0032e632-->
2. **Kleinste Korrelation von $n$ austauschbaren Variablen:** Die Variablen $Y_1, \dots, Y_n$ haben jeweils Varianz $\sigma^2$ und paarweise Korrelation $\rho$. Mit $\mathrm{Var}(\sum_{i=1}^n Y_i) \ge 0$ die kleinstmögliche negative Korrelation $\rho$ bestimmen.

<!--en:9256bd60-->
3. **Unbeschränkte transformierte Dichte:** Sei $U \sim \mathrm{Unif}(0, 1)$. Welche Dichte $p_Y(y)$ hat $Y = U^2$ auf $(0, 1)$? Warum wächst sie nahe $y = 0$ über alle Grenzen, obwohl ihr Integral $1$ ergibt?

<!--en:557b98b6-->
**Lösungen:**

<!--en:e7bfca7d-->
1. $R^2 \le \frac{\mathrm{Var}(\mathbb{E}[X \mid Y])}{\mathrm{Var}(X)}$. Beim Dartwurf ist $\mathbb{E}[X \mid Y] = 0$ konstant, also gilt $\mathrm{Var}(\mathbb{E}[X \mid Y]) = 0$. Damit beträgt die Obergrenze **$R^2 = 0$** für *jeden* Prädiktor $h(Y)$.

<!--en:9713c9b2-->
2. $\mathrm{Var}(\sum_{i=1}^n Y_i) = n\sigma^2(1 + (n-1)\rho) \ge 0 \implies \boldsymbol{\rho \ge -\frac{1}{n-1}}$ (erreicht für $\sum_{i=1}^n Y_i = 0$, z. B. liefert $Y_2 = -Y_1$ die Korrelation $\rho = -1$ für $n=2$).

<!--en:4bc2abf6-->
3. $g^{-1}(y) = \sqrt{y}$ hat die Ableitung $\frac{1}{2\sqrt{y}}$, also gilt **$p_Y(y) = \frac{1}{2\sqrt{y}}$** auf $(0, 1)$. Das Quadrieren staucht Intervalle nahe $0$ stark. Daher wächst die Dichte gegen $\infty$ für $y \downarrow 0$, obwohl $\int_0^1 \frac{1}{2\sqrt{y}}\,dy = [\sqrt{y}]_0^1 = 1$ gilt.

<!--en:c9d5841f-->
::: {.slide title="Zusammenfassung: gemeinsame Verteilungen und Transformationen"}

<!--en:c5a222f8-->
[Wesentliche Ergebnisse]{.kicker}

<!--en:ec071c21-->
| Operation oder Satz | Mathematische Formel | Dartwurf oder Rechenbeispiel | Rolle im Deep Learning |

<!--en:bf9dc3dc-->
| **Marginalisierung** | $p_X(x) = \int_{-\infty}^\infty p_{X,Y}(x, y)\,dy$ | Konstante Dichte auf der Scheibe $1/\pi \to$ skalierter Halbkreis $\frac{2}{\pi}\sqrt{1-x^2}$ | Ausintegrieren latenter Variablen |

<!--en:b21f57ff-->
| **Bedingen und Unabhängigkeit** | $p_{X\mid Y}(x\mid y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$, $X \perp Y \iff p_{X,Y} = p_X p_Y$ | Gleichverteilt auf der Sehne $\|x\| \le \sqrt{1-y^2}$ (abhängig) | Diskriminative und autoregressive Modelle |

<!--en:fd457a2f-->
| **Optimaler MSE-Prädiktor** | $\mathbb{E}[(X-h(Y))^2] = \mathbb{E}[\mathrm{Var}(X\mid Y)] + \mathbb{E}[(\mathbb{E}[X\mid Y]-h(Y))^2]$ | Bester Prädiktor $h^\star(Y) = \mathbb{E}[X \mid Y]$ | Training neuronaler Netze mit $L_2$-Verlust |

<!--en:cca8909e-->
::: {.slide title="Zusammenfassung: gemeinsame Verteilungen und Transformationen (Fortsetzung)"}

<!--en:c5a222f8-2-->
[Wesentliche Ergebnisse]{.kicker}

<!--en:ec071c21-2-->
| Operation oder Satz | Mathematische Formel | Dartwurf oder Rechenbeispiel | Rolle im Deep Learning |

<!--en:43b91bb6-->
| **Satz von der totalen Varianz** | $\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X \mid Y)] + \mathrm{Var}(\mathbb{E}[X \mid Y])$ | Mischung: $1.00\text{ (within)} + 2.56\text{ (between)} = 3.56$ | Varianzanalyse, Obergrenze für $R^2$ und Zerlegung des Rauschens |

<!--en:67f30abd-->
| **Kovarianz und Varianz einer Summe** | $\mathrm{Cov}(X,Y) = \mathbb{E}[XY]-\mu_X\mu_Y$, $\mathrm{Var}(\bar{X}_n) = \sigma^2/n$ | $\mathrm{Cov}(X,Y)=0$ auf der Kreisscheibe trotz Abhängigkeit | Monte-Carlo-Fehler $\sigma/\sqrt{n}$ und Vielfalt im Ensemble |

<!--en:d9aba8ea-->
| **Kovarianzmatrix $\boldsymbol{\Sigma}$** | $\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^\top]$, $\mathrm{Var}(\mathbf{a}^\top\mathbf{X}) = \mathbf{a}^\top\boldsymbol{\Sigma}\mathbf{a} \ge 0$ | Kreisscheibe: $\boldsymbol{\Sigma} = \mathrm{diag}(1/4, 1/4)$ | PCA, Whitening von Merkmalen und Mahalanobis-Metriken |

<!--en:2997e9d5-->
| **Dichtetransformation mit der Jacobi-Determinante** | $\log p_{\mathbf{Y}}(\mathbf{y}) = \log p_{\mathbf{X}}(\mathbf{x}) - \log\|\det J_g(\mathbf{x})\|$ | Maximum der Lognormaldichte bei $e^{-1}$, in Polarkoordinaten $p_{R,\Theta} = 2r \cdot \frac{1}{2\pi}$ | Normalisierende Flüsse und reparametrisierte Dichten |
