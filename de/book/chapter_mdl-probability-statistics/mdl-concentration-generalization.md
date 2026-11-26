<!--en:55ea9050-->
# Konzentrationsungleichungen

<!--en:007949ca-->
Wenn wir die Testgenauigkeit eines Modells angeben, beruht unsere Angabe auf einer Messung an einer endlichen Stichprobe. In :numref:`sec_mdl-statistics` haben wir mithilfe des zentralen Grenzwertsatzes einen Fehlerbalken für diese Messung bestimmt, nämlich $\pm1.96\,\hat\sigma/\sqrt n$. Die Gauß-Näherung ist praktisch und oft genau. Exakt ist sie aber erst im Grenzfall $n\to\infty$, und für die Standardabweichung setzen wir einen Schätzwert $\hat\sigma$ ein. In :numref:`sec_mdl-random_variables` haben wir bereits die Chebyshev-Ungleichung kennengelernt. Sie gilt ohne Näherung für jedes endliche $n$, erwies sich aber als sehr schwach. Wie weit kann also der Fehler auf dem Testdatensatz vom wahren Fehler abweichen? Wir suchen eine Garantie, die für den tatsächlich verfügbaren Stichprobenumfang gilt und selbst dann aussagekräftig bleibt, wenn sie nur äußerst selten versagen darf.

<!--en:f30d535d-->
Betrachten wir einen Klassifikator, der bei 10 000 Testbildern 830 Fehler macht. Seine beobachtete Fehlerrate beträgt 8.3%. Nach der Chebyshev-Ungleichung weicht der Mittelwert von $n$ unabhängigen Ziehungen von seinem Erwartungswert um mindestens $t$ ab, und zwar mit einer Wahrscheinlichkeit von höchstens $\sigma^2/(nt^2)$. Da die Varianz eines 0/1-Ergebnisses höchstens ein Viertel beträgt ($\sigma^2\leq\tfrac14$), garantiert die Chebyshev-Ungleichung eine wahre Fehlerrate im Bereich von $\pm2.2\%$ um 8.3% bei 95% Konfidenz. Schon dieses Intervall ist breiter als das Gauß-Intervall. Verlangen wir eine höhere Zuverlässigkeit, wird die Schranke vollends unbrauchbar. Für eine Garantie, die nur in einem von einer Million Fällen versagt, liefert die Chebyshev-Ungleichung $\pm500\%$. Damit erfahren wir gar nichts. Tatsächlich fällt die Wahrscheinlichkeit, dass der Mittelwert von 0/1-Ergebnissen weit vom Erwartungswert abweicht, *exponentiell* mit $n$ und nicht nur wie $1/n$. Eine Schranke, die diesen exponentiellen Abfall erfasst, liefert $\pm1.4\%$ bei 95% Konfidenz und $\pm2.7\%$ bei einer Versagenswahrscheinlichkeit von eins zu einer Million.

<!--en:5d0fd0fb-->
Der entscheidende Schritt zu diesen schärferen Schranken ist überraschend einfach. Wir wenden die Markov-Ungleichung auf die Exponentialfunktion des Mittelwerts an. So wird aus einer Summe unabhängiger Terme ein Produkt. Sergei Bernstein entwickelte solche Schranken in den 1920er-Jahren. Herman Chernoff untersuchte 1952 mit dieser Methode, wie schnell statistische Tests zuverlässig werden. Wassily Hoeffding gab 1963 die einfache Form an, die im maschinellen Lernen weit verbreitet ist :cite:`Hoeffding.1963`.

<!--en:032028a3-->
Warum sollte man in der Praxis eine Garantie verlangen, die nur in einem von einer Million Fällen versagt? Wir bewerten selten nur ein Modell. Ein Lernalgorithmus vergleicht viele Kandidaten und wählt den besten aus. Um zu garantieren, dass *keiner* von ihnen nur durch Zufall gut abschneidet, brauchen wir für jeden einzelnen eine winzige Versagenswahrscheinlichkeit. Bei realistischem Stichprobenumfang gelingt das nur, wenn die Randwahrscheinlichkeit exponentiell fällt. Diesen Zusammenhang untersuchen wir in :numref:`sec_mdl-generalization-bounds` ausführlich. In diesem Abschnitt bestimmen wir zunächst mit exponentiellen Schranken für Randwahrscheinlichkeiten (engl. tail bounds), wie groß ein Testdatensatz für eine gewünschte Schätzgenauigkeit sein muss. Dann wenden wir dasselbe Verfahren auf die Koordinaten eines Zufallsvektors in $\mathbb{R}^d$ an. Daraus ergeben sich zwei bemerkenswerte geometrische Eigenschaften. Die Länge eines standardnormalverteilten Vektors in $\mathbb{R}^{784}$ weicht fast immer nur um wenige Prozent von 28 ab, und zwei unabhängige Zufallsrichtungen sind nahezu orthogonal. Beide Eigenschaften sind grundlegend für die Initialisierung neuronaler Netze und für hochdimensionale Einbettungen.

<!--en:a6ba4e13-->
Drei Ergebnisse aus diesem Abschnitt sollten Sie sich merken. Es sind die Chernoff-Methode :eqref:`eq_mdl-chernoff`, das Hoeffding-Konfidenzintervall :eqref:`eq_mdl-hoeffding-interval` und die Konzentration der euklidischen Norm :eqref:`eq_mdl-norm-concentration`.

<!--en:b33a5b84-->
## Exponentielle Schranken für Randwahrscheinlichkeiten

<!--en:807af3c1-->
Das Gauß-Konfidenzintervall ist schmal, weil die Randwahrscheinlichkeiten der Normalverteilung wie $e^{-z^2/2}$ fallen. Für den Stichprobenmittelwert liefert der zentrale Grenzwertsatz diese Rate erst im Grenzfall großer $n$. Wir suchen denselben schnellen Abfall mit einer Garantie, die bei jedem endlichen Stichprobenumfang gilt. Bevor wir eine solche Schranke herleiten, sehen wir uns an, wie weit die Chebyshev-Ungleichung vom wahren Wert entfernt liegt.

<!--en:7e9d791b-->
### Die Abfallrate der Chebyshev-Schranke

<!--en:5218419a-->
Die Chebyshev-Schranke ergibt sich aus der Markov-Ungleichung (:numref:`sec_mdl-random_variables`), angewandt auf eine quadrierte Abweichung. Betrachten wir den Stichprobenmittelwert $\bar X$ aus $n$ unabhängigen Ziehungen mit jeweils gleichem Erwartungswert $\mu$ und gleicher endlicher Varianz $\sigma^2$. Für die erwartete quadrierte Abweichung gilt $\mathbb{E}[(\bar X-\mu)^2]=\textrm{Var}(\bar
X)=\sigma^2/n$. Die Markov-Ungleichung liefert daher

<!--en:cf6871db-->
Hier fällt die Schranke für die Wahrscheinlichkeit einer solchen Abweichung nur wie $1/n$. Um die Versagenswahrscheinlichkeit von $1$ auf $10^{-6}$ zu senken, müssen wir den Stichprobenumfang bei fester Abweichung $t$ auf das Millionenfache erhöhen.

<!--en:fa242b33-->
Wie klein ist die Wahrscheinlichkeit einer solchen Abweichung tatsächlich? Am einfachsten lässt sie sich für eine faire Münze exakt berechnen. Sei $\hat p$ der Anteil der Würfe mit Kopf unter $n$ unabhängigen Würfen. Wie wahrscheinlich ist es, dass $\hat p$ von $\tfrac12$ um mindestens $t=0.1$ abweicht? Die Varianz jedes Wurfs beträgt $\tfrac14$, die Chebyshev-Schranke also $0.25/(0.01\,n)=25/n$. Die exakte Wahrscheinlichkeit ist eine Summe von Binomialwahrscheinlichkeiten. Wir berechnen sie im logarithmischen Raum, um bei $n=5{,}000$ einen Unterlauf auf null zu vermeiden.

<!--en:92f4ff83-->
Bei $n=1{,}000$ lässt die Chebyshev-Schranke eine solche Abweichung in einem von vierzig Experimenten zu. Schätzen Sie vor dem Blick auf die Tabelle, wie oft sie tatsächlich auftritt.

<!--en:8384497e-->
Bei $n=10$ beträgt die exakte Randwahrscheinlichkeit $0.75$. Die Schranke von $2.5$ sagt hier nichts aus. Bei $n=100$ sinkt die Schranke auf $0.25$ und beträgt damit etwa das Vierfache des exakten Werts von $0.057$. Danach gehen die beiden Kurven weit auseinander. Bei $n=1{,}000$ beträgt die exakte Randwahrscheinlichkeit $2.7\times10^{-10}$, während die Chebyshev-Schranke mit $0.025$ acht Größenordnungen darüber liegt. Bei $n=5{,}000$ sind es bereits zweiundvierzig Größenordnungen. Verzehnfachen wir $n$, so sinkt die Chebyshev-Schranke nur um den Faktor zehn. Der Logarithmus der exakten Randwahrscheinlichkeit hingegen fällt mit jeweils hundert weiteren Würfen um einen festen Betrag. Auf einer logarithmischen Skala ergibt sich eine abfallende Gerade.

<!--en:6a2cf2c9-->
Ohne weitere Annahmen über die Verteilung lässt sich die Chebyshev-Schranke leider nicht verschärfen. Wie wir in :numref:`sec_mdl-random_variables` gesehen haben, gilt für eine Verteilung auf drei Punkten Gleichheit. Eine Schranke, die allein den Erwartungswert und die Varianz nutzt, kann daher im Allgemeinen nicht schärfer sein. Von einem Münzwurf wissen wir aber weit mehr, als dass seine ersten beiden Momente endlich sind. Seine Werte sind beschränkt, und damit bleiben auch alle höheren Momente klein. Um alle Momente zugleich zu nutzen, betrachten wir die momenterzeugende Funktion (engl. moment generating function, MGF).

<!--en:964149b6-->
### Die Chernoff-Methode

<!--en:759d4665-->
Wie lassen sich höhere Momente nutzen? Betrachten wir dazu die Gemeinsamkeit der Markov- und der Chebyshev-Ungleichung. Ist eine Funktion $g$ überall mindestens so groß wie die Indikatorfunktion $\mathbf 1\{x\geq t\}$, so heißt sie *Majorante* der Indikatorfunktion. Bilden wir die Erwartungswerte, folgt unmittelbar $P(X\geq t)=\mathbb{E}[\mathbf 1\{X\geq t\}]\leq\mathbb{E}[g(X)]$. Die Markov-Ungleichung verwendet die lineare Majorante $g(x)=x/t$ für ein nichtnegatives $X$, die Chebyshev-Ungleichung die quadratische Majorante $g(x)=(x/t)^2$ für die zentrierte Abweichung $X-\mu$. Für jedes $\lambda>0$ liegt auch die Exponentialfunktion $e^{\lambda(x-t)}$ auf der gesamten reellen Achse mindestens auf Höhe der Indikatorfunktion. Teil (a) von :numref:`fig_mdl-conc-majorants` vergleicht alle drei Majoranten für ein nichtnegatives $X$. Die Schranke ist der Erwartungswert von $g(X)$. Wie scharf sie ist, hängt also davon ab, wie klein $g$ dort bleibt, wo die Wahrscheinlichkeitsmasse von $X$ liegt.

<!--en:187c8f2f-->
![Eine Schranke für die Randwahrscheinlichkeit ist der Erwartungswert einer Majorante der Indikatorfunktion. (a) Für die Schwelle $t=2$ zeigt die Abbildung die Indikatorfunktion $\mathbf{1}\{x\geq t\}$ (schwarz) und drei Majoranten, nämlich $x/t$ nach Markov, $(x/t)^2$ nach Chebyshev und $e^{\lambda(x-t)}$ nach Chernoff mit $\lambda=1.5$. Die schattierte Erhebung zeigt, wo die Wahrscheinlichkeitsmasse von $X$ liegt. Jede Schranke ist der Mittelwert der jeweiligen Funktion bezüglich dieser Verteilung. Erhöhen wir $\lambda$, wird die Exponentialfunktion dort klein. (b) Der Chernoff-Exponent $-\lambda nt+n\lambda^2/8$ für den Mittelwert von $n=100$ fairen Münzwürfen bei $t=0.1$. Jedes $\lambda>0$ liefert eine gültige Schranke. Am Minimum der Parabel bei $\lambda=4t$ ergibt sich die schärfste Schranke $e^{-2}$.](../img/mdl-prob-conc-majorants.svg)

<!--en:f86ac4d1-->
Der Erwartungswert dieser Exponentialfunktion hat einen eigenen Namen. Die *momenterzeugende Funktion* (MGF) von $X$ ist der Erwartungswert $\mathbb{E}[e^{\lambda X}]$ als Funktion von $\lambda$, überall dort, wo dieser Erwartungswert endlich ist. Für einen um null zentrierten fairen Münzwurf $Z\in\{-\tfrac12,\tfrac12\}$ lautet sie $\tfrac12(e^{\lambda/2}+e^{-\lambda/2})=\cosh(\lambda/2)$. Die MGF ist für jedes $\lambda$ endlich, wenn $X$ beschränkt ist. Sie ist aber für jedes $\lambda>0$ unendlich, wenn die rechte Randwahrscheinlichkeit nur polynomiell fällt, etwa bei einer Student-$t$-verteilten Zufallsvariablen. Für solche Zufallsvariablen mit schweren Verteilungsrändern ist die nachfolgende Exponentialmethode nicht anwendbar.

<!--en:d407409d-->
Wenden wir die Markov-Ungleichung auf die nichtnegative Zufallsvariable $e^{\lambda X}$ an, erhalten wir die Schranke in einer Zeile. Da $x\mapsto e^{\lambda x}$ für $\lambda>0$ streng monoton wächst, stimmt das Ereignis $\{X\geq t\}$ mit $\{e^{\lambda X}\geq e^{\lambda t}\}$ überein. Daher gilt

<!--en:ac5d0580-->
Jedes $\lambda>0$ liefert eine gültige obere Schranke. Mit dem Infimum erhalten wir die schärfste davon. Dieses zweistufige Verfahren heißt *Chernoff-Methode*.

<!--en:e8e3cfd0-->
Für die Wahl der Exponentialfunktion als Majorante gibt es einen zweiten, ebenso wichtigen Grund. Ist $S=X_1+\cdots+X_n$ eine Summe unabhängiger Zufallsvariablen, so ist $e^{\lambda S}=\prod_{i=1}^n e^{\lambda X_i}$ ein Produkt unabhängiger Faktoren. Der Erwartungswert eines Produkts unabhängiger Zufallsvariablen ist das Produkt ihrer Erwartungswerte.

<!--en:fe91d47f-->
Die momenterzeugende Funktion einer Summe unabhängiger Zufallsvariablen ist also das Produkt der MGFs ihrer Summanden. Damit zerfällt der Logarithmus der Chernoff-Schranke in eine Summe aus $n$ Termen und skaliert linear mit $n$. Keine polynomielle Majorante hat diese Eigenschaft. So lässt sich $\mathbb{E}[S^4]$ nicht in ein Produkt vierter Momente zerlegen. Wir brauchen also nur noch eine Schranke für die MGF eines einzelnen Summanden.

<!--en:f7e3524a-->
### Die faire Münze

<!--en:a275b2d1-->
Rechnen wir das Beispiel der fairen Münze bis zum Ende durch, denn schon dieser einfachste Fall enthält jeden Schritt des allgemeinen Beweises. Ein zentrierter Münzwurf $Z=X-\tfrac12$ nimmt die Werte $\pm\tfrac12$ mit gleicher Wahrscheinlichkeit an. Seine MGF lautet daher $\mathbb{E}[e^{\lambda Z}]=\cosh(\lambda/2)$. Vergleichen wir die Taylorreihen von $\cosh u$ und $e^{u^2/2}$ gliedweise, erhalten wir

<!--en:0f0b4527-->
denn es gilt $(2k)!\geq2\cdot4\cdots(2k)=2^kk!$. Schon die geraden Faktoren von $(2k)!$ ergeben $2^kk!$, und jeder ungerade Faktor ist mindestens $1$. Setzen wir $u=\lambda/2$, so erfüllt ein einzelner zentrierter Wurf die Ungleichung $\mathbb{E}[e^{\lambda Z}]\leq e^{\lambda^2/8}$ für jedes $\lambda$. Zum Vergleich gilt für eine normalverteilte Zufallsvariable mit Verteilung $\mathcal N(0,\sigma^2)$ die Gleichung $\mathbb{E}[e^{\lambda X}]=e^{\lambda^2\sigma^2/2}$, wie die quadratische Ergänzung im Integral zeigt. Die MGF eines zentrierten Münzwurfs ist also nie größer als die einer normalverteilten Zufallsvariablen mit Standardabweichung $\tfrac12$.

<!--en:45ee01a3-->
Betrachten wir nun $n$ unabhängige Würfe. Das Ereignis $\hat p-\tfrac12\geq t$ ist gleichbedeutend mit $\sum_{i=1}^n Z_i\geq nt$. Aus der Chernoff-Methode und der Produktregel folgt daher

<!--en:11afdb44-->
Der Exponent beschreibt als Funktion von $\lambda$ eine Parabel. Teil (b) von :numref:`fig_mdl-conc-majorants` zeigt sie für $n=100$ und $t=0.1$. Der Exponent lautet dann $-10\lambda+12.5\lambda^2$. Jedes $\lambda>0$ liefert eine gültige Schranke, etwa $e^{-0.875}\approx0.42$ bei $\lambda=0.1$. Die schärfste Schranke erhalten wir am Scheitelpunkt der Parabel bei $\lambda=4t$. Dort vereinfacht sich der Exponent zu $-4nt^2+2nt^2=-2nt^2$. Wegen der Symmetrie gilt dieselbe Schranke für die untere Randwahrscheinlichkeit. Addieren wir die beiden Randwahrscheinlichkeiten, erhalten wir

<!--en:c3a2cba4-->
Die Wahrscheinlichkeit einer Abweichung um mindestens $t$ fällt nun exponentiell mit $nt^2$. Bei $n=100$ und $t=0.1$ erhalten wir $2e^{-2}\approx0.27$. Bei $n=1{,}000$ sinkt die Schranke auf $2e^{-20}\approx4.1\times10^{-9}$ und ist damit erheblich schärfer als die Chebyshev-Schranke von $0.025$.

<!--en:85d63a19-->
Warum funktioniert die exponentielle Majorante so gut? An der Schwelle $S=nt$ hat die Funktion $e^{\lambda(S-nt)}$ den Wert $1$. Am Erwartungswert $S=0$, in dessen Nähe fast die gesamte Wahrscheinlichkeitsmasse liegt, sinkt ihr Wert auf $e^{-\lambda nt}$ und ist damit exponentiell klein in $n$. Warum lassen wir $\lambda$ nicht gegen unendlich gehen? Für $S>0$ wächst die Funktion steil an, sodass Schwankungen nach oben ein unverhältnismäßig großes Gewicht erhalten. Das Produkt der MGFs misst diesen Effekt und ist durch $e^{n\lambda^2/8}$ beschränkt. Der optimale Wert $\lambda=4t$ gleicht die beiden Effekte genau aus.

<!--en:e45ea395-->
Die gesamte Herleitung stützt sich auf eine einzige Eigenschaft der Münze. Die MGF eines zentrierten Wurfs ist durch $e^{\lambda^2/8}$ beschränkt, die MGF einer normalverteilten Zufallsvariablen mit Standardabweichung $\tfrac12$. Diese entspricht der Hälfte der Spannweite der Wurfergebnisse. Das Hoeffding-Lemma verallgemeinert diesen einen Schritt. Für *jede* Zufallsvariable mit Werten im Intervall $[a,b]$ gilt nach dem Zentrieren dieselbe Schranke mit $(b-a)/2$ anstelle von $\tfrac12$.

<!--en:a77781d1-->
### Hoeffding-Lemma und Hoeffding-Ungleichung

<!--en:3e396575-->
Der Verlust auf dem Testdatensatz verhält sich selten wie ein fairer Münzwurf. Eine Fehlerrate von 8.3% entspricht einer stark gezinkten Münze, und ein auf ein Intervall begrenzter Verlust kann ein Kontinuum von Werten annehmen statt nur $0$ und $1$. Wir möchten die exponentielle Schranke für die faire Münze auf *beliebige* beschränkte Zufallsvariablen übertragen. Das folgende Lemma leistet das und ist der wichtigste technische Schritt dieses Abschnitts. Sein Beweis beruht auf einer elementaren Eigenschaft der Varianz.

<!--en:fe825fae-->
**Proposition (Varianz einer beschränkten Zufallsvariablen).** *Nimmt $Y$ Werte in $[a,b]$ an, so gilt $\textrm{Var}(Y)\leq(b-a)^2/4$. Gleichheit gilt, wenn $Y$ die Werte $a$ und $b$ jeweils mit Wahrscheinlichkeit $\tfrac12$ annimmt.*

<!--en:63211349-->
**Beweis.** Der Erwartungswert minimiert die erwartete quadrierte Abweichung, denn es gilt $\mathbb{E}[(Y-c)^2]=\textrm{Var}(Y)+(\mathbb{E}[Y]-c)^2$ für jeden Bezugswert $c$. Wählen wir den Mittelpunkt $c=(a+b)/2$, so hat jeder Punkt von $[a,b]$ höchstens den Abstand $(b-a)/2$ zum Mittelpunkt. Daraus folgt

<!--en:5a17d374-->
Beide Ungleichungen werden zu Gleichungen, wenn $Y$ den beiden Endpunkten $a$ und $b$ die gleiche Wahrscheinlichkeitsmasse zuweist. $\blacksquare$

<!--en:a7424577-->
Die faire Münze auf $[0,1]$ ist genau dieser Extremfall. Ihre Varianz beträgt $\tfrac14$. Mit dieser Varianzschranke können wir nun das Hoeffding-Lemma beweisen.

<!--en:ccf753de-->
**Proposition (Hoeffding-Lemma).** *Sei $X$ eine Zufallsvariable mit $\mathbb{E}[X]=0$ und Werten in $[a,b]$. Dann gilt für jedes $\lambda\in\mathbb{R}$*

<!--en:7fcdc6ac-->
**Beweis.** Wir wollen zeigen, dass die logarithmische MGF $\psi(\lambda)=\log\mathbb{E}[e^{\lambda X}]$ höchstens so schnell wächst wie die Parabel $\lambda^2(b-a)^2/8$. Da $X$ beschränkt ist, ist $\psi$ überall endlich und glatt. Wir dürfen die Ableitung also in den Erwartungswert ziehen. Bei $\lambda=0$ gilt $\psi(0)=\log 1=0$ und

<!--en:acfe3fa8-->
Wie bei der logarithmischen Zustandssumme einer Exponentialfamilie (:numref:`sec_mdl-exponential-family`) ist die zweite Ableitung von $\psi$ eine Varianz. Wir definieren die *exponentiell umgewichtete* Verteilung $q_\lambda$, indem wir die Verteilung von $X$ mit $e^{\lambda x}$ umgewichten und anschließend normieren. Unter $q_\lambda$ ist $\psi'(\lambda)$ der Erwartungswert von $X$. Leiten wir noch einmal ab, erhalten wir

<!--en:9158a2ab-->
Entscheidend ist, dass die exponentielle Umgewichtung nur die Gewichte der Punkte verändert. Sie verschiebt keine Masse aus dem Intervall $[a,b]$ hinaus. Unsere Varianzschranke gilt daher für $q_\lambda$ bei jedem $\lambda$. Es folgt $\psi''(\lambda)\leq(b-a)^2/4$. Der Satz von Taylor mit Lagrange-Restglied schließt den Beweis ab. Für ein $\xi$ zwischen $0$ und $\lambda$ gilt

<!--en:469c63e6-->
und durch Exponenzieren beider Seiten erhalten wir :eqref:`eq_mdl-hoeffding-lemma`. $\blacksquare$

<!--en:f9b74ef7-->
In Worten ist die momenterzeugende Funktion jeder beschränkten, zentrierten Zufallsvariablen höchstens so groß wie die einer normalverteilten Zufallsvariablen mit Standardabweichung $(b-a)/2$. Für einen zentrierten Münzwurf gilt $b-a=1$. Das Lemma liefert damit unmittelbar die Schranke $\cosh(\lambda/2)\leq e^{\lambda^2/8}$, die wir mithilfe der Taylorreihe bewiesen haben. Mit diesem einen Ergebnis lässt sich unser Beweis für die Münze Schritt für Schritt auf jeden Mittelwert unabhängiger beschränkter Zufallsvariablen übertragen.

<!--en:7be4ebfc-->
**Proposition (Hoeffding-Ungleichung).** *Seien $X_1,\ldots,X_n$ unabhängige Zufallsvariablen mit $X_i\in[a,b]$ und sei $\bar X = \frac1n\sum_i X_i$. Dann gilt für jedes $t>0$*

<!--en:1b9f371f-->
**Beweis.** Sei $Z_i = X_i - \mathbb{E}[X_i]$. Jede Variable $Z_i$ hat den Erwartungswert null und liegt in einem Intervall der Länge $b-a$. Das Hoeffding-Lemma gilt also für jeden Summanden. Für jedes $\lambda>0$ liefern die Chernoff-Methode :eqref:`eq_mdl-chernoff` und die Produktregel

<!--en:e2d7d4bc-->
Dies ist dieselbe Parabel, die wir beim Münzwurf minimiert haben, jetzt mit $(b-a)^2$ anstelle von $1$. Ihr Minimum liegt bei $\lambda = 4t/(b-a)^2$. Dort beträgt der Exponent $-2nt^2/(b-a)^2$. Wenden wir dasselbe Argument auf $-Z_i$ an, erhalten wir dieselbe Schranke für die untere Randwahrscheinlichkeit. Addieren wir beide Randwahrscheinlichkeiten, ergibt sich der Faktor $2$. $\blacksquare$

<!--en:50cf3031-->
Die Wahrscheinlichkeit, dass der Stichprobenmittelwert von $n$ unabhängigen beschränkten Variablen um mindestens $t$ von seinem Erwartungswert abweicht, fällt also exponentiell in $nt^2$. Dabei messen wir $t$ in Einheiten der Spannweite $b-a$. Was setzt die Schranke *nicht* voraus? Weder eine Normalverteilung noch eine Varianzschätzung noch einen asymptotischen Grenzübergang. Beschränktheit und Unabhängigkeit genügen, und die Schranke gilt für jedes endliche $n$. Die Unabhängigkeit ist allerdings wesentlich. Auf die korrelierten Stichprobenwerte einer Markov-Kette lässt sich die Schranke nicht anwenden.

<!--en:75e82ffc-->
Ein Blick auf den Beweis zeigt, dass die Intervalle gar nicht gleich lang sein müssen. Gilt für die unabhängigen zentrierten Variablen $Y_i\in[a_i,b_i]$, so tritt im Produkt der Schranken aus dem Hoeffding-Lemma an die Stelle von $n(b-a)^2$ die Summe $\sum_i(b_i-a_i)^2$. Minimieren wir wieder die quadratische Funktion, erhalten wir

<!--en:c2053c1e-->
Diese Fassung für unterschiedlich lange Intervalle verwenden wir in Kürze, wenn wir zufällige Richtungen in hohen Dimensionen untersuchen.

<!--en:4443fb29-->
Wie groß muss ein Testdatensatz nach der Hoeffding-Ungleichung in der Praxis sein? Für Verluste in $[0,1]$ gilt $b-a=1$. Die Wahrscheinlichkeit, die gewünschte Genauigkeit zu verfehlen, beträgt also höchstens $\delta$, sobald $n\geq\log(2/\delta)/(2t^2)$ gilt. Um eine Fehlerrate bis auf $t=0.01$ mit einer Irrtumswahrscheinlichkeit von $\delta=0.05$ zu schätzen, brauchen wir $n\geq\log 40/(2\cdot10^{-4})=18{,}444.4$, also $18{,}445$ Testbeispiele. Dies ist die Zahl aus :numref:`chap_classification_generalization`. Sie ist ungefähr doppelt so groß wie die 9604 Beispiele, die sich aus der asymptotischen Normalapproximation ergeben.

<!--en:0ef32e6d-->
Setzen wir die rechte Seite von :eqref:`eq_mdl-hoeffding` gleich $\delta$ und lösen nach $t$ auf, erhalten wir die äquivalente Form als Konfidenzintervall, die in der Praxis gebräuchlich ist. Mit Wahrscheinlichkeit mindestens $1-\delta$ gilt

<!--en:4feee026-->
Wie beim Intervall der Normalapproximation $\hat\mu\pm1.96\,\hat\sigma/\sqrt n$ schrumpft die Halbbreite mit der Rate $1/\sqrt n$. Hier gilt die Garantie bereits bei endlichem Stichprobenumfang.

<!--en:2398c3e5-->
Kehren wir zu unserem Klassifikator mit 830 Fehlern auf 10 000 Bildern zurück. Bei $\delta=0.05$ ergibt die Hoeffding-Ungleichung $\pm1.36\%$, die Normalapproximation dagegen $\pm1.96\sqrt{0.083\cdot0.917/10^4}=\pm0.54\%$. Die beiden Intervalle unterscheiden sich nur in ihren Konstanten. Die Hoeffding-Ungleichung ersetzt die empirische Standardabweichung $\hat\sigma=0.28$ durch ihren größtmöglichen Wert $(b-a)/2=0.5$ und das Quantil der Normalverteilung $1.96$ durch $\sqrt{2\log(2/\delta)}=2.72$. Dadurch ist das Intervall $2.5$-mal so breit. Ist die tatsächliche Streuung der Daten viel kleiner als ihre Spannweite, fällt die Hoeffding-Schranke zwangsläufig konservativ aus. Diese Lücke schließen wir weiter unten mit der Bernstein-Ungleichung.

<!--en:7e5e4b8f-->
Lässt sich die Konstante $\tfrac18$ im Hoeffding-Lemma verbessern? Eine Variable, die jedem der beiden Endpunkte die Wahrscheinlichkeit $\tfrac12$ zuweist, liefert die Antwort. Auf dem Intervall $[a,b]$ erreicht sie die Konstante $\tfrac18$ bis zur zweiten Ordnung in $\lambda$ (Übung :exref:`ex_mdl-concentration-generalization-random-signs`). Eine kleinere Konstante ist also unmöglich.

<!--en:58aa8009-->
### Die faire Münze in Zahlen

<!--en:489840b6-->
Wie nahe kommt die Hoeffding-Schranke $2e^{-2nt^2}$ bei einer fairen Münze der exakten Randwahrscheinlichkeit für $t=0.1$? Wir geben beide Werte und ihren Quotienten aus. Zusammen mit der Chebyshev-Schranke tragen wir sie auf logarithmischer Skala gegen $n$ auf.

<!--en:0cd3e47c-->
Bei $n=10$ liegen beide Schranken über $1$ und sind daher aussagelos. Bei $n=100$ ist die Chebyshev-Schranke mit $0.25$ sogar etwas schärfer als die Hoeffding-Schranke mit $0.27$. Bei kleinen Stichproben geben die Vorfaktoren den Ausschlag, und in die Chebyshev-Schranke geht die exakte Varianz $\tfrac14$ ein. Mit wachsendem $n$ setzt sich jedoch der exponentielle Abfall durch. Bei $n=1{,}000$ übersteigt die Hoeffding-Schranke die exakte Randwahrscheinlichkeit um höchstens den Faktor 15, bei $n=5{,}000$ um höchstens den Faktor 57.

<!--en:43ff527d-->
In der logarithmischen Darstellung verlaufen die exakte Randwahrscheinlichkeit und die Hoeffding-Schranke fast als parallele Geraden. Der Logarithmus der Hoeffding-Schranke sinkt pro Wurf um $2t^2=0.02$, der Logarithmus der exakten Randwahrscheinlichkeit um etwa $0.0201$. Der Betrag dieser steileren Steigung ist die Kullback-Leibler-Divergenz $D_{\textrm{KL}}(0.6\|0.5)=0.6\log1.2+0.4\log0.8$ zwischen einer Münze mit Kopfwahrscheinlichkeit $0.6$ und einer fairen Münze (:numref:`sec_mdl-divergences-distances`). Mit der Chernoff-Methode und der exakten MGF einer Bernoulli-Variablen erhalten wir genau diese Rate (Übung :exref:`ex_mdl-concentration-generalization-chernoff-coins`).

<!--en:c23553ef-->
Bei sehr kleinem $n$ kann keine Ungleichung den Mangel an Daten ausgleichen. Erinnern wir uns an die Münze aus der Kapiteleinleitung, die bei 13 Würfen 9-mal Kopf zeigte. Der Kopfanteil wich also um $t=2.5/13$ von $\tfrac12$ ab. Für eine mindestens so große Abweichung liefert die Hoeffding-Schranke $2e^{-2\cdot13\cdot(2.5/13)^2}=2e^{-0.96}\approx0.77$. Damit lässt sich nichts ausschließen. Selbst die exakte zweiseitige Binomialwahrscheinlichkeit beträgt $0.27$. Dreizehn Würfe reichen hier nicht aus, um eine faire Münze auszuschließen.

<!--en:d65dbf46-->
### Subgaußsche und subexponentielle Variablen

<!--en:d20055f8-->
Für unseren Klassifikator mit 8.3% Fehlerrate war das Hoeffding-Intervall 2.5-mal so breit wie das Intervall der Normalapproximation. In das Hoeffding-Intervall geht nur die Spannweite des 0/1-Verlusts ein, seine kleine Varianz bleibt unberücksichtigt. Die *Bernstein-Ungleichung* berücksichtigt beides.

<!--en:475564f9-->
**Proposition (Bernstein-Ungleichung).** *Seien $X_1,\ldots,X_n$ unabhängig mit $\textrm{Var}(X_i)\leq\sigma^2$ und $|X_i-\mathbb{E}[X_i]|\leq M$. Dann gilt für jedes $t>0$*

<!--en:0946fa8b-->
Der Beweis verwendet wieder die Chernoff-Methode :cite:`Boucheron.Lugosi.Massart.2013`, diesmal mit einer MGF-Schranke, die sowohl die Varianz als auch die Spannweite berücksichtigt (Herleitung in Übung :exref:`ex_mdl-concentration-generalization-bernstein`). Am Nenner erkennen wir zwei Fälle. Bei kleinem $t$ überwiegt $\sigma^2$ gegenüber $Mt/3$, und die Schranke verhält sich wie die Randwahrscheinlichkeit einer Normalverteilung mit der *wahren* Varianz $\sigma^2$. Bei großem $t$ überwiegt der lineare Term $Mt/3$. Die Randwahrscheinlichkeit fällt dann nur noch exponentiell in $t$ statt wie bei einer Normalverteilung exponentiell in $t^2$.

<!--en:c8dd9145-->
Wie viel bringt es in der Praxis, die Varianz zu berücksichtigen? Betrachten wir einen Klassifikator mit einer Fehlerrate von 2% auf $n=10{,}000$ Testbeispielen. Jeder 0/1-Verlust hat die Varianz $\sigma^2=0.02\cdot0.98$, und für den Betrag seiner zentrierten Abweichung genügt die Schranke $M=1$. Setzen wir die rechte Seite von :eqref:`eq_mdl-bernstein` gleich $\delta$, ergibt sich eine quadratische Gleichung in $t$, nämlich $nt^2/(2(\sigma^2+t/3))=\log(2/\delta)$. Im folgenden Code lösen wir diese Gleichung bei $\delta=0.05$ für unsere beiden Fehlerraten von 8.3% und 2% und vergleichen das Ergebnis mit dem Hoeffding-Intervall und dem Intervall der Normalapproximation. Wir berechnen sowohl die Plug-in-Schätzung, bei der wir die beobachtete Rate in die Varianz einsetzen, als auch eine Variante mit strenger Garantie. Letztere erläutern wir nach der Ausgabe.

<!--en:7e040978-->
Werten wir `half_widths(p, 10000)` für die Fehlerraten $8.3\%$ und $2.0\%$ aus, können wir die Halbbreiten der 95%-Konfidenzintervalle für alle vier Verfahren vergleichen.

<!--en:743b7a97-->
Bei einer Fehlerrate von 2% sinkt die Plug-in-Halbbreite nach Bernstein auf $\pm0.39\%$, gegenüber $\pm1.36\%$ nach Hoeffding. Das Hoeffding-Intervall ist also ungefähr dreieinhalbmal so breit. Das Bernstein-Intervall ist nur etwa $1.4$-mal so breit wie das asymptotische Intervall der Normalapproximation mit $\pm0.27\%$. Bei 8.3% Fehlern ist die Verbesserung geringer, aber immer noch beträchtlich. Die Halbbreite sinkt von $\pm1.36\%$ auf $\pm0.76\%$.

<!--en:8f5269f3-->
Streng genommen liefern die beiden Plug-in-Werte ($\pm0.39\%$ und $\pm0.76\%$) noch keine Garantie für endliche Stichproben. Der Code setzt die beobachtete Rate $\hat p$ in die Varianz $p(1-p)$ ein. Die Bernstein-Ungleichung verlangt jedoch eine obere Schranke für die *wahre* Varianz. Genau diesen Einwand hatten wir bereits gegen das Plug-in-Intervall der Normalapproximation erhoben. Das lässt sich mit geringem Aufwand beheben. Wenn die Schranke gilt, beträgt die wahre Fehlerrate höchstens $\hat p+t$. Wir lösen daher dieselbe Gleichung mit der größtmöglichen Varianz $(\hat p+t)(1-\hat p-t)$. Iterieren wir diese Fixpunktgleichung wie in der Codezelle, ändern sich die Halbbreiten nur wenig. Sie betragen jetzt $\pm0.43\%$ und $\pm0.79\%$, und beide Werte sind streng garantiert. Damit können wir die Eingangsfrage genau beantworten. Für den Klassifikator mit 830 Fehlern auf 10 000 Bildern weicht die wahre Fehlerrate mit 95% Konfidenz um höchstens etwa $0.8\%$ von 8.3% ab. Da Modelle in der Praxis meist Fehlerraten von wenigen Prozent haben, lohnt es sich, die Varianz in der Schranke zu berücksichtigen.

<!--en:56216725-->
Blicken wir auf Hoeffdings Beweis zurück. Er nutzt nur eine Eigenschaft der Summanden. Ihre momenterzeugenden Funktionen sind durch die einer Normalverteilung nach oben beschränkt. Geben wir dieser Eigenschaft einen Namen, können wir den Beweis auf unbeschränkte Zufallsvariablen übertragen und genau erkennen, wo er seine Gültigkeit verliert. Eine Zufallsvariable $X$ heißt **subgaußsch** mit *Varianzproxy* $\sigma^2$, wenn die MGF der zentrierten Variablen nirgends größer ist als die von $\mathcal N(0,\sigma^2)$.

<!--en:6a7264a1-->
Bei jeder subgaußschen Variablen fallen die Randwahrscheinlichkeiten mindestens so schnell ab wie bei einer Normalverteilung. Die Chernoff-Methode liefert mit $\lambda=t/\sigma^2$ die Schranke $P(|X-\mathbb{E}[X]|\geq t)\leq2e^{-t^2/(2\sigma^2)}$. Dabei ist $\sigma^2$ ein *Proxy*, der von der wahren Varianz abweichen kann. Vergleichen wir beide Seiten von :eqref:`eq_mdl-subgaussian` bis zur zweiten Ordnung in $\lambda$, erkennen wir, dass der Proxy nie kleiner als $\textrm{Var}(X)$ ist. Er kann sogar wesentlich größer sein. Bei einer 0/1-Variablen mit Fehlerwahrscheinlichkeit 2% beträgt die Varianz $0.0196$, das Hoeffding-Lemma liefert jedoch den Proxy $\tfrac14$. Genau deshalb war das Hoeffding-Intervall so breit.

<!--en:7780d7b9-->
Drei bekannte Familien von Zufallsvariablen sind subgaußsch. Für eine normalverteilte Variable $\mathcal N(\mu,\sigma^2)$ gilt die Schranke mit dem Proxy $\sigma^2$ sogar mit Gleichheit. Jede Variable mit Werten in $[a,b]$ hat nach dem Hoeffding-Lemma den Proxy $(b-a)^2/4$. Ein gleichverteiltes zufälliges Vorzeichen $\xi\in\{-1,+1\}$ hat den Proxy $1$, denn unser früherer Reihenvergleich ergibt $\mathbb{E}[e^{\lambda\xi}]=\cosh\lambda\leq e^{\lambda^2/2}$. Besonders wichtig ist, dass die Eigenschaft bei der Mittelwertbildung erhalten bleibt. Sind $X_1,\ldots,X_n$ unabhängig mit Proxy $\sigma^2$, so hat nach der Produktregel $\bar X$ den Proxy $\sigma^2/n$. Folglich gilt $P(|\bar X-\mathbb{E}\bar X|\geq t)\leq2e^{-nt^2/(2\sigma^2)}$.

<!--en:32024fd7-->
Allerdings sind nicht alle Zufallsvariablen subgaußsch, mit denen sich gut rechnen lässt. Das zeigt schon die nächste Größe, die wir brauchen. Das Quadrat der Länge eines gaußschen Zufallsvektors ist eine Summe von Quadraten standardnormalverteilter Variablen. Für eine einzelne standardnormalverteilte Variable $x$ liefert quadratische Ergänzung

<!--en:ca5e4aed-->
während der Erwartungswert für jedes $\lambda\geq\tfrac12$ gegen unendlich divergiert. Da die MGF bei $\lambda=\tfrac12$ unendlich wird, kann :eqref:`eq_mdl-subgaussian` für $x^2$ mit keinem Varianzproxy gelten. Die rechte Randwahrscheinlichkeit fällt wie $e^{-t/2}$, also zu langsam für eine gaußsche Schranke.

<!--en:b49dc61b-->
In einem beschränkten Intervall um $\lambda=0$ gilt *durchaus* eine MGF-Schranke von gaußscher Form. Eine Zufallsvariable $X$ heißt **subexponentiell** mit Parametern $(\nu,\alpha)$, wenn

<!--en:549b8bcd-->
Hier dient $\nu^2$ in der Nähe von $\lambda=0$ als lokaler Varianzproxy. Die Größe $1/\alpha$ bestimmt die Halbbreite des zulässigen Intervalls und damit den größten Wert von $\lambda$, den die Chernoff-Methode wählen darf. In :numref:`subsec_mdl-thin-shell` sehen wir, dass $x^2$ subexponentiell mit $\nu=2$ und $\alpha=4$ ist. Wenden wir die Chernoff-Methode auf einen Mittelwert subexponentieller Variablen an, ergeben sich bei der Optimierung zwei Fälle, je nachdem, ob die Intervallgrenze erreicht wird. Bei der ersten Lektüre können Sie den Beweis überspringen. Der nächste Unterabschnitt verwendet nur die Aussage.

<!--en:95c2ab41-->
**Proposition (Schranke für subexponentielle Randwahrscheinlichkeiten).** *Seien $X_1,\ldots,X_n$ unabhängig und $(\nu,\alpha)$-subexponentiell. Dann gilt für jedes $t>0$*

<!--en:a1ee5b0e-->
**Beweis.** Für jedes $0<\lambda\leq1/\alpha$ liefern die Chernoff-Methode und die Produktregel für $P(\bar X-\mathbb{E}[\bar X]\geq t)$ die obere Schranke $\exp(-\lambda nt+n\lambda^2\nu^2/2)$. Gilt $t\leq\nu^2/\alpha$, liegt der Minimierer $\lambda=t/\nu^2$ der unbeschränkten Optimierung im zulässigen Intervall und liefert $e^{-nt^2/(2\nu^2)}$. Gilt $t>\nu^2/\alpha$, liegt der Minimierer außerhalb des Intervalls. Wir setzen dann $\lambda$ auf den größten zulässigen Wert $\lambda=1/\alpha$. Der Exponent beträgt nun $-nt/\alpha+n\nu^2/(2\alpha^2)$. Das ist höchstens $-nt/\alpha+nt/(2\alpha)=-nt/(2\alpha)$, denn es gilt $\nu^2/\alpha<t$. In beiden Fällen ist der Exponent also höchstens $-\tfrac n2\min(t^2/\nu^2,t/\alpha)$. Die untere Randwahrscheinlichkeit schätzen wir genauso mit $\lambda\in[-1/\alpha,0)$ ab. $\blacksquare$

<!--en:f699a82d-->
Nun sehen wir den Unterschied zu Mittelwerten subgaußscher Variablen. Bei Mittelwerten beschränkter oder subgaußscher Variablen fallen die Randwahrscheinlichkeiten auf jeder Skala wie bei einer Normalverteilung gemäß $e^{-ct^2}$. Bei Mittelwerten subexponentieller Variablen gilt derselbe gaußsche Abfall für mäßige Abweichungen unterhalb von $\nu^2/\alpha$. Weiter außen gehen die Randwahrscheinlichkeiten in den langsameren exponentiellen Abfall $e^{-nt/(2\alpha)}$ über.

<!--en:45577ad9-->
## Wahrscheinlichkeit in hohen Dimensionen

<!--en:b22ac90a-->
Bisher galten unsere Schranken für einen einzelnen Stichprobenmittelwert. Betrachten wir nun einen Zufallsvektor in $\mathbb{R}^d$ mit $d$ unabhängigen Koordinaten. Sowohl sein Längenquadrat als auch seine inneren Produkte mit anderen Vektoren sind Summen über diese $d$ Koordinaten. Konzentrationsungleichungen bestimmen daher die Geometrie hochdimensionaler Räume. Wer hier einsteigt, braucht nur zwei Ergebnisse aus der ersten Hälfte des Abschnitts. Für Längenquadrate verwenden wir die subexponentielle Schranke :eqref:`eq_mdl-subexponential-tail`, für innere Produkte die Hoeffding-Schranke für unterschiedliche Wertebereiche :eqref:`eq_mdl-hoeffding-unequal`.

<!--en:f7630263-->
### Die dünne Kugelschale

<!--en:1b2538e2-->
Ziehen wir einen standardnormalverteilten Vektor $\mathbf x\sim\mathcal N(\mathbf 0,\mathbf
I_{784})$. Er hat so viele Koordinaten wie ein Bild der Größe $28\times28$ Pixel. Seine Wahrscheinlichkeitsdichte ist am Ursprung am höchsten. Dennoch liegt $\mathbf x$ mit verschwindend kleiner Wahrscheinlichkeit höchstens 14 Einheiten vom Ursprung entfernt, also nur halb so weit wie bei typischer Länge. Diese Wahrscheinlichkeit beträgt ungefähr $10^{-110}$. So wahrscheinlich ist es, dass die Quadratsumme von 784 unabhängigen standardnormalverteilten Variablen höchstens 196 beträgt. Selbst in einer Stichprobe, wie sie ein Rechner bestenfalls ziehen kann, findet sich praktisch kein Punkt in der Nähe des Modus. Ein typischer gezogener Vektor hat vielmehr die Norm $28.0\pm0.7$.

<!--en:99ae31b5-->
Das erklärt sich daraus, dass $\|\mathbf x\|^2=\sum_{i=1}^d x_i^2$ eine Summe von $d$ unabhängigen Termen mit dem Mittelwert $1$ ist. Damit ist $\|\mathbf x\|^2/d$ ein Stichprobenmittelwert. Da $x_i^2$ unbeschränkt ist, greift die Hoeffding-Ungleichung hier nicht. Unsere subexponentielle Schranke lässt sich jedoch direkt anwenden.

<!--en:256dd178-->
**Proposition (Konzentration der Norm).** *Sei $\mathbf x\sim\mathcal N(\mathbf 0,\mathbf I_d)$. Für jedes $0<\varepsilon\leq 1$ gilt*

<!--en:18b98dc0-->
**Beweis.** Wir gehen in drei kurzen Schritten vor. Erstens zeigt ein Vergleich der Potenzreihen (Übung :exref:`ex_mdl-concentration-generalization-chi-square-tail`), dass für die momenterzeugende Funktion (MGF) eines zentrierten Quadrats, also für $e^{-\lambda}/\sqrt{1-2\lambda}$, folgende Schranke gilt:

<!--en:99c1f53d-->
Im Sinne von :eqref:`eq_mdl-subexponential` ist $x_i^2$ also subexponentiell mit $\nu=2$ und $\alpha=4$. Zweitens liefert die subexponentielle Schranke :eqref:`eq_mdl-subexponential-tail` mit $n=d$ und $t=\varepsilon$

<!--en:96f5249c-->
Drittens gehen wir vom Quadrat der Norm zur Norm selbst über. Sei $z=\|\mathbf x\|/\sqrt d\geq0$. Aus $|z-1|\geq\varepsilon$ folgt auch $|z^2-1|=|z-1|\,(z+1)\geq\varepsilon$, denn $z+1\geq1$. Das Ereignis für die Norm ist also im Ereignis für ihr Quadrat enthalten. Damit gilt für seine Wahrscheinlichkeit dieselbe obere Schranke. $\blacksquare$

<!--en:cd78359b-->
Die Wahrscheinlichkeit, dass die Länge eines standardnormalverteilten Vektors um einen festen Anteil von $\sqrt d$ abweicht, fällt also exponentiell mit der Dimension $d$. Die Einschränkung $\varepsilon\leq1$ rührt daher, dass wir das *Quadrat* der Norm betrachten. Jenseits von $\varepsilon=1$ fällt seine obere Randwahrscheinlichkeit nur noch exponentiell wie $e^{-d\varepsilon/8}$ (Übung :exref:`ex_mdl-concentration-generalization-chi-square-tail`). Für die Norm selbst gilt auf jeder Skala eine gaußsche Schranke. Überschreitet die Norm $(1+\varepsilon)\sqrt d$, so überschreitet ihr Quadrat $(1+2\varepsilon+\varepsilon^2)d$. Die Wahrscheinlichkeit dafür beträgt höchstens $e^{-d\min(s,s^2)/8}$ mit $s=2\varepsilon+\varepsilon^2$. Wegen $s\geq\varepsilon^2$ und $s^2\geq4\varepsilon^2$ ist das höchstens $e^{-d\varepsilon^2/8}$.

<!--en:977f9d39-->
Wie dick ist die Schale, in der die typischen Normen liegen? Wegen $\mathbb{E}[x^4]=3$ gilt für jedes Koordinatenquadrat $\textrm{Var}(x^2)=3-1=2$ und damit für die Summe $\textrm{Var}(\|\mathbf x\|^2)=2d$. Linearisieren wir die Quadratwurzel um $d$, erhalten wir $\|\mathbf x\|\approx\sqrt d+(\|\mathbf x\|^2-d)/(2\sqrt d)$. Die Standardabweichung von $\|\mathbf x\|$ beträgt somit unabhängig von der Dimension ungefähr $\sqrt{2d}/(2\sqrt d)=1/\sqrt2
\approx0.71$. Eine Standardnormalverteilung in $\mathbb{R}^d$ konzentriert sich also auf eine dünne Kugelschale mit Radius $\sqrt d$. Ihre Dicke bleibt von der Größenordnung eins, gleichgültig, wie groß $d$ wird.

<!--en:6576a568-->
Warum liegt die Wahrscheinlichkeitsmasse so weit vom Modus entfernt? Die gemeinsame Dichte $(2\pi)^{-d/2}e^{-\|\mathbf x\|^2/2}$ hängt von $\mathbf x$ nur über den Radius $r=\|\mathbf x\|$ ab. Zugleich wächst der Flächeninhalt einer Kugeloberfläche mit Radius $r$ wie $r^{d-1}$. Multiplizieren wir die Dichte mit dem Flächeninhalt, erhalten wir für den Radius $\|\mathbf x\|$ eine Dichte proportional zu $r^{d-1}e^{-r^2/2}$. Mit wachsendem $r$ steigt der Volumenfaktor $r^{d-1}$ steil an, während die Dichte $e^{-r^2/2}$ rasch abfällt. Ihr Produkt ist dort am größten, wo die Ableitung von $(d-1)\log r-r^2/2$ verschwindet, also bei $r=\sqrt{d-1}$. Teilbild (a) von :numref:`fig_mdl-conc-shell` zeigt dieses Zusammenspiel von Volumen und Dichte für $d=20$.

<!--en:9152077f-->
An einem typischen Punkt mit Norm $\sqrt d$ beträgt die Dichte nur das $e^{-d/2}$-Fache ihres Werts am Ursprung (ein Faktor von gerade einmal $10^{-170}$ für $d=784$). Beim Radius $\sqrt d$ ist die Kugeloberfläche jedoch so viel größer, dass dort fast die gesamte Masse liegt. Teilbild (b) zeigt die exakte Dichte von $\|\mathbf x\|$ für Dimensionen von $d=2$ bis $d=512$. Die Maxima wandern mit $\sqrt d$ nach außen, während die Breiten nahe bei $1/\sqrt2$ bleiben.

<!--en:1fa91c78-->
![Die dünne Kugelschale. (a) Für $d=20$ fällt die gaußsche Dichte $e^{-r^2/2}$ mit dem Radius $r$, während das Schalenvolumen $r^{d-1}$ beim Radius $r$ wächst. Ihr Produkt, die Dichte von $\|\mathbf{x}\|$, erreicht sein Maximum bei $\sqrt{d-1}$, weit entfernt vom Ursprung, wo die gaußsche Dichte am höchsten ist. Jede Kurve ist auf den Höchstwert 1 skaliert. (b) Exakte Dichte von $\|\mathbf{x}\|$ für $\mathbf{x}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_d)$ und $d=2,8,32,128,512$. Die Zentren wandern mit $\sqrt{d}$ nach außen, die Breiten bleiben nahe bei $1/\sqrt{2}$.](../img/mdl-prob-conc-shell.svg)

<!--en:214c2c2a-->
Unsere Schranke erfasst zwar die exponentielle Abhängigkeit von $d$, gibt aber den Zahlenwert nur grob wieder. Eine Norm unter 14 bei $d=784$ entspricht einer Abweichung von mindestens $\varepsilon=\tfrac12$. Dafür liefert :eqref:`eq_mdl-norm-concentration` die Schranke $2e^{-784/32}=2e^{-24.5}\approx5\times10^{-11}$ gegenüber dem exakten Wert von rund $10^{-110}$. Die Konstante $8$ im Nenner ist großzügig bemessen. Die qualitative Aussage, dass die Wahrscheinlichkeit exponentiell mit $d$ abfällt, trifft jedoch zu.

<!--en:1aa17ceb-->
### Nahezu orthogonale Zufallsrichtungen

<!--en:eb86ec4c-->
In :numref:`sec_mdl-geometry-linear-algebraic-ops` haben wir gesehen, dass der Kosinus des Winkels zwischen einem festen Einheitsvektor und einer zufälligen Richtung in $\mathbb{R}^d$ die Varianz $1/d$ hat. Mit der Chebyshev-Ungleichung ergab sich daraus nur eine schwache Schranke der Größenordnung $1/(d\varepsilon^2)$ für die Randwahrscheinlichkeit. Mithilfe der Konzentration erhalten wir eine exponentielle Schranke. Damit können wir bestimmen, wie viele zufällige Richtungen gleichzeitig nahezu orthogonal zueinander sein können.

<!--en:fb502286-->
Am einfachsten ist die Rechnung für zufällige Vorzeichenvektoren. Wir wählen einen beliebigen festen Einheitsvektor $\mathbf u$ und ziehen unabhängig voneinander gleichverteilte Vorzeichen $x_i\in\{-1,+1\}$. Jeder Summand $u_ix_i$ des inneren Produkts $\langle\mathbf x,\mathbf u\rangle$ hat den Mittelwert null und liegt in $[-|u_i|,|u_i|]$, einem Intervall der Länge $2|u_i|$. Setzen wir diese Wertebereiche in die Hoeffding-Schranke für unterschiedliche Wertebereiche :eqref:`eq_mdl-hoeffding-unequal` ein, ergibt sich im Nenner $4\sum_i u_i^2=4$. Damit folgt

<!--en:04f2e2e3-->
Das innere Produkt eines zufälligen Vorzeichenvektors mit einem beliebigen festen Einheitsvektor bleibt von der Größenordnung eins. Seine Randwahrscheinlichkeit fällt wie bei einer Gauß-Verteilung, weil sich die Beiträge $\pm|u_i|$ weitgehend aufheben. Die Norm von $\mathbf x$ beträgt hingegen $\sqrt d$. Mit $s=\varepsilon\sqrt d$ folgt unmittelbar $P(|\cos(\mathbf x,\mathbf u)|\geq\varepsilon)\leq2e^{-d\varepsilon^2/2}$. Ein standardnormalverteilter Vektor $\mathbf x$ verhält sich ähnlich. Das innere Produkt $\langle\mathbf x,\mathbf u\rangle$ ist für jeden Einheitsvektor $\mathbf u$ selbst standardnormalverteilt. Es erfüllt daher dieselbe Schranke $2e^{-s^2/2}$ für die Randwahrscheinlichkeit. Wegen der Konzentration auf eine dünne Schale weicht $\|\mathbf x\|$ typischerweise nur um wenige Prozent von $\sqrt d$ ab.

<!--en:db8defef-->
Mit einer exponentiellen Schranke für die Randwahrscheinlichkeit können wir *viele* Paare zugleich erfassen. Dazu nutzen wir die *boolesche Ungleichung*. Sie besagt, dass die Wahrscheinlichkeit, dass mindestens eines von mehreren Ereignissen eintritt, höchstens so groß ist wie die Summe ihrer Wahrscheinlichkeiten. Das gilt auch für abhängige Ereignisse. Betrachten wir $m$ unabhängige zufällige Vorzeichenvektoren. Sie bilden $m(m-1)/2<m^2/2$ verschiedene Paare. Für jedes Paar gilt $|\cos|\geq\varepsilon$ mit einer Wahrscheinlichkeit von höchstens $2e^{-d\varepsilon^2/2}$. Nach der booleschen Ungleichung beträgt die Wahrscheinlichkeit, dass der Kosinusbetrag für *irgendein* Paar die Schwelle $\varepsilon$ überschreitet, höchstens $m^2e^{-d\varepsilon^2/2}$. Mit einer Wahrscheinlichkeit von mindestens $1-\delta$ bleiben also sämtliche paarweisen Kosinusbeträge unter $\varepsilon$, sofern $m\leq\sqrt\delta\,e^{d\varepsilon^2/4}$ gilt. In hohen Dimensionen finden somit *exponentiell* viele nahezu orthogonale Vektoren in $\mathbb{R}^d$ Platz.

<!--en:86b3fd8e-->
Setzen wir $d=784$ und $\delta=0.05$ ein. Selbst diese grobe Schranke garantiert rund 570 Richtungen mit $|\cos|<0.2$ für alle Paare und rund $10^7$ Richtungen mit $|\cos|<0.3$ für alle Paare. Diese Garantien berücksichtigen selbst das ungünstigste Paar. Für ein einzelnes typisches Paar ist $|\cos|$ wesentlich kleiner und liegt bei etwa $\sqrt{2/(\pi d)}\approx0.028$. Das ist der Erwartungswert des Betrags einer zentrierten gaußschen Zufallsvariablen mit Standardabweichung $1/\sqrt d$.

<!--en:bd75c022-->
Einbettungen, die ein neuronales Netz lernt, sind natürlich keine unabhängigen Zufallsvektoren. Gerade deshalb ist diese Rechnung aufschlussreich. Bei $d=784$ liegt eine Kosinusähnlichkeit von $0.3$ mehr als acht Standardabweichungen ($1/\sqrt{784}=0.036$) über dem Erwartungswert für unabhängige Zufallsrichtungen. Zufällige Richtungen liefern also ein Nullmodell, an dem wir die geometrische Struktur erlernter Einbettungen messen können.

<!--en:09fdcf78-->
### Die dünne Kugelschale in Zahlen

<!--en:d9c8a1a5-->
Da die Konstanten in unseren theoretischen Schranken großzügig bemessen sind, messen wir die Werte direkt. Nach unserer Linearisierung sollte die Standardabweichung von $\|\mathbf x\|$ nahe bei $1/\sqrt2\approx0.71$ liegen, unabhängig von der Dimension $d$. Die Gauß-Näherung sagt für den Mittelwert von $|\cos|$ den Wert $\sqrt{2/(\pi d)}$ voraus. Um beides zu prüfen, ziehen wir im Folgenden 10 000 Paare standardnormalverteilter Vektoren für $d=2,20,200,2000$. Eine Zeile weicht von der Vorhersage ab.

<!--en:5b86333e-->
Wie vorhergesagt liegt die Standardabweichung der Norm bei etwa $0.65$ für $d=2$ und pendelt sich danach bei rund $0.7$ ein, obwohl die Norm selbst von etwa 1.3 auf 45 steigt. Nur die Zeile für $d=2$ passt nicht zur asymptotischen Formel. Hier beträgt der Mittelwert von $|\cos|$ etwa $0.64$. In der Ebene ist der Winkel zwischen zwei unabhängigen isotropen Vektoren gleichverteilt. Daher gilt $\mathbb{E}|\cos\theta|=2/\pi$, während $\sqrt{2/(\pi d)}$ eine Näherung für hohe Dimensionen ist. Ab $d=20$ stimmen die Messwerte (etwa $0.18$, $0.057$ und $0.018$) bis auf wenige Prozent mit der Formel überein.

<!--en:688000f7-->
Zwei weitere Fragen veranschaulichen die Geometrie. Erstens: Welcher Anteil der Wahrscheinlichkeitsmasse liegt höchstens 10% von der typischen Norm entfernt, also in der Schale $(1\pm0.1)\sqrt d$? Zweitens: Lassen sich nahe und ferne Punkte anhand ihres euklidischen Abstands noch gut unterscheiden? Dazu berechnen wir den *Kontrast* eines Abfragepunkts als $(d_{\max}-d_{\min})/d_{\min}$ aus seinen Abständen zu 200 zufälligen Punkten. Anschließend bestimmen wir den Median des Kontrasts über 100 Abfragen.

<!--en:951b702a-->
Bei $d=2$ enthält die 10%-Schale nur etwa 15% der Masse, bei $d=20$ schon rund 47% und bei $d=200$ rund 95%. Die 95% stimmen mit unserer Schätzung aus der Linearisierung überein. Bei $d=200$ hat das Verhältnis $\|\mathbf x\|/\sqrt d$ die Standardabweichung $1/\sqrt{2d}=0.05$. Der Bereich $\pm0.1$ reicht daher auf beiden Seiten des Mittelwerts über je zwei Standardabweichungen. Bei $d=2000$ fallen alle 10 000 gezogenen Punkte in die Schale. Zugleich sinkt der Median des Abstandskontrasts von rund 30 bei $d=2$ (je nach Durchlauf zwischen etwa 25 und 40) auf ungefähr 1.2, dann auf 0.27 und schließlich auf 0.08. Bei $d=2000$ ist für einen typischen Abfragepunkt der entfernteste von 200 Zufallspunkten weniger als 10% weiter entfernt als der nächste.

<!--en:01e0df53-->
### Folgen für Deep Learning

<!--en:a56aa009-->
Drei gängige Verfahren im Deep Learning beruhen unmittelbar auf dieser hochdimensionalen Geometrie.

<!--en:abe06a5f-->
**Initialisierung.** Bei der LeCun-Initialisierung :cite:`LeCun.Bottou.Orr.ea.1998` ziehen wir die Einträge einer $d\times d$-Gewichtsmatrix unabhängig mit Varianz $1/d$. Die He-Initialisierung :cite:`He.Zhang.Ren.ea.2015` verwendet für ReLU-Netze die Varianz $2/d$ (:numref:`sec_numerical_stability`). Die klassische Herleitung dieser Skalierungen mittelt über alle zufälligen Ziehungen der Gewichte. Bei der LeCun-Skalierung hat jede Präaktivierung $\langle\mathbf w,\mathbf x\rangle$ die Varianz $\|\mathbf x\|^2/d$. Haben die Eingabekoordinaten Varianz eins, hält die Konzentration der Norm diesen Wert nahe bei $1$. In der Praxis initialisieren und trainieren wir jedoch ein *einzelnes* Netz. Über unendlich viele Netze können wir nicht mitteln. Die Konzentration zeigt uns, wann der Erwartungswert diese eine Realisierung tatsächlich beschreibt.

<!--en:02bebd24-->
Für eine einzelne ReLU-Schicht mit He-Initialisierung und Breite $d$ hat das Verhältnis $\rho=\|\mathbf h_k\|^2/\|\mathbf h_{k-1}\|^2$ der quadrierten Aktivierungsnormen den Mittelwert $1$ und die Varianz $5/d$ (Übung :exref:`ex_mdl-concentration-generalization-depth-norms`). Eine Schicht verändert das Normquadrat also relativ um etwa $\sqrt{5/d}$, das heißt um rund 10% bei $d=512$. Über $K$ Schichten addieren sich diese multiplikativen Schwankungen auf logarithmischer Skala. Damit hat $\log(\|\mathbf h_K\|^2/\|\mathbf h_0\|^2)$ eine Varianz von ungefähr $5K/d$. Die Heuristik auf Grundlage der erwarteten Varianz beschreibt ein typisches Netz also nur, solange $K\ll d$ gilt. Bei $K=50$ und $d=512$ erreicht die Varianz des Logarithmus etwa $0.5$. Das Normquadrat weicht bei einem typischen initialisierten Netz also bereits um einen Faktor zwei nach oben oder unten ab.

<!--en:a0a3b17d-->
**Abstände und nächste Nachbarn.** Für die Differenz zweier unabhängiger standardnormalverteilter Vektoren gilt $\mathbf x-\mathbf y\sim\mathcal N(\mathbf 0,2\mathbf I)$. Die Konzentration der Norm lässt sich daher direkt auf $(\mathbf x-\mathbf y)/\sqrt2$ anwenden. Für jeden paarweisen Abstand ergibt sich das Intervall $(1\pm\varepsilon)\sqrt{2d}$, das er nur mit Wahrscheinlichkeit von höchstens $2e^{-d\varepsilon^2/8}$ verlässt. Wenden wir wie bei den Richtungen die boolesche Ungleichung an, liegen *alle* paarweisen Abstände zwischen $n$ Punkten gleichzeitig in diesem schmalen Band, sobald $d\geq(8/\varepsilon^2)\log(n^2/\delta)$ gilt. Diese Schwelle wächst nur logarithmisch mit $n$. Sind alle Abstände nahezu gleich, unterscheiden sich der nächste und der entfernteste Nachbar relativ gesehen nur wenig. Das haben wir bereits im Experiment zum Abstandskontrast beobachtet. Bei erlernten Einbettungen bleibt der Abstandskontrast nur deshalb aussagekräftig, weil reale Daten nahe an einer Struktur von viel niedrigerer Dimension liegen. Dieselbe Rechnung liefert uns zugleich ein wirkungsvolles Verfahren. Eine zufällige Projektion auf $k$ Dimensionen erhält alle paarweisen Abstände zwischen $n$ Punkten näherungsweise, sobald $k$ die Größenordnung $\log n$ erreicht, unabhängig von der Dimension des umgebenden Raums (Übung :exref:`ex_mdl-concentration-generalization-johnson-lindenstrauss`).

<!--en:95b9cd1a-->
**Latente Codes.** Ein generativer Decoder sieht beim Training mit latenten Codes $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ fast ausschließlich Eingaben, deren Norm nahe bei $\sqrt d$ liegt. Der Mittelwert der A-priori-Verteilung (Norm $0$) liegt tief im leeren Inneren, wo der Decoder nie trainiert wurde. Dasselbe gilt für den Mittelpunkt der geraden Verbindungsstrecke zwischen zwei unabhängigen Codes (Norm nahe bei $\sqrt{d/2}$). In der Praxis interpoliert man deshalb zwischen latenten Codes meist entlang eines Kreisbogens auf der Kugeloberfläche. Den Unterschied zur geradlinigen Interpolation zeigt :numref:`fig_mdl-conc-slerp-vs-lerp` (Übung :exref:`ex_mdl-concentration-generalization-latent-interpolation`).

<!--en:1812a590-->
![Zwei Arten der Interpolation zwischen latenten Codes $\mathbf{z}_1,\mathbf{z}_2\sim\mathcal{N}(\mathbf{0},\mathbf{I}_d)$. (a) Beide Codes liegen in der dünnen Schale mit Radius $\sqrt d$ (schattiert) und sind nahezu orthogonal. Die Sehne (lerp) verläuft durch das Innere. Ihr Mittelpunkt hat ungefähr die Norm $\sqrt{d/2}$. Der Kreisbogen (slerp) bleibt in der Schale. (b) Norm entlang beider Wege, geteilt durch $\sqrt d$, für ein zufälliges Paar bei $d=512$.](../img/mdl-prob-conc-slerp-vs-lerp.svg)

<!--en:e1c7a56c-->
Eine wesentliche Einschränkung bleibt. Jede Schranke in diesem Abschnitt gilt für eine einzelne Größe, die *vor* dem Ziehen der Daten feststeht. Ein Lernalgorithmus sichtet hingegen die Daten und wählt *danach* aus vielen Kandidaten das beste Modell. Selbst unter lauter fairen Münzen wirkt die mit dem höchsten Kopfanteil so, als wäre sie gezinkt. In :numref:`sec_mdl-generalization-bounds` zeigen wir, wie sich diese Auswahl bei Konzentrationsungleichungen berücksichtigen lässt.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3b512fe2-->
Wie weit kann der Mittelwert von $n$ unabhängigen Summanden bei endlichem Stichprobenumfang $n$ von seinem Erwartungswert abweichen? Die Chebyshev-Ungleichung liefert für die Randwahrscheinlichkeit eine Schranke, die nur wie $1/n$ abfällt. Das ist viel zu langsam, wenn wir eine Garantie brauchen, die fast nie versagt. Die Chernoff-Methode wendet die Markov-Ungleichung auf $e^{\lambda X}$ an. Dadurch lässt sich die momenterzeugende Funktion einer Summe unabhängiger Variablen als Produkt schreiben. Anschließend optimieren wir über $\lambda$. Zusammen mit dem Hoeffding-Lemma ergibt das die exponentielle Schranke $2e^{-2nt^2}$ für Verluste in $[0,1]$ und ein Konfidenzintervall mit Halbbreite $\sqrt{\log(2/\delta)/(2n)}$ bei endlichem Stichprobenumfang. Ist die Varianz im Verhältnis zum Quadrat der Spannweite klein, liefert die Bernstein-Ungleichung ein deutlich engeres Intervall. Wenden wir dieselben Überlegungen auf die $d$ Koordinaten eines Zufallsvektors an, zeigt sich, dass eine Standardnormalverteilung auf einer dünnen Kugelschale mit Radius $\sqrt d$ und Dicke von etwa $1/\sqrt2$ konzentriert ist. Außerdem können exponentiell viele Zufallsrichtungen paarweise nahezu orthogonal sein. Mit diesen Ergebnissen können wir die nötige Größe eines Testdatensatzes bestimmen und erkennen, wann Heuristiken zur Initialisierung für ein einzelnes trainiertes Netz gelten. Zudem helfen sie uns, Kosinusähnlichkeiten und Abstände in hochdimensionalen Einbettungen zu beurteilen.

<!--en:2294243a-->
## Übungen

<!--en:227e3d82-->
1. **Subgaußsche zufällige Vorzeichen.** Ein zufälliges Vorzeichen $\xi$ nimmt die Werte $-1$ und $+1$ jeweils mit Wahrscheinlichkeit $\tfrac12$ an. Daher gilt $\mathbb{E}[e^{\lambda\xi}]=\cosh\lambda\leq e^{\lambda^2/2}$.
    1. Folgern Sie mit der Chernoff-Methode :eqref:`eq_mdl-chernoff`, dass für den Mittelwert $\bar\xi$ von $n$ unabhängigen zufälligen Vorzeichen $P(|\bar\xi|\geq t)\leq 2e^{-nt^2/2}$ gilt.
    1. Das Hoeffding-Lemma :eqref:`eq_mdl-hoeffding-lemma` mit $[a,b]=[-1,1]$ liefert ebenfalls einen Varianzproxy für $\xi$. Vergleichen Sie ihn mit der Schranke $\cosh\lambda\leq e^{\lambda^2/2}$. Entwickeln Sie $\log\cosh\lambda=\lambda^2/2-\lambda^4/12+\cdots$ um $\lambda=0$. Erklären Sie, warum keine Konstante kleiner als $\tfrac18$ den Faktor $\tfrac18$ in :eqref:`eq_mdl-hoeffding-lemma` für alle beschränkten Variablen ersetzen kann.
   :exlabel:`ex_mdl-concentration-generalization-random-signs`

<!--en:125e54b9-->
1. **Interpolation im latenten Raum.** Ein generatives Modell dekodiert latente Codes $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_d)$ mit $d=512$. Es seien $\mathbf z_1,\mathbf z_2$ zwei unabhängige Codes und $\mathbf z(\gamma)=(1-\gamma)\mathbf z_1+\gamma\mathbf z_2$ für $\gamma\in[0,1]$ ihre lineare Interpolation.
    1. Bestimmen Sie die Verteilung von $\mathbf z(\gamma)$ und den ungefähren Wert von $\|\mathbf z(\gamma)\|$. Für welches $\gamma$ ist die Norm am kleinsten?
    1. Bestimmen Sie mit :eqref:`eq_mdl-norm-concentration` eine obere Schranke für die Wahrscheinlichkeit, dass der Mittelpunkt $\mathbf z(\tfrac12)$ mindestens die Norm $0.9\sqrt d$ hat, also mindestens den Innenradius der Schale $(1\pm 0.1)\sqrt d$ erreicht, in der typische Codes liegen. Berechnen Sie dann die exakte Wahrscheinlichkeit. Sie beträgt $P(\chi^2_{512}\geq 2\cdot0.81\cdot512)$ für eine Chi-Quadrat-Variable mit 512 Freiheitsgraden (`scipy.stats.chi2.sf`). Um wie viele Größenordnungen ist die Schranke zu groß?
    1. Bei der sphärischen Interpolation :cite:`White.2016` ersetzt ein Kreisbogen die Sehne:

       $$\tilde{\mathbf z}(\gamma)=\frac{\sin((1-\gamma)\theta)}{\sin\theta}\,\mathbf z_1+\frac{\sin(\gamma\theta)}{\sin\theta}\,\mathbf z_2,$$

       Dabei ist $\theta$ der Winkel zwischen $\mathbf z_1$ und $\mathbf z_2$. Zeigen Sie, dass $\|\tilde{\mathbf z}(\gamma)\|=r$ für alle $\gamma$ gilt, wenn $\|\mathbf z_1\|=\|\mathbf z_2\|=r$ ist. Welchen Wert nimmt $\theta$ für $d=512$ ungefähr an und warum?
   :exlabel:`ex_mdl-concentration-generalization-latent-interpolation`

<!--en:bc8f83f3-->
1. [code] **Chernoff-Schranke für Münzwürfe.** ● Sei $\hat p$ der Kopfanteil bei $n$ unabhängigen Würfen einer Münze mit Kopfwahrscheinlichkeit $p$, und sei $a>p$.
    1. Wenden Sie die Chernoff-Methode :eqref:`eq_mdl-chernoff` auf $n\hat p$ an. Die zugehörige momenterzeugende Funktion ist $(1-p+pe^\lambda)^n$. Optimieren Sie über $\lambda$ und zeigen Sie, dass $P(\hat p\geq a)\leq e^{-n\,D_{\textrm{KL}}(a\|p)}$ gilt, wobei $D_{\textrm{KL}}(a\|p)=a\log\frac ap+(1-a)\log\frac{1-a}{1-p}$ ist.
    1. Zeigen Sie, dass $D_{\textrm{KL}}(a\|p)\geq 2(a-p)^2$ gilt, indem Sie die zweite Ableitung nach $a$ untersuchen. Damit ist die Schranke aus (a) stets mindestens so scharf wie die einseitige Hoeffding-Schranke. Berechnen Sie beide Exponenten für $p=\tfrac12$ und $a=0.6$.
    1. Tragen Sie die zweiseitige Schranke $2e^{-n\,D_{\textrm{KL}}(0.6\|0.5)}$ als weitere Spalte in die Tabelle aus :numref:`subsec_mdl-tail-bounds-numerical` ein, und zwar für $n\in\{10,100,1000,5000\}$. Das Verhältnis der exakten Randwahrscheinlichkeit zu dieser Schranke nimmt mit $n$ weiterhin ab. Wie hängt es von $n$ ab und was sagt diese Abhängigkeit über den Faktor aus, der bei der Chernoff-Methode verloren geht?
   :exlabel:`ex_mdl-concentration-generalization-chernoff-coins`

<!--en:4a53418c-->
1. **Randwahrscheinlichkeit der Chi-Quadrat-Verteilung.** ● Der Beweis von :eqref:`eq_mdl-norm-concentration` verwendet die Schranke :eqref:`eq_mdl-chi2-mgf` für $x^2-1$ mit $x\sim\mathcal N(0,1)$ ohne Herleitung.
    1. Zeigen Sie durch quadratische Ergänzung im Gauß-Integral, dass $\mathbb{E}[e^{\lambda(x^2-1)}]=e^{-\lambda}/\sqrt{1-2\lambda}$ für $\lambda<\tfrac12$ gilt.
    1. Entwickeln Sie $\log\mathbb{E}[e^{\lambda(x^2-1)}]$ in eine Potenzreihe in $2\lambda$ und beweisen Sie :eqref:`eq_mdl-chi2-mgf`. Die Reihe beginnt mit dem quadratischen Term.
    1. Der Beweis von :eqref:`eq_mdl-norm-concentration` verwendet $\lambda=\varepsilon/4$. Diese Wahl ist nur für $\varepsilon\leq 1$ zulässig. Setzen Sie für $\varepsilon>1$ den Wert $\lambda=\tfrac14$ ein und bestimmen Sie eine obere Schranke für $P(\|\mathbf x\|^2\geq(1+\varepsilon)d)$. Fällt sie wie $e^{-c\varepsilon^2}$ oder wie $e^{-c\varepsilon}$ mit $\varepsilon$ ab? Welchem der beiden Bereiche in :eqref:`eq_mdl-subexponential-tail` entspricht das?
   :exlabel:`ex_mdl-concentration-generalization-chi-square-tail`

<!--en:38ab07cc-->
1. [code] **Aktivierungsnormen mit zunehmender Tiefe.** ● Betrachten Sie ein ReLU-Netz der Breite $d$ und Tiefe $K$ ohne Bias-Terme. Es gelte $\mathbf h_k=\mathrm{ReLU}(\mathbf W_k\mathbf h_{k-1})$ mit unabhängig aus $\mathcal N(0,2/d)$ gezogenen Gewichten (He-Initialisierung, :numref:`sec_numerical_stability`). Sei $\rho_k=\|\mathbf h_k\|^2/\|\mathbf h_{k-1}\|^2$. Wie diese Normen in Abhängigkeit von Tiefe und Breite schwanken, untersuchten :citet:`Hanin.Nica.2020`.
    1. Zeigen Sie, dass für gegebenes $\mathbf h_{k-1}\ne\mathbf 0$ der Quotient $\rho_k$ dieselbe Verteilung hat wie $\frac2d\sum_{j=1}^d\mathrm{ReLU}(g_j)^2$ mit unabhängigen $g_j\sim\mathcal N(0,1)$, unabhängig von der Richtung von $\mathbf h_{k-1}$. Folgern Sie $\mathbb{E}[\rho_k]=1$ und $\textrm{Var}(\rho_k)=5/d$ mithilfe von $\mathbb{E}[g^2]=1$ und $\mathbb{E}[g^4]=3$.
    1. Schreiben Sie $\log(\|\mathbf h_K\|^2/\|\mathbf h_0\|^2)$ als Summe von $K$ unabhängigen Termen. Vernachlässigen Sie dabei das Ereignis $\rho_k=0$, das in jeder Schicht mit Wahrscheinlichkeit $2^{-d}$ eintritt. Entwickeln Sie $\log\rho$ bis zur zweiten Ordnung um $1$. Sagen Sie Mittelwert und Varianz der Summe in Abhängigkeit von $K/d$ vorher.
    1. Simulieren Sie $1{,}000$ Netze für jede Kombination aus Breite $d\in\{32,128\}$ und Tiefe $K\in\{1,10,50\}$, jeweils mit einer neu gezogenen standardnormalverteilten Eingabe. Vergleichen Sie Mittelwert und Varianz von $\log(\|\mathbf h_K\|^2/\|\mathbf h_0\|^2)$ mit (b). Geben Sie auch den Mittelwert und den Median des Quotienten $\|\mathbf h_K\|^2/\|\mathbf h_0\|^2$ selbst an. Die Varianzrechnung aus :numref:`sec_numerical_stability` liefert $\mathbb{E}[\|\mathbf h_K\|^2/\|\mathbf h_0\|^2]=1$ für jede Tiefe. Wiederholen Sie die Simulation für $K=50$ und $d=32$ mehrmals. Warum streut der Stichprobenmittelwert über $1{,}000$ Netze so stark um $1$? Warum beschreibt der Erwartungswert hier kein typisches Netz?
   :exlabel:`ex_mdl-concentration-generalization-depth-norms`

<!--en:2c6ee064-->
1. [code] **Zufällige Projektionen erhalten Abstände.** ● Die Matrix $\mathbf A\in\mathbb{R}^{k\times d}$ habe unabhängige, $\mathcal N(0,1/k)$-verteilte Einträge.
    1. Zeigen Sie, dass für festes $\mathbf v\neq\mathbf 0$ der Quotient $\|\mathbf A\mathbf v\|^2/\|\mathbf v\|^2$ dieselbe Verteilung hat wie $\|\mathbf g\|^2/k$ mit $\mathbf g\sim\mathcal N(\mathbf 0,\mathbf I_k)$. Nach :eqref:`eq_mdl-norm-concentration` folgt also $P(|\|\mathbf A\mathbf v\|/\|\mathbf v\|-1|\geq\varepsilon)\leq
       2e^{-k\varepsilon^2/8}$. Hinweis: Drehen Sie das Koordinatensystem so, dass $\mathbf v$ auf der ersten Achse liegt.
    1. Wenden Sie für $n$ Punkte das Ergebnis aus (a) auf die $n(n-1)/2$ Differenzen an und nutzen Sie die boolesche Ungleichung. Zeigen Sie, dass alle paarweisen Abstände bis auf einen Faktor $1\pm\varepsilon$ erhalten bleiben, und zwar mit Wahrscheinlichkeit mindestens $1-\delta$, sobald $k\geq(8/\varepsilon^2)\log(n^2/\delta)$ gilt. Diese Schranke hängt nicht von $d$ ab. Das ist das Johnson-Lindenstrauss-Lemma. Werten Sie es für $n=1{,}000$, $\varepsilon=0.3$ und $\delta=0.05$ aus.
    1. Überprüfen Sie das Ergebnis an $1{,}000$ standardnormalverteilten Punkten bei $d=2{,}000$. Projizieren Sie auf $k\in\{50,200,800\}$ Dimensionen. Geben Sie die größte Verzerrung $|\|\mathbf A(\mathbf x_i-\mathbf x_j)\|/\|\mathbf x_i-\mathbf x_j\|-1|$ über alle Paare sowie die typische Schwankung $1/\sqrt{2k}$ an. Berechnen Sie die paarweisen Abstände aus der Gram-Matrix mit $\|\mathbf x_i-\mathbf x_j\|^2=\|\mathbf x_i\|^2+\|\mathbf x_j\|^2-2\,
       \mathbf x_i^\top\mathbf x_j$. Verzichten Sie dabei auf Broadcasting mit einem Array der Größe $1{,}000\times1{,}000\times2{,}000$. Erklären Sie das Verhältnis der beiden Zahlen anhand der Anzahl der Paare. Schon bei $k=200$ liegt die größte Verzerrung unter $\varepsilon=0.3$, während (b) ein wesentlich größeres $k$ verlangt. Welcher Teil der Schranke ist zu großzügig bemessen, die Konstante 8 oder der Faktor $\log n$?
   :exlabel:`ex_mdl-concentration-generalization-johnson-lindenstrauss`

<!--en:75b62b2d-->
1. **Bernstein-Ungleichung.** ● Sei $X$ zentriert mit $|X|\leq M$ und $\textrm{Var}(X)=\sigma^2$.
    1. Zeigen Sie, dass $\mathbb{E}[X^k]\leq\sigma^2M^{k-2}$ für $k\geq2$ gilt. Leiten Sie daraus $\mathbb{E}[e^{\lambda X}]\leq1+\sigma^2(e^{\lambda M}-1-\lambda M)/M^2$ für $\lambda>0$ ab.
    1. Zeigen Sie, dass $e^u-1-u\leq u^2/(2(1-u/3))$ für $0\leq u<3$ gilt. Vergleichen Sie dazu $u^k/k!$ mit $(u^2/2)(u/3)^{k-2}$, verwenden Sie also $k!\geq2\cdot3^{k-2}$. Folgern Sie mithilfe von $1+y\leq e^y$, dass $\mathbb{E}[e^{\lambda X}]\leq\exp(\lambda^2\sigma^2/(2(1-\lambda M/3)))$ für $0<\lambda<3/M$ gilt.
    1. Wenden Sie die Chernoff-Methode auf den Mittelwert von $n$ solchen Variablen an. Wählen Sie $\lambda=t/c$ mit $c=\sigma^2+Mt/3$. Prüfen Sie, dass $\lambda<3/M$ und $1-\lambda M/3=\sigma^2/c$ gelten. Folgern Sie, dass der Exponent genau $-nt^2/(2c)$ beträgt. Damit ist :eqref:`eq_mdl-bernstein` bewiesen.
    1. Zeigen Sie, dass $X$ nach (b) subexponentiell im Sinne von :eqref:`eq_mdl-subexponential` ist, mit $\nu^2=2\sigma^2$ und $\alpha=2M/3$.
    1. Zeigen Sie für $n=10{,}000$ und $\delta=0.05$, dass die Bernstein-Schranke bei der größtmöglichen Varianz $\sigma^2=\tfrac14$ eines 0/1-Ergebnisses schwächer ist als die Hoeffding-Schranke. Unterhalb welcher Varianz ist sie schärfer und welchen Fehlerraten entspricht das?
   :exlabel:`ex_mdl-concentration-generalization-bernstein`

<!--en:f64ec8df-->
[Diskussion](https://d2l.discourse.group/t/36834)

<!--en:7d99d695-->
Konzentrationsungleichungen und Maße in hohen Dimensionen<br>**Chernoff-Methode · Schranken von Hoeffding und Bernstein · subgaußsche und subexponentielle Randwahrscheinlichkeiten · dünne Kugelschale der Gaußverteilung · annähernde Orthogonalität**

<!--en:06612cd9-->
::: {.slide title="830 Fehler bei 10 000 Bildern. Wie verlässlich ist der Mittelwert?"}

<!--en:25d9aa8a-->
[Motivation und Kernfrage]{.kicker}

<!--en:dab830e9-->
- **Asymptotisches Intervall nach dem zentralen Grenzwertsatz ($\pm 0.54\%$ bei $95\%$):** Das Intervall ist schmal, gilt aber nur im Grenzfall $n \to \infty$ exakt und verwendet den eingesetzten Schätzwert $\hat{\sigma}$.

<!--en:c93660e1-->
- **Polynomielle Chebyshev-Schranke ($\sigma^2 / nt^2 \le \delta$):** Sie ergibt $\pm 2.24\%$ bei $\delta = 0.05$ und wächst auf aussagelose **$\pm 500\%$** bei $\delta = 10^{-6}$.

<!--en:ee74bb73-->
- **Exponentielle Konzentration (Hoeffding und Bernstein):** Hoeffding garantiert **$\pm 1.36\%$** bei $\delta = 0.05$ und **$\pm 2.69\%$** bei $\delta = 10^{-6}$. Bernstein berücksichtigt die Varianz und verschärft die Schranken auf **$\pm 0.79\%$** und **$\pm 1.66\%$**.

<!--en:043e3df1-->
::: {.slide title="Warum die polynomielle Chebyshev-Schranke zu grob ist"}

<!--en:ef71b7c8-->
[Polynomieller und exponentieller Abfall]{.kicker}

<!--en:4d7da2fb-->
Die Markov-Ungleichung für die quadrierte Abweichung $(\bar{X} - \mu)^2$ liefert die Chebyshev-Ungleichung $P(\lvert\bar{X} - \mu\rvert \ge t) \le \frac{\sigma^2}{nt^2}$. Ihre Schranke fällt nur wie $1/n$. Wie groß der Unterschied ist, zeigt ein Vergleich der Chebyshev-Schranke $25/n$ mit der exakten binomialen Randwahrscheinlichkeit $P(\lvert\hat{p} - 0.5\rvert \ge 0.1)$ für $n$ Würfe einer fairen Münze.

<!--en:c56bfa83-->
- **Acht Größenordnungen zu groß bei $n = 1{,}000$:** Die exakte Randwahrscheinlichkeit beträgt $2.7 \times 10^{-10}$, Chebyshev lässt $2.5 \times 10^{-2}$ zu. Bei $n = 5{,}000$ beträgt der Abstand bereits $42$ Größenordnungen.

<!--en:0c067d13-->
- **Warum zwei Momente nicht genügen:** Für diskrete Dreipunktverteilungen ist die Chebyshev-Schranke scharf. Eine bessere Rate als $1/n$ ergibt sich, wenn die momenterzeugende Funktion *alle* höheren Momente zugleich erfasst.

<!--en:d4f3ac01-->
::: {.slide title="Exponentielle Majoranten in der Chernoff-Methode"}

<!--en:4493de6f-->
[Alle Momente zugleich nutzen]{.kicker}

<!--en:dc12ae90-->
- **Majorante der Indikatorfunktion:** Für jede nichtnegative Funktion $g(x) \ge \mathbf{1}\{x \ge t\}$ gilt $P(X \ge t) \le \mathbb{E}[g(X)]$. Markov verwendet $x/t$, Chebyshev verwendet $(x/t)^2$ und Chernoff verwendet $e^{\lambda(x-t)}$ für $\lambda > 0$.
  $$P(X \ge t) = P\bigl(e^{\lambda X} \ge e^{\lambda t}\bigr) \le \inf_{\lambda > 0}\, e^{-\lambda t}\,\mathbb{E}\bigl[e^{\lambda X}\bigr]$$

<!--en:9de8a3fa-->
- **Produktregel für Summen unabhängiger Zufallsvariablen:** Sind $X_1, \dots, X_n$ unabhängig und ist $S = \sum_{i=1}^n X_i$, so gilt $\mathbb{E}[e^{\lambda S}] = \prod_{i=1}^n \mathbb{E}[e^{\lambda X_i}]$. Damit wird der Exponent zu einer Summe aus $n$ Termen.

<!--en:342228b7-->
::: {.slide title="Herleitung der exponentiellen Schranke für eine faire Münze"}

<!--en:b20cf19b-->
[Chernoff-Methode für zentrierte Münzwürfe]{.kicker}

<!--en:7d72fb04-->
- **1. MGF-Schranke für einen Wurf:** Für $Z_i \in \{-\frac{1}{2}, +\frac{1}{2}\}$ gilt $\mathbb{E}[e^{\lambda Z_i}] = \cosh(\lambda/2) \le e^{\lambda^2/8}$, denn $(2k)! \ge 2^k k!$.

<!--en:8ec68eeb-->
- **2. Faktorisieren und optimieren:** Für das Ereignis $\sum_{i=1}^n Z_i \ge nt$ liefert die Chernoff-Methode die Schranke $\exp(-\lambda nt + n\lambda^2/8)$. Am Scheitelpunkt $\lambda^* = 4t$ des Exponenten nimmt die Schranke ihr Minimum $e^{-2nt^2}$ an.

<!--en:6eb220c6-->
::: {.slide title="Hoeffdings Lemma für beliebige beschränkte Zufallsvariablen"}

<!--en:fa12bedd-->
[Schranke für die kumulantenerzeugende Funktion]{.kicker}

<!--en:5b9d4717-->
- **Maximale Varianz auf $[a, b]$:** Für jede Verteilung auf $[a, b]$ gilt $\operatorname{Var}(Y) \le (b-a)^2/4$.

<!--en:0b1ddfec-->
- **Krümmung unter der gekippten Verteilung:** Für die logarithmierte MGF $\psi(\lambda) = \log \mathbb{E}[e^{\lambda X}]$ gilt $\psi(0) = \psi'(0) = 0$. Ihre zweite Ableitung erfüllt $\psi''(\lambda) = \operatorname{Var}_{q_\lambda}(X) \le (b-a)^2/4$. Der Satz von Taylor liefert daher $\psi(\lambda) \le \lambda^2(b-a)^2/8$.

<!--en:e9f994a0-->
::: {.slide title="Hoeffding-Ungleichung für Summen unabhängiger beschränkter Zufallsvariablen"}

<!--en:d72cf239-->
[Schranke für die Randwahrscheinlichkeit bei endlicher Stichprobe]{.kicker}

<!--en:1fa03902-->
Die Chernoff-Methode mit dem Hoeffding-Lemma liefert für unabhängige Zufallsvariablen $X_1, \dots, X_n \in [a, b]$ den quadratischen Exponenten $-\lambda nt + \frac{n\lambda^2(b-a)^2}{8}$. Er nimmt sein Minimum bei $\lambda^* = \frac{4t}{(b-a)^2}$ an. Damit folgt die **Hoeffding-Ungleichung**.

<!--en:dc57133f-->
- **Beschränktheit und Unabhängigkeit genügen:** Die Schranke gilt für jeden endlichen Stichprobenumfang $n$ und setzt weder die Kenntnis der wahren Varianz noch den zentralen Grenzwertsatz voraus.

<!--en:b67b4fff-->
- **Erweiterung auf unterschiedlich lange Intervalle:** Gilt für unabhängige zentrierte Variablen $Y_i \in [a_i, b_i]$, so ergibt sich die Gesamtschranke als Produkt der einzelnen MGF-Schranken. An die Stelle von $n(b-a)^2$ tritt dann $\sum_{i=1}^n (b_i - a_i)^2$.
  $$P\Biggl(\Biggl\lvert\sum_{i=1}^n Y_i\Biggr\rvert \ge s\Biggr) \le 2\exp\!\left(-\frac{2s^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)$$

<!--en:6a149c06-->
::: {.slide title="Die Größe des Testdatensatzes mit Hoeffdings Konfidenzintervall bestimmen"}

<!--en:a4984420-->
[Stichprobenkomplexität der Auswertung]{.kicker}

<!--en:d3343867-->
Die Schranke für die Versagenswahrscheinlichkeit soll dem gewünschten Niveau entsprechen, also $2\exp\bigl(-\frac{2nt^2}{(b-a)^2}\bigr) = \delta$. Auflösen nach der Halbbreite $t$ oder dem Stichprobenumfang $n$ liefert für endliche Stichproben eine Garantie mit Wahrscheinlichkeit mindestens $1 - \delta$.

<!--en:e3f67938-->
- **Größe des Testdatensatzes für einen Benchmark ($[a,b] = [0,1]$):** Um die Fehlerrate bis auf $t = \pm 0.01$ ($1\%$) mit $95\%$ Konfidenz ($\delta = 0.05$) zu bestimmen, sind $n = \frac{\log 40}{2 \cdot 10^{-4}} = 18{,}445$ Testbeispiele nötig. Das sind etwa doppelt so viele wie die $9{,}604$ Beispiele der asymptotischen Gauß-Näherung.

<!--en:8f711513-->
- **Logarithmischer Aufwand bei sehr hoher Konfidenz:** Beim Übergang von $\delta = 0.05$ zu $\delta = 10^{-6}$ wächst $\log(2/\delta)$ um einen Faktor von weniger als $4\times$ ($n = 72{,}544$ Beispiele). Nach Chebyshev wächst der Bedarf wie $1/\delta$ und läge bei $2.5 \times 10^9$ Beispielen.

<!--en:75c18040-->
::: {.slide title="Randwahrscheinlichkeiten im Code vergleichen. Binomialverteilung und Hoeffding"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:c461ed69-->
- **Paralleler exponentieller Abfall:** Auf einer logarithmischen Achse verlaufen die Hoeffding-Schranke $2e^{-2nt^2}$ und die exakte binomiale Randwahrscheinlichkeit als nahezu parallele Geraden.

<!--en:2fc78b09-->
- **Abfallrate pro Wurf:** Der Logarithmus der Hoeffding-Schranke fällt um $2t^2 = 0.0200$ Nat pro Wurf. Von der exakten KL-Rate nach Cramér und Chernoff, $D_{\mathrm{KL}}(0.6 \,\|\, 0.5) = 0.0201$ Nat pro Wurf, weicht diese Rate um weniger als $1\%$ ab.

<!--en:7a92d8e3-->
- **Kleines $n$ ($n \le 13$):** Bei $9$-mal Kopf in $13$ Würfen ($t = 2.5/13$) ergibt die Hoeffding-Schranke $0.77$. Die exakte Randwahrscheinlichkeit beträgt $0.27$. Aus $13$ Würfen kann keine Ungleichung eine belastbare Aussage gewinnen.

<!--en:dae5ad76-->
::: {.slide title="Subgaußsche Zufallsvariablen und Varianzproxys"}

<!--en:0478e0c1-->
[Verallgemeinerung auf unbeschränkte Träger]{.kicker}

<!--en:ca128365-->
Hoeffdings Beweis nutzt nur, dass die MGF der zentrierten Variablen durch die MGF einer Gaußverteilung nach oben beschränkt ist. Eine Zufallsvariable $X$ heißt **subgaußsch** mit *Varianzproxy* $\sigma^2$, wenn die folgende Ungleichung gilt.

<!--en:c631f2bb-->
- **Typische subgaußsche Verteilungen:** (1) Für $\mathcal{N}(\mu, \sigma^2)$ gilt die Schranke mit Varianzproxy $\sigma^2$ mit Gleichheit. (2) Für jede beschränkte Variable $X \in [a,b]$ liefert Hoeffdings Lemma den Varianzproxy $\sigma^2 = (b-a)^2/4$. (3) Ein Rademacher-Vorzeichen $\xi \in \{-1, +1\}$ hat den Varianzproxy $\sigma^2 = 1$.

<!--en:26d0b8e1-->
- **Mittelung erhält die Eigenschaft:** Sind $X_1, \dots, X_n$ unabhängig und $\sigma^2$-subgaußsch, so ist $\bar{X}$ wiederum $(\sigma^2/n)$-subgaußsch und erfüllt $P(\lvert\bar{X} - \mathbb{E}[\bar{X}]\rvert \ge t) \le 2e^{-nt^2/(2\sigma^2)}$.

<!--en:af786b82-->
- **Warum $\sigma^2$ ein Proxy ist:** Die Entwicklung beider Seiten bis $O(\lambda^2)$ liefert $\operatorname{Var}(X) \le \sigma^2$. Bei einem $0/1$-Verlust mit $2\%$ Fehlerrate ist die Varianz $\operatorname{Var}(X) = 0.0196$ um den Faktor $12.7\times$ kleiner als Hoeffdings Varianzproxy $1/4$ für den ungünstigsten Fall.

<!--en:0aab8484-->
::: {.slide title="Warum quadrierte gaußsche Zufallsvariablen nur subexponentiell sind"}

<!--en:0142611c-->
[MGF-Schranken für zwei Bereiche]{.kicker}

<!--en:19a017c3-->
- **Polstelle der MGF bei $\lambda = 1/2$:** Für $x_i \sim \mathcal{N}(0, 1)$ ist die MGF durch $\mathbb{E}[e^{\lambda(x_i^2 - 1)}] = e^{-\lambda}/\sqrt{1 - 2\lambda}$ gegeben, solange sie endlich ist. Für $\lambda \ge 1/2$ divergiert der Erwartungswert. Die Schranke $e^{\lambda^2\nu^2/2}$ gilt daher nur im Bereich $|\lambda| \le 1/\alpha$.

<!--en:37ac7479-->
- **Zwei Bereiche für Randwahrscheinlichkeiten:** Die freie Optimierung liefert die gaußsche Schranke $e^{-nt^2/(2\nu^2)}$ für kleine Abweichungen $t \le \nu^2/\alpha$. Das Optimum am Rand liefert die exponentielle Schranke $e^{-nt/(2\alpha)}$ für große Abweichungen $t > \nu^2/\alpha$.

<!--en:297be539-->
::: {.slide title="Wie die Bernstein-Ungleichung eine kleine Varianz nutzt"}

<!--en:3e831985-->
[Konzentration unter Berücksichtigung der Varianz]{.kicker}

<!--en:b1253bc2-->
Erfüllen die Variablen die Schranke $\lvert X_i - \mathbb{E}[X_i]\rvert \le M$ und haben sie zudem eine kleine Varianz $\operatorname{Var}(X_i) \le \sigma^2 \ll M^2$, so berücksichtigt die **Bernstein-Ungleichung** beide Momente und verbindet den subgaußschen mit dem subexponentiellen Bereich.

<!--en:f1a42811-->
- **Annäherung an den zentralen Grenzwertsatz:** Bei einer Fehlerrate von $2\%$ auf $n = 10{,}000$ Bildern berücksichtigt die Fixpunktgleichung die größtmögliche wahre Varianz $(\hat{p}+t)(1-\hat{p}-t)$. Die daraus berechnete garantierte Halbbreite **$\pm 0.43\%$** ist um mehr als den Faktor $3\times$ kleiner als Hoeffdings $\pm 1.36\%$.

<!--en:41b499c0-->
::: {.slide title="Paradoxon in hohen Dimensionen. Kaum Masse am Modus der Gaußverteilung"}

<!--en:8f65deb7-->
[Maßkonzentration in R^d]{.kicker}

<!--en:1ecc0192-->
- **Wo liegt $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{784})$ typischerweise?** Die Dichte $(2\pi)^{-d/2}e^{-\|\mathbf{x}\|^2/2}$ hat ihr Maximum im Ursprung $\mathbf{0}$. Dennoch beträgt die Wahrscheinlichkeit, höchstens den Abstand $14$ vom Ursprung zu haben, nur $\approx 10^{-110}$.

<!--en:28b6c3fe-->
- **Dichte und Kugelvolumen:** Bei der Integration über Kugelschalen mit Radius $r = \|\mathbf{x}\|$ ist die abfallende Dichte $e^{-r^2/2}$ mit dem rasch wachsenden Flächeninhalt der Kugeloberfläche ($\propto r^{d-1}$) zu multiplizieren.
  $$p(r) \propto r^{d-1}e^{-r^2/2}$$

<!--en:6e4a417a-->
- **Die dünne Kugelschale:** Die radiale Dichte erreicht ihr Maximum bei $r_{\mathrm{mode}} = \sqrt{d - 1} \approx \sqrt{d}$. Für $d = 784$ konzentriert sich die Wahrscheinlichkeitsmasse in einer dünnen Kugelschale, deren Radius typischerweise bei $\sqrt{784} = 28.0 \pm 0.7$ liegt.

<!--en:a6f048e8-->
::: {.slide title="Satz über die Konzentration der euklidischen Norm"}

<!--en:6fd8061f-->
[Schranke für die dünne Kugelschale]{.kicker}

<!--en:36ddef04-->
Der Ausdruck $\frac{1}{d}\|\mathbf{x}\|^2 = \frac{1}{d}\sum_{i=1}^d x_i^2$ ist der Stichprobenmittelwert von $d$ unabhängigen quadrierten gaußschen Zufallsvariablen. Damit lässt sich die Geometrie in hohen Dimensionen durch eine Konzentrationsungleichung beschreiben.

<!--en:5f353713-->
1. **Subexponentielle Koordinatenquadrate:** Die Entwicklung in eine Potenzreihe ergibt $\mathbb{E}[e^{\lambda(x_i^2 - 1)}] = \frac{e^{-\lambda}}{\sqrt{1-2\lambda}} \le e^{2\lambda^2}$ für $\lvert\lambda\rvert \le \frac{1}{4}$. Somit ist jedes $x_i^2$ eine $(\nu = 2, \alpha = 4)$-subexponentielle Zufallsvariable.

<!--en:088875f2-->
2. **Konzentration der quadrierten Norm:** Die subexponentielle Schranke für die Randwahrscheinlichkeit liefert mit $n = d$ und $t = \varepsilon \le 1$ die Ungleichung $P\bigl(\bigl\lvert\frac{\|\mathbf{x}\|^2}{d} - 1\bigr\rvert \ge \varepsilon\bigr) \le 2\exp\bigl(-\frac{d}{2}\min(\frac{\varepsilon^2}{4}, \frac{\varepsilon}{4})\bigr) = 2e^{-d\varepsilon^2/8}$.

<!--en:fce6c448-->
3. **Von der quadrierten Norm zur Norm:** Für $z = \|\mathbf{x}\|/\sqrt{d} \ge 0$ folgt aus $\lvert z - 1\rvert \ge \varepsilon$ die Ungleichung $\lvert z^2 - 1\rvert = \lvert z - 1\rvert(z + 1) \ge \varepsilon$, denn $z + 1 \ge 1$.

<!--en:1a7c2330-->
::: {.slide title="Konstante Schalendicke in jeder Dimension"}

<!--en:4c33ac31-->
[Dicke O(1) bei Radius sqrt(d)]{.kicker}

<!--en:adabf65b-->
Wie dick ist die Schale um den Radius $\sqrt{d}$? Wegen $\mathbb{E}[x_i^4] = 3$ hat jedes Koordinatenquadrat die Varianz $\operatorname{Var}(x_i^2) = 3 - 1 = 2$. Daraus folgt $\operatorname{Var}(\|\mathbf{x}\|^2) = 2d$. Die Linearisierung der Quadratwurzel um $d$ zeigt, dass die Schalendicke von der Dimension unabhängig ist:

<!--en:2e9293c4-->
::: {.slide title="Nahezu orthogonale Zufallsvektoren in hohen Dimensionen"}

<!--en:9c71e6d5-->
[Winkel in hohen Dimensionen]{.kicker}

<!--en:d40c5224-->
- **Hoeffding-Schranke für unterschiedliche Wertebereiche:** Für einen Einheitsvektor $\mathbf{u}$ und zufällige Vorzeichen $\mathbf{x} \in \{\pm 1\}^d$ gilt für die Summanden $Y_i = u_i x_i \in [-|u_i|, |u_i|]$ die Beziehung $\sum (2|u_i|)^2 = 4$. Damit bleibt typischerweise $\langle\mathbf{x}, \mathbf{u}\rangle = O(1)$, während $\|\mathbf{x}\| = \sqrt{d}$ gilt.

<!--en:14e713ce-->
- **Vergleichsmaßstab für Einbettungen ($d = 784$):** Für Zufallsvektoren gilt $\mathrm{sd}(\cos) = 1/\sqrt{784} = 0.036$ ($90^\circ \pm 2.1^\circ$). Ein Wert von $\cos = 0.30$ liegt also $>8\sigma$ über dem Mittelwert.

<!--en:3a081a62-->
::: {.slide title="Platz für exponentiell viele nahezu orthogonale Vektoren"}

<!--en:080ca3c6-->
[Die boolesche Ungleichung in R^d]{.kicker}

<!--en:6122c253-->
- **Strenge und angenäherte Orthogonalität:** Bei strenger Orthogonalität ($\cos = 0$) finden höchstens $m = d$ Vektoren in $\mathbb{R}^d$ Platz.

<!--en:0be357c1-->
- **Exponentiell viel Platz in $\mathbb{R}^{784}$:** Unter der Bedingung $|\cos| < \varepsilon$ finden $m \approx 570$ Vektoren bei $\varepsilon = 0.2$ und **$m \approx 10^7$ Vektoren** bei $\varepsilon = 0.3$ Platz ($\delta = 0.05$).

<!--en:0a1541d6-->
::: {.slide title="Schwindender Kontrast paarweiser euklidischer Abstände"}

<!--en:c7cd27e8-->
[Fluch der Dimensionalität bei nächsten Nachbarn]{.kicker}

<!--en:0e182179-->
Für zwei unabhängige gaußsche Zufallsvektoren $\mathbf{x}, \mathbf{y} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ gilt $\mathbf{x} - \mathbf{y} \sim \mathcal{N}(\mathbf{0}, 2\mathbf{I}_d)$. Damit konzentriert sich $\|\mathbf{x} - \mathbf{y}\| / \sqrt{2d}$ um $1$, mit der Schranke $2e^{-d\varepsilon^2/8}$ für die Randwahrscheinlichkeit:

<!--en:23e97b6c-->
- **Zufallspunkte haben annähernd gleiche Abstände:** Nach der booleschen Ungleichung für $\binom{n}{2}$ Paare liegen alle paarweisen Abstände zwischen $n$ Zufallspunkten in $(1 \pm \varepsilon)\sqrt{2d}$, sobald $d \ge \frac{8}{\varepsilon^2}\log(n^2/\delta)$ gilt. Bei $d = 2000$ ist der entfernteste von $200$ zufälligen Nachbarn nur $8\%$ weiter entfernt als der nächste.

<!--en:a21608bc-->
::: {.slide title="Folge 1: Gewichtsinitialisierung in tiefen Netzen"}

<!--en:d2b593aa-->
[Schwankungen bei LeCun- und He-Initialisierung]{.kicker}

<!--en:73ae2fbd-->
- **Schwankung in einer Schicht:** Bei He-Initialisierung ($W_{ij} \sim \mathcal{N}(0, 2/d)$) schwankt das Verhältnis der quadrierten Normen in einer ReLU-Schicht der Breite $d$ um $\operatorname{sd}(\rho_k) = \sqrt{5/d} \approx 10\%$ für $d = 512$.

<!--en:94c20c94-->
- **Verhältnis von Tiefe zu Breite $K/d$:** Bei Tiefe $K$ gilt $\operatorname{Var}\bigl(\log\frac{\|\mathbf{h}_K\|^2}{\|\mathbf{h}_0\|^2}\bigr) \approx \frac{5K}{d}$. Damit die Norm nicht driftet, muss $K \ll d$ gelten (alternativ helfen LayerNorm oder Residualverbindungen).

<!--en:3c004577-->
::: {.slide title="Johnson-Lindenstrauss-Lemma und Kreisbögen im latenten Raum"}

<!--en:3a8e1700-->
[Zufällige Projektionen und Interpolation in generativen Modellen]{.kicker}

<!--en:dabb05d0-->
- **Johnson-Lindenstrauss-Projektionen:** Eine Zufallsmatrix $\mathbf{A} \in \mathbb{R}^{k \times d}$ mit $\mathcal{N}(0, 1/k)$-verteilten Einträgen erhält alle paarweisen Abstände zwischen $n$ Punkten bis auf einen Faktor $1 \pm \varepsilon$, sobald $k \ge \frac{8}{\varepsilon^2}\log\frac{n^2}{\delta}$ gilt (unabhängig von $d$).

<!--en:8d89c60e-->
- **Sphärische (Slerp) und lineare Interpolation (Lerp):** Die Vektoren $\mathbf{z}_1, \mathbf{z}_2 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ haben typischerweise die Norm $\approx \sqrt{d}$ und schließen den Winkel $\theta \approx 90^\circ$ ein. Der Mittelpunkt ihrer Verbindungsstrecke liegt mit der Norm $\sqrt{d/2} \approx 0.71\sqrt{d}$ im leeren Inneren. Slerp bleibt auf der Schale der Trainingsdaten mit Radius $\sqrt{d}$.

<!--en:7defc6e1-->
::: {.slide title="Grenzen der Schranke für einen Mittelwert nach der Modellauswahl"}

<!--en:112320e6-->
[Übergang zur gleichmäßigen Konvergenz]{.kicker}

<!--en:e7b1ad81-->
Alle Konzentrationsungleichungen in diesem Abschnitt (Hoeffding-, Bernstein- und subexponentielle Normschranken) beruhen auf derselben Voraussetzung:

<!--en:364316d4-->
**Voraussetzung einer festen Variablen:** Die zu mittelnde Zufallsvariable (der Klassifikator $f$ auf einem Testdatensatz oder der zu projizierende Vektor $\mathbf{v}$) muss feststehen, **bevor** die Zufallsstichprobe $S$ gezogen wird.

<!--en:52e80efc-->
- **Was passiert beim Training und bei der Modellauswahl?** Ein Lernalgorithmus sichtet den Datensatz $S$, bewertet Tausende oder unendlich viele Funktionen $f \in \mathcal{F}$ und wählt die empirisch beste aus: $\hat{f} = \operatorname{argmin}_{f \in \mathcal{F}} \hat{R}_S(f)$.

<!--en:6309428f-->
- **Selektionsbias des Minimums:** Selbst wenn jeder Kandidat das wahre Risiko $R(f) = 0.5$ hat, liegt der beste von $1{,}000$ empirischen Mittelwerten für $n = 100$ Beispiele systematisch nahe bei $0.34$. Eine Garantie für die datenabhängige Wahl $\hat{f}$ erfordert **gleichmäßige Konvergenz** auf der gesamten Klasse $\mathcal{F}$ (§{sec_mdl-generalization-bounds}).

<!--en:6b26f071-->
::: {.slide title="Schranken für Randwahrscheinlichkeiten im Vergleich"}

<!--en:c231a18c-->
[Markov, Chebyshev, Hoeffding, Bernstein und subexponentielle Schranken]{.kicker}

<!--en:09ad8349-->
- **Polynomieller und exponentieller Abfall:** Die Schranken von Markov ($1/\delta$) und Chebyshev ($1/(\delta t^2)$) nutzen nur ein oder zwei Momente. Der nötige Stichprobenumfang wächst für kleine $\delta$ rasant.

<!--en:2eca162f-->
- **Exponentielle Schranken aus der MGF:** Subexponentielle Schranken sowie die Schranken von Hoeffding und Bernstein nutzen über $e^{\lambda X}$ alle Momente. Der nötige Stichprobenumfang wächst logarithmisch wie $\log(2/\delta)$.

<!--en:ad068ac1-->
- **Einfluss der Varianz:** Bernstein nutzt anstelle von Hoeffdings Abschätzung über die Spannweite $(b-a)^2$ die wahre Varianz $\sigma^2$.

<!--en:650b2a69-->
::: {.slide title="Verständnisfragen zur Konzentration in hohen Dimensionen"}

<!--en:929b7671-->
[Wissenskontrolle]{.kicker}

<!--en:9fe435dd-->
1. **Testdatensatzgröße für $99\%$ Konfidenz:** Wie viele zurückgehaltene Testbeispiele braucht die Hoeffding-Ungleichung bei einem $0/1$-Klassifikationsverlust, damit die empirische Genauigkeit höchstens um $\pm 1\%$ ($t = 0.01$) von der wahren Genauigkeit abweicht, mit $99\%$ Konfidenz ($\delta = 0.01$)?

<!--en:2ccf0686-->
2. **Sehnenmittelpunkt und Schale im latenten Raum:** Das arithmetische Mittel zweier unabhängiger latenter Vektoren $\mathbf{z}_1, \mathbf{z}_2 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{512})$ ist der Mittelpunkt $\mathbf{z}_{\mathrm{mid}} = \frac{1}{2}(\mathbf{z}_1 + \mathbf{z}_2)$. Welche euklidischen Normen haben $\mathbf{z}_1$, $\mathbf{z}_2$ und $\mathbf{z}_{\mathrm{mid}}$ typischerweise?

<!--en:2c616021-->
3. **Warum Bernstein bei niedrigen Fehlerraten eine engere Schranke liefert:** Ein Klassifikator hat eine Fehlerrate von $1\%$ auf $n = 10{,}000$ Bildern. Warum ist das Hoeffding-Konfidenzintervall mehr als $3\times$ so breit wie das Bernstein-Konfidenzintervall?

<!--en:557b98b6-->
**Lösungen:**

<!--en:9f10097e-->
1. $n \ge \frac{\log(2/\delta)}{2t^2} = \frac{\log(200)}{2 \times 10^{-4}} \approx \frac{5.2983}{0.0002} = 26{,}492$ Testbeispiele.

<!--en:32922eda-->
2. Für typische Ziehungen gilt $\|\mathbf{z}_1\| \approx \|\mathbf{z}_2\| \approx \sqrt{512} \approx 22.6 \pm 0.7$. Dagegen hat $\mathbf{z}_{\mathrm{mid}} \sim \mathcal{N}(\mathbf{0}, \frac{1}{2}\mathbf{I}_{512})$ typischerweise die Norm $\sqrt{256} = 16.0 \pm 0.5$. Das liegt mehr als $9$ Standardabweichungen unter dem typischen Radius der Trainingsdaten.

<!--en:6e025eb4-->
3. Hoeffding verwendet für jede Zufallsvariable in $[0,1]$ den ungünstigsten Varianzproxy $\sigma^2 = 1/4 = 0.25$ (nur bei $p = 0.5$ erreicht). Bernstein nutzt die tatsächliche Bernoulli-Varianz $\sigma^2 = 0.01 \times 0.99 \approx 0.01$, die um den Faktor $25\times$ kleiner ist.

<!--en:9a38c6d0-->
::: {.slide title="Zusammenfassung: Konzentration und hochdimensionale Geometrie"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:f932efc8-->
| Konzept | Mathematische Form | Bedeutung für Statistik und Deep Learning |

<!--en:dd3c7004-->
| **Chernoff-Methode** | $P(S \ge nt) \le \inf_{\lambda > 0} e^{-\lambda nt}\prod_{i=1}^n \mathbb{E}[e^{\lambda X_i}]$ | Produkt der MGFs unabhängiger Summanden liefert exponentiellen Abfall |

<!--en:067ca1ef-->
| **Hoeffding-Ungleichung** | $\lvert\bar{X} - \mu\rvert \le (b-a)\sqrt{\frac{\log(2/\delta)}{2n}}$ mit Wahrscheinlichkeit $\ge 1-\delta$ | Verteilungsfreier Fehlerbalken bei endlichem Stichprobenumfang. $\log(1/\delta)$ ermöglicht die Anwendung der booleschen Ungleichung |

<!--en:1858ddd2-->
| **Bernstein-Ungleichung** | $P(\lvert\bar{X} - \mu\rvert \ge t) \le 2\exp\!\left(-\frac{nt^2}{2(\sigma^2 + Mt/3)}\right)$ | Engere Konfidenzintervalle bei kleiner Fehlervarianz $\sigma^2 \ll (b-a)^2/4$ |

<!--en:c29b407a-->
| **Dünne gaußsche Kugelschale** | $P\bigl(\bigl\lvert\frac{\|\mathbf{x}\|}{\sqrt{d}} - 1\bigr\rvert \ge \varepsilon\bigr) \le 2e^{-d\varepsilon^2/8}$, $\operatorname{sd}(\|\mathbf{x}\|) \approx \frac{1}{\sqrt{2}}$ | $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ konzentriert sich auf eine Schale mit Radius $\sqrt{d}$ und Dicke $O(1)$. Slerp berücksichtigt diese Geometrie |

<!--en:ec227f4e-->
| **Nahezu orthogonale Richtungen** | $P(\lvert\cos(\mathbf{x}, \mathbf{u})\rvert \ge \varepsilon) \le 2e^{-d\varepsilon^2/2}$ | Exponentiell viele ($e^{d\varepsilon^2/4}$) nahezu orthogonale Richtungen finden in $\mathbb{R}^d$ Platz |

<!--en:e2b04a0d-->
| **Tiefe und zufällige Projektionen** | $\operatorname{Var}\bigl(\log\frac{\|\mathbf{h}_K\|^2}{\|\mathbf{h}_0\|^2}\bigr) \approx \frac{5K}{d}$, JL: $k \ge \frac{8}{\varepsilon^2}\log\frac{n^2}{\delta}$ | Erklärt, warum die He-Initialisierung $K \ll d$ erfordert und warum Projektionen auf $O(\log n)$ Dimensionen Abstände näherungsweise erhalten |
