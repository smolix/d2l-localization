<!--en:546b4f5c-->
# Numerische Verfahren

<!--en:e43f4612-->
Wer schon einmal ein Netz trainiert hat, kennt Trainingsläufe, die wegen einer zu großen Lernrate (engl. learning rate) divergieren. Da liegt es nahe, die Ursache in den steilen Tälern und Plateaus der Verlustlandschaft (engl. loss landscape) zu suchen. Doch dasselbe passiert schon bei der einfachsten Verlustfunktion. Auf der Parabel $\mathcal L(x)=25x^2$ macht der Gradientenabstieg (engl. gradient descent) den Schritt $x\leftarrow x-\eta\cdot 50x$. Bei jeder Lernrate oberhalb von $\eta=0.04$ wachsen die Beträge der Iterierten unbegrenzt, obwohl die Parabel nur ein Minimum hat und keinerlei Schwierigkeiten birgt. Wenn es nicht an der Verlustfunktion liegt, woran dann?

<!--en:91530045-->
Die Antwort liefert ein älteres Gebiet der Mathematik. Der Gradientenabstieg löst schrittweise die Differentialgleichung $\dot x=-50x$. Ihr zufolge bewegen wir uns bergab, und zwar umso schneller, je steiler der Hang ist. Das ist die Zerfallsgleichung aus :numref:`sec_mdl-odes-solvers` mit fünfzigfacher Zerfallsrate. Ihre exakte Lösung $x(t)=e^{-50t}x_0$ klingt von jedem Anfangswert aus gegen null ab. Euler schlug 1768 vor, zur Lösung einer solchen Gleichung dem Richtungspfeil für eine kurze Zeit $h$ zu folgen und dann die Richtung neu zu bestimmen. Die Lernrate übernimmt die Rolle von $h$. Ein Schritt multipliziert $x$ mit $1-50h$. Für $h=0.01$ beträgt der Faktor $0.5$, und die berechnete Lösung klingt wie die exakte ab. Für $h=0.05$ beträgt er $-1.5$, während der exakte Faktor $e^{-2.5}=0.08$ ist. Die berechnete Lösung wechselt das Vorzeichen und wächst im Betrag. Die Gleichung ist stabil, die Schritte sind es nicht. Im vorigen Abschnitt bestimmten die Eigenwerte einer Gleichung die exakte Bewegung. Hier entscheiden dieselben Eigenwerte, mit welcher Schrittweite wir dieser Bewegung folgen können.

<!--en:fa89fc08-->
Dieser Unterschied zwischen einer Gleichung und ihren diskreten Schritten begegnet uns im Deep Learning überall dort, wo wir schrittweise vorgehen: beim Gradientenabstieg (:numref:`sec_gd`), in den Schichten eines Residualnetzes (:numref:`sec_resnet`) und in den Samplern von Diffusions- und Flow-Matching-Modellen (:numref:`sec_diffusion-samplers`). Ein Sampler wertet pro Schritt ein Netz ein- oder zweimal aus. Die Anzahl der Schritte bestimmt also den Rechenaufwand pro Bild.

<!--en:38fc6e00-->
Drei Fragen entscheiden über die Güte einer schrittweise berechneten Lösung. Wie weit weicht sie von der exakten Lösung ab? Wie lässt sich diese Abweichung bei gleichem Aufwand verringern? Und wie groß darf ein Schritt sein, damit Fehler noch abklingen und nicht anwachsen? Alle drei Fragen beantwortet derselbe Faktor: der Faktor, mit dem ein Schritt die Lösung multipliziert. Beim Euler-Verfahren auf der Parabel war das $1-50h$. Zum Schluss kommen wir auf den Gradientenabstieg zurück und fragen, was ein besseres numerisches Verfahren (engl. numerical solver) für das Training leisten würde.

<!--en:645cb1c5-->
## Explizites Euler-Verfahren

<!--en:5d44593e-->
Die Position auf einer Trajektorie ergibt sich aus der Anfangsposition und dem Integral der Geschwindigkeit entlang des Weges. Für einen kurzen Schritt wählt Euler die gröbste mögliche Näherung: Die Geschwindigkeit behält über den ganzen Schritt ihren Anfangswert. Sehen wir uns zunächst an einer Gleichung mit bekannter Lösung an, was das bewirkt.

<!--en:0f030a11-->
Betrachten wir $\dot x=x$ mit Anfangswert $x_0=1$ bis zum Zeitpunkt $T=1$. Dort ist die Lösung $e=2.718$. Wir teilen das Intervall in $n$ Schritte der Länge $h=1/n$ und bewegen uns bei jedem Schritt mit der Geschwindigkeit an seinem Anfang: $x\leftarrow x+hx=(1+1/n)x$. Nach $n$ Schritten gilt $x_n=(1+1/n)^n$. Genauso wächst ein Dollar bei 100% Jahreszins, wenn die Zinsen $n$-mal im Jahr gutgeschrieben und mitverzinst werden:

<!--en:de84b685-->
| Fehler | 0.718 | 0.468 | 0.277 | 0.153 | 0.080 | 0.041 | 0.0037 |

<!--en:ef86f06a-->
Die Quotienten aufeinanderfolgender Fehler betragen $1.53,1.69,1.82,1.90,1.95$. Eine Verdopplung der Schrittzahl kommt einer Halbierung des Fehlers also immer näher. Selbst bei täglicher Verzinsung fehlt noch mehr als ein Drittel Cent bis zu $e$ Dollar.

<!--en:e1d340b7-->
Dieser langsamen Konvergenz sind wir bereits begegnet. Die wiederholte Multiplikation $(\mathbf I+t\mathbf A/n)^n$ aus :numref:`sec_mdl-odes-solvers` war das Euler-Verfahren, bevor wir es so genannt haben: $n$ Schritte der Länge $t/n$ entlang eines linearen Feldes. Die dort hergeleitete Entwicklung $(1+\lambda t/n)^n\approx e^{\lambda t}\bigl(1-\tfrac{(\lambda t)^2}{2n}\bigr)$ liefert für $\lambda t=1$ den Fehler bei der Verzinsung des Dollars. Er beträgt ungefähr $e/(2n)=1.36\,h$. Ist der Fehler proportional zu $h$, sprechen wir von einem Verfahren erster Ordnung.

<!--en:82b26256-->
Für ein allgemeines Feld $\mathbf f(\mathbf x,t)$ mit Schrittweite $h$ und Zeitgitter $t_n = nh$ zählt $n$ von nun an die bereits ausgeführten Schritte. Das **explizite Euler-Verfahren** (engl. forward Euler) lautet

<!--en:6aeaa37c-->
Das Verfahren folgt dem Pfeil am aktuellen Punkt für die Dauer $h$ und wertet dann das Feld erneut aus. Das entspricht der Taylor-Entwicklung der exakten Lösung bis zum linearen Term. Der erste weggelassene Term bestimmt den führenden Fehler eines Schritts. Gilt entlang der Lösung $\|\ddot{\mathbf{x}}(t)\| \le M$, so folgt

<!--en:bc4a5936-->
Das Restglied $\mathbf r_n$ ist der **lokale Fehler** (engl. local error): die Abweichung nach einem einzigen Schritt, der auf der exakten Lösung beginnt. Uns interessiert der **globale Fehler** (engl. global error), also der Abstand von der exakten Lösung nach allen Schritten bis zu einem festen Endzeitpunkt $T$. Ein lokaler Fehler der Ordnung $h^2$ ergibt keinen globalen Fehler der Ordnung $h^2$. Bis $T$ brauchen wir $T/h$ Schritte. Ihre lokalen Fehler summieren sich, und jeder Schritt kann den bisherigen Fehler verstärken. Eine Potenz von $h$ geht verloren. Der Beweis ist kurz, und sein Aufbau ist wichtiger als die Konstante.

<!--en:81759ae2-->
**Proposition (Konvergenz des Euler-Verfahrens mit Ordnung 1).** *Sei $\mathbf{f}$ Lipschitz-stetig mit Konstante $L$ bezüglich $\mathbf{x}$, das heißt, es gelte $\|\mathbf f(\mathbf x,t)-\mathbf f(\mathbf y,t)\|\le L\|\mathbf x-\mathbf y\|$ für alle $\mathbf x$, $\mathbf y$ und $t$. Für die Lösung von $\dot{\mathbf x}=\mathbf f(\mathbf x,t)$ mit Anfangswert $\mathbf x(0)=\mathbf x_0$ gelte $\|\ddot{\mathbf{x}}(t)\| \le M$ auf $[0, T]$. Dann erfüllen die Euler-Iterierten aus :eqref:`eq_mdl-ode-euler-update` mit demselben Anfangswert die Abschätzung*

<!--en:b42bdd0e-->
**Beweis.** Setzen wir $\varepsilon_n = \|\mathbf{x}_n - \mathbf{x}(t_n)\|$ und ziehen die obige Taylor-Identität von der Aktualisierung :eqref:`eq_mdl-ode-euler-update` ab:

<!--en:d1761ff8-->
Jeder Schritt multipliziert den bisherigen Fehler mit höchstens $1 + hL$ und addiert höchstens $Mh^2/2$. Lösen wir die Rekursion ausgehend von $\varepsilon_0 = 0$ schrittweise auf, so erhalten wir eine geometrische Summe:

<!--en:9a66f6c6-->
Dabei haben wir $(1 + hL)^n \le e^{nhL} \le e^{LT}$ verwendet. $\blacksquare$

<!--en:9273439c-->
Für $\dot x=x$ auf $[0,1]$ gilt $M=\max|\ddot x|=e$ und $L=1$. Die Schranke beträgt also $\tfrac e2(e-1)h=2.3\,h$, der tatsächliche Fehler dagegen $1.36\,h$: Die Ordnung stimmt, die Konstante ist pessimistisch. Bei jedem numerischen Verfahren hat die Schranke denselben Aufbau. Aus einem lokalen Fehler der Ordnung $h^{p+1}$ pro Schritt wird ein globaler Fehler der Ordnung $h^p$ nach $T/h$ Schritten. Hinzu kommt der Faktor $e^{LT}$, der angibt, wie stark die Dynamik frühere Fehler verstärkt. Ein Verfahren mit globalem Fehler der Ordnung $h^p$ hat die **Ordnung** $p$. Das Euler-Verfahren hat die Ordnung $1$.

<!--en:019c23f2-->
Der Faktor $e^{LT}$ macht die Abschätzung grob. Die Konstante $L$ erfasst die Steilheit des Feldes, aber nicht die Richtung seiner Neigung. Daher unterscheidet die Schranke nicht zwischen einem Feld, das Trajektorien auseinandertreibt, und einem, das sie zusammenführt. Für die Gleichung $\dot x=-50x$ der Parabel auf $[0,1]$ beträgt der Faktor $e^{50}\approx5\times10^{21}$. Damit ist die Schranke bei jeder praktisch berechenbaren Schrittweite unbrauchbar, obwohl kleine Schritte den Zerfall gut nachbilden. Ein kontrahierendes Feld erfordert eine eigene Analyse. Dazu kommen wir, nachdem wir uns Verfahren höherer Ordnung angesehen haben.

<!--en:60d9f9de-->
## Runge-Kutta-Verfahren

<!--en:276d4a4e-->
Das Euler-Verfahren behält die Steigung am Anfang des Schritts bei, während die Trajektorie von dieser Richtung abbiegt. Runge-Kutta-Verfahren messen die Steigung auch innerhalb des Schritts und kombinieren die Messwerte. :numref:`fig_mdl-dyn-solver-geometry` zeigt einen Schritt der Weite $h=1$ für $\dot x=x$ vom Anfangswert $x=1$ aus.

<!--en:7a3f026f-->
![Ein Schritt der Weite $h=1$ für $\dot x=x$ vom Anfangswert $x=1$ aus. (a) Euler folgt der Anfangssteigung und erreicht $2$. (b) Das Mittelpunktverfahren macht einen halben Schritt, nur um dort die Steigung $1.5$ zu messen, und erreicht mit dieser Steigung $2.5$. (c) RK4 misst die vier Steigungen $1$, $1.5$, $1.75$ und $2.75$, mittelt sie mit den Gewichten $\tfrac16(1,2,2,1)$ und erreicht $2.708$. Der exakte Wert ist $e=2.718$.](../img/mdl-dyn-solver-geometry.svg)

<!--en:30fba58b-->
### Mittelpunktverfahren, Heun-Verfahren und RK4

<!--en:3fb3d4fc-->
Die einfachste Verbesserung, das **Mittelpunktverfahren** (engl. midpoint method), macht einen halben Euler-Schritt, nur um dort die Steigung zu messen. Mit dieser Steigung macht es dann den eigentlichen Schritt. Schreiben wir $\mathbf f(\mathbf x)$ für ein zeitunabhängiges Feld, so lautet die Aktualisierung $\mathbf{x}_{n+1} = \mathbf{x}_n + h\,\mathbf{f}\!\left(\mathbf{x}_n + \tfrac{h}{2}\mathbf{f}(\mathbf{x}_n)\right)$. In einer Dimension zeigt eine kurze Taylor-Entwicklung, was wir dadurch gewinnen:

<!--en:9ac2361f-->
Die exakte Entwicklung lautet $x(t+h) = x + h\dot{x} + \tfrac{h^2}{2}\ddot{x} + O(h^3)$ mit $\ddot{x} = \tfrac{d}{dt} f(x(t)) = f' f$. Damit ist der Term in $h^2$ korrekt berücksichtigt, der beim Euler-Verfahren fehlt. Der lokale Fehler beträgt $O(h^3)$, die globale Ordnung ist $2$. Dazu brauchen wir zwei Feldauswertungen pro Schritt.

<!--en:e7c79286-->
Das Mittelpunktverfahren nutzt die erste Auswertung für einen halben Schritt. Das **Heun-Verfahren** macht stattdessen einen ganzen Euler-Probeschritt und mittelt die Steigungen an den beiden Enden des Schritts:

<!--en:da5f42f5-->
Das Mittelpunktverfahren entspricht bei der Integration der Mittelpunktregel, das Heun-Verfahren der Trapezregel. Dieselbe Taylor-Entwicklung ergibt dieselbe Ordnung $2$, ebenfalls bei zwei Auswertungen. Von den beiden Verfahren sollten wir uns vor allem das Heun-Verfahren merken, denn viele Diffusionssampler nutzen es (:numref:`sec_diffusion-samplers`).

<!--en:7943ba0a-->
Vier Steigungsmessungen führen zu einem Standardverfahren für das wissenschaftliche Rechnen, dem **klassischen Runge-Kutta-Verfahren (RK4)**. Es geht auf Kutta im Jahr 1901 zurück:

<!--en:e6175620-->
Die Aktualisierung mittelt vier Steigungen und gewichtet dabei die beiden Messwerte in der Mitte doppelt. Das sind die Gewichte der Simpson-Regel $\int_a^b g \approx \tfrac{b-a}{6}\bigl(g(a) + 4g(\tfrac{a+b}{2}) + g(b)\bigr)$, die kubische Polynome exakt integriert. Hier verteilt sich das Gewicht der Mitte auf zwei Messungen.

<!--en:811dac54-->
### Die Stabilitätsfunktion

<!--en:0f1418a3-->
Warum hat das Verfahren die Ordnung $4$? Bei einem linearen Feld genügt eine Rechnung. Betrachten wir die skalare Gleichung $\dot y=\lambda y$ mit reellem oder komplexem $\lambda$. Diese *Testgleichung* steht stellvertretend für jedes lineare Feld: Jede Eigenmode von $\dot{\mathbf x}=\mathbf A\mathbf x$ erfüllt sie, wobei $\lambda$ ein Eigenwert von $\mathbf A$ ist (:numref:`sec_mdl-odes-solvers`). Die exakte Lösung ändert sich um den Faktor $e^{h\lambda}$ in der Zeit $h$. Jedes Verfahren multipliziert $y$ pro Schritt mit einer festen Funktion von $z=h\lambda$. Diese Funktion können wir mit $e^z$ vergleichen.

<!--en:c529b63f-->
**Proposition (Stabilitätsfunktionen).** *Für die Testgleichung mit $z=h\lambda$ multipliziert jeder Schritt eines Verfahrens den Wert $y$ mit der zugehörigen **Stabilitätsfunktion** $R(z)$:*

<!--en:e2bd3ed2-->
**Beweis.** Bei RK4 erhalten wir die vier Messwerte $k_1=\lambda y$, $k_2=\lambda y(1+\tfrac z2)$, $k_3=\lambda y(1+\tfrac z2+\tfrac{z^2}4)$ und $k_4=\lambda y(1+z+\tfrac{z^2}2+\tfrac{z^3}4)$, indem wir jeweils den vorigen Messwert einsetzen. Die Aktualisierung ergibt $y+\tfrac z6\left(6+3z+z^2+\tfrac{z^3}4\right)y$, also das angegebene Polynom mal $y$. Das Euler-Verfahren liefert $y+zy$. Das Mittelpunktverfahren liefert $y+h\lambda(y+\tfrac z2y)$ und das Heun-Verfahren $y+\tfrac z2(y+(1+z)y)$. Beide Ausdrücke ergeben $(1+z+\tfrac{z^2}2)y$. $\blacksquare$

<!--en:606372e4-->
Jede Funktion $R$ ist das Taylor-Polynom des exakten Faktors $e^z$ bis zur Ordnung des Verfahrens. Für RK4 gilt $e^z-R_{\textrm{RK4}}(z)=z^5/120+O(z^6)$. Bei dieser Gleichung hat also ein Schritt einen Fehler von $O(h^5)$, und nach $T/h$ Schritten beträgt der Fehler $O(h^4)$. Für nichtlineare Felder, die viermal stetig differenzierbar und in $\mathbf x$ Lipschitz-stetig sind, liefert ein längerer Vergleich der Taylor-Entwicklungen dieselben Ordnungen: Das Mittelpunkt- und das Heun-Verfahren haben die Ordnung $2$, RK4 hat die Ordnung $4$ :cite:`Hairer.Norsett.Wanner.1993`. Auf den Beweis verzichten wir. Die Entwicklung von vier ineinander verschachtelten Auswertungen füllt Seiten und bringt wenig Einsicht. Merken wir uns aus diesem Abschnitt die Stabilitätsfunktion. Wie gut sie $e^z$ in der Nähe von $z=0$ annähert, bestimmt die Genauigkeit des Verfahrens. Wie groß ihr Betrag fern der Null wird, entscheidet über die Stabilität der Schritte.

<!--en:f6b8e29a-->
### Ordnung bei gleichem Rechenaufwand

<!--en:4b9f8926-->
Messen wir nun die Ordnungen und vergleichen wir die Verfahren bei gleicher Anzahl von Feldauswertungen. Für ein zeitunabhängiges Feld `f` genügen an beiden Enden der Verfahrensfamilie jeweils wenige Zeilen Code.

<!--en:565e2302-->
In doppelt logarithmischer Darstellung ist $\textrm{error} \approx C h^p$ eine Gerade mit Steigung $p$. Die gemessenen Steigungen sollten also $1$ und $4$ betragen. Wir prüfen das an der Spirale $\dot{\mathbf x}=\mathbf A\mathbf x$ aus :numref:`fig_mdl-dyn-ode-field` mit den Eigenwerten $-0.5\pm\mathrm i$. Der Abstand eines Punktes vom Ursprung nimmt mit Rate $\tfrac12$ ab, während er sich mit Winkelgeschwindigkeit eins dreht. Mit dem Anfangswert $\mathbf x_0=(2,0)$ ist der exakte Zustand bei $T=2$ daher $e^{-1}(2\cos 2,\,2\sin 2)=(-0.306,\,0.669)$ mit der Länge $0.736$ (:numref:`sec_mdl-odes-solvers`).

<!--en:2a614077-->
Bei einem linearen Feld können wir neben den Steigungen auch die Konstanten vorhersagen. Auf der Spirale besteht ein Schritt beider Verfahren wie die exakte Bewegung aus einer Drehung und einer Skalierung. Weicht der Faktor eines Verfahrens bei jeder Eigenmode um einen bestimmten Bruchteil ab, so beträgt der Fehler am Endpunkt denselben Bruchteil der Länge $0.736$ des exakten Endvektors. Für Euler kennen wir diesen Bruchteil von der Zinseszinsrechnung: $|\lambda T|^2/(2n)$. Hier gilt $|\lambda T|^2=5$. Der Fehler sollte beim Euler-Verfahren also $2.5/n$ der Länge betragen, das heißt $1.84/n$.

<!--en:ada0ab01-->
Für RK4 erhalten wir den Bruchteil aus der Stabilitätsfunktion. Ein Schritt weicht relativ um $|z|^5/120$ ab, mit $|z|=|\lambda T|/n=\sqrt5/n$, und die Fehler aus $n$ Schritten addieren sich. Insgesamt ergibt das $5^{5/2}/(120\,n^4)$ der Länge, also $0.343/n^4$. Die Codezelle halbiert die Schrittweite sechsmal, von $8$ auf $512$ Schritte.

<!--en:182a4789-->
Die angepassten Geraden haben die Steigungen $1.04$ und $4.02$. Sie liegen etwas über $1$ und $4$, weil die Anpassung auch grobe Schritte bis $h=0.25$ einbezieht. Dort fallen die nächsten Terme der Entwicklungen noch ins Gewicht. Bei $512$ Schritten betragen die Fehler $3.6\times10^{-3}$ und $5.0\times10^{-12}$, wie es $1.84/n$ und $0.343/n^4$ vorhersagen. Allerdings sagen die Steigungen nichts über den Aufwand aus. RK4 wertet das Feld viermal pro Schritt aus. Für einen fairen Vergleich geben wir ihm daher ein Viertel der Schritte. Bei denselben $512$ Feldauswertungen, also $128$ Schritten, erreicht RK4 einen Fehler von $1.3\times10^{-9}$, sechs Größenordnungen unter dem Euler-Verfahren. Für denselben Fehler bräuchte Euler nach dem Fehlergesetz $1.84/n$ mehr als eine Milliarde Schritte. Allgemein benötigt ein Verfahren der Ordnung $p$ eine Schrittzahl proportional zu $\varepsilon^{-1/p}$, um einen Fehler von $\varepsilon$ zu erreichen.

<!--en:55edef41-->
In der Praxis passen numerische Verfahren auch die Schrittweite während der Rechnung an. Ein *eingebettetes Verfahrenspaar* (engl. embedded pair) wie das Dormand-Prince-Verfahren berechnet aus denselben Auswertungen Lösungen der Ordnungen $4$ und $5$. Ihre Differenz dient als Schätzwert für den lokalen Fehler :cite:`Dormand.Prince.1980`. Das Verfahren verkürzt den Schritt, wenn der geschätzte Fehler groß ist, und verlängert ihn, wenn er klein ist. In Aufgabe :exref:`ex_mdl-odes-solvers-adaptive-steps` entwickeln Sie die einfachste Form einer solchen Schrittweitensteuerung.

<!--en:6207ca76-->
Alle bisherigen Verfahren beginnen jeden Schritt von Neuem und verwerfen die im vorigen Schritt gemessenen Steigungen. Ein *Mehrschrittverfahren* (engl. multistep method) behält sie bei. Das einfachste, das zweischrittige **Adams-Bashforth-Verfahren**, wertet das Feld einmal pro Schritt aus und extrapoliert aus den letzten beiden Werten:

<!--en:8f796d9e-->
Der Klammerausdruck lautet $\mathbf f(\mathbf x_n)+\tfrac12[\mathbf f(\mathbf x_n)-\mathbf f(\mathbf x_{n-1})]$. Er verlängert die Gerade durch die beiden Messwerte um einen halben Schritt und schätzt so die Steigung in der Mitte des kommenden Schritts. Das entspricht dem Mittelpunktverfahren, nur stammt die Steigung in der Mitte aus gespeicherten Werten statt aus einer zweiten Auswertung. So erhalten wir Ordnung $2$ zum Aufwand des Euler-Verfahrens. Dafür müssen wir zwei Nachteile in Kauf nehmen. Für den ersten Schritt fehlt ein Vorgänger, sodass wir ein Einschrittverfahren brauchen. Die Gewichte $\tfrac32$ und $-\tfrac12$ gelten nur für gleich große Schritte. Ändert sich die Schrittweite, müssen wir sie neu berechnen. Wenn jede Auswertung einen Durchlauf durch ein großes Netz erfordert, überwiegt die Einsparung beide Nachteile. Die Mehrschritt-Sampler aus :numref:`sec_diffusion-samplers` bauen auf diesem Schritt auf.

<!--en:eaa695e8-->
## Steifheit und implizite Verfahren

<!--en:eb60392d-->
Die Ordnung beschreibt, wie genau ein kleiner Schritt ist. Bei der Stabilität geht es um eine andere Frage: Klingen die Fehler über viele Schritte ab oder schaukeln sie sich auf? Für Euler und RK4 begrenzt die Antwort die Schrittweite, selbst wenn uns eine geringe Genauigkeit genügt.

<!--en:ad8c863b-->
Das kleinste System, an dem das Problem sichtbar wird, hat zwei entkoppelte Eigenmoden. Wir verfolgen $\dot{\mathbf x}=\mathrm{diag}(-50,-1)\,\mathbf x$ vom Anfangswert $(1,1)$ bis $T=1$. Bis dahin ist die schnelle Eigenmode auf $e^{-50}\approx10^{-22}$ abgeklungen, die langsame steht bei $e^{-1}=0.368$. Nach $t\approx0.1$ bleibt praktisch nur die langsame Eigenmode übrig, und Schritte der Weite $h=0.1$ würden sie gut erfassen. Doch für die schnelle Koordinate gilt die Gleichung $\dot x=-50x$ vom Anfang des Abschnitts. Euler multipliziert sie bei jedem Schritt mit $1-50h$, solange wir integrieren. Ein System heißt **steif** (engl. stiff), wenn einige Eigenwerte der Jacobi-Matrix negative Realteile haben, deren Beträge groß sind im Vergleich zur Änderungsrate der Lösung, die uns interessiert. Eine längst abgeklungene Eigenmode bestimmt dann noch immer die Schrittweite.

<!--en:10b3cea0-->
Welche Schritte sind zulässig? Die Antwort gibt die Stabilitätsfunktion. Betrachten wir die Testgleichung $\dot y=\lambda y$. Ihre Lösung klingt genau dann ab, wenn $\operatorname{Re}\lambda<0$ gilt. Das Verfahren überführt den Anfangswert $y_0$ in den Wert $R(z)^ny_0$ nach $n$ Schritten. Die berechnete Eigenmode klingt daher genau dann ab, wenn $|R(z)|<1$ gilt. Für das explizite Euler-Verfahren lautet die Bedingung $|1+z|<1$. Für reelles $\lambda<0$ ergibt sich $-2<h\lambda<0$ oder

<!--en:05b129df-->
Für die Parabel ist $2/|\lambda|=2/50=0.04$ genau die Lernrate, oberhalb derer die Iterierten im Eingangsbeispiel betragsmäßig anwuchsen.

<!--en:e55b4c44-->
Unterhalb dieser Grenze erhalten wir auch das, was der Fehlerschranke des Euler-Verfahrens fehlte: eine Analyse für ein kontrahierendes Feld. Ein Schritt multipliziert den bisherigen Fehler jetzt mit $|1+h\lambda|<1$. Im allgemeinen Beweis konnten wir diesen Faktor nur durch $1+hL$ abschätzen. Für $h\le1/|\lambda|$ bleibt die geometrische Summe aus dem Beweis unter $1/(h|\lambda|)$. Die Schranke vereinfacht sich zu $Mh/(2|\lambda|)$, der alten Schranke ohne Exponentialfaktor. Für die Gleichung der Parabel mit Anfangswert $x_0=1$ gilt $M=2500$. Daraus ergibt sich $25\,h$, höchstens das Dreifache des tatsächlichen Fehlers.

<!--en:19b1ec62-->
### Implizites Euler-Verfahren

<!--en:31d5282a-->
Kleinere Schritte bieten einen Ausweg. Der andere wirkt zunächst ungewohnt: Wir machen den Euler-Schritt mit der Steigung am *Ende* des Schritts,

<!--en:06eda6da-->
Das ist das **implizite Euler-Verfahren** (engl. backward Euler). Die Unbekannte $\mathbf x_{n+1}$ steht auf beiden Seiten. Mit dieser Schwierigkeit befassen wir uns, nachdem wir gesehen haben, was das Verfahren leistet.

<!--en:1b9fb10a-->
**Proposition (Stabilität des impliziten Euler-Verfahrens bei jeder Schrittweite).** *Für die Testgleichung mit $\operatorname{Re}\lambda<0$ hat das implizite Euler-Verfahren die Stabilitätsfunktion $R(z)=1/(1-z)$, und es gilt $|R(z)|<1$ für jede Schrittweite $h>0$.*

<!--en:a485a3d3-->
**Beweis.** Ein Schritt lautet $y_{n+1}=y_n+h\lambda y_{n+1}$. Daraus folgt $y_{n+1}=y_n/(1-z)$, und es gilt $|1-z|\ge\operatorname{Re}(1-z)=1-\operatorname{Re}z>1$ für $\operatorname{Re}z<0$. $\blacksquare$

<!--en:37d55f17-->
Der Grund lässt sich auch ohne Rechnung verstehen. Das explizite Euler-Verfahren extrapoliert mit der Steigung am aktuellen Punkt. Bei einer steilen Eigenmode führt diese Steigung weit über die Null hinaus, wenn wir sie über einen langen Schritt beibehalten. Das implizite Euler-Verfahren sucht den Punkt, von dem uns ein Schritt mit der dortigen Steigung rückwärts zum Ausgangspunkt führt. Ist die Lösung an einem Punkt fast abgeklungen, ist auch ihre Steigung fast null. Daher schießt selbst ein langer Schritt nicht über das Ziel hinaus.

<!--en:6a77b7b3-->
Nun zur Schwierigkeit: Wir werten das Feld an einem noch unbekannten Punkt aus. Deshalb müssen wir in jedem Schritt eine Gleichung lösen. Bei einem linearen Feld lösen wir ein lineares Gleichungssystem, sonst brauchen wir einige Newton-Iterationen (Aufgabe :exref:`ex_mdl-odes-solvers-stiff-newton`). Solche Verfahren heißen *implizit*. Das einfache Euler-Verfahren und RK4 berechnen den neuen Zustand unmittelbar mit einer Formel und heißen *explizit*. Der Aufwand für das Lösen lohnt sich, wenn bei einem expliziten Verfahren die Stabilität und nicht die Genauigkeit die Schrittweite bestimmen würde.

<!--en:dfc130dd-->
Wenden wir beide Verfahren mit fünf verschiedenen Schrittweiten auf das System mit den zwei Eigenmoden an. Beim impliziten Euler-Verfahren lösen wir pro Schritt ein lineares Gleichungssystem. Die erste Zeile lässt sich vorhersagen: Bei $h=0.1$ multipliziert das explizite Euler-Verfahren die schnelle Koordinate zehnmal mit $1-5=-4$.

<!--en:1efc0272-->
Bei $h=0.1$ weichen beide Verfahren für die langsame Eigenmode nur um etwa $5\%$ vom exakten Wert ab ($0.349$ und $0.386$ gegenüber $0.368$). Für viele Zwecke ist das genau genug. Trotzdem liefert das explizite Euler-Verfahren für die schnelle Komponente $4^{10}\approx10^6$ bei einem exakten Wert von $10^{-22}$. Bei $h=0.05$ kommt der Faktor $-1.5$ aus dem Eingangsbeispiel zwanzigmal zur Anwendung. Bei $h=0.04$ beträgt der Faktor $-1$: Die schnelle Komponente wechselt bei jedem Schritt das Vorzeichen, ohne jemals im Betrag abzunehmen. (Die Null in der Zeile für $h=0.02$ ist exakt. Dort gilt $1-50h=0$, sodass schon ein Schritt die schnelle Eigenmode beseitigt.) Das explizite Euler-Verfahren braucht $h<0.04$. Bei dieser Schrittweite beträgt der Fehler der langsamen Eigenmode nur noch etwa $2\%$. Uns hätten $5\%$ genügt, doch die Stabilität zwingt uns zu einer Genauigkeit von $2\%$.

<!--en:ae40c71a-->
Das implizite Euler-Verfahren ist bei jeder Schrittweite stabil. Sein Fehler in der langsamen Eigenmode ist der für ein Verfahren erster Ordnung typische Fehler $O(h)$. Stabilität allein garantiert also keine Genauigkeit.

<!--en:fded8e25-->
### Stabilitätsgebiete

<!--en:c5efb513-->
Für die Schrittweite des expliziten Euler-Verfahrens gibt es eine Obergrenze, für die des impliziten nicht. Wie steht es mit anderen Verfahren? Eine Darstellung in der $z$-Ebene zeigt sie alle auf einen Blick. Das **Stabilitätsgebiet** (engl. stability region) eines Verfahrens ist die Menge $\{z:|R(z)|<1\}$, in der seine Iterierten abklingen. Ein Verfahren heißt **A-stabil**, wenn sein Gebiet die gesamte offene linke Halbebene $\operatorname{Re}z<0$ enthält: Es lässt jede abklingende Eigenmode bei beliebiger Schrittweite abklingen :cite:`Dahlquist.1963`. Das implizite Euler-Verfahren ist A-stabil, denn sein Gebiet umfasst alle Punkte außerhalb der abgeschlossenen Einheitskreisscheibe um $+1$.

<!--en:a0efdaf4-->
In :numref:`fig_mdl-dyn-stability-regions` sind unsere Beispiele markiert. Der Schritt $h=0.05$ für $\dot x=-50x$ aus dem Eingangsbeispiel entspricht dem Punkt $z=-2.5$ außerhalb der Euler-Kreisscheibe, während $h=0.01$ dem Punkt $z=-0.5$ in ihrem Inneren entspricht. Für die Spirale mit $\lambda=-0.5\pm\mathrm i$ gilt $|1+z|^2=(1-h/2)^2+h^2$. Dieser Ausdruck ist genau dann kleiner als $1$, wenn $h<0.8$ gilt. Bei $h=0.8$ liegen die beiden Punkte $-0.4\pm0.8\,\mathrm i$ auf der Kreislinie. Bei einer reinen Drehung liegt $z$ auf der imaginären Achse. Diese berührt die Euler-Kreisscheibe nur im Ursprung, sodass beim expliziten Euler-Verfahren die Amplitude einer ungedämpften Schwingung bei jeder positiven Schrittweite wächst (Aufgaben :exref:`ex_mdl-odes-solvers-rotating-modes` und :exref:`ex_mdl-odes-solvers-symplectic-euler`). Der exakte Schritt $e^{z}$ aus :numref:`subsec_mdl-ode-driven-linear` hat auf der gesamten linken Halbebene einen Betrag kleiner als eins. Dieses Verhalten bildet das implizite Euler-Verfahren nach, das explizite hingegen nicht.

<!--en:88e11a72-->
![Stabilitätsgebiete in der komplexen Ebene von $z = h\lambda$ für die Testgleichung $\dot{y} = \lambda y$. Eine abklingende Eigenmode liegt in der linken Halbebene. Beim expliziten Euler-Verfahren klingt sie nur innerhalb der Kreisscheibe $\lvert 1 + z \rvert < 1$ ab, auf der reellen Achse also für $-2 < h\lambda < 0$. RK4 erweitert dieses Gebiet zu einer ausgebuchteten Fläche, die auf der reellen Achse bis $\approx -2.79$ reicht. Das implizite Euler-Verfahren ist überall außerhalb der Einheitskreisscheibe um $+1$ stabil, insbesondere auf der gesamten linken Halbebene (A-Stabilität). Markiert sind die Schritte $h=0.05$ und $h=0.01$ aus dem Eingangsbeispiel sowie die Punkte für die Spirale bei $h=0.8$ auf dem Rand der Euler-Kreisscheibe.](../img/mdl-dyn-stability-regions.svg)

<!--en:16211bed-->
Kein explizites Runge-Kutta-Verfahren ist A-stabil, unabhängig von der Zahl seiner Stufen. Seine Stabilitätsfunktion $R$ ist ein Polynom (Aufgabe :exref:`ex_mdl-odes-solvers-rotating-modes`). Jedes nichtkonstante Polynom wächst betragsmäßig unbegrenzt, wenn $z$ auf der negativen reellen Achse gegen minus unendlich geht. Eine höhere Ordnung behebt die Steifheit also nicht. Das Stabilitätsgebiet von RK4 endet auf der negativen reellen Achse näherungsweise bei $z=-2.79$, der negativen Lösung von $R_{\textrm{RK4}}(z)=1$. Für die Parabel ergibt sich die Grenze $h<2.79/50=0.056$ gegenüber $0.04$ bei Euler. Dabei wertet RK4 das Feld viermal pro Schritt aus. Daher kommen implizite Verfahren vor allem bei steifen Feldern zum Einsatz, während sonst explizite Verfahren die übliche Wahl sind. Das steife Feld, dem wir in diesem Buch am häufigsten begegnen, ist der Gradient einer Verlustfunktion, deren Krümmung in einigen Richtungen groß und in anderen klein ist.

<!--en:cfd6ef4d-->
## Gradientenabstieg als numerisches Verfahren

<!--en:b47892d7-->
Im Eingangsbeispiel gab es nur eine Koordinate. Bei mehreren Koordinaten tritt derselbe Faktor für jeden Eigenwert der Hesse-Matrix auf. Bewegen wir uns auf einer Verlustfunktion $\mathcal L$ kontinuierlich bergab, immer in Richtung des negativen Gradienten, so erhalten wir den **Gradientenfluss** (engl. gradient flow)

<!--en:9553715e-->
mit einem Feld, das an den kritischen Punkten von $\mathcal L$ verschwindet. Entlang seiner Trajektorien kann der Verlust nur abnehmen: $\tfrac{d}{dt}\mathcal L(\mathbf{x}(t)) = \nabla \mathcal L^\top \dot{\mathbf{x}} = -\|\nabla \mathcal L\|^2 \le 0$. Der Gradientenabstieg mit Lernrate $\eta$ und Aktualisierung $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\, \nabla \mathcal L(\mathbf{x}_k)$ ist das explizite Euler-Verfahren :eqref:`eq_mdl-ode-euler-update` für diesen Fluss mit $h = \eta$.

<!--en:851035c2-->
In :numref:`sec_mdl-gradient-based-optimization` haben wir die Obergrenze $\eta<2/\lambda_{\max}(\mathbf H)$ für die Lernrate bei einer quadratischen Funktion mit Hesse-Matrix $\mathbf H$ hergeleitet und die Konvergenzgeschwindigkeit unterhalb dieser Grenze untersucht. Die Untersuchung der Geschwindigkeit wiederholen wir hier nicht. Die Grenze leiten wir noch einmal in drei Schritten her, denn gerade der Weg dorthin zeigt, was die Sichtweise als numerisches Verfahren beiträgt. Schreiben wir $\boldsymbol\delta=\mathbf x-\mathbf x^\star$ für die Abweichung von einem Minimum $\mathbf x^\star$ mit Hesse-Matrix $\mathbf H$. In der Nähe des Minimums lässt sich der Fluss linearisieren. Jeder Eigenwert $\lambda_i$ von $\mathbf H$ ergibt eine abklingende Eigenmode mit Rate $-\lambda_i$, also eine Kopie der Testgleichung:

<!--en:f96333d6-->
Die Obergrenze für die Lernrate entspricht also dem Rand der Euler-Kreisscheibe. Die steilste Eigenmode erreicht ihn bei $z=-\eta\lambda_{\max}=-2$. Beim ersten Pfeil halten wir die Hesse-Matrix fest. Die Grenze gilt daher für eine quadratische Funktion. Bei einem Netz ändert sich die Hesse-Matrix während des Trainings. Ein Lauf oberhalb der Grenze divergiert häufig nicht: Der größte Eigenwert nähert sich wieder $2/\eta$, und der Verlust sinkt auf lange Sicht weiter, wenn auch nicht bei jedem Schritt (:numref:`subsec_mdl-edge-of-stability`).

<!--en:71f04f56-->
Wenn diese Deutung stimmt, ergibt sich die Grenze aus dem Euler-Verfahren, und ein anderes Verfahren muss eine andere Grenze haben. Prüfen wir diese Vorhersage an dem Tal, für das wir in :numref:`sec_mdl-gradient-based-optimization` die Lernrate variiert haben: $\mathcal L(\mathbf x)=\tfrac12\mathbf x^\top\mathbf H\mathbf x$ mit $\mathbf H=\mathrm{diag}(10,1)$ (dort steht die flache Koordinate an erster Stelle). Sein Gradientenfluss $\dot{\mathbf x}=-\mathbf H\mathbf x$ hat eine steile Eigenmode mit Rate $-10$ und eine flache mit Rate $-1$. Von $(1,1)$ aus machen wir mit drei Verfahren jeweils $100$ Schritte der Weite $\eta$. Das explizite Euler-Verfahren entspricht dem Gradientenabstieg mit der Grenze $2/10=0.2$. RK4 wertet den Gradienten viermal pro Schritt aus. Sein Stabilitätsgebiet reicht auf der reellen Achse bis $-2.79$ (:numref:`fig_mdl-dyn-stability-regions`). Seine Grenze sollte also bei $0.279$ liegen. Das implizite Euler-Verfahren $\mathbf x_{k+1}=(\mathbf I+\eta\mathbf H)^{-1}\mathbf x_k$ sollte keine solche Grenze haben. Die Codezelle gibt für jeden Lauf die Faktoren aus, mit denen ein Schritt die beiden Eigenmoden multipliziert, also $R(-10\eta)$ und $R(-\eta)$, sowie die Länge der Iterierten nach den $100$ Schritten. Die Faktoren messen wir, indem wir einen Schritt auf die Einheitsmatrix anwenden.

<!--en:fc2b09ff-->
Die Grenze jedes Verfahrens liegt dort, wo sein Stabilitätsgebiet endet. Das explizite Euler-Verfahren konvergiert bei $\eta=0.19$ mit dem Faktor $-0.9$ für die steile Eigenmode und divergiert bei $\eta=0.21$ mit dem Faktor $-1.1$. RK4 konvergiert noch bei $\eta=0.27$ und divergiert bei $0.29$. Das implizite Euler-Verfahren konvergiert bei $\eta=0.25$, oberhalb der Grenze des Gradientenabstiegs, und bei $\eta=25$, weit oberhalb beider Grenzen. Dort betragen seine Faktoren $0.004$ und $0.038$. Nach hundert Schritten ist die Iterierte nicht mehr messbar von null verschieden. Am Anfang fragten wir nach der Ursache der Divergenz, wenn sie nicht allein in der Verlustfunktion liegt. Die Grenze ist ein Quotient. Die Verlustfunktion liefert den Nenner $\lambda_{\max}$, das Verfahren den Zähler: $2$ beim expliziten Euler-Verfahren und $2.79$ bei RK4. Das implizite Euler-Verfahren hat überhaupt keine Obergrenze.

<!--en:ff23c38a-->
Sollten wir also mit RK4 trainieren? Nein. Seine Obergrenze ist $1.4$-mal so hoch, und ein Schritt kostet vier Gradienten. Pro Gradient kommen wir in der Zeit des Flusses also höchstens um $0.07$ voran, beim Gradientenabstieg dagegen um $0.2$. Die zusätzlichen Gradienten dienen dazu, der Bahn des Gradientenflusses genauer zu folgen. Für einen Optimierer zählt aber nur, wo die Bahn endet. Die Abweichung von der Bahn ist auch nicht bloß ein Fehler. Bis zur zweiten Ordnung in $\eta$ folgen die Iterierten des Gradientenabstiegs dem Fluss einer modifizierten Verlustfunktion, die große Gradienten bestraft. Der Fehler des Euler-Verfahrens wirkt also regularisierend (Aufgabe :exref:`ex_mdl-odes-solvers-implicit-gradient-regularization`).

<!--en:07317821-->
Die beiden Verfahren divergieren auch auf unterschiedliche Weise. Beim Euler-Verfahren verlässt der Faktor der steilen Eigenmode den stabilen Bereich über $-1$. Die Iterierte springt über die Talsohle und landet jedes Mal höher. Bei RK4 ist der Faktor auf der gesamten reellen Achse positiv und verlässt den stabilen Bereich über $+1$. Bei $\eta=0.29$ beträgt er $1.187$, und die Iterierte entfernt sich vom Minimum, ohne je das Vorzeichen zu wechseln. Auch der Zickzackverlauf ist also eine Eigenschaft des Euler-Verfahrens. Momentum, das übliche Mittel dagegen, löst eine andere Gleichung: Das Heavy-Ball-Verfahren von Polyak diskretisiert $m\ddot{\mathbf x}+c\dot{\mathbf x}=-\nabla\mathcal L(\mathbf x)$, die Bewegung einer Kugel mit Masse und Reibung :cite:`Polyak.1964` (Aufgabe :exref:`ex_mdl-odes-solvers-heavy-ball-ode`).

<!--en:59ac732b-->
Beim impliziten Euler-Verfahren werden die Faktoren $1/(1+\eta\lambda_i)$ mit wachsender Schrittweite kleiner, die Konvergenz wird also immer schneller. Der Preis dafür liegt im Lösen der Gleichung. Die Schrittgleichung $\mathbf x_{k+1}=\mathbf x_k-\eta\nabla\mathcal L(\mathbf x_{k+1})$ besagt, dass der Gradient von $\mathcal L(\mathbf x)+\tfrac1{2\eta}\|\mathbf x-\mathbf x_k\|^2$ an der Stelle $\mathbf x_{k+1}$ verschwindet. Für eine konvexe Verlustfunktion ist das der proximale Schritt $\mathrm{prox}_{\eta\mathcal L}(\mathbf x_k)$ aus :numref:`subsec_mdl-proximal-operators`, angewandt auf die Verlustfunktion selbst. Bei unserer quadratischen Funktion müssen wir dafür ein lineares Gleichungssystem lösen. Bei einem Netz ergibt sich in jedem Schritt ein eigenes Optimierungsproblem. Für große $\eta$ ist das gerade das Problem, das wir ursprünglich lösen wollten.

<!--en:326727c5-->
Das Tal ist eine weniger steife Variante des Systems aus :numref:`sec_mdl-stiffness-implicit`, mit den Raten $10$ und $1$ anstelle von $50$ und $1$ sowie $\eta$ anstelle von $h$. Für jedes $\eta<0.2$ bleibt der Faktor $1-\eta$ des Gradientenabstiegs in der flachen Richtung über $0.8$ (die Tabelle zeigt $0.81$ bei $\eta=0.19$). Diese Komponente verliert also pro Schritt weniger als ein Fünftel ihrer Länge, unabhängig davon, wie schnell die steile Komponente abnimmt. Allgemein verliert sie weniger als den Anteil $2\lambda_{\min}/\lambda_{\max}$. Bei einer Konditionszahl von $1000$ ist das ein Fünfhundertstel. Die Stabilität begrenzt $\eta$ über $\lambda_{\max}$. Dann bestimmt $\lambda_{\min}$ die Geschwindigkeit. Damit beschreibt die Konditionszahl der Hesse-Matrix die Steifheit des Gradientenflusses. Führt eine Lernrate bei einer quadratischen Funktion zur Divergenz, liegt eine Instabilität des numerischen Verfahrens vor.

<!--en:6a75e3de-->
In diesem Abschnitt kannten wir für jedes Feld die Eigenwerte schon vor dem ersten Schritt, weil wir das Feld selbst vorgegeben hatten. Stellt ein Netz das Feld dar, gibt es keine solche Liste. In :numref:`sec_mdl-neural-odes-cnf` dient ein gelerntes Feld selbst als Modell, dessen Schichten Euler-Schritten entsprechen. Wir untersuchen dort, wie sich ein solches Modell trainieren lässt und wie es eine Wahrscheinlichkeitsdichte verändert.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3d6884e8-->
Ein numerisches Verfahren nähert eine Trajektorie durch einzelne Schritte an. Zwei Eigenschaften kennzeichnen seine Güte. Die Ordnung $p$ gibt an, wie schnell der Fehler bei kleinerer Schrittweite sinkt: Lokale Fehler $O(h^{p+1})$ summieren sich zu einem globalen Fehler $O(h^p)$, mit $p=1$ für Euler und $p=4$ für RK4. Das Stabilitätsgebiet gibt an, bei welchen Schritten die Lösung nicht divergiert. Das explizite Euler-Verfahren braucht $|1+h\lambda|<1$. Das implizite Euler-Verfahren ist auf der gesamten linken Halbebene stabil und löst in jedem Schritt eine Gleichung. Ein System heißt steif, wenn eine längst abgeklungene schnelle Eigenmode noch immer die Schrittweite bestimmt. Bei einem linearen Feld lassen sich beide Eigenschaften an der Stabilitätsfunktion $R(h\lambda)$ ablesen. Sie gibt den Faktor an, mit dem ein Schritt eine Eigenmode multipliziert. Der Gradientenabstieg ist das explizite Euler-Verfahren für $\dot{\mathbf x}=-\nabla\mathcal L$. Seine Obergrenze $\eta<2/\lambda_{\max}(\mathbf H)$ entspricht daher dem Rand des Euler-Stabilitätsgebiets, und schlechte Konditionierung bedeutet Steifheit.

<!--en:2294243a-->
## Übungen

<!--en:34455f45-->
1. [code] **Adaptive Schrittweite.** ● Ein eingebettetes Verfahrenspaar schätzt den lokalen Fehler mithilfe zweier Verfahren unterschiedlicher Ordnung. Das einfachste Paar verwendet die Stufen $\mathbf{k}_1 = \mathbf{f}(\mathbf{x})$ und $\mathbf{k}_2 = \mathbf{f}(\mathbf{x} + h\mathbf{k}_1)$ des Heun-Verfahrens. Der Euler-Wert $\mathbf{x} + h\mathbf{k}_1$ und der Heun-Wert $\mathbf{x} + \tfrac{h}{2}(\mathbf{k}_1 + \mathbf{k}_2)$ haben den Abstand $\Delta = \tfrac{h}{2}\|\mathbf{k}_2 - \mathbf{k}_1\|$. Gilt $\Delta \le \texttt{tol}$, so akzeptiert die Schrittweitensteuerung den Schritt und übernimmt den Heun-Wert. In beiden Fällen rechnet sie mit der folgenden Schrittweite weiter:

   $$
   h \leftarrow h \min\!\Big(2,\ \max\!\big(0.2,\ 0.9\sqrt{\texttt{tol}/\Delta}\,\big)\Big).
   $$

    1. Welche Ordnung des lokalen Fehlers setzt der Exponent $1/2$ voraus?
    1. Implementieren Sie die Schrittweitensteuerung mit der anfänglichen Schrittweite $h = 0.1$. Verkürzen Sie den letzten Schritt so, dass er bei $T$ endet. Integrieren Sie das Spiralfeld `f` von `x0` bis $T = 2$ für $\texttt{tol} = 10^{-2}, 10^{-3}, \ldots, 10^{-6}$. Zählen Sie alle Feldauswertungen einschließlich derjenigen für verworfene Schritte. Bestimmen Sie den Fehler gegenüber dem exakten Endpunkt `x_true`. Wie hängt die Anzahl der Auswertungen von $\texttt{tol}$ ab?
    1. Integrieren Sie das Feld von `A_stiff` ausgehend von $(1, 1)$ bis $T = 10$ mit $\texttt{tol} = 10^{-3}$. Tragen Sie die akzeptierten Schrittweiten gegen die Zeit auf. Bei welchem Wert pendeln sie sich ein, nachdem die schnelle Eigenmode abgeklungen ist? Erklären Sie diesen Wert anhand der Stabilitätsfunktion des Heun-Verfahrens $1 + z + z^2/2$ aus :eqref:`eq_mdl-ode-stability-function`.
   :exlabel:`ex_mdl-odes-solvers-adaptive-steps`

<!--en:2313d4b8-->
1. **Rotierende Eigenmoden.** ● Das steife System aus dem Beispiel hat reelle Eigenwerte. Ein komplexer Eigenwert $\lambda$ geht über die Koordinate $z = h\lambda$ in :numref:`fig_mdl-dyn-stability-regions` ein. Sein Imaginärteil kann darüber entscheiden, ob ein Schritt stabil ist.
    1. Für das Spiralfeld aus :numref:`fig_mdl-dyn-ode-field` mit den Eigenwerten $-0.5 \pm \mathrm i$ haben wir gesehen, dass die Iterierten des expliziten Euler-Verfahrens für $h < 0.8$ abklingen. Bestimmen Sie die entsprechende Grenze der Schrittweite für das Heun-Verfahren aus seiner Stabilitätsfunktion in :eqref:`eq_mdl-ode-stability-function` (die Bedingung führt auf eine kubische Gleichung in $h$, die Sie numerisch lösen können). Bestimmen Sie die Grenze auch für RK4. Das Heun-Verfahren wertet das Feld pro Schritt zweimal aus, RK4 viermal. Welches der drei Verfahren überbrückt pro Feldauswertung die längste Zeitspanne, ohne dass die Lösung anwächst?
    1. Betrachten Sie das Drehfeld $\dot{\mathbf{x}} = \left(\begin{smallmatrix}0&-b\\b&0\end{smallmatrix}\right)\mathbf{x}$. Zeigen Sie, dass das explizite Euler-Verfahren die Norm $\|\mathbf{x}\|$ in jedem Schritt mit $\sqrt{1 + h^2 b^2}$ multipliziert und das implizite Euler-Verfahren sie durch denselben Faktor teilt. Um welchen Faktor ändert jedes Verfahren die Norm $\|\mathbf{x}\|$ während einer Umdrehung, also in $2\pi/(hb)$ Schritten, wenn $hb = 0.1$ gilt?
    1. Die letzte Formel in :eqref:`eq_mdl-ode-stability-function` gibt die Stabilitätsfunktion von RK4 an. Ihr Betragsquadrat auf der imaginären Achse bestimmt, wie sich der Radius bei einer Drehung ändert. Zeigen Sie, dass $|R(\mathrm iy)|^2 = 1 - y^6/72 + y^8/576$ für reelles $y$ gilt. Bestimmen Sie den Bereich von $hb$, in dem RK4 die Drehbewegung nicht verstärkt.
    1. Ein explizites Runge-Kutta-Verfahren mit $s$ Stufen berechnet $\mathbf{k}_i = \mathbf{f}\big(\mathbf{x}_n + h\sum_{j<i} a_{ij}\mathbf{k}_j\big)$ für $i = 1, \ldots, s$ und anschließend $\mathbf{x}_{n+1} = \mathbf{x}_n + h\sum_{i} b_i\mathbf{k}_i$, wie es RK4 in :eqref:`eq_mdl-ode-rk4-update` tut. Zeigen Sie, dass seine Stabilitätsfunktion ein Polynom in $z$ mit Grad höchstens $s$ ist. Warum schließt das A-Stabilität für explizite Verfahren aus?
   :exlabel:`ex_mdl-odes-solvers-rotating-modes`

<!--en:26e4fb3e-->
1. [code] **Implizite Schritte für ein steifes nichtlineares Feld.** ● Das steife System aus dem Text ist linear und diagonal. Für jeden impliziten Euler-Schritt genügte es daher, ein lineares Gleichungssystem zu lösen. Betrachten Sie nun das gedämpfte Pendel $\dot\phi=\omega$, $\dot\omega=-\sin\phi-c\,\omega$ aus :numref:`subsec_mdl-ode-linearization` mit starker Dämpfung $c=50$, vom Anfangswert $(\phi,\omega)=(1,0)$ bis $T=2$.
    1. Berechnen Sie die Eigenwerte seiner Jacobi-Matrix $\mathbf J(\phi,\omega)=\left(\begin{smallmatrix}0&1\\-\cos\phi&-c\end{smallmatrix}\right)$ am tiefsten Punkt $(0,0)$. Welche Grenzen für die Schrittweite des expliziten Euler-Verfahrens und von RK4 folgen daraus?
    1. Zeigen Sie, dass die Bewegung nach dem schnellen Einschwingvorgang näherungsweise $\dot\phi=-\sin\phi/c$ erfüllt. Berechnen Sie $\phi(2)$ aus dieser eindimensionalen Gleichung.
    1. Ein impliziter Euler-Schritt löst $\mathbf g(\mathbf y)=\mathbf y-\mathbf x_n-h\mathbf f(\mathbf y)=\mathbf 0$ nach $\mathbf y=\mathbf x_{n+1}$ auf. Formulieren Sie dafür die Newton-Iteration $\mathbf y\leftarrow\mathbf y-(\mathbf I-h\mathbf J(\mathbf y))^{-1}\mathbf g(\mathbf y)$. Implementieren Sie das implizite Euler-Verfahren mit drei Newton-Iterationen pro Schritt, jeweils mit dem Startwert $\mathbf x_n$.
    1. Führen Sie das explizite Euler-Verfahren, RK4 und das implizite Euler-Verfahren mit $n=20$, $33$, $36$ und $100$ gleich großen Schritten aus ($h=0.1$, $0.061$, $0.056$ und $0.02$). Vergleichen Sie die Ergebnisse mit einer Referenzlösung aus `rk4` mit $2^{16}$ Schritten. Welche Läufe divergieren? Sagen die Grenzen aus Teil a diese Fälle voraus? Bestimmen Sie dann für jedes Verfahren die kleinste Schrittzahl $n$ mit einem Fehler am Endpunkt unter $10^{-4}$. Zählen Sie die dafür nötigen Auswertungen des Feldes und der Jacobi-Matrix. Bei welchem Verfahren bestimmt die Genauigkeit die Schrittzahl $n$, bei welchem die Stabilität?
   :exlabel:`ex_mdl-odes-solvers-stiff-newton`

<!--en:b151aa68-->
1. [code] **Symplektisches Euler-Verfahren und Operatorzerlegung.** ● Beim harmonischen Oszillator $\dot q=p$, $\dot p=-q$ bleibt $q^2+p^2$ erhalten. Beim expliziten Euler-Verfahren spiralt die Bahn aus der Kreisbahn heraus, beim impliziten nach innen. Das *symplektische Euler-Verfahren* aktualisiert zuerst $q$ mit dem alten Wert von $p$ und dann $p$ mit dem neuen Wert von $q$: $q_{n+1}=q_n+hp_n$, $p_{n+1}=p_n-hq_{n+1}$.
    1. Schreiben Sie die Aktualisierung als Matrix $\mathbf S_h$, die auf $(q,p)$ wirkt. Zeigen Sie, dass $\det\mathbf S_h=1$ für jedes $h$ gilt. Die Aktualisierung erhält also wie der exakte Fluss den Flächeninhalt, während die Matrix des expliziten Euler-Verfahrens die Determinante $1+h^2$ hat.
    1. Zeigen Sie, dass die Eigenwerte von $\mathbf S_h$ genau dann verschieden sind und auf dem Einheitskreis liegen, wenn $0<h<2$ gilt. Was passiert bei $h=2$ und für $h>2$?
    1. Zeigen Sie, dass die Aktualisierung $q^2+p^2+hqp$ exakt erhält. Wie sieht die zugehörige Niveaulinie durch $(1,0)$ aus? Wie weit kann sich die berechnete Bahn bei $h=0.5$ und bei $h=1$ vom Ursprung entfernen? (Hinweis: Die Achsenlängen der Niveaulinie dieser quadratischen Form stehen im Verhältnis $\sqrt{(2+h)/(2-h)}$.)
    1. Integrieren Sie vom Anfangswert $(1,0)$ bis $T=200$ mit dem expliziten, dem impliziten und dem symplektischen Euler-Verfahren für $h=0.1$ und $h=1$. Geben Sie für jeden Lauf den größten und den kleinsten Wert von $\|(q,p)\|$ an.
    1. Die Aktualisierung folgt zunächst dem exakten Fluss von $\dot q=p$ bei festem $p$ für die Zeit $h$ und anschließend dem exakten Fluss von $\dot p=-q$ bei festem $q$. Geben Sie die beiden Generatoren $\mathbf P$ und $\mathbf Q$ an, deren Summe die Matrix des Oszillators ist. Berechnen Sie $e^{h\mathbf P}$, $e^{h\mathbf Q}$ und $e^{h(\mathbf P+\mathbf Q)}$. Zeigen Sie, dass sich $e^{h\mathbf Q}e^{h\mathbf P}=\mathbf S_h$ von $e^{h(\mathbf P+\mathbf Q)}$ um Terme der Ordnung $h^2$ unterscheidet. Bei der geschlossenen Lösung für die Spirale in :numref:`sec_mdl-odes-solvers` konnten wir eine Matrixexponentialfunktion ohne Fehler zerlegen. Welche dort genutzte Eigenschaft fehlt hier? Die symmetrische Zerlegung $e^{h\mathbf Q/2}e^{h\mathbf P}e^{h\mathbf Q/2}$ ist der *Leapfrog-Integrator* des Hamiltonian-Monte-Carlo-Verfahrens :cite:`Neal.2011`. Welche Ordnung hat er?
   :exlabel:`ex_mdl-odes-solvers-symplectic-euler`

<!--en:2302f9cb-->
1. [code] **Implizite Gradientenregularisierung.** ● Der Gradientenabstieg ist das explizite Euler-Verfahren für den Gradientenfluss (:numref:`sec_mdl-gd-as-solver`). Jeder Schritt weicht daher vom Fluss der Verlustfunktion $\mathcal L$ um $O(\eta^2)$ ab. Die Rückwärtsfehleranalyse sucht ein modifiziertes Feld $\tilde{\mathbf{f}} = \mathbf{f} + \eta\mathbf{f}_1$, dessen exakter Fluss über die Zeit $\eta$ bis auf $O(\eta^3)$ mit einem Euler-Schritt übereinstimmt. :citet:`Barrett.Dherin.2021` zeigen, dass dieses Feld beim Gradientenabstieg wiederum ein Gradientenfeld ist, und zwar das der modifizierten Verlustfunktion

   $$
   \tilde{\mathcal L}(\mathbf{x}) = \mathcal L(\mathbf{x}) + \frac{\eta}{4}\|\nabla \mathcal L(\mathbf{x})\|^2 .
   $$

    1. Entwickeln Sie den exakten Fluss von $\tilde{\mathbf{f}}$ über die Zeit $\eta$ bis zur zweiten Ordnung und zeigen Sie, dass $\mathbf{f}_1 = -\tfrac12(\partial\mathbf{f}/\partial\mathbf{x})\,\mathbf{f}$ gilt. Leiten Sie $\tilde{\mathcal L}$ für $\mathbf{f} = -\nabla \mathcal L$ her. Welche Bereiche des Parameterraums bestraft der zusätzliche Term?
    1. Zeigen Sie für $\mathcal L(x) = \tfrac{\lambda}{2}x^2$ mit $0 < \eta\lambda < 1$, dass die Iterierten des Gradientenabstiegs exakt auf dem Gradientenfluss von $\tfrac{\tilde{\lambda}}{2}x^2$ mit $\tilde{\lambda} = -\log(1 - \eta\lambda)/\eta$ liegen. Entwickeln Sie $\tilde{\lambda}$ nach $\eta$ und vergleichen Sie das Ergebnis mit der Krümmung von $\tilde{\mathcal L}$.
    1. Der Gradientenabstieg konvergiert bei dieser quadratischen Funktion auch für $1 < \eta\lambda < 2$. Die Iterierten wechseln dabei das Vorzeichen. Trajektorien eines skalaren autonomen Feldes mit einer Lipschitz-Konstante können einander nicht kreuzen. Die Flussabbildung zur Zeit $\eta$ erhält daher die Reihenfolge der Punkte. Folgern Sie, dass kein solches Feld eine Trajektorie besitzt, die zu den Zeitpunkten $k\eta$ die Werte der Iterierten annimmt. Betrachten Sie eine quadratische Funktion mit Hesse-Matrix $\mathbf H$ in $d$ Dimensionen. Der Anfangspunkt habe eine von null verschiedene Komponente in Richtung eines Eigenvektors zum größten Eigenwert. Zeigen Sie für jeden solchen Anfangspunkt, dass die Iterierten auf keinem Fluss eines Gradientenfeldes der Form $-\nabla\big(\tfrac12\mathbf{x}^\top\tilde{\mathbf H}\mathbf{x}\big)$ liegen, sobald $\eta \ge 1/\lambda_{\max}(\mathbf H)$ gilt.
    1. Führen Sie für $\mathcal L(x) = x^4/4 + x^2/2$ und $x_0 = 1.5$ den Gradientenabstieg bis $T = 1$ mit $\eta = 0.1, 0.05, 0.025, 0.0125$ aus. Berechnen Sie die Gradientenflüsse von $\mathcal L$ und $\tilde{\mathcal L}$ bis $T = 1$ mit `rk4` und vielen Schritten, den zweiten Fluss für jedes $\eta$ gesondert. Schätzen Sie die Steigungen der beiden Endpunktfehler des Gradientenabstiegs in Abhängigkeit von $\eta$ in doppelt logarithmischer Darstellung.
   :exlabel:`ex_mdl-odes-solvers-implicit-gradient-regularization`

<!--en:de4ebfaa-->
1. **Heavy-Ball-Verfahren als diskretisierte Differentialgleichung.** ● Das Heavy-Ball-Momentum $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\nabla \mathcal L(\mathbf{x}_k) + \beta(\mathbf{x}_k - \mathbf{x}_{k-1})$ diskretisiert die Bewegungsgleichung $m\ddot{\mathbf{x}} + c\dot{\mathbf{x}} = -\nabla \mathcal L(\mathbf{x})$ einer Kugel mit Masse $m$ und Reibungskoeffizient $c$.
    1. Ersetzen Sie $\ddot{\mathbf{x}}(t_k)$ durch den zentralen Differenzenquotienten $(\mathbf{x}_{k+1} - 2\mathbf{x}_k + \mathbf{x}_{k-1})/h^2$ und $\dot{\mathbf{x}}(t_k)$ durch den rückwärts gerichteten Differenzenquotienten $(\mathbf{x}_k - \mathbf{x}_{k-1})/h$. Zeigen Sie, dass sich die Aktualisierung des Heavy-Ball-Verfahrens ergibt, und drücken Sie $\eta$ und $\beta$ durch $m$, $c$ und $h$ aus.
    1. Für $\mathcal L(x) = \tfrac{\lambda}{2}x^2$ beschreibt die Differentialgleichung einen gedämpften Oszillator. Bei $c = 2\sqrt{m\lambda}$ ist die Dämpfung kritisch. Drücken Sie diese Bedingung durch $\eta$, $\beta$ und $\lambda$ aus. Die Heavy-Ball-Rekursion ist kritisch gedämpft, wenn ihr charakteristisches Polynom $z^2 - (1 + \beta - \eta\lambda)z + \beta$ eine doppelte Nullstelle hat. Bestimmen Sie das zugehörige $\beta$ und vergleichen Sie die beiden Bedingungen für kleine Werte von $\eta\lambda$.
    1. Die Heavy-Ball-Rekursion für diese quadratische Funktion konvergiert genau dann, wenn $0 < \eta\lambda < 2(1 + \beta)$ für $0 \le \beta < 1$ gilt (Aufgabe :exref:`ex_mdl-gradient-based-optimization-edge-of-stability-momentum`a in :numref:`sec_mdl-gradient-based-optimization` verlangt den Beweis). Formen Sie die Bedingung in eine Bedingung an $h$, $m$, $c$ und $\lambda$ um. Für jedes $\lambda > 0$ klingt die Lösung der Differentialgleichung ab. Für ihre Diskretisierung gilt das nicht immer. Wo liegt die Grenze der stabilen Schrittweite $h$ für $c \to 0$, ausgedrückt durch die Kreisfrequenz $\sqrt{\lambda/m}$ des Oszillators? Welcher Eigenwert der Hesse-Matrix einer schlecht konditionierten quadratischen Funktion bestimmt die Grenze der Schrittweite?
   :exlabel:`ex_mdl-odes-solvers-heavy-ball-ode`

<!--en:c211ac20-->
[Diskussion](https://d2l.discourse.group/t/36897)

<!--en:bbb83c9e-->
Numerische Verfahren für gewöhnliche Differentialgleichungen: Genauigkeit, Steifheit und Optimierung<br>**Euler- und Runge-Kutta-Verfahren · lokaler und globaler Fehler · Stabilitätsfunktion R(z) · Steifheit und A-Stabilität · Gradientenabstieg**

<!--en:ba7c2bbb-->
::: {.slide title="Warum der Gradientenabstieg auf einer einfachen Parabel divergiert"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:7ea217b6-->
Divergiert ein Trainingslauf bei großer Lernrate $\eta$, liegt es nahe, eine problematische Verlustlandschaft dafür verantwortlich zu machen. Doch schon bei der einfachen konvexen Parabel $\mathcal{L}(x) = 25x^2$ tritt Divergenz auf:

<!--en:0d986400-->
- **Gradientenabstieg als Euler-Verfahren:** Die Aktualisierung $x_{k+1} = x_k - \eta \cdot 50x_k = (1 - 50\eta)x_k$ ist ein expliziter Euler-Schritt der Weite $h = \eta$ für die Gleichung des Gradientenflusses $\dot{x} = -50x$.

<!--en:908210d3-->
- **Exakter Fluss und diskrete Schritte:** Die exakte kontinuierliche Trajektorie $x(t) = e^{-50t}x_0$ nähert sich von jedem Anfangspunkt aus monoton der Null.

<!--en:bb99b826-->
- **Instabilität durch die Schrittweite:** Bei $h = 0.01$ multipliziert Euler mit $1 - 50(0.01) = 0.5$ und konvergiert. Bei $h = 0.05$ verkleinert der exakte Fluss den Wert um den Faktor $e^{-2.5} \approx 0.082$. Euler multipliziert dagegen mit $1 - 50(0.05) = -1.5$: Das Vorzeichen wechselt, der Betrag wächst unbegrenzt.

<!--en:a82f52d0-->
Die Differentialgleichung $\dot{x} = -50x$ ist stabil. Die **numerischen Schritte** sind instabil, sobald $h > 2/50 = 0.04$ gilt.

<!--en:07faf183-->
::: {.slide title="Das explizite Euler-Verfahren als Zinseszins auf einem Vektorfeld"}

<!--en:b9a796ef-->
[Verfahren erster Ordnung]{.kicker}

<!--en:5c8283f8-->
Bei Schrittweite $h = T/n$ und Gitterzeitpunkten $t_n = nh$ hält das **explizite Euler-Verfahren** den Geschwindigkeitspfeil vom Anfang jedes Schritts fest:

<!--en:478ff266-->
- **Zinseszins bei $\dot{x} = x$ ($x_0 = 1, T = 1$):** Nach $n$ Schritten ergibt sich $x_n = (1 + 1/n)^n \to e \approx 2.7183$.

<!--en:9f2059d6-->
- **Genauigkeit erster Ordnung ($p = 1$):** Bei Verdopplung von $n$ halbiert sich der Fehler am Endpunkt $|x_n - e| \approx \frac{e}{2n} \approx 1.36\,h$ näherungsweise.

<!--en:2f40e769-->
::: {.slide title="Lokaler Abbruchfehler und globaler Fehler"}

<!--en:44a9369d-->
[Fehleranalyse I]{.kicker}

<!--en:740bea4f-->
- **Lokaler Abbruchfehler ($\|\mathbf{r}_n\| \le \frac12 M h^2$):** Ein einzelner Euler-Schritt von der exakten Trajektorie aus hat einen Fehler von $O(h^2)$.

<!--en:c04dd962-->
- **Fehlerrekursion für einen Schritt:** Für ein $L$-Lipschitz-stetiges Feld erfüllt der globale Fehler $\varepsilon_n = \|\mathbf{x}_n - \mathbf{x}(t_n)\|$ die Ungleichung $\varepsilon_{n+1} \le (1 + hL)\varepsilon_n + \frac12 M h^2$.

<!--en:1c341ff4-->
- **Warum eine Potenz von $h$ verloren geht:** Das Summieren über $n = T/h$ Schritte führt zur Division durch $h$. Das ergibt die globale Ordnung $p = 1$: $\varepsilon_n \le \frac{Mh}{2L}(e^{LT} - 1) = O(h)$.

<!--en:81157bf2-->
::: {.slide title="Globale Konvergenz des expliziten Euler-Verfahrens mit Ordnung 1"}

<!--en:0c621399-->
[Fehleranalyse II]{.kicker}

<!--en:40db3060-->
Das schrittweise Auflösen der Rekursion $\varepsilon_{n+1} \le (1 + hL)\varepsilon_n + \frac{1}{2}Mh^2$ vom Anfangswert $\varepsilon_0 = 0$ aus über $n \le T/h$ Schritte ergibt eine geometrische Summe:

<!--en:7ffe9e99-->
- **Wo eine Potenz von $h$ verloren geht:** Beim Addieren von $T/h$ lokalen Fehlern der Größe $O(h^{p+1})$ entsteht ein Quotient mit Nenner $h$. Der globale Fehler hat daher die **Ordnung $p$** ($O(h^p)$).

<!--en:23a774b8-->
- **Wann $e^{LT}$ scharf und wann der Faktor pessimistisch ist:** Für das expandierende Feld $\dot{x} = x$ ($L=1, M=e$) ergibt die Schranke $2.3h$ bei einem tatsächlichen Fehler von $1.36h$. Für das kontrahierende Feld $\dot{x} = -50x$ erfasst $L = 50$ das Vorzeichen der Steigung nicht. Die Abschätzung liefert $e^{50} \approx 5 \times 10^{21}$, obwohl stabile Schritte frühere Fehler mit dem Faktor $|1 - 50h| < 1$ verkleinern.

<!--en:4fedd32f-->
::: {.slide title="Steigungen innerhalb eines Schritts und die Geometrie der Runge-Kutta-Verfahren"}

<!--en:571d3e30-->
[Verfahren höherer Ordnung]{.kicker}

<!--en:9cdbaa4a-->
Das explizite Euler-Verfahren hält an der Anfangssteigung fest, während die Trajektorie davon abbiegt. **Runge-Kutta-Verfahren** werten das Vektorfeld an Zwischenpunkten aus, damit sich Fehlerterme höherer Ordnung in der Taylor-Entwicklung aufheben:

<!--en:79046679-->
Für $\dot{x} = x$ mit Anfangswert $x_0 = 1$ und einem einzigen großen Schritt $h = 1$ (exakt $e \approx 2.718$) liefert Euler $2.0$, das Mittelpunktverfahren $2.5$ und das 4-stufige RK4-Verfahren $2.708$ (Abweichung unter $0.4\%$).

<!--en:72a7c90f-->
::: {.slide title="Verfahren zweiter Ordnung: Mittelpunktverfahren und Heun-Verfahren"}

<!--en:48cdf035-->
[Verfahren der Ordnung 2]{.kicker}

<!--en:0cc698ad-->
Die Auswertung des Vektorfelds **zweimal pro Schritt** erfasst auch den Term der Ordnung $O(h^2)$ mit der zweiten Ableitung $\ddot{x} = f'(x)f(x)$ in der exakten Taylor-Entwicklung. Der lokale Fehler beträgt dann $O(h^3)$, der globale Fehler hat die **Ordnung $p = 2$**:

<!--en:c2c62708-->
- **Explizites Mittelpunktverfahren:** Ein halber Euler-Probeschritt liefert die Steigung in der Mitte des Schritts bei $t_n + h/2$:
  $$\mathbf{x}_{n+1} = \mathbf{x}_n + h\,\mathbf{f}\!\left(\mathbf{x}_n + \frac{h}{2}\mathbf{f}(\mathbf{x}_n)\right)$$

<!--en:40551695-->
- **Nachweis durch Taylor-Entwicklung (1D):**
  $$x + h f\!\left(x + \tfrac{h}{2}f\right) = x + hf + \tfrac{h^2}{2}f'f + O(h^3)$$

<!--en:7ff0d111-->
- **Heun-Verfahren (explizite Trapezregel):** Nach einem ganzen Euler-Probeschritt zählt der Mittelwert der Steigungen am Anfang und am Ende:
  $$\mathbf{x}_{n+1} = \mathbf{x}_n + \frac{h}{2}\Big[\mathbf{f}(\mathbf{x}_n) + \mathbf{f}\big(\mathbf{x}_n + h\mathbf{f}(\mathbf{x}_n)\big)\Big]$$

<!--en:d264344f-->
- **Einsatz in Diffusionsmodellen:** Moderne Diffusions- und Flow-Matching-Sampler wie EDM verringern mit dem Heun-Verfahren die Schrittzahl erheblich.

<!--en:283822c8-->
::: {.slide title="Klassisches Runge-Kutta-Verfahren vierter Ordnung (RK4)"}

<!--en:bfe7f948-->
[Ein bewährtes Standardverfahren]{.kicker}

<!--en:a70a245a-->
Das **klassische RK4-Verfahren** bestimmt vier Steigungen pro Schritt (am Anfang $\mathbf{k}_1$, zweimal in der Mitte $\mathbf{k}_2, \mathbf{k}_3$ und am Ende $\mathbf{k}_4$) und kombiniert sie mit den Gewichten der Simpson-Regel:

<!--en:a5848813-->
- **Genauigkeit vierter Ordnung ($O(h^4)$):** Bei Halbierung von $h$ sinkt der globale Fehler um den Faktor $2^4 = 16\times$ (Euler: $2\times$, Heun: $4\times$). Pro Schritt sind $4$ Feldauswertungen nötig.

<!--en:82dc0d4a-->
::: {.slide title="Die Stabilitätsfunktion R(z) für die Testgleichung"}

<!--en:06e0b45b-->
[Ein Polynom für Genauigkeit und Stabilität]{.kicker}

<!--en:6ac81d98-->
Für eine skalare Eigenmode $\dot{y} = \lambda y$ ($z = h\lambda$) multipliziert ein Schritt den Zustand mit der **Stabilitätsfunktion** des Verfahrens: $y_{n+1} = R(z)y_n$.

<!--en:80c84f4e-->
- **Explizite Taylor-Polynome:** Euler liefert $R(z) = 1 + z$ ($p=1$), das Mittelpunkt- und das Heun-Verfahren liefern $R(z) = 1 + z + \tfrac12 z^2$ ($p=2$). RK4 entwickelt $e^z$ bis einschließlich $\frac{1}{24}z^4$ ($p=4$).

<!--en:d4573137-->
- **Zwei Aufgaben von $R(z)$:** Die Übereinstimmung mit $e^z$ nahe $z = 0$ bestimmt die **Genauigkeitsordnung** $p$. Das Gebiet $\{|R(z)| < 1\}$ bestimmt die **numerische Stabilität**.

<!--en:1436ad64-->
::: {.slide title="Implementierung des expliziten Euler-Verfahrens und von RK4"}

<!--en:a5a60910-->
[Umsetzung im Code]{.kicker}

<!--en:28680d2e-->
Beide expliziten Verfahren brauchen nur eine Schleife über $n$ Schritte der Weite $h = T/n$ und Auswertungen des Vektorfeldes `f(x)`:

<!--en:fd9db6c2-->
::: {.slide title="Genauigkeitsordnung bei gleichem Rechenaufwand"}

<!--en:1cf1469e-->
[Verfahrensvergleich an der Spirale]{.kicker}

<!--en:47a8e1a0-->
Für die ebene Spirale $\mathbf{A} = -\frac{1}{2}\mathbf{I} + \mathbf{K}$ ($\lambda = -0.5 \pm \mathrm{i}$, $|\lambda T|^2 = 5$ bei $T = 2$) liefert $R(z)$ als Vorhersage für den globalen Fehler $1.84/n$ bei Euler und $0.343/n^4$ bei RK4:

<!--en:d9f8dd70-->
- **Gemessene Steigungen bei doppelt logarithmischer Darstellung:** Lineare Regression ergibt die Ordnung $1.04$ für Euler und $4.02$ für RK4.

<!--en:f15f84e0-->
- **Gleicher Aufwand (512 Auswertungen):** $512$ Euler-Schritte ergeben den Fehler $3.6 \times 10^{-3}$. Mit $128$ RK4-Schritten ($4 \times 128 = 512$ Auswertungen) beträgt der Fehler $1.3 \times 10^{-9}$, sechs Größenordnungen weniger.

<!--en:31451ab8-->
- **Aufwand für dieselbe Genauigkeit wie bei RK4:** Für einen Fehler von $1.3 \times 10^{-9}$ benötigt Euler $n \approx 1.4 \times 10^9$ Auswertungen.

<!--en:f28a6094-->
::: {.slide title="Adaptive Schrittweiten und symplektische Integratoren"}

<!--en:31657a86-->
[Spezialisierte Verfahren]{.kicker}

<!--en:7a5786de-->
- **Adaptive Schrittweitensteuerung (Dormand-Prince RK45):** Das Verfahren vergleicht die eingebetteten Näherungen der Ordnungen $p$ und $p+1$ aus denselben Stufen ($\Delta = \|\hat{\mathbf{x}} - \mathbf{x}\|$) und skaliert die Schrittweite gemäß $h \leftarrow 0.9\,h(\text{tol}/\Delta)^{1/(p+1)}$.

<!--en:f2c8eea6-->
- **Symplektisches Euler- und Leapfrog-Verfahren (HMC):** Bei hamiltonschen Oszillatoren erhält die Folge der Aktualisierungen $q_{n+1} = q_n + h p_n$ und dann $p_{n+1} = p_n - h q_{n+1}$ den Flächeninhalt im Phasenraum exakt ($\det\mathbf{S}_h = 1$), und zwar für jedes $h$. Für $h < 2$ bleiben die Bahnen beschränkt.

<!--en:f4bd28f2-->
::: {.slide title="Steifheit: Schon abgeklungene schnelle Eigenmoden begrenzen die Schrittweite"}

<!--en:6d547161-->
[Schrittweitenbegrenzung durch Steifheit]{.kicker}

<!--en:b76babdd-->
Ein System heißt **steif**, wenn eine schnell abklingende Eigenmode noch winzige explizite Schritte erzwingt, lange nachdem sie abgeklungen ist:

<!--en:9e46467c-->
- **Stabilitätsgrenze:** Zum Zeitpunkt $t = 0.1$ ist $x_1(t) = e^{-50t}$ praktisch abgeklungen, und mit $h = 0.1$ lässt sich $e^{-t}$ bis auf $5\%$ genau verfolgen. Das explizite Euler-Verfahren multipliziert jedoch $x_1$ in jedem Schritt mit $1 - 50(0.1) = -4$. Nur für $h < 2/50 = 0.04$ klingt die berechnete schnelle Eigenmode ab.

<!--en:8994408e-->
::: {.slide title="Implizites Euler-Verfahren und unbedingte A-Stabilität"}

<!--en:82949b24-->
[Implizite Verfahren]{.kicker}

<!--en:dabc3368-->
Das **implizite Euler-Verfahren** bestimmt den Geschwindigkeitspfeil am *Endpunkt* des Schritts:

<!--en:b2d6b740-->
- **Unbedingte A-Stabilität:** Es gilt $|(1 - z)^{-1}| < 1$ in der gesamten linken Halbebene $\operatorname{Re}z < 0$. Bei $h = 0.1, \lambda = -50$ dämpft das Verfahren pro Schritt mit dem Faktor $1/6$.

<!--en:24e4a138-->
- **Rechenaufwand:** Jeder Schritt erfordert das Lösen eines linearen Gleichungssystems oder das Newton-Verfahren.

<!--en:9cb54836-->
::: {.slide title="Steifheit messen mit expliziten und impliziten Schritten"}

<!--en:f473cea7-->
[Numerisches Experiment]{.kicker}

<!--en:82d1b9fa-->
Vergleich zwischen explizitem und implizitem Euler-Verfahren für $\dot{\mathbf{x}} = \mathrm{diag}(-50, -1)\mathbf{x}$ zum Zeitpunkt $T = 1$ mit Schrittweiten $h \in [0.01, 0.10]$:

<!--en:6b79e2f9-->
Bei $h = 0.10$ nähern beide Verfahren die langsame Eigenmode $e^{-1} \approx 0.3679$ bis auf etwa $5\%$ an ($0.3487$ und $0.3855$). Beim expliziten Euler-Verfahren wächst die schnelle Eigenmode jedoch auf $1.0 \times 10^6$ an, während das implizite sie auf $1.7 \times 10^{-8}$ dämpft.

<!--en:830cd541-->
::: {.slide title="Stabilitätsgebiete in der komplexen Ebene z = hλ"}

<!--en:d317149d-->
[A-Stabilität]{.kicker}

<!--en:456894ca-->
Das **Stabilitätsgebiet** $\{z \in \mathbb{C} : |R(z)| < 1\}$ eines Verfahrens zeigt, bei welchen Kombinationen aus Schrittweite $h$ und Eigenwert $\lambda$ die Eigenmode abklingt:

<!--en:136fe23c-->
- **Kreisscheibe des expliziten Euler-Verfahrens ($|1 + z| < 1$):** Die Einheitskreisscheibe um $-1$ reicht auf der reellen Achse bis $z = -2$. Ungedämpfte Drehungen ($z = \mathrm{i}hb$) liegen außerhalb. Bei jedem $h > 0$ spiralt die berechnete Bahn nach außen.

<!--en:bac227e0-->
- **Ausgebuchtetes Stabilitätsgebiet von RK4:** Es reicht auf der negativen reellen Achse bis $z \approx -2.79$ und enthält ein Intervall der imaginären Achse.

<!--en:0c16df45-->
- **Kein explizites RK-Verfahren ist A-stabil:** Jedes $s$-stufige explizite Runge-Kutta-Verfahren hat ein Stabilitätspolynom $R(z)$ vom Grad $\le s$, dessen Betrag für $|z| \to \infty$ unbegrenzt wächst. Nur implizite Verfahren wie das implizite Euler-Verfahren können die gesamte linke Halbebene $\operatorname{Re}z < 0$ abdecken (**A-Stabilität**).

<!--en:85195b7e-->
::: {.slide title="Gradientenabstieg als explizites Euler-Verfahren für den Gradientenfluss"}

<!--en:1c4ce518-->
[Optimierung als dynamischer Prozess]{.kicker}

<!--en:5ee043f2-->
Der kontinuierliche steilste Abstieg für eine Verlustfunktion $\mathcal{L}(\mathbf{x})$ definiert den **Gradientenfluss** $\dot{\mathbf{x}} = -\nabla\mathcal{L}(\mathbf{x})$. Entlang dieses Flusses gilt $\frac{d}{dt}\mathcal{L}(\mathbf{x}(t)) = -\|\nabla\mathcal{L}\|^2 \le 0$. Der Gradientenabstieg mit Lernrate $\eta$ entspricht dem expliziten Euler-Verfahren mit $h = \eta$:

<!--en:0d3a0a16-->
- **Obergrenze für die Lernrate:** Jeder Eigenwert $\lambda_i > 0$ der Hesse-Matrix $\mathbf{H}$ bestimmt den Wert $z_i = -\eta\lambda_i$ für die zugehörige Eigenmode. Für Stabilität $|1 - \eta\lambda_i| < 1$ in allen Eigenmoden muss gelten:
  $$\eta < \frac{2}{\lambda_{\max}(\mathbf{H})}$$

<!--en:f57ab85f-->
- **Schlechte Konditionierung bedeutet Steifheit:** Da die Lernrate $\eta$ durch die stärkste Krümmung $\lambda_{\max}$ begrenzt ist, schrumpft die Komponente in der flachsten Richtung $\lambda_{\min}$ pro Schritt um höchstens den Anteil $2\lambda_{\min}/\lambda_{\max} = 2/\kappa$.

<!--en:c16e7de7-->
::: {.slide title="Drei Verfahren in einem schlecht konditionierten quadratischen Tal"}

<!--en:f473cea7-2-->
[Numerisches Experiment]{.kicker}

<!--en:45d0e585-->
Integration des Gradientenflusses $\dot{\mathbf{x}} = -\mathbf{H}\mathbf{x}$ für die Verlustfunktion $\mathcal{L}(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\mathrm{diag}(10, 1)\mathbf{x}$ über $100$ Schritte der Weite $\eta$:

<!--en:d0323da4-->
**Die Obergrenze ist ein Quotient:** Die Hesse-Matrix der Verlustfunktion liefert den Nenner $\lambda_{\max} = 10$, die Stabilitätsgrenze des Verfahrens auf der reellen Achse den Zähler. Daraus folgen $2 / 10 = 0.20$ für Euler, $2.79 / 10 = 0.279$ für RK4 und $\infty$ für das implizite Euler-Verfahren.

<!--en:411d1ff6-->
::: {.slide title="Warum Optimierer im Deep Learning Euler-Schritte und Momentum nutzen"}

<!--en:a723b9bd-->
[Rechenaufwand und Verfahrenswahl]{.kicker}

<!--en:735171b8-->
- **Maximale Flusszeit pro Gradient:** Bei $\lambda_{\max} = 10$ kommt Euler pro Gradientenauswertung um $0.200$ Zeiteinheiten voran, das 4-stufige RK4-Verfahren nur um $0.279/4 = 0.070$.

<!--en:54eb5147-->
- **Endpunkt und Weg:** Bei der Optimierung zählt nur, dass die Iterierten $\mathbf{x}^\star$ erreichen. Der Rückwärtsfehler des Euler-Verfahrens bewirkt eine implizite Regularisierung mit der modifizierten Verlustfunktion $\mathcal{L}(\mathbf{x}) + \frac{\eta}{4}\|\nabla\mathcal{L}(\mathbf{x})\|^2$.

<!--en:97d1095f-->
- **Momentum ändert die Differentialgleichung:** Heavy-Ball-Momentum diskretisiert einen gedämpften Oszillator zweiter Ordnung $m\ddot{\mathbf{x}} + c\dot{\mathbf{x}} = -\nabla\mathcal{L}$ und erreicht so eine Skalierung mit $\sqrt{\kappa}$ bei $1$ Gradientenauswertung pro Schritt.

<!--en:397ebff9-->
::: {.slide title="Verständnisfragen zu Ordnung, Steifheit und Stabilitätsgebieten"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:1e5981e1-->
Drei Fragen zu den numerischen Verfahren:

<!--en:1b0cb434-->
1. **Fehler und Rechenaufwand:** Bei einer glatten gewöhnlichen Differentialgleichung halbiert sich die Schrittweite: $h \to h/2$. Um welchen Faktor sinkt der globale Fehler beim **Heun-Verfahren** und bei **RK4**?

<!--en:250cae3c-->
2. **Grenze der Schrittweite für die Spirale:** Für die ebene Spirale mit Eigenwerten $\lambda = -0.5 \pm \mathrm{i}$ gilt beim expliziten Euler-Verfahren $|1 + h\lambda|^2 = (1 - 0.5h)^2 + h^2$. Wo liegt die obere Grenze für die Schrittweite $h$, bis zu der die Euler-Iterierten abklingen?

<!--en:c41a9a82-->
3. **Verlauf der Divergenz:** Die Lernrate $\eta$ liegt für $\dot{x} = -10x$ knapp über der Stabilitätsgrenze. Wie unterscheidet sich der sichtbare Verlauf beim **Euler-Verfahren** mit $\eta = 0.21$ von dem bei **RK4** mit $\eta = 0.29$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:65ad4bbd-->
1. Beim Heun-Verfahren (Ordnung 2) sinkt der Fehler um den Faktor $2^2 = 4\times$, bei RK4 (Ordnung 4) um den Faktor $2^4 = 16\times$.

<!--en:cd9ec00d-->
3. Der Euler-Faktor $R(-2.1) = 1 - 2.1 = -1.1 < -1$ ist negativ. Die Iterierten springen daher **im Zickzack** mit wachsender Amplitude über die Talsohle. Das RK4-Polynom $R(-2.9) = +1.187 > 1$ ist positiv. Die Iterierten bewegen sich auf einer Seite **monoton bergauf**, ohne das Vorzeichen zu wechseln.

<!--en:5ec6dc14-->
::: {.slide title="Zusammenfassung: Numerische Verfahren und Stabilität"}

<!--en:c5a222f8-->
[Das Wichtigste im Überblick]{.kicker}

<!--en:dd344759-->
| Verfahren / Begriff | Aktualisierung oder Formel | Ordnung $p$ | Stabilitätsbedingung ($z = h\lambda$) | Rolle im Deep Learning |

<!--en:6c7f95b6-->
| **Explizites Euler-Verfahren** | $\mathbf{x}_{n+1} = \mathbf{x}_n + h\mathbf{f}(\mathbf{x}_n, t_n)$ | $1$ ($O(h)$) | $\|1 + z\| < 1$ ($h < 2/\|\lambda\|$) | Gradientenabstieg, ResNet-Blöcke, DDIM |

<!--en:eced7ecc-->
| **Heun-Verfahren** | $\mathbf{x}_n + \frac{h}{2}[\mathbf{f}(\mathbf{x}_n) + \mathbf{f}(\mathbf{x}_n + h\mathbf{f}(\mathbf{x}_n))]$ | $2$ ($O(h^2)$) | $\|1 + z + \frac{1}{2}z^2\| < 1$ | 2-stufige Diffusions- und Flow-Matching-Sampler |

<!--en:fe10109e-->
| **Klassisches RK4-Verfahren** | $\mathbf{x}_n + \frac{h}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$ | $4$ ($O(h^4)$) | $\|1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24}\| < 1$ | Kontinuierliche normalisierende Flüsse mit hoher Genauigkeit |

<!--en:b5539711-->
| **Implizites Euler-Verfahren** | $\mathbf{x}_{n+1} = \mathbf{x}_n + h\mathbf{f}(\mathbf{x}_{n+1}, t_{n+1})$ | $1$ ($O(h)$) | $\|(1 - z)^{-1}\| < 1$ (A-stabil für $\operatorname{Re}z < 0$) | Steife Systeme und proximale Optimierung |

<!--en:5c013ba7-->
| **Gradientenfluss** | $\dot{\mathbf{x}} = -\nabla\mathcal{L}(\mathbf{x})$, $\mathbf{J} = -\mathbf{H}$ | – | $\eta < 2/\lambda_{\max}(\mathbf{H})$ beim Euler-Verfahren | Erklärt Divergenz bei großer Lernrate und die Rolle der Konditionszahl $\kappa$ |
