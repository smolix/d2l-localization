<!--en:ac979613-->
# Gewöhnliche Differentialgleichungen

<!--en:bd4e339e-->
Ein Gradientenschritt, ein Residualblock und ein Schritt eines Diffusions-Samplers haben etwas gemeinsam, wie wir in der Kapiteleinleitung festgestellt haben. Jeder macht einen kurzen Schritt entlang eines Pfeils, der angibt, wie schnell sich ein Zustand ändert. Im nächsten Abschnitt fragen wir, wie gut solche Schritte sind. Hier fragen wir zunächst, *welche Bewegung* sie annähern. Stellen wir uns vor, an jedem Punkt sei eine Geschwindigkeit vorgegeben, wie der Wind auf einer Wetterkarte. Wir lassen einen Ballon steigen. Steht seine Bahn damit fest? Können zwei Ballons, die am selben Ort starten, getrennte Wege nehmen? Können wir sagen, wo der Ballon vor einer Stunde war? Und wo landet er schließlich?

<!--en:0ee53314-->
Die Newtonschen Gesetze von 1687 sind Regeln dieser Art. Sie geben an, wie sich der Zustand eines Körpers ändert, und lassen seinen künftigen Ort zunächst offen. Dass eine solche Regel genau eine Lösung hat, ist keineswegs selbstverständlich. Picard und Lindelöf bewiesen in den 1890er-Jahren einen Satz, der dies garantiert. Er setzt voraus, dass sich die Geschwindigkeit von einem Punkt zum nächsten nicht zu abrupt ändert. Wir werden eine harmlos aussehende Regel kennenlernen, die diese Voraussetzung verletzt.

<!--en:ae83bcd0-->
Modelle brauchen diesen Satz. Gibt ein Netz die Geschwindigkeit vor, so ist die Ausgabe des Modells der Ort, an den die Bewegung seine Eingabe trägt. Die Eindeutigkeit macht die Ausgabe zu einer Funktion der Eingabe. Lassen wir die Bewegung rückwärts ablaufen, erhalten wir die Umkehrfunktion, die ein generativer Fluss braucht. Bei einer linearen Regel können wir mehr sagen und die Lösung explizit angeben. Die Eigenwerte ihrer Matrix verraten, ob die Bewegung abklingt, anwächst oder sich dreht. An ihnen lassen sich das Gedächtnis eines Zustandsraummodells (:numref:`sec_ssm`) und die Konvergenz eines Trainingslaufs ablesen.

<!--en:e1e96b0c-->
Wir beginnen mit zwei Bewegungen, die wir von Hand berechnen können, beweisen den Satz und lösen anschließend die linearen Gleichungen vollständig. Alles Weitere beruht auf zwei Ergebnissen. Ändert sich die Geschwindigkeit hinreichend sanft, verläuft durch jeden Punkt genau eine Trajektorie. Bei einer linearen Regel entscheiden die Realteile der Eigenwerte, ob die Bewegung abklingt. Die letzten beiden Unterabschnitte behandeln ein abklingendes System, das zunächst anwächst, und das Pendel. Bei der ersten Lektüre können Sie diese beiden Unterabschnitte überspringen.

<!--en:599e8650-->
## Vektorfelder und Flüsse

<!--en:2f9a9cca-->
### Abklingen, Rotation und Spirale

<!--en:8b01ff8c-->
Die einfachste Bewegung ist das *Abklingen* (engl. decay). Die Regel „Bewege dich mit einer zum Abstand proportionalen Geschwindigkeit auf null zu“ entspricht der Gleichung $\dot x=-x$. Der Punkt bezeichnet die Ableitung nach der Zeit. Mit der Anfangsbedingung $x(0)=x_0$ lautet die Lösung

<!--en:a9000deb-->
wie die Ableitung zeigt: $\dot x=-e^{-t}x_0=-x$. Mit jeder Zeiteinheit verringert sich der Abstand zum Ursprung um den Faktor $e^{-1}\approx0.37$.

<!--en:667375d6-->
Als Nächstes betrachten wir die *Rotation*. In der Ebene erhalten wir den Geschwindigkeitsvektor, indem wir den Ortsvektor um einen rechten Winkel drehen,

<!--en:f7768700-->
also $\dot x_1=-x_2$ und $\dot x_2=x_1$. Die Geschwindigkeit steht senkrecht auf dem Ortsvektor, denn $\mathbf x^\top\mathbf K\mathbf x=-x_1x_2+x_2x_1=0$. Daher bleibt die Länge von $\mathbf x$ konstant:

<!--en:9ea145aa-->
Der Punkt bewegt sich auf einem Kreis. Da sein Geschwindigkeitsbetrag $\|\mathbf K\mathbf x\|$ gleich dem Radius $\|\mathbf x\|$ ist, legt er pro Zeiteinheit einen Winkel von einem Radiant zurück. Ein Umlauf dauert $2\pi$.

<!--en:966ce3da-->
Addieren wir zur Rotation die Hälfte des Abklingterms, erhalten wir die Gleichung, die uns in diesem und den nächsten beiden Abschnitten als Beispiel begleitet:

<!--en:2ec604b3-->
Der Rotationsterm lässt die Länge unverändert, der Abklingterm verringert sie: $\tfrac{d}{dt}\|\mathbf x\|^2=2\mathbf x^\top\mathbf A\mathbf x
=-\|\mathbf x\|^2$. Das ist wieder die Abklinggleichung, diesmal für das Quadrat der Länge. Also gilt $\|\mathbf x(t)\|^2=e^{-t}\|\mathbf x_0\|^2$ und $\|\mathbf x(t)\|=e^{-t/2}\|\mathbf x_0\|$. Der Abstand zum Ursprung nimmt mit der Rate $\tfrac12$ ab, während der Punkt mit Winkelgeschwindigkeit eins rotiert. Er nähert sich dem Ursprung auf einer Spirale (:numref:`fig_mdl-dyn-ode-field`). Die Eigenwerte von $\mathbf A$ sind $-0.5\pm\mathrm i$. Ihr Realteil $-\tfrac12$ bestimmt das Abklingen, ihr Imaginärteil $\pm1$ die Winkelgeschwindigkeit. Das ist kein Zufall. Wie :numref:`sec_mdl-linear-odes-stability` zeigt, setzt sich jede solche lineare Gleichung aus Abkling-, Wachstums- und Rotationsbewegungen zusammen, mit einem Beitrag je Eigenwert.

<!--en:f91522e7-->
![Eine gewöhnliche Differentialgleichung als Geschwindigkeitsfeld. Die blassen Pfeile auf dem Gitter zeigen das Feld $\mathbf{f}(\mathbf{x})=\mathbf{A}\mathbf{x}$ mit $\mathbf{A}=\left(\begin{smallmatrix}-0.5&-1\\1&-0.5\end{smallmatrix}\right)$ und den Eigenwerten $-0.5\pm\mathrm{i}$. Die beiden farbigen Kurven sind Trajektorien. Jede folgt zu jedem Zeitpunkt dem Pfeil an ihrem Ort und läuft auf einer Spirale in den Ursprung. Nur dort verschwindet das Feld (Fixpunkt).](../img/mdl-dyn-ode-field.svg)

<!--en:8473b749-->
### Das Anfangswertproblem

<!--en:d8a01198-->
Die meisten Bewegungsgesetze sind nichtlinear. Wir brauchen daher eine Form, die sie alle erfasst. Das allgemeine Problem gibt ein Geschwindigkeitsfeld und einen Anfangspunkt vor:

<!--en:2370ac69-->
Hier ist $\mathbf x(t)\in\mathbb R^d$ der Zustand. Die Funktion $\mathbf f:\mathbb R^d\times\mathbb R\to\mathbb R^d$ ist ein **Vektorfeld** (engl. vector field). Sie ordnet jedem Punkt zu jeder Zeit einen Pfeil zu. Eine **Trajektorie** ist eine differenzierbare Bahn, deren Tangentenvektor zu jedem Zeitpunkt mit dem Pfeil am jeweiligen Ort übereinstimmt.

<!--en:bc7cc0b4-->
Der Vektor $\mathbf x_0$ ist der **Anfangswert** (engl. initial value), und :eqref:`eq_mdl-ode-ivp` heißt **Anfangswertproblem** (engl. initial-value problem). Hängt $\mathbf f$ nicht von $t$ ab, heißt das System **autonom**. Die Pfeile stehen fest, nur der Zustand bewegt sich. Abklingen, Rotation und Spirale entsprechen den autonomen Feldern $\mathbf f(\mathbf x)=-\mathbf x$, $\mathbf K\mathbf x$ und $\mathbf A\mathbf x$. Eine Wetterkarte ist nicht autonom, denn der Wind ändert sich, während der Ballon fliegt.

<!--en:b719a2ba-->
Integrieren wir beide Seiten von $0$ bis $t$ (:numref:`sec_mdl-integral_calculus`), erhalten wir die äquivalente **Integralform**

<!--en:63fe6b33-->
Die Lösung ist der Anfangswert plus die über die Zeit aufsummierte Geschwindigkeit. Die unbekannte Bahn steht auf beiden Seiten. Wir haben also weiterhin eine Gleichung und noch keine explizite Formel, können damit aber weiter unten die Existenz einer Lösung beweisen.

<!--en:35f1e2e1-->
### Existenz und Eindeutigkeit

<!--en:a76ea117-->
Hat ein Anfangswertproblem eine Lösung, und zwar genau eine? Nicht immer. Betrachten wir $\dot x=\sqrt{|x|}$ mit $x_0=0$. Die konstante Bahn $x\equiv0$ löst die Gleichung. Für jede Wartezeit $c\ge0$ ist auch folgende Bahn eine Lösung:

<!--en:278b3773-->
denn $\dot x_c=(t-c)/2=\sqrt{x_c}$ für $t>c$. Ein ruhendes Teilchen kann sich jederzeit in Bewegung setzen. Die Lösungen fächern sich von der Achse aus auf, eine für jeden Aufbruchszeitpunkt (Teilbild (a) in :numref:`fig_mdl-dyn-uniqueness-fan`). Wir nennen dieses Bild daher den *Fächer*. Ein Modell mit diesem Feld würde seiner Eingabe keine eindeutige Ausgabe zuordnen. Der Grund liegt in der Steigung von $\sqrt{|x|}$, die bei $0$ unbeschränkt ist. Eine kleine Auslenkung $x$ führt zu einer viel größeren Geschwindigkeit $\sqrt{|x|}$.

<!--en:f5768a72-->
![Zwei Felder, die der folgende Satz nicht erfasst. (a) Lösungen von $\dot{x} = \sqrt{|x|}$ mit $x(0) = 0$: die konstante Lösung $x \equiv 0$ und die parabelförmige Lösung $x_c(t) = (t-c)^2/4$ für jeden Aufbruchszeitpunkt $c$. Die Steigung von $\sqrt{|x|}$ ist am Ursprung unbeschränkt. Keine Konstante begrenzt dort die Änderung der Geschwindigkeit im Verhältnis zur Ortsänderung (die Lipschitz-Bedingung des Satzes ist verletzt). Die Lösung ist nicht eindeutig. (b) Lösungen von $\dot x = x^2$ mit den Anfangswerten $x_0 = 2$, $1$ und $1/2$ (Punkte). Hier ist die Steigung des Feldes in der Nähe jedes Punktes beschränkt, aber es gibt keine gemeinsame Schranke für alle $x$. Jede Lösung $x_0/(1 - x_0 t)$ existiert nur bis zum Zeitpunkt $t = 1/x_0$ (gestrichelt), an dem sie gegen unendlich geht.](../img/mdl-dyn-uniqueness-fan.svg)

<!--en:9d595415-->
Für ein hinreichend reguläres Feld lässt sich die Lösung durch Iteration der Integralform :eqref:`eq_mdl-ode-integral-form` konstruieren. Wir wählen eine vorläufige Bahn, lesen entlang der Bahn die vorgegebenen Geschwindigkeiten ab und integrieren sie vom Anfangspunkt aus. So erhalten wir eine neue Bahn und wiederholen den Vorgang. Das ist die **Picard-Iteration**. Für $\dot x=x$ mit $x_0=1$ beginnen wir mit der konstanten Bahn $\varphi_0\equiv1$ und setzen $\varphi_{k+1}(t)=1+\int_0^t\varphi_k(s)\,ds$:

<!--en:7e12dc24-->
Die Iterierten konvergieren gegen die Lösung $e^t$. Hier kommt bei jeder Iteration ein Term der Taylorreihe von $e^t$ hinzu. Bei anderen Feldern sehen die Iterierten anders aus. Sie konvergieren aber ebenfalls, sofern sich das Feld von Punkt zu Punkt nicht zu abrupt ändert. Was „zu abrupt“ bedeutet, präzisiert der folgende Satz.

<!--en:cb1256c8-->
**Satz (Picard–Lindelöf).** *Sei $\mathbf{f}(\mathbf{x}, t)$ stetig in $t$ und global Lipschitz-stetig in $\mathbf{x}$. Es gebe also ein $L$ mit*

<!--en:62a8e952-->
*Dann hat das Anfangswertproblem :eqref:`eq_mdl-ode-ivp` auf $[0, T]$ für jeden Anfangswert $\mathbf{x}_0$ und jedes $T>0$ genau eine Lösung* :cite:`Coddington.Levinson.1955`.

<!--en:46993648-->
Dank dieses Satzes können wir ein Modell durch ein Feld definieren. Die Konstante $L$ begrenzt, wie stark sich die Pfeile von Punkt zu Punkt ändern dürfen. Für das Abklingen gilt $L=1$, für ein lineares Feld $\mathbf A\mathbf x$ gilt $L=\|\mathbf A\|$. Mit der Norm einer Matrix meinen wir hier und im restlichen Abschnitt ihre Spektralnorm, also den größten Faktor, um den die Matrix einen Vektor streckt (:numref:`subsec_mdl-normal-matrices`). Der Fächer zeigt, was geschieht, wenn keine solche Konstante $L$ in einer Umgebung von $0$ existiert. Wir zeigen nun, dass die Picard-Iterierten konvergieren und ihr Grenzwert die einzige Lösung ist.

<!--en:28dd6e72-->
**Beweis.** Eine stetige Bahn $\boldsymbol\varphi$ auf $[0,\delta]$ löst das Problem genau dann, wenn sie ein Fixpunkt des **Picard-Operators** ist:

<!--en:f0293896-->
In Worten entspricht $P$ einem Schritt der Picard-Iteration. Der Operator liest die Geschwindigkeiten entlang einer vorläufigen Bahn ab und integriert sie vom Anfangspunkt aus. (Eine Lösung erfüllt die Integralform. Umgekehrt ist ein stetiger Fixpunkt nach dem Hauptsatz der Differential- und Integralrechnung differenzierbar und hat die geforderte Ableitung und den geforderten Anfangswert.)

<!--en:4055bcbd-->
Der entscheidende Schritt nutzt die Lipschitz-Schranke unter dem Integral. Für zwei Bahnen auf $[0,\delta]$ schreiben wir $d(\boldsymbol\varphi,\boldsymbol\psi)=\max_{t\le\delta}\|\boldsymbol\varphi(t)-\boldsymbol\psi(t)\|$. Dann gilt für $t\le\delta$

<!--en:d089a3c2-->
Für $\delta=1/(2L)$ verringert jede Anwendung von $P$ den Abstand zweier möglicher Bahnen auf höchstens die Hälfte.

<!--en:9df1fa46-->
Wir beginnen mit der konstanten Bahn $\boldsymbol\varphi_0\equiv\mathbf x_0$ und setzen $\boldsymbol\varphi_{k+1}=P\boldsymbol\varphi_k$. Der Abstand zweier aufeinanderfolgender Iterierter beträgt dann höchstens $2^{-k}d(\boldsymbol\varphi_1,\boldsymbol\varphi_0)$. Diese Schranken summieren sich wie eine geometrische Reihe. Die Iterierten konvergieren also bezüglich des Abstands $d$, das heißt gleichmäßig auf $[0,\delta]$, gegen eine Bahn $\boldsymbol\varphi$. Ein Ergebnis der Analysis verwenden wir hier ohne Beweis: Der gleichmäßige Grenzwert stetiger Bahnen ist stetig. Wegen $d(P\boldsymbol\varphi_k,P\boldsymbol\varphi)\le\tfrac12d(\boldsymbol\varphi_k,\boldsymbol\varphi)\to0$ konvergieren die Bahnen $P\boldsymbol\varphi_k=\boldsymbol\varphi_{k+1}$ sowohl gegen $P\boldsymbol\varphi$ als auch gegen $\boldsymbol\varphi$. Somit gilt $\boldsymbol\varphi=P\boldsymbol\varphi$, der Grenzwert ist also eine Lösung. Zwei Lösungen $\boldsymbol\varphi,\boldsymbol\psi$ müssten $d(\boldsymbol\varphi,\boldsymbol\psi)=d(P\boldsymbol\varphi,P\boldsymbol\psi)\le\tfrac12d(\boldsymbol\varphi,\boldsymbol\psi)$ erfüllen und stimmen daher überein.

<!--en:fc07dcde-->
Schließlich beginnen wir bei $t=\delta$ mit dem Anfangswert $\boldsymbol\varphi(\delta)$ erneut. Die Schrittlänge $1/(2L)$ hängt nicht vom Ort der Lösung ab, denn $L$ ist überall gleich. Somit überdecken $\lceil 2LT\rceil$ Teilintervalle dieser Länge das Intervall $[0,T]$, mit genau einer Lösung auf jedem Teilintervall. $\blacksquare$

<!--en:75ec1886-->
Der Satz verlangt eine Lipschitz-Schranke und fordert, dass sie global gilt. Jede der beiden Forderungen schließt ein anderes Problem aus. Der Fächer zeigt, was ohne Lipschitz-Schranke geschehen kann. Ohne globale Schranke kann die Lösung in endlicher Zeit unendlich werden. Das Feld $\dot x=x^2$ ist glatt, und wir können die Gleichung lösen. Wegen $\tfrac{d}{dt}(-1/x)=\dot x/x^2=1$ wächst die Größe $-1/x$ mit der Rate eins: $-1/x(t)=t-1/x_0$. Damit gilt

<!--en:d2b4e5c3-->
Bei einem Anfangswert $x_0>0$ geht diese Lösung zum endlichen Zeitpunkt $t=1/x_0$ gegen unendlich (Teilbild (b) in :numref:`fig_mdl-dyn-uniqueness-fan`). Der letzte Beweisschritt zeigt, warum der Satz diesen Fall nicht erfasst. Auf dem Intervall $|x|\le\rho$ ist das Feld Lipschitz-stetig mit $L=2\rho$, denn $|x^2-y^2|=|x+y|\,|x-y|$. Die Länge $1/(2L)=1/(4\rho)$ der Teilintervalle schrumpft also, wenn die Lösung wächst. Ihre Summe kann endlich bleiben. Ist ein Feld nur auf beschränkten Mengen Lipschitz-stetig, so ist die Existenz einer Lösung nur für eine gewisse Zeit garantiert.

<!--en:0c163fe6-->
### Die Flussabbildung

<!--en:7936817d-->
Ein generatives Modell bewegt eine ganze Wolke von Anfangspunkten zugleich. Wir halten daher nun das Feld fest und variieren den Anfangspunkt. Beginnen wir mit dem einfachen Fall eines zeitunabhängigen Feldes. Auf den Wind kommen wir am Ende des Unterabschnitts zurück. Für ein autonomes, global Lipschitz-stetiges Feld schreiben wir

<!--en:ab63f527-->
für den Zustand zur Zeit $t$ auf der einzigen Trajektorie, die in $\mathbf x_0$ startet. Diese **Flussabbildung** (engl. flow map) $\Phi_t:\mathbb R^d\to\mathbb R^d$ bewegt den gesamten Raum um die Zeitspanne $t$ entlang der Pfeile vorwärts.

<!--en:1d1c4ba7-->
Beim Abklingen lautet die Flussabbildung $\Phi_t(x)=e^{-t}x$. Folgen wir dem Fluss zunächst für die Dauer $s$ und dann für die Dauer $t$, so ergibt sich $\Phi_t(\Phi_s(x))=e^{-t}e^{-s}x=e^{-(t+s)}x=\Phi_{t+s}(x)$. Die Abbildung $\Phi_{-t}(x)=e^{t}x$ kehrt $\Phi_t$ um. Dasselbe Gesetz gilt für jedes autonome Feld:

<!--en:c7637be7-->
Der Grund ist, dass die Pfeile zeitlich unverändert bleiben. Die Bahn $\tau\mapsto\mathbf x(s+\tau)$ beginnt in $\Phi_s(\mathbf x_0)$ und folgt denselben Pfeilen. Wegen der Eindeutigkeit ist sie daher die Trajektorie, die dort startet. Das Beispiel des Abklingens legt nahe, dass das Gesetz auch für negative Zeiten gilt. Das trifft zu, denn für die Rückwärtsbewegung können wir denselben Satz auf ein zweites Feld anwenden.

<!--en:f53277a2-->
**Korollar (die Flussabbildung ist bijektiv).** *Sei $\mathbf f(\mathbf x)$ autonom und global Lipschitz-stetig. Dann existiert $\Phi_t$ für jedes $t\in\mathbb R$, und jedes $\Phi_t$ ist eine Bijektion von $\mathbb R^d$ auf sich selbst mit der Umkehrabbildung $\Phi_{-t}$. Hängt dagegen $\mathbf f(\mathbf x,t)$ von der Zeit ab und erfüllt die Voraussetzungen des Satzes, so gilt für jedes $T>0$: Die Abbildung vom Zustand zur Zeit $0$ auf den Zustand zur Zeit $T$ ist eine Bijektion von $\mathbb R^d$ auf sich selbst.*

<!--en:e3cd9d92-->
**Beweis.** Das Feld $\mathbf g(\mathbf y)=-\mathbf f(\mathbf y)$ ist Lipschitz-stetig mit demselben $L$. Eine Bahn $\mathbf x(t)$ löst $\dot{\mathbf x}=\mathbf f$ auf $[-T,0]$ genau dann, wenn $\mathbf y(s)=\mathbf x(-s)$ die Gleichung $\dot{\mathbf y}=\mathbf g$ auf $[0,T]$ löst, denn $\dot{\mathbf y}(s)=-\dot{\mathbf x}(-s)=-\mathbf f(\mathbf y(s))$. Auf $\mathbf g$ angewandt liefert der Satz für jeden Anfangspunkt eine eindeutige Lösung rückwärts in der Zeit. Also ist $\Phi_t$ für alle reellen $t$ definiert. Beim Beweis des Verknüpfungsgesetzes :eqref:`eq_mdl-ode-flow-group` spielten die Vorzeichen von $s$ und $t$ keine Rolle. Daraus folgt $\Phi_{-t}\circ\Phi_t=\Phi_t\circ\Phi_{-t}=\Phi_0=\mathrm{id}$.

<!--en:94db6126-->
Für die Zeitumkehr mussten die Pfeile nicht stillstehen. Lassen wir bei einem zeitabhängigen Feld den Film vom Zeitpunkt $T$ an rückwärts laufen. Eine Bahn $\mathbf x(t)$ löst $\dot{\mathbf x}=\mathbf f(\mathbf x,t)$ auf $[0,T]$ genau dann, wenn $\mathbf y(s)=\mathbf x(T-s)$ die Gleichung $\dot{\mathbf y}=\mathbf g(\mathbf y,s)$ mit $\mathbf g(\mathbf y,s)=-\mathbf f(\mathbf y,T-s)$ löst. Dieses Feld $\mathbf g$ ist wiederum stetig in der Zeit und Lipschitz-stetig mit demselben $L$. Nach dem Satz geht also von jedem Punkt $\mathbf z$ genau eine rückwärts durchlaufene Bahn aus. Vorwärts gelesen ist sie die einzige Trajektorie, die in $\mathbf z$ zur Zeit $T$ endet. Jeder Endzustand hat genau einen Anfangspunkt. $\blacksquare$

<!--en:c26b9f7b-->
Für ein zeitabhängiges Feld entfällt das Gesetz :eqref:`eq_mdl-ode-flow-group` nur in dieser Form. Eine Stunde Wind ab Mittag bewirkt eine andere Abbildung als eine Stunde Wind ab Mitternacht. Die Abbildung hängt also sowohl von der Anfangszeit als auch von der Dauer ab. Schreiben wir $\Phi_{s\to t}(\mathbf x)$ für den Zustand zur Zeit $t$ auf der Trajektorie, die den Punkt $\mathbf x$ zur Zeit $s$ durchläuft. Der Weg von $s$ nach $r$ und weiter von $r$ nach $t$ entspricht nach demselben Eindeutigkeitsargument wie zuvor dem Weg von $s$ nach $t$:

<!--en:1e330f49-->
Die drei Zeiten können in beliebiger Reihenfolge liegen. Daher ist $\Phi_{t\to s}$ die Umkehrabbildung von $\Phi_{s\to t}$. Für ein autonomes Feld gilt $\Phi_{s\to t}=\Phi_{t-s}$, und :eqref:`eq_mdl-ode-two-time-flow` ergibt das bisherige Gesetz. Bei einem zeitabhängigen Feld behalten wir das Symbol $\Phi_t$ als Kurzform für $\Phi_{0\to t}$ bei. Das Gesetz für zwei Zeitpunkte ist mehr als eine Formalität. Die Sampler mit wenigen Schritten aus :numref:`sec_diffusion-few-step` trainieren ein Netz darauf, die Abbildung $\Phi_{s\to t}$ selbst nachzubilden. Manche nutzen das Gesetz als Trainingssignal: Zwei kurze Sprünge des Netzes müssen zusammen seinen langen Sprung ergeben.

<!--en:d7a7c3a9-->
Daraus folgt, dass zwei verschiedene Trajektorien nie zur selben Zeit am selben Punkt sind. Liefen sie dort zusammen, ließe der Satz von diesem Treffpunkt aus rückwärts nur eine Bahn zu. Für ein Lipschitz-stetiges Windfeld beantwortet das die ersten drei Fragen der Einleitung. Die Bahn des Ballons liegt fest, zwei gemeinsam gestartete Ballons bleiben zusammen, und wenn wir die Wetterkarte rückwärts abspielen, sehen wir, wo sie vor einer Stunde waren.

<!--en:bcd2fcda-->
Ein generativer Fluss braucht eine invertierbare Abbildung vom Rauschen auf die Daten, deren Umkehrabbildung sich berechnen lässt. Die Felder vieler neuronaler Netze erfüllen die Voraussetzungen. Ein Feld aus linearen Abbildungen und Lipschitz-stetigen Aktivierungsfunktionen, etwa $\mathbf f(\mathbf x)=\mathbf W_2\tanh(\mathbf W_1\mathbf x+\mathbf b_1)+\mathbf b_2$, ist global Lipschitz-stetig. Seine Lipschitz-Konstante $L$ ist höchstens das Produkt der Normen der Gewichtsmatrizen und der Lipschitz-Konstanten der Aktivierungsfunktionen (hier $\|\mathbf W_2\|\,\|\mathbf W_1\|$). Das Modell aus :numref:`sec_mdl-neural-odes-cnf`, dessen Feld auch zeitabhängig sein darf, ist daher wohldefiniert. Nach dem Korollar ist seine Abbildung von der Eingabe auf die Ausgabe bijektiv.

<!--en:446f45d2-->
## Lineare gewöhnliche Differentialgleichungen und Stabilität

<!--en:2595a71a-->
Für eine Klasse von Feldern können wir die letzte Frage der Einleitung vollständig beantworten: Wo endet die Bewegung? Für lineare Felder gibt es eine geschlossene Lösung. Zudem beschreiben sie die meisten anderen Felder in der Nähe eines Punktes, an dem das Feld verschwindet.

<!--en:7281cc67-->
### Das Matrixexponential

<!--en:345cfc3f-->
Die skalare Gleichung $\dot x=ax$ hat die Lösung $x(t)=e^{at}x_0$. Für ein lineares System

<!--en:ae3221f4-->
suchen wir dieselbe Form mit einer Matrix im Exponenten. Für jede quadratische Matrix $\mathbf C$ definieren wir das **Matrixexponential** durch dieselbe Potenzreihe, die $e^z$ definiert:

<!--en:1b352714-->
Die Reihe konvergiert für jede Matrix. Strecken wir einen Vektor erst mit einer Matrix und dann mit einer zweiten, ist die gesamte Streckung höchstens das Produkt der beiden Streckfaktoren: $\|\mathbf C\mathbf D\|\le\|\mathbf C\|\,\|\mathbf D\|$. Daher gilt $\|\mathbf C^k\| \le \|\mathbf C\|^k$. Die konvergente skalare Reihe für $e^{\|\mathbf C\|}$ beschränkt die Matrixreihe gliedweise in der Norm.

<!--en:43a6e932-->
**Satz (das Matrixexponential löst lineare gewöhnliche Differentialgleichungen).** *$\mathbf{x}(t) = e^{\mathbf At}\mathbf{x}_0$ ist die eindeutige Lösung von $\dot{\mathbf{x}} = \mathbf A\mathbf{x}$ mit $\mathbf{x}(0) = \mathbf{x}_0$.*

<!--en:ee7a9e8c-->
**Beweis.** Wir differenzieren die Reihe :eqref:`eq_mdl-ode-matrix-exp-series` mit $\mathbf C = \mathbf At$ gliedweise. Das ist zulässig, denn auf jedem beschränkten $t$-Intervall konvergiert auch die abgeleitete Reihe nach derselben Abschätzung gleichmäßig:

<!--en:9e9b1908-->
Also erfüllt $\mathbf{x}(t) = e^{\mathbf At}\mathbf{x}_0$ die Differentialgleichung. Bei $t = 0$ bleibt von der Reihe nur $\mathbf I$ übrig, also gilt $\mathbf{x}(0) = \mathbf{x}_0$. Die Eindeutigkeit folgt aus dem Satz von Picard–Lindelöf, denn das Feld $\mathbf{x} \mapsto \mathbf A\mathbf{x}$ ist Lipschitz-stetig mit $L = \|\mathbf A\|$. $\blacksquare$

<!--en:bc952285-->
Aus einer unendlichen Reihe von Matrixpotenzen lässt sich das Verhalten des Systems noch schwer ablesen. Die Eigenwertzerlegung macht daraus $d$ skalare Exponentialfunktionen. Erinnern wir uns an :numref:`sec_mdl-eigendecompositions`: Hat $\mathbf A$ insgesamt $d$ linear unabhängige Eigenvektoren $\mathbf v_j$ mit $\mathbf A\mathbf v_j=\lambda_j\mathbf v_j$, so gilt mit $\mathbf V=(\mathbf v_1,\ldots,\mathbf v_d)$ und $\boldsymbol\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_d)$ die Zerlegung $\mathbf A=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}$.

<!--en:e4f797a5-->
Die Potenzen vereinfachen sich dann zu $(\mathbf At)^k = \mathbf V (\boldsymbol\Lambda t)^k \mathbf V^{-1}$, weil sich alle benachbarten Faktoren $\mathbf V^{-1} \mathbf V$ aufheben. Summieren wir die Reihe :eqref:`eq_mdl-ode-matrix-exp-series`, erhalten wir

<!--en:77e474f4-->
Im Abschnitt über Eigenwerte haben wir diese Formel bereits angegeben. Nun haben wir sie auch hergeleitet. Sie besagt, dass das System in unabhängige Gleichungen zerfällt. Entwickeln wir den Anfangszustand in der Eigenbasis, $\mathbf{x}_0 = \sum_j c_j \mathbf{v}_j$, so gilt

<!--en:bd858bd3-->
Entlang jeder Eigenrichtung gilt die skalare Gleichung $\dot{x} = \lambda_j x$, unabhängig von den übrigen Richtungen. Jeder Term $c_je^{\lambda_jt}\mathbf v_j$ heißt **Mode** des Systems.

<!--en:dbce9cfe-->
Für die Spirale können wir eine konkretere Formel angeben. Ihre Matrix ist eine Summe, $\mathbf A=-\tfrac12\mathbf I+\mathbf K$. Wir möchten das Exponential in ein Produkt zerlegen, wie es für Zahlen mit $e^{p+q}=e^pe^q$ gelingt. Für Matrizen gilt die Regel $e^{\mathbf P+\mathbf Q}=e^{\mathbf P}e^{\mathbf Q}$, wenn $\mathbf P\mathbf Q=\mathbf Q\mathbf P$ gilt. Dann lässt sich $(\mathbf P+\mathbf Q)^k$ wie bei Zahlen mit dem binomischen Lehrsatz entwickeln. Die Einheitsmatrix kommutiert mit jeder Matrix, also gilt $e^{\mathbf At}=e^{-t/2}e^{\mathbf Kt}$. Wegen $\mathbf K^2=-\mathbf I$ durchlaufen die Potenzen von $\mathbf K$ zyklisch die Werte $\mathbf I,\mathbf K,-\mathbf I,-\mathbf K$. Die Terme mit geraden und ungeraden Potenzen der Reihe für $e^{\mathbf Kt}$ liefern damit die Reihen von $\cos t$ und $\sin t$:

<!--en:df743f23-->
Der Fluss verkürzt den Ortsvektor um den Faktor $e^{-t/2}$ und dreht ihn um $t$ Radiant, wie wir bereits aus dem Feld selbst abgeleitet haben. Ausgehend von $\mathbf x_0=(2,0)$ erhalten wir bei $t=2$ den Zustand $e^{-1}(2\cos2,\,2\sin2)=(-0.306,\,0.669)$ mit der Länge $2e^{-1}=0.736$. Für diese Matrix $\mathbf A$ sind die Eigenvektoren und die Eigenwerte $-0.5\pm\mathrm i$ komplex, das Produkt $\mathbf Ve^{\boldsymbol\Lambda t}\mathbf V^{-1}$ ist jedoch reell.

<!--en:bea92a39-->
Wenn die beiden Matrizen nicht kommutieren, gilt die Produktformel im Allgemeinen nicht. Für $\mathbf P=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$ und $\mathbf Q=\mathbf P^\top$, deren Quadrate beide null sind, gilt $e^{\mathbf P}e^{\mathbf Q}=(\mathbf I+\mathbf P)(\mathbf I+\mathbf Q)
=\left(\begin{smallmatrix}2&1\\1&1\end{smallmatrix}\right)$. Dagegen steht bei $e^{\mathbf P+\mathbf Q}$ in beiden Diagonaleinträgen $\cosh1\approx1.54$.

<!--en:0efa22f7-->
Ein dritter Weg zu $e^{\mathbf At}$ kommt ohne Reihe und ohne Eigenvektoren aus: der aus der Zinseszinsrechnung bekannte Grenzwert von $(\mathbf I+t\mathbf A/n)^n$ für $n\to\infty$. Jeder Faktor bewegt den Zustand für die kurze Zeit $t/n$ entlang des Feldes. Wie schnell nähert sich das Produkt dem Grenzwert? In der Eigenbasis ergibt sich für das Produkt $\mathbf V\,\mathrm{diag}\bigl((1+\lambda_jt/n)^n\bigr)\mathbf V^{-1}$. Es genügt daher, einen einzelnen Eigenwert zu betrachten. Wegen $\log(1+z)=z-z^2/2+\cdots$ gilt

<!--en:cfd8262b-->
Somit strebt jeder Ausdruck $(1+\lambda_jt/n)^n$ gegen $e^{\lambda_jt}$. Die Abweichung beträgt in führender Ordnung den Anteil $|\lambda_jt|^2/(2n)$ am Betrag des Grenzwerts. Für die Spirale bei $t=2$ beträgt dieser Anteil für beide Eigenwerte $2.5/n$. Mit $|e^{\lambda t}|=e^{-1}$ erwarten wir also einen Fehler von etwa $0.9/n$. Die Codezelle berechnet $e^{\mathbf At}$ aus der Eigenwertzerlegung und vergleicht das Ergebnis für drei Werte von $n$ mit der Zinseszinsnäherung.

<!--en:8fe34dd5-->
Der Zustand $(-0.3062,\,0.6690)$ und seine Länge $0.735759$ stimmen mit den von Hand berechneten Werten überein. Die Zinseszinsnäherung konvergiert genau mit der vorhergesagten Rate: Der betragsgrößte Fehlereintrag, multipliziert mit $n$, bleibt bei $0.90$, von $n=2^{10}$ bis $n=2^{24}$. Hier rechnen wir mit Zinseszinsen, deren Zinssatz eine Matrix ist. So lässt sich das Matrixexponential gut definieren, aber nur ineffizient berechnen. Nach fast siebzehn Millionen Zinsperioden beträgt der Fehler noch $5\times10^{-8}$.

<!--en:306500e0-->
### Stabilität anhand der Eigenwerte

<!--en:0aca417e-->
An der entkoppelten Form erkennen wir auf einen Blick, ob ein lineares System zur Ruhe kommt. Schreiben wir jeden Eigenwert als $\lambda_j = a_j + \mathrm i\, b_j$. Der Betrag des zugehörigen Exponentialfaktors ist $|e^{\lambda_j t}| = e^{a_j t}$, denn $e^{\mathrm i b_jt}=\cos b_jt+\mathrm i\sin b_jt$ hat den Betrag eins. Der Realteil bestimmt Wachstum oder Abklingen, der Imaginärteil die Rotation. Für eine diagonalisierbare Matrix $\mathbf A$ ergibt sich folgende Klassifikation.

<!--en:d3a7b15d-->
* Gilt $\operatorname{Re}\lambda_j < 0$ für alle $j$, so klingen alle Moden ab. Von jedem Anfangspunkt aus gilt $\mathbf{x}(t) \to \mathbf{0}$. Der Ursprung ist **asymptotisch stabil**. Die langsamste Mode bestimmt die Abklingrate $-\max_j \operatorname{Re}\lambda_j$.

<!--en:99178dfa-->
* Gilt $\operatorname{Re}\lambda_j > 0$ für mindestens ein $j$, so wächst diese Mode exponentiell. Jeder Anfangszustand mit einer Komponente entlang $\mathbf v_j$ entfernt sich unbegrenzt vom Ursprung. Der Ursprung ist **instabil**.

<!--en:97cb202f-->
* Dazwischen liegt der Fall einer Mode mit $\operatorname{Re}\lambda_j=0$. Sie wächst weder an noch klingt sie ab. Sie steht still oder kreist unbegrenzt wie bei der Rotation.

<!--en:a85812f1-->
Ein komplexes Paar $a\pm\mathrm ib$ ergibt die Spirale mit allgemeinen Raten: eine Rotation mit Winkelgeschwindigkeit $b$, die je nach Vorzeichen von $a$ abklingt oder anwächst. Schreiben wir dazu den Eigenvektor zu $\lambda=a+\mathrm ib$ als $\mathbf v=\mathbf p+\mathrm i\mathbf q$ mit reellen Vektoren $\mathbf p,\mathbf q$ (linear unabhängig, da $b\ne0$). Real- und Imaginärteil von $\mathbf A\mathbf v=(a+\mathrm ib)\mathbf v$ ergeben

<!--en:577c02aa-->
Somit bildet $\mathbf A$ die von $\mathbf p$ und $\mathbf q$ aufgespannte Ebene auf sich selbst ab. In der Basis $(\mathbf q,\mathbf p)$ hat die Abbildung die Matrix $a\mathbf I+b\mathbf K$. Die Rechnung für die Spirale liefert dann

<!--en:bf71c9cc-->
also eine Skalierung um $e^{at}$, verknüpft mit einer Drehung um den Winkel $bt$. Derselbe Block tritt in :numref:`subsec_mdl-complex-rotation` auf, dort als Iteration in diskreter Zeit statt als Fluss.

<!--en:89cabb74-->
In zwei Dimensionen bleiben sechs Phasenporträts. Für jeden Typ zeigt :numref:`fig_mdl-dyn-phase-portraits` ein Feld. Bei reellen Eigenwerten ist der Ursprung ein **stabiler Knoten** (beide negativ), ein **instabiler Knoten** (beide positiv) oder ein **Sattelpunkt** (verschiedene Vorzeichen). Bei einem komplexen Paar ist er ein **stabiler Spiralpunkt**, ein **instabiler Spiralpunkt** oder, bei Realteil null, ein **Zentrum**. (Den Eigenwert null lassen wir aus. Er liefert eine ganze Gerade von Ruhelagen.) Der stabile Spiralpunkt ist unser durchgängiges Beispiel.

<!--en:e6d676d3-->
![Sechs ebene lineare Systeme. Obere Reihe, reelle Eigenwerte: stabiler Knoten (beide negativ), instabiler Knoten (beide positiv) und Sattelpunkt (verschiedene Vorzeichen). Entlang der gestrichelten Eigenrichtungen laufen Trajektorien ein oder aus. Untere Reihe, komplexes Paar $a\pm\mathrm{i}b$: stabiler Spiralpunkt ($a<0$), instabiler Spiralpunkt ($a>0$) und Zentrum ($a=0$, geschlossene Bahnen).](../img/mdl-dyn-phase-portraits.svg)

<!--en:7f1b4832-->
Beim Training begegnet uns am häufigsten die obere Reihe. Für den quadratischen Verlust $\tfrac12\mathbf x^\top\mathbf H\mathbf x$ ist der Gradientenabstieg in kontinuierlicher Zeit das lineare Feld $\dot{\mathbf x}=-\mathbf H\mathbf x$. Eine symmetrische Matrix $\mathbf H$ hat reelle Eigenwerte, sodass ein solcher Verlust nur Knoten und Sattelpunkte erzeugt. Rotation setzt ein Feld voraus, das kein Gradientenfeld ist. Ein Beispiel liefern die beiden Spieler des kleinsten GANs: Sie kreisen um ihr Gleichgewicht, ein Zentrum (:numref:`sec_gan_convergence`).

<!--en:cf9a38ab-->
Hat $\mathbf A$ zu wenige Eigenvektoren für eine Diagonalisierung, kommen bei den Moden Polynomfaktoren hinzu, wie in $t^ke^{\lambda t}$. Sie ändern die Stabilitätsaussage nur für Eigenwerte auf der imaginären Achse. Den kleinsten Fall untersucht :exref:`ex_mdl-odes-solvers-repeated-eigenvalue`.

<!--en:abbd8d90-->
### Lineare Systeme mit Eingabe

<!--en:5d616bcb-->
Bisher wirkte nichts von außen auf das System ein. Der Zustand eines Zustandsraummodells, das eine Sequenz liest (:numref:`sec_ssm`), erfüllt eine lineare Gleichung mit einer Eingabe $\mathbf u(t)$. Diese wirkt über eine Matrix $\mathbf B$ auf den Zustand:

<!--en:f5cf7235-->
Das Matrixexponential löst auch diese Gleichung. Eine Beobachtung führt uns zur Lösung. Ohne Eingabe bleibt die Größe $e^{-\mathbf At}\mathbf x(t)$ konstant bei $\mathbf x_0$. Sie entspricht dem Zustand, wenn wir der freien Bewegung bis zur Zeit $0$ zurück folgen. Mit Eingabe ändert sich diese Größe nur durch die Einwirkung der Eingabe. Wegen $\tfrac{d}{dt}e^{-\mathbf At}=-e^{-\mathbf At}\mathbf A$ ergibt die Produktregel tatsächlich $\tfrac{d}{dt}\bigl(e^{-\mathbf At}\mathbf x\bigr)
=e^{-\mathbf At}(\dot{\mathbf x}-\mathbf A\mathbf x)
=e^{-\mathbf At}\mathbf B\mathbf u$, und die rechte Seite enthält kein $\mathbf x$ mehr. Integrieren wir von $0$ bis $t$ und multiplizieren mit $e^{\mathbf At}$, erhalten wir

<!--en:31f4da8a-->
Das ist die Formel der **Variation der Konstanten** (engl. variation of constants). Sie heißt so, weil wir in der freien Lösung die Konstante $\mathbf x_0$ aus $e^{\mathbf At}\mathbf x_0$ mit der Zeit variieren lassen. Der Zustand setzt sich aus dem abgeklungenen Anfangswert und den Beiträgen aller bisherigen Eingaben zusammen. Jeder Beitrag ist seit seinem Eintreffen entsprechend abgeklungen.

<!--en:cfb77810-->
Ein Sequenzmodell erhält pro Schritt einen Abtastwert der Eingabe. Zwischenwerte stehen ihm nicht zur Verfügung. Teilen wir die Zeit in Schritte der Länge $h$ mit Gitterzeitpunkten $t_n=nh$ und Zuständen $\mathbf x_n=\mathbf x(t_n)$. Nehmen wir an, dass die Eingabe während jedes Schritts konstant bleibt, also $\mathbf u(s)=\mathbf u_n$ für $s\in[t_n,t_n+h)$. Das entspricht einem *Halteglied nullter Ordnung* (engl. zero-order hold, ZOH). Wenden wir :eqref:`eq_mdl-ode-variation-of-constants` auf einen Schritt an, der im Zustand $\mathbf x_n$ zur Zeit $t_n$ beginnt. Wir können die Konstante $\mathbf u_n$ vor das Integral ziehen. Mit $\tau=t_{n+1}-s$ bleibt $\int_0^he^{\mathbf A\tau}d\tau$ übrig, für jeden Schritt dieselbe Matrix. Damit erfüllen die Zustände auf dem Gitter ohne jede Näherung die lineare Rekurrenz

<!--en:c336ec23-->
wobei die letzte Form voraussetzt, dass $\mathbf A$ invertierbar ist. In einer Dimension erhalten wir $\bar a=e^{h\lambda}$ und $\bar b=(e^{h\lambda}-1)\,b/\lambda$. Auf diesen Koeffizienten beruhen die Zustandsraummodelle aus :numref:`subsec_zoh` (dort bezeichnet $\Delta$ die Schrittweite und $h$ den Zustand).

<!--en:52c33c5b-->
Entscheidend ist, dass $|e^{h\lambda}|=e^{h\operatorname{Re}\lambda}<1$ für jede Schrittweite $h>0$ gilt, sobald $\operatorname{Re}\lambda<0$ ist. Die exakte Rekurrenz ist also stabil, wenn die Gleichung stabil ist. Vergleichen wir damit den Faktor $1+h\lambda$ einer Zinsperiode. Damit folgen wir dem Feld für eine Zeitspanne $h$, ohne die Richtung erneut zu prüfen. Für $\dot x=-50x$ mit $h=0.05$ ist der exakte Faktor $e^{-2.5}=0.08$. Dagegen kehrt $1+h\lambda=-1.5$ das Vorzeichen des Zustands um und vergrößert seinen Betrag. Solche Schritte entlang des Feldes kennzeichnen das Euler-Verfahren. In :numref:`sec_mdl-euler-runge-kutta` sehen wir, wie sich der Faktor $-1.5$ auf einen Trainingslauf auswirkt.

<!--en:099e5971-->
Die Rekurrenz ist nur dann exakt, wenn die Eingabe während jedes Schritts konstant bleibt. Ein Sequenzmodell liest jedoch Abtastwerte eines Signals, das sich zwischen den Abtastzeitpunkten ändert. Welchen Fehler verursacht das Halten dann? Die Codezelle wendet die skalare Rekurrenz auf $\dot x=\lambda x+\sin t$ von $x(0)=1$ bis $T=2$ an. Dabei hält sie die Eingabe während jedes Schritts auf ihrem Anfangswert und vergleicht das Ergebnis mit der geschlossenen Lösung. Sie betrachtet eine langsame Rate $\lambda=-1$ und eine schnelle Rate $\lambda=-50$ und gibt $1+h\lambda$ neben dem exakten Faktor $e^{h\lambda}$ aus.

<!--en:7c3dea74-->
Die Rekurrenz selbst ist exakt. Der einzige Fehler entsteht dadurch, dass das Halteglied den Sinus während jedes Schritts als konstant behandelt. Der Fehler schrumpft ungefähr proportional zur Schrittweite. Für $\lambda=-1$ beträgt er $0.068$, $0.0097$ und $0.0018$ bei den Schrittweiten $h=0.5$, $0.1$ und $0.02$. Betrachten wir die Zeile für $\lambda=-50$ mit $h=0.5$. Dieser Schritt dauert fünfundzwanzigmal so lange wie die Zeit $1/|\lambda|$, in der der Zustand sein Gedächtnis verliert. Der exakte Faktor beträgt hier $1.4\times10^{-11}$ und der Fehler $0.0016$. Mit dem Faktor $1+h\lambda=-24$ wäre der Zustand hingegen im Betrag explodiert. Darauf beruhen die Modelle aus :numref:`subsec_zoh`: Solange die Raten negativ bleiben, kann keine beim Training gewählte Schrittweite die Rekurrenz zum Anwachsen bringen.

<!--en:dd19c715-->
### Transientes Wachstum

<!--en:30c707b9-->
Die Stabilitätsaussage anhand der Eigenwerte bedarf noch einer Einschränkung. Wenn jede Mode abklingt, strebt der Zustand gegen null. Seine Länge $\|\mathbf x(t)\|$ muss auf dem Weg dorthin aber nicht monoton abnehmen. Das zeigt eine Dreiecksmatrix:

<!--en:9bc15eb9-->
mit den beiden negativen Eigenwerten $-1$ und $-2$. In der zweiten Zeile tritt nur $x_2$ auf, also gilt $x_2(t)=e^{-2t}$. Die erste Zeile lautet dann $\dot x_1=-x_1+10e^{-2t}$, eine Abklinggleichung mit Eingabe. Aus :eqref:`eq_mdl-ode-variation-of-constants` folgt

<!--en:e086a4b3-->
Die erste Koordinate beginnt bei $0$ und steigt, bis ihre Ableitung $10(2e^{-2t}-e^{-t})$ verschwindet, also bei $e^{-t}=\tfrac12$. Zu diesem Zeitpunkt $t=\log2\approx0.69$ hat sie den Wert $10(\tfrac12-\tfrac14)=2.5$, und es gilt $x_2=\tfrac14$. Der Zustand hatte anfangs die Länge $1$ und hat jetzt die Länge $\sqrt{2.5^2+0.25^2}\approx2.51$. Von da an klingt $x_1$ ab, auf lange Sicht wie $10e^{-t}$, also mit der Rate der langsameren Mode (:numref:`fig_mdl-dyn-nonnormal-transient`).

<!--en:d21267ac-->
![Die Länge $\|\mathbf x(t)\|$ des Zustands im Dreieckssystem mit Anfangszustand $\mathbf x_0=(0,1)$ auf einer logarithmischen Achse. Sie steigt auf $2.51$ bei $t=\log 2$ und klingt danach wie $10e^{-t}$ ab. Gestrichelt ist $\|e^{\boldsymbol\Lambda t}\|_2=e^{-t}$ für $\boldsymbol\Lambda=\mathrm{diag}(-1,-2)$ eingezeichnet, die größte Länge, die ein Einheitsvektor als Anfangszustand zur Zeit $t$ unter der Diagonalmatrix mit denselben Eigenwerten erreichen kann. Am Ende haben beide Kurven dieselbe Steigung. Nur diese legen die Eigenwerte fest.](../img/mdl-dyn-nonnormal-transient.svg)

<!--en:c485631a-->
Der Grund ist derselbe wie in :numref:`subsec_mdl-transient-growth`, wo wir den zeitdiskreten Fall ausführlich behandeln. Die Eigenvektoren $\mathbf v_1=(1,0)$ und $\mathbf v_2=(0.995,\,-0.0995)$ spannen Geraden auf, die nur einen Winkel von $5.7^\circ$ einschließen. Der kurze Vektor $\mathbf x_0=10.0\,\mathbf v_1-10.05\,\mathbf v_2$ ist also die Differenz zweier langer Vektoren. Sie klingen mit unterschiedlichen Raten ab. Daher heben sie sich schon nicht mehr gegenseitig auf, bevor beide abgeklungen sind. Die Eigenwerte bestimmen die asymptotische Abklingrate, lassen aber die Stärke des transienten Wachstums offen. Das ist beim Lesen einer Trainingskurve wichtig: Auch bei einem konvergenten Lauf mit Momentum kann die Norm in den ersten Schritten wachsen (:exref:`ex_mdl-odes-solvers-transient-growth`).

<!--en:ae06ad0c-->
### Linearisierung an Fixpunkten

<!--en:4fd8dcc1-->
Nur wenige nichtlineare Felder haben eine geschlossene Lösung. In der Nähe eines Punktes, an dem das Feld verschwindet, verhalten sie sich jedoch oft wie ein lineares Feld. Ein **Fixpunkt** (engl. fixed point) eines autonomen Feldes ist ein Zustand $\mathbf x^\star$ mit $\mathbf f(\mathbf x^\star)=\mathbf 0$. Ein Teilchen, das wir dort absetzen, bewegt sich nie. Für ein stetig differenzierbares Feld $\mathbf f$ und eine kleine Auslenkung $\boldsymbol\delta(t)=\mathbf x(t)-\mathbf x^\star$ liefert der Satz von Taylor (:numref:`sec_mdl-multivariable_calculus`)

<!--en:a98974d9-->
In der Nähe des Fixpunkts stimmt das Feld bis auf einen Rest mit dem linearen Feld seiner Jacobi-Matrix $\mathbf J$ überein. Der Rest geht schneller gegen null als die Auslenkung. Dürfen wir ihn vernachlässigen? In den meisten Fällen ja. Ein Fixpunkt heißt **hyperbolisch**, wenn kein Eigenwert von $\mathbf J$ den Realteil null hat. Für einen hyperbolischen Fixpunkt gilt der Satz von Hartman–Grobman :cite:`Hartman.1960,Grobman.1959`, den wir ohne Beweis anführen. In der Nähe von $\mathbf x^\star$ stimmen die Trajektorien des nichtlinearen Feldes bis auf einen stetigen Koordinatenwechsel mit denen von $\dot{\boldsymbol\delta}=\mathbf J\boldsymbol\delta$ überein. Die Eigenwerte der Jacobi-Matrix bestimmen dann die lokale Stabilität nach der obigen Klassifikation.

<!--en:0dff507b-->
Für den Gradientenabstieg in kontinuierlicher Zeit, $\dot{\mathbf x}=-\nabla\mathcal L(\mathbf x)$, ist das eine vertraute Aussage in neuer Form. Die Fixpunkte sind die kritischen Punkte der Verlustfunktion, und die Jacobi-Matrix ist $-\mathbf H$, die negative Hesse-Matrix. Ein Minimum mit positiver Krümmung in jeder Richtung ist daher ein stabiler Knoten. Ein Sattelpunkt der Verlustfunktion ist auch ein Sattelpunkt des Flusses.

<!--en:e3d2063f-->
Als nichtlineares Beispiel rechnen wir das gedämpfte Pendel durch. Wählen wir die Zeiteinheit so, dass das Verhältnis $g/\ell$ von Fallbeschleunigung zu Pendellänge eins beträgt. Dann erfüllt der Winkel $\phi$ gegenüber der unteren Ruhelage die Gleichung $\ddot\phi=-\sin\phi-c\dot\phi$ mit Dämpfungskoeffizient $c\ge0$. Mit der Winkelgeschwindigkeit $\omega=\dot\phi$ erhalten wir ein System erster Ordnung. Daneben steht seine Jacobi-Matrix:

<!--en:24a31142-->
An den Fixpunkten gilt $\omega=0$ und $\sin\phi=0$. Das Pendel hängt bei $\phi=0$ senkrecht nach unten oder steht bei $\phi=\pi$ senkrecht nach oben im Gleichgewicht.

<!--en:60785559-->
* In der unteren Ruhelage hat $\mathbf J$ das charakteristische Polynom $\lambda^2+c\lambda+1$ mit den Nullstellen $\lambda=-c/2\pm\mathrm i\sqrt{1-c^2/4}$ für $c<2$. Bei $c=0.5$ sind die Eigenwerte $-0.25\pm0.97\,\mathrm i$. Es liegt ein stabiler Spiralpunkt vor. Das entspricht dem Ausschwingen eines losgelassenen Pendels: Es schwingt mit einer Periode von etwa $2\pi/0.97$, während seine Amplitude wie $e^{-t/4}$ abklingt.

<!--en:cc525fe9-->
* In der oberen Ruhelage lautet das Polynom $\lambda^2+c\lambda-1$. Seine Nullstellen $(-c\pm\sqrt{c^2+4})/2$ haben das Produkt $-1$. Für jedes $c$ ist also eine positiv und eine negativ. Bei $c=0.5$ betragen sie $0.78$ und $-1.28$: ein Sattelpunkt. Nur Anfangszustände auf einer einzigen Kurve nähern sich der oberen Ruhelage. Sie entsprechen einem Pendel mit genau der richtigen Geschwindigkeit, um oben zum Stillstand zu kommen und wie ein Bleistift auf seiner Spitze zu balancieren.

<!--en:366728f2-->
Mit derselben Rechnung, also über die Jacobi-Matrix und ihre Eigenwerte, lassen sich die Fixpunkte jedes glatten Feldes klassifizieren, mit einer Ausnahme. Hat kein Eigenwert einen positiven Realteil, liegt aber einer auf der imaginären Achse, entscheidet die Linearisierung nicht über die Stabilität. Die Felder $\dot x=-x^3$ und $\dot x=x^3$ haben am Ursprung beide die Jacobi-Matrix $0$. Beim ersten ist der Ursprung stabil, mit $x(t)=x_0/\sqrt{1+2x_0^2t}\to0$. Beim zweiten geht die Lösung für einen Anfangswert ungleich null bei $t=1/(2x_0^2)$ gegen unendlich.

<!--en:c4e36fcf-->
Bisher konnten wir die beiden Fälle nur unterscheiden, weil wir die Gleichungen lösen konnten. Lyapunovs Idee von 1892 entscheidet solche Fälle ohne explizite Lösung. Wir suchen eine Größe, die im Laufe der Bewegung nur abnehmen kann. Eine **Lyapunov-Funktion** für einen Fixpunkt $\mathbf x^\star$ ist eine stetig differenzierbare Funktion $V$ des Zustands mit $V(\mathbf x^\star)=0$ und $V(\mathbf x)>0$ in allen anderen Punkten einer Umgebung. Ihr Wert steigt entlang keiner Trajektorie:

<!--en:876dacfc-->
Der mittlere Ausdruck folgt aus der Kettenregel. Wir brauchen dafür nur das Feld zu kennen, keine Trajektorie. Beginnt ein Zustand dort, wo $V$ klein ist, kann er nie einen Ort mit größerem $V$ erreichen. Er bleibt also nahe bei $\mathbf x^\star$, der Fixpunkt ist stabil. Gilt die Ungleichung außerhalb von $\mathbf x^\star$ strikt, so konvergieren die benachbarten Zustände gegen den Fixpunkt. Für die beiden kubischen Felder wählen wir $V(x)=x^2$ mit der Ableitung $2x\dot x$. Für $\dot x=-x^3$ ist diese gleich $-2x^4$, und der Ursprung ist stabil. Für $\dot x=x^3$ ergibt sich $+2x^4$. Nun wächst $x^2$ entlang jeder nichtstationären Trajektorie, und der Ursprung ist instabil.

<!--en:6b03b1f3-->
Tatsächlich haben wir diese Idee bereits genutzt. Bei der Spirale erfüllte das Quadrat der Länge $V=\|\mathbf x\|^2$ die Gleichung $\dot V=-V$. Beim Gradientenabstieg in kontinuierlicher Zeit ist der Verlust abzüglich seines Werts an einem isolierten Minimum in dessen Umgebung eine Lyapunov-Funktion, denn $\tfrac{d}{dt}\mathcal L=\nabla\mathcal L^\top\dot{\mathbf x}=-\|\nabla\mathcal L\|^2$ (:exref:`ex_mdl-odes-solvers-gradient-flow-orbits`). Beim Pendel haben wir die Energie $E=\tfrac12\omega^2+1-\cos\phi$, die Summe aus kinetischer und potenzieller Energie. Sie ist in der unteren Ruhelage null und ändert sich mit der Rate

<!--en:7ad57f3f-->
Die Dämpfung kann dem Pendel nur Energie entziehen. Das gilt für beliebig große Ausschläge, auch weit von der unteren Ruhelage entfernt, wo die Jacobi-Matrix keine Aussage mehr erlaubt. Die Ungleichung ist nicht strikt, denn an jedem Umkehrpunkt gilt $\omega=0$, und die Energie ändert sich in diesem Moment nicht. Dass das gedämpfte Pendel trotzdem zur Ruhe kommt, erfordert ein weiteres Argument, das wir hier auslassen. Für die Jacobi-Matrix gibt es eine Rechenvorschrift. Eine Lyapunov-Funktion müssen wir erraten. Meist versuchen wir es zuerst mit einer Energie oder einer Verlustfunktion.

<!--en:2f244db9-->
Bisher hatten wir stets eine Formel für die Lösung. Für ein durch ein Netz gegebenes Feld fehlt eine solche Formel. Dem Computer bleiben nur einzelne Schritte: den Pfeil ablesen, ein Stück weitergehen, den nächsten Pfeil ablesen. Wie weit weichen solche Schritte von der Trajektorie ab, die sie annähern sollen? Und können sie scheitern, obwohl die Gleichung eine Lösung hat?

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:6323980f-->
Eine gewöhnliche Differentialgleichung gibt an jedem Punkt eine Geschwindigkeit vor. Die erste Frage ist, ob damit die Bewegung feststeht. Ist das Feld global Lipschitz-stetig, lautet die Antwort ja. Durch jeden Punkt verläuft vorwärts wie rückwärts genau eine Trajektorie. Die Flussabbildung ist daher bijektiv, und wir können ein Modell durch ein Feld definieren. Lineare Systeme lösen wir mit $e^{\mathbf At}=\mathbf Ve^{\boldsymbol\Lambda t}\mathbf V^{-1}$. Die Realteile der Eigenwerte bestimmen das Abklingen, die Imaginärteile die Rotation. Halten wir die Eingabe während jedes Schritts konstant, erhalten wir eine exakte Rekurrenz. Ist die Gleichung stabil, bleibt die Rekurrenz bei jeder Schrittweite stabil. Die Eigenwerte bestimmen die langfristige Abklingrate, lassen aber die Stärke des transienten Wachstums offen. An einem hyperbolischen Fixpunkt klassifizieren die Eigenwerte der Jacobi-Matrix das Verhalten eines nichtlinearen Feldes. Wo die Jacobi-Matrix keine Entscheidung erlaubt oder die Auslenkung groß ist, kann eine Lyapunov-Funktion die Stabilität nachweisen. Das ist eine Größe wie eine Energie oder ein Verlust, die entlang keiner Trajektorie wächst. Wie wir eine Trajektorie berechnen, wenn keine Formel vorliegt, bleibt offen.

<!--en:2294243a-->
## Übungen

<!--en:9dc8c524-->
1. **Ein Exponent, zwei Gründe für das Scheitern.** Betrachten Sie $\dot x=|x|^\alpha$ für $\alpha>0$.
    1. Bestimmen Sie für $0<\alpha<1$ alle Lösungen mit $x(0)=0$. Prüfen Sie, dass sich für $\alpha=\tfrac12$ der Fächer aus :numref:`fig_mdl-dyn-uniqueness-fan` ergibt. Welche Voraussetzung des Satzes von Picard–Lindelöf ist verletzt, und an welchem Punkt?
    1. Für $\alpha=1$ gilt der Satz. Wie groß ist $L$, und wie lautet die Lösung mit $x(0)=0$?
    1. Lösen Sie die Gleichung für $\alpha>1$ und $x_0>0$ wie im Text die Gleichung $\dot x=x^2$ (differenzieren Sie $x^{1-\alpha}$). Bestimmen Sie den Zeitpunkt, zu dem die Lösung gegen unendlich geht. Welche Voraussetzung ist jetzt verletzt? Zeigen Sie, dass die Lösung mit $x(0)=0$ dennoch eindeutig ist. (Zwei solche Lösungen bleiben für kurze Zeit beide in $|x|\le1$. Dort ist das Feld Lipschitz-stetig mit der Konstante $\alpha$. Wenden Sie das Eindeutigkeitsargument aus dem Beweis auf ein kurzes Intervall an und setzen Sie am Endpunkt erneut an.)
    1. Kehren Sie für $\alpha=\tfrac12$ die Zeit um. Die Gleichung $\dot x=-\sqrt x$ für $x\ge0$ beschreibt in geeigneten Einheiten den Wasserstand in einem Eimer, aus dem durch ein Loch im Boden Wasser abfließt (Ausflussgesetz von Torricelli). Zeigen Sie, dass jede Lösung in endlicher Zeit $0$ erreicht und dort bleibt. Einem leeren Eimer sehen Sie nicht an, wann er leer wurde. Welche Aussage des Korollars über die Flussabbildung gilt hier nicht?
   :exlabel:`ex_mdl-odes-solvers-uniqueness-exponent`

<!--en:c275d724-->
1. **Abstand zwischen Trajektorien.** Betrachten Sie ein Feld der Form $\mathbf{f}(\mathbf{x}) = \mathbf W_2 \tanh(\mathbf W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$, wie es eine Neural ODE integriert (:numref:`sec_mdl-neural-odes-cnf`).
    1. Zeigen Sie, dass $\mathbf{f}$ bezüglich der Spektralnorm Lipschitz-stetig mit der Konstante $L = \|\mathbf W_2\|\,\|\mathbf W_1\|$ ist. Welche Eigenschaft von $\tanh$ nutzt der Beweis? Ändert sich die Schranke, wenn Sie $\tanh$ durch ReLU ersetzen?
    1. Seien $\mathbf{x}(t)$ und $\mathbf{y}(t)$ zwei Trajektorien eines autonomen Feldes mit Lipschitz-Konstante $L$. Differenzieren Sie $\|\mathbf{x}(t) - \mathbf{y}(t)\|^2$ und zeigen Sie, dass folgende Abschätzung gilt:

       $$
       e^{-Lt}\|\mathbf{x}(0) - \mathbf{y}(0)\|
       \le \|\mathbf{x}(t) - \mathbf{y}(t)\|
       \le e^{Lt}\|\mathbf{x}(0) - \mathbf{y}(0)\| .
       $$

       Was folgt aus der unteren Schranke für zwei Trajektorien mit verschiedenen Anfangspunkten? Was folgt aus den beiden Schranken für die Lipschitz-Konstanten von $\Phi_T$ und seiner Umkehrabbildung?
   :exlabel:`ex_mdl-odes-solvers-trajectory-separation`

<!--en:7a64e5f0-->
1. **Ein mehrfacher Eigenwert.** Die Matrix $\mathbf A=\left(\begin{smallmatrix}\lambda&1\\0&\lambda\end{smallmatrix}\right)$ mit reellem $\lambda$ hat einen einzigen Eigenwert und nur eine Eigenrichtung. Daher ist :eqref:`eq_mdl-ode-matrix-exp-eig` nicht anwendbar.
    1. Schreiben Sie $\mathbf A=\lambda\mathbf I+\mathbf N$ und zeigen Sie mithilfe von $\mathbf N^2=\mathbf 0$, dass $e^{\mathbf At}=e^{\lambda t}\left(\begin{smallmatrix}1&t\\0&1\end{smallmatrix}\right)$ gilt. Warum dürfen Sie das Exponential hier in ein Produkt zerlegen?
    1. Für den Anfangszustand $\mathbf x_0=(0,1)$ lautet die erste Koordinate $te^{\lambda t}$. Bestimmen Sie für $\lambda<0$ ihr Maximum und den Zeitpunkt, zu dem sie es erreicht. Für welche $\lambda$ ändert der Faktor $t$ die Stabilitätsaussage, die allein aus dem Vorzeichen von $\lambda$ folgen würde?
    1. Lassen Sie die beiden Eigenwerte der Matrix aus :numref:`subsec_mdl-ode-transient-growth` aufeinander zugehen. Berechnen Sie für $\left(\begin{smallmatrix}-1&10\\0&-1-\varepsilon\end{smallmatrix}\right)$ und $\mathbf x_0=(0,1)$ die Koordinate $x_1(t)$ und ihren Grenzwert für $\varepsilon\to0$. Was geschieht dabei mit dem Winkel zwischen den beiden Eigenrichtungen? Was sagt das über eine Matrix aus, die sich nicht diagonalisieren lässt?
   :exlabel:`ex_mdl-odes-solvers-repeated-eigenvalue`

<!--en:6c7425ef-->
1. [code] **Das Halteglied nullter Ordnung.** Betrachten Sie die skalare Gleichung $\dot x=\lambda x+b\,u(t)$ mit $\lambda<0$. (Die Übung zu Diskretisierungsregeln in :numref:`sec_ssm` vergleicht das Halteglied mit drei anderen Möglichkeiten, diese Gleichung in eine Rekurrenz zu überführen.)
    1. Wenden Sie :eqref:`eq_mdl-ode-zoh` in einer Dimension an, um $\bar a=e^{h\lambda}$ und $\bar b=(e^{h\lambda}-1)\,b/\lambda$ zu erhalten. Bestimmen Sie den Grenzwert von $\bar b$ für $\lambda\to0$. Zeigen Sie, dass sich Rekurrenz und Differentialgleichung bei konstanter Eingabe auf denselben Wert einpendeln.
    1. Mit den Abtastwerten $u(t_n)$ einer zeitlich veränderlichen Eingabe löst die Rekurrenz die Gleichung für eine treppenförmige Eingabe. Begründen Sie, warum die Treppenfunktion im Mittel der um $h/2$ verzögerten Eingabe entspricht. Folgern Sie, dass der Fehler in erster Ordnung in $h$ zum Zeitpunkt $T$ den Betrag $\tfrac h2\bigl|\int_0^Te^{\lambda(T-s)}\,b\,u'(s)\,ds\bigr|$ hat. Berechnen Sie diesen Ausdruck für $\lambda=-1$, $b=1$, $u=\sin$ und $T=2$. Wie gut stimmt er mit den drei Fehlern überein, die die Codezelle in :numref:`subsec_mdl-ode-driven-linear` für $\lambda=-1$ ausgibt?
    1. Ändern Sie die Codezelle jenes Unterabschnitts so, dass sie die Eingabe in der Mitte jedes Schritts abtastet, also $u(t_n+h/2)$ verwendet. Wie nimmt der Fehler für $\lambda=-1$ jetzt mit $h$ ab, und warum?
   :exlabel:`ex_mdl-odes-solvers-zero-order-hold`

<!--en:af708a52-->
1. [code] **Transientes Wachstum und Nichtnormalität.** ● Im Text haben wir eine Matrix mit transientem Wachstum gesehen. Untersuchen Sie, wie dieses Wachstum vom Winkel zwischen den Eigenvektoren abhängt und wie es beim Momentum-Verfahren erneut auftritt.
    1. Zeigen Sie für $\mathbf A_\nu=\left(\begin{smallmatrix}-1&\nu\\0&-2\end{smallmatrix}\right)$ und $\mathbf x_0=(0,1)$, dass $x_1(t)=\nu(e^{-t}-e^{-2t})$ gilt. Bestimmen Sie das Maximum von $x_1$ und den Zeitpunkt, zu dem es auftritt. Bestimmen Sie numerisch die Schwelle für $\nu$, oberhalb derer $\|\mathbf x(t)\|$ den Anfangswert $\|\mathbf x_0\|$ irgendwann übersteigt.
    1. Berechnen Sie den Winkel zwischen den beiden Eigenvektoren von $\mathbf A_\nu$ als Funktion von $\nu$. Stellen Sie $\mathbf x_0$ in der Eigenbasis dar und erklären Sie das Wachstum anhand der beiden großen Komponenten, die sich anfangs weitgehend aufheben.
    1. Berechnen Sie $\max_t\|e^{\mathbf A_\nu t}\|_2$ für $\nu=1,5,10,20$ numerisch und vergleichen Sie das Ergebnis mit der maximalen Norm der Trajektorie, die in $(0,1)$ startet. Warum kann der erste Wert nie kleiner sein als der zweite?
    1. Heavy-Ball-Momentum auf $\mathcal L(x)=\tfrac\lambda2x^2$ entspricht der Rekurrenz $x_{k+1}=(1+\beta-\eta\lambda)x_k-\beta x_{k-1}$, die Sie in :exref:`ex_mdl-odes-solvers-heavy-ball-ode` in :numref:`sec_mdl-euler-runge-kutta` aus einem gedämpften Oszillator herleiten. Sie ist die lineare Abbildung $(x_{k+1},x_k)=\mathbf M(x_k,x_{k-1})$ mit $\mathbf M=\left(\begin{smallmatrix}1+\beta-\eta\lambda&-\beta\\1&0\end{smallmatrix}\right)$. Zeigen Sie, dass $\mathbf M$ nicht normal ist, das heißt, dass $\mathbf M\mathbf M^\top\ne\mathbf M^\top\mathbf M$ gilt (normale Matrizen besitzen eine orthogonale Eigenbasis und zeigen bei abklingenden Moden kein transientes Wachstum, :numref:`subsec_mdl-transient-growth`). Berechnen Sie für $\beta=0.9$ die Eigenwerte bei $\eta\lambda=0.1$ und bei $\eta\lambda=3$. Prüfen Sie, dass ihr Betrag in beiden Fällen $\sqrt\beta\approx0.95$ ist. Starten Sie beide Läufe aus der Ruhe mit $x_0=x_{-1}=1$. In welchem Lauf wächst $\|(x_{k+1},x_k)\|$, bevor die Norm abnimmt? Über wie viele Schritte wächst sie, welches Maximum erreicht sie, und was geschieht im ersten Schritt mit dem Verlust? Was folgt daraus, wenn Sie die ersten Iterationen eines Laufs mit Momentum als Anzeichen für Divergenz deuten wollen?
   :exlabel:`ex_mdl-odes-solvers-transient-growth`

<!--en:8c220ea4-->
1. **Ein Gradientenfluss kann nicht im Kreis laufen.** Sei $\mathcal L$ zweimal stetig differenzierbar und nach unten beschränkt. Betrachten Sie den Gradientenfluss $\dot{\mathbf x}=-\nabla\mathcal L(\mathbf x)$.
    1. Zeigen Sie, dass $\tfrac{d}{dt}\mathcal L(\mathbf x(t))=-\|\nabla\mathcal L(\mathbf x(t))\|^2$ gilt. Folgern Sie, dass keine Trajektorie zu einem Punkt zurückkehrt, den sie verlassen hat. Ein Gradientenfluss besitzt also weder eine geschlossene Bahn noch ein Zentrum.
    1. Integrieren Sie die Identität von $0$ bis $T$ und zeigen Sie, dass $\min_{t\le T}\|\nabla\mathcal L(\mathbf x(t))\|^2\le(\mathcal L(\mathbf x_0)-\inf\mathcal L)/T$ gilt. Welche Annahmen über die Verlustfunktion haben Sie für diese Schranke nicht gebraucht?
    1. Betrachten Sie das Spiel $\min_{x_1}\max_{x_2}x_1x_2$, bei dem $x_1$ in Abstiegsrichtung und $x_2$ in Aufstiegsrichtung des Spielwerts $x_1x_2$ läuft. Sein Feld $\dot x_1=-x_2$, $\dot x_2=x_1$ beschreibt die Rotation $\mathbf K$. Zeigen Sie, dass dieses Feld für keine Funktion der negative Gradient ist. (Vergleichen Sie $\partial f_1/\partial x_2$ mit $\partial f_2/\partial x_1$.) Was sagt Teil (a) dann über den Unterschied zwischen dem Training mit einer einzigen Verlustfunktion und dem Training zweier Spieler gegeneinander aus?
   :exlabel:`ex_mdl-odes-solvers-gradient-flow-orbits`

<!--en:08d9d6ed-->
1. **Grenzen der Linearisierung.**
    1. Das Feld $\dot x=x^2$ hat am Ursprung die Jacobi-Matrix $0$, genau wie die beiden kubischen Felder im Text. Seine Lösung aus :numref:`sec_mdl-ode-existence-uniqueness` gilt für jeden Anfangswert $x_0$. Bestimmen Sie die Anfangswerte, von denen aus sich der Zustand dem Ursprung nähert. Ist der Ursprung stabil, instabil oder nur von einer Seite stabil? Was sagt der Satz von Hartman–Grobman über die drei Felder aus?
    1. Bestimmen Sie für das gedämpfte Pendel die Dämpfung $c$, bei der die untere Ruhelage vom Spiralpunkt zum Knoten übergeht. Skizzieren Sie das Phasenporträt nahe der unteren Ruhelage für $c=0.5$, $c=2$ und $c=4$. Beschreiben Sie, wie sich ein losgelassenes Pendel im jeweiligen Fall bewegt.
    1. Die Verlustfunktion $\mathcal L(w_1,w_2)=\tfrac12(w_1w_2-1)^2$ gehört zum kleinsten tiefen linearen Netz: Das Produkt zweier Gewichte soll den Zielwert $1$ treffen. Geben Sie den Gradientenfluss $\dot{\mathbf w}=-\nabla\mathcal L(\mathbf w)$ an, bestimmen Sie alle Fixpunkte und klassifizieren Sie den Ursprung. Zeigen Sie, dass die Jacobi-Matrix an jedem Punkt der Kurve $w_1w_2=1$ die Eigenwerte $0$ und $-(w_1^2+w_2^2)$ hat. Warum kann kein Punkt auf einer Kurve von Fixpunkten hyperbolisch sein, und was verrät der negative Eigenwert dennoch?
   :exlabel:`ex_mdl-odes-solvers-linearization-limits`

<!--en:da2ea984-->
[Diskussionen](https://d2l.discourse.group/t/36837)

<!--en:e7f22f91-->
Gewöhnliche Differentialgleichungen, Phasenporträts und Flüsse<br>**Vektorfelder · Eindeutigkeit nach Picard–Lindelöf · bijektive Flussabbildungen · Matrixexponential · Phasenporträts**

<!--en:4e05de68-->
::: {.slide title="Wann bestimmt ein Geschwindigkeitsfeld die Bewegung eindeutig?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:654a794b-->
- **Global Lipschitz-stetig ($\dot{x} = -x, x_0 = 1$):** Die eindeutige Lösung $x(t) = e^{-t}$ auf dem gesamten Intervall $[0, T]$ definiert den invertierbaren Fluss $\Phi_t(x_0) = e^{-t}x_0$.

<!--en:1979111b-->
- **Nicht Lipschitz-stetig ($\dot{x} = \sqrt{|x|}, x_0 = 0$):** Unendlich viele Lösungen $x_c(t) = (t-c)^2/4$ zweigen von $0$ ab. Eindeutigkeit und Invertierbarkeit gehen verloren.

<!--en:90374138-->
- **Nur lokal Lipschitz-stetig ($\dot{x} = x^2, x_0 > 0$):** Die Lösung ist lokal eindeutig, strebt aber in endlicher Zeit gegen $\infty$, und zwar bei $t^\star = 1/x_0$.

<!--en:09fd35c3-->
::: {.slide title="Elementare Bewegungen: exponentielles Abklingen und Rotation"}

<!--en:b47479c8-->
[Grundbausteine]{.kicker}

<!--en:b2ec6167-->
- **Radiales Abklingen ($\dot{x} = -x$):** Die Geschwindigkeit zeigt nach innen. Ihr Betrag ist proportional zum Abstand, der nach dem Gesetz $|x(t)| = e^{-t}|x_0|$ schrumpft.

<!--en:942c478a-->
- **Orthogonale Rotation ($\dot{\mathbf{x}} = \mathbf{K}\mathbf{x}$ mit $\mathbf{K} = \left(\begin{smallmatrix}0 & -1 \\ 1 & 0\end{smallmatrix}\right)$):** Wegen $\mathbf{x}^\top\mathbf{K}\mathbf{x} = 0$ steht die Geschwindigkeit senkrecht auf dem Ortsvektor. Die Länge bleibt erhalten ($\frac{d}{dt}\|\mathbf{x}\|^2 = 0$), die Winkelgeschwindigkeit beträgt eins.

<!--en:c054243a-->
- **Spirale in der Ebene ($\mathbf{A} = -\tfrac12\mathbf{I} + \mathbf{K}$):** Die Eigenwerte $\lambda = -0.5 \pm \mathrm{i}$ beschreiben ein Abklingen gemäß $e^{-t/2}$ bei einer Winkelgeschwindigkeit von $1\text{ rad/s}$.

<!--en:d33fce0d-->
::: {.slide title="Unser Beispiel: Abklingen und Rotation ergeben eine Spirale"}

<!--en:884f9431-->
[Vektorfelder]{.kicker}

<!--en:49299a46-->
Die halbe Abklingrate und eine Rotation mit Winkelgeschwindigkeit eins ergeben das ebene System $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$, das uns als Beispiel begleitet:

<!--en:da3aab77-->
- **Systemmatrix:** $\mathbf{A} = -\tfrac{1}{2}\mathbf{I} + \mathbf{K} = \left(\begin{smallmatrix}-0.5 & -1 \\ 1 & -0.5\end{smallmatrix}\right)$.

<!--en:9e1a3119-->
- **Änderung der Länge:** Der Rotationsterm $\mathbf{K}$ erhält die Länge, während $-\tfrac{1}{2}\mathbf{I}$ sie verringert:
  $$\frac{d}{dt}\|\mathbf{x}\|^2 = 2\,\mathbf{x}^\top \mathbf{A}\mathbf{x} = -\|\mathbf{x}\|^2$$

<!--en:7fa8a8fd-->
- **Spiralförmige Trajektorie:** Die Norm nimmt gemäß $\|\mathbf{x}(t)\| = e^{-t/2}\|\mathbf{x}_0\|$ ab, während sich der Zustand mit $1$ rad/s auf einer Spirale dem Fixpunkt $\mathbf{x}^\star = \mathbf{0}$ nähert.

<!--en:baba7de1-->
- **Eigenwerte:** $\mathbf{A}$ hat die komplexen Eigenwerte $\lambda = -0.5 \pm \mathrm{i}$ (Realteil $=$ Abklingrate $-\tfrac{1}{2}$, Imaginärteil $=$ Winkelgeschwindigkeit $\pm 1$).

<!--en:65c8ab99-->
::: {.slide title="Das Anfangswertproblem und seine Integralform"}

<!--en:b2ddf5af-->
[Definitionen]{.kicker}

<!--en:b21552be-->
Eine allgemeine gewöhnliche Differentialgleichung erster Ordnung gibt an jedem Ort und zu jedem Zeitpunkt einen Geschwindigkeitsvektor vor. Hinzu kommt der Anfangszustand:

<!--en:8e5d6e64-->
- **Vektorfeld ($\mathbf{f}: \mathbb{R}^d \times \mathbb{R} \to \mathbb{R}^d$):** Ordnet jeder Position einen Geschwindigkeitspfeil $\mathbf{f}(\mathbf{x}, t)$ zu, abhängig vom Ort $\mathbf{x}$ und der Zeit $t$. Das System ist **autonom**, wenn $\mathbf{f}(\mathbf{x})$ von $t$ unabhängig ist (die Pfeile stehen still, während sich die Teilchen bewegen).

<!--en:b783c574-->
- **Trajektorie ($\mathbf{x}(t)$):** Eine differenzierbare Kurve durch $\mathbf{x}_0$, deren Tangentenvektor $\dot{\mathbf{x}}(t)$ zu jedem Zeitpunkt mit dem örtlichen Pfeil $\mathbf{f}(\mathbf{x}(t), t)$ übereinstimmt.

<!--en:fee421ef-->
- **Äquivalente Integralform:** Nach dem Hauptsatz der Differential- und Integralrechnung ergibt die Integration beider Seiten von $0$ bis $t$ den Zustand als Summe aus Anfangswert und integrierter Geschwindigkeit:
  $$\mathbf{x}(t) = \mathbf{x}_0 + \int_0^t \mathbf{f}(\mathbf{x}(s), s)\,ds$$

<!--en:f4144cab-->
::: {.slide title="Wenn die Eindeutigkeit fehlt: Verzweigung ohne Lipschitz-Stetigkeit"}

<!--en:db490683-->
[Gegenbeispiel]{.kicker}

<!--en:2d395cab-->
Ändert sich die Geschwindigkeit in der Nähe eines Punktes zu abrupt, kann eine einzige Anfangsbedingung unendlich viele Trajektorien zulassen:

<!--en:2e5e7f0d-->
- **Gleichung des Fächers:** $\dot{x} = \sqrt{|x|}$ beschreibt ein anfangs ruhendes Teilchen ($x_0 = 0$).

<!--en:5a1fcc94-->
- **Unendlich viele Lösungen:** Außer der konstanten Lösung $x(t) \equiv 0$ gibt es weitere: Das Teilchen kann bei $0$ bis zu einem beliebigen Zeitpunkt $c \ge 0$ warten und sich dann auf einer Parabel bewegen:
  $$x_c(t) = \begin{cases} 0, & t \le c \\ (t - c)^2 / 4, & t > c \end{cases}$$

<!--en:1693e18f-->
- **Ursache:** Für die Ableitung gilt $\frac{d}{dx}\sqrt{x} = \frac{1}{2\sqrt{x}} \to \infty$ bei $x \to 0^+$. Es gibt keine Konstante $L$, für die $|f(x) - f(0)| \le L|x - 0|$ in einer Umgebung des Ursprungs gilt.

<!--en:4f4bfefe-->
::: {.slide title="Picard-Iteration: Konstruktion der Lösung durch Integration"}

<!--en:76c65690-->
[Konstruktiver Existenzbeweis]{.kicker}

<!--en:0417702b-->
Der **Picard-Integraloperator** $P$ bildet jede stetige Bahn $\boldsymbol{\varphi}$ auf eine verbesserte Näherung ab:

<!--en:3501dffe-->
- **Fixpunktbedingung:** $\boldsymbol{\varphi} = P\boldsymbol{\varphi}$ gilt genau dann, wenn $\boldsymbol{\varphi}$ die Differentialgleichung löst.

<!--en:83949e4d-->
- **Kontraktion nach Banach:** Ist $\mathbf{f}$ global $L$-Lipschitz-stetig, so halbiert sich für $\delta = 1/(2L)$ der Abstand der Bahnen in jedem Schritt mindestens. Mit $\lceil 2LT \rceil$ gleich langen Teilintervallen lässt sich jedes Zeitintervall $[0, T]$ überdecken.

<!--en:2cda6d9c-->
::: {.slide title="Der Existenz- und Eindeutigkeitssatz von Picard–Lindelöf"}

<!--en:0fcbf466-->
[Zentraler Satz]{.kicker}

<!--en:f041125c-->
Eine globale Schranke für die Änderung des Vektorfeldes von Punkt zu Punkt macht den Picard-Operator $P$ zu einer strikten Kontraktion:

<!--en:e5fa8317-->
**Satz (Picard–Lindelöf):** Ist $\mathbf{f}(\mathbf{x}, t)$ stetig in $t$ und **global $L$-Lipschitz-stetig** in $\mathbf{x}$, gilt also
$$\|\mathbf{f}(\mathbf{x}, t) - \mathbf{f}(\mathbf{y}, t)\| \le L\,\|\mathbf{x} - \mathbf{y}\| \qquad \text{for all } \mathbf{x}, \mathbf{y} \in \mathbb{R}^d,\; t \in [0, T],$$
so hat $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, t)$ mit $\mathbf{x}(0) = \mathbf{x}_0$ auf $[0, T]$ für jeden Anfangswert $\mathbf{x}_0$ und jedes $T > 0$ **genau eine** Lösung.

<!--en:3667f4ce-->
- **Kontraktion auf kurzen Intervallen:** Für den Abstand $d(\boldsymbol{\varphi}, \boldsymbol{\psi}) = \max_{t \le \delta}\|\boldsymbol{\varphi}(t) - \boldsymbol{\psi}(t)\|$ zwischen zwei Bahnen ergibt die Lipschitz-Schranke unter dem Integral:
  $$d(P\boldsymbol{\varphi}, P\boldsymbol{\psi}) \le \int_0^\delta L\,\|\boldsymbol{\varphi}(s) - \boldsymbol{\psi}(s)\|\,ds \le L\delta\,d(\boldsymbol{\varphi}, \boldsymbol{\psi})$$

<!--en:468633f8-->
- **Globale Fortsetzung:** Mit $\delta = \frac{1}{2L}$ halbiert sich der Abstand bei jedem Picard-Schritt mindestens. Das garantiert Konvergenz und Eindeutigkeit auf $[0, \delta]$. Da $L$ global gilt, ist $\delta$ vom Zustand unabhängig. So überdecken $\lceil 2LT \rceil$ Teilintervalle ganz $[0, T]$.

<!--en:80952ad5-->
::: {.slide title="Lokale Lipschitz-Stetigkeit ohne globale Schranke: Explosion in endlicher Zeit"}

<!--en:49c0f019-->
[Grenzen des Satzes]{.kicker}

<!--en:43a03770-->
Ist $f(x) = x^2$ nur *lokal* Lipschitz-stetig ($L_\rho = 2\rho$ auf $|x| \le \rho$), so geht die hier betrachtete Lösung in endlicher Zeit gegen $+\infty$:

<!--en:036df52c-->
- **Grenze der Existenzaussage:** Mit wachsendem $x(t)$ schrumpft die zulässige Picard-Schrittweite $\delta_\rho = 1/(2L_\rho) = 1/(4\rho)$. Die Teilintervalle können sich zu einer endlichen Zeit summieren und garantieren eine Lösung nur für eine gewisse Dauer. Den Explosionszeitpunkt $t^\star = 1/x_0$ lesen wir aus der Formel ab.

<!--en:f72b3a4b-->
::: {.slide title="Die Flussabbildung ist eine Bijektion des Raumes"}

<!--en:be301acf-->
[Flussabbildungen und Invertierbarkeit]{.kicker}

<!--en:83aadfbc-->
Ein festes global Lipschitz-stetiges Vektorfeld definiert bei variablem Anfangspunkt $\mathbf{x}_0$ die **Flussabbildung** $\Phi_t(\mathbf{x}_0) = \mathbf{x}(t)$:

<!--en:7c5b9f23-->
- **Bijektion und Schnittfreiheit:** Bei Zeitumkehr ($\dot{\mathbf{y}} = -\mathbf{f}(\mathbf{y}, T-s)$) bleibt das Feld global $L$-Lipschitz-stetig. Verschiedene Trajektorien können sich daher in der Raumzeit nie schneiden, und $\Phi_T : \mathbb{R}^d \to \mathbb{R}^d$ ist eine Bijektion.

<!--en:f1777f28-->
::: {.slide title="Lineare Systeme und die Potenzreihe des Matrixexponentials"}

<!--en:9b37e004-->
[Lineare Differentialgleichungen]{.kicker}

<!--en:68508eaa-->
So wie $\dot{x} = ax$ die Lösung $x(t) = e^{at}x_0$ hat, lässt sich das $d$-dimensionale lineare System $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$ durch die Potenzreihe des **Matrixexponentials** lösen:

<!--en:006ccca1-->
- **Konvergenz für jede Matrix:** Wegen der Submultiplikativität der Spektralnorm ($\|\mathbf{A}^k\| \le \|\mathbf{A}\|^k$) beschränkt die konvergente skalare Exponentialreihe $e^{\|\mathbf{A}\||t|}$ die Reihe gliedweise in der Norm.

<!--en:927b1c1b-->
- **Gliedweise Differentiation:** Ableiten der Reihe zeigt, dass $\mathbf{x}(t) = e^{\mathbf{A}t}\mathbf{x}_0$ die nach Picard–Lindelöf eindeutige Lösung ist ($L = \|\mathbf{A}\|_2$):
  $$\frac{d}{dt}e^{\mathbf{A}t} = \sum_{k=1}^\infty \frac{\mathbf{A}^k t^{k-1}}{(k-1)!} = \mathbf{A}\sum_{j=0}^\infty \frac{(\mathbf{A}t)^j}{j!} = \mathbf{A}e^{\mathbf{A}t}$$

<!--en:367e8dea-->
::: {.slide title="Spektrale Zerlegung in unabhängige Eigenmoden"}

<!--en:54cce930-->
[Eigenwertzerlegung]{.kicker}

<!--en:5e432331-->
Bei der Diagonalisierung $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ vereinfachen sich die Potenzen zu $(\mathbf{A}t)^k = \mathbf{V}(\boldsymbol{\Lambda}t)^k\mathbf{V}^{-1}$, und das Matrixexponential zerfällt in unabhängige Beiträge:

<!--en:18568d64-->
- **Überlagerung der Moden:** Die Entwicklung $\mathbf{x}_0 = \sum_{j=1}^d c_j\mathbf{v}_j$ zerlegt das System in $d$ unabhängige skalare Exponentialfunktionen: $\mathbf{x}(t) = \sum_{j=1}^d c_j e^{\lambda_j t}\mathbf{v}_j$.

<!--en:ecdd5b1d-->
::: {.slide title="Geschlossene Form des Matrixexponentials für die ebene Spirale"}

<!--en:0b66c979-->
[Kommutierende Matrizen]{.kicker}

<!--en:5529fb2b-->
Da $-\tfrac12\mathbf{I}$ und $\mathbf{K}$ kommutieren, gilt $e^{(-\frac12\mathbf{I} + \mathbf{K})t} = e^{-t/2\mathbf{I}}e^{\mathbf{K}t}$:

<!--en:c79b4d08-->
- **Zyklische Potenzen von $\mathbf{K}$:** Wegen $\mathbf{K}^2 = -\mathbf{I}$ summieren sich die Terme mit geraden Potenzen zu $\cos t\,\mathbf{I}$ und die mit ungeraden zu $\sin t\,\mathbf{K}$.

<!--en:9e1dfab1-->
- **Vorsicht bei nicht kommutierenden Matrizen:** Gilt $\mathbf{P}\mathbf{Q} \neq \mathbf{Q}\mathbf{P}$, so gilt im Allgemeinen $e^{\mathbf{P}+\mathbf{Q}} \neq e^{\mathbf{P}}e^{\mathbf{Q}}$.

<!--en:8426aec9-->
::: {.slide title="Zinseszins mit Matrizen: Konvergenz nur mit O(1/n)"}

<!--en:1b718c47-->
[Numerische Überprüfung]{.kicker}

<!--en:15235b60-->
Mit $n$ kurzen Schritten der Länge $t/n$ entlang des linearen Feldes ergibt sich die Zinseszinsnäherung $(\mathbf{I} + t\mathbf{A}/n)^n \approx e^{\mathbf{A}t}$. Ihr Fehler für jeden Eigenwert ergibt sich aus $\log(1 + z) = z - z^2/2 + \cdots$:

<!--en:afbe814b-->
::: {.slide title="Eigenwerte und Phasenporträts in der Ebene"}

<!--en:72437262-->
[Klassifikation der Stabilität]{.kicker}

<!--en:81a2caed-->
Für einen Eigenwert $\lambda_j = a_j + \mathrm{i}\,b_j$ skaliert der Betrag der Mode mit $|e^{\lambda_j t}| = e^{a_j t}$, während sie mit Winkelgeschwindigkeit $b_j$ rotiert:

<!--en:1043917c-->
- **Asymptotische Stabilität ($\max_j a_j < 0$):** Jede Mode klingt gegen $\mathbf{0}$ ab. Der langsamste Eigenwert bestimmt die Abklingrate $-\max_j \operatorname{Re}\lambda_j$ des Systems.

<!--en:57be10ce-->
- **Instabilität ($\max_j a_j > 0$):** Jeder Anfangszustand mit einer Komponente entlang einer wachsenden Eigenrichtung $\mathbf{v}_j$ entfernt sich exponentiell vom Ursprung.

<!--en:53ba49e5-->
- **Gradientenflüsse und Spiele:** Beim Gradientenabstieg in kontinuierlicher Zeit $\dot{\mathbf{x}} = -\mathbf{H}\mathbf{x}$ hat die symmetrische Hesse-Matrix $\mathbf{H}$ nur reelle Eigenwerte (Knoten und Sattelpunkte, obere Reihe). Bei Minimax-Spielen wie GANs können komplexe Eigenwerte auftreten. Die Trajektorien bilden dann geschlossene Bahnen um ein Zentrum oder Spiralen (untere Reihe).

<!--en:a09199eb-->
::: {.slide title="Lineare Systeme mit Anregung und Variation der Konstanten"}

<!--en:6f1aa65e-->
[Zustandsraummodelle]{.kicker}

<!--en:9477a65c-->
In einem Zustandsraummodell (SSM) in kontinuierlicher Zeit wirkt eine externe Eingabesequenz $\mathbf{u}(t)$ über eine Eingabematrix $\mathbf{B}$ auf den versteckten Zustand:

<!--en:98b5fe48-->
- **Integrierender Faktor ($e^{-\mathbf{A}t}$):** Bei der Rückführung des Zustands entlang der freien Bewegung auf $t = 0$ fällt nach der Produktregel der Term $\mathbf{A}\mathbf{x}$ weg:
  $$\frac{d}{dt}\left(e^{-\mathbf{A}t}\mathbf{x}(t)\right) = e^{-\mathbf{A}t}\big(\dot{\mathbf{x}}(t) - \mathbf{A}\mathbf{x}(t)\big) = e^{-\mathbf{A}t}\mathbf{B}\mathbf{u}(t)$$

<!--en:7e4e36fb-->
- **Variation der Konstanten:** Integration von $0$ bis $t$ und Multiplikation mit $e^{\mathbf{A}t}$ ergeben den Zustand als abgeklungenen Anfangswert plus eine Faltung über alle bisherigen Eingaben:
  $$\mathbf{x}(t) = e^{\mathbf{A}t}\mathbf{x}_0 + \int_0^t e^{\mathbf{A}(t-s)}\,\mathbf{B}\mathbf{u}(s)\,ds$$

<!--en:80628228-->
::: {.slide title="Halteglied nullter Ordnung: exakte Diskretisierung für SSMs"}

<!--en:8b924e9f-->
[Exakte Rekurrenz]{.kicker}

<!--en:f4f354e6-->
Bleibt die Eingabe während jedes Schritts konstant ($\mathbf{u}(s) = \mathbf{u}_n$ auf $[t_n, t_n + h)$), ergibt sich die **exakte ZOH-Rekurrenz** für Zustandsraummodelle (S4, Mamba):

<!--en:6d0092e4-->
- **Unbedingte Stabilität:** Für $\operatorname{Re}\lambda < 0$ gilt $|\bar{a}| = e^{h\operatorname{Re}\lambda} < 1$ bei *jeder* Schrittweite $h > 0$. Bei $\lambda = -50, h = 0.05$ gilt $\bar{a} = e^{-2.5} \approx 0.082$, während der Euler-Faktor $1 + h\lambda = -1.5$ den Betrag in jedem Schritt vergrößert.

<!--en:1e1623a0-->
::: {.slide title="Halteglied nullter Ordnung bei schnellen und langsamen Moden im Vergleich"}

<!--en:f473cea7-->
[Numerisches Experiment]{.kicker}

<!--en:7969fa86-->
Bei der Integration von $\dot{x} = \lambda x + \sin t$ bis $T = 2$ mit schrittweise konstanter Eingabe entsteht beim veränderlichen Sinussignal nur ein Haltefehler der Ordnung $O(h)$. Selbst bei $h|\lambda| = 25$ bleibt die Rekurrenz stabil:

<!--en:47f3c515-->
Bei $\lambda = -50$ und $h = 0.5$ dauert der Schritt $25\times$ so lange wie die Relaxationszeit $1/|\lambda|$. Die ZOH-Rekurrenz bleibt dennoch stabil ($e^{h\lambda} = 1.4 \times 10^{-11}$, Fehler $1.6 \times 10^{-3}$), während der Faktor $1 + h\lambda = -24$ zur Divergenz führt.

<!--en:212a1247-->
::: {.slide title="Nichtnormale Matrizen: transientes Wachstum vor dem Abklingen"}

<!--en:c11ee6eb-->
[Transiente Dynamik]{.kicker}

<!--en:fa2e9b03-->
Gilt $\mathbf{A}\mathbf{A}^\top \neq \mathbf{A}^\top\mathbf{A}$, können fast parallele Eigenvektoren ein starkes transientes Wachstum bewirken, bevor das asymptotische Abklingen einsetzt:

<!--en:bc1fe4fc-->
- **Normspitze ($2.5\times$):** Trotz $\lambda_1 = -1, \lambda_2 = -2 < 0$ schließen die Eigenvektoren nur einen Winkel von $5.7^\circ$ ein. Deshalb wächst $\|\mathbf{x}(t)\|$ auf $2.51$ bei $t = \log 2$, bevor die Norm abnimmt. Das erklärt anfängliche Spitzen beim Momentum-Verfahren.

<!--en:06620a31-->
::: {.slide title="Lokale Linearisierung an hyperbolischen Fixpunkten"}

<!--en:8f51ad16-->
[Nichtlineare Systeme]{.kicker}

<!--en:8e2928ec-->
In der Nähe eines Fixpunkts $\mathbf{f}(\mathbf{x}^\star) = \mathbf{0}$ beschreibt die Linearisierung mit der Jacobi-Matrix $\dot{\boldsymbol{\delta}} = \mathbf{J}(\mathbf{x}^\star)\boldsymbol{\delta}$ das Verhalten einer glatten nichtlinearen Differentialgleichung in erster Ordnung:

<!--en:4a1c5a2d-->
- **Satz von Hartman–Grobman:** Ist $\mathbf{x}^\star$ **hyperbolisch** ($\operatorname{Re}\lambda_j \neq 0$ für alle $j$), so ist der lokale nichtlineare Fluss topologisch konjugiert zu $\dot{\boldsymbol{\delta}} = \mathbf{J}\boldsymbol{\delta}$.

<!--en:2dec53fc-->
- **Gedämpftes Pendel ($\ddot{\phi} + 0.5\dot{\phi} + \sin\phi = 0$):** In der unteren Hängelage $(0, 0)$ gilt $\lambda = -0.25 \pm 0.97\mathrm{i}$ (stabiler Spiralpunkt), in der oberen aufrechten Lage $(\pi, 0)$ gilt $\lambda = +0.78, -1.28$ (instabiler Sattelpunkt).

<!--en:2a22073e-->
::: {.slide title="Verständnisfragen: Eindeutigkeit, Spektren und Diskretisierung"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:807395a5-->
Fragen zu Vektorfeldern, Matrixexponentialen und lokaler Stabilität:

<!--en:9936dd28-->
1. **Globale Existenz und Eindeutigkeit:** Welches der skalaren Felder $\dot{x} = \tanh(x)$, $\dot{x} = x^3$ und $\dot{x} = |x|^{1/3}$ garantiert auf $[0, T]$ für jeden Anfangspunkt $x_0 \in \mathbb{R}$ eine eindeutige Trajektorie?

<!--en:4d0972e1-->
2. **Phasenporträts von Gradientenflüssen:** Kann der Gradientenabstieg in kontinuierlicher Zeit $\dot{\mathbf{x}} = -\nabla\mathcal{L}(\mathbf{x})$ für eine zweimal differenzierbare Verlustfunktion $\mathcal{L}$ auf einer Spirale um einen kritischen Punkt $\mathbf{x}^\star$ laufen?

<!--en:c1574285-->
3. **Stabilität eines SSM-Schritts:** Für $\dot{x} = -100x + u(t)$ bei Schrittweite $h = 0.1$ den exakten ZOH-Faktor $\bar{a}$ für den Zustand mit dem einfachen linearen Faktor $1 + h\lambda$ vergleichen.

<!--en:557b98b6-->
**Lösungen:**

<!--en:812de3bd-->
1. Nur $\dot{x} = \tanh(x)$ (global Lipschitz-stetig mit $L = \max \operatorname{sech}^2(x) = 1$). Für das Feld $x^3$ divergiert die Lösung bei $x_0 \ne 0$ in endlicher Zeit. Für $|x|^{1/3}$ ist die Lipschitz-Bedingung für Eindeutigkeit bei $x_0 = 0$ verletzt.

<!--en:6e02cb6f-->
2. **Nein:** An jedem kritischen Punkt ist die Jacobi-Matrix von $-\nabla\mathcal{L}$ die negative symmetrische Hesse-Matrix $-\mathbf{H}(\mathbf{x}^\star)$. Ihre Eigenwerte sind reell ($\mathrm{i}b_j = 0$). Gradientenflüsse bilden Knoten und Sattelpunkte, aber keine Spiralpunkte oder Zentren.

<!--en:484476c9-->
3. Das exakte Halteglied nullter Ordnung ergibt $\bar{a} = e^{-10} \approx 4.5 \times 10^{-5}$ (stabile Kontraktion). Mit $1 + h\lambda = 1 - 10 = -9$ dagegen wächst der Betrag pro Schritt um den Faktor $9\times$.

<!--en:277fad9d-->
::: {.slide title="Zusammenfassung: gewöhnliche Differentialgleichungen und Flüsse"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:a9beea2b-->
| Begriff | Mathematische Form | Bedeutung für Deep Learning und dynamische Systeme |

<!--en:6e7a2e03-->
| **Satz von Picard–Lindelöf** | $\|\mathbf{f}(\mathbf{x},t) - \mathbf{f}(\mathbf{y},t)\| \le L\|\mathbf{x} - \mathbf{y}\|$ | Garantiert auf $[0, T]$ eindeutige Trajektorien ohne Schnittpunkte in der Raumzeit |

<!--en:9944859b-->
| **Bijektive Flussabbildung** | $\Phi_T(\mathbf{x}_0) = \mathbf{x}(T)$, $\Phi_T^{-1} = \Phi_{-T}$ | Macht Neural ODEs und kontinuierliche normalisierende Flüsse invertierbar |

<!--en:3dcc1f3b-->
| **Matrixexponential** | $e^{\mathbf{A}t} = \mathbf{V}\mathrm{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_d t})\mathbf{V}^{-1}$ | Zerlegt die lineare Dynamik in $d$ unabhängige Eigenmoden |

<!--en:5638d654-->
| **Stabilität anhand der Eigenwerte** | $\lambda_j = a_j + \mathrm{i}b_j \implies |e^{\lambda_j t}| = e^{a_j t}$ | Der Realteil $a_j < 0$ bestimmt die Abklingrate, der Imaginärteil $b_j$ die Rotation |

<!--en:3bf046c9-->
| **Halteglied nullter Ordnung (ZOH)** | $\mathbf{x}_{n+1} = e^{\mathbf{A}h}\mathbf{x}_n + \mathbf{A}^{-1}(e^{\mathbf{A}h}-\mathbf{I})\mathbf{B}\mathbf{u}_n$ | Exakte, unbedingt stabile Rekurrenz für stabile Zustandsraummodelle |

<!--en:50dcccbc-->
| **Linearisierung mit der Jacobi-Matrix** | $\dot{\boldsymbol{\delta}} = \mathbf{J}(\mathbf{x}^\star)\boldsymbol{\delta} + o(\|\boldsymbol{\delta}\|)$ | Klassifiziert Fixpunkte (Minima und Sattelpunkte bei Gradientenflüssen) |
