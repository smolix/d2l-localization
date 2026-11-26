<!--en:3ec0ae36-->
# Neural ODEs, adjungierte Variablen und kontinuierliche Flüsse

<!--en:61c098a0-->
Ein Residualblock addiert eine Korrektur zu seiner Eingabe. Dasselbe tut ein Schritt des Euler-Verfahrens (:numref:`sec_mdl-euler-runge-kutta`). Wenn ein Residualnetz (:numref:`sec_resnet`) eine Differentialgleichung numerisch löst, was gewinnen wir dann, wenn wir die Gleichung selbst als Modell verwenden? Und was kostet uns das?

<!--en:280e6a9e-->
Eine Antwort liefert die generative Modellierung. Für das Maximum-Likelihood-Verfahren brauchen wir die Dichte der vom Modell erzeugten Daten. Ein normalisierender Fluss (engl. normalizing flow) ist ein umkehrbares Netz. Seine Ausgabedichte ergibt sich daher aus der Formel für die Variablentransformation. Diese Formel verlangt für jedes Sample die Determinante der Jacobi-Matrix des Netzes. Bei einem Farbbild mit $32\times32$ Pixeln hat die Matrix $3072$ Zeilen. In kontinuierlicher Zeit wird aus der Determinante eine Spur, die wir mit einem einzigen Rückwärtsdurchlauf schätzen können.

<!--en:e28a9e26-->
Eine *Neural ODE* :cite:`Chen.Rubanova.Bettencourt.ea.2018` verwendet die Gleichung selbst als Modell. In der Differentialgleichung $\dot{\mathbf{x}}=\mathbf{f}_{\boldsymbol\theta}(\mathbf{x},t)$ stellt ein Netz das Vektorfeld dar. Die Lösung zum Endzeitpunkt ist die Ausgabe. Sowohl für das Training als auch für die Berechnung der Dichte integrieren wir neben dem Zustand jeweils eine weitere Gleichung. Beim Training ist das die *adjungierte* Gleichung. Sie entstand in den 1950er-Jahren in der Theorie der optimalen Steuerung, um zu bestimmen, wie sich die Kosten einer Trajektorie mit der Steuerung ändern :cite:`Pontryagin.Boltyanskii.Gamkrelidze.ea.1962`. Für die Dichte integrieren wir die Spur der Jacobi-Matrix.

<!--en:7323cb06-->
Gegenüber einer Folge von Residualblöcken gewinnen wir eine umkehrbare Abbildung, eine vom Modell unabhängige Schrittzahl, einen Gradienten, der ohne gespeicherte Aktivierungen auskommt, und eine berechenbare Dichte. Jeder Vorteil hat seinen Preis. Wir untersuchen alle vier Einschränkungen quantitativ. Ein Fluss auf der Geraden kann zwei Punkte nicht vertauschen. Ein trainiertes Feld ist an sein numerisches Verfahren gebunden. Ohne gespeicherte Zustände driftet der berechnete Gradient, und die kostengünstige Spurschätzung rauscht.

<!--en:039fa3d0-->
Wir trainieren ein solches Modell durch die Schritte eines numerischen Verfahrens hindurch, leiten die adjungierte Gleichung her und verfolgen eine Dichte entlang des Flusses. Die wichtigsten Ergebnisse sind die adjungierten Gleichungen und die Spurregel :eqref:`eq_mdl-ode-instant-cov`, auf der die Likelihoods von Diffusionsmodellen beruhen. Wer sich nur für die Spurregel interessiert, kann bei :numref:`sec_mdl-continuous-normalizing-flows` beginnen. Der Unterabschnitt über Speicherbedarf und Drift ist für den weiteren Text entbehrlich.

<!--en:a4c4b28f-->
In den folgenden Zellen trainieren wir durch die Euler- und RK4-Schritte aus :numref:`sec_mdl-euler-runge-kutta` hindurch. Die Schleife `solve` führt $N$ Schritte der Weite $T/N$ aus. Bei negativem $T$ durchlaufen wir das Feld rückwärts. Weiter unten überprüfen wir die adjungierte Gleichung anhand der Spirale $\mathbf A$, ihres Anfangspunkts $\mathbf{x}_0$ und des Matrixexponentials $e^{\mathbf Cs}=\mathbf Ve^{\boldsymbol\Lambda s}\mathbf V^{-1}$ aus :numref:`sec_mdl-odes-solvers`.

<!--en:004b65b2-->
## Neural ODEs

<!--en:94a4b39f-->
Eine Neural ODE entsteht aus einem Residualnetz, wenn dessen Tiefe in Zeit übergeht.

<!--en:206b61be-->
### Von Residualblöcken zum kontinuierlichen Modell

<!--en:fcc08f69-->
Vergleichen wir einen Residualblock :cite:`He.Zhang.Ren.ea.2016` mit einem Vorwärts-Euler-Schritt für die gewöhnliche Differentialgleichung $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},t)$:

<!--en:dc80332b-->
Mit $h=1$ und $\mathbf{f}(\cdot,t_l)=\mathbf{f}_{\boldsymbol\theta_l}$ stimmen beide Aktualisierungsformeln überein. Ein ResNet mit unterschiedlichen Gewichten in seinen Blöcken entspricht also einem zeitabhängigen Feld. Wir werten es zu den Zeitpunkten der einzelnen Schichten aus und integrieren es mit Euler-Schritten der Weite eins. :numref:`fig_mdl-dyn-resnet-as-euler` zeigt den Fall, dass alle Blöcke dasselbe Feld verwenden. Der Zeithorizont entspricht der Tiefe des Netzes. Zehn Blöcke integrieren das Feld bis zur Zeit $10$.

<!--en:e3318bb5-->
![Links: eine Folge von Residualblöcken mit gemeinsamem Feld. Jeder Block addiert den erlernten Zweig zur unveränderten Eingabe aus der Direktverbindung. Rechts: dieselben Aktualisierungen als Vorwärts-Euler-Schritte der Weite $h=1$ neben dem exakten Fluss desselben Feldes, von dem sie zunehmend abweichen.](../img/mdl-dyn-resnet-as-euler.svg)

<!--en:3fa3454f-->
Für den Grenzübergang halten wir den Zeithorizont $T$ fest, verwenden zu allen Zeiten denselben Parametervektor $\boldsymbol\theta$ und verkleinern die Schritte. Das Netz wird also tiefer, während jede Schicht weniger korrigiert. (Das Feld darf weiterhin von $t$ abhängen. Das Netz erhält $t$ als Eingabe, sodass ein einziger Parametervektor unterschiedliche Blöcke beschreiben kann.) Für $N$ Schritte der Weite $h=T/N$ zu den Zeitpunkten $t_n=nh$ gilt

<!--en:8f3c7b5b-->
Die Konvergenz folgt aus der Fehlerschranke für das Euler-Verfahren aus :numref:`sec_mdl-euler-runge-kutta`. Ist das Feld glatt und Lipschitz-stetig in $\mathbf{x}$, so weicht die Euler-Näherung nach $N$ Schritten höchstens um $O(h)$ von der exakten Lösung ab. Im Grenzfall erhalten wir die **Neural ODE**. Ihre Ausgabe ist die Lösung $\mathbf{x}(T)$ von $\dot{\mathbf{x}}=\mathbf{f}_{\boldsymbol\theta}(\mathbf{x},t)$ mit $\mathbf{x}(0)=\mathbf{x}_0$. Die Parameter legen das Feld fest. Erst danach wählen wir das numerische Verfahren (Euler, RK4 oder ein adaptives Verfahren). Die Wahl beeinflusst die berechnete Abbildung, ihre Kosten und ihren Fehler, lässt aber das Modell unverändert.

<!--en:78e1b1f8-->
Die exakte und die berechnete Abbildung unterscheiden sich mitunter nicht nur in der Genauigkeit, sondern auch in ihren qualitativen Eigenschaften. Der exakte Fluss des Feldes $\dot{x}=-x$ ist $\Phi_t(x)=e^{-t}x$, für jedes $t$ eine Bijektion auf $\mathbb{R}$. Ein Euler-Schritt mit $h=1$ berechnet $x+1\cdot(-x)=0$ und bildet jede Eingabe auf null ab. Die exakte Abbildung hingegen ist bijektiv, sobald das Feld stetig in $t$ und global Lipschitz-stetig in $\mathbf{x}$ ist. Vom Endzustand $\mathbf{x}(T)$ aus rückwärts zu integrieren ergibt wieder ein Problem mit genau einer Lösung. Diese führt zum Anfangspunkt zurück (:numref:`sec_mdl-ode-existence-uniqueness`).

<!--en:8d4b2254-->
### Ordnungserhaltung auf der Geraden

<!--en:d41344bf-->
Umkehrbarkeit ist ein Vorteil, schränkt das Modell aber auch ein. Ist eine Neural ODE mindestens so ausdrucksstark wie die Folge von Residualblöcken, aus der sie hervorgeht? Auf der reellen Geraden lautet die Antwort nein. Das sieht man in der $(t,x)$-Ebene. Dort schneiden sich die Trajektorien eines Lipschitz-stetigen Feldes nie. Zwei Trajektorien, die $-1$ und $1$ vertauschen, müssten sich unterwegs treffen (Teilbild (a) in :numref:`fig_mdl-dyn-flow-order`).

<!--en:1e2e2419-->
![(a) Auf der Geraden schneiden sich die Trajektorien eines Lipschitz-stetigen Feldes nie (hier $\dot x=-x$). Daher erhält die Flussabbildung zur Zeit $T$ die Ordnung. Die gestrichelten Trajektorien, die $-1$ und $1$ vertauschen, würden zwei Lösungen durch denselben Punkt erfordern. (b) Der Residualschritt $x+f(x)$ mit $f(x)=-2x$ vertauscht die Punkte in einem Schritt. Es handelt sich um einen Euler-Schritt mit $hL=2$, nicht um einen Fluss. (c) Eine zusätzliche Dimension beseitigt das Hindernis. Eine Drehung um $\pi$ führt $(x,0)$ auf $(-x,0)$. Die Punkte bewegen sich auf Halbkreisen aneinander vorbei, ohne sich zu treffen (Farben wie in (b)).](../img/mdl-dyn-flow-order.svg)

<!--en:44369b70-->
**Proposition (Flüsse auf $\mathbb{R}$ erhalten die Ordnung).** *Sei $f(x,t)$ stetig in $t$ und global Lipschitz-stetig in $x$. Mit $\Phi_t(x)$ bezeichnen wir die Lösung zur Zeit $t$ mit Anfangswert $x$ zur Zeit $0$. Dann ist für jedes $T>0$ die Abbildung $\Phi_T$ streng monoton steigend.*

<!--en:5f08ee53-->
**Beweis.** Sei $x<y$, und nehmen wir an, es gelte $\Phi_T(x)\ge\Phi_T(y)$. Die Funktion $g(t)=\Phi_t(y)-\Phi_t(x)$ ist stetig mit $g(0)>0$ und $g(T)\le0$. Nach dem Zwischenwertsatz gilt also $g(s)=0$ für ein $s\in(0,T]$. Die Trajektorien treffen sich zur Zeit $s$ in einem Punkt $z$. Lassen wir nun die Zeit von $s$ an rückwärts laufen. Beide rückwärts durchlaufenen Trajektorien $\tau\mapsto\Phi_{s-\tau}(x)$ und $\tau\mapsto\Phi_{s-\tau}(y)$ lösen dasselbe Anfangswertproblem $\dot{w}=-f(w,s-\tau)$, $w(0)=z$, dessen Feld wiederum Lipschitz-stetig ist. Nach dem Satz von Picard-Lindelöf hat das Problem genau eine Lösung. Die rückwärts durchlaufenen Trajektorien stimmen also überein. Für $\tau=s$ folgt daraus $x=y$, ein Widerspruch. $\blacksquare$

<!--en:566fb035-->
Die Beweisidee besteht darin, die Eindeutigkeit vom Treffzeitpunkt aus rückwärts anzuwenden. Folglich kann keine Neural ODE auf $\mathbb{R}$ die Abbildung $x\mapsto-x$ darstellen. Diese bildet $-1$ auf $1$ und $1$ auf $-1$ ab und kehrt so die Reihenfolge um.

<!--en:b5c7fc25-->
Der Residualblock $x+f(x)$ mit $f(x)=-2x$ stellt diese Abbildung in einem einzigen Schritt dar (Teilbild (b) in :numref:`fig_mdl-dyn-flow-order`). Dieser Block ist der Euler-Schritt mit $h=1$ für $\dot{x}=-2x$. Der exakte Fluss $e^{-2t}x$ kehrt die Reihenfolge nie um. Die Aussagen sind miteinander vereinbar. Ein Euler-Schritt $x+hf(x)$ für ein Feld mit Lipschitz-Konstante $L$ ist unter der Bedingung $hL<1$ garantiert streng monoton steigend. Für $x>y$ gilt nämlich

<!--en:9e0d45eb-->
und dieser Ausdruck ist für $hL<1$ positiv. Hier gilt jedoch $hL=2$, sodass der Schritt die Reihenfolge umkehren kann. Ein endlicher Schritt ist kein Fluss und unterliegt daher nicht der Aussage der Proposition.

<!--en:7765bd2c-->
Eine weitere Dimension beseitigt das Hindernis. Das Rotationsfeld $\dot{\mathbf{x}}=\mathbf K\mathbf{x}$ aus :numref:`sec_mdl-odes-solvers` mit $\mathbf K=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$ hat den Fluss $e^{s\mathbf K}=\cos s\,\mathbf{I}+\sin s\,\mathbf K$. Dieser dreht die Ebene um den Winkel $s$. Durchlaufen wir das Feld $\pi$-mal so schnell, also gemäß $\dot{\mathbf{x}}=\pi \mathbf K\mathbf{x}$, dann ist die Abbildung zur Zeit eins gegeben durch

<!--en:ebcc7ca2-->
Sie führt $(x,0)$ auf Halbkreisen nach $(-x,0)$. In der Ebene laufen die Trajektorien aneinander vorbei, ohne sich zu treffen (Teilbild (c) in :numref:`fig_mdl-dyn-flow-order`). Eine solche Erweiterung um zusätzliche Dimensionen heißt *Zustandserweiterung* (engl. augmentation) :cite:`Dupont.Doucet.Teh.2019`. Wir betten die Eingabe $x$ als $(x,0)$ ein, folgen dem Fluss in der Ebene und lesen die erste Koordinate aus. Ein Fluss ordnet den Raum stetig um. Auf einer Geraden fehlt der Platz, damit zwei Punkte aneinander vorbeikommen.

<!--en:e1d42208-->
### Training durch die Schritte des numerischen Verfahrens

<!--en:d63fc094-->
Der Vorwärtsdurchlauf einer Neural ODE ist eine Schleife mit $N$ Schritten des numerischen Verfahrens. Der direkte Weg zum Training lautet: *erst diskretisieren, dann differenzieren*. Dazu nehmen wir das Verfahren in den Berechnungsgraphen auf, berechnen den Verlust am Endzustand $\mathbf{x}_N$ und wenden Backpropagation auf die entrollten Schritte an. Beim Euler-Verfahren entspricht das Programm einem ResNet aus $N$ Blöcken mit gemeinsamen Gewichten. Wir können es also wie gewohnt durch automatisches Differenzieren trainieren.

<!--en:47a782d9-->
Probieren wir das an einer bewusst einfachen Aufgabe aus. Wir wollen $64$ Punkte auf dem Einheitskreis durch Streckung und Verschiebung auf die entsprechenden Punkte einer Ellipse abbilden. Ihr Mittelpunkt ist $(1.5,0)$, ihre Halbachsen haben die Längen $1.2$ und $0.4$. Der exakte Fluss des affinen Feldes $\dot{\mathbf{x}}=\mathrm{diag}(\ln1.2,\ln0.4)\,\mathbf{x}+\mathbf{c}$ löst die Aufgabe bei geeignetem $\mathbf{c}$ fehlerfrei. Ein beobachteter Fehler kann also nicht daran liegen, dass es keine Lösung gibt. Erreicht das trainierte Feld die Ziele auch bei genauer Integration oder nur mit den zehn Schritten aus dem Training? Im ersten Fall hat das Training einen Fluss gefunden, im zweiten nur ein Programm mit zehn Schritten.

<!--en:659806cd-->
Das Feld hat eine versteckte Schicht mit $32$ tanh-Einheiten. Der Verlust ist der mittlere quadratische Fehler an den Endpunkten. Adam (:numref:`sec_mdl-adaptive-stochastic-methods`) verwendet in den Iterationen $0$ bis $400$ jeweils den gesamten Datensatz. Die Lernrate beträgt $0.05$ und sinkt für die letzten hundert Iterationen auf $0.005$. Ausgehend von denselben Anfangsgewichten trainieren wir zweimal: einmal durch $N=10$ Euler-Schritte der Weite $h=0.1$ und einmal durch zehn RK4-Schritte. RK4 wertet das Feld viermal pro Schritt aus.

<!--en:09607c16-->
Wir legen den Ausgangskreis `X0_ring`, die Zielellipse `X1_ring` und die gemeinsame Initialisierung `seed` fest. Dann passt `train` das zweischichtige `tanh`-Vektorfeld `net` mittels Backpropagation durch 10 Schritte von `solve(step, net, X, 1.0, 10)` für jedes Verfahren in `solvers` an:

<!--en:8ad0ddf1-->
Beide Verluste fallen von der Größenordnung eins auf unter $10^{-5}$. Gemessen am Trainingsverlust sind die beiden Felder gleich gut.

<!--en:11fc4897-->
Wir führen zwei Tests durch. Der erste greift unsere Frage direkt auf. Wir integrieren jedes trainierte Feld mit seinem eigenen Verfahren und verwenden $N=100$ und $N=1000$ Schritte (stets mit $Nh=1$). Je größer $N$ ist, desto näher liegt die berechnete Abbildung am exakten Fluss zur Zeit eins. Was erwarten Sie für die beiden Endpunktfehler? Überlegen Sie kurz, bevor Sie die Zelle ausführen. Der zweite Test prüft die Umkehrbarkeit, den ersten Vorteil auf unserer Liste. Wir gehen im mit Euler trainierten Feld $N$ Schritte vorwärts und dann $N$ Schritte rückwärts. Wie weit sind wir danach vom Anfangspunkt entfernt? Ein exakter Fluss würde jeden Punkt an seinen Ausgangsort zurückführen.

<!--en:0f72bf5b-->
Betrachten wir zunächst den Hin- und Rückweg in der letzten Spalte. Der Abstand zum Anfangspunkt beträgt etwa $0.055$ bei $N=10$, $0.0055$ bei $N=100$ und $0.00055$ bei $N=1000$. Zehnmal so viele Schritte verringern den Fehler also auf ein Zehntel. Jedes Paar aus einem Euler-Schritt vorwärts und einem rückwärts hinterlässt ein Residuum der Ordnung $h^2$. Bei $N$ Paaren summieren sich die Residuen zu $O(h)$, der Ordnung des Euler-Verfahrens. Das Feld rückwärts zu durchlaufen macht den exakten Fluss rückgängig. Beim Euler-Programm gelingt dies erst im Grenzfall $h\to0$. Ein Euler-Programm mit $hL<1$ besitzt zwar eine Umkehrabbildung, denn jeder seiner Schritte ist dann bijektiv (Aufgabe :exref:`ex_mdl-odes-solvers-residual-logdet`). Die Umkehrabbildung besteht aber aus impliziten Schritten und nicht aus Euler-Schritten mit $-h$.

<!--en:6114031a-->
Betrachten wir nun die Endpunktfehler. Mit der Schrittzahl aus dem Training verfehlen beide Felder ihre Ziele nur um wenige Tausendstel. Überraschenderweise schneiden feinere Schritte beim mit Euler trainierten Feld schlechter ab. Sein Fehler *steigt* auf etwa $0.02$ und bleibt dann konstant. Das Training hat das Programm mit zehn Schritten angepasst. Der exakte Fluss dieses Feldes zur Zeit eins, den wir mit $N=1000$ annähern, verfehlt die Ziele um $0.02$. Das sind weniger als $2\%$ der großen Halbachse der Ellipse. Der Fluss liegt also nahe an der Abbildung des Programms mit zehn Schritten, stimmt aber nicht mit ihr überein.

<!--en:83c02613-->
Für das mit RK4 trainierte Feld erhalten wir bei zehn, hundert und tausend Schritten denselben Fehler. Bei einem so glatten Feld berechnen schon zehn RK4-Schritte den Fluss auf die angezeigten Stellen genau. Mit dem Programm haben wir also auch den Fluss angepasst und können die Schrittzahl nach dem Training frei wählen. Mit einem groben numerischen Verfahren lernen wir ein Programm, mit einem genauen einen Fluss.

<!--en:373f1702-->
Diese Genauigkeit müssen wir allerdings in jeder Iteration aufbringen. Beim Training mit RK4 werten wir das Feld pro Iteration vierzigmal aus und wenden Backpropagation auf alle vierzig Auswertungen an. Beim Euler-Verfahren sind es zehn. Hundert Euler-Schritte oder ein Verfahren zweiter Ordnung würden nur die Anzahl ändern. Alternativ können wir das numerische Verfahren ganz aus dem Training herausnehmen. Wir passen das Feld durch Regression an Zielwerte an, die in geschlossener Form bekannt sind. Das erfordert eine Feldauswertung pro Beispiel, und wir passen von Anfang an einen Fluss an. So lernen wir die Felder in :numref:`sec_mdl-score-matching-diffusion-flow` und :numref:`sec_mdl-flow-matching-transport`.

<!--en:6a8a6f37-->
## Die adjungierte Methode

<!--en:2d0e6275-->
Beim Training haben wir den Gradienten des Programms mit zehn Schritten erhalten. Programm und Fluss beschreiben jedoch verschiedene Abbildungen. Unterscheiden sich auch ihre Gradienten? Und lässt sich der Gradient des Flusses überhaupt berechnen, wenn kein endliches Programm den Fluss berechnet?

<!--en:b0560b32-->
Ja, mit einer Idee, die bereits im Differenzieren im Rückwärtsmodus steckt. Der Rückwärtsmodus reicht die Ableitung des Verlusts nach den Aktivierungen einer Schicht von Schicht zu Schicht zurück. In :numref:`subsec_mdl-backpropagation` haben wir sie als adjungierte Variable eines Knotens bezeichnet. Die adjungierte Methode verfolgt dieselbe Größe rückwärts durch die Zeit, mit einem Knoten $\mathbf{x}(t)$ für jeden Zeitpunkt. Im Mittelpunkt steht der *adjungierte Zustand* (engl. adjoint state) $\mathbf{a}(t)=\partial\mathcal{L}/\partial\mathbf{x}(t)$, der Gradient des Verlusts bezüglich des Zustands zur Zeit $t$. Wie wir sehen werden, erfüllt er selbst eine lineare Differentialgleichung.

<!--en:ca9aa749-->
### Der skalare Fall

<!--en:9f081418-->
Vor der allgemeinen Aussage rechnen wir ein Beispiel vollständig durch, in dem alle Größen in geschlossener Form vorliegen. Betrachten wir $\dot{x}=\theta x$ mit $x(0)=x_0$ und dem Verlust $\mathcal{L}=\tfrac12x(T)^2$ am Endpunkt. Direkt erhalten wir $x(T)=e^{\theta T}x_0$, also $\partial x(T)/\partial\theta=T\,x(T)$ und

<!--en:58badf27-->
Mit $\theta=T=x_0=1$ ergibt das $e^2=7.389$.

<!--en:ee57688a-->
Die adjungierte Methode liefert denselben Wert auf anderem Weg. Eine kleine Änderung $\delta$ des Zustands zur Zeit $t$ wächst bis zur Zeit $T$ auf $e^{\theta(T-t)}\delta$ an und ändert den Verlust um $x(T)\,e^{\theta(T-t)}\delta$. Also gilt $a(t)=e^{\theta(T-t)}x(T)$. Diese adjungierte Variable löst die ODE $\dot{a}=-\theta a$ rückwärts vom Endwert $a(T)=x(T)$ aus. Dieser ist die Ableitung des Verlusts $\ell(x)=\tfrac12x^2$ an der Stelle $x(T)$. Eine Änderung $\delta\theta$ des Parameters ändert die Geschwindigkeit zu jedem Zeitpunkt um $x(t)\,\delta\theta$, denn es gilt $\partial f/\partial\theta=x$. Die adjungierte Variable übersetzt jede dieser kleinen Änderungen in eine Änderung des Verlusts. Summieren wir über die Zeit, erhalten wir

<!--en:15085c72-->
Der Integrand ist konstant. In Vorwärtsrichtung wächst $x$, während $a$ um denselben Faktor abnimmt. Ihr Produkt bleibt daher bei $x(T)^2$ (:numref:`fig_mdl-dyn-adjoint-timeline`). Der Gradient ist die Fläche unter dieser konstanten Funktion.

<!--en:655bfd40-->
![Die skalare adjungierte Variable für $\dot x=\theta x$ bei $\theta=1$, $x_0=1$, $T=1$. Der Zustand wächst in Vorwärtsrichtung. Die adjungierte Variable wird vom Verlustgradienten $a(1)=x(1)$ aus rückwärts integriert. Ihr Produkt bleibt konstant. Der Parametergradient entspricht der schattierten Fläche $Tx(T)^2=e^2$.](../img/mdl-dyn-adjoint-timeline.svg)

<!--en:e26f1e92-->
### Die adjungierten Gleichungen

<!--en:dfed7c73-->
Der allgemeine Fall hat dieselbe Struktur wie der skalare. Die adjungierte Variable entwickelt sich vom Verlustgradienten aus rückwärts. Der Parametergradient ergibt sich durch Integration der adjungierten Variablen gegen $\partial\mathbf{f}/\partial\boldsymbol\theta$.

<!--en:baa5bc0e-->
**Proposition (adjungierte Gleichungen).** *Sei $\mathbf{f}$ stetig differenzierbar in $\mathbf{x}$ und $\boldsymbol\theta$. Sei $\mathbf{x}(t)$ die Lösung von $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},\boldsymbol\theta,t)$ (das Feld $\mathbf{f}_{\boldsymbol\theta}$ mit seinen Parametern als explizitem Argument) auf $[0,T]$ zu einem festen Anfangswert $\mathbf{x}(0)$. Weiter gelte $\mathcal{L}=\ell(\mathbf{x}(T))$ für ein differenzierbares $\ell$. Dann erfüllt der adjungierte Zustand die lineare ODE*

<!--en:edd4527b-->
*mit Rückwärtsintegration vom Endwert $T$ aus. Der Parametergradient ist das Integral*

<!--en:be73ad5f-->
*wobei wir beide partiellen Ableitungen entlang der Trajektorie $(\mathbf{x}(t),\boldsymbol\theta,t)$ auswerten* :cite:`Chen.Rubanova.Bettencourt.ea.2018`.

<!--en:a7705efa-->
**Beweis durch Variation.** Wir ändern die Parameter um ein infinitesimales $\delta\boldsymbol\theta$. Da $\mathbf{f}$ zur Klasse $C^1$ gehört, ist die Lösung in $\boldsymbol\theta$ differenzierbar. Die Störung erster Ordnung $\delta\mathbf{x}(t)$ der Trajektorie erfüllt die *Linearisierung* der Dynamik. Sie ergibt sich, wenn wir die ODE nach $\boldsymbol\theta$ differenzieren und die Kettenregel anwenden:

<!--en:f85d4ed7-->
Definieren wir $\mathbf{a}(t)$ durch die Rückwärtsgleichung :eqref:`eq_mdl-ode-adjoint` und differenzieren das Skalarprodukt $\mathbf{a}^\top\delta\mathbf{x}$:

<!--en:e1f11d93-->
Die Terme, die an den Zustand koppeln, heben sich auf. Genau dafür haben wir die adjungierte Gleichung so gewählt. Wir integrieren von $0$ bis $T$ und verwenden am linken Rand $\delta\mathbf{x}(0)=\mathbf{0}$:

<!--en:261acfaf-->
Links steht $\nabla\ell(\mathbf{x}(T))^\top\delta\mathbf{x}(T)=\delta\mathcal{L}$, die Änderung des Verlusts erster Ordnung. Die rechte Seite ist linear in $\delta\boldsymbol\theta$ mit dem Koeffizienten :eqref:`eq_mdl-ode-adjoint-grad`. Es bleibt zu zeigen, dass $\mathbf{a}$ der adjungierte Zustand ist. Wenden wir dasselbe Argument mit $\delta\boldsymbol\theta=\mathbf{0}$ und einer Störung $\delta\mathbf{x}(s)$ zur Zeit $s$ an. Das Skalarprodukt $\mathbf{a}^\top\delta\mathbf{x}$ ist dann auf $[s,T]$ konstant. Folglich gilt $\delta\mathcal{L}=\mathbf{a}(T)^\top\delta\mathbf{x}(T)=\mathbf{a}(s)^\top\delta\mathbf{x}(s)$, und die Lösung von :eqref:`eq_mdl-ode-adjoint` ist $\partial\mathcal{L}/\partial\mathbf{x}(s)$. $\blacksquare$

<!--en:bae1c200-->
Die Rückwärtsgleichung ist linear in $\mathbf{a}$. Ihre Koeffizientenmatrix ist die transponierte Jacobi-Matrix. In jedem Schritt berechnen wir $\mathbf{a}^\top\partial\mathbf{f}/\partial\mathbf{x}$, ein Vektor-Jacobi-Produkt. Das kostet einen Durchlauf im Rückwärtsmodus durch $\mathbf{f}$ (:numref:`sec_mdl-matrix-calculus-autodiff`). Derselbe Durchlauf liefert $(\partial\mathbf{f}/\partial\boldsymbol\theta)^\top\mathbf{a}$, den Integranden des Parametergradienten, unabhängig von der Anzahl der Parameter. (Ein ReLU-Feld gehört nur stückweise zur Klasse $C^1$. Die Proposition erfasst es daher nicht, obwohl automatisches Differenzieren die entrollten Schritte des Verfahrens weiterhin stückweise differenziert. Glatte Aktivierungen wie tanh oder GELU erfüllen die Voraussetzung.)

<!--en:5556e8f7-->
### Backpropagation als diskrete adjungierte Methode

<!--en:1b4ded32-->
Nun können wir Backpropagation durch das Euler-Programm Zeile für Zeile mit der adjungierten Gleichung vergleichen. Damit die Formeln kurz bleiben, wählen wir ein autonomes Feld und schreiben $\mathbf J=\partial\mathbf{f}/\partial\mathbf{x}$ für seine Jacobi-Matrix. Wir differenzieren das Programm $\mathbf{x}_{n+1}=\mathbf{x}_n+h\,\mathbf{f}(\mathbf{x}_n,\boldsymbol\theta)$ mit der Kettenregel und setzen $\mathbf{a}_n=\partial\mathcal{L}/\partial\mathbf{x}_n$. Die Jacobi-Matrix eines Schritts ist $\mathbf{I}+h\mathbf J(\mathbf{x}_n)$. Damit folgt

<!--en:1bcdbd46-->
Vergleichen wir dies mit einem expliziten Euler-Schritt für die adjungierte Gleichung :eqref:`eq_mdl-ode-adjoint` von $t_{n+1}$ zurück nach $t_n$:

<!--en:cca13e9b-->
Die Aktualisierungen unterscheiden sich nur darin, wo wir die Jacobi-Matrix auswerten: am Anfang $\mathbf{x}_n$ des Vorwärtsschritts oder an seinem Ende $\mathbf{x}_{n+1}$. Die Summe in :eqref:`eq_mdl-ode-discrete-adjoint` entspricht der Rechteckregel für das Integral :eqref:`eq_mdl-ode-adjoint-grad`. Backpropagation durch ein numerisches Verfahren ist also eine *diskrete adjungierte Methode*. Wenden wir dagegen ein numerisches Verfahren auf die adjungierte Gleichung an, diskretisieren wir die *kontinuierliche adjungierte Gleichung*. In der Literatur heißen die beiden Wege „erst diskretisieren, dann optimieren“ und „erst optimieren, dann diskretisieren“ :cite:`Kidger.2022`. Der erste entspricht unserem Trainingsprinzip „erst diskretisieren, dann differenzieren“. Er liefert den exakten Gradienten des Programms. Der zweite nähert den Gradienten des Flusses an. Beim Euler-Verfahren unterscheiden sich beide Gradienten in erster Ordnung in $h$. Das ist die wichtigste Aussage dieses Unterabschnitts.

<!--en:e8df48fa-->
Im skalaren Beispiel sehen wir beide Gradienten zugleich. Für $\dot{x}=\theta x$ lautet das Euler-Programm $x_N=(1+h\theta)^Nx_0$. Damit gilt $\partial x_N/\partial\theta=Nh(1+h\theta)^{N-1}x_0=T\,x_N/(1+h\theta)$, und die diskrete adjungierte Methode liefert $\partial\mathcal{L}/\partial\theta=T\,x_N^2/(1+h\theta)$. Mit $\theta=T=x_0=1$ erhalten wir $6.116$ bei $N=10$, $7.244$ bei $N=100$ und $7.374$ bei $N=1000$ gegenüber dem exakten Wert $e^2=7.389$. Die Abweichungen $1.27$, $0.145$ und $0.0148$ sinken jeweils auf ein Zehntel, wenn wir $N$ verzehnfachen.

<!--en:1a36c9c6-->
Mit dem Spiralfeld aus :numref:`sec_mdl-odes-solvers`, also $\dot{\mathbf{x}}=\mathbf A\mathbf{x}$ mit $\mathbf A=\left(\begin{smallmatrix}-0.5&-1\\1&-0.5\end{smallmatrix}\right)$, können wir dieselbe Prüfung in zwei Dimensionen durchführen. Alle Größen liegen in geschlossener Form vor. Für ein lineares Feld ist die Jacobi-Matrix an jedem Punkt gleich $\mathbf A$. Die beiden obigen Rekursionen stimmen daher überein. Zu messen bleibt ihr gemeinsamer Fehler gegenüber dem Fluss. Er entsteht, weil wir die Zustände, die adjungierte Variable und die Summe mit dem Euler-Verfahren berechnen.

<!--en:d975588e-->
Wählen wir als $\boldsymbol\theta$ die Einträge von $\mathbf A$ und setzen wir $\mathcal{L}=\tfrac12\|\mathbf{x}(T)\|^2$. Die Trajektorie lautet $e^{\mathbf At}\mathbf{x}_0$, die adjungierte Variable ist $\mathbf{a}(t)=e^{\mathbf A^\top(T-t)}\mathbf{x}(T)$ (die Lösung von $\dot{\mathbf{a}}=-\mathbf A^\top\mathbf{a}$). Wegen $\partial\mathbf{f}/\partial A_{ij}=x_j\mathbf{e}_i$ vereinfacht sich der Gradient :eqref:`eq_mdl-ode-adjoint-grad` zu $\partial\mathcal{L}/\partial \mathbf A=\int_0^T\mathbf{a}(t)\,\mathbf{x}(t)^\top dt$.

<!--en:6bc091a1-->
Die Zelle implementiert den Rückwärtsdurchlauf :eqref:`eq_mdl-ode-discrete-adjoint` für $T=2$ von Hand. Die erste Ausgabezeile vergleicht ihn mit zentralen endlichen Differenzen desselben Programms bei $N=1000$ Schritten. Das prüft lediglich die Implementierung. Entscheidend ist die Abweichung vom Gradienten des exakten Flusses. Dafür wertet die Zelle das kontinuierliche adjungierte Integral mit der Trapezregel auf $2000$ Intervallen aus. Dabei stammen $\mathbf{a}$ und $\mathbf{x}$ aus dem Matrixexponential. Die Zelle vergleicht das Ergebnis für drei Schrittzahlen mit der diskreten adjungierten Methode.

<!--en:5288d26e-->
Der Rückwärtsdurchlauf stimmt bis auf etwa $10^{-10}$ mit den endlichen Differenzen überein. Das entspricht deren eigener Genauigkeit. Die Abweichung vom Gradienten des Flusses fällt von $2.9\times10^{-2}$ über $2.8\times10^{-3}$ auf $2.8\times10^{-4}$. Wie im skalaren Fall sinkt sie auf ein Zehntel, wenn wir $N$ verzehnfachen.

<!--en:96b45bd4-->
### Speicherbedarf und Drift

<!--en:1ae882e5-->
Die adjungierte Gleichung benötigt $\mathbf{x}(t)$ für jeden Rückwärtsschritt, denn dort werten wir die Jacobi-Matrix aus. Wir können diesen Zustand auf zwei Arten bereitstellen. Wie bei der Backpropagation können wir die Zustände des Vorwärtsdurchlaufs speichern. Das kostet Speicher proportional zu $N$. Oder wir integrieren $\dot{\mathbf{x}}=\mathbf{f}$ von $\mathbf{x}(T)$ aus zusammen mit $\mathbf{a}$ erneut rückwärts. Dann bleibt der Speicherbedarf unabhängig von $N$ konstant, doch jeder Schritt erfordert eine zusätzliche Feldauswertung. Die zweite Möglichkeit klingt zu gut. Schon das einfachste Beispiel zeigt ihren Nachteil: Sie verstärkt Fehler.

<!--en:3b047a94-->
Für $\dot{x}=-\lambda x$ mit $\lambda>0$ löst die Rückwärtsintegration die Gleichung $\dot{x}=+\lambda x$. Ein absoluter Fehler $\delta$ im gespeicherten Endzustand $x(T)$ wächst daher auf $e^{\lambda T}\delta$ an, bis die Integration $t=0$ erreicht.

<!--en:0549bb3b-->
Wie groß ist $\delta$? Betrachten wir einen Zustand mit zwei Moden. Einer klingt mit der Rate $\lambda=50$ ab, der schnellen Rate aus dem steifen Beispiel in :numref:`sec_mdl-stiffness-implicit`. Der andere ändert sich kaum. Die Richtung des schnellen Modus liege auf einer Diagonalen, sodass jede Koordinate einen Anteil davon enthält. Zur Zeit $T$ ist der schnelle Modus abgeklungen. Die gespeicherten Koordinaten haben die Größenordnung eins. Bei doppelter Genauigkeit beträgt der Rundungsfehler jeder Koordinate etwa $10^{-16}$. Die Rundung berücksichtigt die Moden nicht. Ein Teil des Fehlers liegt in der schnellen Richtung, also gilt $\delta\approx10^{-16}$. Der rekonstruierte Anfangszustand weicht dann um $2.7\times10^{-11}$ bei $T=0.25$, um $7\times10^{-6}$ bei $T=0.5$ und um $5\times10^{5}$ bei $T=1$ ab. Im letzten Fall übersteigt die Abweichung den Zustand selbst. Doppelte Genauigkeit bietet etwa sechzehn Stellen. Ein Faktor von $e^{50}\approx5\times10^{21}$ hat fünf Stellen mehr. In Aufgabe :exref:`ex_mdl-odes-solvers-backward-reconstruction` messen Sie dieses Wachstum.

<!--en:967a9b07-->
Bei der Spirale haben die Eigenwerte den Realteil $-0.5$. Über $T=2$ beträgt der Faktor daher nur $e$. Ein vorwärts stark kontrahierender Fluss expandiert rückwärts stark. Die erneute Integration spart also Speicher auf Kosten der Drift. Für steife Felder ist *Checkpointing* (:numref:`subsec_mdl-checkpointing`) robuster. Wir speichern den Zustand nur an wenigen Checkpoints. Die Zwischenzustände berechnen wir vom jeweils vorhergehenden Checkpoint aus vorwärts erneut. Damit integrieren wir immer in Flussrichtung. Bei einem Checkpoint alle $\sqrt N$ Schritte hält der Rückwärtsdurchlauf etwa $2\sqrt N$ Zustände im Speicher, also $63$ statt $1000$ für $N=1000$. Das kostet einen zusätzlichen Vorwärtsdurchlauf.

<!--en:8359a983-->
Damit haben wir drei Wege zum Gradienten. Welchen wählen wir? Backpropagation durch das numerische Verfahren, bei knappem Speicher mit Checkpoints. Sie liefert den exakten Gradienten des tatsächlich ausgeführten Programms. Die kontinuierliche adjungierte Methode beschreibt den Grenzwert dieses Gradienten und liefert die Formel für theoretische Überlegungen. Den Zustand erneut rückwärts zu integrieren bleibt ein letzter Ausweg für Felder, die nicht steif sind.

<!--en:bf80cdb3-->
## Kontinuierliche normalisierende Flüsse

<!--en:5216900a-->
Eine Neural ODE bewegt Punkte. Ein generatives Modell, das wir mit Maximum-Likelihood trainieren, muss auch angeben, wie wahrscheinlich seine einzelnen Ausgaben sind. Daher verfolgen wir nun eine Dichte entlang des Flusses.

<!--en:aab8ecf9-->
### Dichte entlang eines Flusses

<!--en:d9b3f234-->
Transportieren wir Samples mit einem Fluss, erhalten wir einen Generator: Wir ziehen $\mathbf{x}(0)$ aus einer einfachen Dichte $p_0$ und integrieren. Für das Maximum-Likelihood-Training eines solchen Generators benötigen wir die Dichte seiner Ausgabe. Ein so trainiertes umkehrbares Netz heißt *normalisierender Fluss*. Seine Dichte folgt aus der Formel für die Variablentransformation in :numref:`sec_mdl-joint-distributions`. Für eine umkehrbare differenzierbare Abbildung $\mathbf{g}$ mit Jacobi-Matrix $\mathbf J_{\mathbf{g}}$ gilt

<!--en:781ca0e0-->
Die Log-Dichte sinkt um den Logarithmus des Faktors, um den $\mathbf{g}$ das Volumen in der Nähe von $\mathbf{x}$ streckt. Ein normalisierender Fluss mit vielen Schichten benötigt pro Schicht und Sample eine Log-Determinante. Eine allgemeine Determinante kostet $O(d^3)$ Operationen, also etwa $2\times10^{10}$ bei der Dimension $d=3072$ eines Farbbilds mit $32\times32$ Pixeln. Diskrete Flüsse beschränken ihre Schichten daher auf Jacobi-Matrizen in Dreiecksform, deren Determinante das Produkt der Diagonaleinträge ist. Alternativ verwenden sie Jacobi-Matrizen aus der Einheitsmatrix plus einem Term von niedrigem Rang, etwa wie in $\det(\mathbf{I}+\mathbf{u}\mathbf{v}^\top)=1+\mathbf{v}^\top\mathbf{u}$. In beiden Fällen kostet die Determinante $O(d)$.

<!--en:86f7c172-->
In kontinuierlicher Zeit vereinfacht sich die Formel. Beginnen wir in einer Dimension. Das Feld $\dot{x}=\theta x$ auf $\mathbb{R}$ hat den Fluss $\Phi_t(x)=e^{\theta t}x$, der jedes Intervall um den Faktor $e^{\theta t}$ streckt. Die mitgeführte Dichte wird durch den Streckfaktor geteilt. Daher gelten $\log p_t(x(t))=\log p_0(x_0)-\theta t$ und $\tfrac{d}{dt}\log p_t(x(t))=-\theta$. Die Zahl $\theta$ ist die Jacobi-Matrix der Größe $1\times1$ des Feldes und damit zugleich ihre Spur.

<!--en:99fdb001-->
In zwei Dimensionen betrachten wir wieder die Spirale. Ihr Fluss $e^{\mathbf At}=e^{-t/2}e^{t\mathbf K}$ dreht die Ebene und staucht sie anschließend mit dem Faktor $e^{-t/2}$ (:numref:`sec_mdl-odes-solvers`). Eine Kreisscheibe geht in eine Kreisscheibe über, deren Radius das $e^{-t/2}$-Fache des ursprünglichen Radius beträgt. Ihr Flächeninhalt wird also mit $e^{-t}$ multipliziert, und die Dichte steigt entlang jeder Trajektorie um den Faktor $e^{t}$. Hier gilt $\operatorname{tr}\mathbf A=-1$. Die Änderungsrate der Log-Dichte ist also wieder gleich der negativen Spur. :numref:`fig_mdl-dyn-divergence-volume` zeigt zum Vergleich ein flächenerhaltendes und ein flächenvergrößerndes Feld. Die dunkelste Kreisscheibe der Spirale zur Zeit $t=1.2$ hat nur das $e^{-1.2}\approx0.30$-Fache der Fläche der hellsten.

<!--en:d39f283b-->
![Eine Kreisscheibe aus Samples unter den exakten Flüssen dreier linearer Felder zu den Zeiten $t=0$, $0.6$ und $1.2$ (je dunkler, desto später). Die gepunktete Linie zeigt die Bahn des Mittelpunkts. Die Dichte entlang einer Trajektorie ändert sich um den Faktor $e^{-t\operatorname{tr}\mathbf A}$. Sie steigt dort, wo der Fluss kontrahiert (links, Spirale), bleibt beim flächenerhaltenden Sattelfeld trotz der Formänderung konstant (Mitte) und fällt dort, wo der Fluss expandiert (rechts).](../img/mdl-dyn-divergence-volume.svg)

<!--en:a7f839ca-->
### Die infinitesimale Variablentransformation

<!--en:70769ca4-->
Beide Beispiele waren linear, und in beiden änderte sich die Log-Dichte mit einer Rate gleich der negativen Spur. Für den allgemeinen Fall brauchen wir eine Eigenschaft der Determinante. Über einen kurzen Schritt liegt die Flussabbildung nahe an der Identität. Bei einer Matrix nahe der Einheitsmatrix bestimmt die Spur des Störterms die Änderung der Determinante in erster Ordnung.

<!--en:ecf0cd39-->
**Lemma.** *Für jede quadratische Matrix $\mathbf J$ gilt $\det(\mathbf{I}+h\mathbf J)=1+h\operatorname{tr}\mathbf J+O(h^2)$.*

<!--en:53f0f366-->
**Beweis.** Die Permutationsformel (:numref:`subsec_mdl-determinant-general`) stellt die Determinante als Summe über alle Permutationen dar. Jeder Summand ist ein vorzeichenbehaftetes Produkt aus je einem Eintrag jeder Zeile und jeder Spalte. Jede Permutation außer der Identität wählt mindestens zwei Einträge außerhalb der Diagonalen. Diese haben jeweils die Größenordnung $O(h)$, sodass die Permutation einen Beitrag der Ordnung $O(h^2)$ liefert. Die identische Permutation trägt $\prod_i(1+hJ_{ii})=1+h\sum_iJ_{ii}+O(h^2)$ bei. $\blacksquare$

<!--en:3e83488d-->
**Proposition (infinitesimale Variablentransformation).** *Es gelte $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x},t)$, wobei $\mathbf{f}$ stetig differenzierbar in $(\mathbf{x},t)$ und global Lipschitz-stetig in $\mathbf{x}$ sei. Dann ist jede Flussabbildung $\Phi_t$ eine $C^1$-Bijektion mit einer Umkehrabbildung der Klasse $C^1$ (die Lösung eines $C^1$-Feldes ist eine $C^1$-Funktion ihres Anfangswerts :cite:`Hairer.Norsett.Wanner.1993`, und für die Umkehrabbildung durchläuft man das Feld rückwärts). Es gelte $\mathbf{x}(0)\sim p_0$ für eine positive differenzierbare Dichte $p_0$. Dann existiert die Dichte $p_t$ von $\mathbf{x}(t)$, sie ist positiv, und entlang jeder Trajektorie gilt* :cite:`Chen.Rubanova.Bettencourt.ea.2018`

<!--en:9f2ca7b2-->
**Beweis.** Die Existenz der Dichte folgt aus :eqref:`eq_mdl-ode-discrete-cov`, angewandt auf die Bijektion $\Phi_t$. Die Dichte ist positiv, weil $p_0$ positiv und die Jacobi-Matrix einer differenzierbar umkehrbaren Abbildung regulär ist. Über ein kurzes Zeitintervall schreitet der Fluss mit der Abbildung $\Phi_{t\to t+h}(\mathbf{x})=\mathbf{x}+\int_t^{t+h}\mathbf{f}(\mathbf{x}(s),s)\,ds$ voran. Durch Differenzieren nach $\mathbf{x}$ erhalten wir wegen der Stetigkeit von $\mathbf J$ die Jacobi-Matrix $\mathbf{I}+h\mathbf J+o(h)$. Dabei werten wir $\mathbf J=\partial\mathbf{f}/\partial\mathbf{x}$ an der Stelle $(\mathbf{x}(t),t)$ aus, und $o(h)$ bezeichnet Terme, die schneller als $h$ verschwinden. Wir wenden :eqref:`eq_mdl-ode-discrete-cov` auf diesen einen Schritt an und entwickeln die Determinante mithilfe des Lemmas. Die Störung spielt dabei keine Rolle: Ersetzen wir $h\mathbf J$ durch $h\mathbf J+o(h)$, ändert sich die Determinante nur um $o(h)$. Für kleine $h$ liegt die Determinante nahe bei $1$ und ist daher positiv. Wir können also die Betragsstriche weglassen:

<!--en:66f6e58c-->
Dabei haben wir $\log(1+u)=u+O(u^2)$ verwendet. Subtrahieren wir $\log p_t(\mathbf{x}(t))$, dividieren durch $h$ und bilden den Grenzwert $h\to0$, so folgt die Behauptung. $\blacksquare$

<!--en:533ac7dd-->
Die Voraussetzungen sind weniger streng, als sie erscheinen. Sie verlangen nur, dass jede Flussabbildung eine glatte Bijektion ist. Ein Feld mit glatten Lipschitz-stetigen Aktivierungen wie tanh erfüllt dies. Die Aussage der Proposition steckt in der einen Zeile mit der Spur.

<!--en:2ebe0b4b-->
Die Spur der Jacobi-Matrix ist die *Divergenz* $\nabla\cdot\mathbf{f}$ des Feldes. Sie beschreibt die lokale Rate der Volumenausdehnung. Wo das Feld die Samples auseinandertreibt ($\operatorname{tr}\mathbf J>0$), fällt die Log-Dichte entlang jeder Trajektorie. Wo es sie zusammendrückt, steigt sie. Integrieren wir :eqref:`eq_mdl-ode-instant-cov` entlang einer Trajektorie, erhalten wir

<!--en:fb553e40-->
Aus der Summe der Log-Determinanten über die Schichten eines diskreten Flusses ist ein Integral über Spuren geworden. Ein Modell, das dieses Integral berechnet, indem es das Paar $(\mathbf{x},\log p)$ mit einem beliebigen ODE-Löser integriert, heißt **kontinuierlicher normalisierender Fluss** (engl. continuous normalizing flow, CNF) :cite:`Chen.Rubanova.Bettencourt.ea.2018`. Die Architektur von $\mathbf{f}$ unterliegt dabei keiner Einschränkung. Um die Dichte an einem gegebenen Punkt $\mathbf{x}(T)$ zu bestimmen, integrieren wir das Feld von dort rückwärts bis $\mathbf{x}(0)$. Unterwegs integrieren wir die Spur mit und lesen am Ende $p_0$ ab. So berechnen wir die Likelihoods von Diffusionsmodellen (:numref:`sec_diffusion-design-space`). Die Formel bevorzugt keine Zeitrichtung. In den Generatoren aus :numref:`sec_mdl-score-matching-diffusion-flow` und :numref:`sec_mdl-flow-matching-transport` liegt das Rauschen am Endzeitpunkt und die Daten liegen bei $t=0$. Für sie gilt dieselbe Formel.

<!--en:f988200a-->
Bei einem linearen Feld ergibt das Spurintegral $t\operatorname{tr}\mathbf A$. Vergleichen wir :eqref:`eq_mdl-ode-cnf-likelihood` mit :eqref:`eq_mdl-ode-discrete-cov` für die Abbildung $e^{\mathbf At}$, so folgt $\log\det e^{\mathbf At}=t\operatorname{tr}\mathbf A$. Für eine diagonalisierbare Matrix $\mathbf A=\mathbf V\boldsymbol\Lambda \mathbf V^{-1}$ ergibt sich die Identität auch direkt aus den Eigenwerten:

<!--en:8998433d-->
Der Vergleich der beiden Formeln liefert sie für jede Matrix $\mathbf A$.

<!--en:df06a3de-->
Die Proposition verfolgt jeweils eine einzelne Trajektorie. Für die gesamte Dichte lautet dasselbe Gesetz in der Strömungsmechanik Kontinuitätsgleichung. In :numref:`sec_mdl-fokker-planck-probability-flow` leiten wir sie zusammen mit dem Term her, der bei zusätzlichem Rauschen entsteht.

<!--en:0ca8b110-->
### Überprüfung des Spurintegrals

<!--en:40874600-->
Ein lineares Feld stellt die Proposition kaum auf die Probe. Seine Spur ist konstant, und das Integral ergibt schlicht $t\operatorname{tr}\mathbf A$. Aussagekräftiger ist ein nichtlineares Feld, dessen Spur sich entlang der Bahn ändert. Betrachten wir das ebene Feld $\mathbf{f}(\mathbf{x})=\mathbf W_2\tanh(\mathbf W_1\mathbf{x}+\mathbf{b}_1)$ mit $8$ versteckten Einheiten und zufälligen Gewichten. Seine Jacobi-Matrix und deren Spur lassen sich explizit angeben:

<!--en:c12583f6-->
Dabei verwenden wir die zyklische Eigenschaft $\operatorname{tr}\big(\mathbf W_2\,\mathrm{diag}(\mathbf{s})\,\mathbf W_1\big)=\operatorname{tr}\big(\mathrm{diag}(\mathbf{s})\,\mathbf W_1\mathbf W_2\big)$. Die Zelle integriert das Paar $(\mathbf{x},\int\operatorname{tr}\mathbf J\,dt)$ mit RK4 bis $T=1$ und speichert die Bahn. Sie gibt die kleinste und die größte Spur entlang der Bahn aus. Anschließend vergleicht sie das Integral mit $\log|\det D\Phi_T|$. Dazu berechnet sie die Jacobi-Matrix $D\Phi_T$ der Flussabbildung aus zentralen endlichen Differenzen der Flussabbildung selbst.

<!--en:2b35f98d-->
Die Spur ändert sich tatsächlich. Die erste Zeile gibt den Wertebereich entlang der Bahn an. Seine Breite hängt von den gezogenen Gewichten ab. Typisch sind ein Zehntel oder mehr. Bisweilen enthält das Intervall die Null, sodass der Fluss Flächen auf einem Teil der Bahn vergrößert und auf einem anderen verkleinert. Trotzdem stimmen das Integral und die Log-Determinante auf mindestens acht Stellen überein. Das entspricht der Genauigkeit des numerischen Verfahrens und der endlichen Differenzen. Die Proposition setzt an keiner Stelle Linearität voraus.

<!--en:4394988b-->
### Der Hutchinson-Spurschätzer

<!--en:baad29a4-->
Die obige Formel für $\operatorname{tr}\mathbf J$ nutzt die Struktur eines Feldes mit einer versteckten Schicht. Bei einem allgemeinen Netz benötigen wir für die Spur die $d$ Diagonaleinträge von $\mathbf J$. Das kostet $d$ Durchläufe im Rückwärtsmodus, einen pro Koordinate. Mit der Spur haben wir die Determinante mit Aufwand $O(d^3)$ bereits durch $d$ Durchläufe ersetzt. Eine zufällige Projektion ersetzt die $d$ Durchläufe durch einen einzigen und liefert eine erwartungstreue, aber verrauschte Schätzung :cite:`Hutchinson.1989`.

<!--en:a6b9d35a-->
**Proposition (Hutchinson-Schätzer).** *Für jede Matrix $\mathbf M\in\mathbb{R}^{d\times d}$ und jeden Zufallsvektor $\boldsymbol\epsilon\in\mathbb{R}^d$ mit $\mathbb{E}[\boldsymbol\epsilon\boldsymbol\epsilon^\top]=\mathbf{I}$ gilt die folgende Gleichung. Beispiele sind ein standardnormalverteilter Vektor und ein Rademacher-Vektor mit unabhängigen Einträgen, die mit gleicher Wahrscheinlichkeit die Werte $\pm1$ annehmen:*

<!--en:3074738e-->
**Beweis.** Es gilt $\mathbb{E}[\boldsymbol\epsilon^\top \mathbf M\boldsymbol\epsilon]
=\mathbb{E}[\operatorname{tr}(\mathbf M\boldsymbol\epsilon\boldsymbol\epsilon^\top)]
=\operatorname{tr}(\mathbf M\,\mathbb{E}[\boldsymbol\epsilon\boldsymbol\epsilon^\top])
=\operatorname{tr}(\mathbf M)$. Dies folgt aus der zyklischen Eigenschaft und der Linearität der Spur. Dank der Linearität dürfen wir Spur und Erwartungswert vertauschen. $\blacksquare$

<!--en:1fd7b7eb-->
Der Schätzer greift auf $\mathbf J$ nur über $\boldsymbol\epsilon^\top \mathbf J$ zu, also über ein Vektor-Jacobi-Produkt. Dafür genügt ein Durchlauf im Rückwärtsmodus durch $\mathbf{f}$ mit dem Startvektor $\boldsymbol\epsilon$. Die Jacobi-Matrix stellen wir nicht auf. Für das Feld mit einer versteckten Schicht kennen wir das Produkt explizit:

<!--en:c53f2a8c-->
Es kostet etwa so viel wie eine Auswertung von $\mathbf{f}$. Ein CNF berechnet bei jeder Feldauswertung ein solches Produkt. Die Likelihood kostet also pro Feldauswertung einen zusätzlichen Rückwärtsdurchlauf. Diesem Verfahren gaben :citet:`Grathwohl.Chen.Bettencourt.ea.2018` den Namen FFJORD.

<!--en:a2050259-->
Nach der Proposition konvergiert der Mittelwert über die Testvektoren gegen die Spur. Wie schnell das geschieht, sagt die Proposition nicht. Für einen einzelnen Testvektor ist aber gerade das entscheidend. Mit $\mathbf S=(\mathbf J+\mathbf J^\top)/2$ beträgt die Varianz bei einem Rademacher-Testvektor $2\sum_{i\ne j}S_{ij}^2$ und bei einem gaußschen Testvektor $2\sum_{i,j}S_{ij}^2$. Letzteres ist dieselbe Summe einschließlich der Diagonale (Aufgabe :exref:`ex_mdl-odes-solvers-hutchinson-variance`). Die Spur hängt nur von der Diagonale von $\mathbf J$ ab. Zum Rauschen eines Rademacher-Testvektors tragen nur Einträge außerhalb der Diagonalen bei.

<!--en:5ed4ca9a-->
Erproben wir den Schätzer an einem Feld in $d=50$ Dimensionen, dessen Spur wir abschätzen können. Es besteht aus einem zufälligen Anteil mit einer versteckten Schicht aus $64$ tanh-Einheiten und einem Zug zum Ursprung: $\mathbf{f}(\mathbf{x})=\mathbf W_2\tanh(\mathbf W_1\mathbf{x}+\mathbf{b}_1)-\mathbf{x}$. Seine Jacobi-Matrix ist $\mathbf W_2\,\mathrm{diag}(\mathbf{s})\,\mathbf W_1-\mathbf{I}$. Der Zug trägt also $-50$ zur Spur bei, der zufällige Anteil einen Wert der Größenordnung eins. Zu jeder Schätzung addiert der Zug den Term $-\boldsymbol\epsilon^\top\boldsymbol\epsilon$. Für einen Rademacher-Testvektor beträgt er exakt $-50$. Das gesamte Rauschen stammt daher vom zufälligen Anteil. Für einen gaußschen Testvektor ist $\boldsymbol\epsilon^\top\boldsymbol\epsilon$ selbst zufällig. Die beiden Varianzformeln zeigen, was das ausmacht. Sie unterscheiden sich um den Diagonalterm $2\sum_iS_{ii}^2$. Hier liegt jedes $S_{ii}=J_{ii}$ nahe bei $-1$. Die Varianz eines gaußschen Testvektors sollte die eines Rademacher-Testvektors also um etwa $2\cdot50=100$ übersteigen. Die Zelle zieht $m=1000$ Testvektoren jeder Art und berechnet jedes Produkt $\boldsymbol\epsilon^\top\mathbf J$ im Rückwärtsmodus.

<!--en:28dcd5a5-->
Wir werten das Vektor-Jacobi-Produkt $\boldsymbol\epsilon_k^\top \mathbf{J}$ im Rückwärtsmodus an der Stelle `x` für alle $m = 1000$ Rademacher-Testvektoren (`E`) und gaußschen Testvektoren (`E_gauss`) aus. Wir vergleichen die Spurschätzungen $\boldsymbol\epsilon_k^\top \mathbf{J}\boldsymbol\epsilon_k$ und ihre Standardabweichungen für einen einzelnen Testvektor mit der exakten Spur `tr_tanh - d`:

<!--en:c4fc66f0-->
Beide Mittelwerte liegen innerhalb weniger Standardfehler um die exakte Spur, wie bei erwartungstreuen Schätzungen zu erwarten. Ein Rademacher-Testvektor ergibt eine Standardabweichung von etwa $5$, einem Zehntel des Betrags der Spur. Ein einziger Rückwärtsdurchlauf bestimmt die Spur also auf etwa zehn Prozent genau, während der exakte Wert $50$ Durchläufe erfordert. Bei einem gaußschen Testvektor beträgt die Standardabweichung etwa $11$. Die Differenz $11^2-5^2=96$ liegt nahe beim vorhergesagten Wert $100$. Rademacher-Testvektoren rauschen nie stärker als gaußsche und hier deutlich weniger.

<!--en:87424c23-->
Die zehn Prozent sind eine Eigenschaft dieses Feldes, nicht des Schätzers. Ohne den Zug zum Ursprung würde sich jede Rademacher-Schätzung um genau $+50$ verschieben. Die Standardabweichung bliebe bei $5$, die Spur hätte die Größenordnung eins, und ein einzelner Testvektor würde über sie kaum etwas aussagen. Welchem der beiden Fälle ein hochdimensionales Modell ähnelt, untersuchen Sie in Aufgabe :exref:`ex_mdl-odes-solvers-hutchinson-variance`.

<!--en:f0b08144-->
Für eine Likelihood genügt eine einzelne Spur nicht. Wir brauchen das Integral $\int_0^T\operatorname{tr}\mathbf J\,dt$ entlang einer Trajektorie mit einer Schätzung pro Integrationsschritt. Dabei können wir einen Testvektor während der gesamten Integration festhalten oder in jedem Schritt einen neuen ziehen. Was erwarten wir? Ein fester Testvektor behandelt die Summe $\sum_nh\,\mathbf J(\mathbf x_n)$ als eine einzige Matrix und schätzt deren Spur einmal. Das Rauschen entspricht daher unabhängig von $N$ dem einer einzelnen Schätzung. Neue Testvektoren erzeugen unabhängige Fehler, jeder mit dem Gewicht $h=1/N$. Das Rauschen ihrer Summe nimmt daher wie $1/\sqrt N$ ab. Die nächste Zelle folgt einer Euler-Trajektorie desselben Feldes bis $T=1$ mit $N=10$, $100$ und $1000$ Schritten. Sie wiederholt jedes Vorgehen $200$-mal mit Rademacher-Testvektoren und berechnet $\boldsymbol\epsilon^\top\mathbf J$ über das oben angegebene explizite Produkt.

<!--en:913446d5-->
Das exakte Integral liegt wieder nahe bei $-50$. Die Mittelwerte beider Verfahren stimmen bis auf den Stichprobenfehler damit überein. Beide Schätzverfahren sind erwartungstreu. Auch die Streuungen fallen wie vorhergesagt aus. Für einen festen Testvektor liegt die Standardabweichung bei jedem $N$ zwischen etwa $5$ und $6$. Mit jeweils neuen Testvektoren liegt sie zwischen $1.5$ und $1.8$ bei $N=10$, zwischen etwa $0.5$ und $0.6$ bei $N=100$ und knapp unter $0.2$ bei $N=1000$. In jeder Zeile entspricht das dem Wert für den festen Testvektor geteilt durch $\sqrt N$.

<!--en:d6fa6158-->
Bei $N=1000$ rauschen neu gezogene Testvektoren etwa dreißigmal weniger und kosten genauso viel. Warum also einen Testvektor festhalten? Bei festem Testvektor bleibt die Gleichung für das numerische Verfahren eine gewöhnliche Differentialgleichung. Halten wir $\boldsymbol\epsilon$ fest, ist die Schätzung $\boldsymbol\epsilon^\top\mathbf J(\mathbf{x})\boldsymbol\epsilon$ eine deterministische Funktion des Zustands. Jedes numerische Verfahren ist anwendbar, ein adaptives Verfahren kann seinen Fehler kontrollieren, und die adjungierten Gleichungen liefern den Gradienten. Ziehen wir bei jeder Auswertung einen neuen Testvektor, rauscht die rechte Seite. Ein Verfahren, das anhand zweier Auswertungen seine Schrittweite beurteilt, würde zwei zufällige Ziehungen vergleichen. In unserer Zelle konnten wir neue Testvektoren ziehen, weil die Euler-Schritte im Voraus feststanden. FFJORD hält den Testvektor fest und nimmt das Rauschen in Kauf.

<!--en:9e9bfc2e-->
Beim Training ist dieses Rauschen hinnehmbar. Wir brauchen keine genaue, sondern nur eine erwartungstreue Schätzung der Log-Likelihood. Die Schätzung geht linear in die Log-Likelihood ein. Daher sind auch die geschätzte Log-Likelihood und ihr Gradient erwartungstreu. Mehr verlangt ein stochastischer Gradient nicht (:numref:`sec_mdl-stochastic-gradient-methods`). Das Rauschen der Testvektoren mittelt sich wie das Rauschen des Minibatchs selbst über den Minibatch und die Iterationen heraus. Für die Likelihood eines einzelnen Punkts reicht ein fester Testvektor jedoch nicht aus. Ein Fehler von $5$ in der Log-Likelihood entspricht einem Faktor $e^5\approx150$ in der Dichte. Hierfür mitteln wir daher über viele Testvektoren, ziehen bei festem Zeitgitter in jedem Schritt neue oder berechnen die Spur exakt.

<!--en:879b4963-->
In diesem Abschnitt stammte das gesamte Rauschen von den Testvektoren. Die Trajektorien selbst waren deterministisch. Ein Diffusionsmodell fügt dem Zustand in jedem Schritt neues Rauschen hinzu. Zuwächse der Größenordnung $\sqrt{h}$ machen die hier verwendeten Taylor-Entwicklungen unbrauchbar. Für die Integration einer solchen Gleichung brauchen wir einen eigenen Kalkül (:numref:`sec_mdl-sdes`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:96efdf57-->
Ein Residualblock entspricht einem Euler-Schritt. Verwenden wir ein gemeinsames Feld und lassen die Schrittweite gegen null gehen, erhalten wir eine Neural ODE. Ihr exakter Fluss ist umkehrbar, und das numerische Verfahren legt die Schrittzahl fest. Beide Vorteile haben Grenzen. Auf $\mathbb{R}$ erhält ein Fluss die Ordnung, sodass wir für $x\mapsto-x$ eine zweite Dimension brauchen. Mit zehn Euler-Schritten passen wir beim Training das Programm mit zehn Schritten an. Mit RK4 passen wir den Fluss an, brauchen aber viermal so viele Feldauswertungen. Der Gradient eines Verlusts am Endzustand $\mathbf{x}(T)$ ist das Integral von $(\partial\mathbf{f}/\partial\boldsymbol\theta)^\top\mathbf{a}$ entlang der Trajektorie. Dabei löst die adjungierte Variable $\mathbf{a}$ eine lineare ODE rückwärts vom Endwert $\nabla\ell(\mathbf{x}(T))$ aus. Backpropagation durch Euler liefert den exakten Gradienten des Programms und weicht um $O(h)$ vom Gradienten des Flusses ab. Den Zustand erneut rückwärts zu integrieren spart Speicher, verstärkt aber einen Fehler im gespeicherten Zustand um den Faktor $e^{\lambda T}$, wenn der Modus mit Rate $\lambda$ abklingt. Entlang einer Trajektorie gilt $\tfrac{d}{dt}\log p_t=-\operatorname{tr}\mathbf J$. Hier genügt also eine Spur, wo ein diskreter Fluss eine Determinante braucht. Der Hutchinson-Schätzer bestimmt die Spur mit einem Vektor-Jacobi-Produkt pro Testvektor. Die Schätzung ist erwartungstreu, rauscht aber. Beim Training mittelt sich das Rauschen heraus, bei einer einzelnen Likelihood nicht.

<!--en:2294243a-->
## Übungen

<!--en:858e94b6-->
1. [code] **Was ein Fluss nicht darstellen kann.** ● Zwei Trajektorien eines Feldes mit Lipschitz-Konstante $L$ erfüllen $e^{-Lt}\|\mathbf{x}-\mathbf{y}\|\le\|\Phi_t(\mathbf{x})-\Phi_t(\mathbf{y})\|\le e^{Lt}\|\mathbf{x}-\mathbf{y}\|$ (Aufgabe :exref:`ex_mdl-odes-solvers-trajectory-separation` in :numref:`sec_mdl-odes-solvers`). In :numref:`subsec_mdl-flow-order` haben wir gesehen, dass kein Fluss auf $\mathbb{R}$ die Reihenfolge umkehrt :cite:`Dupont.Doucet.Teh.2019`.
    1. Sei $d=1$. Geben Sie mithilfe der unteren Schranke einen zweiten Beweis dafür, dass $\Phi_T$ streng monoton steigt, und zeigen Sie $\Phi_T(1)-\Phi_T(-1)\ge2e^{-LT}$. Welche der beiden Aussagen, die zur Reihenfolge oder die zum Abstand, liefert die Proposition im Text? Für welche brauchen Sie die Schranke?
    1. Im Trainingsbeispiel werden die Kreispunkte $(0,\pm1)$ mit Abstand $2$ auf $(1.5,\pm0.4)$ mit Abstand $0.8$ abgebildet. Zeigen Sie, dass der exakte Fluss zur Zeit $T$ eines Feldes mit Lipschitz-Konstante $L$ dies nur für $LT\ge\ln2.5\approx0.92$ leisten kann. Die Streckung der langen Achse von $2$ auf $2.4$ ergibt dagegen nur $LT\ge\ln1.2$. Für ein tanh-Feld mit einer versteckten Schicht gilt $L\le\|\mathbf W_2\|_2\|\mathbf W_1\|_2$, weil $\tanh$ die Lipschitz-Konstante $1$ hat. Berechnen Sie diese Schranke für das mit RK4 trainierte Feld und überprüfen Sie die Ungleichung.
    1. Betten Sie $\mathbb{R}$ in $\mathbb{R}^2$ durch $x\mapsto(x,0)$ ein. Zeigen Sie, dass für jedes lineare Feld $\dot{\mathbf{x}}=\mathbf C\mathbf{x}$, dessen Fluss zur Zeit eins $(x,0)$ auf $(-x,0)$ abbildet, und zwar für jedes $x$, die Gleichung $e^{\mathbf C}=-\mathbf{I}$ gilt. *Hinweis: Ein Eigenwert $-1$ von $e^{\mathbf C}$ entsteht aus einem Eigenwert $\mu$ von $\mathbf C$ mit $e^{\mu}=-1$. Dieser Eigenwert der Ausgangsmatrix ist nicht reell.* Finden Sie ein solches Feld, dessen Trajektorien Ellipsen statt Kreise sind.
    1. Ändern Sie die Trainingszelle so, dass sie die Abbildung $x\mapsto-x$ an $64$ gleichmäßig verteilten Punkten aus $[-1,1]$ anpasst. Verwenden Sie ein Feld mit Eingabe- und Ausgabedimension $1$ sowie $N=10$ Euler-Schritte. Was gibt das trainierte Modell aus? Warum pendelt sich der Verlust bei einem Wert ein, den Sie schon vor dem Training vorhersagen können? Wiederholen Sie den Versuch mit einem Feld der Dimension $2$, den Eingaben $(x,0)$ und dem Verlust für die erste Koordinate.
    1. Trainieren Sie das eindimensionale Modell erneut mit $N=1$ und $h=1$. Werten Sie es mit $N=1$ und mit $N=100$ Schritten der Weite $h=0.01$ aus. Setzen Sie beide Ergebnisse in Beziehung zur Proposition und zur Bedingung $hL<1$.
   :exlabel:`ex_mdl-odes-solvers-flow-expressivity`

<!--en:673b0e48-->
1. **Adjungierte Gleichung mit laufendem Verlust.** ● Die adjungierten Gleichungen :eqref:`eq_mdl-ode-adjoint` und :eqref:`eq_mdl-ode-adjoint-grad` behandeln einen Verlust am Endpunkt. Regularisierungsterme wie die kinetische Energie $\int_0^T\|\mathbf{f}(\mathbf{x}(t),\boldsymbol\theta,t)\|^2\,dt$ bei :citet:`Finlay.Jacobsen.Nurbekyan.ea.2020` summieren Kosten entlang der Trajektorie. Sei

   $$
   \mathcal{L} = \ell_T(\mathbf{x}(T)) + \int_0^T \ell(\mathbf{x}(t), \boldsymbol\theta, t)\,dt .
   $$

    1. Erweitern Sie den Variationsbeweis von :eqref:`eq_mdl-ode-adjoint`. Zeigen Sie, dass die adjungierte Gleichung einen Quellterm erhält, also $\dot{\mathbf{a}} = -(\partial\mathbf{f}/\partial\mathbf{x})^\top\mathbf{a} - \nabla_{\mathbf{x}}\ell$ mit $\mathbf{a}(T) = \nabla\ell_T(\mathbf{x}(T))$ gilt, und dass $\partial\mathcal{L}/\partial\boldsymbol\theta = \int_0^T \big((\partial\mathbf{f}/\partial\boldsymbol\theta)^\top\mathbf{a} + \nabla_{\boldsymbol\theta}\ell\big)\,dt$ folgt.
    1. :citet:`Finlay.Jacobsen.Nurbekyan.ea.2020` ergänzen den Zustand stattdessen um eine Koordinate $E$ mit $\dot{E} = \ell$ und $E(0) = 0$ und addieren $E(T)$ zum Endverlust. Wenden Sie :eqref:`eq_mdl-ode-adjoint` auf den erweiterten Zustand $(\mathbf{x},E)$ an und zeigen Sie, dass sich das Ergebnis aus (a) ergibt.
    1. Für die kinetische Energie gilt $\ell = \|\mathbf{f}\|^2$. Schreiben Sie $\nabla_{\mathbf{x}}\ell$ als Vektor-Jacobi-Produkt. Wie viele Durchläufe im Rückwärtsmodus durch $\mathbf{f}$ benötigt dann ein Schritt der adjungierten ODE?
   :exlabel:`ex_mdl-odes-solvers-running-loss-adjoint`

<!--en:7e8e8432-->
1. [code] **Rückwärtsrekonstruktion der Trajektorie.** ● Bei der kontinuierlichen adjungierten Methode müssen wir die Zustände des Vorwärtsdurchlaufs nicht speichern, wenn wir $\mathbf{x}(t)$ von $\mathbf{x}(T)$ aus zusammen mit $\mathbf{a}(t)$ rückwärts integrieren. Wie :numref:`sec_mdl-adjoint-method` zeigt, verstärkt dann ein mit Rate $\lambda$ abklingender Modus einen Fehler $\delta$ im gespeicherten Endzustand $\mathbf{x}(T)$ auf $e^{\lambda T}\delta$.
    1. Sei $\mathbf R$ die Drehung um $0.6$ Radiant, und sei $\mathbf C=\mathbf R\,\mathrm{diag}(-50,-1)\,\mathbf R^\top$. Integrieren Sie $\dot{\mathbf{x}}=\mathbf C\mathbf{x}$ von $(1,1)$ bis zur Zeit $T$ mit $2000$ RK4-Schritten (`solve` mit `rk4_step`). Integrieren Sie anschließend vom Ergebnis aus über dieselbe Zeitspanne mit negativem Zeithorizont rückwärts. Geben Sie den Rekonstruktionsfehler für $T=0.25$, $0.5$, $1$ und $2$ an und vergleichen Sie sein Wachstum mit $e^{50T}\delta$. Welches $\delta$ passt? Warum liegt es näher bei $10^{-17}$ als beim Wert $10^{-16}$ aus dem Text? *Hinweis: Wie groß sind die Koordinaten von $\mathbf{x}(T)$?*
    1. Wiederholen Sie den Versuch mit $\mathrm{diag}(-50,-1)$ selbst. Ab $T=0.5$ erhalten Sie so den Anfangspunkt wesentlich genauer zurück als mit $\mathbf C$. Warum?
    1. Aktivierungs-Checkpointing (:numref:`subsec_activation_checkpointing`) speichert den Zustand am Anfang jedes Abschnitts der Trajektorie. Ein gemischtes Verfahren rekonstruiert die Zustände innerhalb eines Abschnitts durch Rückwärtsintegration vom Checkpoint an dessen Ende. Wie lang darf bei der Matrix $\mathbf C$ aus (a) ein Abschnitt höchstens dauern, damit die Drift $e^{50T}\delta$ bei $\delta=10^{-16}$ unter $10^{-8}$ bleibt?
   :exlabel:`ex_mdl-odes-solvers-backward-reconstruction`

<!--en:d7fbb18e-->
1. **Log-Determinante eines Residualschritts.** ●● Die infinitesimale Variablentransformation :eqref:`eq_mdl-ode-instant-cov` ergibt sich im Grenzübergang zu unendlich kleinen Schritten. Für einen endlichen Residualschritt $F(\mathbf{x})=\mathbf{x}+\mathbf{g}(\mathbf{x})$ benötigen wir die exakte Log-Determinante wie bei den umkehrbaren Residualnetzen von :citet:`Behrmann.Grathwohl.Chen.ea.2019`.
    1. Sei $\mathbf{g}$ Lipschitz-stetig mit Konstante $c<1$. Für $d=1$ zeigt das kurze Argument aus :numref:`subsec_mdl-flow-order` (ein Euler-Schritt ist für $hL<1$ streng monoton steigend), dass $F$ streng monoton steigend und damit injektiv ist. Schreiben Sie für beliebiges $d$ die Gleichung $F(\mathbf{x})=\mathbf{y}$ als Fixpunktgleichung für $\mathbf{x}$. Zeigen Sie wie im Beweis des Satzes von Picard-Lindelöf mit dem Banachschen Fixpunktsatz, dass $F$ eine Bijektion auf $\mathbb{R}^d$ ist. Wie schnell konvergiert die Fixpunktiteration?
    1. Zeigen Sie $\|\mathbf J_{\mathbf{g}}\|\le c$, wobei $\mathbf J_{\mathbf{g}}$ die Jacobi-Matrix von $\mathbf{g}$ ist. Zeigen Sie anschließend mithilfe der Eigenwerte von $\mathbf J_{\mathbf{g}}$ die Identität

       $$
       \log\det(\mathbf{I} + \mathbf J_{\mathbf{g}})
       = \sum_{k=1}^\infty \frac{(-1)^{k+1}}{k}\operatorname{tr}\big(\mathbf J_{\mathbf{g}}^k\big).
       $$

    1. Für einen Euler-Schritt gelten $\mathbf{g}=h\mathbf{f}$ und $\mathbf J_{\mathbf{g}}=h\mathbf J$. Vergleichen Sie $\log\det(\mathbf{I}+h\mathbf J)$ mit dem Term $h\operatorname{tr}\mathbf J$, den :eqref:`eq_mdl-ode-cnf-likelihood` in jedem Schritt aufsummiert. Wie groß ist die Differenz pro Schritt und wie groß ist sie insgesamt über $T/h$ Schritte?
    1. Der Hutchinson-Schätzer :eqref:`eq_mdl-ode-hutchinson` schätzt $\operatorname{tr}(\mathbf J_{\mathbf{g}}^k)$ durch $\boldsymbol{\epsilon}^\top \mathbf J_{\mathbf{g}}^k\boldsymbol{\epsilon}$. Wie viele Vektor-Jacobi-Produkte pro Testvektor kostet die nach $K$ Gliedern abgeschnittene Reihe?
   :exlabel:`ex_mdl-odes-solvers-residual-logdet`

<!--en:ffb74c1c-->
1. [code] **Varianz des Hutchinson-Schätzers.** ● Der Schätzer :eqref:`eq_mdl-ode-hutchinson` schätzt $\operatorname{tr}\mathbf M$ durch $\boldsymbol{\epsilon}^\top \mathbf M\boldsymbol{\epsilon}$ mit einem einzigen Testvektor $\boldsymbol{\epsilon}$. Sei $\mathbf S=(\mathbf M+\mathbf M^\top)/2$, sodass $\boldsymbol{\epsilon}^\top \mathbf M\boldsymbol{\epsilon}=\boldsymbol{\epsilon}^\top \mathbf S\boldsymbol{\epsilon}$ gilt.
    1. Zeigen Sie, dass die Varianz von $\boldsymbol{\epsilon}^\top \mathbf M\boldsymbol{\epsilon}$ für gaußsche Testvektoren $2\|\mathbf S\|_F^2$ und für Rademacher-Testvektoren $2\sum_{i\neq j}S_{ij}^2$ beträgt. Für welche Matrizen liefert schon ein einziger Rademacher-Testvektor den exakten Wert?
    1. Stellen Sie die Jacobi-Matrix $\mathbf J=\mathbf W_2\,\mathrm{diag}(\mathbf{s})\,\mathbf W_1-\mathbf{I}$ des Feldes mit Zug zum Ursprung explizit auf. Verwenden Sie die Gewichte und den Punkt aus dem Hutchinson-Experiment (die Matrix hat die Größe $50\times50$). Berechnen Sie beide Varianzen und vergleichen Sie ihre Quadratwurzeln mit den beiden im Experiment ausgegebenen Standardabweichungen für einen Testvektor. Wie viele Testvektoren aus jeder Verteilung benötigen Sie für einen Standardfehler von $0.01$?
    1. In einem CNF ist $\mathbf M$ die Jacobi-Matrix $\partial\mathbf{f}/\partial\mathbf{x}$ in Dimension $d$. Nehmen Sie an, alle Einträge haben vergleichbare Beträge. Wie skaliert der relative Standardfehler $\sqrt{\textrm{Var}}/|\operatorname{tr}\mathbf M|$ eines Testvektors mit $d$, wenn alle Diagonaleinträge dasselbe Vorzeichen haben? Wie skaliert er bei zufälligen Vorzeichen?
    1. Beim Training hat jeder der $B$ Punkte eines Minibatchs eine eigene Trajektorie und einen eigenen festen Testvektor. Wie hängt das Rauschen der mittleren Log-Likelihood von $B$ ab? Bei welchem $B$ ist es genauso groß wie bei einem einzelnen Punkt mit neuen Testvektoren in jedem der $N=100$ Schritte?
   :exlabel:`ex_mdl-odes-solvers-hutchinson-variance`

<!--en:f0af9483-->
1. [code] **Diskrete und kontinuierliche adjungierte Methode.** ●● Backpropagation durch ein numerisches Verfahren differenziert das ausgeführte Programm. Wenden wir dagegen ein numerisches Verfahren auf die adjungierte ODE :eqref:`eq_mdl-ode-adjoint` an, nähern wir den Gradienten des exakten Flusses an :cite:`Kidger.2022`. Vergleichen Sie die beiden Rekursionen Schritt für Schritt für ein autonomes Feld mit Jacobi-Matrix $\mathbf J(\mathbf{x})=\partial\mathbf{f}/\partial\mathbf{x}$ und Zuständen $\mathbf{x}_n$ aus dem Vorwärtsdurchlauf.
    1. Beim Vorwärts-Euler-Verfahren berechnet die diskrete adjungierte Gleichung :eqref:`eq_mdl-ode-discrete-adjoint` den Wert $\mathbf{a}_n=(\mathbf{I}+h\mathbf J(\mathbf{x}_n))^\top\mathbf{a}_{n+1}$. Ein expliziter Euler-Schritt für die adjungierte ODE von $t_{n+1}$ zurück nach $t_n$ berechnet dagegen $\mathbf{a}_n=\mathbf{a}_{n+1}+h\mathbf J(\mathbf{x}_{n+1})^\top\mathbf{a}_{n+1}$. Geben Sie für ein glattes $\mathbf{f}$ eine Schranke für die Differenz der Aktualisierungen an. Bei einem linearen Feld stimmen beide überein. Nach dem Text entspricht die verbleibende Abweichung vom Gradienten des Flusses dann dem Fehler des Euler-Verfahrens selbst. Präzisieren Sie dies für das skalare Programm $x_N=(1+h\theta)^Nx_0$. Zeigen Sie, dass die Rekursion den im Text gefundenen Gradienten $T\,x_N^2/(1+h\theta)$ berechnet und dass dieser gleich $T\,x(T)^2\big(1-h\theta(1+\theta T)\big)+O(h^2)$ ist.
    1. Differenzieren Sie einen Heun-Schritt $\mathbf{x}_{n+1}=\mathbf{x}_n+\tfrac{h}{2}\big[\mathbf{f}(\mathbf{x}_n)+\mathbf{f}(\mathbf{u}_n)\big]$ mit $\mathbf{u}_n=\mathbf{x}_n+h\mathbf{f}(\mathbf{x}_n)$, um seine diskrete adjungierte Rekursion herzuleiten. Zeigen Sie, dass diese Rekursion einem Heun-Schritt der adjungierten ODE von $t_{n+1}$ nach $t_n$ entspricht, bei dem $\mathbf J(\mathbf{x}_{n+1})$ durch $\mathbf J(\mathbf{u}_n)$ ersetzt wird. In welcher Ordnung in $h$ unterscheiden sich die beiden Aktualisierungen?
    1. Bei einem linearen Feld $\mathbf{f}(\mathbf{x})=\mathbf A\mathbf{x}$ multipliziert ein RK4-Schritt den Zustand mit $R(h\mathbf A)$. Dabei ist $R(z)=1+z+z^2/2+z^3/6+z^4/24$ die Stabilitätsfunktion von RK4 (siehe :numref:`sec_mdl-euler-runge-kutta`, wo Aufgabe :exref:`ex_mdl-odes-solvers-rotating-modes` sie auf der imaginären Achse untersucht). Zeigen Sie, dass der diskrete adjungierte Schritt von RK4 ein RK4-Schritt der adjungierten ODE ist.
    1. Ersetzen Sie den Euler-Schritt in `loss_and_grad_unrolled` durch einen RK4-Schritt. Berechnen Sie den Gradienten des Verlusts für das entrollte Programm mit zentralen endlichen Differenzen statt mit einem von Hand implementierten Rückwärtsdurchlauf. Messen Sie die Abweichung vom Gradienten `g_cont` der kontinuierlichen adjungierten Methode für $N=5$, $10$, $20$, $40$ und $80$ Schritte. Die Trapezregel zur Berechnung von `g_cont` verwendet `n_quad = 2000` Intervalle und ist nur bis auf etwa $10^{-7}$ genau. Verfeinern Sie daher zuerst die Unterteilung. Welche Steigung messen Sie? Welche Ordnung erbt der Gradient, den Sie durch ein numerisches Verfahren der Ordnung $p$ berechnen?
   :exlabel:`ex_mdl-odes-solvers-re-derive-adjoint-equations`

<!--en:cd6fda23-->
1. [code] **Liouville-Formel.** ●● Bei der infinitesimalen Variablentransformation haben wir eine Dichte entlang einer Trajektorie verfolgt. In dieser Aufgabe beweisen Sie dieselbe Formel auf einem zweiten Weg über die Jacobi-Matrix $\mathbf M(t)=D\Phi_t(\mathbf{x}_0)$ der Flussabbildung selbst.
    1. Differenzieren Sie $\tfrac{d}{dt}\Phi_t(\mathbf{x}_0)=\mathbf{f}(\Phi_t(\mathbf{x}_0),t)$ nach $\mathbf{x}_0$, um die Variationsgleichung $\dot{\mathbf M}=\mathbf J(\mathbf{x}(t),t)\,\mathbf M$ mit $\mathbf M(0)=\mathbf{I}$ zu erhalten.
    1. Beweisen Sie die Jacobi-Formel $\tfrac{d}{dt}\det \mathbf M=\operatorname{tr}(\mathbf M^{-1}\dot{\mathbf M})\det \mathbf M$ für invertierbare $\mathbf M(t)$. *Hinweis: Schreiben Sie $\det \mathbf M(t+h)=\det \mathbf M(t)\,\det\big(\mathbf{I}+h\mathbf M^{-1}\dot{\mathbf M}+O(h^2)\big)$ und wenden Sie das Lemma an.*
    1. Folgern Sie $\log|\det D\Phi_T(\mathbf{x}_0)|=\int_0^T\operatorname{tr}\mathbf J(\mathbf{x}(t),t)\,dt$. Leiten Sie :eqref:`eq_mdl-ode-cnf-likelihood` aus der diskreten Variablentransformation :eqref:`eq_mdl-ode-discrete-cov` her, indem Sie diese auf die Bijektion $\Phi_T$ anwenden. Für ein lineares Feld ergibt sich die Formel :eqref:`eq_mdl-ode-jacobi`.
    1. Integrieren Sie $\dot{\mathbf M}=\mathbf J\mathbf M$ mit RK4 zusammen mit dem tanh-Feld mit $8$ Einheiten aus :numref:`sec_mdl-continuous-normalizing-flows`. Verwenden Sie die Zelle zur Überprüfung des Spurintegrals, bevor das Hutchinson-Experiment die Variablennamen der Gewichte erneut verwendet. Vergleichen Sie $\log|\det \mathbf M(1)|$ mit dem Spurintegral und der Log-Determinante aus der Zellenausgabe.
   :exlabel:`ex_mdl-neural-odes-liouville`

<!--en:d7268f3f-->
[Diskussion](https://d2l.discourse.group/t/36868)

<!--en:089815a4-->
Neural ODEs, adjungierte Sensitivität und kontinuierliche Flüsse<br>**ResNets mit kontinuierlicher Tiefe · adjungierte ODE nach Pontryagin · Spurregel von Liouville · Hutchinson-Schätzung in FFJORD**

<!--en:7b8ccb28-->
::: {.slide title="Warum diskrete normalisierende Flüsse an die O(d³)-Grenze stoßen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:c0c27b4b-->
Das Maximum-Likelihood-Training einer umkehrbaren diskreten Schicht $\mathbf{g} : \mathbb{R}^d \to \mathbb{R}^d$ erfordert die Jacobi-Determinante:

<!--en:adfa5c5d-->
- **$O(d^3)$ für die Determinante:** Bei einem RGB-Bild mit $32 \times 32$ Pixeln ($d = 3072$) kostet $\det\mathbf{J}_{\mathbf{g}}$ $\approx 2 \times 10^{10}$ FLOPs pro Schicht. Das erzwingt Einschränkungen wie Dreiecksstruktur oder Kopplungsschichten.

<!--en:110c1108-->
- **Ausweg in kontinuierlicher Zeit:** Tritt $\dot{\mathbf{x}} = \mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ an die Stelle der diskreten Schichten, wird aus $\log|\det\mathbf{J}|$ das Integral der **Spur** $-\int_0^T \operatorname{tr}(\mathbf{J})\,dt$. 1 Vektor-Jacobi-Produkt (VJP) liefert eine erwartungstreue Schätzung.

<!--en:8aae0207-->
::: {.slide title="Residualnetze als Diskretisierungen mit dem Vorwärts-Euler-Verfahren"}

<!--en:9d2ab3cf-->
[Von ResNets zu ODEs]{.kicker}

<!--en:e1dec338-->
Der Vergleich eines Residualblocks mit einem Vorwärts-Euler-Schritt für eine ODE zeigt: Ein ResNet ist ein diskretisiertes dynamisches System mit Schrittweite $h = 1$:

<!--en:5ab5a57a-->
- **Diskreter Residualblock ($l = 0, \ldots, L-1$):**
  $$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathbf{f}_{\boldsymbol{\theta}_l}(\mathbf{x}_l)$$

<!--en:e82272b9-->
- **Vorwärts-Euler-Schritt ($t_n = nh$):**
  $$\mathbf{x}_{n+1} = \mathbf{x}_n + h\,\mathbf{f}(\mathbf{x}_n, t_n)$$

<!--en:d697a073-->
- **Qualitativer Unterschied zwischen Schritten und Flüssen:** Für $\dot{x} = -x$ ist der exakte Fluss $\Phi_t(x) = e^{-t}x$ für jedes $t \in \mathbb{R}$ eine glatte Bijektion. Ein einziger Euler-Schritt mit $h = 1$ bildet dagegen $x \mapsto x - x = 0$ ab und führt so ganz $\mathbb{R}$ auf den Ursprung.

<!--en:a6b72098-->
::: {.slide title="Die Neural ODE als Grenzfall: vier Vorteile und ihre Grenzen"}

<!--en:e87e59a5-->
[Das kontinuierliche Modell]{.kicker}

<!--en:14312d2d-->
Beim Übergang zum kontinuierlichen Grenzfall $\mathbf{x}(T) = \mathbf{x}_0 + \int_0^T \mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}(t), t)\,dt$ stehen vier theoretischen Vorteilen vier praktische Einschränkungen gegenüber:

<!--en:a30a93f2-->
- **1. Bijektiver Fluss $\Phi_T$** $\leftrightarrow$ Trajektorien ohne Schnittpunkte erhalten in einer Dimension die Reihenfolge.

<!--en:dd33f6ba-->
- **2. Adaptives Verfahren zur Testzeit** $\leftrightarrow$ Das Training mit groben Euler-Schritten passt ein diskretes Programm an.

<!--en:7418d847-->
- **3. Adjungierte ODE mit Speicherbedarf $O(1)$** $\leftrightarrow$ Die Rückwärtsintegration des Zustands verstärkt Rundungsfehler um den Faktor $e^{\lambda T}$.

<!--en:decef7ec-->
- **4. Likelihood aus der Spur $-\operatorname{tr}(\mathbf{J})$** $\leftrightarrow$ Hutchinson-Testvektoren bringen Monte-Carlo-Varianz mit sich.

<!--en:1109e72b-->
::: {.slide title="Einschränkung 1: Kontinuierliche Flüsse auf der Geraden erhalten die Ordnung"}

<!--en:00351b3f-->
[Topologische Ausdrucksstärke]{.kicker}

<!--en:795a5d87-->
Die Trajektorien einer Lipschitz-stetigen ODE können sich im Raum-Zeit-Diagramm nicht schneiden. Eine Neural ODE in einer Dimension ist daher weniger ausdrucksstark als ein Residualblock mit nur 1 Schritt:

<!--en:c6a0cba0-->
- **Strenge Monotonie auf $\mathbb{R}$:** Für $x < y$ treffen sich die Trajektorien unter $\dot{z} = f(z, t)$ nie. Das folgt aus der Eindeutigkeit nach Picard-Lindelöf, rückwärts von einem angenommenen Schnittpunkt aus angewandt. Daher gilt $\Phi_T(x) < \Phi_T(y)$ für alle $T > 0$.

<!--en:9ddf84aa-->
- **Warum die Spiegelung $x \mapsto -x$ in einer Dimension scheitert:** Wer $-1$ und $+1$ vertauscht, kehrt ihre Reihenfolge um. Ein diskreter Residualschritt $x + f(x)$ mit $f(x) = -2x$ realisiert $x \mapsto -x$, weil er mit $hL = 2 > 1$ über den Schnittpunkt hinwegspringt.

<!--en:48e010a7-->
- **Erweiterung auf zwei Dimensionen:** Die Einbettung $x \mapsto (x, 0) \in \mathbb{R}^2$ und die Drehung gemäß $\dot{\mathbf{x}} = \pi\mathbf{K}\mathbf{x}$ ergeben $\Phi_1 = e^{\pi\mathbf{K}} = -\mathbf{I}$. Damit bewegen sich die Punkte gemäß $(x, 0) \to (-x, 0)$ auf Halbkreisen aneinander vorbei.

<!--en:4505c23d-->
::: {.slide title="Erst diskretisieren, dann optimieren: Training durch ein numerisches Verfahren"}

<!--en:aec61804-->
[Trainingsaufbau]{.kicker}

<!--en:31dc0fee-->
Ziel ist die Abbildung von $64$ Punkten des Einheitskreises auf eine Ellipse mit Mittelpunkt $(1.5, 0)$ und Halbachsen $(1.2, 0.4)$. Das `tanh`-Netz mit 32 Einheiten lernt durch Backpropagation direkt durch $N = 10$ entrollte Schritte ($h = 0.1$) von **Euler** oder **RK4**:

<!--en:87472d31-->
Bei der im Training verwendeten Zielfunktion mit 10 Schritten senken beide Trainingsläufe den mittleren quadratischen Endpunktfehler von $O(1)$ auf unter $10^{-5}$.

<!--en:365e1832-->
::: {.slide title="Einschränkung 2: Lernt das Modell einen kontinuierlichen Fluss oder ein Programm?"}

<!--en:50dded7f-->
[Feinere Schritte zur Testzeit]{.kicker}

<!--en:6069aa9b-->
Was geschieht bei höherer Schrittzahl ($N = 100$ und $N = 1000$ bei festem $Nh = 1$), wenn sich die berechneten Abbildungen den exakten kontinuierlichen Flüssen zur Zeit 1 nähern? Wie gut macht die Rückwärtsintegration den Hinweg rückgängig?

<!--en:8f14f44b-->
- **Euler trainiert ein Programm mit 10 Schritten:** Bei kleinerer Schrittweite wird das mit Euler trainierte Netz *schlechter* (der Fehler steigt von $9 \times 10^{-4}$ auf $2 \times 10^{-2}$). Seine Gewichte haben den Abbruchfehler der Ordnung $h = 0.1$ bei groben Schritten ausgeglichen.

<!--en:801b1b5b-->
- **RK4 trainiert einen Fluss:** Schon 10 RK4-Schritte waren bis auf die verwendete Genauigkeit exakt. Daher bleibt der Endpunktfehler bei etwa $10^{-3}$, unabhängig von der Schrittzahl $N \in \{10, 100, 1000\}$.

<!--en:58243b4e-->
- **Fehler auf dem diskreten Hin- und Rückweg:** Nach $N$ Euler-Schritten vorwärts und $N$ Euler-Schritten rückwärts beträgt die Abweichung von $\mathbf{x}_0$ noch $O(h) = O(1/N)$ ($5.5 \times 10^{-2} \to 5.5 \times 10^{-4}$).

<!--en:0d3a0d4e-->
::: {.slide title="Drei Ansätze zum Training kontinuierlicher Vektorfelder"}

<!--en:1498b5e7-->
[Rechenaufwand und Genauigkeit]{.kicker}

<!--en:b9133b2a-->
- **Entrolltes Vorwärts-Euler-Verfahren ($N = 10$):** Erfordert $10$ Vorwärts- $+ 10$ Rückwärtsauswertungen und Speicher der Ordnung $O(N)$. Es passt die grobe Diskretisierung mit 10 Schritten an (eine Verfeinerung von $N$ zur Testzeit erhöht den Fehler).

<!--en:fbecd518-->
- **Entrolltes klassisches RK4-Verfahren ($N = 10$):** Erfordert $40$ Vorwärts- $+ 40$ Rückwärtsauswertungen. Bei allen zur Testzeit verwendeten Verfahren stimmt das Ergebnis bis auf $O(h^4)$ mit dem exakten kontinuierlichen Fluss überein.

<!--en:3732bd81-->
- **Score- und Flow-Matching (§§{sec_mdl-score-matching-diffusion-flow}–§{num:sec_mdl-flow-matching-transport}):** Regression direkt auf geschlossene Zielausdrücke mit **1 Vorwärts- und 1 Rückwärtsauswertung** bei Speicherbedarf $O(1)$, ohne ODE-Löser in der Trainingsschleife.

<!--en:1240b7f9-->
::: {.slide title="Kontinuierliche Backpropagation: der adjungierte Zustand a(t)"}

<!--en:7136f201-->
[Die adjungierte Methode]{.kicker}

<!--en:5ef57e8d-->
In einem diskreten Netz überträgt automatisches Differenzieren im Rückwärtsmodus die Sensitivität $\mathbf{a}_l = \frac{\partial\mathcal{L}}{\partial\mathbf{x}_l}$ von Schicht $l+1$ zurück zu Schicht $l$. In kontinuierlicher Zeit beschreibt der **adjungierte Zustand** diese Sensitivität zu jedem Zeitpunkt $t \in [0, T]$:

<!--en:4c731ce5-->
- **Skalares Beispiel ($\dot{x} = \theta x$, $x(0) = x_0$, $\mathcal{L} = \frac{1}{2}x(T)^2$):**
  - **Direktes Differenzieren:** Wegen $x(T) = e^{\theta T}x_0$ gilt $\frac{\partial\mathcal{L}}{\partial\theta} = x(T)\frac{\partial x(T)}{\partial\theta} = T\,x(T)^2$ ($= e^2 \approx 7.389$ bei $\theta = T = x_0 = 1$).
  - **Sicht der adjungierten Methode:** Eine Zustandsstörung $\delta x(t)$ zur Zeit $t$ verstärkt sich durch den verbleibenden Fluss um den Faktor $e^{\theta(T-t)}$ bis zur Zeit $T$. Also gilt $a(t) = e^{\theta(T-t)}x(T)$. Dies erfüllt die Rückwärtsgleichung $\dot{a}(t) = -\theta a(t)$.

<!--en:23fef549-->
::: {.slide title="Geometrie im skalaren Fall: konstantes Produkt a(t)x(t)"}

<!--en:dcd3b746-->
[Geometrie der adjungierten Variablen]{.kicker}

<!--en:f47ab2d5-->
Eine Parameterstörung $\delta\theta$ ändert die Geschwindigkeit zum Zeitpunkt $t$ um $\frac{\partial f}{\partial\theta}\delta\theta = x(t)\delta\theta$. Die adjungierte Variable $a(t)$ übersetzt diese Änderung in ihre endgültige Wirkung auf $\mathcal{L}$:

<!--en:e18ecb4d-->
- **Vorwärts wachsender Zustand:** $x(t) = e^{\theta t}x_0$ wächst exponentiell von $t = 0$ auf $T$.

<!--en:9ae4bb02-->
- **In Vorwärtsrichtung abklingende adjungierte Variable:** $a(t) = e^{\theta(T-t)}x(T)$ wächst rückwärts von $T$ auf $0$ (von links nach rechts gelesen nimmt sie ab).

<!--en:58c0ff51-->
- **Exakte Aufhebung im Produkt:**
  $$\frac{\partial\mathcal{L}}{\partial\theta} = \int_0^T a(t)\,\frac{\partial f}{\partial\theta}(x(t), \theta)\,dt = \int_0^T x(T)^2\,dt = T\,x(T)^2$$

<!--en:c5422a85-->
::: {.slide title="Die kontinuierlichen adjungierten Gleichungen von Pontryagin"}

<!--en:d0c01ef8-->
[Allgemeiner Satz]{.kicker}

<!--en:0ba99427-->
Für $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \boldsymbol{\theta}, t)$ und den Endverlust $\mathcal{L} = \ell(\mathbf{x}(T))$ ergeben sich die adjungierte Variable $\mathbf{a}(t) = \frac{\partial\mathcal{L}}{\partial\mathbf{x}(t)}$ und der Parametergradient durch Rückwärtsintegration ausgehend von $\mathbf{a}(T) = \nabla\ell(\mathbf{x}(T))$:

<!--en:64fb8d88-->
- **Ein VJP pro Schritt:** Beide Produkte sind gewöhnliche Vektor-Jacobi-Produkte im Rückwärtsmodus mit Startvektor $\mathbf{a}(t)$. Die explizite Jacobi-Matrix der Größe $d \times d$ entfällt.

<!--en:120cf72e-->
::: {.slide title="Variationsbeweis der kontinuierlichen adjungierten Gleichungen"}

<!--en:5d5ccc2a-->
[Herleitung]{.kicker}

<!--en:ae603824-->
Woher kommen das Minuszeichen und die transponierte Jacobi-Matrix $-(\frac{\partial\mathbf{f}}{\partial\mathbf{x}})^\top$ in der adjungierten Rückwärtsgleichung?

<!--en:ab6a3122-->
1. **Parameterstörung linearisieren ($\delta\boldsymbol{\theta}$):** Differenzieren von $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \boldsymbol{\theta}, t)$ zeigt, wie sich die Störung der Trajektorie $\delta\mathbf{x}(t)$ ausgehend von $\delta\mathbf{x}(0) = \mathbf{0}$ entwickelt:
   $$\delta\dot{\mathbf{x}}(t) = \frac{\partial\mathbf{f}}{\partial\mathbf{x}}\,\delta\mathbf{x}(t) + \frac{\partial\mathbf{f}}{\partial\boldsymbol{\theta}}\,\delta\boldsymbol{\theta}$$

<!--en:001f8da6-->
2. **Skalarprodukt $\mathbf{a}(t)^\top\delta\mathbf{x}(t)$ differenzieren:** Nach der Produktregel heben sich bei der Wahl $\dot{\mathbf{a}} = -(\frac{\partial\mathbf{f}}{\partial\mathbf{x}})^\top\mathbf{a}$ die Kopplungsterme des Zustands exakt auf:
   $$\frac{d}{dt}\left(\mathbf{a}^\top\delta\mathbf{x}\right) = \dot{\mathbf{a}}^\top\delta\mathbf{x} + \mathbf{a}^\top\delta\dot{\mathbf{x}} = -\mathbf{a}^\top\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\delta\mathbf{x} + \mathbf{a}^\top\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\delta\mathbf{x} + \mathbf{a}^\top\frac{\partial\mathbf{f}}{\partial\boldsymbol{\theta}}\delta\boldsymbol{\theta} = \mathbf{a}^\top\frac{\partial\mathbf{f}}{\partial\boldsymbol{\theta}}\delta\boldsymbol{\theta}$$

<!--en:642205c9-->
3. **Von $0$ bis $T$ integrieren:** Aus $\delta\mathbf{x}(0) = \mathbf{0}$ und $\mathbf{a}(T)^\top\delta\mathbf{x}(T) = \delta\mathcal{L}$ folgt $\delta\mathcal{L} = \left(\int_0^T \mathbf{a}(t)^\top\frac{\partial\mathbf{f}}{\partial\boldsymbol{\theta}}\,dt\right)\delta\boldsymbol{\theta}$.

<!--en:dd12bc7a-->
::: {.slide title="Erst diskretisieren oder erst optimieren?"}

<!--en:e06cfaa8-->
[Diskrete und kontinuierliche adjungierte Gleichung]{.kicker}

<!--en:d4585228-->
- **Erst diskretisieren, dann optimieren (Backpropagation):** Das Euler-Programm mit $N$ Schritten lässt sich über $\mathbf{a}_n = (\mathbf{I} + h\mathbf{J}(\mathbf{x}_n))^\top\mathbf{a}_{n+1}$ exakt differenzieren (ein Checkpoint alle $\sqrt{N}$ Schritte erfordert Speicher für etwa $2\sqrt{N}$ Zustände).

<!--en:3ab86c91-->
- **Erst optimieren, dann diskretisieren (kontinuierliche adjungierte Gleichung):** $\dot{\mathbf{a}} = -\mathbf{J}^\top\mathbf{a}$ rückwärts integrieren. Bei erneuter Rückwärtsintegration von $\mathbf{x}(t)$ ohne Checkpoints verstärkt ein abklingender Modus $\lambda = 50$ den Rundungsfehler von `fp64` um den Faktor $e^{50T} \approx 5 \times 10^{21}$.

<!--en:42592fb6-->
::: {.slide title="Diskrete und kontinuierliche adjungierte Methode im 2D-Vergleich"}

<!--en:1b718c47-->
[Numerische Überprüfung]{.kicker}

<!--en:07cedd21-->
Für die ebene Spirale $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$ mit Verlust $\mathcal{L} = \frac{1}{2}\|\mathbf{x}(T)\|^2$ bei $T = 2$ ergibt das kontinuierliche adjungierte Integral $\frac{\partial\mathcal{L}}{\partial\mathbf{A}} = \int_0^T \mathbf{a}(t)\mathbf{x}(t)^\top\,dt$:

<!--en:d7204d29-->
::: {.slide title="Einschränkung 3: Rückwärtsintegration verstärkt die Drift um e^(λT)"}

<!--en:0b618086-->
[Speicherbedarf und numerische Drift]{.kicker}

<!--en:29a0a0e2-->
Die Auswertung der Jacobi-Matrix $\mathbf{J}(\mathbf{x}(t))$ im adjungierten Rückwärtsdurchlauf erfordert die Zustände $\mathbf{x}(t)$ des Vorwärtsdurchlaufs. Was geschieht, wenn das Tape entfällt und die Rückwärtsintegration $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, t)$ von $\mathbf{x}(T)$ aus bei Speicherbedarf $O(1)$ rekonstruiert?

<!--en:cd8dfc72-->
- **Kontraktion vorwärts bedeutet Wachstum rückwärts:** Ein vorwärts stabiler Modus $\dot{x} = -\lambda x$ ($\lambda > 0$) wird bei Rückwärtsintegration von $T$ bis $0$ zum instabilen Modus $\dot{x} = +\lambda x$.

<!--en:91ce63f4-->
- **Katastrophale Verstärkung der Rundungsfehler bei doppelter Genauigkeit:** Bei einem steifen Modus $\lambda = 50$ wächst ein `float64`-Rundungsfehler der Größe $\delta \approx 10^{-16}$ im gespeicherten Endzustand $\mathbf{x}(T)$ auf $e^{50T}\delta$ an:
  - Bei $T = 0.25$ beträgt die Drift $e^{12.5} \cdot 10^{-16} \approx 2.7 \times 10^{-11}$.
  - Bei $T = 1.00$ beträgt sie $e^{50} \cdot 10^{-16} \approx 5 \times 10^{5}$ (Trajektorie und Gradient sind völlig unbrauchbar).

<!--en:97aa55fa-->
**Für die Praxis:** **Erst diskretisieren, dann optimieren** (Backpropagation durch das numerische Verfahren) und **Aktivierungs-Checkpointing** alle $\sqrt{N}$ Schritte verwenden. So genügen $2\sqrt{N} \approx 63$ gespeicherte Zustände für $N = 1000$. Die Neuberechnung der Abschnitte läuft *vorwärts*, für jeden Modus in seiner stabilen Richtung.

<!--en:59a580f8-->
::: {.slide title="Liouville-Gleichung: infinitesimale Variablentransformation"}

<!--en:f63dcac2-->
[Kontinuierliche normalisierende Flüsse]{.kicker}

<!--en:f51b625b-->
Über einen kurzen Schritt $h$ erfüllt die Jacobi-Matrix des Flusses $\mathbf{I} + h\mathbf{J} + o(h)$ die Beziehung $\det(\mathbf{I} + h\mathbf{J}) = 1 + h\operatorname{tr}(\mathbf{J}) + O(h^2)$. Der Grenzübergang $h \to 0$ liefert die **Spurregel von Liouville**:

<!--en:a28eef25-->
- **Keine Determinanten der Größe $O(d^3)$:** Zur Integration der Log-Dichte entlang der Trajektorie genügt der Spurterm erster Ordnung.

<!--en:e530ae49-->
::: {.slide title="Divergenz als Volumenkontraktion und Dichteänderung"}

<!--en:9914b045-->
[Geometrische Anschauung]{.kicker}

<!--en:fffe81dd-->
Bei Integration von $-\operatorname{tr}(\mathbf{J})$ über das Intervall von $0$ bis $T$ ersetzt ein einziges Kurvenintegral die Summe der Log-Determinanten von $O(d^3)$ Schichten:

<!--en:277c5376-->
- **Linearer Spezialfall (Jacobi-Formel):** Für $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$ gilt $\det(e^{\mathbf{A}t}) = e^{t\operatorname{tr}\mathbf{A}}$. Bei der kontrahierenden Spirale ($\operatorname{tr}\mathbf{A} = -0.5 - 0.5 = -1$) sinkt die Kreisfläche um den Faktor $e^{-1.2} \approx 0.30$ bei $t = 1.2$, während die Dichte entlang jeder Trajektorie um den Faktor $e^{+1.2} \approx 3.32$ steigt.

<!--en:c98c53ce-->
::: {.slide title="Überprüfung des Spurintegrals entlang eines nichtlinearen Flusses"}

<!--en:1b718c47-2-->
[Numerische Überprüfung]{.kicker}

<!--en:3915a277-->
Im nichtlinearen `tanh`-Vektorfeld $\mathbf{f}(\mathbf{x}) = \mathbf{W}_2\tanh(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)$ mit 8 Einheiten ändert sich die Divergenz $\operatorname{tr}\mathbf{J}(\mathbf{x})$ stetig entlang der Trajektorie. Dennoch stimmt ihr mit RK4 berechnetes Zeitintegral auf 8 Stellen mit der Log-Determinante $\log|\det D\Phi_1(\mathbf{x}_0)|$ aus endlichen Differenzen überein:

<!--en:178fed37-->
::: {.slide title="Hutchinson-Spurschätzer und FFJORD"}

<!--en:892c09d3-->
[Stochastische Spurschätzung]{.kicker}

<!--en:b3110c1f-->
Für jeden isotropen Testvektor $\boldsymbol{\epsilon}$ mit $\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top] = \mathbf{I}$ schätzt der **Hutchinson-Schätzer** die Spur $\operatorname{tr}(\mathbf{J})$ mit **einem** VJP im Rückwärtsmodus:

<!--en:29a8ec71-->
- **Rademacher ($\epsilon_i \in \{\pm 1\}$) gegenüber Gauß im Vorteil:** Da stets $\epsilon_i^2 \equiv 1$ gilt, ist bei Rademacher-Testvektoren der **Varianzbeitrag der Diagonalen null** ($2\sum_{i \neq j} S_{ij}^2$ gegenüber $2\sum_{i,j} S_{ij}^2$ bei gaußschen Testvektoren).

<!--en:e45d9dfc-->
::: {.slide title="Rademacher- und gaußsche Testvektoren im Vergleich bei d = 50"}

<!--en:6326b828-->
[Varianz des Schätzers]{.kicker}

<!--en:65c12617-->
Im Feld $\mathbf{f}(\mathbf{x}) = \mathbf{W}_2\tanh(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) - \mathbf{x}$ in $d = 50$ Dimensionen trägt der Zug $-\mathbf{x}$ zu jedem Diagonaleintrag $J_{ii}$ den Wert $-1$ bei ($\operatorname{tr}\mathbf{J} \approx -50$). Da bei Rademacher-Testvektoren stets $\boldsymbol{\epsilon}^\top\boldsymbol{\epsilon} = d = 50$ gilt, entfällt der zusätzliche Varianzbeitrag der Diagonalen von $2\sum_i J_{ii}^2 \approx 100$ bei gaußschen Testvektoren:

<!--en:ff1dd51c-->
::: {.slide title="Einschränkung 4: feste oder neue Testvektoren entlang einer Bahn?"}

<!--en:49ab8b88-->
[Verfahrenswahl bei FFJORD]{.kicker}

<!--en:126e6c22-->
Bei der Integration von $\int_0^T \boldsymbol{\epsilon}^\top\mathbf{J}(\mathbf{x}(t))\boldsymbol{\epsilon}\,dt$ über $N$ Schritte: einen Testvektor $\boldsymbol{\epsilon}$ für die gesamte Trajektorie festhalten oder in jedem Schritt einen neuen Vektor $\boldsymbol{\epsilon}_n$ ziehen?

<!--en:94dfc910-->
- **Neue Testvektoren ($\text{SD} \propto 1/\sqrt{N}$):** Unabhängige Schrittfehler mitteln sich heraus (die Standardabweichung sinkt von $1.7$ bei $N=10$ auf $0.17$ bei $N=1000$). Auf der rechten Seite der ODE entsteht aber weißes Rauschen, sodass die adaptive Schrittweitensteuerung versagt.

<!--en:70f3774d-->
- **Fester Testvektor (Standard in FFJORD):** Bei festem $\boldsymbol{\epsilon}$ bleibt die erweiterte ODE für RK45 und adjungierte Backpropagation deterministisch und glatt. Minibatch-SGD mittelt den Varianzbeitrag der Testvektoren der Ordnung $O(1)$ über Beispiele und Trainingsiterationen heraus.

<!--en:30c238e5-->
::: {.slide title="Umkehrbare Residualblöcke und kontinuierliche Flüsse"}

<!--en:d2e8dcc7-->
[Reihenentwicklung für diskrete Kontraktionen]{.kicker}

<!--en:c32ff1ee-->
Was ändert sich, wenn die diskreten Residualschritte $F(\mathbf{x}) = \mathbf{x} + \mathbf{g}(\mathbf{x})$ erhalten bleiben und der kontinuierliche Grenzübergang $h \to 0$ entfällt?

<!--en:8a783be6-->
- **Umkehrbarkeit nach dem Banachschen Fixpunktsatz ($c < 1$):** Hat $\mathbf{g}$ die Lipschitz-Konstante $c = \|\mathbf{J}_{\mathbf{g}}\|_2 < 1$ (spektrale Normierung), so ist $\mathbf{x} = \mathbf{y} - \mathbf{g}(\mathbf{x})$ eine Kontraktion. Damit ist $F$ global bijektiv. Die Fixpunktiteration berechnet die Umkehrabbildung mit geometrischer Konvergenzrate $c^k$.

<!--en:f9e34fbb-->
- **Potenzreihe für die Log-Determinante (i-ResNet):** Für $\|\mathbf{J}_{\mathbf{g}}\|_2 \le c < 1$ liefert die Entwicklung von $\log(1 + \lambda_i)$ nach den Eigenwerten von $\mathbf{J}_{\mathbf{g}}$ die diskrete Log-Determinante als unendliche Reihe von Spuren:
  $$\log\det(\mathbf{I} + \mathbf{J}_{\mathbf{g}}) = \sum_{k=1}^\infty \frac{(-1)^{k+1}}{k}\operatorname{tr}\!\left(\mathbf{J}_{\mathbf{g}}^k\right)$$

<!--en:87249839-->
- **Vorteil kontinuierlicher Flüsse:** Ein diskretes umkehrbares ResNet muss $\|\mathbf{J}_{\mathbf{g}}\|_2 < 1$ einhalten und für die abgeschnittene Potenzreihe $K$ aufeinanderfolgende VJPs pro Schicht berechnen. Ein kontinuierlicher normalisierender Fluss benötigt **nur den linearen Spurterm für $k = 1$**, also $h\operatorname{tr}(\mathbf{J})$, ohne eine solche Obergrenze für die Lipschitz-Konstante.

<!--en:a0363035-->
::: {.slide title="Adjungierte Gleichungen mit laufenden Kosten und Regularisierung durch kinetische Energie"}

<!--en:9ec76c86-->
[Erweiterung der adjungierten ODE]{.kicker}

<!--en:54cd0d73-->
In der Praxis ergänzen kontinuierliche normalisierende Flüsse den Endverlust $\ell_T(\mathbf{x}(T))$ häufig um einen über die Bahn integrierten Strafterm $\mathcal{L} = \ell_T(\mathbf{x}(T)) + \int_0^T \ell(\mathbf{x}(t), \boldsymbol{\theta}, t)\,dt$. Das begünstigt geradlinige, nicht steife Trajektorien:

<!--en:e453a38a-->
- **Adjungierte ODE mit laufendem Quellterm:** Im erweiterten Variationsbeweis tritt $-\nabla_{\mathbf{x}}\ell(\mathbf{x}(t), \boldsymbol{\theta}, t)$ als kontinuierliche Quelle auf, die im gesamten Rückwärtsdurchlauf zur Sensitivität beiträgt:
  $$\dot{\mathbf{a}}(t) = -\left(\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\right)^{\!\top}\mathbf{a}(t) - \nabla_{\mathbf{x}}\ell(\mathbf{x}(t), \boldsymbol{\theta}, t), \qquad \mathbf{a}(T) = \nabla\ell_T(\mathbf{x}(T))$$

<!--en:c16d833e-->
- **Regularisierung mit kinetischer Energie und Frobeniusnorm der Jacobi-Matrix (RNODE):** Der Strafterm $\ell(\mathbf{x}, \boldsymbol{\theta}, t) = \|\mathbf{f}(\mathbf{x}, \boldsymbol{\theta}, t)\|^2 + \lambda\|\mathbf{J}(\mathbf{x}, t)\|_F^2$ verhindert gekrümmte, steife Trajektorien. Dadurch benötigen adaptive Verfahren um Größenordnungen weniger Feldauswertungen.

<!--en:2bf2e644-->
::: {.slide title="Verständnisfragen zu Ausdrucksstärke, adjungierten Variablen und Dichten"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:5301bcce-->
Fragen zu Neural ODEs und kontinuierlichen normalisierenden Flüssen:

<!--en:acdd2a93-->
1. **Einschränkung für eindimensionale Flüsse:** Warum kann ein Residualblock mit 1 Schicht $x \mapsto x - 3x = -2x$ die Orientierung der reellen Geraden umkehren, während keine Lipschitz-stetige Neural ODE $\dot{x} = f(x, t)$ auf $\mathbb{R}$ die Abbildung $(-1, 1) \mapsto (2, -2)$ realisieren kann?

<!--en:179fb183-->
2. **Checkpointing und Rückwärtsintegration:** Ein System hat einen schnell abklingenden Modus mit $\lambda = 50$ über $T = 1$. Warum bleibt Aktivierungs-Checkpointing stabil, während eine erneute Rückwärtsintegration des Zustands alle 16 Stellen der `float64`-Genauigkeit verliert?

<!--en:dbfb637a-->
3. **Wahl des Spurschätzers:** Welche Varianz hat für eine diagonale Jacobi-Matrix $\mathbf{J} = \mathrm{diag}(c_1, \ldots, c_d)$ eine Hutchinson-Schätzung $\boldsymbol{\epsilon}^\top\mathbf{J}\boldsymbol{\epsilon}$ mit einem einzelnen **Rademacher-Testvektor**?

<!--en:557b98b6-->
**Lösungen:**

<!--en:45690742-->
1. Um in einer Dimension die Reihenfolge umzukehren, müssten sich die stetigen Trajektorien von $-1$ und $+1$ zu einem Zeitpunkt $s \in (0, T)$ schneiden. Das widerspräche der Eindeutigkeit nach Picard-Lindelöf. Ein diskreter Schritt mit $hL = 3 > 1$ überspringt den Ursprung in einem Satz.

<!--en:f71290ac-->
2. Checkpointing speichert Zustände während des Vorwärtsdurchlaufs und berechnet nur kurze Abschnitte erneut **vorwärts** ($e^{-50\Delta t}$ dämpft Fehler). So entfällt die Rückwärtsintegration, bei der $e^{+50T} \approx 5 \times 10^{21}$ die Rundungsfehler explosionsartig verstärken würde.

<!--en:18858dd5-->
3. **Null.** Für eine Diagonalmatrix gilt $S_{ij} = 0$ für alle $i \ne j$. Daher ergibt schon ein einziger Rademacher-Testvektor exakt $\sum_i \epsilon_i^2 c_i = \sum_i (\pm 1)^2 c_i = \sum_i c_i = \operatorname{tr}(\mathbf{J})$.

<!--en:e58b2a6d-->
::: {.slide title="Zusammenfassung: Neural ODEs, adjungierte Variablen und kontinuierliche Flüsse"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:043dda0e-->
| Bestandteil | Kontinuierliche Gleichung | Rechenaufwand und praktische Konsequenzen |

<!--en:21023a7a-->
| **Vorwärtsabbildung der Neural ODE** | $\mathbf{x}(T) = \mathbf{x}_0 + \int_0^T \mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}(t), t)\,dt$ | Bijektive Abbildung $\Phi_T$. Lernen des tatsächlichen Flusses mit genauem Verfahren (RK4) oder direkter Regression |

<!--en:d446f1bf-->
| **Kontinuierliche adjungierte ODE** | $\dot{\mathbf{a}}(t) = -\left(\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\right)^\top\mathbf{a}(t)$, $\mathbf{a}(T) = \nabla\ell(\mathbf{x}(T))$ | Überträgt die Sensitivität mit 1 VJP pro Schritt rückwärts. Checkpointing mit Vorwärtsintegration vermeidet die Drift mit Faktor $e^{\lambda T}$. |

<!--en:d0958aa9-->
| **Parametergradient** | $\frac{\partial\mathcal{L}}{\partial\boldsymbol{\theta}} = \int_0^T \left(\frac{\partial\mathbf{f}}{\partial\boldsymbol{\theta}}\right)^\top\mathbf{a}(t)\,dt$ | Backpropagation durch Euler liefert den exakten Gradienten des diskreten Programms und weicht um $O(h)$ vom Gradienten des Flusses ab. |

<!--en:8d2cec5d-->
| **Momentane Änderung der Log-Dichte** | $\frac{d}{dt}\log p_t(\mathbf{x}(t)) = -\operatorname{tr}\!\left(\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\right) = -\nabla\cdot\mathbf{f}$ | Ein eindimensionales Kurvenintegral der Spur ersetzt diskrete Jacobi-Determinanten mit Aufwand $O(d^3)$. |

<!--en:f671b1a5-->
| **Hutchinson-Spurschätzung (FFJORD)** | $\operatorname{tr}(\mathbf{J}) = \mathbb{E}\!\left[\boldsymbol{\epsilon}^\top\mathbf{J}\boldsymbol{\epsilon}\right]$ ($\epsilon_i \in \{\pm 1\}$) | Erwartungstreue Spurschätzung mit 1 VJP pro Schritt. Bei Rademacher-Testvektoren entfällt der Varianzbeitrag der Diagonalen. |
