<!--en:5deb788f-->
# Die Fokker-Planck-Gleichung und der Wahrscheinlichkeitsfluss

<!--en:723a8449-->
Im vorigen Abschnitt haben wir den Ornstein-Uhlenbeck-Prozess simuliert. Seine stochastische Differentialgleichung (engl. stochastic differential equation, SDE) $dX=-X\,dt+\sqrt2\,dW$ zieht ein Teilchen zum Ursprung und versetzt ihm zugleich zufällige Stöße. Jeder Lauf ergab andere Pfade, und kein einzelner Pfad lässt sich vorhersagen. Dennoch wiederholt sich etwas von Lauf zu Lauf. Starten hunderttausend Teilchen bei $x=2$, so zeigt ihr Histogramm zum Zeitpunkt $t=0.5$ jedes Mal dieselbe Glockenkurve mit einem Mittelwert nahe $1.21$ und einer Standardabweichung nahe $0.80$. Die zufälligen Schwankungen der Pfade mitteln sich heraus, und die Dichte der Wolke entwickelt sich deterministisch. Welche Gleichung beschreibt diese Entwicklung?

<!--en:1a69af30-->
Adriaan Fokker und Max Planck stellten diese Gleichung für Teilchen auf, die einer stetig wirkenden Kraft und zufälligen Stößen ausgesetzt sind :cite:`Fokker.1914,Planck.1917`. Andrey Kolmogorov zeigte, dass sie die Dichte jedes Diffusionsprozesses beschreibt :cite:`Kolmogorov.1931`. Sie hat zwei Terme. Die Drift transportiert die Dichte, das Rauschen glättet sie. Auch die Glättung lässt sich als Transport schreiben, als Fluss von Wahrscheinlichkeit aus Bereichen hoher in Bereiche niedriger Dichte. Auf der Ebene der Dichte wirkt das Rauschen wie eine deterministische Geschwindigkeit.

<!--en:f2ff599e-->
Für generative Modelle ist das entscheidend. Ein Diffusionsmodell erzeugt Daten, indem es einen Prozess, der Rauschen hinzufügt, rückwärts laufen lässt. Kein Anfangspunkt lässt sich eindeutig aus dem Endpunkt eines verrauschten Pfades bestimmen, denn viele Anfangspunkte führen zum selben Endpunkt. Wir wissen jedoch, wie sich die Dichte der gesamten Wolke verändert hat, und können Prozesse konstruieren, deren Dichte sich umgekehrt entwickelt. Einen davon erhalten wir als gewöhnliche Differentialgleichung (engl. ordinary differential equation, ODE), indem wir das Rauschen durch den nach außen gerichteten Fluss ersetzen. Ihre Teilchen haben zu jedem Zeitpunkt dasselbe Histogramm wie die des stochastischen Prozesses. Der andere ist wiederum ein stochastischer Prozess und verwandelt Rauschen in Daten. In beiden Fällen können wir die Drift und die Stärke des Rauschens wählen. Eine Größe ist jedoch vorgegeben, nämlich die Richtung, in der die Log-Dichte an jedem Ort und zu jeder Zeit am stärksten zunimmt.

<!--en:3fe6f6c5-->
Wir überprüfen jeden Schritt am Ornstein-Uhlenbeck-Prozess und an einer bimodalen „Datendichte“, für die diese Richtung in geschlossener Form bekannt ist. Bei realen Daten ist sie unbekannt. Wie wir sie dort bestimmen, bleibt offen. Auch wenn die Einzelheiten der Herleitungen verblassen, sollten drei Gleichungen im Gedächtnis bleiben: die Fokker-Planck-Gleichung :eqref:`eq_mdl-dyn-fokker-planck`, die Identität :eqref:`eq_mdl-dyn-diffusion-as-transport`, die ihren Glättungsterm als Transport ausdrückt, und die Familie der Rückwärtsprozesse :eqref:`eq_mdl-dyn-reverse-family`. Wer mit dem Divergenzsatz vertraut ist, kann :numref:`sec_mdl-dyn-vector-calculus` überspringen. Die Likelihood-Formel und die Berechnung der Verzerrung am Ende können bis zur zweiten Lektüre warten.

<!--en:3d4de2cc-->
## Von Pfaden zu Dichten

<!--en:b8bb2d34-->
### Pfade und Randverteilungen

<!--en:6373d13a-->
Jede SDE in diesem Abschnitt hat die Form

<!--en:43f40f67-->
mit einer Drift $\mathbf{f} : \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$, einer skalaren Rauschamplitude $g(t)$, einer $d$-dimensionalen brownschen Bewegung $\mathbf W$ und einem zufälligen Anfangswert $\mathbf{X}_0 \sim p_0$. In einem kurzen Zeitschritt bewegt sich das Teilchen um $\mathbf f\,dt$ und erhält zusätzlich in jeder Koordinate einen unabhängigen gaußschen Stoß mit Varianz $g^2\,dt$.

<!--en:8acbd7ce-->
Eine Realisierung der brownschen Bewegung liefert einen Pfad $\mathbf{X}_t$, der bei jedem Lauf anders aussieht. Uns interessiert hier die Dichte $p_t(\mathbf{x})$ der Zufallsvariablen $\mathbf{X}_t$, die **Randdichte zum jeweiligen Zeitpunkt** (engl. time-marginal). Sie mittelt über beide Quellen des Zufalls, über die Ziehung von $\mathbf{X}_0$ und über das Rauschen entlang des Pfades. Diese Mittelung macht die Dichte deterministisch.

<!--en:3aac8262-->
Als durchgängiges Beispiel verwenden wir den Ornstein-Uhlenbeck-Prozess (OU-Prozess) aus :numref:`sec_mdl-ornstein-uhlenbeck` mit der Gleichung $dX = -\theta X\,dt + g\,dW$. Startet er in einem Punkt $X_0 = x_0$, so ist seine Randdichte der gaußsche Übergangskern, den wir schreiben als

<!--en:6649c4d8-->
Wir wählen durchgehend $\theta = 1$ und $g = \sqrt{2}$. Die Varianz nähert sich damit dem Grenzwert $g^2/(2\theta)=1$, und die stationäre Verteilung ist $\mathcal{N}(0, 1)$. Die Drift verschiebt das Zentrum der Wolke, das Rauschen verbreitert sie. Die Fokker-Planck-Gleichung beschreibt beide Wirkungen durch Operatoren auf $p_t$.

<!--en:24fd990a-->
### Simulierte Randverteilungen

<!--en:a2355ded-->
Im Abschnitt über SDEs haben wir die Momente einer solchen Wolke mit denen des Kerns verglichen. Hier betrachten wir das ganze Histogramm. Wir starten $100{,}000$ Teilchen bei $x_0 = 2$ und bewegen sie mit Euler-Maruyama-Schritten der Weite $\Delta t = 0.002$ (:numref:`sec_mdl-euler-maruyama`). Zu den Zeitpunkten $t=0.1$, $0.5$ und $2$ stellen wir die Wolke zusammen mit ihrer exakten Gauß-Dichte dar.

<!--en:41271ca0-->
Zu jedem der drei Zeitpunkte liegt das Histogramm auf seiner Kurve. Nach :eqref:`eq_mdl-dyn-ou-marginal` haben die Kurven den Mittelwert $2e^{-t}$ und die Standardabweichung $\sqrt{1-e^{-2t}}$. Wir haben sie direkt aus der Formel gezeichnet, ohne sie an die Wolke anzupassen. Bei $t = 2$ beträgt der Mittelwert noch $2e^{-2}\approx0.27$. Er liegt damit etwa eine viertel Standardabweichung vom Mittelwert der stationären Verteilung $\mathcal{N}(0, 1)$ entfernt. Es dauert also, bis der Einfluss des Anfangswerts verschwindet. Die Histogramme stammen aus einem einzigen Lauf, doch die drei zugehörigen Kurven enthalten keinen Zufall. Ein zweiter Lauf ergäbe lauter andere Pfade und dieselben Kurven. Welche Gleichung beschreibt ihre Entwicklung?

<!--en:44670dc0-->
## Die Fokker-Planck-Gleichung

<!--en:04849f06-->
Ein einzelner Pfad lässt sich nicht nach der Zeit differenzieren, der Mittelwert einer glatten Funktion von $\mathbf X_t$ über alle Pfade hingegen schon. Um von den Mittelwerten zur Dichte überzugehen, müssen wir Ableitungen von der Funktion auf die Dichte übertragen. Dazu brauchen wir drei Eigenschaften von Feldern in $\mathbb R^d$.

<!--en:f6675c26-->
### Divergenz, Laplace-Operator und partielle Integration

<!--en:da7698ce-->
Alle drei beruhen auf dem Gradienten aus :numref:`sec_mdl-multivariable_calculus`, und jede hat eine physikalische Bedeutung, auf die wir zurückkommen. Sei $\mathbf{u} : \mathbb{R}^d \to \mathbb{R}^d$ ein glattes Vektorfeld, das wir uns als Geschwindigkeit einer Flüssigkeit vorstellen, und $\psi : \mathbb{R}^d \to \mathbb{R}$ ein glattes Skalarfeld.

<!--en:c20c886b-->
**Divergenz.** Wie viel Flüssigkeit verlässt einen kleinen Quader pro Zeiteinheit? Die **Divergenz** von $\mathbf u$ ist das Skalarfeld

<!--en:ff5cd6fa-->
Sie gibt an, wie viel Flüssigkeit pro Zeiteinheit netto aus einem infinitesimalen Quader um $\mathbf{x}$ austritt, geteilt durch dessen Volumen. An einer Quelle ist sie positiv, an einer Senke negativ. Zwei Felder in der Ebene veranschaulichen den Unterschied. Für das expandierende Feld $\mathbf u=(x_1,x_2)$ gilt $\nabla\cdot\mathbf u=1+1=2$. Jedes Rechteck mit Fläche $A$ verliert daher unabhängig von seiner Lage Flüssigkeit mit der Rate $2A$. Für die Rotation $\mathbf u=(-x_2,x_1)$ gilt $\nabla\cdot\mathbf u=0+0=0$. Die Flüssigkeit zirkuliert, und kein Rechteck gewinnt oder verliert Flüssigkeit. Ein divergenzfreies Feld muss also kein Nullfeld sein. :numref:`fig_mdl-dyn-divergence-boxes` überprüft beide Aussagen an einem Quadrat.

<!--en:8596b229-->
![Die Divergenz als Nettoabfluss am Quadrat mit Seitenlänge $0.5$ und Mittelpunkt $(0.6,0.4)$. Jeder grüne Pfeil zeigt die Normalkomponente des Feldes in der Mitte einer Seite. Bei Abfluss weist er von der Seite weg, bei Zufluss auf sie zu. (a) Für $\mathbf u=(x_1,x_2)$ tritt links und unten Flüssigkeit ein, rechts und oben tritt sie aus. Der Abfluss übersteigt den Zufluss um $2\times$ die Fläche. (b) Bei der Rotation $\mathbf u=(-x_2,x_1)$ entspricht der Zufluss rechts und unten dem Abfluss durch die beiden anderen Seiten.](../img/mdl-dyn-divergence-boxes.svg)

<!--en:9ab22b6a-->
Summieren wir den Abfluss über viele kleine Quader, heben sich die Beiträge an jeder Innenfläche auf. Übrig bleibt der Fluss durch den äußeren Rand. Das ist der **Divergenzsatz** (engl. divergence theorem). Für ein beschränktes Gebiet $\Omega$ mit äußerer Einheitsnormale $\mathbf{n}$ gilt

<!--en:dc9fdda4-->
Das Integral der Divergenz über ein Gebiet ist gleich dem Nettoabfluss durch seinen Rand. In einer Dimension ist das der Hauptsatz der Differential- und Integralrechnung, $\int_a^b u'(x)\,dx = u(b) - u(a)$. Wir verwenden den Satz ohne Beweis. Einen Beweis findet man in jedem Lehrbuch der mehrdimensionalen Analysis.

<!--en:9d08c5cb-->
**Laplace-Operator.** Wir suchen ein Maß dafür, wie sich ein Funktionswert von den Werten in seiner Umgebung unterscheidet. Der **Laplace-Operator** ist die Divergenz des Gradienten. Vor einer Funktion bezeichnet $\Delta$ diesen Operator, anders als das Inkrement in $\Delta t$. Es gilt

<!--en:5ef247a9-->
Er vergleicht $\psi$ an einem Punkt mit dem Mittelwert auf einer kleinen Kugeloberfläche mit Radius $r$ um diesen Punkt. Der Mittelwert beträgt $\psi(\mathbf{x}) + \frac{r^2}{2d}\Delta \psi(\mathbf{x}) + O(r^4)$, wie Sie in Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-laplacian-averaging` beweisen. In einer Dimension besteht die „Kugeloberfläche“ für $\psi(x)=x^2$ aus den beiden Punkten $x\pm r$. Der Mittelwert ist $\frac12\big((x-r)^2+(x+r)^2\big)=x^2+r^2$, und es gilt $\frac{r^2}{2}\psi''=r^2$. Die Formel ist also ohne Restglied exakt (für $x=1$ und $r=0.3$ beträgt der Mittelwert $1.09$). Somit bedeutet $\Delta \psi > 0$, dass $\psi$ unter dem Mittelwert der Umgebung liegt. Verschwindet der Laplace-Operator, muss die Funktion keineswegs konstant sein. Für $\psi=x_1^2-x_2^2$ gilt $\Delta \psi=2-2=0$, und ihr Wert stimmt mit ihrem Mittelwert auf jedem Kreis um den betrachteten Punkt überein.

<!--en:d5f8ad32-->
**Partielle Integration.** Dies ist die $d$-dimensionale Form der eindimensionalen Regel aus :numref:`subsec_mdl-integration-by-parts`. Mit ihr übertragen wir eine Ableitung von einem Faktor auf den anderen. Sind $\psi$ und $\mathbf{u}$ glatt und fällt $\psi\,\mathbf u$ im Unendlichen schnell genug ab, so gilt die folgende Identität. Hier ist die Abnahme gesichert, weil $\psi$ außerhalb einer beschränkten Menge verschwindet oder weil beide Faktoren im Unendlichen gaußsch abfallen.

<!--en:46d98936-->
Die Ableitung geht von $\mathbf u$ auf $\psi$ über, und das Vorzeichen wechselt.

<!--en:0929cfd6-->
**Beweis.** Die Produktregel liefert $\nabla \cdot (\psi \mathbf{u}) = \nabla \psi \cdot \mathbf{u} + \psi\, \nabla \cdot \mathbf{u}$. Wir integrieren beide Seiten über eine Kugel mit Radius $R$ und lassen $R \to \infty$. Nach dem Divergenzsatz ist die linke Seite der Fluss von $\psi\mathbf{u}$ durch die Kugeloberfläche mit Radius $R$. Unter der angenommenen Abklingbedingung geht er gegen null. $\blacksquare$

<!--en:9df5453d-->
Wenden wir die Regel zweimal an, zunächst mit $\mathbf{u} = \nabla p$ und dann mit vertauschten Rollen, so übertragen wir den gesamten Laplace-Operator auf den anderen Faktor im Integral. Die beiden Vorzeichenwechsel heben sich auf, und wir erhalten $\int \psi\, \Delta p = -\int\nabla \psi\cdot\nabla p = \int p\, \Delta \psi$.

<!--en:a6416302-->
### Vom Lemma von Itô zur partiellen Differentialgleichung

<!--en:5d25c5a3-->
Nun leiten wir die Gleichung für $p_t$ her. Dabei setzen wir durchgehend voraus, dass $\mathbf f$ in $\mathbf x$ stetig differenzierbar und in $t$ stetig ist. Außerdem sei $g$ stetig und $p_t$ glatt und positiv. Ableitungen nach $t$ dürfen wir unter das Integralzeichen ziehen.

<!--en:00f83b33-->
Wir betrachten die Wolke über ihre Mittelwerte. Sei $\phi$ eine glatte Buckelfunktion mit dem Wert $1$ auf einem kleinen Quader und dem Wert $0$ außerhalb eines etwas größeren Quaders. Dann entspricht $\mathbb{E}[\phi(\mathbf{X}_t)]$ ungefähr der Wahrscheinlichkeit, dass das Teilchen im kleinen Quader liegt. Wissen wir, wie sich jeder solche Mittelwert mit der Zeit ändert, so kennen wir auch die zeitliche Änderung der Dichte. Betrachten wir deshalb eine beliebige glatte **Testfunktion** $\phi : \mathbb{R}^d \to \mathbb{R}$, die außerhalb einer beschränkten Menge verschwindet.

<!--en:f991ded3-->
Das Lemma von Itô (:numref:`sec_mdl-ito-lemma`) ist die Kettenregel für SDE-Pfade. Für $d\mathbf X=\mathbf f\,dt+\mathbf G\,d\mathbf W$ ergänzt es die gewöhnliche Kettenregel um den Term zweiter Ordnung $\tfrac12\sum_{ij}(\mathbf G\mathbf G^\top)_{ij}\,\partial_i\partial_j\phi$, denn das quadrierte brownsche Inkrement $(dW)^2$ hat die Größenordnung $dt$ und ist daher nicht vernachlässigbar. Für $\mathbf{G} = g(t)\, \mathbf I$ entspricht dieser Term dem $\tfrac12 g^2$-Fachen des Laplace-Operators angewandt auf $\phi$:

<!--en:5a8063f0-->
Die Änderung der Funktion setzt sich aus einem Driftanteil und einem Rauschanteil zusammen. Der Driftanteil ergibt sich aus der Kettenregel mit der Laplace-Korrektur. Bilden wir nun den Erwartungswert. Das Itô-Integral von $g\nabla\phi$ hat den Mittelwert null, denn jedes brownsche Inkrement ist vom Integranden zu Beginn dieses Inkrements unabhängig (:numref:`sec_mdl-ito-integral`). Damit entfällt der Rauschanteil:

<!--en:c8811a6f-->
Der Mittelwert von $\phi$ ändert sich mit dem Mittelwert der Rate aus dem Lemma von Itô. Beide Seiten sind Integrale bezüglich der Randdichte. Schreiben wir $\mathbb{E}[\,\cdot\,] = \int (\cdot)\, p_t(\mathbf{x})\, d\mathbf{x}$ und ziehen $d/dt$ unter das Integralzeichen, so erhalten wir

<!--en:890b0a15-->
Rechts wirken die Ableitungen noch auf $\phi$. Wir übertragen sie auf $p_t$ durch partielle Integration gemäß :eqref:`eq_mdl-dyn-integration-by-parts`. Da $\phi$ außerhalb einer beschränkten Menge verschwindet, treten keine Randterme auf, und $p_t$ muss im Unendlichen nicht abfallen. Beim Driftterm integrieren wir einmal partiell mit $\psi=\phi$ und $\mathbf u=\mathbf fp_t$. Beim Diffusionsterm integrieren wir zweimal partiell, sodass sich die Vorzeichenwechsel aufheben:

<!--en:e9bb5405-->
Setzen wir beides ein, erhalten wir ein einziges Integral gegen $\phi$:

<!--en:bcd9e346-->
Das gilt für jede Testfunktion $\phi$. Der Klammerausdruck ist stetig. Eine stetige Funktion, deren Integral gegen jede Buckelfunktion null ist, verschwindet überall. Wäre sie etwa irgendwo positiv, könnten wir dort eine Buckelfunktion platzieren und erhielten ein positives Integral. Der Klammerausdruck verschwindet also punktweise, und $p_t$ löst die **Fokker-Planck-Gleichung**, auch *Kolmogorov-Vorwärtsgleichung* genannt:

<!--en:438eb6f8-->
Der erste Term stammt von der Drift, der zweite vom Rauschen. Hat das Rauschen die Matrixform $\mathbf G\,d\mathbf W$, wird der zweite Term zu $\tfrac12\sum_{ij}\partial_i\partial_j(a_{ij}p_t)$ mit $\mathbf a=\mathbf G\mathbf G^\top$. In Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-state-dependent-noise` untersuchen Sie, was sich dadurch ändert. :citet:`Risken.1996` stellt zahlreiche Varianten der Gleichung aus der Physik zusammen.

<!--en:b359bd42-->
Die Herleitung zeigt nur eine Richtung. Die Randdichte der SDE löst die Gleichung. Später brauchen wir die Umkehrung, dass eine Lösung mit Anfangsdichte $p_0$ die Randdichte *ist*. Dazu muss die Gleichung eine eindeutige Lösung haben. Das gilt unter schwachen Voraussetzungen, die alle hier verwendeten Driftfelder erfüllen. Es gelte $g>0$, und die Drift sei bezüglich $\mathbf x$ Lipschitz-stetig, gleichmäßig in $t$. Ihre Steigung in $\mathbf x$ ist also für alle $t$ durch dieselbe Konstante beschränkt. Unter den Wahrscheinlichkeitsdichten mit gegebener Anfangsverteilung ist die Lösung dann eindeutig (präzise Aussagen finden sich bei :citet:`Bogachev.Krylov.Rockner.ea.2015`). Wir verwenden diese Eindeutigkeit ohne Beweis und weisen an den betreffenden Stellen darauf hin.

<!--en:98c1c432-->
### Die Wärmeleitungsgleichung und die beiden Terme

<!--en:4ee8c4fe-->
Schalten wir die Drift ab, so erhalten wir aus der Fokker-Planck-Gleichung die **Wärmeleitungsgleichung** (engl. heat equation),

<!--en:d1b88882-->
Die Dichte steigt überall dort, wo sie unter dem Mittelwert ihrer Umgebung liegt. Prüfen wir das zunächst an einem Punkt. Wir wählen $g=1$, $d=1$ und $p_t=\mathcal N(0,t)$, die Verteilung der bei $0$ gestarteten brownschen Bewegung. Für $p=e^{-x^2/(2t)}/\sqrt{2\pi t}$ liefert direktes Ableiten $\partial_tp=p\,\big(\tfrac{x^2}{2t^2}-\tfrac1{2t}\big)$ und $\partial_{xx}p=p\,\big(\tfrac{x^2}{t^2}-\tfrac1t\big)$. Bei $x=0$ und $t=1$ gilt $p=0.399$, $\partial_tp=-0.1995$ und $\tfrac12\partial_{xx}p=-0.1995$. Der Gipfel sinkt mit der vom Laplace-Operator vorhergesagten Rate.

<!--en:b30f59aa-->
Allgemein löst eine Gauß-Verteilung $\mathcal N(\mathbf m,v_t\mathbf I)$ die Wärmeleitungsgleichung, wenn ihre Varianz gemäß $v_t=v_0+g^2t$ bei konstantem $g$ wächst oder gemäß $v_t=v_0+\int_0^tg(\tau)^2\,d\tau$, wenn sich $g$ mit der Zeit ändert. Dies entspricht dem Gesetz $\textrm{Var}(W_t)=t$ der brownschen Bewegung. Rauschen allein ergibt die *varianzexplodierende* Familie aus :numref:`sec_mdl-why-randomness`, die Diffusionsmodelle mit wachsendem $g(t)$ verwenden.

<!--en:2959e017-->
Betrachten wir nun die beiden Terme in :eqref:`eq_mdl-dyn-fokker-planck`. Der erste, $-\nabla \cdot (\mathbf{f} p)$, beschreibt *Advektion*. Die Dichte sinkt dort, wo die Drift mehr Masse aus einem kleinen Quader hinausträgt als hinein. Der zweite Term, $\tfrac12 g^2 \Delta p$, beschreibt *Diffusion*. Er erhöht die Dichte dort, wo sie unter dem Mittelwert ihrer Umgebung liegt, und senkt sie dort, wo sie darüber liegt. Das ist die Itô-Korrektur $(dW)^2 = dt$ auf der Ebene der Dichte.

<!--en:8212a5d8-->
### Gaußsche Lösungen der OU-Gleichung

<!--en:d37b718d-->
Für den OU-Prozess kennen wir die Antwort in geschlossener Form. Damit können wir die Gleichung an einer ganzen Familie von Lösungen prüfen. Mit $f(x) = -\theta x$ in einer Dimension lautet :eqref:`eq_mdl-dyn-fokker-planck`

<!--en:9f2dd602-->
Die Unbekannte einer partiellen Differentialgleichung ist eine ganze Funktion und lässt sich selten von Hand bestimmen. Hier haben wir Glück. Für eine Gauß-Dichte reduziert sich die Gleichung auf zwei gewöhnliche Differentialgleichungen, eine für den Mittelwert und eine für die Varianz.

<!--en:cf23432a-->
**Proposition (gaußsche Lösungen der OU-Gleichung).** *Sei $p(x, t) = \mathcal{N}(x; m(t), v(t))$ eine Gauß-Dichte mit differenzierbarem Mittelwert $m(t)$ und differenzierbarer Varianz $v(t) > 0$. Dann löst $p$ die Fokker-Planck-Gleichung des OU-Prozesses genau dann, wenn*

<!--en:4ad842c0-->
**Beweis.** Für eine Gauß-Dichte gilt $\partial_x p = -\frac{x - m}{v}\, p$. Durch direktes Ableiten erhalten wir auf beiden Seiten der partiellen Differentialgleichung jeweils $p$ mal ein quadratisches Polynom in $(x - m)$:

<!--en:5e319ad4-->
Schreiben wir im Driftterm $x = (x - m) + m$, so ergibt sich auf der anderen Seite

<!--en:4996f511-->
Die Funktionen $1, (x - m), (x - m)^2$ sind linear unabhängig. Beide Seiten stimmen also nur dann für alle $x$ überein, wenn die Koeffizienten gleich sind. Der lineare Term liefert $\dot{m} = -\theta m$, der quadratische Term $\dot{v} = g^2 - 2\theta v$ (der konstante Term ergibt dieselbe zweite Bedingung). $\blacksquare$

<!--en:95cbf5a7-->
Der Mittelwert klingt ab, und die Varianz nähert sich $g^2/(2\theta)$. Lösen wir die beiden gewöhnlichen Differentialgleichungen (ODEs) mit den Anfangswerten $m(0)=m_0$ und $v(0)=v_0$, so erhalten wir

<!--en:751a0016-->
Für einen festen Anfangspunkt gilt $v_0=0$. Damit erhalten wir den Kern :eqref:`eq_mdl-dyn-ou-marginal` für jedes $t>0$. Die Übergangsdichte aus :numref:`sec_mdl-ornstein-uhlenbeck` löst also die Fokker-Planck-Gleichung.

<!--en:2db35c03-->
Der Fixpunkt $\dot m = \dot v = 0$ liegt bei $m = 0$ und $v = g^2/(2\theta)$. Die stationäre Gauß-Verteilung entspricht also dem stationären Zustand $\partial_t p = 0$ der partiellen Differentialgleichung (Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-stationary-density` leitet ihn ohne gaußschen Ansatz her). Mit $\theta = 0$ erhalten wir wieder $\dot v = g^2$ aus der Wärmeleitungsgleichung.

<!--en:28a8b1f4-->
Dieses Glück hat einen Grund und eine Grenze. Weil die Drift linear ist, bleibt eine Gauß-Verteilung gaußsch, und ihre ersten beiden Momente bilden ein geschlossenes System. Bei der Drift $f(x)=x-x^3$ eines Doppelmuldenpotentials aus derselben Aufgabe enthält die Gleichung für $\mathbb{E}[X^2]$ auch $\mathbb{E}[X^4]$, die für $\mathbb{E}[X^4]$ auch $\mathbb{E}[X^6]$, und die Kette schließt sich nie.

<!--en:2dfe2d3a-->
### Linearität und Mischungen

<!--en:c6f01e8a-->
Auch reale Daten sind nicht gaußverteilt. Wir brauchen also eine Dichte mit mehr Struktur, deren Entwicklung wir dennoch in geschlossener Form angeben können. Bei linearer Drift erfüllt eine Gauß-Mischung diese Bedingung. Als „Datendichte“ verwenden wir im Folgenden

<!--en:c2cc3da6-->
mit zwei gut getrennten Modi der Standardabweichung $0.5$. Für eine Ziehung aus $p_0$ wählen wir eine Komponente $\mathcal N(\mu_k,v_k)$ und ziehen daraus einen Punkt. Nach dem Kern :eqref:`eq_mdl-dyn-ou-marginal` ist $X_t$ die Summe aus $\alpha_tX_0$ und einer unabhängigen $\mathcal N(0,\sigma_t^2)$-verteilten Zufallsvariablen. Bei gegebener Komponente ist dies also die Summe zweier unabhängiger gaußverteilter Größen mit Gesamtmittelwert $\alpha_t\mu_k$ und Gesamtvarianz $\alpha_t^2v_k+\sigma_t^2$. Mitteln wir über die Komponente, erhalten wir als Randverteilung die Mischung der zeitlich entwickelten Komponenten. Für $\theta=1$ und $g=\sqrt2$ ergibt sich zu jeder Zeit die Mischung aus $\mathcal N\big(\pm2e^{-t},\,1-0.75\,e^{-2t}\big)$ in geschlossener Form.

<!--en:ee4190d7-->
Dasselbe folgt aus der Fokker-Planck-Gleichung, die linear in $p$ ist. Eine Mischung $w_1p^{(1)}_t+w_2p^{(2)}_t$ von Lösungen ist deshalb ebenfalls eine Lösung. Nach der Eindeutigkeitsaussage im Anschluss an :eqref:`eq_mdl-dyn-fokker-planck` ist sie die Randdichte zur gemischten Anfangsverteilung. Im OU-Prozess folgt jede Gauß-Komponente für sich den Momentengleichungen :eqref:`eq_mdl-dyn-moment-odes`.

<!--en:884adef6-->
Alle folgenden Experimente verwenden diese Randverteilung. Wir definieren sie daher einmal zusammen mit ihrer Verteilungsfunktion und einem Sampler. Die Parameter sind explizite Argumente, damit sie sich in den Aufgaben ändern lassen.

<!--en:37e79d75-->
Mit der geschlossenen Formel im Code können wir die Lösung direkt in die partielle Differentialgleichung einsetzen. Wir werten beide Seiten auf einem Gitter zu den Zeitpunkten $t=0.25$, $1$ und $3$ aus. Für $\partial_t$ verwenden wir eine zentrale Differenz in der Zeit, für die Ableitungen nach $x$ verwenden wir `np.gradient`. Endliche Differenzen hinterlassen selbst bei einer exakten Lösung ein Residuum. Daher verwendet die Codezelle zwei Gitter, das zweite doppelt so fein wie das erste. Der Abbruchfehler dieser Differenzenschemata zweiter Ordnung muss um den Faktor vier sinken.

<!--en:edc9233c-->
Das Residuum liegt zu jedem Zeitpunkt drei bis fünf Größenordnungen unter $\partial_tp$ ($4\times10^{-6}$ gegenüber $0.44$ bei $t=0.25$, bei $t=3$ bewegt sich die Dichte kaum noch, und $10^{-6}$ steht $3\times10^{-3}$ gegenüber). Auf dem feineren Gitter sinkt das Residuum zu allen drei Zeitpunkten auf ein Viertel. Was bleibt, ist der Abbruchfehler. Für die Herleitung war diese Rechnung entbehrlich. Die Codezelle prüft jedoch den Code, auf dem alle weiteren Experimente beruhen.

<!--en:132c8017-->
## Die Score-Funktion

<!--en:5e1bb060-->
Der Glättungsterm der Fokker-Planck-Gleichung beschreibt, wie sich zeigen wird, einen Transport. Dessen Geschwindigkeit ergibt sich aus einem einzigen Feld, dem Gradienten von $\log p_t$. Über die Identität $\nabla p=p\,\nabla\log p$ tritt dieses Feld im Laplace-Operator auf (:numref:`sec_mdl-continuity-equation`). Wir brauchen es für alle weiteren Konstruktionen und untersuchen es deshalb zuerst, vor allem am Beispiel der Mischung.

<!--en:1479a3a8-->
### Definition und Geometrie

<!--en:f3caede8-->
In welcher Richtung gelangen wir von einem Punkt $\mathbf x$ am schnellsten zu Punkten höherer Dichte? Die Antwort gibt der **Score** einer Dichte $p_t$:

<!--en:3709530f-->
Der Score ist ein Vektorfeld auf dem Datenraum. Er zeigt in Richtung des steilsten Anstiegs der Log-Dichte. Wir bilden den Gradienten bezüglich der Eingabe $\mathbf{x}$. In :numref:`sec_mdl-fisher-divergence` haben wir diesen Daten-Score eingeführt und vom Score bezüglich der Parameter bei der Maximum-Likelihood-Schätzung unterschieden.

<!--en:ca846ffe-->
Die Länge von $\mathbf s_t$ ist die Anstiegsrate von $\log p_t$, also $\|\nabla p_t\|/p_t$. Sie ist daher dort groß, wo die Dichte klein ist und sich dennoch ändert, etwa an den Verteilungsrändern und in den Tälern zwischen den Modi. Eine hohe Dichte allein ergibt keinen großen Score. An den kritischen Punkten von $p_t$ verschwindet der Score. Anhand seiner Ableitung lassen sich Modi, Minima und Sattelpunkte unterscheiden.

<!--en:25857132-->
Betrachten wir zunächst eine Gauß-Verteilung. Für $p(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, v\mathbf I)$ lautet die Log-Dichte $-\|\mathbf{x} - \boldsymbol{\mu}\|^2 / (2v) + \textrm{const}$. Somit gilt

<!--en:e101fcf5-->
Das ist die lineare Feder aus :numref:`sec_mdl-fisher-divergence`. Sie zieht zum Mittelwert und ist bei kleinerer Varianz steifer. Mit $\mu=0$, $v=0.25$ und $x=1$ ergibt sich $-4$, die vierfache Zugkraft im Vergleich zur Einheitsvarianz. Für eine allgemeine Kovarianz lautet die Federkraft $-\boldsymbol\Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})$. :numref:`fig_mdl-dyn-score-field` zeigt das Score-Feld einer bimodalen Dichte in zwei Dimensionen.

<!--en:106c121a-->
![Der Score $\mathbf{s}(\mathbf{x})=\nabla\log p(\mathbf{x})$ einer Mischung aus zwei Gauß-Komponenten über den Höhenlinien der Dichte. Die Pfeile weisen bergauf und verschwinden an kritischen Punkten. Die orangefarbenen Punkte markieren die beiden Modi. Der Symmetriepunkt dazwischen ist eine weitere Nullstelle mit anderer Krümmung. Die Pfeile zeigen die tatsächliche Länge $\|\nabla p\|/p$. Nur weit außen, wo diese unbegrenzt wächst, ist die Darstellung auf $4$ begrenzt. Nahe den Modi und dem Sattelpunkt sind die Pfeile kurz.](../img/mdl-dyn-score-field.svg)

<!--en:7e1f7b00-->
### Der Score einer Gauß-Mischung

<!--en:1b9bcac3-->
Unsere Datendichte ist eine Mischung. Wir brauchen ihren Score deshalb in einer Form, die sich berechnen und deuten lässt. Für $p(x) = \sum_k w_k\, N_k(x)$ mit Gewichten $w_k$ und Komponenten $N_k(x)=\mathcal{N}(x; \mu_k, v_k)$ differenzieren wir $\log p$ direkt:

<!--en:21d566ba-->
Der Score der Mischung ist ein gewichteter Mittelwert der Federkräfte ihrer Komponenten. Das Gewicht $r_k(x)$ ist die *Zugehörigkeitswahrscheinlichkeit* (engl. responsibility) der Komponente $k$, also die A-posteriori-Wahrscheinlichkeit, dass ein Punkt bei $x$ aus ihr stammt, wie in :numref:`sec_mdl-fisher-divergence`. Bei zwei gleich gewichteten Komponenten mit Mittelwerten $\pm\mu$ und gemeinsamer Varianz $v_0$ vereinfachen sich die Zugehörigkeitswahrscheinlichkeiten zu einer logistischen Funktion von $x$. Der Score hat dann die geschlossene Form

<!--en:d031ed00-->
In der Nähe jedes Modus gilt $\tanh \to \pm 1$, und der Score entspricht der lokalen Federkraft $-(x \mp \mu)/v_0$. Für unsere Beispielmischung mit $\mu=2$ und $v_0=0.25$ ergibt sich $s_0(x)=8\tanh(8x)-4x$. Bei $x = 0$ verschwindet der Score ebenfalls. Dort liegt der dritte kritische Punkt, und der Score wächst von null aus mit der Steigung

<!--en:3abfdc56-->
Eine positive Steigung bedeutet, dass $x=0$ ein Minimum der Dichte ist. Der bergauf gerichtete Fluss $\dot x=s_0(x)$ bewegt sich von diesem Punkt weg. Die Mischung hat genau dann zwei Modi, wenn $\mu^2>v_0$ gilt. Für $\mu^2\le v_0$ verschmelzen sie zu einem Modus (Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging`).

<!--en:636765c7-->
Im Code verwenden wir die allgemeine Formel :eqref:`eq_mdl-dyn-mixture-score` mit einer numerisch stabilen Softmax-Funktion für die Zugehörigkeitswahrscheinlichkeiten. Denn in den späteren Experimenten brauchen wir den Score von $p_t$ zu jedem Zeitpunkt. Mit der tanh-Formel, den Nullstellen und der Steigung $60$ prüfen wir die Implementierung.

<!--en:79b9e352-->
Alle Prüfungen gelingen. Die beiden Formeln stimmen bis auf Maschinengenauigkeit überein, und die Nullstellen sowie die Steigung entsprechen unserer Rechnung. Die Grafik zeigt darüber hinaus, wo der Score groß ist. Das gilt an den Verteilungsrändern und im Tal zwischen den Modi. Dort ist die Dichte fast null, doch ihr Logarithmus steigt zu beiden Modi hin steil an.

<!--en:c59094b0-->
## Die Kontinuitätsgleichung und die Wahrscheinlichkeitsfluss-ODE

<!--en:b301868d-->
Die Fokker-Planck-Gleichung beschreibt Teilchen, die sich zufällig bewegen. Welches Gesetz gilt für Teilchen, die sich deterministisch bewegen? Wie wir gleich sehen, stimmen die beiden Gesetze überein, sobald wir das Rauschen als Geschwindigkeit ausdrücken.

<!--en:5434c366-->
### Erhaltung der Wahrscheinlichkeit

<!--en:d5af3bd9-->
Lassen wir das Rauschen kurz beiseite. Jedes Teilchen folge einem Geschwindigkeitsfeld, $\dot{\mathbf{x}} = \mathbf{u}(\mathbf{x}, t)$. Welcher partiellen Differentialgleichung genügt die Dichte $q_t$ der Teilchen? Wahrscheinlichkeit entsteht und verschwindet nicht. Die Masse in einem festen Gebiet $\Omega$ kann sich also nur durch einen Fluss über dessen Rand ändern. Die Rate, mit der Teilchen ein kleines Randstück passieren, ergibt sich pro Flächeneinheit aus Dichte mal Normalgeschwindigkeit. Diese Flussdichte nennen wir **Wahrscheinlichkeitsstrom** (engl. probability current), $\mathbf j=q\,\mathbf u$. An jedem Punkt gibt seine Normalkomponente an, wie viel Wahrscheinlichkeit dort pro Zeit- und Flächeneinheit eine Fläche durchquert. Die Massenbilanz des Gebiets lautet

<!--en:1d6af6d8-->
Dabei haben wir mit dem Divergenzsatz :eqref:`eq_mdl-dyn-divergence-theorem` den Fluss durch den Rand in ein Volumenintegral umgewandelt. Die Masse im Inneren ändert sich mit dem negativen Nettoabfluss. Da $\Omega$ beliebig ist, stimmen die Integranden punktweise überein. Das ergibt die **Kontinuitätsgleichung**:

<!--en:13d525ca-->
Die Dichte an einem Punkt sinkt mit einer Rate, die der Divergenz des Stroms an diesem Punkt entspricht. In einer Dimension lautet die Bilanz $\partial_t \int_a^b q\,dx = j(a) - j(b)$, also Zufluss bei $a$ abzüglich Abfluss bei $b$. :numref:`fig_mdl-dyn-fokker-planck-flux` veranschaulicht sie.

<!--en:fd321551-->
![Die Kontinuitätsgleichung als Flussbilanz. Die Geschwindigkeit ist hier eine konstante Drift $f$, und die Dichte trägt die Bezeichnung $p$. Die Dichte $p(x)$ (blau) bewegt sich mit $f$. Daraus ergibt sich der Wahrscheinlichkeitsstrom $j(x)=f\,p(x)$. Im Intervall von $a$ bis $b$ auf der rechten Flanke des Dichtegipfels wächst die eingeschlossene Masse (orange) mit der Rate $j(a)-j(b)$, dem Nettozufluss. Bei $a$ fließt mehr hinein als bei $b$ hinaus, weil die Dichte bei $a$ höher ist.](../img/mdl-dyn-fokker-planck-flux.svg)

<!--en:1c5e2f77-->
Die Bilanz beschreibt auch, was am Rand des Gebiets geschieht. Auf ganz $\mathbb R^d$ setzen wir voraus, dass der Strom im Unendlichen abklingt. Für die Fokker-Planck-Gleichung müssen dazu $\mathbf fp_t$ und $\nabla p_t$ abklingen. Gaußsch abfallende Verteilungsränder sichern dies selbst bei der linear wachsenden OU-Drift. Integration von :eqref:`eq_mdl-dyn-continuity` über $\mathbb{R}^d$ ergibt dann $\frac{d}{dt}\int q_t = 0$. Die Gesamtwahrscheinlichkeit bleibt also erhalten. Auf einem beschränkten Gebiet gilt an einem *reflektierenden* Rand $\mathbf j\cdot\mathbf n=0$, sodass die Masse auch dort erhalten bleibt. An einem *absorbierenden* Rand gilt dagegen $q=0$, und Masse kann austreten.

<!--en:77fd497c-->
Zwei einfache Fälle zeigen, wie die Gleichung in einer Dimension wirkt. Für eine Translation $u=c$ lautet sie $\partial_tq+c\,\partial_xq=0$ mit Lösung $q_t(x)=q_0(x-ct)$. Das Profil verschiebt sich, ohne seine Form zu ändern. Bei einer Kontraktion $u=-x$ bewegt sich jedes Teilchen gemäß $x(t)=x(0)e^{-t}$, und die Wolke schrumpft um den Faktor $e^{-t}$. Da die Masse jedes mitbewegten Intervalls erhalten bleibt, steigt die Dichte entlang jeder Trajektorie um den Faktor $e^{t}$.

<!--en:3112051c-->
Die Kontraktion legt eine allgemeine Regel für die Dichte nahe, die ein bewegtes Teilchen erfährt. Schreiben wir die Divergenz mit der Produktregel aus,

<!--en:858624cb-->
so wird die Kontinuitätsgleichung zu $\partial_tq+\mathbf u\cdot\nabla q=-q\,\nabla\cdot\mathbf u$. Die linke Seite ist nach der Kettenregel die Änderungsrate von $q$ entlang einer Trajektorie $\mathbf x(t)$ des Feldes $\dot{\mathbf x}=\mathbf u$. Division durch $q$ liefert

<!--en:328ec41f-->
Entlang einer Trajektorie sinkt die Log-Dichte mit der Rate der lokalen Ausdehnung. Bei der Kontraktion gilt $\nabla\cdot u=-1$, und die Log-Dichte steigt wie oben berechnet mit der Rate $1$. Die Trajektorien von $\dot{\mathbf x}=\mathbf u$ heißen **Charakteristiken** der Kontinuitätsgleichung. Diese Regel ist die infinitesimale Variablentransformation aus :numref:`sec_mdl-continuous-normalizing-flows`, die wir dort entlang von Trajektorien bewiesen haben.

<!--en:e3734599-->
### Diffusion als Transport

<!--en:c3ae7de3-->
Die Fokker-Planck-Gleichung :eqref:`eq_mdl-dyn-fokker-planck` hat noch nicht die Form einer Kontinuitätsgleichung. Neben dem Transportterm $-\nabla \cdot (\mathbf{f} p)$ enthält sie einen Laplace-Operator. Mit der Kettenregel verwandeln wir ihn in einen Transportterm. Diese Rechnung braucht nur eine Zeile und hat doch die weitreichendsten Folgen in diesem Abschnitt. Überall, wo $p > 0$ gilt, ist

<!--en:24bf620d-->
Das ist die umgestellte Ableitungsregel für $\log p$. Daher gilt

<!--en:432932dd-->
Der Diffusionsterm ist die Divergenz eines Stroms, der sich als Produkt der Dichte mit $+\tfrac12 g^2\nabla\log p$ schreiben lässt. Auf der linken Seite einer Kontinuitätsgleichung erhält er ein Minuszeichen und entspricht somit einer Geschwindigkeit $-\tfrac12 g^2\nabla\log p$.

<!--en:4bc28830-->
**Proposition (die Fokker-Planck-Gleichung als Kontinuitätsgleichung).** *Überall, wo $p_t > 0$ gilt, ist die Fokker-Planck-Gleichung :eqref:`eq_mdl-dyn-fokker-planck` äquivalent zu*

<!--en:f59a5933-->
**Beweis.** Wir setzen :eqref:`eq_mdl-dyn-diffusion-as-transport` in :eqref:`eq_mdl-dyn-fokker-planck` ein und fassen beide Divergenzen zusammen:

<!--en:4e081798-->
Auf der Ebene der Dichten transportiert die SDE Wahrscheinlichkeit so, als folgte jedes Teilchen der Geschwindigkeit $\mathbf u_t$. Zur Drift kommt also eine Bewegung entgegen dem Gradienten der Log-Dichte hinzu. Der zugehörige Strom lautet

<!--en:a3744fe6-->
Die Wärmeleitungsgleichung zeigt, wie diese Geschwindigkeit aussieht. Wählen wir $\mathbf f=0$ und $p_t=\mathcal N(0,v_t)$ in einer Dimension mit $v_t=v_0+g^2t$. Der Score ist $-x/v_t$, also gilt

<!--en:8dcf4565-->
Die Geschwindigkeit zeigt vom Zentrum weg und ist proportional zum Abstand vom Zentrum. Je breiter die Dichte wird, desto schwächer wird diese Bewegung. In der ODE $\dot x=g^2x/(2v_t)$ lassen sich die Variablen trennen. Es gilt $\frac{d}{dt}\log x=\frac{g^2}{2v_t}=\frac12\frac{d}{dt}\log v_t$ und daher

<!--en:d862a973-->
Der Abstand jedes Teilchens vom Zentrum wächst um denselben Faktor. Strecken wir eine Gauß-Verteilung um $\sqrt{v_t/v_0}$, so hat sie die Varianz $v_t$, genau wie bei der brownschen Bewegung. Für $v_0=g=1$ und $t=3$ verdoppelt sich der Abstand jedes Teilchens vom Zentrum. Ein brownsches Teilchen mit Startpunkt $x=1$ landet dagegen bei $1$ plus einer Zufallsgröße mit Standardabweichung $\sqrt3\approx1.73$. Beide Prozesse liefern zu jedem Zeitpunkt dasselbe Histogramm, bewegen die einzelnen Teilchen aber unterschiedlich (:numref:`fig_mdl-dyn-diffusion-outward-flow`).

<!--en:b52985ed-->
![Die Wärmeleitungsgleichung $\partial_tp=\frac12\partial_{xx}p$ mit Anfangsverteilung $p_0=\mathcal N(0,1)$, gelöst als Fluss. (a) Die Dichte bei $t=0$ und $t=3$. Die Pfeile auf der Achse zeigen die Geschwindigkeit $u_0(x)=x/2$. Sie weist nach außen und ist proportional zum Abstand. Das Teilchen bei $x=1$ erreicht $x=2$ zum Zeitpunkt $t=3$, und die mitbewegte Dichte halbiert sich. (b) Die Flusstrajektorien $x(0)\sqrt{1+t}$ sind glatt und kreuzen sich nie. Zum Vergleich drei brownsche Pfade ab $x=1$ und das Band von einer Standardabweichung $1\pm\sqrt t$.](../img/mdl-dyn-diffusion-outward-flow.svg)

<!--en:3f0d4629-->
Diese Konstruktion hat eine Grenze. Für $v_0\to0$ ist die Geschwindigkeit bei kleinen Zeiten $g^2x/(2g^2t)=x/(2t)$ und wächst für $t\to0$ unbeschränkt. Ein deterministischer Fluss kann eine Punktmasse nur dann zu einer Gauß-Verteilung verbreitern, wenn das Feld zu Beginn unbeschränkt wird. Denn ein Lipschitz-stetiger Fluss bildet einen Punkt auf genau einen Punkt ab.

<!--en:03838ec2-->
Die nächste Codezelle setzt dieses Beispiel um. Aus einer Stichprobe von $20{,}000$ Punkten aus $\mathcal N(0,1)$ erzeugt sie zum Zeitpunkt $t=3$ eine brownsche Wolke, indem sie unabhängige Inkremente mit Verteilung $\mathcal N(0,3)$ addiert. Eine zweite Wolke erhält sie durch Integration von $\dot x=x/(2(1+t))$ mit dem Heun-Verfahren (:numref:`sec_mdl-euler-runge-kutta`). Sie gibt beide Standardabweichungen aus und zeigt, wie weit die zweite Wolke von den verdoppelten Werten der Ausgangsstichprobe abweicht.

<!--en:2bae5f92-->
Beide Wolken haben die Standardabweichung $2.00$ bis auf einen Stichprobenfehler der Größenordnung $0.01$. Die Flussabbildung stimmt mit $x\mapsto2x$ überein, bis auf einen Fehler des Heun-Verfahrens unter $10^{-5}$. Zufällige Stöße und eine deterministische Streckung haben dasselbe Histogramm erzeugt.

<!--en:d7a1b7fe-->
### Die Wahrscheinlichkeitsfluss-ODE

<!--en:4cdcd580-->
Die Wärmeleitungsgleichung war ein Beispiel. Nach dem obigen Satz funktioniert dieselbe Konstruktion für jede SDE der hier betrachteten Form. Folgen wir der Geschwindigkeit $\mathbf u_t$, erhalten wir eine ODE,

<!--en:ab8dde3c-->
die **Wahrscheinlichkeitsfluss-ODE** (engl. probability-flow ODE) :cite:`song2021score`. Die nächste Proposition ist das erste von zwei Ergebnissen dieses Abschnitts, die wir für generative Modelle brauchen.

<!--en:bcbf2838-->
**Proposition (der deterministische Zwilling).** *Sei $p_t$ die Randdichte von $d\mathbf{X} = \mathbf{f}\,dt + g\,d\mathbf{W}$ mit $\mathbf{X}_0 \sim p_0$. Wir setzen voraus, dass sie glatt und positiv ist, sodass sie die Fokker-Planck-Gleichung löst. Das Feld $\mathbf u_t$ sei stetig in $t$, stetig differenzierbar in $\mathbf x$ und Lipschitz-stetig in $\mathbf x$ mit einer von $t$ unabhängigen Konstanten. Sei $\mathbf{x}(t)$ eine Lösung von :eqref:`eq_mdl-dyn-pf-ode` mit $\mathbf{x}(0) \sim p_0$. Dann gilt $\mathbf{x}(t) \sim p_t$ für jedes $t$. ODE und SDE haben also zu jedem Zeitpunkt dieselbe Randverteilung.*

<!--en:fa7688b1-->
**Beweis.** Nach dem Existenz- und Eindeutigkeitssatz aus :numref:`sec_mdl-ode-existence-uniqueness` hat ein Lipschitz-stetiges Feld mit einer von $t$ unabhängigen Konstanten vorwärts wie rückwärts genau eine Trajektorie durch jeden Punkt. Seine Flussabbildung zur Zeit $t$ ist daher eine Bijektion auf $\mathbb R^d$. Sei $q_t$ die Dichte des ODE-Zustands. Hier liegt kein Zirkelschluss vor. Das Feld $\mathbf{u}_t$ bestimmen wir aus $p$, unabhängig von $q$. Die Geschwindigkeit ist also bekannt, und $q_t$ erfüllt die Kontinuitätsgleichung $\partial_t q + \nabla \cdot (q \mathbf{u}_t)=0$. Nach dem vorigen Satz löst $p_t$ dieselbe Gleichung, und beide beginnen bei $q_0 = p_0$. Für die Herleitung der Regel :eqref:`eq_mdl-dyn-log-density-rule` für die Log-Dichte brauchten wir nur diese partielle Differentialgleichung. Entlang jeder Charakteristik ändern sich $\log q_t$ und $\log p_t$ daher mit derselben Rate $-\nabla\cdot\mathbf u_t$ und haben denselben Anfangswert. Da die Charakteristiken jeden Punkt erreichen, gilt überall $q_t = p_t$. $\blacksquare$

<!--en:52f109a3-->
Die Lipschitz-Bedingung gilt für die OU-Drift und für den Score einer Gauß-Verteilung oder einer Gauß-Mischung mit positiven Komponentenvarianzen. Die Steigung des Scores ist für $t$ in $[0,T]$ beschränkt.

<!--en:1dc37f5b-->
Die Randverteilungen stimmen überein, die Pfade unterscheiden sich. Ein SDE-Pfad ist rau und kreuzt andere Pfade. Eine ODE-Trajektorie ist glatt. Wegen der Eindeutigkeit treffen zwei verschiedene Trajektorien nie zur selben Zeit am selben Punkt zusammen (Teil (b) von :numref:`fig_mdl-dyn-diffusion-outward-flow`).

<!--en:02bb7aee-->
Die ODE liefert auch eine Likelihood. Integrieren wir die Regel :eqref:`eq_mdl-dyn-log-density-rule` für die Log-Dichte mit $q_t=p_t$ entlang einer Trajektorie von $0$ nach $T$, erhalten wir

<!--en:ad9a3717-->
Die Log-Dichte eines Datenpunkts ergibt sich aus der Log-Dichte seines Bildes zum Zeitpunkt $T$ zuzüglich der entlang des Weges aufsummierten Ausdehnung. Die Dichte zum Endzeitpunkt ist annähernd gaußsch und leicht auszuwerten. Ersetzen wir den exakten Score durch einen erlernten, erhalten wir bis auf den Fehler des Lösungsverfahrens die Log-Dichte des durch dieses Feld definierten *Modells* :cite:`song2021score`. Das ist eine der beiden Größen, mit denen man die Likelihood eines Diffusionsmodells angibt. Die andere ist eine Variationsschranke (:numref:`sec_diffusion-design-space`).

<!--en:5c5184eb-->
### Randverteilungen von SDE und ODE im Vergleich

<!--en:b1e8732b-->
Beim Wärmefluss haben wir eine einzige Gauß-Verteilung gestreckt. Bei der Mischung sagt der Satz mehr aus. Die Dichte ändert ihre Form von zwei Modi zu einem. Dennoch müssen der Prozess mit rauen Pfaden und der mit glatten Pfaden zu jedem Zeitpunkt dieselbe Dichte haben. Aus einer Stichprobe aus $p_0$ starten wir zwei Wolken mit je $20{,}000$ Teilchen. Eine folgt der SDE mit Euler-Maruyama-Schritten, die andere der Wahrscheinlichkeitsfluss-ODE mit Heun-Schritten und dem exakten Score. Als Maßstab berechnen wir für jede Wolke den Einstichproben-Kolmogorov-Smirnov-Abstand zur exakten Verteilungsfunktion von $p_t$, also die größte Differenz zwischen empirischer und exakter Verteilungsfunktion. Bei einer exakten Stichprobe der Größe $n$ übersteigt dieser Abstand den Wert $1.36/\sqrt n$ nur in $5\%$ der Fälle. Schätzen Sie vor dem Blick auf die Ausgabe, bei welcher Wolke sich der Abstand von einer Momentaufnahme zur nächsten stärker ändert.

<!--en:ea80f902-->
In den meisten Läufen bleiben beide Wolken zu allen drei Zeitpunkten innerhalb des $0.0096$-Bandes. Die Abstände liegen etwa zwischen $0.003$ und $0.01$. Ein Wert knapp oberhalb des Bandes ist unbedenklich. Das Band entspricht einer $5\%$-Schwelle, die auch eine exakte Stichprobe in einem von zwanzig Fällen überschreitet. Bei $t=0.25$ ist die Randverteilung noch bimodal, und beide Histogramme geben ihre beiden Modi wieder. Bei $t=1$ hat sie nur noch einen Modus (die Modi verschmelzen bei $0.78$, Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging`).

<!--en:a737b9c2-->
Nun zur Vermutung. Der Abstand der ODE ist zu allen drei Zeitpunkten bis auf die letzte ausgegebene Stelle gleich. Eine streng monoton wachsende Abbildung, die $p_0$ in $p_t$ überführt, transportiert auch die empirische Verteilungsfunktion der Stichprobe mit. Der Abstand der ODE-Wolke bleibt also bis auf den Fehler des Lösungsverfahrens gleich dem Abstand der Anfangsstichprobe. Bei der SDE-Wolke wird in jedem Schritt neues Rauschen gezogen, und ihr Abstand schwankt.

<!--en:3a916e91-->
## Zeitumkehr

<!--en:72df63eb-->
Die Vorwärts-SDE verwandelt Daten in Rauschen. Um Daten zu erzeugen, brauchen wir den Prozess in umgekehrter Richtung, vom Rauschen zu den Daten. Das mag unmöglich erscheinen. Die Wärmeleitungsgleichung lässt sich von einer beliebigen Enddichte aus nicht rückwärts lösen, denn die Glättung vernichtet Details unwiederbringlich. Wir lösen sie hier auch nicht rückwärts. Die Vorwärtsfamilie $p_t$ ist gegeben, und wir suchen nur einen Prozess, der sie in umgekehrter Reihenfolge durchläuft (:numref:`fig_mdl-dyn-forward-reverse`). Seine Drift enthält $\nabla\log p_t$ und damit die Information über die verlorenen Details. Die rückwärts integrierte Wahrscheinlichkeitsfluss-ODE ist ein solcher Prozess. Bei der SDE ist die Antwort weniger offensichtlich. Der Rückwärtsprozess ist wieder ein Diffusionsprozess, und seine Drift enthält die doppelte Score-Korrektur der ODE.

<!--en:5d32bcbb-->
![Exakte Randdichten unserer Beispielmischung $p_0=\frac12\mathcal N(-2,0.25)+\frac12\mathcal N(2,0.25)$ zu verschiedenen Zeiten im OU-Prozess mit $\theta=1$ und $g=\sqrt2$. Die Rückwärtsprozesse dieses Abschnitts durchlaufen dieselben Randdichten von rechts nach links. Beim flachen Gipfel bei $t\approx0.78$ verschmelzen die beiden Modi.](../img/mdl-dyn-forward-reverse.svg)

<!--en:422b0f12-->
### Die Bayes-Regel für einen infinitesimalen Schritt

<!--en:ce5f1f52-->
Um einen kleinen Schritt umzukehren, fragen wir mit der Bayes-Regel, woher ein Teilchen kam. Über ein kurzes Intervall $\tau$ hat der Vorwärtsschritt nach Euler-Maruyama eine Gauß-Verteilung (:numref:`sec_mdl-euler-maruyama`):

<!--en:d5d184dc-->
Bedingt auf den Endpunkt $\mathbf{x}_{t+\tau} = \mathbf{y}$ liefert die Bayes-Regel die Verteilung der möglichen Anfangspunkte. Dazu gewichten wir den Vorwärtskern mit der Randdichte zum Zeitpunkt $t$:

<!--en:c7687878-->
Ein Anfangspunkt ist plausibel, wenn das Teilchen von dort $\mathbf y$ erreichen konnte und der Punkt schon zu Beginn wahrscheinlich war. Wir logarithmieren und entwickeln $\log p_t(\mathbf{x})$ um den Endpunkt. Das ergibt $\log p_t(\mathbf{x}) \approx \log p_t(\mathbf{y}) + (\mathbf{x} - \mathbf{y}) \cdot \nabla \log p_t(\mathbf{y})$. Der Schritt $\mathbf x-\mathbf y$ hat die Größenordnung $\sqrt{\tau}$. Der vernachlässigte quadratische Term ist also von der Ordnung $\tau$ und ändert die A-posteriori-Verteilung erst in höherer Ordnung. Mit derselben Genauigkeit dürfen wir $\mathbf f$ bei $\mathbf y$ statt bei $\mathbf x$ auswerten, denn $\mathbf f(\mathbf x,t)-\mathbf f(\mathbf y,t)$ hat die Größenordnung $\sqrt\tau$ und wird mit $\tau$ multipliziert. Ergänzen wir den Exponenten der Gauß-Dichte um einen linearen Term, verschiebt sich der Mittelwert um Varianz mal Steigung. Die Varianz bleibt gleich, wie die quadratische Ergänzung zeigt. Der Exponent lautet $-\|\mathbf{x} - (\mathbf{y} - \mathbf{f}(\mathbf y,t)\,\tau)\|^2 / (2 g^2 \tau)$, die Varianz ist $g^2\tau$ und die Steigung $\nabla \log p_t(\mathbf{y})$:

<!--en:708e3d72-->
Der Rückwärtsschritt enthält einen gaußschen Stoß mit derselben Varianz $g^2\tau$. Seine Drift ergibt sich aus der Vorwärtsdrift durch eine Korrektur um das Produkt aus dem vollen Faktor $g^2$ und dem Score. Die Randdichte $p_t$ wirkt wie eine A-priori-Verteilung. Sie lenkt den Rückwärtsschritt dorthin, wo die Datendichte zuvor hoch war.

<!--en:b2871f17-->
### Die zeitumgekehrte SDE

<!--en:87065731-->
Die Rechnung für einen Schritt ist heuristisch. Sie beginnt mit einem näherungsweisen Kern und entwickelt bis zur ersten Ordnung. :citet:`Anderson.1982` bewies, dass das Ergebnis für den gesamten Prozess gilt. Wir geben seinen Satz ohne Beweis an. Die Zeitumkehr des Diffusionsprozesses $d\mathbf{X} = \mathbf{f}\,dt + g\,d\mathbf{W}$ ist die **zeitumgekehrte SDE** (engl. reverse-time SDE)

<!--en:c01c2576-->
integriert von $t = T$ abwärts bis $t = 0$. Dabei ist $\bar{\mathbf{W}}$ eine in der Zeit rückwärts laufende brownsche Bewegung. In der Simulation ziehen wir dafür bei jedem Rückwärtsschritt ein neues gaußsches Inkrement. Startet der Prozess mit $\mathbf{X}_T \sim p_T$, hat er zu jeder Zwischenzeit die Randdichte $p_t$ und liefert bei $t=0$ die Datendichte. Der Satz bestimmt den gesamten umgekehrten Prozess. Im Folgenden beweisen wir nur die Aussage über die Randverteilungen, die wir für das Sampling brauchen.

<!--en:c3b9e5aa-->
Schrittweise gelesen entspricht die Gleichung der Rechnung im vorigen Unterabschnitt. Der Übergang von $t+\tau$ abwärts zu $t$ ist ein Schritt mit $dt=-\tau$. Das Teilchen bewegt sich also um $-[\mathbf f-g^2\nabla\log p_t]\,\tau$ und erhält einen Stoß der Varianz $g^2\tau$. Das ergibt die Gauß-Verteilung :eqref:`eq_mdl-dyn-reverse-step`. Einfacher rechnen wir mit einer vorwärts laufenden Uhr. In der **umgekehrten Zeit** $\bar t=T-t$ ist derselbe Prozess eine gewöhnliche SDE mit Drift $-\mathbf f+g^2\,\nabla\log p_t$ und Rauschen $g$. Mit der Zeitrichtung wechselt die Drift ihr Vorzeichen, auf das Rauschen hat die Richtung keinen Einfluss.

<!--en:8bee5a33-->
Den einfachsten Fall können wir von Hand prüfen. Betrachten wir den stationären OU-Prozess mit $p_t=\mathcal N(0,1)$ für alle $t$, mit $\theta=1$, $g=\sqrt2$ und Score $-x$. Die Geschwindigkeit des Wahrscheinlichkeitsflusses ist $-x-\tfrac12\cdot2\cdot(-x)=0$. Die ODE-Teilchen bewegen sich also nicht, und die Wolke bleibt $\mathcal N(0,1)$-verteilt. In der umgekehrten Zeit hat die zeitumgekehrte SDE die Drift $-f+g^2s_t=x-2x=-x$ und das Rauschen $\sqrt2$. Das ergibt denselben OU-Prozess. Der stationäre OU-Prozess ist seine eigene Zeitumkehr.

<!--en:7a1b6361-->
Mit $\tfrac12g^2$ anstelle von $g^2$ wäre die Rückwärtsdrift $x-x=0$, und die Wolke würde sich wie bei einer brownschen Bewegung ausbreiten. Die zeitumgekehrte SDE braucht also den vollen Faktor $g^2$ und damit die doppelte Korrektur der ODE. Eine Familie von Samplern erklärt diesen Faktor.

<!--en:23970973-->
**Proposition (eine Familie mit festen Randverteilungen).** *Sei $p_t$ die Randdichte der Vorwärts-SDE unter den Voraussetzungen des Satzes über den deterministischen Zwilling. Außerdem sei $\mathbf f$ Lipschitz-stetig in $\mathbf x$. Für jedes $\lambda \ge 0$ hat die zeitumgekehrte SDE mit Rauschamplitude $\lambda g$ und Drift*

<!--en:184f6410-->
*die Randdichten $p_t$, wenn sie mit $p_T$ startet. Für $\lambda = 1$ ergibt sich Andersons zeitumgekehrte SDE :eqref:`eq_mdl-dyn-reverse-sde`, für $\lambda = 0$ die Wahrscheinlichkeitsfluss-ODE :eqref:`eq_mdl-dyn-pf-ode`.*

<!--en:c90b185e-->
Je mehr Rauschen ein Sampler hinzufügt, desto größer ist seine Score-Korrektur.

<!--en:67a709fd-->
**Beweis.** Wir verwenden die umgekehrte Zeit $\bar t = T - t$ und setzen $q_{\bar t} = p_{T-\bar t}$. Damit durchlaufen wir die Entwicklung der Dichte rückwärts. Es gilt also $\partial_{\bar t} q = -\partial_t p = \nabla \cdot (\mathbf{f} q) - \tfrac12 g^2 \Delta q$, wobei wir $\mathbf f$ und $g$ zur Vorwärtszeit $T-\bar t$ auswerten. Ein Prozess $d\mathbf{Y} = \mathbf{b}'\,d\bar t + \lambda g\, d\mathbf{W}$ in der umgekehrten Zeit hat seine eigene Fokker-Planck-Gleichung $\partial_{\bar t} q = -\nabla \cdot (q\, \mathbf{b}') + \tfrac{\lambda^2}{2} g^2 \Delta q$. Damit die beiden Gleichungen übereinstimmen, brauchen wir

<!--en:b2df838d-->
Dabei haben wir *beide* Laplace-Terme mit :eqref:`eq_mdl-dyn-diffusion-as-transport` in Transportform geschrieben. Die Gleichung gilt für $\mathbf{b}' = -\mathbf{f} + \tfrac{1+\lambda^2}{2} g^2 \nabla \log q$. Eine Drift $\mathbf b'$ in der umgekehrten Zeit entspricht einer Drift $-\mathbf b'$ in der Vorwärtszeit. Daraus folgt :eqref:`eq_mdl-dyn-reverse-family`. Nun müssen wir aus gleichen Entwicklungsgleichungen mit gleicher Anfangsdichte $q_0 = p_T$ auf gleiche Randdichten schließen. Dazu brauchen wir Eindeutigkeit. Für $\lambda = 0$ greift das Argument entlang der Charakteristiken aus dem Beweis für den deterministischen Zwilling. Für $\lambda > 0$ verwenden wir die Eindeutigkeitsaussage nach :eqref:`eq_mdl-dyn-fokker-planck`, die wir nicht bewiesen haben. Sie gilt hier, weil $\mathbf b'$ Lipschitz-stetig ist, wenn $\mathbf f$ und $\mathbf u_t$ Lipschitz-stetig sind. $\blacksquare$

<!--en:e2b67213-->
Der Beweis erklärt auch den Faktor zwei. Ein Rückwärtsprozess mit Rauschen $\lambda g$ glättet seine eigene Dichte. Seine Drift muss also zwei Glättungen rückgängig machen. Ein Term $\tfrac12 g^2 \nabla \log p_t$ hebt die Vorwärtsglättung auf. Das ist die gesamte Korrektur der Wahrscheinlichkeitsfluss-ODE, die kein Rauschen hinzufügt. Ein weiterer Term $\tfrac{\lambda^2}2 g^2 \nabla \log p_t$ gleicht die Glättung durch das eigene Rauschen des Rückwärtsprozesses aus. Andersons $\lambda = 1$ enthält beide Hälften. Die Randdichten legen die Drift allerdings nicht fest. Jede andere Drift mit denselben Randdichten unterscheidet sich von $\mathbf b'$ um ein Feld $\mathbf w$ mit $\nabla\cdot(q\mathbf w)=0$. In Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-divergence-free` konstruieren Sie solche Felder.

<!--en:83b394fb-->
Wozu dient diese Wahlfreiheit? Rauschen lässt einen Sampler Fehler vergessen. Ein Fehler in der Zustandsverteilung kann aus einer geringfügig von $p_T$ abweichenden Anfangsverteilung oder aus einem früheren Schritt stammen. Die ODE trägt ihn bis $t=0$ mit, denn ihre Abbildung ist bijektiv und verliert keine Information. Das Rauschen einer SDE baut einen Teil des Fehlers ab (Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-prior-mismatch` präzisiert beide Aussagen). Dafür ist der Score-Term größer, und ein langer verrauschter Schritt verursacht selbst einen Fehler. Bei Samplern für Diffusionsmodelle berücksichtigt man dies bei der Wahl von $\lambda$ (:numref:`sec_diffusion-samplers`). Unabhängig davon braucht jeder Sampler dasselbe Feld $\nabla\log p_t$. Eine einzige Schätzung des Scores genügt also für die gesamte Familie.

<!--en:18047dc2-->
:numref:`fig_mdl-dyn-lambda-family` zeigt drei Werte von $\lambda$ für unsere Beispielmischung. Die deterministischen Pfade teilen sich nahe $\bar t\approx2.2$ auf die beiden Modi auf. Das entspricht ungefähr der Vorwärtszeit $0.78$, zu der die Randverteilung ihre Bimodalität verliert.

<!--en:41ac9ef8-->
![Trajektorien in umgekehrter Zeit von denselben vier Anfangspunkten bei $t=T$ mit dem exakten Score der Beispielmischung, Drift $\mathbf{f} - \tfrac{1+\lambda^2}{2} g^2 \nabla \log p_t$ und Rauschen $\lambda g$. Die Wahrscheinlichkeitsfluss-ODE ($\lambda = 0$) ist deterministisch. Für $\lambda = 0.5$ und bei Andersons zeitumgekehrter SDE ($\lambda = 1$) kommen Rauschen und eine größere Score-Korrektur hinzu. Die drei Histogramme bei $t=0$ mit je $20{,}000$ Teilchen (rechts) stimmen mit der Datendichte $p_0$ überein.](../img/mdl-dyn-lambda-family.svg)

<!--en:c519badb-->
### Sampling in umgekehrter Zeit

<!--en:94cdf334-->
Erzeugen wir nun Daten mit dem exakten Score anstelle eines trainierten Netzes. Für dieses eine Experiment wählen wir eine asymmetrische Datenverteilung. Die Modi behalten ihre Position und Breite, aber vier Fünftel der Masse liegen links: $p_0=0.8\,\mathcal N(-2,0.25)+0.2\,\mathcal N(2,0.25)$. Ein Sampler, der mit symmetrischem Rauschen startet, muss also neben den Modi auch ihre Anteile treffen. An diesem Test kann er scheitern.

<!--en:1574b215-->
Wir ziehen $20{,}000$ Punkte aus $\mathcal{N}(0, 1)$ als Näherung für $p_T$ bei $T = 3$. Dann integrieren wir Andersons zeitumgekehrte SDE mit Euler-Maruyama-Schritten in der umgekehrten Zeit bis $t = 0$. Die Codezelle misst zuerst, wie weit $p_T$ von $\mathcal N(0,1)$ entfernt ist. Anschließend vergleicht sie die Stichprobe mit den Daten anhand der Masse links vom Ursprung, der Mittelwerte und Breiten der Modi sowie des Kolmogorov-Smirnov-Abstands (KS-Abstands) zu $p_0$. Daneben gibt sie den Abstand einer exakten Stichprobe gleicher Größe aus.

<!--en:37c994ca-->
Der Sampler rekonstruiert die Datenverteilung. Seine Modi liegen höchstens etwa $0.02$ von $\pm2$ entfernt, ihre Breiten liegen nahe $0.5$, und der Anteil der Teilchen, die links enden, beträgt bis auf ein Hundertstel vier Fünftel. Auf diese letzte Zahl kommt es an. Die Hälfte der Teilchen startete rechts vom Ursprung, und allein der Score enthält im Sampler Informationen über die Gewichte. Wo steckt diese Information? Kaum in der Nähe von $t=0$. Dort sind die Zugehörigkeitswahrscheinlichkeiten in :eqref:`eq_mdl-dyn-mixture-score` überall, wo die Dichte nennenswert ist, praktisch $0$ oder $1$. Unabhängig von den Gewichten entspricht der Score einer der beiden Federkräfte $-(x\pm2)/0.25$. Diese Unempfindlichkeit des Scores gegenüber den Gewichten weit getrennter Modi haben wir in :numref:`sec_mdl-fisher-divergence` festgestellt. Auch bei $t=3$ entscheiden sich die Anteile nicht. Dort entsprechen die beiden Komponenten fast derselben Gauß-Verteilung. Wo auch immer ein Teilchen liegt, die Zugehörigkeitswahrscheinlichkeit der linken Komponente liegt nahe an ihrem Gewicht $0.8$. Aber auch die Federkräfte stimmen fast überein. Ihre Differenz beträgt $2m_t/v_t$ bei Komponentenmittelwerten $\mp m_t$ und Varianz $v_t$, also $0.2$ bei $t=3$ gegenüber $16$ bei $t=0$. Mitteln wir zwei fast gleiche Federkräfte mit den Gewichten $0.8$ und $0.2$, ergibt sich ein kleiner stetiger Schub nach links. Er legt für kein Teilchen den Modus fest.

<!--en:f53b09bb-->
Die Gewichte wirken dazwischen, während sich die Modi trennen. Am Punkt $x^*_t=v_t\ln(0.8/0.2)/(2m_t)$ hat ein Teilchen für beide Komponenten dieselbe Zugehörigkeitswahrscheinlichkeit. Ein Teilchen links davon stammt mit höherer Wahrscheinlichkeit aus der linken Komponente und endet im Rückwärtsprozess auch mit höherer Wahrscheinlichkeit dort. Bei $t=3$ liegt die Trennlinie bei $x^*\approx7$, weit rechts von allen Teilchen. Bei $t=1$ hat sie bereits $0.85$ erreicht. Ein Teilchen bei $x=+0.5$ endet also immer noch eher links als rechts. Bei $t=0$ liegt sie bei $0.09$ im Tal, das inzwischen leer ist. Bei starkem Rauschen wirken die Gewichte überall, entscheiden aber nichts. Bei schwachem Rauschen könnten sie den Ausschlag geben, doch dann ist kein Teilchen mehr im Tal.

<!--en:612bbd4c-->
Die Anfangsverteilung war nicht ganz richtig, doch das fiel kaum ins Gewicht. Bei ungleichen Gewichten hat $p_T$ noch nicht jede Spur der Daten verloren. Ihr Mittelwert beträgt $-1.2\,e^{-3}\approx-0.06$, und die erste Zeile zeigt einen Abstand von bis zu $0.0145$ zur Dichte von $\mathcal N(0,1)$ bei einer Gipfelhöhe von $0.40$. Das Rauschen der zeitumgekehrten SDE gleicht diese Abweichung größtenteils aus. Der linke Modus verliert dadurch etwa $0.002$ an Masse, weniger als die binomiale Schwankung $0.003$ des ausgegebenen Anteils. Bei der Wahrscheinlichkeitsfluss-ODE, die nichts vergisst, wäre der Verlust um ein Mehrfaches größer (Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-prior-mismatch`).

<!--en:39077580-->
Der Sampler ist nicht exakt. Die Zeile zum KS-Abstand lässt das allenfalls erahnen. Beide Abstände haben die Größenordnung des $0.0096$-Bandes. Der Abstand des Samplers ist meist größer und überschreitet das Band etwa in jedem sechsten Lauf. Eine exakte Stichprobe überschreitet es nur in jedem zwanzigsten Lauf. Mit einem einzigen Lauf dieser Größe lässt sich der Sampler nicht von einem exakten unterscheiden. Neben der Anfangsverteilung gibt es eine zweite Fehlerquelle, die Diskretisierung. Euler-Maruyama hat schwache Ordnung eins (:numref:`sec_mdl-euler-maruyama`). Die Ausgabe weist daher eine zur Schrittweite proportionale Verzerrung auf. Die Verzerrung steckt in den Breiten, lässt sich aber mit einem Lauf nicht messen. Der Monte-Carlo-Fehler einer aus $M$ Samples geschätzten Breite beträgt die Breite geteilt durch $\sqrt{2M}$. Das ergibt etwa $0.003$ für den linken Modus mit $16{,}000$ Samples und $0.006$ für den rechten mit $4{,}000$.

<!--en:e4f1abe9-->
Die Verzerrung lässt sich ohne Sampling isolieren. Für gaußsche Daten $\mathcal N(2,0.25)$, also für den rechten Modus allein, ist die Randverteilung eine Gauß-Verteilung $\mathcal N(m_t,v_t)$ mit Score $-(x-m_t)/v_t$. Jeder Rückwärtsschritt ist dann affin in $x$. Mittelwert $m$ und Varianz $V$ des Samplerzustands genügen daher einer exakten Rekursion. Mit der Steigung $\kappa_t=\theta-g^2/v_t$ der Rückwärtsdrift lautet sie $m\leftarrow m+\Delta t\,(\kappa_tm+g^2m_t/v_t)$ und $V\leftarrow(1+\kappa_t\Delta t)^2V+g^2\Delta t$. Starten wir mit der exakten Verteilung $p_T$, bleibt nur der Diskretisierungsfehler. Die Codezelle berechnet die Rekursion auf gleichmäßigen Gittern mit $300$ bis $2{,}400$ Schritten. Anschließend verteilt sie $1{,}500$ Schritte auf zwei Arten, einmal gleichmäßig und einmal mit $1{,}000$ Schritten im Bereich $t\le0.5$.

<!--en:b707ca9a-->
Auf den ersten vier Gittern überschreitet die Breite den Sollwert $0.5$ um $0.0096$, $0.0047$, $0.0024$ und $0.0012$. Bei halbierter Schrittweite halbiert sich die Abweichung, wie es bei einer Verzerrung erster Ordnung sein muss. Bei $600$ Schritten, dem Gitter unseres Sampling-Laufs, beträgt sie $0.005$. Eine Abweichung dieser Größe ist auch bei den ausgegebenen Breiten zu erwarten. Sie liegt im Bereich ihrer Monte-Carlo-Fehler. Ein einzelner Sampling-Lauf kann sie daher nicht auflösen, die Rekursion schon.

<!--en:77f9f8fc-->
Die letzten beiden Zeilen untersuchen, wo die Schritte liegen sollten. Verteilen wir $1{,}500$ Schritte gleichmäßig, beseitigen wir sechs Zehntel der Verzerrung des Gitters mit $600$ Schritten. Verfeinern wir die Schritte nur auf $t\le0.5$ um den Faktor zehn, entfallen bei ebenfalls insgesamt $1{,}500$ Schritten fast neun Zehntel der Verzerrung. Eine zehnfache Verfeinerung reduziert den Beitrag aus dem verfeinerten Bereich auf ein Zehntel. Der Rückgang von $0.0047$ auf $0.0006$ zeigt also, dass fast die gesamte Verzerrung der Ausgabe aus den Schritten auf $t\le0.5$ stammt, dem letzten Sechstel des Rückwärtslaufs.

<!--en:b0e28ba9-->
Zwei Effekte kommen zusammen. Die Rückwärtsdrift ist dort am steilsten, mit einer Steigung von $\theta-g^2/v_0=1-8=-7$ bei $t=0$ gegenüber $-1$ nahe $t=T$. Daher verursacht jeder Schritt einen größeren Fehler. Außerdem verkleinert jeder spätere Schritt einen früh im Rückwärtslauf entstandenen Fehler um den Faktor $(1+\kappa_t\Delta t)^2<1$ der Rekursion. Fehler aus den letzten Schritten erreichen die Ausgabe dagegen fast ungeschwächt. Bei schmaleren Datenverteilungen wächst der Betrag der Steigung $\theta-g^2/v_0$ unbegrenzt. Die Schrittweite stößt dann an die Stabilitätsgrenze aus :numref:`sec_mdl-stiffness-implicit`, von der wir hier weit entfernt sind. In Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-step-placement` messen Sie beide Effekte und suchen die beste Verteilung der Schritte.

<!--en:069819aa-->
All dies beruhte auf einem Score, für den wir eine Formel hatten. Die Daten bildeten eine Mischung mit zwei Komponenten. Als Fehler blieben nur die Verzerrung des Euler-Maruyama-Verfahrens und der kleine Unterschied zwischen $p_T$ und $\mathcal N(0,1)$. Für reale Daten gibt es keine solche Formel. Auch die Berechnung von $p_t$ aus der Fokker-Planck-Gleichung scheidet aus. Ein Gitter mit $100$ Zellen pro Achse hat $100^d$ Unbekannte, und $d$ ist die Anzahl der Pixel. Wir müssen den Score lernen (:numref:`sec_mdl-score-matching-diffusion-flow`). Bei Bildern stehen uns nur Samples aus $p_0$ zur Verfügung.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:a280bae1-->
Die Dichte einer SDE $d\mathbf X=\mathbf f\,dt+g\,d\mathbf W$ erfüllt die Fokker-Planck-Gleichung $\partial_tp=-\nabla\cdot(\mathbf fp)+\tfrac12g^2\Delta p$. Die Drift transportiert die Dichte, das Rauschen glättet sie. Wegen $\nabla p=p\nabla\log p$ ist auch die Glättung ein Transport mit Geschwindigkeit $-\tfrac12g^2\nabla\log p$. Dieselbe Dichte entwickelt sich daher gemäß einer gewöhnlichen Differentialgleichung, der Wahrscheinlichkeitsfluss-ODE $\dot{\mathbf x}=\mathbf f-\tfrac12g^2\nabla\log p_t$. Diese ODE hat zu jeder Zeit dieselbe Randverteilung wie die SDE. Ihre Trajektorien sind glatt und kreuzen sich nie, und sie liefert eine Likelihood. Eine Familie von Prozessen mit Rauschen $\lambda g$ und Drift $\mathbf f-\tfrac{1+\lambda^2}2g^2\nabla\log p_t$ durchläuft dieselben Dichten in umgekehrter Reihenfolge. Die erste Hälfte der Score-Korrektur macht die Vorwärtsglättung rückgängig, der Rest kompensiert das eigene Rauschen des Samplers. Deshalb hat Andersons zeitumgekehrte SDE ($\lambda=1$) die doppelte Korrektur der ODE ($\lambda=0$). Für die Randverteilungen haben wir dies unter der Voraussetzung einer eindeutigen Lösung der Fokker-Planck-Gleichung bewiesen. Für den gesamten Prozess haben wir Andersons Satz zitiert. Mit dem exakten Score verwandelte die zeitumgekehrte SDE symmetrisches Rauschen in eine Mischung mit den Gewichten $0.8$ und $0.2$. Sie rekonstruierte sowohl die Modi als auch deren Anteile. Die Anteile entscheiden sich bei mittlerem Rauschen, während sich die Modi trennen. Bei all diesen Samplern ist der Score $\nabla\log p_t$ die einzige Unbekannte.

<!--en:2294243a-->
## Übungen

<!--en:4f5c5338-->
1. **Der Laplace-Operator als Mittelung.** Mitteln wir eine glatte Funktion $\psi$ auf der Kugeloberfläche mit Radius $r$ um $\mathbf{x}$, so ergibt sich $\psi(\mathbf{x}) + \frac{r^2}{2d}\Delta \psi(\mathbf{x}) + O(r^4)$, wie in :numref:`sec_mdl-dyn-vector-calculus` angegeben.
    1. Sei $\mathbf{e}$ auf der Einheitssphäre des $\mathbb{R}^d$ gleichverteilt. Zeigen Sie anhand der Symmetrie, dass $\mathbb{E}[e_i] = 0$, $\mathbb{E}[e_i e_j] = \delta_{ij}/d$ und $\mathbb{E}[e_i e_j e_k] = 0$ gelten. Beweisen Sie die Mittelwertformel, indem Sie diese Momente in einer Taylor-Entwicklung von $\psi(\mathbf{x} + r\mathbf{e})$ verwenden. Warum tritt kein Term mit $r^3$ auf?
    1. In einer Dimension besteht die Kugeloberfläche aus den beiden Punkten $x \pm \delta$. Betrachten Sie einen Vorwärts-Euler-Schritt für die Wärmeleitungsgleichung $\partial_t p = D\,\partial_{xx} p$. Ersetzen Sie $\partial_{xx} p$ durch die zentrale Differenz $(p_{i+1} - 2p_i + p_{i-1})/\delta^2$. Zeigen Sie, dass jeder neue Wert $p_i$ eine gewichtete Kombination aus $p_{i-1}$, $p_i$ und $p_{i+1}$ ist, deren Gewichte sich zu eins summieren. Bestimmen Sie die größte Schrittweite $\Delta t$, für die alle drei Gewichte nichtnegativ sind. Warum kann die Iteration dann nie ein neues Maximum erzeugen?
   :exlabel:`ex_mdl-fokker-planck-probability-flow-laplacian-averaging`

<!--en:f1cd7307-->
1. **Momente aus der schwachen Form.** Das Entwicklungsgesetz :eqref:`eq_mdl-dyn-expectation-evolution` haben wir für Testfunktionen hergeleitet, die außerhalb einer beschränkten Menge verschwinden. Nehmen Sie an, dass es auch für Polynome $\phi$ gilt. Das ist hier gerechtfertigt, weil die Dichten im Unendlichen gaußsch abfallen.
    1. Setzen Sie für den eindimensionalen OU-Prozess $\phi(x) = x$ und $\phi(x) = x^2$ ein. Leiten Sie ODEs für den Mittelwert $m(t)$ und die Varianz $v(t)$ von $X_t$ her. Zeigen Sie, dass sich die Momentengleichungen :eqref:`eq_mdl-dyn-moment-odes` für jede Anfangsverteilung $p_0$ ergeben, auch wenn die Verteilung nicht gaußsch ist. Prüfen Sie das Ergebnis mit `mixture_params`, indem Sie den Mittelwert und die Varianz der bimodalen Mischung zum Zeitpunkt $t$ berechnen.
    1. Betrachten Sie die lineare SDE $d\mathbf{X} = \mathbf A\mathbf{X}\,dt + g\,d\mathbf{W}$ in $\mathbb{R}^d$ mit einer konstanten Matrix $\mathbf A$. Zeigen Sie mit $\phi(\mathbf{x}) = x_i$ und $\phi(\mathbf{x}) = x_i x_j$, dass Mittelwert und Kovarianz die Gleichungen $\dot{\mathbf{m}} = \mathbf A\mathbf{m}$ und $\dot{\boldsymbol\Sigma} = \mathbf A\boldsymbol\Sigma + \boldsymbol\Sigma\mathbf A^\top + g^2 \mathbf I$ erfüllen. Welche Matrixgleichung muss eine stationäre Kovarianz lösen?
   :exlabel:`ex_mdl-fokker-planck-probability-flow-moments-weak-form`

<!--en:aeb683b2-->
1. **Stationäre Dichte bei verschwindendem Strom.**
    1. Setzen Sie in der Fokker-Planck-Gleichung des OU-Prozesses $\partial_t p = 0$, sodass $0 = \theta\,\partial_x(x p) + \tfrac12 g^2 \partial_{xx} p$ gilt. Integrieren Sie einmal nach $x$. Begründen Sie mit dem Abklingen im Unendlichen, warum die Integrationskonstante verschwindet, die gleich dem negativen Wahrscheinlichkeitsstrom :eqref:`eq_mdl-dyn-probability-current` ist. Lösen Sie die resultierende ODE erster Ordnung und zeigen Sie ohne gaußschen Ansatz, dass $p_\infty = \mathcal{N}(0, g^2/(2\theta))$ gilt.
    1. Wiederholen Sie das Argument für eine allgemeine Drift $f(x)$ bei konstantem $g$ und drücken Sie $p_\infty$ durch ein Potential $U$ mit $f = -U'$ aus. Bestimmen Sie umgekehrt zu einer gegebenen Dichte $p \propto e^{-E(x)}$ die Drift, für die $p$ stationär ist, und drücken Sie sie durch den Score von $p$ aus.
    1. Wenden Sie das Ergebnis auf die Drift $f(x) = x - x^3$ eines Doppelmuldenpotentials mit $g = \sqrt{2}$ an und bestimmen Sie die Modi von $p_\infty$. Was geht bei der instabilen Drift $f(x) = \theta x$ mit $\theta > 0$ schief?
   :exlabel:`ex_mdl-fokker-planck-probability-flow-stationary-density`

<!--en:7e2ec036-->
1. **Verschmelzen der Modi.** Betrachten Sie die symmetrische Mischung aus zwei Komponenten mit Score :eqref:`eq_mdl-dyn-tanh-score`.
    1. Leiten Sie :eqref:`eq_mdl-dyn-tanh-score` aus der Formel :eqref:`eq_mdl-dyn-mixture-score` für die Zugehörigkeitswahrscheinlichkeiten her.
    1. Zeigen Sie, dass der Score für $\mu^2 > v_0$ drei Nullstellen hat und nur die Nullstelle $x = 0$, wenn $\mu^2 \le v_0$ gilt. Ordnen Sie jede Nullstelle als lokales Maximum oder lokales Minimum der Dichte ein. (Hinweis: Die Steigung bei $0$ beträgt $(\mu^2-v_0)/v_0^2$.)
    1. Im OU-Prozess bleibt die Mischung symmetrisch. Ihre Komponenten haben die Mittelwerte $\pm\mu e^{-\theta t}$ und die mit `mixture_params` berechnete Varianz. Bestimmen Sie den Zeitpunkt $t^*$, zu dem $p_t$ für die Parameter dieses Abschnitts aufhört, bimodal zu sein. Verwenden Sie $\mu = 2$, $v_0 = 0.25$, $\theta = 1$ und $g = \sqrt{2}$.
    1. Was bedeutet die Änderung bei $t^*$ für einen Rückwärtssampler, der mit Rauschen startet? (:citet:`Biroli.Bonnaire.deBortoli.ea.2024` untersuchen das hochdimensionale Gegenstück zu diesem Symmetriebruch und nennen es Speziation.)
   :exlabel:`ex_mdl-fokker-planck-probability-flow-mode-merging`

<!--en:710a6b8a-->
1. [code] **Zustandsabhängiges Rauschen.** ● Für eine SDE $d\mathbf X=\mathbf f\,dt+\mathbf G(\mathbf X,t)\,d\mathbf W$ mit matrixwertiger Rauschamplitude schreiben wir $\mathbf a=\mathbf G\mathbf G^\top$. Die Itô-Korrektur lautet dann $\tfrac12\sum_{ij}a_{ij}\,\partial_i\partial_j\phi$. Zweimalige partielle Integration, $\int p\,a_{ij}\,\partial_i\partial_j\phi=\int\phi\,\partial_i\partial_j(a_{ij}p)$, liefert $\partial_tp=-\nabla\cdot\mathbf j$ mit dem Strom

   $$
   j_i=f_i\,p-\tfrac12\sum_k\partial_k(a_{ik}\,p)
   $$

   und nach Division durch $p$ die Geschwindigkeit des Wahrscheinlichkeitsflusses

   $$
   \mathbf u=\mathbf f-\tfrac12\,\mathbf a\,\nabla\log p-\tfrac12\,\nabla\cdot\mathbf a,
   \qquad(\nabla\cdot\mathbf a)_i=\sum_k\partial_ka_{ik}.
   $$

   Bei konstanter Matrix ändert sich im Score-Term nur der Faktor $g^2$ zu $\mathbf a$. Bei zustandsabhängigem $\mathbf a$ kommt der letzte Term hinzu. Betrachten Sie nun $dX = g(X)\,dW$ ohne Drift mit einer glatten positiven Funktion $g$ auf dem Kreis $[0, 1)$ mit periodischen Randbedingungen.
    1. Geben Sie den Strom für Matrixrauschen und die allgemeine Geschwindigkeit von oben für diese SDE an. Zeigen Sie, dass eine stationäre Dichte einen konstanten Strom hat und die Periodizität diese Konstante zum Verschwinden zwingt. Bestimmen Sie $p_\infty$.
    1. Ersetzen wir die Konstante $g^2$ in :eqref:`eq_mdl-dyn-fokker-planck` durch $g(x)^2$, erhalten wir die partielle Differentialgleichung $\partial_t p = \tfrac12 g(x)^2\,\partial_{xx} p$. Welche stationäre Dichte sagt sie voraus?
    1. Simulieren Sie $50{,}000$ Teilchen mit Anfangswerten aus der Gleichverteilung mit Euler-Maruyama, $g(x) = 1 + \tfrac12 \sin(2\pi x)$ und $\Delta t = 10^{-3}$ bis $t = 2$. Nehmen Sie die Positionen modulo $1$. Vergleichen Sie das Histogramm mit beiden Vorhersagen. Wo sammeln sich die Teilchen und warum?
   :exlabel:`ex_mdl-fokker-planck-probability-flow-state-dependent-noise`

<!--en:1d1bfa73-->
1. [code] **Die Fokker-Planck-Gleichung auf einem Gitter.** ● Teilen Sie $[-6, 6]$ in $N$ Zellen der Breite $\delta = 12/N$ mit Mittelpunkten $x_i$. Dabei nähere $p_i(t)$ die Dichte in Zelle $i$ an. Eine konservative Diskretisierung der OU-Gleichung ordnet jeder inneren Zellgrenze $x_{i+1/2}$ den Strom

   $$j_{i+1/2} = f(x_{i+1/2})\,\frac{p_i + p_{i+1}}{2} - \frac{g^2}{2}\,\frac{p_{i+1} - p_i}{\delta},$$

   zu, setzt den Strom durch die beiden äußeren Grenzen auf null und entwickelt die Dichte gemäß $\dot{p}_i = -(j_{i+1/2} - j_{i-1/2})/\delta$. Das ergibt das lineare Gleichungssystem $\dot{\mathbf{p}} = \mathbf F\mathbf{p}$.
    1. Zeigen Sie, dass $\sum_i p_i\,\delta$ für jedes $\mathbf{p}$ erhalten bleibt. Das ist das diskrete Gegenstück zum Divergenzsatz :eqref:`eq_mdl-dyn-divergence-theorem`. Welche Randbedingung aus :numref:`sec_mdl-continuity-equation` erzwingen die verschwindenden äußeren Ströme?
    1. Stellen Sie $\mathbf F$ für $\theta = 1$, $g = \sqrt{2}$ und $\delta = 0.05$ auf. Starten Sie mit der bimodalen Mischung `p_t(x, 0.0)` an den Zellmittelpunkten. Integrieren Sie bis $t = 0.5$ mit dem Vorwärts-Euler-Verfahren bei Schrittweite $\Delta t = 10^{-3}$ und mit dem Rückwärts-Euler-Verfahren bei Schrittweite $\Delta t = 10^{-2}$ (ein Aufruf von `np.linalg.solve` pro Schritt). Vergleichen Sie beide Ergebnisse mit `p_t(x, 0.5)`.
    1. Erhöhen Sie die Schrittweite des Vorwärts-Euler-Verfahrens ausgehend von $10^{-3}$. Bestimmen Sie die größte Schrittweite, bei der ein Lauf bis $t = 5$ stabil bleibt. Diese Zeit reicht aus, damit eine durch Rundungsfehler angeregte instabile Mode anwächst. Setzen Sie diese Schrittweite mit dem am weitesten im Negativen liegenden Eigenwert $-\nu$ von $\mathbf F$ in Beziehung. Verwenden Sie dazu die Schranke $\Delta t < 2/\nu$ aus :numref:`sec_mdl-stiffness-implicit`. Wie ändert sich die Grenze bei Halbierung von $\delta$? Vergleichen Sie sie mit der Bedingung aus Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-laplacian-averaging`b.
    1. Ein Gitter mit $100$ Zellen pro Achse in $d$ Dimensionen hat $100^d$ Unbekannte. Berechnen Sie diese Zahl für $d = 2$ und für $d = 784$, die Pixelzahl eines Bildes der Größe $28 \times 28$. Was folgt daraus für die Verwendung der Fokker-Planck-Gleichung in Diffusionsmodellen?

   *Nach :citet:`Sarkka.Solin.2019`, Aufgabe 5.3.*
   :exlabel:`ex_mdl-fokker-planck-probability-flow-grid`

<!--en:f4321fd8-->
1. **Divergenzfreie Geschwindigkeitsfelder.** ● Sei $\mathbf R$ eine konstante antisymmetrische $d \times d$-Matrix.
    1. Zeigen Sie, dass $\nabla \cdot (p\,\mathbf R\nabla \log p) = 0$ für jede glatte positive Dichte $p$ gilt. Folgern Sie daraus, dass für jedes $\beta \in \mathbb{R}$ die ODE $\dot{\mathbf{x}} = \mathbf{u}_t(\mathbf{x}) + \beta \mathbf R \nabla \log p_t(\mathbf{x})$ mit $\mathbf{u}_t$ aus :eqref:`eq_mdl-dyn-pf-velocity` dieselben Randverteilungen hat wie die Wahrscheinlichkeitsfluss-ODE.
    1. Betrachten Sie den zweidimensionalen OU-Prozess im stationären Zustand mit $p_t = \mathcal{N}(\mathbf{0}, \frac{g^2}{2\theta} \mathbf I)$ für alle $t$ und $\mathbf R = \left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$. Zeigen Sie, dass die Geschwindigkeit des Wahrscheinlichkeitsflusses verschwindet, und beschreiben Sie die Trajektorien der geänderten ODE.
    1. Nehmen Sie die Rotation nun in die SDE selbst auf, also $d\mathbf{X} = (-\theta \mathbf I + \beta \mathbf R)\mathbf{X}\,dt + g\,d\mathbf{W}$ in $\mathbb{R}^2$. Zeigen Sie mit der Kovarianzgleichung aus Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-moments-weak-form`b, dass $\frac{g^2}{2\theta} \mathbf I$ für jedes $\beta$ eine stationäre Kovarianz ist. Die SDE hat also dieselbe stationäre Dichte wie in (b). Berechnen Sie ihren stationären Strom, also den Strom :eqref:`eq_mdl-dyn-probability-current` dieser SDE. Warum versagt in zwei Dimensionen das Argument aus Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-stationary-density`a, das einen stationären Strom zum Verschwinden zwingt?
    1. Nehmen Sie $\mathbf{f} = -\nabla U$ an, sodass $\mathbf{u}_t$ ein Gradientenfeld ist. Betrachten Sie alle Geschwindigkeitsfelder $\mathbf{w}$, deren Kontinuitätsgleichung :eqref:`eq_mdl-dyn-continuity` die Lösung $p_t$ hat. Zeigen Sie, dass $\mathbf{u}_t$ unter ihnen die kleinste kinetische Energie $\mathbb{E}_{p_t}\|\mathbf{w}\|^2$ hat. Die Formulierung von Benamou und Brenier :cite:`Benamou.Brenier.2000` minimiert dieselbe Energie zusätzlich über den Pfad $p_t$.
   :exlabel:`ex_mdl-fokker-planck-probability-flow-divergence-free`

<!--en:00cec83e-->
1. **Exakter gaußscher Rückwärtsschritt.** ● Sei $p_t = \mathcal{N}(m_t, v_t)$ eine gaußsche Randverteilung des OU-Prozesses. Bei einem Schritt der Weite $\tau$ ist der exakte Vorwärtsübergang $X_{t+\tau} \mid X_t = x \sim \mathcal{N}(\alpha_\tau x, \sigma_\tau^2)$ mit $\alpha_\tau = e^{-\theta\tau}$ und $\sigma_\tau^2 = \frac{g^2}{2\theta}(1 - e^{-2\theta\tau})$ gemäß dem Kern :eqref:`eq_mdl-dyn-ou-marginal`.
    1. Zeigen Sie durch Konditionieren der Gauß-Verteilung, dass

       $$X_t \mid X_{t+\tau} = y \;\sim\; \mathcal{N}\!\left(\frac{\sigma_\tau^2\, m_t + \alpha_\tau v_t\, y}{v_{t+\tau}},\; \frac{\sigma_\tau^2\, v_t}{v_{t+\tau}}\right).$$

       gilt. Dabei ist $v_{t+\tau} = \alpha_\tau^2 v_t + \sigma_\tau^2$ die Randvarianz zum Zeitpunkt $t + \tau$.
    1. Entwickeln Sie den Mittelwert und die Varianz bis zur ersten Ordnung in $\tau$. Leiten Sie so den Rückwärtsschritt :eqref:`eq_mdl-dyn-reverse-step` einschließlich des vollen Faktors $g^2$ vor dem Score her.
   :exlabel:`ex_mdl-fokker-planck-probability-flow-gaussian-reverse-step`

<!--en:6e3a5872-->
1. [code] **Verteilung der Zeitschritte.** ● Die Funktion `em_reverse_width` liefert für jedes Zeitgitter die exakte Ausgabebreite des rückwärts angewandten Euler-Maruyama-Verfahrens für die Daten $\mathcal N(2, 0.25)$. Im Text haben wir zwei Gitter mit $1{,}500$ Schritten verglichen.
    1. Starten Sie mit dem gleichmäßigen Gitter mit $600$ Schritten und verfeinern Sie die Schritte nur auf $t \le c$ um den Faktor zehn, jeweils für $c \in \{0.1, 0.25, 0.5, 1, 2\}$. Notieren Sie die Anzahl der Schritte und die Überschreitung der Sollbreite. Eine zehnfache Verfeinerung reduziert die Verzerrung aus dem verfeinerten Bereich auf ein Zehntel. Berechnen Sie damit, welcher Anteil der Verzerrung des Gitters mit $600$ Schritten auf die Schritte im Bereich $t \le c$ entfällt. Wie viele zusätzliche Schritte auf $t \le 0.1$ bewirken so viel wie die $900$ zusätzlichen gleichmäßig verteilten Schritte im Text? Der Text nennt zwei Gründe, weshalb diese Schritte den größten Beitrag liefern. Messen Sie beide Effekte auf dem gleichmäßigen Gitter. Berechnen Sie für den einzelnen Schritt, der bei $t \in \{0.1, 0.5, 3\}$ beginnt, den Fehler in $V$ bei einem Start mit dem exakten Wert $v_t$. Bestimmen Sie auch den Faktor, mit dem die folgenden Schritte diesen Fehler bis $t = 0$ multiplizieren.
    1. Probieren Sie mit einem Budget von $N = 1{,}500$ Schritten die Gitter $t_i = T\,(i/N)^\rho$ für $\rho \in \{1, 1.5, 2, 3, 4, 6\}$ aus. Welches $\rho$ ergibt die geringste Überschreitung? Warum schneidet ein großes $\rho$ wieder schlechter ab?
    1. Ist die Verzerrung auf dem quadratischen Gitter mit $\rho = 2$ noch von erster Ordnung in $1/N$? Prüfen Sie dies mit $N = 750$, $1{,}500$ und $3{,}000$. Was hat das Gitter verändert, die Ordnung oder die Konstante?
    1. Wiederholen Sie (b) für schmalere Daten mit $v_0 = 0.01$. Dort beträgt die Steigung der Rückwärtsdrift bei $t = 0$ genau $1 - 2/0.01 = -199$. Vergleichen Sie jede Überschreitung mit der Sollbreite $0.1$. Wie verschiebt sich der beste Wert für $\rho$? Was legt das für Daten nahe, deren Varianz in einigen Richtungen klein ist? (In :numref:`sec_diffusion-samplers` häufen sich die Schritte von Bildsamplern bei schwachem Rauschen. Das Gitter für den Rauschpegel folgt dort einem Potenzgesetz.)
   :exlabel:`ex_mdl-fokker-planck-probability-flow-step-placement`

<!--en:fffb6fde-->
1. [extended] **Abweichende A-priori-Verteilung bei ODE und SDE.** ●● Sampler starten mit der Referenzverteilung $q_T = \mathcal{N}(0, 1)$ anstelle der exakten Verteilung $p_T$. In diesem Projekt vergleichen Sie, wie die zeitumgekehrte SDE und die rückwärts integrierte Wahrscheinlichkeitsfluss-ODE diese Abweichung bis $t = 0$ forttragen. Verwenden Sie die ungleichen Gewichte $w=(0.8, 0.2)$ des Rückwärtssamplers aus :numref:`sec_mdl-time-reversal`. Damit unterscheidet sich $p_T$ sichtbar von $q_T$, wenn $T$ klein ist. Übergeben Sie `**uneven` an `p_t`, `score_t`, `mixture_cdf` und `sample_p0` sowie `score=lambda x, t: score_t(x, t, **uneven)` an `pf_velocity`.
    1. Simulieren Sie Andersons zeitumgekehrte SDE wie im Text und die rückwärts integrierte Wahrscheinlichkeitsfluss-ODE mit Heun-Schritten. Starten Sie beide mit $q_T$ und wählen Sie $T \in \{0.25, 0.5, 1, 2, 3\}$ mit Schrittweite $\Delta t = 0.005$. Notieren Sie den Anteil der Samples mit $x < 0$. Der Sollwert ist $0.8$. Die SDE bei $T = 3$ entspricht dem Lauf im Text.
    1. Andersons Satz gilt für den gesamten Prozess. Beim Start in $X_T = y$ liefert die zeitumgekehrte SDE $X_0 \sim p(x_0 \mid x_T = y)$. Sagen Sie damit die Masse des linken Modus für die SDE als Erwartungswert über $y \sim q_T$ einer Zugehörigkeitswahrscheinlichkeit von $p_T$ voraus (:eqref:`eq_mdl-dyn-mixture-score`). Nutzen Sie für die ODE, dass ihre Abbildung von $x_T$ nach $x_0$ monoton wächst und $p_T$ in $p_0$ überführt. Sagen Sie die Masse des linken Modus aus der Verteilungsfunktion der Standardnormalverteilung und den Verteilungsfunktionen von $p_T$ und $p_0$ voraus. Werten Sie beide Vorhersagen durch Quadratur aus und vergleichen Sie sie mit den Messwerten.
    1. Sei $q_0$ die Verteilung der Ausgabe eines Samplers. Beweisen Sie mit der Datenverarbeitungsungleichung aus :numref:`sec_mdl-tv-pinsker`, dass für die ODE-Ausgabe $D_{\textrm{KL}}(q_0 \| p_0) = D_{\textrm{KL}}(q_T \| p_T)$ gilt, für die SDE-Ausgabe hingegen $D_{\textrm{KL}}(q_0 \| p_0) \le D_{\textrm{KL}}(q_T \| p_T)$.
    1. Berechnen Sie $D_{\textrm{KL}}(q_T \| p_T)$ sowie $D_{\textrm{KL}}(q_0 \| p_0)$ für die SDE durch Quadratur für jedes $T$. Die A-posteriori-Verteilung $p(x_0 \mid x_T = y)$ ist eine Mischung aus zwei Gauß-Komponenten, die sich aus Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-gaussian-reverse-step`a ergeben. Damit ist $q_0$ ein eindimensionales Integral über $y$. Um welchen Faktor verkleinert die SDE die Abweichung bei $T = 1$?
    1. Wiederholen Sie die Messung aus (a) bei $T = 1$ für die Sampler der Familie :eqref:`eq_mdl-dyn-reverse-family` mit $\lambda \in \{0, 0.5, 1, 2\}$. Welchen Sampler würden Sie verwenden, wenn der Zeithorizont $T$ des Vorwärtsprozesses kurz sein muss? Was begrenzt bei fester Schrittweite die Größe von $\lambda$?
   :exlabel:`ex_mdl-fokker-planck-probability-flow-prior-mismatch`

<!--en:2c61b585-->
[Diskussion](https://d2l.discourse.group/t/36839)

<!--en:1a871a8c-->
Die Fokker-Planck-Gleichung, zeitumgekehrte SDEs und Wahrscheinlichkeitsfluss<br>**Partielle Differentialgleichung der Randdichte · Score-Funktion · Diffusion als Transport · Wahrscheinlichkeitsfluss-ODE · Zeitumkehr nach Anderson**

<!--en:f650b20c-->
::: {.slide title="100 000 zufällige Pfade ergeben immer dasselbe Histogramm"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:bbe0f240-->
Bei der Ornstein-Uhlenbeck-SDE $dX = -X\,dt + \sqrt{2}\,dW$ ist jeder einzelne Teilchenpfad unvorhersehbar. Starten jedoch $100{,}000$ Teilchen bei $x_0 = 2$, ergibt sich in jedem Lauf dasselbe glockenförmige Histogramm $\mathcal{N}(2e^{-t}, 1 - e^{-2t})$:

<!--en:6d088baf-->
**Von zufälligen Pfaden zu deterministischen Dichten:** Welche partielle Differentialgleichung beschreibt die Entwicklung der Randdichte $p_t(\mathbf{x})$ einer SDE-Teilchenwolke? Wie lässt sich diese Entwicklung umkehren, um vom Rauschen zu den Daten zu gelangen?

<!--en:574bcf43-->
::: {.slide title="Vektoranalysis: Divergenz, Laplace-Operator und partielle Integration"}

<!--en:b6b1294c-->
[Mathematische Grundlagen]{.kicker}

<!--en:b9624fe0-->
Um vom Lemma von Itô für Teilchenmittelwerte zu einer partiellen Differentialgleichung für $p_t(\mathbf{x})$ zu gelangen, genügen drei Werkzeuge der mehrdimensionalen Analysis:

<!--en:66198edd-->
- **Divergenz ($\nabla \cdot \mathbf{u} = \sum_{i=1}^d \partial_i u_i$):** Nettoabfluss pro Volumeneinheit ($\nabla\cdot(x_1, x_2) = 2$, bei reiner Rotation $\nabla\cdot(-x_2, x_1) = 0$). Nach dem **Divergenzsatz** gilt $\int_\Omega \nabla\cdot\mathbf{u}\,d\mathbf{x} = \oint_{\partial\Omega} \mathbf{u}\cdot\mathbf{n}\,dS$.

<!--en:b59cbca8-->
- **Laplace-Operator ($\Delta \psi = \nabla \cdot \nabla \psi = \sum_{i=1}^d \partial_i^2 \psi$):** Er misst, wie weit $\psi(\mathbf{x})$ unter dem Mittelwert auf einer kleinen Kugeloberfläche liegt ($\bar{\psi}_r(\mathbf{x}) - \psi(\mathbf{x}) \approx \frac{r^2}{2d}\Delta\psi(\mathbf{x})$).

<!--en:56eb066a-->
- **Partielle Integration:** Bei im Unendlichen abklingenden Feldern wechselt beim Übertragen von $\nabla\cdot$ das Vorzeichen einmal, beim Übertragen von $\Delta$ zweimal:
  $$\int \psi\,(\nabla\cdot\mathbf{u})\,d\mathbf{x} = -\int \nabla\psi \cdot \mathbf{u}\,d\mathbf{x}, \quad \int \psi\,\Delta p\,d\mathbf{x} = \int p\,\Delta\psi\,d\mathbf{x}$$

<!--en:cf59c634-->
::: {.slide title="Herleitung der Fokker-Planck-Gleichung I: Erwartungswert einer Testfunktion"}

<!--en:fbb3dd92-->
[Herleitung der schwachen Form]{.kicker}

<!--en:42d389d2-->
Ein rauer SDE-Pfad $\mathbf{X}_t$ hat keine Zeitableitung. Der Erwartungswert $\mathbb{E}[\phi(\mathbf{X}_t)]$ jeder glatten Testfunktion („Buckelfunktion“) $\phi: \mathbb{R}^d \to \mathbb{R}$ mit kompaktem Träger ist jedoch nach der Zeit differenzierbar:

<!--en:c541b8d2-->
1. **Mehrdimensionales Lemma von Itô anwenden:** Entlang von $d\mathbf{X} = \mathbf{f}(\mathbf{X}, t)\,dt + g(t)\,d\mathbf{W}$ liefert die Korrektur aus $(dW)^2 = dt$ den Zusatzterm $\frac{1}{2}g(t)^2\Delta\phi$:
   $$d\phi(\mathbf{X}_t) = \left(\nabla\phi(\mathbf{X}_t) \cdot \mathbf{f}(\mathbf{X}_t, t) + \frac{1}{2}g(t)^2\,\Delta\phi(\mathbf{X}_t)\right)dt + g(t)\,\nabla\phi(\mathbf{X}_t) \cdot d\mathbf{W}_t$$

<!--en:e5d90590-->
2. **Erwartungswert über alle Pfade bilden:** Weil das Itô-Integral den Mittelwert null hat ($\mathbb{E}[\int g\nabla\phi \cdot d\mathbf{W}] = 0$), entfällt der stochastische Term vollständig:
   $$\frac{d}{dt}\mathbb{E}\!\left[\phi(\mathbf{X}_t)\right] = \mathbb{E}\!\left[\nabla\phi(\mathbf{X}_t) \cdot \mathbf{f}(\mathbf{X}_t, t) + \frac{1}{2}g(t)^2\,\Delta\phi(\mathbf{X}_t)\right]$$

<!--en:0e60aa0e-->
::: {.slide title="Herleitung der Fokker-Planck-Gleichung II: partielle Integration"}

<!--en:4fd84fb0-->
[Partielle Differentialgleichung der Dichte]{.kicker}

<!--en:ab315503-->
Partielle Integration überträgt die Ableitungen von der Testfunktion $\phi(\mathbf{x})$ auf $p_t(\mathbf{x})$. Daraus folgt die **Fokker-Planck-Gleichung**:

<!--en:6e5180dc-->
- **Zwei konkurrierende Operatoren:** Die **Advektion** erster Ordnung $-\nabla\cdot(\mathbf{f}p_t)$ transportiert Masse entlang der Drift $\mathbf{f}$. Die **Diffusion** zweiter Ordnung $\tfrac12 g^2\Delta p_t$ glättet Spitzen und füllt Täler auf.

<!--en:6d56de9e-->
::: {.slide title="Advektion, Diffusion und Wärmeleitungsgleichung"}

<!--en:9a713599-->
[Physikalische Deutung]{.kicker}

<!--en:66024b42-->
Die beiden Operatoren auf der rechten Seite von $\partial_t p_t = -\nabla\cdot(\mathbf{f}p_t) + \frac{1}{2}g(t)^2\Delta p_t$ verändern die Wahrscheinlichkeitswolke auf unterschiedliche, einander ergänzende Weise:

<!--en:118e9e58-->
- **Advektion ($-\nabla\cdot(\mathbf{f}p_t)$):** Deterministischer Transport erster Ordnung durch die Drift. Die Dichte steigt dort, wo das Vektorfeld $\mathbf{f}$ mehr Wahrscheinlichkeitsmasse in einen kleinen Quader hineinträgt als hinaus.

<!--en:fa237b86-->
- **Diffusion ($\frac{1}{2}g(t)^2\Delta p_t$):** Glättung zweiter Ordnung durch den Laplace-Operator. Sie entspricht der Itô-Korrektur $(dW)^2 = dt$ auf der Ebene der Dichte, senkt Spitzen ($\Delta p_t < 0$) und füllt Täler auf ($\Delta p_t > 0$).

<!--en:bee3c388-->
- **Wärmeleitungsgleichung ($\mathbf{f} = \mathbf{0}$):** Ohne Drift bleibt $\partial_t p_t = \frac{1}{2}g(t)^2\Delta p_t$. Jede isotrope Gauß-Verteilung $\mathcal{N}(\mathbf{m}, v_t\mathbf{I})$ mit linear wachsender Varianz $v_t = v_0 + \int_0^t g(\tau)^2\,d\tau$ löst sie (bei $x=0, t=1, g=1$ haben $\partial_t p$ und $\frac{1}{2}\partial_{xx}p$ beide den Wert $-\frac{1}{2\sqrt{2\pi}} \approx -0.1995$).

<!--en:74e1d709-->
::: {.slide title="Momentengleichungen für gaußsche Ornstein-Uhlenbeck-Prozesse"}

<!--en:22211144-->
[Exakt geschlossenes Momentensystem]{.kicker}

<!--en:e4dbf22a-->
Mit dem gaußschen Ansatz $p(x, t) = \mathcal{N}(x; m(t), v(t))$ reduziert sich die eindimensionale Fokker-Planck-Gleichung des OU-Prozesses $\partial_t p = \theta\partial_x(xp) + \tfrac12 g^2\partial_{xx}p$ auf zwei skalare ODEs:

<!--en:07c60fda-->
- **Stationäres Gleichgewicht ($\dot{m} = \dot{v} = 0$):** Bei $m_\infty = 0$ und $v_\infty = \frac{g^2}{2\theta}$ gleichen sich die nach innen gerichtete Kompression durch die Drift und die nach außen gerichtete Diffusion durch den Laplace-Operator überall aus.

<!--en:bbeed52a-->
::: {.slide title="Linearität der Fokker-Planck-Gleichung und gemischte Randverteilungen"}

<!--en:2ce89724-->
[Bimodaler Testfall]{.kicker}

<!--en:354e9947-->
Da die Fokker-Planck-Gleichung linear in $p_t$ ist, lässt sich jede konvexe Mischung gaußscher Lösungen $p_0(x) = \sum_k w_k \mathcal{N}(x; \mu_k, v_{0,k})$ zeitlich entwickeln, indem jede Komponente ihren eigenen Momentengleichungen folgt:

<!--en:2aae11fc-->
Verdoppelt sich die Zahl der räumlichen Gitterpunkte von $4{,}001$ auf $8{,}001$, sinkt das Residuum der zentralen Differenzen für die partielle Differentialgleichung um den Faktor $4.0\times$ zu jedem Zeitpunkt $t$. Das bestätigt die Konvergenz gegen null mit Ordnung zwei.

<!--en:7eb37a64-->
::: {.slide title="Die Score-Funktion zeigt in Richtung der steilsten Zunahme der Log-Dichte"}

<!--en:80f5b137-->
[Das Score-Feld]{.kicker}

<!--en:fb066839-->
Der **Score** einer Wahrscheinlichkeitsdichte $p_t(\mathbf{x})$ ist der Gradient ihrer Log-Dichte bezüglich des Zustands $\mathbf{x}$:

<!--en:33ece05f-->
- **Der gaußsche Score wirkt wie eine lineare Feder:** Für $p(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, v\mathbf{I})$ gilt
  $$\nabla_{\mathbf{x}}\log p(\mathbf{x}) = -\frac{\mathbf{x} - \boldsymbol{\mu}}{v}$$
  Die Feder zieht die Teilchen zum Mittelwert $\boldsymbol{\mu}$ und hat die Steifigkeit $1/v$.

<!--en:7a48d211-->
- **Kritische Punkte und Verteilungsränder:** Es gilt $\mathbf{s}_t(\mathbf{x}) = \mathbf{0}$ an Modi, lokalen Minima und Sattelpunkten von $p_t$. Am größten ist $\|\mathbf{s}_t(\mathbf{x})\|$ in Tälern geringer Dichte und weit außen an den Rändern, wo sich $\log p_t$ steil ändert.

<!--en:1a6a452e-->
::: {.slide title="Geschlossene Formel für den Score einer Gauß-Mischung"}

<!--en:8449628a-->
[Federn mit Zugehörigkeitswahrscheinlichkeiten als Gewichten]{.kicker}

<!--en:93d47b2a-->
Für eine Mischung $p(x) = \sum_k w_k N_k(x)$ mit Gauß-Komponenten $N_k(x) = \mathcal{N}(x; \mu_k, v_k)$ ergibt die Ableitung von $\log p(x)$ den Score als Mittelwert der einzelnen Federkräfte, gewichtet mit den A-posteriori-Wahrscheinlichkeiten:

<!--en:30f4e3e3-->
- **Symmetrische Formel für zwei Modi ($w_1 = w_2 = \frac{1}{2}$, $\mu_k = \pm\mu$, $v_k = v_0$):** Die Zugehörigkeitswahrscheinlichkeiten vereinfachen sich zu einer logistischen Sigmoidfunktion. Der Score ergibt sich damit aus dem hyperbolischen Tangens:
  $$s_0(x) = \frac{\mu\tanh(\mu x / v_0) - x}{v_0} = 8\tanh(8x) - 4x \qquad (\mu = 2, v_0 = 0.25)$$

<!--en:ab63569d-->
- **Kriterium für Bimodalität:** Am Ursprung $x = 0$ beträgt die Steigung $s_0'(0) = \frac{\mu^2 - v_0}{v_0^2} = 60 > 0$ (die Dichte hat dort ein Minimum zwischen zwei Modi bei $\approx \pm 2$, sofern $\mu^2 > v_0$ gilt, für $\mu_t^2 \le v_t$ verschmelzen die Modi zu einem).

<!--en:9bb70960-->
::: {.slide title="Der Score der Mischung und sein Verlauf im Tal"}

<!--en:1b718c47-->
[Numerische Überprüfung]{.kicker}

<!--en:fa3b70a1-->
Der Vergleich des allgemeinen Scores aus den Softmax-Zugehörigkeitswahrscheinlichkeiten `score_t(x, 0)` mit der analytischen `tanh`-Formel zeigt Übereinstimmung bis auf Maschinengenauigkeit. Im zentralen Tal ist die rücktreibende Kraft besonders stark:

<!--en:5a12088d-->
::: {.slide title="Erhaltung der Wahrscheinlichkeit und Kontinuitätsgleichung"}

<!--en:2697c2dd-->
[Deterministischer Transport]{.kicker}

<!--en:c8b69982-->
Bewegen sich Teilchen deterministisch entlang eines ODE-Geschwindigkeitsfelds $\dot{\mathbf{x}} = \mathbf{u}(\mathbf{x}, t)$, bleibt die Wahrscheinlichkeitsmasse erhalten. Der Fluss durch eine Fläche pro Flächen- und Zeiteinheit ergibt sich aus dem **Wahrscheinlichkeitsstrom** $\mathbf{j}(\mathbf{x}, t) = q_t(\mathbf{x})\mathbf{u}(\mathbf{x}, t)$:

<!--en:a1d92721-->
- **Integrale Massenbilanz:** Nach dem Divergenzsatz ist die Änderungsrate der Wahrscheinlichkeit in einem beliebigen Gebiet $\Omega$ gleich dem negativen Nettoabfluss durch den Rand:
  $$\frac{d}{dt}\int_\Omega q_t\,d\mathbf{x} = -\oint_{\partial\Omega} \mathbf{j}\cdot\mathbf{n}\,dS = -\int_\Omega \nabla\cdot(q_t\mathbf{u})\,d\mathbf{x}$$

<!--en:cb5f3398-->
- **Kontinuitätsgleichung:** Für die punktweise Erhaltung muss gelten:
  $$\partial_t q_t + \nabla \cdot (q_t\,\mathbf{u}) = 0$$

<!--en:f70beae0-->
- **Log-Dichte entlang von Charakteristiken:** Die Produktregel für $\nabla\cdot(q_t\mathbf{u}) = \mathbf{u}\cdot\nabla q_t + q_t\nabla\cdot\mathbf{u}$ ergibt die Liouville-Spurformel $\frac{d}{dt}\log q_t(\mathbf{x}(t)) = -\nabla\cdot\mathbf{u}(\mathbf{x}(t), t)$.

<!--en:0b4439a1-->
::: {.slide title="Die entscheidende Identität: Diffusion als Transport"}

<!--en:299904ec-->
[Verbindung zwischen SDEs und ODEs]{.kicker}

<!--en:d0798315-->
Mit $\nabla p_t = p_t\nabla\log p_t$ lässt sich den Laplace-Term zweiter Ordnung $\tfrac12 g^2\Delta p_t$ als Divergenz eines Transports erster Ordnung schreiben:

<!--en:5fbb5b45-->
- **Geschwindigkeit des Wahrscheinlichkeitsflusses $\mathbf{u}_t$:** Auf die Randdichte wirken zufällige brownsche Stöße wie eine deterministische Geschwindigkeit $-\tfrac12 g^2\nabla\log p_t$, die Wahrscheinlichkeit in Richtung abnehmender Dichte bewegt.

<!--en:90d206d8-->
::: {.slide title="Beispiel: die Wärmeleitungsgleichung als Fluss lösen"}

<!--en:a8607d2b-->
[Nach außen gerichteter Wahrscheinlichkeitsfluss]{.kicker}

<!--en:07448e14-->
Für eine reine brownsche Bewegung ($\mathbf{f} = \mathbf{0}, g = 1$) mit Anfangsverteilung $p_0 = \mathcal{N}(0, v_0)$ gilt $p_t = \mathcal{N}(0, v_t)$ mit $v_t = v_0 + t$ und Score $\nabla\log p_t(x) = -x/v_t$:

<!--en:2665687d-->
- **Nach außen gerichtete Flussgeschwindigkeit:**
  $$u_t(x) = -\frac{1}{2}g^2\nabla\log p_t(x) = \frac{x}{2v_t} = \frac{x}{2(v_0 + t)}$$

<!--en:00c4231c-->
- **Exakte ODE-Trajektorien:** Variablentrennung in $\dot{x} = \frac{x}{2(v_0 + t)}$ ergibt glatte Strahlen, die sich nie kreuzen:
  $$x(t) = x(0)\sqrt{\frac{v_0 + t}{v_0}}$$

<!--en:74eac005-->
- **Gleiches Histogramm, verschiedene Pfade:** Für $v_0 = 1$ bei $t = 3$ streckt die ODE jeden Teilchenabstand deterministisch um $\sqrt{4} = 2\times$. Das ergibt dieselbe Randverteilung $\mathcal{N}(0, 4)$ wie die rauen brownschen Pfade.

<!--en:6bdf1eea-->
::: {.slide title="Simulation des Wärmeflusses: brownsche Stöße und Heun-Verfahren"}

<!--en:1b718c47-2-->
[Numerische Überprüfung]{.kicker}

<!--en:4fcb6a25-->
Ausgangspunkt sind $20{,}000$ Ziehungen $z \sim \mathcal{N}(0, 1)$. Sowohl die Addition eines brownschen Inkrements $\mathcal{N}(0, 3)$ als auch die Integration von $\dot{x} = \frac{x}{2(1+t)}$ mit dem Heun-Verfahren ergeben eine $\mathcal{N}(0, 2^2)$-verteilte Wolke bei $T = 3$:

<!--en:4ba932b1-->
::: {.slide title="Die Wahrscheinlichkeitsfluss-ODE als deterministischer Zwilling einer SDE"}

<!--en:44f0b790-->
[Wahrscheinlichkeitsfluss-ODE]{.kicker}

<!--en:f9862ba6-->
Zu jeder SDE $d\mathbf{X} = \mathbf{f}(\mathbf{X}, t)\,dt + g(t)\,d\mathbf{W}$ definiert die ODE mit Geschwindigkeit $\mathbf{u}_t$ einen deterministischen Prozess. Er hat dieselbe Randdichte $p_t(\mathbf{x})$ zu jedem Zeitpunkt $t \in [0, T]$:

<!--en:0a70aa1a-->
- **Warum die Randdichten übereinstimmen:** Da $\mathbf{u}_t$ die Gleichung $\partial_t p_t + \nabla\cdot(p_t\mathbf{u}_t) = 0$ erfüllt, lösen die ODE-Dichte $q_t$ und die SDE-Dichte $p_t$ dieselbe Kontinuitätsgleichung mit Anfangsdichte $q_0 = p_0$. Entlang jeder Charakteristik ändern sich $\log q_t$ und $\log p_t$ daher mit derselben Rate $-\nabla\cdot\mathbf{u}_t$.

<!--en:0cd4e6b1-->
- **Exakte Berechnung der Likelihood:** Die Integration der Liouville-Spurformel von $0$ bis $T$ macht jedes scorebasierte Diffusionsmodell zu einem kontinuierlichen normalisierenden Fluss:
  $$\log p_0(\mathbf{x}(0)) = \log p_T(\mathbf{x}(T)) + \int_0^T \nabla \cdot \mathbf{u}_t(\mathbf{x}(t))\,dt$$

<!--en:8bfbe9b3-->
::: {.slide title="Randverteilungen von SDE und Wahrscheinlichkeitsfluss-ODE für die Mischung"}

<!--en:e8c39627-->
[Entwicklung einer bimodalen Verteilung]{.kicker}

<!--en:119bdd86-->
$20{,}000$ Teilchen starten aus der bimodalen Mischung $p_0 = \frac{1}{2}\mathcal{N}(-2, 0.25) + \frac{1}{2}\mathcal{N}(2, 0.25)$. Der Vergleich zeigt Euler-Maruyama-Schritte für die OU-SDE neben Heun-Schritten für die Wahrscheinlichkeitsfluss-ODE:

<!--en:acb3b65f-->
::: {.slide title="Randverteilungen von SDE und Wahrscheinlichkeitsfluss-ODE für die Mischung (Fortsetzung)"}

<!--en:e8c39627-2-->
[Entwicklung einer bimodalen Verteilung]{.kicker}

<!--en:9bac3446-->
Warum ist `KS(ODE)` für jedes $t$ gleich? In einer Dimension ist die Flussabbildung $\Phi_t$ der ODE streng monoton wachsend und überführt die exakte Verteilungsfunktion $F_0$ in $F_t$. Sie transportiert damit die empirischen Quantile der Anfangsstichprobe, ohne durch erneutes Ziehen zusätzliches Rauschen zu erzeugen.

<!--en:9efce19b-->
::: {.slide title="Zeitumkehr: die Randverteilungen in umgekehrter Reihenfolge durchlaufen"}

<!--en:d10dc090-->
[Datenerzeugung durch Zeitumkehr]{.kicker}

<!--en:438bd955-->
Die Vorwärts-SDE glättet die Daten $p_0$ zu Rauschen $p_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$ (bei $t^* \approx 0.78$ verschmelzen die beiden Modi). Wie lässt sich der Prozess von rechts nach links durchlaufen ($t: T \to 0$), um aus Rauschen Daten zu erzeugen?

<!--en:61e7dca1-->
- **Warum der Wärmefluss hier umkehrbar ist:** Die reine Wärmeleitungsgleichung $\partial_t p = \frac{1}{2}g^2\Delta p$ von einem unbekannten Profil aus rückwärts zu lösen ist ein schlecht gestelltes Problem. Hier stehen die Randdichten $p_t$ des Vorwärtsprozesses **bereits fest**. Ihr Score $\nabla\log p_t$ liefert genau die bergauf gerichtete Rückstellkraft, die die Glättung rückgängig macht.

<!--en:611653a0-->
::: {.slide title="Die Bayes-Regel für einen infinitesimalen Rückwärtsschritt"}

<!--en:307dbea5-->
[Herleitung der Rückwärtsdrift]{.kicker}

<!--en:c3c4c136-->
Nach der Bayes-Regel gilt $p(\mathbf{x}_t = \mathbf{x} \mid \mathbf{x}_{t+\tau} = \mathbf{y}) \propto p(\mathbf{y} \mid \mathbf{x})\,p_t(\mathbf{x})$. Die Entwicklung von $\log p_t(\mathbf{x}) \approx \log p_t(\mathbf{y}) + (\mathbf{x} - \mathbf{y})\cdot\nabla\log p_t(\mathbf{y})$ und quadratische Ergänzung im Exponenten der Gauß-Dichte verschieben den Rückwärtsmittelwert um $+g^2\tau\nabla\log p_t(\mathbf{y})$. Das ergibt **Andersons zeitumgekehrte SDE** ($t : T \to 0$):

<!--en:3b6c243a-->
::: {.slide title="Andersons zeitumgekehrte SDE und der Faktor zwei"}

<!--en:342b2927-->
[Diffusion in umgekehrter Zeit]{.kicker}

<!--en:b7318c67-->
Im kontinuierlichen Grenzfall $\tau \to 0$ ergibt sich **Andersons zeitumgekehrte SDE**. Die Integration läuft rückwärts von $t = T$ bis $t = 0$:

<!--en:375ab9e9-->
- **Vorwärts in der umgekehrten Zeit ($\bar{t} = T - t \in [0, T]$):** Ein Vorwärtsschritt in $\bar{t}$ ($d\bar{t} = -dt > 0$) kehrt das Vorzeichen der deterministischen Drift um und erhält die Rauschamplitude $g$:
  $$d\mathbf{Y}_{\bar{t}} = \Big[-\mathbf{f}(\mathbf{Y}_{\bar{t}}, T - \bar{t}) + g(T - \bar{t})^2\,\nabla\log p_{T-\bar{t}}(\mathbf{Y}_{\bar{t}})\Big]\,d\bar{t} + g(T - \bar{t})\,d\mathbf{W}_{\bar{t}}$$

<!--en:4f8a437d-->
- **Prüfung am stationären OU-Prozess ($p_t \equiv \mathcal{N}(0, 1)$, $f = -x$, $g = \sqrt{2}$, $s_t = -x$):**
  - **Wahrscheinlichkeitsfluss-ODE.** $u_t(x) = f - \frac{1}{2}g^2 s_t = -x - 1(-x) = 0$ (die Teilchen stehen still).
  - **Zeitumgekehrte SDE in $\bar{t}$.** Die Drift ist $-f + g^2 s_t = +x + 2(-x) = -x$ mit Rauschen $\sqrt{2}$ (der stationäre OU-Prozess ist seine eigene Zeitumkehr). Mit nur $\frac{1}{2}g^2$ wäre die Rückwärtsdrift $0$, und die Wolke würde wie bei einer reinen brownschen Bewegung unbegrenzt auseinanderlaufen.

<!--en:20c708bf-->
::: {.slide title="Die λ-Familie von Rückwärtssamplern mit gleichen Randverteilungen"}

<!--en:d97b7a5b-->
[Gemeinsame Darstellung von ODE- und SDE-Samplern]{.kicker}

<!--en:751d3f38-->
Warum hat die zeitumgekehrte SDE eine **doppelt** so große Score-Korrektur ($g^2\nabla\log p_t$) wie die Wahrscheinlichkeitsfluss-ODE ($\frac{1}{2}g^2\nabla\log p_t$)? Beide gehören zu einer einparametrigen Familie von Rückwärtsprozessen mit Rauschamplitude $\lambda g$ ($\lambda \ge 0$):

<!--en:ef5e0dcb-->
- **Drift der Rückwärtsprozesse ($t: T \to 0$):**
  $$\mathbf{b}_\lambda(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \frac{1 + \lambda^2}{2}g(t)^2\,\nabla\log p_t(\mathbf{x})$$

<!--en:486ec2b8-->
- **Warum zwei Hälften nötig sind:** In der umgekehrten Zeit $\bar{t} = T - t$ hebt die erste Korrektur $+\frac{1}{2}g^2\nabla\log p_t$ die Glättung der vorwärts gerichteten Fokker-Planck-Gleichung auf ($\lambda = 0$, die **Wahrscheinlichkeitsfluss-ODE**). Zusätzliches Rückwärtsrauschen $\lambda g\,d\mathbf{W}_{\bar{t}}$ erzeugt eine weitere Laplace-Glättung $+\frac{\lambda^2}{2}g^2\Delta p_t$. Um sie aufzuheben, braucht es eine zusätzliche Kraft $+\frac{\lambda^2}{2}g^2\nabla\log p_t$ in Richtung höherer Dichte ($\lambda = 1$, **Andersons zeitumgekehrte SDE**).

<!--en:ad0689b9-->
::: {.slide title="Asymmetrische Modusgewichte aus symmetrischem Rauschen"}

<!--en:4b09b2eb-->
[Sampling in umgekehrter Zeit]{.kicker}

<!--en:3d27183e-->
Es starten $20{,}000$ Teilchen mit symmetrischem A-priori-Rauschen $\mathcal{N}(0, 1)$ bei $T = 3$ und folgen Andersons zeitumgekehrter SDE bis $t = 0$ für die **ungleich gewichtete** Mischung $p_0 = 0.8\,\mathcal{N}(-2, 0.25) + 0.2\,\mathcal{N}(2, 0.25)$:

<!--en:537810b7-->
::: {.slide title="Asymmetrische Modusgewichte aus symmetrischem Rauschen (Fortsetzung)"}

<!--en:4b09b2eb-2-->
[Sampling in umgekehrter Zeit]{.kicker}

<!--en:300ee2e5-->
- **Wann verteilt der Score die Teilchen im Verhältnis 80% zu 20% auf die Modi?**
  - **Kaum bei $t = 0$.** In jedem Modus gilt $r_k(x) \in \{0, 1\}$. Daher ist $s_0(x) \approx -(x \pm 2)/0.25$ unempfindlich gegenüber den Mischungsgewichten $(0.8, 0.2)$.
  - **Auch kaum bei $t = 3$.** Beide Komponenten überlagern sich nahe $\mathcal{N}(0, 1)$, sodass ihre Federkräfte fast übereinstimmen.
  - **Bei mittlerem Rauschen ($t \approx 1$).** Die Grenze gleicher Zugehörigkeitswahrscheinlichkeiten $x_t^* = \frac{v_t}{2m_t}\ln\frac{0.8}{0.2}$ durchläuft $+0.85$. Während sich die Gipfel trennen, lenkt sie Teilchen bei $x \in (0, 0.85)$ in den stärker gewichteten linken Modus.

<!--en:a76fd3fa-->
::: {.slide title="Diskretisierungsverzerrung und Verteilung der Schritte nahe t = 0"}

<!--en:183c9730-->
[Wo die Schrittweite am meisten zählt]{.kicker}

<!--en:e061e2ab-->
Die exakte Varianzrekursion des rückwärts angewandten Euler-Maruyama-Verfahrens für einen einzelnen gaußschen Modus $\mathcal{N}(2, 0.25)$ ($\text{true width} = 0.5$) liefert die Diskretisierungsverzerrung ohne Monte-Carlo-Rauschen:

<!--en:c6a19b9b-->
- **97% der Verzerrung stammen aus $t \le 0.5$:** Eine Verfeinerung der Schritte ($10\times$) nur im letzten Sechstel des Rückwärtslaufs ($t \in [0, 0.5]$) senkt die Überschreitung der Sollbreite von $0.0047$ auf $0.0006$.

<!--en:c6ac5a24-->
- **Zwei Ursachen verstärken sich:** (1) Die Steigung $\kappa_t = \theta - g^2/v_t$ der Rückwärtsdrift hat bei $t = 0$ den größten Betrag ($1 - 2/0.25 = -7$ gegenüber $-1$ nahe $t = T$). (2) Frühe Fehler bei großem $t$ schrumpfen mit jedem weiteren Schritt um den Faktor $(1 + \kappa_t\Delta t)^2 < 1$. Fehler nahe $t = 0$ erreichen die Ausgabe ungeschwächt.

<!--en:18ffc04c-->
::: {.slide title="Verständnisfragen zu Fokker-Planck-Gleichung, Scores und Zeitumkehr"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:46ed138e-->
Fragen zur Dichteentwicklung und zu Rückwärtsprozessen:

<!--en:281303fa-->
1. **Score einer schmalen Gauß-Verteilung:** Wie groß ist für $p(x) = \mathcal{N}(x; 3, 0.1^2)$ der Score $s(x) = \nabla_x \log p(x)$ bei $x = 3.2$, und in welche Richtung zeigt er?

<!--en:15889d86-->
2. **Frei gewähltes Rückwärtsrauschen ($\lambda = \sqrt{3}$):** Ein Rückwärtssampler für $d\mathbf{X} = \mathbf{f}\,dt + g\,d\mathbf{W}$ fügt Rauschen der Amplitude $\sqrt{3}\,g$ hinzu. Welche Drift $\mathbf{b}(\mathbf{x}, t)$ braucht er, um die exakten Randdichten $p_t$ zu erhalten?

<!--en:80c65ec3-->
3. **Abweichende A-priori-Verteilung bei $t = T$:** Warum rekonstruiert Andersons zeitumgekehrte SDE ($\lambda = 1$) die Modusgewichte $0.8 / 0.2$ aus $\mathcal{N}(0, 1)$ bei $T = 3$ genauer als die deterministische Wahrscheinlichkeitsfluss-ODE ($\lambda = 0$)?

<!--en:557b98b6-->
**Lösungen:**

<!--en:75eb8786-->
1. Es gilt $s(x) = -\frac{x - 3}{0.01} = -100(x - 3)$ und damit $s(3.2) = -20$. Der Score zeigt mit großer Stärke nach links, zurück zum Modus $\mu = 3$.

<!--en:d7c5fb5b-->
2. Aus $\mathbf{b}_\lambda = \mathbf{f} - \frac{1+\lambda^2}{2}g^2\nabla\log p_t$ mit $\lambda^2 = 3$ folgt $\mathbf{b}_{\sqrt{3}} = \mathbf{f} - 2g^2\nabla\log p_t$.

<!--en:867a3d20-->
3. Bei endlichem $T = 3$ hat $p_T$ noch den Mittelwert $-1.2e^{-3} \approx -0.06 \ne 0$. Die Wahrscheinlichkeitsfluss-ODE ist eine deterministische Bijektion. Sie erhält daher die exakten Quantile der abweichenden Anfangsverteilung $\mathcal{N}(0, 1)$ bis $t = 0$. Das zusätzliche Rauschen der zeitumgekehrten SDE durchmischt die Teilchen und baut den anfänglichen Unterschied zur A-priori-Verteilung ab.

<!--en:96152d03-->
::: {.slide title="Zusammenfassung: Fokker-Planck-Gleichung, Wahrscheinlichkeitsfluss und Zeitumkehr"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:50c29906-->
| Gleichung oder Größe | Mathematische Form | Rolle in generativen Modellen |

<!--en:95222129-->
| **Fokker-Planck-Gleichung** | $\partial_t p_t = -\nabla\cdot(\mathbf{f}p_t) + \frac{1}{2}g(t)^2\Delta p_t$ | Deterministische Entwicklung der Randdichte $p_t(\mathbf{x})$ einer SDE |

<!--en:f6c3a5c8-->
| **Score-Funktion** | $\mathbf{s}_t(\mathbf{x}) = \nabla_{\mathbf{x}}\log p_t(\mathbf{x}) = \frac{\nabla p_t(\mathbf{x})}{p_t(\mathbf{x})}$ | Zeigt in Richtung steigender Log-Dichte. Einzige Unbekannte für die Zeitumkehr |

<!--en:4f46bbc5-->
| **Diffusion als Transport** | $\frac{1}{2}g^2\Delta p_t = \nabla\cdot\big(p_t \cdot \frac{1}{2}g^2\nabla\log p_t\big)$ | Übersetzt die Laplace-Glättung in eine Geschwindigkeit $-\frac{1}{2}g^2\mathbf{s}_t$ der Kontinuitätsgleichung |

<!--en:2ab80947-->
| **Wahrscheinlichkeitsfluss-ODE** | $\dot{\mathbf{x}} = \mathbf{u}_t(\mathbf{x}) = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2\nabla\log p_t(\mathbf{x})$ | Deterministischer Zwilling mit allen Randdichten $p_t$ der SDE. Liefert exakte Likelihoods |

<!--en:4fc36745-->
| **Andersons zeitumgekehrte SDE** | $d\mathbf{X} = \big[\mathbf{f}(\mathbf{X}, t) - g(t)^2\nabla\log p_t(\mathbf{X})\big]\,dt + g(t)\,d\bar{\mathbf{W}}$ | Erzeugt Daten $p_0$ aus Rauschen $p_T$. Der volle Faktor $g^2$ kompensiert Vorwärts- und Rückwärtsrauschen |

<!--en:79a6903a-->
| **Familie der $\lambda$-Sampler** | Drift $\mathbf{f} - \frac{1+\lambda^2}{2}g^2\nabla\log p_t$, Rauschen $\lambda g$ | Interpoliert zwischen deterministischer ODE ($\lambda=0$) und zeitumgekehrter SDE ($\lambda=1$) |
