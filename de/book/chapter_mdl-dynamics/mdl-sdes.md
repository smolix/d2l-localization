<!--en:ff7f2011-->
# Stochastische Differentialgleichungen

<!--en:7c0f24b1-->
In den vorigen Abschnitten legte der Anfangszustand jede Trajektorie fest. Ein Fluss vergisst seinen Ursprung nie: Zwei verschiedene Anfangszustände führen stets zu zwei verschiedenen Endzuständen. Die Diffusionsmodelle aus :numref:`chap_diffusion` beruhen auf einem Prozess, der seinen Ursprung vergisst. Ihr *Vorwärtsprozess* (engl. forward process) mischt dem Zustand nach und nach neues Rauschen bei, bis vom Anfangszustand nichts mehr übrig ist. Für die Erzeugung von Daten läuft der Prozess rückwärts und hat damit einen bekannten Ausgangspunkt. Auch ein solcher Prozess löst eine Differentialgleichung, doch die gewöhnliche Analysis reicht dafür nicht aus.

<!--en:b435d507-->
Beginnen wir mit einem Beispiel, das sich von Hand nachvollziehen lässt. Wir starten mit einer Zahl, $x_0 = 2$. In jedem Schritt behalten wir $99\%$ ihres Werts und addieren eine unabhängige Ziehung aus einer Normalverteilung mit Standardabweichung $\sqrt{0.02} \approx 0.14$:

<!--en:e2f2b0d5-->
Nach $400$ Schritten beträgt der Mittelwert $2 \cdot 0.99^{400} \approx 0.036$. Die Varianz liegt nahe bei $1$, denn dort gleichen die $2\%$, die ein Schritt an Varianz wegnimmt, die hinzukommenden $0.02$ aus. Jeder andere Anfangswert führt zur gleichen Verteilung, aus der der Endwert stammt, annähernd $\mathcal{N}(0, 1)$. Eine Eigenschaft der Rekursion überrascht. Beschreibt sie einen Prozess in kontinuierlicher Zeit mit Zeitschritt $h = 0.01$, so ändert der Abklingfaktor $0.99 = 1 - h$ den Wert von $x$ um einen Betrag der Ordnung $h$. Das Rauschen hat dagegen die Größenordnung $\sqrt{h}$ statt $h$. Diese Wahl ist zwingend. Rauschen der Größenordnung $h$ verschwände für $h \to 0$. Rauschen der Größenordnung $\sqrt{h}$ bleibt erhalten, und sein Quadrat hat die Ordnung $h$, dieselbe wie die Drift. Die gewöhnliche Analysis vernachlässigt Quadrate kleiner Änderungen. Schon deshalb brauchen wir einen neuen Kalkül.

<!--en:0538cde5-->
Im Grenzwert führt solches Rauschen zur brownschen Bewegung (engl. Brownian motion). Ihr Namensgeber, der Botaniker Robert Brown, beobachtete 1827, wie Teilchen aus Pollenkörnern im Wasser zitterten. Norbert Wiener fasste die Bewegung 1923 mathematisch. Kiyoshi Itô entwickelte in den 1940er-Jahren den zugehörigen Kalkül :cite:`Ito.1944`. Er beruht auf einer Regel: Die Quadrate kurzer brownscher Zuwächse summieren sich zur verstrichenen Zeit. Dadurch erhält die Kettenregel einen Term mit der zweiten Ableitung.

<!--en:61777fbc-->
Wir konstruieren die brownsche Bewegung aus Münzwürfen und leiten den Kalkül aus dieser Regel her. Dann lösen wir die Gleichung des Prozesses hinter der Rekursion, des Vorwärtsprozesses aus :numref:`sec_diffusion-gaussian-paths`. Bei gegebenem Anfangswert ist seine Verteilung zu jedem Zeitpunkt eine Normalverteilung. Ein Diffusionsmodell kann seine Daten also mit einer einzigen Ziehung beliebig stark verrauschen. Die Simulation und die Messung ihres Fehlers können bis zur zweiten Lektüre warten. Nur die Regel für die Quadrate ist schwer zu glauben, und ihr Beweis braucht sechs Zeilen.

<!--en:30ccf6fd-->
## Brownsche Bewegung

<!--en:f453f989-->
### Rauschen als Vorwärtsprozess

<!--en:ee5b333a-->
Klären wir zunächst, was ein Vorwärtsprozess leisten muss. Er beginnt mit Daten, die gemäß $p_0 = p_{\textrm{data}}$ verteilt sind. Unabhängig von der ursprünglichen Datenverteilung muss er zu einer bekannten Verteilung führen,

<!--en:153024a1-->
damit der Ausgangspunkt für die Erzeugung neuer Daten stets bekannt ist. Kein fester Fluss leistet das für zwei verschiedene Datensätze zugleich. Ein Fluss ist eine umkehrbare Abbildung $\Phi$ (:numref:`sec_mdl-odes-solvers`). Würde er zwei Verteilungen $p$ und $q$ auf dieselbe Verteilung abbilden, ergäbe die Anwendung von $\Phi^{-1}$, dass $p = q$.

<!--en:373a4ef7-->
Erproben wir drei Prozesse an zwei möglichst einfachen „Datensätzen“, je einer Punktmasse bei $x_0 = 2$ und bei $x_0 = -2$.

<!--en:173cba01-->
- **Drift allein**, $\dot{x} = -x$. Die Lösung $x(t) = x_0 e^{-t}$ führt von den beiden Anfangswerten zu $\pm 2e^{-3} = \pm 0.0996$ zur Zeit $t = 3$. Beide nähern sich dem Punkt $0$, also keiner Normalverteilung, und bleiben zu jedem endlichen Zeitpunkt verschieden.

<!--en:3321b299-->
- **Rauschen allein**, $dX = \sqrt{2}\, dW$. Das bedeutet $X_t = x_0 + \sqrt{2}\, W_t$ für ein Rauschen $W$, dessen Varianz linear mit der Zeit wächst. Im nächsten Unterabschnitt konstruieren wir es. Für $X_t$ lautet das *Verteilungsgesetz* $\mathcal{N}(x_0, 2t)$. Jede Punktwolke behält ihren Mittelpunkt bei $\pm 2$, während ihre Standardabweichung $\sqrt{2t}$ unbegrenzt wächst ($2.45$ bei $t = 3$). Nur relativ zu dieser wachsenden Breite verliert der Anfangswert an Bedeutung. Eine Grenzverteilung gibt es nicht.

<!--en:83bfd0b9-->
- **Drift und Rauschen zusammen**, $dX = -X\,dt + \sqrt{2}\, dW$. Für die beiden Anfangswerte ergeben sich zur Zeit $t = 3$ die Verteilungen $\mathcal{N}(\pm 0.0996,\, 0.9975)$, wie wir in :numref:`sec_mdl-ornstein-uhlenbeck` beweisen. Die beiden Verteilungen lassen sich kaum voneinander und von $\mathcal{N}(0, 1)$ unterscheiden.

<!--en:35ba17f3-->
![Drei Prozesse mit den Anfangswerten $x_0=\pm2$: Einer verfehlt das Ziel, einer ist brauchbar, und der dritte kommt hier zum Einsatz. Drift allein führt jeden Anfangswert zum Punkt $0$. Bei reinem Rauschen bleibt jeder Anfangswert der Mittelpunkt einer Wolke. Ihr Band von zwei Standardabweichungen $x_0\pm2\sqrt{2t}$ (gestrichelt) wächst unbegrenzt. Zusammen führen Drift und Rauschen von jedem Anfangswert zur selben Grenzverteilung $\mathcal N(0,1)$, rechts seitlich dargestellt. Die grünen Linien zeigen die Mittelwerte.](../img/mdl-dyn-sde-paths.svg)

<!--en:cf1ad5c4-->
:numref:`fig_mdl-dyn-sde-paths` zeigt alle drei Prozesse. Nur der erste scheidet ganz aus. Der zweite ist brauchbar und wird auch eingesetzt. Sobald seine Varianz $2t$ die Datenvarianz weit übersteigt, spielt der Anfangswert kaum noch eine Rolle. Solche Prozesse heißen *varianzexplodierend* (engl. variance-exploding), weil sie ohne Grenzverteilung eine sehr große Endvarianz erreichen müssen. Der dritte Prozess begleitet uns durch diesen Abschnitt. Die Rekursion vom Anfang simuliert ihn. Seine Drift zieht jeden Anfangswert gegen null, während das Rauschen die von der Drift abgebaute Varianz wieder zuführt. Bei der Varianz $1$ halten sich beide die Waage.

<!--en:45d6d527-->
Allgemein setzt sich ein solcher Vorwärtsprozess aus einem deterministischen Vektorfeld und einem Rauschterm zusammen. Für Daten in $\mathbb{R}^d$ schreiben wir

<!--en:0af15bdd-->
Das heißt: In einem kurzen Zeitintervall $dt$ verschiebt sich der Zustand um die Geschwindigkeit $\mathbf{f}$ mal $dt$ und erhält einen zufälligen Stoß. Dessen Stärke bestimmt die Rauschamplitude $g$. Die Drift steuert den Mittelwert, die Diffusion führt Varianz zu. Bislang ist der Stoß $d\mathbf{W}$ nur ein Symbol. Wir geben ihm zunächst in einer Koordinate eine Bedeutung und kehren erst am Ende der Behandlung des Itô-Kalküls zu $\mathbb{R}^d$ zurück.

<!--en:e0cc7d24-->
### Von Irrfahrten zum Wiener-Prozess

<!--en:5c4b8f66-->
Um dem Symbol eine Bedeutung zu geben, betrachten wir den Grenzwert eines denkbar einfachen stochastischen Prozesses: einer Irrfahrt (engl. random walk), deren Schritte Münzwürfe bestimmen. Der Grenzwert erklärt die Quadratwurzel in unserer Rekursion.

<!--en:2ea9e82e-->
Legen wir eine Schrittdauer $\Delta t$ fest. Bei einer Irrfahrt seien die Schritte unabhängig und jeweils gleich $\pm\sqrt{\Delta t}$. Das Vorzeichen bestimmt eine faire Münze. Nach $k$ Schritten, also zur Zeit $t = k\,\Delta t$, ist die Position eine Summe aus $k$ unabhängigen Schritten mit Mittelwert null und Varianz $\Delta t$. Die Varianzen unabhängiger Summanden addieren sich (:numref:`subsec_mdl-covariance`). Also gilt

<!--en:eaac1a0f-->
unabhängig von der Schrittweite. Bei $\Delta t = 0.01$ ergeben beispielsweise $100$ Schritte von $\pm 0.1$ die Varianz $100 \cdot 0.01 = 1$ zur Zeit $1$. Schritte von $\pm 0.01$ im selben Zeittakt ergäben die Varianz $0.01$, Schritte von $\pm 1$ dagegen $100$. Allgemein ergibt eine Schrittgröße von $c\,\Delta t^{a}$ die Varianz $c^2\, t\, \Delta t^{2a - 1}$ zur Zeit $t$. Sie geht gegen $0$ für $a > \tfrac12$ und gegen $\infty$ für $a < \tfrac12$. Nur mit der Quadratwurzel bleibt im Grenzwert eine endliche, von null verschiedene Varianz. Deshalb steht die Wurzel im Rauschterm jedes folgenden numerischen Verfahrens.

<!--en:e45351f9-->
Für $\Delta t \to 0$ wächst die Zahl der Schritte bis zur Zeit $t$ unbegrenzt. Nach dem zentralen Grenzwertsatz (:numref:`subsec_mdl-gaussian-clt`) wird die Position im Grenzwert normalverteilt: Eine Summe vieler unabhängiger Terme mit Mittelwert null und Gesamtvarianz $t$ hat näherungsweise die Verteilung $\mathcal{N}(0, t)$.

<!--en:59b3439d-->
Der Grenzprozess heißt **brownsche Bewegung** oder **Wiener-Prozess** :cite:`Wiener.1923`. Vier Eigenschaften definieren ihn:

<!--en:7ed5f095-->
2. **Unabhängige Zuwächse**: Für $s < t$ ist der Zuwachs $W_t - W_s$ vom gesamten Pfad bis zur Zeit $s$ unabhängig.

<!--en:8608fa13-->
3. **Normalverteilte Zuwächse**: $W_t - W_s \sim \mathcal{N}(0,\, t - s)$.

<!--en:3570656f-->
4. Stetige Pfade.

<!--en:fadae081-->
Der Satz von Wiener sichert die Existenz eines solchen Prozesses, der Satz von Donsker die Konvergenz der umskalierten Irrfahrten gegen diesen Prozess. Beide setzen wir voraus. Der Prozess breitet sich mit der Standardabweichung $\sqrt{t}$ immer weiter aus (:numref:`fig_mdl-dyn-brownian-paths`). Seine Zuwächse haben die Form

<!--en:c5870210-->
einer standardnormalverteilten Ziehung, skaliert mit $\sqrt{\Delta t}$.

<!--en:258ca581-->
![Eine durch Münzwürfe bestimmte Irrfahrt mit $1000$ Schritten von $\pm\sqrt{0.001}$ auf Gittern mit den Maschenweiten $\Delta t=0.1$, $0.01$ und $0.001$ (hellgrau der vollständige Pfad). Jedes Gitter zeigt wiederum eine Irrfahrt mit Schrittvarianz $\Delta t$. Ein feineres Gitter zeigt mehr Einzelheiten, der Verlauf im Großen bleibt erhalten. Zu jedem festen Zeitpunkt folgt die Streuung den Hüllkurven $\pm\sqrt t$ und $\pm2\sqrt t$.](../img/mdl-dyn-brownian-paths.svg)

<!--en:3aab2198-->
Die unabhängigen Zuwächse bestimmen auch den Zusammenhang zwischen den Werten von $W$ zu zwei Zeitpunkten. Für $s \le t$ schreiben wir $W_t = W_s + (W_t - W_s)$. Der Zuwachs ist unabhängig von $W_s$ und hat Mittelwert null. Daher gilt $\textrm{Cov}(W_s, W_t) = \mathbb{E}[W_s^2] = s = \min(s, t)$. Die Irrfahrt übernimmt ihre Position zur Zeit $s$ und addiert dazu unkorreliertes Rauschen. Diese Formel leistet mehr, als ihr einzeiliger Beweis vermuten lässt. Mittelwerte und Kovarianzen legen gemeinsam normalverteilte Werte eindeutig fest. Hat ein Prozess an jeder endlichen Menge von Zeitpunkten gemeinsam normalverteilte Werte mit Mittelwert null und Kovarianz $\min(s, t)$ sowie stetige Pfade, so ist er eine brownsche Bewegung.

<!--en:8a51c6d8-->
Dieselben Zuwächse machen den Pfad rau. Brownsche Pfade sind stetig, aber nirgends differenzierbar. Das besagt ein Satz von Paley, Wiener und Zygmund :cite:`Paley.Wiener.Zygmund.1933`. Eine anschauliche Begründung liefert die Skalierung der Zuwächse :eqref:`eq_mdl-sde-increment`:

<!--en:86b22cbf-->
Ein Differenzenquotient wächst dem Betrag nach also unbegrenzt, wenn der Schritt schrumpft. Das ist kein Beweis: Die Überlegung schließt eine Ableitung zu jedem einzeln festgelegten Zeitpunkt aus, „nirgends“ betrifft dagegen alle Zeitpunkte zugleich. Sie zeigt aber, wo die Schwierigkeit liegt.

<!--en:f7a154dd-->
Strecken wir die Zeit um den Faktor $c$ und teilen die Raumkoordinate durch $\sqrt{c}$, erhalten wir wieder eine brownsche Bewegung (Übung :exref:`ex_mdl-sdes-brownian-scaling-exit-times`). Beim Vergrößern zeigt ein Pfad daher auf jeder Skala dieselbe Rauheit. Eine differenzierbare Kurve würde sich dabei ihrer Tangente anschmiegen. Deshalb existiert $dW/dt$ nicht. Wir müssen $dW$ innerhalb von Integralen eine Bedeutung geben. Das leistet der Itô-Kalkül.

<!--en:b5bcac24-->
### Simulation brownscher Pfade

<!--en:34c5c28c-->
Die Darstellung der Zuwächse in :eqref:`eq_mdl-sde-increment` liefert auch ein Simulationsverfahren. Wir summieren Schritte der Form $\sqrt{\Delta t}\,\xi$ fortlaufend auf. Wir erzeugen $20{,}000$ Pfade auf $[0, 1]$ in $500$ Schritten und untersuchen eine Frage. Zu jedem einzelnen Zeitpunkt gilt $W_t \sim \mathcal{N}(0, t)$. Daher liegen $95.4\%$ der Pfade innerhalb von $\pm 2\sqrt{t}$, dem äußeren Band in :numref:`fig_mdl-dyn-brownian-paths`. Welcher Anteil bleibt zu allen Zeitpunkten im Band? Schätzen Sie den Anteil, bevor Sie die Zelle ausführen.

<!--en:0b61dce5-->
Zum einzelnen Zeitpunkt $t = 1$ liegt der Anteil wie erwartet bei etwa $95\%$. Aber nur etwa sieben von zehn Pfaden bleiben zu jedem von $50$ gleichmäßig verteilten Zeitpunkten im Band, und nur etwas mehr als die Hälfte zu allen $500$. Ein Band für einzelne Zeitpunkte erfasst also nicht mit derselben Wahrscheinlichkeit den ganzen Pfad. Bei jedem weiteren Blick kann der Pfad außerhalb liegen. Beobachten wir ihn in jedem Augenblick, bleibt fast sicher kein Pfad im Band. Nahe $t = 0$ überschreitet ein brownscher Pfad $\pm 2\sqrt{t}$ immer wieder (Gesetz des iterierten Logarithmus, hier ohne Beweis). Eine Aussage über die Verteilung zu einem Zeitpunkt ist viel schwächer als eine Aussage über ganze Pfade. Beide Sichtweisen begegnen uns beim schwachen und beim starken Fehler einer Simulation wieder (:numref:`sec_mdl-sde-strong-weak`).

<!--en:c0dbd53c-->
## Itô-Kalkül

<!--en:46e155d7-->
### Quadratische Variation

<!--en:b75cb6df-->
In der gewöhnlichen Differentialrechnung genügt die Taylor-Entwicklung erster Ordnung, weil das Quadrat einer kleinen Änderung vernachlässigbar ist. Für die brownsche Bewegung gilt das nicht. Wie groß der Beitrag der Quadrate ist, zeigt das Ergebnis dieses Unterabschnitts. Darauf baut der gesamte weitere Abschnitt auf.

<!--en:67452993-->
Betrachten wir eine Zerlegung $0 = t_0 < t_1 < \cdots < t_n = t$ mit Teilintervallen der Länge $\Delta t_i = t_{i+1} - t_i$ und Maschenweite $h_{\max} = \max_i \Delta t_i$. Wir summieren die quadrierten Zuwächse eines Pfades entlang der Zerlegung. Für einen differenzierbaren Pfad $x$ mit $|x'| \le M$ liefert der Mittelwertsatz $|\Delta x_i| \le M\, \Delta t_i$. Also gilt

<!--en:59fc8cc0-->
für verschwindende Maschenweite: Das Quadrat eines Zuwachses auf einem glatten Pfad hat die Ordnung $\Delta t^2$. Ein quadrierter brownscher Zuwachs hat dagegen die Ordnung $\Delta t$, denn $\mathbb{E}[(\Delta W_i)^2] = \Delta t_i$. Unabhängig von der Maschenweite summieren sich diese Erwartungswerte zu $t$ (:numref:`fig_mdl-dyn-qv-convergence`). Die Summe der Quadrate ist zufällig, ihre Schwankungen verschwinden jedoch im Grenzwert.

<!--en:7cf34bb8-->
**Proposition (quadratische Variation der brownschen Bewegung).** *Für $h_{\max} \to 0$ gilt*

<!--en:c0743fc3-->
**Beweis.** Jeder Zuwachs hat die Verteilung $\mathcal{N}(0, \Delta t_i)$. Daher gelten $\mathbb{E}[(\Delta W_i)^2] = \Delta t_i$ und $\mathbb{E}[Q_n] = \sum_i \Delta t_i = t$ für jede Zerlegung. Für die Streuung nutzen wir das vierte Moment der Normalverteilung, $3\,\Delta t_i^2$. Daraus folgt $\textrm{Var}\!\left((\Delta W_i)^2\right) = 3\Delta t_i^2 - \Delta t_i^2
= 2\Delta t_i^2$. Wegen der Unabhängigkeit der Zuwächse addieren sich die Varianzen ihrer Quadrate:

<!--en:1128b7ca-->
Damit folgt die Konvergenz im quadratischen Mittel. $\blacksquare$

<!--en:ba1cb219-->
Betrachten wir diese Aussage genauer: Jeder Zuwachs ist zufällig, doch der Grenzwert der Summe ihrer Quadrate ist deterministisch. Auf einem gleichmäßigen Gitter mit $n$ Teilintervallen von $[0, 1]$ ergibt der Beweis für die Summe eine Standardabweichung von $\sqrt{2/n}$. Sie beträgt $0.35$ für $n = 16$ und $0.003$ für $n = 2^{18}$.

<!--en:e3320174-->
![Fortlaufende Summen der quadrierten Zuwächse entlang eines brownschen Pfades für Gitter mit $n = 16$, $256$ und $4096$ Teilintervallen (dunklere Farbtöne für feinere Gitter). Die Summen nähern sich $y = t$. Beim glatten Pfad $\sin 2\pi t$ (grau) nehmen sie dagegen wie $2\pi^2/n$ ab.](../img/mdl-dyn-qv-convergence.svg)

<!--en:cf3592df-->
Meist schreibt man die Proposition als **Itô-Multiplikationstabelle**. Sie gibt die Rechenregeln für Produkte von Differentialen an:

<!--en:bd648622-->
Die erste Regel fasst :eqref:`eq_mdl-sde-qv` zusammen und bezieht sich auf Summen. Ein einzelner quadrierter Zuwachs beträgt $\Delta t\,\xi^2$ und liegt selten nahe bei $\Delta t$: Etwa jedes vierte Mal liegt er unter einem Zehntel von $\Delta t$ und etwa jedes sechste Mal über dem Doppelten von $\Delta t$. Erst die Summe über viele Schritte stabilisiert sich, wie der Beweis zeigt. (Bei der Irrfahrt mit Münzwürfen ist jedes Schrittquadrat gleich $\Delta t$, sodass keine Schwankungen auftreten.) Die beiden anderen Regeln halten fest, dass $dW\,dt$ von der Größenordnung $\Delta t^{3/2}$ sowie $(dt)^2$ schneller verschwinden als $\Delta t$. Im Grenzwert tragen sie nichts bei.

<!--en:30afdaba-->
Auf feinen Gittern sind die Schwankungen so klein, dass schon ein einziger Pfad den Grenzwert erkennen lässt. Wir tasten einen Pfad auf $2^{18}$ Teilintervallen von $[0, 1]$ ab und betrachten ihn auf gröberen Teilgittern. Seine Quadratsumme geben wir neben der vorhergesagten Standardabweichung aus. Zum Vergleich stellen wir dieselbe Summe für den glatten Pfad $x(t) = \sin 2\pi t$ dem Wert $2\pi^2/n$ gegenüber, dem Produkt aus der Maschenweite $1/n$ und $\int_0^1 x'(t)^2\,dt = 2\pi^2$.

<!--en:9c6cf51c-->
In der Spalte für den brownschen Pfad streuen die Werte um $1$ mit der im Beweis vorhergesagten Standardabweichung $\sqrt{2/n}$. Bei $16$ Teilintervallen liegt der Wert typischerweise zwischen etwa $0.5$ und $1.6$. Bei $n = 2^{18}$ weicht er nur um etwa $0.01$ von $1$ ab. Für den glatten Pfad beträgt der Wert noch $1.22$ bei $n = 16$, fällt aber wie $2\pi^2/n$ und stimmt ab $n = 256$ auf drei Stellen mit dieser Vorhersage überein.

<!--en:ec6da67a-->
### Das Itô-Integral

<!--en:9b573eeb-->
Wegen der Rauheit des Pfades müssen wir Integrale bezüglich $dW$ direkt definieren. Beginnen wir mit dem einfachsten, $\int_0^t W\,dW$. Die übliche Definition über Riemann-Summen liefert hier kein eindeutiges Ergebnis.

<!--en:b5a4fad2-->
Bei einem glatten Integrator konvergieren die Riemann-Summen gegen denselben Grenzwert, ob wir den Integranden am linken Rand, am rechten Rand oder in der Mitte jedes Teilintervalls auswerten. Hier ist das anders. Wählen wir $W$ selbst als Integranden, so übersteigt die Summe mit rechten Stützstellen diejenige mit linken Stützstellen um

<!--en:f95d2e80-->
nach der gerade bewiesenen Proposition. Die beiden Diskretisierungen „desselben“ Integrals unterscheiden sich im Grenzwert um einen endlichen, deterministischen Betrag. Betrachten wir einen Pfad mit $1000$ Schritten.

<!--en:432fa0c6-->
Die linke Summe ist eine Zufallszahl, die sich von Lauf zu Lauf ändert. Die rechte Summe übersteigt sie bis zur letzten ausgegebenen Stelle um die Summe der quadrierten Zuwächse, wie es die obige Identität verlangt. Der Lauf zeigt zudem, wie groß die Differenz ist. Meist weicht sie nur um etwa $0.1$ von $1$ ab (ihre Standardabweichung beträgt $\sqrt{2/1000} \approx 0.045$). Auch ein feineres Gitter würde die Differenz nicht beseitigen.

<!--en:6191f2fb-->
Beide Summen haben Grenzwerte, die wir bereits bestimmen können. Ihre Summe ist eine Teleskopsumme, denn $\left(W_{t_{i+1}} + W_{t_i}\right)\Delta W_i = W_{t_{i+1}}^2 - W_{t_i}^2$ summiert sich auf jeder Zerlegung zu $W_t^2$. Ihre Differenz geht gegen $t$. Also konvergiert die linke Summe gegen $\tfrac12\left(W_t^2 - t\right)$ und die rechte gegen $\tfrac12\left(W_t^2 + t\right)$, jeweils im quadratischen Mittel. Die gewöhnliche Analysis ergäbe $\tfrac12 W_t^2$, also das Mittel der beiden Grenzwerte. Wir müssen die Stützstelle wählen, und diese Wahl ändert das Ergebnis. Welche der beiden Summen soll das Integral definieren?

<!--en:8000af2e-->
Itô wählt den linken Rand. So kennt der Integrand den kommenden Zuwachs noch nicht. Ein faires Spiel erklärt, warum das sinnvoll ist. Stellen wir uns $\Delta W_i$ als Ergebnis der $i$-ten fairen Wette und $H_i$ als den zugehörigen Einsatz vor. Wer wettet, muss den Einsatz nach dem Kenntnisstand zur Zeit $t_i$ festlegen, bevor das Ergebnis feststeht. Der Gesamtgewinn beträgt $\sum_i H_i\,\Delta W_i$. Der Zuwachs $\Delta W_i$ ist unabhängig von allem, was zur Zeit $t_i$ bekannt ist, einschließlich $H_i$. Daher hat jede Wette den erwarteten Gewinn $\mathbb{E}[H_i\,\Delta W_i] = \mathbb{E}[H_i]\,\mathbb{E}[\Delta W_i] = 0$. Keine Wettstrategie erzielt einen positiven erwarteten Gewinn. Wer den Einsatz in Kenntnis des Ergebnisses wählen dürfte, könnte im Mittel Geld gewinnen. Die rechte Summe verwendet einen solchen Einsatz und hat tatsächlich den Mittelwert $t$ statt null.

<!--en:a3c0dd36-->
Die Definition hält diese Regel fest. Ein Integrand heißt *vorhersagbar* (engl. predictable), wenn sein Wert auf jedem Teilintervall feststeht, bevor der zugehörige Zuwachs gezogen wird. Ein **einfacher vorhersagbarer Prozess** $H$ ist auf jedem Teilintervall konstant: $H_s = H_i$ für $s \in (t_i, t_{i+1}]$. Dabei hängt $H_i$ nur vom Pfad bis zur Zeit $t_i$ ab, und es gilt $\mathbb{E}[H_i^2] < \infty$. Sein **Itô-Integral** ist die Summe der Gewinne,

<!--en:a7902a57-->
Nach dem Argument für das faire Spiel hat es den Mittelwert null, $\mathbb{E}\!\left[\int_0^t H\,dW\right] = 0$. Auch seine Varianz lässt sich berechnen.

<!--en:f76364ea-->
**Proposition (Itô-Isometrie).** *Für einen einfachen vorhersagbaren Prozess $H$ gilt*

<!--en:a4fff5f4-->
**Beweis.** Multiplizieren wir das Quadrat der Summe in :eqref:`eq_mdl-sde-ito-integral` aus. Besondere Aufmerksamkeit verlangen die gemischten Terme. Für $i < j$ enthält ein solcher Term den Faktor $\Delta W_j$. Er ist unabhängig von allem, was zur Zeit $t_j$ bekannt ist, einschließlich des Produkts $H_i\,\Delta W_i\,H_j$. Daher zerfällt der Erwartungswert in ein Produkt: $\mathbb{E}\!\left[H_i\,\Delta W_i\, H_j\, \Delta W_j\right] =
\mathbb{E}\!\left[H_i\,\Delta W_i\, H_j\right]\mathbb{E}[\Delta W_j] = 0$. Einen Diagonalterm zerlegen wir ebenso in $\mathbb{E}\!\left[H_i^2\right]\mathbb{E}\!\left[(\Delta W_i)^2\right] =
\mathbb{E}\!\left[H_i^2\right]\Delta t_i$. Die Summe der Diagonalterme ergibt $\sum_i \mathbb{E}[H_i^2]\,\Delta t_i = \int_0^t \mathbb{E}[H_s^2]\,ds$. $\blacksquare$

<!--en:7d4392d9-->
Mit der Isometrie lässt sich die Definition über Treppenprozesse hinaus erweitern, etwa auf den Integranden $H = W$ selbst. Die Isometrie besagt, dass $H \mapsto \int H\,dW$ die Norm $\|H\|^2 = \int_0^t \mathbb{E}[H_s^2]\,ds$ erhält. Konvergieren einfache vorhersagbare Prozesse in dieser Norm gegen $H$, so konvergieren ihre Integrale im quadratischen Mittel. Ihren Grenzwert wählen wir als Definition von $\int_0^t H\,dW$. Der Mittelwert null und die Isometrie bleiben beim Grenzübergang erhalten und gelten für jeden vorhersagbaren Prozess $H$ mit endlicher Norm. Für einen stetigen Integranden dieser Art konvergieren die Summen mit linken Stützstellen gegen sein Integral. Die Beweise finden sich in :citet:`Oksendal.2003`, Kapitel 3.

<!--en:9283919a-->
Unser Ausgangsintegral hat nun einen Wert, den Grenzwert seiner Summen mit linken Stützstellen:

<!--en:3cd55abe-->
Das ist das Ergebnis der gewöhnlichen Analysis vermindert um $t/2$. Der Term $-t/2$ sorgt für den Mittelwert null, den ein faires Spiel verlangt, denn $\mathbb{E}[W_t^2] = t$. Auch die Isometrie lässt sich daran prüfen. Sie drückt die Varianz eines Integrals bezüglich des Rauschens als gewöhnliches Integral über ein zweites Moment aus. Hier ergibt sich $\int_0^1 \mathbb{E}[W_s^2]\,ds = \int_0^1 s\,ds = \tfrac12$ bei $t = 1$. Die geschlossene Formel bestätigt dies: $W_1^2$ ist das Quadrat einer standardnormalverteilten Zufallsvariablen und hat die Varianz $2$. Also gilt $\textrm{Var}\big(\tfrac12(W_1^2 - 1)\big) = \tfrac14 \cdot 2 = \tfrac12$.

<!--en:a9432953-->
Andere Stützstellen definieren andere Integrale. Die Wahl des Mittelpunkts oder die Mittelung der linken und rechten Summen definiert das *Stratonovich-Integral* :cite:`Stratonovich.1966`. Es erfüllt die gewöhnliche Kettenregel, wie das obige Mittel $\tfrac12 W_t^2$ nahelegt. Der Mittelwert null geht jedoch verloren, da der Integrand den kommenden Zuwachs zur Hälfte berücksichtigt. Wir verwenden in diesem Buch Itôs Konvention aus zwei Gründen. Der Mittelwert null vereinfacht die Berechnung von Erwartungswerten. Außerdem ergibt der einfachste Simulationsschritt eine Itô-Summe, weil er die Koeffizienten am Schrittanfang auswertet (Übung :exref:`ex_mdl-sdes-ito-versus-stratonovich`).

<!--en:b9f07385-->
### Drift, Diffusion und Lösungen

<!--en:47ea1c17-->
Mit dem Integral erhält der allgemeine Ansatz :eqref:`eq_mdl-sde-template` für den Vorwärtsprozess eine Bedeutung. Eine **stochastische Differentialgleichung** (engl. stochastic differential equation, SDE) schreiben wir als

<!--en:5fbc76c3-->
und die Differentialform steht kurz für die Integralgleichung

<!--en:7c7c35dc-->
also für die Summe aus einem gewöhnlichen Integral und einem Itô-Integral. Halten wir die Koeffizienten während eines kurzen Schritts auf ihren Anfangswerten fest, so gelten $\mathbb{E}[dX] = f\,dt$ und $\textrm{Var}(dX) = g^2\,dt$. Die **Drift** $f$ ist also die bedingte mittlere Geschwindigkeit. Der Koeffizient der **Diffusion** $g$ bestimmt die Rate $g^2$, mit der Varianz hinzukommt. Für $g \equiv 0$ erhalten wir die gewöhnliche Differentialgleichung aus :numref:`sec_mdl-odes-solvers`. Für $f \equiv 0$ und konstantes $g$ erhalten wir den reinen Rauschprozess $X_t = X_0 + g\,W_t$ aus :numref:`fig_mdl-dyn-sde-paths`.

<!--en:c1903223-->
Was ist eine Lösung? Bei einer gewöhnlichen Differentialgleichung ist es eine Kurve. Hier ist es je eine Kurve für jeden brownschen Pfad. Seien $f$ und $g$ Lipschitz-stetig in $x$ und höchstens linear wachsend, $|f(x, t)| + |g(x, t)| \le C(1 + |x|)$. Sei ferner $X_0$ unabhängig von $W$ mit $\mathbb{E}[X_0^2] < \infty$. Dann legen der Anfangswert und der brownsche Pfad den Lösungspfad eindeutig fest (:citet:`Oksendal.2003`, Satz 5.2.1). Der Beweis ist eine stochastische Variante der Picard-Iteration aus :numref:`sec_mdl-ode-existence-uniqueness`. An die Stelle der Integralabschätzung tritt die Isometrie. Somit steckt der gesamte Zufall in $W$. Lösen wir die Gleichung vom selben Anfangswert aus wiederholt mit neuem Rauschen, entsteht eine Wolke von Pfaden wie in :numref:`fig_mdl-dyn-sde-paths`. Für ein generatives Modell zählt die Verteilung dieser Wolke zu jedem Zeitpunkt.

<!--en:2c0938be-->
Für die Simulation ist eine Unterscheidung wichtig. Rauschen heißt **additiv**, wenn seine Amplitude nur von der Zeit abhängt, $g = g(t)$. Das gilt für den Ansatz :eqref:`eq_mdl-sde-template` und für alle stochastischen Differentialgleichungen der Diffusionsmodelle in diesem Kapitel, auch für unser durchgehendes Beispiel. Rauschen heißt **multiplikativ**, wenn seine Amplitude vom Zustand abhängt, $g = g(X, t)$. Ein Beispiel ist ein Aktienkurs, dessen Schwankungen proportional zu seinem Wert sind. Der Unterschied wirkt geringfügig, entscheidet aber darüber, wie genau ein simulierter Pfad dem wahren folgt (:numref:`sec_mdl-sde-strong-weak`).

<!--en:11c1a042-->
### Itô-Lemma

<!--en:b9dfa6b4-->
Um mit einer stochastischen Differentialgleichung zu rechnen, brauchen wir die Kettenregel. Wenn $X$ eine solche Gleichung löst, welche Gleichung löst dann eine Funktion $\phi(X_t, t)$? Die gewöhnliche Kettenregel ergäbe $d(W^2) = 2W\,dW$. Wir wissen aber, dass das falsch ist, denn es gilt $\int_0^t 2W\,dW = W_t^2 - t$ nach :eqref:`eq_mdl-sde-wdw`. Es fehlt also etwas.

<!--en:ab8ee5bc-->
Ein einzelner Schritt einer Irrfahrt zeigt, woher der fehlende Term kommt und wie er für andere Funktionen aussieht. Betrachten wir $\phi(x) = x^2$ und einen Schritt $\pm\sqrt{\Delta t}$ von $x$ aus, mit Wahrscheinlichkeit $\tfrac12$ für jedes Vorzeichen. Der Mittelwert von $\phi$ nach dem Schritt beträgt

<!--en:3a685135-->
denn $\phi'' = 2$. Der Anstieg stammt vom Quadrat des Schritts. Beim Münzwurf beträgt dieses Quadrat jedes Mal $\Delta t$, doch die gewöhnliche Kettenregel vernachlässigt es.

<!--en:7e430f27-->
Mit $x = 0.7$ und $\Delta t = 0.25$ führt der Schritt zu $0.2$ oder $1.2$. Dort hat $\phi$ den Wert $0.04$ oder $1.44$. Der Mittelwert $0.74$ liegt über $\phi(0.7) = 0.49$, und zwar um $0.25 = \Delta t$ (:numref:`fig_mdl-dyn-ito-jensen`). Symmetrisches Rauschen erhöht also den Mittelwert einer konvexen Funktion. Weil ein quadrierter brownscher Zuwachs die Ordnung $dt$ hat, verschwindet der Zuwachs pro Zeiteinheit, $\tfrac12\phi''$, im Grenzwert nicht.

<!--en:f140fa45-->
![Die Itô-Korrektur als Differenz in der Jensen-Ungleichung. Ein Schritt von $\pm\sqrt{\Delta t}$ führt von $x$ aus für eine konvexe Funktion $\phi$ zu einem Mittelwert, der über $\phi(x)$ liegt, und zwar um $\tfrac12\phi''(x)\,\Delta t$. Hier gelten $\phi=x^2$, $x=0.7$ und $\Delta t=0.25$. Die Differenz beträgt $0.25$.](../img/mdl-dyn-ito-jensen.svg)

<!--en:0330b8ed-->
Dieselbe Rechnung liefert die allgemeine Formel. Sei $X$ eine Lösung von $dX = f\,dt + g\,dW$ und sei $\phi(x, t)$ glatt. Eine kleine Änderung von $\phi$ entwickeln wir bis zur zweiten Ordnung, da die erste nicht mehr genügt:

<!--en:02647c3f-->
Setzen wir $dX = f\,dt + g\,dW$ ein und multiplizieren wir $(dX)^2$ mithilfe der Tabelle :eqref:`eq_mdl-sde-ito-table` aus:

<!--en:e44c5aa2-->
Der Driftterm und die gemischten Terme verschwinden, während der Rauschterm in erster Ordnung erhalten bleibt. Fassen wir die Terme zusammen, erhalten wir die Kettenregel der stochastischen Analysis. Unsere Herleitung ist heuristisch. Wir formulieren das Ergebnis deshalb als Satz ohne Beweis.

<!--en:7e9636b5-->
**Satz (Itô-Lemma).** *Sei $X$ eine Lösung von $dX = f\,dt + g\,dW$ unter den Voraussetzungen des vorigen Unterabschnitts. Sei $\phi \in C^{1,2}$, also einmal stetig differenzierbar in der Zeit und zweimal im Zustand. Dann erfüllt $Y_t = \phi(X_t, t)$ die Gleichung*

<!--en:d515740c-->
Mit dieser Formel rechnen wir im Folgenden: Zur gewöhnlichen Kettenregel kommt die halbe zweite Ableitung hinzu, multipliziert mit der Rate, mit der Varianz hinzukommt. Der neue Term $\tfrac12 g^2 \phi_{xx}\,dt$ heißt **Itô-Korrektur**. Der Beweis muss die Restterme der Taylor-Entwicklung im quadratischen Mittel kontrollieren (:citet:`Oksendal.2003`, Kapitel 4). Er benötigt beide Ableitungen nach dem Zustand. Für eine Funktion mit einem Knick, etwa $\phi(x) = |x|$, gilt das Lemma in dieser Form nicht.

<!--en:9f2202fa-->
**Beispiel 1: $W^2$.** Wählen wir $X = W$ (also $f = 0$, $g = 1$) und $\phi(x) = x^2$. Das Lemma liefert $d(W^2) = 2W\,dW + dt$, einschließlich des Terms $dt$, den die gewöhnliche Kettenregel übergeht. In Integralform ergibt sich $W_t^2 = \int_0^t 2W_s\,dW_s + t$, also erneut :eqref:`eq_mdl-sde-wdw`. Eine Zeile mit dem Lemma ersetzt die Teleskopsummen.

<!--en:4a7dcef9-->
Wir können beobachten, wie sich die Korrektur aufsummiert. Auf jedem Gitter vereinfacht sich die Differenz zwischen $W_t^2$ und der linken Summe für $\int_0^t 2W\,dW$ durch Teleskopieren: $W_t^2 - \sum_{t_i < t} 2W_{t_i}\Delta W_i = \sum_{t_i < t} (\Delta W_i)^2$. Die Differenz ist also die fortlaufende quadratische Variation und sollte der Geraden $t$ folgen. Wir verwenden denselben Pfad wie für die linke und die rechte Summe.

<!--en:65e79715-->
Die mit „naive“ beschriftete Kurve zeigt das Itô-Integral $\int_0^t 2W\,dW$, das die gewöhnliche Kettenregel mit $W_t^2$ gleichsetzen würde. Die Kurve liegt unter $W_t^2$. Addieren wir $t$, liegt sie wieder auf der Vergleichskurve. Auf dem ganzen Pfad weicht die Differenz meist nur um etwa $0.1$ von $t$ ab. Nach dem Beweis von :eqref:`eq_mdl-sde-qv` beträgt ihre Standardabweichung zur Zeit $t$ auf diesem Gitter $\sqrt{2t\,\Delta t} \le 0.045$. Bei $t = 1$ ist die Differenz gleich der Summe der quadrierten Zuwächse, die die frühere Zelle ausgegeben hat.

<!--en:af10d78e-->
**Beispiel 2: geometrische brownsche Bewegung.** Die stochastische Differentialgleichung $dX = \mu X\,dt + \nu X\,dW$ mit $X_0 > 0$ modelliert eine Größe, die mit der mittleren Rate $\mu$ wächst und deren Rauschen proportional zu ihrem Wert ist. Dies ist Samuelsons Modell für einen Aktienkurs :cite:`Samuelson.1965`. Darauf beruht die Optionspreisformel von Black und Scholes :cite:`Black.Scholes.1973`. Das Rauschen ist multiplikativ. Wenden wir das Lemma auf $\phi(x) = \log x$ an, mit $\phi' = 1/x$, $\phi'' = -1/x^2$ und $g^2 = \nu^2 x^2$. Der Logarithmus ist nur für $x > 0$ glatt. Nehmen wir deshalb zunächst an, dass $X$ positiv bleibt:

<!--en:91a7fa37-->
Die so gewonnene Formel liefert positive Werte. Wenden wir das Lemma auf sie als Funktion von $W_t$ und $t$ an, bestätigt sich, dass sie die Gleichung löst. Damit entfällt die Annahme. Der Term $-\tfrac12\nu^2$ ist die Korrektur mit $\phi'' < 0$: Symmetrisches Rauschen senkt den Mittelwert einer konkaven Funktion. Das Rauschen drückt die typische Wachstumsrate $\mu - \tfrac12\nu^2$ also unter die mittlere Wachstumsrate $\mu$. Übung :exref:`ex_mdl-sdes-geometric-brownian-motion-correction` zeigt, wie weit Mittelwert und Median von $X_t$ auseinanderlaufen. An dieser exakten Lösung messen wir später Simulationsfehler.

<!--en:02564ffd-->
In mehreren Dimensionen bleibt das Prinzip gleich. Wir setzen das Ergebnis voraus :cite:`Oksendal.2003`. Betrachten wir $d\mathbf{X} = \mathbf{f}\,dt + \mathbf{G}\,d\mathbf{W}$ mit $\mathbf{X} \in \mathbb{R}^d$, einer Matrix $\mathbf{G}$ und einer brownschen Bewegung $\mathbf{W}$, deren Koordinaten unabhängige skalare brownsche Bewegungen sind. Die Multiplikationstabelle lautet dann $dW_i\, dW_j = \delta_{ij}\,dt$, und für eine zweimal stetig differenzierbare Funktion $\phi$ gilt

<!--en:c8cc6353-->
Die Korrektur gewichtet jede zweite Ableitung mit der Kovarianzrate $\mathbf{G}\mathbf{G}^\top$ des Rauschens. Die stochastischen Differentialgleichungen der Diffusionsmodelle in diesem Kapitel behandeln alle Koordinaten gleich, $\mathbf{G} = g(t)\,\mathbf{I}$. Hier vereinfacht sich die Korrektur zu $\tfrac12 g(t)^2 \sum_i \partial^2\phi/\partial x_i^2\; dt$, einem Vielfachen des Laplace-Operators angewandt auf $\phi$. Im nächsten Abschnitt brauchen wir das Lemma in dieser Form.

<!--en:11c515af-->
## Der Ornstein-Uhlenbeck-Prozess

<!--en:af719b6c-->
Unser durchgehendes Beispiel erhält nun einen Namen, und wir bestimmen seine Lösung. Der Ornstein-Uhlenbeck-Prozess (OU-Prozess) :cite:`Uhlenbeck.Ornstein.1930` erfüllt

<!--en:780b2058-->
also die Abklinggleichung $\dot{x} = -\theta x$ aus :numref:`sec_mdl-linear-odes-stability`, ergänzt um Rauschen mit konstanter Amplitude $g$. Die Autoren beschrieben damit die Geschwindigkeit eines Teilchens, das durch Molekülstöße beschleunigt und durch Reibung abgebremst wird. Die Drift zieht Auslenkungen mit der Rate $\theta$ gegen null zurück, während das Rauschen immer neue Auslenkungen erzeugt (:numref:`fig_mdl-dyn-ou-mean-reversion`). Die Rekursion vom Anfang beschreibt diesen Prozess mit $\theta = 1$ und $g = \sqrt{2}$ auf einem Gitter der Schrittweite $h = 0.01$. Dabei halten wir die Drift während jedes Schritts konstant: Ein Schritt multipliziert den Zustand mit $1 - \theta h = 0.99$ und addiert Rauschen mit Standardabweichung $g\sqrt{h} = \sqrt{0.02}$.

<!--en:d507611c-->
![Rückkehr zum Mittelwert. Die Drift $-\theta x$ (Pfeile) zieht jeden Pfad gegen $0$ zurück, und zwar mit der Rate $\theta=1$. Rauschen der Amplitude $g=\sqrt2$ hält ihn in Bewegung. Innerhalb weniger Vielfacher von $1/\theta$ verlieren die drei Anfangswerte ihren Einfluss. Danach schwanken die Pfade im Bereich von zwei Standardabweichungen der Grenzverteilung, $\pm2\sqrt{g^2/2\theta}=\pm2$ (als „stationary“ beschriftet). Rechts ist ihre Dichte dargestellt.](../img/mdl-dyn-ou-mean-reversion.svg)

<!--en:84154bbb-->
### Der Übergangskern

<!--en:78af57e3-->
Die Gleichung des OU-Prozesses gehört zu den wenigen stochastischen Differentialgleichungen, deren Lösungsverteilung zu jedem Zeitpunkt in geschlossener Form bekannt ist. Deshalb bauen Diffusionsmodelle auf diesem Prozess auf. Die Verteilung von $X_t$ bei gegebenem Anfangswert $X_0 = x_0$ heißt als Funktion des Anfangswerts *Übergangskern* (engl. transition kernel). Mit einem Faltungskern hat er nichts zu tun. Beim OU-Prozess ist sie eine Normalverteilung. Von allen Ergebnissen dieses Abschnitts verwenden wir dieses im weiteren Buch am häufigsten.

<!--en:64c36531-->
**Proposition (Übergangskern des OU-Prozesses).** *Für den OU-Prozess :eqref:`eq_mdl-sde-ou` gilt*

<!--en:93858840-->
**Beweis.** Mit dem integrierenden Faktor $e^{\theta t}$ haben wir in :numref:`subsec_mdl-ode-driven-linear` die Abklinggleichung mit einem Eingangssignal gelöst (Variation der Konstanten). Er funktioniert auch, wenn das Eingangssignal Rauschen ist. Wenden wir das Itô-Lemma :eqref:`eq_mdl-sde-ito-lemma` auf $\phi(x, t) = e^{\theta t} x$ an. Hier gelten $\phi_t = \theta e^{\theta t} x$, $\phi_x = e^{\theta t}$ und $\phi_{xx} = 0$, sodass die Korrektur entfällt:

<!--en:3eb10e7a-->
Die Drift hebt sich weg. Integration von $0$ bis $t$ und Multiplikation mit $e^{-\theta t}$ liefern

<!--en:b8331baa-->
Das verbleibende Integral hat einen deterministischen Integranden. Es ist also ein Grenzwert gewichteter Summen unabhängiger normalverteilter Zuwächse und selbst normalverteilt mit Mittelwert null. Dabei setzen wir voraus, dass ein Grenzwert normalverteilter Zufallsvariablen im quadratischen Mittel wieder normalverteilt ist. Die Itô-Isometrie :eqref:`eq_mdl-sde-ito-isometry` liefert die Varianz:

<!--en:e089bc74-->
Die Lösung :eqref:`eq_mdl-sde-ou-solution` beschreibt ein nachlassendes Gedächtnis. Der Einfluss des Anfangswerts klingt mit der Rate $\theta$ ab. Ein Rauschstoß zur Zeit $s$ wirkt mit dem Gewicht $e^{-\theta(t-s)}$ nach. Frisches Rauschen zählt, altes wird vergessen. In der Rekursion hat ein Stoß, der $k$ Schritte zurückliegt, noch das Gewicht $0.99^k \approx e^{-0.01k}$. In allen folgenden Abschnitten und in :numref:`sec_diffusion-gaussian-paths` im Haupttext schreiben wir den Kern mit zwei Koeffizienten, dem Faktor für den Mittelwert und der Standardabweichung:

<!--en:c67668ee-->
Die Daten gehen nur über den Mittelwert $\alpha_t x_0$ ein, das Rauschen nur über die Varianz $\sigma_t^2$. Für unser Beispiel ($\theta = 1$, $g = \sqrt{2}$, $\sigma_t^2 = 1 - e^{-2t}$) mit Anfangswert $x_0 = 2$ ist der Kern $\mathcal{N}(1.213,\, 0.795^2)$ bei $t = 0.5$, $\mathcal{N}(0.736,\, 0.930^2)$ bei $t = 1$ und $\mathcal{N}(0.037,\, 1.000^2)$ bei $t = 4$. Für jeden Zeitpunkt und jeden Anfangswert ist der Kern eine Normalverteilung. Damit kennen wir die Verteilung des verrauschten Zustands bei gegebenem Anfangswert in geschlossener Form, einschließlich des Gradienten ihrer Log-Dichte, $-(x-\alpha_t x_0)/\sigma_t^2$. Wie sich zeigen wird, hängt die Erzeugung neuer Daten von diesem Gradienten ab.

<!--en:923c6380-->
Die Rekursion diskretisiert den Prozess nur. Deshalb sollte ihre Punktwolke dem Kern nahekommen, muss ihm aber nicht exakt folgen. Wir wenden die Rekursion in einer Schleife auf $10{,}000$ Kopien von $x_0 = 2$ an und speichern die Wolke zu fünf Zeitpunkten. Ihren Mittelwert und ihre Standardabweichung geben wir neben $2\alpha_t$ und $\sigma_t$ aus, dazu den Anteil der Wolke innerhalb von $2\alpha_t \pm 2\sigma_t$.

<!--en:dc3d66c6-->
Mittelwerte und Standardabweichungen stimmen bis auf zwei oder drei Standardfehler mit dem Kern überein (bei einem Mittelwert etwa $0.01$ und bei einer Standardabweichung $0.007$ für $10{,}000$ Ziehungen). Wie bei jeder Normalverteilung liegen zu jedem Zeitpunkt etwa $95\%$ der Wolke im Band. Die Wolke beginnt im einzelnen Punkt $2$, wird rasch breiter und erreicht dann eine nahezu konstante Breite. Bei diesem Stichprobenumfang ist die Diskretisierung nicht zu erkennen.

<!--en:59f58997-->
### Die stationäre Verteilung

<!--en:ca240089-->
Ein Vorwärtsprozess muss seinen Anfangswert vergessen. Der Kern :eqref:`eq_mdl-sde-ou-kernel` zeigt, dass der OU-Prozess dies tut. Für $t \to \infty$ fällt der Mittelwert $x_0 e^{-\theta t}$ auf null, und die Varianz nähert sich $g^2/(2\theta)$. Also gilt

<!--en:e218d924-->
Das ist die für alle Daten gleiche Endverteilung, die wir in :numref:`sec_mdl-why-randomness` gefordert haben. Die Grenzverteilung ist mehr als ein Endpunkt. Ein Prozess, der mit dieser Verteilung startet, behält sie bei. Eine solche Verteilung heißt *stationär*. Sei dazu $X_0 \sim \mathcal{N}(0, g^2/2\theta)$ unabhängig von $W$. Nach :eqref:`eq_mdl-sde-ou-solution` ist $X_t$ dann die Summe von $X_0 e^{-\theta t}$ und einer unabhängigen normalverteilten Zufallsvariablen mit Varianz $\sigma_t^2$. Die Summe ist also normalverteilt mit Mittelwert null und Varianz

<!--en:abf917f1-->
Das Abklingen des Anfangswerts baut so viel Varianz ab, wie das Rauschen hinzufügt. Wie lange dauert das Vergessen? Seine Zeitskala ist $1/\theta$, die *Relaxationszeit*. Nach wenigen Vielfachen davon ist $e^{-\theta t}$ vernachlässigbar. Für unser Beispiel gelten $g^2/2\theta = 1$ und $1/\theta = 1$. Die stationäre Verteilung ist daher $\mathcal{N}(0, 1)$, und bei $t = 4$ ist der Mittelwert von $2$ auf $0.037$ gefallen.

<!--en:05714b79-->
Die Rekursion hat eine eigene, etwas andere stationäre Verteilung. Ein Schritt bildet die Varianz $v$ auf $(1 - \theta h)^2 v + g^2 h$ ab. Ausgehend von $v = 1$ ergibt ein Schritt in unserem Beispiel $0.9801 + 0.02 = 1.0001$, etwas mehr als $1$. Der Fixpunkt der Abbildung ist

<!--en:c7caefea-->
Die Varianz der Rekursion pendelt sich also ein halbes Prozent über der exakten Varianz $1$ ein. In einer Wolke aus $10{,}000$ Punkten fällt ein halbes Prozent nicht auf, auf dem Papier lässt es sich nachweisen. Dies ist das erste Anzeichen eines Diskretisierungsfehlers. In :numref:`sec_mdl-sde-strong-weak` sehen wir, dass er proportional zu $h$ schrumpft.

<!--en:5aebc59f-->
### Die varianzerhaltende Normierung

<!--en:fbcfb4d3-->
Die stationäre Varianz $g^2/2\theta$ ist ein freier Parameter. Diffusionsmodelle setzen sie auf $1$, durch die Wahl

<!--en:2b5e9e1b-->
Auch in unserem Beispiel treffen wir diese Wahl. Dann gilt $\sigma_t^2 = 1 - \alpha_t^2$, sodass ein Koeffizient den Kern beschreibt.

<!--en:d3cf7cdc-->
Bisher war der Anfangswert eine feste Zahl $x_0$. Nun sei er eine Ziehung $X_0$ aus den Daten. Die Lösung :eqref:`eq_mdl-sde-ou-solution` lautet dann in Verteilung $X_t = \alpha_t X_0 + \sqrt{1 - \alpha_t^2}\,\epsilon$, mit einer standardnormalverteilten Zufallsvariablen $\epsilon$, unabhängig von $X_0$. Die Verteilung von $X_t$ ergibt sich durch Mittelung des Kerns über die Daten. Sie heißt *Randverteilung* (engl. marginal) zur Zeit $t$. Die meisten Arbeiten verwenden ein anderes Symbol. Sie folgen dem Denoising Diffusion Probabilistic Model (DDPM) von :citet:`ho2020denoising`, dem zeitdiskreten Diffusionsmodell aus :numref:`sec_diffusion-gaussian-paths`. Es schreibt $\bar{\alpha}_t$ für unser $\alpha_t^2 = e^{-2\theta t}$. Der Haupttext folgt dieser Konvention und wir ebenfalls:

<!--en:298e5a08-->
Die Randverteilung entsteht aus einer gewichteten Summe von Daten und frischem Rauschen. Die Quadrate der Gewichte ergeben zusammen eins. Nehmen wir an, die Daten haben Mittelwert null und Varianz eins. Aus den Varianzen der beiden unabhängigen Summanden folgt

<!--en:5752e78a-->
Das bedeutet **varianzerhaltend** (engl. variance-preserving, VP). Für Daten mit Varianz eins beträgt die Randvarianz zu jedem Zeitpunkt $1$, auch vor dem Grenzübergang. Der Prozess überträgt Varianz von den Daten auf das Rauschen. Den Anteil $\bar{\alpha}_t$ tragen noch die Daten, den Anteil $1 - \bar{\alpha}_t$ hat schon das Rauschen übernommen. Die Summe bleibt gleich. Die Bezeichnung trifft nur für Daten mit Varianz eins zu. Für Daten mit Varianz $s^2$ wandert die Randvarianz von $s^2$ nach $1$ (Übung :exref:`ex_mdl-sdes-variance-preservation-exactly`). Die Skalierung der Daten spielt also eine Rolle. Auf $[-1, 1]$ skalierte Bilder, wie sie in :numref:`sec_diffusion-denoising` vorkommen, haben eine Varianz unter eins. Latente Diffusionsmodelle skalieren ihre latenten Variablen auf eine Varianz von ungefähr eins :cite:`Rombach.Blattmann.Lorenz.ea.2022`.

<!--en:ee8a9630-->
Die Varianz bleibt erhalten, die Form der Verteilung ändert sich. Betrachten wir Daten mit den Werten $\pm 1$, die jeweils mit Wahrscheinlichkeit $\tfrac12$ auftreten, wie beim Münzwurf. Ihr viertes Moment beträgt $1$ gegenüber $3$ bei der Standardnormalverteilung. Im Verlauf des Prozesses steigt es gemäß $\mathbb{E}[X_t^4] = 3 - 2\bar{\alpha}_t^2$. In unserem Beispiel ergibt das $2.264$ bei $t = 0.25$ und $2.963$ bei $t = 1$ (dieselbe Übung). Aus dem Münzwurf wird eine Glockenkurve, und die Varianz bleibt stets gleich.

<!--en:88cc19c8-->
Bei Diffusionsmodellen hängt die Rate meist von der Zeit ab. Die **VP-SDE** der scorebasierten Diffusion :cite:`song2021score`,

<!--en:f9c3b012-->
ist ein OU-Prozess mit zeitabhängiger Rate $\theta(t) = \beta(t)/2$ und Rauschamplitude $g(t)^2 = \beta(t)$. Unser Beispiel verwendet den konstanten Verlauf $\beta = 2$. Zu jedem Zeitpunkt gilt $g^2 = 2\theta$, sodass die Varianz während des gesamten Verlaufs bei $1$ bleibt. Die Randverteilung ergibt sich aus :eqref:`eq_mdl-sde-vp-marginal` mit $\bar{\alpha}_t = e^{-\int_0^t \beta(s)\,ds}$ (Übung :exref:`ex_mdl-sdes-exact-vp-discretization` zeigt, dass die Übergänge exakt normalverteilt sind). Das ist der varianzerhaltende Pfad aus :numref:`sec_diffusion-gaussian-paths`, also der Vorwärtsprozess, dessen Umkehrung ein Diffusionsmodell lernt.

<!--en:3bce61f4-->
## Simulation

<!--en:46dcb863-->
### Das Euler-Maruyama-Verfahren

<!--en:6a836bb8-->
Die meisten stochastischen Differentialgleichungen haben keine geschlossene Lösung. Deshalb müssen wir sie simulieren. Unsere Rekursion zeigt bereits, wie das geht. Um :eqref:`eq_mdl-sde-general` auf einem Gitter $t_n = n\,\Delta t$ zu simulieren, halten wir die Koeffizienten während eines Schritts konstant, wie beim expliziten Euler-Verfahren für gewöhnliche Differentialgleichungen (:numref:`sec_mdl-euler-runge-kutta`). Dazu kommt ein normalverteilter Stoß, dessen Größe die Darstellung der Zuwächse :eqref:`eq_mdl-sde-increment` vorgibt. Der Schritt nach **Euler-Maruyama** :cite:`Maruyama.1955` lautet

<!--en:e1ff17da-->
wobei $\hat{X}_n$ den Wert $X_{t_n}$ approximiert und $\hat X_0 = X_0$ gilt. Der Stoß skaliert mit $\sqrt{\Delta t}$, denn bei jedem anderen Exponenten verschwindet oder explodiert das Rauschen im Grenzwert (:numref:`sec_mdl-wiener-process`). Für $g = 0$ erhalten wir das explizite Euler-Verfahren.

<!--en:241ee22c-->
Für den OU-Prozess lautet der Schritt

<!--en:c065d864-->
wobei die zweite Form unserem durchgehenden Beispiel mit $\Delta t = h = 0.01$ entspricht, also der bisher verwendeten Rekursion.

<!--en:b8a9f24c-->
Für diesen Prozess können wir genauer rechnen. Mit $g^2 = 2\theta$ lautet der Kern :eqref:`eq_mdl-sde-ou-kernel` für ein einzelnes Intervall $\mathcal{N}(e^{-\theta\Delta t}x, 1 - e^{-2\theta\Delta t})$. Eine Ziehung daraus ist der Vorwärtsschritt des DDPM (:numref:`sec_diffusion-gaussian-paths`), $x_k = \sqrt{1 - \beta_k}\, x_{k-1} + \sqrt{\beta_k}\,\epsilon_k$ mit $\beta_k = 1 - e^{-2\theta\Delta t}$. Der DDPM-Schritt ist also für jede Schrittweite exakt, Euler-Maruyama ist seine Näherung erster Ordnung. Für $\theta = 1$ und $\Delta t = 0.01$ multipliziert der exakte Schritt mit $e^{-0.01} = 0.99005$ und addiert Rauschen der Varianz $\beta_k = 1 - e^{-0.02} = 0.0198$. Bei der Näherung sind es $0.99$ und $0.02$. Die beiden Beta-Größen haben verschiedene Bedeutungen. Das $\beta_k$ des Schritts ist der Varianzanteil, den ein Schritt durch Rauschen ersetzt. Er beträgt ungefähr $\beta(t)\,\Delta t$ für die Rate $\beta(t)$ aus :eqref:`eq_mdl-sde-vp-sde`, hier also $0.0198$ gegenüber $2 \cdot 0.01$. Exakte Schritte setzen hier eine lineare Gleichung voraus (Übung :exref:`ex_mdl-sdes-exact-vp-discretization`). Für eine allgemeine Drift kennen wir die Übergangsverteilung nicht in geschlossener Form. Dann bleibt uns das Näherungsverfahren.

<!--en:c19c3a1c-->
Die folgende Hilfsfunktion implementiert :eqref:`eq_mdl-sde-em` für einen ganzen Batch von Pfaden. Sie erhält die Zuwächse $\Delta W$ als Argument. So können wir zwei Simulationen mit demselben Rauschen antreiben. Das brauchen wir gleich: Sobald Rauschen hinzukommt, müssen wir festlegen, was „Fehler“ bedeutet. Dafür gibt es zwei Definitionen.

<!--en:bb80f981-->
### Starke und schwache Konvergenz

<!--en:9cd8bc55-->
Ein simulierter Pfad ist zufällig, der wahre Pfad ebenso. Die beiden Fehlerdefinitionen unterscheiden sich darin, was sie vergleichen: Pfade oder Verteilungen. Diesen Unterschied haben wir im Experiment mit dem Band gesehen.

<!--en:64820ec8-->
Sei $N = T/\Delta t$ die Anzahl der Schritte, sodass $\hat{X}_N$ den Wert $X_T$ approximiert. Verwenden wir dieselben brownschen Zuwächse für das Verfahren und die exakte Lösung. Der **starke Fehler** (engl. strong error) $\mathbb{E}\,|\hat{X}_N - X_T|$ misst die Abweichung auf einzelnen Pfaden: Folgt die simulierte Trajektorie der wahren Trajektorie für dieses Rauschen? Der **schwache Fehler** (engl. weak error) $|\mathbb{E}\,\varphi(\hat{X}_N) - \mathbb{E}\,\varphi(X_T)|$ misst für glatte Testfunktionen $\varphi$ nur, wie gut die Verteilung zur Zeit $T$ stimmt. Ein generatives Modell benötigt schwache Genauigkeit, weil es nur die Samples zum Endzeitpunkt nutzt. Starke Genauigkeit ist wichtig, wenn wir Pfade auf verschiedenen Auflösungsstufen koppeln, etwa beim Multilevel-Monte-Carlo-Verfahren. Dort stammt der Großteil der Schätzung von kostengünstigen groben Pfaden. Einige teure feine Pfade liefern die Korrektur (Übung :exref:`ex_mdl-sdes-multilevel-monte-carlo`). :numref:`fig_mdl-dyn-strong-weak` stellt die beiden Begriffe gegenüber.

<!--en:5afb5655-->
![Starke und schwache Konvergenz von Euler-Maruyama für unser Beispiel mit Anfangswert $X_0=2$ auf $[0,1]$. Links ein grober Pfad ($16$ Schritte, orange) und eine feine Referenz ($4096$ Schritte, blau) mit denselben brownschen Zuwächsen. Der Streifen darunter vergrößert ihre Differenz an den groben Gitterzeitpunkten, die der starke Fehler erfasst. Rechts stimmen die Histogramme der Endwerte von $20{,}000$ Pfaden bei beiden Schrittweiten fast miteinander und mit der exakten Verteilung (schwarz) überein, obwohl bei $16$ Schritten die Varianz um etwa $4\%$ zu groß ist.](../img/mdl-dyn-strong-weak.svg)

<!--en:af6c929a-->
Ein Verfahren hat starke Ordnung $p$, wenn sein starker Fehler höchstens $C\,\Delta t^{p}$ beträgt. Analog definieren wir die schwache Ordnung $p$. Die Konvergenzordnungen von Euler-Maruyama sind klassische Ergebnisse.

<!--en:4e84df19-->
**Satz (Konvergenzordnungen von Euler-Maruyama, ohne Beweis).** *Seien $f$, $g$ und $\varphi$ glatt. Für das Wachstum und die Momente von $X_0$ gelten die in :citet:`Kloeden.Platen.1992` präzisierten Voraussetzungen.*

<!--en:29dc6d4b-->
1. *Euler-Maruyama hat die starke Ordnung $\tfrac12$: $\mathbb{E}\,|\hat{X}_N - X_T| \le C\,\Delta t^{1/2}$.*

<!--en:96f1c7f2-->
2. *Bei additivem Rauschen, $g = g(t)$, beträgt die starke Ordnung $1$: $\mathbb{E}\,|\hat{X}_N - X_T| \le C\,\Delta t$.*

<!--en:d6c24f09-->
3. *Die schwache Ordnung beträgt bei additivem wie bei multiplikativem Rauschen $1$: $|\mathbb{E}\,\varphi(\hat{X}_N) - \mathbb{E}\,\varphi(X_T)| \le C\,\Delta t$.*

<!--en:ad330462-->
Eine Abschätzung der Einzelschrittfehler erklärt die Exponenten. Für einen Beweis reicht sie nicht: Sie addiert die Fehler der Schritte, als bliebe jeder Fehler dort, wo er entstanden ist. Ein Beweis muss auch erfassen, wie die spätere Dynamik einen frühen Fehler fortpflanzt.

<!--en:a5f62fe5-->
Euler-Maruyama hält $g$ während des Schritts konstant. Tatsächlich ändert sich $g$ aber mit dem Pfad. Zu einer Zeit $s$ innerhalb des Schritts hat sich der Pfad um etwa $g\,(W_s - W_{t_n})$ verschoben. Der Rauschkoeffizient beträgt dort also ungefähr $g + g\,\partial_x g\,(W_s - W_{t_n})$. Der größte vernachlässigte Term ist das Integral des zweiten Anteils bezüglich $dW$. Es ergibt sich aus :eqref:`eq_mdl-sde-wdw` für die brownsche Bewegung $W_s - W_{t_n}$, die bei $t_n$ neu beginnt:

<!--en:d3e1ae72-->
Das Milstein-Verfahren nimmt diesen Term wieder auf :cite:`Milstein.1975`. Sein Zufallsfaktor zeigt die Regel für Quadrate auf der Ebene eines einzelnen Schritts. Die Beziehung $(\Delta W)^2 = \Delta t$ gilt in Summen. In jedem Schritt hat die Abweichung den Mittelwert null und die Größenordnung $\Delta t$. Die $N = T/\Delta t$ Schritte liefern unkorrelierte Terme dieser Art. Sie summieren sich wie bei einer Irrfahrt zu einem Fehler der Größenordnung $\sqrt{N}\,\Delta t = \sqrt{T\,\Delta t}$. Das ergibt die Ordnung $\tfrac12$.

<!--en:f2f660d0-->
Bei additivem Rauschen verschwindet dieser Term wegen $\partial_x g = 0$. Die nächsten vernachlässigten stochastischen Terme haben pro Schritt die Größe $\Delta t^{3/2}$. Zusammen ergeben sie $\sqrt{N}\,\Delta t^{3/2} = O(\Delta t)$, ebenso wie die deterministischen Reste, $N\,\Delta t^2 = O(\Delta t)$. Die Ordnung beträgt also $1$. Für den schwachen Fehler zählen nur Mittelwerte. Jeder vernachlässigte stochastische Term hat den Mittelwert null. Die Verzerrung eines Schritts hat die Ordnung $\Delta t^2$, und $N$ Schritte ergeben $O(\Delta t)$, unabhängig von der Art des Rauschens.

<!--en:9d6d27ac-->
Wir messen beide starken Konvergenzordnungen mit demselben Rauschen. Die geometrische brownsche Bewegung ($\mu = \nu = 1$, $X_0 = 1.5$) vergleichen wir mit ihrer exakten Lösung :eqref:`eq_mdl-sde-gbm`. Für unser Beispiel mit Anfangswert $X_0 = 2$ verwenden wir als Referenz ein Gitter mit $4096$ Schritten, denn auch das Rauschintegral in der exakten Lösung :eqref:`eq_mdl-sde-ou-solution` könnten wir nur auf einem Gitter auswerten. Die groben Zuwächse sind Summen aufeinanderfolgender feiner Zuwächse. So liegt jeder Auflösung derselbe brownsche Pfad zugrunde.

<!--en:49423cff-->
Die gemessenen Steigungen liegen nahe bei $1$ für den OU-Prozess mit additivem Rauschen und nahe bei $\tfrac12$ für die geometrische brownsche Bewegung mit multiplikativem Rauschen. Dies sind die beiden starken Ordnungen des Satzes. Die Kurven liegen unterschiedlich hoch, weil die Prozesse unterschiedliche Skalen haben. Vergleichbar ist, wie schnell sie fallen: Jede Halbierung der Schrittweite halbiert den OU-Fehler und verkleinert den Fehler der geometrischen brownschen Bewegung etwa um den Faktor $\sqrt{2}$.

<!--en:15edc738-->
Für die schwache Ordnung brauchen wir kein Sampling. Euler-Maruyama liefert für den OU-Prozess eine lineare gaußsche Rekursion: Ist $\hat{X}_n$ normalverteilt, so auch $\hat{X}_{n+1}$. Damit ist die simulierte Randverteilung in jedem Schritt eine Normalverteilung mit Mittelwert und Varianz

<!--en:1c30aaeb-->
Der Mittelwert erhält in jedem Schritt den Faktor $1 - \theta\Delta t$, bei der exakten Lösung dagegen $e^{-\theta\Delta t}$ ($0.900$ gegenüber $0.905$ bei $\Delta t = 0.1$). Die Varianzrekursion summiert eine geometrische Reihe, $v_n = g^2\Delta t\,(1 - a^{2n})/(1 - a^2)$ mit $a = 1 - \theta\Delta t$ und $v_0 = 0$. Ihr Grenzwert ist der bereits bestimmte Fixpunkt $v_\infty$. Wir werten diese geschlossenen Ausdrücke bei $T = 1$ mit Anfangswert $X_0 = 2$ für sechs verschiedene Schrittzahlen aus. So erhalten wir die Fehler der simulierten Verteilung, ohne dass ein Stichprobenfehler ihre Steigungen verdeckt.

<!--en:bb422d4f-->
Bei $10$ Schritten hat die simulierte Verteilung den Mittelwert $0.697$ und die Varianz $0.925$. Die exakten Werte sind $0.736$ und $0.865$ (der bereits bekannte Kern $\mathcal{N}(0.736,\, 0.930^2)$). Bei $320$ Schritten erhalten wir $0.735$ und $0.866$. Beide angepassten Steigungen betragen $1.01$: Eine Halbierung der Schrittweite halbiert den Fehler der simulierten Verteilung. Das entspricht der schwachen Ordnung $1$.

<!--en:cf934efd-->
Dieselbe Rechnung ergibt auch für die geometrische brownsche Bewegung die schwache Ordnung $1$. Multiplikatives Rauschen kostet bei den Pfaden also eine halbe Ordnung, bei der Verteilung nichts. Ein scheinbar genaueres Verfahren kann sogar gegen die Lösung der falschen Gleichung konvergieren. Das Heun-Verfahren mittelt den Rauschkoeffizienten über den Schritt und konvergiert zur Stratonovich-Lösung statt zur Itô-Lösung (Übung :exref:`ex_mdl-sdes-ito-versus-stratonovich`).

<!--en:78628b03-->
Die Simulation liefert Samples des Vorwärtsprozesses, der Kern seine Verteilung bei einem einzelnen Anfangswert. Für die Erzeugung neuer Daten brauchen wir mehr. Die Daten bestehen nicht aus einem einzelnen Punkt. Die Randverteilung $p_t$ der verrauschten Daten ergibt sich durch Mittelung des Kerns über alle Anfangswerte. Für reale Daten, die wir nur durch Samples kennen, liegt sie nicht in geschlossener Form vor. Um den Prozess rückwärts laufen zu lassen, brauchen wir jedoch den Gradienten von $\log p_t$. Für einen Anfangswert konnten wir ihn direkt angeben. Bisher haben wir nicht einmal eine Gleichung für $p_t$.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3b0da499-->
Ein Vorwärtsprozess muss seinen Anfangszustand vergessen. Da kein deterministischer Fluss das leisten kann, treiben wir ihn mit Rauschen an. Rauschen, das im Grenzwert kleiner Schritte erhalten bleibt, hat die Größenordnung $\sqrt{\Delta t}$. Die Quadrate seiner Zuwächse summieren sich zur verstrichenen Zeit, $(dW)^2 = dt$. Ein Integral bezüglich $dW$ hängt daher davon ab, wo wir seinen Integranden auswerten. Mit Itôs Wahl des linken Randes erhalten wir den Mittelwert null und die Isometrie $\mathbb{E}[(\int H\,dW)^2] = \int \mathbb{E}[H^2]\,ds$. Aus demselben Grund erhält die Kettenregel die Korrektur $\tfrac12 g^2 \phi_{xx}\,dt$. Damit lässt sich die Gleichung des Ornstein-Uhlenbeck-Prozesses $dX = -\theta X\,dt + g\,dW$ lösen. Bei gegebenem Anfangswert $x_0$ ist er zu jedem Zeitpunkt normalverteilt mit Mittelwert $x_0 e^{-\theta t}$ und Varianz $\tfrac{g^2}{2\theta}(1 - e^{-2\theta t})$. Für $g^2 = 2\theta$ ist er der varianzerhaltende Vorwärtsprozess von Diffusionsmodellen. Gleichungen ohne geschlossene Lösung simulieren wir mit Euler-Maruyama. Das Verfahren ergänzt den Euler-Schritt um einen Stoß der Größenordnung $\sqrt{\Delta t}$. Es approximiert Pfade mit der Ordnung $1$ bei additivem und $\tfrac12$ bei multiplikativem Rauschen. Verteilungen gibt es in beiden Fällen mit der Ordnung $1$ wieder.

<!--en:2294243a-->
## Übungen

<!--en:92f8c4cc-->
1. **Unendliche totale Variation.** Die totale Variation eines Pfades $x$ entlang einer Zerlegung $0 = t_0 < t_1 < \cdots < t_n = t$ lautet $V_n = \sum_i |x(t_{i+1}) - x(t_i)|$. Pfadweise Integrale $\int H\,dx$ definieren wir als Grenzwerte von Summen $\sum_i H(s_i)\,\Delta x_i$ mit beliebigen Stützstellen $s_i \in [t_i, t_{i+1}]$. Sie existieren für jeden stetigen Integranden $H$, wenn $V_n$ bei abnehmender Maschenweite beschränkt bleibt.
    1. Setzen Sie $x = W$ und verwenden Sie die gleichmäßige Zerlegung mit $\Delta t_i = t/n$. Berechnen Sie $\mathbb{E}[V_n]$ und $\textrm{Var}(V_n)$ mithilfe von $\mathbb{E}|\xi| = \sqrt{2/\pi}$ für $\xi \sim \mathcal{N}(0, 1)$. Folgern Sie, dass $V_n \to \infty$ in Wahrscheinlichkeit für $n \to \infty$ gilt.
    1. Sei $x$ stetig mit $V_n \le V$ für jede Zerlegung. Zeigen Sie, dass $\max_i |\Delta x_i| \to 0$ für verschwindende Maschenweite gilt, und folgern Sie $\sum_i (\Delta x_i)^2 \le \max_i |\Delta x_i| \cdot V \to 0$. Ein solcher Pfad hat also quadratische Variation null. Warum liefert :eqref:`eq_mdl-sde-qv` damit einen zweiten Beweis dafür, dass brownsche Pfade unbeschränkte totale Variation haben?
    1. Sei $x$ stetig mit beschränkter totaler Variation. Zeigen Sie, dass die linken Summen $\sum_i x(t_i)\,\Delta x_i$ und die rechten Summen $\sum_i x(t_{i+1})\,\Delta x_i$ beide gegen $\tfrac12\left(x(t)^2 - x(0)^2\right)$ konvergieren. Vergleichen Sie dies mit den beiden Grenzwerten für $x = W$ in :numref:`sec_mdl-ito-integral`. Welcher Schritt Ihrer Begründung versagt bei einem brownschen Pfad?
   :exlabel:`ex_mdl-sdes-infinite-total-variation`

<!--en:009f2a64-->
1. [code] **Skalierung der brownschen Bewegung und Austrittszeiten.**
    1. Für $c > 0$ sei $\tilde{W}_t = W_{ct}/\sqrt{c}$. Prüfen Sie die Eigenschaften 1 bis 4 aus :numref:`sec_mdl-wiener-process` und folgern Sie, dass $\tilde{W}$ eine brownsche Bewegung ist. Welche Eigenschaft der Irrfahrt mit Schritten von $\pm\sqrt{\Delta t}$ gilt damit exakt?
    1. Sei $\tau_a$ der erste Zeitpunkt, zu dem $|W_t|$ ein Niveau $a > 0$ erreicht. Zeigen Sie mithilfe von (a), dass $\tau_a$ genauso verteilt ist wie $a^2 \tau_1$ und daher $\mathbb{E}[\tau_a] = a^2\, \mathbb{E}[\tau_1]$ gilt.
    1. Setzen Sie $\mathbb{E}[\tau_1] = 1$ voraus. Dies folgt aus $\mathbb{E}[W_{\tau}^2] = \mathbb{E}[\tau]$, einer Konsequenz aus :eqref:`eq_mdl-sde-wdw` und dem Mittelwert null des Itô-Integrals bis zu einer Stoppzeit. Simulieren Sie $4{,}000$ Irrfahrten mit normalverteilten Schritten der Varianz $\Delta t = 10^{-3}$ und schätzen Sie $\mathbb{E}[\tau_a]$ für $a = 0.5, 1, 2$ samt Monte-Carlo-Standardfehlern. Die Schätzwerte liegen tendenziell etwas über $a^2$, besonders deutlich für kleine $a$. Erklären Sie das Vorzeichen der Verzerrung anhand der Art, wie eine diskrete Irrfahrt ein Niveau überschreitet. Geben Sie an, wie die Verzerrung mit $a$ und $\Delta t$ skaliert.
    1. Im Experiment im Text bleibt etwas mehr als die Hälfte der Pfade innerhalb von $\pm 2\sqrt{t}$ zu allen $500$ Gitterzeitpunkten. Zeigen Sie mit (a), dass dieser Anteil nur von der Zahl $n$ der Gitterzeitpunkte abhängt und nicht von der Intervalllänge. Messen Sie ihn für $n = 50$, $500$ und $5000$. Um welchen Faktor fällt er, wenn sich $n$ jeweils verzehnfacht? Was bedeutet das für den Grenzübergang $n \to \infty$?
   :exlabel:`ex_mdl-sdes-brownian-scaling-exit-times`

<!--en:4753adec-->
1. **Geometrische brownsche Bewegung und die Korrektur $-\nu^2/2$.** Betrachten Sie $dX = \mu X\,dt + \nu X\,dW$ mit $X_0 > 0$ und die zugehörige Lösung aus :eqref:`eq_mdl-sde-gbm`.
    1. Zeigen Sie mithilfe von $\mathbb{E}\big[e^{\lambda Z}\big] = e^{\lambda^2 v/2}$ für $Z \sim \mathcal{N}(0, v)$, dass $\mathbb{E}[X_t] = X_0 e^{\mu t}$ und $\mathbb{E}[X_t^2] = X_0^2 e^{(2\mu + \nu^2) t}$ gelten. Bestimmen Sie den Median von $X_t$.
    1. Setzen Sie $\mu = 0.1$ und $\nu = 0.6$. Was geschieht mit dem Mittelwert, dem Median und einem typischen Pfad für $t \to \infty$? Wie lassen sich diese Aussagen vereinbaren?
    1. Für welche $\nu$ hört der Median bei festem $\mu > 0$ auf zu wachsen? Prüfen Sie die Bedingung für die Parameter aus (b) an $100$ Pfaden. Simulieren Sie diese mit `euler_maruyama` bis $t = 20$ bei Schrittweite $\Delta t = 10^{-2}$. Welcher Anteil der Pfade endet unter $X_0$? Welchen Anteil sagt die exakte Lösung :eqref:`eq_mdl-sde-gbm` voraus?

   *Nach Simo Särkkä und Arno Solin, [Applied Stochastic Differential Equations](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf), Übung 4.3.*
   :exlabel:`ex_mdl-sdes-geometric-brownian-motion-correction`

<!--en:2d0da9b6-->
1. [code] **Starke Ordnung und Milstein-Korrektur.** Wie in :numref:`sec_mdl-sde-strong-weak` erläutert, lässt Euler-Maruyama den Milstein-Term $\tfrac12 g\,\partial_x g\left((\Delta W)^2 - \Delta t\right)$ weg. Seine starke Ordnung beträgt $\tfrac12$, sofern dieser Term nicht verschwindet.
    1. Nach :eqref:`eq_mdl-sde-gbm` erfüllt $Y = \log X$ eine SDE mit additivem Rauschen. Wenden Sie `euler_maruyama` mit den Zuwächsen `dW_f` und den Schrittweiten aus dem Experiment zur starken Ordnung auf diese SDE an. Exponenzieren Sie die Endwerte und messen Sie den starken Fehler gegenüber `gbm_exact`. Erklären Sie das Ergebnis.
    1. Zeigen Sie mit dem Itô-Lemma, dass für $dX = f(X)\,dt + g(X)\,dW$ mit $g > 0$ die Variable $Y = \psi(X)$ mit $\psi' = 1/g$ additives Rauschen der Amplitude eins hat. Bestimmen Sie ihre Drift.
    1. Das Milstein-Verfahren :cite:`Milstein.1975` ergänzt den Euler-Maruyama-Schritt um den zuvor ausgelassenen Term:

       $$\hat X_{n+1} = \hat X_n + f\,\Delta t + g\,\Delta W_n + \tfrac12\, g\,\partial_x g\left((\Delta W_n)^2 - \Delta t\right),$$

       Dabei werden $f$, $g$ und $\partial_x g$ an der Stelle $(\hat X_n, t_n)$ ausgewertet. Implementieren Sie das Verfahren für die geometrische brownsche Bewegung aus dem Experiment zur starken Ordnung. Messen Sie die Steigung für die starke Ordnung mit denselben Zuwächsen und Schrittweiten. Um welchen Faktor verringert das Verfahren den starken Fehler bei $\Delta t = 2^{-8}$?
   :exlabel:`ex_mdl-sdes-milstein-correction`

<!--en:2450ea4a-->
1. **Exakte Erhaltung der Varianz.** Sei $X_t$ durch :eqref:`eq_mdl-sde-vp-marginal` gegeben, mit Daten $X_0$ vom Mittelwert null.
    1. Nehmen Sie an, die Daten haben Varianz eins, und zeigen Sie $\mathbb{E}[X_t^4] = 3 + \bar{\alpha}_t^2\left(\mathbb{E}[X_0^4] - 3\right)$. Leiten Sie für Rademacher-Daten $X_0 = \pm 1$ die Vorhersage $3 - 2\bar{\alpha}_t^2$ aus dem Text her.
    1. Der Exzess $\mathbb{E}[X^4]/\textrm{Var}(X)^2 - 3$ beträgt für jede Normalverteilung null. Mit welcher Rate in $t$ klingt er für $X_t$ ab? Vergleichen Sie diese Rate mit der Rate, mit der der bedingte Mittelwert $\mathbb{E}[X_t \mid X_0]$ die Daten vergisst.
    1. Nehmen Sie nun an, die Daten haben die Varianz $s^2$. Bestimmen Sie $\textrm{Var}(X_t)$. Wie lange muss der Prozess für $s = 2$ und $\theta = 1$ laufen, bis $\textrm{Var}(X_T)$ um höchstens $1\%$ von der Varianz der Referenzverteilung $\mathcal{N}(0, 1)$ abweicht?
   :exlabel:`ex_mdl-sdes-variance-preservation-exactly`

<!--en:84dcc604-->
1. [code] **Itô und Stratonovich.** ● Die linken und rechten Summen für $\int_0^t W\,dW$ ergeben zusammen auf jeder Zerlegung $W_t^2$ (:numref:`sec_mdl-ito-integral`). Ihr Mittelwert ist also $\tfrac12 W_t^2$. Solche Summen definieren das Stratonovich-Integral :cite:`Stratonovich.1966`, das der gewöhnlichen Kettenregel genügt.
    1. Das Heun-Verfahren aus :numref:`sec_mdl-euler-runge-kutta` lässt sich auf SDEs übertragen, wenn wir den Rauschterm wie eine zweite Steigung behandeln. Der Prädiktor ist der Euler-Maruyama-Schritt $\tilde{X} = \hat X_n + f(\hat X_n)\,\Delta t + g(\hat X_n)\,\Delta W_n$, und der Korrektor mittelt beide Koeffizienten:

       $$\hat X_{n+1} = \hat X_n + \tfrac12\left[f(\hat X_n) + f(\tilde{X})\right]\Delta t + \tfrac12\left[g(\hat X_n) + g(\tilde{X})\right]\Delta W_n.$$

       Implementieren Sie das Verfahren für die geometrische brownsche Bewegung aus dem Experiment zur starken Ordnung mit dessen Zuwächsen `dW_f` und Schrittweiten. Vergleichen Sie die Endwerte mit der exakten Itô-Lösung :eqref:`eq_mdl-sde-gbm` und mit $X_0 \exp(\mu T + \nu W_T)$. Gegen welche der beiden Lösungen konvergiert das Verfahren?
    1. Entwickeln Sie den Rauschterm des Korrektors bis zur ersten Ordnung in $\tilde{X} - \hat X_n$. Zeigen Sie mithilfe von $(\Delta W_n)^2 \approx \Delta t$, dass sich das Verfahren wie Euler-Maruyama für eine Itô-SDE mit veränderter Drift verhält. Bestimmen Sie diese Drift und prüfen Sie, ob sie den Grenzwert aus (a) erklärt.
    1. Warum liefern beide Konventionen für jede SDE der Form :eqref:`eq_mdl-sde-template` dieselbe Lösung?
   :exlabel:`ex_mdl-sdes-ito-versus-stratonovich`

<!--en:f02d88d6-->
1. **Exakte Übergänge linearer SDEs.** ● Betrachten Sie die lineare SDE $dX = -\theta(t) X\,dt + g(t)\,dW$ mit stetigen Funktionen $\theta$ und $g$. Sei $\Theta(t) = \int_0^t \theta(u)\,du$.
    1. Wenden Sie das Itô-Lemma auf $\phi(x, t) = e^{\Theta(t)} x$ an. Zeigen Sie, dass bei gegebenem $X_s$ der Zustand $X_t$ zu einem späteren Zeitpunkt $t$ normalverteilt ist, mit Mittelwert $e^{-(\Theta(t) - \Theta(s))} X_s$ und Varianz $\int_s^t e^{-2(\Theta(t) - \Theta(u))}\, g(u)^2\, du$.
    1. Wenden Sie (a) auf die VP-SDE :eqref:`eq_mdl-sde-vp-sde` an. Zeigen Sie, dass ein Übergang beliebiger Länge einen Zustand mit Varianz eins wieder auf einen Zustand mit Varianz eins abbildet. Nehmen Sie umgekehrt für die lineare SDE aus (a) an, dass $\textrm{Var}(X_t) = 1$ für jedes $t \ge 0$ gilt, sobald $\textrm{Var}(X_0) = 1$. Zeigen Sie, dass dann $g(t)^2 = 2\,\theta(t)$ zu jedem Zeitpunkt $t$ gilt.
    1. Betrachten Sie die semilineare SDE $dX = \left(-\theta X + q(X)\right)dt + g\,dW$ mit Konstanten $\theta, g > 0$ und einer glatten Funktion $q$. Der exponentielle Euler-Maruyama-Schritt hält $q$ auf dem Wert $q(\hat X_n)$ während eines Schritts der Länge $\Delta t$ fest. Die verbleibende lineare SDE löst er mit dem Verfahren aus (a) exakt. Geben Sie den Schritt an. Zeigen Sie, dass er mit dem Euler-Maruyama-Schritt :eqref:`eq_mdl-sde-em` bis zur ersten Ordnung in $\Delta t$ übereinstimmt und für konstantes $q$ exakt ist. Für $q \equiv 0$ dämpft Euler-Maruyama den Mittelwert nur bei $\Delta t < 2/\theta$ (:numref:`sec_mdl-stiffness-implicit`). Was begrenzt die Schrittweite beim exponentiellen Schritt?
    1. Exakte Schritte haben bei jeder Schrittweite den schwachen Fehler null. Welche Eigenschaft der Drift und der Diffusion der VP-SDE ermöglicht solche Schritte? Der generative Sampler eines Diffusionsmodells simuliert eine SDE rückwärts in der Zeit. Ihre Drift lautet $-\tfrac12\beta(t)\,x - \beta(t)\,\nabla \log p_t(x)$, also die VP-Drift plus ein Vielfaches von $\nabla \log p_t$, dem *Score* der Randverteilung des Vorwärtsprozesses. Bei nicht normalverteilten Daten ist dieser Score eine nichtlineare Funktion von $x$. Warum lassen sich die Schritte hier nicht exakt ausführen? Exponentielle Integratoren für Diffusionssampler :cite:`Zhang.Chen.2023` übertragen die Idee aus (c) auf das deterministische Gegenstück dieses Samplers. Das ist eine gewöhnliche Differentialgleichung, deren Drift aus denselben zwei Teilen besteht, mit halbiertem Score-Term (:numref:`sec_mdl-fokker-planck-probability-flow`). Welchen Teil dieser Drift behandelt der Schritt exakt, und welchen approximiert er?
   :exlabel:`ex_mdl-sdes-exact-vp-discretization`

<!--en:91a99091-->
1. [extended] **Multilevel-Monte-Carlo-Verfahren.** ●● Schätzen wir $\mathbb{E}[\varphi(X_T)]$ durch Mittelung über Euler-Maruyama-Pfade, gibt es zwei Fehlerquellen: die schwache Verzerrung des Verfahrens und das Stichprobenrauschen. Das Multilevel-Monte-Carlo-Verfahren :cite:`Giles.2008` senkt die Kosten, indem es Pfade über verschiedene Schrittweiten hinweg koppelt. Dies ist eine der in :numref:`sec_mdl-sde-strong-weak` genannten Anwendungen starker Genauigkeit. In dieser Aufgabe hat der starke Fehler im quadratischen Mittel $\mathbb{E}\big[(\hat X_N - X_T)^2\big]^{1/2}$ dieselben Ordnungen wie der starke Fehler aus jenem Abschnitt.
    1. Das gewöhnliche Monte-Carlo-Verfahren mittelt $\varphi(\hat X_N)$ über $M$ unabhängige Pfade. Die Kosten betragen $MT/\Delta t$ Schritte. Nehmen Sie eine schwache Verzerrung von $c_1\,\Delta t$ und eine Varianz $v$ von $\varphi(\hat X_N)$ an. Zeigen Sie, dass ein mittlerer quadratischer Fehler unter $\varepsilon^2$ bei optimaler Wahl der Parameter eine Schrittzahl der Ordnung $\varepsilon^{-3}$ erfordert. Die Parameter sind $M$ und $\Delta t$.
    1. Sei $P_l$ der Wert von $\varphi$ am Endpunkt des Euler-Maruyama-Verfahrens mit Schrittweite $h_l = T\,2^{-l}$. Schreiben Sie $\mathbb{E}[P_L] = \mathbb{E}[P_0] + \sum_{l=1}^{L}\mathbb{E}[P_l - P_{l-1}]$. Schätzen Sie jeden Term aus $M_l$ unabhängigen Pfadpaaren. Wie im Experiment zur starken Ordnung verwendet der grobe Pfad die Summen je zweier aufeinanderfolgender feiner Zuwächse. Geben Sie für eine Lipschitz-stetige Funktion $\varphi$ eine obere Schranke für $V_l = \textrm{Var}(P_l - P_{l-1})$ mithilfe des starken Fehlers im quadratischen Mittel an. Setzen Sie $\varphi(x) = x$ und sagen Sie voraus, wie $V_l$ mit $h_l$ skaliert. Betrachten Sie dazu den OU-Prozess und die geometrische brownsche Bewegung mit den Parametern aus dem Experiment. Messen Sie $V_l$ für $l = 1, \ldots, 8$ mit `euler_maruyama` und bestimmen Sie die Steigungen durch Anpassung auf den feineren Stufen.
    1. Minimieren Sie die Gesamtkosten $\sum_l M_l / h_l$ unter der Nebenbedingung $\sum_l V_l / M_l \le \varepsilon^2 / 2$ mit einem Lagrange-Multiplikator. Wählen Sie die feinste Stufe $L$ so, dass die Verzerrung unter $\varepsilon / \sqrt{2}$ bleibt. Zeigen Sie unter der Annahme $V_l \propto h_l^{r}$, dass die Kosten die Ordnung $\varepsilon^{-2}$ für $r = 2$ und die Ordnung $\varepsilon^{-2} (\log \varepsilon)^2$ für $r = 1$ haben.
    1. Implementieren Sie beide Schätzer für $\mathbb{E}[X_T] = X_0 e^{\mu T}$ der geometrischen brownschen Bewegung aus dem Experiment zur starken Ordnung. Verwenden Sie $M_l$ aus (c) und Ihre Messwerte für $V_l$. Erfassen Sie für $\varepsilon \in \{0.2, 0.1, 0.05, 0.02\}$ die gesamte Schrittzahl jedes Schätzers und prüfen Sie seinen Fehler anhand einiger Wiederholungen. Tragen Sie $\varepsilon^2$ mal die Kosten gegen $\varepsilon$ auf. Wiederholen Sie das Experiment mit dem Milstein-Verfahren aus Übung :exref:`ex_mdl-sdes-milstein-correction`c anstelle von Euler-Maruyama. Da `euler_maruyama` jeden Pfad vollständig speichert, simulieren Sie große $M_l$ in Batches.
    1. Wiederholen Sie die Messung aus (b) für die unstetige Funktion $\varphi(x) = \mathbf{1}(x > X_0)$. Warum klingt $V_l$ langsamer ab, und wie wirkt sich das auf die Kosten aus?
   :exlabel:`ex_mdl-sdes-multilevel-monte-carlo`

<!--en:620e008f-->
[Diskussion](https://d2l.discourse.group/t/36838)

<!--en:20d5f936-->
Stochastische Differentialgleichungen, brownsche Bewegung und Itô-Lemma<br>**Wiener-Prozess · quadratische Variation (dW)² = dt · Itô-Integral und Itô-Isometrie · Itô-Lemma · Ornstein-Uhlenbeck-Prozess**

<!--en:76824be1-->
::: {.slide title="Eine Rekursion, die von jedem Anfangswert zu N(0, 1) führt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:667aadf1-->
10 000 Kopien bei $x_0 = 2$ starten und $x_{k+1} = 0.99\,x_k + \sqrt{0.02}\,\xi_k$ mit $\xi_k \sim \mathcal{N}(0, 1)$ iterieren ($h = 0.01$, $t = kh$). Nach 400 Schritten ist von $x_0 = 2$ in der annähernden Verteilung $\mathcal{N}(0, 1)$ kaum noch etwas zu erkennen:

<!--en:bea83f3a-->
**Warum $\sqrt{h}$ statt $h$?** Im Schritt $h = 0.01$ ändert der deterministische Abklingfaktor $0.99 = 1 - h$ den Wert von $x$ um $O(h)$. Der Rauschstoß $\sqrt{2h} = \sqrt{0.02} \approx 0.141$ hat dagegen die Größe $O(\sqrt{h})$. Sein **Quadrat** ist $O(h)$. Daher versagt die gewöhnliche Kettenregel.

<!--en:3372cbd5-->
::: {.slide title="Warum deterministische Flüsse ihren Anfangszustand behalten"}

<!--en:3933e91a-->
[Vorwärtsprozesse mit Rauschen]{.kicker}

<!--en:c2f4136f-->
Ein generatives Diffusionsmodell braucht einen Vorwärtsprozess, der **jede** Datenverteilung $p_0 = p_{\text{data}}$ auf dieselbe handhabbare A-priori-Verteilung $p_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$ abbildet. Kein umkehrbarer ODE-Fluss $\Phi_T$ kann das: Würde $\Phi_T$ zwei Verteilungen $p \ne q$ auf dieselbe Verteilung abbilden, folgte durch Anwenden von $\Phi_T^{-1}$ der Widerspruch $p = q$.

<!--en:2fe0490c-->
- **Drift allein ($\dot{x} = -x$):** Führt $x_0 = \pm 2$ zur Punktmasse bei $0$ ($\pm 0.0996$ bei $t=3$), also zu keiner Normalverteilung.

<!--en:aca2d67e-->
- **Rauschen allein ($dX = \sqrt{2}\,dW$):** Verbreitert jeden Anfangswert gemäß $\mathcal{N}(x_0, 2t)$ (**varianzexplodierend**). Ab $\sqrt{2t} \gg |x_0|$ brauchbar, aber ohne stationäre Grenzverteilung.

<!--en:75cd536e-->
- **Drift und Rauschen ($dX = -X\,dt + \sqrt{2}\,dW$):** Abklingen des Mittelwerts und Zufuhr von Rauschen führen zur stationären Verteilung $\mathcal{N}(0, 1)$ ($\mathcal{N}(\pm 0.0996, 0.9975)$ bei $t = 3$).

<!--en:77479dd9-->
::: {.slide title="Von Irrfahrten zur Skalierung mit √Δt"}

<!--en:3ba610f8-->
[Konstruktion der brownschen Bewegung]{.kicker}

<!--en:0699e8e8-->
Nach $k = t/\Delta t$ unabhängigen Schritten der Größe $\pm c(\Delta t)^a$ skaliert die Gesamtvarianz zur Zeit $t$ wie folgt:

<!--en:9ade3d15-->
- **Einziger kritischer Exponent ($a = 1/2$):** Für $a > 1/2$ verschwindet die Varianz im Grenzwert $\Delta t \to 0$. Für $a < 1/2$ wächst sie gegen $\infty$. Nur Schritte von $\pm\sqrt{\Delta t}$ ergeben den Grenzwert $W_t \sim \mathcal{N}(0, t)$.

<!--en:b95d8fc4-->
::: {.slide title="Wiener-Prozess: Axiome, Kovarianz und Rauheit"}

<!--en:6c410b17-->
[Brownsche Bewegung]{.kicker}

<!--en:608ec6fc-->
Den Grenzprozess $W_t$ in kontinuierlicher Zeit (**brownsche Bewegung** oder **Wiener-Prozess**) charakterisieren vier Axiome: (1) $W_0 = 0$, (2) **unabhängige Zuwächse** $W_t - W_s$ für $s < t$, (3) **normalverteilte Zuwächse** $W_t - W_s \sim \mathcal{N}(0, t - s)$ und (4) **stetige Pfade**.

<!--en:67d9d233-->
- **Exakte Zuwächse:** Für jedes Intervall der Länge $\Delta t$ gilt
  $$\Delta W = W_{t + \Delta t} - W_t = \sqrt{\Delta t}\,\xi, \quad \xi \sim \mathcal{N}(0, 1)$$

<!--en:df4454e4-->
- **Kovarianz zweier Zeitpunkte:** Für $s \le t$ folgt aus $W_t = W_s + (W_t - W_s)$ die Kovarianz $\operatorname{Cov}(W_s, W_t) = \mathbb{E}[W_s^2] = \min(s, t)$.

<!--en:9548ed42-->
- **Nirgends differenzierbare Pfade:** Der Betrag des Differenzenquotienten wächst für $\Delta t \to 0$ unbegrenzt:
  $$\left|\frac{\Delta W}{\Delta t}\right| = \frac{|\xi|}{\sqrt{\Delta t}} \;\xrightarrow{\Delta t \to 0}\; \infty$$

<!--en:1766f82f-->
::: {.slide title="Randverteilungen und Bänder für ganze Pfade"}

<!--en:06508599-->
[Simulation brownscher Pfade]{.kicker}

<!--en:547b7a4e-->
Zu jedem festen Zeitpunkt $t$ liegt die Zufallsvariable $W_t \sim \mathcal{N}(0, t)$ im Band $\pm 2\sqrt{t}$ von zwei Standardabweichungen mit Wahrscheinlichkeit $95.4\%$. Welcher Anteil von $20{,}000$ simulierten Pfaden bleibt innerhalb von $\pm 2\sqrt{t}$ zu *allen* Gitterzeitpunkten auf $[0, 1]$?

<!--en:9d52a2d8-->
**Einzelne Zeitpunkte und ganze Pfade:** Mit jedem weiteren Beobachtungszeitpunkt kann ein rauer Pfad erneut außerhalb der Grenzen liegen. Bei 50 Zeitpunkten bleiben $70\%$ im Band, bei 500 Zeitpunkten $54\%$ und in kontinuierlicher Zeit $0\%$, da brownsche Pfade die Grenzen $\pm 2\sqrt{t}$ nahe $t = 0$ unendlich oft überschreiten.

<!--en:44b26d4d-->
::: {.slide title="Quadratische Variation der brownschen Bewegung: (dW)² = dt"}

<!--en:8e75e3d7-->
[Grundlage des Itô-Kalküls]{.kicker}

<!--en:84603f1c-->
Für $[0, t]$ die Zerlegung $0 = t_0 < t_1 < \cdots < t_n = t$ mit Maschenweite $h_{\max} = \max_i \Delta t_i$ wählen. Für einen glatten Pfad mit $|x'(t)| \le M$ verschwindet die Summe der quadrierten Zuwächse ($\sum_i (\Delta x_i)^2 \le M^2 h_{\max} t \to 0$). Bei der brownschen Bewegung konvergiert die zufällige Summe gegen die deterministische Zeit $t$:

<!--en:f3c12dd2-->
- **Mittelwert gleich verstrichener Zeit:** Aus $\Delta W_i \sim \mathcal{N}(0, \Delta t_i)$ folgt
  $$\mathbb{E}[Q_n] = \sum_{i=0}^{n-1} \mathbb{E}\!\left[(\Delta W_i)^2\right] = \sum_{i=0}^{n-1} \Delta t_i = t$$

<!--en:b5d9b412-->
- **Varianz verschwindet mit Ordnung $O(h_{\max})$:** Aus $\mathbb{E}[\xi^4] = 3$ folgt $\operatorname{Var}((\Delta W_i)^2) = 2\Delta t_i^2$. Wegen der Unabhängigkeit gilt daher
  $$\mathbb{E}\!\left[(Q_n - t)^2\right] = \operatorname{Var}(Q_n) = 2\sum_{i=0}^{n-1} \Delta t_i^2 \le 2h_{\max}t \to 0$$

<!--en:2a0e94bf-->
::: {.slide title="Quadratische Variation an einem einzelnen Pfad messen"}

<!--en:1b718c47-->
[Numerische Überprüfung]{.kicker}

<!--en:a71a297d-->
Einen einzigen brownschen Pfad auf $[0, 1]$ auf Teilgittern mit $n = 16$ bis $2^{18}$ Intervallen betrachten. Es bestätigt sich $\sum_{i=0}^{n-1} (\Delta W_i)^2 \to 1$ mit Standardabweichung $\sqrt{2/n}$, während für den glatten Vergleichspfad gilt: $\sum (\Delta \sin 2\pi t_i)^2 \approx 2\pi^2 / n \to 0$.

<!--en:e561605a-->
**Itô-Multiplikationstabelle:** Bei stochastischen Differentialen bleiben quadrierte brownsche Zuwächse in erster Ordnung erhalten. Produkte höherer Ordnung verschwinden:
$$(dW)^2 = dt, \qquad dW\,dt = 0, \qquad (dt)^2 = 0$$

<!--en:b3073999-->
::: {.slide title="Warum Riemann-Summen für stochastische Integrale unterschiedliche Werte liefern"}

<!--en:c7e31ce8-->
[Die Wahl der Stützstelle]{.kicker}

<!--en:e0b6d591-->
Für eine glatte Kurve konvergieren Riemann-Summen mit linken und mit rechten Stützstellen gegen dasselbe Integral. Bei $\int_0^t W_s\,dW_s$ ist ihre Differenz die quadratische Variation:

<!--en:7ce63e77-->
- **Zwei verschiedene Grenzwerte:** Zusammen ergeben linke und rechte Näherung eine Teleskopsumme ($\sum_i (W_{t_{i+1}} + W_{t_i})\Delta W_i = \sum_i (W_{t_{i+1}}^2 - W_{t_i}^2) = W_t^2$). Die linke Summe konvergiert daher gegen $\frac{1}{2}(W_t^2 - t)$, die rechte gegen $\frac{1}{2}(W_t^2 + t)$.

<!--en:9e130fd3-->
::: {.slide title="Itô-Integral und Itô-Isometrie"}

<!--en:70094b5f-->
[Stochastische Integration]{.kicker}

<!--en:7be09125-->
Das **Itô-Integral** wertet $H_s$ am **linken Rand** $t_i$ aus. Daher ist $H_i$ unabhängig vom kommenden Stoß $\Delta W_i = W_{t_{i+1}} - W_{t_i}$:

<!--en:d5d6278f-->
- **Martingal und Itô-Isometrie:** Linke Stützstellen sichern den Mittelwert null. Beim Quadrieren der Summe verschwinden im Erwartungswert alle gemischten Terme mit $i < j$.

<!--en:e7f2a5bb-->
::: {.slide title="Stochastische Differentialgleichungen nach Itô: Drift und Rauschen"}

<!--en:28e7d8c8-->
[Aufbau einer SDE]{.kicker}

<!--en:0f1f3be9-->
Eine **stochastische Differentialgleichung nach Itô (SDE)** verbindet einen deterministischen Geschwindigkeitsvektor mit einem zufälligen brownschen Stoß:

<!--en:e380486b-->
- **Drift $f(X, t)$:** Die bedingte mittlere Geschwindigkeit über ein kurzes Intervall, $\mathbb{E}[dX \mid X_t] = f(X_t, t)\,dt$.

<!--en:a8224196-->
- **Diffusion $g(X, t)$:** Die Rauschamplitude bestimmt die lokale Rate der Varianzzufuhr, $\operatorname{Var}(dX \mid X_t) = g(X_t, t)^2\,dt$.

<!--en:980103ea-->
- **Additives und multiplikatives Rauschen:**
  - **Additives Rauschen ($g = g(t)$):** Die Amplitude hängt nur von der Zeit ab (wie in allen gängigen Diffusionsmodellen).
  - **Multiplikatives Rauschen ($g = g(X, t)$):** Die Amplitude hängt vom Zustand ab (z. B. bei der geometrischen brownschen Bewegung mit $g(X) = \nu X$).

<!--en:e6c2b101-->
::: {.slide title="Warum die gewöhnliche Kettenregel versagt: die Jensen-Ungleichung"}

<!--en:9f64ac72-->
[Geometrische Deutung des Itô-Lemmas]{.kicker}

<!--en:35527ae0-->
Die gewöhnliche Kettenregel sagt $d(W_t^2) = 2W_t\,dW_t$ voraus. Tatsächlich ergibt das Integral $\int_0^t 2W_s\,dW_s = W_t^2 - t$. In der naiven Rechnung fehlt der Term $+t$. Ein Schritt einer Irrfahrt mit Münzwürfen zeigt, warum symmetrisches Rauschen bei einer gekrümmten Funktion $\phi(x)$ eine deterministische Drift bewirkt:

<!--en:e15e7ce4-->
- **Symmetrischer Stoß auf einer konvexen Kurve:** Von $x$ aus führt ein Schritt von $\pm\sqrt{\Delta t}$ nach $x + \sqrt{\Delta t}$ oder $x - \sqrt{\Delta t}$, jeweils mit Wahrscheinlichkeit $\frac{1}{2}$.

<!--en:2a2e92a5-->
- **Differenz zweiter Ordnung in der Jensen-Ungleichung:** Für $\phi(x) = x^2$ ($\phi''(x) = 2$) beträgt der Mittelwert nach dem Schritt
  $$\frac{1}{2}\left[(x + \sqrt{\Delta t})^2 + (x - \sqrt{\Delta t})^2\right] = x^2 + \Delta t = \phi(x) + \frac{1}{2}\phi''(x)\,\Delta t$$

<!--en:52a12212-->
- **Von null verschiedene Rate:** Wegen $(\Delta W)^2 \approx \Delta t$ bleibt der krümmungsbedingte Zuwachs $+\frac{1}{2}\phi''(x)$ pro Zeiteinheit im Grenzübergang $\Delta t \to 0$ erhalten.

<!--en:9075a0b2-->
::: {.slide title="Das Itô-Lemma (die stochastische Kettenregel)"}

<!--en:0fcbf466-->
[Zentraler Satz]{.kicker}

<!--en:2bd74413-->
Die Entwicklung von $\phi(X_t, t)$ bis zur zweiten Ordnung in $dX = f\,dt + g\,dW$ liefert mit $(dX)^2 = g^2(dW)^2 = g^2\,dt$ das **Itô-Lemma**:

<!--en:d4e551db-->
- **Mehrdimensionale isotrope Form ($\mathbb{R}^d$):** $d\phi = (\partial_t\phi + \nabla\phi\cdot\mathbf{f} + \tfrac12 g(t)^2\Delta\phi)\,dt + g(t)\nabla\phi\cdot d\mathbf{W}_t$. Dabei ist $\tfrac12 g^2\Delta\phi$ die Itô-Korrektur für die Krümmung.

<!--en:a3b5abee-->
::: {.slide title="Wie sich die Itô-Korrektur auf einem Pfad aufsummiert"}

<!--en:27b6be3c-->
[Numerische Überprüfung des Itô-Lemmas]{.kicker}

<!--en:c5557f79-->
Für $X_t = W_t$ ($f = 0, g = 1$) und $\phi(x) = x^2$ ergibt das Itô-Lemma $d(W_t^2) = 2W_t\,dW_t + dt$. Also gilt auf jedem einzelnen Pfad deterministisch $W_t^2 - \int_0^t 2W_s\,dW_s = t$:

<!--en:27a5bbd1-->
Die naive Kettenregel liefert die Kurve $\int_0^t 2W_s\,dW_s$. Sie liegt unter $W_t^2$, und zwar um die fortlaufende quadratische Variation $\sum (\Delta W_i)^2 \approx t$. Mit der Korrektur $+t$ stimmen die Pfade im Grenzwert exakt überein.

<!--en:084a3349-->
::: {.slide title="Anwendungsbeispiel: geometrische brownsche Bewegung"}

<!--en:4c31277e-->
[Multiplikatives Rauschen und Driftänderung durch Konkavität]{.kicker}

<!--en:6d4cb371-->
Die SDE $dX_t = \mu X_t\,dt + \nu X_t\,dW_t$ ($X_0 > 0$) modelliert eine positive Größe, deren Rauschamplitude $g(X) = \nu X$ proportional zum aktuellen Wert ist (**multiplikatives Rauschen**):

<!--en:0f295729-->
- **Logarithmieren mit dem Itô-Lemma:** Das Itô-Lemma auf $\phi(x) = \log x$ anwenden. Mit $\phi'(x) = \frac{1}{x}$ und $\phi''(x) = -\frac{1}{x^2}$ folgt
  $$d\log X_t = \frac{1}{X_t}\,dX_t + \frac{1}{2}\left(-\frac{1}{X_t^2}\right)(\nu^2 X_t^2)\,dt = \left(\mu - \frac{1}{2}\nu^2\right)dt + \nu\,dW_t$$

<!--en:e7f05325-->
- **Exakte geschlossene Lösung:** Integration von $0$ bis $t$ und anschließendes Exponenzieren ergeben
  $$X_t = X_0 \exp\!\left(\left(\mu - \frac{1}{2}\nu^2\right)t + \nu W_t\right)$$

<!--en:ebbb50ab-->
- **Mittelwert und Median laufen auseinander:** Da $\log x$ streng konkav ist ($\phi'' < 0$), senkt symmetrisches Rauschen die Wachstumsrate des Medians auf $\mu - \frac{1}{2}\nu^2$. Der Erwartungswert wächst weiter mit der Driftrate gemäß $\mathbb{E}[X_t] = X_0 e^{\mu t}$.

<!--en:06c17415-->
::: {.slide title="Ornstein-Uhlenbeck-Prozess und Rückkehr zum Mittelwert"}

<!--en:ef07149e-->
[Der prototypische Diffusionsprozess]{.kicker}

<!--en:2be01088-->
Brownsches Rauschen $g\,dW$ mit konstanter Amplitude ergänzt die lineare Abklinggleichung $\dot{x} = -\theta x$ ($\theta > 0$) zum **Ornstein-Uhlenbeck-Prozess (OU-Prozess)**:

<!--en:6b36e8a2-->
- **Exakte Lösung mit integrierendem Faktor:** Das Itô-Lemma auf $\phi(x, t) = e^{\theta t}x$ anwenden. Da $\phi$ linear in $x$ ist ($\phi_{xx} = 0$), ist die Itô-Korrektur **null**:
  $$d\!\left(e^{\theta t}X_t\right) = \theta e^{\theta t}X_t\,dt + e^{\theta t}(-\theta X_t\,dt + g\,dW_t) = g\,e^{\theta t}\,dW_t$$

<!--en:aa0aed6c-->
- **Integral mit abklingendem Gedächtnis:** Integration von $0$ bis $t$ ergibt
  $$X_t = X_0\,e^{-\theta t} + g\int_0^t e^{-\theta(t-s)}\,dW_s$$

<!--en:7e194dd6-->
::: {.slide title="Gaußscher Übergangskern und stationäre Verteilung des OU-Prozesses"}

<!--en:5d61fcfa-->
[Übergangskern]{.kicker}

<!--en:08ad679a-->
Da der Integrand $e^{-\theta(t-s)}$ deterministisch ist, ist $\int_0^t e^{-\theta(t-s)}\,dW_s$ eine normalverteilte Zufallsvariable mit Mittelwert $0$. Ihre Varianz ergibt sich direkt aus der **Itô-Isometrie**:

<!--en:ff48566c-->
**Proposition (OU-Übergangskern):** Bei gegebenem $X_0 = x_0$ ist der Zustand zu jedem Zeitpunkt $t > 0$ exakt normalverteilt:
$$X_t \mid X_0 = x_0 \;\sim\; \mathcal{N}\!\left(\alpha_t x_0,\; \sigma_t^2\right), \qquad \alpha_t = e^{-\theta t}, \qquad \sigma_t^2 = \frac{g^2}{2\theta}\left(1 - e^{-2\theta t}\right)$$

<!--en:fba642dd-->
- **Stationäre Verteilung ($t \to \infty$):** Nach wenigen Vielfachen der Relaxationszeit $1/\theta$ ist der Mittelwert fast verschwunden ($x_0 e^{-\theta t} \to 0$). Die Varianz nähert sich $\sigma_\infty^2 = \frac{g^2}{2\theta}$. Daher gilt $X_t \to \mathcal{N}\!\left(0, \frac{g^2}{2\theta}\right)$ für jeden Anfangszustand $x_0$.

<!--en:1aa97064-->
::: {.slide title="Varianzerhaltende Normierung und VP-SDE"}

<!--en:af937992-->
[Grundlage des DDPM]{.kicker}

<!--en:1243e560-->
Die Wahl $g^2 = 2\theta$ legt die stationäre Varianz auf eins fest ($\frac{g^2}{2\theta} = 1$) und koppelt die Rauschvarianz über $\sigma_t^2 = 1 - \alpha_t^2$ direkt an das Abklingen des Signals. Mit $\bar{\alpha}_t = \alpha_t^2 = e^{-2\theta t}$ (DDPM-Notation) gilt:

<!--en:e949f841-->
- **Warum heißt der Prozess varianzerhaltend (VP)?** Haben die Daten $X_0$ den Mittelwert $0$ und die Varianz eins, $\operatorname{Var}(X_0) = 1$, so gilt zu **jedem** Zeitpunkt $t \ge 0$:
  $$\operatorname{Var}(X_t) = \bar{\alpha}_t \operatorname{Var}(X_0) + (1 - \bar{\alpha}_t) = \bar{\alpha}_t + 1 - \bar{\alpha}_t = 1$$

<!--en:c61845c4-->
- **Änderung der Verteilungsform:** Höhere Momente nähern sich denen der Normalverteilung. Bei Rademacher-Daten $X_0 \in \{\pm 1\}$ steigt das vierte Moment stetig vom Anfangswert $\mathbb{E}[X_0^4] = 1$ gemäß der Formel $\mathbb{E}[X_t^4] = 3 - 2\bar{\alpha}_t^2 \to 3$.

<!--en:34245be2-->
- **Zeitabhängige Rate (VP-SDE):** Wächst die Rate $\beta(t) = 2\theta(t) = g(t)^2$ mit der Zeit, ergibt sich $dX = -\frac{1}{2}\beta(t)X\,dt + \sqrt{\beta(t)}\,dW$. Für die Randverteilung gilt $\bar{\alpha}_t = \exp\!\big(-\int_0^t \beta(s)\,ds\big)$.

<!--en:b3ec95de-->
::: {.slide title="Simulation von SDEs: Euler-Maruyama und exakte Übergänge"}

<!--en:4ea66207-->
[Numerische Verfahren für SDEs]{.kicker}

<!--en:4e0b6dcb-->
**Euler-Maruyama** hält $f$ und $g$ während jedes Schritts der Länge $\Delta t$ auf ihren Anfangswerten fest und addiert einen normalverteilten Stoß der Größenordnung $\sqrt{\Delta t}$:

<!--en:003b6f63-->
- **Exakter OU-Schritt und Euler-Maruyama ($\theta = 1, g = \sqrt{2}, \Delta t = 0.01$):** Exakte DDPM-Schritte nutzen $a = e^{-\theta\Delta t}$ und $b = 1 - e^{-2\theta\Delta t}$ ($v_\infty = 1.000$). Der Faktor $1 - \theta\Delta t$ bei Euler-Maruyama führt zur erhöhten Varianz $v_\infty = 1.005$ (Verzerrung $+0.5\%$).

<!--en:7b14f5c0-->
::: {.slide title="Euler-Maruyama für Pfade im Batch mit gemeinsamem Rauschen"}

<!--en:038c5778-->
[Simulation stochastischer Differentialgleichungen]{.kicker}

<!--en:32f3aee1-->
Die brownschen Zuwächse `dW` als eigenes Argument übergeben. So lässt sich die starke pfadweise Konvergenz auf groben und feinen Zeitgittern mit **genau derselben Realisierung des Rauschens** messen:

<!--en:c476c19a-->
- **Milstein-Korrektur höherer Ordnung ($p_{\text{strong}} = 1$):** Das vernachlässigte Diffusionsintegral $\frac{1}{2}g(\hat{X}_n)\,\partial_x g(\hat{X}_n)\big((\Delta W_n)^2 - \Delta t\big)$ zum Euler-Maruyama-Schritt addieren. Das ergibt selbst bei multiplikativem Rauschen wieder die starke Ordnung $1$.

<!--en:76ffcbdc-->
::: {.slide title="Starker Fehler der Pfade und schwacher Fehler der Verteilung"}

<!--en:8ad08df4-->
[Zwei Konvergenzbegriffe]{.kicker}

<!--en:9a076ce0-->
Da die Lösungen einer SDE Zufallsvariablen sind, gibt es zwei Maße für die numerische Genauigkeit. Wie gut folgt ein simulierter Pfad der konkreten Realisierung von $W_t$? Wie gut stimmt das Histogramm zur Zeit $T$ mit der wahren Randverteilung überein?

<!--en:2d512ff9-->
- **Starker Fehler ($\mathbb{E}\,|\hat{X}_N - X_T| \le C\,\Delta t^p$):** Koppelt grobe und feine Läufe über **genau denselben brownschen Pfad** $W_t$. Benötigt für das Multilevel-Monte-Carlo-Verfahren.

<!--en:29ce00b0-->
- **Schwacher Fehler ($|\mathbb{E}\,\varphi(\hat{X}_N) - \mathbb{E}\,\varphi(X_T)| \le C\,\Delta t^p$):** Vergleicht Erwartungswerte oder Histogramme zur Zeit $T$ über unabhängige Ziehungen des Rauschens. **Generative Diffusionssampler benötigen nur schwache Konvergenz.**

<!--en:3e9a4d21-->
::: {.slide title="Starke Ordnung bei additivem und multiplikativem Rauschen"}

<!--en:10d0da67-->
[Pfadweise Konvergenz]{.kicker}

<!--en:022123aa-->
Warum erreicht Euler-Maruyama beim OU-Prozess mit additivem Rauschen die starke Ordnung $p = 1$, bei der geometrischen brownschen Bewegung mit multiplikativem Rauschen aber nur die starke Ordnung $p = 1/2$ ($g(X) = \nu X$)?

<!--en:3ed729ed-->
- **Der fehlende Milstein-Term:** Bleibt $g(X)$ während des Schritts konstant, fehlt seine Änderung $g(X_s) - g(X_{t_n}) \approx g\,\partial_x g\,(W_s - W_{t_n})$ innerhalb des Schritts. Ihr Itô-Integral über $[t_n, t_{n+1}]$ lautet
  $$g\,\partial_x g \int_{t_n}^{t_{n+1}} (W_s - W_{t_n})\,dW_s = \frac{1}{2}g\,\partial_x g\left((\Delta W_n)^2 - \Delta t\right)$$

<!--en:aadcfa03-->
- **Irrfahrt der Schrittfehler:** Pro Schritt fehlt ein Term mit Mittelwert null und Größe $O(\Delta t)$. Die Summe von $N = T/\Delta t$ unabhängigen Fehlern hat die Größenordnung $\sqrt{N}\,\Delta t = O(\Delta t^{1/2})$. Ausnahme: $\partial_x g = 0$ (additives Rauschen mit starker Ordnung $O(\Delta t)$).

<!--en:fd805158-->
::: {.slide title="Schwache Ordnung 1 mit exakten Momentenrekursionen messen"}

<!--en:d1b9c9ed-->
[Konvergenz der Verteilung]{.kicker}

<!--en:7c641fb2-->
Da jeder vernachlässigte stochastische Term den **Mittelwert null** hat, beträgt die Verzerrung der Verteilung pro Euler-Maruyama-Schritt $O(\Delta t^2)$. Über $N = T/\Delta t$ Schritte ergibt sich die **schwache Ordnung $1$** ($O(\Delta t)$), sowohl bei additivem als auch bei multiplikativem Rauschen:

<!--en:64908a6a-->
**Fehlerquelle beim Wechsel von Itô zu Stratonovich:** ODE-Löser höherer Ordnung lassen sich bei multiplikativem Rauschen nicht direkt übernehmen. Das Prädiktor-Korrektor-Verfahren von Heun mittelt $g(X)$ am linken und rechten Rand. Es konvergiert zur **Stratonovich**-Lösung $X_0 e^{\mu T + \nu W_T}$ statt zur Itô-Lösung $X_0 e^{(\mu - \nu^2/2)T + \nu W_T}$.

<!--en:1cc0de35-->
::: {.slide title="Verständnisfragen: Itô-Kalkül und Ornstein-Uhlenbeck-Prozess"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:f5ee458d-->
Drei Aufgaben zur stochastischen Analysis lösen:

<!--en:ecd2d85d-->
1. **Itô-Lemma für die dritte Potenz:** Für die brownsche Standardbewegung $W_t$ das stochastische Differential $d(W_t^3)$ berechnen und $\mathbb{E}[W_t^3]$ und $\mathbb{E}[W_t^2]$ bestimmen.

<!--en:f106d5cc-->
2. **Stationäre Varianz:** Gegeben sei der OU-Prozess $dX_t = -3X_t\,dt + \sqrt{3}\,dW_t$ mit Anfangswert $X_0 = 5$. Wie lautet seine Grenzverteilung für $t \to \infty$?

<!--en:8e2ca0aa-->
3. **Starke und schwache Ordnung:** Die Schrittweite ($\Delta t \to \Delta t / 4$) bei Euler-Maruyama für eine SDE mit multiplikativem Rauschen vierteln. Um welchen Faktor sinkt der **starke Fehler**, um welchen der **schwache Fehler**?

<!--en:557b98b6-->
**Lösungen:**

<!--en:5b4fbf91-->
1. Für $\phi(x) = x^3$ gelten $\phi' = 3x^2$ und $\phi'' = 6x$, also $d(W_t^3) = 3W_t^2\,dW_t + \frac{1}{2}(1)^2(6W_t)\,dt = 3W_t^2\,dW_t + 3W_t\,dt$. Aus Symmetriegründen gilt $\mathbb{E}[W_t^3] = 0$. Für das zweite Moment folgt $d(W_t^2) = 2W_t\,dW_t + dt \implies \mathbb{E}[W_t^2] = t$.

<!--en:cce96e62-->
2. Für $\theta = 3$ und $g^2 = 3$ lautet die stationäre Verteilung $\mathcal{N}\!\left(0, \frac{g^2}{2\theta}\right) = \mathcal{N}\!\left(0, \frac{1}{2}\right)$.

<!--en:22daccd1-->
3. Der starke Fehler (Ordnung $1/2$) sinkt um den Faktor $4^{1/2} = 2\times$, der schwache Fehler (Ordnung $1$) um den Faktor $4^1 = 4\times$.

<!--en:f2569b54-->
::: {.slide title="Zusammenfassung: brownsche Bewegung, Itô-Kalkül und SDEs"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:6b834c03-->
| Begriff | Formel | Bedeutung für Diffusion und stochastische Modellierung |

<!--en:da492180-->
| **Brownscher Zuwachs** | $\Delta W = W_{t+\Delta t} - W_t = \sqrt{\Delta t}\,\xi$, $\xi \sim \mathcal{N}(0, 1)$ | Einzige Skalierung mit endlicher, von null verschiedener Varianz $\operatorname{Var}(W_t) = t$ |

<!--en:1fc2436a-->
| **Quadratische Variation** | $\sum_i (\Delta W_i)^2 \to t$ in $L^2$ $\iff (dW)^2 = dt$ | Taylor-Terme zweiter Ordnung bleiben in der Ordnung $dt$ erhalten |

<!--en:6c90996a-->
| **Itô-Integral und Itô-Isometrie** | $\mathbb{E}\!\left[\int_0^t H_s\,dW_s\right] = 0$, $\mathbb{E}\!\left[\left(\int_0^t H_s\,dW_s\right)^2\right] = \int_0^t \mathbb{E}[H_s^2]\,ds$ | Linke Stützstellen sichern den Mittelwert null und ermöglichen die Berechnung der Varianz |

<!--en:99352bae-->
| **Itô-Lemma** | $d\phi = (\phi_t + f\phi_x + \frac{1}{2}g^2\phi_{xx})\,dt + g\phi_x\,dW$ | Stochastische Kettenregel mit Laplace-Term $\frac{1}{2}g^2\Delta\phi$ in $\mathbb{R}^d$ |

<!--en:29b4953e-->
| **Übergangskern von OU-Prozess und VP-SDE** | $X_t \mid X_0 \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}X_0, (1-\bar{\alpha}_t)\mathbf{I})$ ($g^2 = 2\theta$) | Verrauschen im Vorwärtsprozess bis hin zu $\mathcal{N}(\mathbf{0}, \mathbf{I})$ mit einer einzigen Ziehung aus der Normalverteilung |

<!--en:7d43962e-->
| **Euler-Maruyama** | $\hat{X}_{n+1} = \hat{X}_n + f\Delta t + g\sqrt{\Delta t}\,\xi_n$ | Starke Ordnung $1$ (additiv) oder $1/2$ (multiplikativ), schwache Ordnung $1$ |
