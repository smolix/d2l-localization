<!--en:e68f4615-->
# Optimale Regelung, LQR und modellprädiktive Regelung

<!--en:25f0563b-->
Mit dynamischer Programmierung (:numref:`sec_valueiter`) haben wir das Problem des rutschigen Sees gelöst. Dazu haben wir eine Tabelle mit sechzehn Zustandswerten so oft durchlaufen, bis die Bellman-Optimalitätsgleichung für jedes Feld galt. Alle seither betrachteten Verfahren, vom Q-Learning bis zum Verhaltensklonen, verwenden eine Tabellenzeile pro Zustand. Was geschieht, wenn der Zustand wie beim Fahrzeug aus :numref:`sec_mdp` ein kontinuierlicher Vektor $\mathbf{x}_t \in \mathbb{R}^d$ aus Positionen, Winkeln und Geschwindigkeiten ist? Und wenn die Aktion eine kontinuierliche Stellgröße $\mathbf{u}_t \in \mathbb{R}^m$ ist, die über die bekannte Dynamik $\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$ wirkt? Ein Gitter stößt hier schnell an seine Grenzen. Teilen wir jede der $d$ Koordinaten in $100$ Intervalle, erhalten wir $100^d$ Gitterzellen. Das ist Bellmans *Fluch der Dimensionalität* (engl. curse of dimensionality) :cite:`BellmanDPBook`.

<!--en:2d9364b2-->
Die Regelungstheorie löste dieses Problem für kontinuierliche Zustands- und Aktionsräume lange vor dem tiefen bestärkenden Lernen. Ausgangspunkt war eine algebraische Beobachtung von :citet:`Kalman.1960`. Ist die Übergangsdynamik linear in Zustand und Stellgröße und sind die Kosten eines Schritts in beiden quadratisch, so bleibt die quadratische Form der Wertfunktion bei jedem Rückwärtsschritt der dynamischen Programmierung nach Bellman erhalten. Statt einer Tabelle mit $100^d$ Einträgen brauchen wir für die exakte Wertfunktion $V_t(\mathbf{x}) = \mathbf{x}^\top \mathbf{P}_t \mathbf{x}$ nur eine symmetrische $d \times d$-Matrix $\mathbf{P}_t$ mit $d(d+1)/2$ unabhängigen Einträgen. Das Bellman-Backup vereinfacht sich zur *Riccati-Rekursion*. Die optimale Strategie (engl. optimal policy) hat die Form einer linearen Zustandsrückführung $\mathbf{u}_t^* = -\mathbf{K}_t \mathbf{x}_t$. Dies ist der *linear-quadratische Regler* (engl. linear-quadratic regulator, LQR) :cite:`Kalman.1960,Anderson.Moore.2007`.

<!--en:347d1d6a-->
Die meisten physikalischen Systeme sind nichtlinear, und ihre Aktoren erreichen Sättigungsgrenzen. Zwei Ideen erweitern LQR auf nichtlineare Systeme mit kontinuierlichen Stellgrößen. Beide sind grundlegend für die moderne Trajektorienoptimierung und Robotik :cite:`Tedrake.2023,Bertsekas.2025`. Erstens linearisiert das *iterative LQR-Verfahren* (iLQR) :cite:`Li.Todorov.2004` die nichtlineare Dynamik wiederholt um eine vorläufige Trajektorie und nähert dort die Kosten quadratisch an. Im Rückwärtsdurchlauf löst es ein zeitvariantes LQR-Problem, um im Vorwärtsdurchlauf die Trajektorie zu verbessern. Zweitens macht die *modellprädiktive Regelung* (engl. model predictive control, MPC) :cite:`Rawlings.Mayne.Diehl.2017` aus der Trajektorienoptimierung mit endlichem Horizont eine Strategie mit Rückführung. In jedem Schritt $t$ löst sie das Regelungsproblem über einen gleitenden Horizont neu. Sie geht dabei vom aktuell gemessenen Zustand $\mathbf{x}_t$ aus, wendet nur die erste Stellgröße $\mathbf{u}_{t|t}^*$ an und plant im Schritt $t+1$ erneut. In diesem Abschnitt leiten wir LQR, iLQR und MPC mit gleitendem Horizont her. Wir untersuchen, wann eine lokale lineare Rückführung ausreicht und wann wir dem System durch nichtlineare Regelung Energie zuführen müssen. Außerdem sehen wir, wie MPC die Planung mit erlernten neuronalen Dynamikmodellen in :numref:`sec_model_based` vorbereitet.

<!--en:ad8ea4d7-->
## Linear-quadratische Regelung

<!--en:f01a288f-->
### Dynamik im Zustandsraum und quadratische Kosten

<!--en:fc623c45-->
In der Regelungstechnik bezeichnen wir einen kontinuierlichen Zustand mit $\mathbf{x}_t \in \mathbb{R}^d$ und eine Aktion (die Stellgröße) mit $\mathbf{u}_t \in \mathbb{R}^m$. Beim bestärkenden Lernen maximieren wir eine Belohnung $r(\mathbf{x}_t, \mathbf{u}_t)$. Bei der optimalen Regelung minimieren wir die Stufenkosten (engl. stage cost) $c(\mathbf{x}_t, \mathbf{u}_t) = -r(\mathbf{x}_t, \mathbf{u}_t)$. Für ein zeitdiskretes lineares System gilt

<!--en:7c80c399-->
Dabei ist $\mathbf{A} \in \mathbb{R}^{d \times d}$ die Zustandsübergangsmatrix und $\mathbf{B} \in \mathbb{R}^{d \times m}$ die Eingangsmatrix. Das System ist das zeitdiskrete Gegenstück zu den linearen gewöhnlichen Differentialgleichungen in :numref:`sec_mdl-odes-solvers`. Der Zielzustand ist der Ursprung $\mathbf{x} = \mathbf{0}$. Über einen endlichen Horizont von $H$ Schritten minimieren wir die quadratischen Gesamtkosten

<!--en:8e23e83b-->
Dabei sind $\mathbf{Q} \succeq \mathbf{0}$ und $\mathbf{Q}_H \succeq \mathbf{0}$ symmetrische, positiv semidefinite $d \times d$-Matrizen für die Zustandskosten. Für die Stellkosten verwenden wir eine symmetrische, positiv definite Matrix $\mathbf{R} \succ \mathbf{0}$ der Größe $m \times m$. Die Bedingung $\mathbf{R} \succ \mathbf{0}$ stellt sicher, dass ein großer Stellaufwand stets positive Kosten verursacht. Damit hat die Minimierung über $\mathbf{u}_t$ in einem einzelnen Schritt eine eindeutige endliche Lösung.

<!--en:6f1c41ad-->
### Exakte dynamische Programmierung im kontinuierlichen Raum und Riccati-Rekursion

<!--en:be89e91c-->
Mit $V_t(\mathbf{x})$ bezeichnen wir die optimalen Restkosten (engl. cost-to-go) vom Zustand $\mathbf{x}$ im Schritt $t \in \{0, \ldots, H\}$ an. Im Endschritt $t = H$ gibt es keine Stellgröße mehr. Es gilt $V_H(\mathbf{x}) = \mathbf{x}^\top \mathbf{P}_H \mathbf{x}$ mit $\mathbf{P}_H = \mathbf{Q}_H$. Gehen wir von $t+1$ einen Schritt zurück zu $t$, so liefert die Bellman-Optimalitätsgleichung für den endlichen Horizont (:numref:`sec_valueiter`)

<!--en:1458829c-->
Als Induktionsannahme für die Rückwärtsinduktion gelte $V_{t+1}(\mathbf{x}') = {\mathbf{x}'}^\top \mathbf{P}_{t+1} \mathbf{x}'$ mit einer symmetrischen, positiv semidefiniten Matrix $\mathbf{P}_{t+1} \succeq \mathbf{0}$. Setzen wir $\mathbf{x}' = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$ ein und multiplizieren aus, so erhalten wir für die Aktionswertfunktion $Q_t(\mathbf{x}, \mathbf{u})$ in der Klammer eine streng konvexe quadratische Funktion von $\mathbf{u}$:

<!--en:b406a119-->
Wir setzen den Gradienten nach $\mathbf{u}$ gleich null,

<!--en:36c6c7ef-->
und erhalten die eindeutige optimale Stellgröße in geschlossener Form:

<!--en:92801560-->
Setzen wir $\mathbf{u} = -\mathbf{K}_t \mathbf{x}$ wieder in $Q_t(\mathbf{x}, \mathbf{u})$ ein, so sehen wir, dass $V_t(\mathbf{x}) = \mathbf{x}^\top \mathbf{P}_t \mathbf{x}$ rein quadratisch bleibt. Dabei erfüllt $\mathbf{P}_t$ die rückwärts laufende *zeitdiskrete Riccati-Rekursion*:

<!--en:bad37c53-->
Die Rekursion läuft ausgehend von $\mathbf{P}_H = \mathbf{Q}_H$ einmal rückwärts und benötigt dazu keinen Zustandswert. Der spätere Betrieb erfordert dann wenig Rechenaufwand. In jedem Schritt misst der Regler $\mathbf{x}_t$ und multipliziert diesen Zustand mit einer gespeicherten Verstärkung, wie :numref:`fig_rl_lqr_loop` zeigt.

<!--en:5e9dfeb3-->
![LQR im geschlossenen Regelkreis. Der Regler multipliziert den gemessenen Zustand $\mathbf{x}_t$ mit der Verstärkung $-\mathbf{K}_t$. Das System (die *Regelstrecke*) reagiert mit dem nächsten Zustand, den der Regler wiederum misst. Alle Verstärkungen liefert die rückwärts laufende Riccati-Rekursion, bevor der Regelkreis in Betrieb geht.](../img/mdl-rl-lqr-loop.svg)

<!--en:f92e6997-->
Drei Eigenschaften von :eqref:`eq_lqr_gain` und :eqref:`eq_riccati_recursion` sind besonders wichtig:

<!--en:2a5e40b4-->
1. **Kein Fluch der Dimensionalität:** Jedes Bellman-Backup ersetzt die Minimierung über einen unendlichen Aktionsraum $\mathbb{R}^m$ für alle Zustände in $\mathbb{R}^d$ durch $O(d^3 + m^3)$ Matrixoperationen mit $\mathbf{P}_{t+1}$.

<!--en:984fbcb8-->
2. **Lineare Rückführung:** Die optimale Strategie $\pi_t^*(\mathbf{x}) = -\mathbf{K}_t \mathbf{x}$ ist linear im Zustand. Die Verstärkungsmatrix $\mathbf{K}_t \in \mathbb{R}^{m \times d}$ hängt nur von den System- und Kostenmatrizen $(\mathbf{A}, \mathbf{B}, \mathbf{Q}, \mathbf{R})$ ab und ist unabhängig vom Anfangszustand $\mathbf{x}_0$.

<!--en:f5827be8-->
3. **Sicherheitsäquivalenz:** Selbst wenn zum Zustandsübergang unabhängiges Rauschen $\mathbf{w}_t$ mit Mittelwert null und Kovarianz $\boldsymbol{\Sigma}$ hinzukommt ($\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t + \mathbf{w}_t$), verschwindet der gemischte Term $\mathbb{E}[\mathbf{w}_t^\top \mathbf{P}_{t+1}(\mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t)]$. Zur Wertfunktion kommt nur die Konstante $\textrm{tr}(\mathbf{P}_{t+1}\boldsymbol{\Sigma})$ hinzu. Die optimale Rückführmatrix $\mathbf{K}_t$ bleibt unverändert (Übung :exref:`ex_optimal-control-certainty-equivalence`).

<!--en:02354bfa-->
Für einen Horizont $H \to \infty$ konvergiert $\mathbf{P}_t$ unter den üblichen Bedingungen an Steuerbarkeit und Beobachtbarkeit gegen eine stationäre Lösung $\mathbf{P}_\infty$ der *diskreten algebraischen Riccati-Gleichung* (engl. discrete algebraic Riccati equation, DARE). Der zeitinvariante Regler $\mathbf{u}_t = -\mathbf{K}_\infty \mathbf{x}_t$ stabilisiert die Dynamik des geschlossenen Regelkreises $\mathbf{x}_{t+1} = (\mathbf{A} - \mathbf{B}\mathbf{K}_\infty)\mathbf{x}_t$. Jeder Eigenwert von $\mathbf{A} - \mathbf{B}\mathbf{K}_\infty$ liegt dann strikt innerhalb des Einheitskreises, also $\rho(\mathbf{A} - \mathbf{B}\mathbf{K}_\infty) < 1$.

<!--en:b52bbe6e-->
### Der Doppelintegrator

<!--en:48209fb7-->
Wir testen `riccati_lqr` am Fahrzeug aus dem Beispiel zur Zustandserweiterung in :numref:`sec_mdp`. Ein Körper mit Einheitsmasse bewegt sich auf einer Geraden. Sein Zustand $\mathbf{x}_t = (p_t, v_t)^\top$ umfasst Position und Geschwindigkeit, die Stellgröße $u_t$ ist die Beschleunigung und der Zeitschritt beträgt $\Delta t = 0.1$. Nach den Bewegungsgesetzen von Newton gilt $p_{t+1} = p_t + v_t \Delta t + \frac{1}{2} u_t \Delta t^2$ und $v_{t+1} = v_t + u_t \Delta t$. Mit $\mathbf{Q} = \textrm{diag}(1.0, 0.1)$ gewichten wir Abweichungen in Position und Geschwindigkeit. Für den Stellaufwand variieren wir das Gewicht $r_u \in \{0.01, 0.10, 1.00\}$ in $\mathbf{R} = [r_u]$:

<!--en:593574f7-->
Für die drei Gewichtungen des Stellaufwands können wir auch die Positionsverläufe $p_t$ im geschlossenen Regelkreis mit Anfangszustand $\mathbf{x}_0 = (1, 0)^\top$ plotten:

<!--en:deb10f65-->
Betrachten wir die beiden Terme des stationären Rückführgesetzes $u_t^* = -k_p p_t - k_v v_t$. Die dynamische Programmierung hat automatisch einen *Proportional-Differential-Regler* (PD-Regler) hergeleitet. Er wirkt dem Positionsfehler $p_t$ mit der Verstärkung $k_p$ entgegen und dämpft die Geschwindigkeit $v_t$ mit der Verstärkung $k_v$. Ist der Stellaufwand günstig ($R = 0.01$), wählt LQR hohe Verstärkungen $\mathbf{K}_\infty = [7.613, 4.585]$. Die Zustandsnorm klingt mit der Rate $\rho(\mathbf{A} - \mathbf{B}\mathbf{K}_\infty) = 0.761$ pro Schritt ab. Ist der Stellaufwand teuer ($R = 1.00$), beschleunigt der Regler mit den Verstärkungen $\mathbf{K}_\infty = [0.930, 1.395]$ sanfter. Die Abklingrate beträgt dann $\rho = 0.930$ und die kumulierten Kosten $V_0(\mathbf{x}_0) = \mathbf{x}_0^\top \mathbf{P}_\infty \mathbf{x}_0$ sind höher ($P_{00} = 15.00$ gegenüber $6.02$).

<!--en:38ae19ce-->
## Linearisierung eines nichtlinearen Systems um ein Gleichgewicht

<!--en:d141faa3-->
### Analytische Linearisierung von `Pendulum-v1`

<!--en:5ba48bcf-->
Betrachten wir nun einen physikalischen Benchmark, der uns im restlichen Kapitel und in :numref:`chap_deep_rl` begleitet: das inverse Pendel (`Pendulum-v1`). Ein starrer Stab der Masse $m = 1$ und Länge $l = 1$ ist an einem Ende drehbar gelagert. Auf ihn wirken die Schwerkraft mit der Fallbeschleunigung $g = 10$ und ein begrenztes Motordrehmoment $u_t \in [-2, 2]$. Sein kontinuierlicher Zustand ist $\mathbf{x}_t = (\vartheta_t, \dot\vartheta_t)^\top$. Dabei messen wir den Winkel $\vartheta_t$ von der aufrechten Lage aus ($\vartheta = 0$ senkrecht nach oben, $\vartheta = \pi$ senkrecht nach unten). Die Winkelgeschwindigkeit beträgt $\dot\vartheta_t \in [-8, 8]$. In der Waagerechten ($\vartheta = \pi/2$) erreicht das Drehmoment der Schwerkraft sein Maximum von $mgl/2 = 5 > 2$. Der Antrieb ist also *zu schwach*: Er kann den Stab nicht mit einem einzigen Stoß von unten nach oben heben. Für einen Simulationsschritt der Dauer $\Delta t = 0.05\text{ s}$ liefert das semiimplizite Euler-Verfahren den Zustandsübergang $\mathbf{x}_{t+1} = f(\mathbf{x}_t, u_t)$:

<!--en:4ad09bd2-->
Dabei begrenzen wir die Winkelgeschwindigkeit auf $[-8, 8]$. In jedem Schritt fallen die quadratischen Stufenkosten $c(\vartheta_t, \dot\vartheta_t, u_t) = \vartheta_t^2 + 0.1\,\dot\vartheta_t^2 + 0.001\,u_t^2$ an. Hierbei führen wir $\vartheta_t$ periodisch auf das Intervall $[-\pi, \pi]$ zurück. Die Gewichtung entspricht $\mathbf{Q} = \textrm{diag}(1.0, 0.1)$ und $\mathbf{R} = [[0.001]]$. Gleichwertig können wir die Belohnung $r_t = -c(\vartheta_t, \dot\vartheta_t, u_t)$ angeben. Eine Episode im Standardbenchmark dauert $200$ Schritte ($10\text{ s}$). Ein ungeregeltes Pendel erzielt etwa $-1{,}200$ bis $-1{,}500$ Punkte. Ein Regler, der den Stab aufschwingt und aufrecht hält, erreicht über die zufälligen Anfangszustände des Benchmarks im Mittel etwa $-200$ Punkte oder mehr. Hier planen wir direkt mit den bekannten Bewegungsgleichungen :eqref:`eq_pendulum_discrete`. In :numref:`sec_deeprl`, :numref:`sec_sac` und :numref:`sec_model_based` betrachten wir denselben Benchmark erneut. Dort muss der Agent handeln oder ein Dynamikmodell erlernen, ohne die zugrunde liegende Physik zu kennen.

<!--en:8552cd51-->
Die aufrechte Ruhelage $\mathbf{x}^* = (0, 0)^\top$ bei Drehmoment $u^* = 0$ ist ein Fixpunkt, also $f(\mathbf{x}^*, u^*) = \mathbf{x}^*$. In der Nähe eines beliebigen Referenzpaars $(\bar{\mathbf{x}}, \bar{u})$ lässt sich der nichtlineare Übergang durch eine Taylor-Entwicklung erster Ordnung annähern. Das ergibt ein lokales lineares Modell in den Abweichungen $\delta\mathbf{x}_t = \mathbf{x}_t - \bar{\mathbf{x}}$ und $\delta u_t = u_t - \bar{u}$:

<!--en:87d91525-->
Die Ableitungen von :eqref:`eq_pendulum_discrete` nach $(\vartheta_t, \dot\vartheta_t)$ und $u_t$ liefern $\partial\dot\vartheta_{t+1}/\partial\vartheta_t = \frac{3g}{2l}\cos(\vartheta_t)\Delta t$ und $\partial\dot\vartheta_{t+1}/\partial u_t = \frac{3}{ml^2}\Delta t$:

<!--en:6ff9e26c-->
Am aufrechten Fixpunkt hat die Jacobi-Matrix $\mathbf{A}$ des ungeregelten Systems einen instabilen Eigenwert $\lambda_1 = 1.2133 > 1$. Jede kleine Neigung wächst exponentiell um mehr als $21\%$ pro Schritt, wie das Phasenporträt in :numref:`sec_mdl-odes-solvers` zeigt. Lösen wir die Riccati-Rekursion für `Pendulum-v1` mit den Kostenmatrizen $\mathbf{Q} = \textrm{diag}(1, 0.1)$ und $\mathbf{R} = [[0.001]]$, erhalten wir die Rückführverstärkung $\mathbf{K}_\infty = [19.693, 5.263]$. Damit liegen beide Eigenwerte des geschlossenen Regelkreises innerhalb des Einheitskreises ($\lambda_1 = 0.8537, \lambda_2 = 0.2467$).

<!--en:929b3736-->
### Drehmomentsättigung und Einzugsgebiet

<!--en:bf42e44a-->
Bis zu welchem Abstand von der aufrechten Lage kann der lineare Regler das Pendel auffangen, wenn das Motordrehmoment auf $[-2, 2]$ begrenzt ist? Das Drehmoment der Schwerkraft, das das Pendel nach unten zieht, beträgt $\frac{1}{2}mgl\sin\vartheta = 5\sin\vartheta$. Der Motor liefert höchstens $|u| = 2$. Gilt beim Auslenken aus der aufrechten Lage $|\vartheta| > \arcsin(2/5) \approx 0.412\text{ rad}$ ($23.6^\circ$), übersteigt das Drehmoment der Schwerkraft zunächst das maximale Haltemoment des Motors. Wir testen die LQR-Strategie mit begrenztem Drehmoment $u_t = \textrm{clip}(-\mathbf{K}_\infty \mathbf{x}_t, -2, 2)$ am vollständigen nichtlinearen Pendelmodell. Dazu lassen wir das Pendel aus fünf Anfangswinkeln $\vartheta_0$ aus der Ruhe los:

<!--en:e0b29754-->
Innerhalb der statischen Haltegrenze ($\vartheta_0 = 0.20$ und $0.40\text{ rad}$) führt der lineare LQR-Regler das Pendel direkt in die aufrechte Lage ($\vartheta_{200} = 0.000$). Die Kosten sind nahezu null (Erträge $-0.32$ und $-2.97$). Knapp außerhalb dieser Grenze ($\vartheta_0 = 0.50\text{ rad}$) kann selbst das volle negative Drehmoment $u = -2$ den Fall nicht aufhalten. Nach 200 Schritten schwingt der Stab noch immer durch die untere Lage (Ertrag $-770.94$). Jenseits der Grenze hängt das Ergebnis nicht mehr monoton von $\vartheta_0$ ab. Auch bei einem Startwinkel von $0.60\text{ rad}$ fällt der Stab zunächst. Nach einigen weiteren Schwingungen steigt er zufällig langsam genug auf, dass der Regler ihn trotz der Drehmomentgrenze auffängt (Ertrag $-546.37$). Das ist Glück und kein Verdienst der Regelung. Aus der unteren Lage ($\vartheta_0 = \pi$) erzielt der lineare Regler einen Ertrag von $-1471.86$ und erreicht die obere Lage nie. Wegen des begrenzten Motordrehmoments ($u_{\max} = 2 < 5$) müssen wir dem Pendel zum Aufschwingen aus $\vartheta_0 = \pi$ *gezielt Energie zuführen* (engl. energy pumping). Dazu stoßen wir es in eine Richtung an und kehren das Drehmoment um, wenn es durch die untere Lage zurückschwingt. So bauen wir kinetische und potentielle Energie auf. Erst wenn der Stab das Einzugsgebiet der aufrechten Lage erreicht, schalten wir auf deren Stabilisierung um.

<!--en:6782f321-->
## Nichtlineare Trajektorienoptimierung mit dem iterativen LQR-Verfahren

<!--en:eaf66320-->
### Linearisierung entlang einer Trajektorie

<!--en:7af0af04-->
Eine einzige Linearisierung bei $(\mathbf{0}, 0)$ erfasst die nichtlineare Krümmung der Sinusfunktion sin$(\vartheta)$ beim Aufschwingen aus $\vartheta_0 = \pi$ nicht. Das *iterative linear-quadratische Regelungsverfahren* (engl. iterative linear-quadratic regulation, iLQR) :cite:`Li.Todorov.2004` linearisiert dagegen entlang einer ganzen nominellen Trajektorie. Es ist eine Gauß-Newton-Variante der differentiellen dynamischen Programmierung :cite:`Mayne.1966`.

<!--en:38d13582-->
Um die Drehmomentgrenze $u_t \in [-2, 2]$ ohne Knickstellen einzuhalten, parametrisieren wir $u_t = 2\tanh(v_t)$ über eine unbeschränkte Stellvariable $v_t \in \mathbb{R}$. Ausgangspunkt ist eine beliebige nominelle Stellfolge $\bar{v}_{0:H-1}$ und die zugehörige vorwärts simulierte Trajektorie $\bar{\mathbf{x}}_{0:H}$ der tatsächlichen nichtlinearen Dynamik $\mathbf{x}_{t+1} = f(\mathbf{x}_t, v_t)$. Die Begrenzung der Winkelgeschwindigkeit lassen wir weg, da das Pendel sie beim folgenden Aufschwung nie erreicht. Jede iLQR-Iteration umfasst zwei Durchläufe:

<!--en:afcc99c6-->
1. **Riccati-Rückwärtsdurchlauf entlang von $(\bar{\mathbf{x}}_{0:H}, \bar{v}_{0:H-1})$:** In jedem Schritt $t = H-1, \ldots, 0$ berechnen wir die lokalen Jacobi-Matrizen $\mathbf{A}_t = \frac{\partial f}{\partial \mathbf{x}}(\bar{\mathbf{x}}_t, \bar{v}_t)$ und $\mathbf{B}_t = \frac{\partial f}{\partial v}(\bar{\mathbf{x}}_t, \bar{v}_t)$. Für die Änderung des Aktionswerts $\delta Q_t(\delta\mathbf{x}_t, \delta v_t)$ bilden wir eine Taylor-Näherung zweiter Ordnung. Dazu verwenden wir die Gradienten $(\boldsymbol{\ell}_{\mathbf{x},t}, \ell_{v,t})$ und Hesse-Matrizen $(\mathbf{L}_{\mathbf{x}\mathbf{x},t}, \ell_{vv,t})$ der Stufenkosten sowie den Gradienten $\mathbf{v}_{\mathbf{x},t+1}$ und die Hesse-Matrix $\mathbf{V}_{\mathbf{x}\mathbf{x},t+1}$ der Wertfunktion im nächsten Schritt:

   $$
   \begin{aligned}
   \mathbf{q}_{\mathbf{x},t} &= \boldsymbol{\ell}_{\mathbf{x},t} + \mathbf{A}_t^\top \mathbf{v}_{\mathbf{x},t+1}, \qquad q_{v,t} = \ell_{v,t} + \mathbf{B}_t^\top \mathbf{v}_{\mathbf{x},t+1}, \\
   \mathbf{Q}_{\mathbf{x}\mathbf{x},t} &= \mathbf{L}_{\mathbf{x}\mathbf{x},t} + \mathbf{A}_t^\top \mathbf{V}_{\mathbf{x}\mathbf{x},t+1}\mathbf{A}_t, \qquad \mathbf{q}_{v\mathbf{x},t} = \mathbf{B}_t^\top \mathbf{V}_{\mathbf{x}\mathbf{x},t+1}\mathbf{A}_t, \\
   q_{vv,t} &= \ell_{vv,t} + \mu + \mathbf{B}_t^\top \mathbf{V}_{\mathbf{x}\mathbf{x},t+1}\mathbf{B}_t,
   \end{aligned}
   $$

   Dabei ist $\mu > 0$ ein Levenberg-Marquardt-Dämpfungsterm. Die Minimierung über $\delta v_t$ liefert einen Vorsteuerschritt $k_t = -q_{v,t}/q_{vv,t}$ und eine lokale lineare Rückführverstärkung $\mathbf{K}_t = -\mathbf{q}_{v\mathbf{x},t}/q_{vv,t}$.

<!--en:361fe6f1-->
2. **Vorwärtssimulation mit Backtracking-Liniensuche:** Das lineare Modell ist nur in der Nähe von $(\bar{\mathbf{x}}_t, \bar{v}_t)$ genau. Deshalb simulieren wir die *tatsächliche nichtlineare Dynamik* mit der Schrittweite $\alpha \in \{1, 1/2, 1/4, \ldots\}$,

   $$
   \begin{aligned}
   v_t^{\textrm{new}} &= \bar{v}_t + \alpha\, k_t + \mathbf{K}_t(\mathbf{x}_t^{\textrm{new}} - \bar{\mathbf{x}}_t), \\
   \mathbf{x}_{t+1}^{\textrm{new}} &= f(\mathbf{x}_t^{\textrm{new}}, v_t^{\textrm{new}}),
   \end{aligned}
   $$
   :eqlabel:`eq_ilqr_forward`

   und akzeptieren den ersten Wert von $\alpha$, der die tatsächlichen Kosten der nichtlinearen Trajektorie senkt.

<!--en:d6d8b592-->
Ein Detail bei der Wahl der Trajektorienkosten ist aufschlussreich. Um aus $\vartheta_0 = \pi$ Energie aufzubauen, muss das Pendel zunächst auf der Gegenseite bis $\vartheta > \pi$ ausschwingen, bevor es mit Schwung in Richtung $\vartheta = 0$ zurückkehrt. Gewichten wir den quadratischen Kostenterm $\vartheta_t^2$ in jedem frühen Schritt $t$ stark, vermeidet ein lokales Gradientenverfahren jede Bewegung weg von $0$, denn $\vartheta_t^2$ steigt beim Ausholen. Deshalb gewichten wir die Zustandskosten in der Schwungphase niedrig ($q_{\textrm{run}} = 0.02$ für $t < H - 20$) und in der abschließenden Fangphase hoch ($q_T = 100$ für $t \geq H - 20$). So kann iLQR dem Pendel in den frühen Schritten Energie zuführen und den Stab in den letzten Schritten bei $(\vartheta, \dot\vartheta) = (0, 0)$ fixieren:

<!--en:6c2fa944-->
Betrachten wir die Trajektorie, die iLQR in 35 Iterationen aus der unteren Ruhelage $(\pi, 0)$ findet:

<!--en:eae7aee9-->
- **Schritte 0 bis 10 (Ausholen):** Der Motor drückt den Stab nach links ($u_0 = -1.782$) bis $\vartheta_{10} = 2.652\text{ rad}$.

<!--en:cfc48410-->
- **Schritte 10 bis 30 (Energiezufuhr):** Der Motor kehrt sein Drehmoment um ($u_{10} = +0.965 \to u_{20} = +1.681$). Er treibt das Pendel durch die untere Lage bis $\vartheta_{30} = 4.038\text{ rad}$ auf der anderen Seite und baut so potentielle Energie im Schwerefeld auf.

<!--en:dc509d95-->
- **Schritte 30 bis 60 (Aufschwingen und Auffangen):** Der Motor kehrt sein Drehmoment erneut um ($u_{30} = -1.807 \to u_{40} = -1.832$). Der Stab durchläuft die untere Lage mit bis zu $5.7\text{ rad/s}$. Beim Aufstieg über die Waagerechte ($\vartheta_{40} = 1.567\text{ rad}$) bewegt er sich noch mit $\dot\vartheta_{40} = -4.440\text{ rad/s}$. Er erreicht das Einzugsgebiet der aufrechten Lage ($\vartheta_{50} = 0.375\text{ rad}$) und kommt oben sanft zur Ruhe ($\vartheta_{60} = 0.0009\text{ rad}$, $\vartheta_{80} = 0.0001\text{ rad}$, $u_{80} = 0.000$).

<!--en:7e6e2474-->
Dieser Plan ist ein lokales Optimum. Er liefert also nicht unbedingt den besten Aufschwung. Jede iLQR-Iteration verbessert die vorgegebene Trajektorie. Das Ergebnis hängt deshalb von der anfänglichen Stellfolge ab. `ilqr_swingup` startet mit einem kleinen Schub während der ersten 15 Schritte ($\bar{v}_t = 0.2$), gefolgt von einer Stellgröße null. Die Dämpfung $\mu$ ist fest. Der anfängliche Schub ist entscheidend. Beginnen wir bei $\bar{v} = 0$, bringen dieselben 35 Iterationen den Stab oben nicht zur Ruhe. Weiter unten begegnet uns diese Abhängigkeit vom Startwert erneut. Dort entscheidet sie darüber, ob der Regler einen Stoß ausgleichen kann.

<!--en:64648dcc-->
## Modellprädiktive Regelung mit gleitendem Horizont

<!--en:0701b5a3-->
### Steuerung ohne Rückführung, lokale Rückführung entlang der Bahn und Neuplanung im Betrieb

<!--en:a4d41115-->
Eine optimale Trajektorie $(\bar{\mathbf{x}}_{0:H}, \bar{u}_{0:H-1})$ allein liefert noch keinen robusten Regler. Wir können den Plan auf drei Arten umsetzen:

<!--en:474eb915-->
1. **Steuerung ohne Rückführung:** Wir wenden die vorberechnete Folge $\bar{u}_t$ unabhängig vom tatsächlichen Zustand $\mathbf{x}_t$ an. Die aufrechte Lage ist ein instabiler Sattelpunkt. Daher genügt schon eine kleine Störung oder numerische Drift, um die Trajektorie von der geplanten Bahn abzubringen.

<!--en:2f4c7d72-->
2. **Bahnfolgeregelung mit zeitvariantem LQR (TV-LQR):** Wir verwenden die lokalen Rückführmatrizen $\mathbf{K}_t$ aus dem letzten Rückwärtsdurchlauf von iLQR und setzen $v_t = \bar{v}_t + \mathbf{K}_t(\mathbf{x}_t - \bar{\mathbf{x}}_t)$. So wirkt die lineare Rückführung in einer schlauchförmigen Umgebung der nominellen Trajektorie. Im Betrieb entsteht kein zusätzlicher Planungsaufwand. Die Rückführung versagt jedoch, wenn ein starker Stoß den Zustand $\mathbf{x}_t$ aus dieser Umgebung heraus in einen Bereich wirft, in dem das Pendel erneut Schwung holen muss.

<!--en:2f5b9dfe-->
3. **Modellprädiktive Regelung mit gleitendem Horizont (MPC):** In jedem Zeitschritt $t$ beobachten wir den tatsächlichen Zustand $\mathbf{x}_t$. Von $\mathbf{x}_t$ aus lösen wir das Regelungsproblem über einen endlichen Vorschauhorizont neu. Als Startwert verschieben wir den bisherigen Plan um einen Schritt. Wir wenden *nur die erste Stellgröße* $u_{t|t}^*$ an und wiederholen das Verfahren bei $t+1$ :cite:`Rawlings.Mayne.Diehl.2017`.

<!--en:1b060e36-->
Unsere Funktion `mpc_action` legt zwei Einzelheiten des dritten Verfahrens fest. Die erste betrifft die Länge des Planungsfensters. Der verschobene Plan ist einen Schritt kürzer als sein Vorgänger. Das Fenster beginnt daher mit den 80 Schritten des nominellen Plans und schrumpft mit ihm. Sobald noch `H_min` $= 25$ Schritte übrig sind, füllen wir den Plan mit Nullen auf. Von da an gleitet das Fenster bei fester Länge weiter. Die zweite Frage ist, was geschieht, wenn die Initialisierung mit dem bisherigen Plan (engl. warm start) scheitert. Da iLQR ein lokales Verfahren ist, bleibt ein so initialisierter Plan in der Nähe der alten Trajektorie, selbst wenn diese nicht mehr nach oben führt. Endet der erneut optimierte Plan mehr als $0.2\text{ rad}$ von der aufrechten Lage entfernt, verwirft `mpc_action` ihn und plant einen neuen Aufschwung über `H_mpc` $= 60$ Schritte ($3\text{ s}$). Als Startwert dient wieder der anfängliche Schub, den `ilqr_swingup` standardmäßig verwendet.

<!--en:64104767-->
Wir vergleichen die drei Verfahren über $T = 120$ Schritte vom Anfangszustand $\mathbf{x}_0 = (\pi, 0)$ aus. Ein äußerer Stoß ändert die Winkelgeschwindigkeit des Pendels um $\Delta\dot\vartheta \in \{0.0, +0.5, +2.2\}\text{ rad/s}$. Er trifft das Pendel im Schritt $t = 45$, gerade als es über die Waagerechte hinaus zum Einzugsgebiet der aufrechten Lage aufsteigt:

<!--en:a744764b-->
:numref:`fig_rl_optimal_control` zeigt die drei Etappen dieses Abschnitts nebeneinander. Zuerst sehen wir die lokalen LQR-Niveaulinien der Wertfunktion und die Grenze der Drehmomentsättigung um das aufrechte Gleichgewicht. Es folgen die iLQR-Trajektorie mit gezielter Energiezufuhr in der $(\vartheta, \dot\vartheta)$-Phasenebene und die Trajektorien beim Ausregeln des starken Stoßes $\Delta\dot\vartheta = +2.2\text{ rad/s}$ zum Zeitpunkt $t = 45$.

<!--en:f1f9ce22-->
![Optimale Regelung vom lokalen LQR bis zur MPC mit gleitendem Horizont. (a) Niveaulinien der LQR-Restkosten $V_\infty(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}_\infty\mathbf{x}$ um die aufrechte Lage und statische Haltegrenze $|\vartheta| \leq \arcsin(2/5) \approx 0.412\text{ rad}$ (gepunktet). Vom Startwinkel $\vartheta_0 = 0.40\text{ rad}$ aus nähert sich das Pendel dem Ursprung (blau). Bei $\vartheta_0 = 0.50\text{ rad}$ erreicht das Motordrehmoment seine Grenze $|u| \leq 2$, und das Pendel fällt (orange gestrichelt). (b) iLQR-Aufschwung über 80 Schritte in der $(\vartheta, \dot\vartheta)$-Phasenebene. Das Pendel holt von $(\pi, 0)$ bis $\vartheta_{30} = 4.04\text{ rad}$ Schwung und schwingt dann durch die untere Lage bis ins aufrechte Gleichgewicht $(0, 0)$. (c) Auf das Hauptintervall zurückgeführter Winkel $\vartheta_t$ bei einem Stoß $\Delta\dot\vartheta = +2.2\text{ rad/s}$ zum Zeitpunkt $t = 45$ ($2.25\text{ s}$, senkrechte gepunktete Linie). Die Steuerung ohne Rückführung und TV-LQR versagen. MPC mit gleitendem Horizont plant einen neuen Aufschwung und führt das Pendel zurück zu $\vartheta = 0$.](../img/mdl-rl-optimal-control.svg)

<!--en:c494df1b-->
Bei den drei Störstärken im Pendelbenchmark treten die Unterschiede zwischen den drei Verfahren deutlich hervor:

<!--en:827bdd9c-->
- **Ohne Störung ($\Delta\dot\vartheta = 0.0$):** Alle drei Verfahren schwingen das Pendel auf und erzielen über die 120 Schritte Erträge zwischen $-332$ und $-334$. Diese liegen aus zwei Gründen unter dem Richtwert von $-200$ für den Benchmark. Erstens starten wir in der unteren Lage, dem ungünstigsten Zustand, während der Benchmark über zufällige Anfangszustände mittelt. Zweitens minimiert iLQR seine eigenen Trajektorienkosten mit geringer Gewichtung des Zustands während der Schwungphase. Der Ertrag ergibt sich dagegen aus den Belohnungen des Pendelbenchmarks. In den letzten 40 Schritten nach $t=80$ driftet das Pendel ohne Rückführung am instabilen Sattelpunkt bis $\vartheta_{120} = +0.075\text{ rad}$ ab. TV-LQR und MPC halten es bei $\vartheta_{120} = 0.000\text{ rad}$.

<!--en:bf667b07-->
- **Mäßige Störung ($\Delta\dot\vartheta = +0.5\text{ rad/s}$):** Ohne Rückführung kommt das Pendel von der vorberechneten Bahn ab, fällt zurück und endet bei $\vartheta_{120} = -2.939\text{ rad}$ (Ertrag $-502.3$). Sowohl TV-LQR ($\vartheta_{120} = -0.001\text{ rad}$, Ertrag $-333.9$) als auch MPC ($\vartheta_{120} = +0.000\text{ rad}$, Ertrag $-331.7$) gleichen den Stoß aus und halten den Stab aufrecht.

<!--en:9ba62eeb-->
- **Starke Störung ($\Delta\dot\vartheta = +2.2\text{ rad/s}$):** Der Stoß stoppt den Aufstieg und lässt das Pendel zurückfallen. Bei der linearen Rückführung um die alte nominelle Trajektorie erreicht das Drehmoment die Sättigungsgrenze. TV-LQR versagt ($\vartheta_{120} = +1.959\text{ rad}$, Ertrag $-797.9$), denn das Pendel müsste erneut Schwung holen. In der Umgebung der alten Trajektorie, in der die lineare Näherung gilt, ist das nicht möglich. MPC bringt das Pendel dank der Regel zur Neuinitialisierung wieder ins aufrechte Gleichgewicht ($\vartheta_{120} = 0.000\text{ rad}$, Ertrag $-558.2$). Im Schritt des Stoßes verfehlt der mit dem alten Plan initialisierte Aufschwung die obere Lage. Deshalb plant `mpc_action` vom Zustand nach der Störung aus einen neuen Aufschwung und folgt diesem Plan. Eine Neuplanung allein auf Grundlage des verschobenen Plans würde nicht ausreichen, da sie in der Nähe der alten Trajektorie bleibt.

<!--en:d06401f6-->
### Von der klassischen Regelung zum bestärkenden Lernen

<!--en:ef0d2db7-->
:numref:`tab_control_vs_rl` stellt die Begriffe der klassischen optimalen Regelung denen des bestärkenden Lernens gegenüber. Die Übersicht zeigt, wie die folgenden Abschnitte dieses Kapitels und :numref:`chap_deep_rl` die einzelnen Annahmen von LQR und iLQR lockern:

<!--en:de521a8c-->
:Zusammenhang zwischen klassischer optimaler Regelung und bestärkendem Lernen.

<!--en:b095358b-->
| Begriff | Klassische optimale Regelung | Bestärkendes Lernen |

<!--en:feae8bf9-->
| Variablen | Zustand $\mathbf{x}_t \in \mathbb{R}^d$, Stellgröße $\mathbf{u}_t \in \mathbb{R}^m$ | Zustand $s_t \in \mathcal{S}$, Aktion $a_t \in \mathcal{A}$ |

<!--en:a470ed08-->
| Ziel | Kumulierte Kosten $\sum_t c(\mathbf{x}_t, \mathbf{u}_t)$ minimieren | Erwarteten diskontierten Ertrag $\mathbb{E}[\sum_t \gamma^t r(s_t, a_t)]$ maximieren |

<!--en:97fdce49-->
| Wertfunktion | Restkosten $V_t(\mathbf{x}) = \mathbf{x}^\top \mathbf{P}_t \mathbf{x}$ (LQR) | Zustandswert $V^\pi(s)$ oder Aktionswert $Q^\pi(s, a)$ |

<!--en:23e91ca9-->
| Bellman-Backup | Riccati-Rekursion :eqref:`eq_riccati_recursion` | Bellman-Erwartungswertoperator oder Bellman-Optimalitätsoperator (:numref:`sec_valueiter`) |

<!--en:34993305-->
| Bekannte nichtlineare Dynamik $f$ | Gradientenbasierte Trajektorienoptimierung (iLQR oder DDP) mit MPC | Baumsuche oder stichprobenbasierte MPC (:numref:`sec_model_based`) |

<!--en:555994e6-->
| Unbekannte Dynamik, erlerntes Modell $\hat{f}_\phi$ | Systemidentifikation und sicherheitsäquivalente MPC | Modellbasiertes bestärkendes Lernen mit Random Shooting oder CEM (:numref:`sec_model_based`) |

<!--en:a912dbc4-->
| Unbekannte Dynamik, kein Modell | Modellfreie adaptive Regelung | Policy-Gradient (:numref:`sec_policygradient`, :numref:`sec_deeprl`), PPO (:numref:`sec_ppo`), DQN (:numref:`sec_dqn`), SAC (:numref:`sec_sac`) |

<!--en:a9ae8950-->
Ist die Dynamik $f$ unbekannt und müssen wir handeln, ohne ein Übergangsmodell anzupassen, fehlen uns die Jacobi-Matrizen $\mathbf{A}_t = \partial f/\partial \mathbf{x}$ und $\mathbf{B}_t = \partial f/\partial \mathbf{u}$ für den Riccati-Rückwärtsdurchlauf. :numref:`sec_policygradient` zeigt, wie eine stochastische Strategie $\pi_\theta(a \mid s)$ ohne diese unbekannten Ableitungen auskommt. Dazu differenzieren wir die Log-Wahrscheinlichkeit der Aktionen, die wir aus der Strategie selbst gezogen haben.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:74c7b621-->
Für lineare Dynamik $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t$ und quadratische Kosten $\mathbf{x}^\top\mathbf{Q}\mathbf{x} + \mathbf{u}^\top\mathbf{R}\mathbf{u}$ liefert die rückwärts laufende dynamische Programmierung nach Bellman im kontinuierlichen Raum eine exakte geschlossene Lösung. Die Restkosten sind quadratisch, $V_t(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}_t\mathbf{x}$. Dabei erfüllt $\mathbf{P}_t$ die rückwärts laufende Riccati-Rekursion. Die optimale Strategie ist der lineare Regler mit Zustandsrückführung $\mathbf{u}_t^* = -\mathbf{K}_t\mathbf{x}_t$ (LQR). Linearisieren wir ein nichtlineares System mithilfe seiner analytischen Jacobi-Matrizen um einen instabilen Fixpunkt, so stabilisiert der Regler Zustände innerhalb des durch den Aktor begrenzten Einzugsgebiets. Außerhalb dieses Gebiets versagt er, wenn der begrenzte Antrieb zum Aufschwingen über mehrere Schwingungen Energie zuführen muss. Das iterative LQR-Verfahren (iLQR) linearisiert die Dynamik wiederholt entlang einer nominellen Trajektorie und nähert dort die Kosten quadratisch an. Mit Riccati-Rückwärtsdurchläufen und nichtlinearen Vorwärtssimulationen findet es Trajektorien, die dem System in mehreren Phasen Energie zuführen. Eine Steuerung ohne Rückführung versagt bei Störungen. Eine zeitvariante LQR-Rückführung entlang einer festen Trajektorie gleicht nur kleine Störungen aus. Die modellprädiktive Regelung (MPC) mit gleitendem Horizont löst das Problem mit endlichem Horizont in jedem Schritt vom aktuellen Zustand aus neu. So bringt sie das System auch dann ins Gleichgewicht zurück, wenn eine große Störung eine völlig neue Trajektorie erfordert.

<!--en:2294243a-->
## Übungen

<!--en:dc95d141-->
1. **Quadratische Ergänzung im Riccati-Backup.** Setzen Sie $\mathbf{S}_{t+1} = \mathbf{R} + \mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{B}$ und $\mathbf{K}_t = \mathbf{S}_{t+1}^{-1}\mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{A}$.
    1. Zeigen Sie durch Ausmultiplizieren, dass für alle $\mathbf{x} \in \mathbb{R}^d$ und $\mathbf{u} \in \mathbb{R}^m$ gilt:

       $$Q_t(\mathbf{x}, \mathbf{u}) = \mathbf{x}^\top\mathbf{P}_t\mathbf{x} + (\mathbf{u} + \mathbf{K}_t\mathbf{x})^\top \mathbf{S}_{t+1}(\mathbf{u} + \mathbf{K}_t\mathbf{x}),$$

       Dabei ist $\mathbf{P}_t$ durch :eqref:`eq_riccati_recursion` gegeben.
    1. Folgern Sie aus $\mathbf{R} \succ \mathbf{0}$ und $\mathbf{P}_{t+1} \succeq \mathbf{0}$, dass $\mathbf{S}_{t+1} \succ \mathbf{0}$ gilt, dass $\mathbf{u}_t^* = -\mathbf{K}_t\mathbf{x}$ der eindeutige globale Minimierer von $Q_t(\mathbf{x}, \mathbf{u})$ ist und dass $\mathbf{P}_t \succeq \mathbf{0}$ gilt.
    1. Überprüfen Sie die stabilisierte Form nach Joseph

       $$\mathbf{P}_t = \mathbf{Q} + \mathbf{K}_t^\top\mathbf{R}\mathbf{K}_t + (\mathbf{A} - \mathbf{B}\mathbf{K}_t)^\top\mathbf{P}_{t+1}(\mathbf{A} - \mathbf{B}\mathbf{K}_t),$$

       und erklären Sie, warum jeder Term auf der rechten Seite offensichtlich positiv semidefinit ist, selbst wenn Sie $\mathbf{K}_t$ durch eine beliebige Rückführmatrix $\mathbf{K}$ ersetzen.

<!--en:feed7733-->
1. **Sicherheitsäquivalenz bei additivem gaußschem Rauschen.** Unabhängige Rauschvektoren $\mathbf{w}_0, \ldots, \mathbf{w}_{H-1}$ mit Mittelwert null und Kovarianz $\mathbb{E}[\mathbf{w}_t\mathbf{w}_t^\top] = \boldsymbol{\Sigma}$ stören die lineare Dynamik :eqref:`eq_lqr_dynamics`, sodass $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t + \mathbf{w}_t$ gilt.
    1. Beweisen Sie durch Rückwärtsinduktion, dass die optimalen Restkosten $V_t(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}_t\mathbf{x} + c_t$ betragen. Dabei erfüllt $\mathbf{P}_t$ genau dieselbe deterministische Riccati-Rekursion :eqref:`eq_riccati_recursion`, und es gilt $c_t = \sum_{k=t+1}^{H} \textrm{tr}(\mathbf{P}_k \boldsymbol{\Sigma})$.
    1. Folgern Sie, dass die optimale Strategie mit Rückführung $\mathbf{u}_t^* = -\mathbf{K}_t\mathbf{x}_t$ unabhängig von der Rauschkovarianz $\boldsymbol{\Sigma}$ ist. An welcher Stelle bricht die Herleitung der Sicherheitsäquivalenz zusammen, wenn das Rauschen multiplikativ eingeht, also $\mathbf{x}_{t+1} = (\mathbf{A} + \tilde{\mathbf{A}}_t)\mathbf{x}_t + (\mathbf{B} + \tilde{\mathbf{B}}_t)\mathbf{u}_t$ gilt?
   :exlabel:`ex_optimal-control-certainty-equivalence`

<!--en:36b5f72e-->
1. [code] **Steuerbarkeit und Grenzübergang zum unendlichen Horizont.** Ein lineares System $(\mathbf{A}, \mathbf{B})$ mit $d$ Zustandsgrößen heißt *steuerbar*, wenn die Steuerbarkeitsmatrix $\mathcal{C} = [\mathbf{B}, \mathbf{A}\mathbf{B}, \ldots, \mathbf{A}^{d-1}\mathbf{B}]$ vollen Zeilenrang $d$ hat.
    1. Zeigen Sie, dass das System vom Anfangszustand $\mathbf{x}_0 = \mathbf{0}$ aus nach $d$ Schritten den Zustand $\mathbf{x}_d = \mathcal{C}\,[\mathbf{u}_{d-1}^\top, \mathbf{u}_{d-2}^\top, \ldots, \mathbf{u}_0^\top]^\top$ erreicht. Folgern Sie, dass voller Zeilenrang von $\mathcal{C}$ notwendig und hinreichend ist, um jeden Zielzustand in $\mathbb{R}^d$ in $d$ Schritten zu erreichen.
    1. Berechnen Sie $\textrm{rank}(\mathcal{C})$ und $\det(\mathcal{C})$ für den Doppelintegrator $(\mathbf{A}_{\textrm{di}}, \mathbf{B}_{\textrm{di}})$ und für das um die aufrechte Lage linearisierte Pendel $(\mathbf{A}_p, \mathbf{B}_p)$.
    1. Setzen Sie den Eingang für die erste Koordinate auf null. Verwenden Sie dazu $\mathbf{B}_{\textrm{un}} = [0, 0]^\top$ als angegebenen Eingangsvektor für das entkoppelte instabile System $\mathbf{A} = \textrm{diag}(1.2, 0.8), \mathbf{B} = [0, 1]^\top$ und führen Sie `riccati_lqr` für $H \in \{10, 50, 100\}$ aus. Welcher Eintrag von $\mathbf{P}_0$ divergiert für $H \to \infty$ und warum?

<!--en:62e380fc-->
1. [code] **Energiebasierte Regelung und Trajektorienoptimierung.** In der klassischen nichtlinearen Regelung schwingen wir ein Pendel häufig auf, indem wir seine gesamte mechanische Energie $E(\vartheta, \dot\vartheta) = \frac{1}{6}ml^2\dot\vartheta^2 + \frac{1}{2}mgl\cos\vartheta$ regeln. Sie beträgt $E^* = \frac{1}{2}mgl = 5$ in der aufrechten Ruhelage $(\vartheta=0, \dot\vartheta=0)$ und $-5$ in der unteren Ruhelage $(\vartheta=\pi, \dot\vartheta=0)$ :cite:`Tedrake.2023`.
    1. Zeigen Sie, dass die zeitliche Änderung der Energie entlang zeitkontinuierlicher, reibungsfreier Trajektorien ($\ddot\vartheta = \frac{3g}{2l}\sin\vartheta + \frac{3}{ml^2}u$) durch $\dot E = u\dot\vartheta$ gegeben ist. Folgern Sie, dass das Drehmoment $u = -k_E (E - E^*)\dot\vartheta$ für $k_E > 0$ den quadrierten Energiefehler $(E - E^*)^2$ monoton verringert, wann immer $\dot\vartheta \neq 0$ gilt.
    1. Implementieren Sie auf der Grundlage von `pendulum_step` einen hybriden Regler. Er soll mit begrenztem Drehmoment Energie zuführen, gemäß $u = \textrm{clip}(-k_E (E - E^*)\dot\vartheta, -2, 2)$ (mit einem kurzen Anfangsstoß $u_0 = 2$), solange $|\textrm{wrap}(\vartheta)| > 0.35$ gilt. Sobald $|\textrm{wrap}(\vartheta)| \leq 0.35$ gilt, soll er auf den LQR-Regler $u = \textrm{clip}(-\mathbf{K}_\infty \mathbf{x}, -2, 2)$ für die aufrechte Lage umschalten. Vergleichen Sie seinen Ertrag über 200 Schritte aus $(\pi, 0)$ mit dem Ertrag von iLQR.

<!--en:3ae20b74-->
1. [code] **Horizontlänge und Endkosten bei MPC.** `mpc_action` optimiert den verschobenen Plan erneut über mindestens `H_min` $= 25$ Schritte. Verfehlt dieser Plan die obere Lage, plant die Funktion einen neuen Aufschwung über `H_mpc` $= 60$ Schritte ($3\text{ s}$).
    1. Wiederholen Sie den Versuch mit $\Delta\dot\vartheta = +2.2$ aus diesem Abschnitt für `H_mpc` $\in \{15, 25, 40, 60\}$. Bei welchen Horizonten bringt MPC das Pendel wieder nach oben? Vergleichen Sie diese Horizonte mit der Anzahl der Schritte, die der nominelle Aufschwung bis zur oberen Lage braucht. Erklären Sie, warum der Planer bei kurzem Horizont in der Nähe der unteren Lage stecken bleibt.
    1. Ersetzen Sie die Endkostenmatrix $q_T \mathbf{Q}_{\textrm{base}}$ in `ilqr_swingup` durch die DARE-Matrix $\mathbf{P}_\infty$ für den unendlichen Horizont, die `riccati_lqr` am aufrechten Gleichgewicht berechnet. Betrachten Sie Anfangszustände, die bereits im Einzugsgebiet der aufrechten Lage liegen ($|\vartheta_0| \leq 0.35$). Zeigen Sie, dass ein LQR-Regler mit gleitendem Horizont und Endkosten $\mathbf{x}_H^\top \mathbf{P}_\infty \mathbf{x}_H$ die *exakte* optimale Stellgröße $\mathbf{u}_t^* = -\mathbf{K}_\infty \mathbf{x}_t$ für den unendlichen Horizont liefert, selbst wenn der Vorschauhorizont nur einen Schritt umfasst, also $H = 1$ gilt.

<!--en:b2493278-->
1. **Kozustände nach Pontryagin und Gradienten der Wertfunktion.** Gegeben seien ein glattes deterministisches System $\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$ und die Kosten $J = \sum_{t=0}^{H-1} c(\mathbf{x}_t, \mathbf{u}_t) + c_H(\mathbf{x}_H)$. Führen Sie Lagrange-Multiplikatoren (auch *Kozustände* oder *adjungierte Variablen* genannt) $\boldsymbol{\lambda}_1, \ldots, \boldsymbol{\lambda}_H \in \mathbb{R}^d$ für die Nebenbedingungen $\mathbf{x}_{t+1} - f(\mathbf{x}_t, \mathbf{u}_t) = \mathbf{0}$ ein.
    1. Differenzieren Sie die Lagrange-Funktion $\mathcal{L} = J + \sum_{t=0}^{H-1} \boldsymbol{\lambda}_{t+1}^\top (f(\mathbf{x}_t, \mathbf{u}_t) - \mathbf{x}_{t+1})$ nach $\mathbf{x}_t$ und $\mathbf{u}_t$. Leiten Sie so die rückwärts laufende adjungierte Gleichung

       $$\boldsymbol{\lambda}_H = \nabla_{\mathbf{x}} c_H(\mathbf{x}_H), \qquad \boldsymbol{\lambda}_t = \nabla_{\mathbf{x}} c(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{A}_t^\top \boldsymbol{\lambda}_{t+1},$$

       und den Gradienten $\nabla_{\mathbf{u}_t} J = \nabla_{\mathbf{u}} c(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{B}_t^\top \boldsymbol{\lambda}_{t+1}$ her :cite:`Pontryagin.Boltyanskii.Gamkrelidze.ea.1962`.
    1. Vergleichen Sie die rückwärts laufende adjungierte Rekursion mit den Größen erster Ordnung $\mathbf{v}_{\mathbf{x},t}$ und $q_{v,t}$ im Rückwärtsdurchlauf von `ilqr_swingup` (und mit der Backpropagation durch die Zeit in :numref:`sec_bptt`). Welche zusätzliche Information liefert die Rekursion für die Hesse-Matrix $\mathbf{V}_{\mathbf{x}\mathbf{x},t}$ bei iLQR gegenüber der Backpropagation durch die Zeit erster Ordnung?

<!--en:5c578ac8-->
[Diskussion](https://d2l.discourse.group/t/36887)

<!--en:964cfee6-->
Optimale Regelung, LQR und modellprädiktive Regelung<br>**linear-quadratische Regelung · rückwärts laufende Riccati-Rekursion · lokale Linearisierung · iteratives LQR-Verfahren · MPC mit gleitendem Horizont**

<!--en:0fe41d6b-->
::: {.slide title="Kontinuierliche Zustandsräume und Bellmans Fluch der Dimensionalität"}

<!--en:b0445376-->
[Motivation und Grundidee]{.kicker}

<!--en:519a32a5-->
- **Fluch der Dimensionalität:** Bei $d$ Zustandskoordinaten mit je $100$ Intervallen braucht das Gitter $100^d$ Zellen ($10^{28}$ für einen Roboterarm mit 7 Gelenken und $d = 14$).

<!--en:5f7aa81b-->
- **Kalmans algebraische Einsicht (1960):** Bei linearer Dynamik und quadratischen Stufenkosten bleibt die Wertfunktion exakt quadratisch. Für $V_t(\mathbf{x}) = \mathbf{x}^\top\mathbf{P}_t\mathbf{x}$ genügen $d(d+1)/2$ unabhängige Matrixeinträge.

<!--en:becc2d74-->
::: {.slide title="Das Regelungsproblem für den linear-quadratischen Regler (LQR)"}

<!--en:e82de56e-->
[Lineare Dynamik und quadratische Kosten]{.kicker}

<!--en:b1606d16-->
- **Dynamik und Kosten:** $\mathbf{A}, \mathbf{B}$ bestimmen den Zustandsverlauf. $\mathbf{Q} \succeq \mathbf{0}$ gewichtet den Zustandsfehler, und mit $\mathbf{R} \succ \mathbf{0}$ verursacht jeder Stellaufwand ungleich null positive Kosten.

<!--en:7be432e0-->
::: {.slide title="Exakte dynamische Programmierung im kontinuierlichen Raum"}

<!--en:3b1a2852-->
[Ausmultiplizieren des Bellman-Backups]{.kicker}

<!--en:f853f082-->
Die Rückwärtsinduktion beginnt bei $t = H$ ($\mathbf{P}_H = \mathbf{Q}_H$). Unter der Annahme quadratischer Restkosten im Schritt $t+1$, also $V_{t+1}(\mathbf{x}') = {\mathbf{x}'}^\top\mathbf{P}_{t+1}\mathbf{x}'$, lautet die Bellman-Optimalitätsgleichung:

<!--en:100e9be2-->
- **Streng konvexe quadratische Funktion von $\mathbf{u}$:** Ausmultiplizieren der Klammer für den Aktionswert $Q_t(\mathbf{x}, \mathbf{u})$ liefert quadratische Terme in $\mathbf{x}$, gemischte Terme $2\mathbf{u}^\top\mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{A}\mathbf{x}$ und einen quadratischen Term für die Stellgröße $\mathbf{u}^\top(\mathbf{R} + \mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{B})\mathbf{u}$.

<!--en:7d5edaaf-->
- **Optimalität erster Ordnung:** Die Bedingung $\nabla_{\mathbf{u}} Q_t(\mathbf{x}, \mathbf{u}) = 2(\mathbf{R} + \mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{B})\mathbf{u} + 2\mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{A}\mathbf{x} = \mathbf{0}$ liefert die analytische Lösung des Backups in geschlossener Form.

<!--en:4f00e5de-->
::: {.slide title="Rückwärts laufende Riccati-Rekursion und Verstärkungsmatrix"}

<!--en:bcdef964-->
[Lineare Zustandsrückführung in geschlossener Form]{.kicker}

<!--en:757c5394-->
- **Drei strukturelle Eigenschaften:** (1) Aufwand $O(d^3 + m^3)$ pro Schritt. (2) $\mathbf{K}_t$ ist unabhängig vom Anfangszustand $\mathbf{x}_0$. (3) Additives gaußsches Rauschen $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ lässt $\mathbf{K}_t$ unverändert (*Sicherheitsäquivalenz*).

<!--en:0e33df82-->
::: {.slide title="Doppelintegrator: LQR liefert einen PD-Regler" layout="tight"}

<!--en:5bb47a6b-->
[Stationäre Rückführung bei unendlichem Horizont]{.kicker}

<!--en:01b93a7d-->
::: {.slide title="Doppelintegrator: LQR liefert einen PD-Regler (Fortsetzung)" layout="tight"}

<!--en:5bb47a6b-2-->
[Stationäre Rückführung bei unendlichem Horizont]{.kicker}

<!--en:1453b7fd-->
Für ein Fahrzeug mit einer Bewegungsrichtung, Zustand $\mathbf{x}_t = (p_t, v_t)^\top$ und Beschleunigung $u_t$ liefert `riccati_lqr` automatisch einen Proportional-Differential-Regler (PD-Regler) $u_t^* = -k_p p_t - k_v v_t$:

<!--en:20e15f8f-->
- **Asymptotische Stabilität:** Für $H \to \infty$ gilt $\mathbf{P}_t \to \mathbf{P}_\infty$ (Lösung der diskreten algebraischen Riccati-Gleichung). Damit liegen alle Eigenwerte von $\mathbf{A} - \mathbf{B}\mathbf{K}_\infty$ strikt innerhalb des Einheitskreises ($\rho < 1$).

<!--en:c047ff2f-->
::: {.slide title="Kosten des Stellaufwands und Abklinggeschwindigkeit"}

<!--en:4db807b3-->
[Trajektorien des Doppelintegrators im geschlossenen Regelkreis]{.kicker}

<!--en:338b2db6-->
Die Verläufe für verschiedene Stellgewichte $R \in \{0.01, 0.10, 1.00\}$ beim Anfangszustand $\mathbf{x}_0 = (1, 0)^\top$ zeigen, wie $\mathbf{R}$ Stellaufwand und Konvergenzgeschwindigkeit gegeneinander abwägt:

<!--en:b0cb2959-->
- **Günstiger Stellaufwand ($R = 0.01$):** Hohe Verstärkungen $\mathbf{K}_\infty = [7.613, 4.585]$ bauen den Zustandsfehler schnell ab ($\rho = 0.761$, Wert $P_{00} = 6.02$).

<!--en:071055c2-->
- **Teurer Stellaufwand ($R = 1.00$):** Kleine Verstärkungen $\mathbf{K}_\infty = [0.930, 1.395]$ bremsen langsam ab ($\rho = 0.930$, Wert $P_{00} = 15.00$).

<!--en:33c15fab-->
::: {.slide title="Pendulum-v1 als Benchmark für kontinuierliche Regelung"}

<!--en:0a6dd942-->
[Benchmark für nichtlineare Regelung]{.kicker}

<!--en:5981ea43-->
- **Stark begrenzter Antrieb:** Das Motordrehmoment ist auf $u_t \in [-2, 2]$ begrenzt. In der Waagerechten ($\vartheta = \pi/2$) erreicht das Drehmoment der Schwerkraft sein Maximum von $\frac{1}{2}mgl = 5 > 2$.

<!--en:347d6790-->
- **Stufenkosten und Richtwert für das Balancieren:** $c_t = \vartheta_t^2 + 0.1\,\dot{\vartheta}_t^2 + 0.001\,u_t^2$. Zufällige Aktionen erzielen $\approx -1{,}200$ Punkte. Aufschwingen und Balancieren erreichen bei zufälligen Anfangszuständen im Mittel **etwa $-200$ Punkte oder mehr**.

<!--en:58f3bf72-->
::: {.slide title="Linearisierung nichtlinearer Dynamik um ein Gleichgewicht"}

<!--en:97b94b3f-->
[Taylor-Näherung erster Ordnung]{.kicker}

<!--en:87cbcc76-->
Für ein glattes nichtlineares System $\mathbf{x}_{t+1} = f(\mathbf{x}_t, u_t)$ in der Nähe eines Fixpunkts $f(\mathbf{x}^*, u^*) = \mathbf{x}^*$ liefert eine Taylor-Entwicklung erster Ordnung ein lokales lineares Modell in $(\delta\mathbf{x}_t, \delta u_t)$:

<!--en:550f9f3d-->
- **Aufrechtes Gleichgewicht von `Pendulum-v1`:** Bei $(\vartheta^*, \dot{\vartheta}^*) = (0, 0)$ und $u^* = 0$ liefert $\sin\vartheta_t \approx \vartheta_t$ das exakte linearisierte Modell $(\mathbf{A}_p, \mathbf{B}_p)$ um den instabilen oberen Sattelpunkt.

<!--en:f8ac0d72-->
::: {.slide title="Stabilisierung des aufrechten Pendels mit lokalem LQR"}

<!--en:d8688f4d-->
[Verschiebung instabiler Eigenwerte ins Innere des Einheitskreises]{.kicker}

<!--en:c7255ef2-->
Die exakten Jacobi-Matrizen $(\mathbf{A}_p, \mathbf{B}_p)$ von `Pendulum-v1` am aufrechten Gleichgewicht $\vartheta = 0$ liefern mit der Riccati-Rekursion einen Regler, der den Sattelpunkt stabilisiert:

<!--en:8e266f8d-->
- **Ungeregelt instabil, geregelt stabil:** Die Jacobi-Matrix des ungeregelten Systems hat einen instabilen Eigenwert $\lambda_1 = 1.2133 > 1$ (die Neigung wächst um $21\%$ pro Schritt). Mit der LQR-Verstärkung $\mathbf{K}_\infty = [19.693, 5.263]$ liegen beide Eigenwerte innerhalb des Einheitskreises ($0.8537$ und $0.2467$).

<!--en:36d0f5ab-->
::: {.slide title="Drehmomentsättigung als Grenze der lokalen LQR-Regelung"}

<!--en:8da34ba4-->
[Begrenzter Antrieb und Einzugsgebiet]{.kicker}

<!--en:94fe11da-->
Die Schwerkraft zieht das Pendel mit dem Drehmoment $\frac{1}{2}mgl\sin\vartheta = 5\sin\vartheta$ nach unten. Das Motordrehmoment ist auf $|u_t| \leq 2$ begrenzt. Daraus ergibt sich die statische Haltegrenze $|\vartheta| = \arcsin(2/5) \approx 0.412\text{ rad}$ ($23.6^\circ$):

<!--en:0c650c16-->
- **Scharfe Grenze:** Bei einem Startwinkel von $\vartheta_0 = 0.20$ oder $0.40\text{ rad}$ fängt der lineare LQR-Regler den Stab trotz Drehmomentgrenze auf und hält ihn bei $\vartheta_{200} = 0.000\text{ rad}$.

<!--en:ce823331-->
- **Versagen außerhalb des Einzugsgebiets:** Bei $\vartheta_0 = 0.50\text{ rad}$ ($5\sin 0.50 = 2.40 > 2$) oder beim Start von unten ($\vartheta_0 = \pi$) reicht das volle Drehmoment $u = -2$ gegen die Schwerkraft nicht aus. Zum Aufschwingen ist **gezielte Energiezufuhr** über mehrere Schwingungen nötig. Auch bei $0.60\text{ rad}$ fällt der Stab und wird erst nach mehreren Schwingungen durch Zufall aufgefangen.

<!--en:b2e8a7f7-->
::: {.slide title="Iteratives LQR-Verfahren (iLQR): Linearisierung entlang der Trajektorie"}

<!--en:99799131-->
[Trajektorienoptimierung nach Gauß-Newton]{.kicker}

<!--en:fce7b53d-->
- **Glatte Drehmomentbegrenzung:** $u_t = 2\tanh(v_t)$ hält die Schranke $u_t \in (-2, 2)$ auf differenzierbare Weise ein.

<!--en:ef69c0d5-->
- **Riccati-Rekursion rückwärts, Liniensuche vorwärts:** Zeitvariante Jacobi-Matrizen $(\mathbf{A}_t, \mathbf{B}_t)$ berechnen, rückwärts nach $(k_t, \mathbf{K}_t)$ auflösen und die tatsächliche nichtlineare Dynamik mit schrittweise verkleinertem $\alpha$ simulieren.

<!--en:fe085d34-->
::: {.slide title="iLQR findet eine mehrphasige Trajektorie zur Energiezufuhr" layout="tight"}

<!--en:87b66eb8-->
[Nichtlineares Aufschwingen aus der unteren Ruhelage]{.kicker}

<!--en:ae54042f-->
In 35 Iterationen aus der hängenden Ruhelage $(\pi, 0)$ findet iLQR eine Trajektorie mit drei Phasen, die das Pendel in $H = 80$ Schritten ($4.0\text{ s}$) aufschwingt und auffängt:

<!--en:f71d0087-->
- **Schritte 0–10 (Ausholen):** Schub nach links ($u_0 = -1.78$) bis $\vartheta_{10} = 2.65\text{ rad}$

<!--en:70870944-->
- **Schritte 10–30 (Energiezufuhr):** Umkehr des Drehmoments ($u_{20} = +1.68$), Schwung durch die untere Lage bis $\vartheta_{30} = 4.04\text{ rad}$

<!--en:40ab622b-->
- **Schritte 30–80 (Aufschwingen und Auffangen):** Erneute Umkehr ($u_{40} = -1.83$), beim Aufstieg noch eine Winkelgeschwindigkeit von $\dot{\vartheta}_{40} = -4.44\text{ rad/s}$, dann Abbremsen bis zur Ruhe bei $\vartheta_{80} = 0.0001\text{ rad}$

<!--en:db3d3971-->
::: {.slide title="iLQR findet eine mehrphasige Trajektorie zur Energiezufuhr (Fortsetzung)"}

<!--en:87b66eb8-2-->
[Nichtlineares Aufschwingen aus der unteren Ruhelage]{.kicker}

<!--en:53bae74d-->
Die Aufschwungtrajektorie in Abständen von 10 Schritten, von der hängenden Ausgangslage bis zum aufgefangenen, aufrechten Stab:

<!--en:f5f1f813-->
::: {.slide title="Steuerung, TV-LQR und MPC mit gleitendem Horizont"}

<!--en:845a98a8-->
[Vom Trajektorienplan zur Regelung mit Rückführung]{.kicker}

<!--en:f72870f8-->
1. **Steuerung ohne Rückführung ($u_t = \bar{u}_t$):** Die vorberechneten Drehmomente wirken unabhängig vom Zustand $\mathbf{x}_t$. Am instabilen oberen Sattelpunkt bringt jeder Stoß den Stab zu Fall.

<!--en:90d2aeaa-->
2. **TV-LQR ($v_t = \bar{v}_t + \mathbf{K}_t(\mathbf{x}_t - \bar{\mathbf{x}}_t)$):** Lokale Rückführung stabilisiert die Bewegung in einer Umgebung der Bahn ohne zusätzlichen Planungsaufwand im Betrieb. Außerhalb dieser Umgebung versagt sie.

<!--en:11787734-->
3. **MPC mit gleitendem Horizont:** iLQR in jedem Schritt vom gemessenen Zustand $\mathbf{x}_t$ aus neu lösen, initialisiert mit dem verschobenen Plan. $u_{t|t}^*$ anwenden und bei $t+1$ neu planen. Verfehlt der so initialisierte Plan die obere Lage, einen neuen Aufschwung über 60 Schritte planen.

<!--en:14f9d41a-->
::: {.slide title="Ausregeln eines Stoßes bei t = 45" layout="tight"}

<!--en:16d9137a-->
[Steuerung ohne Rückführung, TV-LQR und MPC im Test]{.kicker}

<!--en:44b82a94-->
Ein äußerer Stoß ändert die Winkelgeschwindigkeit um $\Delta\dot{\vartheta} \in \{0.0, +0.5, +2.2\}\text{ rad/s}$, und zwar im Schritt $t = 45$, wenn das Pendel zum Einzugsgebiet aufsteigt. Dabei zeigen sich die Unterschiede zwischen den drei Verfahren:

<!--en:6990b962-->
::: {.slide title="Ausregeln eines Stoßes bei t = 45 (Fortsetzung)" layout="tall-output"}

<!--en:16d9137a-2-->
[Steuerung ohne Rückführung, TV-LQR und MPC im Test]{.kicker}

<!--en:8097d6eb-->
Endwinkel und Ertrag für jedes Verfahren und jede Stoßstärke:

<!--en:38c7aac4-->
::: {.slide title="Phasenporträts und Rückkehr ins Gleichgewicht nach starken Stößen"}

<!--en:caa4028d-->
[Warum die Neuplanung im Betrieb nötig ist]{.kicker}

<!--en:1c64d539-->
Die drei Teilbilder zeigen den Zusammenhang zwischen lokalem LQR, Trajektorienoptimierung und Neuplanung mit gleitendem Horizont:

<!--en:b7118125-->
- **Mäßiger Stoß ($\Delta\dot{\vartheta} = +0.5$):** Ohne Rückführung fällt das Pendel ($\vartheta_{120} = -2.94$). TV-LQR und MPC gleichen die Störung innerhalb der Umgebung aus, in der die lineare Näherung gilt ($\vartheta_{120} = 0.000$).

<!--en:4cb7cdc2-->
- **Starker Stoß ($\Delta\dot{\vartheta} = +2.2$):** Der Aufstieg bricht ab. Die lineare Rückführung von TV-LQR um den alten Plan erreicht die Sättigungsgrenze und versagt ($\vartheta_{120} = +1.96$). **MPC erkennt, dass der mit dem alten Plan initialisierte Aufschwung die obere Lage verfehlt, plant einen neuen Aufschwung und erreicht wieder $\vartheta_{120} = 0.000$**.

<!--en:6f19a7bd-->
::: {.slide title="Klassische optimale Regelung und bestärkendes Lernen im Vergleich"}

<!--en:92be91c3-->
[Begriffe der beiden Fachgebiete]{.kicker}

<!--en:933f09ed-->
- **Variablen und Ziel:** Die optimale Regelung minimiert die Summe der Stufenkosten $\sum_t c(\mathbf{x}_t, \mathbf{u}_t)$ über $(\mathbf{x}_t, \mathbf{u}_t) \in \mathbb{R}^d \times \mathbb{R}^m$. Bestärkendes Lernen maximiert den diskontierten Ertrag $\mathbb{E}[\sum_t \gamma^t r(s_t, a_t)]$.

<!--en:726d12c5-->
- **Backups:** Die Riccati-Rekursion für $\mathbf{x}^\top\mathbf{P}_t\mathbf{x}$ entspricht dem Bellman-Operator $\mathcal{T}$ für $V(s), Q(s, a)$.

<!--en:49549736-->
- **Unbekannte Physik:** Systemidentifikation und MPC stellen die Verbindung zum modellbasierten bestärkenden Lernen (§{sec_model_based}) und zu modellfreien Policy-Gradienten (§{sec_policygradient}) her.

<!--en:46bca58d-->
::: {.slide title="Verständnisfragen zu LQR, iLQR und MPC"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:b28f9f39-->
Fragen zu Riccati-Backups, Drehmomentsättigung und Regelung mit gleitendem Horizont:

<!--en:799e1572-->
1. **Dimension des Riccati-Problems:** Ein humanoider Roboter hat $d = 36$ Zustandsdimensionen. Wie viele Einträge braucht ein grobes Gitter mit $10$ Intervallen pro Achse? Wie viele braucht die symmetrische Riccati-Matrix $\mathbf{P}_t$?

<!--en:8b11b9a3-->
2. **Sicherheitsäquivalenz:** Gaußverteilte Windböen $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ mit Mittelwert null stören den linearen Doppelintegrator $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t + \mathbf{B}\mathbf{u}_t + \mathbf{w}_t$. Wie ändert sich die optimale Rückführverstärkung $\mathbf{K}_t$?

<!--en:7699d875-->
3. **TV-LQR und MPC:** Warum konnte TV-LQR den Stoß von $+0.5\text{ rad/s}$ bei $t = 45$ ausgleichen, versagte aber beim Stoß von $+2.2\text{ rad/s}$ vollständig?

<!--en:557b98b6-->
**Lösungen:**

<!--en:be245716-->
1. Das Gitter mit 10 Intervallen pro Achse braucht $10^{36}$ Zellen. Dagegen hat $\mathbf{P}_t \in \mathbb{R}^{36 \times 36}$ nur $d(d+1)/2 = 36 \times 37 / 2 = 666$ unabhängige Einträge.

<!--en:3b675c10-->
2. $\mathbf{K}_t$ bleibt unverändert. In $\mathbb{E}[(\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{w})^\top\mathbf{P}_{t+1}(\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{w})]$ verschwindet der in $\mathbf{w}$ lineare gemischte Term. Der Ausdruck $\mathbb{E}[\mathbf{w}^\top\mathbf{P}_{t+1}\mathbf{w}] = \textrm{tr}(\mathbf{P}_{t+1}\boldsymbol{\Sigma})$ trägt nur eine von $\mathbf{u}$ unabhängige Konstante bei.

<!--en:3e196cab-->
3. Beim Stoß von $+0.5\text{ rad/s}$ blieb der Stab in der Umgebung der nominellen Trajektorie, in der die lineare Näherung gilt. Der Stoß von $+2.2\text{ rad/s}$ warf ihn zurück nach unten. Dort erreicht das Drehmoment seine Sättigungsgrenze, und das Pendel muss über mehrere Schwingungen erneut Energie aufbauen. MPC plant diesen Aufschwung neu.

<!--en:bf971734-->
::: {.slide title="Klassische Verfahren der optimalen Regelung im Überblick"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:a026a255-->
| Verfahren | Dynamik und Kostenmodell | Form und Leistungsfähigkeit der Strategie |

<!--en:44776645-->
| **LQR mit endlichem oder unendlichem Horizont** | Lineare Dynamik $\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$, quadratische Kosten $\mathbf{x}^\top\mathbf{Q}\mathbf{x} + \mathbf{u}^\top\mathbf{R}\mathbf{u}$ | Exakte lineare Rückführung $\mathbf{u}_t^* = -\mathbf{K}_t\mathbf{x}_t$ durch rückwärts laufende Riccati-Rekursion |

<!--en:2f6b126b-->
| **Lokaler LQR-Regler am Gleichgewicht** | Linearisierung mit den Jacobi-Matrizen $(\mathbf{A}, \mathbf{B})$ am Fixpunkt $(\mathbf{x}^*, u^*)$ | Stabilisiert instabile Gleichgewichte lokal innerhalb der Sättigungsgrenzen des Aktors |

<!--en:7c20eacc-->
| **Iteratives LQR-Verfahren (iLQR)** | Zeitvariante Jacobi-Matrizen $(\mathbf{A}_t, \mathbf{B}_t)$ entlang der nominellen simulierten Bahn | Findet nichtlineare Trajektorien mit mehreren Schwingungen durch Riccati-Rückwärtsdurchlauf und Liniensuche |

<!--en:405cc2b2-->
| **Bahnfolgeregelung mit TV-LQR** | Lineare Näherung für Störungen in einer Umgebung der nominellen Bahn $(\bar{\mathbf{x}}_t, \bar{v}_t)$ | Rückführung $v_t = \bar{v}_t + \mathbf{K}_t(\mathbf{x}_t - \bar{\mathbf{x}}_t)$ bei kleinen Störungen |

<!--en:43d673c8-->
| **MPC mit gleitendem Horizont** | Löst die Optimierung mit endlichem Horizont von $\mathbf{x}_t$ aus im Betrieb neu | Rückkehr ins Gleichgewicht auch aus entfernten Zuständen durch Rückführung und Planung neuer Aufschwünge nach starken Stößen |
