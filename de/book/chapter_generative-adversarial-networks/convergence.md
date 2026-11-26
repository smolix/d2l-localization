<!--en:ab23166c-->
# Gradientenstrafen und Konvergenz

<!--en:a0dadcc0-->
Im Experiment mit Fashion-MNIST in :numref:`sec_basic_gan` pendelten sich weder der Diskriminatorverlust noch der Generatorverlust auf ihren jeweiligen theoretischen Gleichgewichtswert ein. Konvergiert der alternierende Gradientenabstieg (engl. alternating gradient descent) nur langsam, oder kann er ein Gleichgewicht unbegrenzt umkreisen, ohne es zu erreichen? Die Analysen der Divergenzen auf Populationsebene in :numref:`sec_basic_gan`–:numref:`sec_gan_relativistic` beschreiben die Minimax-Zielfunktion für einen Kritiker in seinem exakten Optimum $D^\star$. In der Praxis machen beide Netze kleine Gradientenschritte, entweder gleichzeitig oder abwechselnd. Jeder Schritt verändert die Zielfunktion des anderen Netzes.

<!--en:f7e90f73-->
Um die Dynamik des Trainings mit Gradienten direkt zu untersuchen, führten :citet:`Mescheder.Geiger.Nowozin.2018` das *Dirac-GAN* ein. Dieses minimale Problem hat zwei Parameter: Die Datenverteilung und die Generatorverteilung sind jeweils eine Punktmasse, der Kritiker ist linear. Schon bei diesem System mit zwei Parametern konvergiert der unregularisierte Gradientenabstieg nicht. Bestrafen wir dagegen die quadrierte Norm des Eingabegradienten des Kritikers ($R_1$ und $R_2$), so wird das Gleichgewicht lokal asymptotisch stabil. Kombinieren wir diese um null zentrierten Gradientenstrafterme (engl. zero-centered gradient penalties) mit der Zielfunktion für die relativistische Paarbildung aus :numref:`sec_gan_relativistic`, so erhalten wir das Trainingsverfahren von R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. Dieses Verfahren verwenden wir im weiteren Verlauf des Kapitels.

<!--en:e7fdb15b-->
## Das Dirac-GAN

<!--en:2ab44fb6-->
### Ein Minimax-System mit zwei Parametern

<!--en:81aac297-->
Die Datenverteilung sei eine Punktmasse im Ursprung, $p = \delta_0$. Der Generator platziere eine Punktmasse bei $\theta \in \mathbb{R}$, also $q_\theta = \delta_\theta$ (somit gilt $q_\theta = p$ genau dann, wenn $\theta = 0$). Der Kritiker sei linear und ohne Bias, $D_\psi(x) = \psi x$. Für die symmetrische Margin-Zielfunktion $V_\ell(D) = \mathbb{E}_{x \sim p}[\ell(D(x))] + \mathbb{E}_{x' \sim q}[\ell(-D(x'))]$ (:numref:`sec_gan_objectives`) genügt für jeden der beiden Erwartungswerte eine einzige Funktionsauswertung:

<!--en:46617fa8-->
Wir wählen die logistische Auszahlungsfunktion $\ell(t) = \log \sigma(t)$. Für sie gelten $\ell'(t) = \sigma(-t) > 0$ und $\ell''(t) = -\sigma(t)\sigma(-t) < 0$, insbesondere $\ell'(0) = \tfrac12$ und $\ell''(0) = -\tfrac14$.

<!--en:377554df-->
Für jedes $\theta \neq 0$ sind die Träger von $p$ und $q_\theta$ disjunkt. Maximieren wir $V(\theta, \psi)$ über $\psi$, so gilt $\psi\theta \to -\infty$. Damit ergibt sich der konstante Wert $\sup_\psi V(\theta, \psi) = \ell(0)$ für alle $\theta \neq 0$. Sein Gradient bezüglich $\theta$ ist null.

<!--en:c37801fd-->
### Ungedämpfte Schwingungen in kontinuierlicher Zeit

<!--en:38583078-->
Statt $\sup_\psi V(\theta, \psi)$ zu bestimmen, aktualisieren wir beim Training mit Gradienten $\theta$ und $\psi$ schrittweise ausgehend von ihren aktuellen Werten. Der einzige stationäre Punkt von :eqref:`eq_gan_dirac_value` ist das Gleichgewicht $(\theta, \psi) = (0, 0)$. Für den simultanen Gradientenfluss (engl. gradient flow) in kontinuierlicher Zeit ($\dot\theta = -\partial_\theta V$, $\dot\psi = +\partial_\psi V$) gilt:

<!--en:7e7cf81a-->
Da der Geschwindigkeitsvektor $\mathbf{u}(\theta, \psi) = \ell'(-\psi\theta)\,(\psi, -\theta)$ überall orthogonal zum Zustandsvektor $(\theta, \psi)$ steht, bleibt das Quadrat des Abstands zum Ursprung erhalten:

<!--en:c1c55a43-->
Jede Trajektorie verläuft auf einem Kreis mit konstantem Radius um $(0, 0)$. Das gilt für jede streng monoton wachsende Auszahlungsfunktion $\ell$. Der Generatorparameter $\theta$ schwingt um $0$, ohne jemals zu konvergieren.

<!--en:81fa7ba7-->
Linearisieren wir :eqref:`eq_gan_dirac_flow` um $(\theta, \psi) = (0, 0)$ (:numref:`subsec_mdl-ode-linearization`), so erhalten wir folgende Jacobi-Matrix und Eigenwerte:

<!--en:92dc98f3-->
Rein imaginäre Eigenwerte entsprechen einem linearen *Zentrum* (bilineares Sattelpunktproblem $\min_\theta \max_\psi \theta\psi$, :citet:`Daskalakis.Ilyas.Syrgkanis.ea.2018`). Der skalare Faktor $\ell'(-\psi\theta) = \sigma(D_\psi(\theta)) = w_{\textrm{sat}}$ multipliziert beide Komponenten gleichermaßen und skaliert daher nur die Winkelgeschwindigkeit auf der Kreisbahn.

<!--en:35fff7fe-->
### Nach außen laufende Spiralen und Grenzzyklen in diskreter Zeit

<!--en:7ca86f7c-->
In diskreter Zeit lautet der simultane Gradientenabstieg mit Schrittweite $h > 0$ wie folgt: $(\theta, \psi) \mapsto (\theta, \psi) + h\,\mathbf{u}(\theta, \psi)$. Wegen $\mathbf{u}(\theta, \psi) \perp (\theta, \psi)$ liefert der Satz des Pythagoras:

<!--en:fa5b8075-->
Jeder simultane diskrete Schritt *vergrößert* also den Abstand zum Gleichgewicht für jede Schrittweite $h > 0$ strikt. Die Iterierten bewegen sich daher auf einer Spirale nach außen (Übung :exref:`ex_convergence-discrete-simultaneous-update-dirac`).

<!--en:a8f0af25-->
Alternierende Gradientenschritte verhalten sich etwas günstiger. Wir aktualisieren zuerst $\psi$ und dann $\theta$ mit dem neuen Wert von $\psi$. In der Nähe von $(0, 0)$ ist jeder Halbschritt eine flächentreue lineare Scherung. Die Jacobi-Matrix der zusammengesetzten Abbildung hat daher die Determinante $1$. Für $h\,\ell'(0) \leq 2$ liegen beide Eigenwerte auf dem Einheitskreis :cite:`Mescheder.Geiger.Nowozin.2018`. Die alternierenden Iterierten umkreisen den Ursprung in einem beschränkten Band, ohne zu konvergieren (Übung :exref:`ex_convergence-alternating-updates`). Der nicht sättigende Generatorverlust hat im Ursprung dieselbe Linearisierung $J$. Der relativistische Paarverlust $\ell(D_\psi(0) - D_\psi(\theta)) = \ell(-\psi\theta)$ liefert sogar dasselbe Gradientenfeld. Beide lassen die lokalen Eigenwerte unverändert.

<!--en:b69554aa-->
## Um null zentrierte Gradientenstrafterme

<!--en:72d3a980-->
Damit das Gleichgewicht $(\theta, \psi) = (0, 0)$ lokal asymptotisch stabil wird, müssen wir die Eigenwerte der Jacobi-Matrix von der imaginären Achse in die offene linke Halbebene $\operatorname{Re}(\nu) < 0$ verschieben. Im Gleichgewicht muss der Kritiker flach sein ($\nabla_x D^\star(x) = 0$ für $q = p$). Bestrafen wir die quadrierte Norm seines Eingabegradienten, so dämpfen wir den Kritiker, ohne das Gleichgewicht $q = p$ zu verschieben.

<!--en:35f868a2-->
:citet:`Mescheder.Geiger.Nowozin.2018` definieren die beiden *um null zentrierten Gradientenstrafterme* $R_1$ (ausgewertet an echten Daten $x \sim p$) und $R_2$ (ausgewertet an generierten Samples $x' \sim q$):

<!--en:2892305e-->
Dabei ist $\gamma > 0$ das Gewicht der Strafterme. Der Kritiker maximiert $V_\ell(D) - R_1 - R_2$, während die Zielfunktion des Generators unverändert bleibt. Schreiben wir $m = (p + q)/2$, so ergibt sich die Summe der Strafterme aus dem Erwartungswert der quadrierten Gradientennorm unter der Mischverteilung:

<!--en:6ad8898e-->
### Dämpfung des Dirac-GAN

<!--en:2d68e6af-->
Für den linearen Kritiker $D_\psi(x) = \psi x$ gilt überall $\nabla_x D_\psi(x) = \psi$, also $R_1 = R_2 = \tfrac{\gamma}{2}\psi^2$. Ziehen wir $\tfrac{\gamma}{2}\psi^2$ von der Zielfunktion des Kritikers ab, so kommt der lineare Dämpfungsterm $-\gamma\psi$ zu $\dot\psi$ hinzu (oder $-2\gamma\psi$, wenn wir sowohl $R_1$ als auch $R_2$ berücksichtigen):

<!--en:c488a4cc-->
Für die Jacobi-Matrix bei $(0, 0)$ und ihre Eigenwerte ergibt sich:

<!--en:0dffd9df-->
Für jedes $\gamma > 0$ haben beide Eigenwerte einen strikt negativen Realteil ($\operatorname{Re}(\nu_{1,2}) < 0$). Das Gleichgewicht $(0, 0)$ ist daher lokal exponentiell stabil. Die charakteristische Gleichung $\nu^2 + \gamma\nu + \ell'(0)^2 = 0$ ist dieselbe wie bei einem gedämpften harmonischen Oszillator (:numref:`subsec_mdl-ode-linearization`). Wir unterscheiden drei Fälle:

<!--en:af678754-->
1. *Unterkritische Dämpfung* ($\gamma < 2\ell'(0)$): Die Eigenwerte sind konjugiert komplex mit Realteil $-\gamma/2$. Die Trajektorien verlaufen spiralförmig nach innen (bei simultanen Euler-Schritten konvergieren sie für $h < \gamma / \ell'(0)^2$, Übung :exref:`ex_convergence-discrete-simultaneous-update-dirac`).

<!--en:68937d32-->
2. *Kritische Dämpfung* ($\gamma = 2\ell'(0)$, also $\gamma = 1$ beim logistischen Verlust): Der doppelte reelle Eigenwert $\nu = -\ell'(0)$ führt zum schnellsten Abklingen ohne Schwingung.

<!--en:2cc32bd2-->
3. *Überkritische Dämpfung* ($\gamma > 2\ell'(0)$): Beide Eigenwerte sind reell und negativ. Die langsame Eigenmode mit Eigenwert $\nu \approx -\ell'(0)^2/\gamma$ klingt umso langsamer ab, je größer $\gamma$ wird.

<!--en:8169131c-->
### Phasenporträts

<!--en:91ef77a5-->
Wir überprüfen diese vier Fälle, indem wir :eqref:`eq_gan_dirac_flow` ausgehend von $(\theta_0, \psi_0) = (1, 1)$ (Anfangsradius $\sqrt{2} \approx 1.41$) integrieren. Dazu verwenden wir das Runge-Kutta-Verfahren vierter Ordnung (`rk4_step`), simultane Vorwärts-Euler-Schritte (`euler_step`) und alternierende Gradientenschritte (`alt_step`).

<!--en:95e99743-->
Alle Bahnen beginnen am roten Punkt $(\theta, \psi) = (1, 1)$ im Abstand $\sqrt 2 \approx 1.41$ vom Gleichgewicht am schwarzen Kreuz. Links sehen wir den Fluss. Die mittlere Grafik zeigt 300 simultane und 2000 alternierende Schritte der Schrittweite $h = 0.2$. Rechts sehen wir 300 simultane Schritte derselben Schrittweite mit Strafterm, einmal bei unterkritischer Dämpfung mit $\gamma = 0.3$ und einmal bei kritischer Dämpfung mit $\gamma = 1$.

<!--en:f405f265-->
Die Phasenporträts bestätigen alle vier Vorhersagen. Der kontinuierliche Fluss (`RK4`) verläuft auf dem Kreis mit Radius $1.41$. Die simultanen Euler-Schritte führen spiralförmig nach außen bis zum Radius $2.87$ nach 300 Schritten. Bei alternierenden Schritten bleibt der Radius im Intervall $[1.34, 1.42]$. Mit dem um null zentrierten Strafterm führen die simultanen Euler-Schritte direkt zum Ursprung (Restradius $6.5 \times 10^{-4}$ bei unterkritischer Dämpfung mit $\gamma = 0.3$ und $7 \times 10^{-12}$ bei kritischer Dämpfung mit $\gamma = 1$).

<!--en:76f9a98c-->
### Lokale Konvergenz allgemeiner Netze und Vergleich der Strafterme

<!--en:ed725751-->
:citet:`Mescheder.Geiger.Nowozin.2018` übertrugen diese lokale Stabilitätsanalyse vom Dirac-GAN mit zwei Parametern auf allgemeine neuronale Netze. Betrachten wir die Umgebung eines Gleichgewichts, in dem der Generator die Datenverteilung $p$ darstellt und der Kritiker auf $\operatorname{supp}(p)$ flach ist. Simultane und alternierende Gradientenschritte mit $R_1$ oder $R_2$ und einem strikt konkaven Verlust mit $\ell'(0) \neq 0, \ell''(0) < 0$ konvergieren dort exponentiell gegen die Gleichgewichtsmannigfaltigkeit. Voraussetzung ist, dass die lokale Kapazität des Kritikers ausreicht, um Störungen des Generators in erster Ordnung zu erkennen. :citet:`Huang.Gokaslan.Kuleshov.ea.2024` beweisen einen analogen lokalen Satz für die Zielfunktion der relativistischen Paarbildung.

<!--en:822c60b1-->
Die Analyse erklärt auch zwei Entscheidungen für die Praxis:

<!--en:48cc943c-->
- **Um null ($R_1, R_2$) oder um eins (WGAN-GP) zentrierte Strafterme:** WGAN-GP bestraft $(\|\nabla_x D\| - 1)^2$ und zwingt den Kritiker selbst bei $q = p$ zu einer Steigung vom Betrag eins. Beim Dirac-GAN mit linearer Auszahlungsfunktion $\ell(t) = t$ verhindert der um eins zentrierte Strafterm die Konvergenz gegen $\theta = 0$. Für $\gamma \geq 2$ bleibt der Generator bei $|\theta| = \gamma$ stehen, wo sich Wasserstein-Gradient und Strafterm die Waage halten. Für kleinere $\gamma$ schießt der Generator über diesen Punkt hinaus und bewegt sich weiter auf einer Umlaufbahn (Übung :exref:`ex_convergence-one-centered-dynamics`).

<!--en:83810c83-->
- **$R_1$ und $R_2$ zusammen oder $R_1$ allein:** In der Nähe von $q = p$ sind $R_1$ und $R_2$ gleichwertig. Weit vom Gleichgewicht entfernt bestraft $R_1$ die Steigung des Kritikers jedoch nur an echten Datenpunkten $x \sim p$. An generierten Punkten $x' \sim q$ bleibt die Steigung unbeschränkt. Zum Beispiel hat $D_c(x) = -cx^2$ bei $p = \delta_0$ die Steigung null, also gilt $R_1 = 0$. Damit sind $c \to \infty$ und explodierende Generatorgradienten bei $x' = 1$ möglich (Übung :exref:`ex_convergence-one-sided-penalty`). Mit $R_2$ beschränken wir den Gradienten des Kritikers auch an generierten Samples.

<!--en:515a8d1f-->
## Einfluss der Gradientenstrafen auf die Wertfunktion

<!--en:c04e957e-->
Wie verändern die um null zentrierten Strafterme $R_1 + R_2 = \gamma\,\mathbb{E}_m[\|\nabla_x D\|^2]$ die Divergenz, die der Generator in der Nähe des Gleichgewichts minimiert?

<!--en:a0a8be0a-->
### Lokale quadratische Näherung und Verschiebung einer Gauß-Verteilung

<!--en:fcd9d021-->
In der Nähe von $q = p$ nimmt der optimale Kritiker $D$ kleine Werte an. Setzen wir die Entwicklung $\ell(t) \approx \ell(0) + \ell'(0)t - \tfrac12|\ell''(0)|t^2$ bis zur zweiten Ordnung in $D$ in $V_\ell(D)$ ein, so erhalten wir:

<!--en:9622b18f-->
Dabei gilt $\langle p - q, D \rangle = \mathbb{E}_p[D] - \mathbb{E}_q[D]$. Betrachten wir zwei Gauß-Verteilungen $p = \mathcal{N}(0, 1)$ und $q = \mathcal{N}(\mu, 1)$ mit Varianz eins und einer kleinen Verschiebung $\mu$ sowie einen linearen Kritiker $D(x) = cx$. In führender Ordnung in $\mu$ gelten $\langle p - q, D \rangle = -c\mu$ und $\int m\,D^2 \approx c^2$, und der Strafterm beträgt $\gamma\int m\,\|\nabla_x D\|^2 = \gamma c^2$. Maximieren wir die konkave quadratische Funktion $-\ell'(0)\mu c - (|\ell''(0)| + \gamma)c^2$ über $c$, so erhalten wir die optimale Steigung $c^\star = -\frac{\ell'(0)\mu}{2(|\ell''(0)| + \gamma)}$ und den Wert:

<!--en:5b4ac788-->
Der Strafterm verringert also sowohl die Steigung des Kritikers als auch den lokalen Wert um den Faktor $1 + 4\gamma$. Bei $\gamma = 0$ erhalten wir wieder $\mu^2/4 \approx 2\,\mathrm{JS}(p, q)$. Für große $\gamma$ skaliert der Wert wie $\mu^2/(16\gamma) = \tfrac{\ell'(0)^2}{4\gamma} W_2(p, q)^2$. Wir prüfen diese geschlossene Näherungsformel anhand des exakten Optimums mit Strafterm. Dazu berechnen wir das Optimum mit dem Newton-Verfahren auf einem eindimensionalen Gitter.

<!--en:704835f9-->
Für kleine Verschiebungen ($\mu = 0.2$) weicht die quadratische Näherung $\mu^2/(4 + 16\gamma)$ bei allen betrachteten Werten von $\gamma$ um höchstens 1% von der exakten numerischen Lösung ab.

<!--en:f850962c-->
### Duale Sobolev-Norm und Tiefpassfilterung

<!--en:5490029c-->
Für allgemeine Verteilungen $p$ und $q$ in der Nähe des Gleichgewichts können wir die Gradientenstrafe durch partielle Integration als $\int m\,\|\nabla_x D\|^2 = \langle D, \mathcal{L}_m D \rangle$ schreiben. Dabei ist $\mathcal{L}_m D = -\nabla \cdot (m \nabla D)$ der mit $m$ gewichtete Laplace-Operator. Maximieren wir die lokale quadratische Näherung über alle Funktionen $D$, so ergibt sich (Übung :exref:`ex_convergence-prove-eqref-eq-gan`):

<!--en:ae769464-->
Für $\gamma \to 0$ vereinfacht sich der Ausdruck zu $\tfrac{\ell'(0)^2}{4|\ell''(0)|} \int (p - q)^2/m \approx 2\,\mathrm{JS}(p, q)$. Für $\gamma \to \infty$ nähert er sich einem Vielfachen des Quadrats der *dualen Sobolev-Norm* $\dot H^{-1}(m)$:

<!--en:43a22a09-->
In Bereichen, in denen $m$ annähernd konstant ist, multipliziert $\mathcal{L}_m$ eine Fourier-Mode der Wellenzahl $\omega$ mit $m\omega^2$. Die Mode erhält dadurch das Gewicht $(|\ell''(0)| + \gamma\omega^2)^{-1}$ anstelle von $|\ell''(0)|^{-1}$. Somit wirkt $R_1 + R_2$ auf $p - q$ wie ein räumlicher *Tiefpassfilter* (engl. low-pass filter) mit der charakteristischen Grenzlänge $\sqrt{\gamma / |\ell''(0)|}$ ($2\sqrt{\gamma}$ beim logistischen Verlust). Da $\nabla_x D$ die Dimension einer inversen Länge hat, hat $\gamma$ die Dimension einer quadrierten Länge und skaliert mit der Auflösung und der Norm der Daten. Außerdem bestraft $R_1 + R_2$ den Gradienten $\nabla_x D$ nur auf dem Träger von $m = (p+q)/2$. Im leeren Raum zwischen disjunkten Trägern hängt der Gradient weiterhin von der Glattheit des neuronalen Netzes ab, das als Kritiker dient (Übung :exref:`ex_convergence-penalized-best-response`).

<!--en:4eb8ea01-->
## Das Trainingsverfahren von R3GAN

<!--en:67f75286-->
R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` kombiniert den relativistischen Paarverlust (`d2l.rpgan_loss_D` und `d2l.rpgan_loss_G` aus :numref:`sec_gan_relativistic`) mit den beiden um null zentrierten Gradientenstraftermen $R_1$ und $R_2$.

<!--en:1e671c9a-->
### Implementierung der Verlustfunktionen und Strafterme

<!--en:ce74f528-->
Zum Vergleich speichern wir auch `gan_loss_D` und `gan_loss_G`. Sie implementieren den klassischen logistischen Diskriminatorverlust und den nicht sättigenden Generatorverlust aus :numref:`sec_basic_gan` mit derselben Signatur `(critic, real, fake)`.

<!--en:afaabf95-->
Die Funktion für den Strafterm gibt für beide Batches die quadrierte Norm des Eingabegradienten jedes Samples zurück, noch ohne den Faktor $\gamma/2$. Dazu differenziert sie die *Summe* der Scores des Kritikers nach dem Eingabebatch. Jeder Score hängt nur von seiner eigenen Zeile ab. Daher liefert ein einziger Rückwärtsdurchlauf die Eingabegradienten aller Samples. Bei Batchnormalisierung wäre das nicht mehr der Fall, da diese Informationen aus mehreren Zeilen verknüpft. Der Eingabegradient bleibt im Berechnungsgraphen der automatischen Differentiation. So kann der Optimierer den Strafterm nach den Parametern des Kritikers differenzieren. Die Samples bleiben dabei konstant. Wegen der zweiten Ableitung kostet ein Kritikerschritt mit Strafterm etwa so viel wie zwei Rückwärtsdurchläufe.

<!--en:8699414f-->
`create_graph=True` erhält den Berechnungsgraphen des Eingabegradienten für die zweite Differentiation. Mit `detach` behandeln wir die Samples als Konstanten.

<!--en:a6741bf1-->
`jax.grad` liefert für die Summe der Scores die Eingabegradienten. Um den Strafterm nach den Kritikerparametern zu differenzieren, differenzieren wir ein zweites Mal durch diese Gradientenberechnung. Mit `jax.lax.stop_gradient` behandeln wir die Samples als Konstanten.

<!--en:d053ca34-->
### Die Trainingsschritte

<!--en:589072f4-->
In der Trainingsschleife machen die beiden Spieler abwechselnd je einen Schritt. `critic_step` macht einen Optimierungsschritt für die Summe aus Kritikerverlust und den mit $\gamma/2$ gewichteten beiden Straftermen. Dabei bleibt der generierte Batch fest. Die Funktion gibt den Verlust und die Batchmittelwerte der quadrierten Eingabegradientennormen für echte und generierte Samples zurück ($R_1$ und $R_2$ vor Multiplikation mit $\gamma/2$). Damit kann die Schleife die Steilheit des Kritikers protokollieren. Anschließend macht `generator_step` einen Schritt für den Generatorverlust mit dem aktualisierten Kritiker. Die optionalen Labels `y` sind für den bedingten Kritiker aus :numref:`sec_gan_conditional` vorgesehen.

<!--en:240cf81a-->
Die Verlustfunktionen sind statische Argumente der kompilierten Schritte. Daher ist für jedes Paar von Verlustfunktionen nur eine Kompilierung nötig. Der Tracer erfasst $\gamma$ als dynamisches Argument, dessen Wert sich ohne erneute Kompilierung ändern kann.

<!--en:871700d8-->
### Modusabdeckung auf einem Gitter aus 25 Gauß-Verteilungen

<!--en:bf6d0766-->
Wir untersuchen, wie $R_1 + R_2$ und paarweises Ranking die Modusabdeckung (engl. mode coverage) in neuronalen Netzen beeinflussen. Dazu trainieren wir als Generator ein MLP mit 3 Schichten und 64-dimensionalem latentem Raum. Die beiden versteckten Schichten enthalten je 256 ReLU-Einheiten. Als Kritiker dient ein MLP mit 3 Schichten und Leaky ReLU. Die Daten stammen von einem $5 \times 5$-Gitter zweidimensionaler Gauß-Verteilungen (Abstand $2.0$, Standardabweichung $\sigma = 0.05$). Wir bewerten 10 000 generierte Samples anhand von vier Metriken:

<!--en:71d36f1a-->
1. *Abgedeckte Modi*: Anzahl der Modi, auf die mindestens $1/5$ des bei Gleichverteilung erwarteten Anteils ($1/25$) der modusnahen Samples entfällt.

<!--en:d8335a76-->
2. *Anteil modusnaher Samples*: Anteil der Samples, deren Abstand zum nächsten Moduszentrum höchstens $3\sigma = 0.15$ beträgt ($98.9\%$ bei der wahren Verteilung).

<!--en:a3a2a7d4-->
3. *Mittlerer Abstand* zum nächsten Zentrum ($\approx 0.06$ bei der wahren Verteilung).

<!--en:3fa0bf75-->
4. *Steilheit des Kritikers* $\mathbb{E}_q[\|\nabla_x D(x')\|^2]$ an generierten Samples.

<!--en:5a3e0c32-->
Um die generierten Punktwolken mit dem Gitter `centers` zu vergleichen, ordnet `mode_stats` jedes Sample dem nächstgelegenen Moduszentrum zu. Die Funktion liefert die Anzahl der abgedeckten Modi, den Anteil der Samples im Umkreis von $3\sigma$ um das nächste Zentrum und den mittleren Abstand zu diesem Zentrum:

<!--en:e66879ba-->
`train_grid` führt abwechselnd `critic_step` und `generator_step` über 10 000 Schritte mit Batchgröße 1024 aus. Wie bei R3GAN verwenden wir Adam mit $\beta_1 = 0$, denn positives Momentum im ersten Moment kann bei einem rotierenden Vektorfeld den Stabilitätsbereich verkleinern. Die Lernrate sinkt nach einem Kosinusverlauf von $2 \times 10^{-3}$ auf $2 \times 10^{-4}$.

<!--en:50f73ea8-->
In einem faktoriellen $2 \times 2$-Versuch kombinieren wir jede der beiden Zielfunktionen (klassischer nicht sättigender Verlust `NS` oder Paarverlust `pairing`) mit $\gamma = 0$ (ohne Strafterm) und $\gamma = 1$ ($R_1 + R_2$). Jede Kombination trainieren wir mit drei Zufallsstartwerten. Hinzu kommt ein Lauf mit überkritischer Dämpfung bei $\gamma = 10$.

<!--en:a4879047-->
Die folgende Tabelle zeigt die Ergebnisse von `mode_stats` für die 10 000 generierten Samples jedes Laufs und die abschließende quadrierte Gradientennorm des Kritikers $\mathbb{E}_q[\|\nabla_x D(x')\|^2]$:

<!--en:0056605a-->
Für jede der vier Hauptvarianten tragen wir 3000 generierte Samples aus dem ersten Lauf zusammen mit den 25 Moduszentren (`grid`) auf. So sehen wir die Unterschiede in der Modusabdeckung direkt:

<!--en:5ab7920e-->
Die Ergebnisse zeigen, welchen Beitrag die Gradientenstrafen und der Paarverlust jeweils leisten:

<!--en:cd9d7eeb-->
- **Gradientenstrafen verbessern die Modusabdeckung:** Ohne Strafterme ($\gamma = 0$) bleiben die Gradienten des Kritikers groß ($\mathbb{E}_q\|\nabla_x D\|^2 \in [0.7, 17]$). Die Modusabdeckung ist unzuverlässig und schwankt je nach Zufallsstartwert zwischen 12 und 25 Modi. Mit $R_1 + R_2$ bei $\gamma = 1$ sinkt die Steilheit des Kritikers um drei Größenordnungen ($\sim 10^{-3}$). Sowohl mit `NS` als auch mit `pairing` deckt jeder Lauf alle 25 Modi ab.

<!--en:93c0b679-->
- **Überkritische Dämpfung bei großem $\gamma$:** Erhöhen wir $\gamma$ auf $10$, so wird der Kritiker zu flach ($\sim 10^{-4}$). Die Konvergenz verlangsamt sich so stark, dass nach 10 000 Schritten nur 20–30% der Samples im Umkreis von $3\sigma$ um einen Modus liegen.

<!--en:7b9081cb-->
Betrachten wir abschließend die Messwerte an den Checkpoints. Sie zeigen, warum wir einen ruhigen Verlauf des Kritikerverlusts mit Vorsicht deuten müssen:

<!--en:d6fa9238-->
In den Läufen mit Strafterm liegt der Kritikerverlust schon ab dem ersten Checkpoint nahe seinem Gleichgewichtswert ($2\log 2 \approx 1.39$ für `NS`, $\log 2 \approx 0.69$ für `pairing`), selbst wenn nur 1–3% der generierten Samples in der Nähe eines Modus liegen. Der Grund ist, dass die Gradientenstrafe $D(x)$ überall klein hält. Ein Kritikerverlust beim Gleichgewichtswert zeigt also einen flachen Kritiker an. Ob der Generator bereits konvergiert ist, bleibt offen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:f3db6f33-->
Beim Dirac-GAN mit zwei Parametern ($p = \delta_0, q_\theta = \delta_\theta, D_\psi(x) = \psi x$) verläuft der Gradientenfluss in kontinuierlicher Zeit auf Kreisen mit konstantem Radius. Simultane diskrete Schritte führen spiralförmig nach außen, während die Bahnen bei alternierenden diskreten Schritten in einem beschränkten Band bleiben. Ziehen wir die um null zentrierten Gradientenstrafterme $R_1 = \frac{\gamma}{2}\mathbb{E}_p[\|\nabla_x D(x)\|^2]$ und $R_2 = \frac{\gamma}{2}\mathbb{E}_q[\|\nabla_{x'} D(x')\|^2]$ ab, so verschieben sich die Eigenwerte der Jacobi-Matrix in die offene linke Halbebene ($\operatorname{Re}(\nu_{1,2}) < 0$). Das gilt für jedes $\gamma > 0$ und macht das Gleichgewicht lokal exponentiell stabil. Nahe dem Gleichgewicht wirkt $R_1 + R_2$ auf $p - q$ wie ein räumlicher Tiefpassfilter mit der Grenzlänge $2\sqrt{\gamma}$ beim logistischen Verlust. Der lokale Wert interpoliert damit zwischen $2\,\mathrm{JS}(p, q)$ auf großen Längenskalen und der quadrierten dualen Sobolev-Norm $\|p - q\|_{\dot H^{-1}(m)}^2$ auf kleinen Längenskalen. Im Experiment mit dem Gitter aus 25 Gauß-Verteilungen verringert $R_1 + R_2$ bei $\gamma = 1$ die Steilheit des Kritikers um drei Größenordnungen. Sowohl mit der klassischen logistischen Zielfunktion als auch mit der relativistischen Paarbildung erreichen wir eine vollständige Modusabdeckung.

<!--en:2294243a-->
## Übungen

<!--en:6e8d31fd-->
1. [code] **Stabilität des Dirac-GAN in diskreter Zeit.** Der simultane Gradientenabstieg mit Schrittweite $h$ verwendet die Abbildung $(\theta, \psi) \mapsto (\theta, \psi) + h\,\mathbf{u}(\theta, \psi)$.
    1. Zeigen Sie, dass für $\gamma = 0$ die Jacobi-Matrix des diskreten Schritts bei $(0, 0)$ die Eigenwerte $1 \pm i\,h\,\ell'(0)$ besitzt. Weisen Sie für ihren Betrag die Beziehung $\sqrt{1 + h^2\ell'(0)^2} > 1$ für jede Schrittweite $h > 0$ nach.
    1. Für $\gamma > 0$ lauten die Eigenwerte $1 + h\nu_{1,2}$ mit $\nu_{1,2}$ aus :eqref:`eq_gan_dirac_pen`. Zeigen Sie, dass bei unterkritischer Dämpfung $\gamma < 2\ell'(0)$ der diskrete Schritt genau dann lokal stabil ist ($|1 + h\nu_{1,2}| < 1$), wenn $h < \gamma / \ell'(0)^2$ gilt.
    1. Überprüfen Sie diese Schwelle bei $\gamma = 0.3$ ($\ell'(0) = \tfrac12$, Schwelle $h = 1.2$). Führen Sie dazu `trajectory(euler_step, (0.1, 0.1), h, 3000, gamma=0.3)` für $h = 1.1$ und $h = 1.3$ aus.
   :exlabel:`ex_convergence-discrete-simultaneous-update-dirac`

<!--en:1e4201a7-->
1. **Beste Antwort beim Dirac-GAN mit Strafterm.** Betrachten Sie $V_\gamma(\theta) = \sup_\psi \{V(\theta, \psi) - \gamma\psi^2\}$ beim Dirac-GAN mit $\ell = \log \sigma$.
    1. Zeigen Sie, dass für kleine $\theta$ die Steigung $\psi^\star \approx -\ell'(0)\theta / (2\gamma)$ den Wert maximiert und $V_\gamma(\theta) = 2\ell(0) + \frac{\ell'(0)^2}{4\gamma}\theta^2 + O(\theta^4)$ gilt. Vergleichen Sie das Ergebnis mit :eqref:`eq_gan_sobolev` für $W_2(\delta_0, \delta_\theta) = |\theta|$.
    1. Nun durchlaufe $D$ alle differenzierbaren Funktionen mit Strafterm $\gamma\int m\,\|\nabla_x D\|^2$. Zeigen Sie, dass für die Punktmassen $p = \delta_0$ und $q_\theta = \delta_\theta$ ($\theta \neq 0$) der Wert $\sup_D \{V_\ell(D) - R_1 - R_2\} = 2\sup_t \ell(t)$ unabhängig von $\theta$ ist. Warum liefert ein endliches neuronales Netz als Kritiker dennoch über die Lücke hinweg einen von null verschiedenen Gradienten?
   :exlabel:`ex_convergence-penalized-best-response`

<!--en:38d187f2-->
1. **Vergleich der Strafterme $R_1$ allein und $R_1 + R_2$.** Es seien $p = \delta_0$, $q = \delta_1$ und $\ell = \log \sigma$ gegeben. Betrachten Sie die quadratischen Kritiker $D_c(x) = -cx^2$ ($c \geq 0$).
    1. Berechnen Sie $R_1$ und $R_2$ für $D_c$. Zeigen Sie, dass $V_\ell(D_c) - R_1$ mit $c$ streng monoton wächst und daher an keiner endlichen Stelle ein Maximum annimmt.
    1. Zeigen Sie, dass beim nicht sättigenden Generatorverlust $-\log \sigma(D_c(x'))$ der Betrag des Generatorgradienten an der Stelle $x' = 1$ mit $c$ unbeschränkt wächst. Zeigen Sie auch, dass nach Abzug von $R_1 + R_2$ das Maximum an einer endlichen Stelle $c^\star < 1/(8\gamma)$ liegt.
   :exlabel:`ex_convergence-one-sided-penalty`

<!--en:761818a1-->
1. [code] **Alternierende Schritte beim Dirac-GAN.** ● Bei alternierenden Schritten berechnen wir zunächst $\psi_{k+1} = \psi_k + h\,u_\psi(\theta_k, \psi_k)$ und anschließend $\theta_{k+1} = \theta_k + h\,u_\theta(\theta_k, \psi_{k+1})$.
    1. Linearisieren Sie den zusammengesetzten Schritt bei $(0, 0)$ für $\gamma = 0$. Zeigen Sie, dass seine Jacobi-Matrix die Determinante $1$ hat und dass ihre Eigenwerte genau dann auf dem Einheitskreis liegen, wenn $h\,\ell'(0) \leq 2$ gilt.
    1. Führen Sie `trajectory(alt_step, (0.1, 0.1), h, 10**5)` bei $\gamma = 0$ für $h \in \{0.2, 1, 3, 5\}$ aus. Halten Sie den kleinsten und den größten Radius über die letzten $10^4$ Schritte fest. Vergleichen Sie die Ergebnisse mit denen für $\gamma = 0.3$ bei $h = 0.2$.
   :exlabel:`ex_convergence-alternating-updates`

<!--en:75285a96-->
1. [code] **Lokaler quadratischer Wert und Tiefpassfilterung.** ● Schreiben Sie $\mathcal{A} = |\ell''(0)|\,m + \gamma\mathcal{L}_m$ mit $\mathcal{L}_m D = -\nabla \cdot (m \nabla D)$.
    1. Zeigen Sie durch partielle Integration, dass die folgende Gleichung gilt:

       $$|\ell''(0)|\int m\,D^2 + \gamma\int m\,\|\nabla_x D\|^2 = \langle D, \mathcal{A}D \rangle$$

       Leiten Sie daraus :eqref:`eq_gan_local_value` her, indem Sie entlang der Strahlen $t D_0$ maximieren und die Cauchy-Schwarz-Ungleichung anwenden.
    1. Es seien $p = \mathcal{N}(0, 1)$ und $q(x) = p(x)(1 + 0.1\sin(\omega x))$ gegeben. Berechnen Sie mit `penalized_value` das Verhältnis des Wertes mit Strafterm bei $\gamma \in \{0.01, 0.1, 1\}$ zum Wert ohne Strafterm bei $\gamma = 0$ für $\omega \in \{0.5, 2, 8\}$. Vergleichen Sie es mit dem Fourier-Dämpfungsfaktor für konstantes $m$, also $(1 + 4\gamma\omega^2)^{-1}$.
   :exlabel:`ex_convergence-prove-eqref-eq-gan`

<!--en:92f68bc6-->
1. [code] **Dynamik des Dirac-GAN mit einem um eins zentrierten Strafterm.** ● Betrachten Sie die lineare WGAN-Auszahlungsfunktion $\ell(t) = t$ ($\ell'(0) = 1$). Ziehen Sie den um eins zentrierten Strafterm $\frac{\gamma}{2}(|\psi| - 1)^2$ von der Zielfunktion des Kritikers ab.
    1. Zeigen Sie, dass das simultane Gradientenfeld für $\psi \neq 0$ nie verschwindet. Bestimmen Sie die Positionen $\theta$ des Generators, an denen die einseitigen Grenzwerte des Feldes für $\psi \to 0^\pm$ verschwinden.
    1. Ändern Sie `dirac_field` so, dass die Funktion den um eins zentrierten Strafterm verwendet. Führen Sie `trajectory` ab $(1, 1)$ mit $h = 0.01$ für jeweils 40 000 Schritte bei $\gamma = 1$ und $\gamma = 3$ aus. Überprüfen Sie, dass bei $\gamma = 3$ der Generator bei $\theta = 3$ stehen bleibt.
   :exlabel:`ex_convergence-one-centered-dynamics`

<!--en:7602ff77-->
[Diskussion](https://d2l.discourse.group/t/36811)

<!--en:dab7527a-->
Dynamik von Minimax-Spielen und Gradientenstrafen<br>**Dirac-GAN · rotierende Vektorfelder · nach außen laufende Spiralen und Grenzzyklen · um null zentrierte Strafterme $R_1$ und $R_2$ · Filterung mit der dualen Sobolev-Norm · R3GAN**

<!--en:8f508d5d-->
::: {.slide title="Existenz eines Gleichgewichts und Konvergenz des Gradientenverfahrens"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:4785a917-->
- **Statisches Optimum und gekoppelte Dynamik:** In §{num:sec_basic_gan}–§{num:sec_gan_relativistic} galt die Annahme, dass $D$ im Optimum $D^\star$ liegt. In der Praxis machen beide Spieler kleine Gradientenschritte. Jeder Schritt verändert die Zielfunktion des anderen Spielers.

<!--en:9ae46741-->
- **Minimales Gegenbeispiel (Mescheder et al., 2018):** Selbst in einem Minimax-Spiel mit zwei Parametern und eindeutigem Gleichgewicht bei $(0, 0)$ genügen $300$ simultane Gradientenschritte ($h = 0.2$) ab $(1, 1)$: Der Abstand zu $(0, 0)$ **verdoppelt** sich von $1.41$ auf $2.87$.

<!--en:60893f85-->
- **Leitfrage:** Wie machen um null zentrierte Strafterme ($R_1, R_2$) das Gleichgewicht lokal exponentiell stabil?

<!--en:995a545c-->
::: {.slide title="Das Dirac-GAN als Minimax-System mit zwei Parametern"}

<!--en:501c00ab-->
[Minimales analytisches Modell]{.kicker}

<!--en:f42a6077-->
- **Zwei skalare Parameter (Mescheder et al., 2018):** Echte Daten $p = \delta_0$, Generator $q_\theta = \delta_\theta$ ($q_\theta = p \iff \theta = 0$) und linearer Kritiker $D_\psi(x) = \psi x$.

<!--en:f22b439f-->
- **Unstetige beste Antwort:** Für jedes $\theta \neq 0$ gilt bei Maximierung über $\psi$ der Grenzübergang $\psi\theta \to -\infty$. Der Wert $\sup_\psi V(\theta, \psi) = \ell(0)$ ist konstant und hat daher die Steigung null.

<!--en:e5cb70a9-->
::: {.slide title="Kreisbahnen des Gradientenflusses in kontinuierlicher Zeit"}

<!--en:0feff368-->
[Rotierendes Vektorfeld]{.kicker}

<!--en:8ca68fe9-->
- **Exakte Erhaltung des Radius:** Die Geschwindigkeit $\mathbf{u}(\theta, \psi) = \ell'(-\psi\theta)(\psi, -\theta)$ steht orthogonal zu $(\theta, \psi)$. Daher gilt $\frac{d}{dt}(\theta^2 + \psi^2) = 0$.

<!--en:045b52f0-->
- **Imaginäre Eigenwerte der Jacobi-Matrix:** Die Linearisierung bei $(0, 0)$ ergibt $J = \left(\begin{smallmatrix} 0 & \ell'(0) \\ -\ell'(0) & 0 \end{smallmatrix}\right)$ mit den Eigenwerten $\nu_{1,2} = \pm i\,\ell'(0)$. Es liegt ein lineares **Zentrum** vor, dessen Trajektorien dauerhaft auf Kreisbahnen verlaufen.

<!--en:4484cc5c-->
::: {.slide title="Diskrete Zeit: Spiralen nach außen und beschränkte Umlaufbahnen"}

<!--en:4b0f8e8a-->
[Fallstricke der Diskretisierung]{.kicker}

<!--en:943ae092-->
- **Simultane Euler-Schritte führen für alle $h > 0$ spiralförmig nach außen:** Wegen $\mathbf{u} \perp (\theta, \psi)$ vergrößert jeder Schritt den Radius (die Eigenwerte $1 \pm i\,h\,\ell'(0)$ haben Betrag $> 1$).

<!--en:8bbb9d5f-->
- **Alternierende Schritte sind flächentreu:** Die Verkettung der Halbschritte für $\psi$ und $\theta$ ergibt eine $2 \times 2$-Matrix mit $\det = 1$. Die Iterierten bleiben in einem beschränkten Kreisring, ohne zu konvergieren.

<!--en:50792677-->
::: {.slide title="Phasenporträts des Flusses mit und ohne Strafterm"}

<!--en:7c839ab5-->
[Numerische Integration]{.kicker}

<!--en:99917632-->
Die Integration des Dirac-GAN ab $(\theta_0, \psi_0) = (1, 1)$ (Anfangsradius $\sqrt{2} \approx 1.41$) bestätigt alle vier Fälle:

<!--en:86493375-->
- **Links (`flow`):** Bei kontinuierlicher Dynamik ergibt die RK4-Integration eine Kreisbahn mit Radius $1.41$.

<!--en:92f2c76a-->
- **Mitte (`steps`, $h = 0.2$):** $300$ simultane Schritte führen spiralförmig nach außen bis zum Radius $2.87$. Bei $2{,}000$ alternierenden Schritten bleibt der Radius in $[1.34, 1.42]$.

<!--en:c624dff0-->
- **Rechts (`penalized`, $h = 0.2$):** Mit einem um null zentrierten Strafterm für den Kritiker führen simultane Euler-Schritte direkt zu $(0, 0)$ (Restradius $6.5 \times 10^{-4}$ bei $\gamma = 0.3$ und $7 \times 10^{-12}$ bei $\gamma = 1.0$).

<!--en:fee07822-->
::: {.slide title="Um null zentrierte Gradientenstrafterme (R₁ und R₂)"}

<!--en:409a615b-->
[Dämpfung des Kritikers]{.kicker}

<!--en:354c6a88-->
- **Warum die Eingabegradienten bestrafen?** Bei $q = p$ ist der optimale Kritiker für das logarithmierte Dichteverhältnis $D^\star(x) = \log(p/q) = 0$ **überall flach** ($\nabla_x D^\star = \mathbf{0}$). Ein Strafterm für $\|\nabla_x D\|^2$ verschiebt das Gleichgewicht $q = p$ daher nicht.

<!--en:954617e6-->
- **Gemeinsamer Strafterm für die Mischverteilung:** Bei der Maximierung von $V_\ell(D) - R_1 - R_2$ geht $R_1 + R_2 = \gamma\,\mathbb{E}_{x \sim m}[\|\nabla_x D(x)\|^2]$ mit negativem Vorzeichen ein.

<!--en:70845e5f-->
::: {.slide title="Eigenwerte für drei Dämpfungsfälle des Oszillators"}

<!--en:7f8dad08-->
[Gedämpfter harmonischer Oszillator]{.kicker}

<!--en:4d678fa7-->
- **Lokale exponentielle Stabilität für alle $\gamma > 0$:** Es gilt $\operatorname{Re}(\nu_{1,2}) < 0$ in allen drei Fällen (unterkritisch bei $0 < \gamma < 2\ell'(0)$, kritisch bei $\gamma = 2\ell'(0)$ und überkritisch bei $\gamma > 2\ell'(0)$).

<!--en:8ced3575-->
- **Nachteil überkritischer Dämpfung ($\gamma \gg 1$):** Der Eigenwert $\nu_{\textrm{slow}} \approx -\ell'(0)^2/\gamma \to 0$ der langsamen Mode bremst die Konvergenz des Generators, wenn $\gamma$ zu groß ist.

<!--en:8d1e4504-->
::: {.slide title="Lokale Konvergenz neuronaler Netze und Wahl der Strafterme"}

<!--en:8ef87564-->
[Generalisierung und praktische Regeln]{.kicker}

<!--en:1d878eae-->
- **Lokale exponentielle Konvergenz (Mescheder et al., 2018):** Nahe dem Gleichgewicht $q_{\theta^\star} = p$ sei $D_{\psi^\star}$ auf $\operatorname{supp}(p)$ flach. Dann konvergieren simultane und alternierende Schritte mit $R_1$ oder $R_2$ exponentiell.

<!--en:cccd6a7a-->
- **Zentrierung um null ($R_1, R_2$) oder um eins (WGAN-GP):** WGAN-GP erzwingt die Steigung $\|\nabla_x D\| = 1$ selbst bei $q = p$. Das Dirac-GAN erreicht den Ursprung dann nie. Es bleibt bei $|\theta| = \gamma$ stehen, falls $\gamma \geq 2$ gilt, und durchläuft für kleinere $\gamma$ eine Umlaufbahn.

<!--en:8103b8de-->
- **Vorteil von $R_1 + R_2$ gegenüber $R_1$ allein:** Fern vom Gleichgewicht ($p = \delta_0, q = \delta_1$) gilt für den quadratischen Kritiker $D_c(x) = -cx^2$ die Beziehung $R_1 = 0$ bei $x = 0$. Damit ist $c \to \infty$ möglich. Mit $R_2$ gilt dagegen die Schranke $c^\star < 1/(8\gamma)$.

<!--en:d901e143-->
::: {.slide title="Einfluss der Strafterme auf die lokale Wertfunktion"}

<!--en:fc6c1469-->
[Quadratische Entwicklung und Verschiebung einer Gauß-Verteilung]{.kicker}

<!--en:11c219c6-->
Wie verändert der Abzug von $R_1 + R_2 = \gamma\int m\,\|\nabla_x D\|^2$ die Divergenz nahe dem Gleichgewicht? Die Entwicklung $\ell(t) \approx \ell(0) + \ell'(0)t - \frac{1}{2}|\ell''(0)|t^2$ bis zur zweiten Ordnung in $D$ ergibt:

<!--en:29ea9840-->
- **Verschiebung einer eindimensionalen Gauß-Verteilung ($p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(\mu, 1)$, linearer Kritiker $D(x) = cx$):** Für kleine $\mu$ ist $-\ell'(0)\mu c - (|\ell''(0)| + \gamma)c^2$ zu maximieren beim logistischen Verlust ($\ell'(0) = \frac{1}{2}, |\ell''(0)| = \frac{1}{4}$). Sowohl die optimale Steigung als auch der lokale Wert sinken um den Faktor $1 + 4\gamma$:
  $$c^\star = -\frac{\mu}{1 + 4\gamma}, \qquad V_\gamma(D^\star) - 2\ell(0) \approx \frac{\mu^2}{4 + 16\gamma}$$

<!--en:3ca8effe-->
::: {.slide title="Einfluss der Strafterme auf die lokale Wertfunktion (Überprüfung)" layout="tall-output"}

<!--en:fc6c1469-2-->
[Quadratische Entwicklung und Verschiebung einer Gauß-Verteilung]{.kicker}

<!--en:b26d963c-->
Das Newton-Verfahren zur Maximierung über $D$ liefert die exakte Lösung des Spiels mit Strafterm auf einem Gitter. Zum Vergleich dienen der lokale Wert $\mu^2/(4 + 16\gamma)$ und für große $\gamma$ der Wert $\mu^2/(16\gamma)$:

<!--en:51723cc2-->
::: {.slide title="Duale Sobolev-Norm und räumliche Tiefpassfilterung"}

<!--en:593d3556-->
[Interpretation auf verschiedenen Längenskalen]{.kicker}

<!--en:97b8be21-->
- **Interpolation zwischen JS und dualer Sobolev-Norm:** Mit $\mathcal{L}_m D = -\nabla \cdot (m\nabla D)$ geht der lokale Wert von $2\,\mathrm{JS}(p, q)$ für $\gamma \to 0$ in $\frac{\ell'(0)^2}{4\gamma}\|p - q\|_{\dot{H}^{-1}(m)}^2$ für $\gamma \to \infty$ über.

<!--en:882511bf-->
- **Räumlicher Tiefpassfilter:** Die Fourier-Mode der Wellenzahl $\omega$ erhält das Gewicht $(|\ell''(0)| + \gamma\omega^2)^{-1}$. Der Filter dämpft Strukturen unterhalb der Grenzlänge $2\sqrt{\gamma}$ ($\gamma$ hat die Dimension einer **quadrierten Länge**).

<!--en:d7a3e679-->
::: {.slide title="Implementierung der differenzierbaren Strafterme R₁ + R₂"}

<!--en:ee7e92be-->
[Automatische Differentiation höherer Ordnung]{.kicker}

<!--en:068e79da-->
Ohne Batchnormalisierung hängt der Score $D(x_i)$ jedes Samples nur von seiner Zeile $x_i$ ab. Die Ableitung der Summe $\sum_i D(x_i)$ nach dem Eingabebatch $x$ liefert in einem einzigen Rückwärtsdurchlauf den Eingabegradienten $\nabla_{x_i} D(x_i)$ jedes Samples:

<!--en:725e9360-->
**Rechenaufwand:** Der Eingabegradient bleibt im Berechnungsgraphen, damit der Optimierer $\|\nabla_x D(x)\|^2$ nach den Kritikerparametern differenzieren kann. Ein Kritikerschritt mit Strafterm kostet daher ungefähr so viel wie **zwei Rückwärtsdurchläufe**.

<!--en:d2f17ffe-->
::: {.slide title="Funktionen für die Kritiker- und Generatorschritte von R3GAN" layout="tight"}

<!--en:d05abed0-->
[Das Trainingsverfahren im Code]{.kicker}

<!--en:192f1bc6-->
R3GAN (Huang et al., 2024) kombiniert den relativistischen Paarverlust (`rpgan_loss_D`, `rpgan_loss_G`) mit $R_1 + R_2$. Die Funktion `critic_step` aktualisiert `net_D` anhand von `loss_D + gamma / 2 * (r1 + r2).mean()`. Zum Protokollieren gibt sie die unskalierten Batchmittelwerte von $R_1$ und $R_2$ zurück:

<!--en:1faf0046-->
::: {.slide title="Funktionen für die Kritiker- und Generatorschritte von R3GAN (Fortsetzung)" layout="tight"}

<!--en:d05abed0-2-->
[Das Trainingsverfahren im Code]{.kicker}

<!--en:1db90cfe-->
`generator_step` macht mit dem aktuellen Kritiker einen Schritt für den Generatorverlust:

<!--en:043c6037-->
::: {.slide title="Benchmark zur Modusabdeckung: 25 Gauß-Verteilungen auf einem 5×5-Gitter"}

<!--en:5e7332f1-->
[Versuchsaufbau]{.kicker}

<!--en:72c9740d-->
- **$5 \times 5$-Gitter aus Gauß-Verteilungen (Gitterabstand und Standardabweichung: $2.0, \sigma = 0.05$):** Generator (ReLU-MLP mit 3 Schichten, $64 \to 256 \to 256 \to 2$) und Kritiker mit Leaky ReLU. Training über $10{,}000$ Schritte mit Adam und **$\beta_1 = 0, \beta_2 = 0.99$**.

<!--en:e5d93ca4-->
- **Faktorieller $2 \times 2$-Versuch (3 Zufallsstartwerte je Variante):** Jede Zielfunktion (`NS` oder `pairing`) wird mit jeder Strafstärke ($\gamma = 0$ oder $\gamma = 1$) kombiniert. Als Kontrolle dient ein Lauf mit überkritischer Dämpfung bei $\gamma = 10$.

<!--en:5302c065-->
- **Vier quantitative Metriken:** Abgedeckte Modi (von $25$), Anteil modusnaher Samples (im Umkreis von $3\sigma = 0.15$), mittlerer Abstand zum nächsten Zentrum und Steilheit des Kritikers $\mathbb{E}_q[\|\nabla_x D(x')\|^2]$.

<!--en:14467407-->
::: {.slide title="Ergebnisse für 25 Gauß-Verteilungen: Strafterme verbessern die Abdeckung" layout="code"}

<!--en:c0b0d553-->
[Quantitativer Vergleich]{.kicker}

<!--en:b3c4d662-->
Die Auswertung von $10{,}000$ Samples bei Schritt $10{,}000$ zeigt den Beitrag der einzelnen Komponenten:

<!--en:e89e817b-->
- **Gradientenstrafen ($R_1 + R_2$ bei $\gamma = 1$) verbessern die Modusabdeckung:** Ohne Strafterme ($\gamma = 0$) bleiben die Gradienten des Kritikers groß ($\mathbb{E}_q\|\nabla_x D\|^2 \in [0.7, 17]$) und die Abdeckung schwankt ($12\text{--}25$ Modi). Mit $\gamma = 1$ sinkt die Steilheit des Kritikers um drei Größenordnungen ($\sim 10^{-3}$). **Jeder Lauf deckt alle $25$ Modi ab**, sowohl mit `NS` als auch mit `pairing`.

<!--en:9c10ac9f-->
::: {.slide title="Modusabdeckung und überkritische Dämpfung im Bild"}

<!--en:b71a2f41-->
[Räumliche Verteilungen]{.kicker}

<!--en:68eac498-->
- **Mit und ohne Strafterme:** Beide Varianten mit $\gamma = 1$ (`NS + R1 + R2` und `pairing + R1 + R2`) besetzen alle $25$ Gitterzentren. Ohne Strafterme fehlen Modi oder die Cluster sind ungleichmäßig besetzt.

<!--en:1293cab6-->
- **Warum dünne Stege benachbarte Modi verbinden:** Ein stetiger ReLU-Generator bildet den zusammenhängenden latenten Gauß-Raum $\mathcal{N}(0, I_{64})$ auf eine zusammenhängende Fläche ab. Daher bleiben zwischen den Modi dünne Verbindungen bestehen.

<!--en:e165e485-->
- **Überkritische Dämpfung ($\gamma = 10$):** Ein zu flacher Kritiker ($\mathbb{E}_q\|\nabla_x D\|^2 \sim 10^{-4}$) verlangsamt die Konvergenz. Nur $20\text{--}30\%$ der Samples liegen im Umkreis von $3\sigma$ um einen Modus, selbst nach $10{,}000$ Schritten.

<!--en:7797546c-->
::: {.slide title="Diagnostischer Fallstrick: Ein ruhiger Verlustverlauf belegt keine Konvergenz" layout="tall-output"}

<!--en:531426db-->
[Fehlinterpretation beim Überwachen des Trainings]{.kicker}

<!--en:e6a19a5a-->
Der Vergleich des Anteils modusnaher Samples mit dem Kritikerverlust an allen $8$ Checkpoints (alle $1{,}250$ Schritte) zeigt, wie leicht der Verlustverlauf in die Irre führen kann:

<!--en:e67e6383-->
**Warum der Gleichgewichtswert des Kritikerverlusts keine Konvergenz belegt:** In allen Läufen mit Strafterm liegt der Kritikerverlust beim exakten Gleichgewichtswert ($2\log 2 \approx 1.39$ für `NS`, $\log 2 \approx 0.69$ für `pairing`), und zwar schon am ersten Checkpoint bei Schritt $1{,}250$. Zu diesem Zeitpunkt liegen nur **$1\text{--}3\%$** der generierten Samples in der Nähe eines Modus. Da $R_1 + R_2$ die Funktion $D(x)$ überall nahezu flach hält, bestätigt der Gleichgewichtswert des Verlusts nur die Dämpfung des Kritikers. Er belegt nicht, dass $q$ mit $p$ übereinstimmt.

<!--en:2df0d67b-->
::: {.slide title="Verständnisfragen zu Minimax-Dynamik und Straftermen"}

<!--en:929b7671-->
[Verständnisprüfung]{.kicker}

<!--en:5cfed2a1-->
Fragen zur Rotationsdynamik von Minimax-Spielen und zu um null zentrierten Gradientenstraftermen:

<!--en:c1fb0c56-->
1. **Stabilitätsschwelle für diskrete Euler-Schritte:** Beim Dirac-GAN mit Strafterm und logistischem Verlust ($\ell'(0) = \frac{1}{2}$) hat die Jacobi-Matrix des kontinuierlichen Systems die Eigenwerte $\nu_{1,2} = -\frac{\gamma}{2} \pm i\sqrt{\ell'(0)^2 - \gamma^2/4}$, wenn die Dämpfung unterkritisch ist ($\gamma < 1$). Unter welcher Schranke muss die Schrittweite $h$ liegen, damit simultane diskrete Euler-Schritte konvergieren ($|1 + h\nu_{1,2}| < 1$)? Welchen Zahlenwert hat die Schranke bei $\gamma = 0.3$?

<!--en:28d5853c-->
2. **Verschiebung einer Gauß-Verteilung mit Strafterm:** Gegeben seien $p = \mathcal{N}(0, 1)$ und $q = \mathcal{N}(\mu, 1)$ mit kleiner Verschiebung $\mu = 0.2$ und logistischem Verlust. Um welchen Faktor verringert $\gamma = 1$ die optimale Steigung $c^\star$ des linearen Kritikers gegenüber $\gamma = 0$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:dc114740-->
1. Wegen $\nu_1 \nu_2 = |\nu|^2 = \ell'(0)^2$ und $2\operatorname{Re}(\nu) = -\gamma$ gilt $|1 + h\nu|^2 = 1 - h\gamma + h^2\ell'(0)^2 < 1$ $\iff$ $h < \frac{\gamma}{\ell'(0)^2} = \mathbf{4\gamma}$. Bei $\gamma = 0.3$ konvergieren simultane Euler-Schritte also für **$h < 1.2$**.

<!--en:17ba2658-->
2. Wegen $c^\star = -\frac{\mu}{1 + 4\gamma}$ verringert $\gamma = 1$ sowohl die Steigung des Kritikers als auch den lokalen quadratischen Wert $\frac{\mu^2}{4 + 16\gamma}$ um den Faktor $1 + 4(1) = \mathbf{5\times}$.

<!--en:3923d6bc-->
::: {.slide title="Verständnisfragen zu Minimax-Dynamik und Straftermen (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständnisprüfung]{.kicker}

<!--en:56978812-->
3. **Paarbildung beim Dirac-GAN:** Ändert sich das Vektorfeld des Dirac-GAN ohne Strafterm, wenn der relativistische Paarverlust $\ell(D_\psi(0) - D_\psi(\theta))$ den logistischen Verlust für einzelne Samples ersetzt?

<!--en:23db34b8-->
**Lösung:**

<!--en:4a9026d1-->
3. **Nein.** Wegen $D_\psi(0) = 0$ unterscheidet sich die Zielfunktion für die Paarbildung $\ell(D_\psi(0) - D_\psi(\theta)) = \ell(-\psi\theta)$ von der Zielfunktion für einzelne Samples $\ell(0) + \ell(-\psi\theta)$ nur um die Konstante $\ell(0)$. Das rotierende Vektorfeld ist daher genau dasselbe.

<!--en:4b595563-->
::: {.slide title="Zusammenfassung: Dynamik, Gradientenstrafen und R3GAN"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:25042699-->
| Dynamik oder Mechanismus | Mathematische Form | Eigenwerte der Jacobi-Matrix oder Wert | Trainingsverhalten und praktische Bedeutung |

<!--en:fb30d42d-->
| **Fluss des Dirac-GAN ohne Strafterm** | $\dot{\theta} = \psi\ell'(-\psi\theta)$, $\dot{\psi} = -\theta\ell'(-\psi\theta)$ | $\nu_{1,2} = \pm i\,\ell'(0)$ (lineares Zentrum) | $\frac{d}{dt}(\theta^2 + \psi^2) = 0$, Kreisbahnen mit konstantem Radius |

<!--en:d94ca99e-->
| **Diskrete simultane Schritte** | $(\theta, \psi) \mapsto (\theta, \psi) + h\,\mathbf{u}(\theta, \psi)$ | $1 \pm i\,h\,\ell'(0)$ ($|\lambda| = \sqrt{1 + h^2\ell'(0)^2} > 1$) | Spiralen nach außen bei jeder Schrittweite $h > 0$ |

<!--en:96136fd7-->
| **Diskrete alternierende Schritte** | Halbschritt des Kritikers $\to$ Halbschritt des Generators | $\det = 1$, auf dem Einheitskreis für $h\ell'(0) \le 2$ | Flächentreue Scherungen, Umlaufbahnen in einem beschränkten Band |

<!--en:3a4d613b-->
| **Um null zentrierte Strafterme ($R_1, R_2$)** | $R_1 + R_2 = \gamma\,\mathbb{E}_m[\|\nabla_x D(x)\|^2]$ | $\nu_{1,2} = -\frac{\gamma}{2} \pm \sqrt{\frac{\gamma^2}{4} - \ell'(0)^2}$ ($\operatorname{Re} < 0$) | Lokale exponentielle Konvergenz für alle $\gamma > 0$ |

<!--en:ca520296-->
| **Lokaler Wert mit Strafterm** | $\frac{\ell'(0)^2}{4}\langle p - q, (|\ell''(0)|m + \gamma\mathcal{L}_m)^{-1}(p - q)\rangle$ | Bei verschobener Gauß-Verteilung um den Faktor $1 + 4\gamma$ kleiner | Tiefpassfilterung von $p - q$ unterhalb der Grenzlänge $2\sqrt{\gamma}$ ($\dot{H}^{-1}(m)$) |

<!--en:d5b411bc-->
| **Trainingsverfahren von R3GAN** | $\operatorname{softplus}(D(x') - D(x)) + \frac{\gamma}{2}(R_1 + R_2)$ | Adam mit $\beta_1 = 0$ und ohne Batchnormalisierung | Verringert die Steilheit des Kritikers um den Faktor $1000\times$ und deckt alle $25$ Modi ab |
