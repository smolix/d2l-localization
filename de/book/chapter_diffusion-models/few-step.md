<!--en:5b1cad36-->
# Generierung in wenigen Schritten

<!--en:1c229abd-->
Ein Diffusionsmodell ruft sein Netz immer wieder auf, um ein Sample zu ziehen. Die Sampler aus :numref:`sec_diffusion-samplers` kamen mit rund fünfzig Aufrufen aus, wo die ancestrale Kette tausend brauchte. Die latenten Modelle aus :numref:`sec_diffusion-latent` machten jeden Aufruf günstiger. Doch fünfzig Aufrufe sind immer noch fünfzig, und der Generator eines GAN (:numref:`chap_gans`) brauchte nur einen. Warum also nicht einen einzigen langen Schritt machen? Das scheitert selbst mit einem guten Netz. Sogar im exakten Geschwindigkeitsfeld führt ein Euler-Schritt der Wahrscheinlichkeitsfluss-ODE von jedem Startpunkt zum Mittelwert der Daten (:numref:`sec_diffusion-integration`). Bei einer Mischung aus drei Punktwolken liegt dieser Punkt zwischen den Wolken, wo keine Daten liegen.

<!--en:850f78ed-->
Die Geschwindigkeit gibt dem Sampler die Bewegungsrichtung vor. Für einen einzigen Schritt muss er aber das Ziel kennen. Deshalb lassen wir das Netz die Lösung der ODE lernen statt ihrer rechten Seite. Die gesuchte Abbildung ordnet einem Punkt zu einem gegebenen Zeitpunkt den Punkt zu, an dem seine Trajektorie zu einem früheren Zeitpunkt verläuft. :citet:`Song.Dhariwal.Chen.ea.2023` lernten den Sprung zu den Daten, indem sie ein trainiertes Diffusionsmodell, den *Lehrer* (engl. teacher), in ein neues Netz, den *Schüler* (engl. student), destillierten. Den Schüler nannten sie *Konsistenzmodell* (engl. consistency model). :citet:`Geng.Deng.Bai.ea.2025` lernten Sprünge über beliebige Intervalle allein aus Daten und nannten das Verfahren *MeanFlow*. Das Training bleibt eine Regression mit einem Netz. Dasselbe Rauschen ergibt immer dasselbe Sample, sodass wir den Schüler Bild für Bild mit seinem Lehrer vergleichen können. In :numref:`sec_gan_beyond` trainierten wir Einschrittgeneratoren gegen einen Kritiker. Auf diesen Ansatz kommen wir am Ende zurück. Zunächst untersuchen wir die Fehler bei einem und bei wenigen Euler-Schritten. Dann leiten wir einen gemeinsamen Verlust für beide Verfahren her und trainieren Schüler am durchgehenden Beispiel und auf Fashion-MNIST.

<!--en:521b2d3b-->
## Lange Euler-Schritte

<!--en:a9700a6f-->
In einem exakten Feld macht ein langer Euler-Schritt einen Fehler. Er folgt der Tangente, während die Trajektorie eine Kurve beschreibt. Bei $t = 1$ ist dieser Fehler am größten. Von jedem Startpunkt aus weist die Rückwärtsbewegung zum Mittelwert der Daten, und ein Schritt der Länge eins endet dort (:numref:`sec_diffusion-integration`). Bei späteren Schritten zeichnen sich die Zielmodi an den Startpunkten bereits ab. Die Richtung stimmt dann besser, doch jeder Schritt kürzt die Kurve weiterhin ab.

<!--en:4d26998e-->
### Euler-Schritte im exakten Feld

<!--en:54548251-->
In :numref:`sec_diffusion-integration` sahen wir einen, zehn und hundert Schritte. Welche Fehler bleiben bei zwei und vier Schritten? `mix.velocity` wertet das exakte Feld der Mischverteilung aus. `d2l.make_pred` macht daraus den Prädiktor, den `d2l.ode_sample` integriert. Wir führen das Euler-Verfahren mit 1, 2, 4 und 10 Schritten von denselben 20 000 gaußverteilten Startpunkten aus. Die Zeilen für zwei und vier Schritte liefern die Vergleichswerte, die die Schüler dieses Abschnitts übertreffen müssen.

<!--en:1fe83372-->
Nach einem Schritt liegt jedes Sample auf dem Mittelwert $(-0.5, -0.7)$. Dort zählt `summary` es als modusfern und ordnet es dem nächsten Modus zu, dem mit dem größten Gewicht. Daher ergeben sich die Gewichte 1/0/0. Zwei Schritte erreichen die Modi, lassen aber etwa ein Viertel der Samples dazwischen liegen. Die Gewichte betragen etwa 0.64/0.22/0.14 statt 0.5/0.3/0.2. Der erste halbe Schritt zielt noch auf den Mittelwert. Diesem liegt der ohnehin stärkste Modus am nächsten, und so gewinnt er an Gewicht. Nach vier Schritten sind noch etwa 6.5% der Samples modusfern, nach zehn etwa 2%. Bei den Daten selbst sind es rund 1%. Da das Feld exakt ist, geht der Überschuss auf den Schrittfehler zurück.

<!--en:e53c3e6a-->
### Reflow

<!--en:f9300b8c-->
Warum krümmen sich die Trajektorien überhaupt? Jedes Trainingsbeispiel verbindet einen Datenpunkt über eine gerade Strecke mit einem unabhängig gezogenen Rauschvektor. Die Strecken verschiedener Beispiele schneiden sich jedoch. An den Schnittpunkten mittelt das Feld ihre Richtungen. Die Trajektorien können einander nicht schneiden und weichen deshalb im Bogen aus (:numref:`sec_mdl-rectified-flow`). Die Krümmung entsteht durch die Paarung von Rauschen und Daten, unabhängig vom Verlust.

<!--en:2dce239f-->
*Reflow* :cite:`Liu.Gong.Liu.2022` ändert die Paarung. Jeder Rauschvektor bildet ein Paar mit dem Sample, das ein trainiertes Modell von dort aus erreicht. Mit diesen Paaren wird das Netz erneut trainiert. Jede neue Strecke ist die Sehne einer Trajektorie des alten Modells. Da sich diese Trajektorien nie schneiden, schneiden sich auch ihre Sehnen nur selten. :numref:`subsec_mdl-reflow` zeigt, dass Reflow mit dem exakten Feld die Datenverteilung erhält und die Transportkosten nie erhöht. Bei Daten in Form zweier Halbmonde schneidet das mit Reflow trainierte Modell mit einem Euler-Schritt ebenso gut ab wie das ursprüngliche mit vielen.

<!--en:b6af0b58-->
Für ein gelerntes Feld ist die Garantie schwächer. Die neuen Paare enden bei Samples des Modells statt bei den Daten. Dadurch erbt das mit Reflow trainierte Modell die Fehler des ursprünglichen Modells. Bei den beiden Halbmonden reproduziert es dessen Samples samt ihren Mängeln (:numref:`subsec_mdl-reflow`). Ebenso gibt ein Bildmodell jeden Mangel seiner Samples weiter. Jede Runde kostet zudem einen Trainingslauf. Ihre besten Ergebnisse mit einem Schritt erzielten :citet:`Liu.Gong.Liu.2022` nach wie vor durch Destillation des mit Reflow trainierten Modells. Im Rest dieses Abschnitts ändern wir die Ausgabe des Netzes und behalten die Paarung der Daten bei.

<!--en:a37a065d-->
## Flussabbildungen

<!--en:40ace210-->
Ein Geschwindigkeitsmodell gibt an, in welche Richtung eine Trajektorie weist. Eine Flussabbildung (engl. flow map) gibt an, wo sie ankommt.

<!--en:269eef68-->
### Das Integral der Geschwindigkeit

<!--en:1c48246f-->
Integrieren wir die Wahrscheinlichkeitsfluss-ODE $d\mathbf{x}_\tau / d\tau =
\mathbf{u}_\tau(\mathbf{x}_\tau)$ von einem Punkt $\mathbf{x}$ zur Zeit $t$ rückwärts bis zu einem früheren Zeitpunkt $s \leq t$. Den Zielpunkt liefert die *Flussabbildung* $\Phi_{t \to s}(\mathbf{x})$. Ist das Feld in $\mathbf{x}$ Lipschitz-stetig, so ist die exakte Flussabbildung bijektiv (:numref:`sec_mdl-vector-fields-trajectories`). Der gesuchte Sampler ist $\Phi_{1 \to 0}$.

<!--en:55ce71ad-->
Integrieren wir die ODE entlang der Trajektorie $\mathbf{x}_\tau$ durch $\mathbf{x}_t$, so können wir die Flussabbildung mithilfe einer Verschiebung schreiben:

<!--en:3c363139-->
wobei $\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$ die *Durchschnittsgeschwindigkeit* (engl. average velocity) entlang der Trajektorie zwischen beiden Zeitpunkten ist. Das Euler-Verfahren ersetzt sie durch die Geschwindigkeit am Startpunkt, $\mathbf{u}_t(\mathbf{x}_t)$. Der Fehler wächst mit der Schrittweite (:numref:`sec_mdl-euler-runge-kutta`). Auf einem Intervall der Länge null stimmen beide überein, $\bar{\mathbf{u}}(\mathbf{x}, t, t) =
\mathbf{u}_t(\mathbf{x})$. Ein Netz für $\bar{\mathbf{u}}$ ist also ein Geschwindigkeitsmodell mit einer zusätzlichen Eingabe, der Endzeit.

<!--en:332f2d8a-->
Schon der einfachste Fall zeigt, welche Information die Durchschnittsgeschwindigkeit der momentanen Geschwindigkeit voraushat. Die Daten seien eindimensional und gaußverteilt mit Mittelwert $m$ und Standardabweichung $\varsigma$. Dann ist $x_t$ gaußverteilt mit Mittelwert $(1 - t)\,
m$ und Standardabweichung $k_t = \sqrt{(1 - t)^2 \varsigma^2 + t^2}$. In einer Dimension können Trajektorien einander nicht überholen. Daher erhält der Fluss das Quantil jedes Punktes. Bei einer Gauß-Verteilung legt die standardisierte Position $(x - (1 - t) m) / k_t$ das Quantil fest. Die Flussabbildung ist also affin:

<!--en:230810bd-->
Für $m = 2$ und $\varsigma = 0.5$ bildet der vollständige Sprung $\Phi_{1 \to 0}(x) = 2 +
0.5\, x$ standardnormalverteiltes Rauschen auf die Daten $\mathcal{N}(2, 0.5^2)$ ab. Die Durchschnittsgeschwindigkeit beträgt $\bar{u}(x, 0, 1) = 0.5\, x - 2$. Die Geschwindigkeit am Start ist $u_1(x) = x - 2$. Der zugehörige Euler-Schritt ergibt für jedes $x$ den Wert 2. Die Durchschnittsgeschwindigkeit entfernt nur die Hälfte des Rauschens, denn die halbe Streuung des Rauschens muss als Streuung der Daten erhalten bleiben.

<!--en:2feb08c7-->
Wie kann ein Netz $\Phi$ lernen, wenn niemand die Werte vorgeben kann? Wir nutzen eine Identität, die nur die wahre Flussabbildung erfüllt, und leiten daraus einen Verlust ab. Für jede Art, die beiden Zeitpunkte entlang einer Trajektorie zu verschieben, gibt es eine solche Identität (:numref:`fig_diffusion-flow-map`). Sei $\mathbf{x}_\tau$ die Trajektorie durch $\mathbf{x}$ zur Zeit $t$, und sei $s \leq r \leq t$.

<!--en:30435d4c-->
Nach der *Kompositionsidentität* ergeben zwei Sprünge zusammen einen einzigen, $\Phi_{r \to s}(\Phi_{t \to r}(\mathbf{x})) = \Phi_{t \to s}(\mathbf{x})$. Ein darauf beruhender Verlust braucht keine Ableitung, nur das Modell zu einem Zwischenzeitpunkt. Der Zielwert ist so genau wie die kürzeren Sprünge, aus denen er sich zusammensetzt. Progressive Destillation :cite:`Salimans.Ho.2022` und Shortcut-Modelle :cite:`Frans.Hafner.Levine.ea.2025` nutzen diese Identität.

<!--en:c9cf5b9f-->
Die *lagrangesche* Identität hält den Start fest und verschiebt das Ende. Der Bildpunkt bewegt sich dann wie ein Teilchen mit dem Feld, $\partial_s \Phi_{t \to s}(\mathbf{x}) = \mathbf{u}_s(\Phi_{t \to s}(\mathbf{x}))$. Ein darauf beruhender Verlust braucht die Geschwindigkeit am Zielpunkt des Sprungs. Ein Trainingsbeispiel liefert seine bedingte Geschwindigkeit nur am eigenen Punkt $\mathbf{x}_t$. Die Geschwindigkeit am Ziel muss also ein Lehrer liefern oder die Flussabbildung selbst (Übung :exref:`ex_few-step-lagrangian`).

<!--en:843e9b63-->
Die *eulersche* Identität hält das Ende fest und verschiebt den Start entlang der Trajektorie. Alle Punkte derselben Trajektorie haben dasselbe Bild:

<!--en:f41aab77-->
Ein darauf beruhender Verlust braucht ein Jacobi-Vektor-Produkt (engl. Jacobian-vector product, JVP) und die Geschwindigkeit am Start, wo das Trainingsbeispiel sie liefert. In den Zielwert geht ein einziges Paar von Zeitpunkten ein, ohne Zwischenzeitpunkt und ohne Schrittweite.

<!--en:e4d43372-->
Im Gauß-Beispiel lassen sich alle drei Identitäten von Hand überprüfen. Die Verhältnisse $k_s / k_t$ lassen sich durch Multiplikation verketten. Verfolgen wir den Punkt $\Phi_{t \to s}(x)$ bei veränderlichem $s$, so behält er seine standardisierte Position und bewegt sich damit wie ein Punkt des Flusses. Schließlich ist die standardisierte Position entlang einer Trajektorie konstant.

<!--en:e2ccd38d-->
![Die Flussabbildung. Links: Trajektorien der Wahrscheinlichkeitsfluss-ODE für eine eindimensionale Mischung zweier Gauß-Verteilungen, vom Rauschen bei $t = 1$ (oben) zu den Daten bei $t = 0$ (unten). Die Sehne von $(\mathbf{x}_t, t)$ nach $(\mathbf{x}_s, s)$ entspricht einem Sprung der Flussabbildung. Rechts: drei Sichtweisen auf eine Trajektorie. Zwei Sprünge über $r$ ergeben einen (Komposition). Der Start bei $t$ bleibt fest und das Ende bewegt sich (lagrangesch). Das Ende bei $s$ bleibt fest und der Start bewegt sich (eulersch).](../img/mdl-diffusion-flow-map.svg)

<!--en:2324de23-->
Zusammen mit der Bedingung bei $s = t$ bestimmt jede der drei Identitäten die Flussabbildung. Dort ist die Abbildung die Identität und ihre Steigung in $s$ die Geschwindigkeit. Die drei unterscheiden sich nur darin, was für den Verlust zu berechnen ist. Die Komposition ist am einfachsten. Nur die eulersche Identität kommt sowohl ohne Zwischenzeitpunkt als auch ohne die Geschwindigkeit am Zielpunkt aus. Dafür braucht sie ein Jacobi-Vektor-Produkt. Sowohl Konsistenzmodelle als auch MeanFlow nutzen die eulersche Identität. :citet:`Boffi.Albergo.VandenEijnden.2025` ordnen die Verfahren mit wenigen Schritten danach, welche Identität sie durchsetzen.

<!--en:b6303e1c-->
### Konsistenzmodelle

<!--en:0dd99107-->
Das erste Verfahren mit einer Flussabbildung beschränkte sich auf den Sprung zu den Daten. Ein *Konsistenzmodell* :cite:`Song.Dhariwal.Chen.ea.2023` ist ein Netz $f(\mathbf{x}, t)$, das darauf trainiert wird, zu jedem Zeitpunkt $\Phi_{t \to 0}(\mathbf{x})$ auszugeben. Beim Training soll die Ausgabe gleich bleiben, wenn sich die Eingabe entlang einer Trajektorie bewegt. Es gilt also $\frac{d}{dt} f(\mathbf{x}_t, t) = \mathbf{0}$ mit $f(\mathbf{x}, 0) = \mathbf{x}$. Das ist die eulersche Identität :eqref:`eq_diffusion-eulerian` für $s = 0$.

<!--en:3eae2600-->
Bei der *Konsistenzdestillation* lassen Song et al. den Lehrer einen ODE-Schritt zwischen benachbarten Zeitpunkten eines Gitters machen. Die Ausgaben des Schülers an diesen beiden Punkten sollen übereinstimmen. Als Zielwert dient die Ausgabe am früheren Punkt. Sie wird mit einem gleitenden Mittelwert der Schülergewichte berechnet und festgehalten. Verfeinern wir das Gitter, geht die Differenz geteilt durch die Schrittweite in die obige Ableitung über. :citet:`Lu.Song.2025` verwenden diese direkt. Das *Konsistenztraining* kommt ohne Lehrer aus. Es ersetzt dessen Geschwindigkeit durch die Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$ entlang der geraden Strecke jedes Trainingsbeispiels, also durch die bedingte Geschwindigkeit des Flow-Matchings.

<!--en:72593339-->
Eine Auswertung eines Konsistenzmodells ergibt ein Sample. Ein zweiter Aufruf kann keinen Sprung von $t = 1$ nach $t = 0.5$ ausführen, da das Modell nur die Endpunkte der Trajektorien kennt. Stattdessen wird das Sample *erneut verrauscht*. Frisches Rauschen versetzt es zurück nach $t = 0.5$, also $\mathbf{x}_{0.5} = 0.5\, \mathbf{x}_0 + 0.5\,
\boldsymbol\epsilon'$. Von dort, wo die Trajektorien weniger gekrümmt sind, bildet das Modell es erneut auf die Daten ab. Eine Flussabbildung mit beiden Zeitpunkten kommt ohne erneutes Verrauschen aus. Zwei Auswertungen entsprechen zwei Sprüngen auf derselben Trajektorie. Das Sample hängt weiterhin allein vom anfänglichen Rauschen ab.

<!--en:5ed1008e-->
### Durchschnittsgeschwindigkeit und MeanFlow-Identität

<!--en:234bb6b8-->
Drücken wir die Flussabbildung wie in :eqref:`eq_diffusion-flow-map` durch die Durchschnittsgeschwindigkeit aus, so wird die eulersche Identität zum Regressionsziel. Wir multiplizieren die Durchschnittsgeschwindigkeit mit der Länge ihres Intervalls und erhalten das Integral der Geschwindigkeit. Dann differenzieren wir entlang der Trajektorie nach $t$ und halten $s$ fest:

<!--en:e534d2e7-->
Die Produktregel auf der linken Seite ergibt die *MeanFlow-Identität* von :citet:`Geng.Deng.Bai.ea.2025`:

<!--en:6b4c4de3-->
Die Durchschnittsgeschwindigkeit ergibt sich aus der Anfangsgeschwindigkeit und einer Korrektur. Diese hängt davon ab, wie schnell sich der Durchschnitt ändert, wenn der Start entlang der Trajektorie wandert. Setzen wir :eqref:`eq_diffusion-flow-map` in :eqref:`eq_diffusion-eulerian` ein, sehen wir, dass beide Identitäten übereinstimmen. Die totale Ableitung ist $\partial_t \bar{\mathbf{u}} + (\nabla_{\mathbf{x}} \bar{\mathbf{u}})\,
\mathbf{u}_t$, das Jacobi-Vektor-Produkt des Netzes $(\mathbf{x}, t)
\mapsto \bar{\mathbf{u}}(\mathbf{x}, s, t)$ mit der Tangente $(\mathbf{u}_t, 1)$. Differentiation im Vorwärtsmodus berechnet dieses Produkt zusammen mit der Ausgabe zu etwa den Kosten eines Rückwärtsdurchlaufs. Ein Trainingsschritt kostet damit etwa einen Rückwärtsdurchlauf mehr als bei einem Geschwindigkeitsmodell. Falls es einen Lehrer gibt, kommt eine Auswertung des Lehrers hinzu.

<!--en:1d0eabaf-->
Aus der Identität erhalten wir einen Verlust, indem wir das Netz durch Regression an die rechte Seite anpassen. Diese berechnen wir mit dem Netz selbst und halten sie fest. Die marginale Geschwindigkeit $\mathbf{u}_t$ steht nicht zur Verfügung, doch es gibt zwei Alternativen. Ein Lehrer liefert seine Schätzung $\hat{\mathbf{u}}$ für $\mathbf{u}_t(\mathbf{x}_t)$. Ohne Lehrer genügt die bedingte Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$ des Trainingsbeispiels, aus demselben Grund wie beim Flow-Matching. Der Zielwert hängt affin von der Geschwindigkeit ab, sowohl direkt als auch über die Tangente im Jacobi-Vektor-Produkt. Mitteln wir über die Beispiele, aus denen $\mathbf{x}_t$ hätte entstehen können, erhalten wir daher den Zielwert mit $\mathbf{u}_t(\mathbf{x}_t)$ anstelle der bedingten Geschwindigkeit. Dieser Mittelwert minimiert den quadratischen Fehler. Bezeichnen wir die verwendete Geschwindigkeit, die *Referenzgeschwindigkeit*, mit $\mathbf{u}_{\textrm{ref}}$, so ist der Verlust an einem Trainingspunkt

<!--en:7879cfa7-->
wobei $\mathrm{sg}$, der *Gradientenstopp* (engl. stop-gradient), eine Größe kennzeichnet, die beim Differenzieren des Verlusts als konstant gilt. Das Argument für die Mittelung setzt dies voraus. Es gilt nur, wenn das Netz allein über die Vorhersage in den Verlust eingeht. Würden wir auch durch den Zielwert differenzieren, könnte das Netz die Jacobi-Matrix im Zielwert verändern. Es könnte so die Schwankungen des Zielwerts zwischen Beispielen beeinflussen und den Verlust senken, ohne die Identität zu erfüllen (Übung :exref:`ex_few-step-stop-gradient`). Bei einer Neuimplementierung unterläuft hier am leichtesten ein Fehler, denn der Code unterscheidet sich nur durch einen einzigen Aufruf zum Gradientenstopp.

<!--en:80bf9430-->
Der Aufruf lautet `.detach()`.

<!--en:d31efb89-->
Der Aufruf lautet `jax.lax.stop_gradient`.

<!--en:15b4f873-->
Der Verlust lässt zwei Entscheidungen offen. Woher kommt $\mathbf{u}_{\textrm{ref}}$, und ist die Endzeit $s$ frei oder auf null festgelegt? Die wichtigsten Verfahren entsprechen den vier Kombinationen. Einige setzen die Identität auf anderem Weg als über diesen Verlust durch. Jedes Verfahren ergänzt eigene Details, die :numref:`sec_diffusion-further-reading` beschreibt.

<!--en:ea0749c3-->
| Geschwindigkeit $\mathbf{u}_{\textrm{ref}}$ | ein Zeitpunkt, $s = 0$ | zwei Zeitpunkte, $s \leq t$ |

<!--en:f75485aa-->
| $\hat{\mathbf{u}}$ des Lehrers | Konsistenzdestillation | Destillation der Flussabbildung :cite:`Kim.Lai.Liao.ea.2024` |

<!--en:5773ffa6-->
| $\boldsymbol\epsilon - \mathbf{x}_0$ des Beispiels | Konsistenztraining | MeanFlow |

<!--en:438ccfea-->
Bei $s = 0$ lautet :eqref:`eq_diffusion-flow-map` somit $f(\mathbf{x}, t) =
\mathbf{x} - t\, \bar{\mathbf{u}}(\mathbf{x}, 0, t)$. Die Zelle oben links entspricht also der Konsistenzdestillation in der Form für kontinuierliche Zeit von :citet:`Lu.Song.2025`. Für alle vier Kombinationen implementiert `flow_map_error` den Verlust :eqref:`eq_diffusion-flow-map-loss` für jedes Beispiel auf dem linearen Pfad. Das Netz `u` berechnet $\bar{\mathbf{u}}$, und `teacher=None` wählt die bedingte Geschwindigkeit. Der aufrufende Code wählt die Zeitpunkte und mittelt die Ergebnisse.

<!--en:e9216092-->
Die Ableitung nach $t$ erfasst auch die Zeitmerkmale des Netzes. Deshalb verwenden die Flussabbildungen langsamere Zeitmerkmale als im übrigen Kapitel. Ein Merkmal $\sin(\omega t)$ hat die Ableitung $\omega \cos(\omega t)$. Jedes Merkmal geht also mit seiner Frequenz als Faktor in den Zielwert ein. Die übrigen Netze sehen $t$ mit Frequenzen bis 64 (:numref:`sec_diffusion-denoising`). Die Flussabbildungen verwenden `time_freqs=slow` mit Frequenzen von $10^{-4}$ bis eins, die der folgende Code als `slow` definiert. Hohe Frequenzen führen zu großen Faktoren in der Ableitung. Bettet der Code etwa $1000\, t$ ein, wächst der Faktor bis auf tausend. :citet:`Lu.Song.2025` berichten bei Konsistenzmodellen von Instabilitäten aus diesem Grund.

<!--en:e394de6f-->
Eine Flussabbildung bringt $\mathbf{x}_t$ in einem Sprung nach $\mathbf{x}_s$, also $\mathbf{x}_s = \mathbf{x}_t - (t - s)\, \bar{\mathbf{u}}(\mathbf{x}_t, s,
t)$. Ein Sampler mit wenigen Schritten teilt daher $[0, 1]$ in gleich lange Intervalle und überspringt jedes davon in einem Zug.

<!--en:4436e780-->
Mit der Geschwindigkeit $\mathbf{u}_t$ anstelle von $\bar{\mathbf{u}}$ entspricht dieselbe Funktion dem Euler-Verfahren. Ein Sampler genügt also für Lehrer und Schüler. Die Sprünge gehen nur in eine Richtung. Bei allen Verlusten in diesem Abschnitt ziehen wir $s \leq t$. Um ein Sample so zu invertieren wie die ODE in :numref:`sec_diffusion-samplers`, müssten wir die Abbildung auch in der Gegenrichtung trainieren.

<!--en:3bac0a19-->
Beim Bildexperiment am Ende des nächsten Abschnitts startet der Schüler als Kopie des U-Nets des Lehrers. Der Zeitabstand $t - s$ erhält eine eigene Zeiteinbettung, die eine mit null initialisierte lineare Abbildung durchläuft. Deren Ausgabe addieren wir zur Einbettung des U-Nets. So gilt zu Beginn $\bar{\mathbf{u}}(\mathbf{x}, s, t)
= \hat{\mathbf{u}}(\mathbf{x}, t)$, der Schüler liefert also die Geschwindigkeit des Lehrers.

<!--en:21765bae-->
## Training von Flussabbildungen

<!--en:6dd0082b-->
### Die Mischverteilung mit drei Modi

<!--en:a5a2604c-->
Funktioniert der Verlust, und spielt es eine Rolle, woher die Referenzgeschwindigkeit kommt? An der Mischverteilung lässt sich beides prüfen. Ihr exaktes Feld verrät uns, welche Werte ein perfekter Sampler liefert. Der Lehrer ist ein Geschwindigkeitsmodell wie in :numref:`sec_diffusion-gaussian-paths`. Es ist ein `TimeMLP`, das den Punkt und die Sinusmerkmale von $t$ durch Schichten der Breite 256 führt. Hier trainieren wir es über 20 000 Schritte.

<!--en:f26a1270-->
Drei Schüler belegen drei Zellen von :numref:`tab_diffusion-flow-map-methods`. Alle drei sind Netze vom Typ `TimeMLP` mit der Startzeit $t$ und der Intervalllänge $t - s$ als Eingaben. Wir trainieren sie wie den Lehrer, jeweils sechsmal mit den Zufallsstartwerten 2 bis 7. So können wir Eigenschaften eines Verfahrens von Zufällen eines einzelnen Laufs unterscheiden. Der Konsistenzschüler zieht $t$ aus einer Gleichverteilung, setzt $s = 0$ fest und verwendet die Geschwindigkeit des Lehrers. Bei der Destillation der Flussabbildung zieht der Schüler zwei gleichverteilte Zeitpunkte, nimmt den kleineren als $s$ und verwendet ebenfalls die Geschwindigkeit des Lehrers. Der MeanFlow-Schüler zieht seine Zeitpunkte ebenso, kommt aber ohne Lehrer aus. Für die vierte Zelle, das Konsistenztraining, brauchen wir keinen eigenen Schüler. Seine beiden Merkmale, ein einzelner Zeitpunkt und die Geschwindigkeit des Beispiels, prüfen wir bereits mit je einem der anderen Schüler.

<!--en:b55f9ce4-->
Welche Verluste sind zu erwarten? Der Zielwert eines destillierten Schülers ist eine Funktion des Punktes und der Zeitpunkte. Sein Verlust kann daher gegen null gehen. In den Zielwert von MeanFlow geht die Geschwindigkeit des jeweiligen Beispiels zweimal ein, einmal als $\mathbf{u}_{\textrm{ref}}$ und einmal als Richtung der Tangente. Der Verlust kann nicht unter die Varianz dieses Zielwerts zwischen Beispielen am selben Punkt zu denselben Zeiten sinken.

<!--en:02d839cd-->
Die destillierten Schüler erreichen Verluste nahe null. Beim Schüler mit zwei Zeitpunkten sind es etwa 0.03, beim Konsistenzschüler 0.06 bis 0.08. MeanFlow endet in den sechs Läufen bei 76 bis 85, etwa dem Tausendfachen des Verlusts des Konsistenzschülers. Der größte Teil davon könnte auf unvermeidbare Unterschiede zwischen den Beispielen zurückgehen, die kein Netz ausgleichen kann. Die Codezelle misst diese Untergrenze nicht. Der Wert sagt also wenig über den Schüler aus. MeanFlow-Läufe beurteilen wir anhand ihrer Samples.

<!--en:7e0e784e-->
Alle Modelle starten nun an denselben 20 000 Punkten wie im Euler-Experiment. Der Lehrer macht fünfzig Euler-Schritte. Der Konsistenzschüler macht einen Sprung oder zwei mit erneutem Verrauschen bei $t = 0.5$. Die Schüler mit zwei Zeitpunkten machen einen, zwei oder vier Sprünge. Eine neue Stichprobe aus der Mischverteilung zeigt, welche Werte ein perfekter Sampler liefern würde. Jede Tabellenzeile gibt für jeden Lauf den größten Fehler unter den drei Modusgewichten an. Bei einem Schüler stehen dort sechs Werte in der Reihenfolge der Zufallsstartwerte, bei den Daten und beim Lehrer je einer. Es folgen die Wertebereiche über alle Läufe für zwei Größen. Die erste ist der Anteil modusferner Samples, deren Abstand von jedem Moduszentrum mehr als drei Standardabweichungen beträgt. Die zweite ist die Streuung, der mittlere quadrierte Abstand der übrigen Samples zum Zentrum ihres jeweiligen Modus (:numref:`sec_diffusion-denoising`).

<!--en:14aa570c-->
Nach einer Auswertung verteilt jeder Schüler seine Samples in jedem Lauf auf die drei Modi. Die Gewichte weichen um höchstens 0.007 von 0.5/0.3/0.2 ab. Eine neue Stichprobe aus 20 000 Punkten weicht um 0.004 bis 0.005 ab. Ein Sprung trifft die Gewichte also bis auf etwa das Stichprobenrauschen. Der destillierte Schüler mit zwei Zeitpunkten bleibt bei zwei und vier Sprüngen ebenso nahe daran. In der neuen Stichprobe sind rund 1% der Samples modusfern. Sie liegen in den Ausläufern der Gauß-Verteilungen jenseits von drei Standardabweichungen. Beim Lehrer mit fünfzig Schritten beträgt der Anteil rund 1.5%. Beide destillierten Schüler liefern nach einer Auswertung rund 2% modusferne Samples und erhalten die Streuung der Daten mit 0.45 bis 0.46 gegenüber 0.47. Eine zweite Auswertung senkt den Anteil modusferner Samples auf das Niveau des Lehrers oder darunter, sowohl mit erneutem Verrauschen (1.2% bis 1.5%) als auch mit einem zweiten Sprung (1.0% bis 1.1%). Bei der Mischverteilung hat das erneute Verrauschen nach diesen Kennzahlen keinen messbaren Nachteil. Der zweite Sprung erhält allerdings die Paarung jedes Samples mit seinem Rauschvektor. MeanFlow liefert nach einem Sprung rund 4% modusferne Samples. Die Abbildung zeigt einen zweiten Mangel. Die Modi liegen zu eng um ihre Zentren, mit einer Streuung von 0.31 bis 0.34 nach einem Sprung und 0.35 bis 0.40 nach zwei und vier Sprüngen. Daher liegen nach vier Sprüngen auch nicht mehr Samples fern der Modi als bei den Daten selbst.

<!--en:74d1d19f-->
Die sechs Läufe zeigen auch, wo ein einzelner Lauf in die Irre führen würde. Bei zwei und vier Sprüngen weichen die Gewichte des destillierten Schülers in jedem Lauf um höchstens 0.006 ab. Bei MeanFlow ergibt sich ein anderes Bild. Nach vier Sprüngen liegen ein oder zwei der sechs Läufe um 0.013 bis 0.023 daneben, das Zwei- bis Sechsfache der Abweichung einer neuen Stichprobe. Bei den übrigen beträgt die Abweichung höchstens 0.010. Nach mehreren Sprüngen hängen die Gewichte bei MeanFlow also vom Trainingslauf ab. Ein einzelner Lauf ist dafür wenig aussagekräftig. Beim destillierten Schüler beobachten wir diese Abhängigkeit nicht.

<!--en:46f9bd89-->
In diesem Reiter weicht der Lauf mit Zufallsstartwert 6 ab. Die fünfte Tabellenspalte zeigt nach vier Sprüngen einen Fehler von 0.023.

<!--en:3d45b42b-->
In diesem Reiter weichen die Läufe mit den Zufallsstartwerten 3 und 6 ab. Die zweite und die fünfte Tabellenspalte zeigen nach vier Sprüngen Fehler von 0.020 und 0.013. Beim Zufallsstartwert 3 beträgt die Abweichung schon nach zwei Sprüngen 0.012.

<!--en:5873a2dc-->
Der Vergleich grenzt die Ursache ein. Die beiden Schüler mit zwei Zeitpunkten verwenden dasselbe Netz, dieselben Zeitpaare und dieselben Aktualisierungen. Sie unterscheiden sich nur in der Geschwindigkeit im Zielwert. Die Wertebereiche ihrer Streuungen überlappen sich nie. Jeder Lauf mit Destillation liegt nach einem Sprung um mehr als 0.1 über jedem MeanFlow-Lauf, nach zwei und vier Sprüngen um mindestens 0.06. Innerhalb der sechs Läufe eines Schülers beträgt die Spannweite dagegen höchstens 0.013 (Destillation) bzw. rund 0.04 (MeanFlow). In diesem Versuch geht die verringerte Streuung also auf das Training mit den Geschwindigkeiten der Beispiele zurück. Warum ein verrauschter Zielwert eine zu enge statt einer verrauschten Abbildung ergibt, zeigen die Läufe nicht. Lernt ein Schüler den Korrekturterm aus :eqref:`eq_diffusion-meanflow` nur teilweise, so entspricht sein Sprung zum Teil einem Euler-Schritt. Große Euler-Schritte ziehen die Modi genau auf diese Weise zusammen (:numref:`sec_diffusion-samplers`). Das ist eine plausible Deutung, kein Versuchsergebnis. Bei der Mischverteilung lohnt sich ein Lehrer. Training allein aus Daten spart zwar dessen Trainingslauf, doch der Schüler muss über deutlich stärker verrauschte Zielwerte mitteln.

<!--en:e44c6205-->
### Fashion-MNIST

<!--en:5a8b5886-->
Als Lehrer für die Bilder dient ein unbedingtes U-Net für die Geschwindigkeit, `UNet(ch=64)` mit rund 1.2 Millionen Parametern. Wir trainieren es über 10 000 Aktualisierungen mit den langsamen Zeitmerkmalen, da der Schüler als Kopie des Lehrers startet. `train_regression` gibt mit dem trainierten Netz auch einen gleitenden Mittelwert der Gewichte zurück. Das Netz mit diesen gemittelten Gewichten dient als Lehrer.

<!--en:fd0aba3f-->
Wir bewerten 1000 Samples mit dem Fréchet-Abstand (FD) aus :numref:`sec_diffusion-images` gegenüber 1000 Testbildern. Ein Fashion-MNIST-Klassifikator extrahiert Merkmale aus beiden Mengen. Der FD misst den Abstand zwischen den an diese Merkmale angepassten Gauß-Verteilungen. Als unterer Referenzwert dient der Abstand zwischen zwei disjunkten Mengen von je 1000 Testbildern.

<!--en:caed21cb-->
Als Schüler verwenden wir das oben definierte `UNetFlowMap` mit einer Kopie des Lehrers. Wir trainieren ihn mit der Geschwindigkeit des Lehrers und zwei freien Zeitpunkten. Für die Hälfte der Paare gilt $s = t$. Hier ist der Zielwert die Geschwindigkeit des Lehrers selbst. Der Schüler startet als Kopie des Lehrers, und diese Paare sollen dafür sorgen, dass er beim Erlernen der Sprünge die Geschwindigkeit behält. Für die andere Hälfte ziehen wir wie bei der Mischverteilung $s < t$. Ausgehend von den trainierten Gewichten lernt der Schüler mit einer Lernrate von $2 \cdot 10^{-4}$ über 5000 Aktualisierungen mit je 128 Bildern. Der Lehrer benötigte mit 10 000 doppelt so viele Aktualisierungen. Wir trainieren zwei Schüler desselben Lehrers mit den Zufallsstartwerten 2 und 3, um den Einfluss des einzelnen Laufs zu untersuchen.

<!--en:07c8f955-->
Von denselben 1000 Rauschbildern aus macht der Lehrer 1 bis 50 Euler-Schritte und jeder Schüler 1, 2 oder 4 Sprünge. Wir tragen den FD gegen die Zahl der Netzauswertungen auf. Ein einziger Schritt des Lehrers sollte beim Mittelwertbild landen. Das entspricht auf Pixelebene dem Fehler bei der Mischverteilung. Sein FD sollte hier also weit über allen anderen Werten liegen.

<!--en:f9f09d61-->
Der FD des Lehrers sinkt von etwa 145 bis 170 bei einem Euler-Schritt auf etwa 80 bei zwei, rund 7 bei zehn und rund 2.6 bei fünfzig Schritten. Der untere Referenzwert liegt bei etwa 0.8. Bei wenigen Auswertungen liegen die Schüler weit unter dieser Kurve. Ein Sprung erreicht 11 bis 12, das Vier- bis Fünffache des FD des Lehrers bei fünfzig Schritten. Zwei Sprünge erreichen knapp das Doppelte und vier das 1.4- bis 1.5-Fache, nahe am Wert des Lehrers nach zwanzig Schritten. Lesen wir das Diagramm horizontal, so entsprechen zwei Sprünge eines Schülers einem Wert zwischen zehn und zwanzig Euler-Schritten des Lehrers, ein Sprung einem Wert zwischen vier und zehn. Bei jedem Budget unterscheiden sich die Schüler um höchstens ein Zehntel ihres FD. Die Vergleiche gelten daher für beide.

<!--en:265c9156-->
Ein destillierter Schüler kann bestenfalls die ODE seines Lehrers reproduzieren. Sein Ziel ist daher der FD der exakten Lehrer-ODE, dem sich fünfzig Euler-Schritte von oben nähern. Der untere Referenzwert liegt tiefer. Welcher Anteil der verbleibenden Lücke auf das Training zurückgeht, zeigen unsere Läufe nicht. Jeder Schüler sah 640 000 Bilder. In der Literatur werden Schüler mit viel mehr Bildern trainiert und kommen ihren Lehrern in einem Schritt deutlich näher :cite:`Song.Dhariwal.Chen.ea.2023,Lu.Song.2025`.

<!--en:448f60f6-->
Die folgenden Zeilen gehen von denselben acht Rauschbildern aus. Sie zeigen den Lehrer mit fünfzig und mit zwei Euler-Schritten, dann den Schüler mit Zufallsstartwert 2 mit einem und mit zwei Sprüngen.

<!--en:cef5a803-->
Die Zeilen lassen sich Spalte für Spalte vergleichen. Mit zwei Euler-Schritten zeichnet der Lehrer graue Mittelwerte von Kleidungsstücken. Hier zeigt sich der Fehler der Mischverteilung auf Pixelebene. In manchen Spalten stimmt nicht einmal die Art des Kleidungsstücks mit dem Ergebnis nach fünfzig Schritten überein. Aus demselben Rauschen zeichnet der Schüler schon mit einer Auswertung in fast jeder Spalte dasselbe Kleidungsstück wie der Lehrer. Bei einem Sprung gehen Details verloren. Aufdrucke und Streifen verschwimmen, hier und da franst ein Umriss aus, und vereinzelt entstehen Flecken. Ein zweiter Sprung kommt in den meisten Spalten den fünfzig Schritten des Lehrers nahe.

<!--en:b01123f9-->
## Verteilungsabgleich

<!--en:b53f5081-->
Wie gut eine Flussabbildung sein kann, hängt von den Trajektorien ab, an denen sie lernt. Einem destillierten Schüler setzt die ODE seines Lehrers die Grenze. Bei einem Schüler, der allein aus Daten lernt, kommt es darauf an, wie gut er das exakte Feld lernt. Bei der Mischverteilung reichte das für die Streuung nicht aus.

<!--en:744ba2b7-->
Der andere Ansatz trainiert einen Einschrittgenerator $g_{\boldsymbol\theta}(\mathbf{z})$ darauf, eine Verteilung statt einer Trajektorie nachzubilden. Passen wir ihn mit dem quadratischen Fehler an die Samples eines Lehrers an, erzeugt er Mittelwerte. Kann das Netz die lange Berechnung des Lehrers nicht in einem Durchlauf nachbilden, erscheinen ihm mehrere Rauschvektoren gleich. Der quadratische Fehler begünstigt dann den Mittelwert der zugehörigen Bilder (:numref:`sec_gan_beyond`). Dort haben wir auch zwei Gegenmittel kennengelernt. Beim ersten vergleicht ein Kritiker die Ergebnisse mit echten Bildern :cite:`Sauer.Lorenz.Blattmann.ea.2023`. Das zweite, die Destillation durch Verteilungsabgleich (engl. distribution matching distillation, DMD) :cite:`Yin.Gharbi.Zhang.ea.2024`, kommt ohne Kritiker aus. Sie verrauscht die Samples des Generators gemäß $\mathbf{x}_t = (1 - t)\,
g_{\boldsymbol\theta}(\mathbf{z}) + t\, \boldsymbol\epsilon$ und verringert die Kullback-Leibler-Divergenz ihrer Verteilung $q_{\boldsymbol\theta, t}$ von der Verteilung der verrauschten Daten $p_t$. Ihr Gradient lautet

<!--en:75b29f79-->
Der Gradient ergibt sich aus dem Score des Schülers abzüglich des Scores des Lehrers. Ein Abstiegsschritt verschiebt jedes verrauschte Sample dorthin, wo die Daten mehr Masse haben, als der Schüler ihnen zuweist. Der Lehrer liefert $\nabla \log p_t$, denn auf dem linearen Pfad ist der Score eine affine Funktion der Geschwindigkeit (:numref:`sec_diffusion-gaussian-paths`). Ein zweites Diffusionsmodell, das parallel mit den Samples des Schülers trainiert wird, liefert $\nabla \log q_{\boldsymbol\theta, t}$ (:numref:`fig_diffusion-dmd`). Wie gut dieser Schüler werden kann, hängt von der Verteilung des Lehrers ab und nicht mehr von dessen einzelnen Trajektorien. Nur wenn echte Bilder in den Verlust eingehen, kann der Schüler den Lehrer übertreffen, etwa durch einen Kritiker wie bei der adversariellen Destillation oder als Ergänzung zum Verteilungsabgleich :cite:`Yin.Gharbi.Park.ea.2024` (:numref:`sec_gan-further-reading`). Bei keiner Variante folgt das Training den Trajektorien der ODE. Für einen gegebenen Rauschvektor kann der Generator daher ein anderes Sample als der Lehrer liefern. Inversion und Bearbeitung über die ODE sind dann nicht mehr möglich.

<!--en:7248aa96-->
![Destillation durch Verteilungsabgleich. Der Schüler bildet Rauschen $\mathbf{z}$ in einem Schritt auf ein Sample ab. Anschließend wird das Sample bis zum Zeitpunkt $t$ verrauscht. Der Lehrer mit festen Gewichten liefert den Score $\nabla \log p_t$ der verrauschten Daten bei $\mathbf{x}_t$. Ein zweites Diffusionsmodell lernt parallel aus den Samples des Schülers (orange) und liefert den Score $\nabla \log q_{\boldsymbol\theta, t}$ der verrauschten Samples. Die Differenz der beiden Scores ergibt den Gradienten, der den Schüler erreicht (rot).](../img/mdl-diffusion-dmd.svg)

<!--en:8995d7f6-->
Welches Verfahren sich anbietet, hängt vom Rechenbudget ab. Bei fünfzig Auswertungen genügt ein Lehrer mit einem guten ODE-Löser, ein zweiter Trainingslauf erübrigt sich. Kommt es auf die Latenz an, bietet sich die Destillation einer Flussabbildung mit zwei Zeitpunkten aus dem Lehrer an. Auf Fashion-MNIST lagen zwei ihrer Sprünge zwischen zehn und zwanzig Euler-Schritten des Lehrers. Ohne Lehrer trainiert MeanFlow eine Flussabbildung allein aus Daten. Mit dem einfachen quadratischen Fehler, den wir verwendet haben, ergaben sich bei der Mischverteilung jedoch zu enge Modi. Das veröffentlichte Verfahren gewichtet den Verlust neu (Übung :exref:`ex_few-step-stop-gradient`). Auch der Verteilungsabgleich kommt mit oder ohne Kritiker mit einem Schritt aus. Dabei geht allerdings die für Inversion und Bearbeitung benötigte Zuordnung von Rauschen zu Samples verloren.

<!--en:467d7b29-->
Beide Ansätze gehen von einem kontinuierlichen Pfad aus, auf dem eine Geschwindigkeit oder ein Score definiert ist. Ist $\mathbf{x}$ eine Sequenz von Token, gibt es weder diese Geschwindigkeit noch diesen Score. Damit befasst sich :numref:`sec_diffusion-discrete`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:da278d8b-->
Ein einzelner Euler-Schritt der Wahrscheinlichkeitsfluss-ODE von $t = 1$ nach $t = 0$ führt von jedem Startpunkt zum Mittelwert der Daten $\mathbb{E}[\mathbf{x}_0]$, weil er der Anfangstangente einer gekrümmten Trajektorie folgt. Eine Flussabbildung $\Phi_{t \to s}(\mathbf{x}_t) = \mathbf{x}_t - (t - s)\, \bar{\mathbf{u}}(\mathbf{x}_t, s, t)$ sagt dagegen mithilfe der Durchschnittsgeschwindigkeit $\bar{\mathbf{u}}$ die über die ODE-Trajektorie integrierte Verschiebung vorher. Differenzieren nach $t$ ergibt die eulersche Identität in ihrer MeanFlow-Form $\bar{\mathbf{u}} = \mathbf{u}_t - (t - s)\, \frac{d}{dt}\bar{\mathbf{u}}$. Für das Training genügen ein Jacobi-Vektor-Produkt im Vorwärtsmodus und ein Zielwert mit Gradientenstopp. Die Wahl der Referenzgeschwindigkeit $\mathbf{u}_{\textrm{ref}}$ (Schätzung $\hat{\mathbf{u}}$ eines vortrainierten Lehrers oder bedingte Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$) und des Zeitbereichs ($s = 0$ oder $s \le t$) führt vier Verfahren zusammen: Konsistenzdestillation, Konsistenztraining, Destillation der Flussabbildung und MeanFlow. Bei der Mischung mit drei Modi und bei Fashion-MNIST erhält die Destillation der Flussabbildung aus einem Lehrer die Streuung innerhalb der Modi und erreicht mit 2–4 Schülersprüngen die Güte von 10–20 Euler-Schritten des Lehrers. MeanFlow ohne Lehrer zieht dagegen die Modi der Mischung zusammen, wenn es mit einfachem quadratischem Fehler und den verrauschten Geschwindigkeiten einzelner Beispiele trainiert wird.

<!--en:2294243a-->
## Übungen

<!--en:3e4abab7-->
1. **Ein und zwei Euler-Schritte.**
    1. Ein einziger Schritt des deterministischen Samplers von reinem Rauschen bis $t = 0$ ergibt von jedem Startpunkt aus $\mathbb{E}[\mathbf{x}_0]$, unabhängig vom Pfad (:numref:`sec_diffusion-integration`). Gilt das auch für einen Euler-Schritt in $t$, also $\mathbf{x} - \mathbf{u}_1(\mathbf{x})$, auf dem Kosinuspfad $\alpha_t = \cos\frac{\pi t}{2}$, $\sigma_t = \sin\frac{\pi t}{2}$? Was ergibt dieser Schritt? Warum hängt ein Euler-Schritt in $t$ vom Pfad ab, ein Schritt in $\bar\sigma$ hingegen nicht?
    1. Betrachten Sie für den linearen Pfad eindimensionale Daten, die mit gleicher Wahrscheinlichkeit die beiden Werte $\pm a$ annehmen. Zeigen Sie, dass $\mathbb{E}[x_0 \mid x_t = x] = a \tanh\big(a (1 - t) x / t^2\big)$ gilt und dass zwei Euler-Schritte der Länge $\tfrac12$ einen Startpunkt $x$ auf $a \tanh(a x)$ abbilden. Den exakten Fluss für dieses Beispiel finden Sie in :numref:`subsec_mdl-fm-two-point`.
    1. Welcher Anteil der Samples liegt für $a = 2$ nach zwei Schritten höchstens $0.1$ von einem Datenpunkt entfernt? Wohin bildet die exakte Flussabbildung $\Phi_{1 \to 0}$ den Punkt $x$ ab?
   :exlabel:`ex_few-step-one-step-mean`

<!--en:e8965bb6-->
1. [code] **Reflow für die Mischverteilung.** ● Reflow :cite:`Liu.Gong.Liu.2022` trainiert ein Geschwindigkeitsmodell erneut. Jedes Trainingspaar besteht aus einem Rauschvektor und dem Sample, das das aktuelle Modell daraus erzeugt.
    1. Ziehen Sie 20 000 Rauschvektoren, integrieren Sie den Lehrer dieses Abschnitts mit fünfzig Euler-Schritten und trainieren Sie ein neues `TimeMLP` mit den resultierenden Paaren. Verwenden Sie den Geschwindigkeitsverlust und die Einstellungen des Lehrers.
    1. Geben Sie die Modusgewichte und den Anteil modusferner Samples des neuen Modells bei 1, 2, 4 und 10 Euler-Schritten an. Bestimmen Sie außerdem den mittleren Abstand seiner Trajektorien mit fünfzig Schritten von ihren Sehnen. Stellen Sie alle Größen den entsprechenden Werten des Lehrers gegenüber.
    1. Welche Fehler des Lehrers behebt Reflow, welche erbt es und welche kommen hinzu? Wie würde sich eine zweite Runde auf die jeweiligen Fehler auswirken?
   :exlabel:`ex_few-step-reflow`

<!--en:c31bae40-->
1. **Der lagrangesche Verlust.** ● :citet:`Boffi.Albergo.VandenEijnden.2025` trainieren Flussabbildungen mithilfe der lagrangeschen Identität.
    1. Setzen Sie :eqref:`eq_diffusion-flow-map` in $\partial_s
       \Phi_{t \to s}(\mathbf{x}) = \mathbf{u}_s(\Phi_{t \to s}(\mathbf{x}))$ ein. Leiten Sie eine Identität für $\bar{\mathbf{u}}$ her, in die die Geschwindigkeit am Zielpunkt des Sprungs eingeht.
    1. Warum eignet sich die bedingte Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$ nicht für diese Identität, sodass ein Geschwindigkeitsmodell nötig ist? Zeigen Sie, dass die Geschwindigkeit $\bar{\mathbf{u}}(\cdot, s, s)$ der Flussabbildung selbst ausreicht und somit ein einziges Netz sich selbst trainieren kann.
    1. [code] Trainieren Sie einen solchen Schüler auf der Mischverteilung. Vergleichen Sie seinen Anteil modusferner Samples und seine Trainingskosten pro Schritt mit denen des MeanFlow-Schülers.
   :exlabel:`ex_few-step-lagrangian`

<!--en:48baa9c3-->
1. [code] **Gradientenstopp und Gewichtung.** Zwei Details von :eqref:`eq_diffusion-flow-map-loss` bestimmen, wo der Verlust sein Minimum annimmt.
    1. Kopieren Sie `flow_map_error` in eine Codezelle und entfernen Sie den Gradientenstopp am Zielwert. Trainieren Sie den MeanFlow-Schüler auf der Mischverteilung mit diesem Verlust. Geben Sie seine Modusgewichte, den Anteil modusferner Samples und die Streuung nach einem Sprung an. Schreiben Sie den Gradienten des Verlusts aus, wenn auch durch den Zielwert differenziert wird. Bestimmen Sie den Term, der beim Gradientenstopp entfällt. Was kann das Netz ändern, um den Verlust zu senken, ohne :eqref:`eq_diffusion-meanflow` zu erfüllen?
    1. Song et al. berechnen den Zielwert stattdessen mit einem gleitenden Mittelwert der Schülergewichte. Implementieren Sie dies mit `d2l.EMA` und vergleichen Sie die Ergebnisse.
    1. Das MeanFlow-Verfahren für ImageNet :cite:`Geng.Deng.Bai.ea.2025` teilt den quadratischen Fehler $e$ jedes Beispiels durch den beim Differenzieren festgehaltenen Wert $e + c$ mit kleinem $c$. Der Gradient entspricht dadurch dem von $\log(e + c)$. Der bedingte Mittelwert des Zielwerts minimiert den quadratischen Fehler. In der Nähe welchen Werts liegt der Minimierer des gewichteten Verlusts? Ohne Lehrer verteilen sich die Zielwerte an einem verrauschten Punkt der Mischung über die drei Modi. Sagen Sie die Modusgewichte nach einem Sprung vorher. Trainieren Sie dann den MeanFlow-Schüler mit dieser Gewichtung und $c = 10^{-3}$ und prüfen Sie Ihre Vorhersage.
    1. Bei einem Bild der Größe $28 \times 28$ summiert $e$ insgesamt 784 quadratische Fehler. Begründen Sie, warum die Summe bei gleichem $(s, t)$ zwischen Beispielen viel weniger schwankt als in zwei Dimensionen. Die Gewichtung wirkt dann hauptsächlich zwischen verschiedenen Zeitpunkten. Was sagt das Experiment auf der Mischverteilung über das Verfahren für Bilder aus? Welche Schlüsse lässt es nicht zu?
   :exlabel:`ex_few-step-stop-gradient`

<!--en:57410adb-->
1. [code] **Ein Schüler mit Guidance.** ● Classifier-Free Guidance wie in :numref:`subsec_diffusion-images-guided` wertet das Netz zweimal pro Schritt aus. :citet:`Meng.Rombach.Gao.ea.2023` destillieren die Geschwindigkeit mit Guidance in ein einziges Netz. Das halbiert die Kosten jedes Schritts schon vor der Verringerung der Schrittzahl. Trainieren Sie das klassenbedingte U-Net aus :numref:`sec_diffusion-images` mit `time_freqs=slow` wie den Bildlehrer in diesem Abschnitt. Destillieren Sie es in einen Schüler mit zwei Zeitpunkten. Verwenden Sie als Referenzgeschwindigkeit $\mathbf{u}_{\textrm{ref}}$ die Geschwindigkeit mit Guidance $\tilde{\mathbf{u}}$, berechnet mit `d2l.cfg(teacher, c, 2.0)` für zufällige Label $c$. Geben Sie dem Schüler das Label als Eingabe.
    1. Vergleichen Sie den FD und die Übereinstimmung der vorhergesagten mit der gewünschten Klasse. Verwenden Sie für den Lehrer fünfzig Euler-Schritte mit Guidance und für den Schüler einen, zwei und vier Sprünge.
    1. Wie viele Netzauswertungen kostet ein Sample jeweils?
    1. Meng et al. geben dem Schüler auch die Guidance-Stärke als Eingabe. Wie würden Sie die Stärken beim Training ziehen? Was bietet der Schüler dann, was ein bei $\gamma = 2$ destillierter Schüler nicht bietet?
   :exlabel:`ex_few-step-guided-student`

<!--en:8b826e5b-->
1. **Abgleich verrauschter Verteilungen.** ● Die Destillation durch Verteilungsabgleich :cite:`Yin.Gharbi.Zhang.ea.2024` trainiert einen Einschrittgenerator $\mathbf{x} = g_{\boldsymbol\theta}(\mathbf{z})$ entlang des Gradienten :eqref:`eq_diffusion-dmd-gradient`. Dabei ist $q_{\boldsymbol\theta, t}$ die Verteilung der verrauschten Samples $\mathbf{x}_t$.
    1. Leiten Sie :eqref:`eq_diffusion-dmd-gradient` her. Warum verschwindet der Term, der sich aus der Abhängigkeit der Verteilung $q_{\boldsymbol\theta, t}$ selbst von $\boldsymbol\theta$ ergibt?
    1. In einer Dimension sei der Generator $m + b z$ mit $b > 0$, die Daten seien gemäß $\mathcal{N}(\mu, v)$ verteilt. Geben Sie $D_{\textrm{KL}}(q_t \,\|\, p_t)$ auf dem linearen Pfad in geschlossener Form an. Zeigen Sie, dass $m = \mu$ und $b^2 = v$ für jedes $t < 1$ den einzigen Minimierer festlegen.
    1. Die Daten seien gemäß $\tfrac12 \mathcal{N}(-2, 0.25) + \tfrac12 \mathcal{N}(2, 0.25)$ verteilt. Berechnen Sie $D_{\textrm{KL}}(q_t \,\|\, p_t)$ durch numerische Quadratur auf einem Gitter von $(m, b)$ bei $t \in \{0.1, 0.3, 0.5, 0.9\}$ und bestimmen Sie für jedes $t$ den Minimierer. Warum liegt das Minimum nahe bei $\log 2$, wenn $t$ klein ist? Setzen Sie das Ergebnis in Beziehung zum Modusausfall bei GANs (:numref:`subsec_gan_mode_dropping`) und zum Verschmelzen von Modi unter Rauschen (Übung :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging` in :numref:`sec_mdl-fokker-planck-probability-flow`). Was gewinnt ein gaußscher Generator, wenn die Gewichtung der Rauschpegel große $t$ bevorzugt? Was kann er durch keine Gewichtung gewinnen?
   :exlabel:`ex_few-step-matching-noised`

<!--en:5903b8ca-->
[Diskussion](https://d2l.discourse.group/t/36893)

<!--en:27c0d252-->
Generierung in wenigen Schritten: Flussabbildungen, Konsistenz und Destillation<br>**gekrümmte ODE-Trajektorien · Rectified Flow und Reflow · Flussabbildungen und Durchschnittsgeschwindigkeit · MeanFlow-Identität · Konsistenzmodelle · DMD**

<!--en:e9910006-->
::: {.slide title="Warum genügt einem gewöhnlichen Diffusionsmodell kein einzelner Schritt?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:2feb9d48-->
- **Geschwindigkeit $\neq$ Verschiebung:** Ein Geschwindigkeitsnetz $\mathbf{u}_{\boldsymbol\theta}(\mathbf{x}_t, t)$ sagt die momentane Tangente $d\mathbf{x}_t/dt$ zur Zeit $t$ vorher, nicht die Sehne bis $t = 0$.

<!--en:37249119-->
- **Bei gekrümmten Trajektorien verfehlt die Tangente das Ziel:** Selbst im exakten Feld $\mathbf{u}_t$ folgt 1 Euler-Schritt ab $t = 1$ der Geschwindigkeit $\mathbf{u}_1(\mathbf{x}_1) = \mathbf{x}_1 - \mathbb{E}[\mathbf{x}_0]$. Alle Samples fallen auf den globalen Mittelwert $\mathbb{E}[\mathbf{x}_0]$.

<!--en:831c010a-->
- **Zwei Wege zu 1–4 Schritten:** Trajektorien mit **Reflow oder Flussabbildungen** begradigen oder integrieren, oder Dichten mit **DMD oder ADD** direkt aneinander anpassen.

<!--en:974c824b-->
::: {.slide title="Fehler bei wenigen Euler-Schritten im exakten Feld"}

<!--en:9da8fb83-->
[1, 2, 4 und 10 Euler-Schritte auf der zweidimensionalen Mischverteilung]{.kicker}

<!--en:89bd23a5-->
Das Euler-Verfahren mit der exakten Geschwindigkeit `mix.velocity` der Mischverteilung und $20{,}000$ gaußverteilten Startpunkten liefert die Vergleichswerte für Generatoren mit wenigen Schritten:

<!--en:bf5a1771-->
- **1 Euler-Schritt:** $100\%$ der Startpunkte fallen auf $(-0.5, -0.7)$ im leeren Tal zwischen den Modi (`weights` $[1, 0, 0]$, $100\%$ modusferne Samples).

<!--en:ebe49c62-->
- **2 Euler-Schritte:** Drei Cluster entstehen, doch **$\approx 25\%$ der Samples bleiben modusfern**. Die Modusgewichte verschieben sich stark auf **$[0.64, 0.22, 0.14]$** (gegenüber $[0.50, 0.30, 0.20]$), weil der erste halbe Schritt alle Punkte zum Mittelwert $(-0.5, -0.7)$ zieht. Diesem liegt Modus 1 am nächsten.

<!--en:d60f8ae8-->
- **4 und 10 Euler-Schritte:** Der Anteil modusferner Samples beträgt noch $\approx 6.5\%$ bei $4$ Schritten und sinkt auf $\approx 2\%$ bei $10$ Schritten ($\approx 1\%$ in den echten Daten).

<!--en:44fb27e0-->
::: {.slide title="Rectified Flow und Reflow zur Begradigung von Trajektorien"}

<!--en:76f3c8c6-->
[Neue Paare aus Rauschen und Daten entlang der Sehnen von ODE-Trajektorien]{.kicker}

<!--en:628e3c59-->
- **Warum unabhängige Sehnen die ODE-Trajektorien krümmen:** Unabhängig gezogene Strecken $(1-t)\mathbf{x}_0 + t\boldsymbol\epsilon$ schneiden sich überall. Die marginalen ODE-Trajektorien können sich nicht schneiden und müssen um die Schnittpunkte ausweichen.

<!--en:0efdcd20-->
- **Reflow (Liu, Gong und Liu, 2022):** Paart Rauschen $\mathbf{z}$ mit den Endpunkten $\hat{\mathbf{x}}_0 = \Phi_{1 \to 0}(\mathbf{z})$ der Lehrer-ODE. Die Sehnen schneiden sich kaum noch. Dafür muss ein ganzer synthetischer Datensatz erzeugt werden, und die Fehler des Lehrers bleiben erhalten.

<!--en:085dfa47-->
::: {.slide title="Flussabbildungen und das Feld der Durchschnittsgeschwindigkeit"}

<!--en:46254f66-->
[Die Lösung der ODE statt ihrer Ableitung lernen]{.kicker}

<!--en:dbfea6e7-->
Statt Trajektorien durch neue Datenpaare zu begradigen, kann ein Netz die **Flussabbildung** $\Phi_{t \to s}(\mathbf{x}_t)$ lernen. Sie liefert genau den Punkt, den die ODE-Trajektorie durch $(\mathbf{x}_t, t)$ zur Zeit $s \le t$ erreicht:

<!--en:ac2a95af-->
- **Durchschnittsgeschwindigkeit $\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$:** Zeitliches Mittel der Geschwindigkeit entlang der gekrümmten Trajektorie über $[s, t]$

<!--en:9f8c486b-->
- **Randbedingung bei Schrittweite null ($s \to t$):** Für $s \to t$ geht die Durchschnittsgeschwindigkeit in die Momentangeschwindigkeit über:
  $$\bar{\mathbf{u}}(\mathbf{x}_t, t, t) = \mathbf{u}_t(\mathbf{x}_t)$$

<!--en:c4cd88ab-->
- **Sampling mit einem oder mehreren Schritten:** Eine einzige Auswertung $\mathbf{x}_1 - \bar{\mathbf{u}}(\mathbf{x}_1, 0, 1)$ springt direkt vom Rauschen zu den Daten.

<!--en:887b2032-->
::: {.slide title="Rechenbeispiel: exakte Flussabbildung für eindimensionale gaußverteilte Daten"}

<!--en:1a2b1e35-->
[Warum die Durchschnittsgeschwindigkeit die Streuung erhält und der Euler-Schritt sie verliert]{.kicker}

<!--en:dd4c9df7-->
- **Exakte Erhaltung der Quantile:** Für $x_0 \sim \mathcal{N}(2, 0.5^2)$ mit $k_t = \sqrt{0.25(1-t)^2 + t^2}$ erhält die eindimensionale ODE den standardisierten z-Wert von $t=1$ ($k_1 = 1$) bis $s=0$ ($k_0 = 0.5$).

<!--en:0d139b3f-->
- **1 Euler-Schritt ($u_1(x) = x - 2$):** Er endet bei $x - (x - 2) = \mathbf{2}$. Die Varianz fällt auf null.

<!--en:2ed7c331-->
- **1 Sprung der Flussabbildung ($\bar{u}(x, 0, 1) = 0.5x - 2$):** Er endet bei $x - (0.5x - 2) = \mathbf{2 + 0.5x} \sim \mathcal{N}(2, 0.5^2)$.

<!--en:68a9a2e7-->
::: {.slide title="Drei Identitäten zur Charakterisierung von Flussabbildungen"}

<!--en:66b729a6-->
[Komposition, lagrangesche und eulersche Beschreibung]{.kicker}

<!--en:84e2afd4-->
- **1. Komposition (Halbgruppe):** $\Phi_{r \to s}(\Phi_{t \to r}(\mathbf{x}_t)) = \Phi_{t \to s}(\mathbf{x}_t)$, mit zwei Schülerdurchläufen über $r$ und ohne Ableitungen (progressive Destillation, Shortcut-Modelle).

<!--en:89007d56-->
- **2. Lagrangesche Identität (bewegliche Endzeit $s$):** $\partial_s \Phi_{t \to s}(\mathbf{x}_t) = \mathbf{u}_s(\Phi_{t \to s}(\mathbf{x}_t))$, mit Auswertung der Geschwindigkeit $\mathbf{u}_s$ am Zielpunkt $\Phi_{t \to s}(\mathbf{x}_t)$.

<!--en:81563c96-->
- **3. Eulersche Identität (bewegliche Startzeit $t$):** $\partial_t \Phi_{t \to s} + (\nabla_{\mathbf{x}}\Phi_{t \to s})\mathbf{u}_t = \mathbf{0}$, mit der Geschwindigkeit $\mathbf{u}_t$ am Startpunkt $\mathbf{x}_t$ und 1 Jacobi-Vektor-Produkt (JVP) im Vorwärtsmodus (Konsistenzmodelle, MeanFlow).

<!--en:766b5296-->
::: {.slide title="Konsistenzmodelle: den Sprung zu t = 0 lernen"}

<!--en:91947990-->
[Invarianz entlang von Trajektorien]{.kicker}

<!--en:65d67025-->
- **Feste Endzeit $s = 0$ (Song et al., 2023):** Entlang jeder Trajektorie muss $f_{\boldsymbol\theta}(\mathbf{x}_t, t) = \Phi_{t \to 0}(\mathbf{x}_t)$ gelten, mit der Randbedingung $f_{\boldsymbol\theta}(\mathbf{x}_0, 0) = \mathbf{x}_0$.

<!--en:ea2fc585-->
- **Warum mehrere Schritte erneutes stochastisches Verrauschen erfordern:** Da $f_{\boldsymbol\theta}$ nur nach $s = 0$ springt, muss für einen zweiten Schritt das Sample $\hat{\mathbf{x}}_0$ bis $t = 0.5$ mit frischem Rauschen $\boldsymbol\epsilon'$ verrauscht werden. Dabei geht die deterministische Bijektion der ODE verloren.

<!--en:58d1ab89-->
::: {.slide title="Herleitung der MeanFlow-Identität für zwei Zeitpunkte"}

<!--en:90b806c5-->
[Exaktes Regressionsziel für die Durchschnittsgeschwindigkeit]{.kicker}

<!--en:e176a08d-->
- **Herleitung mit der Produktregel (Geng et al., 2025):** Die Ableitung von $(t - s)\,\bar{\mathbf{u}}(\mathbf{x}_t, s, t) = \int_s^t \mathbf{u}_\tau(\mathbf{x}_\tau)\,d\tau$ nach $t$ bei festem $s$ liefert ein explizites Fixpunktziel für $\bar{\mathbf{u}}$.

<!--en:51f93131-->
- **Ein JVP im Vorwärtsmodus:** Entlang $d\mathbf{x}_t/dt = \mathbf{u}_t$ entspricht die totale Ableitung $\frac{d}{dt}\bar{\mathbf{u}} = \partial_t\bar{\mathbf{u}} + (\nabla_{\mathbf{x}}\bar{\mathbf{u}})\mathbf{u}_t$ einem einzigen JVP in Richtung der Tangente $(\mathbf{u}_t, 1)$.

<!--en:117e3ad0-->
::: {.slide title="Gemeinsamer Verlust für Flussabbildungen und Gradientenstopp"}

<!--en:a43f5721-->
[Vier Verfahren mit wenigen Schritten in einer Gleichung]{.kicker}

<!--en:d243c163-->
- **Vier Kombinationen (2×2):** Lehrergeschwindigkeit $\hat{\mathbf{u}}_{\boldsymbol\phi}$ oder bedingte Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$, jeweils mit $s = 0$ oder $s \le t$, ergeben **Konsistenzdestillation**, **Destillation der Flussabbildung**, **Konsistenztraining** und **MeanFlow**.

<!--en:c81eeecb-->
- **Notwendiger Gradientenstopp $\mathrm{sg}[\cdot]$:** Feste Parameter $\boldsymbol\theta$ im Zielwert verhindern, dass der Schüler die Varianz durch einen Kollaps von $\nabla_{\mathbf{x}}\bar{\mathbf{u}}_{\boldsymbol\theta}$ minimiert.

<!--en:266e9593-->
::: {.slide title="Fehler der Flussabbildung und langsame Zeiteinbettungen im Code" layout="tight"}

<!--en:63942fb9-->
[Stabile JVPs für Zeitmerkmale]{.kicker}

<!--en:ff3ca392-->
Im Code berechnet `flow_map_error` die Ausgabe $\bar{\mathbf{u}}_{\boldsymbol\theta}$ und ihre Richtungsableitung $d\bar{\mathbf{u}}_{\boldsymbol\theta}/dt$ gemeinsam mit einem einzigen Aufruf von `jvp`:

<!--en:94fc7b67-->
::: {.slide title="Fehler der Flussabbildung und langsame Zeiteinbettungen im Code (Fortsetzung)" layout="tight"}

<!--en:63942fb9-2-->
[Stabile JVPs für Zeitmerkmale]{.kicker}

<!--en:9a939d0b-->
- **Warum Flussabbildungen Sinusmerkmale mit niedrigen Zeitfrequenzen verwenden (`slow = (1e-4, 1.0)`):**
  - Bei der Ableitung von $\sin(\omega t)$ nach $t$ tritt die Kreisfrequenz als Faktor auf: $\frac{d}{dt}\sin(\omega t) = \omega\cos(\omega t)$.
  - Übliche Zeiteinbettungen in Diffusionsmodellen verwenden $\omega$ bis $64$ oder $1{,}000$. Dadurch werden die JVP-Gradienten im Zielwert um den Faktor $64\times$ bis $1{,}000\times$ verstärkt. Das destabilisiert das Konsistenztraining (Lu und Song, 2025). Mit der Beschränkung $\omega \le 1.0$ bleiben die Zeitableitungen von der Größenordnung $O(1)$.

<!--en:886854c4-->
::: {.slide title="Training dreier Schüler für wenige Schritte auf der zweidimensionalen Mischung"}

<!--en:90d5136c-->
[Destillation und MeanFlow ohne Lehrer im Vergleich]{.kicker}

<!--en:eb5ae780-->
Mit einem über 20 000 Schritte trainierten Lehrer `TimeMLP` entstehen für sechs Zufallsstartwerte (`seeds 2..7`) jeweils drei Schüler: `consistency` ($s = 0$, mit Lehrer), `distillation` ($s \le t$, mit Lehrer) und `MeanFlow` ($s \le t$, ohne Lehrer):

<!--en:e51530de-->
- **Warum der Verlust bei Destillation gegen $0$ geht und bei MeanFlow um $\approx 80$ stagniert:**
  - Die destillierten Schüler lernen durch Regression eine deterministische Funktion von $(\mathbf{x}_t, s, t)$, die der Lehrer vorgibt. Ihr Verlust erreicht $\approx 0.03$–$0.08$.
  - MeanFlow verwendet die verrauschte bedingte Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$ jedes Beispiels sowohl für $\mathbf{u}_{\textrm{ref}}$ als auch in der JVP-Tangente $(\mathbf{u}_{\textrm{ref}}, 1)$. Die hohe bedingte Varianz setzt dem Verlust eine unvermeidbare Untergrenze ($\approx 76$–$85$).

<!--en:bd038e9f-->
::: {.slide title="Vergleich der Schüler bei wenigen Schritten auf der zweidimensionalen Mischung"}

<!--en:c785151a-->
[1, 2 und 4 Auswertungen bei sechs Zufallsstartwerten im Training]{.kicker}

<!--en:e042bd6e-->
Vergleich aller 18 trainierten Schülernetze an $20{,}000$ gaußverteilten Startpunkten mit 50 Euler-Schritten des Lehrers und den echten Daten:

<!--en:8c984381-->
::: {.slide title="Vergleich der Schüler auf der zweidimensionalen Mischung (Ergebnisse)" layout="tall-output"}

<!--en:c785151a-2-->
[1, 2 und 4 Auswertungen bei sechs Zufallsstartwerten im Training]{.kicker}

<!--en:250b2c66-->
Fehler der Modusgewichte für jeden der sechs Zufallsstartwerte, Anteil modusferner Samples zwischen den Modi und Streuung innerhalb der Modi:

<!--en:08c5edb8-->
::: {.slide title="Warum die Modi beim Training ohne Lehrer schrumpfen"}

<!--en:e3258ce8-->
[Der Nutzen eines vortrainierten Geschwindigkeitsmodells als Lehrer]{.kicker}

<!--en:5dc9d79a-->
Der Vergleich von `distillation` mit zwei Zeitpunkten und `MeanFlow` ohne Lehrer über je sechs unabhängige Zufallsstartwerte isoliert den Effekt eines einzigen Austauschs: An die Stelle der marginalen Geschwindigkeit $\hat{\mathbf{u}}_t(\mathbf{x}_t)$ des Lehrers tritt $\boldsymbol\epsilon - \mathbf{x}_0$.

<!--en:e65bc74c-->
| Verfahren und Schrittzahl | Größter Fehler der Modusgewichte (6 Zufallsstartwerte) | Anteil modusferner Samples ($>3s$) | Streuung innerhalb der Modi | Ergebnis |

<!--en:e607fb1d-->
| **Echte Daten der Mischverteilung** | $0.005$ (Stichprobenrauschen) | $1.1\%$ | **$0.474$** | *(Referenzwert der Daten)* |

<!--en:f3d8be69-->
| **Lehrer (50 Euler-Schritte)** | $0.003$ | $1.5\%$ | **$0.46$** | Gute Übereinstimmung mit der Datenverteilung |

<!--en:764444a3-->
| **Konsistenz ($s=0$), 1 Sprung** | $0.002$–$0.006$ | $1.9\%$–$2.1\%$ | **$0.453$–$0.463$** | Genaue Gewichte und Streuung nach 1 Sprung |

<!--en:e482c9f9-->
| **Destillation ($s \le t$), 1 / 2 / 4 Sprünge** | $0.001$–$0.006$ (alle Zufallsstartwerte) | $2.0\% \to \mathbf{1.0\%}$ | **$0.45$–$0.47$** | Erhält Streuung und deterministische Abbildung |

<!--en:cf5191fe-->
| **MeanFlow (ohne Lehrer), 1 / 4 Sprünge** | $\le 0.007 \to$ bis zu $\mathbf{0.02}$ | $\approx 4\% \to \approx 1\%$ | **$0.31$–$0.34 \to 0.35$–$0.39$** | **Verringert die Streuung der Modi um $\approx 30\%$ (1 Sprung) bzw. $15$–$25\%$ (4 Sprünge)** |

<!--en:36470879-->
**Warum verrauschte JVP-Tangenten die Streuung verringern:** Gibt $\boldsymbol\epsilon - \mathbf{x}_0$ die JVP-Tangente vor, hat der Korrekturterm $(t - s)\,d\bar{\mathbf{u}}/dt$ eine hohe Varianz. Das könnte erklären, warum der Schüler die Krümmungskorrektur nur teilweise lernt. Sein Sprung ähnelt wieder stärker einem großen Euler-Schritt, der die Varianz innerhalb der Modi verringert.

<!--en:182acf08-->
::: {.slide title="Destillation eines U-Nets für wenige Schritte auf Fashion-MNIST" layout="tight"}

<!--en:d93768c7-->
[Konditionierung auf den Zeitabstand mit Nullinitialisierung]{.kicker}

<!--en:97f31606-->
Zur Destillation des Geschwindigkeits-U-Nets mit $1.2\text{M}$ Parametern für Fashion-MNIST in eine Flussabbildung `UNetFlowMap` mit zwei Zeitpunkten startet der Schüler als exakte Kopie des Lehrers. Der Zeitabstand $t - s$ fließt über eine **mit null initialisierte lineare Schicht** ein:

<!--en:b94f7df2-->
::: {.slide title="Destillation eines U-Nets für wenige Schritte auf Fashion-MNIST (Fortsetzung)" layout="tight"}

<!--en:d93768c7-2-->
[Konditionierung auf den Zeitabstand mit Nullinitialisierung]{.kicker}

<!--en:52a70e40-->
- **Start mit der Geschwindigkeit des Lehrers:** Da `self.gap` mit null initialisiert ist, gilt $\bar{\mathbf{u}}_{\boldsymbol\theta}(\mathbf{x}_t, s, t) = \hat{\mathbf{u}}_{\textrm{teacher}}(\mathbf{x}_t, t)$ bei Schritt $0$.

<!--en:69df9fb3-->
- **50% der Paare auf der Diagonalen ($s = t$):** Für die Hälfte jedes Minibatches gilt $s = t$. Der Zielwert ist dort die Momentangeschwindigkeit $\hat{\mathbf{u}}_{\textrm{teacher}}$ des Lehrers. So behält der Schüler das Tangentenfeld, während er lange Sprünge mit $s < t$ über $5{,}000$ Aktualisierungen lernt.

<!--en:9e3f80c5-->
::: {.slide title="Euler-Schritte des Lehrers und Sprünge des Schülers auf Fashion-MNIST" layout="tight"}

<!--en:b5aa76d0-->
[FD bei 1 bis 50 Auswertungen]{.kicker}

<!--en:3962846e-->
Der Vergleich des Lehrers ($1$ bis $50$ Euler-Schritte) mit zwei destillierten Schülern (`seed 2, 3` bei $1, 2, 4$ Sprüngen) auf denselben $1{,}000$ Rauschbildern zeigt den großen Gewinn bei wenigen Schritten:

<!--en:64785e68-->
- **1 Auswertung:** Der Euler-Schritt des Lehrers endet auf dem grauen Mittelwertbild des Datensatzes ($\mathrm{FD} \approx 160$). 1 Schülersprung erreicht **$\mathrm{FD} \approx 11.5$** ($14\times$ niedriger).

<!--en:e149c5d1-->
- **2 Auswertungen:** Der Lehrer erreicht mit Euler-Schritten $\mathrm{FD} \approx 80$, der Schüler mit 2 Sprüngen **$\mathrm{FD} \approx 4.5$ bis $5$**. Das ist besser als der Wert von etwa $7$ bei 10 Lehrerschritten.

<!--en:ab9f30bf-->
- **4 Auswertungen:** Der Schüler erreicht **$\mathrm{FD} \approx 3.7$** und liegt damit gleichauf mit 20 Euler-Schritten des Lehrers ($3.5$ bis $4$). Der Abstand zum FD des Lehrers beträgt höchstens einen Faktor von $1.4\times$ bei 50 Lehrerschritten ($2.6$).

<!--en:14e677c8-->
::: {.slide title="Erhaltung der Trajektorien: Euler-Schritte des Lehrers und Sprünge des Schülers"}

<!--en:3a98bc9d-->
[50 und 2 Lehrerschritte im Vergleich zu 1–2 Schülersprüngen]{.kicker}

<!--en:c15b83d0-->
`UNetFlowMap` folgt der deterministischen ODE des Lehrers. In jeder Spalte führt dasselbe Rauschen $\mathbf{z}_i$ daher zum **selben Kleidungsstück**:

<!--en:7cc92fc9-->
- **Zeile 1 und 2 (Lehrer, 50 und 2 Euler-Schritte):** Zwei Euler-Schritte erzeugen verschwommene, kontrastarme Schemen. Oft lässt sich nicht einmal die richtige Art von Kleidungsstück erkennen.

<!--en:959aea82-->
- **Zeile 3 und 4 (Schüler, 1 und 2 Sprünge):** Schon **ein einziger Sprung** (Zeile 3) trifft in fast jeder Spalte Umriss und Grauton des Kleidungsstücks nach 50 Lehrerschritten. **Zwei Sprünge** (Zeile 4) schärfen Kanten und Texturen bis fast auf die Qualität des Lehrers.

<!--en:a8767445-->
::: {.slide title="Destillation durch Verteilungsabgleich (DMD) und Kritiker"}

<!--en:dca14e55-->
[Abgleich marginaler Dichten statt ODE-Trajektorien]{.kicker}

<!--en:bc57c726-->
- **Destillation durch Verteilungsabgleich (DMD, Yin et al., 2024):** Verrauscht die in 1 Schritt erzeugten künstlichen Samples erneut, $\mathbf{x}_t = (1-t)g_{\boldsymbol\theta}(\mathbf{z}) + t\boldsymbol\epsilon$. Der Abstieg für die umgekehrte KL-Divergenz nutzt die Score-Differenz über die Rauschpegel $t$.

<!--en:77e23247-->
- **Abwägung gegenüber Flussabbildungen:** Ein zusätzlicher adversarieller Kritiker mit echten Bildern (ADD / DMD2) kann in 1 Schritt einen besseren FD als der Lehrer erzielen. Die deterministische Bijektion $\mathbf{z} \leftrightarrow \mathbf{x}_0$ entlang der ODE-Trajektorien geht dabei verloren.

<!--en:a5c4b78c-->
::: {.slide title="Verständnisfragen zu Flussabbildungen, MeanFlow und Gradientenstopp"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:1a4350d9-->
Fragen zu den Identitäten der Flussabbildung und zur Destillation auf wenige Schritte:

<!--en:6177c482-->
1. **Die MeanFlow-Identität für eindimensionale Gauß-Verteilungen prüfen:** Für $x_0 \sim \mathcal{N}(2, 0.5^2)$ bei $s = 0, t = 1$ ergeben sich $\bar{u}(x, 0, 1) = 0.5x - 2$ und die Momentangeschwindigkeit $u_1(x) = x - 2$. Welchen Wert muss die totale Ableitung gemäß der MeanFlow-Identität $\bar{u} = u_1 - (1 - 0)\frac{d}{dt}\bar{u}$ annehmen, wenn man $\frac{d}{dt}\bar{u}(x_t, 0, t)$ bei $t = 1$ auswertet?

<!--en:d0546e71-->
2. **Warum zwei Schritte den stärksten Modus begünstigen:** Warum weisen 2 Euler-Schritte im exakten Geschwindigkeitsfeld der zweidimensionalen Mischung mit Gewichten $(0.5, 0.3, 0.2)$ dem Modus 1 das Gewicht $0.64$ statt $0.50$ zu?

<!--en:557b98b6-->
**Lösungen:**

<!--en:ffb1ef36-->
1. Nach der MeanFlow-Identität gilt $\frac{d}{dt}\bar{u} = u_1(x) - \bar{u}(x, 0, 1) = (x - 2) - (0.5x - 2) = \mathbf{0.5\,x}$. (Direkt gerechnet: Entlang einer Trajektorie gilt $\bar{u}(x_t, 0, t) = (x_t - x_0)/t$ und damit $\frac{d}{dt}\bar{u} = (u_t - \bar{u})/t = 0.5x$ bei $t = 1$.)

<!--en:c15c69e6-->
2. Bei $t = 1$ lenkt der erste halbe Schritt jeden Punkt zum globalen Mittelwert der Daten $\mathbb{E}[\mathbf{x}_0] = (-0.5, -0.7)$. Diesem liegt der Modus $(-2.5, -1.5)$ mit dem größten Gewicht am nächsten. So gelangt bei $t = 0.5$ zusätzliche Masse in das Einzugsgebiet von Modus 1.

<!--en:951704c8-->
::: {.slide title="Verständnisfragen zu Flussabbildungen, MeanFlow und Gradientenstopp (Fortsetzung)"}

<!--en:929b7671-2-->
[Wissen überprüfen]{.kicker}

<!--en:96b8286c-->
3. **Erhaltung der Zuordnung von Rauschen zu Bild bei 2 Netzauswertungen:** Zur Wahl stehen ein Konsistenzmodell ($s = 0$), eine Flussabbildung mit zwei Zeitpunkten ($s \le t$) und ein adversariell destillierter Einschrittgenerator. Bei welchem Modell hängt das Sample nach 2 Auswertungen weiterhin rein deterministisch vom anfänglichen Rauschen $\mathbf{z}$ ab?

<!--en:23db34b8-->
**Lösung:**

<!--en:8f6ac6ae-->
3. **Nur bei der Flussabbildung mit zwei Zeitpunkten ($\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$):** Sie verkettet zwei deterministische Sprünge $\Phi_{0.5 \to 0}(\Phi_{1 \to 0.5}(\mathbf{z}))$ entlang derselben ODE-Trajektorie. Konsistenzmodelle und Einschrittgeneratoren müssen für die zweite Auswertung frisches Rauschen $\boldsymbol\epsilon'$ bei $t = 0.5$ hinzufügen.

<!--en:2bd73df3-->
::: {.slide title="Zusammenfassung: Verfahren zur Generierung in wenigen Schritten"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:48593f33-->
| Verfahren | Netzausgabe und Definitionsbereich | Voraussetzungen für das Training | Mehrere Schritte | Stärke und Grenze |

<!--en:e0c8e903-->
| **Euler-Verfahren als ODE-Löser** | Momentangeschwindigkeit $\mathbf{u}_t(\mathbf{x}_t)$ | Übliche CFM-Regression | $K$ Tangentenschritte | Kollaps auf $\mathbb{E}[\mathbf{x}_0]$ bei $K = 1$, benötigt $20$–$50$ Netzauswertungen |

<!--en:3864e296-->
| **Reflow (Rectified Flow)** | Momentangeschwindigkeit $\mathbf{u}_t(\mathbf{x}_t)$ | Erneutes Training mit Lehrerpaaren $(\hat{\mathbf{x}}_0, \mathbf{z})$ | Begradigte Euler-Schritte | Begradigt Trajektorien, erbt aber die Fehler der Lehrersamples |

<!--en:572a9391-->
| **Konsistenzdestillation** | Endpunkt $\Phi_{t \to 0}(\mathbf{x}_t)$ ($s = 0$) | Vortrainierter Lehrer + 1 JVP | Erneutes stochastisches Verrauschen bei $t'$ | Gutes Ergebnis mit 1 Schritt, der 2. Schritt braucht frisches Rauschen $\boldsymbol\epsilon'$ |

<!--en:25529e5c-->
| **Destillation der Flussabbildung** | Durchschnittsgeschwindigkeit $\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$ | Vortrainierter Lehrer + 1 JVP | Deterministische Sprünge $t \to r \to s$ | Erhält die Streuung der Modi und die Bijektion von Rauschen zu Samples ($2$ Sprünge $\approx 15$ Euler-Schritte) |

<!--en:bdb0ff6f-->
| **MeanFlow** | Durchschnittsgeschwindigkeit $\bar{\mathbf{u}}(\mathbf{x}_t, s, t)$ | Ohne Lehrer ($\boldsymbol\epsilon - \mathbf{x}_0$) + 1 JVP | Deterministische Sprünge $t \to r \to s$ | Spart den Trainingslauf des Lehrers, doch die verrauschte Tangente verringert die Streuung |

<!--en:c5c2f7a5-->
| **DMD / adversarielle Destillation (ADD)** | Einschrittgenerator $g_{\boldsymbol\theta}(\mathbf{z})$ | Lehrer + Score künstlicher Samples / Kritiker | Erneutes Verrauschen bei $>1$ Schritten | Kann in 1 Schritt den FD des Lehrers erreichen oder unterbieten, verliert die Invertierbarkeit der ODE |
