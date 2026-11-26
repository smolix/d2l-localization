<!--en:846cfc4f-->
# Likelihood, Gewichtung und Gestaltungsspielraum

<!--en:1251f191-->
In den vorangegangenen Abschnitten haben wir Diffusions- und Flow-Matching-Modelle mit dem ungewichteten quadratischen Fehler für Rauschen oder Geschwindigkeit als Zielgröße trainiert und anhand ihrer Samples bewertet. Anders als die adversariellen Modelle aus :numref:`chap_gans` ordnen Diffusionsmodelle Datenpunkten auch Wahrscheinlichkeitsdichten zu. :citet:`sohl2015deep` und :citet:`ho2020denoising` zeigten, dass die rückwärts durchlaufene Markov-Kette ein latentes Variablenmodell definiert. Dessen untere Evidenzschranke (engl. evidence lower bound, ELBO) zerfällt in eine gewichtete Summe von Regressionstermen für die Rauschvorhersage bei verschiedenen Rauschpegeln. Die Wahrscheinlichkeitsfluss-ODE (engl. probability-flow ODE) wiederum definiert über die Formel für die momentane Variablentransformation eine exakte Log-Likelihood :cite:`song2021score`.

<!--en:f1341191-->
Wie :citet:`Kingma.Salimans.Poole.ea.2021` und :citet:`Kingma.Gao.2023` außerdem zeigten, erhalten wir für alle üblichen Zielfunktionen von Diffusionsmodellen und Flow-Matching dasselbe Integral, wenn wir sie durch das logarithmierte Signal-Rausch-Verhältnis (engl. log signal-to-noise ratio, Log-SNR) $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$ ausdrücken. Der Fehler der Rauschvorhersage geht dabei mit einer Gewichtungsfunktion $w(\lambda)$ ein. Neben dieser Verlustgewichtung identifizierten :citet:`Karras.Aittala.Aila.ea.2022` (EDM) zwei unabhängige Gestaltungsmöglichkeiten, die Vorkonditionierung von Ein- und Ausgabe und die Verteilung der Rauschpegel beim Training. In diesem Abschnitt leiten wir die diskrete und die kontinuierliche ELBO her und vergleichen exakte Fluss-Likelihoods und Schranken für die Mischverteilung mit drei Modi. Wir führen veröffentlichte Zielfunktionen auf ihre jeweilige Log-SNR-Gewichtung $w(\lambda)$ zurück und variieren in einer kontrollierten Ablationsstudie auf Fashion-MNIST jeweils einen Faktor.

<!--en:f4fd67ad-->
## Die Likelihood-Schranke

<!--en:a0d2ecd0-->
Schreiben wir zunächst die Kette aus :numref:`sec_diffusion-gaussian-paths` als generatives Modell und leiten wir eine Schranke für ihre Likelihood her. Die Regression auf das Rauschen ergibt sich dann von selbst.

<!--en:d5fa52ba-->
### Die Kette als latentes Variablenmodell

<!--en:ed2f0518-->
Schritt $k$ der Vorwärtskette skaliert den aktuellen Punkt in Richtung Ursprung und addiert Rauschen, $\mathbf{x}_k = \sqrt{1 - \beta_k}\, \mathbf{x}_{k-1} + \sqrt{\beta_k}\, \boldsymbol\epsilon_k$. Alle bisherigen Schritte zusammen entsprechen einer einzigen Ziehung aus einer Gauß-Verteilung, $\mathbf{x}_k = \alpha_k \mathbf{x}_0 + \sigma_k \boldsymbol\epsilon$ mit $\alpha_k^2 = \prod_{j \le k}(1 - \beta_j)$ und $\sigma_k^2 = 1 - \alpha_k^2$. Das ist der VP-Pfad zu den Zeitpunkten $t_k = k/T$. Mit $q$ bezeichnen wir die Verteilungen dieser Vorwärtskette. Das generative Modell durchläuft sie mit gelernten gaußschen Schritten rückwärts,

<!--en:6ea44d1c-->
mit $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, einem vom Netz berechneten Mittelwert $\boldsymbol\mu_{\boldsymbol\theta}$ und festen Varianzen $\nu_k$ ($\sigma_k$ bezeichnet weiterhin die Rauschskala des Pfades).

<!--en:dfe8309f-->
Die Dichte von $\mathbf{x}_0$ erhalten wir, indem wir diese gemeinsame Dichte über die latenten Variablen $\mathbf{x}_{1:T}$ integrieren. Dieses Integral über $T$ Kopien des Datenraums können wir nicht berechnen. Mithilfe einer beliebigen Verteilung $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ der latenten Variablen schreiben wir es als Erwartungswert, $p_{\boldsymbol\theta}(\mathbf{x}_0) = \mathbb{E}_{q}\big[p_{\boldsymbol\theta}(\mathbf{x}_{0:T}) / q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\big]$. Da der Logarithmus konkav ist, ist der Logarithmus eines Mittelwerts mindestens so groß wie der Mittelwert der Logarithmen (Jensen-Ungleichung). Für die negative Log-Likelihood ergibt das eine obere Schranke,

<!--en:c5e598b9-->
also die *untere Evidenzschranke* (ELBO) der Variationsinferenz (:numref:`sec_mdl-latent-em-elbo`) mit umgekehrtem Vorzeichen. Bei den meisten latenten Variablenmodellen müssen wir $q$ lernen, um die wahre A-posteriori-Verteilung anzunähern. Hier gibt die Vorwärtskette diese Verteilung vollständig vor. Die Differenz zwischen beiden Seiten der Ungleichung beträgt $D_{\textrm{KL}}\big(q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \,\|\, p_{\boldsymbol\theta}(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\big)$. Die Schranke ist also exakt, wenn die gelernte Rückwärtskette bei gegebenem $\mathbf{x}_0$ die Vorwärtskette rückwärts nachzeichnet. Im Mittel über die Daten ist die Schranke mindestens so groß wie die Kreuzentropie $\mathbb{E}_{p}[-\log p_{\boldsymbol\theta}(\mathbf{x}_0)] = H(p) + D_{\textrm{KL}}(p \,\|\, p_{\boldsymbol\theta})$. Lässt ein Modell einen Modus aus, fällt das also bei jedem Punkt dieses Modus ins Gewicht.

<!--en:db15ae47-->
Um die beiden Ketten zu vergleichen, setzen wir ihre Faktorisierungen ein, das obige Produkt für $p_{\boldsymbol\theta}$ sowie $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_k q(\mathbf{x}_k \mid \mathbf{x}_{k-1})$. Dadurch zerfällt die Schranke in je einen Term pro Schritt:

<!--en:3e30ae1d-->
Den $k$-ten Term können wir noch nicht als Divergenz lesen, denn der gelernte Schritt führt von $\mathbf{x}_k$ nach $\mathbf{x}_{k-1}$ und der Vorwärtsschritt in die Gegenrichtung. Mit dem Satz von Bayes kehren wir den Vorwärtsschritt um. Dank der Markov-Eigenschaft bleibt der Vorwärtsschritt unverändert, wenn wir zusätzlich auf $\mathbf{x}_0$ bedingen. Dann gilt

<!--en:9e32b743-->
Setzen wir dies für $k \ge 2$ ein, kürzen sich die Randdichten im Produkt der Quotienten, $\prod_{k=2}^{T} q(\mathbf{x}_{k-1} \mid \mathbf{x}_0) / q(\mathbf{x}_k \mid \mathbf{x}_0) = q(\mathbf{x}_1 \mid \mathbf{x}_0) / q(\mathbf{x}_T \mid \mathbf{x}_0)$. Der Zähler kürzt sich mit dem verbleibenden Vorwärtsschritt bei $k = 1$. Der Erwartungswert jedes verbleibenden Logarithmus ist eine Divergenz:

<!--en:f21b9390-->
Der *Prior-Term* $L_T$ prüft, ob die Kette bei der Gauß-Verteilung endet, mit der der Sampler startet. Die *Entrauschungsterme* $L_{k-1}$ vergleichen jeden gelernten Schritt mit dem *Vorwärts-Posterior*, der Umkehrung von Schritt $k$ bei bekanntem rauschfreiem Punkt. Der *Rekonstruktionsterm* $L_0$ bewertet den letzten Schritt. Bedingen wir auf $\mathbf{x}_0$, lässt sich der Vorwärts-Posterior berechnen. Er ist gaußsch, denn er ergibt sich aus dem Produkt zweier Gauß-Dichten in $\mathbf{x}_{k-1}$, dem Vorwärtsschritt und $q(\mathbf{x}_{k-1} \mid \mathbf{x}_0)$ (quadratische Ergänzung in Aufgabe :exref:`ex_design-space-forward-posterior`):

<!--en:8a58f511-->
Der Mittelwert ist ein gewichtetes Mittel der Positionen, die sich für $\mathbf{x}_{k-1}$ aus dem rauschfreien bzw. dem verrauschten Punkt ergeben. Die Varianz ist kleiner als die Varianz $\beta_k$ des Vorwärtsschritts. Kennen wir beide Endpunkte, ist der Punkt dazwischen besser bestimmt.

<!--en:c88c9d42-->
Prior-Term und Entrauschungsterme vergleichen Gauß-Verteilungen, sodass wir sie mit derselben Formel berechnen können. Für isotrope Gauß-Verteilungen in $d$ Dimensionen gilt

<!--en:22acdd36-->
als Spezialfall der Formel aus :numref:`subsec_gan_fit`. Der erste Teil bestraft eine falsche Varianz, der zweite eine Abweichung des Mittelwerts, deren Quadrat wir in Einheiten der zweiten Varianz messen.

<!--en:583b8091-->
Wie groß sind die einzelnen Terme? Den einfachsten Fall können wir von Hand durchrechnen. Betrachten wir eine Koordinate mit Daten aus $\mathcal{N}(0, 1)$, eine Kette aus $T = 2$ Schritten mit $\beta_1 = \beta_2 = \tfrac12$, also $\alpha_1^2 = \tfrac12$ und $\alpha_2^2 = \tfrac14$, und die bestmöglichen gelernten Mittelwerte. Der Prior-Term vergleicht die Endverteilung $\mathcal{N}(\alpha_2 x_0, \tfrac34)$ der Kette mit $\mathcal{N}(0, 1)$ und beträgt $x_0^2/8 + 0.019$, im Mittel also $0.144$. Selbst nach zwei großen Schritten bleibt viel von $x_0$ erhalten. Der Vorwärts-Posterior des zweiten Schritts hat den Mittelwert $\tfrac{\sqrt 2}{3}(x_0 + x_2)$ und die Varianz $\tilde\beta_2 = \tfrac13$. Der bestmögliche gelernte Schritt ersetzt $x_0$ in diesem Mittelwert durch die beste Schätzung aus $x_2$, deren Unsicherheit immer noch einer Varianz von $\tfrac34$ entspricht. Mit $\nu_k = \beta_k$, der Varianz des Vorwärtsschritts selbst, ergibt sich im Erwartungswert die Schranke

<!--en:38306be6-->
und damit die wahre negative Log-Likelihood von $\mathcal{N}(0, 1)$. Bei gaußschen Daten und diesen Varianzen ist die Schranke schon mit zwei Schritten exakt. Am leichtesten lässt sich der Rekonstruktionsterm nachprüfen. Bei gegebenem $x_1$ hat der rauschfreie Punkt noch eine Varianz von $\tfrac12$. Das ist gerade $\nu_1$, also gilt $L_0 = \tfrac12 \log(2\pi \cdot \tfrac12) + \tfrac12 = 1.072$. Wählen wir stattdessen $\nu_2 = \tilde\beta_2$, wächst $L_1$ auf $0.25$ und die Schranke auf $1.466$. Der Grund dafür ist wichtig. Der Vorwärts-Posterior kennt $x_0$, daher ist seine Varianz klein. Dem gelernten Schritt von $x_2$ aus fehlt diese Information, und seine Varianz muss auch diese Unsicherheit erfassen. Bei $T = 1000$ ist der Prior-Term nahezu null, jeder Schritt ist klein, und die Entrauschungsterme liefern das Trainingssignal.

<!--en:a64206d7-->
### Von der Schranke zur gewichteten Regression

<!--en:6fe9c849-->
Jeder Entrauschungsterm vergleicht zwei Gauß-Verteilungen, deren Varianzen unabhängig vom Netz sind. Nach :eqref:`eq_diffusion-gauss-kl` hängt also nur der Abstand der Mittelwerte von $\boldsymbol\theta$ ab:

<!--en:f9421773-->
Dabei verschwindet die Konstante für $\nu_k = \tilde\beta_k$. In jedem Schritt soll das Netz also den Mittelwert des Vorwärts-Posteriors liefern.

<!--en:87610964-->
Das Netz muss diesen Mittelwert nicht direkt ausgeben. Setzen wir $\mathbf{x}_0 = (\mathbf{x}_k - \sigma_k \boldsymbol\epsilon)/\alpha_k$ in :eqref:`eq_diffusion-posterior` ein und nutzen $\sigma_k^2 = \beta_k + (1 - \beta_k)\sigma_{k-1}^2$, so ergeben die Koeffizienten von $\mathbf{x}_k$ zusammen $1/\sqrt{1 - \beta_k}$, und für den Mittelwert erhalten wir

<!--en:09f5e99e-->
Bis auf $\boldsymbol\epsilon$ sind hier alle Größen bekannt, wenn das Netz $\mathbf{x}_k$ und $k$ erhält. Wir lassen es also das Rauschen vorhersagen und bilden seinen Mittelwert auf dieselbe Weise,

<!--en:f6c18877-->
und die Differenz der beiden Mittelwerte ist ein Vielfaches des Fehlers der Rauschvorhersage:

<!--en:773c3093-->
Jeder Term der Schranke entspricht der Rauschregression aus :numref:`sec_diffusion-gaussian-paths` bei einem Rauschpegel, mit einem Gewicht, das für $\nu_k = \tilde\beta_k$ gleich $\beta_k / (2(1 - \beta_k)\sigma_{k-1}^2)$ ist. In Schritt $k$ sinkt das Signal-Rausch-Verhältnis $\alpha_k^2/\sigma_k^2$ um den Faktor $\sigma_k^2 / ((1 - \beta_k)\sigma_{k-1}^2) = 1 + \beta_k/((1 - \beta_k)\sigma_{k-1}^2)$. Mit $\lambda_k = \log(\alpha_k^2/\sigma_k^2)$ ergibt sich also das Gewicht

<!--en:72a3a39d-->
Die Schranke gewichtet jeden Schritt danach, wie stark er die Log-SNR senkt. Die Codezelle gibt beide Gewichte entlang unserer VP-Kette ($T = 1000$) neben der halben Log-SNR-Schrittweite aus.

<!--en:70dfbb6d-->
Bei kleinen Schritten entspricht das Gewicht für $\nu_k = \tilde\beta_k$ der halben Log-SNR-Schrittweite, und das Gewicht für $\nu_k = \beta_k$ weicht nur am Anfang davon ab. Beide sind bei den ersten Schritten am größten, wo die Log-SNR am schnellsten fällt. Verdoppeln wir die ausgegebenen halben Schrittweiten, erhalten wir $0.78$ bei $k = 2$ und $0.157$ bei $k = 10$. Das lässt sich mit keinem Gitter aus gleich großen Zeitschritten vermeiden. Selbst ein konstantes $\beta$ senkt die Log-SNR um mindestens $\log(k/(k-1))$ in Schritt $k$, also um $\log 2 \approx 0.69$ bei $k = 2$ und um $\log(10/9) \approx 0.105$ bei $k = 10$. Unser $\beta$ wächst entlang der Kette, sodass die Log-SNR in den ersten Schritten noch stärker fällt. Nach den ersten paar Dutzend Schritten bleiben beide Gewichte bei etwa $0.01$ oder darunter. Der einfache Verlust aus :numref:`sec_diffusion-gaussian-paths`,

<!--en:886be93e-->
mit $k$ gleichverteilt auf $\{1, \ldots, T\}$, setzt alle Gewichte auf eins und vernachlässigt damit im Vergleich zur Schranke die Schritte mit dem geringsten Rauschen. Wie :citet:`ho2020denoising` feststellten, liefert er dennoch bessere Samples. Diese Schritte enthalten die feinsten Bilddetails (:numref:`subsec_diffusion-design-space-scales`), die voll in die Likelihood eingehen, dem Auge aber kaum auffallen.

<!--en:e9ed5cb8-->
Werden die Schritte immer kleiner, geht die gewichtete Summe in ein Integral über die Log-SNR über,

<!--en:e2b46504-->
wobei $\mathbf{x}_\lambda$ der Punkt des Pfades mit der Log-SNR $\lambda$ ist :cite:`Kingma.Salimans.Poole.ea.2021`. Die Log-SNR verläuft von $\lambda_{\max}$ beim kleinsten Rauschpegel bis $\lambda_{\min}$ am Pfadende. In Worten ergibt sich die Schranke, indem wir den Fehler der Rauschvorhersage mit Gewicht eins über die Log-SNR integrieren und die beiden Randterme addieren. Der Grenzübergang setzt kleine Schritte in $\lambda$ voraus. Am datenbehafteten Ende des Gitters fehlen sie jedoch. Vom ersten zum zweiten Schritt fällt die Log-SNR um mehr als $\log 2$, unabhängig von $T$. Dort ist die Schranke der Kette also nicht eng (Aufgabe :exref:`ex_design-space-weights-of-the-bound`).

<!--en:12482cc8-->
### Ein Zahlenwert für die Mischverteilung

<!--en:63ba440e-->
Zwei Zahlen beschreiben die Likelihood eines trainierten Netzes. Die Schranke :eqref:`eq_diffusion-elbo-continuous` gehört zum stochastischen Rückwärtsprozess. Die Wahrscheinlichkeitsfluss-ODE aus :numref:`sec_diffusion-integration` bildet Rauschen deterministisch auf Daten ab. Ihre Dichte folgt aus einer Variablentransformation. Für das exakte Feld stimmen beide Ergebnisse überein, für ein trainiertes Netz müssen sie das nicht. Wir trainieren mit dem Verfahren für die Mischverteilung aus :numref:`sec_diffusion-gaussian-paths` einen Rauschprädiktor für $20{,}000$ Schritte auf dem VP-Pfad.

<!--en:225f161a-->
Entlang einer Trajektorie von $d\mathbf{x}/dt = \mathbf{u}_t(\mathbf{x})$ ändert sich die Log-Dichte mit der Rate $-\nabla \cdot \mathbf{u}_t$. Wo der Fluss die Punkte auseinandertreibt, sinkt die Dichte. Das ist die momentane Variablentransformation aus :numref:`sec_mdl-continuous-normalizing-flows`. Integrieren wir von einem Datenpunkt zu einem frühen Zeitpunkt $t_{\min}$ bis $t = 1$, wo die Dichte standardnormal ist, erhalten wir

<!--en:d39b2bd2-->
`pf_nll` integriert den Punkt und seine aufsummierte Divergenz gemeinsam in 100 Schritten. Zum Datenende hin liegen die Schritte dichter, weil sich das Feld dort am schnellsten ändert. Die Funktion nutzt das klassische Runge-Kutta-Verfahren vierter Ordnung (:numref:`sec_mdl-euler-runge-kutta`), das mit dem Heun-Verfahren aus :numref:`sec_diffusion-samplers` verwandt ist und das Feld viermal pro Schritt auswertet. Die Geschwindigkeit erhalten wir mit den Umrechnungen aus :numref:`sec_diffusion-gaussian-paths` aus der Rauschvorhersage. Die Divergenz ist die Spur der $2 \times 2$-Jacobi-Matrix. Wir brauchen eine Ableitung im Vorwärtsmodus pro Koordinate, in $d$ Dimensionen also $d$ Ableitungen pro Auswertung. Für Bilder ist daher ein günstigerer Schätzer nötig (Aufgabe :exref:`ex_design-space-hutchinson`). Wir beginnen bei $t_{\min} = 0.001$, wo das Rauschen etwa $1\%$ des Signals beträgt, und verwenden die exakte Rauschvorhersage der Mischverteilung als Referenz.

<!--en:3835c530-->
Wir ziehen nun 20 000 Testpunkte aus `mix` und wenden `pf_nll` auf die ersten 4000 an, einmal mit dem exakten Rauschprädiktor `eps_exact` und einmal mit dem trainierten Netz `ema`. Ihre negativen Log-Likelihoods aus dem Wahrscheinlichkeitsfluss vergleichen wir mit dem wahren Wert `-mix.log_prob(test)`:

<!--en:735e69e5-->
Der Fluss des exakten Feldes liefert die wahren 2.5 Nat pro Punkt auf etwa vier Nachkommastellen genau und bestätigt damit den Integrator. Die negative Log-Likelihood des trainierten Netzes weicht davon um wenige Tausendstel Nat ab. Der Standardfehler der gepaarten Differenz beträgt etwa ein Tausendstel. Das Netz hat die Dichte selbst gelernt und kann daraus Samples ziehen.

<!--en:57f2db83-->
Zur Schranke gehören die beiden Randterme. Der Prior-Term beträgt etwa $\tfrac12 \alpha_1^2 \|\mathbf{x}_0\|^2 \approx 2 \cdot 10^{-4}$ Nat. Der letzte Schritt liefert eine Gauß-Verteilung um $\hat{\mathbf{x}}_0$ mit der Varianz $e^{-\lambda_{\max}} = \sigma^2/\alpha^2$ des kleinsten Rauschpegels. Wegen $\mathbf{x}_0 - \hat{\mathbf{x}}_0 = (\sigma/\alpha)(\hat{\boldsymbol\epsilon} - \boldsymbol\epsilon)$ gilt

<!--en:6354cc87-->
Wie sich die Schranke aufteilt, können wir schon vor der Ausgabe der Codezelle vorhersagen. In zwei Dimensionen liegt der erste Teil bei einem Rauschpegel von $0.01$ bei etwa $-7.3$ Nat, dem logarithmierten Normierungsfaktor einer sehr schmalen Gauß-Verteilung. Der zweite Teil beträgt etwa $1$. Das ist die Hälfte des erwarteten quadratischen Fehlers eines Netzes, das so schwaches Rauschen nicht von den Schwankungen der Daten selbst unterscheiden kann. Damit ist $L_0 \approx -6.3$. Liegt die gesamte Schranke nahe bei 2.5 Nat, muss das Integral allein etwa $2.5 + 6.3 \approx 8.8$ Nat betragen. Keiner der beiden Teile ist eine Likelihood. Erst ihre Summe liefert eine Schranke dafür.

<!--en:647d8fe1-->
`bound` schätzt das Integral mit der Monte-Carlo-Methode, wobei wir $\lambda$ geschichtet aus einer Gleichverteilung ziehen. Um die Schranke auf $0.01$ Nat genau zu bestimmen, müssen wir das Netz einige Millionen Mal auswerten. Für die Differenz zweier Prädiktoren genügen viel weniger Auswertungen, wenn beide dieselben Punkte, Rauschpegel und Rauschwerte erhalten. Das Monte-Carlo-Rauschen ist dann größtenteils beiden gemeinsam und fällt in der Differenz weg (:numref:`subsec_mdl-paired-comparisons`). Deshalb wertet `bound` eine Liste von Prädiktoren mit denselben Ziehungen aus.

<!--en:872fd08e-->
Die Codezelle bestätigt die Vorhersage. Das Integral beträgt etwa 8.75 Nat, die Randterme ergeben zusammen etwa $-6.3$. Für das exakte Feld bilden wir an jedem der 20 000 Testpunkte die Differenz zwischen Schranke und wahrer negativer Log-Likelihood. Sie ist innerhalb ihres Standardfehlers von $0.015$ mit null vereinbar. Mit dem bestmöglichen Prädiktor zeigt die kontinuierliche Schranke bei dieser Genauigkeit also keine Abweichung. Die Schranke des Netzes liegt etwa ein halbes Hundertstel Nat über der des exakten Feldes. Diese gepaarte Differenz lässt sich auf ein halbes Tausendstel genau bestimmen. Beide Maße ergeben für dieses Netz auf der Mischverteilung einen Abstand von höchstens einem halben Hundertstel Nat zum Maximum-Likelihood-Wert.

<!--en:54f9595e-->
### Einfacher Verlust und Schranke im Vergleich

<!--en:2d5f8698-->
Damit ist die eingangs gestellte Frage noch offen. Bei hinreichender Kapazität hat jede Gewichtung, die alle Rauschpegel erfasst, denselben Minimierer, die exakte Rauschvorhersage. Jede Gewichtung käme dem Optimum also so nahe. Die Gewichtung kann erst dann eine Rolle spielen, wenn das Netz zu klein ist, um überall exakt zu sein. Sie bestimmt dann, wo der Fehler anfällt. Deshalb trainieren wir Rauschprädiktoren mit zwei Gewichtungen, dem einfachen Verlust und der Gewichtung der Schranke selbst ($\lambda$ gleichverteilt, gezogen von `bound_loss` mithilfe von `time_at`). Wir verwenden jeweils die volle Breite sowie 32 versteckte Einheiten und je zwei Zufallsstartwerte. Jedes der acht Modelle erhält 10 000 Schritte, halb so viele wie das obige Modell, damit der Rechenaufwand überschaubar bleibt. Schon vor der Auswertung können wir das Ergebnis abschätzen. Bei voller Breite sollten die beiden Gewichtungen fast übereinstimmen. Bei 32 Einheiten sollte das mit der Schranke gewichtete Modell die niedrigere Schranke erreichen, denn darauf trainieren wir es. Beim einfachen Verlust entfallen weit weniger Ziehungen auf fast rauschfreie Punkte und mehr auf die verrauschten Bereiche, die jedes Sample und jede Flusstrajektorie durchlaufen. Dort könnte er also besser abschneiden.

<!--en:fcf88413-->
Wir bewerten jedes Modell auf drei Arten. Für die Schranke und die Fluss-Likelihood bilden wir jeweils die gepaarte Differenz zum exakten Feld an den Testpunkten. Außerdem ziehen wir mit der ancestralen Kette Samples aus demselben Ausgangsrauschen und bewerten sie anhand der wahren Dichte. Für die Samples geben wir ihre mittlere negative Log-Likelihood unter der wahren Dichte an sowie den Anteil der Ausreißer, also der Punkte, die mehr als drei Standardabweichungen von jedem Modus entfernt liegen. Für die Daten selbst ergeben sich etwa 2.47 Nat und 1%.

<!--en:928b5178-->
Bei voller Breite spielt die Gewichtung kaum eine Rolle. Bei beiden Maßen liegen alle vier Modelle höchstens zwei Hundertstel Nat vom exakten Feld entfernt. Ihre Samples sind nicht zu unterscheiden, und ihr Ausreißeranteil liegt nahe bei dem der Daten. Dennoch liegt jede Gewichtung bei ihrem eigenen Maß schon vorn, bei der Schranke um ein bis fünf Tausendstel Nat, beim Fluss um ein halbes bis ein ganzes Hundertstel. Bei 32 Einheiten wächst der Unterschied und zeigt sich auch bei den Samples. Das mit der Schranke gewichtete Modell erreicht die niedrigere Schranke, um ein bis zwei Hundertstel Nat. Der einfache Verlust liefert trotz fehlender Likelihood-Schranke die bessere Fluss-Likelihood. Die negative Log-Likelihood liegt um ein bis zwei Hundertstel niedriger. Auch die Samples sind besser. Ihre negative Log-Likelihood liegt um zwei bis sechs Hundertstel niedriger, und es gibt weniger Ausreißer, etwa 1.5 bis 1.8% gegenüber 2.0 bis 2.3%. Für diese Mischverteilung erreicht das kleine Netz beim Training mit der Schranke also nicht seine beste Likelihood. Das liegt vermutlich wie erwartet an der Verteilung der Ziehungen. Bei gleichverteiltem $\lambda$ auf $[-10.05, 9.12]$ liegen etwa $7/19$, also mehr als ein Drittel der Ziehungen für die Schranke, über $\lambda = 2$, wo der Punkt fast rauschfrei ist. Mit gleichverteiltem $t$ auf unserem VP-Pfad liegt dort etwa ein Zehntel.

<!--en:b7e9512b-->
Die Antwort auf die Eingangsfrage lautet also Nein, mit einer Einschränkung. Der einfache Verlust entspricht dem Integranden der Likelihood-Schranke mit einem anderen Gewicht. Bei reichlich Kapazität verändert das Gewicht beide Likelihood-Maße um höchstens etwa ein Hundertstel Nat. Bei geringer Kapazität entscheidet es zwischen einer besseren Schranke einerseits und einer besseren Fluss-Likelihood und besseren Samples andererseits.

<!--en:b77ff78d-->
## Zielfunktionen als Gewichtungen

<!--en:9e8895f3-->
Die Schranke gewichtet jede Einheit der Log-SNR gleich, der einfache Verlust jeden Schritt der Kette. Beim Flow-Matching lernen wir durch Regression eine Geschwindigkeit auf einem anderen Pfad. Drücken wir alle drei in derselben Variablen aus, erhalten wir eine Zielfunktion mit drei Gewichtungen.

<!--en:16a96aed-->
### Darstellung durch die Log-SNR

<!--en:bb35f8a3-->
Für jeden unserer Verluste ziehen wir einen Zeitpunkt $t$, bilden $\mathbf{x}_t$ und berechnen den quadratischen Fehler für eine Zielgröße. Bei festem $\mathbf{x}_t$ hängt jede Zielgröße affin von der Rauschvorhersage ab. Die skalare Steigung hängt nur von $t$ ab. Jeder quadratische Fehler hat also die Form $c(t)\, \|\hat{\boldsymbol\epsilon} - \boldsymbol\epsilon\|^2$ mit einem Faktor $c(t)$, den Zielgröße und Pfad bestimmen. Wechseln wir von der Variablen $t$ zu $\lambda$, erhalten wir die Darstellung von :citet:`Kingma.Gao.2023`,

<!--en:73d84973-->
wobei $p(t)$ die Dichte der Trainingszeitpunkte ist. Eine einzige Funktion beschreibt den Verlust also vollständig. Sie gibt an, wie stark der Fehler der Rauschvorhersage bei jeder Log-SNR ins Gewicht fällt. Die Schranke :eqref:`eq_diffusion-elbo-continuous` entspricht der Gewichtung $w \equiv 1$.

<!--en:448409c4-->
Der Erwartungswert im Integral hängt nicht vom Pfad ab. Bei gleichem $\lambda$ stimmen die verrauschten Punkte zweier Pfade bis auf die Skalierung überein (:numref:`sec_diffusion-gaussian-paths`). Das Netz löst also dieselbe Vorhersageaufgabe. In $\mathcal{L}_w$ wirkt sich ein Rauschplan somit nur über die Gewichtung und die Endpunkte $\lambda_{\min}$ und $\lambda_{\max}$ aus. Bei der Schranke liefern Rauschpläne mit denselben Endpunkten denselben Verlust :cite:`Kingma.Salimans.Poole.ea.2021`.

<!--en:ada8b379-->
Betrachten wir als einfachstes Rechenbeispiel den Rauschverlust mit gleichverteiltem $t$ auf dem linearen Pfad, $c = p = 1$. Dort gilt $\sigma_t/\alpha_t = t/(1 - t) = e^{-\lambda/2}$, also $t = 1/(1 + e^{\lambda/2})$ und $|dt/d\lambda| = \tfrac18 \operatorname{sech}^2(\lambda/4)$. Die Gewichtung $w(\lambda) = \tfrac14 \operatorname{sech}^2(\lambda/4)$ hat ihr Maximum bei $\lambda = 0$, wo Signal und Rauschen gleich stark sind. Auf beiden Seiten fällt sie wie $e^{-|\lambda|/2}$ ab.

<!--en:d1c27fb2-->
### Gewichtungen veröffentlichter Zielfunktionen

<!--en:fc11ddbc-->
Für die anderen Zielfunktionen brauchen wir zwei Umrechnungen. Auf dem VP-Pfad erfüllt die $\mathbf{v}$-Zielgröße aus :citet:`Salimans.Ho.2022` die Beziehung $\hat{\mathbf{v}} - \mathbf{v} = (\hat{\boldsymbol\epsilon} - \boldsymbol\epsilon)/\alpha_t$, also gilt $c = 1/\alpha_t^2 = 1 + e^{-\lambda}$. Für die Geschwindigkeit auf dem linearen Pfad gilt dieselbe Beziehung und dort ist $1/\alpha_t = 1 + e^{-\lambda/2}$. Bei gleichverteiltem $t$ ergeben sich die folgenden Gewichtungen, jeweils bis auf einen konstanten Faktor (Herleitung in Aufgabe :exref:`ex_design-space-implied-weighting`). EDM zieht die Rauschpegel aus einer eigenen Verteilung. In der entsprechenden Tabellenzeile steht $\sigma_{\mathrm{data}}$ für die Skala der Daten, die wir unten mit EDM einführen.

<!--en:da256756-->
| Zielfunktion | Pfad und Rauschpegel | $w(\lambda)$ |

<!--en:c3891fb2-->
| Schranke (ELBO) | beliebiger Pfad | $1$ |

<!--en:b4980c9a-->
| Rauschen, DDPM | VP, lineares $\beta$, $t$ gleichverteilt | $\lvert dt/d\lambda\rvert$ |

<!--en:578f4130-->
| Rauschen | linear, $t$ gleichverteilt | $\operatorname{sech}^2(\lambda/4)$, Maximum bei $\lambda = 0$ |

<!--en:710fa150-->
| Rauschen | Kosinus, $t$ gleichverteilt | $\operatorname{sech}(\lambda/2)$ |

<!--en:6028811a-->
| $\mathbf{v}$ | Kosinus, $t$ gleichverteilt | $e^{-\lambda/2}$ |

<!--en:878d5ddc-->
| Geschwindigkeit (Flow-Matching) | linear, $t$ gleichverteilt | $e^{-\lambda/2}$ |

<!--en:8974faf6-->
| EDM | $\log(\sigma_t/\alpha_t) \sim \mathcal{N}(-1.2, 1.2^2)$ | $\mathcal{N}(\lambda; 2.4, 2.4^2)\,(e^{-\lambda} + \sigma_{\mathrm{data}}^2)$ |

<!--en:fec6f9e5-->
Bei solchen Umformungen übersieht man leicht einen Faktor. Prüfen wir daher zwei Tabellenzeilen und das Maximum der DDPM-Gewichtung mit dem Rechner. `weighting` berechnet $w(\lambda)$ für gleichverteiltes $t$ ganz ohne algebraische Umformungen. Der Faktor $c(t)$ ist das Quadrat der Steigung der Zielgröße als Funktion des vorhergesagten Rauschens. Diese Steigung liefern `x0_eps` und `target` bei festem $\mathbf{x}_t$. Den Faktor $d\lambda/dt$ erhalten wir wie in `rates` durch Differenzieren im Vorwärtsmodus.

<!--en:6469e729-->
Die Geschwindigkeitsgewichtung stimmt auf alle ausgegebenen Stellen mit $e^{-\lambda/2}$ überein. Die $\mathbf{v}$-Gewichtung entspricht derselben Kurve mal $0.64$, und zwar von $\lambda = -8$ bis $4$. Die Konstante, am verrauschten Ende $0.642$, entspricht $2 \cdot 1.008/\pi$. Wir verwenden den Kosinuspfad von :citet:`Nichol.Dhariwal.2021`, also $\alpha_t \propto \cos\big(\tfrac{\pi}{2}(t + 0.008)/1.008\big)$ mit einem kleinen Versatz von $0.008$ und so skaliert, dass er bei eins beginnt. Sein Winkel wächst daher etwas langsamer als bei $\cos\frac{\pi t}{2}$. Der Versatz fällt nur bei fast rauschfreien Punkten ins Gewicht. Dort sinkt das Verhältnis auf $0.62$ bei $\lambda = 6$ und auf $0.53$ bei $\lambda = 8$. Bis auf eine Konstante und das rauschfreie Ende sind die $\mathbf{v}$-Vorhersage auf dem Kosinuspfad und Flow-Matching auf dem linearen Pfad also dieselbe Zielfunktion :cite:`Gao.Hoogeboom.Heek.ea.2024`. Der Rauschverlust auf dem VP-Pfad hat sein Maximum bei $\lambda = -0.9$, etwas auf der verrauschten Seite des Punkts, an dem Signal und Rauschen gleich stark sind. Die Geschwindigkeits- und Rauschmodelle aus :numref:`sec_diffusion-gaussian-paths` unterscheiden sich in der Gewichtung und lernen dieselbe Größe (:numref:`fig_diffusion-weightings`).

<!--en:dd2a113f-->
![Zielfunktionen für Diffusionsmodelle als Gewichtungen des Fehlers der Rauschvorhersage über dem logarithmierten Signal-Rausch-Verhältnis $\lambda$. (a) $w(\lambda)$ auf logarithmischer Skala. Die Schranke verläuft flach. Der Rauschverlust auf dem VP-Pfad mit linearem $\beta$ hat sein Maximum nahe $\lambda \approx -1$. Die $\mathbf{v}$-Vorhersage und Flow-Matching folgen derselben Kurve $e^{-\lambda/2}$. Die EDM-Gewichtung (mit $\sigma_{\mathrm{data}} = 0.5$ aus der Originalarbeit) ist glockenförmig. (b) Dichte von $\lambda$ bei gleichverteiltem $t$ auf den drei Pfaden sowie für die logit-normalverteilten Zeitpunkte von :citet:`Esser.Kulal.Blattmann.ea.2024` auf dem linearen Pfad.](../img/mdl-diffusion-weightings.svg)

<!--en:846b6517-->
Welche dieser Gewichtungen liefern Likelihood-Schranken? Überraschenderweise mehrere. Eine Gewichtung, die unterhalb eines Pegels $\mu$ eins und darüber null ist, liefert die Schranke :eqref:`eq_diffusion-elbo-continuous` für Daten, die bis zur Log-SNR $\mu$ verrauscht wurden. Ihr Pfad beginnt bei $\mu$, sodass nur Pegel unterhalb von $\mu$ in ihre Schranke eingehen. Eine Gewichtung, die mit $\lambda$ monoton fällt, ist eine positive Mischung solcher Stufenfunktionen. Ein solches $\mathcal{L}_w$ liefert daher die Schranke für Daten, denen wir gaußsches Rauschen mit zufälligem Pegel hinzufügen :cite:`Kingma.Gao.2023`. Für $w = e^{-\lambda/2}$ steigt die Mischdichte $-w'(\mu) = \tfrac12 e^{-\mu/2}$ zu niedrigen Log-SNR-Werten hin an. Die $\mathbf{v}$-Vorhersage und Flow-Matching trainieren also die Likelihood von Daten, die wir zufällig, meist stark, verrauschen. Sie lernen die groben Strukturen, die bei starkem Rauschen erhalten bleiben, und geben den feinsten Details, die eine Likelihood dominieren, wenig Gewicht. Der Rauschverlust von DDPM und die Gewichtung von EDM haben beide einen Gipfel und liefern keine Schranken dieser Art.

<!--en:08a7608c-->
### Die Zeitverteilung als Importance-Sampling

<!--en:30949ba4-->
In :eqref:`eq_diffusion-weighted-loss` gehen die Trainingszeitpunkte nur über das Produkt $w(\lambda) = 2c\, p\, |dt/d\lambda|$ ein. Grundsätzlich können wir die Dichte der Trainingszeitpunkte und die Gewichtung getrennt wählen. Ziehen wir $\lambda$ aus einer beliebigen Dichte $p(\lambda)$, die den gesamten Bereich abdeckt, gilt

<!--en:d596c229-->
die Importance-Sampling-Identität aus :numref:`sec_mdl-bayes-importance`. Damit verändert $p(\lambda)$ nur die Varianz der Schätzung. Die Wahl bleibt dennoch wichtig. Bei gleichverteiltem $t$ auf dem VP-Pfad mit linearem $\beta$ liegt fast die Hälfte der Ziehungen unter $\lambda = -3$ (Teilbild (b) in :numref:`fig_diffusion-weightings`), wo der Integrand der Schranke klein ist. Dort entspricht $\mathbf{x}_t$ fast dem Rauschen selbst. Das Rauschen lässt sich daraus also leicht vorhersagen. Deshalb wechselten :citet:`Nichol.Dhariwal.2021` zum Kosinuspfad. Um die Schranke selbst zu schätzen, zogen sie auch deren Schritte mit Importance-Sampling (Aufgabe :exref:`ex_design-space-weights-of-the-bound`).

<!--en:f4a1e89d-->
In der Praxis teilt man nicht durch $p(\lambda)$. Die Zeitverteilung wirkt also als implizite Gewichtung. Häufig verwendet man die *Logit-Normalverteilung* von :citet:`Esser.Kulal.Blattmann.ea.2024`, also $t = 1/(1 + e^{-(m + s n)})$ mit $n \sim \mathcal{N}(0, 1)$. Das Zentrum $m$ verschiebt die Zeitpunkte entlang des Pfades, die Streuung $s$ bestimmt, wie dicht sie beieinanderliegen, und $m = 0$ konzentriert sie um $t = \tfrac12$. Verwenden wir die Verteilung bei $m = 0$ ohne Division, multipliziert sie die Verlustgewichtung mit einer glockenförmigen Funktion um die Pfadmitte.

<!--en:d6330026-->
## Gestaltungsspielraum

<!--en:a22fd4be-->
Wer ein Diffusionsmodell vorstellt, legt einen Pfad, eine Zielgröße, eine Gewichtung, die Rauschpegel beim Training, ein Netz und einen Sampler fest. :citet:`Karras.Aittala.Aila.ea.2022` betrachteten diese Entscheidungen einzeln und zeigten, dass sie sich weitgehend unabhängig voneinander treffen lassen. Deshalb können wir sie einzeln vergleichen.

<!--en:4347bbe1-->
Eine dieser Entscheidungen, die Wahl der Zielgröße, scheint im vorigen Teil verschwunden zu sein. Dort führte jede Zielgröße zu einer Gewichtung desselben Fehlers der Rauschvorhersage. Aus der Gewichtung geht hervor, welche Funktion den Verlust minimiert. Sie sagt jedoch nichts darüber aus, wie schwer es einem endlichen Netz fällt, diese Funktion darzustellen, oder wie stark ein Fehler bei der Umrechnung zwischen zwei Ausgaben wächst. Zwei Verluste mit derselben Gewichtung können daher zu unterschiedlichen trainierten Netzen führen (im letzten Teil von Aufgabe :exref:`ex_design-space-implied-weighting` untersuchen Sie den Unterschied). Es spielt also weiterhin eine Rolle, was das Netz ausgeben soll.

<!--en:9e162eeb-->
### Die EDM-Tabelle

<!--en:0417e37d-->
Karras et al. bezeichnen ihr eigenes Verfahren als EDM. In den Spalten ihrer Tabelle lassen sich drei Modelle aus :numref:`tab_diffusion_models` und EDM selbst wie folgt darstellen:

<!--en:6f21075b-->
| Verfahren | Training: Pfad, Rauschpegel, Gewichtung | Netzausgabe | Sampler |

<!--en:fae3b5a9-->
| DDPM | VP, lineares $\beta$, $t$ gleichverteilt, einfacher Verlust | Rauschen | ancestrale Kette, 1000 Schritte |

<!--en:94398cef-->
| NCSN | $\mathbf{x}_0 + \sigma\boldsymbol\epsilon$, geometrische Folge von $\sigma$, Gewicht $\sigma^2$ | Score | Annealed Langevin Dynamics |

<!--en:f2a6dd45-->
| Flow-Matching | linear, $t$ gleichverteilt | Geschwindigkeit | Euler-Verfahren oder adaptiver ODE-Löser |

<!--en:f6561eeb-->
| EDM | $\log \bar\sigma \sim \mathcal{N}(-1.2, 1.2^2)$, Gewicht $1/c_{\mathrm{out}}^2$ | vorkonditioniertes rauschfreies Bild | Heun-Verfahren auf dem $\rho = 7$-Gitter |

<!--en:b9134297-->
NCSN :cite:`song2019generative` (:numref:`sec_mdl-score-based-generative-modeling`) skaliert die Daten nie ($\alpha = 1$). Das ist die *varianzexplodierende* Darstellung. EDM arbeitet auf jedem Pfad in diesen Koordinaten, also mit $\bar{\mathbf{x}} = \mathbf{x}/\alpha$ beim Rauschpegel $\bar\sigma = \sigma/\alpha$, genau wie `ode_sample`.

<!--en:6837bc90-->
EDM führt als neue Spalte die Parametrisierung ein. Sie legt fest, was das Netz ausgibt und wie wir seine Ein- und Ausgabe skalieren. Zwei Überlegungen bestimmen die Wahl. Die erste betrifft die Skala. Das Netz $F$ verwendet bei allen Rauschpegeln dieselben Gewichte. Deshalb sollten sowohl Eingabe als auch Ausgabe bei jedem Pegel Varianz eins haben. Für Daten mit Mittelwert null und Standardabweichung $\sigma_{\mathrm{data}}$ hat der Entrauscher die Form

<!--en:18a3f227-->
und wir trainieren ihn mit $\|D - \mathbf{x}_0\|^2$ und dem Gewicht $1/c_{\mathrm{out}}^2$. Damit hat auch die Zielgröße von $F$ Varianz eins (Aufgabe :exref:`ex_design-space-edm-preconditioning` leitet die Koeffizienten her). Bei schwachem Rauschen reicht der Entrauscher seine Eingabe durch ($c_{\mathrm{skip}} \to 1$, $c_{\mathrm{out}} \to 0$), und $F$ korrigiert sie nur. Bei starkem Rauschen liefert $F$ die Schätzung, skaliert auf die Daten ($c_{\mathrm{skip}} \to 0$, $c_{\mathrm{out}} \to \sigma_{\mathrm{data}}$). Für nicht zentrierte Daten gilt dieselbe Herleitung mit dem quadratischen Mittel anstelle von $\sigma_{\mathrm{data}}$, da dann an die Stelle jeder Varianz das zweite Moment um null tritt. Karras et al. verwenden für ihre natürlichen Bilder $\sigma_{\mathrm{data}} = 0.5$. Unten messen wir den Wert für unsere eigenen Daten.

<!--en:57e688b3-->
An der Skalierung erkennen wir auch das Problem eines einfachen Rauschprädiktors. Er rechnet seine Ausgabe in ein rauschfreies Bild um, indem er durch den Signalkoeffizienten teilt, $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t \hat{\boldsymbol\epsilon})/\alpha_t$. Auf dem VP-Pfad gilt $\alpha_1 \approx 0.007$, sodass die Division nahe dem reinen Rauschen den Fehler der Rauschvorhersage etwa auf das 150-Fache verstärkt. Dort lässt sich der Rauschverlust schon durch Kopieren der Eingabe fast minimieren. Er bietet kaum Anreiz, den Fehler unter das für die Division erforderliche Verhältnis 1 zu 150 zu senken. EDM teilt nie durch eine kleine Zahl.

<!--en:7e510ca4-->
Die zweite Überlegung betrifft das Darstellungsvermögen eines endlichen Netzes. Bilder liegen nahe einer Menge, deren Dimension weit unter der Pixelzahl liegt, während das Rauschen alle Dimensionen ausfüllt. Bei starkem Rauschen muss das Netz aus wenig Signal ein ganzes Bild erzeugen. Das rauschfreie Bild ist dann die einfachere Ausgabe. Dort gilt $c_{\mathrm{skip}} \to 0$, und $F$ sagt ein skaliertes rauschfreies Bild vorher. Bei schwachem Rauschen sagt $F$ ein Vielfaches des Rauschens vorher, eine kleine Korrektur seiner Eingabe. :citet:`Li.He.2026` führen diese Überlegung bis an ihre Grenze. Transformer auf großen Pixel-Patches scheitern an der Vorhersage des Rauschens oder der Geschwindigkeit, können das rauschfreie Bild aber vorhersagen.

<!--en:95b3ab72-->
`EDM` versieht ein Netz $F(\mathbf{x}, t)$ in den Koordinaten eines Pfades mit dieser Vorkonditionierung ($c_{\mathrm{in}} \bar{\mathbf{x}} = \mathbf{x}/n$ mit $n^2 = \sigma^2 + \alpha^2 \sigma_{\mathrm{data}}^2$, selbst bei $\alpha = 0$ endlich) und sagt das rauschfreie Bild vorher. `edm_loss` zieht die log-normalverteilten Rauschpegel und wendet das Gewicht an. Karras et al. geben dem Netz den Rauschpegel außerdem als $\tfrac14 \log \bar\sigma$. Unser `EDM` übergibt wie alle anderen Modelle dieses Kapitels den Zeitpunkt $t$ des Pfades, damit wir in den folgenden Vergleichen nur die Ausgabe verändern.

<!--en:57930892-->
### Rauschpegel und räumliche Skalen im Bild

<!--en:ca5a8afc-->
Welche Bildinformation geht bei einem bestimmten Rauschpegel verloren? Bei einer orthonormalen zweidimensionalen Fourier-Transformation hat weißes Rauschen mit Standardabweichung $\sigma$ bei jeder Frequenz die Leistung $\sigma^2$. Bei Bildern konzentriert sich die Leistung dagegen auf niedrige Frequenzen. Eine Frequenz wird verdeckt, sobald dort die Rauschleistung die Bildleistung übersteigt. Die Codezelle stellt die Leistung der Testbilder, gemittelt über Ringe mit Radius $r$ in Zyklen pro Bild, der Leistung bei drei Rauschpegeln gegenüber.

<!--en:c802195a-->
Von einem bis vierzehn Zyklen pro Bild fällt die Bildleistung um drei Größenordnungen. Bei $\sigma = 0.1$ bleibt die Rauschleistung überall darunter. Bei $\sigma = 0.3$ verdeckt das Rauschen die Ringe ab etwa 9 Zyklen pro Bild, bei $\sigma = 1$ schon ab etwa 4. Rauschen hinzuzufügen wirkt also wie ein Tiefpassfilter, dessen Grenzfrequenz mit wachsendem $\sigma$ sinkt. Das Sampling verläuft vom Groben zum Feinen. Die Schritte bei starkem Rauschen legen die Silhouette eines Kleidungsstücks fest, die letzten Schritte seine Textur :cite:`Dieleman.2024`.

<!--en:6a735f81-->
So erklärt sich auch, warum ein Rauschplan von der Auflösung abhängt. Bei $56 \times 56$ und gleichem Rauschen pro Pixel erhalten wir durch Mittelung über jeden $2 \times 2$-Block wieder das $28 \times 28$-Bild. Der Mittelwert der vier Rauschwerte hat aber nur die Varianz $\sigma^2/4$. Für die grobe Struktur hat sich das Signal-Rausch-Verhältnis vervierfacht. Ein auf eine bestimmte Auflösung abgestimmter Rauschplan macht die Aufgabe bei höherer Auflösung also zu leicht. :citet:`Hoogeboom.Heek.Salimans.2023` verschieben $\lambda_t$ um den Logarithmus des Verhältnisses der Pixelzahlen nach unten (Aufgabe :exref:`ex_design-space-resolution-schedule`, :numref:`fig_diffusion-schedules`).

<!--en:d362d8b2-->
![Rauschpläne auf der Zeitskala der Log-SNR: $\lambda_t$ über $t$ für den VP-Pfad mit linearem $\beta$, den Kosinuspfad und den linearen Pfad sowie für den bei vierfacher Seitenlänge verschobenen Kosinuspfad ($\lambda_t - \log 16$). Der VP-Pfad endet bei $\lambda \approx -10$, die anderen bei $-\infty$. Das Band markiert die Log-SNR, bei der das Rauschen in Fashion-MNIST die Bildanteile oberhalb von 6 Zyklen pro Bild verdeckt.](../img/mdl-diffusion-schedules.svg)

<!--en:34ace9e9-->
Blockweise Mittelung verstärkt auch das Signal, das am Ende eines Rauschplans übrig bleibt. Unser VP-Pfad endet bei $\lambda \approx -10$. Dort beträgt das Signal-Rausch-Verhältnis für den Mittelwert aller 784 Pixel höchstens $784\, e^{-10} \approx 0.04$, viel zu wenig für ein erkennbares Signal. Der Rauschplan von Stable Diffusion endete bei einer Log-SNR von etwa $-5$ für latente Darstellungen mit $64 \times 64$ Werten pro Kanal. Der Mittelwert eines Kanals erreicht dort $4096\, e^{-5} \approx 28$. Das Modell lernte, die verbleibende Helligkeitsinformation zu nutzen. Da sein Sampler mit Rauschen mit Mittelwert null startet, konnte es weder sehr helle noch sehr dunkle Bilder erzeugen :cite:`Lin.Liu.Li.ea.2024`.

<!--en:54be2b45-->
### Vergleich einzelner Faktoren

<!--en:7bc87fae-->
Nun können wir die Entscheidungen einzeln an Bildern prüfen. Als Basis verwenden wir das Verfahren aus :numref:`sec_diffusion-images`, die Geschwindigkeit auf dem linearen Pfad mit gleichverteiltem $t$. Jede weitere Zeile ändert einen Faktor:

<!--en:5021597c-->
| Zeile | Änderung gegenüber der Basis |

<!--en:3f5d6a23-->
| `t logit-normal` | logit-normalverteilte Zeitpunkte mit $m = 0$ und $s = 1$ wie bei :citet:`Esser.Kulal.Blattmann.ea.2024` |

<!--en:b7266e7d-->
| `t logit-normal, m = -0.5` | wie oben, aber mit Zentrum bei $t \approx 0.38$, näher an den Daten |

<!--en:84eb6986-->
| `cosine path, v` | $\mathbf{v}$ auf dem Kosinuspfad, laut Gewichtungstabelle dieselbe Zielfunktion wie bei der Basis |

<!--en:4d3eb97f-->
| `t at EDM's levels` | log-normalverteilte Rauschpegel von EDM als Kontrolle für die nächste Zeile |

<!--en:2f9bbdbc-->
| `EDM, clean image` | vorkonditioniertes rauschfreies Bild nach EDM mit den zugehörigen Rauschpegeln und Gewichten |

<!--en:e48d361a-->
Jedes Modell ist ein unkonditioniertes `d2l.UNet` mit 64 Kanälen. Wir trainieren es mit zwei Zufallsstartwerten über je 5000 Aktualisierungen, ein Drittel des Rechenbudgets aus :numref:`sec_diffusion-images`. Für das Basismodell, das Kontrollmodell und EDM, deren Vergleich am schwierigsten zu deuten ist, verwenden wir vier Zufallsstartwerte. Mit dem Heun-Verfahren erzeugen wir für alle Modelle aus denselben 5000 Rauschbildern Samples und vergleichen sie anhand der Fréchet-Distanz (FD) mit 5000 Testbildern. Dabei stellt sich die Frage, ob der Sampler einen fairen Vergleich erlaubt. Die EDM-Originalarbeit verwendet für das Sampling das $\rho = 7$-Gitter. Für ein mit gleichverteilten Zeitpunkten trainiertes Modell schnitt dieses Gitter bei gleichem Rechenbudget in :numref:`sec_diffusion-samplers` jedoch nicht besser ab als das lineare. Für ein mit EDM-Rauschpegeln trainiertes Modell könnte es dennoch besser geeignet sein. Das prüfen wir mit der Kontrollzeile. Da ein einzelnes Gitter ein Verfahren begünstigen könnte, ziehen wir für jedes Modell Samples auf beiden Gittern, dem EDM-Gitter (49 Auswertungen) und dem linearen Gitter (48). Zunächst bestimmt die Codezelle die Pixelstatistik für den EDM-Parameter $\sigma_{\mathrm{data}}$ sowie den Grundwert von :numref:`subsec_diffusion-fd-reference`, die FD zwischen zwei Mengen echter Bilder.

<!--en:ad2a398a-->
Fashion-MNIST besteht größtenteils aus schwarzem Hintergrund. Die Pixel haben einen Mittelwert von $-0.43$ und ein quadratisches Mittel von $0.83$, deutlich mehr als der in der Veröffentlichung verwendete Wert $0.5$. Die Codezelle übergibt den Messwert an `EDM`. Für zwei Mengen von je 5000 echten Bildern ergibt sich im Vergleich eine FD von knapp $0.2$.

<!--en:558a9e89-->
`logit_normal_loss` entspricht `regression_loss` mit logit-normalverteilten statt gleichverteilten Zeitpunkten. Auf dem linearen Pfad gilt $\bar\sigma = t/(1 - t)$ und damit $\log \bar\sigma = m + s n$. Mit $m = -1.2$ und $s = 1.2$ trainieren wir das Geschwindigkeitsmodell also bei den log-normalverteilten Rauschpegeln von EDM. Die so bezeichnete Zeile unterscheidet sich von der EDM-Zeile nur in Parametrisierung und Gewichtung. Auch hier sind die Zeitpunkte logit-normalverteilt. Ihr Zentrum liegt weiter in Richtung der Daten als in der Zeile $m = -0.5$, nämlich bei $t \approx 0.23$.

<!--en:000ebe4f-->
Die Schleife trainiert jede Variante und erzeugt für jedes Modell auf beiden Gittern Samples aus demselben Rauschen. Für jede Zeile und jedes Gitter gibt sie die mittlere FD über die Zufallsstartwerte und in Klammern die FD jedes einzelnen Zufallsstartwerts aus. Ein Ergebnis lässt sich vorab abschätzen. Der $\mathbf{v}$-Verlust auf dem Kosinuspfad entspricht der Zielfunktion der Basis in anderen Koordinaten und sollte daher nicht schlechter abschneiden.

<!--en:ec06881c-->
Nach 5000 Aktualisierungen, einem Viertel der 20 000 Schritte für die latente Diffusion in :numref:`sec_diffusion-latent`, zeigt die Tabelle die Sample-Qualität früh im Training für mehrere Zufallsstartwerte und zwei Sampling-Gitter:

<!--en:f77f3b0c-->
1. **Einfluss des Sampling-Gitters:** Jedes auf dem linearen Pfad trainierte Geschwindigkeitsmodell erreicht auf dem linearen Gitter eine um $0.3$–$1.3$ niedrigere FD als auf dem EDM-Gitter. Das gilt auch für das mit EDM-Rauschpegeln trainierte Kontrollmodell. Das vorkonditionierte EDM-Modell ist dagegen unempfindlich gegenüber der Gitterwahl. Das Training bei EDM-Rauschpegeln allein macht das EDM-Gitter also nicht zur besseren Wahl.

<!--en:3b9a57ca-->
2. **Verteilung der Rauschpegel beim Training:** Zentrieren wir die logit-normalverteilten Trainingszeitpunkte bei $m = 0$ ($t = 0.5$), verschlechtert sich die FD auf dem EDM-Gitter um $1.1$–$1.3$ (auf dem linearen Gitter um $0.5$–$0.75$) gegenüber gleichverteiltem $t$. Dieser Nachteil verschwindet, wenn wir das Zentrum in Richtung der Daten verschieben ($m = -0.5$, $t \approx 0.38$) oder die EDM-Rauschpegel verwenden ($m = -1.2$, $t \approx 0.23$).

<!--en:d393d409-->
3. **Zielgröße und Vorkonditionierung:** Auf dem linearen Gitter überlappen sich die FD-Bereiche über die Zufallsstartwerte für das Basismodell mit Geschwindigkeitsvorhersage, die $\mathbf{v}$-Vorhersage auf dem Kosinuspfad und EDM.

<!--en:720c53b4-->
Die Samples für den ersten Zufallsstartwert auf dem EDM-Gitter entstehen aus denselben zehn Rauschbildern. Jede Zeile zeigt ein Verfahren in der Reihenfolge der Tabelle:

<!--en:3dfb4f58-->
In den meisten Spalten erzeugen alle sechs Verfahren aus demselben Rauschen dasselbe Kleidungsstück mit Unterschieden im Detail. Nur in wenigen Spalten ändert sich die Art des Kleidungsstücks. Die Verfahren lernen fast dieselbe Abbildung vom Rauschen zum Bild, wie für denselben A-posteriori-Mittelwert bei verschiedenen Gewichtungen zu erwarten ist.

<!--en:380c91a7-->
Der Testverlust bei den einzelnen Rauschpegeln zeigt, wo das zentrierte logit-normalverteilte Modell schlechter abschneidet. Die Codezelle wertet es und das Basismodell bei einzelnen Rauschpegeln auf denselben 2000 Testbildern mit demselben Rauschen aus.

<!--en:a380813f-->
Bei $t = \tfrac12$, wo die meisten seiner Trainingszeitpunkte liegen, schneidet das Modell mit Logit-Normalverteilung etwas besser ab als das Basismodell (Verlust 127 gegenüber 131). Bei $t = 0.01$ schneidet es deutlich schlechter ab (etwa 670 bis 700 gegenüber 420 bis 440). Weniger als einer von dreitausend Trainingszeitpunkten liegt unter $t = 0.03$, bei der Gleichverteilung sind es drei von hundert. Die letzten Schritte jedes Samplers nutzen diese Rauschpegel, um feine Details festzulegen. Esser et al. wählten das Zentrum für die latenten Darstellungen wesentlich größerer Bilder. Nach unserer Überlegung zur Skala bildet sich die Struktur eines $28 \times 28$-Bildes bei schwächerem Rauschen. Genau dorthin verlagert die Variante mit verschobenem Zentrum die Trainingspegel.

<!--en:87a4e31b-->
### Wahl des Verfahrens

<!--en:00eb6a2c-->
Aus unserer Analyse ergeben sich drei Regeln für den Entwurf eines neuen Modells:

<!--en:4754eb46-->
1. **Wählen Sie eine gut konditionierte Zielgröße auf einem Pfad, der reines Rauschen erreicht ($\alpha_1 = 0$):** Die Geschwindigkeitsvorhersage auf dem linearen Pfad, die $\mathbf{v}$-Vorhersage auf dem Kosinuspfad und die EDM-Vorkonditionierung bleiben an beiden Enden von $[0, 1]$ gut skaliert. Die einfache Rauschvorhersage teilt dagegen durch $\alpha_t \to 0$, wenn sie $\hat{\mathbf{x}}_0$ bei starkem Rauschen rekonstruiert.

<!--en:60be8586-->
2. **Legen Sie die Trainingspegel dorthin, wo die räumliche Struktur entsteht:** Bei $28 \times 28$-Bildern schneiden gleichverteiltes $t$ oder eine zu schwachem Rauschen verschobene Logit-Normalverteilung ($m < 0$) besser ab als eine um $t = 0.5$ zentrierte Logit-Normalverteilung. Steigt die Bildauflösung um den Faktor $r$, verschieben Sie die Log-SNR um $\log r^2$ nach unten (:numref:`subsec_diffusion-design-space-scales`).

<!--en:c79081ca-->
3. **Richten Sie die Verlustgewichtung nach den Samples aus, nicht nach der Schranke:** Die Gewichtung $w(\lambda) \equiv 1$ trainiert die Likelihood-Schranke und verwendet einen großen Teil des Trainingsaufwands auf feine Details bei schwachem Rauschen. Gewichtungen, die hohe Log-SNR-Werte schwächer gewichten, etwa $w(\lambda) = e^{-\lambda/2}$ (Flow-Matching und $\mathbf{v}$-Vorhersage), haben in der Praxis gute Bildsamples geliefert :cite:`Kingma.Gao.2023`. Bei unserer Mischverteilung erzielte das kleine Netz mit der Gewichtung der Schranke nicht einmal seine beste Fluss-Likelihood.

<!--en:4b2b8d49-->
Das Standardverfahren dieses Kapitels, die Geschwindigkeitsregression auf dem linearen Pfad mit gleichverteiltem $t$ und Integration auf dem linearen Gitter, erfüllt diese Kriterien ohne zusätzliche Hyperparameter. In :numref:`sec_diffusion-latent` betrachten wir, wie sich Netzauswertungen für hochfrequente Pixeldetails einsparen lassen, indem wir die Diffusion in einen gelernten latenten Raum verlagern.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:711539b9-->
Kehren wir die Vorwärtskette um, die den Daten Rauschen hinzufügt, erhalten wir ein latentes Variablenmodell. Seine negative Log-Likelihood ist durch die ELBO mit umgekehrtem Vorzeichen nach oben beschränkt. Die Schranke summiert die Fehler der Rauschvorhersage, gewichtet mit dem halben Rückgang der Log-SNR pro Schritt $\tfrac12(\lambda_{k-1} - \lambda_k)$. Im Grenzübergang zu kontinuierlicher Zeit ergibt sich $\frac12 \int \mathbb{E}\|\boldsymbol\epsilon - \hat{\boldsymbol\epsilon}_{\boldsymbol\theta}\|^2\, d\lambda$. Allgemeiner lässt sich jede übliche Zielfunktion für Diffusionsmodelle und Flow-Matching in Log-SNR-Koordinaten als $\mathcal{L}_w = \frac12 \int w(\lambda)\, \mathbb{E}\|\hat{\boldsymbol\epsilon}_{\boldsymbol\theta} - \boldsymbol\epsilon\|^2\, d\lambda$ schreiben. Monoton fallende Gewichtungen $w(\lambda)$ entsprechen Likelihood-Schranken für durch Rauschen erweiterte Daten. Insbesondere haben Flow-Matching auf dem linearen Pfad und die $\mathbf{v}$-Vorhersage auf dem Kosinuspfad dieselbe Gewichtung $w(\lambda) \propto e^{-\lambda/2}$. Auf der Mischverteilung mit drei Modi erreicht ein trainiertes Netz die negative Log-Likelihood des exakten Orakels bis auf $0.005$ Nat, sowohl mit der kontinuierlichen ELBO als auch mit der Variablentransformation der Wahrscheinlichkeitsfluss-ODE. Bei begrenzter Kapazität liefert $w(\lambda) \equiv 1$ eine engere ELBO, während die ungewichtete Rauschregression bessere Fluss-Likelihoods und weniger Ausreißer unter den Samples ergibt. Die EDM-Vorkonditionierung skaliert Netzeingaben, Skip-Verbindungen und Zielgrößen über alle Rauschpegel auf Einheitsvarianz. Da weißes Rauschen auf natürliche Bilder wie ein Tiefpassfilter wirkt, müssen wir bei einer Erhöhung der Bildauflösung um den Faktor $r$ den Log-SNR-Plan um $2 \log r$ nach unten verschieben.

<!--en:2294243a-->
## Aufgaben

<!--en:ea86337d-->
1. [code] **Die Gewichte der Schranke.** ● Sei $\omega_k$ das Gewicht in :eqref:`eq_diffusion-lt-eps`.
    1. Berechnen Sie $\omega_k$ für $\nu_k = \beta_k$ und für $\nu_k = \tilde\beta_k$ in jedem Schritt $k = 2, \ldots, 1000$ der VP-Kette und tragen Sie beide Gewichte auf einer logarithmischen Achse auf. Welcher Anteil von $\sum_k \omega_k$ entfällt jeweils auf die Schritte $k \le 20$? Welchen Anteil weist der einfache Verlust diesen Schritten zu?
    1. Werten Sie die Schranke :eqref:`eq_diffusion-elbo-terms` der Kette für die Mischverteilung aus. Verwenden Sie den exakten Rauschprädiktor, $\nu_k = \tilde\beta_k$ und den Rekonstruktionsterm aus dem Text beim Rauschpegel von Schritt 1. Vergleichen Sie das Ergebnis mit der wahren negativen Log-Likelihood und mit der kontinuierlichen Schranke aus dem Text. Welche Schritte dürften nach dem Beispiel mit zwei Schritten am meisten zur Überschätzung beitragen und warum?
    1. Für eine Minibatch-Schätzung der Schranke der Kette ziehen wir $k$ aus einer Gleichverteilung. :citet:`Nichol.Dhariwal.2021` ziehen stattdessen $k$ mit einer Wahrscheinlichkeit $p_k$ proportional zum quadratischen Mittel der einzelnen Sample-Werte von $L_{k-1}$ und teilen den gezogenen Term durch $p_k$. Warum verringert das die Varianz der Schätzung? Welche Schritte würde ein solcher Sampler bei der Mischverteilung am häufigsten ziehen?
   :exlabel:`ex_design-space-weights-of-the-bound`

<!--en:2c7944c6-->
1. **Die implizite Gewichtung.** Verwenden Sie :eqref:`eq_diffusion-weighted-loss`.
    1. Zeigen Sie, dass für den Geschwindigkeitsverlust auf dem linearen Pfad mit gleichverteiltem $t$ die Gewichtung $w(\lambda) \propto e^{-\lambda/2}$ gilt und dass der $\mathbf{v}$-Verlust auf dem Kosinuspfad $\alpha_t = \cos\frac{\pi t}{2}$, $\sigma_t = \sin\frac{\pi t}{2}$ dieselbe Gewichtung hat.
    1. Zeigen Sie, dass für den Rauschverlust auf dem Kosinuspfad mit gleichverteiltem $t$ die Gewichtung $w(\lambda) \propto \operatorname{sech}(\lambda/2)$ gilt. Berechnen Sie $w$ für den Rauschverlust und den $\mathbf{v}$-Verlust auf dem VP-Pfad mit linearem $\beta$ (`vp`).
    1. Welche dieser Gewichtungen sind monoton in $\lambda$? Welche Rauschpegel gewichtet der Rauschverlust auf dem VP-Pfad schwächer als der $\mathbf{v}$-Verlust auf demselben Pfad? Um welchen Faktor unterscheiden sich die Gewichte im letzten Schritt?
    1. (Code.) Nach dem ersten Teil bilden der $\mathbf{v}$-Verlust auf dem Kosinuspfad und der Geschwindigkeitsverlust auf dem linearen Pfad eine einzige Zielfunktion. Dennoch trainieren wir damit im Vergleich aus dem Text zwei verschiedene Netze. Werten Sie wie in der letzten Codezelle des Vergleichs die beiden Modelle für den ersten Zufallsstartwert bei einzelnen Rauschpegeln aus. Rechnen Sie dazu beide Ausgaben bei derselben Log-SNR $\lambda$ mit `make_pred` in Rauschvorhersagen um. Bei welchen Pegeln unterscheiden sie sich? Auf beiden Pfaden erhält das Netz denselben Punkt mit unterschiedlicher Skalierung. Wie ist die Eingabe auf jedem Pfad bei $t = 1/2$ skaliert und wie könnte der Unterschied Ihre Beobachtung erklären?
   :exlabel:`ex_design-space-implied-weighting`

<!--en:4a91eff7-->
1. [code] **Likelihoods in vielen Dimensionen.** ● In $d$ Dimensionen benötigt die exakte Divergenz in `pf_nll` bei jeder Auswertung $d$ Ableitungen im Vorwärtsmodus. Ersetzen Sie sie durch den Hutchinson-Spurschätzer $\mathbf{r}^\top (\partial \mathbf{u}/\partial \mathbf{x})\, \mathbf{r}$ aus :numref:`sec_mdl-neural-odes-cnf` mit einem Rademacher-Vektor $\mathbf{r}$ pro Schritt. Dafür genügt eine Ableitung :cite:`Grathwohl.Chen.Bettencourt.ea.2018`.
    1. Zeigen Sie, dass die resultierende Schätzung von $\log p$ erwartungstreu ist. Die Varianz für einen einzelnen Testvektor wird im ersten Teil von Aufgabe :exref:`ex_mdl-odes-solvers-hutchinson-variance` in :numref:`sec_mdl-neural-odes-cnf` hergeleitet. Messen Sie die Standardabweichung der Schätzung pro Punkt auf der Mischverteilung. Vergleichen Sie sie mit dem Abstand von wenigen Tausendstel Nat zwischen dem trainierten Netz und dem exakten Feld. Wie viele Testvektoren pro Schritt oder wie viele Punkte brauchen Sie, um diesen Abstand aufzulösen?
    1. Addieren Sie zu jedem Pixel von 100 Testbildern gleichverteiltes Rauschen mit der Breite einer Graustufe, $2/255$. Wenden Sie die Schätzung dann auf das Basismodell des Vergleichs an (Geschwindigkeit auf dem linearen Pfad). Geben Sie das Ergebnis in Bit pro Dimension an, $\big(-\log p(\mathbf{x})/d + \log 127.5\big)/\log 2$. Das entspricht der Anzahl Bit pro Pixel des 8-Bit-Bildes. Eine über die 256 Graustufen gleichförmige Dichte ergibt genau 8 Bit. Ein funktionierendes Modell erreicht deutlich weniger, nur wenige Bit oder weniger. Bei einem Wert nahe oder über 8 fehlt meist der Term $\log 127.5$ oder ein Vorzeichen ist falsch. Was würde sich bei 15 000 Aktualisierungen ändern?
   :exlabel:`ex_design-space-hutchinson`

<!--en:063a1299-->
1. **Auflösung und Rauschplan.** Schreiben Sie $\mathrm{SNR}_t = \alpha_t^2/\sigma_t^2$ für Daten mit Einheitsvarianz pro Pixel.
    1. Zeigen Sie, dass die Multiplikation der Daten mit einer Konstanten $\kappa$ das Verhältnis von Signal- zu Rauschvarianz in $\mathbf{x}_t$ mit $\kappa^2$ multipliziert, und zwar zu jedem Zeitpunkt $t$.
    1. Vergrößern Sie jedes Bild auf $28 r \times 28 r$ Pixel, indem Sie jedes Pixel in einem $r \times r$-Block wiederholen, und verrauschen Sie das große Bild mit demselben Rauschplan. Zeigen Sie, dass die blockweise Mittelung des verrauschten Bildes das ursprüngliche Bild mit dem Signal-Rausch-Verhältnis $r^2\, \mathrm{SNR}_t$ ergibt.
    1. Betrachten Sie $r = 32$, also ein $896 \times 896$-Bild. Berechnen Sie $r^2\, \mathrm{SNR}_1$ für `vp` und bestimmen Sie den Zeitpunkt auf dem $28 \times 28$-Pfad mit demselben Verhältnis. Was bedeutet das Ergebnis für ein mit diesem Rauschplan trainiertes $896 \times 896$-Modell, dessen Sampler mit reinem Rauschen startet?
    1. Bei welcher Verschiebung von $\lambda_t$ durchläuft das gemittelte große Bild dieselben Signal-Rausch-Verhältnisse wie das $28 \times 28$-Modell? Vergleichen Sie die Verschiebung mit der von :citet:`Hoogeboom.Heek.Salimans.2023`.
    1. (Code.) Vergrößern Sie Fashion-MNIST durch Wiederholen der Pixel auf $56 \times 56$. Trainieren Sie das Basismodell des Vergleichs darauf in zwei Varianten mit je zwei Zufallsstartwerten, einmal auf dem linearen Pfad und einmal mit einer Verschiebung von $\lambda_t$ um $-\log 4$. Übersteigt die Verbesserung durch die Verschiebung die Streuung zwischen den Zufallsstartwerten, wie der vierte Teil erwarten lässt?
   :exlabel:`ex_design-space-resolution-schedule`

<!--en:23666b3c-->
1. **Der Vorwärts-Posterior.** Leiten Sie :eqref:`eq_diffusion-posterior` her.
    1. Schreiben Sie

       $$q(\mathbf{x}_{k-1} \mid \mathbf{x}_k, \mathbf{x}_0) \propto q(\mathbf{x}_k \mid \mathbf{x}_{k-1})\, q(\mathbf{x}_{k-1} \mid \mathbf{x}_0)$$

       als Produkt zweier Gauß-Dichten in $\mathbf{x}_{k-1}$. Addieren Sie ihre Präzisionen und kombinieren Sie ihre Mittelwerte mit Gewichten proportional zur jeweiligen Präzision.
    1. Zeigen Sie, dass $\tilde\beta_k \le \beta_k$ gilt, mit Gleichheit nur für $\sigma_{k-1}^2 = 1$. Wo in der Kette ist $\tilde\beta_k$ wesentlich kleiner als $\beta_k$?
    1. Die Daten im Beispiel mit zwei Schritten seien nun gemäß $\mathcal{N}(0, a^2)$ verteilt. Zeigen Sie, dass die Varianz $\nu_2$, die die Schranke minimiert, gleich $\mathrm{Var}(x_1 \mid x_2)$ ist. Zeigen Sie außerdem, dass sie gegen $\tilde\beta_2$ konvergiert, wenn $a \to 0$, und den Wert $\beta_2$ annimmt, wenn $a = 1$ gilt. Das sind die beiden Varianten von :citet:`ho2020denoising`.
   :exlabel:`ex_design-space-forward-posterior`

<!--en:fbc3f79d-->
1. **Die Vorkonditionierung von EDM.** ● Die Daten $\mathbf{x}_0$ seien unabhängig vom Rauschen, mit Mittelwert null und Varianz $\sigma_{\mathrm{data}}^2$ pro Koordinate.
    1. Wählen Sie $c_{\mathrm{in}}$ so, dass die Eingabe $c_{\mathrm{in}}(\mathbf{x}_0 + \bar\sigma\boldsymbol\epsilon)$ von $F$ Einheitsvarianz hat. Schreiben Sie die Trainingszielgröße von $F$ als $(\mathbf{x}_0 - c_{\mathrm{skip}}\, \bar{\mathbf{x}})/c_{\mathrm{out}}$. Wählen Sie $c_{\mathrm{skip}}$ so, dass die Varianz der Zielgröße minimal ist, und dann $c_{\mathrm{out}}$ so, dass diese Varianz eins beträgt. Vergleichen Sie das Ergebnis mit :eqref:`eq_diffusion-edm-precond`. Wiederholen Sie die Rechnung für Daten mit einem von null verschiedenen Mittelwert mit zweiten Momenten anstelle der Varianzen. Zeigen Sie, dass $\sigma_{\mathrm{data}}$ dann das quadratische Mittel ist.
    1. Zeigen Sie, dass der Verlust $\|D - \mathbf{x}_0\|^2 / c_{\mathrm{out}}^2$ gleich $\|F - \textrm{target}\|^2$ ist. Zeigen Sie außerdem, dass er als Rauschverlust dem Fehler der Rauschvorhersage multipliziert mit $(e^{-\lambda} + \sigma_{\mathrm{data}}^2)/\sigma_{\mathrm{data}}^2$ entspricht.
    1. Zeigen Sie für $\log \bar\sigma \sim \mathcal{N}(-1.2, 1.2^2)$, dass die EDM-Gewichtung $w(\lambda) \propto \mathcal{N}(\lambda; 2.4, 2.4^2)\, (e^{-\lambda} + \sigma_{\mathrm{data}}^2)$ beträgt. Wo liegt ihr Maximum? :citet:`Kingma.Gao.2023` machen sie monoton, indem sie für alle $\lambda$ unterhalb der Maximalstelle den Maximalwert beibehalten. Was wird nach dem Satz aus dem Text durch diese Änderung aus der EDM-Zielfunktion?
   :exlabel:`ex_design-space-edm-preconditioning`

<!--en:c9739d21-->
[Diskussionen](https://d2l.discourse.group/t/36891)

<!--en:0d841739-->
Likelihood, Gewichtung und Gestaltungsspielraum bei Diffusionsmodellen<br>**Variationsschranke (ELBO) · Log-SNR-Gewichtungen w(λ) · Likelihood der Wahrscheinlichkeitsfluss-ODE · EDM-Vorkonditionierung · Skalierung der Auflösung**

<!--en:4281c7c2-->
::: {.slide title="Wie hängt Rauschregression mit Maximum-Likelihood zusammen?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:a61dd408-->
- **1. Variationsschranke (ELBO):** Sohl-Dickstein et al. (2015) und Ho et al. (2020) zeigten, dass die Rückwärtskette ein latentes Variablenmodell bildet. Gewichtete Fehler der Rauschvorhersage beschränken seine negative Log-Likelihood (NLL).

<!--en:c4e1be51-->
- **2. Likelihood der Wahrscheinlichkeitsfluss-ODE:** Song et al. (2021) integrierten $-\nabla \cdot \mathbf{u}_t(\mathbf{x}_t)$ entlang der deterministischen ODE-Trajektorie und erhielten die exakte Log-Dichte $\log p_{\boldsymbol\theta}(\mathbf{x}_0)$.

<!--en:092fc48b-->
- **3. Log-SNR-Gewichtung $w(\lambda)$:** Kingma und Gao (2023) bewiesen, dass sich jede Diffusions- und Flusszielfunktion auf $\frac{1}{2}\int w(\lambda)\,\mathbb{E}\|\hat{\boldsymbol\epsilon}_{\boldsymbol\theta} - \boldsymbol\epsilon\|^2\,d\lambda$ zurückführen lässt.

<!--en:c410087f-->
::: {.slide title="Die Rückwärtskette als latentes Variablenmodell (ELBO)"}

<!--en:37bbea7c-->
[Obere Variationsschranke für die negative Log-Likelihood]{.kicker}

<!--en:309c4dd4-->
- **Exakte Differenz in der Jensen-Ungleichung:** Der Abstand zur Schranke beträgt $D_{\textrm{KL}}\big(q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \,\|\, p_{\boldsymbol\theta}(\mathbf{x}_{1:T} \mid \mathbf{x}_0)\big)$. Er verschwindet, wenn die Rückwärtskette mit der wahren zeitumgekehrten A-posteriori-Verteilung übereinstimmt.

<!--en:a95eaeed-->
- **Strafe für Modusausfall:** Im Mittel über $p(\mathbf{x}_0)$ gilt $\mathbb{E}_p[L(\mathbf{x}_0)] \ge H(p) + D_{\textrm{KL}}(p \,\|\, p_{\boldsymbol\theta})$. Jeder ausgelassene Modus fällt stark ins Gewicht.

<!--en:7e0bcafe-->
::: {.slide title="Zerlegung der ELBO in KL-Divergenzen pro Schritt"}

<!--en:f7e11043-->
[Prior-Term, Entrauschungsterme und Rekonstruktionsterm]{.kicker}

<!--en:2fc0c7b3-->
- **Kürzen mit dem Satz von Bayes:** Beim Umschreiben von $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{x}_0)$ nach dem Satz von Bayes kürzen sich die dazwischenliegenden Randdichten über $k = 2, \ldots, T$.

<!--en:2f71a1eb-->
- **Berechenbarer Gauß-Posterior:** Bedingt auf beide Endpunkte $(\mathbf{x}_0, \mathbf{x}_k)$ ist $q(\mathbf{x}_{k-1} \mid \mathbf{x}_k, \mathbf{x}_0) = \mathcal{N}(\tilde{\boldsymbol\mu}_k, \tilde\beta_k\mathbf{I})$ exakt gaußsch mit $\tilde\beta_k = \beta_k\sigma_{k-1}^2/\sigma_k^2 < \beta_k$.

<!--en:1a7300f4-->
::: {.slide title="Rechenbeispiel: exakte ELBO in zwei Schritten für N(0, 1)-Daten"}

<!--en:348c8689-->
[Jeden Term von Hand nachrechnen]{.kicker}

<!--en:83f3a18f-->
Gegeben sind $1\text{D}$-Daten aus der Standardnormalverteilung $x_0 \sim \mathcal{N}(0, 1)$, eine Kette mit $T = 2$ und $\beta_1 = \beta_2 = \frac{1}{2}$ ($\alpha_1^2 = \frac{1}{2}, \alpha_2^2 = \frac{1}{4}$), optimale Prädiktoren für den A-posteriori-Mittelwert und die Rückwärtsvarianz $\nu_k = \beta_k$:

<!--en:003b251a-->
- **Exakt in zwei Schritten:** Selbst bei großen Schritten $\beta_k = 0.5$ entspricht die Summe der exakten differentiellen Entropie von $\frac{1}{2}\log(2\pi e) \approx 1.419$ Nat für $\mathcal{N}(0, 1)$.

<!--en:1442141e-->
- **Warum $\nu_2 = \beta_2 = \frac{1}{2}$ hier besser ist als $\nu_2 = \tilde\beta_2 = \frac{1}{3}$:** Der Vorwärts-Posterior $q(x_1 \mid x_2, x_0)$ kennt $x_0$ (bedingte Varianz $\tilde\beta_2 = \frac{1}{3}$). Der Rückwärtsschritt $p_{\boldsymbol\theta}(x_1 \mid x_2)$ kennt $x_0$ *nicht*. Die Wahl $\nu_2 = \tilde\beta_2$ unterschätzt die wahre Rückwärtsvarianz, und die Schranke steigt auf $1.466$ Nat.

<!--en:58db2a50-->
::: {.slide title="Von der gaußschen KL-Divergenz zur Log-SNR-gewichteten Regression"}

<!--en:30ed93cd-->
[Herleitung des ELBO-Integrals in kontinuierlicher Zeit]{.kicker}

<!--en:16fe3620-->
- **Das ELBO-Gewicht entspricht dem halben Rückgang der Log-SNR:** Für $\nu_k = \tilde\beta_k$ vereinfacht sich das Gewicht der gaußschen KL-Divergenz jedes Schritts zu $\frac{1}{2}(e^{\lambda_{k-1} - \lambda_k} - 1) \approx \frac{1}{2}(\lambda_{k-1} - \lambda_k)$.

<!--en:aceb3a82-->
- **Grenzübergang zu kontinuierlicher Zeit ($T \to \infty$, Kingma et al., 2021):** Die Riemann-Summe über $\Delta\lambda_k$ konvergiert gegen ein vom Rauschplan unabhängiges Integral mit konstanter Gewichtung $w(\lambda) \equiv 1$.

<!--en:141494df-->
::: {.slide title="Diskrete ELBO-Gewichte entlang der VP-Kette"}

<!--en:40785f49-->
[Warum DDPM auf die Variationsgewichte verzichtet]{.kicker}

<!--en:f730876f-->
Der Vergleich der diskreten ELBO-Gewichte $\omega_k$ für $\nu_k = \beta_k$ und $\nu_k = \tilde\beta_k$ mit $\frac{1}{2}(\lambda_{k-1} - \lambda_k)$ auf der VP-Kette mit $T = 1{,}000$ zeigt ein extremes Übergewicht bei $k \to 0$:

<!--en:03fe670d-->
- **Sehr hohe Gewichte nahe den rauschfreien Daten ($k \le 10$):** Da $\sigma_k^2$ nahe null beginnt, fällt die Log-SNR um $\approx 0.78$ bei $k = 2$ und um $\approx 0.16$ bei $k = 10$, gegenüber etwa $0.01$–$0.02$ für $k \ge 100$.

<!--en:a030dcb4-->
- **Warum $L_{\textrm{simple}}$ alle Gewichte auf $1$ setzt:** Bei der eigentlichen ELBO entfällt rund ein Fünftel des Gewichts auf die ersten 20 von 1000 Schritten ($k \le 20$), etwa das Zehnfache ihres Anteils bei Gleichgewichtung. Diese Schritte entsprechen kaum wahrnehmbaren Pixelstörungen bei schwachem Rauschen. Gleich gewichtete Schritte konzentrieren die Kapazität auf mittlere Rauschpegel, bei denen die globale Bildstruktur entsteht.

<!--en:6a08f491-->
::: {.slide title="Exakte Fluss-Likelihood und kontinuierliche ELBO"}

<!--en:4dcc0c27-->
[Gelernte Dichte im Vergleich zum wahren Wert von 2.47 Nat]{.kicker}

<!--en:11f9d680-->
Entlang jeder Trajektorie der Wahrscheinlichkeitsfluss-ODE $d\mathbf{x}/dt = \mathbf{u}_t(\mathbf{x})$ ändert sich die Log-Dichte gemäß der Formel für die momentane Variablentransformation $d\log p_t(\mathbf{x}_t)/dt = -\nabla \cdot \mathbf{u}_t(\mathbf{x}_t)$:

<!--en:c6e4ee17-->
**Höchstens $0.005$ Nat vom Maximum-Likelihood-Wert entfernt:** Die Integration von `pf_nll` in 100 RK4-Schritten und die geschichtete Schätzung der kontinuierlichen ELBO `bound` bestätigen: Das trainierte MLP erreicht das exakte analytische Feld der Mischverteilung bis auf **$0.002$–$0.005$ Nat**.

<!--en:7bfce0e4-->
::: {.slide title="Begrenzte Kapazität: ELBO-Gewichtung und einfacher Verlust"}

<!--en:5809455f-->
[Wann spielt die Verlustgewichtung eine Rolle?]{.kicker}

<!--en:7c226828-->
Bei reichlich Kapazität (`width = 256`) erreicht jede positive Gewichtung das exakte Orakel. Bei begrenzter Kapazität (`width = 32`) entscheidet die Gewichtung, wo der Approximationsfehler anfällt:

<!--en:eeed3623-->
- **Bei `width = 32` gewinnt jede Gewichtung für ihre eigene Zielfunktion:** Gleichverteiltes $\lambda$ (`bound`) liefert eine um $0.01$–$0.02$ Nat niedrigere ELBO, da $>35\%$ der Trainingsziehungen bei $\lambda > 2$ liegen (fast rauschfreie Daten).

<!--en:f59480e4-->
- **Einfacher Verlust liefert bessere Fluss-Likelihood und Samples:** Bei gleichverteiltem $t$ (`simple`) liegen $90\%$ der Ziehungen im verrauschten Transportbereich ($\lambda < 2$). Das ergibt bessere exakte ODE-Likelihoods (`flow - exact`) und weniger Ausreißer ($1.6\%$ gegenüber $2.2\%$).

<!--en:5a8d2c84-->
::: {.slide title="Einheitliche Darstellung aller Zielfunktionen mit der Log-SNR-Gewichtung w(λ)"}

<!--en:1ce34103-->
[Die kanonische Integralform (Kingma und Gao, 2023)]{.kicker}

<!--en:8b74b111-->
- **Kontinuierliche ELBO ($w(\lambda) = 1$):** Auf allen Skalen konstant. Das Training optimiert die Likelihood-Schranke für rauschfreie Daten.

<!--en:153bde80-->
- **Flow-Matching und $\mathbf{v}$-Vorhersage auf dem Kosinuspfad ($w(\lambda) = e^{-\lambda/2}$):** Monoton fallende Gewichtung. Die Zielfunktion entspricht einer ELBO für durch Rauschen erweiterte Daten.

<!--en:e22abeaf-->
- **Einfacher DDPM-Verlust, Rauschvorhersage auf dem linearen Pfad und EDM:** Glockenförmige Gewichte betonen wahrnehmbare Strukturen bei mittlerer SNR und liefern keine Likelihood-Schranken.

<!--en:cbff5474-->
::: {.slide title="Übereinstimmung von Flow-Matching und v-Vorhersage"}

<!--en:ab304e64-->
[Überprüfung von w(λ) durch automatisches Differenzieren]{.kicker}

<!--en:7aeff404-->
Die numerische Berechnung von $w(\lambda) = 2\,c(t)\,|dt/d\lambda|$ durch automatisches Differenzieren im Vorwärtsmodus bestätigt, dass die Geschwindigkeitsregression auf dem linearen Pfad und die $\mathbf{v}$-Vorhersage auf dem Kosinuspfad genau dieselbe $e^{-\lambda/2}$-Gewichtung haben:

<!--en:53294845-->
::: {.slide title="Warum monotone Gewichtungen Likelihood-Schranken liefern"}

<!--en:207e7df2-->
[Interpretation durch verrauschte Daten und Ziehen der Zeitpunkte]{.kicker}

<!--en:676be692-->
- **Stufenfunktionen als Bausteine:** Die Stufengewichtung $w_\mu(\lambda) = \mathbf{1}\{\lambda \le \mu\}$ liefert die exakte kontinuierliche ELBO für Daten, die bereits bis zur Log-SNR $\mu$ verrauscht sind.

<!--en:848fb639-->
- **Monoton fallendes $w(\lambda)$ als Mischung von Schranken (Kingma und Gao, 2023):** Jedes $w'(\lambda) \le 0$ lässt sich als $\int w_\mu(\lambda)\,(-w'(\mu))\,d\mu$ schreiben, also als Mittelwert von ELBOs über durch Rauschen erweiterte Daten.

<!--en:5db3897b-->
- **Ziehen der Zeitpunkte aus $p(t)$ als zusätzlicher Faktor:** Das Ziehen gemäß $\operatorname{logit}(t) \sim \mathcal{N}(m, s^2)$ multipliziert $w(\lambda)$ mit einer Glockenkurve in $\lambda$, deren Zentrum bei $\lambda = -2m$ liegt (auf dem linearen Pfad $\lambda = -2\operatorname{logit} t$).

<!--en:363d7217-->
::: {.slide title="Gestaltungsspielraum bei EDM und Vorkonditionierung des Netzes"}

<!--en:f0925f8f-->
[Eingaben mit Einheitsvarianz, Skip-Verbindungen und Zielgrößen]{.kicker}

<!--en:33213342-->
- **Exakte Normierung auf Einheitsvarianz (Karras et al., 2022):** Für $n = \sqrt{\bar\sigma^2 + \sigma_{\mathrm{data}}^2}$ normieren die Einstellungen $c_{\mathrm{in}} = 1/n$, $c_{\mathrm{skip}} = \sigma_{\mathrm{data}}^2/n^2$ und $c_{\mathrm{out}} = \bar\sigma\sigma_{\mathrm{data}}/n$ sowohl Netzeingaben als auch Trainingsziele für alle $\bar\sigma \in [0.002, 80]$ auf Varianz eins.

<!--en:b8dce2d1-->
- **Adaptive Interpolation der Zielgröße:** Bei starkem Rauschen sagt $F_{\boldsymbol\theta}$ die rauschfreien Daten $\mathbf{x}_0/\sigma_{\mathrm{data}}$ vorher. Bei abnehmendem Rauschen geht die Zielgröße stetig in $-\boldsymbol\epsilon$ über.

<!--en:854aede5-->
::: {.slide title="Implementierung der EDM-Vorkonditionierung und des Verlusts bei log-normalverteilten Pegeln"}

<!--en:2b9c6183-->
[Koordinaten mit Einheitsvarianz für beliebige Pfade]{.kicker}

<!--en:457d9dc1-->
In den Pfadkoordinaten $\mathbf{x} = \alpha\mathbf{x}_0 + \sigma\boldsymbol\epsilon$ bleibt der Nenner $n = \sqrt{\sigma^2 + (\alpha\sigma_{\mathrm{data}})^2}$ für die Normierung selbst dann strikt positiv, wenn $\alpha_1 = 0$:

<!--en:4914589b-->
::: {.slide title="Implementierung der EDM-Vorkonditionierung und des Verlusts bei log-normalverteilten Pegeln (Fortsetzung)"}

<!--en:2b9c6183-2-->
[Koordinaten mit Einheitsvarianz für beliebige Pfade]{.kicker}

<!--en:c0bc0f44-->
`edm_loss` zieht log-normalverteilte Rauschpegel $\bar\sigma$ und gewichtet jeden quadratischen Fehler mit $1/c_{\mathrm{out}}^2$:

<!--en:10fe255f-->
::: {.slide title="Warum weißes Rauschen wie ein räumlicher Tiefpassfilter wirkt" layout="tight"}

<!--en:3d6f8902-->
[Fourier-Leistungsspektrum natürlicher Bilder]{.kicker}

<!--en:b01a6590-->
Bei einer orthonormalen 2D-Fourier-Transformation (`fft2`) hat isotropes gaußsches Rauschen $\sigma\boldsymbol\epsilon$ die gleiche erwartete Leistung $\sigma^2$ bei allen Raumfrequenzen $k$. Bei natürlichen Bildern und Bildern von Kleidungsstücken fällt die Leistung dagegen nach einem steilen Potenzgesetz ab:

<!--en:c71aef0c-->
- **Spektraler Abfall um drei Größenordnungen:** Von $k = 1$ bis $14$ Zyklen pro Bild sinkt die Signalleistung auf Fashion-MNIST um den Faktor $1{,}000\times$.

<!--en:fb35c2a7-->
- **SNR und Grenzfrequenz:**
  - Bei $\sigma = 1.0$ verdeckt das Rauschen alle Frequenzen $k \ge 4$ Zyklen pro Bild. Nur die grobe Silhouette bleibt.
  - Bei $\sigma = 0.3$ sind Frequenzen $k \ge 9$ Zyklen pro Bild verdeckt.
  - Bei $\sigma = 0.1$ überwiegt das Signal in allen $14$ Frequenzringen.

<!--en:3cfe02bc-->
::: {.slide title="Auflösung und verbleibendes Signal am Ende des Rauschplans"}

<!--en:ed51feab-->
[Verschiebung der Log-SNR um 2 log r bei höherer Auflösung]{.kicker}

<!--en:a53030b6-->
Benachbarte Pixel sind stark korreliert. Eine Erhöhung der Bildauflösung von $H \times W$ auf $(rH) \times (rW)$ verändert daher das effektive Signal-Rausch-Verhältnis grober räumlicher Strukturen grundlegend:

<!--en:d9fba5dd-->
- **SNR-Anstieg durch blockweise Mittelung ($r^2\times$):** Mitteln über einen Block aus $r \times r$ Pixeln erhält das grobe Signal und teilt die Varianz des unabhängigen Pixelrauschens durch $r^2$. Die Log-SNR der groben Struktur steigt um $+2\log r$.

<!--en:ef44c4c5-->
- **Rauschplan an die Auflösung anpassen (Hoogeboom et al., 2023):** Damit die grobe Struktur zum selben Zeitpunkt $t$ entsteht, den Rauschplan gemäß $\lambda_t^{(r)} = \lambda_t^{(1)} - 2\log r$ nach unten verschieben (z. B. um $-\log 16 \approx -2.77$ bei $4\times$ Seitenlänge).

<!--en:8f952d7a-->
- **Verbleibende Helligkeitsinformation (Lin et al., 2024):** Endet der Rauschplan bei $\lambda_1 = -5$ auf einem $64 \times 64$-Gitter, bleibt für den Kanalmittelwert $\mathrm{SNR} = 4096\,e^{-5} \approx 28$. Beginnt das Sampling mit Rauschen mit Mittelwert null, entstehen daher keine sehr dunklen oder sehr hellen Bilder.

<!--en:9f4ddae2-->
::: {.slide title="Kontrollierte Ablationsstudie mit je einem Faktor auf Fashion-MNIST"}

<!--en:f0fca970-->
[Zeitverteilungen, Zielgrößen und Gitter im Vergleich]{.kicker}

<!--en:cf2c37dc-->
Ausgehend vom Basismodell (Geschwindigkeit auf dem linearen Pfad, $t \sim U(0, 1)$, U-Net mit 64 Kanälen, $5{,}000$ Aktualisierungen) ändert sich je Zeile ein Faktor. Das Training läuft mit 2–4 Zufallsstartwerten. Die FD beruht auf je 5000 Heun-Samples auf dem EDM-Gitter ($\rho = 7$) und dem linearen Gitter:

<!--en:380353a1-->
::: {.slide title="Warum bei der Logit-Normalverteilung niedrige Rauschpegel zu kurz kommen"}

<!--en:eff9785e-->
[Warum zentrierte logit-normalverteilte Zeitpunkte bei 28 × 28 Pixeln schlechter abschneiden]{.kicker}

<!--en:63d18fa7-->
Warum verschlechterte die Zentrierung der logit-normalverteilten Trainingszeitpunkte bei $m = 0$ ($t = 0.5$) die FD um $+0.5$ bis $+1.3$, während eine Verschiebung des Zentrums auf $m = -0.5$ ($t \approx 0.38$) oder den EDM-Wert $m = -1.2$ ($t \approx 0.23$) wieder die beste Qualität lieferte?

<!--en:72f1b51a-->
- **Zu wenig Training für feine Details ($t \le 0.03$):** Bei einer Standard-Logit-Normalverteilung ($m = 0, s = 1$) liegen $< 1/3000$ der Trainingsziehungen unter $t = 0.03$ (gegenüber $3\%$ bei gleichverteiltem $t$). Der Testverlust der Geschwindigkeitsvorhersage bei $t = 0.01$ steigt dadurch von **$430$ auf $690$**.

<!--en:9bc31ffb-->
- **Abhängigkeit von der Auflösung:** Große latente Modelle für $512^2$ benötigen $m = 0$, weil eine hohe Auflösung die SNR der groben Struktur erhöht. Bei kleinen $28 \times 28$-Bildern bildet sich die Struktur bei schwächerem Rauschen ($m < 0$).

<!--en:2d7a029d-->
::: {.slide title="Drei Regeln für den Entwurf eines Diffusionsmodells"}

<!--en:3feb0d15-->
[Ergebnisse aus Theorie und Experiment]{.kicker}

<!--en:f0b78cbc-->
Für Diffusionsmodelle und Flow-Matching in einem neuen Anwendungsbereich oder bei einer neuen Auflösung gelten drei Gestaltungsregeln:

<!--en:4a5788e4-->
1. **Eine gut konditionierte Zielgröße auf einem Pfad bis zum reinen Rauschen ($\alpha_1 = 0$) wählen:** Geschwindigkeitsvorhersage auf dem linearen Pfad, $\mathbf{v}$-Vorhersage auf dem Kosinuspfad und EDM-Vorkonditionierung bleiben an beiden Enden des Intervalls $[0, 1]$ gut skaliert. Das vermeidet sowohl die Fehlerverstärkung durch Division bei $\alpha_1 \to 0$ als auch verbleibende Helligkeitsinformation ($\lambda_1 > -\infty$).

<!--en:eec46856-->
2. **Trainingspegel dort wählen, wo die räumliche Struktur entsteht:** Kleine $28 \times 28$-Bilder brauchen gleichverteiltes $t$ oder eine zu schwachem Rauschen verschobene Logit-Normalverteilung ($m < 0$). Wächst die Seitenlänge um den Faktor $r$, den Log-SNR-Plan um $2\log r$ nach unten verschieben.

<!--en:53295774-->
3. **Die Verlustgewichtung $w(\lambda)$ nach den Samples ausrichten, nicht nach der Schranke:** Konstantes $w(\lambda) \equiv 1$ (`bound`) dient dem Training der Likelihood-Schranke. Gute Samples liefern in der Praxis Gewichtungen, die hohe Log-SNR-Werte schwächer gewichten, etwa $w(\lambda) = e^{-\lambda/2}$ (Flow-Matching oder $\mathbf{v}$-Vorhersage, Kingma und Gao, 2023). Bei der Mischverteilung erreichte das kleine Netz mit der Gewichtung der Schranke nicht einmal seine beste Fluss-Likelihood.

<!--en:6cb6e02d-->
::: {.slide title="Verständnisfragen: Zerlegung der ELBO, Gewichtungen und Auflösung"}

<!--en:929b7671-->
[Verständnisprüfung]{.kicker}

<!--en:d4ef7bc2-->
Fragen zur Likelihood-Schranke und zu den Gestaltungsmöglichkeiten für Diffusionsmodelle:

<!--en:235bd789-->
1. **Warum $L_0$ im kontinuierlichen Raum negativ sein kann:** Für zweidimensionale Daten ($d = 2$) beim kleinsten Rauschverhältnis $e^{-\lambda_{\max}} = \sigma_{\min}^2/\alpha_{\min}^2 = 10^{-4}$ beträgt der unvermeidbare Fehler der Rauschvorhersage $\mathbb{E}\|\boldsymbol\epsilon - \hat{\boldsymbol\epsilon}\|^2 \approx 2$. Wie groß ist der Rekonstruktionsterm $L_0 = \frac{d}{2}\log(2\pi e^{-\lambda_{\max}}) + \frac{1}{2}\mathbb{E}\|\boldsymbol\epsilon - \hat{\boldsymbol\epsilon}\|^2$?

<!--en:dae5616d-->
2. **Gültige Likelihood-Schranken erkennen:** Auf dem linearen Pfad mit $t \sim U(0, 1)$ gilt für den Rauschverlust $w(\lambda) \propto \operatorname{sech}^2(\lambda/4)$ und für den Geschwindigkeitsverlust $w(\lambda) \propto e^{-\lambda/2}$. Welcher liefert eine gültige Likelihood-Schranke für durch Rauschen erweiterte Daten und warum?

<!--en:557b98b6-->
**Lösungen:**

<!--en:f58a381b-->
1. $L_0 = \log(2\pi \times 10^{-4}) + \frac{1}{2}(2) \approx -7.37 + 1.00 = \mathbf{-6.37\text{ nats}}$ (ein Term der differentiellen Log-Dichte). Zusammen mit dem gemessenen Integral über die Log-SNR ($\approx +8.75$ Nat, beim in der Zelle tatsächlich verwendeten Wert $e^{-\lambda_{\max}} \approx 1.1 \times 10^{-4}$ betragen die Endterme $\approx -6.3$ Nat) ergibt sich die Schranke $\approx \mathbf{+2.47\text{ nats}}$.

<!--en:cf754d6a-->
2. **Der Geschwindigkeitsverlust ($w(\lambda) \propto e^{-\lambda/2}$):** Da $e^{-\lambda/2}$ in $\lambda$ **monoton fällt**, ist die Gewichtung eine positive Mischung von ELBOs mit Stufenfunktionsgewichten für durch Rauschen erweiterte Daten. Dagegen hat $\operatorname{sech}^2(\lambda/4)$ bei $\lambda = 0$ ein Maximum.

<!--en:bddada72-->
::: {.slide title="Verständnisfragen: Zerlegung der ELBO, Gewichtungen und Auflösung (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständnisprüfung]{.kicker}

<!--en:052be68a-->
3. **Rauschpläne an andere Auflösungen anpassen:** Ein Log-SNR-Plan $\lambda_t$ ist auf $28 \times 28$-Bilder abgestimmt. Um wie viele Log-SNR-Einheiten muss sich $\lambda_t$ für $56 \times 56$-Bilder und für $112 \times 112$-Bilder verschieben?

<!--en:23db34b8-->
**Lösung:**

<!--en:d4c14d15-->
3. Den Plan um $\log(r^2) = 2\log r$ nach unten verschieben. Für $56 \times 56$ ($r = 2$) beträgt die Verschiebung **$-\log 4 \approx -1.39$**, für $112 \times 112$ ($r = 4$) beträgt sie **$-\log 16 \approx -2.77$**.

<!--en:47e475e6-->
::: {.slide title="Zusammenfassung: Likelihood, Gewichtung und Gestaltungsspielraum"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:62f50a9a-->
| Gestaltungsentscheidung | Mathematische Formulierung | Zentrales Ergebnis aus Theorie oder Experiment | Regel für die Praxis |

<!--en:b8b5e132-->
| **Diskrete und kontinuierliche ELBO** | $\frac{1}{2}\int_{\lambda_{\min}}^{\lambda_{\max}} \mathbb{E}\|\boldsymbol\epsilon - \hat{\boldsymbol\epsilon}_{\boldsymbol\theta}\|^2\,d\lambda + L_T + L_0$ | Gewicht pro Schritt $\frac{1}{2}(\lambda_{k-1} - \lambda_k)$, mit $w(\lambda) \equiv 1$ | Bei der zweidimensionalen Mischverteilung bis auf $0.005$ Nat gleich der exakten NLL |

<!--en:2d53b279-->
| **NLL der Wahrscheinlichkeitsfluss-ODE** | $\log p_1(\mathbf{x}_1) + \int_{t_{\min}}^1 \nabla \cdot \mathbf{u}_t(\mathbf{x}_t)\,dt$ | Exakte Log-Dichte über die Spur der Jacobi-Matrix ($\text{RK4}$) | Hutchinson-Spurschätzer $\mathbf{r}^\top(\nabla\mathbf{u}_t)\mathbf{r}$ bei großem $d$ verwenden |

<!--en:51878abe-->
| **Kanonische Gewichtung $w(\lambda)$** | $w(\lambda) = 2\,c(t)\,p(t)\,|dt/d\lambda|$ | Flow-Matching und $\mathbf{v}$-Vorhersage auf dem Kosinuspfad haben dieselbe Gewichtung $w(\lambda) \propto e^{-\lambda/2}$ | Monoton fallendes $w(\lambda)$ begünstigt eine niedrige FD für die wahrgenommene Sample-Qualität |

<!--en:77c44d3d-->
| **EDM-Vorkonditionierung** | $c_{\mathrm{skip}}\bar{\mathbf{x}} + c_{\mathrm{out}}F_{\boldsymbol\theta}(c_{\mathrm{in}}\bar{\mathbf{x}}; \bar\sigma)$ | Normiert Eingabe- und Zielvarianz auf $1$ für alle $\bar\sigma$ | Quadratisches Mittel $\sigma_{\mathrm{data}}$ des Datensatzes messen ($0.83$ auf Fashion-MNIST) |

<!--en:4771741c-->
| **Verschiebung des Rauschplans mit der Auflösung** | $\lambda_t^{(r)} = \lambda_t^{(1)} - 2\log r$ | $r \times r$-Pooling erhöht die SNR der groben Struktur um den Faktor $r^2$ | Bei höherer Auflösung Trainingspegel zu stärkerem Rauschen verschieben |
