<!--en:158dbceb-->
# Sampler, Inversion und Bildbearbeitung

<!--en:acaeef46-->
Jeder Schritt eines Diffusionssamplers wertet das Score- oder Geschwindigkeitsnetz einmal aus. Die Rechenkosten pro erzeugtem Bild wachsen daher linear mit der Schrittzahl. In :numref:`sec_diffusion-images` erreichten 50 Euler-Schritte für die Wahrscheinlichkeitsfluss-ODE (engl. probability-flow ODE) fast dieselbe Fréchet-Distanz (engl. Fréchet distance, FD) wie die ancestrale Kette (engl. ancestral chain) mit 1000 Schritten, bei einem Zwanzigstel der Kosten. Die Regression beim Training in :numref:`sec_diffusion-gaussian-paths` hängt nur von den Randverteilungen von $\mathbf{x}_t$ zu einzelnen Zeitpunkten $t$ ab. Gemeinsame Verteilungen über mehrere Schritte gehen nicht ein. Daher genügt ein einziges trainiertes Netz für eine kontinuierliche Familie stochastischer und deterministischer Rückwärtsprozesse :cite:`Song.Meng.Ermon.2020`.

<!--en:78243f30-->
In diesem Abschnitt leiten wir die Familie der *Denoising Diffusion Implicit Models* (DDIM) her. Wir zeigen, dass ihr deterministischer Vertreter ($\eta = 0$) in den Koordinaten $(\bar{\mathbf{x}}, \bar\sigma)$ mit dem Euler-Verfahren für die Wahrscheinlichkeitsfluss-ODE übereinstimmt. Für verschiedene Schrittzahlen und Zeitgitter vergleichen wir Verfahren erster Ordnung (Euler/DDIM), zweiter Ordnung (Heun) und Mehrschrittverfahren (engl. multistep methods, Adams–Bashforth / DPM-Solver++). Schließlich nutzen wir die Umkehrbarkeit der deterministischen ODE, um Bilder auf Rauschen zurückzuführen, im latenten Raum zu interpolieren, Bilder durch partielles Verrauschen zu bearbeiten und fehlende Bildbereiche per Zero-Shot-Inpainting zu ergänzen.

<!--en:ca9307f2-->
## Deterministische und stochastische Sampler

<!--en:1b424c95-->
Die beiden Sampler aus :numref:`sec_diffusion-integration`, die ancestrale Kette und das Euler-Verfahren, nutzen die Netzausgabe auf dieselbe Weise. Dabei fügt die Kette bei jedem Schritt frisches Rauschen hinzu, das Euler-Verfahren keines. An einem Punkt $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ eines Gauß-Pfads liefert das Netz eine Schätzung $\hat{\mathbf{x}}_0$ des rauschfreien Punkts und eine Schätzung $\hat{\boldsymbol\epsilon}$ des Rauschens. Jede der beiden bestimmt die andere:

<!--en:c68879b7-->
Unabhängig davon, welche Vorhersage das Netz beim Training gelernt hat, wandelt `d2l.make_pred` aus :numref:`sec_diffusion-integration` das Netz in eine Funktion `pred(x,
alpha, sigma)` um. Sie gibt dieses Paar bei jedem Rauschpegel zurück. Mehr benötigen die Sampler in diesem Abschnitt nicht.

<!--en:215a7914-->
### Die DDIM-Familie und das Euler-Verfahren

<!--en:e9438683-->
Betrachten wir einen Zeitpunkt $t$ und einen früheren Zeitpunkt $s < t$. Ein Sampler-Schritt von $t$ nach $s$ muss einen wie $\mathbf{x}_t$ verteilten Punkt in einen wie $\mathbf{x}_s$ verteilten Punkt überführen. Weitere Bedingungen stellt die Regression nicht. Wir schreiben $\mathbf{x}_t
= \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$, wählen eine Rauschskala $\tilde\sigma$ zwischen $0$ und $\sigma_s$ und setzen

<!--en:7072320c-->
mit $\boldsymbol\xi$ unabhängig von $\boldsymbol\epsilon$. Die beiden Rauschvarianzen addieren sich zu $\sigma_s^2$. Bei gegebenem $\mathbf{x}_0$ hat der Punkt $\mathbf{x}_s$ daher für jedes $\tilde\sigma$ die richtige Verteilung. Wegen $\boldsymbol\epsilon = (\mathbf{x}_t - \alpha_t \mathbf{x}_0) /
\sigma_t$ legt jedes $\tilde\sigma$ einen Schritt von $\mathbf{x}_t$ nach $\mathbf{x}_s$ fest. Ein Sampler setzt die Schätzungen des Netzes für $\mathbf{x}_0$ und $\boldsymbol\epsilon$ ein:

<!--en:2b3153c0-->
Der Schritt bringt den vorhergesagten rauschfreien Punkt zusammen mit dem vorhergesagten Rauschen auf den Pegel $s$ und ersetzt einen Teil dieses Rauschens durch frisches Rauschen. :citet:`Song.Meng.Ermon.2020` beschreiben den frischen Anteil durch die Stärke $\eta \geq 0$,

<!--en:ffb608c3-->
wobei $\kappa^2 = \mathrm{SNR}_t / \mathrm{SNR}_s < 1$ mit $\mathrm{SNR} =
\alpha^2 / \sigma^2$ gilt. Für $\eta = 1$ ist $\tilde\sigma^2$ die Varianz von $\mathbf{x}_s$ bei gegebenem $\mathbf{x}_t$ und $\mathbf{x}_0$ in der Vorwärtskette aus :numref:`sec_diffusion-gaussian-paths` (Aufgabe :exref:`ex_samplers-forward-step`). Der Schritt zieht dann ein Sample aus der A-posteriori-Verteilung dieser Kette und setzt dabei die Schätzungen für die wahren Werte ein. Bei $\eta = 0$ kommt kein frisches Rauschen hinzu, der Schritt ist also deterministisch. Dies ist das *Denoising Diffusion Implicit Model* (DDIM). Es heißt implizit, weil es seine Samples durch eine Abbildung ausgehend vom Rauschen definiert und keine Kette von Dichten benötigt.

<!--en:0b73c700-->
`ddim_sample` führt die Sampler der DDIM-Familie auf dem Gitter von `path.grid` mit einem beliebigen Prädiktor `pred(x, alpha, sigma) -> (x0, eps)` aus. Die Funktion hält bei `t_end` an (standardmäßig bei den Daten), sodass eine Schleife das Sample zwischen den Pegeln bearbeiten kann.

<!--en:461df87b-->
Wie groß ist der Fehler bei einem langen deterministischen Schritt? Betrachten wir auf dem varianzerhaltenden Pfad Daten, die selbst standardnormalverteilt sind. Dann ist $\mathbf{x}_t$ zu jedem Zeitpunkt standardnormalverteilt, der exakte Sampler sollte also jeden Punkt an Ort und Stelle lassen. Die exakten Schätzungen sind $\hat{\mathbf{x}}_0 = \alpha_t
\mathbf{x}_t$ und $\hat{\boldsymbol\epsilon} = \sigma_t \mathbf{x}_t$. Ein einziger deterministischer Schritt von $t = 1$ nach $t = 0$ liefert $\hat{\mathbf{x}}_0
= \alpha_1 \mathbf{x}_1$. Bei $\alpha_1 \approx 0.0066$ fällt die ganze Punktwolke praktisch auf ihren Mittelwert zusammen. Der Schritt hält die Rauschschätzung über die gesamte Schrittweite fest, während sie sich entlang der exakten Trajektorie ändert.

<!--en:27c57093-->
Für $\eta = 0$ lautet der Schritt $\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 +
\sigma_s \hat{\boldsymbol\epsilon}$. Genau diesen Schritt macht `ode_sample` in :numref:`sec_diffusion-integration`. In den dort hergeleiteten Koordinaten $\bar{\mathbf{x}} =
\mathbf{x} / \alpha_t$ und $\bar\sigma_t = \sigma_t / \alpha_t$ lautet er

<!--en:bc0b235f-->
und ist damit ein Euler-Schritt für die Wahrscheinlichkeitsfluss-ODE $d\bar{\mathbf{x}} / d\bar\sigma =
\hat{\boldsymbol\epsilon}$. Das Verhältnis $\kappa$ von oben ist gleich $\bar\sigma_s / \bar\sigma_t$. Auf jedem Gauß-Pfad entspricht DDIM also dem Euler-Verfahren. Bei $\eta = 0$ berechnen `ddim_sample` und `ode_sample` auf denselben Pegeln dieselbe Funktion in zwei Koordinatensystemen. :numref:`sec_mdl-ddim-guidance` beweist die Äquivalenz und behandelt die beiden Extremfälle. Für einen einzelnen Datenpunkt sind Schritte beliebiger Länge exakt, weil $\hat{\boldsymbol\epsilon}$ dann entlang jeder Trajektorie konstant ist. Bei gaußverteilten Daten schrumpft die Punktwolke mit jedem Schritt (:numref:`fig_mdl-dyn-ddim-strides`).

<!--en:548b0480-->
### Frisches Rauschen und lange Schritte

<!--en:840ca763-->
Wenn ein Netz für jedes $\eta$ ausreicht, welche Stärke sollten wir dann wählen? Mit $f = \tilde\sigma^2 / \sigma_s^2$ lautet der Schritt :eqref:`eq_diffusion-ddim-step`

<!--en:7cb0f626-->
Jeder Vertreter der Familie bringt die Schätzung des rauschfreien Punkts auf dieselbe Weise auf den Pegel $s$. Der Anteil $f$ der zugehörigen Rauschvarianz stammt aus frischem Rauschen. Auf einem feinen Gitter ist $f$ selbst bei $\eta = 1$ klein. Auf einem Gitter mit zehn Schritten ersetzt der Sampler bei $\eta = 1$ in jedem Schritt den Großteil des Rauschens.

<!--en:b0ca4de4-->
Für frisches Rauschen spricht ein gutes Argument. In erster Ordnung zerfällt der ancestrale Schritt aus :numref:`sec_diffusion-integration` in zwei Schritte. Ein Langevin-Schritt erhält die aktuelle Verteilung $p_t$ und zieht abgedriftete Punkte zu ihr zurück. Hinzu kommt ein Euler-Schritt für die Wahrscheinlichkeitsfluss-ODE. Jeder Rückwärtsprozess, der Rauschen hinzufügt und die Randverteilungen erhält, lässt sich so zerlegen (:numref:`sec_mdl-time-reversal`). Ein stochastischer Sampler zieht seinen Zustand also immer wieder zur Modellverteilung hin. Dadurch kann ein Fehler aus einem Schritt beim nächsten teilweise verschwinden. Ein deterministischer Sampler trägt jeden Fehler bis zum Ende mit.

<!--en:fa20a712-->
Ein ebenso gutes Argument spricht dagegen. Die rücktreibende Wirkung des Langevin-Schritts gilt für kurze Schritte. Bei einem langen Schritt weicht der exakte Rückwärtsübergang stark von einer Gauß-Verteilung ab (Aufgabe :exref:`ex_gaussian-paths-gaussian-reversal` in :numref:`sec_diffusion-integration`). Ein langer stochastischer Schritt verursacht also einen zusätzlichen Fehler. Bei kleinem Rechenbudget sollte der deterministische Sampler besser abschneiden. Ob ihn der stochastische bei kurzen Schritten überholt, untersuchen wir in :numref:`subsec_diffusion-samplers-quality` anhand von Bildern. :numref:`fig_diffusion-two-samplers` zeigt die beiden Extremfälle der Familie an der Mischverteilung.

<!--en:3536d68d-->
![Die beiden Extremfälle der DDIM-Familie bei der Mischung mit drei Modi und exaktem Prädiktor. Von drei Startpunkten bei $t = 1$ (Kreise) führen jeweils 200 Schritte auf dem varianzerhaltenden Pfad bis $t = 0$ (Kreuze). Die Trainingsdaten sind grau dargestellt. (a) Bei $\eta = 0$ führt von jedem Startpunkt eine glatte Kurve zu einem Modus. (b) Bei $\eta = 1$ verlaufen die Pfade im Zickzack. Zwei der drei enden hier in einem anderen Modus als in (a). Die erste Hälfte jedes Pfads, in der das Rauschen überwiegt, ist blasser gezeichnet. Beide Sampler erzeugen bei $t = 0$ dieselbe Verteilung.](../img/mdl-diffusion-two-samplers.svg)

<!--en:bff31552-->
Zwei stochastische Sampler müssen wir auseinanderhalten. Die ancestrale Kette fügt Rauschen mit der Varianz des Vorwärtsschritts $\sigma_t^2 - (\alpha_t / \alpha_s)^2
\sigma_s^2$ hinzu. Bei $\eta = 1$ verwendet DDIM dagegen die um den Faktor $\sigma_s^2 / \sigma_t^2$ kleinere A-posteriori-Varianz. Beide steuern denselben Mittelwert an. Die ancestrale Kette gehört somit nicht zur Familie, denn die Varianzen des beibehaltenen und des hinzugefügten Rauschens ergeben zusammen mehr als $\sigma_s^2$. Welche Varianz ist richtig? Im Allgemeinen keine von beiden. Die A-posteriori-Varianz ist exakt, wenn die Daten aus einem einzigen Punkt bestehen, denn dann bestimmt $\mathbf{x}_t$ den Wert von $\mathbf{x}_0$ eindeutig. Die Varianz des Vorwärtsschritts ist für standardnormalverteilte Daten auf dem varianzerhaltenden Pfad exakt. Streuen die Daten weniger als das Rauschen, mit einer Varianz unter eins je Koordinate, so liegt die beste gaußsche Varianz zwischen den beiden Werten. Bei breiter gestreuten Daten kann sie sogar die Varianz des Vorwärtsschritts übertreffen (Aufgabe :exref:`ex_samplers-optimal-variance`). Auf einem Gitter mit tausend Schritten liegt der Faktor fast überall nahe eins. Nur in den letzten Schritten weicht er davon ab, dort sind beide Varianzen allerdings winzig. Die Wahl fällt daher kaum ins Gewicht, bei fünfzig Schritten dagegen schon.

<!--en:4539a4a9-->
## Weniger Schritte

<!--en:4970bfe2-->
Bei einem Verfahren erster Ordnung ist der Fehler proportional zur Schrittweite. Wenn wir Schritte überspringen, sinkt die Genauigkeit also auf vorhersehbare Weise. Wir messen diesen Verlust zunächst dort, wo wir ihn exakt bestimmen können. Dann untersuchen wir, wie viel Genauigkeit sich mit besseren Verfahren und günstiger gewählten Gitterpunkten zurückgewinnen lässt. Die Messwerte für Bilder gelten für ein einzelnes kleines Modell. Auf andere Modelle lässt sich das Messverfahren übertragen.

<!--en:e0621817-->
### Überspringen von Schritten im bisherigen Beispiel

<!--en:724835e1-->
Wir kehren zur Mischung `mix` mit drei Modi aus :numref:`sec_diffusion-denoising` zurück. Für sie können wir auf jedem Pegel den A-posteriori-Mittelwert und damit das exakte Paar $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ berechnen. `pred_exact` stellt das Paar über die Schnittstelle der Sampler bereit. Wir ziehen 20 000 Startpunkte und ebenso viele Datenpunkte. Dann geben wir die *Streuung* (engl. spread) der Daten aus, den mittleren quadrierten Abstand eines Punkts zum nächstgelegenen Moduszentrum (:numref:`sec_diffusion-denoising`). Jeder Modus hat je Koordinate die Varianz $0.25$, die Streuung liegt daher knapp unter $0.5$.

<!--en:5d8e08cc-->
Beim exakten Prädiktor geht jede Abweichung zwischen den Samples nach zehn und nach tausend Schritten vom selben Startpunkt auf die endliche Schrittweite zurück. Bei einem Verfahren erster Ordnung sollte sich der Fehler mit jeder Verdopplung der Schrittzahl halbieren. Wir führen DDIM auf dem varianzerhaltenden Gitter mit 5 bis 1000 Schritten aus und geben den mittleren Abstand zu den Samples nach tausend Schritten sowie die Streuung aus. Neben der Streuung für $\eta = 0$ stehen die Werte für $\eta = 1$ und für die ancestrale Kette mit denselben Startpunkten.

<!--en:11c8d4a0-->
Wie vorhergesagt halbiert sich der Abstand mit jeder Verdopplung der Schrittzahl. Er beträgt etwa 0.4 bei fünf Schritten, 0.2 bei zehn und 0.04 bei fünfzig. Die Modi liegen auf dieser Skala etwa fünf Einheiten auseinander. Der Fehler hat auch eine Richtung. Bei zehn Schritten streuen die deterministischen Samples mit etwa 0.38 gegenüber 0.47 bis 0.48 bei den Daten. Im linken Teilbild sind die Punktwolken sichtbar enger als die dahinterliegenden Daten. Ein Schritt hält die Rauschschätzung auf ihrem Anfangswert fest. Bei gaußverteilten Daten entfernt er dadurch stets zu viel Rauschen (Aufgabe :exref:`ex_samplers-gaussian-data`). Sobald das Rauschen so gering ist, dass sich die Modi unterscheiden lassen, verhält sich jeder Modus der Mischung wie gaußverteilte Daten mit Varianz $0.25$ je Koordinate.

<!--en:01a4c406-->
Die stochastischen Sampler weichen in beide Richtungen ab. Bei zehn Schritten zieht $\eta = 1$ die Modi noch enger zusammen, auf eine Streuung von etwa 0.32. Die ancestrale Kette weitet sie auf etwa 0.52 und damit über die Streuung der Daten hinaus. Zunächst überrascht es, dass $\eta = 1$ trotz des zusätzlichen Rauschens eine engere Verteilung ergibt als $\eta = 0$ ganz ohne zusätzliches Rauschen. Jeder Vertreter der Familie teilt dieselbe Varianz $\sigma_s^2$ zwischen beibehaltenem und frischem Rauschen auf. Bei gaußverteilten Daten sind sowohl die Schätzung des rauschfreien Punkts als auch die Rauschschätzung positive Vielfache von $\mathbf{x}_t$. Das beibehaltene Rauschen zeigt daher in dieselbe Richtung wie der geschätzte rauschfreie Punkt und erhöht dessen Streuung über einen Kreuzterm. Frisches Rauschen ist von beiden unabhängig und trägt nur seine Varianz bei. Verlagert man einen Teil der Varianz auf frisches Rauschen, wie bei $\eta = 1$ auf einem groben Gitter, so streuen die Samples daher weniger (Beweis in Aufgabe :exref:`ex_samplers-gaussian-data`). Die ancestrale Kette behält ebenso viel Rauschen wie $\eta = 1$ bei, fügt aber über das Budget hinaus frisches Rauschen hinzu. Der Überschuss passt zu Daten, die ebenso stark streuen wie das Rauschen. Für einen engeren Modus ist er zu groß. Auf feinen Gittern erreichen alle drei Sampler die Streuung der Daten. Eine Kennzahl, die in beide Richtungen abweichen kann, liefert ihrer Größe nach keine Rangfolge der Sampler. Für Bilder verwenden wir deshalb ein Maß, das Abweichungen in beiden Richtungen bestraft.

<!--en:0d6b50b6-->
### Verfahren zweiter Ordnung und Wahl der Gitterpunkte

<!--en:cf3519d3-->
Das Euler-Verfahren hält die Steigung über einen ganzen Schritt fest. Sein Fehler über den gesamten Lauf nimmt deshalb nur proportional zur Schrittweite ab. Das Heun-Verfahren (:numref:`sec_mdl-euler-runge-kutta`) verwendet den Euler-Schritt als Probeschritt, wertet die Steigung am Probepunkt erneut aus und verwendet für den eigentlichen Schritt den Mittelwert beider Steigungen:

<!--en:aba03f99-->
Sein Fehler nimmt quadratisch mit der Schrittweite ab, bei zwei Auswertungen pro Schritt. Für standardnormalverteilte Daten lautet die Rauschschätzung $\hat{\boldsymbol\epsilon} = \bar\sigma \bar{\mathbf{x}} / (1 +
\bar\sigma^2)$, und der exakte Fluss multipliziert $\bar{\mathbf{x}}$ mit $\sqrt{(1 + \bar\sigma_s^2) / (1 + \bar\sigma_t^2)}$. Von $\bar\sigma_t =
1$ nach $\bar\sigma_s = 0.5$ beträgt der exakte Faktor $0.791$. Der Euler-Schritt ergibt $0.75$, der Heun-Schritt $0.80$. Die Fehler betragen $0.041$ und $0.009$, bei doppelt so vielen Auswertungen also rund ein Viertel des Fehlers. In zwei Fällen entfällt die zweite Steigung. Bei $\bar\sigma = 0$ ist kein Rauschen mehr zu schätzen. Ein Schritt von $\bar\sigma = \infty$ aus ist unendlich lang und lässt sich nur in der Form $\hat{\mathbf{x}}_0 + \bar\sigma_s \hat{\boldsymbol\epsilon}$ schreiben. Der letzte Schritt nach $\bar\sigma = 0$ und ein erster Schritt von $\bar\sigma = \infty$ aus sind daher Euler-Schritte. Somit benötigt `heun_sample` insgesamt $2n - 1$ Auswertungen für $n$ Schritte oder $2n - 2$ bei einem Start aus reinem Rauschen auf dem linearen Pfad.

<!--en:c31e1113-->
Ob sich die zweite Auswertung lohnt, hängt von der Lage der Gitterpunkte ab. Bei einer eindimensionalen Mischung mit exaktem Feld ist das Heun-Verfahren auf einem gleichmäßigen Gitter unterhalb von etwa vierzig Auswertungen dem Euler-Verfahren unterlegen. Auf einem Gitter, dessen Punkte zum Datenende hin dichter liegen, gewinnt es bei jedem Budget. Dieselbe Verdichtung erhöht den Fehler des Euler-Verfahrens auf mehr als das Doppelte (:numref:`sec_mdl-sampling-learned-dynamics`). :citet:`Karras.Aittala.Aila.ea.2022` trennten zwei Entscheidungen, die bei früheren Samplern gekoppelt waren, nämlich die Ordnung des Verfahrens und die Lage der Gitterpunkte. Sie verdichten das Gitter, indem sie die Punkte in $\bar\sigma^{1/\rho}$ gleichmäßig verteilen,

<!--en:7a98d71a-->
gefolgt von einem letzten Schritt nach $\bar\sigma = 0$, mit $\bar\sigma_{\max} = 80$, $\bar\sigma_{\min} = 0.002$ und $\rho = 7$ für Bilder mit Werten in $[-1, 1]$. Der Sampler kann das Gitter unabhängig vom Trainingspfad wählen, da `make_pred` jeden Pegel auf die Zeitskala des Netzes umrechnet. `edm_grid` liefert diese Pegel, und `heun_sample` führt Heun-Schritte entlang einer beliebigen Pegelfolge aus. Wir geben drei Gitter mit je zehn Schritten aus. Die ersten beiden teilen die Zeit auf dem varianzerhaltenden beziehungsweise dem linearen Pfad gleichmäßig ein. Das dritte ist das Gitter von Karras et al., das wir nach ihrer Arbeit *Elucidating the Design Space of Diffusion-Based Generative Models* als EDM-Gitter bezeichnen.

<!--en:7c7180ce-->
Gitter mit gleichmäßigen Zeitabständen sehen in $\bar\sigma$ völlig unterschiedlich aus. Beim varianzerhaltenden Gitter liegen vier der zehn Schritte zwischen $152$ und $6.1$. Der letzte Schritt führt von $0.34$ zu den Daten. Das lineare Gitter überbrückt den gesamten Bereich oberhalb von $9$ mit einem einzigen Schritt aus dem Unendlichen. Sieben Schritte liegen zwischen $4$ und $0.11$. Das EDM-Gitter beginnt bei $80$, verwendet ebenfalls vier Schritte oberhalb von $4$ und drängt seine letzten drei Pegel unterhalb von $0.12$ zusammen. Welche Aufteilung zu einem Modell passt, lässt sich an den Pegeln allein nicht ablesen. Wir messen es weiter unten an Bildern.

<!--en:298b0b39-->
### Exponentielle Integratoren und Mehrschrittverfahren

<!--en:536b5947-->
Für seine zweite Ordnung benötigt das Heun-Verfahren eine zweite Auswertung pro Schritt. Einen günstigeren Weg eröffnen die bisher verwendeten Koordinaten. Tatsächlich haben wir diesen Weg schon beschritten. In $\mathbf{x}$ hat die Wahrscheinlichkeitsfluss-ODE einen Anteil, der linear in $\mathbf{x}$ und vom Netz unabhängig ist. Auf dem varianzerhaltenden Pfad etwa lässt die Drift $\mathbf{x}$ unabhängig vom Score mit der Rate $\tfrac12\beta(t)$ schrumpfen. Dabei ist $\beta(t)$ die Rate des Rauschplans. Die exakte Lösung dieses linearen Teils ist ein exponentieller Abfall, hier der Faktor $\alpha_t$. Durch Division durch $\alpha_t$ fällt er weg. Übrig bleibt $d\bar{\mathbf{x}} /
d\bar\sigma = \hat{\boldsymbol\epsilon}$, worin nur noch das Netz eingeht. Einen *exponentiellen Integrator* kennzeichnet genau diese Idee: den linearen Teil exakt zu integrieren und nur den Netzterm anzunähern :cite:`Zhang.Chen.2023,Lu.Zhou.Bao.ea.2022`. DDIM hält $\hat{\boldsymbol\epsilon}$ über den Schritt konstant und ist damit der Vertreter erster Ordnung. Der Koordinatenwechsel, durch den DDIM als Euler-Verfahren erkennbar wurde, war also bereits die Grundidee dieser Löser. Die Vertreter höherer Ordnung unterscheiden sich nur darin, wie sie die Änderung von $\hat{\boldsymbol\epsilon}$ im Laufe des Schritts berücksichtigen. DPM-Solver :cite:`Lu.Zhou.Bao.ea.2022` wertet die Größe ein zweites Mal aus, und zwar in der Schrittmitte bezüglich $\log \bar\sigma$ (Aufgabe :exref:`ex_samplers-dpm-solver`).

<!--en:779a3e66-->
Die zweite Auswertung lässt sich einsparen. Der vorige Schritt hat $\hat{\boldsymbol\epsilon}$ bereits an dem Pegel ausgewertet, den wir gerade verlassen haben. Ein *Mehrschrittverfahren* nutzt diesen Wert erneut. Aus den letzten beiden Schätzungen extrapoliert es $\hat{\boldsymbol\epsilon}$ linear zur Mitte des aktuellen Schritts. Bezeichnen $\Delta = \bar\sigma_s - \bar\sigma_t$ den aktuellen Schritt, $\Delta'$ den vorigen und $\hat{\boldsymbol\epsilon}_t$ sowie $\hat{\boldsymbol\epsilon}'$ die jeweiligen Schätzungen am Anfang, dann lautet der Adams–Bashforth-Schritt zweiter Ordnung

<!--en:cbc435ea-->
Ändert sich die Rauschschätzung von einem Schritt zum nächsten nicht, ergibt sich DDIM. Andernfalls folgt der Korrekturterm dieser Änderung, bei einer Auswertung pro Schritt. Für den ersten Schritt fehlt eine frühere Schätzung. Er ist daher ein Euler-Schritt, ebenso wie der Schritt nach einem Start bei $\bar\sigma = \infty$, für den $\Delta'$ unendlich ist. DPM-Solver++(2M) :cite:`Lu.Zhou.Bao.ea.2025` wendet dieselbe Idee auf die Schätzung des rauschfreien Bilds an und extrapoliert in $\log \bar\sigma$. Dieses Verfahren und seine Varianten sind bei zehn bis fünfundzwanzig Auswertungen die übliche Wahl. `multistep_sample` führt den oben angegebenen Schritt für eine beliebige Pegelfolge aus.

<!--en:e2465226-->
### Bildqualität in Abhängigkeit von der Zahl der Auswertungen

<!--en:f5619541-->
Bei Bildern lernen wir das Feld. Wir trainieren das klassenbedingte U-Net aus :numref:`sec_diffusion-images` (etwa 1.2 Millionen Parameter) mit zehntausend Updates auf die Geschwindigkeit des linearen Pfads. Auf einer GPU dauert das wenige Minuten. Wir ersetzen ein Zehntel der Label durch ein Null-Label und verwenden für das Sampling den gleitenden Mittelwert der Gewichte. Dieses eine Netz genügt für den Rest des Abschnitts. Sampler ohne Guidance (unten $\gamma = 1$) fordern jede Klasse gleich oft an (Sample $i$ für Klasse $i$ modulo 10). Sampler mit Guidance ($\gamma > 1$) werten zusätzlich das Null-Label aus. Bei Inversion und Bildbearbeitung erhält jedes Bild sein eigenes Label.

<!--en:604989db-->
Wir bewerten 1000 Samples anhand ihrer Fréchet-Distanz (FD) zu 1000 Testbildern im Merkmalsraum von `d2l.FeatureCNN` (:numref:`sec_diffusion-images`). Bei dieser Anzahl ergibt schon der Vergleich zweier Mengen echter Bilder einen Wert knapp unter 1. Von einer Bildauswahl zur nächsten schwankt der Wert um 0.15 bis 0.2 (:numref:`subsec_diffusion-fd-reference`). Deshalb wiederholen wir jede Messung mit drei Sätzen von je 1000 Rauschbildern. Alle Sampler verwenden dieselben Sätze, sodass jeder Vergleich gepaart ist. Wir geben den Mittelwert plus/minus die halbe Spannweite an. Als Untergrenze dient der Vergleich der Testbilder mit drei weiteren Mengen echter Bilder. Unterschiede innerhalb der halben Spannweiten sind nicht aussagekräftig. `fd_runs` akzeptiert außerdem eine Guidance-Stärke $\gamma$, die wir im nächsten Unterabschnitt verwenden. $\gamma = 1$ entspricht dem unveränderten bedingten Feld.

<!--en:2e1878d0-->
Die Untergrenze liegt bei etwa 0.9. Die erste Tabelle beantwortet die oben offengebliebene Frage. Wir verwenden den varianzerhaltenden Pfad mit gleichmäßigen Zeitschritten, sodass DDIM mit $\eta = 0$ dem Euler-Verfahren auf dem ersten oben ausgegebenen Gitter entspricht. Wenn Rauschen Fehler korrigiert, die ein deterministischer Sampler bis zum Ende mitträgt, sollten $\eta = 1$ und die ancestrale Kette bei kürzeren Schritten an $\eta = 0$ herankommen und es noch vor tausend Schritten überholen.

<!--en:fe584eee-->
Sie holen auf, überholen es aber nie. Bis einschließlich zweihundert Schritte liegt die FD für $\eta = 0$ bei jedem Budget um ein Drittel bis fast die Hälfte unter der für $\eta = 1$. Das liegt weit außerhalb der halben Spannweiten. Bei tausend Schritten liegen die drei Werte höchstens etwa 0.2 auseinander, zwischen rund 1.3 und 1.5. Bei diesem Modell half frisches Rauschen nie. Bei keinem Budget übertraf ein stochastischer Sampler $\eta = 0$ um mehr als die halben Spannweiten. Bei tausend Schritten zieht die ancestrale Kette lediglich gleich. Bis hundert Schritte schneidet sie am schlechtesten ab. Mit etwa 8 bis 11 bei fünfzig Schritten liegt sie sogar hinter DDIM mit zehn Schritten (etwas über 5). Ihre Varianz des Vorwärtsschritts passt zu Daten, die so stark streuen wie das Rauschen. Bei geringem Rauschen steht das Kleidungsstück aber schon nahezu fest. Die überschüssige Varianz sollte daher als Körnung sichtbar werden. Die Bildreihen am Ende des nächsten Unterabschnitts zeigen das. Bei DDIM selbst ändert sich der Wert nach hundert Schritten noch um etwa ein Zehntel. Schon bei fünfzig Schritten liegt er nur rund ein Drittel vom Wert der Kette mit tausend Schritten entfernt, bei einem Zwanzigstel der Auswertungen. Das ist das zentrale Ergebnis von :citet:`Song.Meng.Ermon.2020`.

<!--en:2c9aa0fb-->
Das Urteil über frisches Rauschen gilt für dieses Modell und lässt sich nicht auf große Modelle übertragen. Auf ImageNet mit $64 \times 64$ Pixeln zeigten :citet:`Karras.Aittala.Aila.ea.2022`, dass eine kleine, abgestimmte Menge Rauschen vor jedem Heun-Schritt bessere Ergebnisse lieferte als ihr deterministischer Sampler. Die Prädiktor-Korrektor-Sampler von :citet:`song2021score` fügen mit derselben Wirkung Langevin-Schritte zwischen den Schritten des Lösers ein. Restart-Sampling :cite:`Xu.Deng.Cheng.ea.2023` profitiert am meisten bei großen Budgets. Dahinter steht die oben beschriebene rücktreibende Wirkung des Langevin-Schritts. Ein großes Modell für schwierige Daten macht Fehler, die späteres Rauschen noch korrigieren kann. Bei genügend vielen Schritten wiegt die Korrektur schwerer als die zusätzliche Varianz.

<!--en:687bfc98-->
Die zweite Tabelle zählt Netzauswertungen, denn diese bestimmen die Kosten pro Sample. Wir testen das Euler-, das Heun- und das Mehrschrittverfahren bei etwa 10 bis 100 Auswertungen auf drei Gittern. Wir verwenden das lineare Gitter, das EDM-Gitter von Karras et al. mit $\rho = 7$ und dasselbe Gitter mit $\rho = 3$. Letzteres verschiebt Pegel vom Datenende hin zu stärkerem Rauschen. Das Heun-Verfahren macht etwa halb so viele Schritte. Auf den EDM-Gittern, die bei einem endlichen Pegel beginnen, benötigt es eine Auswertung weniger als vorgesehen.

<!--en:91c99e59-->
Wie hängen die beiden Tabellen zusammen? Die Grafik zeigt die Sampler der ersten Tabelle und die drei Löser auf dem linearen Gitter über einer gemeinsamen Achse für die Zahl der Netzauswertungen.

<!--en:a1410a17-->
Die Verfahren zweiter Ordnung erreichen das asymptotische FD-Plateau des Modells ($\approx 1.1$ bis $1.3$, etwa $0.3$ über der Untergrenze für endlich viele Samples) zuerst. Das Heun-Verfahren auf dem linearen Gitter erreicht $\approx 1.2$ bis $1.3$ bei 20 Funktionsauswertungen. Das Mehrschrittverfahren zweiter Ordnung erreicht das Plateau nach spätestens 50 Auswertungen, das Euler-Verfahren benötigt 50 bis 100.

<!--en:29e9efbe-->
Die Lage der Gitterpunkte ist so wichtig wie die Ordnung des Lösers. Bei 10 Auswertungen erreicht das Euler-Verfahren $\approx 3.2$ auf dem linearen Gitter, $\approx 5$ auf dem VP-Gitter und $\approx 6.5$ auf dem EDM-Gitter ($\rho = 7$). Das Heun-Verfahren erreicht $\approx 1.6$ auf dem linearen Gitter gegenüber $12$ bis $15$ auf dem EDM-Gitter. Senken wir $\rho$ auf $3$, wandern Schritte vom Datenende zu stärkerem Rauschen. Bei 10 und 20 Auswertungen verschlechtert sich die Güte dadurch weiter. Oberhalb von $\bar\sigma \approx 4$ (ein Mehrfaches der Pixelstandardabweichung $\approx 0.7$) besteht das verrauschte Bild fast nur aus gaußschem Rauschen, und die Trajektorien sind nahezu gerade. Das lineare Gitter überbrückt $\bar\sigma > 4$ in zwei Schritten ($\infty \to 9 \to 4$). Das EDM-Gitter mit 10 Schritten verwendet dort vier Schritte. So bleiben zu wenige für den mittleren Bereich, in dem die Struktur des Kleidungsstücks entsteht.

<!--en:c41b2359-->
### Integrationsfehler, Guidance und Wahl des Samplers

<!--en:55bf586a-->
Bisher haben wir die Sampler allein anhand der FD beurteilt. Die FD bewertet eine Menge von Samples. Sie sagt nichts darüber aus, wie weit jedes Sample von dem Punkt entfernt ist, den eine exakte Integration des Felds vom selben Rauschbild aus erreichen würde. Um auch Fehler zu erfassen, die der FD entgehen, integrieren wir jedes Rauschbild als Referenz mit 200 Auswertungen des Heun-Verfahrens. Für jeden Löser messen wir den quadratischen Mittelwert der Abstände pro Pixel zur Referenz.

<!--en:b4bcbdb1-->
Wir tun dies für zwei Felder, da Guidance das zu integrierende Feld verändert. Classifier-Free Guidance (:numref:`subsec_diffusion-images-guided`) extrapoliert von der Vorhersage ohne Label zur Vorhersage mit Label gemäß $\tilde{\mathbf{u}} = \mathbf{u}_\varnothing +
\gamma(\mathbf{u}_c - \mathbf{u}_\varnothing)$. Damit ergibt $\gamma = 1$ das bedingte Feld, und jede Auswertung mit Guidance kostet zwei Netzdurchläufe. Ein Feld, das über seine beiden Bestandteile hinaus extrapoliert, sollte sich entlang einer Trajektorie auch schneller ändern. Wir erwarten daher, dass jeder Löser bei $\gamma = 2$ mehr Auswertungen benötigt, um ihm zu folgen.

<!--en:6d2cdeba-->
Die beiden Maße liefern unterschiedliche Urteile. Daran erkennen wir eine Eigenschaft der FD. Bei zehn Auswertungen ohne Guidance liegen die Samples des Heun-Verfahrens weiter von der Referenz entfernt als die des Euler-Verfahrens (Abstand 0.12 bis 0.13 gegenüber etwa 0.1). Ihre FD ist dennoch nur halb so groß (etwa 1.6 gegenüber etwa 3.2). Die FD bewertet die Verteilung der Samples. Der Fehler eines einzelnen Samples geht nicht direkt ein. Ein Löser kann also zu einem bestimmten Rauschbild das falsche Kleidungsstück erzeugen und insgesamt dennoch realistische Bilder liefern.

<!--en:d80aa3e6-->
Wie erwartet ist es mit Guidance schwieriger, dem Feld zu folgen. Bei $\gamma = 2$ erreicht die Referenz selbst etwa 2.1 bis 2.3, denn Guidance verbessert die Übereinstimmung mit der Klasse auf Kosten des Realismus. Bei zwanzig Auswertungen liegen alle Löser noch 0.4 bis 1 darüber. Bei fünfzig erreicht das Mehrschrittverfahren den Bereich der Referenz oder dessen Rand. Das Heun-Verfahren liegt noch 0.15 bis 0.2 darüber, das Euler-Verfahren etwa 0.3. Die Abstände bestätigen die Vorhersage nur teilweise. Bei zwanzig Auswertungen ist der Abstand des Heun-Verfahrens zur Referenz um ein Drittel bis die Hälfte größer als ohne Guidance. Beim Euler- und beim Mehrschrittverfahren hat er sich dagegen höchstens um ein Sechstel geändert. Bei zehn und zwanzig Auswertungen liegt das Mehrschrittverfahren der Referenz am nächsten. Mit Guidance erreicht also nur das Mehrschrittverfahren das Plateau innerhalb von fünfzig Auswertungen, entsprechend hundert Netzdurchläufen. Das Heun- und das Euler-Verfahren brauchen mehr. Wie viel mehr, lässt die Tabelle offen.

<!--en:4d59de78-->
Der Integrationsfehler ist auch sichtbar. Unten sehen wir je acht Bilder von vier Samplern, alle aus denselben acht Rauschbildern und mit denselben Labeln: die ancestrale Kette mit 50 Schritten, DDIM mit 10 und mit 50 Schritten sowie das Heun-Verfahren auf dem linearen Gitter mit 20 Auswertungen.

<!--en:b293731e-->
Die drei deterministischen Reihen stimmen Spalte für Spalte überein, denn die deterministischen Sampler nähern alle dieselbe Abbildung von Rauschen auf Bilder an. Zehn Schritte glätten Details wie Aufdrucke und Kanten, die bei fünfzig Schritten und beim Heun-Verfahren übereinstimmen. Die Reihe der ancestralen Kette wirkt körnig. Hintergründe und Stoffe sind gesprenkelt. Hier wird die überschüssige Varianz aus der ersten Tabelle sichtbar.

<!--en:5bfc1105-->
Wie wählen wir nun einen Sampler für ein Modell, das wir noch nicht vermessen haben? Wir legen Rauschbilder und Bewertungsmaß fest und beginnen mit einem deterministischen Löser auf dem Gitter des Trainingspfads. Dann verdoppeln wir die Zahl der Auswertungen, bis sich der Messwert höchstens noch um die Spannweite zwischen den Läufen ändert. Übrig bleibt der Modellfehler. Ein Sampler kann bestenfalls seinen eigenen Fehler vernachlässigbar klein halten. Bei unserem Modell endet das Verfahren dort, wo die Kurven in der Grafik abflachen, für das Heun-Verfahren bei etwa zwanzig Auswertungen. Frisches Rauschen hilft hier nie. Mit Guidance endet es später, bei etwa fünfzig Auswertungen des Mehrschrittverfahrens. Mit einer Auswertung pro Schritt und dem kleinsten Integrationsfehler ist dieses dort die bessere Standardwahl. Ein Gitter aus der Literatur sollte dieselbe Prüfung bestehen, bevor es das Trainingsgitter ersetzt. Bei zehn Auswertungen hat noch kein Löser das Plateau erreicht. Um es in einem oder zwei Schritten zu erreichen, brauchen wir ein anderes Modell (:numref:`sec_diffusion-few-step`).

<!--en:93fd633d-->
## Inversion und Bildbearbeitung

<!--en:c51fb937-->
Ein deterministischer Sampler integriert eine Differentialgleichung, und diese lässt sich in beiden Richtungen lösen. In umgekehrter Richtung ordnet sie jedem Bild ein Rauschbild zu, seinen *Code*. Aus ihm rekonstruiert der Sampler das Bild. Mithilfe der Codes können wir zwischen Bildern interpolieren. Starten wir denselben Sampler mit einem teilweise verrauschten Bild, können wir es bearbeiten. Geben wir bei jedem Schritt einen Teil des Bilds vor, können wir fehlende Bereiche ergänzen. Beides gelingt ohne erneutes Training.

<!--en:5a1fee50-->
### Der latente Code

<!--en:676102e7-->
`ddim_invert` führt den deterministischen Sampler aufwärts aus, von einem Bild bei $t = 0$ in Richtung Rauschen. Vom Pegel $t$ zum nächsthöheren Pegel $t' > t$ führt jeweils der Euler-Schritt

<!--en:79434d1b-->
mit der Rauschschätzung am unteren Ende. Nur diesen Punkt kennt der Encoder. Beim ersten Schritt, bei $\bar\sigma = 0$, ist der Prädiktor undefiniert. Deshalb wertet der Encoder das Bild mit dem Rauschpegel der nächsten Stufe aus.

<!--en:0c6b723d-->
Der Decoder geht mit $\hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_{t'}, \bar\sigma_{t'})$ wieder abwärts und wertet die Schätzung am oberen Ende aus. Die beiden Schätzungen unterscheiden sich um die Änderung von $\hat{\boldsymbol\epsilon}$ über einen Schritt. Selbst bei exaktem Feld führt der Hin- und Rückweg also nicht exakt zum Ausgangspunkt zurück. Wie jeder Fehler erster Ordnung sollte die Abweichung proportional zur Schrittweite abnehmen. Beide Richtungen verwenden dasselbe Feld. Ob dieses Feld stimmt oder nicht, die Abweichung misst den Fehler der Schritte auf diesem Feld. Allerdings könnte sich das gelernte Feld entlang seiner Trajektorien schneller ändern als ein exaktes Feld. In diesem Fall würde die Rekonstruktion langsamer konvergieren als bei einem exakten Feld. Wir kodieren und dekodieren 100 Testbilder mit jeweils eigenem Label und 10 bis 200 Schritten auf dem linearen Pfad. Die 20 000 Punkte der Mischung durchlaufen denselben Hin- und Rückweg mit dem exakten Feld.

<!--en:871a1a0f-->
In beiden Fällen ist die Rekonstruktion ungenau, und beide Fehler nehmen stetig ab. Von zehn auf hundert Schritte sinkt die Wurzel des mittleren quadratischen Fehlers in beiden Fällen etwa um den Faktor sechs. Das ist deutlich weniger als der für einen Fehler erster Ordnung erwartete Faktor zehn. Ab fünfzig Schritten fällt der Fehler der Bilder mindestens ebenso schnell wie der der Mischung. Das gelernte Feld ist entlang seiner Trajektorien also nicht rauer als ein exaktes Feld. Übrig bleibt der Fehler der Schritte. Bei fünfzig Schritten beträgt er 0.05 bis 0.06 pro Pixel auf einer Skala von $-1$ bis $1$, und die dritte Reihe kommt den Originalen nahe. Bei zehn Schritten beträgt er etwa 0.19. In der zweiten Reihe gehen Details wie das Karomuster des letzten Hemds verloren.

<!--en:38a37ed3-->
Weniger zufriedenstellend sind die Codes. Ihre Standardabweichung beträgt bei zehn Schritten etwa 0.7 und liegt selbst bei zweihundert Schritten noch unter 0.9. Codes, die ihre Bilder rekonstruieren, streuen also weniger als das Rauschen, für das der Sampler ausgelegt ist. Ein solcher Code ist kein typisches Sample aus standardnormalverteiltem Rauschen. Das müssen wir berücksichtigen, wenn wir zwischen Codes interpolieren oder neue Codes in ihrer Nähe ziehen.

<!--en:65e47e36-->
Verfahren, die echte Fotos über ihre Codes bearbeiten, stoßen auf eine weitere Schwierigkeit. Die Inversion folgt der Trajektorie eines Felds. Ein Sampler mit Guidance integriert ein anderes Feld. Geht er vom Code aus, verlässt er daher die Trajektorie und rekonstruiert das Bild nicht mehr. Die Null-Text-Inversion :cite:`Mokady.Hertz.Aberman.ea.2023` behebt dies, indem sie in jedem Schritt die Einbettung des Null-Labels optimiert, bis der Sampler mit Guidance das Bild aus seinem Code rekonstruiert.

<!--en:7c9f7278-->
Da der Sampler eine stetige Funktion seines Codes ist, können wir durch Interpolation zwischen den Codes von einem Bild zum anderen gelangen. Eine Gerade ist dafür ungünstig. In der Mitte zwischen zwei Codes mit gleicher Norm $r$ und Winkel $\omega$ beträgt die Norm $r \cos(\omega/2)$. Sie ist um etwa den Faktor $\sqrt{2}$ kleiner, wenn die Codes wie unabhängige Rauschbilder nahezu orthogonal sind. Mit einem so kurzen Code ist der Decoder nie gestartet. Die sphärische Interpolation folgt stattdessen dem Großkreis zwischen den beiden Codes,

<!--en:52a5422e-->
wodurch die Norm der Endpunkte erhalten bleibt. Wir dekodieren acht Punkte zwischen den Codes zweier Stiefeletten aus den Testbildern, die wir mit 50 Schritten bestimmt haben. Da die ganze Reihe zu einer Klasse gehört, genügt dem Decoder ein einziges Label.

<!--en:875825fc-->
Die Form geht allmählich von einer Stiefelette in die andere über. Die mittleren Bilder wirken jedoch eher wie eine Überlagerung beider Stiefeletten als wie eine dritte. Die sphärische Interpolation hält die Norm der Codes auf dem richtigen Wert. Nichts zwingt den Decoder dazu, die Codes zwischen zwei Stiefeletten auf eine dritte abzubilden.

<!--en:e3c292ab-->
### Bildbearbeitung durch partielles Verrauschen

<!--en:9490ddbf-->
Die Inversion sucht das Rauschen, aus dem sich ein Bild exakt rekonstruieren lässt. Für die Bearbeitung wollen wir das Gegenteil: einen Startpunkt, der die Details der Eingabe vergisst und ihren groben Aufbau bewahrt. :citet:`Meng.He.Song.ea.2022` verrauschen deshalb eine grobe Vorlage $\mathbf{y}$, etwa eine Skizze oder eine einfache Collage, direkt mit frischem Rauschen bis zu einem Zwischenzeitpunkt $t_0$ gemäß $\mathbf{x}_{t_0} = \alpha_{t_0}
\mathbf{y} + \sigma_{t_0} \boldsymbol\epsilon$. Dann führen sie den Sampler von $t_0$ zurück zu den Daten. Das Rauschen verdeckt Details der Eingabe, die in echten Bildern nicht vorkommen. Der Sampler ersetzt sie durch realistische Details, während der grobe Aufbau erhalten bleibt. Der Zeitpunkt $t_0$ bestimmt den Kompromiss zwischen Vorlagentreue (engl. fidelity) und Realismus. Wie gut lässt sich beides vereinbaren?

<!--en:86d978e1-->
Als Vorlagen dienen die zehn Mittelwertbilder der Klassen. Es sind verschwommene Silhouetten, denen kein echtes Kleidungsstück gleicht. Jede verrauschen wir hundertmal bis $t_0 \in \{0.3,
0.5, 0.7, 1\}$ und integrieren unter ihrem Klassenlabel mit 25 DDIM-Schritten zurück. Wir geben den quadrierten Abstand zur Eingabe pro Pixel (Vorlagentreue) und die FD der 1000 Ausgaben (Realismus) aus. Die Reihe $t_0 = 0$ zeigt die Eingaben.

<!--en:c3e8b8ef-->
Die FD der Eingaben liegt bei 85 bis 87. Mit wachsendem $t_0$ nimmt der Abstand zur Eingabe zu und die FD ab. Bei $t_0 = 0.3$ sind die Ausgaben schärfer als die Mittelwertbilder, wirken aber fleckig. Ihre FD von etwa 30 entspricht der ancestralen Kette mit zehn Schritten und liegt weit über den Werten aller deterministischen Sampler. Bei $0.5$ sehen sie wie Kleidungsstücke mit ungewöhnlichen Texturen aus (11 bis 12). Erst bei $0.7$ nähert sich die FD mit etwa 4 dem Wert der deterministischen Sampler mit zehn Schritten. Dann bleiben von der Eingabe nur Umriss und Haltung übrig. Bei $t_0 = 1$ ist die Eingabe vollständig vergessen. Jede Ausgabe ist ein neues Sample ihrer Klasse, mit einer FD unter 2. Da die beiden Reihen mit denselben Rauschbildern beginnen, sehen die Bilder bei $0.7$ und $1$ oft ähnlich aus. Bei einer so groben Vorlage wie einem Klassenmittelwert entsteht ein realistisches Bild erst bei einem $t_0$, das wenig von der Eingabe übrig lässt.

<!--en:d8ef96b5-->
### Inpainting

<!--en:6eeabbd8-->
Angenommen, wir kennen einen Teil eines Bilds, etwa die obere Hälfte, und möchten den Rest vom Modell ergänzen lassen. Mit $\mathbf{m}$ bezeichnen wir die Maske, die auf den bekannten Pixeln eins ist, und mit $\mathbf{y}$ deren Werte. Auf jedem Pegel wissen wir, wie die bekannten Pixel von $\mathbf{x}_s$ aussehen sollten: $\alpha_s
\mathbf{y}$ plus Rauschen mit der Skala $\sigma_s$. Beim Inpainting durch Ersetzen (engl. replacement inpainting) :cite:`song2021score,Lugmayr.Danelljan.Romero.ea.2022` führt der Sampler daher jeden Schritt wie gewohnt aus. Anschließend überschreibt er die bekannten Pixel mit einer auf dem neuen Pegel frisch verrauschten Kopie von $\mathbf{y}$:

<!--en:cfa8c511-->
Das Ersetzen ist eine Näherung. Jeder Schritt bewegt alle Pixel, als wären auch die bekannten frei. Dann überschreibt das Verfahren die bekannten Pixel mit Rauschen, das unabhängig davon ist, wohin der Schritt sie geführt hat. Die bekannten Pixel beeinflussen die unbekannten erst, wenn das Netz das verrauschte Bild das nächste Mal auswertet. Der Sampler zieht die unbekannten Pixel also nur näherungsweise aus ihrer bedingten Verteilung bei gegebenen bekannten Pixeln. `inpaint` führt den Sampler schrittweise aus, damit die Pixel nach jedem Schritt überschrieben werden können.

<!--en:15bf8d07-->
Wir behalten die obere Hälfte von acht Testbildern bei und erzeugen für jedes Bild drei Ergänzungen aus unterschiedlichem Rauschen mit seinem jeweiligen Label. Die Reihen zeigen die bekannten Hälften, die drei Ergänzungen und die Originale.

<!--en:a08b45c5-->
Die meisten Ergänzungen setzen die bekannte Hälfte zu einem plausiblen Kleidungsstück ihrer Klasse fort. Dabei unterscheiden sich die drei Ergänzungen eines Bilds. An der Nahtstelle wird die Näherung sichtbar. Bei mehreren Ergänzungen, am deutlichsten bei den Hemden, weicht die untere Hälfte in Textur oder Farbton von der oberen ab (Aufgabe :exref:`ex_samplers-repaint` untersucht, wie sich diese Abweichung messen und beheben lässt).

<!--en:89226247-->
Alle Sampler in diesem Abschnitt integrierten dasselbe gelernte Feld. Sie unterschieden sich nur in der Art der Integration. Das Feld selbst haben wir als gegeben angenommen. Wir hatten es mit einem ungewichteten quadratischen Fehler für eine von uns gewählte Zielgröße gelernt. Im nächsten Abschnitt untersuchen wir, warum dieser Verlust der richtige ist und was sich mit der Wahl von Zielgröße, Pfad und Gewichtung ändert.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:6469307d-->
Das Training hängt nur von der Randverteilung von $\mathbf{x}_t$ zum jeweiligen Zeitpunkt $t$ ab. Ein einziges trainiertes Netz genügt daher für die gesamte DDIM-Familie, deren Parameter $\eta \ge 0$ die Stärke der Stochastizität bestimmt. Bei $\eta = 0$ ist DDIM deterministisch und entspricht dem Euler-Verfahren für die Wahrscheinlichkeitsfluss-ODE $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$ in den umskalierten Koordinaten $(\bar{\mathbf{x}}, \bar\sigma) = (\mathbf{x}/\alpha_t, \sigma_t/\alpha_t)$. Bei unserem Fashion-MNIST-Modell sind deterministische ODE-Löser den stochastischen Samplern ($\eta = 1$ und ancestrale Kette) bei kleinen und mittleren Rechenbudgets durchweg überlegen. Auf dem linearen Gitter erreicht das Heun-Verfahren zweiter Ordnung das FD-Plateau des Modells in etwa 20 Auswertungen. Mit Guidance erreicht das Adams–Bashforth-Mehrschrittverfahren zweiter Ordnung das Plateau nach spätestens 50 Auswertungen, bei einer Auswertung pro Schritt. In umgekehrter Richtung bildet die deterministische ODE (`ddim_invert`) jedes Bild auf einen latenten Rauschcode ab. So lassen sich ohne erneutes Training Codes sphärisch interpolieren, Bilder durch partielles Verrauschen bearbeiten und fehlende Bereiche per Inpainting durch Ersetzen ergänzen.

<!--en:2294243a-->
## Übungen

<!--en:9856a47d-->
1. **Der Vorwärtsschritt und die Markov-Eigenschaft:** ● Gegeben sei $\mathbf{x}_0$. Ziehen Sie $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ sowie $\mathbf{x}_s$ aus :eqref:`eq_diffusion-ddim-family` mit demselben $\boldsymbol\epsilon$.
    1. Berechnen Sie den Mittelwert und die Kovarianz von $\mathbf{x}_t$ bei gegebenem $\mathbf{x}_s$ und $\mathbf{x}_0$ durch Konditionieren der gemeinsamen Gauß-Verteilung.
    1. Zeigen Sie, dass der Mittelwert von $\mathbf{x}_0$ abhängt, außer wenn $\tilde\sigma$ seinen Wert für $\eta = 1$ annimmt. Zeigen Sie, dass die bedingte Verteilung dann dem Vorwärtsschritt $\mathbf{x}_t = (\alpha_t / \alpha_s) \mathbf{x}_s +
       \sqrt{\sigma_t^2 - (\alpha_t / \alpha_s)^2 \sigma_s^2}\,
       \boldsymbol\nu$ der Vorwärtskette aus :numref:`sec_diffusion-gaussian-paths` entspricht. Folgern Sie, dass nur bei dieser Kette der Vorwärtsprozess innerhalb der Familie die Markov-Eigenschaft besitzt.
    1. Zeigen Sie, dass der Vorwärtsprozess bei $\tilde\sigma = 0$ in jedem Schritt die Form $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t
       \boldsymbol\epsilon$ hat, mit demselben $\boldsymbol\epsilon$ für alle Zeitpunkte. Warum kann die Regression diese Vorwärtsprozesse nicht unterscheiden?
   :exlabel:`ex_samplers-forward-step`

<!--en:3dd5c07c-->
1. **Gaußverteilte Daten:** Die Daten seien eindimensional und gemäß $\mathcal{N}(0, v)$ verteilt.
    1. Zeigen Sie in den Koordinaten $(\bar x, \bar\sigma)$, dass die exakte Rauschschätzung $\hat\epsilon = \bar\sigma \bar x / (v + \bar\sigma^2)$ lautet. Zeigen Sie weiter, dass ein deterministischer Schritt von $\bar\sigma$ nach $\bar\sigma'$ die Koordinate $\bar x$ mit $(v + \bar\sigma \bar\sigma') / (v +
       \bar\sigma^2)$ multipliziert und der exakte Fluss sie mit $\sqrt{(v +
       \bar\sigma'^2) / (v + \bar\sigma^2)}$ multipliziert.
    1. Zeigen Sie, dass der Faktor des Schritts den exakten Faktor nie überschreitet und Gleichheit nur bei $\bar\sigma' = \bar\sigma$ gilt. Was folgt daraus für die Varianz deterministisch erzeugter Samples bei gaußverteilten Daten auf einem beliebigen Gitter?
    1. Betrachten Sie nun einen beliebigen Vertreter der Familie :eqref:`eq_diffusion-ddim-step` mit $0 \leq \eta \leq 1$ und exakten Schätzungen. Schreiben Sie die Varianz von $x_s$ nach einem Schritt ausgehend von $x_t \sim \mathcal{N}(0, \alpha_t^2 v +
       \sigma_t^2)$ als Funktion von $c = \sqrt{\sigma_s^2 -
       \tilde\sigma^2}$. Zeigen Sie mit der Cauchy-Schwarz-Ungleichung, dass sie unter dem exakten Wert $\alpha_s^2 v + \sigma_s^2$ bleibt und mit wachsendem $\eta$ fällt. Welche der in diesem Abschnitt ausgegebenen Streuungen erklärt das?
    1. Setzen Sie $v = 0.25$, die Varianz einer Koordinate eines Modus von `mix`. Berechnen Sie die Standardabweichung $s$ deterministischer Samples ausgehend von $x_1
       \sim \mathcal{N}(0, 1)$ auf den varianzerhaltenden Gittern mit 200, 50, 20, 10 und 5 Schritten. Vergleichen Sie $2s^2$, die Streuung einer zweidimensionalen Punktwolke mit dieser Standardabweichung je Koordinate, mit dem Datenwert $0.5$ und den in diesem Abschnitt ausgegebenen Streuungen. Sagt ein gaußverteilter Modus die Kontraktion der Mischung korrekt voraus, überschätzt er sie oder unterschätzt er sie?
   :exlabel:`ex_samplers-gaussian-data`

<!--en:934801a8-->
1. [extended] **Ein Sampler zweiter Ordnung in der Log-SNR.** ●● DPM-Solver-2 :cite:`Lu.Zhou.Bao.ea.2022` macht mit zwei Auswertungen einen Schritt von $\bar\sigma$ nach $\bar\sigma'$. Die zweite liegt am Mittelpunkt $\bar\sigma_m = \sqrt{\bar\sigma \bar\sigma'}$ bezüglich $\log \bar\sigma$. Es gilt $\bar{\mathbf{x}}_m = \bar{\mathbf{x}} + (\bar\sigma_m - \bar\sigma)\,
   \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}, \bar\sigma)$ und $\bar{\mathbf{x}}' = \bar{\mathbf{x}} + (\bar\sigma' - \bar\sigma)\,
   \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_m, \bar\sigma_m)$.
    1. Sei $h = \log(\bar\sigma / \bar\sigma')$. Schreiben Sie die exakte Änderung von $\bar{\mathbf{x}}$ über den Schritt als Integral von $\hat{\boldsymbol\epsilon}$ entlang der Trajektorie. Zeigen Sie, dass der lokale Fehler des Updates $O(\bar\sigma h^3)$ beträgt und der des Euler-Schritts $O(\bar\sigma h^2)$. Nehmen Sie an, dass $\hat{\boldsymbol\epsilon}$ entlang der Trajektorie eine glatte Funktion von $\log \bar\sigma$ und bezüglich $\mathbf{x} =
       \bar{\mathbf{x}} / \sqrt{1 + \bar\sigma^2}$ Lipschitz-stetig ist. Die Lipschitz-Konstante ist also von der Ordnung $1/\bar\sigma$ bezüglich $\bar{\mathbf{x}}$ für große $\bar\sigma$.
    1. Implementieren Sie das Update mit der Schnittstelle `pred(x, alpha, sigma)`. Verwenden Sie $\alpha = 1 / \sqrt{1 + \bar\sigma^2}$ und $\sigma = \bar\sigma
       \alpha$ und führen Sie den letzten Schritt nach $\bar\sigma = 0$ mit dem Euler-Verfahren aus. Vergleichen Sie Ihr Verfahren mit `ode_sample`, `heun_sample` und `multistep_sample` bei etwa 11, 21 und 51 Auswertungen. Verwenden Sie auf der Mischung mit `pred_exact` den Abstand zu einer Referenz mit 1000 Schritten und auf Bildern die FD.
    1. Wiederholen Sie den Versuch mit Pegeln, die in $\log \bar\sigma$ zwischen 80 und 0.002 gleichmäßig verteilt sind. Welche Kombination aus Verfahren und Gitter ist bei welchem Budget am besten?
   :exlabel:`ex_samplers-dpm-solver`

<!--en:512c2a12-->
1. **Die optimale Varianz des Rückwärtsschritts:** ● Ein stochastischer Schritt von $t$ nach $s$ führt zum A-posteriori-Mittelwert $\alpha_s \hat{\mathbf{x}}_0 + \sigma_s \kappa
   \hat{\boldsymbol\epsilon}$ und addiert gaußsches Rauschen mit Varianz $v$ je Koordinate. :citet:`ho2020denoising` erprobten die Varianz des Vorwärtsschritts und die A-posteriori-Varianz. :citet:`Bao.Li.Zhu.ea.2022` fanden das beste $v$ für beliebige Daten. Nehmen Sie exakte Schätzungen an. Dabei sei $\bar e_t =
   \mathbb{E}\|\boldsymbol\epsilon - \hat{\boldsymbol\epsilon}\|^2 / d$ der Fehler je Koordinate der exakten Rauschschätzung zum Zeitpunkt $t$ in $d$ Dimensionen.
    1. Aufgabe :exref:`ex_gaussian-paths-gaussian-reversal`a in :numref:`sec_diffusion-integration` zeigt, dass der wahre Rückwärtsübergang $q(\mathbf{x}_s \mid \mathbf{x}_t)$ die Kovarianz $\tau^2
       \mathbf{I} + c^2 \textrm{Cov}(\mathbf{x}_0 \mid \mathbf{x}_t)$ hat. Dabei ist $c$ der Koeffizient von $\mathbf{x}_0$ im Mittelwert von $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$. Schreiben Sie diesen Mittelwert als $\alpha_s \mathbf{x}_0 + \sigma_s \kappa \boldsymbol\epsilon$ mit $\boldsymbol\epsilon = (\mathbf{x}_t - \alpha_t
       \mathbf{x}_0)/\sigma_t$. Folgern Sie, dass $\tau^2 = \tilde\sigma^2$ die A-posteriori-Varianz angibt und $c = k = \alpha_s - \sigma_s \kappa\,
       \alpha_t / \sigma_t$ gilt. Wie lautet der Mittelwert des wahren Rückwärtsübergangs?
    1. Zeigen Sie, dass der Wert von $v$, der die erwartete KL-Divergenz von $q(\mathbf{x}_s \mid \mathbf{x}_t)$ zum gaußschen Schritt minimiert, durch $v^\star = \tilde\sigma^2 + k^2 \bar\sigma_t^2 \bar e_t$ gegeben ist. Zeigen Sie außerdem, dass $\bar e_t = 1 - \sigma_t^2\, \mathbb{E}\|\nabla \log
       p_t(\mathbf{x}_t)\|^2 / d$ gilt, sodass sich $v^\star$ allein aus dem Score ergibt.
    1. Zeigen Sie, dass $v^\star$ für Daten aus einem einzigen Punkt die A-posteriori-Varianz ist und für standardnormalverteilte Daten auf dem varianzerhaltenden Pfad die Varianz des Vorwärtsschritts.
    1. (Optional, mit Code.) Schätzen Sie $\bar e_t$ für die Mischung mit `pred_exact` auf dem varianzerhaltenden Gitter mit 50 Schritten. Vergleichen Sie die Streuung von je 20 000 Samples des stochastischen Schritts für die drei Varianzen.
   :exlabel:`ex_samplers-optimal-variance`

<!--en:cbf31aa2-->
1. [code] **Inpainting mit Angleichung der Bildhälften.** ● Die von `inpaint` erzeugten Ergänzungen können an der Nahtstelle von der bekannten Bildhälfte abweichen.
    1. Messen Sie an der Naht die mittlere absolute Differenz zwischen den Zeilen 13 und 14 der ergänzten Bilder. Vergleichen Sie sie mit derselben Größe für echte Testbilder.
    1. RePaint :cite:`Lugmayr.Danelljan.Romero.ea.2022` ermöglicht einen Informationsaustausch zwischen den Hälften. Nach jedem Schritt von $t$ nach $s$ verrauscht es das Ergebnis mit dem Vorwärtsschritt wieder bis $t$ und wiederholt den Schritt, insgesamt $r$ Mal pro Pegel. Implementieren Sie dies mit $r \in \{1, 5, 10\}$ bei 50 Pegeln und vermessen Sie die Naht erneut. Was kostet jede Wiederholung?
    1. Ziehen Sie für jedes $r$ fünf Ergänzungen desselben Testbilds. Wie vielfältig sind sie?
   :exlabel:`ex_samplers-repaint`

<!--en:ec7506f3-->
1. [code] **Ein Bild in Richtung einer anderen Klasse bearbeiten.** Verwenden Sie das klassenbedingte Modell aus diesem Abschnitt und steuern Sie die Erzeugung mit `d2l.make_pred(d2l.cfg(ema, c, gamma), lin, 'velocity')`, wobei `c` ein Label pro Bild enthält.
    1. Verrauschen Sie 100 Testbilder von Hosen bis $t_0 \in \{0.3, 0.5, 0.7, 0.9\}$. Integrieren Sie mit 25 DDIM-Schritten unter dem Label für Kleider zurück, einmal mit $\gamma = 1$ und einmal mit $\gamma = 2$. Geben Sie wie in der Codezelle zur Bildbearbeitung den Anteil der Ausgaben an, die `cnn` als Kleider klassifiziert, sowie den quadrierten Abstand pro Pixel zur Eingabe.
    1. Beginnen Sie mit reinem Rauschen. Verwenden Sie beim Sampling das Label für Hosen und $\gamma = 1$ von $t = 1$ bis $t^*$, danach das Label für Kleider und die beiden Guidance-Stärken aus (a) bis $t = 0$ (`ddim_sample` akzeptiert `t_start` und `t_end`). Wählen Sie $t^* \in \{0.3, 0.5, 0.7, 0.9\}$. Tragen Sie den als Kleider klassifizierten Anteil gegen $t^*$ auf und zeichnen Sie daneben die Kurve aus (a) in Abhängigkeit von $t_0$. Vergleichen Sie beide mit dem Zeitpunkt der Festlegung auf einen Modus aus Aufgabe :exref:`ex_gaussian-paths-commitment-time` in :numref:`sec_diffusion-integration` für die Mischung.
    1. Zeigen Sie, dass beide Experimente mit einem exakten Netz bei gleichem $\gamma$ und $t_0 = t^*$ Ausgaben mit derselben Verteilung liefern. Welche Unterschiede zwischen den Kurven bleiben, und woher kommen sie?

    *Teil (a) beruht auf Berkeley CS180/280A, [Projekt 5A](https://cal-cs180.github.io/fa25/hw/proj5/parta.html) (Herbst 2025), Abschnitt 1.7.3.*
   :exlabel:`ex_samplers-class-edit`

<!--en:98daaf49-->
[Diskussion](https://d2l.discourse.group/t/36890)

<!--en:98ceeff4-->
Schnelle Diffusionssampler, Inversion und Bildbearbeitung<br>**DDIM-Familie (η) · Heun- und Mehrschrittverfahren für ODEs · Wahl der Gitterpunkte · ODE-Inversion und Slerp · Bearbeitung durch partielles Verrauschen · Inpainting**

<!--en:83d68880-->
::: {.slide title="Warum ein trainiertes Netz für viele Sampler genügt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:ee74887b-->
- **Das Training hängt nur von den Randverteilungen $p_t(\mathbf{x}_t)$ zu einzelnen Zeitpunkten ab:** Die gemeinsame Übergangsverteilung $q(\mathbf{x}_s, \mathbf{x}_t \mid \mathbf{x}_0)$ zweier Rauschpegel $s < t$ geht nie in den Regressionsverlust ein.

<!--en:4a488913-->
- **Familien von Rückwärtsprozessen ohne Markov-Eigenschaft (Song, Meng & Ermon, 2020):** Jeder Übergang $t \to s$, der die Randverteilung $\mathcal{N}(\alpha_s\mathbf{x}_0, \sigma_s^2\mathbf{I})$ erhält, ist mit dem trainierten Prädiktor $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ vereinbar.

<!--en:bd3829ad-->
- **Drei Stellgrößen für schnelles Sampling:** Stochastizität $\eta$, ODE-Löser höherer Ordnung (Heun- oder Mehrschrittverfahren) und Lage der Gitterpunkte ($\bar\sigma_i$).

<!--en:927163f2-->
::: {.slide title="Wie die DDIM-Familie die Rauschvarianz aufteilt"}

<!--en:b72e0277-->
[Rückwärtsschritte mit unveränderten Randverteilungen]{.kicker}

<!--en:b973674d-->
- **Die Randverteilung bleibt für alle $\tilde\sigma \in [0, \sigma_s]$ erhalten:** Bei gegebenem $\mathbf{x}_0$ addieren sich die Varianzen der orthogonalen Terme für beibehaltenes und frisches Rauschen gemäß $(\sigma_s^2 - \tilde\sigma^2) + \tilde\sigma^2 = \sigma_s^2$.

<!--en:d9e6501c-->
- **DDIM-Parameter für die Stochastizität $\eta \ge 0$:** Er skaliert $\tilde\sigma$ relativ zur Standardabweichung des Vorwärts-Posteriors mithilfe von $\kappa = \bar\sigma_s/\bar\sigma_t < 1$.

<!--en:e231c8f4-->
::: {.slide title="DDIM (η = 0, 1) und die ancestrale Kette im Vergleich"}

<!--en:7b41db4e-->
[Deterministische ODE und stochastische Ketten]{.kicker}

<!--en:0891a3fa-->
Mit $\eta = 0$, $\eta = 1$ oder der ancestralen Varianz von Ho et al. erhalten wir aus genau denselben Netzvorhersagen $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ drei verschiedene Rückwärtsprozesse:

<!--en:23630098-->
- **$\eta = 0$ (deterministisches DDIM = Euler-Verfahren für die ODE):** Es gilt $\tilde\sigma = 0$, also $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$ oder $\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\hat{\boldsymbol\epsilon}$. Das entspricht genau einem Euler-Schritt für $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$.

<!--en:90054787-->
- **$\eta = 1$ (DDIM mit A-posteriori-Varianz):** $\tilde\sigma^2 = \sigma_s^2(1 - \kappa^2)$ ist die exakte bedingte Varianz von $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ (exakt für Daten aus einem einzigen Punkt).

<!--en:e45ea57b-->
- **Ancestrale Kette:** Sie verwendet die größere Varianz des Vorwärtsschritts $\sigma_t^2(1 - \kappa^2) = (\sigma_t^2/\sigma_s^2)\tilde\sigma_{\eta=1}^2$ (exakt für $\mathcal{N}(\mathbf{0}, \mathbf{I})$-verteilte Daten auf dem varianzerhaltenden Pfad).

<!--en:06b4d705-->
::: {.slide title="Fehler eines Verfahrens erster Ordnung bei der 2D-Mischung" layout="tight"}

<!--en:b3f03045-->
[Exakter Prädiktor bei verschiedenen Schrittzahlen]{.kicker}

<!--en:3d5ba352-->
DDIM ($\eta = 0, 1$) und die ancestrale Kette verwenden den exakten Prädiktor `pred_exact` der Mischung. Bei $20{,}000$ Startpunkten und $N \in \{5, 10, 20, 50, 200, 1000\}$ Schritten bleibt so nur der Diskretisierungsfehler:

<!--en:a7467eb7-->
- **Konvergenz erster Ordnung mit $O(1/N)$:** Der mittlere $L_2$-Abstand der Samples bei $\eta = 0$ zur Trajektorie mit 1000 Schritten halbiert sich, wenn sich $N$ verdoppelt ($0.40$ bei $5$ Schritten $\to$ $0.21$ bei $10$ $\to$ $0.04$ bei $50$).

<!--en:e13a16b0-->
- **Grobe Schritte ziehen die Modi systematisch zusammen:** Bei $N = 10$ Schritten verringert deterministisches DDIM ($\eta = 0$) die Streuung innerhalb der Modi auf **$0.38$** (gegenüber $0.48$ bei den echten Daten). Wer $\hat{\boldsymbol\epsilon}$ über den Schritt konstant hält, entfernt zu viel Rauschen.

<!--en:a93e9303-->
::: {.slide title="Fehler eines Verfahrens erster Ordnung bei der 2D-Mischung (Fortsetzung)" layout="tight"}

<!--en:b3f03045-2-->
[Exakter Prädiktor bei verschiedenen Schrittzahlen]{.kicker}

<!--en:f0d504a6-->
- **Warum $\eta = 1$ die Modi noch stärker zusammenzieht ($0.32$):** Ersetzt man das korrelierte beibehaltene Rauschen $\hat{\boldsymbol\epsilon}$ durch orthogonales frisches Rauschen $\boldsymbol\xi$, entfällt der positive Kreuzterm mit $\hat{\mathbf{x}}_0$. Die ancestrale Kette erhöht die Streuung dagegen zu stark auf **$0.52$**.

<!--en:03f24b62-->
::: {.slide title="Integration zweiter Ordnung mit dem Prädiktor-Korrektor-Verfahren von Heun"}

<!--en:b7783c2c-->
[Diskretisierungsfehler von O(h) auf O(h²) senken]{.kicker}

<!--en:540d9aaa-->
- **Steigungen nach der Trapezregel mitteln:** Nach einem Euler-Probeschritt bis $\bar{\mathbf{x}}'$ mittelt das Verfahren die Steigungen am Anfang und am Ende ($2$ NFE/Schritt). So sinkt der lokale Fehler von $O(\Delta^2)$ auf $O(\Delta^3)$.

<!--en:c81a22c5-->
- **Zahlenbeispiel für $\mathcal{N}(\mathbf{0}, \mathbf{I})$ ($\bar\sigma: 1 \to 0.5$):** Der exakte Faktor beträgt $0.7906$. Euler liefert $0.7500$ (Fehler $-0.0406$), Heun dagegen $0.8000$ (um den Faktor $4.3\times$ kleinerer Fehler von $+0.0094$).

<!--en:df160eb3-->
::: {.slide title="Exponentielle Integratoren und Mehrschrittverfahren"}

<!--en:9d90d635-->
[Genauigkeit zweiter Ordnung bei einer Auswertung pro Schritt]{.kicker}

<!--en:f3b32664-->
- **Koordinaten des exponentiellen Integrators ($\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$):** Die lineare Drift wird analytisch integriert. Damit bestimmt allein das neuronale Netz die Gleichung $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}, \bar\sigma)$.

<!--en:fa9977c7-->
- **Adams–Bashforth zweiter Ordnung (`multistep_sample` / DPM-Solver++):** Mit der gespeicherten Steigung $\hat{\boldsymbol\epsilon}'$ aus dem vorigen Schritt $\Delta'$ extrapoliert das Verfahren linear zur Schrittmitte, bei **1 NFE pro Schritt**.

<!--en:7d1a2af9-->
::: {.slide title="Lage der Gitterpunkte auf linearen, VP- und EDM-Gittern"}

<!--en:e636bdeb-->
[Wo lohnen sich die Auswertungen eines Lösers?]{.kicker}

<!--en:d014ea05-->
Karras et al. (2022) trennten das **Gitter für das Sampling** $\{\bar\sigma_0, \ldots, \bar\sigma_N\}$ vom Trainingsplan. Sie schlugen das polynomiale **EDM-Gitter** mit gleichmäßigen Abständen in $\bar\sigma^{1/\rho}$ vor ($\bar\sigma_{\max} = 80, \bar\sigma_{\min} = 0.002, \rho = 7$):

<!--en:221b167d-->
- **Unterschiedliche Aufteilung von $\bar\sigma = \sigma/\alpha$:** Von 10 Schritten verwenden das VP- und das EDM-Gitter jeweils $4$ bei sehr starkem Rauschen ($\bar\sigma > 4$, fast rein gaußsches Rauschen und gerade Trajektorien). Das lineare Gitter springt über $\infty \to 9 \to 4$ und legt **$7$ seiner $10$ Schritte in den entscheidenden Bereich $\bar\sigma \in [0.11, 4.0]$, in dem die Struktur entsteht**.

<!--en:479480f9-->
::: {.slide title="Deterministische und stochastische Sampler für Kleidungsbilder"}

<!--en:f61326be-->
[Empirischer Vergleich bei verschiedenen Schrittzahlen]{.kicker}

<!--en:3a9c6776-->
Das trainierte klassenbedingte U-Net für Fashion-MNIST liefert die FD-Werte für 1000 Samples mit drei gepaarten Rauschbildsätzen (Mittelwert $\pm$ halbe Spannweite, Untergrenze aus echten Bildern etwa $0.9$ bei einer halben Spannweite von $0.15$ bis $0.2$):

<!--en:de1648c2-->
- **Deterministisches DDIM ($\eta = 0$) liegt bei jeder Schrittzahl $\le 200$ vorn:** Seine FD liegt um ein Drittel bis fast die Hälfte unter der für $\eta = 1$ bei $10$ bis $200$ Schritten. Schon bei $50$ Schritten ($\approx 1.7$ bis $1.8$) nähert es sich dem Plateau für $1{,}000$ Schritte ($\approx 1.3$).

<!--en:9ec3b90f-->
- **Warum die ancestrale Kette bei $\le 100$ Schritten schlecht abschneidet:** Ihre Vorwärtsschrittvarianz $\sigma_t^2(1 - \kappa^2)$ fügt auf groben Gittern für Bilder viel zu viel Rauschen hinzu. Sie erreicht $\mathrm{FD} \approx 8$ bis $11$ bei $50$ Schritten (schlechter als DDIM mit 10 Schritten bei etwa $5.4$) und zieht mit $\eta = 0$ erst bei $1{,}000$ Schritten gleich.

<!--en:b62e53aa-->
::: {.slide title="Ordnung des Lösers und Wahl des Gitters bei gleicher Zahl von Auswertungen"}

<!--en:46fccc65-->
[Euler-, Heun- und Mehrschrittverfahren im Vergleich]{.kicker}

<!--en:6c65ad2f-->
Der Vergleich von Euler, Heun ($2$ NFE/Schritt) und dem Adams–Bashforth-Mehrschrittverfahren zweiter Ordnung ($1$ NFE/Schritt) bei gleicher Zahl von **Netzauswertungen (NFE)** auf drei Gittern zeigt, wie stark die Lage der Gitterpunkte und die Ordnung des Lösers zusammenwirken:

<!--en:287de2eb-->
::: {.slide title="Konvergenz der FD mit steigender Zahl von Netzauswertungen"}

<!--en:fbaeacd5-->
[Wie schnell erreichen die Löser das asymptotische Plateau des Modells?]{.kicker}

<!--en:cd0329d5-->
Die doppelt logarithmische Darstellung der FD über der Gesamtzahl der Netzauswertungen zeigt, wie schnell jeder Löser die asymptotische Qualitätsgrenze des Modells erreicht ($\mathrm{FD} \approx 1.15$ bis $1.30$):

<!--en:0e455f05-->
- **20 Auswertungen genügen bei Heun zweiter Ordnung:** Auf dem linearen Gitter erreicht Heun $\mathrm{FD} \approx 1.2$ bis $1.3$ mit nur $20$ Auswertungen ($11$ Schritte). Das entspricht $1{,}000$ ancestralen Schritten bei **$1/50\text{th}$ des Rechenaufwands**.

<!--en:0f6bc60d-->
- **50 Auswertungen für das Mehrschrittverfahren, 50 bis 100 für Euler:** Das Mehrschrittverfahren erreicht $\approx 1.2$ bis $1.3$ auf dem linearen Gitter nach spätestens $50$ Auswertungen. Euler liegt dann noch bei $\approx 1.4$ bis $1.5$.

<!--en:c9a3263b-->
- **Warum EDM-Gitter hier bei $10$ bis $20$ NFE zurückliegen:** Mit $40\%$ aller Schritte bei $\bar\sigma > 4$ bleiben zu wenige für den mittleren Bereich, in dem die $28 \times 28$-Bilder der Kleidungsstücke Gestalt annehmen.

<!--en:b6849935-->
::: {.slide title="Trajektorienfehler und FD mit Guidance (γ = 1, 2)"}

<!--en:a684a361-->
[Vergleich mit einer Referenz aus 200 Netzauswertungen]{.kicker}

<!--en:d5c4a253-->
Bedeutet eine kleinere FD immer einen kleineren ODE-Integrationsfehler pro Bild? Neben der FD dient als Maß der quadratische Mittelwert der Pixelabstände zu einer Heun-Referenz mit 200 NFE, bei $\gamma = 1$ (bedingt, ohne Guidance) und $\gamma = 2$ (CFG):

<!--en:f9e1a285-->
- **FD der Verteilung und punktweise Genauigkeit der Trajektorie:** Bei $10$ NFE ($\gamma = 1$) hat Heun einen *größeren* punktweisen RMS-Fehler als Euler ($0.12$ bis $0.13$ gegenüber etwa $0.1$), aber **nur die halbe FD** (etwa $1.6$ gegenüber etwa $3.2$). Euler erzeugt durch Glättung zum Mittelwert hin unscharfe Bilder. Heun erzeugt scharfe Kleidungsbilder, die etwas über die exakte Trajektorie hinausschießen.

<!--en:66920238-->
- **Mit Guidance ($\gamma = 2$) gewinnt das Mehrschrittverfahren:** Die Extrapolation $(1-\gamma)\mathbf{u}_\varnothing + \gamma\mathbf{u}_c$ macht die ODE steifer. Das Mehrschrittverfahren zweiter Ordnung hat den kleinsten RMS-Trajektorienfehler und erreicht für $\gamma = 2$ das FD-Plateau nach spätestens $50$ Auswertungen.

<!--en:fead584f-->
::: {.slide title="Artefakte der Sampler bei identischem Ausgangsrauschen"}

<!--en:36b77850-->
[Deterministische Konvergenz und Körnung bei der ancestralen Kette]{.kicker}

<!--en:6b03ffdb-->
Vier Sampler erzeugen jeweils acht Kleidungsbilder aus genau demselben Ausgangsrauschen $\mathbf{z}$ und denselben Klassenlabeln. Die Werte der Tabelle lassen sich so auf einen Blick nachvollziehen:

<!--en:4c0fab78-->
- **Reihe 1 (ancestrale Kette, 50 Schritte):** Die zu hohe Varianz des Vorwärtsschritts hinterlässt sichtbares Salz-und-Pfeffer-Rauschen auf Hintergründen und Stoffen.

<!--en:07fdb450-->
- **Reihe 2 (DDIM, 10 Schritte):** Feine Aufdrucke und Kragenkanten verschwinden durch die Glättung.

<!--en:83d39a60-->
- **Reihen 3 und 4 (DDIM mit 50 gegenüber Heun mit 20 NFE):** Haltung, Schattierung und scharfe Details stimmen Spalte für Spalte überein. Beide Verfahren konvergieren also gegen dieselbe deterministische ODE-Abbildung.

<!--en:56d0473d-->
::: {.slide title="Auswahl eines Samplers in der Praxis"}

<!--en:ddfced43-->
[Sampling für ein neues Modell abstimmen]{.kicker}

<!--en:1e410eda-->
Die optimale Schrittzahl hängt von der Bildauflösung, dem Trainingsplan und der Guidance-Stärke $\gamma$ ab. Daher empfiehlt sich für jedes neue Modell ein gepaarter Vergleich mit schrittweiser Verdopplung:

<!--en:88b1ce0e-->
1. **Gemeinsame Zufallsstartwerte und ein kalibriertes Maß festlegen:** Einen festen Rauschtensor $\mathbf{Z}$ zur Validierung ziehen ($n \ge 1{,}000$). Die FD mit der Untergrenze aus echten Bildern bei gleichem $n$ vergleichen.

<!--en:c9793c4d-->
2. **Deterministisch auf dem Trainingsgitter des Modells beginnen:** Auf dem Gitter des Trainingspfads Heun zweiter Ordnung (für $\gamma = 1$) oder ein Mehrschrittverfahren zweiter Ordnung wie DPM-Solver++(2M) (bei Erzeugung mit Guidance, $\gamma > 1$) verwenden.

<!--en:d0b27102-->
3. **NFE verdoppeln, bis die FD weniger stark schwankt als zwischen den Läufen:** Ändert die Verdopplung ($10 \to 20 \to 50 \to 100$) die FD um weniger als die Spannweite über mehrere Zufallsstartwerte, ist der Diskretisierungsfehler vernachlässigbar. Der verbleibende Abstand zur Untergrenze für echte Bilder geht auf die Modellkapazität oder den Trainingsfehler zurück.

<!--en:3b9842a6-->
::: {.slide title="Deterministische ODE-Inversion vom Bild zum latenten Code"}

<!--en:95218bad-->
[Die Wahrscheinlichkeitsfluss-ODE vorwärts integrieren]{.kicker}

<!--en:2b3bfc29-->
Die Wahrscheinlichkeitsfluss-ODE ist deterministisch und umkehrbar. Euler-Schritte aufwärts von $t = 0$ bis $1$ (`ddim_invert`) kodieren daher jedes echte Bild $\mathbf{x}_0$ in einen latenten Rauschcode $\mathbf{z} = \mathbf{x}_1$:

<!--en:35d96e43-->
::: {.slide title="Rekonstruktionsfehler nach Inversion und Norm der Codes"}

<!--en:27f9d156-->
[Warum invertierte Codes weniger streuen als N(0, I)]{.kicker}

<!--en:b6a4250b-->
- **Lokal verschiedene Steigungen:** Der Euler-Schritt vorwärts ($t \to t'$) wertet $\hat{\boldsymbol\epsilon}$ am unteren Ende $t$ aus, der Euler-Schritt rückwärts ($t' \to t$) dagegen $\hat{\boldsymbol\epsilon}$ am oberen Ende $t'$.

<!--en:b4cb1a1f-->
- **Diskretisierungsfehler bei glattem Feld:** Mit wachsender Schrittzahl $10 \to 50 \to 200$ sinkt der mittlere quadratische Rekonstruktionsfehler um mehr als den Faktor $100\times$ ($\approx 0.035 \to \approx 0.003 \to 0.0002$), sowohl bei unserem U-Net als auch beim exakten 2D-Orakel.

<!--en:54f5ad84-->
- **Die Norm der Codes schrumpft (`std` $\approx 0.7$ bis $0.85 < 1$):** Auf gekrümmten Pfaden verstärken die Euler-Schritte in Vorwärtsrichtung die Varianz zu wenig.

<!--en:05f31c43-->
::: {.slide title="Sphärische Interpolation (Slerp) im latenten Raum"}

<!--en:919a23ac-->
[Übergang zwischen zwei echten Bildern]{.kicker}

<!--en:c3acfe97-->
Warum ist eine lineare Interpolation zwischen zwei durch Inversion gewonnenen Rauschcodes ungeeignet ($(1 - u)\mathbf{z}_0 + u\mathbf{z}_1$), wenn $\mathbf{z}_0, \mathbf{z}_1 \in \mathbb{R}^{784}$ gilt?

<!--en:50f414de-->
- **In hohen Dimensionen kann die Norm stark schrumpfen:** Zwei unabhängige gaußverteilte Vektoren in $d = 784$ Dimensionen sind nahezu orthogonal ($\omega \approx \pi/2$) und haben die Norm $r \approx \sqrt{d}$. Ihr linearer Mittelpunkt $\frac{1}{2}(\mathbf{z}_0 + \mathbf{z}_1)$ hat die Norm $r / \sqrt{2} \approx 0.71\,r$. Er liegt damit ganz außerhalb der Kugelschale der Verteilung $\mathcal{N}(\mathbf{0}, \mathbf{I})$, auf der der Entrauscher trainiert wurde.

<!--en:583f37ca-->
- **Sphärische lineare Interpolation (Slerp):** Sie folgt dem Großkreis mit konstantem Radius $r$ zwischen $\mathbf{z}_0$ und $\mathbf{z}_1$. Für den eingeschlossenen Winkel gilt $\cos\omega = \frac{\mathbf{z}_0^\top \mathbf{z}_1}{\|\mathbf{z}_0\|\|\mathbf{z}_1\|}$:
  $$\mathbf{z}_u = \frac{\sin\big((1 - u)\omega\big)\,\mathbf{z}_0 + \sin(u\omega)\,\mathbf{z}_1}{\sin\omega}, \qquad u \in [0, 1]$$

<!--en:a65a007d-->
::: {.slide title="Bildbearbeitung durch partielles Verrauschen (SDEdit)" layout="tight"}

<!--en:817483fd-->
[Vorlagentreue und Realismus der Samples]{.kicker}

<!--en:b5bc8d8d-->
Gegeben sei eine grobe Skizze oder eine verschwommene Bildmontage $\mathbf{y}$ (hier die 10 unscharfen Silhouetten der Klassenmittelwerte, $\mathrm{FD} \approx 87$). **SDEdit** (Meng et al., 2022) verrauscht $\mathbf{y}$ bis zu einem Zwischenzeitpunkt $t_0 \in (0, 1]$ und integriert von $t_0$ bis $0$ zurück:

<!--en:cc6d139e-->
- **Kleines $t_0 = 0.3$ (hohe Vorlagentreue, geringer Realismus):** Der mittlere quadratische Pixelfehler gegenüber der Eingabe bleibt klein ($\approx 0.02$), doch das schwache Rauschen beseitigt die unnatürliche Unschärfe nicht ($\mathrm{FD} \approx 30$).

<!--en:e0d7ade2-->
- **Günstiger Kompromiss bei $t_0 = 0.7$:** Das Rauschen beseitigt Unschärfe in den hohen Frequenzen, während $\alpha_{0.7} = 0.3$ die grobe Haltung und Silhouette erhält. So entstehen scharfe Texturen ($\mathrm{FD} \approx 4$).

<!--en:6a14dc9b-->
- **Reines Rauschen bei $t_0 = 1.0$:** Die Eingabe $\mathbf{y}$ geht vollständig verloren ($\mathrm{FD}$ unter $2$).

<!--en:cf2dca5a-->
::: {.slide title="Bildbearbeitung durch partielles Verrauschen (SDEdit, Fortsetzung)"}

<!--en:817483fd-2-->
[Vorlagentreue und Realismus der Samples]{.kicker}

<!--en:7e58945b-->
Mit wachsendem $t_0$ steigt der Abstand zur Eingabe $\mathbf{y}$, während die FD fällt:

<!--en:53c32f85-->
::: {.slide title="Zero-Shot-Inpainting durch Ersetzen"}

<!--en:96b81cd3-->
[Bedingte Erzeugung bei teilweise bekannten Pixeln]{.kicker}

<!--en:0a69ec22-->
Gegeben seien bekannte Pixel $\mathbf{y}$ auf einer binären Maske $\mathbf{m} \in \{0, 1\}^{28 \times 28}$ (hier die obere Bildhälfte). Beim **Inpainting durch Ersetzen** (Song et al., 2021) folgen auf reguläre DDIM-Schritte Ersetzungen. Auf jedem Pegel $s$ überschreibt eine frisch verrauschte Kopie von $\mathbf{y}$ den maskierten Bereich:

<!--en:45c76bff-->
- **Kein erneutes Training nötig:** Unterschiedliches Ausgangsrauschen liefert vielfältige Ergänzungen für die untere Hälfte.

<!--en:42d2eda1-->
- **Sichtbare Nahtstelle:** Der Schritt ohne Maske und das Überschreiben sind getrennt. Dadurch können die Texturen an der Naht in Zeile 14 voneinander abweichen (RePaint behebt dies durch wiederholte Vorwärts- und Rückwärtsschritte, Lugmayr et al., 2022).

<!--en:567c26bf-->
::: {.slide title="Verständnisfragen zu DDIM, Heun und der Geometrie von Slerp"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:6b539ce5-->
Fragen zu schnellen Diffusionslösern und zur deterministischen Inversion:

<!--en:8ef7ae23-->
1. **Anteil des ersetzten Rauschens bei $\eta = 1$:** Auf einem VP-Gitter mit 10 Schritten fällt der Pegel im ersten Schritt von $\bar\sigma_t = 152$ auf $\bar\sigma_s = 59$. Es gilt $\mathrm{SNR} = 1/\bar\sigma^2$. Welchen Anteil $f = 1 - \mathrm{SNR}_t/\mathrm{SNR}_s$ der Rauschvarianz auf dem Pegel $s$ ersetzt frisches Rauschen $\boldsymbol\xi$, wenn $\eta = 1$ gilt?

<!--en:5ccbe724-->
2. **Warum erzeugt Ancestral Sampling bei 50 Schritten körnige Bilder?** Warum liefert die ancestrale Kette mit 50 Schritten körnige Bilder ($\mathrm{FD} \approx 8$ bis $11$), DDIM mit 50 Schritten ($\eta = 0$) dagegen scharfe ($\mathrm{FD} \approx 1.7$ bis $1.8$), obwohl beide bei $1{,}000$ Schritten gleichauf liegen?

<!--en:7b1b1db8-->
3. **Wann rekonstruiert DDIM nach der Inversion exakt?** Für welche Datenverteilung ist `ddim_sample(ddim_invert(x0))` mit exaktem Geschwindigkeitsorakel schon nach $1$ Schritt exakt? Warum scheitert die lineare Interpolation zwischen zwei Codes $\mathbf{z}_0, \mathbf{z}_1$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:bf7b2738-->
1. Es gilt $\kappa^2 = \mathrm{SNR}_t / \mathrm{SNR}_s = (\bar\sigma_s / \bar\sigma_t)^2 = (59 / 152)^2 \approx 0.151$. Frisches Rauschen ersetzt also $f = 1 - \kappa^2 \approx \mathbf{84.9\%}$ der Rauschvarianz.

<!--en:e459bb30-->
2. Die ancestrale Kette fügt die Varianz des Vorwärtsschritts $\sigma_t^2(1 - \kappa^2) = (\sigma_t^2/\sigma_s^2)\tilde\sigma_{\eta=1}^2$ hinzu. Bei geringem Rauschen auf einem Gitter mit 50 Schritten gilt $\sigma_t^2/\sigma_s^2 \gg 1$. Dadurch kommt übermäßig viel Rauschen hinzu, obwohl das Kleidungsstück schon nahezu feststeht. Bei $1{,}000$ Schritten gilt $\sigma_t^2/\sigma_s^2 \approx 1$.

<!--en:b916dd2e-->
3. Die Inversion ist in 1 Schritt exakt, wenn $\hat{\boldsymbol\epsilon}$ entlang der Trajektorien konstant ist (ein einziger Datenpunkt). Für zwei orthogonale Codes mit Norm $r$ sinkt die Norm am linearen Mittelpunkt $\frac{1}{2}(\mathbf{z}_0 + \mathbf{z}_1)$ auf $r/\sqrt{2} \approx 0.71r$. Dieser liegt außerhalb der typischen Kugelschale der Standardnormalverteilung.

<!--en:077f6107-->
::: {.slide title="Zusammenfassung zu schnellen Samplern, Inversion und Bildbearbeitung"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:2749fff8-->
| Verfahren / Werkzeug | Aktualisierungsregel in $(\bar{\mathbf{x}}, \bar\sigma) = (\mathbf{x}/\alpha, \sigma/\alpha)$ | Ordnung und Kosten pro Schritt | Wichtigstes empirisches Ergebnis und Einsatz |

<!--en:1f42dc49-->
| **DDIM ($\eta = 0$)** | $\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\hat{\boldsymbol\epsilon}_t$ | 1. Ordnung ($O(h)$), 1 NFE | Entspricht genau Euler für die Wahrscheinlichkeitsfluss-ODE und erreicht das Plateau in $50$ NFE |

<!--en:b2f79ceb-->
| **Stochastisches DDIM ($\eta > 0$)** | $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sqrt{\sigma_s^2 - \tilde\sigma^2}\hat{\boldsymbol\epsilon} + \tilde\sigma\boldsymbol\xi$ | SDE-Verfahren 1. Ordnung, 1 NFE | $\eta = 1$ verwendet die A-posteriori-Varianz und schneidet schlechter ab als $\eta = 0$ bei $\le 200$ Schritten |

<!--en:dd654f99-->
| **Heun-Verfahren** | Mittelwert der Steigungen am Start- und am Probepunkt des Euler-Schritts | 2. Ordnung ($O(h^2)$), 2 NFE | Erreicht das asymptotische FD-Plateau auf dem linearen Gitter in **$20$ NFE** |

<!--en:ab614edd-->
| **Mehrschrittverfahren (AB2 / DPM++)** | $\bar{\mathbf{x}}_t + \Delta\big(\hat{\boldsymbol\epsilon}_t + \frac{\Delta}{2\Delta'}(\hat{\boldsymbol\epsilon}_t - \hat{\boldsymbol\epsilon}')\big)$ | 2. Ordnung ($O(h^2)$), 1 NFE | Verwendet die vorige Steigung erneut und ist mit CFG der beste Löser ($\gamma = 2$) |

<!--en:8689de0c-->
| **DDIM-Inversion und Slerp** | Euler vorwärts $t: 0 \to 1$, Großkreisbogen | Rekonstruktionsfehler $O(h)$ | Kodiert echte Bilder als Rauschcodes $\mathbf{z}$ und erhält bei der Interpolation $\|\mathbf{z}\|$ |

<!--en:4258547b-->
| **SDEdit und Inpainting** | Bei $t_0 \in (0, 1)$ starten oder im Bereich der Maske $\mathbf{m}$ überschreiben | Zero-Shot-Verfahren (ohne erneutes Training) | $t_0$ bestimmt den Kompromiss zwischen Vorlagentreue und Realismus. Überschreiben im Maskenbereich ermöglicht die Ergänzung fehlender Bereiche. |
