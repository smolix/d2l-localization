<!--en:0962c676-->
# Sampling durch Integration

<!--en:86042f8f-->
Ein Netz, das entlang eines Gauß-Pfads (engl. Gaussian path) trainiert wurde, sagt für jeden Rauschpegel $t \in [0, 1]$ die marginale Geschwindigkeit $\mathbf{u}_t(\mathbf{x})$ vorher (oder gleichwertig dazu den Score $\nabla \log p_t(\mathbf{x})$). Warum erhalten wir Samples aus $p_0$, wenn wir dieses deterministische Vektorfeld von $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ bei $t = 1$ rückwärts bis $t = 0$ integrieren? Und warum stimmen dabei sogar die Modusgewichte $0.5, 0.3, 0.2$ exakt, die die Langevin-Ketten bei festem Rauschpegel verfehlt haben?

<!--en:a7f344f8-->
Die Garantie liefert die *Kontinuitätsgleichung* (engl. continuity equation) für den Transport von Wahrscheinlichkeit. Jede bedingte gaußsche Punktwolke $p_t(\mathbf{x} \mid \mathbf{x}_0)$ erhält unter ihrer bedingten Geschwindigkeit die Masse. Mitteln wir über $p(\mathbf{x}_0)$, so sehen wir, dass der A-posteriori-Mittelwert der Geschwindigkeit $\mathbf{u}_t(\mathbf{x})$ die Randdichte $p_t$ transportiert. :citet:`song2021score` formulierten das deterministische Sampling über diese *Wahrscheinlichkeitsfluss-ODE* (engl. probability-flow ODE). :citet:`ho2020denoising` erzeugten Samples mit einer stochastischen rückwärts laufenden Markov-Kette. Wie wir sehen, zerfällt jeder ancestrale Schritt von DDPM in erster Ordnung in einen Euler-Schritt der Wahrscheinlichkeitsfluss-ODE und einen Langevin-Schritt beim aktuellen Rauschpegel. In diesem Abschnitt leiten wir beide Sampler her, schreiben die ODE in pfadunabhängigen Koordinaten $(\bar{\mathbf{x}}, \bar\sigma)$ und vergleichen die trainierten Netze an der Mischung mit drei Modi mit dem exakten Orakel.

<!--en:ee3537de-->
Jeder Abschnitt läuft als eigenes Notebook. Die folgende Zelle erstellt deshalb die Mischung mit drei Modi aus :numref:`sec_diffusion-gaussian-paths` erneut, zusammen mit zwei ihrer Pfade, dem varianzerhaltenden (engl. variance-preserving, VP) und dem linearen Pfad. Außerdem definiert sie für jeden Punkt und Rauschpegel die exakten A-posteriori-Mittelwerte $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ der Mischung. Wie im vorigen Abschnitt heißt die zugehörige Funktion im Code `oracle`. Anschließend trainiert die Zelle für jeden der beiden Sampler ein Netz neu. Für den Fluss trainiert sie ein Geschwindigkeitsmodell auf dem linearen Pfad, für die DDPM-Kette ein Rauschmodell auf dem VP-Pfad. Wir verwenden das Verfahren aus dem vorigen Abschnitt mit halb so vielen Schritten. Zwanzigtausend Schritte genügen. Das zeigt der mittlere Trainingsverlust über die letzten tausend Schritte, den die Zelle ausgibt. Als Vergleich dienen die Untergrenzen aus dem vorigen Abschnitt, etwa $4.9$ für die Geschwindigkeit und $0.54$ für das Rauschen. Beide Trainingsläufe zusammen dauern auf einer GPU etwa eine Minute.

<!--en:3322648f-->
Beide Verluste liegen an ihrer Untergrenze. Aus der Regression können die beiden Netze kaum noch etwas lernen. Wie nahe sie dem Orakel bei den Samples kommen, auf die es letztlich ankommt, untersuchen wir am Ende dieses Abschnitts.

<!--en:f4c9350a-->
## Transport von Wahrscheinlichkeitsmasse

<!--en:c111e835-->
Betrachten wir zunächst, wie ein Geschwindigkeitsfeld eine Dichte verändert. Nehmen wir eine Punktwolke mit der Verteilung $p_t$. Jeder Punkt bewegt sich kurze Zeit mit der Geschwindigkeit $\mathbf{u}_t(\mathbf{x})$ an seiner Position. Dabei entstehen und verschwinden keine Punkte. Die Dichte bei $\mathbf{x}$ kann sich also nur durch die einströmende Masse abzüglich der ausströmenden Masse ändern. Die Masse, die pro Flächen- und Zeiteinheit strömt, beschreibt das Produkt aus Dichte und Geschwindigkeit, $p_t \mathbf{u}_t$. Wie viel Masse netto pro Volumeneinheit aus einem kleinen Kasten um $\mathbf{x}$ strömt, gibt die Divergenz $\nabla \cdot (p_t \mathbf{u}_t) = \sum_i \partial (p_t u_{t,i}) / \partial x_i$ an. Genau mit dieser Rate nimmt dort die Dichte ab. Das ist die *Kontinuitätsgleichung* der Strömungsmechanik:

<!--en:43d1a915-->
Erfüllen eine Dichte und ein Feld diese Gleichung, so haben Punkte, die dem Feld folgen, zu jeder Zeit die Dichte des Pfads, vorwärts wie rückwärts. Erfüllt die marginale Geschwindigkeit aus :numref:`sec_diffusion-gaussian-paths` zusammen mit $p_t$ diese Gleichung?

<!--en:6d8a6d00-->
Betrachten wir zunächst einen einzelnen unverrauschten Punkt $\mathbf{x}_0$. Seine verrauschten Varianten bilden die gaußsche Punktwolke $\mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$. Jeder ihrer Punkte bewegt sich mit der bedingten Geschwindigkeit $\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon$. Dabei ist $\boldsymbol\epsilon = (\mathbf{x} - \alpha_t \mathbf{x}_0)/\sigma_t$ das Rauschen, das den Punkt an die Stelle $\mathbf{x}$ gebracht hat. Nennen wir dieses Feld $\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_0)$. Es transportiert diese eine Wolke korrekt. Behält ein Punkt sein Rauschen $\boldsymbol\epsilon$ bei, so liegt er zu jeder Zeit bei $\alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ und ist damit ein Sample aus der Wolke. Die Wolke und ihr Feld erfüllen also die Kontinuitätsgleichung. Die Randdichte $p_t$ ergibt sich durch Mittelung der Wolken über die Daten. Da die Gleichung in der Dichte linear ist, liefert das Mitteln der einzelnen Gleichungen eine Gleichung für $p_t$. Auf der rechten Seite steht der gemittelte Massenfluss. Nach dem Satz von Bayes gilt $p(\mathbf{x}_0)\, p_t(\mathbf{x} \mid \mathbf{x}_0) = p_t(\mathbf{x})\, p(\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x})$. Der Massenfluss beträgt also

<!--en:bc73525b-->
Der A-posteriori-Mittelwert im mittleren Ausdruck ist genau die Definition der marginalen Geschwindigkeit. Die Masse, die alle Wolken zusammen durch $\mathbf{x}$ transportieren, ergibt sich also aus der dortigen Dichte mal der marginalen Geschwindigkeit. Somit erfüllen $p_t$ und $\mathbf{u}_t$ die Kontinuitätsgleichung. Auf diesem Ergebnis beruhen die Sampler der folgenden Abschnitte. Den Beweis mit den nötigen Voraussetzungen liefert :numref:`sec_mdl-flow-matching-transport`.

<!--en:3115aed0-->
Man könnte einwenden, dass gemittelte Geschwindigkeiten keine gemittelten Trajektorien ergeben. Dieser Einwand ist berechtigt. Die bedingten Pfade (auf dem linearen Pfad sind es Geraden) verlaufen zu jeder Zeit durch jeden Punkt und kreuzen sich überall. Zwei Trajektorien desselben Feldes treffen dagegen nie zur selben Zeit aufeinander, denn an jedem Punkt gibt es zu einer gegebenen Zeit nur eine Geschwindigkeit. Ein Sample, das $\mathbf{u}_t$ folgt, bewegt sich auf keinem der bedingten Pfade. :numref:`fig_diffusion-crossing-paths` zeigt beide Kurvenarten für eine Mischung zweier eindimensionaler Gauß-Verteilungen. Der orange eingezeichnete Rauschpunkt gehört zu einem unverrauschten Punkt im rechten Modus, seine Trajektorie endet aber im linken. Unser Argument macht keine Aussage über einzelne Trajektorien. Es besagt, dass das Feld zu jeder Zeit die *Dichte* reproduziert. Mehr braucht ein Sampler nicht.

<!--en:90fccf45-->
![Bedingte Pfade und Trajektorien auf dem linearen Pfad für eine Mischung zweier eindimensionaler Gauß-Verteilungen. Die Zeit läuft von den Daten bei $t = 0$ (unten) zum Rauschen bei $t = 1$ (oben). Die Schattierung gibt die Dichte $p_t$ an. (a) Für neun Paare aus einem unverrauschten Punkt $\mathbf{x}_0$ (Kreuz) und einem Rauschpunkt $\boldsymbol\epsilon$ (Punkt) sind die bedingten Pfade Geraden, die sich kreuzen. (b) Die Trajektorien des marginalen Feldes $\mathbf{u}_t$ treffen von denselben neun Rauschpunkten aus nie aufeinander. Vom orangefarbenen Rauschpunkt aus endet der bedingte Pfad im rechten Modus, die Trajektorie im linken.](../img/mdl-diffusion-crossing-paths.svg)

<!--en:d2dd39ce-->
Damit ist unsere Frage beantwortet. Bei exakter Lösung liefert der Fluss zwangsläufig die richtigen Modusgewichte, weil er die Masse in jedem Augenblick erhält. Die Hälfte der Rauschpunkte muss im Modus mit Gewicht ½ enden. Hätten die Langevin-Ketten die Datenverteilung einmal erreicht, hätten sie sie erhalten. Doch nichts trug sie über die Täler dorthin. Die Leiter mit abnehmendem Rauschpegel überwand die Täler auf den groben Stufen, auf denen das Rauschen sie auffüllt. Wie viele Stufen und Schritte dafür nötig waren, mussten wir aber abschätzen. Der Fluss beginnt bei einer bekannten Dichte, der Standardnormaldichte, und überführt sie garantiert in die Datenverteilung. Zwei Fehlerquellen bleiben. Unsere Schritte nähern die Lösung nur an, und unser Netz nähert das Feld nur an. Im Rest des Abschnitts messen wir beide Fehler.

<!--en:86fa4127-->
Der Sampler zieht also $\mathbf{x}$ aus $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ und löst die gewöhnliche Differentialgleichung

<!--en:62728969-->
Zu jeder Zeit ist ihre Lösung ein Sample aus $p_t$ und bei $t = 0$ ein Sample aus der Datenverteilung. :citet:`song2021score` nannten sie die *Wahrscheinlichkeitsfluss-ODE* (:numref:`sec_mdl-probability-flow-ode`). Nach dem Start geschieht nichts Zufälliges mehr. Der gesamte Zufall eines Samples steckt in seinem Startpunkt.

<!--en:8090c07d-->
## Das Euler-Verfahren für die Wahrscheinlichkeitsfluss-ODE

<!--en:5d08d962-->
Eine ODE lässt sich numerisch mit vielen kleinen geradlinigen Schritten lösen. Das *Euler-Verfahren* (:numref:`sec_mdl-euler-runge-kutta`) unterteilt die Zeit von $1$ bis $0$ in $K$ gleich große Schritte. Bei jedem Schritt bewegt sich der Punkt entgegen der Geschwindigkeit, also gemäß $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_t(\mathbf{x})/K$ zu den Zeiten $t = 1, 1 - 1/K, \ldots, 1/K$. Wie viele Schritte braucht der Fluss? Für allgemeine Daten ist die Antwort schwer abzuleiten. Wie :citet:`Nakkiran.Bradley.Zhou.ea.2025` beginnen wir daher mit Daten, für die sie auf der Hand liegt, und gehen dann zu schwierigeren Fällen über.

<!--en:95c35aa1-->
### Von einem Punkt zum Datenmittelwert

<!--en:4e3536d0-->
Die Daten sollen aus einem einzigen Punkt $\mathbf{x}_0$ bestehen. Auf dem linearen Pfad ($\alpha_t = 1 - t$, $\sigma_t = t$) trägt ein Punkt $\mathbf{x}$ zur Zeit $t$ das Rauschen $\boldsymbol\epsilon = (\mathbf{x} - (1 - t)\mathbf{x}_0)/t$ und bewegt sich mit der Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0 = (\mathbf{x} - \mathbf{x}_0)/t$. Jede Trajektorie führt geradlinig mit konstantem Tempo nach $\mathbf{x}_0$. Ein einziger Euler-Schritt der Länge eins, $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_1(\mathbf{x}) = \mathbf{x}_0$, ist daher exakt. Bei zwei Datenpunkten ist das Feld der mit den A-posteriori-Wahrscheinlichkeiten gewichtete Mittelwert zweier solcher geradlinigen Felder. Die Trajektorien krümmen sich dort, wo die A-posteriori-Verteilung vom einen Punkt zum anderen wechselt (:numref:`subsec_mdl-fm-two-point`).

<!--en:9efebc2a-->
Jede Verteilung ist eine Mischung von Punktmassen. Ihre Trajektorien krümmen sich überall dort, wo die A-posteriori-Verteilung keine eindeutige Zuordnung ergibt. Bei $t = 1$ ist die Zuordnung völlig offen. Ein reiner Rauschpunkt verrät nichts über den unverrauschten Punkt, aus dem er hervorging. Die A-posteriori-Verteilung stimmt daher mit der A-priori-Verteilung überein, und es gilt

<!--en:bfa66923-->
Ein Euler-Schritt der Länge eins führt also jeden Startpunkt zum Datenmittelwert. Für unsere Mischung kennen wir das Ergebnis schon vor dem ersten Programmlauf. Es ist $0.5\,(-2.5, -1.5) + 0.3\,(2.5, -1.5) + 0.2\,(0, 2.5) = (-0.5, -0.7)$, ein Punkt zwischen den Modi, mehr als vier Standardabweichungen vom nächsten Modus entfernt. Was ändern weitere Schritte? Wir wenden das Euler-Verfahren mit einem, zehn und hundert Schritten auf das exakte Feld der Mischung auf dem linearen Pfad an, das `mix.velocity` liefert. Wir wählen fünf feste Startpunkte, den Ursprung und vier Punkte im Abstand $1.5$ vom Ursprung.

<!--en:29bfa2e1-->
Die Spalte für einen Schritt bestätigt die Vorhersage. Alle fünf Startpunkte landen bei $(-0.50, -0.70)$, wo keine Daten liegen. Nach zehn Schritten liegt jeder Endpunkt bereits in einem Modus, höchstens etwa $0.2$ vom Endpunkt nach hundert Schritten entfernt und stets auf der dem Datenmittelwert zugewandten Seite. Bei groben Schritten bleibt also ein Teil des Fehlers erhalten, den ein einziger Schritt macht. Betrachten wir nun, *welchen* Modus die einzelnen Startpunkte erreichen. Vom Ursprung sowie von den Punkten darunter und links davon führen die Trajektorien in den Modus mit dem größten Gewicht. Der rechte Startpunkt landet im rechten Modus, der obere im oberen. Hier zeigt sich die Massenerhaltung. Die Hälfte der Masse der Standardnormalverteilung muss im Modus mit Gewicht ein halb enden. Damit hat dieser Modus das größte Einzugsgebiet, zu dem hier auch das Zentrum der Normalverteilung gehört. Die Endpunkte fallen auch nicht mit den Moduszentren zusammen. Vom Ursprung aus endet die Trajektorie bei $(-1.98, -1.30)$, etwa eine Standardabweichung des Modus von seinem Zentrum entfernt. Der Fluss bildet Rauschen auf Daten ab, und jeder Startpunkt liefert sein eigenes Sample.

<!--en:b4103b81-->
### Koordinaten für jeden Pfad

<!--en:8f2bb8b1-->
Euler-Schritte in $t$ hängen vom Pfad ab. Auf dem VP-Pfad beträgt der Rauschpegel noch $\sigma_t = 0.96$ bei $t = 0.5$ und ist bereits auf $0.32$ gefallen, wenn $t = 0.1$ erreicht ist (:numref:`sec_diffusion-gaussian-paths`). In der ersten Hälfte der Zeit verschwindet also fast kein Rauschen, in den nächsten zwei Fünfteln der Zeit dagegen zwei Drittel davon. Ein Schritt in $t$ verwendet durchgehend die Änderungsrate am Startpunkt. Auf einer solchen Kurve verfehlt ein langer Schritt sein Ziel deutlich (Übung :exref:`ex_integration-coordinates`). Der Sampler der Bibliothek arbeitet deshalb in Koordinaten, in denen jeder Gauß-Pfad gleich aussieht. Teilen wir den Pfad durch $\alpha_t$:

<!--en:d911a327-->
In diesen Koordinaten behalten die Daten ihren Maßstab. Nur das Rauschen nimmt zu, von $\bar\sigma = 0$ bei den Daten bis $\bar\sigma = \infty$ bei reinem Rauschen. Jeder Pfad wird zum varianzexplodierenden (engl. variance-exploding, VE) Pfad aus :numref:`sec_diffusion-gaussian-paths`, den wir schon auf der Leiter mit abnehmendem Rauschpegel hinabgestiegen sind. Sein Rauschpegel ist $\bar\sigma$. :citet:`Karras.Aittala.Aila.ea.2022` erzeugen auf diese Weise Samples für alle Pfade. Der Querstrich bezeichnet hier die Division durch $\alpha_t$. Bei DDPM bezeichnet er dagegen das Quadrat in $\bar\alpha_t = \alpha_t^2$ (:numref:`tab_diffusion-notation`). Auf einem bedingten Pfad bewegt sich $\bar{\mathbf{x}}$ als Funktion von $\bar\sigma$ auf einer Geraden mit Steigung $\boldsymbol\epsilon$. Das marginale Feld ist der A-posteriori-Mittelwert dieser Steigungen, also die Rauschvorhersage:

<!--en:c97c14fa-->
Bilden wir in der ersten Gleichung den A-posteriori-Mittelwert, so erhalten wir $\bar{\mathbf{x}}_t = \hat{\mathbf{x}}_0 + \bar\sigma_t \hat{\boldsymbol\epsilon}$. Ein Euler-Schritt von $\bar\sigma_t$ hinunter zu $\bar\sigma_s$ ergibt also $\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\hat{\boldsymbol\epsilon} = \hat{\mathbf{x}}_0 + \bar\sigma_s \hat{\boldsymbol\epsilon}$. Multiplikation mit $\alpha_s$ liefert

<!--en:15fd5d90-->
Ein Schritt sagt also den unverrauschten Punkt und das Rauschen vorher und setzt beide auf dem nächsten Pegel zusammen. In :numref:`sec_diffusion-samplers` begegnet uns diese Aktualisierung unter dem Namen DDIM wieder. Auf dem linearen Pfad stimmt sie mit dem Euler-Schritt in $t$ überein, wie das Einsetzen von $\mathbf{x}_t = (1 - t)\hat{\mathbf{x}}_0 + t\hat{\boldsymbol\epsilon}$ in $\mathbf{x}_t - (t - s)(\hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0)$ zeigt. Auf anderen Pfaden ergibt sich ein anderes Verfahren. Bestehen die Daten aus einem einzigen Punkt, ist es bei jeder Schrittweite exakt. Der Euler-Schritt in $t$ auf dem oben betrachteten VP-Intervall liegt dagegen weit daneben (Übung :exref:`ex_integration-coordinates`).

<!--en:df5a835b-->
`ode_sample` geht mit diesen Schritten eine Liste von Rauschpegeln $\bar\sigma$ durch. Die Funktion `sigma_bars` berechnet sie aus dem Zeitgitter des Pfads in $t$. Bei reinem Rauschen ist $\bar\sigma$ unendlich und $\bar{\mathbf{x}}$ undefiniert. Der Schritt benötigt jedoch nur $\hat{\mathbf{x}}_0$ und $\hat{\boldsymbol\epsilon}$. Diese Größen berechnet der Prädiktor aus dem Rauschen selbst, sodass der erste Schritt bei $\mathbf{x}$ beginnt.

<!--en:e100c2ed-->
Ein Sampler benötigt $\hat{\mathbf{x}}_0$ und $\hat{\boldsymbol\epsilon}$ für jedes Paar $(\alpha, \sigma)$. Die Funktion `make_pred` macht aus einem beliebigen trainierten Netz einen solchen Prädiktor. In diesem Abschnitt arbeitet jedes Netz auf seinem eigenen Pfad. Daher muss `make_pred` seine Vorhersage nur in das gesuchte Paar umrechnen. In späteren Abschnitten verwenden wir ein Netz auf dem Gitter eines anderen Pfads. Dann bestimmt `make_pred` zunächst durch Bisektion in `time_at` die Zeit auf dem eigenen Pfad des Netzes, die dieselbe Log-SNR hat, und skaliert den Punkt auf diesen Pfad um. Ein Rauschmodell hat einen blinden Fleck. Bei reinem Rauschen, $\alpha = 0$, ist der unverrauschte Punkt $\hat{\mathbf{x}}_0 = (\mathbf{x} - \sigma\hat{\boldsymbol\epsilon})/\alpha$ undefiniert. Deshalb kann ein Rauschmodell auf dem linearen Pfad keinen Sampler starten. Auf dem VP-Pfad bleibt $\alpha$ positiv und beträgt etwa $0.007$ bei $t = 1$. Dort setzen wir das Rauschmodell ein.

<!--en:fcd347d3-->
### Das gelernte Feld

<!--en:7d579c3a-->
Verhält sich das trainierte Geschwindigkeitsnetz wie das exakte Feld? Erreicht es nach einem Schritt den Datenmittelwert und nach zehn Schritten die Modi? Wir ziehen $2{,}000$ Startpunkte aus der Standardnormalverteilung und führen mit `ode_sample` einen, zehn und hundert Schritte aus. Zwei weitere Zeilen erleichtern den Vergleich. Eine zeigt $2{,}000$ Samples direkt aus den Daten (die grauen Punkte im Diagramm), die andere das Orakel mit denselben zehn Schritten von denselben Startpunkten aus. Zuletzt vergleicht die Zelle `ode_sample` mit dem Orakel und die einfache Euler-Schleife mit `mix.velocity`. Auf dem linearen Pfad machen beide dieselben Schritte. Ihre Ergebnisse sollten also bis auf Rundungsfehler übereinstimmen.

<!--en:19ba1928-->
Ein einziger Schritt führt alle Startpunkte zum Datenmittelwert $\approx (-0.5, -0.7)$, bis auf einen kleinen Approximationsfehler des Netzes von wenigen Hundertsteln bei $t = 1$. Zehn Euler-Schritte trennen die drei Modi mit Gewichten nahe $0.5, 0.3, 0.2$. Allerdings liegen $\approx 3\%$ der Samples fernab der Modi (gegenüber $1.1\%$–$1.5\%$ bei den echten Daten), und die Streuung innerhalb der Modi fällt etwas kleiner aus ($0.45$ gegenüber $0.48$–$0.49$). Mit 10 Euler-Schritten liefert das exakte Orakel `oracle` dieselben Modusgewichte (bis auf $0.003$) und eine Streuung von $0.46$. Das exakte Feld verengt die Modi fast ebenso stark wie das Netz. Die Verengung geht also auf das Lösungsverfahren zurück. Zum Anteil modusferner Samples tragen dagegen zwei Fehlerquellen bei. Beim Orakel liegt er mit $2\%$–$3\%$ höher als bei den Daten, aber niedriger als beim Netz. Hier wirken sich also sowohl das Lösungsverfahren als auch das Netz aus. Mit 100 Euler-Schritten trifft das trainierte Netz den Anteil modusferner Samples ($\approx 1.3\%$), die Streuung und die Modusgewichte der echten Daten innerhalb des Monte-Carlo-Fehlers. Mit dem exakten Orakel stimmt `ode_sample` bis auf Rundungsfehler der Gleitkommarechnung mit einfacher Genauigkeit ($10^{-4}$) mit der expliziten Euler-Schleife überein.

<!--en:0dd21ccd-->
## Die ancestrale Kette und die zeitumgekehrte SDE

<!--en:134342a1-->
DDPM erzeugt Samples auf andere Weise. Es durchläuft die Rückwärtskette aus :numref:`fig_diffusion-forward-reverse` von $t = 1$ bis $t = 0$ und fügt in jedem Schritt Rauschen hinzu. Warum funktioniert eine Kette verrauschter Schritte überhaupt, und wie hängt sie mit dem Fluss zusammen?

<!--en:1292cc6b-->
### Der ancestrale Schritt

<!--en:6cde4c64-->
Die Umkehrung eines kleinen Vorwärtsschritts ist annähernd gaußverteilt :cite:`sohl2015deep`. Ist der unverrauschte Punkt gegeben, so ist sie exakt gaußverteilt. Für $s < t$ ist $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ nach dem Satz von Bayes proportional zum Produkt zweier Gauß-Funktionen in $\mathbf{x}_s$. Die Faktoren sind der Vorwärtsschritt $q(\mathbf{x}_t \mid \mathbf{x}_s)$ und die Verrauschung $q(\mathbf{x}_s \mid \mathbf{x}_0)$. Damit ist die Verteilung gaußsch und ihr Mittelwert linear in $\mathbf{x}_t$ und $\mathbf{x}_0$. Im Schritt von Ho et al. steht in diesem Mittelwert $\hat{\mathbf{x}}_0$ anstelle von $\mathbf{x}_0$. Hinzu kommt neues Rauschen mit derselben Varianz $\beta = 1 - \alpha_t^2/\alpha_s^2$, die der Vorwärtsschritt von $s$ nach $t$ hinzufügt. Der Schritt heißt *ancestral*, weil sich jedes $\mathbf{x}_s$ bedingt auf seinen Elternknoten $\mathbf{x}_t$ ziehen lässt, wie beim Ancestral Sampling in einem grafischen Modell. Setzen wir auf dem VP-Pfad $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t \hat{\boldsymbol\epsilon})/\alpha_t$ und den Score $\nabla \log p_t = -\hat{\boldsymbol\epsilon}/\sigma_t$ in den Mittelwert ein und vereinfachen den Ausdruck, so lautet der Schritt

<!--en:24bff9d3-->
Ist das ein neuer Sampler oder der bekannte Sampler mit zusätzlichem Rauschen? Um das zu sehen, verwenden wir die Entwicklung $1/\sqrt{1 - \beta} \approx 1 + \beta/2$ und behalten die Terme erster Ordnung in $\beta$ sowie das Rauschen $\sqrt\beta\, \mathbf{z}$ bei. Der Schritt zerfällt dann in zwei Teile:

<!--en:5444b49e-->
Die Klammer entspricht dem Langevin-Schritt $\mathbf{x} \leftarrow \mathbf{x} + \frac{h}{2} \nabla \log p + \sqrt{h}\, \boldsymbol\xi$ aus :numref:`sec_diffusion-denoising` mit Schrittweite $h = \beta$ für die Dichte $p_t$. Er erhält $p_t$ und zieht abgeirrte Punkte zu dieser Verteilung zurück. Der zweite Term entspricht einem Euler-Schritt der Wahrscheinlichkeitsfluss-ODE von $t$ zurück nach $s$. Auf dem VP-Pfad beträgt die marginale Geschwindigkeit $\mathbf{u}_t = -\tfrac12 \beta(t)\, (\mathbf{x} + \nabla \log p_t)$. Das folgt aus der Identität für Score und Geschwindigkeit aus :numref:`sec_diffusion-gaussian-paths` mit $\dot\alpha_t/\alpha_t = -\beta(t)/2$ und $\alpha_t^2 + \sigma_t^2 = 1$. Hier ist $\beta(t) = 0.1 + 19.9\, t$ die Rate des VP-Zeitplans. Für einen kurzen Schritt gilt $\beta \approx \beta(t)\, (t - s)$. Der Euler-Schritt von $t$ zurück nach $s$ verschiebt den Punkt also um $-(t - s)\, \mathbf{u}_t = \tfrac12 \beta(t)\, (t - s)\, (\mathbf{x}_t + \nabla \log p_t) \approx \tfrac{\beta}{2} (\mathbf{x}_t + \nabla \log p_t)$. Das ist der zweite Term. Ein ancestraler Schritt ist somit ein Flussschritt plus eine Langevin-Korrektur beim aktuellen Rauschpegel. Die Kette transportiert die Punkte wie die ODE. Ihr Langevin-Anteil zieht die Punkte zusätzlich zu $p_t$ zurück und kann dadurch einen früheren Fehler teilweise ausgleichen. Ob das in der Praxis gelingt, messen wir in :numref:`sec_diffusion-samplers`. Dort zeigt :numref:`fig_diffusion-two-samplers` deterministische und verrauschte Trajektorien von denselben Startpunkten aus.

<!--en:12dbc865-->
### Die zeitumgekehrte SDE

<!--en:46c430fc-->
In kontinuierlicher Zeit wird die Umkehrung exakt. Der Code in diesem Abschnitt benötigt diesen Grenzwert nicht. Da wissenschaftliche Arbeiten ihre Sampler jedoch in dieser Form angeben, lohnt sich ein Blick auf die Schreibweise. Eine *stochastische Differentialgleichung* (engl. stochastic differential equation, SDE) $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\, dt + g(t)\, d\mathbf{W}$ steht kurz für den Grenzwert der folgenden Schritte für $\Delta t \to 0$:

<!--en:65baf2b6-->
Jeder Schritt enthält eine deterministische *Drift* $\mathbf{f}$ und gaußsches Rauschen, dessen Varianz proportional zum Zeitschritt wächst. Die Vorwärtskette von DDPM besteht aus solchen Schritten mit $\Delta t = 1/T$, $\mathbf{f} = -\tfrac12 \beta(t)\, \mathbf{x}$ und $g^2 = \beta(t)$, denn es gilt $\sqrt{1 - \beta_k} \approx 1 - \beta_k/2$ und $\beta_k \approx \beta(t)\, \Delta t$. :citet:`Anderson.1982` zeigte, wie sich ein solcher Prozess von $t = 1$ bis $t = 0$ umkehren lässt. Die Schritte $\mathbf{x} \leftarrow \mathbf{x} - [\mathbf{f} - g^2 \nabla \log p_t(\mathbf{x})]\, \Delta t + g \sqrt{\Delta t}\, \mathbf{z}$ haben im Grenzwert $\Delta t \to 0$ dieselben Randdichten $p_t$ wie der Vorwärtsprozess. In der Literatur schreibt man die Umkehrung als SDE, in der die Zeit rückwärts läuft:

<!--en:82da04d9-->
Dabei steht $d\bar{\mathbf{W}}$ für das neue Rauschen jedes Rückwärtsschritts. Warum haben beide Gleichungen dieselben Randdichten? Ihre Rückwärtsschritte unterscheiden sich um $\tfrac12 g^2 \nabla \log p_t\, \Delta t + g \sqrt{\Delta t}\, \mathbf{z}$. Das ist ein Langevin-Schritt der Schrittweite $h = g^2 \Delta t$ für $p_t$, und ein Langevin-Schritt erhält $p_t$. Damit haben wir die Zerlegung des ancestralen Schritts in kontinuierlicher Zeit. Auf dem VP-Pfad lautet die ODE rechts $\mathbf{u}_t = -\tfrac12 \beta(t)\, (\mathbf{x} + \nabla \log p_t)$. Das ist die Wahrscheinlichkeitsfluss-ODE :eqref:`eq_diffusion-flow-ode` dieses Pfads. Die Rückwärtsschritte ergeben $\mathbf{x} + \tfrac12 \beta(t) \Delta t\, \mathbf{x} + \beta(t) \Delta t\, \nabla \log p_t + \sqrt{\beta(t) \Delta t}\, \mathbf{z}$, also die Entwicklung des ancestralen Schritts mit $\beta = \beta(t)\, \Delta t$. Die ancestrale Kette entspricht damit in erster Ordnung der einfachsten Diskretisierung der zeitumgekehrten SDE, dem Euler-Maruyama-Verfahren (:numref:`sec_mdl-euler-maruyama`). Zwischen der ODE ohne zusätzliches Rauschen und der SDE liegt eine ganze Familie von Samplern mit denselben Randdichten. Je mehr Rauschen ein Sampler hinzufügt, desto stärker wirkt der Score-Term seiner Drift der zusätzlichen Streuung entgegen (:numref:`sec_mdl-time-reversal`, :numref:`fig_mdl-dyn-lambda-family`).

<!--en:7cf3032f-->
`ancestral_sample` führt auf jedem Pfad mit jedem Prädiktor `pred(x, alpha, sigma) -> (x0, eps)` den exakten Schritt aus, also die Formel vor der Entwicklung erster Ordnung. Die Funktion geht zum Mittelwert von $\mathbf{x}_s$ unter der Bedingung $\mathbf{x}_t$ und $\hat{\mathbf{x}}_0$ und addiert Rauschen mit der Varianz $\sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ des Vorwärtsschritts von $s$ nach $t$. Auf dem VP-Pfad beträgt diese Varianz $\beta$.

<!--en:77c5d3dd-->
Wir führen mit dem Rauschmodell zweihundert Schritte der Kette aus und beginnen bei denselben $2{,}000$ Startpunkten wie beim Fluss. Ho et al. verwendeten für Bilder tausend Schritte. In der Ebene reichen zweihundert gut aus.

<!--en:a12f50ed-->
Die Kette reproduziert die drei Modi mit den richtigen Gewichten, bis auf den Stichprobenfehler bei $2{,}000$ Punkten. Der Anteil modusferner Samples liegt bei etwa einem Prozent oder darunter, ähnlich wie beim Fluss mit hundert Schritten.

<!--en:05668a05-->
## Samples beider Sampler

<!--en:33d87545-->
Bisher lief jeder Sampler einmal mit $2{,}000$ Punkten. Das sind zu wenige, um einen Fehler des Netzes von einer zufälligen Schwankung der Stichprobe zu unterscheiden. Vergleichen wir nun das Netz und das Orakel unter identischen Bedingungen. Von $20{,}000$ Startpunkten aus machen wir zweihundert ancestrale Schritte mit dem Rauschmodell und hundert Euler-Schritte mit dem Geschwindigkeitsmodell. Jeden Lauf wiederholen wir mit `oracle` anstelle des Netzes, mit denselben Startpunkten und bei der Kette auch mit demselben zusätzlichen Rauschen. Jeder Unterschied zwischen der Zeile eines Netzes und der zugehörigen Orakelzeile geht dann auf das Netz zurück. Bei diesem Stichprobenumfang hat ein Modusgewicht einen Standardfehler von etwa $0.0035$. Die Zelle zählt auch die Startpunkte, die mit Netz und Orakel in unterschiedlichen Modi enden.

<!--en:294cc471-->
Bei beiden Samplern führen das trainierte Netz und das exakte Orakel nur $0.2\%$ bis $0.5\%$ der 20 000 Startpunkte in verschiedene Modi. Jedes Modusgewicht eines Netzlaufs weicht um höchstens $0.004$ vom zugehörigen Orakellauf ab. Das entspricht der Größenordnung des binomialen Standardfehlers von $\approx 0.0035$. Beide Sampler treffen auch die Streuung der echten Daten (bis auf $0.01$) und den Anteil modusferner Samples ($\approx 1.1\%$, die gaußsche Restwahrscheinlichkeit jenseits von $3s$).

<!--en:04b63e24-->
Mit 100–200 Schritten reproduzieren die deterministische ODE und die stochastische ancestrale Kette die Mischverteilung mit praktisch gleicher Genauigkeit. Sie unterscheiden sich jedoch in ihren Eigenschaften. Die Wahrscheinlichkeitsfluss-ODE definiert eine deterministische Bijektion zwischen Rauschen $\mathbf{x}_1$ und Daten $\mathbf{x}_0$. Das ermöglicht die Interpolation im latenten Raum und die exakte Berechnung der Likelihood. Die ancestrale Kette fügt dagegen Langevin-Rauschen hinzu, das Diskretisierungs- oder Score-Fehler entlang der Trajektorie dämpfen kann (:numref:`sec_diffusion-samplers`). :numref:`tab_diffusion_models` fasst zusammen, wie die wichtigsten generativen Modelle dieses Kapitels einen Pfad, eine Zielgröße und einen Sampler kombinieren.

<!--en:7f7788bc-->
:Die Modelle dieses Kapitels mit Zielgröße und Pfad für das Training des Netzes sowie dem Sampler zum Erzeugen von Daten.

<!--en:49e62904-->
| Modell | Zielgröße und Pfad | Sampler | Abschnitt |

<!--en:ba306d9b-->
| Score-Modell zur Entrauschung | Score $-\boldsymbol\epsilon/\sigma$ bei einem Rauschpegel | Langevin-Dynamik | :numref:`sec_diffusion-denoising` |

<!--en:10c9ea6d-->
| auf den Rauschpegel bedingtes Score-Netz | Score auf jeder Stufe einer Leiter entlang des VE-Pfads, gewichtet mit $\tilde\sigma^2$ | Annealed Langevin Dynamics | :numref:`sec_diffusion-gaussian-paths` |

<!--en:722fa829-->
| DDPM | $\boldsymbol\epsilon$ auf dem VP-Pfad | ancestrale Kette oder Wahrscheinlichkeitsfluss-ODE | :numref:`sec_diffusion-gaussian-paths` und dieser Abschnitt |

<!--en:b4f11a93-->
| Flow-Matching, Rectified Flow | $\boldsymbol\epsilon - \mathbf{x}_0$ auf dem linearen Pfad | Euler-Schritte für die Wahrscheinlichkeitsfluss-ODE | :numref:`sec_diffusion-gaussian-paths` und dieser Abschnitt |

<!--en:23f12ad9-->
| $\mathbf{v}$-Vorhersage | $\mathbf{v}$ auf dem Kosinuspfad | beide Verfahren möglich | :numref:`sec_diffusion-gaussian-paths`, :numref:`sec_diffusion-design-space` |

<!--en:7a36f1cc-->
| EDM | unverrauschter Punkt auf dem VE-Pfad, für jeden Rauschpegel umskaliert | Lösungsverfahren zweiter Ordnung für die Wahrscheinlichkeitsfluss-ODE | :numref:`sec_diffusion-samplers`, :numref:`sec_diffusion-design-space` |

<!--en:c622172e-->
In :numref:`sec_diffusion-images` gehen wir von zwei Dimensionen zu Bildern der Größe $28 \times 28$ über. Dort steht kein exaktes Orakel zur Verfügung, und jede Funktionsauswertung erfordert einen Vorwärtsdurchlauf durch ein konvolutionäres U-Net.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:5d8ea303-->
Die marginale Geschwindigkeit $\mathbf{u}_t(\mathbf{x}) = \mathbb{E}[\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon \mid \mathbf{x}_t = \mathbf{x}]$ erfüllt mit der Randdichte $p_t$ die Kontinuitätsgleichung $\partial_t p_t + \nabla \cdot (p_t \mathbf{u}_t) = 0$. Integrieren wir die Wahrscheinlichkeitsfluss-ODE $d\mathbf{x}/dt = \mathbf{u}_t(\mathbf{x})$ von $t = 1$ bis $t = 0$, so transportiert sie $\mathcal{N}(\mathbf{0}, \mathbf{I})$ nach $p_0$ und erhält dabei die Modusgewichte. Ein einziger Euler-Schritt von $t = 1$ bis $t = 0$ führt allerdings jeden Startpunkt zum Datenmittelwert $\mathbb{E}[\mathbf{x}_0]$. In den umskalierten Koordinaten $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$ und $\bar\sigma = \sigma_t/\alpha_t$ gilt für jeden Gauß-Pfad dieselbe ODE $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$. Ihr Euler-Schritt lautet $\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sigma_s \hat{\boldsymbol\epsilon}$. In erster Ordnung in der Schrittvarianz $\beta$ zerfällt der stochastische ancestrale Schritt von DDPM in diesen Euler-Schritt des Wahrscheinlichkeitsflusses und einen Langevin-Schritt für $p_t$. Das entspricht einer Euler-Maruyama-Diskretisierung von Andersons zeitumgekehrter SDE.

<!--en:2294243a-->
## Übungen

<!--en:0c014eac-->
1. **Ein Schritt in zwei Koordinatensystemen.**
    1. Zeigen Sie, dass auf dem linearen Pfad der Schritt $\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sigma_s \hat{\boldsymbol\epsilon}$ mit dem Euler-Schritt $\mathbf{x}_t - (t - s)\, \mathbf{u}_t(\mathbf{x}_t)$ in $t$ übereinstimmt.
    1. Die Daten bestehen aus dem einzigen Punkt $\mathbf{x}_0 = \mathbf{0}$. Wählen Sie einen beliebigen Gauß-Pfad. Zeigen Sie, dass $\hat{\boldsymbol\epsilon} = \mathbf{x}/\sigma_t$ gilt und dass die exakte Trajektorie durch $\mathbf{x}_t$ die Form $\mathbf{x}_s = (\sigma_s/\sigma_t)\, \mathbf{x}_t$ hat. Zeigen Sie, dass der Schritt aus dem ersten Aufgabenteil ihr für jedes $s < t$ exakt folgt.
    1. Auf dem VP-Pfad beträgt die marginale Geschwindigkeit dieser Daten $\mathbf{u}_t(\mathbf{x}) = (\dot\sigma_t/\sigma_t)\, \mathbf{x}$ mit $d\sigma_t^2/dt = \beta(t)\, \alpha_t^2$ und $\beta(t) = 0.1 + 19.9\, t$. Machen Sie einen Schritt von $t = 0.5$ mit $\alpha_t \approx 0.28$ und $\sigma_t \approx 0.96$ nach $s = 0.1$ mit $\sigma_s \approx 0.32$. Mit welchem Faktor multipliziert der Euler-Schritt in $t$ den Punkt $\mathbf{x}$, und welcher Faktor wäre richtig? Welche Koordinaten bieten sich demnach für einen Sampler an, der mit wenigen Schritten auskommen soll?
   :exlabel:`ex_integration-coordinates`

<!--en:ef6c7b02-->
1. **Der ancestrale Schritt in erster Ordnung für gaußverteilte Daten.** ● Die Daten folgen der Verteilung $\mathcal{N}(\mathbf{0}, v_0 \mathbf{I})$. Auf dem VP-Pfad gilt also $p_t = \mathcal{N}(\mathbf{0}, v_t \mathbf{I})$ mit $v_t = \alpha_t^2 v_0 + \sigma_t^2$.
    1. Zeigen Sie, dass ein ancestraler Schritt :eqref:`eq_diffusion-ancestral-score` von $t$ nach $s$ mit dem exakten Score und $\beta = 1 - \alpha_t^2/\alpha_s^2$ die Verteilung $p_t$ auf $\mathcal{N}\big(\mathbf{0}, (v_s + \delta)\mathbf{I}\big)$ mit $\delta = \beta^2 (1/v_t - 1)/(1 - \beta)$ abbildet.
    1. Für welche Daten ist der Schritt exakt? Ist die Varianz zu groß oder zu klein, wenn die Daten weniger streuen als die Standardnormalverteilung, etwa bei einer Komponente der Mischung mit $v_0 = 0.25$?
    1. Wie ändert sich der über die Kette aufgelaufene Fehler, wenn Sie das Gitter auf doppelt so viele Schritte verfeinern und $\beta$ pro Schritt ungefähr halbieren?
   :exlabel:`ex_gaussian-paths-ancestral-first-order`

<!--en:62048611-->
1. **Schrittweite und gaußsche Umkehrung.** ● Die ancestrale Kette verwendet gaußsche Übergänge. Für kleine Schritte ist das gerechtfertigt. Untersuchen Sie, wie klein die Schritte sein müssen. Bezeichnen Sie mit $\sigma_{t|s}^2 = \sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ die Varianz des Vorwärtsschritts von $s$ nach $t > s$.
    1. Zeigen Sie, dass die exakte Umkehrung die Mischung $q(\mathbf{x}_s \mid \mathbf{x}_t) = \mathbb{E}_{\mathbf{x}_0 \mid \mathbf{x}_t}\big[q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)\big]$ mit Kovarianz $\tau^2 \mathbf{I} + c^2\, \textrm{Cov}[\mathbf{x}_0 \mid \mathbf{x}_t]$ ist. Dabei ist $\tau^2 = \sigma_{t|s}^2 \sigma_s^2/\sigma_t^2$ die Varianz von $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ und $c = \alpha_s \sigma_{t|s}^2/\sigma_t^2$ der Koeffizient von $\mathbf{x}_0$ im zugehörigen Mittelwert.
    1. Die eindimensionalen Daten nehmen mit gleicher Wahrscheinlichkeit die Werte $\pm 1$ an, und es gelte $x_t = 0$. Zeigen Sie, dass die Umkehrung eine gleichgewichtete Mischung der Verteilungen $\mathcal{N}(\pm c, \tau^2)$ ist und genau dann zwei Modi hat, wenn $\zeta^2 = c^2/\tau^2$ größer als eins ist. Zeigen Sie außerdem, dass $\zeta^2 = \alpha_s^2 \sigma_{t|s}^2 / (\sigma_t^2 \sigma_s^2)$ gilt. Wie nimmt $\zeta$ ab, wenn Sie den Schritt $t - s$ verkleinern?
    1. Berechnen Sie $\zeta$ auf dem VP-Pfad für Schritte von $t = 0.5$ nach $s = 0.495$, $0.45$ und $0.3$ sowie für einen Schritt von $t = 0.3$ nach $s = 0$. Welche Umkehrungen haben zwei Modi? Was erzeugt im letzten Fall ein gaußscher Übergang mit optimalem Mittelwert bei $x_t = 0$, und was sollte er erzeugen?
   :exlabel:`ex_gaussian-paths-gaussian-reversal`

<!--en:b637d11e-->
1. [code] **Zeitpunkt der Festlegung auf einen Modus.** ● Wann steht auf dem Pfad fest, in welchem Modus ein Sample endet? :citet:`Biroli.Bonnaire.deBortoli.ea.2024` messen das durch Klonen. Starten Sie bei einem Punkt aus $p_t$, lassen Sie zwei Kopien der Rückwärtskette mit unabhängigem Rauschen laufen und halten Sie fest, ob beide im selben Modus enden.
    1. Ziehen Sie $2{,}000$ unverrauschte Punkte und verrauschen Sie diese auf dem VP-Pfad zu $\mathbf{x}_t$ für $t = 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1$. Führen Sie von jedem Punkt aus zwei Läufe von `ancestral_sample(pred_eps, x_t, vp, steps, t_start=t)` mit jeweils etwa $200\, t$ Schritten aus. Zeichnen Sie den Anteil $\varphi(t)$ der Paare, deren Endpunkten derselbe Mittelwert am nächsten liegt. Welchem Wert nähert sich $\varphi$ für $t \to 1$ und warum?
    1. Sei $\Lambda$ der größte Eigenwert der Kovarianzmatrix der Daten. Biroli et al. definieren den Zeitpunkt der Aufspaltung durch $\alpha_t^2 \Lambda = 1$. Dort überwiegt das Signal entlang der ersten Hauptrichtung nicht länger das Rauschen. Berechnen Sie diesen Zeitpunkt für die Mischung. Vergleichen Sie ihn mit Ihrer Kurve und mit der Leiter abnehmender Rauschpegel aus :numref:`sec_diffusion-gaussian-paths`, deren grobe Stufen die Modusgewichte bestimmten. Übung :exref:`ex_mdl-flow-matching-transport-late-decisions` aus :numref:`sec_mdl-flow-matching-transport` und Übung :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging` aus :numref:`sec_mdl-fokker-planck-probability-flow` untersuchen dieselbe Frage für zwei Modi in einer Dimension.
   :exlabel:`ex_gaussian-paths-commitment-time`

<!--en:f35e681c-->
1. **Gleiches Verhältnis, gleicher Fluss.** ● Die Koeffizienten zweier Gauß-Pfade seien durch $\alpha^B_t = c_t\, \alpha^A_t$ und $\sigma^B_t = c_t\, \sigma^A_t$ für eine glatte positive Funktion $c_t$ verknüpft. Damit haben beide Pfade zu jeder Zeit dieselbe Log-SNR.
    1. Zeigen Sie, dass $p^B_t(\mathbf{y}) = c_t^{-d}\, p^A_t(\mathbf{y}/c_t)$ gilt. Verwenden Sie die Identität für Score und Geschwindigkeit aus :numref:`sec_diffusion-gaussian-paths`, $\mathbf{u}_t(\mathbf{x}) = (\dot\alpha_t/\alpha_t)\, \mathbf{x} - \big(\sigma_t \dot\sigma_t - \sigma_t^2 \dot\alpha_t/\alpha_t\big) \nabla \log p_t(\mathbf{x})$, um $\mathbf{u}^B_t(\mathbf{y}) = (\dot c_t / c_t)\, \mathbf{y} + c_t\, \mathbf{u}^A_t(\mathbf{y}/c_t)$ zu zeigen. Folgern Sie, dass die Trajektorien der Wahrscheinlichkeitsfluss-ODE über $\mathbf{y}_t = c_t\, \mathbf{x}_t$ zusammenhängen.
    1. Sei $\tau(t)$ monoton wachsend. Zeigen Sie, dass das Feld $\dot\tau(t)\, \mathbf{u}_{\tau(t)}$ den Pfad $p_{\tau(t)}$ mit denselben Trajektorien wie $\mathbf{u}$ erzeugt.
    1. Bestimmen Sie die Zeittransformation $\tau(t)$, bei der der Kosinuspfad $(\cos\frac{\pi\tau}{2}, \sin\frac{\pi\tau}{2})$ zur Zeit $\tau(t)$ dieselbe Log-SNR hat wie der lineare Pfad zur Zeit $t$. Zeigen Sie, dass $c_t = 1/\sqrt{(1 - t)^2 + t^2}$ gilt. Wo auf dem Pfad ist die Umskalierung am größten?
    1. Wie verändert die Wahrscheinlichkeitsfluss-ODE des Kosinuspfads Daten mit der Verteilung $\mathcal{N}(\mathbf{0}, \mathbf{I})$ und warum? Wie verändert die ODE des linearen Pfads dieselben Daten?
   :exlabel:`ex_gaussian-paths-same-ratio-same-flow`

<!--en:709ed053-->
[Diskussion](https://d2l.discourse.group/t/36889)

<!--en:b17891ec-->
Integration in umgekehrter Zeitrichtung: ODEs und ancestrale Ketten<br>**Kontinuitätsgleichung · Wahrscheinlichkeitsfluss-ODE · Euler-Schritte und Datenmittelwert · pfadunabhängige Koordinaten · ancestrale Kette und zeitumgekehrte SDE**

<!--en:acdd0d69-->
::: {.slide title="Warum das Geschwindigkeitsfeld die Modusgewichte reproduziert"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:28a851ed-->
- **Grenze der Langevin-Dynamik bei festem Pegel:** Lokale Scores $\nabla\log p(\mathbf{x})$ halten $p$ stationär, sobald diese Verteilung erreicht ist. Über die $10\text{--}12$ Nat tiefen Täler zwischen den Modi fließt global keine Masse.

<!--en:e9eb191e-->
- **Stetiger Transport aus einer bekannten A-priori-Verteilung:** Rückwärtsintegration transportiert die bekannte A-priori-Verteilung $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ stetig entlang $p_t$ bis zu $p_0$.

<!--en:d51d7799-->
- **Erhaltung der Wahrscheinlichkeitsmasse:** Da keine Trajektorien entstehen oder verschwinden, endet von den gaußverteilten Startpunkten genau der Anteil $\pi_k$ im Modus $k$.

<!--en:98b92a54-->
::: {.slide title="Zwei trainierte Prädiktoren und das exakte Orakel"}

<!--en:5e7332f1-->
[Versuchsaufbau]{.kicker}

<!--en:29220287-->
Der Vergleich zweier trainierter `TimeMLP(256, 3)`-Modelle mit dem exakten Orakel `oracle` der Mischung trennt den Diskretisierungsfehler vom Approximationsfehler des neuronalen Netzes:

<!--en:78aeab7a-->
- **Geschwindigkeitsnetz auf dem linearen Pfad (`ema_v`):** Das Training mit der Zielgröße $\boldsymbol\epsilon - \mathbf{x}_0$ entlang $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$ erreicht die theoretische Untergrenze des Geschwindigkeitsverlusts ($\approx 4.91$).

<!--en:e7103092-->
- **Rauschnetz auf dem VP-Pfad (`ema_eps`):** Das Training mit der Zielgröße $\boldsymbol\epsilon$ entlang des varianzerhaltenden Zeitplans erreicht die theoretische Untergrenze des Rauschverlusts ($\approx 0.54$).

<!--en:578bde99-->
- **Exaktes A-posteriori-Orakel (`oracle`):** Berechnet die A-posteriori-Mittelwerte $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ der Mischung für jedes $(\mathbf{x}, \alpha_t, \sigma_t)$ in geschlossener Form.

<!--en:81bee7cd-->
::: {.slide title="Massenerhaltung und Kontinuitätsgleichung"}

<!--en:5ebecffe-->
[Transport von Wahrscheinlichkeit]{.kicker}

<!--en:18c20a3a-->
- **Massenflussdichte $\mathbf{J}_t = p_t(\mathbf{x})\mathbf{u}_t(\mathbf{x})$:** Gibt die Wahrscheinlichkeitsmasse an, die pro Zeiteinheit eine Einheitsfläche durchströmt. Ihre negative Divergenz ist die Zunahmerate der Dichte.

<!--en:8cd80059-->
- **Exakter Transport der Randdichte:** Erfüllen $(p_t, \mathbf{u}_t)$ die Kontinuitätsgleichung, dann haben Teilchen mit Startpunkten $\mathbf{x}_1 \sim p_1$ unter der Dynamik $d\mathbf{x}/dt = \mathbf{u}_t(\mathbf{x})$ genau die Randverteilung $\mathbf{x}_t \sim p_t$ für alle $t \in [0, 1]$.

<!--en:5a7e8b25-->
::: {.slide title="Beweis der Kontinuitätsgleichung für die marginale Geschwindigkeit"}

<!--en:bd547dcc-->
[Von bedingten Punktwolken zur Randdichte]{.kicker}

<!--en:5476d0df-->
- **Schritt 1: Bedingte Massenerhaltung:** Für jedes feste $\mathbf{x}_0$ erfüllt die gaußsche Punktwolke $p_t(\mathbf{x} \mid \mathbf{x}_0)$ die Gleichung $\partial_t p_t(\mathbf{x} \mid \mathbf{x}_0) = -\nabla \cdot \big(p_t(\mathbf{x} \mid \mathbf{x}_0)\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_0)\big)$.

<!--en:12a5ca55-->
- **Schritt 2: Mittelung mit dem Satz von Bayes:** Mithilfe von $p(\mathbf{x}_0)p_t(\mathbf{x} \mid \mathbf{x}_0) = p_t(\mathbf{x})p(\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x})$ lässt sich $p_t(\mathbf{x})$ aus dem Erwartungswert ausklammern. Aus der Linearität folgt $\partial_t p_t = -\nabla \cdot (p_t\mathbf{u}_t)$.

<!--en:c0ff42dd-->
::: {.slide title="Warum sich bedingte Geraden kreuzen und ODE-Trajektorien einander nie schneiden"}

<!--en:239731d1-->
[Geometrie der Wahrscheinlichkeitsfluss-ODE]{.kicker}

<!--en:d7717dce-->
- **Bedingte Geraden kreuzen sich überall:** Geradlinige Verbindungen $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol\epsilon$ aus verschiedenen Modi schneiden sich überall in $\mathbb{R}^d \times (0, 1)$.

<!--en:9da3e09a-->
- **Trajektorien der marginalen ODE schneiden sich nie:** $\mathbf{u}_t(\mathbf{x})$ weist jedem Paar $(\mathbf{x}, t)$ genau einen Geschwindigkeitsvektor zu. Nach dem Eindeutigkeitssatz von Picard-Lindelöf weichen die Trajektorien an Kreuzungen aus und ergeben eine glatte Bijektion $\mathcal{N}(\mathbf{0}, \mathbf{I}) \leftrightarrow p_0$.

<!--en:fc540905-->
::: {.slide title="Euler-Verfahren und Kollaps auf den Mittelwert in einem Schritt"}

<!--en:f8432f18-->
[Diskretisierung der Wahrscheinlichkeitsfluss-ODE]{.kicker}

<!--en:183fe896-->
- **Euler-Diskretisierung:** $\mathbf{x}_{t_{k-1}} = \mathbf{x}_{t_k} - \frac{1}{K}\mathbf{u}_{t_k}(\mathbf{x}_{t_k})$ in $K$ Schritten

<!--en:c60ce480-->
- **Warum $K = 1$ bei allgemeinen Daten zum Kollaps führt:** Bei $t = 1$ ist $\mathbf{x}_1 = \boldsymbol\epsilon$ unabhängig von $\mathbf{x}_0$. Der A-posteriori-Mittelwert ist also der globale Datenmittelwert $\mathbb{E}[\mathbf{x}_0]$, und ein einziger Euler-Schritt führt **jeden Startpunkt** nach $\mathbb{E}[\mathbf{x}_0] = (-0.50, -0.70)$.

<!--en:ab9a783f-->
::: {.slide title="Fünf Startpunkte nach 1, 10 und 100 Euler-Schritten"}

<!--en:c3906c83-->
[Trajektorien des exakten Geschwindigkeitsfelds]{.kicker}

<!--en:fc7bd0f5-->
Die Integration des exakten Mischungsfeldes `mix.velocity` von fünf festen Startpunkten aus, dem Ursprung $(0, 0)$ und $(\pm 1.5, 0), (0, \pm 1.5)$, zeigt, wie die über mehrere Schritte sichtbare Krümmung die A-priori-Verteilung aufteilt:

<!--en:5ffa2c6d-->
- **1 Schritt:** Alle fünf Startpunkte landen bei $\mathbb{E}[\mathbf{x}_0] = (-0.50, -0.70)$ im leeren Tal zwischen den Modi.

<!--en:cc9c734c-->
- **10 gegenüber 100 Schritten:** Nach 10 Schritten liegt jede Trajektorie in ihrem Modus, höchstens $\approx 0.2$ vom Endpunkt nach 100 Schritten entfernt und leicht zum globalen Mittelwert hin verschoben.

<!--en:8f664d33-->
- **Massenerhaltende Einzugsgebiete:** Drei der fünf Startpunkte, darunter der Ursprung, führen in den Modus mit dem größten Gewicht ($\pi_1 = 0.5$ bei $(-2.5, -1.5)$). Von $(1.5, 0)$ aus erreicht die Trajektorie Modus 2, von $(0, 1.5)$ aus Modus 3.

<!--en:d7075b8c-->
::: {.slide title="Pfadunabhängige Koordinaten durch Umskalierung mit α_t"}

<!--en:8278fb79-->
[Einheitliche ODE-Integration für alle Zeitpläne]{.kicker}

<!--en:ba7a3e65-->
- **Universelle VE-Koordinaten (Karras et al., 2022):** Division von $\mathbf{x}_t$ durch $\alpha_t$ absorbiert bei einem Rauschverhältnis von $\bar\sigma_t = \sigma_t/\alpha_t = e^{-\lambda_t/2}$ die lineare Drift. Jeder Pfad erfüllt dann $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$.

<!--en:23682afb-->
- **Pfadunabhängiger Euler-Schritt (= DDIM bei $\eta = 0$):** Ein Schritt von $\bar\sigma_t$ nach $\bar\sigma_s$ und anschließende Multiplikation mit $\alpha_s$ ergeben $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$.

<!--en:beb70719-->
::: {.slide title="Implementierung des pfadunabhängigen ODE-Samplings"}

<!--en:9e22d402-->
[Universeller Prädiktor und Euler-Schleife]{.kicker}

<!--en:f9e034f5-->
`make_pred` rechnet jede Netzausgabe bei gleicher Log-SNR in $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ um. In Verbindung mit `ode_sample` kann damit jedes Lösungsverfahren jedes trainierte Modell für jeden Zeitplan integrieren:

<!--en:e33a1273-->
- **Reines Rauschen ($\alpha_1 = 0, \bar\sigma_1 = \infty$):** Bei $t = 1$ auf dem linearen Pfad ist $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$ unendlich. Die Aktualisierung $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$ benötigt jedoch nur $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ an der Stelle $(\mathbf{x}_1, \alpha_1 = 0, \sigma_1 = 1)$.

<!--en:9ba36eb3-->
- **Warum Rauschmodelle $\alpha_1 > 0$ brauchen:** Der Prädiktor `'eps'` berechnet $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t\hat{\boldsymbol\epsilon})/\alpha_t$ und kann daher bei $\alpha_1 = 0$ nicht starten. Auf dem VP-Pfad gilt $\alpha_1 \approx 0.007 > 0$.

<!--en:0f6ba40c-->
- **Äquivalenz auf dem linearen Pfad:** Einsetzen von $\alpha_s = 1 - s$ und $\sigma_s = s$ ergibt $\mathbf{x}_s = (1 - s)\hat{\mathbf{x}}_0 + s\hat{\boldsymbol\epsilon} = \mathbf{x}_t - (t - s)(\hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0)$, also genau den expliziten Euler-Schritt in $t$.

<!--en:434a9f80-->
- **Exaktheit bei einem einzigen Datenpunkt:** Anders als das Euler-Verfahren in $t$ auf dem VP-Pfad integriert $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$ Daten aus einem einzigen Punkt bei jeder Schrittweite ohne Diskretisierungsfehler.

<!--en:0cf4e516-->
::: {.slide title="Sampling mit dem gelernten Geschwindigkeitsfeld in 1, 10 und 100 Schritten"}

<!--en:4c78025d-->
[Empirische Konvergenz des ODE-Verfahrens]{.kicker}

<!--en:2ca2669f-->
Die Anwendung von `ode_sample` auf das für den linearen Pfad trainierte Geschwindigkeitsnetz `ema_v` mit $2{,}000$ gaußverteilten Startpunkten zeigt, wie der Diskretisierungsfehler mit wachsendem $K$ verschwindet:

<!--en:31673eb0-->
::: {.slide title="Diskretisierungsfehler und Netzfehler"}

<!--en:a90dec1f-->
[Vergleich der Läufe mit 10 und 100 Schritten]{.kicker}

<!--en:07c50e5b-->
- **10 Euler-Schritte (trainiertes MLP und exaktes Orakel):** Beide verringern die Streuung innerhalb der Modi ($0.45$ und $0.46$ gegenüber $0.48\text{--}0.49$ bei den Daten). Die Verengung entsteht also durch die Diskretisierung im Lösungsverfahren. Der Anteil modusferner Samples steigt auf $\approx 3\%$ (Netz) bzw. $2\text{--}3\%$ (Orakel) gegenüber $1.1\text{--}1.5\%$ bei den Daten.

<!--en:47e7cf2d-->
- **100 Euler-Schritte:** Die Modusgewichte stimmen innerhalb des Stichprobenfehlers mit den Daten überein, ebenso die Streuung ($0.48$) und der Anteil modusferner Samples ($\approx 1.3\%$).

<!--en:9d392751-->
::: {.slide title="Ancestral Sampling mit DDPM: Zerlegung in erster Ordnung"}

<!--en:29f6544b-->
[Stochastische rückwärts laufende Markov-Kette]{.kicker}

<!--en:bd265ff0-->
- **Entwicklung erster Ordnung ($1/\sqrt{1-\beta} \approx 1 + \beta/2$):** Die Drift des DDPM-Rückwärtsschritts zerfällt in zwei gleichzeitige Aktualisierungen.

<!--en:32226540-->
- **Aufbau des Schritts:** Jeder ancestrale Schritt besteht aus **1 Euler-Schritt der Wahrscheinlichkeitsfluss-ODE** (Transport $p_t \to p_s$) und **1 Langevin-MCMC-Schritt** (Dämpfung von Driftfehlern).

<!--en:72ff9031-->
::: {.slide title="Kontinuierliche Zeit: zeitumgekehrte SDE und Wahrscheinlichkeitsfluss-ODE"}

<!--en:858e602e-->
[Andersons Satz zur Zeitumkehr (1982)]{.kicker}

<!--en:a36fd7ae-->
- **Zeitumgekehrte SDE (stochastisch):** Die ancestrale DDPM-Kette diskretisiert diese Gleichung. Neues brownsches Rauschen $g(t)\,d\bar{\mathbf{W}}$ mischt die Trajektorien fortlaufend und dämpft Score-Fehler.

<!--en:70b7777f-->
- **Wahrscheinlichkeitsfluss-ODE (deterministisch):** Ohne $g\,d\bar{\mathbf{W}}$ und mit halbiertem Score-Term ($g^2 \to \frac{1}{2}g^2$) ergibt sich eine umkehrbare Bijektion $\mathbf{x}_1 \leftrightarrow \mathbf{x}_0$ mit genau denselben Randdichten $p_t(\mathbf{x})$.

<!--en:98e69a0e-->
::: {.slide title="Paarweiser Vergleich mit dem exakten Orakel bei 20 000 Startpunkten"}

<!--en:f8a88645-->
[Ancestrale Kette und Wahrscheinlichkeitsfluss-ODE]{.kicker}

<!--en:d2bf15aa-->
$200$ ancestrale Schritte (`ema_eps` auf dem VP-Pfad) und $100$ Euler-Schritte der ODE (`ema_v` auf dem linearen Pfad) beginnen bei denselben $20{,}000$ gaußverteilten Punkten $\mathbf{x}_1$. Die zugehörigen exakten Läufe mit `oracle` verwenden identische Startpunkte und identisches zusätzliches Rauschen:

<!--en:bf6d6c11-->
::: {.slide title="ODE und ancestrale Kette als Sampler für die Mischung"}

<!--en:5c0bad45-->
[Auswertung des Experiments mit 20 000 Samples]{.kicker}

<!--en:fbc777c4-->
Bei $N = 20{,}000$ Samples beträgt der binomiale Standardfehler eines Modusgewichts $\sqrt{w(1-w)/N} \approx 0.0035$. Damit lässt sich die Genauigkeit der erzeugten Verteilung eng eingrenzen:

<!--en:61dff380-->
- **Bei weniger als 1% der Startpunkte ein anderer Modus als beim exakten Orakel:** Für $99.5\%$ bis $99.8\%$ der $20{,}000$ Startpunkte führen das trainierte neuronale Netz und der exakte analytische A-posteriori-Mittelwert die Trajektorie in **genau denselben Modus**.

<!--en:bfbca854-->
- **Modusgewichte innerhalb der Stichprobenschwankung:** Die ancestrale Kette mit 200 Schritten und das Euler-Verfahren mit 100 Schritten treffen die Modusgewichte des Orakels bis auf $\le 0.004$ ($\approx 1\,\mathrm{SE}$), die Streuung innerhalb der Modi bis auf $0.01$ und den Anteil modusferner Samples ($\approx 1.1\%$).

<!--en:37500ae2-->
- **Deterministische Abbildung und stochastische Korrektur:** Die ODE bildet jeden Startpunkt $\mathbf{x}_1$ deterministisch auf genau einen Punkt $\mathbf{x}_0$ ab. Das ermöglicht Umkehrung und Interpolation im latenten Raum in §{sec_diffusion-samplers}. Die ancestrale Kette fügt bei jedem Schritt Langevin-Rauschen hinzu.

<!--en:aef88098-->
::: {.slide title="Die generativen Modelle aus Kapitel §{num:chap_diffusion} im Überblick"}

<!--en:52fa418b-->
[Pfad, Zielgröße und Sampler]{.kicker}

<!--en:0d4afdef-->
Jedes Diffusions- und Flow-Matching-Modell in der Literatur lässt sich durch drei Bausteine angeben: einen Vorwärtspfad, ein Regressionsziel und ein Verfahren zur Rückwärtsintegration.

<!--en:ffcd8459-->
| Modellfamilie | Vorwärtspfad | Trainingsziel | Rückwärtssampler | Abschnitt |

<!--en:aad6e6bc-->
| **DSM mit festem Rauschpegel** | Feste Verrauschung $\mathbf{x}_0 + \sigma\boldsymbol\epsilon$ | Score $-\boldsymbol\epsilon/\sigma$ | Langevin-MCMC bei festem $\sigma$ | §{sec_diffusion-denoising} |

<!--en:93922094-->
| **NCSN** (Song und Ermon, 2019) | Geometrische VE-Leiter $\tilde\sigma_i$ | Score $-\boldsymbol\epsilon/\tilde\sigma_i$ (Gewicht $\tilde\sigma_i^2$) | Langevin-Leiter mit abnehmendem Rauschpegel | §{sec_diffusion-gaussian-paths} |

<!--en:b77bb6a7-->
| **DDPM** (Ho et al., 2020) | VP mit linearem $\beta$ ($\alpha_t^2 + \sigma_t^2 = 1$) | Rauschen $\boldsymbol\epsilon$ | Ancestrale Kette (zeitumgekehrte SDE) | §{sec_diffusion-gaussian-paths}–§{num:sec_diffusion-integration} |

<!--en:8c841b56-->
| **Flow-Matching / Rectified Flow** | Linear $(1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$ | Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$ | Euler- oder Heun-Verfahren für die Wahrscheinlichkeitsfluss-ODE | §{sec_diffusion-gaussian-paths}–§{num:sec_diffusion-samplers} |

<!--en:e0f5b193-->
| **$\mathbf{v}$-Vorhersage** (Salimans und Ho) | VP mit Kosinuspfad ($\cos\frac{\pi t}{2}, \sin\frac{\pi t}{2}$) | Winkelgeschwindigkeit $\mathbf{v} = \alpha_t\boldsymbol\epsilon - \sigma_t\mathbf{x}_0$ | ODE oder ancestraler Sampler | §{sec_diffusion-gaussian-paths}, §{sec_diffusion-design-space} |

<!--en:9174e2c5-->
| **EDM** (Karras et al., 2022) | VE-Koordinaten $\bar{\mathbf{x}} = \mathbf{x}_0 + \bar\sigma\boldsymbol\epsilon$ | Vorkonditionierter unverrauschter Punkt $\hat{\mathbf{x}}_0$ | Heun-Verfahren zweiter Ordnung auf einem $\bar\sigma^{1/\rho}$-Gitter | §{sec_diffusion-samplers}–§{num:sec_diffusion-design-space} |

<!--en:c2d81b72-->
::: {.slide title="Verständnisfragen zu Transport, Euler-Schritten und Zerlegung der SDE"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:99f18c36-->
Fragen zur Integration in umgekehrter Zeitrichtung und zur Funktionsweise der Sampler:

<!--en:763156f3-->
1. **Ein Euler-Schritt von reinem Rauschen aus:** Gegeben sei die Mischung mit den drei Mittelwerten $(-2.5, -1.5)$, $(2.5, -1.5)$, $(0, 2.5)$ und den Gewichten $(0.5, 0.3, 0.2)$. Wohin führt ein einziger Euler-Schritt $\mathbf{x}_0 = \mathbf{x}_1 - \mathbf{u}_1(\mathbf{x}_1)$ von einem beliebigen Startpunkt $\mathbf{x}_1$ aus?

<!--en:277552ed-->
2. **Fehlerverstärkung nahe reinem Rauschen:** Auf dem linearen Pfad gilt $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - t\hat{\boldsymbol\epsilon})/(1 - t)$. Ein Rauschprädiktor $\hat{\boldsymbol\epsilon}$ hat den Fehler $\boldsymbol\delta$ bei $t = 0.95$. Wie groß ist dann der Fehler in der Geschwindigkeit $\mathbf{u}_t = \hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0$?

<!--en:e37075dc-->
3. **Gleiche Randdichten bei zeitumgekehrter SDE und Wahrscheinlichkeitsfluss-ODE:** Um welchen Term unterscheiden sich die Rückwärtsschritte, und warum lässt dieser Term $p_t$ unverändert?

<!--en:557b98b6-->
**Lösungen:**

<!--en:a0401291-->
1. Wegen $\mathbf{u}_1(\mathbf{x}_1) = \mathbf{x}_1 - \mathbb{E}[\mathbf{x}_0]$ fällt jeder Startpunkt $\mathbf{x}_1$ auf den globalen Datenmittelwert $\mathbb{E}[\mathbf{x}_0] = 0.5(-2.5, -1.5) + 0.3(2.5, -1.5) + 0.2(0, 2.5) = \mathbf{(-0.50, -0.70)}$.

<!--en:1cd57152-->
2. Wegen $\mathbf{u}_t = (\hat{\boldsymbol\epsilon} - \mathbf{x}_t)/(1 - t)$ verstärkt sich ein Fehler $\boldsymbol\delta$ in $\hat{\boldsymbol\epsilon}$ um den Faktor $1/(1 - t) = 1/0.05 = \mathbf{20\times}$ (auf $\mathbf{20\,\boldsymbol\delta}$). Bei $t = 1$ ($\alpha_1 = 0$) divergiert der Fehler.

<!--en:e3350239-->
3. Die Rückwärtsschritte unterscheiden sich um $\frac{1}{2}g(t)^2\nabla \log p_t(\mathbf{x})\,\Delta t + g(t)\sqrt{\Delta t}\,\mathbf{z}$. Das ist ein infinitesimaler **Langevin-Schritt** mit Schrittweite $h = g(t)^2\Delta t$ und stationärer Verteilung $p_t$.

<!--en:4ef7ab1b-->
::: {.slide title="Zusammenfassung der Integration in umgekehrter Zeitrichtung"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:90706fd0-->
| Konzept / Sampler | Gleichung | Mathematische Eigenschaft | Verhalten in der Praxis |

<!--en:1768ccc8-->
| **Kontinuitätsgleichung** | $\partial_t p_t = -\nabla \cdot (p_t \mathbf{u}_t)$ | Erhält die gesamte Wahrscheinlichkeitsmasse | Garantiert, dass $\mathbf{u}_t = \mathbb{E}[\dot{\mathbf{x}}_t \mid \mathbf{x}_t]$ die Modusgewichte exakt reproduziert |

<!--en:b5b44c7a-->
| **Wahrscheinlichkeitsfluss-ODE** | $d\mathbf{x}/dt = \mathbf{u}_t(\mathbf{x})$ ($t: 1 \to 0$) | Glatte deterministische Bijektion $\mathbf{x}_1 \leftrightarrow \mathbf{x}_0$ | Reproduzierbare Samples, latente Codes, exakte Likelihoods |

<!--en:c9604a8b-->
| **Kollaps bei einem Euler-Schritt** | $\mathbf{x}_1 - \mathbf{u}_1(\mathbf{x}_1) = \mathbb{E}[\mathbf{x}_0]$ | Die Tangente bei $t = 1$ zeigt zum A-priori-Mittelwert | Alle Startpunkte landen bei $\mathbb{E}[\mathbf{x}_0]$. Gekrümmte Pfade brauchen $\ge 10$ Schritte |

<!--en:2733e331-->
| **Umskalierte VE-Koordinaten** | $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t, \;\bar\sigma = \sigma_t/\alpha_t$ | $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}, \bar\sigma)$ auf jedem Pfad | Euler-Schritt $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$ (= DDIM bei $\eta = 0$) |

<!--en:e223fbf7-->
| **Ancestraler DDPM-Schritt** | $\frac{\mathbf{x}_t + \beta\nabla \log p_t}{\sqrt{1-\beta}} + \sqrt{\beta}\mathbf{z}$ | $= \text{Flow ODE Euler step} + \text{Langevin step}$ | Diskretisiert Andersons zeitumgekehrte SDE und dämpft Driftfehler |
