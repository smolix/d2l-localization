<!--en:17747603-->
# Diskrete Diffusion

<!--en:9859e393-->
Alle Modelle der vorangegangenen Abschnitte addierten gaußsches Rauschen zu kontinuierlichen Vektoren, etwa zu Punkten in $\mathbb{R}^2$, Pixelintensitäten oder latenten Darstellungen eines Autoencoders. Diskrete Sequenzen wie Texttoken, Quellcode oder quantisierte Bildtoken lassen sich nicht durch Addition kleiner reeller Zahlen verrauschen. Autoregressive Modelle (:numref:`sec_gpt`) erzeugen $L$ Token in $L$ aufeinanderfolgenden Vorwärtsdurchläufen von links nach rechts. Diskrete Diffusionsmodelle erzeugen pro Netzauswertung mehrere Token parallel und nutzen dabei bidirektionalen Kontext.

<!--en:26a59434-->
Um Diffusion auf einem endlichen Vokabular zu definieren, ersetzten :citet:`Hoogeboom.Nielsen.Jaini.ea.2021` und :citet:`Austin.Johnson.Ho.ea.2021` die gaußsche Störung durch eine Markov-Kette mit diskreten Zuständen und Übergangsmatrizen $\mathbf{Q}_t$. Für Text und diskrete Codes ist der *absorbierende (maskierende) Übergangskern* (engl. absorbing kernel) am wirksamsten. Er ersetzt die Token unabhängig voneinander durch ein Maskensymbol $[\textrm{M}]$, bis die gesamte Sequenz gelöscht ist. Für diese absorbierende Kette vereinfacht sich die untere Variationsschranke aus :numref:`sec_diffusion-design-space` zu einer Zielfunktion aus gewichteten Kreuzentropien an maskierten Positionen :cite:`Sahoo.Arriola.Schiff.ea.2024,Shi.Han.Wang.ea.2024`. In diesem Abschnitt leiten wir die absorbierende Markov-Kette und ihre Variationsschranke her. An einer quantisierten Variante der trimodalen Mischung mit zwei Token und an binären Fashion-MNIST-Bildern untersuchen wir den Fehler, der beim parallelen Demaskieren mehrerer Token durch die Annahme der Unabhängigkeit entsteht. Schließlich stellen wir den Zusammenhang zwischen maskierter Diffusion und autoregressiven sowie maskierten Sprachmodellen her (:numref:`sec_transformer` und :numref:`sec_gpt`).

<!--en:a18c4881-->
## Verrauschen von Token

<!--en:6038689a-->
Vom Vorwärtsprozess für Token erwarten wir dasselbe wie von der gaußschen Kette aus :numref:`sec_diffusion-gaussian-paths`: Randverteilungen und A-posteriori-Verteilungen in geschlossener Form sowie einen Endzustand, der keine Information über die Daten enthält.

<!--en:965725d3-->
### Übergangsmatrizen

<!--en:71b55519-->
Zunächst ändern wir die Notation. Wie in den Arbeiten zur diskreten Diffusion zählt nun die ganze Zahl $t = 1, \ldots, T$ die Schritte. Bei der gaußschen Kette übernahm $k$ diese Rolle. Das Netz erhält weiterhin $t/T$, die Zeit auf der in diesem Kapitel verwendeten Skala.

<!--en:cd5d66ac-->
Ein Token nehme einen von $K$ Werten an. Wir schreiben es als One-Hot-Zeilenvektor $\mathbf{x} \in \{0, 1\}^K$. Rauschen bedeutet für ein Token, dass sich sein Wert zufällig ändert. Dies beschreiben *Übergangsmatrizen* $\mathbf{Q}_t \in [0, 1]^{K \times K}$, deren Zeilensummen jeweils eins sind. Dabei ist $[\mathbf{Q}_t]_{ij}$ die Wahrscheinlichkeit für den Übergang vom Wert $i$ zum Wert $j$ im Schritt $t$. Das sind die zeilenstochastischen Matrizen aus dem Markov-Chain-Monte-Carlo-Verfahren (:numref:`sec_mdl-bayes-mcmc`). Eine Verteilung schreiben wir als Zeilenvektor, den wir von rechts mit $\mathbf{Q}_t$ multiplizieren. Damit gilt

<!--en:3c1676ba-->
wobei $\mathrm{Cat}(\mathbf{x}; \mathbf{p})$ die Kategorialverteilung mit Wahrscheinlichkeitsvektor $\mathbf{p}$ bezeichnet. Hier ist das die Zeile von $\mathbf{Q}_t$, die dem aktuellen Wert entspricht. Eine Sequenz aus $L$ Token verrauschen wir, indem wir die Kette an jeder Position unabhängig laufen lassen. Verketten wir zwei Übergänge, multiplizieren sich ihre Matrizen. Per Induktion über $t$ ergibt sich damit für die Randverteilung nach $t$ Schritten

<!--en:94675da4-->
Die A-posteriori-Verteilung für einen Schritt bei gegebenem Anfangs- und Endzustand folgt aus der Bayes-Regel und der Markov-Eigenschaft:

<!--en:5ec85c22-->
Dabei bezeichnet $\odot$ das elementweise Produkt. Das Gewicht jedes möglichen Vorgängers ergibt sich aus seiner Wahrscheinlichkeit bei gegebenem Start und der Wahrscheinlichkeit, von ihm zum beobachteten $\mathbf{x}_t$ zu gelangen.

<!--en:a6ee19f1-->
Betrachten wir die A-posteriori-Verteilung für ein binäres Token, das in jedem Schritt mit Wahrscheinlichkeit $0.1$ seinen Wert wechselt. Hier gilt $\mathbf{Q}_1 = \mathbf{Q}_2 = \left(\begin{smallmatrix} 0.9 & 0.1 \\ 0.1 & 0.9 \end{smallmatrix}\right)$. Wir beginnen beim ersten Wert, $\mathbf{x}_0 = (1, 0)$. Angenommen, nach zwei Schritten beobachten wir den zweiten Wert, $\mathbf{x}_2 = (0, 1)$. Die Wahrscheinlichkeit dafür ist $0.9 \cdot 0.1 + 0.1 \cdot 0.9 = 0.18$, unser Nenner. Für den Zwischenzustand erhalten wir $\mathbf{x}_2 \mathbf{Q}_2^\top = (0.1, 0.9)$ und $\mathbf{x}_0 \bar{\mathbf{Q}}_1 = (0.9, 0.1)$. Das elementweise Produkt ist $(0.09, 0.09)$. Teilen wir es durch $0.18$, erhalten wir $(1/2, 1/2)$. Das Token hat genau einmal den Wert gewechselt. Bei gegebenen Endpunkten sind der erste und der zweite Schritt dafür gleich wahrscheinlich.

<!--en:2d2af037-->
### Der absorbierende Übergangskern

<!--en:51bd260e-->
Für die Schranke brauchen wir eine geschlossene Form für $\bar{\mathbf{Q}}_t$, für den Sampler einen bekannten Anfangszustand. Der *absorbierende Übergangskern* von :citet:`Austin.Johnson.Ho.ea.2021` liefert beides. Er erweitert das Vokabular um einen Wert, die *Maske* $[\textrm{M}]$. Mit Wahrscheinlichkeit $\beta_t$ geht jedes Token im Schritt $t$ in die Maske über. Ein maskiertes Token bleibt maskiert:

<!--en:4758fb21-->
Dabei ist $\mathbf{1}$ der Spaltenvektor aus lauter Einsen und $\mathbf{e}_{\textrm{M}}$ der One-Hot-Spaltenvektor der Maske. Jede Zeile von $\mathbf{1} \mathbf{e}_{\textrm{M}}^\top$ legt also ihre gesamte Wahrscheinlichkeitsmasse auf die Maske.

<!--en:ce437910-->
Produkte solcher Matrizen lassen sich vereinfachen. Setzen wir $\mathbf{P} = \mathbf{1} \mathbf{e}_{\textrm{M}}^\top$. Wegen $\mathbf{e}_{\textrm{M}}^\top \mathbf{1} = 1$ gilt $\mathbf{P}^2 = \mathbf{P}$ und damit $(a\mathbf{I} + b\mathbf{P})(c\mathbf{I} + d\mathbf{P}) = ac\, \mathbf{I} + (ad + bc + bd)\, \mathbf{P}$. Gilt $a + b = c + d = 1$, so ist der zweite Koeffizient gleich $1 - ac$. Entlang der Kette multiplizieren sich also die Koeffizienten von $\mathbf{I}$. Sei $\bar{\alpha}_t = \prod_{s \leq t} (1 - \beta_s)$ dasselbe Produkt wie bei der gaußschen Kette aus :numref:`sec_diffusion-gaussian-paths`. Dort war es das quadrierte Gewicht des rauschfreien Punkts. Hier hat es eine unmittelbarere Bedeutung: die Wahrscheinlichkeit, dass ein Token $t$ Schritte ohne Maskierung übersteht. Dann gilt

<!--en:1cf6454b-->
Die Kette löscht Token nur und verwechselt nie zwei Token miteinander. Ein Zeitplan mit $\bar{\alpha}_T = 0$ endet bei der vollständig maskierten Sequenz (:numref:`fig_diffusion-masking`). Diese enthält ebenso wenig Information über die Daten wie das reine Rauschen am Ende der gaußschen Kette. Die andere gängige Wahl ist der *Gleichverteilungskern* (engl. uniform kernel). Er zieht ein Token erneut aus der Gleichverteilung über dem Vokabular :cite:`Hoogeboom.Nielsen.Jaini.ea.2021` (Aufgabe :exref:`ex_discrete-diffusion-uniform-kernel`).

<!--en:b888af41-->
![Die absorbierende Kette für neun Token. In Vorwärtsrichtung ersetzt die Kette jedes Token unabhängig mit Wahrscheinlichkeit $\beta_t$ pro Schritt durch die Maske (als ? dargestellt). Danach bleibt es maskiert. Nach $T$ Schritten sind alle Token maskiert. In Rückwärtsrichtung sagt das Modell die maskierten Token aus der teilweise maskierten Sequenz vorher. Ein aufgedecktes Token ändert sich nie wieder.](../img/mdl-diffusion-masking.svg)

<!--en:69e1ef78-->
Von nun an verwenden wir den absorbierenden Übergangskern. In den Vergleichen von Austin et al. lieferte er bei Text die besten Likelihoods, und seine A-posteriori-Verteilung ist denkbar einfach. Das hat einen Preis. In Rückwärtsrichtung deckt die Kette Token nur auf. Wie ein Links-nach-rechts-Modell legt sich der Sampler damit auf jedes geschriebene Token fest. Ketten, die Token erneut ziehen, etwa mit dem Gleichverteilungskern, können einen frühen Fehler korrigieren. Dasselbe gilt für Sampler wie ReMDM :cite:`Wang.Schiff.Sahoo.ea.2025`, die aufgedeckte Token erneut maskieren (:numref:`sec_diffusion-further-reading`).

<!--en:76511b7f-->
## Die Schranke für maskierte Diffusion

<!--en:288b968e-->
Um eine Rückwärtskette zu trainieren, brauchen wir einen Verlust. Die Variationsschranke aus :numref:`sec_diffusion-design-space` liefert ihn.

<!--en:ff7849fa-->
### Gleiche Schranke, andere Zielgröße

<!--en:3118e594-->
Das Rückwärtsmodell ist eine Markov-Kette $p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$, die bei der vollständig maskierten Sequenz startet. In :numref:`sec_diffusion-design-space` haben wir die negative Log-Likelihood einer solchen Kette nach oben beschränkt durch

<!--en:e85c4b9b-->
Das ist die untere Evidenzschranke aus :numref:`sec_mdl-latent-em-elbo` mit umgekehrtem Vorzeichen und der Vorwärtskette als Variationsverteilung. Die Herleitung nutzte nur die Markov-Eigenschaft und die Bayes-Regel. Sie gilt daher unverändert für Token.

<!--en:a633538f-->
Was sich ändert, ist die Zielgröße. Bei gaußschem Rauschen sagte das Netz das Rauschen, die Geschwindigkeit oder den rauschfreien Punkt vorher. Alle drei sind affine Funktionen desselben A-posteriori-Mittelwerts $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$ (:numref:`sec_diffusion-denoising`). Bei einem One-Hot-kodierten Token enthält dieser Mittelwert für jeden möglichen ursprünglichen Wert dessen Wahrscheinlichkeit. Die Vorhersage dieser Verteilung ist also die diskrete Entsprechung des Entrauschers vom Anfang des Kapitels. Wie bei :citet:`Austin.Johnson.Ho.ea.2021` gibt das Netz für jede Position $i$ eine Verteilung $p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_t)$ über die ursprünglichen Werte aus, bedingt auf die gesamte verrauschte Sequenz. Für den Rückwärtsübergang mitteln wir die wahre A-posteriori-Verteilung über diese Vorhersage,

<!--en:fd2a5c6e-->
und zwar für jede Position einzeln. An unmaskierten Positionen übernehmen wir das beobachtete Token als Vorhersage. Für ein einzelnes Token ergibt eine perfekte Vorhersage so den wahren Rückwärtsübergang (Aufgabe :exref:`ex_discrete-diffusion-two-forms`a). Bei einer Sequenz bleibt ein Unterschied: Für Positionen, die im selben Schritt zum ursprünglichen Wert zurückkehren, ziehen wir die Werte unabhängig voneinander. Der wahre Rückwärtsübergang zieht sie gemeinsam. Ein Großteil dieses Abschnitts untersucht die Folgen dieses Unterschieds.

<!--en:3042d67f-->
### Die A-posteriori-Verteilung der absorbierenden Kette

<!--en:44d7502f-->
**Satz (A-posteriori-Verteilung der absorbierenden Kette).** *Für den absorbierenden Übergangskern gelte $\beta_1 > 0$ und somit $\bar{\alpha}_t < 1$ für jedes $t \geq 1$. Betrachten wir eine Position mit ursprünglichem Wert $\mathbf{x}_0 \neq [\textrm{M}]$. Wenn $\mathbf{x}_t = \mathbf{x}_0$, dann gilt $\mathbf{x}_{t-1} = \mathbf{x}_0$ mit Wahrscheinlichkeit eins. Wenn $\mathbf{x}_t = [\textrm{M}]$, dann gilt*

<!--en:1acf3433-->
*und kein anderer Wert hat positive Wahrscheinlichkeit.*

<!--en:990f78fb-->
**Beweis.** Wir werten :eqref:`eq_diffusion-discrete-posterior` mithilfe von :eqref:`eq_diffusion-absorbing-kernel` und :eqref:`eq_diffusion-discrete-qbar` aus. Der Vektor $\mathbf{x}_0 \bar{\mathbf{Q}}_{t-1}$ legt die Masse $\bar{\alpha}_{t-1}$ auf $\mathbf{x}_0$ und $1 - \bar{\alpha}_{t-1}$ auf die Maske. Nur diese beiden Vorgänger sind also möglich. Gilt $\mathbf{x}_t = \mathbf{x}_0$, so hat der Vektor $\mathbf{x}_t \mathbf{Q}_t^\top$ den Eintrag $1 - \beta_t$ bei $\mathbf{x}_0$ und den Eintrag $0$ bei der Maske, denn aus einer Maske wird nie wieder ein Token. Die A-posteriori-Verteilung ist somit eine Punktmasse. Gilt $\mathbf{x}_t = [\textrm{M}]$, so lauten die Einträge $\beta_t$ bei $\mathbf{x}_0$ und $1$ bei der Maske. Die unnormierte A-posteriori-Verteilung legt daher $\bar{\alpha}_{t-1}\beta_t$ auf $\mathbf{x}_0$ und $1 - \bar{\alpha}_{t-1}$ auf die Maske. Ihre Summe $1 - \bar{\alpha}_{t-1}(1 - \beta_t) = 1 - \bar{\alpha}_t$ ist der Nenner $q(\mathbf{x}_t = [\textrm{M}] \mid \mathbf{x}_0)$, und es gilt $\bar{\alpha}_{t-1}\beta_t = \bar{\alpha}_{t-1} - \bar{\alpha}_t$. $\blacksquare$

<!--en:c12e5b76-->
Gehen wir einen Schritt zurück, so bleibt ein maskiertes Token entweder maskiert oder nimmt wieder seinen ursprünglichen Wert an. Das Rückwärtsmodell :eqref:`eq_diffusion-x0-parameterization` kopiert daher die unmaskierten Positionen von $\mathbf{x}_t$. An jeder maskierten Position setzt es unabhängig mit der Rückkehrwahrscheinlichkeit einen aus $p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_t)$ gezogenen Wert ein.

<!--en:8d1277d7-->
### Eine gewichtete Kreuzentropie

<!--en:48424cae-->
Das Rückwärtsmodell kopiert die sichtbaren Token und füllt nur Masken. Zur Schranke können daher nur die maskierten Positionen beitragen. Das folgende zentrale Ergebnis dieses Abschnitts geht in dieser Form auf :citet:`Sahoo.Arriola.Schiff.ea.2024` und :citet:`Shi.Han.Wang.ea.2024` zurück: Der Beitrag ist eine Kreuzentropie.

<!--en:bf6e837c-->
**Satz (Schranke für maskierte Diffusion).** *Für den absorbierenden Übergangskern mit $\bar{\alpha}_0 = 1$, $\bar{\alpha}_t < 1$ für $t \geq 1$ und $\bar{\alpha}_T = 0$ sowie für das soeben beschriebene Rückwärtsmodell gilt*

<!--en:5fb06c66-->
**Beweis.** Der Prior-Term verschwindet. Wegen $\bar{\alpha}_T = 0$ ist $q(\mathbf{x}_T \mid \mathbf{x}_0)$ die Punktmasse auf der vollständig maskierten Sequenz, ebenso wie $p(\mathbf{x}_T)$. Sowohl $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ als auch $p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ faktorisieren über die Positionen. Jede Divergenz ist daher eine Summe von Divergenzen der einzelnen Positionen. An einer unmaskierten Position sind beide Verteilungen dieselbe Punktmasse und die Divergenz ist null. An einer maskierten Position setzen wir $w_t = (\bar{\alpha}_{t-1} - \bar{\alpha}_t) / (1 - \bar{\alpha}_t)$. Die wahre A-posteriori-Verteilung legt $1 - w_t$ auf die Maske und $w_t$ auf $\mathbf{x}_0^i$. Das Modell legt $1 - w_t$ auf die Maske und $w_t\, p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}_t)$ auf die Token. Somit gilt

<!--en:f437fce7-->
Damit sind die Schritte $t = 2, \ldots, T$ erfasst. Der Rekonstruktionsterm lautet $\mathbb{E}[-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0 \mid \mathbf{x}_1)]$. Wegen $w_1 = (1 - \bar{\alpha}_1)/(1 - \bar{\alpha}_1) = 1$ ersetzt das Modell jede im Schritt $1$ maskierte Position durch einen aus $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}_1)$ gezogenen Wert und kopiert den Rest. Dieser Term ist somit die Summe von $-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_1)$ über die maskierten Positionen, also der Summand für $t = 1$. $\blacksquare$

<!--en:e834479c-->
Dies ist der Verlust eines maskierten Sprachmodells mit zwei Änderungen. Die Maskierungsrate durchläuft alle Stufen von keiner bis zur vollständigen Maskierung, und jede Stufe hat ihr eigenes Gewicht. Teilen wir die Schranke durch $\log 2$ und die Anzahl der Token, erhalten wir eine Codelänge in Bit pro Token (:numref:`sec_mdl-coding-compression`). In dieser Einheit geben wir die Schranke für Bilder an.

<!--en:94dcb8ab-->
Beim *linearen Zeitplan* $\bar{\alpha}_t = 1 - t/T$ ist der Anteil $t/T$ der Token im Schritt $t$ maskiert, und es gilt $w_t = (1/T)/(t/T) = 1/t$. Bei starker Maskierung zählt ein Fehler weniger als derselbe Fehler bei schwacher Maskierung. Da die Anzahl maskierter Positionen proportional zu $t$ wächst, ist ihre erwartete gewichtete Anzahl in jedem Schritt gleich. Sie beträgt $L/T$ für $L$ Token. Die Schritte unterscheiden sich nur darin, wie schwer ihre maskierten Token vorherzusagen sind.

<!--en:1c449fd9-->
Die Summe ist zugleich eine Riemann-Summe. Bezeichnen wir mit $\ell(\mathbf{x}_0, \mathbf{x})$ die Kreuzentropie an den maskierten Positionen von $\mathbf{x}$, also die innere Summe in :eqref:`eq_diffusion-masked-loss`. Betrachten wir den Grenzübergang $T \to \infty$ mit $\bar{\alpha}_t = \bar{\alpha}(t/T)$ für eine fallende Funktion auf der Zeitskala dieses Kapitels. Zur Zeit $0$ liegen die Daten vor, zur Zeit $1$ die vollständig maskierte Sequenz. Da $t$ jetzt die Schritte zählt, schreiben wir $\tau = t/T$ für die Zeit. Dann gilt $w_t \approx -\bar{\alpha}'(\tau)/(T(1 - \bar{\alpha}(\tau)))$. Der Faktor $1/T$ ist die Breite $d\tau$ eines Schritts. Damit konvergiert die Schranke gegen

<!--en:9dfab90c-->
Setzen wir nun die Maskierungswahrscheinlichkeit $u = 1 - \bar{\alpha}(\tau)$ ein (in diesem Abschnitt bezeichnet $u$ eine Wahrscheinlichkeit, keine Geschwindigkeit). Wegen $du = -\bar{\alpha}'(\tau)\, d\tau$ wird das Integral zu

<!--en:13dd5814-->
wobei $\mathbb{E}_u$ jedes Token unabhängig mit Wahrscheinlichkeit $u$ maskiert. Erhält das Netz die Maskierungswahrscheinlichkeit anstelle der Zeit, so hängt der Grenzwert nicht vom Zeitplan ab. Dies entspricht im diskreten Fall der Invarianz der gaußschen Schranke gegenüber dem Zeitplan (:numref:`sec_diffusion-design-space`) :cite:`Kingma.Salimans.Poole.ea.2021`.

<!--en:60bfca58-->
Tatsächlich braucht das optimale Netz keine der beiden Angaben. Welche Positionen maskiert sind, entscheiden Münzwürfe unabhängig von den Tokenwerten. Sobald die Eingabe zeigt, welche Positionen maskiert sind und welche Werte die übrigen enthalten, liefert die Zeit keine weitere Information über die verborgenen Token. Die beste Vorhersage an einer maskierten Position ist daher völlig unabhängig von der Zeit: Es ist die bedingte Verteilung des Tokens bei gegebenen sichtbaren Token :cite:`Ou.Nie.Xue.ea.2025` (Aufgabe :exref:`ex_discrete-diffusion-time-agnostic`).

<!--en:e87b8afb-->
In neueren Arbeiten begegnet uns dieser Grenzwert in anderer Form. Eine *Markov-Kette in kontinuierlicher Zeit* maskiert jedes verbliebene Token mit der Rate $-\bar{\alpha}'(\tau)/\bar{\alpha}(\tau)$ :cite:`Campbell.Benton.DeBortoli.ea.2022`. Ihre Umkehrung nutzt Wahrscheinlichkeitsverhältnisse zwischen Sequenzen, die sich in einem Token unterscheiden :cite:`Lou.Meng.Ermon.2024`. Beim absorbierenden Übergangskern ist ein solches Verhältnis das Produkt aus einer bekannten Funktion der Zeit und der bedingten Verteilung der ursprünglichen Werte, die unser Netz vorhersagt :cite:`Ou.Nie.Xue.ea.2025`. Beide Ansätze trainieren also dieselbe Größe.

<!--en:8f35283e-->
Die Funktion `masked_loss` schätzt die Schranke mit einem Schritt $t$ pro Sequenz. Vorgegeben sind der Zeitplan `alpha_bar` (die Werte $\bar{\alpha}_0, \ldots, \bar{\alpha}_T$) und der Index `mask_id` der Maske. Dank des Faktors $T$ liefert ein gleichverteilt gezogener Schritt einen erwartungstreuen Schätzer für die Summe über alle Schritte. Die Funktion summiert über alle Achsen außer der ersten und eignet sich daher sowohl für Tokenpaare als auch für Bilder. Damit die Schnittstellen des Kapitels einheitlich bleiben, erhalten unsere Netze weiterhin $t/T$. In Aufgabe :exref:`ex_discrete-diffusion-time-agnostic` messen wir, wie wenig das trainierte Modell für Tokenpaare diese Eingabe nutzt.

<!--en:abad2af4-->
## Sampling durch Demaskieren

<!--en:4906586e-->
Ein trainiertes Netz liefert für jede maskierte Position eine Verteilung. Beim Generieren müssen wir entscheiden, welche Positionen wir wann füllen.

<!--en:abd82a27-->
### Paralleles Aufdecken und Unabhängigkeitsfehler

<!--en:b91d9d37-->
Das Generieren beginnt bei der vollständig maskierten Sequenz. Im Schritt $t$ liefert ein Vorwärtsdurchlauf an jeder maskierten Position $p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}_t)$. Jede dieser Positionen zieht ein Token als Kandidaten und nimmt es mit Wahrscheinlichkeit $w_t$ an. Wie bei den gaußschen Samplern aus :numref:`sec_diffusion-samplers` möchten wir uns $T$ Netzauswertungen ersparen. Wir wählen daher Stufen $T = t_0 > t_1 > \cdots > t_S = 0$ und springen zwischen ihnen. Überspringen wir Stufen, verketten wir Übergangskerne. Wie oben gezeigt, ist ein Produkt absorbierender Übergangskerne selbst wieder absorbierend. Die gröbere Kette hat auf ihren Stufen daher die Randverteilungen :eqref:`eq_diffusion-discrete-qbar`, und der Satz gilt unverändert. Die Funktion `unmask` springt von $t_k$ nach $t_{k+1}$, indem sie jedes maskierte Token mit Wahrscheinlichkeit $(\bar{\alpha}_{t_{k+1}} - \bar{\alpha}_{t_k}) / (1 - \bar{\alpha}_{t_k})$ aufdeckt.

<!--en:0a499ed3-->
Die Entscheidung, ob wir eine Position aufdecken, ist unabhängig vom Netz. Wir können also vorab bestimmen, wann jedes Token erscheint. Die Wahrscheinlichkeiten $(1 - \bar{\alpha}_{t_{k+1}})/(1 - \bar{\alpha}_{t_k})$ dafür, dass ein Token maskiert bleibt, multiplizieren sich über die Sprünge hinweg. Die Zwischenfaktoren kürzen sich: Auf Stufe $t_k$ ist das Token mit Wahrscheinlichkeit $1 - \bar{\alpha}_{t_k}$ noch maskiert, wie bei der Vorwärtskette. Im Sprung $k$ mit $k = 0, \ldots, S - 1$ wird es also mit Wahrscheinlichkeit $\Delta_k = \bar{\alpha}_{t_{k+1}} - \bar{\alpha}_{t_k}$ aufgedeckt, unabhängig von den anderen Positionen.

<!--en:3b3d2d0f-->
Die Vorhersage des Netzes ist ein Produkt über die Positionen. Decken wir mehrere Token im selben Sprung auf, ziehen wir sie daher unabhängig voneinander, jedes aus seiner eigenen vorhergesagten Verteilung. Token aus verschiedenen Sprüngen ziehen wir dagegen nacheinander, jeweils bedingt auf die bereits aufgedeckten Token. Was kostet diese Unabhängigkeit? Beginnen wir mit dem kleinsten Fall, in dem sie überhaupt einen Unterschied machen kann: zwei Token.

<!--en:4ce8897c-->
Unser durchgängiges Beispiel liefert uns ein solches Paar. Es ist die Mischung dreier Gauß-Verteilungen in der Ebene mit Mittelwerten $(-2.5, -1.5)$, $(2.5, -1.5)$ und $(0, 2.5)$, Gewichten $0.5$, $0.3$ und $0.2$ sowie Standardabweichung $0.5$ (:numref:`sec_diffusion-denoising`). Wir quantisieren die Ebene auf einem $32 \times 32$-Gitter über $[-5, 5]^2$. Jedes Sample wird so zu einem Tokenpaar, den beiden Zellindizes entlang der Achsen. Die Tabelle $p$ der Paare zeigt drei Häufungen, eine pro Modus. Ihre Randverteilungen $p^1$ und $p^2$ haben entlang der ersten Achse drei Gipfel, entlang der zweiten aber nur zwei, weil zwei Modi auf gleicher Höhe liegen. Das Produkt der Randverteilungen, $p^1 \otimes p^2$, zieht die beiden Token unabhängig. An jedem Schnittpunkt zweier Gipfel entsteht eine Häufung: insgesamt sechs, drei davon an Stellen ohne Daten.

<!--en:9229e90e-->
Betrachten wir nun ein *exaktes* Netz. Seine Vorhersage an jeder maskierten Position ist die wahre bedingte Verteilung des Tokens bei gegebenen unmaskierten Token. Bei $S$ gleich großen Sprüngen nach dem linearen Zeitplan gilt $\Delta_k = 1/S$ für jeden Sprung. Beide Token werden also mit Wahrscheinlichkeit $S \cdot (1/S)^2 = 1/S$ im selben Sprung aufgedeckt. Dann stammt das Paar aus $p^1 \otimes p^2$. Andernfalls ziehen wir ein Token aus seiner Randverteilung und das andere später aus seiner bedingten Verteilung. Das Paar stammt dann aus $p$. Der Sampler zieht also aus der Mischung

<!--en:98d9a40f-->
Als Abstand verwenden wir den Totalvariationsabstand, die halbe Summe der absoluten Differenzen zwischen den Einträgen zweier Wahrscheinlichkeitstabellen. Er beträgt für diese Mischung $\mathrm{TV}(p^1 \otimes p^2, p)/S$ gegenüber $p$. Der Fehler fällt umgekehrt proportional zur Schrittzahl und wächst mit der Abhängigkeit zwischen den Token.

<!--en:aa78435a-->
Den Fehler bei einem Schritt können wir schon vor der Rechnung abschätzen. Auf die drei unechten Häufungen der Produktverteilung entfällt $0.5 \cdot 0.2 + 0.3 \cdot 0.2 + 0.2 \cdot 0.8 = 0.32$ der Masse. Ohne Überlappung der Häufungen wäre dies der Totalvariationsabstand. Die Zellwahrscheinlichkeiten faktorisieren in eindimensionale gaußsche Integrale. Daher berechnet die folgende Codezelle $p$ exakt, zusammen mit der Entropie, der gegenseitigen Information der beiden Token und dem Fehler des Samplers eines exakten Netzes für vier Schrittzahlen.

<!--en:62cb2dd9-->
Der Einschritt-Sampler platziert $0.314$ der Masse falsch. Das ist etwas weniger als die vorhergesagten $0.32$, weil sich benachbarte Häufungen leicht überlappen. Mit 32 Schritten sinkt der Fehler auf $0.010$.

<!--en:bee37495-->
### Aufdecken nach Konfidenz

<!--en:9db219f4-->
Viele Sampler decken Token in der Praxis in einer gezielt gewählten Reihenfolge auf. MaskGIT :cite:`Chang.Zhang.Jiang.ea.2022` und das Diffusionssprachmodell LLaDA :cite:`Nie.Zhu.You.ea.2025` ziehen an jeder maskierten Position einen Kandidaten. Sie behalten die Kandidaten mit der höchsten Wahrscheinlichkeit unter ihrer jeweils vorhergesagten Verteilung und maskieren die übrigen erneut. Bei dieser *erneuten Maskierung nach Konfidenz* (engl. confidence remasking) bestimmen also die gezogenen Werte, welche Token aufgedeckt werden. Ist das besser als eine zufällige Reihenfolge? Die Tabelle für zwei Token liefert die Antwort ganz ohne Training. Das Verfahren zieht beide Token aus ihren Randverteilungen $A \sim p^1$ und $B \sim p^2$. Es behält den Wert mit der höheren Wahrscheinlichkeit unter seiner eigenen Randverteilung. Das andere Token zieht es erneut, nun aus der bedingten Verteilung bei gegebenem beibehaltenem Wert. Die Ausgabetabelle lautet

<!--en:2dd998ac-->
wobei der erste Term den gezogenen Wert $a$ des ersten Tokens beibehält und der zweite den Wert $b$ des zweiten Tokens. Die Codezelle vergleicht diese Tabelle mit den Tabellen für einen Schritt, für zwei zufällige Sprünge und für eine feste Reihenfolge (erstes Token aus seiner Randverteilung, zweites aus seiner bedingten Verteilung).

<!--en:1d3c8edf-->
Sind die bedingten Verteilungen exakt und decken wir pro Schritt ein Token auf, so genügt es, die Reihenfolge vor der Ziehung festzulegen. Ob von links nach rechts oder zufällig: Das Ergebnis ist exakt, wie das Beispiel der festen Reihenfolge zeigt. Zwei zufällige Sprünge decken in der Hälfte der Fälle beide Token gemeinsam auf und halbieren den Einschrittfehler. Überraschenderweise führt die erneute Maskierung nach Konfidenz weiter von $p$ weg als zwei zufällige Sprünge: Der Abstand beträgt $0.179$ gegenüber $0.157$. Auch sie deckt ein Token pro Schritt auf, wählt die Reihenfolge aber anhand der gezogenen Werte. Die Formel zeigt, warum das schadet. Ein Wert aus einem kleinen Modus hat eine geringe Randwahrscheinlichkeit und verliert daher oft den Vergleich. Dann wird er erneut gezogen, bedingt auf den Gewinner, was selten zum kleinen Modus zurückführt. Der Sampler bevorzugt die ausgeprägten Modi: Das Gewicht des kleinsten Modus fällt von $0.2$ auf etwa $0.07$, die Entropie von $4.84$ auf $4.41$ Nat. Bei Text wirkt eine geringere Entropie oft wie größere sprachliche Flüssigkeit. Bei einem unvollkommenen Netz kann es helfen, die einfachen Positionen zuerst aufzudecken. Die Schranke beschreibt diesen Sampler aber nicht und kann seine Verzerrung nicht erkennen. Sie bewertet Vorhersagen unter zufälligen Masken und beschreibt damit den Sampler mit zufälliger Reihenfolge.

<!--en:87124d53-->
## Trainierte Entrauscher

<!--en:dca92cf6-->
Bisher stammten alle Zahlen aus einer exakten Tabelle. Nun trainieren wir Netze, zunächst für das Tokenpaar, bei dem wir jeden Fehler seiner Ursache zuordnen können, dann für Bilder.

<!--en:cefbc8a2-->
### Die quantisierte Mischung

<!--en:68f906cf-->
Das Netz für das Tokenpaar erhält die One-Hot-Kodierung beider Token mit der Maske als 33. Wert am Index $K$ sowie die Zeit $t/T$. Für jede Position gibt es Logits für die $32$ ursprünglichen Werte aus. Der lineare Zeitplan umfasst $T = 32$ Schritte.

<!--en:7c0e1ab8-->
Wir trainieren es mit `d2l.train_regression` (:numref:`sec_diffusion-denoising`) auf frisch gezogenen Samples aus der Mischung. Jeder Batch enthält 512 Paare, für die wir jeweils einen Schritt gleichverteilt ziehen. Die Lernrate fällt über 6000 Aktualisierungen entlang einer Kosinuskurve.

<!--en:bd4e1baf-->
Bei zwei Positionen lässt sich der Erwartungswert in der Schranke exakt berechnen. In jedem Schritt gibt es drei Maskenmuster mit mindestens einer Maske. Ein Muster mit $m$ Masken hat die Wahrscheinlichkeit $u^m (1 - u)^{2 - m}$, wobei $u = 1 - \bar{\alpha}_t$ gilt. Die 1024 Paare gewichten wir gemäß der Tabelle. Dieser Mittelwert kann die Entropie der Tabelle nicht unterschreiten, denn sie ist die kleinste mittlere negative Log-Likelihood, die ein Modell erreichen kann (:numref:`sec_mdl-information_theory`). Wie nahe kommt ihr ein Netz?

<!--en:06450b94-->
Rechnen wir das für ein exaktes Netz durch. Mit $X^1$ und $X^2$ bezeichnen wir die beiden Token eines aus der Tabelle gezogenen Paars. Die Schranke bewertet jedes Token in dem Schritt, in dem die Rückwärtskette es aufdeckt: Das Gewicht $w_t$ mal der Wahrscheinlichkeit $1 - \bar{\alpha}_t$, dass das Token im Schritt $t$ maskiert ist, ergibt $\bar{\alpha}_{t-1} - \bar{\alpha}_t$. Das ist die Wahrscheinlichkeit, dass die Rückwärtskette es im Schritt $t$ aufdeckt. Werden die Token in verschiedenen Schritten aufgedeckt, liefert die Randverteilung die Vorhersage für das erste, die bedingte Verteilung bei gegebenem ersten die für das zweite. Zusammen kosten sie genau $H(X^1) + H(X^2 \mid X^1) = H(X^1, X^2)$. Werden beide Token im selben Schritt aufgedeckt, liefert jede Randverteilung unabhängig ihre Vorhersage. Das kostet $H(X^1) + H(X^2)$ statt $H(X^1, X^2)$. Die Mehrkosten entsprechen der gegenseitigen Information $I(X^1; X^2)$ (:numref:`sec_mdl-mutual-information`). Beim linearen Zeitplan wird jedes Token in einem bestimmten Schritt mit Wahrscheinlichkeit $1/T$ aufgedeckt. Beide fallen also mit Wahrscheinlichkeit $T \cdot (1/T)^2 = 1/T$ in denselben Schritt, und ein exaktes Netz erreicht

<!--en:7304284a-->
(In Aufgabe :exref:`ex_discrete-diffusion-revealed-together` wird daraus ein Beweis für jeden Zeitplan.) Kein Netz, das die Positionen unabhängig vorhersagt, kann diese Untergrenze von $4.840 + 0.479/32 \approx 4.855$ Nat unterschreiten. Für $T \to \infty$ werden keine zwei Token mehr im selben Schritt aufgedeckt, und die Untergrenze sinkt auf die Entropie.

<!--en:e0b5d0a8-->
Die Schranke liegt weniger als zwei Hundertstel Nat über der Entropie. Das Modell hat die Randverteilung jedes Tokens und seine bedingte Verteilung bei gegebenem anderen Token gelernt. Fast der gesamte Abstand geht auf die Untergrenze zurück. Der zusätzliche Beitrag des Netzes liegt in der Größenordnung eines Tausendstel Nat. Die Untergrenze ergibt sich nicht allein daraus, dass die Schranke Spielraum lässt: Der Sampler eines exakten Netzes modelliert bei 32 Schritten $(1 - 1/32)\, p + (1/32)\, p^1 \otimes p^2$ anstelle von $p$. Schranke und Sampler zeigen also denselben Verlust an Information.

<!--en:9c2e5bc8-->
Was verliert der trainierte Sampler? Bei zwei Token können wir seine Ausgabetabelle durch Summieren exakt bestimmen. Jedes Token wird im Sprung $k$ mit Wahrscheinlichkeit $\Delta_k$ (`pi` im Code) aufgedeckt, unabhängig vom anderen. Gleichzeitig aufgedeckte Token ziehen wir aus den beiden Randverteilungen, die das Netz auf dieser Stufe vorhersagt. Ein später aufgedecktes Token ziehen wir aus seiner bedingten Verteilung bei gegebenem ersten Token. Die Funktion `conditionals` liest diese Vorhersagen aus einem Modell für Tokenpaare aus, und `sampler_table` summiert über die Sprünge. Mit den wahren Randverteilungen und bedingten Verteilungen muss die Rechnung die zuvor ausgegebenen Fehler des exakten Netzes reproduzieren.

<!--en:5e326cb0-->
Für eine Funktion `cond` mit der Signatur von `conditionals` summiert `sampler_table` über alle Kombinationen von Sprüngen. So erhalten wir die exakte Ausgabeverteilung des Samplers, der in $S$ Schritten demaskiert. Wir vergleichen sie mit der wahren gemeinsamen Verteilung `table`, sowohl für `pair_model` als auch für `exact` und jeweils für $S \in \{32, 8, 2, 1\}$:

<!--en:387b6941-->
Die Spalte für das exakte Netz reproduziert $0.314/S$. Der trainierte Sampler folgt ihr bei jeder Schrittzahl. Der Eigenfehler des Netzes, also der Abstand zwischen trainierter und exakter Tabelle, beträgt etwa $0.01$ für jedes $S$. Bei 32 Schritten verdoppelt dieser Fehler den Abstand ($0.020$ gegenüber $0.010$), bei einem oder zwei Schritten fällt er kaum ins Gewicht. Nur ein besseres Netz kann ihn beseitigen.

<!--en:96a67ca5-->
Die folgenden Diagramme zeigen 2000 Datenpaare und je 2000 mit `unmask` gezogene Paare pro Sampler. Jedes Paar liegt an einer zufälligen Stelle innerhalb seiner Zelle. Der Einschritt-Sampler füllt die drei unechten Häufungen dort, wo die Spalte eines Modus die Zeile eines anderen kreuzt.

<!--en:5eee50d2-->
### Bilder als binäre Token

<!--en:a21e47ef-->
An zwei Token erkennen wir den Mechanismus, an einem Bild sein Verhalten in größerem Maßstab. Maskierte Diffusion für Bilder arbeitet mit den diskreten Codes eines gelernten Tokenisierers :cite:`Chang.Zhang.Jiang.ea.2022`. Hier vertreten binäre Pixel diese Codes, weil sie keinen Tokenisierer benötigen. Mit einem Schwellenwert bei halber Intensität machen wir jedes Fashion-MNIST-Pixel zu einem Token mit zwei Werten. Ein maskiertes Pixel hat einen dritten Wert. Ein Bild ist dann eine Sequenz aus 784 Token auf einem Gitter. Damit eignet sich das U-Net dieses Kapitels (:numref:`sec_diffusion-images`): Es erhält die One-Hot-Kodierung über drei Eingabekanäle und liefert die Logits des ursprünglichen Pixels über zwei Ausgabekanäle. Verlust und Sampler übernehmen wir vom Tokenpaar, mit einem linearen Zeitplan über $T = 100$ Schritte und der Maske am Index 2.

<!--en:f8b0eec9-->
Das Training umfasst 6000 Aktualisierungen mit Batches von 128 Bildern und konstanter Lernrate. Für die Gewichte verwenden wir einen gleitenden Mittelwert mit einer Halbwertszeit von 1000 Aktualisierungen.

<!--en:35b02d21-->
Wir schätzen die Schranke auf 1000 zurückgehaltenen Bildern, mit einer zufälligen Maske pro Bild und Stufe. Zum Vergleich berechnen wir die Kreuzentropie eines Modells, das Pixel als unabhängig behandelt. Es nutzt dazu die Häufigkeit weißer Pixel an jeder Position im Trainingsdatensatz.

<!--en:72143753-->
Die Schranke beträgt etwa 0.17 Bit pro Pixel gegenüber etwa 0.70 bei unabhängigen Pixeln. Als Codelänge sind das rund 135 statt 550 Bit für ein zurückgehaltenes Bild eines Kleidungsstücks. Unabhängige Pixel sind zugegebenermaßen ein schwacher Maßstab. Fast jedes Modell, das die Nachbarn eines Pixels berücksichtigt, schneidet deutlich besser ab. Der Versuch zeigt aber, wie groß der Unterschied ist: Ein Modell der Abhängigkeiten zwischen Pixeln spart mindestens drei Viertel der Codelänge für unabhängige Pixel ein. Das Netz hat genug über diese Abhängigkeiten gelernt, um diese Einsparung zu erreichen. Ein fairer Vergleich wäre ein gleich großes Modell, das die Pixel einzeln in Rasterreihenfolge aufdeckt. Ein solches Modell trainieren wir hier nicht. Wie maskierte Diffusion und Links-nach-rechts-Modelle bei Text abschneiden, besprechen wir in :numref:`subsec_diffusion-autoregression`.

<!--en:26ba011e-->
Der Sampler zieht 1000 Bilder in 100 Schritten. Das Raster zeigt die ersten 48.

<!--en:adc0cdf4-->
Die meisten Samples sind als binäre Bilder von Kleidungsstücken erkennbar, wirken aber gröber als die Graustufenbilder aus den vorigen Abschnitten. Etwa jedes fünfte zeigt nur ein Fragment oder ist fast leer. Nach 6000 Aktualisierungen ist das U-Net noch nicht ausreichend trainiert. Für unser Experiment genügt das, denn wir untersuchen die Schranke und die Schrittzahl. Die bestmögliche Darstellung der Kleidungsstücke ist hier zweitrangig. Jeder der 100 Sprünge deckt etwa acht Pixel auf. Ihre Vorhersagen berücksichtigen alle in früheren Sprüngen aufgedeckten Pixel. Das nächste Raster verwendet vier Sprünge mit je etwa 200 Pixeln.

<!--en:dd641660-->
Kein Kleidungsstück bleibt erkennbar. Der erste Sprung zieht seine Pixel unabhängig aus Vorhersagen für ein vollständig maskiertes Bild. Diese geben nur an, wo sich Kleidungsstücke typischerweise befinden. Jeder spätere Sprung ist auf dieses Rauschen bedingt. Übrig bleibt ein mit Textur gefüllter Umriss, der häufiger einem durchschnittlichen Kleidungsstück ähnelt als einem bestimmten.

<!--en:423a8961-->
Um den Fehler bei vier Schritten zu messen, zählen wir *isolierte Pixel* (engl. speckles), deren Wert sich von dem aller vier Nachbarn unterscheidet. Für jede Variante bestimmen wir ihren Anteil an 1000 Bildern und geben den Standardfehler über die 1000 Bilder an. Ein dritter Batch verwendet 784 Sprünge, im Mittel ein Pixel pro Sprung. Er zeigt, was bei 100 Schritten noch verloren geht. Die 100 Stufen der Kette reichen für so viele Sprünge nicht aus: `unmask` rundet seine Stufen auf die der Kette, und ein Sprung zwischen zwei gleichen Stufen deckt nichts auf. Für diese Sprünge verwenden wir daher einen eigenen linearen Zeitplan mit 784 Stufen. Die meisten ihrer Zeitpunkte $k/784$ liegen zwischen den 100 Trainingszeitpunkten, die letzten sieben unterhalb des kleinsten. Die Anzahl maskierter Pixel gibt dem Netz ohnehin Aufschluss über die Zeit.

<!--en:630b8e2c-->
In den Daten kommen auf tausend innere Pixel etwa vier isolierte Pixel, in den Samples mit vier Schritten etwa neunmal so viele. Sie entsprechen den unechten Häufungen beim Tokenpaar mit einem Schritt. Die Samples mit 100 Schritten enthalten etwa doppelt so viele isolierte Pixel wie die Daten. Hier kommen zwei Ursachen zusammen: Pro Sprung erscheinen noch etwa acht Pixel zugleich, und nach 6000 Aktualisierungen macht das Netz selbst Fehler. Die Zeile mit 784 Schritten erlaubt uns, beide Ursachen auseinanderzuhalten. Wie beim Tokenpaar verringern mehr Sprünge den Unabhängigkeitsfehler, während der Eigenfehler des Netzes bestehen bleibt. Bei im Mittel einem Pixel pro Sprung verschwinden zwei Drittel bis drei Viertel des Überschusses gegenüber den Daten, den wir bei 100 Schritten beobachtet haben. Könnte allein die zufällige Auswahl der Bilder diesen Rückgang erklären? Beide Zeilen stammen vom selben Netz. Die jeweils 1000 gezogenen Bilder sind daher die einzige Zufallsquelle im Vergleich. Jeder Anteil ist ein Mittelwert über unabhängige Bilder mit einem Standardfehler von 0.00016 bis 0.0002. Für die Differenz zweier solcher Mittelwerte beträgt der Standardfehler etwa 0.00026. Der Rückgang ist rund zehnmal so groß. Der größte Teil des Fehlers bei 100 Schritten geht somit auf den Unabhängigkeitsfehler zurück. Bei 784 Schritten enthalten die Samples noch etwa 1.3-mal so viele isolierte Pixel wie die Daten. Der Abstand beträgt immer noch etwa vier Standardfehler der Differenz. Dieser Rest setzt sich aus dem Eigenfehler des Netzes und dem kleinen verbleibenden Unabhängigkeitsfehler zusammen, denn ein Pixel teilt seinen Sprung nur selten mit einem Nachbarn. Aufgabe :exref:`ex_discrete-diffusion-pairs-revealed`e trennt die beiden Beiträge.

<!--en:e449282d-->
## Verwandte Modelle und Rückblick

<!--en:ebfe418b-->
Bisher haben wir die maskierte Diffusion für sich betrachtet. Eng verwandt sind die Sprachmodelle aus :numref:`sec_gpt` und :numref:`sec_transformer`. Ihre Grundlagen hat die maskierte Diffusion in den vorangegangenen Abschnitten dieses Kapitels.

<!--en:3e62b17a-->
### Diffusion und Autoregression

<!--en:c39c2aeb-->
Wer das Sprachmodell aus :numref:`sec_gpt` trainiert hat, wird sich fragen, wie es mit unserem Modell zusammenhängt. Die Tabellen für zwei Token geben die Antwort, wenn wir die Sampler danach ordnen, wie viele Token sie pro Schritt aufdecken.

<!--en:ab02578d-->
Decken wir pro Schritt ein Token in fester Reihenfolge von links nach rechts auf, so hat der Sampler die Form des Sprachmodells aus :numref:`sec_gpt`. Seine bedingten Verteilungen sind die Vorhersagen des Netzes für Eingaben mit unmaskiertem Präfix und maskiertem Rest. Für das Tokenpaar entspricht dies der Zeile für die feste Reihenfolge in der Tabelle der vier Verfahren. Der Abstand beträgt $0.000$ gegenüber $p$: Sind die bedingten Verteilungen exakt, so ist auch die Kette exakt, die jeweils ein Token aufdeckt, und ihr Verlust ist die Entropie.

<!--en:2e15e121-->
Decken wir ein Token pro Schritt in einer gleichverteilt zufälligen Reihenfolge auf, so ist der Sampler ein *autoregressives Modell für beliebige Reihenfolgen* :cite:`Uria.Murray.Larochelle.2014`. Sein Verlust mittelt die negativen Log-Likelihoods aller Reihenfolgen. Wie :citet:`Hoogeboom.Gritsenko.Bastings.ea.2022` zeigten, ist dieser Verlust die Schranke in kontinuierlicher Zeit aus :numref:`subsec_diffusion-masked-ce` für ein Netz, das die Zeit ignoriert (Aufgabe :exref:`ex_discrete-diffusion-order-agnostic` leitet sie mit einem Betaintegral her). Dabei geht, wie :citet:`Ou.Nie.Xue.ea.2025` zeigten, nichts verloren: Die beste Vorhersage für ein maskiertes Token hängt nur von den bereits aufgedeckten Token ab. Ein maskiertes Diffusionsmodell wird also als Sprachmodell für alle Reihenfolgen zugleich trainiert. Bei einem exakten Netz ergibt jede Reihenfolge dieselbe gemeinsame Likelihood. Die erwartete negative Log-Likelihood ist damit die Entropie der Daten, und die Schranke in kontinuierlicher Zeit ist scharf. So lässt sich die Untergrenze für das Tokenpaar anhand der Reihenfolgen erklären: Für $T \to \infty$ werden keine zwei Token gemeinsam aufgedeckt, und der Mehrbetrag $I/T$ verschwindet.

<!--en:302980fc-->
Decken wir mehrere Token pro Schritt auf, entsteht beim Sampling der Unabhängigkeitsfehler. Bei einem Paar ziehen $S$ Sprünge einen Anteil $1/S$ der Masse aus dem Produkt der Randverteilungen. Die Schranke für $T$ Schritte steigt um $I/T$. Dafür genügen $S$ Netzauswertungen, um $L$ Token zu erzeugen, wo ein Links-nach-rechts-Modell $L$ benötigt.

<!--en:6510aee3-->
Allerdings ist der Aufwand pro Auswertung verschieden. Ein Links-nach-rechts-Modell, das seine Schlüssel und Werte zwischenspeichert (:numref:`sec_kv-cache`), verarbeitet pro Schritt ein neues Token. Der Entrauscher hat keine kausale Maske. Ein aufgedecktes Token verändert daher die Aktivierungen an jeder Position. Nichts davon lässt sich zwischenspeichern, und jede Auswertung durchläuft alle $L$ Positionen. Zählen wir die Tokendurchläufe, betragen die Kosten ungefähr $L$ gegenüber $S L$. Paralleles Dekodieren spart also sequenzielle Schritte und verringert damit die Latenz. Der gesamte Rechenaufwand ist meist höher. Fast-dLLM :cite:`Wu.Zhang.Xue.ea.2026` spart einen Teil davon ein, indem es näherungsweise gültige Cache-Einträge über mehrere Schritte wiederverwendet. Blockdiffusion :cite:`Arriola.Gokaslan.Chiu.ea.2025` dekodiert dazu Block für Block und nutzt einen exakten Cache für die fertigen Blöcke.

<!--en:5ca15a75-->
Am anderen Ende steht BERT :cite:`Devlin.Chang.Lee.ea.2018`. Es verbirgt 15% der Token eines Textes und sagt sie aus dem Kontext auf beiden Seiten vorher. Das maskierte Sprachmodell aus :numref:`sec_transformer` geht ebenso vor und maskiert jedes Byte mit Wahrscheinlichkeit 15%. Beschränken wir den Verlust :eqref:`eq_diffusion-masked-loss` auf diese eine Maskierungsrate, erhalten wir die Zielfunktion beider Modelle: die Summe der negativen Log-Likelihoods der verborgenen Token. Eine Besonderheit bleibt: BERT ersetzt manche Masken durch zufällige oder unveränderte Token. Ein maskiertes Sprachmodell ist somit ein diskretes Diffusionsmodell mit einem Schritt :cite:`Austin.Johnson.Ho.ea.2021`, das allerdings nur eine Maskierungsrate kennt. Ein Sampler, der mit der vollständig maskierten Sequenz beginnt, würde ihm Eingaben zeigen, die sich stark von sämtlichen Trainingseingaben unterscheiden.

<!--en:969f7786-->
Zwischen diesen Extremen liegen heutige Diffusionssprachmodelle. Ihr Entrauscher ist ein Transformer ohne kausale Maske, mit einem zusätzlichen Eintrag für $[\textrm{M}]$ im Vokabular und ohne Zeiteingabe :cite:`Nie.Zhu.You.ea.2025`. Was gewinnen sie damit? Bei reichlich Daten ergibt sich kein Vorteil für die Likelihood. Stehen durchgehend frische Daten zur Verfügung, braucht die maskierte Diffusion für denselben Verlust um einen konstanten Faktor mehr Rechenaufwand als ein autoregressives Modell :cite:`Nie.Zhu.Du.ea.2025`. Sind Daten knapp und werden häufig wiederverwendet, kann sie besser abschneiden :cite:`Prabhudesai.Wu.Zadeh.ea.2025`. Die Autoren führen dies darauf zurück, dass die zufälligen Masken als Datenaugmentierung wirken. Für diese Modelle sprechen das parallele Dekodieren und das Füllen von Textlücken. Ersteres schreibt viele Token pro Netzauswertung und benötigt dadurch weniger sequenzielle Schritte. Beim Füllen einer Lücke lässt sich der Text auf beiden Seiten als Bedingung nutzen.

<!--en:f855e60b-->
### Diffusionsmodelle im Rückblick

<!--en:e235f60e-->
Mit Ausnahme der weiterführenden Literatur endet hier das Kapitel und damit der Teil für Fortgeschrittene. Blicken wir zurück. Ein Gedanke zog sich durch alle Abschnitte: Ein Netz, das bei jedem Rauschpegel aus einem verrauschten Punkt den rauschfreien Punkt vorhersagt, ist ein Generator. Ein Sampler integriert die Vorhersagen und erzeugt so Daten. Bei Token wurde aus dem verrauschten Punkt eine teilweise gelöschte Sequenz, aus der Vorhersage eine Verteilung über die gelöschten Werte und aus der Integration das schrittweise Aufdecken. Die Schranke ließ sich unverändert übertragen.

<!--en:63601d02-->
Jeder Teil dieses Ansatzes hat einen Nachteil, und in der Praxis begegnen uns alle. Für die Dichte erhalten wir eine Schranke. Die Gewichtung mit der besten Schranke unterscheidet sich von der mit den besten Samples (:numref:`sec_diffusion-design-space`). Eine Regression ersetzt das adversarielle Spiel aus :numref:`chap_gans`, aber beim Sampling werten wir das Netz viele Male aus: tausendmal für die ursprüngliche DDPM-Kette, einige Dutzend Mal mit einem guten Löser (:numref:`sec_diffusion-samplers`). Eine Handvoll Auswertungen genügt erst nach einem zweiten Trainingslauf, in dem wir ein Schülermodell destillieren (:numref:`sec_diffusion-few-step`). Ein latenter Raum macht große Bilder erschwinglich, erfordert aber ein zweites Modell, den Autoencoder. Diesen trainieren wir zuerst und frieren ihn dann ein (:numref:`sec_diffusion-latent`). Bei Text bringt das Erzeugen vieler Token pro Auswertung den Unabhängigkeitsfehler mit sich. Der Sampler kann einmal geschriebene Token nicht zurücknehmen.

<!--en:9f6a4a6e-->
Mehrere Fragen bleiben offen. Schülermodelle mit wenigen Schritten, die entlang der Trajektorien ihres Lehrermodells trainieren, bleiben hinter ihm zurück. Schülermodelle, die ihre Lehrermodelle erreichen oder übertreffen, ergänzen den Verteilungsabgleich um einen Kritiker und geben die Abbildung vom Rauschen zum Sample auf (:numref:`sec_diffusion-few-step`). Bisher ist keine Gewichtung bekannt, die zugleich die beste Likelihood und die besten Samples liefert. Bleibt der Unterschied im Rechenaufwand zwischen maskierter Diffusion und Autoregression bei Text auch in den größten Maßstäben bestehen? Darüber entscheiden Experimente, die weit über die Größenordnung in diesem Buch hinausgehen. Der Teil über Sprachmodelle greift Diffusionssprachmodelle in seinem Überblick über Architekturen wieder auf (:numref:`sec_lm_architectures_fr`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:149affee-->
Diskrete Diffusion verrauscht kategoriale Token mithilfe zeilenstochastischer Übergangsmatrizen $\mathbf{Q}_t$. Beim absorbierenden Übergangskern übersteht jedes Token $t$ Schritte mit Wahrscheinlichkeit $\bar\alpha_t = \prod_{s \le t}(1 - \beta_s)$. Andernfalls wird es durch $[\textrm{M}]$ ersetzt. Die Variationsschranke für die negative Log-Likelihood vereinfacht sich zu einer gewichteten Summe der Kreuzentropien maskierter Token mit Gewicht $w_t = (\bar\alpha_{t-1} - \bar\alpha_t)/(1 - \bar\alpha_t)$ ($1/t$ beim linearen Zeitplan). Ihr Grenzwert in kontinuierlicher Zeit ist $\int_0^1 \frac{1}{u}\, \mathbb{E}_u[\ell]\, du$. Da die Maskierung unabhängig erfolgt, hängt die optimale Verteilung der ursprünglichen Tokenwerte an einer maskierten Position nicht von der Zeit ab. Das Rückwärtsmodell faktorisiert in jedem Schritt über die Positionen. Daher verursacht das gleichzeitige Demaskieren mehrerer Token einen Unabhängigkeitsfehler. Bei einer Verteilung $p$ über zwei Token ziehen $S$ gleich große Sprünge im linearen Zeitplan aus $(1 - 1/S)p + (1/S)p^1 \otimes p^2$. Der Totalvariationsabstand ist proportional zu $1/S$, und die Untergrenze der Schranke für $T$ Schritte übersteigt die Entropie $H(X^1, X^2)$ um $I(X^1; X^2)/T$. Über die Wahl der Reihenfolge und der Maskierungsrate verbindet die maskierte Diffusion klassische Sequenzmodelle. Decken wir pro Schritt ein Token von links nach rechts auf, erhalten wir die autoregressive Modellierung (:numref:`sec_gpt`). Bei zufälliger Reihenfolge ergibt sich die Autoregression in beliebiger Reihenfolge. Trainieren wir mit einer einzigen Maskierungsrate $u = 0.15$, erhalten wir BERT (:numref:`sec_transformer`).

<!--en:2294243a-->
## Aufgaben

<!--en:26f0478c-->
1. **Der Gleichverteilungskern.** Die Vorwärtskette verwende den Gleichverteilungskern $\mathbf{Q}_t = (1 - \beta_t)\, \mathbf{I} + \beta_t\, \tfrac{1}{K}\mathbf{1}\mathbf{1}^\top$ mit $\beta_t > 0$ in jedem Schritt.
    1. Zeigen Sie, dass $\bar{\mathbf{Q}}_t = \bar{\alpha}_t\, \mathbf{I} + (1 - \bar{\alpha}_t)\, \tfrac{1}{K}\mathbf{1}\mathbf{1}^\top$ gilt und dass $\mathbf{x}_t = \mathbf{x}_0$ mit Wahrscheinlichkeit $\bar{\alpha}_t + (1 - \bar{\alpha}_t)/K$ eintritt. Ein Token, dessen Wert im Schritt $t$ sichtbar ist, kann also neu gezogen worden sein. Welche Frage muss ein Netz bei diesem Kern an jeder Position beantworten, die sich beim absorbierenden Übergangskern nicht stellt?
    1. Leiten Sie die A-posteriori-Verteilung $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ aus :eqref:`eq_diffusion-discrete-posterior` her. Behandeln Sie die Fälle $\mathbf{x}_t = \mathbf{x}_0$ und $\mathbf{x}_t \neq \mathbf{x}_0$ getrennt. Zeigen Sie, dass für $t \geq 2$ jeder der $K$ Werte positive Wahrscheinlichkeit hat, im Gegensatz zur A-posteriori-Verteilung :eqref:`eq_diffusion-masked-posterior` mit nur zwei möglichen Werten.
    1. Die Divergenzterme der Schranke vergleichen diese A-posteriori-Verteilung an jeder Position mit dem Rückwärtsmodell. Warum ist ihre Auswertung aufwendiger als die des Summanden in :eqref:`eq_diffusion-masked-loss`?
   :exlabel:`ex_discrete-diffusion-uniform-kernel`

<!--en:5c2cb4c8-->
1. [code] **Gemeinsam aufgedeckte Tokenpaare.** Der Sampler verwende die Stufen $T = t_0 > t_1 > \cdots > t_S = 0$ mit einem Zeitplan, für den $\bar{\alpha}_0 = 1$ und $\bar{\alpha}_T = 0$ gelten. Mit $\Delta_k = \bar{\alpha}_{t_{k+1}} - \bar{\alpha}_{t_k}$ bezeichnen wir den Zuwachs von $\bar{\alpha}$ im Sprung $k$, wobei $k = 0, \ldots, S - 1$ gilt.
    1. In der gröberen Kette auf den Stufen $t_k$ wird ein Token im Sprung $k$ mit Wahrscheinlichkeit $\Delta_k$ aufgedeckt, unabhängig von den anderen Positionen. Zeigen Sie, dass zwei Token mit Wahrscheinlichkeit $\rho = \sum_k \Delta_k^2$ im selben Sprung aufgedeckt werden. Zeigen Sie ferner, dass unter allen Folgen aus $S$ Zuwächsen mit Summe eins gleich große Zuwächse $\Delta_k = 1/S$ den Wert von $\rho$ minimieren.
    1. Zeigen Sie, dass der Sampler mit einem exakten Netz die beiden Token der quantisierten Mischung aus $(1 - \rho)\, p + \rho\, p^1 \otimes p^2$ zieht.
    1. Berechnen Sie für eine Sequenz aus $L$ Token und gleich große Zuwächse die erwartete Anzahl von Tokenpaaren, die im selben Sprung aufgedeckt werden. Wie viele Schritte sind nötig, damit sie für $L = 1024$ unter eins liegt? Warum überschätzt diese Anzahl den Fehler des Samplers bei Text?
    1. Deckt ein Sprung kein Token auf, bleibt die Eingabe für den nächsten Sprung unverändert. Ignoriert das Netz die Zeit, wie das exakte Netz nach Aufgabe :exref:`ex_discrete-diffusion-time-agnostic`a, kann der Sampler daher vorab auslosen, welche Positionen er wann aufdeckt. Das Netz muss er dann nur in Sprüngen auswerten, die mindestens ein Token aufdecken. Zeigen Sie, dass bei $S$ gleich großen Sprüngen im linearen Zeitplan im Mittel $S\, (1 - (1 - 1/S)^L)$ Auswertungen für $L$ Token nötig sind. Berechnen Sie diesen Wert für das Paar mit $S = 32$, für die binären Bilder mit $S = 100$ und $S = 784$ sowie für $L = S = 1024$.
    1. In den Samples mit 784 Schritten aus :numref:`subsec_diffusion-binary-images` bleibt ein Viertel bis ein Drittel der überschüssigen isolierten Pixel übrig, die wir bei 100 Schritten beobachtet haben. Zeigen Sie: Bei $S$ gleich großen Sprüngen fällt ein inneres Pixel mit mindestens einem seiner vier Nachbarn mit Wahrscheinlichkeit $1 - (1 - 1/S)^4$ in denselben Sprung. Berechnen Sie diese Wahrscheinlichkeit für $S = 4$, $100$ und $784$. Trainieren Sie dann `unet` mit doppelt so vielen Aktualisierungen und zählen Sie die isolierten Pixel bei 100 und 784 Sprüngen erneut. Welche Anzahl sinkt? Was sagt das darüber aus, welcher Anteil des verbleibenden Überschusses auf das Netz zurückgeht?
   :exlabel:`ex_discrete-diffusion-pairs-revealed`

<!--en:33f94ad5-->
1. **Zwei Parametrisierungen des Rückwärtsübergangs.** ● Das Rückwärtsmodell :eqref:`eq_diffusion-x0-parameterization` mittelt die A-posteriori-Verteilung über die Vorhersage des Netzes. :citet:`Austin.Johnson.Ho.ea.2021` setzen stattdessen

   $$p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \propto \sum_{\hat{\mathbf{x}}_0} q(\mathbf{x}_{t-1}, \mathbf{x}_t \mid \hat{\mathbf{x}}_0)\, \tilde{p}_{\boldsymbol{\theta}}(\hat{\mathbf{x}}_0 \mid \mathbf{x}_t)$$

   für eine Netzausgabe $\tilde{p}_{\boldsymbol{\theta}}$. Damit erhält jeder Kandidat den zusätzlichen Gewichtsfaktor $q(\mathbf{x}_t \mid \hat{\mathbf{x}}_0)$. Bei den meisten Übergangskernen unterscheiden sich die beiden Formen. Seien $q(\mathbf{x}_0 \mid \mathbf{x}_t)$ und $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ die bedingten Verteilungen des Vorwärtsprozesses, der bei den Daten startet.
    1. Betrachten Sie ein einzelnes Token. Zeigen Sie: Ist die Vorhersage in :eqref:`eq_diffusion-x0-parameterization` gleich $q(\mathbf{x}_0 \mid \mathbf{x}_t)$, so entspricht das Rückwärtsmodell für jeden Übergangskern dem wahren Rückwärtsübergang $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$.
    1. Zeigen Sie, dass das Gewicht $q(\mathbf{x}_t \mid \hat{\mathbf{x}}_0)$ beim absorbierenden Übergangskern an einer maskierten Position für alle Kandidaten gleich ist. Damit stimmen die beiden Formen überein.
   :exlabel:`ex_discrete-diffusion-two-forms`

<!--en:cca70f0f-->
1. [code] **Zeitunabhängige Vorhersagen.** ● Wir haben in diesem Abschnitt begründet, warum die beste Vorhersage an einer maskierten Position beim absorbierenden Übergangskern nicht von $t$ abhängt :cite:`Ou.Nie.Xue.ea.2025`.
    1. Formulieren Sie das Argument als kurzen Beweis. Welche Eigenschaft des absorbierenden Vorwärtsprozesses nutzt er? Gilt dasselbe für den Gleichverteilungskern?
    1. Das maskierte Sprachmodell aus :numref:`sec_transformer` maskiert jedes Token unabhängig mit Wahrscheinlichkeit 15%. Nach (a) sind seine exakten Vorhersagen gerade diejenigen, die der Sampler für Maskenmuster dieser Dichte braucht. Bestimmen Sie für Sequenzen aus $L = 512$ Token den Bereich der maskierten Anteile, die ein solches Modell im Training sieht. Betrachten Sie dazu etwa den Bereich innerhalb von drei Standardabweichungen um 15%. Ein Lauf startet bei der vollständig maskierten Sequenz und verwendet $S = 512$ gleichmäßig verteilte Stufen im linearen Zeitplan. In welchem Anteil seiner Schritte sieht das Netz einen maskierten Anteil in diesem Bereich?
    1. `MaskedMLP` erhält die Zeit als Eingabe. Maskieren Sie das erste Token und lassen Sie das zweite sichtbar. Berechnen Sie die Vorhersagen von `pair_model` für alle 32 Werte des zweiten Tokens bei $t \in \{1, 16, 31\}$, also für die Zeiteingaben $t/32$. Berechnen Sie dann zwei Arten von Kullback-Leibler-Divergenzen, jeweils gemittelt über die Randverteilung des zweiten Tokens. Bestimmen Sie zum einen die Divergenz der Vorhersagen bei $t = 16$ und bei $t = 31$ gegenüber denen bei $t = 1$. Bestimmen Sie zum anderen die Divergenz der Vorhersagen bei jedem $t$ gegenüber den exakten bedingten Verteilungen aus `table` (begrenzen Sie die Wahrscheinlichkeiten nach unten auf $10^{-12}$). Was sagen die Größen der beiden Divergenzen darüber aus, wie das trainierte Netz seine Zeiteingabe nutzt?
   :exlabel:`ex_discrete-diffusion-time-agnostic`

<!--en:833e0253-->
1. [code] **Untergrenze für ein exaktes Netz.** ● Gegeben sei ein Zeitplan mit $\bar{\alpha}_0 = 1$, $\bar{\alpha}_T = 0$ und Zuwächsen $\Delta_t = \bar{\alpha}_{t-1} - \bar{\alpha}_t$. Betrachten Sie die beiden Token der quantisierten Mischung.
    1. Zeigen Sie, dass die über die wahre Tabelle gemittelte Schranke :eqref:`eq_diffusion-masked-loss` eines exakten Netzes folgenden Wert hat:

       $$H(X^1, X^2) + I(X^1; X^2) \sum_{t=1}^{T} \Delta_t^2.$$

       Schreiben Sie den Term für den Schritt $t$ anhand der drei Maskenmuster und ihrer Wahrscheinlichkeiten auf. Mit einer Teleskopsumme von $(1 - \bar{\alpha}_t)^2$ lässt sich die Summe über $t$ auswerten. Prüfen Sie, dass sich für den linearen Zeitplan $H(X^1, X^2) + I(X^1; X^2)/T$ ergibt.
    1. Berechnen Sie die Summe für den Zeitplan $\bar{\alpha}_t = 1 - (t/T)^2$ in geschlossener Form und für $T = 32$. Vergleichen Sie die Untergrenze mit der des linearen Zeitplans. Warum spielt die Wahl des Zeitplans für $T \to \infty$ keine Rolle mehr?
    1. Schreiben Sie eine Funktion mit der Aufrufsignatur von `MaskedMLP`. Sie soll die Logarithmen der exakten bedingten Verteilungen aus `table` zurückgeben. Sind beide Token maskiert, soll sie die Logarithmen der Randverteilungen zurückgeben (begrenzen Sie die Wahrscheinlichkeiten nach unten auf $10^{-12}$). Übergeben Sie die Funktion an `exact_bound` und verwenden Sie für `alpha_bar` nacheinander beide Zeitpläne.
    1. Trainieren Sie `MaskedMLP` mit `masked_loss` nach dem Zeitplan aus (b) und mit den Einstellungen von `pair_model`. Vergleichen Sie seine exakte Schranke mit seiner Untergrenze und mit der exakten Schranke von `pair_model`.
   :exlabel:`ex_discrete-diffusion-revealed-together`

<!--en:9baa5984-->
1. **Die Schranke für beliebige Reihenfolgen.** ● Ein Netz ignoriere die Zeit wie das exakte Netz aus Aufgabe :exref:`ex_discrete-diffusion-time-agnostic`a. Bezeichnen wir mit $\ell(i, U) = -\log p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_0^U)$ seinen Verlust an Position $i$, wenn genau die Positionen aus der Menge $U$ unmaskiert sind. Der Grenzwert in kontinuierlicher Zeit der Schranke aus :numref:`subsec_diffusion-masked-ce` lässt sich als Integral über die Maskierungswahrscheinlichkeit $u$ schreiben. Er ist dann eine gewichtete Summe der Verluste $\ell(i, U)$ über $i \notin U$. :citet:`Hoogeboom.Gritsenko.Bastings.ea.2022` und :citet:`Ou.Nie.Xue.ea.2025` zeigen, dass er gleich der erwarteten negativen Log-Likelihood eines autoregressiven Modells für beliebige Reihenfolgen ist :cite:`Uria.Murray.Larochelle.2014`,

   $$\mathbb{E}_{\sigma} \Big[ \sum_{k=1}^{L} -\log p_{\boldsymbol{\theta}}\big(\mathbf{x}_0^{\sigma(k)} \mid \mathbf{x}_0^{\sigma(1)}, \ldots, \mathbf{x}_0^{\sigma(k-1)}\big) \Big],$$

   wobei $\sigma$ eine gleichverteilt zufällige Permutation der $L$ Positionen ist.
    1. Bestimmen Sie das Gewicht von $\ell(i, U)$ im Integral für $|U| = L - m$. Verwenden Sie dazu das Betaintegral $\int_0^1 u^{m-1} (1 - u)^{L-m}\, du = (m-1)!\, (L-m)! / L!$.
    1. Bestimmen Sie die Wahrscheinlichkeit, dass eine zufällige Reihenfolge zuerst die Positionen von $U$ und dann die Position $i$ aufdeckt. Folgern Sie, dass die beiden Ausdrücke übereinstimmen.
    1. Bei einem exakten Netz ergibt jede Reihenfolge denselben Wert. Welchen? Vergleichen Sie ihn mit Aufgabe :exref:`ex_discrete-diffusion-revealed-together`a für $T \to \infty$.
    1. Zeigen Sie mithilfe der Jensen-Ungleichung, dass der Ausdruck auch die negative Log-Likelihood des folgenden Modells nach oben beschränkt: Das Modell zieht eine Reihenfolge gleichverteilt und anschließend die Token in dieser Reihenfolge.
   :exlabel:`ex_discrete-diffusion-order-agnostic`

<!--en:330b9b57-->
1. [extended] **Ein Sprachmodell mit maskierter Diffusion.** ●● :citet:`Prabhudesai.Wu.Zadeh.ea.2025` berichten, dass maskierte Diffusionsmodelle wiederholt verwendete Daten besser nutzen als autoregressive Modelle. In dieser Aufgabe verwenden wir dieselben Daten bewusst mehrfach. Verwenden Sie die Zeichenfenster `d2l.TimeMachine(batch_size=64, num_steps=128,
   tokenization='char', num_train=100000, num_val=3000)` aus :numref:`sec_text-sequence`. Die 100 000 Trainingsfenster beginnen an den ersten 100 000 Positionen von *The Time Machine*. Bei 2000 Aktualisierungen mit Minibatches zu 64 Fenstern mit je 128 Zeichen durchläuft das Training diese Positionen etwa 160-mal.
    1. Trainieren Sie das Modell `GPT` aus :numref:`sec_gpt` auf diesen Fenstern über 2000 Aktualisierungen nach seinem Trainingsverfahren mit `LMOptimizer`. Erfassen Sie alle 200 Aktualisierungen den Trainings- und Validierungsverlust. Bei welcher Aktualisierung erreicht der Validierungsverlust sein Minimum, und wie entwickeln sich die beiden Verluste danach?
    1. Implementieren Sie den Entrauscher als Unterklasse von `d2l.GPT` mit `vocab_size + 1` Token, von denen das letzte die Maske ist. Der Vorwärtsdurchlauf soll die Zeit entgegennehmen und ignorieren, wie es die Herleitung in diesem Abschnitt erlaubt. (Hinweis: Um die kausale Maske wegzulassen, überschreiben Sie die verschachtelte Klasse `CausalAttention` und rufen Sie die fusionierte Attention des Frameworks ohne Maske auf.) Trainieren Sie das Netz mit `masked_loss` und dem linearen Zeitplan `alpha_bar` mit $T = 128$ Schritten über 2000 Aktualisierungen.
    1. Schätzen Sie alle 200 Aktualisierungen die Schranke in Bit pro Zeichen auf den Validierungsfenstern mit mehreren Masken pro Fenster. Tragen Sie zum Vergleich den Validierungsverlust von `GPT` in Bit pro Zeichen auf. Vergleichen Sie die Bestwerte der beiden Kurven und die Aktualisierungen, bei denen sie auftreten. Die Schranke des Entrauschers sollte bei jeder Aktualisierung deutlich über dem Verlust von `GPT` liegen, bis bei `GPT` die Überanpassung einsetzt. Dieser Abstand entspricht dem Aufwand, den die Skalierungsstudien messen, und ist kein Programmfehler.
    1. Fällt die Schranke bei der 2000. Aktualisierung noch, trainieren Sie den Entrauscher bis zur 8000. Aktualisierung weiter. Bei welchem Modell setzt die Überanpassung später ein? Wie beeinflusst das Training mit zufällig maskierten Fenstern, was sich der Entrauscher von einem oft gesehenen Fenster einprägen kann?
    1. Ziehen Sie Fenster mit $S \in \{128, 32, 8\}$ Schritten und vergleichen Sie sie mit Samples von `GPT`. Wo scheitern die Samples mit wenigen Schritten? Stellen Sie den Bezug zu Aufgabe :exref:`ex_discrete-diffusion-pairs-revealed`c her.
   :exlabel:`ex_discrete-diffusion-language-model`

<!--en:f3582890-->
[Diskussionen](https://d2l.discourse.group/t/36825)

<!--en:c85731cf-->
Diskrete und maskierte Diffusionsmodelle<br>**Übergangsmatrizen Q_t · absorbierender (maskierender) Übergangskern · ELBO als gewichtete Kreuzentropie · paralleles Demaskieren und Unabhängigkeitsfehler · BERT und GPT im Vergleich**

<!--en:53c2d012-->
::: {.slide title="Warum diskrete Token nichtgaußsches Rauschen brauchen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:ea924ed9-->
- **Keine kontinuierliche Vektoraddition:** Die Addition von $0.3\,\boldsymbol\epsilon$ zum Tokenindex `412` (*„cat“*) ist bedeutungslos. Kategoriale Kennungen haben keine euklidische Metrik.

<!--en:43a97739-->
- **Sequenzieller Flaschenhals bei GPT:** Autoregressives Dekodieren von $L = 1{,}024$ Token erfordert $1{,}024$ aufeinanderfolgende Aufrufe, jeweils einen Durchlauf mit Cache für ein neues Token.

<!--en:eacbfd40-->
- **Diskrete Diffusion (Austin et al., 2021; Sahoo et al., 2024):** Eine Markov-Kette ersetzt Token durch Masken $[\textrm{M}]$. Ein bidirektionaler Entrauscher lernt, Token parallel zu demaskieren.

<!--en:22bea60c-->
::: {.slide title="Diskrete Vorwärtsketten mit Übergangsmatrizen"}

<!--en:cb4c730e-->
[Markov-Ketten auf einem endlichen Vokabular]{.kicker}

<!--en:f7f2f967-->
- **Zeilenstochastische Übergänge:** Für den One-Hot-Zeilenvektor $\mathbf{x} \in \{0, 1\}^K$ verrauscht $[\mathbf{Q}_t]_{ij} = P(i \to j)$ jede Position unabhängig.

<!--en:434d3343-->
- **Exakte A-posteriori-Verteilung nach Bayes:** Bei gegebenem $(\mathbf{x}_0, \mathbf{x}_t)$ ist die Verteilung des Vorgängers das elementweise Produkt ($\odot$) der Rückwärts- und Vorwärtswahrscheinlichkeiten:
  $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathrm{Cat}\!\left(\frac{\mathbf{x}_t\mathbf{Q}_t^\top \odot \mathbf{x}_0\bar{\mathbf{Q}}_{t-1}}{\mathbf{x}_0\bar{\mathbf{Q}}_t\mathbf{x}_t^\top}\right)$$

<!--en:96d89b4e-->
::: {.slide title="Rechenbeispiel: Binärer Wertwechsel in zwei Schritten"}

<!--en:174040a0-->
[Der Vorwärts-Posterior von Hand berechnet]{.kicker}

<!--en:cd136347-->
- **Symmetrischer binärer Wertwechsel mit $10\%$ ($\mathbf{Q}_1 = \mathbf{Q}_2$):** Start bei $\mathbf{x}_0 = (1, 0)$ und Beobachtung von $\mathbf{x}_2 = (0, 1)$ nach zwei Schritten bedeuten genau einen Wechsel.

<!--en:9e8d65bd-->
- **A-priori-Verteilung vorwärts $\odot$ Likelihood rückwärts:** Das elementweise Produkt von $\mathbf{x}_0\bar{\mathbf{Q}}_1 = (0.9, 0.1)$ und $\mathbf{x}_2\mathbf{Q}_2^\top = (0.1, 0.9)$ ist $(0.09, 0.09)$, die marginale Evidenz beträgt $0.18$.

<!--en:206ab350-->
- **Normierte A-posteriori-Verteilung $(0.50, 0.50)$:** Der einzige Wechsel ist in Schritt 1 ($\mathbf{x}_1 = 1$) ebenso wahrscheinlich wie in Schritt 2 ($\mathbf{x}_1 = 0$).

<!--en:71994719-->
::: {.slide title="Der absorbierende (maskierende) Übergangskern"}

<!--en:f4ef5b82-->
[Rauschen durch Löschen von Token]{.kicker}

<!--en:aea72c22-->
Für Text und diskrete Bildcodes erweitert der **absorbierende Übergangskern** (Austin et al., 2021) das Vokabular um ein Maskentoken $[\textrm{M}]$ (Index $K$). Im Schritt $t$ geht jedes unmaskierte Token unabhängig in $[\textrm{M}]$ über, und zwar mit Wahrscheinlichkeit $\beta_t$. Maskierte Token bleiben für immer maskiert:

<!--en:7fe8f4fe-->
- **Vereinfachung durch idempotente Projektion:** Für $\mathbf{P} = \mathbf{1}\mathbf{e}_{\textrm{M}}^\top$ gilt $\mathbf{P}^2 = \mathbf{P}$. Das Produkt über $t$ Schritte, $\bar{\mathbf{Q}}_t$, vereinfacht sich mit der Überlebenswahrscheinlichkeit $\bar\alpha_t = \prod_{s=1}^t (1 - \beta_s)$ zu
  $$\bar{\mathbf{Q}}_t = \bar\alpha_t\,\mathbf{I} + (1 - \bar\alpha_t)\,\mathbf{1}\mathbf{e}_{\textrm{M}}^\top$$

<!--en:d387f2db-->
- **Vollständig maskierter Endzustand ($\bar\alpha_T = 0$):** Bei $t = T$ ist jedes Token gleich $[\textrm{M}]$ mit Wahrscheinlichkeit $1$.

<!--en:66e82968-->
::: {.slide title="Zweipunktverteilung der Vorgänger in der absorbierenden Kette"}

<!--en:f42bf360-->
[Warum sichtbare Token rückwärts unverändert bleiben]{.kicker}

<!--en:e89b2c21-->
- **Falls $\mathbf{x}_t^i \neq [\textrm{M}]$ (sichtbar):** Das Token wurde nie maskiert. Also gilt $\mathbf{x}_{t-1}^i = \mathbf{x}_t^i$ mit Wahrscheinlichkeit $1$ (aufgedeckte Token bleiben unverändert).

<!--en:e263eff7-->
- **Falls $\mathbf{x}_t^i = [\textrm{M}]$ (maskiert):** Es gibt nur zwei mögliche Vorgänger: weiterhin $[\textrm{M}]$ mit Wahrscheinlichkeit $1 - w_t$ oder das demaskierte Token $\mathbf{x}_{t-1}^i \sim p_{\boldsymbol\theta}(\cdot \mid \mathbf{x}_t)$ mit Wahrscheinlichkeit $w_t$.

<!--en:a89f84ff-->
::: {.slide title="Die ELBO der maskierten Diffusion als gewichtete Kreuzentropie"}

<!--en:faec8a59-->
[Vereinfachung der Variationsschranke (Sahoo et al., 2024; Shi et al., 2024)]{.kicker}

<!--en:d966aef1-->
- **Terme in $D_{\textrm{KL}}$ heben sich exakt auf:** An unmaskierten Positionen verschwindet die Divergenz ($0$). Wegen $(1-w_t)\log\frac{1-w_t}{1-w_t} = 0$ bleibt an maskierten Positionen $-w_t\log p_{\boldsymbol\theta}(\mathbf{x}_0^i \mid \mathbf{x}_t)$ übrig.

<!--en:8c567823-->
- **Linearer Zeitplan ($\bar\alpha_t = 1 - t/T$):** Der Anteil $t/T$ der Token ist maskiert und hat Gewicht $w_t = 1/t$. Die erwartete gewichtete Anzahl der Masken beträgt in jedem Schritt $L/T$.

<!--en:0d053d2c-->
::: {.slide title="Invarianz gegenüber dem Zeitplan und zeitunabhängiges Optimum"}

<!--en:b4f5a2ac-->
[Grenzübergang zur Markov-Kette in kontinuierlicher Zeit (T → ∞)]{.kicker}

<!--en:2344b8b3-->
- **Invarianz gegenüber dem Zeitplan:** Der Übergang zur Maskierungsrate $u(\tau) = 1 - \bar\alpha(\tau) \in [0, 1]$ eliminiert den Zeitplan $\bar\alpha(\tau)$ vollständig.

<!--en:917a2d17-->
- **Zeitunabhängiges Optimum (Ou et al., 2025):** Sobald $\mathbf{x}_t$ die maskierten Positionen zeigt, liefert der Schrittindex $t$ keine weitere Information über $\mathbf{x}_0$. Daher braucht $p^\star(\mathbf{x}_0^i \mid \mathbf{x}_t)$ die Angabe $t$ nicht.

<!--en:3b26966a-->
::: {.slide title="Implementierung von Verlust und Sampler für maskierte Diffusion"}

<!--en:a894be85-->
[Vektorisierte Kreuzentropie und Demaskieren in Sprüngen]{.kicker}

<!--en:701ab880-->
Da die Verkettung absorbierender Übergangskerne wieder einen solchen Kern ergibt, kann `unmask` jedes grobe Gitter $T = t_0 > t_1 > \cdots > t_S = 0$ in $S$ Auswertungen durchlaufen. Dabei deckt der Sampler jedes maskierte Token mit Wahrscheinlichkeit $(\bar\alpha_s - \bar\alpha_t)/(1 - \bar\alpha_t)$ auf:

<!--en:f3044103-->
- **Teleskopsumme der Aufdeckwahrscheinlichkeiten:** Bei $S$ Sprüngen wird jede Position $i$ im Sprung $k$ mit Wahrscheinlichkeit $\Delta_k = \bar\alpha_{t_{k+1}} - \bar\alpha_{t_k}$ aufgedeckt ($1/S$ auf einem gleichmäßigen linearen Gitter), unabhängig von allen anderen Positionen.

<!--en:58d1967e-->
::: {.slide title="Der Unabhängigkeitsfehler beim parallelen Aufdecken zweier Token"}

<!--en:2a91c572-->
[Analytische Herleitung für die quantisierte 32×32-Mischung]{.kicker}

<!--en:15cd2517-->
- **Getrenntes Aufdecken (Wahrscheinlichkeit $1 - 1/S$):** $X^1$ und $X^2$ werden in verschiedenen Sprüngen demaskiert (erst Randverteilung, dann bedingte Verteilung). Das ergibt ein exaktes Sample aus $p$.

<!--en:16d3d4b0-->
- **Gleichzeitiges Aufdecken (Wahrscheinlichkeit $S \cdot (1/S)^2 = 1/S$):** Beide Token werden im selben Sprung ausgehend von $([\textrm{M}], [\textrm{M}])$ demaskiert. Unabhängige Ziehungen aus $p^1 \otimes p^2$ erzeugen zusätzliche, unechte Modi.

<!--en:373870c1-->
::: {.slide title="Unechte Modi im Produkt der Randverteilungen"}

<!--en:924b514e-->
[Überprüfung des Gesetzes 0.314 / S für den Totalvariationsabstand]{.kicker}

<!--en:e147bd47-->
Zwei unserer drei gaußschen Modi haben denselben vertikalen Mittelwert ($y = -1.5$). Die Randverteilung $p^1$ hat daher $3$ Gipfel und $p^2$ hat $2$ Gipfel. Das unabhängige Produkt $p^1 \otimes p^2$ erzeugt somit **$3 \times 2 = 6$ Häufungen** ($3$ echte Modi + $3$ unechte Modi mit $0.32$ der Masse):

<!--en:e459aa63-->
::: {.slide title="Zufällige Reihenfolge und erneute Maskierung nach Konfidenz"}

<!--en:e35b5032-->
[Warum die Auswahl nach Konfidenz bei MaskGIT und LLaDA ausgeprägte Modi bevorzugt]{.kicker}

<!--en:7d38edcc-->
In der Praxis ziehen MaskGIT (Chang et al., 2022) und LLaDA (Nie et al., 2025) an allen maskierten Positionen Kandidaten. Sie **behalten den Kandidaten mit der höchsten Konfidenz** (höchste vorhergesagte Randwahrscheinlichkeit) und maskieren den Rest erneut. An der exakten Tabelle für 2 Token zeigt sich eine verborgene Verzerrung:

<!--en:5949ac13-->
- **Feste Reihenfolge (1 Token/Schritt):** Kein Unabhängigkeitsfehler ($\mathrm{TV} = 0.000$). Die exakte Entropie von $4.840$ Nat und die Modusgewichte $[0.500, 0.300, 0.200]$ bleiben erhalten.

<!--en:92531e14-->
- **Erneute Maskierung nach Konfidenz ($\mathrm{TV} = 0.179$, schlechter als 2 zufällige Sprünge mit $0.157$):** Werte aus dem schwächsten Modus ($\pi_3 = 0.20$) haben eine geringere Randwahrscheinlichkeit. Sie verlieren daher den Konfidenzvergleich und werden überschrieben. Das Gewicht von Modus 3 bricht **von $0.200$ auf $0.072$** ein, die Entropie sinkt auf $4.41$ Nat.

<!--en:bd6f9a4b-->
::: {.slide title="Untergrenze der negativen ELBO für faktorisierte Entrauscher"}

<!--en:15da59d1-->
[Warum die negative diskrete ELBO bei endlichem T um I / T über der wahren Entropie liegt]{.kicker}

<!--en:60809271-->
Das Training von `MaskedMLP` ($K = 32, T = 32$) umfasst $6{,}000$ Aktualisierungen mit quantisierten Paaren aus der Mischung. Anschließend ergibt sich die exakte negative diskrete ELBO als Summe über alle $T = 32$ Schritte und Maskenmuster:

<!--en:08ed2835-->
- **Analytische Herleitung der Untergrenze $H(X^1, X^2) + I(X^1; X^2)/T$:**
  - Werden $X^1$ und $X^2$ in verschiedenen Schritten demaskiert (Wahrscheinlichkeit $1 - 1/T$), beträgt ihre erwartete Kreuzentropie $H(X^1) + H(X^2 \mid X^1) = H(X^1, X^2)$.
  - Werden beide im selben Schritt demaskiert (Wahrscheinlichkeit $1/T$), kostet ihre unabhängige Vorhersage aus $([\textrm{M}], [\textrm{M}])$ insgesamt $H(X^1) + H(X^2) = H(X^1, X^2) + I(X^1; X^2)$.
  - Kein faktorisiertes Rückwärtsnetz kann also $H + I/T = 4.840 + 0.479/32 = \mathbf{4.855\text{ nats}}$ unterschreiten. Unser trainiertes MLP erreicht **$4.856\text{ nats}$** und liegt höchstens $0.001$ Nat über der theoretischen Untergrenze.

<!--en:677a48fe-->
::: {.slide title="Trainiertes und exaktes Demaskieren bei verschiedenen Schrittzahlen"}

<!--en:8ed4ec7d-->
[Unterscheidung zwischen parallelem Unabhängigkeitsfehler und Netzfehler]{.kicker}

<!--en:d34b9134-->
Summieren liefert die exakte Ausgabeverteilung von `unmask` für unser trainiertes `pair_model` und für das exakte Tabellenorakel bei $S \in \{32, 8, 2, 1\}$ Schritten. So lässt sich der parallele Fehler mit Abhängigkeit $1/S$ vom Approximationsfehler des Netzes trennen:

<!--en:235d36ff-->
- **1 Schritt ($\mathrm{TV} = 0.315$):** Der parallele Unabhängigkeitsfehler ($0.314$) dominiert vollständig. Im Streudiagramm erscheinen Punkte in den **drei unechten Modi**, an denen sich die Spalten und Zeilen verschiedener Modi kreuzen.

<!--en:cc067256-->
- **32 Schritte ($\mathrm{TV} = 0.020$):** Der parallele Fehler sinkt um den Faktor $32\times$ auf $0.010$. Der konstante Approximationsfehler des Netzes von $\approx 0.010$ bleibt.

<!--en:9b06ac2f-->
::: {.slide title="Binäre Fashion-MNIST-Bilder von Kleidungsstücken mit je 784 Token"}

<!--en:f0e47f31-->
[Schranke in Bit pro Pixel und Demaskieren in 100 Schritten]{.kicker}

<!--en:c3601525-->
Durch Binarisieren von Fashion-MNIST erhalten wir Pixel $\{0, 1\}$ mit Maskentoken $[\textrm{M}] = 2$. Jedes Bild besteht damit aus $L = 784$ Token. Das Training eines `MaskedUNet` mit 3 Eingabe- und 2 Ausgabekanälen ($T = 100$) umfasst $6{,}000$ Aktualisierungen:

<!--en:cb636943-->
- **$4\times$ höhere Kompression gegenüber unabhängigen Pixeln:** Die Variationsschranke erreicht **$\approx 0.170\text{ bits/pixel}$** ($\approx 133$ Bit pro Kleidungsbild), gegenüber $0.702\text{ bits/pixel}$ ($\approx 550$ Bit) für unabhängige Pixel mit ihren Randverteilungen.

<!--en:e84840d5-->
- **Generierung in 100 Schritten ($\approx 8$ demaskierte Pixel pro Sprung):** Es entstehen zusammenhängende binäre Silhouetten von Kleidungsstücken.

<!--en:cf3d13f3-->
::: {.slide title="Fehleranalyse beim parallelen Demaskieren von Bildern"}

<!--en:d54de35c-->
[Warum 4 Sprünge Kleidungsstücke in isolierte Pixel zerlegen]{.kicker}

<!--en:683a6c2f-->
Demaskieren von $L = 784$ binären Pixeln in $S = 4$ Sprüngen ($\approx 196$ pro Sprung) im Vergleich zu $S = 100$ oder $784$ Sprüngen (bei Letzteren $\approx 1$ pro Sprung):

<!--en:91d6d9fa-->
- **4 Schritte ($9\times$ so viele isolierte Pixel wie in den realen Daten):** In Sprung 1 zieht der Sampler $\approx 196$ Pixel unabhängig aus den Randverteilungen des vollständig maskierten Bildes. Das erzeugt unkorreliertes Salz-und-Pfeffer-Rauschen, das spätere Sprünge nicht mehr löschen können.

<!--en:02fa7647-->
- **100 gegenüber 784 Schritten (Rückgang der isolierten Pixel um $10\,\mathrm{SE}$):** Der Wechsel von $100$ auf $784$ Schritte beseitigt $70\%$ der überschüssigen isolierten Pixel bei *genau demselben trainierten U-Net*. Dieser Anteil der Artefakte bei 100 Schritten geht also auf den parallelen Unabhängigkeitsfehler zurück.

<!--en:a725fb6c-->
::: {.slide title="GPT, Autoregression in beliebiger Reihenfolge und BERT"}

<!--en:94a88708-->
[Sequenzmodelle als Spezialfälle der maskierten Diffusion]{.kicker}

<!--en:c7de0916-->
- **Autoregression (GPT) und Autoregression in beliebiger Reihenfolge:** Pro Aufruf $1$ Token demaskieren (insgesamt $L$ Aufrufe, $\mathrm{TV} = 0$). Entweder wächst ein kausales Präfix von links nach rechts, oder die Positionen folgen einer zufälligen Permutation $\sigma$. Im zweiten Fall erfolgt das Training über $u \sim U(0, 1)$ mit Gewicht $1/u$.

<!--en:432211d9-->
- **Sprachmodell mit maskierter Diffusion (MDLM / LLaDA):** Training über $u \sim U(0, 1)$ mit Gewicht $1/u$. Pro Aufruf deckt das Modell $\approx L/S$ Token parallel auf. Für einen Unabhängigkeitsfehler $\propto 1/S$ sinkt die Anzahl sequenzieller Aufrufe um den Faktor $L/S\times$. Jeder Aufruf durchläuft alle $L$ Positionen ohne KV-Cache.

<!--en:b2116ba3-->
- **Maskiertes Sprachmodell (BERT):** Eine feste Maskierungsrate $u = 0.15$. Damit kann das Modell vom vollständig maskierten Anfangszustand bei $u = 1.0$ aus keine Sequenz generieren.

<!--en:0510fd7c-->
::: {.slide title="Kapitel §{num:chap_diffusion} im Rückblick: Stärken und offene Fragen"}

<!--en:8fe36042-->
[Diffusions- und Flussmodelle im Vergleich]{.kicker}

<!--en:ca65da3f-->
In allen neun Abschnitten folgten die kontinuierlichen und diskreten Generatoren demselben Prinzip: **Den rauschfreien Punkt bei allen Rauschpegeln durch Regression vorhersagen und dann von reinem Rauschen oder vollständiger Maskierung aus rückwärts integrieren**.

<!--en:75b92111-->
| Fähigkeit des Generators | Umsetzung mit Diffusion oder Flow-Matching | Technischer Zielkonflikt und offene Frage |

<!--en:116280d5-->
| **Stabiles Training eines einzigen Netzes** | Kleinste-Quadrate-Regression auf $\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon}, \mathbf{u}_t$ oder $\log p(\mathbf{x}_0^i \mid \mathbf{x}_t)$ | Vermeidet die Minimax-Instabilität von GANs, erfordert bei der Inferenz aber mehrere Integrationsschritte |

<!--en:3bf226d6-->
| **Exakte Likelihoods und Abdeckung der Modi** | Kontinuierliche ELBO ($w(\lambda) \equiv 1$) und Variablentransformation für die gewöhnliche Differentialgleichung | Optimale Gewichtung für die Likelihood ($w \equiv 1$) unterscheidet sich von der für die wahrgenommene Samplequalität ($e^{-\lambda/2}$) |

<!--en:1e557e9c-->
| **Schnelles ODE-Sampling in $20$ bis $50$ Schritten** | Deterministisches DDIM, Heun-Verfahren zweiter Ordnung und Mehrschrittverfahren | Ohne zusätzliche Destillation der Flussabbildung noch um den Faktor $20$ bis $50\times$ langsamer als Generatoren mit 1 Schritt |

<!--en:4cacbd64-->
| **Skalierung auf hohe Auflösungen ($512^2+$)** | Latente Diffusion (VAE) und Diffusion Transformer mit `adaLN-Zero` | Zweistufiges Verfahren: Wie gut sich die Darstellungen des Autoencoders modellieren lassen, begrenzt den erreichbaren Fréchet-Abstand der Samples |

<!--en:ee3a1a0a-->
| **Parallele Generierung diskreter Token** | Maskierte Diffusion mit absorbierendem Übergangskern ($w_t = (\bar\alpha_{t-1}-\bar\alpha_t)/(1-\bar\alpha_t)$) | Gleichzeitiges Aufdecken verursacht einen Unabhängigkeitsfehler $\propto 1/S$. Festgelegte Token lassen sich nicht korrigieren |

<!--en:a9b7773e-->
::: {.slide title="Verständnisfragen: A-posteriori-Verteilungen der absorbierenden Kette und paralleler Fehler"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:7b309179-->
Fragen zur Funktionsweise diskreter und maskierter Diffusion:

<!--en:09b9ac8e-->
1. **Sichtbare Token bei absorbierendem Übergangskern und Gleichverteilungskern:** Beim Gleichverteilungskern $\mathbf{Q}_t = (1 - \beta_t)\mathbf{I} + (\beta_t/K)\mathbf{1}\mathbf{1}^\top$ hat jeder Tokenwert selbst an unmaskierten Positionen positive A-posteriori-Wahrscheinlichkeit. Welche Frage muss das Netz bei diesem Kern für jedes sichtbare Token beantworten, beim absorbierenden Übergangskern aber nie?

<!--en:8bd11106-->
2. **ELBO-Gewichte beim linearen Zeitplan:** Gegeben seien ein Tokenpaar ($L = 2$) und ein linearer Zeitplan $\bar\alpha_t = 1 - t/T$ mit $T = 32$ Schritten. Wie groß sind die Schrittgewichte $w_1$ und $w_{32}$? Wie groß ist die erwartete gewichtete Anzahl maskierter Token in einem beliebigen Schritt $t$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:2c954566-->
1. Beim Gleichverteilungskern muss das Netz für jedes sichtbare Token erschließen, **ob es den ursprünglichen Wert zeigt oder durch das Rauschen neu gezogen wurde**. Beim absorbierenden Übergangskern hat jedes sichtbare Token ($\neq [\textrm{M}]$) zu $100\%$ seinen ursprünglichen Wert.

<!--en:af7e3e43-->
2. Wegen $w_t = 1/t$ gilt **$w_1 = 1$** und **$w_{32} = 1/32$**. Im Schritt $t$ beträgt die erwartete Anzahl maskierter Token $L(t/T) = 2t/32$. Ihre erwartete gewichtete Anzahl beträgt also $(2t/32) \cdot (1/t) = \mathbf{1/16 = L/T}$ in jedem Schritt $t$.

<!--en:215bece6-->
::: {.slide title="Verständnisfragen: A-posteriori-Verteilungen der absorbierenden Kette und paralleler Fehler (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:1a1cd0da-->
3. **Paralleler Fehler und Netzfehler:** Für das Tokenpaar aus unserer quantisierten Mischung beträgt der Totalvariationsabstand des exakten Orakelsamplers $0.314 / S$. Unser trainiertes `pair_model` erreicht $\mathrm{TV} = 0.020$ bei $S = 32$ Schritten. Genügen $S = 1{,}024$ Schritte, damit der Totalvariationsabstand des trainierten Netzes unter $0.005$ sinkt?

<!--en:23db34b8-->
**Lösung:**

<!--en:0e8da30b-->
3. **Nein:** Bei $S = 32$ beträgt der parallele Unabhängigkeitsfehler $0.314/32 \approx 0.010$, der Approximationsfehler der vom Netz gelernten bedingten Verteilungen $\approx 0.010$. Bei $S = 1{,}024$ ist der Term $0.314/S$ vernachlässigbar klein. Der Netzfehler von $\approx 0.010$ bleibt jedoch bei jeder Schrittzahl bestehen.

<!--en:67572005-->
::: {.slide title="Zusammenfassung: Diskrete und maskierte Diffusionsmodelle"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:faf9cb35-->
| Begriff oder Formel | Mathematischer Ausdruck | Eigenschaft oder Skalierung | Rolle in der diskreten Diffusion |

<!--en:affff6b7-->
| **Absorbierende Vorwärtskette** | $\bar{\mathbf{Q}}_t = \bar\alpha_t\mathbf{I} + (1 - \bar\alpha_t)\mathbf{1}\mathbf{e}_{\textrm{M}}^\top$ | Token übersteht $t$ Schritte mit Wahrscheinlichkeit $\bar\alpha_t = \prod_{s \le t}(1-\beta_s)$ | Bildet jede Sequenz auf den vollständig maskierten Zustand $[\textrm{M}]^L$ bei $\bar\alpha_T = 0$ ab |

<!--en:cd2431a4-->
| **A-posteriori-Verteilung eines absorbierenden Schritts** | $q(\mathbf{x}_{t-1} = \mathbf{x}_0 \mid [\textrm{M}], \mathbf{x}_0) = w_t = \frac{\bar\alpha_{t-1}-\bar\alpha_t}{1-\bar\alpha_t}$ | Unmaskierte Token werden kopiert ($\mathbf{x}_t$), Masken kehren mit Wahrscheinlichkeit $w_t$ zum ursprünglichen Wert zurück | Die gesamte KL-Divergenz entfällt auf maskierte Positionen |

<!--en:a767ec5a-->
| **ELBO der maskierten Diffusion** | $\sum_{t=1}^T w_t\,\mathbb{E}_q\!\big[\sum_{i:\mathbf{x}_t^i=[\textrm{M}]} -\log p_{\boldsymbol\theta}(\mathbf{x}_0^i \mid \mathbf{x}_t)\big]$ | Kontinuierlicher Grenzwert $\int_0^1 \frac{1}{u}\mathbb{E}_u[\ell]\,du$, bei linearem Zeitplan $w_t = 1/t$ | Schranke unabhängig vom Zeitplan, optimale Vorhersage unabhängig von der Zeit |

<!--en:c94ca324-->
| **Untergrenze der negativen ELBO bei Faktorisierung** | $H(X^1, X^2) + \frac{1}{T}I(X^1; X^2)$ | Abstand zur Entropie verschwindet für $1/T \to 0$ | Das trainierte MLP für 2 Token liegt höchstens $0.001$ Nat über der theoretischen Untergrenze |

<!--en:6dc120df-->
| **Paralleler Unabhängigkeitsfehler** | $q_S = (1 - \frac{1}{S})p + \frac{1}{S}p^1 \otimes p^2$ | $\mathrm{TV}(q_S, p) = \mathrm{TV}(p^1 \otimes p^2, p)/S \propto 1/S$ | Gleichzeitiges Aufdecken zieht aus dem Produkt der Randverteilungen (isolierte Pixel bzw. unechte Modi) |
