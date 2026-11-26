<!--en:db139de8-->
# Score-Matching und Diffusionsmodelle

<!--en:9cebfb25-->
Wir wissen, wie wir aus gaußschem Rauschen Daten erzeugen: Wir integrieren die zeitumgekehrte SDE oder die Wahrscheinlichkeitsfluss-ODE. Beide benötigen an jedem Punkt, den sie durchlaufen, den Score $\nabla\log p_t$ der verrauschten Daten. Für die Mischung mit zwei Modi aus dem vorigen Abschnitt lieferte ihn eine Formel. Für Bilder haben wir Samples, aber keine Formel. Hier vertreten 4096 Ziehungen aus dieser Mischung einen Trainingsdatensatz von Bildern. Wir müssen ein Feld lernen, das wir an keinem einzigen Punkt auswerten können.

<!--en:fe14b46a-->
Es hilft uns nicht, eine Dichte mit Maximum-Likelihood anzupassen und dann abzuleiten. Denn eine flexible Dichte, die wir als Exponentialfunktion einer Netzausgabe darstellen, braucht eine Normierungskonstante: ein Integral über den gesamten Raum. Aapo Hyvärinen erkannte 2005, dass der Score ohne Normierungskonstante auskommt und dass sich der unbekannte Score der Daten durch partielle Integration aus der Zielfunktion entfernen lässt :cite:`Hyvarinen.2005`. Pascal Vincent machte eine noch einfachere Beobachtung :cite:`Vincent.2011`. Zu einem verrauschten Punkt können verschiedene unverrauschte Punkte geführt haben. Mitteln wir über diese Punkte die Richtung zurück zum jeweiligen Ausgangspunkt, erhalten wir den Score der verrauschten Daten. Diese Richtung kennen wir für jedes Trainingspaar. Daher können wir den Score allein aus Samples mit der gewöhnlichen Methode der kleinsten Quadrate lernen.

<!--en:70b63969-->
Genau so trainieren wir Diffusionsmodelle: Ihr üblicher Verlust, „das hinzugefügte Rauschen vorhersagen“, entspricht dieser Regression mit einer bestimmten Gewichtung. Auch ihre beiden bekanntesten Sampler, DDPM und DDIM, führen keine neuen Gleichungen ein. Jeder diskretisiert eine Gleichung, die wir schon kennen. An einer eindimensionalen Gauß-Verteilung erhalten wir für jede Aussage eine Zahl, die wir von Hand nachprüfen können, bevor wir die Aussage an der Mischung testen.

<!--en:685ac24e-->
In diesem Abschnitt trainieren wir kein Netz. Wir lernen den Score bei einem Rauschpegel mit der Methode der kleinsten Quadrate. Das Modell ist klein genug, um die Lösung exakt zu berechnen. Dann untersuchen wir die Sampler anhand des exakten Scores der Mischung. So stammt jeder gemessene Fehler vom Sampler selbst. Wir beginnen mit einem einzigen Rauschpegel, der sich als unzureichend erweist, und verwenden dann alle Pegel zusammen. Gegen Ende führen wir einmal einen Sampler mit Scores aus, die wir bei jedem Pegel mit kleinsten Quadraten angepasst haben. Netze und ihre Fehler behandeln wir in :numref:`chap_diffusion`. Dafür brauchen wir drei Ergebnisse aus diesem Abschnitt: den Satz von Vincent, die Identifikation von DDPM mit der varianzerhaltenden SDE aus :numref:`sec_mdl-sdes` und die von DDIM mit dem Euler-Verfahren. Beim ersten Lesen können Sie den Beweis der Identität von Hyvärinen und den Unterabschnitt zur Steuerung überspringen.

<!--en:10879995-->
## Lernen des Scores

<!--en:dfd823bd-->
Wir suchen ein Vektorfeld $\mathbf{s}_{\boldsymbol{\theta}}$, in der Praxis ein Netz mit $d$ Eingaben und $d$ Ausgaben, das den Score der Daten annähert. Wir können aber nur Samples ziehen. Zuerst brauchen wir eine Zielfunktion, die das Modell mit dem wahren Feld vergleicht und die wir anhand von Samples auswerten können.

<!--en:b73880d9-->
### Eine Zielfunktion ohne Normierungskonstante

<!--en:401c0cfd-->
Am flexibelsten beschreibt ein neuronales Netz eine Dichte über eine Energie: Ein *energiebasiertes Modell* (engl. energy-based model) hat die Form $p_{\boldsymbol{\theta}}(\mathbf{x}) = e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}/Z_{\boldsymbol{\theta}}$ mit einem Netz $E_{\boldsymbol{\theta}}$ und der Normierungskonstante $Z_{\boldsymbol{\theta}} = \int e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}\, d\mathbf{x}$ :cite:`LeCun.Chopra.Hadsell.ea.2006`. Maximum-Likelihood benötigt $\nabla_{\boldsymbol{\theta}} \log Z_{\boldsymbol{\theta}}$, wofür wir ein $d$-dimensionales Integral berechnen müssen, das sich bei jeder Aktualisierung von $\boldsymbol{\theta}$ ändert. Für den Score bezüglich $\mathbf{x}$ spielt die Normierung keine Rolle: $Z_{\boldsymbol{\theta}}$ ist bezüglich $\mathbf{x}$ konstant, also gilt $\nabla_{\mathbf{x}} \log p_{\boldsymbol{\theta}} = -\nabla_{\mathbf{x}} E_{\boldsymbol{\theta}}$.

<!--en:f0c1265e-->
Wir vergleichen deshalb Felder statt Dichten. Bezeichnet $\nabla\log p$ den Score der Daten, so ist die Zielfunktion beim **expliziten Score-Matching** der mittlere quadratische Abstand zwischen den beiden Feldern, gewichtet mit der Datendichte:

<!--en:04c6cab0-->
Sie ist genau dann null, wenn das Modellfeld überall dort mit dem Score der Daten übereinstimmt, wo die Daten Masse haben. Ist $\mathbf{s}_{\boldsymbol{\theta}}$ der Score einer Dichte $q$, so ist dies die Fisher-Divergenz aus :numref:`sec_mdl-fisher-divergence`. Dort besprechen wir auch, welche Unterschiede eine solche Divergenz erfasst und welche ihr entgehen. Hier kann $\mathbf{s}_{\boldsymbol{\theta}}$ ein beliebiges Vektorfeld sein, denn ein Netz mit $d$ Ausgaben muss keinen Gradienten darstellen. Die Zielfunktion kommt nun ohne Normierungskonstante aus, enthält aber noch $\nabla \log p$, also gerade das Feld, das wir nicht kennen. Hyvärinens partielle Integration beseitigt es.

<!--en:8cf629a0-->
**Satz (Identität von Hyvärinen).** *Seien $p$ eine glatte positive Dichte und $\mathbf{s}_{\boldsymbol{\theta}}$ ein glattes Vektorfeld. Es gelte $p(\mathbf{x})\, \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \to \mathbf{0}$ für $\|\mathbf{x}\| \to \infty$, und alle folgenden Erwartungswerte seien endlich. Dann gilt*

<!--en:65751b4b-->
*wobei $\nabla \cdot \mathbf{s}_{\boldsymbol{\theta}} = \sum_i \partial s_i / \partial x_i$ die Divergenz bezeichnet und $C = \tfrac12\mathbb{E}_p\|\nabla\log p\|^2$ nicht von $\boldsymbol{\theta}$ abhängt.*

<!--en:cf5f4716-->
**Beweis.** Wir multiplizieren das Quadrat in :eqref:`eq_mdl-esm-objective` aus. Der Term $\tfrac12 \mathbb{E}\|\nabla \log p\|^2$ ist die Konstante $C$, und der Term $\tfrac12 \mathbb{E}\|\mathbf{s}_{\boldsymbol{\theta}}\|^2$ erscheint unverändert in :eqref:`eq_mdl-hyvarinen`. Es bleibt der gemischte Term umzuformen. In einer Dimension gilt

<!--en:e5c63501-->
und partielle Integration mit dem Randterm $\left[ s_{\boldsymbol{\theta}} \, p \right]_{-\infty}^{\infty} = 0$ ergibt $+\int s_{\boldsymbol{\theta}}'(x)\, p(x)\, dx = \mathbb{E}_p[s_{\boldsymbol{\theta}}']$. Das ist die Stein-Identität $\mathbb{E}_p[s_{\boldsymbol{\theta}}' + s_{\boldsymbol{\theta}}\, (\log p)'] = 0$ aus :numref:`sec_mdl-stein-discrepancy`: Die Ableitung wirkt nun auf das Modell statt auf die unbekannte Dichte. In $d$ Dimensionen wiederholen wir diesen Schritt für jede Koordinate $i$. Wir verwenden $s_i$ anstelle von $s_{\boldsymbol{\theta}}$ und $\partial_i p$ anstelle von $p'$ und integrieren zuerst über $x_i$. Das erlaubt der Satz von Fubini unter der vorausgesetzten Integrierbarkeit. Nach Summation über $i$ ergibt der gemischte Term $\mathbb{E}_p[\sum_i \partial_i s_i] = \mathbb{E}_p[\nabla \cdot \mathbf{s}_{\boldsymbol{\theta}}]$. $\blacksquare$

<!--en:72ae5495-->
Jeder Term auf der rechten Seite von :eqref:`eq_mdl-hyvarinen` ist ein Erwartungswert unter $p$, dessen Integranden wir für ein gegebenes Modell auswerten können. Wir können ihn also durch einen Stichprobenmittelwert schätzen, ganz ohne $Z_{\boldsymbol{\theta}}$ und ohne $\nabla \log p$. Die rechte Seite heißt **implizites Score-Matching**.

<!--en:2ca6f65d-->
An einer Gauß-Verteilung können wir die Identität überprüfen. Die Daten seien $x \sim \mathcal{N}(0, v)$ mit Score $-x/v$. Wir wählen das Modell $s_w(x) = -x/w$ mit einem Parameter. Dann gilt $\tfrac12 s_w^2 + s_w' = x^2/(2w^2) - 1/w$, und die implizite Zielfunktion lautet $J(w) = v/(2w^2) - 1/w$. Ihre Ableitung $-v/w^3 + 1/w^2$ verschwindet bei $w = v$: Die Zielfunktion liefert die Varianz, obwohl sie den Score der Daten nie verwendet hat. Auch die Konstante stimmt. Im Minimum gilt $J(v) = -1/(2v)$ und $C = \tfrac12\mathbb{E}[x^2/v^2] = 1/(2v)$. Die explizite Zielfunktion hat dort also den Wert $J(v) + C = 0$, wie es sein muss, wenn das Modell mit den Daten übereinstimmt.

<!--en:6d182166-->
In der Praxis verwenden wir für den Erwartungswert einen Stichprobenmittelwert. Ersetzen wir $v$ in $J$ durch den Mittelwert von $x^2$ über 10 000 Ziehungen aus $\mathcal{N}(0, 2)$ und minimieren wir auf einem Gitter von $w$-Werten. Der Minimierer ist dann dieser Stichprobenmittelwert mit Standardfehler $v\sqrt{2/10^4} \approx 0.03$. Wir erwarten also $2$ mit einer Abweichung von einigen Hundertsteln.

<!--en:cad539c3-->
Das bestätigt die Rechnung. An der Gauß-Verteilung sehen wir, wie die beiden Terme in :eqref:`eq_mdl-hyvarinen` einander entgegenwirken. Der Divergenzterm $-1/w$ begünstigt ein Feld, das an den Samples nach innen zeigt und so bei den Daten eine Senke bildet. Je steiler die Senke, desto günstiger. Die quadrierte Norm $x^2/(2w^2)$ verhindert, dass das Feld unbegrenzt wächst. Das Gleichgewicht liegt bei der wahren Varianz.

<!--en:5c2dacbc-->
Implizites Score-Matching hat zwei Grenzen. Der Beweis setzt voraus, dass der Randterm verschwindet. Bei Daten auf einem beschränkten Intervall ist das im Allgemeinen nicht der Fall. Die implizite Zielfunktion wählt dann das falsche Feld. Die zweite Grenze sind die Kosten. Die Divergenz ist die Spur der Jacobi-Matrix, $\nabla \cdot \mathbf{s}_{\boldsymbol{\theta}} = \operatorname{tr}\, (\partial \mathbf{s}_{\boldsymbol{\theta}} / \partial \mathbf{x})$. Ihre exakte Berechnung erfordert pro Zeile der Jacobi-Matrix einen Durchlauf im Rückwärtsmodus, also insgesamt $d$ Durchläufe. Bei Bildern geht $d$ in die Millionen. Der Schätzer von Hutchinson (:numref:`sec_mdl-continuous-normalizing-flows`) ersetzt die Spur durch ein Vektor-Jacobi-Produkt je Zufallsrichtung, erhöht aber die Varianz. Denoising-Score-Matching kommt ganz ohne Spur aus.

<!--en:18db6ce8-->
### Denoising-Score-Matching

<!--en:79f92d90-->
Wir suchen eine Zielfunktion ohne Spur und brauchen den Score dort, wo ein Sampler ihn abfragt. Dazu gehören auch Punkte, die nahe bei den Daten liegen, ohne selbst Datenpunkte zu sein. Passen wir den Score *verrauschter* Daten an, erfüllen wir beide Anforderungen. Wir stören jedes Sample mit gaußschem Rauschen der Stärke $\sigma$:

<!--en:8659b431-->
Die verrauschte Randdichte $p_\sigma(\tilde{\mathbf{x}}) = \int p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\, p(\mathbf{x})\, d\mathbf{x}$ entsteht durch Glättung der Datendichte mit einer Gauß-Verteilung. Sie nähert sich $p$, wenn $\sigma$ abnimmt. Ihr Score ist ebenso unbekannt wie der Score der Daten. Den Score der *bedingten Dichte* erhalten wir dagegen aus dem Logarithmus einer Gauß-Dichte:

<!--en:03b7c231-->
Dieser Vektor zeigt von der verrauschten Beobachtung zurück zu dem unverrauschten Sample, aus dem sie entstanden ist. Wir kennen ihn für jedes Paar, das wir erzeugen. **Denoising-Score-Matching** (DSM) verwendet ihn als Regressionsziel:

<!--en:dbdca5c0-->
Um den Verlust zu minimieren, muss das Netz allein aus dem verrauschten Punkt die Richtung zurück zu einem unverrauschten Punkt vorhersagen, den es nicht sieht. Der Zielwert hängt vom unverrauschten Punkt ab, der gesuchte Score dagegen nur vom verrauschten. Die Methode der kleinsten Quadrate verbindet beides, denn sie passt den bedingten Mittelwert des Zielwerts an. Dies besagt das folgende elementare Lemma. Auf ihm beruhen dieser und der nächste Abschnitt. Dort wenden wir es auf eine Geschwindigkeit an.

<!--en:fa16efec-->
**Lemma (Regression auf den bedingten Mittelwert).** *Das Paar $(X, Y)$ habe eine gemeinsame Verteilung mit $\mathbb{E}\|Y\|^2 < \infty$, und es sei $\mathbf{m}(X) = \mathbb{E}[Y \mid X]$. Dann gilt für jede messbare Funktion $\mathbf{v}$ mit $\mathbb{E}\|\mathbf{v}(X)\|^2 < \infty$:*

<!--en:ce9cbddd-->
*Der zweite Term hängt nicht von $\mathbf{v}$ ab: Bis auf eine additive Konstante ist die Minimierung des quadratischen Verlusts bezüglich des verrauschten Zielwerts $Y$ dieselbe Aufgabe wie seine Minimierung bezüglich des bedingten Mittelwerts $\mathbf{m}(X)$.*

<!--en:5886cf1f-->
**Beweis.** Wir fügen $\pm\mathbf{m}(X)$ ein und multiplizieren aus. Der gemischte Term ist $2\, \mathbb{E}\left[ (\mathbf{v}(X) - \mathbf{m}(X))^\top (\mathbf{m}(X) - Y) \right]$. Nach der Turmeigenschaft $\mathbb{E}[Z] = \mathbb{E}[\mathbb{E}[Z \mid X]]$ (:numref:`sec_mdl-joint-distributions`) dürfen wir zunächst über $Y$ bei festem $X$ mitteln. Wegen $\mathbb{E}[\mathbf{m}(X) - Y \mid X] = \mathbf{0}$ verschwindet der gemischte Term. $\blacksquare$

<!--en:78250d33-->
**Satz (Vincent).** *Der Übergangskern für die Verrauschung $p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$ sei in $\tilde{\mathbf{x}}$ differenzierbar, und Gradient und Integral über $\mathbf{x}$ seien vertauschbar. Für den Gauß-Kern gilt beides. Das DSM-Ziel und $\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})$ mögen endliche zweite Momente haben. Dann gilt*

<!--en:b4fc0540-->
*wobei $C$ unabhängig von $\boldsymbol{\theta}$ ist. Die rechte Seite ist die explizite Zielfunktion :eqref:`eq_mdl-esm-objective` für $p_\sigma$. Denoising-Score-Matching und explizites Score-Matching auf den verrauschten Daten haben also dieselben Minimierer und dieselben Gradienten.*

<!--en:14934b7c-->
**Beweis.** Wir wenden das Lemma mit $X = \tilde{\mathbf{x}}$ und $Y = \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$ an. Es bleibt der bedingte Mittelwert von $Y$ zu bestimmen. Darin liegt die Aussage des Satzes: *Der Score der Randdichte ist der A-posteriori-Mittelwert der bedingten Scores.* Schreiben wir die A-posteriori-Verteilung $p(\mathbf{x} \mid \tilde{\mathbf{x}})$ mit dem Satz von Bayes und vertauschen dann Gradient und Integral, so erhalten wir

<!--en:54573144-->
Der Beweis nutzt von der Gauß-Verteilung nur die Differenzierbarkeit und setzt keine Eigenschaft der Datenverteilung voraus. Für einen endlichen Trainingsdatensatz wird das Integral zur Summe. Der Satz liefert dann den Score des geglätteten Trainingsdatensatzes. Das hat eine ernüchternde Konsequenz. Erreichte ein Netz bei jedem Rauschpegel das Minimum und folgte ein Sampler seinem Score fehlerfrei, so lieferte er ausschließlich Trainingspunkte (Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-bayes-risk-floor` untersucht den Grenzfall $\sigma\to0$). Erzeugt ein Modell neue Samples, verdankt es diese also einer Regression, die ihr Minimum nicht ganz erreicht, und nicht der Zielfunktion. :numref:`sec_diffusion-further-reading` fasst zusammen, was über die Bedingungen dafür bekannt ist. Wir wählen gaußsches Rauschen, weil sein bedingter Score :eqref:`eq_mdl-dsm-target` das Rauschen selbst ist, skaliert mit $-1/\sigma$. Den Score oder das Rauschen $\boldsymbol{\epsilon}$ vorherzusagen ist daher dieselbe Aufgabe. Ein Netz $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$ zur Rauschvorhersage definiert das Score-Modell $\mathbf{s}_{\boldsymbol{\theta}} = -\boldsymbol{\epsilon}_{\boldsymbol{\theta}}/\sigma$. Diese Form der *Rauschvorhersage* verwenden Diffusionsmodelle.

<!--en:ada92ade-->
**Beispiel: die Gauß-Verteilung.** Die Daten seien wieder $x \sim \mathcal{N}(0, v)$, diesmal mit Rauschpegel $\sigma$. Die verrauschte Randverteilung ist $p_\sigma = \mathcal{N}(0, v + \sigma^2)$ mit Score $-\tilde x/(v + \sigma^2)$. Der Satz von Bayes für zwei Gauß-Verteilungen liefert die A-posteriori-Verteilung des unverrauschten Punkts,

<!--en:c8653e1f-->
also eine Schrumpfung des verrauschten Punkts zum Mittelwert der Daten hin, mit einer Streuung, die nicht von $\tilde x$ abhängt. Jede Aussage dieses Unterabschnitts lässt sich nun als Formel schreiben. Der bedingte Mittelwert des Ziels $Y = (x - \tilde x)/\sigma^2$ ist $\big(\tfrac{v}{v + \sigma^2} - 1\big)\tilde x/\sigma^2 = -\tilde x/(v + \sigma^2)$, der Score der Randdichte, wie es der Satz von Vincent besagt. Der kleinstmögliche mittlere quadratische Fehler, der zweite Term in :eqref:`eq_mdl-regression-lemma`, ist die A-posteriori-Varianz des Ziels, $\textrm{Var}(x \mid \tilde x)/\sigma^4 = \tfrac{1}{\sigma^2} - \tfrac{1}{v + \sigma^2}$. Für $v = 1$ und $\sigma = 0.5$ liegt die Untergrenze bei $4 - 0.8 = 3.2$, und der A-posteriori-Mittelwert des unverrauschten Punkts ist $0.8\,\tilde x$. Ein Monte-Carlo-Versuch mit 100 000 Paaren überprüft beide Zahlen.

<!--en:684af425-->
Die beste Regressionsfunktion erreicht für die Zielwerte einen mittleren quadratischen Fehler von $3.2$ statt $0$. Denn viele unverrauschte Punkte erklären denselben verrauschten Punkt, und die Regression kann nur ihren Mittelwert liefern. Ein Verlust von $3.2$ entspricht hier einer perfekten Anpassung. Ein Verlust von null ist unmöglich. Die Steigung $0.80$ ist der Schrumpfungsfaktor $v/(v+\sigma^2)$. Keine dieser Beobachtungen ist auf die Gauß-Verteilung beschränkt.

<!--en:5d6882e7-->
Die Steigung ist ein Beispiel für *Tweedies Formel*, die Robbins veröffentlichte und Tweedie zuschrieb :cite:`Robbins.1956,Efron.2011`. Stellen wir die Identität von Vincent $\mathbb{E}[(\mathbf{x} - \tilde{\mathbf{x}})/\sigma^2 \mid \tilde{\mathbf{x}}] = \nabla \log p_\sigma(\tilde{\mathbf{x}})$ um, erhalten wir

<!--en:38b03cfb-->
Der beste Entrauscher für den quadratischen Fehler verschiebt den verrauschten Punkt um einen Schritt der Weite $\sigma^2$ in Richtung des Scores der verrauschten Daten (:numref:`fig_mdl-dyn-tweedie`). Für die Gauß-Verteilung ergibt das $\tilde x - \sigma^2 \tilde x/(v + \sigma^2) = 0.8\,\tilde x$ mit der gerade gemessenen Steigung.

<!--en:6628d1c0-->
![Tweedies Formel für die Mischung $\tfrac12\mathcal{N}(-2, 0.5^2) + \tfrac12\mathcal{N}(2, 0.5^2)$ bei Rauschpegel $\sigma = 0.5$. Eine verrauschte Beobachtung $\tilde{x}$ liegt im Tal der verrauschten Dichte $p_\sigma$. Die A-posteriori-Verteilung des unverrauschten Punkts ist bimodal, aber asymmetrisch. Der Schritt $\tilde{x} + \sigma^2 \nabla \log p_\sigma(\tilde{x})$ führt zu ihrem Mittelwert $\hat{x}_0 = \mathbb{E}[x \mid \tilde{x}]$, der optimalen Entrauschung.](../img/mdl-dyn-tweedie.svg)

<!--en:d81e5331-->
Auch die Untergrenze gilt allgemein. Im Optimum ist der mittlere quadratische Fehler $2J_{\mathrm{DSM}}$ gleich $\mathbb{E}\|Y - \mathbf{m}(X)\|^2$, der mittleren A-posteriori-Varianz des Ziels und damit dem *Bayes-Risiko* der Regression. Zur Berechnung wenden wir das Lemma :eqref:`eq_mdl-regression-lemma` auf das Feld $\mathbf{v} = \mathbf{0}$ an und erhalten $\mathbb{E}\|Y\|^2 = \mathbb{E}\|\mathbf{m}(X)\|^2 + \mathbb{E}\|Y - \mathbf{m}(X)\|^2$. Für das Ziel $Y = -\boldsymbol{\epsilon}/\sigma$ gilt $\mathbb{E}\|Y\|^2 = d/\sigma^2$ in $d$ Dimensionen. Sein bedingter Mittelwert ist der Score, also gilt

<!--en:999664a0-->
Für die Gauß-Verteilung ergibt sich wieder $\tfrac{1}{\sigma^2} - \tfrac{1}{v + \sigma^2}$. Ein beobachteter Verlust enthält zusätzlich den Fehler der Anpassung selbst. Der richtige Vergleichswert ist also diese Untergrenze und nicht null.

<!--en:1cb8de53-->
Die Untergrenze zeigt auch, was bei sehr kleinem Rauschpegel schiefgeht. Für Daten mit glatter Dichte wächst sie wie $d/\sigma^2$ für $\sigma \to 0$. Der Zielwert $-\boldsymbol{\epsilon}/\sigma$ wird groß, während das mittlere Quadrat des Scores, der sich durch Mittelung ergibt, endlich bleibt. Ein einzelner kleiner Rauschpegel führt deshalb zu einer schlecht konditionierten Regression, deren Zielwerte größtenteils aus Rauschen bestehen.

<!--en:69becc2c-->
Tweedies Formel gibt an, wohin der beste Entrauscher einen verrauschten Punkt verschiebt. Eine weitere Ableitung zeigt, wie schnell sich das Ergebnis mit dem verrauschten Punkt ändert. Daraus ergibt sich eine Schranke für die Krümmung einer verrauschten Dichte. Wir brauchen sie nur einmal, im letzten Unterabschnitt, wenn wir einen Sampler auf eine Klasse zusteuern. Solange uns die A-posteriori-Verteilung noch vorliegt, genügen für den Beweis wenige Zeilen.

<!--en:42709db5-->
**Satz (Krümmung einer verrauschten Dichte).** *Für jede Datenverteilung und jedes $\sigma > 0$ gilt*

<!--en:586cf582-->
*also übersteigt kein Eigenwert von $-\nabla^2 \log p_\sigma$ die Schranke $1/\sigma^2$.*

<!--en:17714a0d-->
**Beweis.** Nach Tweedies Formel :eqref:`eq_mdl-tweedie` gilt $\nabla \log p_\sigma(\tilde{\mathbf{x}}) = (\mathbb{E}[\mathbf{x} \mid \tilde{\mathbf{x}}] - \tilde{\mathbf{x}})/\sigma^2$. Wir müssen also noch den A-posteriori-Mittelwert $\mathbb{E}[\mathbf{x} \mid \tilde{\mathbf{x}}] = \int \mathbf{x}\, p(\mathbf{x} \mid \tilde{\mathbf{x}})\, d\mathbf{x}$ ableiten. Die Abhängigkeit der A-posteriori-Verteilung von $\tilde{\mathbf{x}}$ ergibt sich aus dem Satz von Bayes, der $\nabla_{\tilde{\mathbf{x}}} \log p(\mathbf{x} \mid \tilde{\mathbf{x}}) = \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) - \nabla \log p_\sigma(\tilde{\mathbf{x}}) = (\mathbf{x} - \mathbb{E}[\mathbf{x} \mid \tilde{\mathbf{x}}])/\sigma^2$ liefert. Differenzieren wir unter dem Integral mit $\nabla p = p\, \nabla \log p$, so erhalten wir für die Jacobi-Matrix des A-posteriori-Mittelwerts

<!--en:f434c58a-->
Wir ziehen die Jacobi-Matrix $\mathbf{I}$ von $\tilde{\mathbf{x}}$ ab und teilen durch $\sigma^2$. Das ergibt :eqref:`eq_mdl-noised-curvature`. Eine Kovarianzmatrix ist positiv semidefinit. Daraus folgt die Schranke. $\blacksquare$

<!--en:345fc9f6-->
Eine verrauschte Dichte kann sich nicht stärker krümmen, als ihr Rauschen es zulässt: Keiner ihrer Gipfel ist schmaler als die Gauß-Dichte, die sie geglättet hat. Für gaußverteilte Daten lautet die rechte Seite von :eqref:`eq_mdl-noised-curvature` wie erwartet $-\tfrac{1}{\sigma^2} + \tfrac{v}{\sigma^2 (v + \sigma^2)} = -\tfrac{1}{v + \sigma^2}$, die Krümmung von $\log \mathcal{N}(0, v + \sigma^2)$.

<!--en:e4407508-->
### Score-Schätzung mit kleinsten Quadraten in einer Dimension

<!--en:a1aa3900-->
Die Gauß-Verteilung hat einen linearen Score, den ein Modell mit einem Parameter darstellen kann. Rekonstruiert die Methode der kleinsten Quadrate aus verrauschten Paaren auch einen gekrümmten Score-Verlauf? Wir nehmen als Daten die Mischung aus dem vorigen Abschnitt, $p = \tfrac12 \mathcal{N}(-2, 0.5^2) + \tfrac12 \mathcal{N}(2, 0.5^2)$, mit Rauschpegel $\sigma = 0.5$. Das Rauschen erhöht die Varianz jeder Komponente um $\sigma^2$. Damit ist $p_\sigma$ wieder eine Mischung, nun mit Komponentenvarianz $0.25 + 0.25 = 0.5$. Wie in :numref:`sec_mdl-score-function` hergeleitet, hat eine symmetrische Mischung mit Mittelwerten $\pm\mu$ und gemeinsamer Komponentenvarianz $v$ den Score $(\mu \tanh(\mu x/v) - x)/v$. Für $\mu = 2$ und $v = 0.5$ lautet die gesuchte Kurve

<!--en:048cd611-->
In der Nähe jedes Modus liegt sie nahe bei $-(\tilde x \mp 2)/0.5$, dem linearen Score einer einzelnen Komponente. Zwischen den Modi steigt sie von etwa $-2.9$ bei $\tilde x = -0.45$ auf $+2.9$ bei $\tilde x = 0.45$ und treibt Punkte aus dem Tal.

<!--en:9499493c-->
Als Modell verwenden wir eine Linearkombination aus 17 festen Merkmalen statt eines Netzes: sechzehn Gauß-Glocken der Breite $0.6$ mit Zentren im Intervall $[-4.5, 4.5]$ und die Identität $x$. So können wir den DSM-Verlust :eqref:`eq_mdl-dsm-loss` mit der gewöhnlichen Methode der kleinsten Quadrate auf den Paaren $(\tilde x, -\epsilon/\sigma)$ minimieren. Das löst `np.linalg.lstsq` exakt. Eine Abweichung zwischen Anpassung und Score ist damit ein Stichprobenfehler, für den wir keinen Optimierer verantwortlich machen können. Neues Rauschen zu ziehen kostet praktisch nichts. Deshalb verrauschen wir jeden der 4096 unverrauschten Punkte viermal. Wir passen das Modell an diese 16 384 Paare an, vergleichen es mit dem analytischen Score und werten beide weit außerhalb der Daten bei $x = 8$ aus. Die ausgegebenen mittleren quadratischen Fehler entsprechen $2J_{\mathrm{DSM}}$, der Größe, die das Regressionslemma zerlegt.

<!--en:11ca79b0-->
Die Grafik zeigt alles, was der Regression zur Verfügung steht: eine Wolke von Zielwerten $-\epsilon/\sigma$, die bei jedem $\tilde x$ über mehrere Einheiten streuen. Durch diese Wolke verläuft ihr bedingter Mittelwert, der Score von $p_\sigma$. Auf ihm liegt die Kurve der Kleinste-Quadrate-Anpassung. Die Ausgabe entspricht dem Regressionslemma. Die Untergrenze :eqref:`eq_mdl-dsm-floor` für die Grundgesamtheit, hier $\tfrac{1}{\sigma^2} - \mathbb{E}_{p_\sigma}(\nabla\log p_\sigma)^2$ durch Quadratur berechnet, beträgt $2.11$. Sie misst die mittlere quadratische Streuung der Wolke in vertikaler Richtung um die Kurve. Der ausgegebene mittlere quadratische Fehler weicht sowohl für die Anpassung als auch für den exakten Score um höchstens ein Zehntel von dieser Untergrenze ab. Untereinander stimmen beide bis auf etwa ein Hundertstel überein. Keiner der beiden Werte muss genau $2.11$ treffen, denn jeder ist ein Mittelwert über eine Stichprobe von Paaren und schwankt von Stichprobe zu Stichprobe. Was die Anpassung vom Score trennt, ist das *Überschussrisiko*, der erste Term des Lemmas. Es beträgt rund $0.01$, weniger als ein Prozent der Untergrenze.

<!--en:7fab15e8-->
Die letzte Zahl zeigt die Grenze der Methode. Kein verrauschtes Paar liegt jenseits von $|\tilde x| \approx 5$. Bei $x = 8$ liefert die Anpassung also den Wert, den die Basisfunktionen durch Extrapolation ergeben. Die Ausgabe kann nahe am exakten Wert $-12$ liegen oder weit davon entfernt. Eine einzelne Anpassung verrät uns nicht, womit wir rechnen sollten. Wiederholen wir den Versuch deshalb zwanzigmal, jedes Mal mit neuen Daten und neuem Rauschen. Wir erfassen den mittleren quadratischen Fehler des exakten Scores, um zu sehen, wie weit eine Stichprobe ihn verschiebt. Außerdem erfassen wir die Anpassung an zwei Punkten: bei $x = 1$ an der Flanke des rechten Modus, wo es viele Paare gibt, und bei $x = 8$, wo es keine gibt.

<!--en:b3dc2877-->
Über die zwanzig Ziehungen hat der mittlere quadratische Fehler des exakten Scores eine Standardabweichung von $0.03$ bis $0.05$. Mit einer Stichprobe von Paaren lässt sich die Untergrenze also nur bis auf einige Hundertstel bestimmen (Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-bayes-risk-floor` erhält dieselbe Zahl aus einer einzigen Stichprobe). Bei $x = 1$ bleiben die zwanzig Anpassungen bis auf etwa zwei Zehntel beim exakten Score $+2.0$. Bei $x = 8$ streuen sie über mehrere Dutzend Einheiten und nehmen beide Vorzeichen an. Bei manchen Ziehungen ist der angepasste Score positiv und zeigt von den Daten weg. Wo die Regression keine Paare gesehen hat, gibt sie trotzdem eine Zahl aus, obwohl dort nichts ihren Wert festlegt. Die Methode der kleinsten Quadrate schränkt die Anpassung nur dort ein, wo Paare liegen. Ein Sampler, der weit von den Daten startet, trifft dort also auf ein beliebiges Feld.

<!--en:942f660d-->
## Sampling mit einem Score

<!--en:8428f227-->
Die Sampler aus dem vorigen Abschnitt brauchen den Score bei jedem Rauschpegel eines Vorwärtsprozesses. Kennen wir den Score nur bei einem Pegel oder den exakten Score von $p$ selbst, können wir trotzdem Samples ziehen. Das Verfahren dafür braucht gar keinen Vorwärtsprozess. Es ist nach Paul Langevin benannt, der 1908 die Bewegung eines Teilchens in einer Flüssigkeit durch eine Kraft und zufällige Stöße beschrieb.

<!--en:9c98222f-->
### Langevin-Dynamik

<!--en:2f47ba16-->
Stellen wir uns ein Teilchen vor, das auf der Log-Dichte bergauf driftet und durch Rauschen zufällige Stöße erhält. Die Drift zieht es in Richtung hoher Dichte. Das Rauschen verhindert, dass es sich in einem Modus festsetzt. Die **Langevin-Dynamik** ist die SDE

<!--en:e3ebc585-->
mit dem halben Score als Drift und einer standardmäßigen brownschen Bewegung $\mathbf{W}$ als Rauschen. Wir behaupten, dass $p$ ihre stationäre Verteilung ist.

<!--en:057214b1-->
Für die Gauß-Verteilung $p = \mathcal{N}(0, v)$ mit Score $-x/v$ wissen wir das schon. Dann lautet :eqref:`eq_mdl-langevin` nämlich $dX = -\tfrac{X}{2v}\, dt + dW$, ein Ornstein-Uhlenbeck-Prozess. Ein OU-Prozess $dX = -\theta X\, dt + g\, dW$ hat die stationäre Verteilung $\mathcal{N}(0, g^2/(2\theta))$ (:numref:`sec_mdl-ornstein-uhlenbeck`). Hier ergeben $\theta = 1/(2v)$ und $g = 1$ die Varianz $v$. Der folgende Satz erweitert das Ergebnis auf glatte Zieldichten, deren Score höchstens linear wächst.

<!--en:fb9bc8e8-->
**Satz (Stationarität).** *$p$ sei eine glatte positive Dichte, deren Score $\nabla \log p$ lokal Lipschitz-stetig ist und höchstens linear wächst. Damit besitzt :eqref:`eq_mdl-langevin` eine eindeutige Lösung, die nicht explodiert, und die zugehörige Fokker-Planck-Gleichung hat für jede Anfangsdichte eine eindeutige Lösung. Gilt $\mathbf{X}_0 \sim p$, so gilt $\mathbf{X}_t \sim p$ für alle $t \ge 0$.*

<!--en:ecb486c2-->
**Beweis.** Für eine SDE $d\mathbf{X} = \mathbf{b}\, dt + d\mathbf{W}$ mit Einheitsdiffusion löst die Dichte $q_t$ von $\mathbf{X}_t$ die Fokker-Planck-Gleichung $\partial_t q = -\nabla \cdot (q\, \mathbf{b}) + \tfrac12 \Delta q$ (:numref:`sec_mdl-fokker-planck`). Wir setzen $\mathbf{b} = \tfrac12 \nabla \log p$ und $q = p$ ein und verwenden $p\, \nabla \log p = \nabla p$:

<!--en:dfd7d3d9-->
Damit löst $q_t \equiv p$ die Gleichung zur Anfangsdichte $p$. Aus der Eindeutigkeit folgt der Schluss von „$p$ löst die Gleichung“ auf „die Verteilung von $\mathbf{X}_t$ ist $p$“. $\blacksquare$

<!--en:d1c0df54-->
Die Drift transportiert Masse auf der Log-Dichte genau so schnell bergauf, wie das Rauschen sie verteilt. Die Stationarität allein sagt noch nichts über eine Kette mit einem anderen Start aus. Ein Sampler hat aber immer einen anderen Start. Für Zieldichten wie im Satz, die glatt und positiv sind und deren Score höchstens linear wächst, konvergiert die Verteilung von $\mathbf{X}_t$ tatsächlich von jedem Startpunkt :cite:`Roberts.Tweedie.1996` aus gegen $p$. Wir verzichten auf den Beweis. Wie lange die Konvergenz dauert, ist eine andere Frage. Das messen wir im folgenden Versuch.

<!--en:0a7df5c6-->
Um Samples zu ziehen, diskretisieren wir. Der Euler-Maruyama-Schritt für :eqref:`eq_mdl-langevin` mit Schrittweite $h$ ergibt den **Langevin-Sampler**

<!--en:4f35355f-->
der nur einen Score $\mathbf{s}$ benötigt, ob exakt oder gelernt. Die diskretisierte Kette erbt die Stationarität der SDE nicht. Für die Gauß-Verteilung ergibt der Schritt die lineare Rekursion $x \leftarrow (1 - \tfrac{h}{2v})\, x + \sqrt{h}\, \xi$, deren stationäre Varianz $V$ die Gleichung $V = (1 - \tfrac{h}{2v})^2 V + h$ löst. Also gilt $V = v/(1 - h/(4v))$, und der Sampler überschätzt die Varianz um etwa $h/4$ (Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-langevin-stationarity` misst diese Verzerrung).

<!--en:eeefe5b4-->
Die Gauß-Verteilung hat nur einen Modus, und damit bleibt die Schwierigkeit verborgen. Geben wir dem Sampler die ungleich gewichtete Mischung aus :numref:`sec_mdl-time-reversal`, also $p = 0.8\, \mathcal{N}(-2, 0.5^2) + 0.2\, \mathcal{N}(2, 0.5^2)$, und den exakten Score von $p$. Ohne Anpassungsfehler können wir jeden Fehler dem Sampler zuschreiben. Ein Fünftel der Masse von $p$ liegt rechts von null. Unabhängig von den Gewichten gilt $\mathbb{E}X^2 = 4 + 0.25 = 4.25$. Wir lassen 10 000 Ketten je 2000 Schritte der Weite $h = 0.01$ laufen und verwenden drei Initialisierungen: die breite Anfangsverteilung $\mathcal{N}(0, 3^2)$, den linken Mittelwert und den rechten Mittelwert. Schätzen Sie vor dem Lesen der Ausgabe, welcher Anteil bei breiter Anfangsverteilung rechts von null endet und wie viele der in einem Modus gestarteten Ketten im anderen enden.

<!--en:e796afd2-->
Von den 10 000 Ketten, die im stärker gewichteten Modus gestartet sind, landet etwa jede zweihundertste auf der anderen Seite. Beim Start im schwächer gewichteten Modus ist es etwa jede vierzigste bis fünfzigste. Im Tal ist die Dichte winzig, und die Drift zeigt zurück zu dem Modus, aus dem die Kette kommt. Um das Tal zu überqueren, braucht es eine seltene Folge zufälliger Stöße in dieselbe Richtung. Die Wartezeit wächst exponentiell mit der Höhe der Barriere in $-\log p$.

<!--en:32ceb6e2-->
Die beiden Anteile unterscheiden sich, wie es sein muss. Im Gleichgewicht fließt ebenso viel Masse von links nach rechts wie von rechts nach links. Der schwächer gewichtete Modus hat ein Viertel der Masse des stärkeren und muss seine Ketten daher viermal so leicht abgeben. Das Verhältnis der Übergangsraten stimmt beim Sampler ungefähr. Die Raten selbst sind viel zu klein. Ein System mit zwei Zuständen vergisst seine Anfangsanteile mit einer Rate, die der Summe seiner beiden Übergangsraten entspricht. Die beiden ausgegebenen Anteile summieren sich auf etwa $0.03$ pro Lauf. Ein Fehler in den Anteilen schrumpft also erst nach jeweils $2{,}000/0.03 \approx 67{,}000$ Schritten um den Faktor $e$. Das ist mehr als das Dreißigfache der Länge unseres Laufs.

<!--en:e2c48d04-->
Damit bekommt die erste Ausgabezeile eine andere Bedeutung. Das zweite Moment stimmt bis auf einige Hundertstel mit $4.25$ überein. Das entspricht dem Stichprobenfehler von 10 000 Ketten (die Verzerrung $h/4 \approx 0.0025$ durch die Schrittweite ist viel kleiner). Jede Kette hat innerhalb ihres Modus ein Gleichgewicht erreicht. Fast die Hälfte der Ketten endet jedoch rechts von null, wo nur ein Fünftel der Masse von $p$ liegt. Die Anfangsverteilung ist symmetrisch. Jede Kette gleitet in den Modus auf ihrer Seite des Tals, und kaum eine verlässt ihn wieder. Die Anteile entsprechen der Anfangsverteilung statt $p$ (:numref:`sec_diffusion-denoising` misst dieselbe Unempfindlichkeit mit einem gelernten Score bei drei Modi).

<!--en:587a7600-->
Die Zeit, die eine Kette braucht, um ihren Startpunkt zu vergessen, heißt *Mischzeit* (engl. mixing time). Die langsame Durchmischung ist eine Eigenschaft der SDE selbst. Eine kleinere Schrittweite behebt sie daher nicht. Sie verringert nur die Diskretisierungsverzerrung.

<!--en:9bdd861c-->
### Grenzen eines einzelnen Rauschpegels

<!--en:2d4b4db7-->
Ein Score bei nur einem Rauschpegel hat sich in dreierlei Hinsicht als unzureichend erwiesen. Wo keine verrauschten Paare lagen, ist der gelernte Score beliebig. Selbst mit dem richtigen Score überquert die Langevin-Dynamik das Tal zwischen den Modi kaum. Und die Anteile ihrer Samples entsprechen der Anfangsverteilung: Fast die Hälfte der Masse liegt in einem Modus, der nur ein Fünftel enthält. Das dritte Problem liegt nicht allein am Sampler. Wo eine Mischung gut getrennter Modi Masse hat, entspricht ihr Score fast dem des nächsten Modus, unabhängig von den Gewichten (:numref:`sec_mdl-fisher-divergence`). Bei geringem Rauschen verrät der Score also kaum etwas über die Anteile.

<!--en:9685c48e-->
Der Rückwärtssampler aus :numref:`sec_mdl-time-reversal` traf bei ebenso symmetrischer Anfangsverteilung auf dieselbe ungleich gewichtete Mischung und brachte vier Fünftel der Masse nach links zurück. Er verfügte über etwas, das der Langevin-Dynamik fehlt: den Score bei jedem Rauschpegel. Die Anteile ergaben sich schon bei mittlerem Rauschen. Dort überlappen die Komponenten noch, ziehen die Teilchen aber bereits unterschiedlich stark an. Daher entscheiden die Gewichte, wohin ein Teilchen wandert.

<!--en:f91203c6-->
Eine absteigende Folge von Rauschpegeln begegnet allen drei Problemen. Bei großem $\sigma$ hat die verrauschte Dichte kein Tal, und die verrauschten Paare decken den gesamten Bereich ab, den ein Sampler besucht. Für die symmetrische Mischung hat $p_\sigma$ die Komponentenvarianz $0.25 + \sigma^2$. Die Mischung ist unimodal, sobald die Standardabweichung der Komponenten den Abstand $2$ vom Zentrum zum jeweiligen Mittelwert erreicht (Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging` im vorigen Abschnitt), also für $\sigma \ge 1.94$. Bei ungleichen Gewichten schließt sich das Tal früher. Mit abnehmendem $\sigma$ treten die Details wieder hervor. Ein Sampler, der seine Masse bereits auf beide Modi verteilt hat, muss die Verteilung nur noch verfeinern.

<!--en:ab960d69-->
Die *Annealed Langevin Dynamics* :cite:`song2019generative` durchlaufen eine solche absteigende Folge und verwenden für jeden Pegel dessen eigenen Score. In Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-annealing` lösen sich Ketten auf diese Weise aus einem einzelnen Modus. :numref:`sec_diffusion-gaussian-paths` zeigt, dass eine solche Folge ungleiche Gewichte reproduziert. Wählen wir die Pegel einzeln, bleiben zwei Fragen offen: Welchen Abstand sollen sie haben, und wie gelangen wir von einem zum nächsten? Ein Vorwärtsprozess beantwortet beide. Er liefert ein Kontinuum von Rauschpegeln, und seine Zeitumkehr ergibt den Sampler.

<!--en:56a53d9a-->
## Scores über mehrere Rauschpegel

<!--en:eeb352c8-->
Von nun an besteht das Modell aus einem einzigen Feld $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t)$. Es erhält den Rauschpegel als zweite Eingabe, und wir trainieren es mit dem Verlust des nächsten Unterabschnitts für alle Pegel zugleich. Als Sampler dienen wieder die beiden Gleichungen vom Anfang: die zeitumgekehrte SDE aus :numref:`sec_mdl-time-reversal` und die Wahrscheinlichkeitsfluss-ODE aus :numref:`sec_mdl-probability-flow-ode`.

<!--en:b561bbb5-->
### Vorwärtsprozesse und rauschpegelabhängiger Verlust

<!--en:48334e94-->
DSM benötigt die bedingte Verteilung des verrauschten Punkts bei gegebenem unverrauschtem Punkt. Damit sich der Zielwert explizit angeben lässt, soll diese Verteilung gaußsch sein. Jede SDE, deren Drift linear in $\mathbf{x}$ ist und deren Rauschamplitude nicht von $\mathbf{x}$ abhängt, hat gaußsche Übergangskerne. Das haben wir in :numref:`sec_mdl-ornstein-uhlenbeck` für den OU-Prozess gesehen. Die beiden üblichen Vorwärtsprozesse aus :numref:`sec_mdl-sdes` gehören zu dieser Familie. Ihre Übergangskerne können wir gemeinsam schreiben als

<!--en:6151b804-->
Der Koeffizient $\alpha_t$ schwächt das Signal ab, und $\sigma_t$ skaliert das Rauschen. Nur in diesen beiden Zahlen unterscheiden sich die Prozesse.

<!--en:b5a1a6b8-->
* Die **varianzexplodierende SDE (VE)** $d\mathbf{X} = \sqrt{\tfrac{d}{dt}\sigma^2(t)}\; d\mathbf{W}$ mit wachsendem $\sigma^2(t)$ und $\sigma(0) = 0$ fügt Rauschen hinzu, ohne die Daten zu schrumpfen: $\alpha_t = 1$ und $\sigma_t = \sigma(t)$. Sie ist die kontinuierliche Entsprechung der absteigenden Folge von Rauschpegeln bei den Annealed Langevin Dynamics.

<!--en:550938de-->
* Die **varianzerhaltende SDE (VP)** ist ein Ornstein-Uhlenbeck-Prozess mit zeitabhängiger Rate,

  $$
  d\mathbf{X} = -\tfrac12 \beta(t)\, \mathbf{X}\, dt + \sqrt{\beta(t)}\; d\mathbf{W},
  $$

  mit $\alpha_t = e^{-\frac12 \int_0^t \beta(r)\, dr}$ und $\sigma_t^2 = 1 - \alpha_t^2$. Daten mit Einheitsvarianz behalten daher für jedes $t$ ihre Einheitsvarianz (:numref:`sec_mdl-sdes`).

<!--en:dacf9567-->
Im Rest des Abschnitts arbeiten wir mit dem VP-Prozess. Dem VE-Prozess begegnen wir noch einmal in den Koordinaten, in denen DDIM eine einfache Form annimmt.

<!--en:9328ca53-->
Mit einem Gauß-Kern bei jedem $t$ lautet der bedingte Score $-\boldsymbol{\epsilon}/\sigma_t$, und wir können DSM auf jedem Pegel anwenden. Mitteln wir mit einem Gewicht $w(t) > 0$ über die Pegel, erhalten wir den **rauschpegelabhängigen DSM-Verlust**

<!--en:6d73fc52-->
wobei wir $t$ gleichverteilt aus dem Trainingsbereich ziehen. Jede Ziehung von $(t, \mathbf{x}_0, \boldsymbol{\epsilon})$ ergibt ein Regressionspaar. Diesen Verlust verwenden wir im restlichen Abschnitt. Die Wahl des Gewichts, die wir nun behandeln, kann beim ersten Lesen in den Hintergrund treten.

<!--en:449328d0-->
Das Gewicht entscheidet, welche Pegel das Netz am besten bedient. Was es lernen soll, bleibt gleich: Für jedes positive $w$ ist der Minimierer für die Grundgesamtheit auf jeder Stufe $t$ durch $\mathbf{s}(\cdot, t) = \nabla \log p_t$ gegeben, sofern das Netz ihn darstellen kann. Das folgt aus dem Satz von Vincent für jeden einzelnen Pegel. Das Gewicht verändert den Beitrag der einzelnen Pegel. Nach :eqref:`eq_mdl-dsm-floor` liegt die Untergrenze bei Pegel $t$ bei $d/\sigma_t^2 - \mathbb{E}\|\nabla \log p_t\|^2$. Für kleine $\sigma_t$ wächst sie wie $d/\sigma_t^2$. Ohne Gewichtung dominieren daher die rauschärmsten Pegel den Verlust.

<!--en:6dc1a85b-->
Zwei Gewichtungen sind üblich. Das zeitdiskrete Modell des nächsten Unterabschnitts verwendet $w(t) = \sigma_t^2$. Das hebt dieses Wachstum auf, sodass jeder Pegel einen Verlust derselben Größenordnung beiträgt.

<!--en:c84501fa-->
Bei der *Likelihood-Gewichtung* $w = g^2$ dient das Quadrat des Diffusionskoeffizienten der Vorwärts-SDE als Gewicht. Mit dieser Gewichtung liefert der Score-Fehler eine Schranke für eine Likelihood. Integrieren wir den so gewichteten quadratischen Score-Fehler über die Pegel, erhalten wir eine obere Schranke für die KL-Divergenz von der Datenverteilung zur Modellverteilung der zeitumgekehrten SDE. Hinzu kommt ein Term für die Abweichung zwischen $p_T$ und der Gauß-Verteilung, von der der Sampler startet (Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-likelihood-bound`). Der berechenbare Verlust enthält neben dem Fehler noch die Untergrenze, und diese bereitet bei der Likelihood-Gewichtung Schwierigkeiten. Für den VP-Prozess gilt $g^2 = \beta$. Die gewichtete Untergrenze wächst wie $d/t$ für $t \to 0$, sodass ihr Integral bis hinunter zu $t = 0$ unendlich ist. Das Training beginnt deshalb bei einer kleinen Zeit $\varepsilon > 0$ (Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-weightings`).

<!--en:561f34bf-->
Für die Erzeugung von Samples brauchen wir nichts Neues. Die beiden Sampler verwenden das gelernte Feld anstelle des exakten Scores. Für eine Vorwärts-SDE $d\mathbf{X} = \mathbf{f}(\mathbf{X}, t)\, dt + g(t)\, d\mathbf{W}$ auf $[0, T]$ starten wir mit der Gauß-Verteilung, die $p_T$ annähert. Dann integrieren wir entweder die zeitumgekehrte SDE (:numref:`sec_mdl-time-reversal`) oder die Wahrscheinlichkeitsfluss-ODE (:numref:`sec_mdl-probability-flow-ode`) von $t = T$ zurück bis $0$:

<!--en:34cb312c-->
Dabei ist $\bar{\mathbf{W}}$ eine in der Zeit rückwärts laufende brownsche Bewegung. Für VP gelten $\mathbf{f} = -\tfrac12 \beta \mathbf{x}$ und $g^2 = \beta$.

<!--en:4a477ee6-->
Ein generatives Modell besteht also aus drei Teilen: einem Vorwärtsprozess, einem gelernten Score und einem Sampler. :numref:`fig_mdl-dyn-noising-denoising` zeigt ihr Zusammenspiel in zwei Dimensionen und ergänzt die Vorwärts- und Rückwärtsdichten aus :numref:`fig_mdl-dyn-forward-reverse`.

<!--en:ea695284-->
![Eine Verteilung in Form zweier Halbmonde unter dem VP-Vorwärtsprozess bei $t/T = 0$, $0.02$ und $1$ (obere Reihe, von links nach rechts). Sie nähert sich einer isotropen Gauß-Verteilung. Die untere Reihe durchläuft dieselben Randverteilungen rückwärts, von rechts nach links. Kurze Pfeile zeigen die Richtung des exakten Scores $\nabla\log p_t$ der verrauschten Daten. Bei starkem Rauschen zieht der Score die Punkte zum Ursprung, bei schwachem Rauschen zeigt er zu den Daten.](../img/mdl-dyn-noising-denoising.svg)

<!--en:02d00894-->
### DDPM und die varianzerhaltende SDE

<!--en:a5c9afc2-->
Das Denoising Diffusion Probabilistic Model (DDPM) :cite:`ho2020denoising` ist zeitdiskret formuliert und erscheint zunächst als eigenständige Konstruktion. Es hat drei Teile: eine Verrauschungskette, einen Verlust und einen Sampler. Alle drei kennen wir bereits in anderer Form. Die Kette ist eine Markov-Kette mit $N = 1000$ Verrauschungsschritten nach dem Plan $\beta_1, \ldots, \beta_N \in (0, 1)$,

<!--en:9bd9454b-->
die in jedem Schritt den Zustand etwas schrumpft und ein wenig Rauschen addiert. In diesem und im nächsten Unterabschnitt zählt $t$ die Pegel, $t = 0, 1, \ldots, N$. Der Pegel $t$ entspricht der Zeit $t\, \Delta$ der SDE mit $\Delta = T/N$.

<!--en:416208ec-->
Wie hängt die Kette mit der VP-SDE zusammen? Für $\beta_t = \beta((t-1)\Delta)\, \Delta$ stimmt ihr Schritt bis zur ersten Ordnung in $\beta_t$ mit dem Euler-Maruyama-Schritt $(1 - \tfrac12 \beta_t)\, \mathbf{x}_{t-1} + \sqrt{\beta_t}\, \boldsymbol{\epsilon}_t$ der VP-SDE überein. Es gilt sogar mehr, wie :numref:`sec_mdl-euler-maruyama` für eine konstante Rate gezeigt hat: Der Schritt ist ein *exakter* Übergang des VP-Prozesses über ein Intervall fast derselben Länge. Zwischen zwei beliebigen Zeitpunkten hat der VP-Übergangskern dieselbe Form wie für den Anfangszeitpunkt $0$, nur integrieren wir $\beta$ zwischen diesen beiden Zeitpunkten. Wählen wir ein Intervall mit $\int \beta(r)\, dr = -\log(1 - \beta_t)$. Der Mittelwertkoeffizient seines Übergangskerns ist $e^{-\frac12 \int \beta} = \sqrt{1 - \beta_t}$ und die Varianz ist $1 - (1 - \beta_t) = \beta_t$. Genau das ist die Verteilung aus :eqref:`eq_mdl-ddpm-forward` bei gegebenem $\mathbf{x}_{t-1}$. Für den größten Schritt des Standardplans, $\beta_t = 0.02$, ergibt sich auf einem solchen Intervall $\int \beta = 0.0202$ statt $0.02$. Die DDPM-Kette ist also der zu diskreten Zeitpunkten beobachtete VP-Prozess. Die Verteilung von $\mathbf{x}_t$ bei gegebenem $\mathbf{x}_0$ ist ein VP-Übergangskern ab der Zeit $0$. Dies ist das zentrale Ergebnis dieses Unterabschnitts. Daraus folgt in zwei Zeilen der DDPM-Trainingsverlust. In der DDPM-Notation können wir die Randverteilung auch direkt an der Kette ohne die SDE überprüfen.

<!--en:e18c61b2-->
**Satz ($\bar{\alpha}$-Randverteilung).** *Es sei $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$. Dann gilt bedingt auf $\mathbf{x}_0$:*

<!--en:1d6a08de-->
**Beweis.** Wir verwenden Induktion über $t$. Der Fall $t = 0$ gilt mit $\bar\alpha_0 = 1$. Nehmen wir :eqref:`eq_mdl-ddpm-marginal` für $t - 1$ an und setzen dies in :eqref:`eq_mdl-ddpm-forward` ein:

<!--en:b15edeeb-->
Dabei sind $\boldsymbol{\epsilon}'$ und $\boldsymbol{\epsilon}_t$ unabhängige standardnormalverteilte Größen. Eine Summe unabhängiger gaußverteilter Größen ist wieder gaußverteilt, und ihre Varianzen addieren sich zu $(1 - \beta_t)(1 - \bar{\alpha}_{t-1}) + \beta_t = 1 - (1 - \beta_t)\bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$. Außerdem gilt $(1 - \beta_t)\bar{\alpha}_{t-1} = \bar{\alpha}_t$. $\blacksquare$

<!--en:69cb532b-->
In der Schreibweise für die Übergangskerne des vorigen Unterabschnitts gilt bei DDPM $\alpha_t = \sqrt{\bar{\alpha}_t}$ und $\sigma_t = \sqrt{1 - \bar{\alpha}_t}$. Beim Training simulieren wir die Kette daher nie. Mit :eqref:`eq_mdl-ddpm-marginal` erreichen wir jeden Pegel in einem Schritt. Da der Übergangskern gaußsch ist, können wir DSM direkt anwenden: Der bedingte Score lautet $-\boldsymbol{\epsilon}/\sigma_t$. Parametrisieren wir das Score-Modell durch ein Netz zur Rauschvorhersage, $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) = -\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) / \sigma_t$. Dann gilt

<!--en:486594bc-->
Damit ist der *einfache Verlust* (engl. simple loss) $\tfrac12\mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \|^2$ von DDPM der rauschpegelabhängige DSM-Verlust :eqref:`eq_mdl-ncsm-loss` mit Gewicht $w(t) = \sigma_t^2 = 1 - \bar{\alpha}_t$. „Das hinzugefügte Rauschen vorhersagen“ ist also Denoising-Score-Matching mit einem Gewicht, das das Wachstum der Untergrenze proportional zu $1/\sigma_t^2$ aufhebt. Jeder Pegel trägt einen Verlust der Größenordnung $d$ bei. (Ein Hinweis zur Notation in :cite:`ho2020denoising`: Dort steht $\alpha_t$ für $1 - \beta_t$, während unser $\alpha_t$ den Signalkoeffizienten $\sqrt{\bar\alpha_t}$ bezeichnet. Außerdem fehlt im einfachen Verlust dort unser Faktor $\tfrac12$.)

<!--en:ce3bfc89-->
Der dritte Teil ist der Sampler. Beim *Ancestral Sampling* durchlaufen wir die Pegel von $N$ bis hinunter zu $1$,

<!--en:f135fc94-->
wobei das Rauschen im letzten Schritt entfällt. Bis zur ersten Ordnung in $\beta_t$ ergibt sich $\mathbf{x}_t + \tfrac12 \beta_t \mathbf{x}_t + \beta_t \mathbf{s}_{\boldsymbol{\theta}} + \sqrt{\beta_t}\, \boldsymbol{\xi}$, der Euler-Maruyama-Schritt der zeitumgekehrten SDE aus :eqref:`eq_mdl-score-samplers` für den VP-Prozess. Ursprünglich leitete man das Modell auf einem anderen Weg her :cite:`sohl2015deep,ho2020denoising`: als Schranke für die Log-Likelihood eines Modells mit latenten Variablen. Diesen Weg geht :numref:`sec_diffusion-design-space`. Er liefert dieselbe Regression mit einer anderen Gewichtung der einzelnen Pegel.

<!--en:c15f9050-->
Die folgenden Sampler verwenden den Standardrauschplan, bei dem $\beta_t$ über $N = 1000$ Pegel linear von $10^{-4}$ auf $0.02$ steigt. Der VP-Prozess ist auf Daten mit Einheitsvarianz ausgelegt. Wir verwenden die Mischung deshalb von nun an *standardisiert*: Wir teilen durch $\sqrt{4.25}$, sodass die Modi bei $\pm 2/\sqrt{4.25} = \pm 0.970$ liegen und die Standardabweichung der Komponenten $0.243$ beträgt. Sehen wir uns vor dem Sampling den Plan selbst an. Wir geben $\bar\alpha_t$ für drei Pegel aus. Für den ersten und letzten Pegel berechnen wir außerdem das Verhältnis $\sqrt{(1 - \bar\alpha_t)/\bar\alpha_t}$ von Rauschen zu Signal, das uns im nächsten Unterabschnitt als Zeitvariable dient.

<!--en:61faaf07-->
Das Signal klingt zunächst langsam ab und verschwindet am Ende fast vollständig. Bei $t = 100$ gilt $\bar{\alpha} = 0.897$, und die verrauschte Verteilung ist immer noch bimodal, mit Modi bei $\pm\sqrt{0.897} \cdot 0.970 = \pm 0.92$ und Komponentenstandardabweichung $\sqrt{0.897 \cdot 0.243^2 + 0.103} = 0.39$. Bei $t = N$ gilt $\bar\alpha_N \approx 4 \times 10^{-5}$, und unabhängig von den ursprünglichen Daten ist das Sample praktisch standardnormalverteilt. Vom ersten bis zum letzten Pegel wächst das Rausch-Signal-Verhältnis von $0.01$ auf $157$, also um vier Größenordnungen.

<!--en:178a0882-->
## Deterministisches Sampling und Steuerung

<!--en:de3cc786-->
Ancestral Sampling ruft das Netz einmal je Pegel auf, also $N = 1000$ Mal pro Sample. Kann dasselbe Netz Samples mit weit weniger Aufrufen erzeugen? DDIM gelingt das dank einer Eigenschaft numerischer ODE-Löser. Wir beweisen dies für exakte Scores und führen DDIM dann einmal mit angepassten Scores aus. Zum Schluss fragen wir, welcher Verteilung ein Sampler folgt, wenn wir ihn auf eine Klasse zusteuern.

<!--en:b633571d-->
### Denoising Diffusion Implicit Models

<!--en:92be2562-->
Die Wahrscheinlichkeitsfluss-ODE ist deterministisch und glatt. Für die Integration sollte daher ein grobes Gitter von Pegeln genügen. Denoising Diffusion Implicit Models (DDIM) :cite:`Song.Meng.Ermon.2020` setzen dies mit einem Schritt um, der das trainierte Netz zur Rauschvorhersage wiederverwendet und Pegel überspringt. Auf Pegel $t$ liefert die Rauschschätzung des Netzes durch Umstellen von :eqref:`eq_mdl-ddpm-marginal` eine Schätzung des unverrauschten Samples,

<!--en:f56f70f6-->
Um einen beliebigen früheren Pegel $t' < t$ zu erreichen, setzt DDIM das Sample aus dieser Schätzung und *derselben* Rauschschätzung neu zusammen, diesmal mit den Koeffizienten des Pegels $t'$:

<!--en:e38095ca-->
Der Schritt ist deterministisch, denn es kommt kein neues Rauschen hinzu. Wörtlich gelesen bewegt er ein Sample entlang der Vorwärtskurve $\sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\epsilon}$ des Paars $(\hat{\mathbf{x}}_0, \boldsymbol{\epsilon}_{\boldsymbol{\theta}})$. Ein mit kleinsten Quadraten trainiertes Netz liefert jedoch den bedingten Mittelwert $\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$ statt des Rauschens, das genau dieses Sample erzeugt hat. Die Schätzung ändert sich von Pegel zu Pegel. Was integriert DDIM dann, und wie groß ist der Fehler bei einem langen Schritt?

<!--en:16aa4b71-->
Ein Koordinatenwechsel beantwortet beide Fragen. Wir teilen durch $\sqrt{\bar\alpha_t}$. Der umskalierte Punkt $\bar{\mathbf{x}}_t = \mathbf{x}_t/\sqrt{\bar{\alpha}_t}$ ist gleich $\mathbf{x}_0 + \bar\sigma_t \boldsymbol{\epsilon}$ mit $\bar\sigma_t = \sqrt{(1 - \bar{\alpha}_t)/\bar{\alpha}_t}$, dem Rausch-Signal-Verhältnis aus der Ausgabe zum Rauschplan. In diesen Koordinaten ist der VP-Prozess ein VE-Prozess: Zu den unverrauschten Daten kommt Rauschen der Stärke $\bar\sigma$ hinzu. Sie steigt von $0.01$ auf dem ersten Pegel bis $157$ auf dem letzten. Der Rauschplan legt nur fest, welche Werte von $\bar\sigma$ das Gitter enthält. Bei $\bar{\mathbf{x}}_t$ und $\bar\sigma_t$ kennzeichnet der Überstrich den umskalierten Punkt und seinen Rauschpegel. Der Überstrich bei $\bar\alpha_t$ stammt aus der DDPM-Notation und hat eine andere Bedeutung.

<!--en:897a729c-->
**Satz (DDIM ist das Euler-Verfahren).** *In den Koordinaten $(\bar\sigma, \bar{\mathbf{x}})$ lautet der DDIM-Schritt :eqref:`eq_mdl-ddim-update`*

<!--en:2544ed25-->
*also ein Euler-Schritt für die ODE $d\bar{\mathbf{x}}/d\bar\sigma = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}$. Ist $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$ die exakte Rauschvorhersage $\hat{\boldsymbol{\epsilon}}(\mathbf{x}_t, t) = \mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$, dann ist diese ODE die Wahrscheinlichkeitsfluss-ODE des VP-Prozesses in den Koordinaten $(\bar\sigma, \bar{\mathbf{x}})$.*

<!--en:907bf4ea-->
**Beweis.** Schritt (i) ist eine algebraische Umformung. Die entscheidenden Schritte (ii) und (iii) zeigen, dass die Wahrscheinlichkeitsfluss-ODE in den neuen Koordinaten diese Form annimmt. In (ii) ist $t$ die kontinuierliche Zeit der SDE.

<!--en:9db20bb9-->
(i) Wir teilen :eqref:`eq_mdl-ddim-update` durch $\sqrt{\bar\alpha_{t'}}$ und setzen $\hat{\mathbf{x}}_0 = \bar{\mathbf{x}}_t - \bar\sigma_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}$ ein. Das ergibt $\bar{\mathbf{x}}_{t'} = \hat{\mathbf{x}}_0 + \bar\sigma_{t'}\boldsymbol{\epsilon}_{\boldsymbol{\theta}} = \bar{\mathbf{x}}_t + (\bar\sigma_{t'} - \bar\sigma_t)\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$.

<!--en:5f8a23af-->
(ii) In der VP-Wahrscheinlichkeitsfluss-ODE $d\mathbf{x}/dt = -\tfrac12\beta\mathbf{x} - \tfrac12\beta\nabla_{\mathbf{x}}\log p_t$ aus :eqref:`eq_mdl-score-samplers` wechseln wir die Variablen mit $\alpha_t = \sqrt{\bar\alpha(t)}$ und $\bar\alpha(t) = e^{-\int_0^t\beta}$. Sei $\bar p_t$ die Dichte von $\bar{\mathbf{x}}_t$. Dann gilt $\nabla_{\mathbf{x}}\log p_t = \nabla_{\bar{\mathbf{x}}}\log\bar p_t/\sqrt{\bar\alpha}$. Wegen $d\sqrt{\bar\alpha}/dt = -\tfrac12\beta\sqrt{\bar\alpha}$ heben sich die Schrumpfungsterme auf, und es gilt $d\bar{\mathbf{x}}/dt = -\tfrac{\beta}{2\bar\alpha}\nabla\log\bar p_t$. Aus $\bar\sigma^2 = 1/\bar\alpha - 1$ folgt $d\bar\sigma/dt = \beta/(2\bar\alpha\bar\sigma)$. Division liefert $d\bar{\mathbf{x}}/d\bar\sigma = -\bar\sigma\, \nabla \log \bar p(\bar{\mathbf{x}})$.

<!--en:a6d8c079-->
(iii) Tweedies Formel für den Übergangskern $\mathcal{N}(\mathbf{x}_0, \bar\sigma^2\mathbf{I})$ lautet $\mathbb{E}[\mathbf{x}_0 \mid \bar{\mathbf{x}}] = \bar{\mathbf{x}} + \bar\sigma^2\nabla\log\bar p$. Also gilt $\mathbb{E}[\boldsymbol{\epsilon} \mid \bar{\mathbf{x}}] = (\bar{\mathbf{x}} - \mathbb{E}[\mathbf{x}_0\mid\bar{\mathbf{x}}])/\bar\sigma = -\bar\sigma\nabla\log\bar p$, die rechte Seite von (ii). Auf $\bar{\mathbf{x}}_t$ zu bedingen ist dasselbe wie auf $\mathbf{x}_t$ zu bedingen. $\blacksquare$

<!--en:e4cceada-->
DDIM ist also ein Verfahren erster Ordnung für die Wahrscheinlichkeitsfluss-ODE mit dem Rauschpegel als Zeitvariable. Sein Fehler ist der des Euler-Verfahrens: Während eines Schritts halten wir $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$ fest, während sich das exakte $\hat{\boldsymbol{\epsilon}}$ entlang der Trajektorie ändert. Zwei Korollare beschreiben die Extremfälle.

<!--en:385f5529-->
**Korollar (fehlerfreie Integration einer Punktmasse).** *Bestehen die Daten aus einem einzigen Punkt $\mathbf{x}_0$, so endet jeder DDIM-Schritt mit exaktem $\hat{\boldsymbol{\epsilon}}$ auf der Trajektorie des Wahrscheinlichkeitsflusses, unabhängig von seiner Länge.* Hier gilt $\hat{\boldsymbol{\epsilon}} = (\bar{\mathbf{x}} - \mathbf{x}_0)/\bar\sigma$, und entlang einer Lösung von $d\bar{\mathbf{x}}/d\bar\sigma = (\bar{\mathbf{x}} - \mathbf{x}_0)/\bar\sigma$ ist das Verhältnis $(\bar{\mathbf{x}} - \mathbf{x}_0)/\bar\sigma$ konstant. Ein Euler-Schritt bei konstanter rechter Seite ist exakt.

<!--en:0c5be765-->
**Korollar (Schrumpfung gaußverteilter Daten).** *Für Daten $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ist der exakte Fluss die Identität. Ein DDIM-Schritt mit exakter Rauschvorhersage multipliziert dagegen jedes Sample mit*

<!--en:e9d3fda4-->
*einem Faktor, der für jede positive Schrittlänge kleiner als $1$ ist.* Der VP-Prozess lässt solche Daten unverändert. Auf jedem Pegel ist die Randverteilung $\mathcal{N}(\mathbf{0}, \mathbf{I})$, der Score ist $-\mathbf{x}$, und die Drift $-\tfrac12 \beta \mathbf{x} - \tfrac12 \beta (-\mathbf{x})$ des Wahrscheinlichkeitsflusses verschwindet. Im exakten Fluss bewegt sich also kein Sample. Nach Tweedies Formel gilt $\hat{\boldsymbol{\epsilon}}(\mathbf{x}_t, t) = \sqrt{1 - \bar\alpha_t}\, \mathbf{x}_t$. Einsetzen in :eqref:`eq_mdl-ddim-update` liefert den Faktor. Schreiben wir $\bar\alpha_t = \cos^2\phi_t$, so wird er zum Kosinus des zurückgelegten Winkels. In den Koordinaten mit Überstrich lautet die exakte Lösung $\bar{\mathbf{x}} \propto \sqrt{1 + \bar\sigma^2}$. Entlang dieser Lösung ändert sich also $\hat{\boldsymbol{\epsilon}} = \bar\sigma\bar{\mathbf{x}}/(1 + \bar\sigma^2)$. Halten wir diese Größe einen ganzen Schritt lang fest, schwächen wir das Signal um den obigen Faktor zu stark ab.

<!--en:b2f47474-->
Das Korollar gibt den Faktor für einen Schritt an. Über einen Sampling-Lauf multiplizieren sich die Faktoren. Wir berechnen das Produkt für 10, 50 und 999 gleich lange Schritte des DDPM-Rauschplans. Mit 999 Schritten durchlaufen wir seine 1000 Pegel einzeln. Wie viel von diesem Fehler bleibt bei nicht gaußverteilten Daten? Dazu führen wir DDIM auf der standardisierten Mischung mit ihrer exakten Rauschvorhersage $\hat{\epsilon} = -\sqrt{1 - \bar\alpha_t}\, \nabla \log p_t$ aus. Die verrauschte Verteilung auf Pegel $t$ hat Modi bei $\pm\sqrt{\bar\alpha_t}\, m$ und die Komponentenvarianz $\bar\alpha_t v_0 + 1 - \bar\alpha_t$, mit den unverrauschten Werten $m = 0.970$ und $v_0 = 0.243^2$. Wir starten mit denselben Ziehungen und vergleichen 10 und 50 Schritte auf zwei Arten mit 999 Schritten: anhand des mittleren Abstands der paarweise zugeordneten Endpunkte und anhand der Kolmogorov-Smirnov-Statistik (KS) der beiden Stichproben. Sie ist der größte Abstand zwischen ihren empirischen Verteilungsfunktionen.

<!--en:77b278ab-->
Zehn Schritte schrumpfen gaußverteilte Daten mit Einheitsvarianz um den Faktor $0.832$, fünfzig um $0.964$ und 999 um $0.998$. Diese Faktoren sind deterministisch und messen allein den Euler-Fehler, da die Rauschvorhersage exakt ist. Fünfmal so viele Schritte verringern den Fehlbetrag von $0.17$ auf $0.04$, wie wir es von einem Verfahren erster Ordnung erwarten. Die Mischung verhält sich ebenso. Nach zehn Schritten beträgt der mittlere Abstand zum Ergebnis mit 999 Schritten $0.08$ auf einer Skala mit Modi bei $\pm 0.97$. Auch die KS-Statistik beträgt $0.08$. Fünfzig Schritte senken beide Werte auf etwa $0.02$. :numref:`fig_mdl-dyn-ddim-strides` zeigt beide Versuche.

<!--en:1f4cadbb-->
![(a) Zu jedem Vorwärtspaar $(x_0,\epsilon)$ der standardisierten Mischung gehört eine Kurve. Das blaue Paar hat das Tal überquert. Vom selben verrauschten Endpunkt aus folgt DDIM mit exakter Rauschvorhersage der Wahrscheinlichkeitsfluss-ODE (10 Schritte als Punkte, 999 Schritte, einer je Pegel, gestrichelt) in den anderen Modus. Die ursprüngliche Kurve des Samples kennt DDIM nicht. (b) Für gaußverteilte Daten mit Einheitsvarianz ist der exakte Fluss die Identität (gestrichelt). Zehn DDIM-Schritte mit exakter Rauschvorhersage schrumpfen jedes Sample um den Faktor $0.83$. Das ist der Euler-Fehler aus dem Korollar für gaußverteilte Daten.](../img/mdl-dyn-ddim-strides.svg)

<!--en:10554afc-->
Teilbild (a) klärt, wie die Formel für den Schritt zu lesen ist. Das Sample folgt der Trajektorie des Wahrscheinlichkeitsflusses durch seinen verrauschten Endpunkt, nicht der Vorwärtskurve, die es hervorgebracht hat. DDIM ist der deterministische Grenzfall der Familie von Rückwärtsprozessen (:numref:`sec_mdl-time-reversal`). Ihre Mitglieder fügen eine frei gewählte Menge Rauschen wieder hinzu und haben zu jedem einzelnen Zeitpunkt dieselben Randverteilungen wie der Vorwärtsprozess. Ein trainiertes Netz genügt daher für alle. :numref:`sec_diffusion-samplers` behandelt die Wahl des Verfahrens.

<!--en:39894750-->
### DDIM mit angepassten Scores

<!--en:e5d281c8-->
Bisher haben wir jedem Sampler den exakten Score gegeben. Jetzt lernen wir den Score und ziehen damit Samples. Dafür nehmen wir die Daten, an denen die Langevin-Dynamik gescheitert ist: die beiden standardisierten Modi bei $-m$ und $+m$. Wir gewichten sie so, dass links $0.8$ der Masse liegt und rechts $0.2$.

<!--en:90df1cc6-->
Wir verwenden das Modell, das bei einem einzelnen Pegel funktioniert hat: die 17 festen Merkmale. Diesmal passen wir es separat auf jedem der 50 Pegel an, die DDIM in einem Lauf mit 50 Schritten besucht. Je Pegel verrauschen wir 4096 unverrauschte Ziehungen viermal und regressieren das Rauschen $\epsilon$ auf die Merkmale des verrauschten Punkts. Wir werten die Merkmale bei $\sqrt{4.25}\, x_t$ aus, damit die Glocken die standardisierten Daten ebenso abdecken wie zuvor die ursprünglichen. Ein schwacher Gewichtszerfall (:numref:`sec_weight_decay`) erfasst die Glocken, die bei einem Pegel kein Paar erreicht: Bei kleinsten Quadraten wären ihre Gewichte beliebig, wie zuvor die Anpassung bei $x = 8$. Der Strafterm setzt sie auf null. Dann starten wir DDIM zweimal mit dem Rauschen der vorigen Zelle, einmal mit der exakten Rauschvorhersage der ungleich gewichteten Mischung und einmal mit der angepassten. Lernt die Regression über alle Pegel, was ein einzelner Pegel nicht erfassen konnte, zeigen beide Zeilen einen schwächer gewichteten Modus mit Masse $0.2$.

<!--en:5dc41536-->
Das bestätigen die Ergebnisse. In beiden Zeilen liegt bis auf ein oder zwei Hundertstel ein Fünftel der Samples im schwächer gewichteten Modus. Bei der Langevin-Dynamik mit dem exakten Score eines einzigen Pegels war es fast die Hälfte. Der Sampler mit angepassten Scores hat weder eine Dichte noch einen Score noch die Zahl $0.2$ als Trainingsinformation erhalten. Er kannte nur verrauschte Punkte und das Rauschen, aus dem sie entstanden waren.

<!--en:30f1d632-->
Beide Zeilen gehen vom selben Rauschen aus. Die gemeinsamen Fehler stammen daher vom Sampler, die Unterschiede von der Anpassung. In beiden liegt der Mittelwert im schwächer gewichteten Modus einige Hundertstel unter $0.97$. Der stärker gewichtete Modus ist etwas zu schmal, etwa $0.23$ gegenüber $0.243$. Bei exakter Rauschvorhersage bleibt nur der Euler-Fehler der fünfzig Schritte. Seine Größe entspricht der Erwartung aus dem Korollar für gaußverteilte Daten: eine Schrumpfung um wenige Prozent, wie bei den gaußverteilten Daten, die fünfzig Schritte mit $0.964$ multiplizierten. Bei der Anpassung kommt ein sichtbarer Fehler hinzu: Der schwächer gewichtete Modus hat eine Breite von $0.25$ bis $0.28$ gegenüber etwa $0.24$ in der Zeile mit exakter Vorhersage. Die beiden Modi unterscheiden sich nur im Anteil der Daten. Der Unterschied in der Breite geht daher auf den geringeren Datenanteil zurück. Die Methode der kleinsten Quadrate passt ihre 17 Gewichte dort an, wo die Paare liegen. Nur ein Fünftel der Paare fällt in den schwächer gewichteten Modus. Die KS-Statistiken bestätigen das Bild. Beide Zeilen haben von neuen Ziehungen aus der Datenverteilung einen Abstand von einigen Hundertsteln. Auch der Abstand zwischen den Zeilen ist nicht größer.

<!--en:6a1ccf6a-->
Ein Diffusionsmodell passt nicht jeden Pegel separat an. Sein Netz erhält den Pegel als Eingabe und verwendet auf allen Pegeln dieselben Gewichte (Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-learned-noise`). Die Regression, der Sampler und die beiden Fehlerarten bleiben dieselben.

<!--en:52a214ed-->
### Steuerung mit dem Satz von Bayes

<!--en:bf738c32-->
Ein Sampler für $p$ liefert Samples aller Arten, die in $p$ vorkommen. Oft wünschen wir nur eine Art: Bilder einer Klasse oder Bilder, die zu einer Bildbeschreibung passen. Bei der *Steuerung* (engl. guidance) lenken wir den Sampler auf eine Klasse $y$, indem wir den Score ändern, dem er folgt. In :numref:`sec_diffusion-images` bauen wir gesteuerte Sampler und bewerten sie an Bildern. Hier beschränken wir uns auf zwei Aussagen, die wir an der Mischung von Hand überprüfen können: Steuerung entspricht dem Satz von Bayes für Scores, und mit einem Skalenfaktor größer als eins folgt der Sampler Dichten, die keine verrauschten Daten haben können.

<!--en:bc1aded3-->
In der standardisierten Mischung mit gleichen Gewichten stehe $y$ für den rechten Modus. Wir wünschen also ausschließlich Samples aus der Klasse $\mathcal{N}(m, v_0)$ mit $m = 0.970$ und $v_0 = 0.243^2$. Der Satz von Bayes bei Rauschpegel $t$ lautet $p_t(\mathbf{x} \mid y) \propto p_t(\mathbf{x})\, p_t(y \mid \mathbf{x})$. Für Scores ergibt sich daraus eine Summe, denn die Evidenz $p(y)$ hängt nicht von $\mathbf{x}$ ab:

<!--en:1cd41e75-->
Der bedingte Score ist der unbedingte Score plus der Gradient der Log-Wahrscheinlichkeit eines Klassifikators. Alle Sampler dieses Abschnitts können ihn ohne weitere Änderung verwenden. Die **Klassifikator-Guidance** :cite:`Dhariwal.Nichol.2021` bezieht den zweiten Term von einem Klassifikator $p_{\boldsymbol{\phi}}(y \mid \mathbf{x}, t)$ für verrauschte Eingaben und skaliert ihn mit einer **Guidance-Skala** $\gamma$:

<!--en:983d47be-->
Für $\gamma = 1$ ist das der Satz von Bayes. Für $\gamma > 1$ ist das gesteuerte Feld der Score einer *umgewichteten* Dichte, in die die Wahrscheinlichkeit des Klassifikators als Potenz eingeht. **Classifier-Free Guidance (CFG)** :cite:`Ho.Salimans.2022` liest :eqref:`eq_mdl-guidance-bayes` von rechts nach links, $\nabla \log p_t(y \mid \mathbf{x}) = \nabla \log p_t(\mathbf{x} \mid y) - \nabla \log p_t(\mathbf{x})$, und braucht daher keinen Klassifikator. Es genügen ein bedingtes und ein unbedingtes Score-Modell $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t, y)$ und $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t, \varnothing)$, die ein einziges Netz liefern kann (:numref:`subsec_diffusion-images-guided`):

<!--en:0e102b10-->
gleichwertig dazu $(1 - \gamma)\, \mathbf{s}_\varnothing + \gamma\, \mathbf{s}_y$. Für $\gamma > 1$ extrapoliert diese Kombination über den bedingten Score hinaus.

<!--en:f38285fc-->
Bei der Interpretation von $\gamma > 1$ gibt es eine Einschränkung. Im Allgemeinen gibt es *keine* unverrauschte Verteilung, deren verrauschte Randdichte die umgewichtete Dichte $\tilde p_t$ ist. Die Randdichte $p_t$ beschreibt die Verteilung von $\sqrt{\bar\alpha_t}\, \mathbf{x}_0$ nach Verrauschung mit Stärke $\sigma_t$. Nach der Krümmungsschranke :eqref:`eq_mdl-noised-curvature` übersteigt deshalb kein Eigenwert von $-\nabla^2 \log p_t$ die Schranke $1/\sigma_t^2$, unabhängig von den Daten. Potenzieren wir die Klassifikatorwahrscheinlichkeit mit $\gamma$, erhöht sich die Krümmung, vor allem auf halbem Weg zwischen den Klassen. Bei der Mischung ist die Schranke dort verletzt, sobald $\gamma > 1 + v_0 v_t/(\sigma_t^2 m^2)$ gilt. Dabei ist $v_t = \bar\alpha_t v_0 + \sigma_t^2$ die Varianz einer verrauschten Klasse (Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-guided-marginals`). Diese Schwelle liegt auf den rauschstärksten Pegeln bei $1.0625$ und bei $t = 100$ bei $1.09$. Für $\gamma = 3$ hält die Dichte die Schranke nur auf etwa zehn der tausend Pegel ein, nämlich auf den rauschärmsten. Für $\gamma = 10$ sind es noch weniger.

<!--en:084b533d-->
Warum spielt das eine Rolle? Wären die umgewichteten Dichten die verrauschten Randdichten derselben unverrauschten Verteilung, dann würde eine Rückwärtskette mit ihren Scores bei dieser Verteilung enden, wie zuvor bei $p$. Das ist hier nicht der Fall. Wir wissen also nicht im Voraus, wo eine gesteuerte Kette endet. Probieren wir es aus. Wir führen das Ancestral Sampling aus :numref:`sec_mdl-ddpm-discretized-sde` mit exakten Scores aus: dem unbedingten Score der Mischung und dem klassenbedingten Score der einzelnen Gauß-Verteilung $\mathcal{N}(\sqrt{\bar\alpha_t}\, m,\ v_t)$, kombiniert nach :eqref:`eq_mdl-cfg` mit $\gamma \in \{1, 3, 10\}$.

<!--en:5350a944-->
Bei $\gamma = 1$ reproduziert die Kette die Klasse: Die gesamte Masse liegt rechts von null, wo die unbedingte Mischung nur die Hälfte ihrer Masse hat. Mittelwert und Standardabweichung liegen bis auf etwa $0.01$ bei den exakten Werten $0.970$ und $0.243$. Bei $\gamma = 3$ und $10$ bleibt keine Masse übrig, die zwischen den Modi wechseln könnte. Die Steuerung verschiebt daher den Modus selbst von der Klassengrenze weg, zunächst auf etwa $1.03$, dann auf einen Wert zwischen $1.07$ und $1.08$. Die Breite ändert sich kaum und weicht um höchstens etwa $5\%$ von $0.24$ ab.

<!--en:911bafa3-->
Hier zeigt sich, was die Verletzung der Schranke bedeutet. Ohne Rauschen liegen die Klassen acht Standardabweichungen auseinander, und der Klassifikator entscheidet praktisch sicher. Die umgewichtete Dichte entspricht dort also der Klasse selbst, mit Mittelwert $0.970$ für jedes $\gamma$. Eine Kette, die den Randdichten eines einzigen Vorwärtsprozesses folgte, würde auf dieser Verteilung enden. Bei $\gamma = 3$ und $10$ endet unsere etwa $0.06$ beziehungsweise $0.1$ weiter rechts, weil die Dichten, deren Scores sie folgt, nicht zu einem einzigen Prozess gehören. Die Verschiebung entsteht auf mittleren Pegeln, wo die verrauschten Klassen überlappen (Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-guidance-levels`). Die Steuerung mit $\gamma > 1$ verschiebt Samples also von der anderen Klasse weg und über die Lage der eigenen Klasse hinaus. Eine Klasse mit einem einzigen gaußschen Modus kann wenig Vielfalt verlieren. Bei Bildern lässt sich dieser Verlust messen (:numref:`subsec_diffusion-images-guided`).

<!--en:4baa454f-->
Mit oder ohne Steuerung wertet jeder Sampler dieses Abschnitts den Score oft aus, weil seine Pfade gekrümmt sind. Das Korollar für gaußverteilte Daten hat gezeigt, dass selbst die einfachsten Daten in der Zeitvariablen von DDIM gekrümmte Pfade haben. Wir haben den Vorwärtsprozess wegen seiner gaußschen Übergangskerne gewählt, ohne zu fragen, ob seinen Pfaden leicht zu folgen ist. Welche Pfade vom Rauschen zu den Daten verlaufen gerade?

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:de6ef577-->
Der Score benötigt keine Normierungskonstante. Für verrauschte Daten ist er der bedingte Mittelwert eines Ziels, das wir für jedes Trainingspaar kennen: $-\boldsymbol{\epsilon}/\sigma$. Die Methode der kleinsten Quadrate lernt ihn daher allein aus verrauschten Sample-Paaren. Der Verlust erreicht als Untergrenze die A-posteriori-Varianz und fällt nicht auf null. Ein Rauschpegel genügt nicht: Fern von den Daten ist die Anpassung beliebig. Die Langevin-Dynamik behält die Anteile ihrer Anfangsverteilung bei, sodass fast die Hälfte ihrer Samples in einem Modus liegt, der nur ein Fünftel der Masse enthält. Mit Scores auf jedem Pegel eines Vorwärtsprozesses diskretisieren die beiden bekanntesten Sampler Gleichungen, die wir bereits kennen. DDPM ist die zu diskreten Zeiten beobachtete varianzerhaltende SDE und verwendet als Verlust Denoising-Score-Matching mit Gewicht $\sigma_t^2$. DDIM ist das Euler-Verfahren für die Wahrscheinlichkeitsfluss-ODE in den Koordinaten $(\bar\sigma, \bar{\mathbf{x}})$. Für eine Punktmasse ist es exakt, gaußverteilte Daten zieht es zu stark zusammen. Mit Scores, die wir auf jedem Pegel durch kleinste Quadrate angepasst haben, reproduzierte DDIM die Gewichte $0.8$ und $0.2$, die ein einzelner Pegel nicht erfassen konnte. Die Steuerung entspricht dem Satz von Bayes für Scores. Für $\gamma > 1$ folgt sie Dichten, die sich stärker krümmen, als es bei einer verrauschten Dichte möglich ist.

<!--en:2294243a-->
## Übungen

<!--en:ca210c8e-->
1. [code] **Diskretisierungsverzerrung beim Langevin-Sampling.** Für die gaußsche Zielverteilung $p = \mathcal{N}(0, v)$ ist der Langevin-Sampler aus :eqref:`eq_mdl-langevin` mit Schrittweite $h$ die lineare Rekursion $x \leftarrow (1 - \tfrac{h}{2v})\, x + \sqrt{h}\, \xi$.
    1. Bestimmen Sie die Schrittweiten, für die die Rekursion eine stationäre Verteilung besitzt. Berechnen Sie deren Varianz und entwickeln Sie die Verzerrung bis zur ersten Ordnung in $h$.
    1. Behandeln Sie jeden Modus der Mischung in der Zelle, die `langevin` definiert, als separate Gauß-Verteilung mit $v = 0.25$. Sagen Sie das stationäre zweite Moment $\mathbb{E}[X^2]$ für $h \in \{0.01, 0.1, 0.25, 0.5\}$ vorher. Vergleichen Sie jede vorhergesagte Verzerrung mit dem Monte-Carlo-Standardfehler einer Schätzung aus 10 000 Ketten. Führen Sie dann `langevin` bei jeder dieser Schrittweiten für 2000 Schritte aus, ausgehend von den Startpunkten `rng.normal(0.0, 3.0, 10000)` aus `warm`. Ab welcher Schrittweite lässt sich die Verzerrung im Versuch erkennen?
    1. Der Metropolis-korrigierte Langevin-Algorithmus :cite:`Roberts.Tweedie.1996` beseitigt die Verzerrung mit einem Akzeptanztest, der das Dichteverhältnis $p(\mathbf{x}') / p(\mathbf{x})$ benötigt. Warum kann ein Sampler, der nur über ein trainiertes Score-Netz $\mathbf{s}_{\boldsymbol{\theta}}$ verfügt, diesen Test nicht direkt anwenden?
   :exlabel:`ex_mdl-score-matching-diffusion-flow-langevin-stationarity`

<!--en:16a83b42-->
1. [code] **Durch Abkühlung aus einem Modus entkommen.** In der Zelle, die `langevin` definiert, bleiben Ketten, die in einem Modus von $0.8\, \mathcal{N}(-2, 0.5^2) + 0.2\, \mathcal{N}(2, 0.5^2)$ starten, dort. Bei breiter Anfangsverteilung bleibt fast die Hälfte der Masse rechts. Die Annealed Langevin Dynamics führen den Sampler zunächst auf geglätteten Dichten $p_\sigma$ aus, hier auf der Mischung mit Komponentenvarianz $0.25 + \sigma^2$, und durchlaufen die absteigende Folge von Rauschpegeln $\sigma = 2, 1, 0.5, 0.25, 0.1, 0$.
    1. Bei gleichen Gewichten hat $p_\sigma$ für $\sigma \ge 1.94$ kein Tal, wie im Text gezeigt. Bei den Gewichten $0.8$ und $0.2$ schließt sich das Tal früher. Zählen Sie die Vorzeichenwechsel von `mixture_score` mit `ws=(0.8, 0.2)` auf einem feinen Gitter, um das kleinste $\sigma$ ohne Tal zu finden. Auf welchen Pegeln der Folge gibt es kein Tal?
    1. Führen Sie auf jedem Pegel der Folge $300$ Schritte von `langevin` aus und übergeben Sie die Komponentenvarianz $0.25 + \sigma^2$ als `var`. Starten Sie 10 000 Ketten bei $-2$ und weitere 10 000 Ketten bei $+2$. Verwenden Sie einmal die feste Schrittweite $h = 0.01$ und einmal $h = 0.04\, (0.25 + \sigma^2)$, also $0.01$ bei $\sigma = 0$. Geben Sie für jeden der vier Läufe den Anteil der Ketten mit $X > 0$ am Ende an. Bei welcher Schrittweitenregel verliert der Start seinen Einfluss? Der zugehörige Anteil kommt dem Ziel $0.2$ nahe, ohne es zu erreichen. Berechnen Sie die Masse von $p_\sigma$ rechts von null auf jedem Pegel und geben Sie an, auf welchem Pegel sich die Anteile festgelegt haben.
    1. :citet:`song2019generative` skalieren die Schrittweite jedes Pegels mit $\sigma^2$. Erklären Sie anhand des Kontraktionsfaktors der linearen Rekursion aus Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-langevin-stationarity`, warum die feste Schrittweite bei hohen Rauschpegeln versagt.
   :exlabel:`ex_mdl-score-matching-diffusion-flow-annealing`

<!--en:73e4404d-->
1. [code] **Das Bayes-Risiko als Untergrenze.** ● Die Zelle, die `w_ls` anpasst, gibt den mittleren quadratischen Fehler der Anpassung und den des exakten Scores aus, beide gemittelt über dieselben 16 384 verrauschten Paare. Außerdem berechnet sie die Untergrenze für die Grundgesamtheit durch Quadratur.
    1. Im Text erhalten wir die Untergrenze :eqref:`eq_mdl-dsm-floor`, das Bayes-Risiko der Regression, aus dem Lemma mit $\mathbf{v} = \mathbf{0}$. Zeigen Sie mit Tweedies Formel :eqref:`eq_mdl-tweedie`, dass sie auch gleich $\mathbb{E}\, \operatorname{tr} \textrm{Cov}[\mathbf{x} \mid \tilde{\mathbf{x}}] / \sigma^4$ ist. Prüfen Sie, dass die beiden Ausdrücke für $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, v \mathbf{I})$ übereinstimmen. Wie verhält sich die Untergrenze für $\sigma \to 0$ bei Daten mit glatter Dichte und wie bei einem Trainingsdatensatz mit $n$ verschiedenen Punkten? Was liefert der optimale Entrauscher des Trainingsdatensatzes in diesem Grenzfall?
    1. Der ausgegebene Mittelwert für den exakten Score ist nicht die Untergrenze. Berechnen Sie seinen Standardfehler auf zwei Arten: aus den 16 384 quadratischen Fehlern der einzelnen Paare, als wären sie unabhängig, und aus den 4096 Mittelwerten über die vier verrauschten Kopien jedes unverrauschten Punkts. Warum unterscheiden sich die Ergebnisse? Welches stimmt mit `errs.std()` überein, der im Text ausgegebenen Standardabweichung über zwanzig neue Ziehungen? Ist der ausgegebene Mittelwert mit der Untergrenze für die Grundgesamtheit vereinbar? Erklären Sie dann, warum sich die Fehler der Anpassung und des exakten Scores bei Auswertung an denselben Paaren um weit weniger als diesen Standardfehler unterscheiden.
   :exlabel:`ex_mdl-score-matching-diffusion-flow-bayes-risk-floor`

<!--en:f6fde0c2-->
1. **Gewichtung der Rauschpegel.** ● Betrachten Sie die VP-SDE mit dem linearen Rauschplan $\beta(t) = \beta_{\min} + t\, (\beta_{\max} - \beta_{\min})$ auf $[0, 1]$, mit $\beta_{\min} = 0.1$ und $\beta_{\max} = 20$, sodass $\bar{\alpha}(t) = \exp\big(-\beta_{\min} t - \tfrac12 (\beta_{\max} - \beta_{\min}) t^2\big)$ und $\sigma_t^2 = 1 - \bar{\alpha}(t)$ gelten. Die Daten in $\mathbb{R}^d$ mögen eine glatte Dichte haben, und $\mathbb{E} \| \nabla \log p_t \|^2$ bleibe für $t \to 0$ beschränkt. Bezeichnen Sie mit $R(t)$ das Bayes-Risiko des rauschpegelabhängigen DSM-Verlusts :eqref:`eq_mdl-ncsm-loss` auf Pegel $t$, also die Untergrenze :eqref:`eq_mdl-dsm-floor` mit $\sigma_t$ anstelle von $\sigma$.
    1. Zeigen Sie, dass $R(t) \approx d / (\beta_{\min} t)$ für $t \to 0$ gilt.
    1. Bestimmen Sie das Verhalten von $w(t)\, R(t)$ für $t \to 0$ bei $w = 1$, bei der DDPM-Gewichtung $w = \sigma_t^2$ und bei der Likelihood-Gewichtung $w = g^2 = \beta(t)$. Für welche Gewichtungen ist $\int_0^1 w(t)\, R(t)\, dt$ endlich, und welche Rauschpegel betont jede Gewichtung?
    1. Beim Training ziehen wir $t$ gleichverteilt aus $[\varepsilon, 1]$. Zeigen Sie für die Likelihood-Gewichtung, dass das zweite Moment von $w(t)\, R(t)$ bei dieser Ziehung wie $1/\varepsilon$ wächst. :citet:`Song.Durkan.Murray.ea.2021` (Abschnitt 5.1) ziehen stattdessen $t$ aus der zu $\beta(t) / \sigma_t^2$ proportionalen Dichte auf $[\varepsilon, 1]$ und multiplizieren den DSM-Term des gezogenen Pegels mit $c\, \sigma_t^2$, wobei die Konstante $c = \tfrac{1}{1 - \varepsilon} \int_\varepsilon^1 \beta(r) / \sigma_r^2\, dr$ ist. Zeigen Sie, dass dies einen erwartungstreuen Schätzer für den Likelihood-gewichteten Verlust ergibt und dass sein Wert im Optimum, $c\, \sigma_t^2\, R(t)$, für $t \to 0$ beschränkt bleibt.
   :exlabel:`ex_mdl-score-matching-diffusion-flow-weightings`

<!--en:5a8efb9c-->
1. [code] **Steuerung nach Rauschpegel.** ● Erhöhen wir in der Zelle, die `cfg_sample` definiert, $\gamma$ von $1$ auf $10$, so verschiebt sich der Mittelwert der gesteuerten Samples von etwa $0.97$ auf einen Wert zwischen $1.07$ und $1.08$.
    1. Im Text heißt es, dass auf dem unverrauschten Pegel die umgewichtete Dichte mit der Klasse selbst übereinstimmt. Überprüfen Sie das. Auf dem letzten Pegel der Kette, $t = 0$ in `cfg_sample`, ist die A-posteriori-Wahrscheinlichkeit der rechten Klasse eine logistische Funktion von $x$. Geben Sie die Funktion an und berechnen Sie durch Quadratur den Mittelwert der zu $p_t(x)\, p_t(y \mid x)^{\gamma}$ proportionalen umgewichteten Dichte auf diesem Pegel für $\gamma \in \{1, 3, 10\}$. Wie weit liegt jeder Mittelwert von $m$ entfernt, und wie weit liegen die ausgegebenen Mittelwerte davon entfernt?
    1. Ändern Sie `cfg_sample` so, dass $\gamma = 10$ nur für die Schritte $t \ge 500$, nur für $200 \le t < 500$ oder nur für $t < 200$ gilt und $\gamma = 1$ für alle übrigen Schritte. In welchem Rauschpegelbereich entsteht die Verschiebung? Setzen Sie Ihre Antwort in Beziehung zum Abstand $\sqrt{\bar{\alpha}_t}\, m / \sqrt{v_t}$ der beiden verrauschten Klassen. Dabei entsprechen $m$ und $v_0$ den Variablen `m` und `v0`. Die verrauschte Komponentenvarianz $v_t = \bar{\alpha}_t\, v_0 + 1 - \bar{\alpha}_t$ ist die Variable `v` in `cfg_sample`.
    1. :citet:`Kynkaanniemi.Aittala.Karras.ea.2024` zeigen an Bildmodellen, dass die Steuerung bei hohen Rauschpegeln schadet, bei niedrigen weitgehend entbehrlich ist und dazwischen nützt. Welche Teile dieses Musters reproduziert das eindimensionale Beispiel, und welche kann es nicht zeigen?
   :exlabel:`ex_mdl-score-matching-diffusion-flow-guidance-levels`

<!--en:ca7f414d-->
1. **Gesteuerte Dichten und verrauschte Randverteilungen.** ● Das gesteuerte Feld :eqref:`eq_mdl-cfg` mit $\gamma > 1$ ist der Score der umgewichteten Dichte $\tilde{p}_t(\mathbf{x}) \propto p_t(\mathbf{x})\, p_t(y \mid \mathbf{x})^{\gamma}$. Eine Dichte heiße verrauschte Randdichte auf Pegel $t$, wenn sie die Verteilung von $\sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$ mit $\sigma_t^2 = 1 - \bar{\alpha}_t$ für eine unverrauschte Verteilung von $\mathbf{x}_0$ beschreibt.
    1. Wenden Sie :eqref:`eq_mdl-noised-curvature` auf die unverrauschte Variable $\sqrt{\bar{\alpha}_t}\, \mathbf{x}_0$ an und zeigen Sie, dass für eine verrauschte Randdichte $p_t$ gilt:

       $$\nabla^2 \log p_t(\mathbf{x}) = -\frac{\mathbf{I}}{\sigma_t^2} + \frac{\bar{\alpha}_t}{\sigma_t^4} \textrm{Cov}\left[ \mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x} \right].$$

       Folgern Sie, dass kein Eigenwert von $-\nabla^2 \log p_t$ die Schranke $1 / \sigma_t^2$ übersteigt, unabhängig von der unverrauschten Verteilung.
    1. In `cfg_sample` sind die beiden verrauschten Klassen $\mathcal{N}(\pm \sqrt{\bar{\alpha}_t}\, m, v_t)$. Dabei entsprechen $m$ und $v_0$ den Variablen `m` und `v0`, und $v_t = \bar{\alpha}_t\, v_0 + \sigma_t^2$ ist die verrauschte Komponentenvarianz. Das Label $y$ bezeichnet die rechte Klasse. Zeigen Sie, dass $-(\log \tilde{p}_t)''(x) = 1/v_t + (\gamma - 1)\, a_t^2\, \varsigma(a_t x)\, (1 - \varsigma(a_t x))$ gilt, wobei $a_t = 2 \sqrt{\bar{\alpha}_t}\, m / v_t$ ist und $\varsigma(u) = 1 / (1 + e^{-u})$ die logistische Funktion bezeichnet.
    1. Bestimmen Sie die Schwelle für $\gamma$, oberhalb derer $\tilde{p}_t$ die Schranke aus (a) an einem Punkt $x$ verletzt. Werten Sie sie auf den Pegeln $t = 0$, $100$ und $500$ von `alpha_bar` aus und bestimmen Sie ihren Grenzwert für $t \to N$. Was sagen die Werte über die Dichten aus, denen die Kette von `cfg_sample` bei $\gamma = 3$ folgt?
   :exlabel:`ex_mdl-score-matching-diffusion-flow-guided-marginals`

<!--en:3eeb069e-->
1. **Score-Fehler und Likelihood.** ●● Nach Satz 1 von :citet:`Song.Durkan.Murray.ea.2021` liefert der Score-Verlust für die Grundgesamtheit mit der Likelihood-Gewichtung $w = g^2$ eine obere Schranke für die Kullback-Leibler-Divergenz von der Datenverteilung zur Modellverteilung. Die Arbeit beweist die Schranke mit Pfadmaßen. In dieser Aufgabe leiten Sie sie aus der Fokker-Planck-Gleichung aus :numref:`sec_mdl-fokker-planck` her. Dabei folgen Sie :citet:`Lu.Zheng.Bao.ea.2022`, die diesen Weg für die SDE (dort Satz C.1) und für die Wahrscheinlichkeitsfluss-ODE (dort Satz 3.1) einschlagen. Die Vorwärts-SDE $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\, dt + g(t)\, d\mathbf{W}$ habe auf $[0, T]$ die Randdichten $p_t$. In umgekehrter Zeit $\bar t = T - t$ lauten die exakte Zeitumkehr und das Modell

   $$\begin{aligned}d\mathbf{x} &= \left[ -\mathbf{f} + g^2\, \nabla \log p_t \right] d\bar t + g\, d\mathbf{B}_{\bar t}, \\ d\mathbf{x} &= \left[ -\mathbf{f} + g^2\, \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t) \right] d\bar t + g\, d\mathbf{B}_{\bar t},\end{aligned}$$

   jeweils ausgewertet bei $t = T - \bar t$. Die erste Gleichung hat die Randdichten $p_t$. Die zweite startet zum Zeitpunkt $\bar t = 0$ mit der A-priori-Verteilung $\pi$ und hat die Randdichten $q_t$. Nehmen Sie an, dass alle Dichten glatt und positiv sind und für jede partielle Integration hinreichend schnell abfallen.
    1. Schreiben Sie beide Fokker-Planck-Gleichungen in $\bar t$ in der Form $\partial_{\bar t} \rho = -\nabla \cdot (\rho\, \mathbf{u})$ und bestimmen Sie die Geschwindigkeiten $\mathbf{u}_p$ und $\mathbf{u}_q$.
    1. Zeigen Sie, dass $\frac{d}{d\bar t} D_{\textrm{KL}}(p_t \,\|\, q_t) = \mathbb{E}_{p_t}\left[ (\mathbf{u}_p - \mathbf{u}_q)^\top \nabla \log (p_t / q_t) \right]$ gilt.
    1. Beschränken Sie die rechte Seite nach oben durch $\tfrac12 g^2\, \mathbb{E}_{p_t} \| \mathbf{s}_{\boldsymbol{\theta}} - \nabla \log p_t \|^2$, integrieren Sie über $\bar t$ und leiten Sie daraus folgende Ungleichung her:

       $$\begin{aligned}D_{\textrm{KL}}(p_0 \,\|\, q_0) &\le D_{\textrm{KL}}(p_T \,\|\, \pi) \\ &\quad + \frac12 \int_0^T g(t)^2\, \mathbb{E}_{p_t} \left\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t) - \nabla \log p_t(\mathbf{x}) \right\|^2 dt.\end{aligned}$$

       Unter welcher Bedingung an $q_t$ gilt in der Schranke für die Ableitung Gleichheit?
    1. Wiederholen Sie (a) und (b) für die Wahrscheinlichkeitsfluss-ODE mit $\mathbf{s}_{\boldsymbol{\theta}}$ anstelle des Scores. Welcher Term, der (c) ermöglichte, fehlt? Warum liefert der Score-Verlust allein keine obere Schranke für $D_{\textrm{KL}}(p_0 \,\|\, q_0)$ mehr?
   :exlabel:`ex_mdl-score-matching-diffusion-flow-likelihood-bound`

<!--en:eef13283-->
1. [extended] **Gelernte Rauschvorhersage.** ●● Die Zelle, die `fit_eps` definiert, lernt die Rauschvorhersage mit 17 festen Merkmalen durch separate Anpassung auf jedem der $50$ Pegel. Ein Diffusionsmodell hat ein einziges Netz für alle Pegel. Bauen Sie es für dieselben Daten, die Ziehungen `draw(size)` aus der ungleich gewichteten standardisierten Mischung, mit dem DDPM-Rauschplan aus `beta` und `alpha_bar`.
    1. Trainieren Sie im Framework Ihres Tabs ein Netz zur Rauschvorhersage $\epsilon_{\boldsymbol{\theta}}(x_t, t)$ mit Eingabe $(x_t, t/1000)$ und zwei versteckten Schichten aus je $64$ tanh-Einheiten. Verwenden Sie Adam mit Lernrate $3\times10^{-3}$ für 4000 Schritte auf Batches aus je $256$ neuen Ziehungen. Minimieren Sie den einfachen Verlust mit $t$ gleichverteilt über die 1000 Pegel. Berechnen Sie die Wurzel des mittleren quadratischen Fehlers gegenüber der exakten Rauschvorhersage `eps_hat(xt, a, ws)` auf Samples aus $p_t$ für mehrere Pegel $t$. Wiederholen Sie die Berechnung für das Modell mit den angepassten Merkmalen aus dem Text auf den Pegeln, auf denen es trainiert wurde. Wo ist der jeweilige Fehler am größten?
    1. Erzeugen Sie mit `ddim` aus dem Rauschen `z` Samples bei $K = 10$ und $K = 50$ und außerdem mit dem Schritt des Ancestral Samplings aus :numref:`sec_mdl-ddpm-discretized-sde`. Verwenden Sie dafür die Schleife von `cfg_sample` mit dem Score des Netzes anstelle des gesteuerten Scores. Vergleichen Sie jede Stichprobe mit `x_exact`, `x_fit` und `x_true` anhand der Masse im schwächer gewichteten Modus, der Breiten beider Modi und der Statistik `ks`. Welcher Sampler reagiert am empfindlichsten auf den Fehler aus (a), und in welchem Schritt dieses Samplers geht der Fehler ein?
    1. Trainieren Sie erneut mit den Gewichten je Pegel, die $w = 1$ und der Likelihood-Gewichtung $w = \beta$ entsprechen. Multiplizieren Sie dazu den Verlust der Rauschvorhersage auf Pegel $t$ mit $w(t) / (1 - \bar{\alpha}_t)$ und wiederholen Sie (a) und (b). Aufgabe :exref:`ex_mdl-score-matching-diffusion-flow-weightings` sagt vorher, welche Pegel jede Gewichtung bevorzugt. Stimmen die Fehler damit überein?

   Geben Sie die Statistiken aus (b) für jede Kombination aus Gewichtung und Sampler in einer Tabelle an und stellen Sie den Fehler aus (a) als Funktion von $t$ grafisch dar.
   :exlabel:`ex_mdl-score-matching-diffusion-flow-learned-noise`

<!--en:796c9793-->
[Diskussionen](https://d2l.discourse.group/t/36840)

<!--en:4c36f03c-->
Denoising-Score-Matching und scorebasierte Diffusionsmodelle<br>**Sätze von Hyvärinen und Vincent · Tweedies Formel · Durchmischung bei der Langevin-Dynamik · DDPM und DDIM · Classifier-Free Guidance**

<!--en:85a36436-->
::: {.slide title="Wie lernen wir ein Score-Feld, das wir nicht auswerten können?"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:41e3fbf0-->
Beide generativen Rückwärtsprozesse, Andersons zeitumgekehrte SDE und die Wahrscheinlichkeitsfluss-ODE, benötigen nur eine unbekannte Größe: den Score $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ der verrauschten Datenverteilung.

<!--en:d193abc5-->
- **Warum Maximum-Likelihood bei energiebasierten Modellen an Grenzen stößt:** Eine flexible Dichte $p_{\boldsymbol{\theta}}(\mathbf{x}) = \frac{e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}}{Z_{\boldsymbol{\theta}}}$ benötigt die Normierungskonstante $Z_{\boldsymbol{\theta}} = \int_{\mathbb{R}^d} e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}\,d\mathbf{x}$, die in hohen Dimensionen praktisch nicht berechenbar ist.

<!--en:d70a001b-->
- **Der Score beseitigt die Normierungskonstante:** Beim Ableiten von $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ nach dem **Zustand $\mathbf{x}$** fällt $\log Z_{\boldsymbol{\theta}}$ weg:
  $$\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) = \nabla_{\mathbf{x}}\log p_{\boldsymbol{\theta}}(\mathbf{x}) = -\nabla_{\mathbf{x}}E_{\boldsymbol{\theta}}(\mathbf{x})$$

<!--en:7bcc9184-->
- **Das Trainingsparadox:** Wir wollen die Zielfunktion $J_{\text{ESM}}(\boldsymbol{\theta}) = \frac{1}{2}\mathbb{E}_{\mathbf{x}\sim p}\big[\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) - \nabla\log p(\mathbf{x})\|^2\big]$ des **expliziten Score-Matchings (ESM)** minimieren, haben aber nur $n = 4{,}096$ Datensamples $\mathbf{x}_i \sim p$ und können $\nabla\log p(\mathbf{x}_i)$ an keinem einzigen Punkt auswerten.

<!--en:704030a0-->
::: {.slide title="Implizites Score-Matching durch partielle Integration"}

<!--en:fefa6f64-->
[Identität von Hyvärinen (2005)]{.kicker}

<!--en:c7792de2-->
Multiplizieren wir das Quadrat in $J_{\text{ESM}}(\boldsymbol{\theta})$ aus und integrieren den gemischten Term $-\int \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \cdot (\nabla\log p(\mathbf{x}))\,p(\mathbf{x})\,d\mathbf{x} = -\int \mathbf{s}_{\boldsymbol{\theta}} \cdot \nabla p\,d\mathbf{x}$ partiell, so geht die Ableitung von der unbekannten Datendichte $p$ auf das Netz $\mathbf{s}_{\boldsymbol{\theta}}$ über:

<!--en:9d533eba-->
- **Geometrische Wirkung:** Die Minimierung von $\nabla \cdot \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) < 0$ erzeugt an den beobachteten Samples eine nach innen gerichtete **Senke** ($\nabla\cdot\mathbf{s}_{\boldsymbol{\theta}}$ negativ). Der quadratische Strafterm $\frac{1}{2}\|\mathbf{s}_{\boldsymbol{\theta}}\|^2$ verhindert, dass die Senke unendlich steil wird.

<!--en:6c9350e5-->
- **Rechenaufwand als Flaschenhals:** Die Divergenz $\nabla \cdot \mathbf{s}_{\boldsymbol{\theta}} = \operatorname{tr}(\partial\mathbf{s}_{\boldsymbol{\theta}}/\partial\mathbf{x})$ erfordert $d$ Rückwärtsdurchläufe pro Trainingsbeispiel (Millionen Durchläufe pro Bild).

<!--en:8770c1e8-->
::: {.slide title="Implizites Score-Matching an einer eindimensionalen Gauß-Verteilung überprüfen"}

<!--en:a7f9a231-->
[Plausibilitätsprüfung]{.kicker}

<!--en:702bf32a-->
Für Daten $x \sim \mathcal{N}(0, v)$ und das lineare Score-Modell $s_w(x) = -x/w$ lautet die implizite Zielfunktion $J(w) = \mathbb{E}[\frac{1}{2}s_w^2 + s_w'] = \frac{v}{2w^2} - \frac{1}{w}$. Ihre Ableitung $-\frac{v}{w^3} + \frac{1}{w^2} = 0$ liefert allein aus Samples $w = v$:

<!--en:d5c3d4c9-->
::: {.slide title="Regressionslemma: Kleinste Quadrate liefern E[Y | X]"}

<!--en:509d03bc-->
[Grundlegende Identität]{.kicker}

<!--en:08c77b39-->
Für jedes Paar $(X, Y)$ mit bedingtem Erwartungswert $\mathbf{m}(X) = \mathbb{E}[Y \mid X]$ verschwindet der gemischte Term nach der Turmeigenschaft:

<!--en:b989c76b-->
- **Folgerung ohne Spurberechnung:** Die Regression von $\mathbf{v}(X)$ auf den verrauschten Zielwert $Y$ jedes Samples hat **genau dieselben Gradienten und denselben Minimierer** wie die Regression auf den praktisch nicht berechenbaren bedingten Mittelwert $\mathbf{m}(X) = \mathbb{E}[Y \mid X]$.

<!--en:1798ab3b-->
::: {.slide title="Denoising-Score-Matching und der Satz von Vincent"}

<!--en:8efde7a9-->
[Score-Schätzung ohne Spurberechnung]{.kicker}

<!--en:757be9c9-->
Wir stören jedes unverrauschte Sample $\mathbf{x} \sim p$ mit gaußschem Rauschen der Stärke $\sigma$: $\tilde{\mathbf{x}} = \mathbf{x} + \sigma\boldsymbol{\epsilon}$ ($\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$), also $p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2\mathbf{I})$. Der Score der Randdichte $\nabla\log p_\sigma(\tilde{\mathbf{x}})$ ist unbekannt, doch den **bedingten Score** kennen wir für jedes Trainingspaar in geschlossener Form:

<!--en:48df4255-->
- **Identität von Vincent (Score der Randdichte = A-posteriori-Mittelwert der bedingten Scores):** Nach dem Satz von Bayes gilt
  $$\mathbb{E}\!\left[\nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \,\middle|\, \tilde{\mathbf{x}}\right] = \int \frac{\nabla_{\tilde{\mathbf{x}}}p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}{p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}\frac{p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})p(\mathbf{x})}{p_\sigma(\tilde{\mathbf{x}})}\,d\mathbf{x} = \frac{\nabla_{\tilde{\mathbf{x}}}\int p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})p(\mathbf{x})\,d\mathbf{x}}{p_\sigma(\tilde{\mathbf{x}})} = \nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}})$$

<!--en:cfabf417-->
- **Satz (Vincent, 2011):** Nach dem Regressionslemma lernen wir durch Minimierung des Verlusts beim **Denoising-Score-Matching (DSM)**
  $$J_{\text{DSM}}(\boldsymbol{\theta}) = \frac{1}{2}\mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}}\!\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x} + \sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2\right] = \frac{1}{2}\mathbb{E}_{\tilde{\mathbf{x}} \sim p_\sigma}\!\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) - \nabla\log p_\sigma(\tilde{\mathbf{x}})\right\|^2\right] + C$$
  den Score $\nabla\log p_\sigma$ mit der gewöhnlichen Methode der kleinsten Quadrate, **ohne die Spur einer Jacobi-Matrix zu berechnen**.

<!--en:8dac1ee7-->
::: {.slide title="Tweedies Formel und die positive Untergrenze des DSM-Verlusts"}

<!--en:45f2af35-->
[Optimale Entrauschung]{.kicker}

<!--en:667f5262-->
Umstellen der Identität von Vincent $\nabla\log p_\sigma(\tilde{\mathbf{x}}) = \mathbb{E}\!\left[\frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2} \,\middle|\, \tilde{\mathbf{x}}\right]$ ergibt **Tweedies Formel** für den MMSE-Schätzer $\hat{\mathbf{x}}_0(\tilde{\mathbf{x}}) = \mathbb{E}[\mathbf{x} \mid \tilde{\mathbf{x}}]$ des unverrauschten Punkts:

<!--en:e817eb86-->
- **Tweedies Formel:**
  $$\mathbb{E}[\mathbf{x} \mid \tilde{\mathbf{x}}] = \tilde{\mathbf{x}} + \sigma^2\,\nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}})$$

<!--en:29737b0b-->
- **Rauschvorhersage ($\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$):** Mit der Parametrisierung $\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) = -\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})/\sigma$ ist Score-Matching gleichbedeutend mit der Vorhersage des hinzugefügten Rauschens $\boldsymbol{\epsilon}$.

<!--en:868b1f4c-->
- **Irreduzible Untergrenze des DSM-Verlusts ($2J_{\text{DSM}}^\star$):** Setzen wir $\mathbf{v} = \mathbf{0}$ im Regressionslemma, so erhalten wir
  $$\mathbb{E}\!\left[\left\|-\frac{\boldsymbol{\epsilon}}{\sigma} - \nabla\log p_\sigma(\tilde{\mathbf{x}})\right\|^2\right] = \frac{d}{\sigma^2} - \mathbb{E}_{p_\sigma}\|\nabla\log p_\sigma\|^2$$

<!--en:5b022c88-->
::: {.slide title="Untergrenze des DSM-Verlusts und Krümmungsschranke überprüfen"}

<!--en:66fbf18e-->
[Gauß-Beispiel und Krümmung]{.kicker}

<!--en:d264d32b-->
Für gaußverteilte Daten $x \sim \mathcal{N}(0, v)$ mit $v = 1$ und Rauschen $\sigma = 0.5$ lautet die verrauschte Randverteilung $p_\sigma = \mathcal{N}(0, 1.25)$. Tweedies optimaler Entrauscher liefert den geschrumpften Schätzwert $\frac{v}{v+\sigma^2}\tilde{x} = 0.8\,\tilde{x}$, und die Untergrenze für den mittleren quadratischen Fehler beträgt $\frac{1}{\sigma^2} - \frac{1}{v+\sigma^2} = 4 - 0.8 = 3.2 > 0$:

<!--en:b2e6ec1f-->
**Krümmungsschranke für jede verrauschte Dichte:** Eine weitere Ableitung von Tweedies Formel nach $\tilde{\mathbf{x}}$ ergibt
$$\nabla^2\log p_\sigma(\tilde{\mathbf{x}}) = -\frac{\mathbf{I}}{\sigma^2} + \frac{1}{\sigma^4}\operatorname{Cov}[\mathbf{x} \mid \tilde{\mathbf{x}}] \quad \Longrightarrow \quad -\nabla^2\log p_\sigma(\tilde{\mathbf{x}}) \preceq \frac{1}{\sigma^2}\mathbf{I}$$
Kein Gipfel einer mit $\sigma$ geglätteten Dichte kann stärker gekrümmt sein als der Gauß-Kern $\mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$, der sie geglättet hat.

<!--en:952b36b0-->
::: {.slide title="Gekrümmten Score einer bimodalen Verteilung mit kleinsten Quadraten anpassen"}

<!--en:e18bfdaf-->
[Versuch mit eindimensionaler Mischung]{.kicker}

<!--en:52594469-->
Wir verrauschen $4{,}096$ Ziehungen aus $p = \frac{1}{2}\mathcal{N}(-2, 0.5^2) + \frac{1}{2}\mathcal{N}(2, 0.5^2)$ je viermal bei $\sigma = 0.5$ ($16{,}384$ Paare). Die Anpassung von 17 festen Basisfunktionen mit `np.linalg.lstsq` an $y = -\epsilon/\sigma$ rekonstruiert den exakten Score $4\tanh(4\tilde{x}) - 2\tilde{x}$:

<!--en:9eb9eec6-->
::: {.slide title="Gekrümmten Score einer bimodalen Verteilung mit kleinsten Quadraten anpassen (Fortsetzung)"}

<!--en:e18bfdaf-2-->
[Versuch mit eindimensionaler Mischung]{.kicker}

<!--en:61a64320-->
Die Wolke der Zielwerte $-\epsilon/\sigma$ streut vertikal mit Varianz $\approx 2.11$ (der Untergrenze für die Grundgesamtheit). Ihr bedingter Mittelwert ist die exakte Score-Kurve, und die Kleinste-Quadrate-Anpassung erreicht ein Überschussrisiko von nur $0.01$.

<!--en:1cbd73c5-->
::: {.slide title="Risiko der Extrapolation in Bereichen ohne Trainingspaare"}

<!--en:ea196aad-->
[Varianz außerhalb des Trägers]{.kicker}

<!--en:45131209-->
Wiederholen wir die Kleinste-Quadrate-Anpassung für 20 unabhängige Ziehungen von Daten und Rauschen, zeigt sich eine grundlegende Grenze des Score-Matchings bei nur einem Rauschpegel:

<!--en:90684651-->
- **Genau innerhalb des Trägers ($x = 1$):** Wo viele verrauschte Paare liegen, liegen alle 20 Anpassungen im Bereich $\pm 0.2$ um den wahren Score $+2.0$.

<!--en:4aabbecc-->
- **Beliebig weit draußen in den Ausläufern ($x = 8$):** Jenseits von $|\tilde{x}| \approx 5$ liegt kein Trainingspaar. Der Regressionsverlust schränkt den Wert bei $x = 8$ daher überhaupt nicht ein. Der angepasste Score schwankt um mehrere Dutzend Einheiten und wechselt sogar das Vorzeichen (er zeigt von den Daten *weg*).

<!--en:0eb40958-->
::: {.slide title="Sampling mit einem Score durch Langevin-Dynamik"}

<!--en:3d8de7e6-->
[MCMC mit dem Score]{.kicker}

<!--en:b9dd1749-->
Bei gegebenem Score-Feld $\nabla\log p(\mathbf{x})$ für einen einzigen Rauschpegel steigt die **Langevin-Dynamik** entlang der Log-Dichte auf. Brownsches Rauschen mit Einheitsstärke verhindert, dass alle Punkte in einem Modus zusammenfallen:

<!--en:343e4161-->
- **Satz ($p$ ist die stationäre Verteilung):** Setzen wir die Drift $\mathbf{b} = \frac{1}{2}\nabla\log p$, die Einheitsdiffusion $g = 1$ und $q_t = p$ in die Fokker-Planck-Gleichung ein, erhalten wir
  $$\partial_t q_t = -\nabla \cdot \left(\frac{1}{2}p\,\nabla\log p\right) + \frac{1}{2}\Delta p = -\frac{1}{2}\nabla\cdot(\nabla p) + \frac{1}{2}\Delta p = 0$$

<!--en:9989a89b-->
- **Verzerrung der Varianz durch die Schrittweite:** Für $p = \mathcal{N}(0, v)$ hat die diskrete Rekursion $x_{k+1} = (1 - \frac{h}{2v})x_k + \sqrt{h}\xi_k$ die stationäre Varianz $V = \frac{v}{1 - h/(4v)} \approx v + \frac{h}{4}$ und überschätzt die wahre Varianz um $O(h)$.

<!--en:77ddc27c-->
::: {.slide title="Warum die Langevin-Dynamik bei einem Rauschpegel an multimodalen Daten scheitert"}

<!--en:7ea3c2a9-->
[Langsame Durchmischung zwischen Modi]{.kicker}

<!--en:60f90a36-->
Was geschieht, wenn wir $10{,}000$ Langevin-Ketten für $2{,}000$ Schritte ($h = 0.01$) mit dem **exakten Score** der ungleich gewichteten Mischung $p = 0.8\,\mathcal{N}(-2, 0.5^2) + 0.2\,\mathcal{N}(2, 0.5^2)$ laufen lassen?

<!--en:095426d4-->
- **Exponentiell seltene Talquerung:** Um das Tal niedriger Dichte um $x = 0$ gegen die rücktreibende Kraft zu durchqueren, braucht es eine exponentiell seltene Folge zufälliger Stöße (Übergangsrate $0.5\%$ aus dem stärkeren Modus, $2$–$2.5\%$ aus dem schwächeren. Die Durchmischung braucht $\sim 67{,}000$ Schritte).

<!--en:8f6c5281-->
- **Falsche Anteile der Modi ($0.48$ gegenüber $0.20$):** Ketten aus der symmetrischen Anfangsverteilung $\mathcal{N}(0, 3^2)$ gleiten in den nächsten Modus und bleiben dort. So behalten sie die Anteile $50/50$ der Initialisierung anstelle der Datengewichte $80/20$.

<!--en:5f9ddf53-->
::: {.slide title="Wie mehrere Rauschpegel alle drei Probleme lösen"}

<!--en:2cbc2217-->
[Score-Matching bedingt auf den Rauschpegel]{.kicker}

<!--en:7248639d-->
Trainieren wir ein einziges zeitabhängiges Score-Netz $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ auf einem Kontinuum von Pegeln $\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\boldsymbol{\epsilon}$ gaußschen Rauschens, beheben wir alle drei Probleme des Score-Matchings bei nur einem Pegel:

<!--en:5b61374d-->
1. **Keine leeren Bereiche bei starkem Rauschen:** Bei großem $\sigma_t$ füllt die verrauschte Dichte den gesamten Raum aus, in dem der Sampler startet.

<!--en:77f5f2bb-->
2. **Keine Barriere im Tal ($\sigma \ge 1.94$):** Sobald für die Komponentenvarianz $0.25 + \sigma^2 \ge \mu^2 = 4$ gilt, verschwindet das Tal zwischen den beiden Modi. Übrig bleibt eine einzige Glockenkurve.

<!--en:65af7374-->
3. **Richtige Anteile der Modi:** Wenn das Rauschen beim Rückwärtssampling abnimmt, teilt die Trennlinie im Score-Feld bei mittlerem $\sigma_t$ die Teilchen genau im Verhältnis der Mischungsgewichte $(0.8, 0.2)$ auf.

<!--en:295c4ca9-->
::: {.slide title="Rauschpegelabhängiger DSM-Verlust und Gewichtung der Pegel"}

<!--en:c73ccb7c-->
[Training über alle Pegel]{.kicker}

<!--en:d6533f4c-->
Wir mitteln den DSM-Verlust über die Rauschpegel $t \sim \mathcal{U}[\varepsilon, T]$ mit Gewichtungsfunktion $w(t) > 0$:

<!--en:9c6ed8c2-->
- **Drei Gewichtungen:** Ohne Gewichtung ($w(t) = 1$) wächst der Verlust wie $d/\sigma_t^2$. **Einfacher DDPM-Verlust:** $w(t) = \sigma_t^2$ gleicht die Untergrenze über alle Pegel auf $d = O(1)$ an ($\tfrac12\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\|^2$). **Likelihood:** $w(t) = g(t)^2$ liefert eine Schranke für die KL-Divergenz der Rückwärts-SDE.

<!--en:5795d9e0-->
::: {.slide title="DDPM ist die varianzerhaltende SDE in diskreter Zeit"}

<!--en:94d4e0b3-->
[Von diskreter zu kontinuierlicher Zeit]{.kicker}

<!--en:f1fb2243-->
Das **Denoising Diffusion Probabilistic Model (DDPM)** durchläuft eine Markov-Kette mit $N = 1000$ Schritten nach dem Plan $\beta_1, \ldots, \beta_N \in [10^{-4}, 0.02]$:

<!--en:00a0dbb3-->
- **Exakter Übergang der VP-SDE:** Jeder DDPM-Schritt ist ein *exakter* Übergang der VP-SDE $d\mathbf{X} = -\frac{1}{2}\beta(s)\mathbf{X}\,ds + \sqrt{\beta(s)}\,d\mathbf{W}$ über ein Intervall mit $\int \beta(s)\,ds = -\log(1 - \beta_t) \approx \beta_t$.

<!--en:deeeb2d8-->
- **Mit einer Ziehung zu jedem Pegel:** Das Training simuliert die Kette mit 1000 Schritten nie. Es zieht $\mathbf{x}_t$ direkt ausgehend von $\mathbf{x}_0$ in Zeit $O(1)$, während das Rausch-Signal-Verhältnis $\bar{\sigma}_t = \sqrt{(1-\bar{\alpha}_t)/\bar{\alpha}_t}$ von $0.01$ auf $157$ wächst.

<!--en:423f39bb-->
::: {.slide title="Einfacher DDPM-Verlust und Ancestral Sampling für die Rückwärts-SDE"}

<!--en:14d08712-->
[Rauschvorhersage und Sampling]{.kicker}

<!--en:015aed61-->
Die Parametrisierung des Score-Modells durch ein Netz zur Rauschvorhersage $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) = -\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) / \sigma_t$ ($\sigma_t = \sqrt{1 - \bar{\alpha}_t}$) verbindet Training und Sampling bei DDPM mit scorebasierten SDEs:

<!--en:49ce4262-->
- **DDPM-Verlust zur Rauschvorhersage:** Setzen wir $\mathbf{s}_{\boldsymbol{\theta}} = -\boldsymbol{\epsilon}_{\boldsymbol{\theta}}/\sigma_t$ in den rauschpegelabhängigen DSM-Verlust mit Gewicht $w(t) = \sigma_t^2 = 1 - \bar{\alpha}_t$ ein, kürzt sich der Faktor $1/\sigma_t^2$ heraus:
  $$\frac{1}{2}\sigma_t^2\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \frac{\boldsymbol{\epsilon}}{\sigma_t}\right\|^2 = \frac{1}{2}\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)\right\|^2$$

<!--en:4b3d0aff-->
- **Ancestral Sampling als Euler-Maruyama-Verfahren in Rückwärtszeit:** Der Übergang von $t = N$ bis hinunter zu $1$ mit
  $$\mathbf{x}_{t-1} = \frac{\mathbf{x}_t + \beta_t\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)}{\sqrt{1 - \beta_t}} + \sqrt{\beta_t}\,\boldsymbol{\xi}_t \approx \mathbf{x}_t + \left(\frac{1}{2}\beta_t\mathbf{x}_t + \beta_t\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)\right) + \sqrt{\beta_t}\,\boldsymbol{\xi}_t$$
  entspricht bis zur ersten Ordnung in $\beta_t$ der Euler-Maruyama-Diskretisierung von Andersons zeitumgekehrter VP-SDE ($1{,}000$ Netzauswertungen pro erzeugtem Sample).

<!--en:59523ac5-->
::: {.slide title="DDIM: schnelles deterministisches Sampling mit dem expliziten Euler-Verfahren"}

<!--en:cf502968-->
[Pegel mit der Wahrscheinlichkeitsfluss-ODE überspringen]{.kicker}

<!--en:1466e929-->
**DDIM** sagt $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}})/\sqrt{\bar{\alpha}_t}$ vorher und springt direkt zu einem früheren Pegel $t' < t$. In den umskalierten Koordinaten $\bar{\mathbf{x}}_t = \mathbf{x}_t/\sqrt{\bar{\alpha}_t}$ und $\bar{\sigma}_t = \sqrt{(1-\bar{\alpha}_t)/\bar{\alpha}_t}$ gilt:

<!--en:9e12dd92-->
- **Exakter expliziter Euler-Schritt:** DDIM ist das explizite Euler-Verfahren erster Ordnung für die VP-Wahrscheinlichkeitsfluss-ODE $\frac{d\bar{\mathbf{x}}}{d\bar{\sigma}} = \mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$. Dadurch genügen 20–50 deterministische Schritte.

<!--en:822e4d54-->
::: {.slide title="Schrittfehler bei DDIM: Punktmasse und Schrumpfung bei gaußverteilten Daten"}

<!--en:802972f9-->
[Analyse des Diskretisierungsfehlers von DDIM]{.kicker}

<!--en:6f61b8af-->
DDIM verwendet das Euler-Verfahren erster Ordnung mit $\bar{\sigma}$ als Zeitvariable. Sein Schrittfehler hängt daher davon ab, wie stark sich $\hat{\boldsymbol{\epsilon}}(\mathbf{x}_t, t) = \mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$ entlang der Trajektorie des Wahrscheinlichkeitsflusses krümmt:

<!--en:ae187f07-->
- **Exakt für eine einzelne Punktmasse:** Gilt $p_0 = \delta_{\mathbf{x}_0}$, so ist $\hat{\boldsymbol{\epsilon}} = (\bar{\mathbf{x}} - \mathbf{x}_0)/\bar{\sigma}$ entlang des exakten Strahls konstant. Daher ist **ein DDIM-Schritt beliebiger Länge exakt**.

<!--en:3f9af5c2-->
- **Kontraktion bei Daten aus $\mathcal{N}(\mathbf{0}, \mathbf{I})$:** Der exakte Fluss ist die Identität. Halten wir aber $\hat{\boldsymbol{\epsilon}} = \sqrt{1-\bar{\alpha}_t}\mathbf{x}_t$ während eines ganzen Schritts fest, schrumpft jedes Sample um den Faktor
  $$\sqrt{\bar{\alpha}_t\bar{\alpha}_{t'}} + \sqrt{(1-\bar{\alpha}_t)(1-\bar{\alpha}_{t'})} = \cos(\phi_t - \phi_{t'}) < 1$$
  mit $\bar{\alpha}_t = \cos^2\phi_t$ ($0.832\times$ bei 10 Schritten, $0.964\times$ bei 50 Schritten).

<!--en:6af1bc06-->
::: {.slide title="Kontraktion und Konvergenz von DDIM messen"}

<!--en:60ed9f76-->
[10, 50 und 999 Schritte im Vergleich]{.kicker}

<!--en:18501b5f-->
Wir vergleichen 10, 50 und 999 gleich lange DDIM-Schritte (bei 999 einer je Pegel) mit exakter Rauschvorhersage $\hat{\epsilon} = -\sqrt{1-\bar{\alpha}_t}\nabla\log p_t$ auf Daten aus $\mathcal{N}(0, 1)$ und auf der standardisierten bimodalen Mischung:

<!--en:0a53831d-->
- **Euler-Konvergenz erster Ordnung:** Erhöhen wir die Schrittzahl von $10$ auf $50$ (um den Faktor $5\times$ feinere Schritte), sinkt der Schrumpfungsfehler der Gauß-Verteilung um den Faktor $4.7\times$ ($1 - 0.832 = 0.168 \to 1 - 0.964 = 0.036$). Bei der Mischung sinkt der Endpunktabstand von etwa $0.08$ auf etwa $0.015$.

<!--en:a6df0def-->
::: {.slide title="DDIM mit Kleinste-Quadrate-Scores rekonstruiert die Mischungsgewichte"}

<!--en:89b09f02-->
[Durchgängiges Lernen über mehrere Pegel]{.kicker}

<!--en:f2e9fb3a-->
Wir passen unser lineares Modell mit 17 Merkmalen durch Ridge-Regression auf jedem der $50$ Rauschpegel an, die ein DDIM-Sampler mit 50 Schritten auf der ungleich gewichteten $0.8 / 0.2$-Mischung besucht:

<!--en:d7086db5-->
**Regression über mehrere Pegel erfasst die Mischungsgewichte:** Die Langevin-Dynamik mit dem *exakten* Score eines einzelnen Pegels ließ $48\%$ der Samples im schwächer gewichteten Modus. DDIM mit 50 Schritten und *durch kleinste Quadrate angepassten* Scores reproduziert seine wahre Masse $20\%$ bis auf ein oder zwei Hundertstel. Der KS-Abstand zum DDIM-Ergebnis mit exakten Scores beträgt nur wenige Hundertstel.

<!--en:0f3c9b85-->
::: {.slide title="Klassifikator-Guidance und Classifier-Free Guidance mit dem Satz von Bayes"}

<!--en:a4a59926-->
[Klassenbedingte Steuerung]{.kicker}

<!--en:79cc1b91-->
Nach dem Satz von Bayes gilt $\nabla\log p_t(y \mid \mathbf{x}) = \mathbf{s}_y - \mathbf{s}_\varnothing$. **Classifier-Free Guidance (CFG)** verstärkt diesen Klassenunterschied mit der Guidance-Skala $\gamma \ge 1$:

<!--en:9aea8e3c-->
- **Verletzte Krümmungsschranke ($\gamma > 1$):** Bei großem $\gamma$ überschreitet $-\nabla^2\log\tilde{p}_t$ die universelle Schranke $1/\sigma_t^2$ für jede tatsächlich verrauschte Dichte. Der Modus am Ende wird schärfer und verschiebt sich nach außen.

<!--en:3d9a63eb-->
::: {.slide title="Verschiebung des Modus bei starker Steuerung messen"}

<!--en:dd02b6e1-->
[Versuch zur Steuerung]{.kicker}

<!--en:29f62c69-->
Wir steuern die Rückwärtskette beim Ancestral Sampling in 1000 Schritten zum rechten Modus ($y = \mathcal{N}(+0.970, 0.243^2)$) und variieren die Guidance-Skala $\gamma \in \{1, 3, 10\}$:

<!--en:dc29e42e-->
Bei $\gamma = 1$ (exakter Satz von Bayes) reproduziert die Kette den exakten Klassenmittelwert $0.970$ und die Standardabweichung $0.243$. Bei $\gamma = 3$ und $\gamma = 10$ treibt die lineare Extrapolation die Samples vom gegenüberliegenden linken Modus *weg*. Der Mittelwert am Ende verschiebt sich nach außen auf $1.03$ beziehungsweise $1.08$.

<!--en:54fd97ba-->
::: {.slide title="Verständnisfragen zu Score-Matching, DDIM und Steuerung"}

<!--en:26a317d8-->
[Verständnisfragen]{.kicker}

<!--en:00903f92-->
Prüfen Sie Ihr Verständnis von Score-Matching und Diffusionssamplern:

<!--en:8dfdd261-->
1. **DSM-Verlust größer als null:** Sie trainieren ein Score-Netz hoher Kapazität auf $d = 100$-dimensionalen Daten bei Rauschpegel $\sigma = 0.1$. Der DSM-Verlust $2J_{\text{DSM}}$ pendelt sich nahe $9{,}800$ ein. Deutet dieser große Verlust auf Unteranpassung hin?

<!--en:c5957d32-->
2. **Tweedies Entrauscher:** Bei einer verrauschten Beobachtung $\tilde{\mathbf{x}}$ mit Rauschpegel $\sigma = 0.5$ liefert der gelernte Score den Wert $\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) = (+4, -2)$. Wie lautet die MMSE-Schätzung $\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}]$?

<!--en:3a916364-->
3. **Linearkombination bei CFG:** Welche Gewichte erhalten bei Classifier-Free Guidance mit Guidance-Skala $\gamma = 7.5$ der bedingte Score $\mathbf{s}_y$ und der unbedingte Score $\mathbf{s}_\varnothing$?

<!--en:557b98b6-->
**Antworten:**

<!--en:1b79ae84-->
1. **Nein.** Nach dem Regressionslemma liegt die irreduzible Untergrenze, das Bayes-Risiko, bei $\frac{d}{\sigma^2} - \mathbb{E}\|\nabla\log p_\sigma\|^2 = \frac{100}{0.01} - \mathbb{E}\|\nabla\log p_\sigma\|^2 = 10{,}000 - \mathbb{E}\|\nabla\log p_\sigma\|^2$. Ein Verlust von $9{,}800$ kann einer global optimalen Anpassung entsprechen.

<!--en:e7974a6b-->
2. Nach Tweedies Formel gilt $\hat{\mathbf{x}}_0 = \tilde{\mathbf{x}} + \sigma^2 \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) = \tilde{\mathbf{x}} + 0.25\,(4, -2) = \tilde{\mathbf{x}} + (1, -0.5)$.

<!--en:dee76533-->
3. $\tilde{\mathbf{s}} = (1 - \gamma)\mathbf{s}_\varnothing + \gamma\mathbf{s}_y = -6.5\,\mathbf{s}_\varnothing + 7.5\,\mathbf{s}_y$ (lineare Extrapolation über $\mathbf{s}_y$ hinaus, weg von $\mathbf{s}_\varnothing$).

<!--en:0e956392-->
::: {.slide title="Zusammenfassung: Score-Matching und Diffusionssampler"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:ac413e33-->
| Methode oder Satz | Formel | Funktion und praktische Eigenschaften |

<!--en:274f0faa-->
| **Identität von Hyvärinen (ISM)** | $\mathbb{E}_p\big[\frac{1}{2}\|\mathbf{s}_{\boldsymbol{\theta}}\|^2 + \nabla\cdot\mathbf{s}_{\boldsymbol{\theta}}\big] + C$ | Beseitigt den unbekannten Score $\nabla\log p$, erfordert aber $O(d)$ Rückwärtsdurchläufe für $\nabla\cdot\mathbf{s}_{\boldsymbol{\theta}}$ |

<!--en:a38de083-->
| **Satz von Vincent (DSM)** | $\frac{1}{2}\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\big\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \boldsymbol{\epsilon}/\sigma_t\big\|^2$ | Regression auf $-\boldsymbol{\epsilon}/\sigma_t$ ohne Spurberechnung. Die Untergrenze im Optimum beträgt $\frac{d}{\sigma_t^2} - \mathbb{E}\|\nabla\log p_t\|^2$ |

<!--en:8979db5a-->
| **Tweedies Formel** | $\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}] = \tilde{\mathbf{x}} + \sigma^2\nabla\log p_\sigma(\tilde{\mathbf{x}})$ | Liefert aus jedem Score oder Rauschprädiktor den MMSE-Entrauscher für den unverrauschten Punkt |

<!--en:f5b8f8ac-->
| **DDPM-Vorwärtsprozess und Verlust** | $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$, $w(t) = \sigma_t^2$ | Exakte diskrete Übergänge der VP-SDE. Das Gewicht $\sigma_t^2$ hebt das Wachstum der Untergrenze wie $1/\sigma_t^2$ auf |

<!--en:47d4fb7d-->
| **Deterministischer DDIM-Schritt** | $\bar{\mathbf{x}}_{t'} = \bar{\mathbf{x}}_t + (\bar{\sigma}_{t'} - \bar{\sigma}_t)\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ | Explizites Euler-Verfahren für die VP-Wahrscheinlichkeitsfluss-ODE in den Koordinaten $(\bar{\sigma}, \bar{\mathbf{x}})$ |

<!--en:31944f5b-->
| **Classifier-Free Guidance** | $\tilde{\mathbf{s}} = (1 - \gamma)\mathbf{s}_\varnothing + \gamma\mathbf{s}_y$ | Verstärkt die Richtung zur Klasse nach dem Satz von Bayes. Bei $\gamma > 1$ wird die Krümmungsschranke $1/\sigma_t^2$ verletzt |
