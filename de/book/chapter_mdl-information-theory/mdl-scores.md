<!--en:49cd66f0-->
# Score-basierte Vergleiche

<!--en:f73ee0d3-->
Eine Softmax-Schicht wandelt zehn Logits in Wahrscheinlichkeiten um. Dazu exponenziert sie die Logits und teilt die Ergebnisse durch ihre Summe. Über diese Summe denkt kaum jemand nach. Stellen wir uns nun ein neuronales Netz vor, das jeder möglichen Eingabe einen Score zuweist statt nur zehn Klassen. Üblicherweise kehrt man das Vorzeichen um. Das Netz gibt eine Energie $E(\mathbf{x})$ aus, die für plausible Eingaben niedrig ist (wo ein Logit hoch wäre) und für unplausible Eingaben hoch. Damit definiert es eine Dichte proportional zu $\tilde{q}(\mathbf{x}) = e^{-E(\mathbf{x})}$. Aus der Summe über zehn Klassen wird ein Integral über alle möglichen Eingaben, $Z = \int \tilde{q}(\mathbf{x})\, d\mathbf{x}$. In hohen Dimensionen ist es praktisch unmöglich, dieses Integral zu berechnen. Ein solches *energiebasiertes Modell* gibt an, wie plausibel eine Eingabe im Vergleich zu einer anderen ist. Ihre tatsächliche Wahrscheinlichkeit liefert es uns nicht.

<!--en:8b06a213-->
Keiner unserer bisherigen Verteilungsvergleiche eignet sich für diesen Fall. Maximum-Likelihood setzt die normierte Dichte $q = \tilde{q}/Z$ voraus. Stichprobenbasierte Verfahren kommen ohne die Normierung aus, ob sie auf einem Kritiker (:numref:`sec_mdl-f-gan-dual`), einem Kernel oder einem Transportplan (:numref:`sec_mdl-test-functions-transport`) beruhen. Sie umgehen zwar $Z$, brauchen aber Samples aus dem Modell. Eine Energiefunktion liefert diese nicht direkt. Für jedes Sample ist eine lange Irrfahrt nötig, die in vielen kleinen, verrauschten Schritten zu niedrigeren Energien führt. Jeder Schritt erfordert eine Gradientenberechnung durch das Netz (:numref:`sec_mdl-langevin-sampling`).

<!--en:cd2e061c-->
Ein einzelner Gradient ist dagegen günstig zu berechnen. Er kostet nur einen Rückwärtsdurchlauf, und wie sich zeigt, genügt er bereits. Der Gradient von $\log \tilde{q}$ nach der Eingabe $\mathbf{x}$ misst die Steigung der Log-Dichte. Dabei fällt $Z$ vollständig weg, weil die Steigung einer Konstanten null ist. Dieser Gradient heißt *Score*. In diesem Abschnitt vergleichen wir Verteilungen allein anhand ihrer Scores. Die mittlere quadratische Differenz zweier Scores ergibt die Zielfunktion des Score-Matchings, das Diffusionsmodellen zugrunde liegt (:numref:`chap_diffusion`). Wer ein Score-basiertes Modell trainiert, sollte allerdings eine wichtige Grenze dieses Vergleichs kennen. Eine Steigung ist rein lokal und verrät daher nicht, wie sich die Wahrscheinlichkeitsmasse auf weit voneinander entfernte Modi verteilt. Wir messen, wie stark sich dieser blinde Fleck auswirkt.

<!--en:41157235-->
Wie in den vorigen Abschnitten bezeichnet $P$ die Datenverteilung, die wir nur anhand von Stichproben beobachten. Mit $Q$ bezeichnen wir das Modell, von dem wir jetzt nur den Score $s_Q$ kennen. Zwei Fragen stehen im Mittelpunkt. Was verrät uns die Differenz der Steigungen über zwei Verteilungen, und was entgeht uns dabei? Wie können wir Steigungen überhaupt vergleichen, wenn unsere Daten als endliche Punktmenge gar keine Steigung haben? :numref:`sec_mdl-score-matching` baut direkt auf den Antworten auf. Wir untersuchen auch, wie sich prüfen lässt, ob eine Stichprobe aus einem Modell stammt, aus dem wir selbst keine Samples ziehen können. Bei der ersten Lektüre können Sie diese dritte Frage überspringen. Zum Schluss fassen wir alle Vergleiche dieses Kapitels in einer Tabelle der Trainingsziele zusammen und geben Hinweise für die Auswahl.

<!--en:3ff466c4-->
Als durchgängige Beispiele dienen uns eindimensionale Normalverteilungen und Mischungen aus zwei Normalverteilungen. Wir beginnen mit drei kleinen Hilfsfunktionen. `log_components` und `log_mix` berechnen die Log-Dichte einer Gauß-Mischung numerisch stabil. Sie klammern den größten Term aus und vermeiden so einen Unterlauf in den Randbereichen (:numref:`subsec_mdl-stable-softmax`). Die Funktion `kl` berechnet auf einem Gitter die KL-Divergenz zwischen zwei Dichten in Nat.

<!--en:dce25bcd-->
## Der Score und die Fisher-Divergenz

<!--en:c442c349-->
### Der Score

<!--en:c7d0ebac-->
An jedem Punkt im Raum beantwortet der Score eine geometrische Frage. In welche Richtung steigt die Dichte, und wie steil? Für eine Verteilung $P$ mit differenzierbarer Dichte $p$ ist der **Score** formal der Gradient der Log-Dichte *nach dem Datenpunkt*:

<!--en:1d394a27-->
In der klassischen Statistik hat „Score“ eine andere Bedeutung. Dort bezeichnet er den Gradienten der Log-Likelihood nach den *Parametern* (:numref:`sec_mdl-maximum_likelihood`). Hier leiten wir nach $\mathbf{x}$ ab. Für eine eindimensionale Normalverteilung $\mathcal{N}(\mu, \sigma^2)$ ergibt sich der Score

<!--en:1573d98d-->
der wie eine Feder wirkt. Sie zieht $x$ zum Mittelwert $\mu$ zurück und hat die Federkonstante $1/\sigma^2$. Für eine Mischung $p = \sum_k w_k\, p_k$ ergibt die Kettenregel $s(x) = \sum_k r_k(x)\, s_k(x)$. Dabei ist $r_k(x) = w_k p_k(x) / p(x)$ die A-posteriori-Wahrscheinlichkeit oder *Zugehörigkeitswahrscheinlichkeit* (engl. responsibility) dafür, dass $x$ aus Komponente $k$ stammt. Auf jeden Punkt wirkt also eine gewichtete Kombination der Federkräfte der Komponenten. In der Nähe eines Modus liegt die zugehörige Wahrscheinlichkeit nahe bei $1$, sodass praktisch nur die Feder dieses Modus wirkt. Im Tal zwischen den Modi verschieben sich die Zugehörigkeitswahrscheinlichkeiten stetig, und der Einfluss geht von einem Attraktor auf den nächsten über, wie :numref:`fig_mdl-score-field` zeigt. Im folgenden Code setzen wir die Formel direkt um. Für die numerische Stabilität berechnen wir die Zugehörigkeitswahrscheinlichkeiten im logarithmischen Raum.

<!--en:9dfb0f56-->
![Score-Feld einer Gauß-Mischung mit zwei Komponenten. Die Pfeile zeigen den Gradienten der Log-Dichte. In der Nähe jedes Modus wirkt das Feld wie eine Feder, die den Punkt zum jeweiligen Mittelwert zieht. Im Tal zwischen den Modi bestimmen die Zugehörigkeitswahrscheinlichkeiten den Übergang vom einen Attraktor zum anderen. Zur Orientierung sind die Höhenlinien der Dichte hinterlegt.](../img/mdl-it-score-field.svg)

<!--en:dc3df66e-->
Für energiebasierte Modelle ist der Score deshalb so nützlich, weil jede Normierungskonstante wegfällt. Gilt $q(\mathbf{x}) = \tilde{q}(\mathbf{x}) / Z$, so folgt

<!--en:b6ee908d-->
denn $Z$ hängt nicht von $\mathbf{x}$ ab. Die unnormierte Dichte mit quartischem Energieterm $\tilde{q}(x) = e^{-x^4/4}$ hat zum Beispiel den Score $-x^3$. Wir können ihn überall auswerten, ohne den Wert $Z \approx 2.56$ zu kennen.

<!--en:da560fc6-->
### Die Fisher-Divergenz

<!--en:7986d107-->
Da $Z$ beim Score wegfällt, lässt sich jeder Score-basierte Vergleich direkt auf unnormierte Modelle anwenden. Naheliegend ist die über die Datenverteilung gemittelte quadratische Differenz der beiden Score-Felder, die **Fisher-Divergenz**:

<!--en:7fd99530-->
Offensichtlich gilt $D_{\textrm{F}}(P\|Q) \geq 0$. Es handelt sich um eine echte Divergenz, die genau dann null ist, wenn $P = Q$ gilt (:numref:`sec_mdl-what-is-a-divergence`), sofern beide Dichten auf demselben zusammenhängenden Gebiet strikt positiv sind. Stimmen die Scores auf einem zusammenhängenden Gebiet überein, muss $\log p - \log q$ konstant sein. Da das Integral beider Dichten jeweils $1$ ergeben muss, ist diese Konstante null.

<!--en:878951da-->
Der Vergleich zweier Normalverteilungen zeigt einen wichtigen Unterschied zwischen Fisher-Divergenz und KL-Divergenz. Haben beide Verteilungen die Varianz $\sigma^2$ und liegen ihre Mittelwerte im Abstand $\Delta$ voneinander, so unterscheiden sich ihre Scores um die Konstante $\Delta/\sigma^2$ an jeder Stelle $x$. Somit gilt

<!--en:66024ad3-->
wobei wir die KL-Formel für Normalverteilungen aus :numref:`sec_mdl-information_theory` verwendet haben. Für $\sigma = 1$ stimmen beide Werte überein. Skalieren wir nun die Koordinate $x$ mit dem Faktor $2$. Damit verdoppeln sich sowohl $\Delta$ als auch $\sigma$. Die KL-Divergenz bleibt gleich, die Fisher-Divergenz sinkt auf ein Viertel ihres bisherigen Werts. Die KL-Divergenz ist dimensionslos (gemessen in Nat). Die Fisher-Divergenz hat dagegen wie jede quadrierte Steigung hier die Einheit $1/x^2$. Ihr Zahlenwert hängt also von der Skalierung der Daten ab. Daher lässt sich der Verlust beim Score-Matching weder direkt über verschiedene Datensätze hinweg vergleichen noch als Kodierbudget in Nat deuten. In :exref:`ex_mdl-divergences-distances-fisher-stein`a untersuchen Sie den Fall ungleicher Varianzen.

<!--en:2ed2c400-->
### Mischungsgewichte weit auseinanderliegender Modi

<!--en:042641f7-->
Der Erwartungswert in :eqref:`eq_mdl-fisher-div-def` mittelt eine Differenz von Steigungen über die Datenverteilung $P$. Haben die Daten mehrere deutlich getrennte Modi, ist diese Kombination ungünstig. Betrachten wir zwei Mischungen aus genau denselben gaußschen Komponenten, aber mit vertauschten Mischungsgewichten:

<!--en:295c480a-->
Als Verteilungen liegen die beiden Mischungen weit auseinander. Sobald die Modi deutlich getrennt sind, nähert sich ihre KL-Divergenz derjenigen der diskreten Gewichtsvektoren $(0.7, 0.3)$ und $(0.3, 0.7)$. Diese beträgt $0.7 \log\tfrac{7}{3} + 0.3 \log\tfrac{3}{7} = 0.339$ Nat, unabhängig davon, wie groß $m$ wird.

<!--en:d273abe4-->
Ihre Scores unterscheiden sich dagegen kaum. In der Nähe jedes Modus liegt dessen Zugehörigkeitswahrscheinlichkeit nahe bei $1$, und zwar in *beiden* Mischungen. Beide Score-Felder reduzieren sich dort auf die Feder dieses Modus. Wie Sie in :exref:`ex_mdl-divergences-distances-score-blindness`b herleiten, lautet der Score von $P_w$ explizit

<!--en:a8f5547d-->
Weit links verhält sich dieser Ausdruck wie die Feder $-(x + m)$, weit rechts wie die Feder $-(x - m)$. Der Übergang liegt in der Nähe von $x = c_w/m$. Das Mischungsgewicht $w$ geht *nur* über die Lage dieses Übergangs ein. Bei $m = 5$ liegt er an der Stelle $x = 0.085$ für $w = 0.7$ und an der Stelle $x = -0.085$ für $w = 0.3$. Das ist mitten im leeren Tal zwischen den Modi, wo die Dichte $e^{-m^2/2}/\sqrt{2\pi} \approx 1.5 \times 10^{-6}$ beträgt. Genau dort, wo die beiden Scores voneinander abweichen, liegen fast nie Datenpunkte. Berechnen wir beide Divergenzen durch numerische Quadratur für wachsendes $m$. Wir erwarten, dass die KL-Divergenz nahe bei $0.339$ Nat bleibt, während die Fisher-Divergenz ungefähr wie $e^{-m^2/2}$ abfällt.

<!--en:affe2025-->
Wie erwartet pendelt sich die KL-Divergenz bei $0.339$ Nat ein. Die Fisher-Divergenz fällt um mehr als vier Größenordnungen, von etwa $0.1$ auf rund $4 \times 10^{-6}$. Bei $m = 5$ liegt sie in der Größenordnung von $e^{-m^2/2} = e^{-12.5} \approx 4 \times 10^{-6}$, wie das fast leere Tal erwarten ließ. Ein Verlust, der nur auf Scores beruht, erfasst nicht, wie sich die Wahrscheinlichkeitsmasse auf weit auseinanderliegende Modi verteilt. Deshalb haben wir zuvor ein *zusammenhängendes* Gebiet vorausgesetzt. Auf zwei getrennten Gebieten kann $\log p - \log q$ trotz identischer Scores jeweils eine andere Konstante annehmen. Minimieren wir beim Training die Fisher-Divergenz, kann das Modell Lage und Form jedes Modus richtig erfassen und dennoch $30\%$ seiner Masse dort platzieren, wo bei den Daten $70\%$ liegen :cite:`Wenliang.Kanagawa.2020`.

<!--en:49f4f657-->
Rauschen füllt das Tal wieder auf. Addieren wir zu beiden Mischungen unabhängiges gaußsches Rauschen mit Standardabweichung $2$, steigt die Varianz jeder Komponente von $1$ auf $5$. Modi, die zuvor zehn Standardabweichungen auseinanderlagen, trennen nun nur noch $4.5$ Standardabweichungen. Dadurch steigt die Dichte im Tal von $1.5 \times 10^{-6}$ auf etwa $0.015$. Nun liegen auch dort Datenpunkte, wo die beiden Scores voneinander abweichen (in :exref:`ex_mdl-divergences-distances-score-blindness`c berechnen Sie, wie viel das ausmacht). Das ist ein wesentlicher Grund dafür, dass Diffusionsmodelle die Daten vor dem Score-Matching auf vielen Skalen verrauschen. In :numref:`sec_diffusion-denoising` messen wir genau diesen blinden Fleck an trainierten neuronalen Netzen.

<!--en:bff47777-->
## Die Stein-Identität

<!--en:368973b1-->
Bei der Fisher-Divergenz stoßen wir auf ein zweites, noch unmittelbareres Hindernis. Sie hängt von $s_P$ ab, dem Score der Datenverteilung. Wir beobachten aber nur Stichproben, und eine endliche Punktmenge hat keine Steigung. Um zu sehen, wo $s_P$ eingeht, multiplizieren wir das Quadrat in einer Dimension aus:

<!--en:b9f2fff3-->
Der dritte Term ist der Erwartungswert einer vom Modell gegebenen Funktion unter der Datenverteilung. Wir können ihn daher direkt aus Stichproben schätzen. Der erste Term hängt überhaupt nicht von $Q$ ab. Beim Anpassen oder Vergleichen von Modellen spielt diese Konstante also keine Rolle. Die Schwierigkeit liegt allein im mittleren Kreuzterm, dem Produkt aus unbekanntem Daten-Score und Modell-Score. Erstaunlicherweise genügt eine einzige Identität für glatte Dichten, um diesen Kreuzterm vollständig zu eliminieren. Wir formulieren und beweisen sie für eine eindimensionale Dichte. Dann wenden wir sie zweimal an: zunächst auf die Datenverteilung, um den Kreuzterm zu beseitigen, und dann auf die Modellverteilung, um zu prüfen, ob eine Stichprobe aus $Q$ stammt.

<!--en:a09f87b1-->
**Satz (Stein-Identität).** *Sei $p > 0$ eine differenzierbare Dichte auf $\mathbb{R}$ mit Score $s = (\log p)'$. Die Funktion $g$ sei differenzierbar, und es gelte $g(x)\, p(x) \to 0$ für $x \to \pm\infty$ sowie $\mathbb{E}_p|g'| < \infty$ und $\mathbb{E}_p|g\, s| < \infty$. Dann gilt*

<!--en:6c7530d9-->
**Beweis.** Nach Definition gilt $s = p'/p$, also $p' = s\, p$. Partielle Integration (:numref:`subsec_mdl-integration-by-parts`) ergibt

<!--en:dc2395f0-->
wobei der Randterm verschwindet und beide Integrale aufgrund der Voraussetzungen an $g$ konvergieren. $\blacksquare$

<!--en:d294f4e4-->
Für die Standardnormalverteilung gilt $s(x) = -x$, und die Stein-Identität lautet $\mathbb{E}[g'(X)] = \mathbb{E}[X g(X)]$. Setzen wir zum Beispiel $g(x) = x$ ein, erhalten wir unmittelbar $\mathbb{E}[X^2] = 1$. Dieser Spezialfall für die Normalverteilung ist als Steins Lemma bekannt :cite:`Stein.1981`. Die allgemeine Form geht auf :cite:`Stein.1986` zurück. Die Stärke der Identität liegt darin, dass sie aus einer bekannten Steigung eine Gleichung für Erwartungswerte macht. Erwartungswerte können wir aus Stichproben schätzen.

<!--en:22b6e53e-->
### Score-Matching

<!--en:918639eb-->
Nehmen wir an, die Datenverteilung $P$ habe eine glatte Dichte, auch wenn wir nur Stichproben daraus sehen. Wir wenden die Stein-Identität auf $P$ an (mit $s = s_P$) und wählen den Modell-Score als Testfunktion ($g = s_Q$). Das ergibt $\mathbb{E}_P[s_Q' + s_Q\, s_P] = 0$. Im Kreuzterm von :eqref:`eq_mdl-fisher-expanded` können wir daher den Erwartungswert des Produkts der Scores durch $-\mathbb{E}_P[s_Q']$ ersetzen. Damit entfällt $s_P$ vollständig:

<!--en:3aa239e1-->
Die partielle Integration hat die Ableitung von den Daten, die selbst keine Ableitung haben, auf das Modell verlagert. Der letzte Term $\tfrac{1}{2}\mathbb{E}_P[s_P^2]$ ist für jedes Modell gleich. Die Fisher-Divergenz zu minimieren ist also gleichbedeutend damit, den ersten Erwartungswert zu minimieren. Dazu mitteln wir über die Datenpunkte Größen, die das Modell selbst liefert. Wir brauchen weder $Z$ noch $s_P$. Dieses Verfahren heißt *Score-Matching*. :citet:`Hyvarinen.2005` führte es eigens zur Anpassung unnormierter Modelle ein und demonstrierte es an Ausschnitten natürlicher Bilder.

<!--en:1617dc64-->
Das unnormierte quartische Modell aus :numref:`sec_mdl-fisher-divergence` zeigt, wie einfach das in der Praxis geht. Wir führen einen Skalenparameter $\theta > 0$ ein und setzen $\tilde{q}_\theta(x) = e^{-\theta x^4/4}$. Damit erhalten wir $s_Q(x) = -\theta x^3$ und $s_Q'(x) = -3\theta x^2$. Ein Überstrich bezeichne den Stichprobenmittelwert über die Daten. Den modellabhängigen Teil von :eqref:`eq_mdl-score-matching-line` schätzen wir dann durch eine quadratische Funktion von $\theta$:

<!--en:43290b1d-->
Zwei Stichprobenmomente genügen also, um den Parameter zu schätzen, ohne $e^{-\theta x^4/4}$ zu integrieren. Maximum-Likelihood benötigt $Z(\theta)$ und führt zu einem anderen Schätzer (:exref:`ex_mdl-scores-unnormalized-quartic`). Bei einem gaußschen Modell stimmen die beiden Schätzer hingegen überein (:exref:`ex_mdl-scores-gaussian-score-matching`).

<!--en:bbe7e83e-->
In einer Dimension erfordert $s_Q'$ nur eine zusätzliche Ableitung. In $\mathbb{R}^n$ geht dieser Term jedoch in die Spur der Jacobi-Matrix über, eine Summe von $n$ zweiten Ableitungen, eine je Koordinate. Für den Score genügte ein Rückwärtsdurchlauf. Nun sind $n$ Rückwärtsdurchläufe nötig, was für Bilder viel zu teuer ist. In :numref:`sec_mdl-score-matching` erweitern wir das Score-Matching auf $\mathbb{R}^n$. Wir zeigen, wie sich diese zweiten Ableitungen durch Verrauschen der Daten vermeiden lassen, und leiten daraus den Trainingsverlust von Diffusionsmodellen ab.

<!--en:614d41d7-->
### Ein Test für Stichproben

<!--en:ce40aaaa-->
Die Stein-Identität lässt sich auch in umgekehrter Richtung nutzen, um zu prüfen, ob eine gegebene Stichprobe aus einem Modell $Q$ stammt. Für das Modell selbst besagt die Identität: Für jede zulässige Testfunktion $g$ hat die Größe $g' + g\, s_Q$ unter $Q$ den Erwartungswert null. Der lineare Operator, der $g$ auf $g' + g\, s_Q$ abbildet, heißt *Stein-Operator* von $Q$. Liegt der Stichprobenmittelwert von $g' + g\, s_Q$ für eine Testfunktion $g$ weit von null entfernt, spricht das gegen die Annahme, dass die Stichprobe aus $Q$ stammt. Für diese Prüfung brauchen wir nur den Modell-Score an den Stichprobenpunkten, weder die Normierungskonstante $Z$ noch Samples aus dem Modell.

<!--en:081a9594-->
Was kann eine einzelne Testfunktion erkennen? Wählen wir als Modell $Q = \mathcal{N}(0, 1)$ mit Score $-x$. Der Stein-Operator bildet dann $g$ auf $g'(x) - x\, g(x)$ ab. Mit $g(x) = 1$ erhalten wir $-x$ und prüfen damit den Stichprobenmittelwert. Mit $g(x) = x$ erhalten wir $1 - x^2$ und prüfen das zweite Moment gegen $1$. Als dritte Testfunktion ohne einfache Interpretation über Momente wählen wir $g(x) = \sin x$. Im folgenden Code ziehen wir jeweils $200{,}000$ Punkte aus drei Verteilungen: dem wahren Modell $\mathcal{N}(0, 1)$, einer verschobenen Verteilung $\mathcal{N}(0.5, 1)$ und einer Verteilung mit größerer Streuung $\mathcal{N}(0, 4)$. Wir geben für jeden Test den Mittelwert und seinen Standardfehler aus. Allein aus den ersten beiden Momenten erwarten wir $-0.5$ und $1 - 1.25 = -0.25$ für die verschobene Stichprobe sowie $0$ und $1 - 4 = -3$ für die stärker streuende Stichprobe.

<!--en:a620ead4-->
Für die Stichprobe aus dem Modell selbst liegen alle Mittelwerte höchstens ein paar Standardfehler von null entfernt. Alle drei Testfunktionen erkennen die Verschiebung deutlich. Sie liefern etwa $-0.50$, $-0.25$ und $-0.15$. Bei der stärker streuenden Stichprobe verhält es sich anders. Da ihr Erwartungswert weiterhin null ist, liefert $g = 1$ einen Wert nahe null, mit einer Abweichung von etwa einem Hundertstel oder weniger. Dass die Varianz um einen Faktor vier abweicht, entgeht diesem Test vollständig. Nur $g(x) = x$ (nahe $-3$) und $g(x) = \sin x$ (nahe $-0.4$) zeigen die Abweichung an. Jede einzelne Testfunktion prüft nur einen Aspekt der Stichprobe. In diesem einen Aspekt kann auch eine falsche Verteilung leicht mit dem Modell übereinstimmen.

<!--en:ac982d5e-->
Warum erkennt $g = 1$ die Verschiebung, aber nicht die abweichende Varianz? Jede glatte Datenverteilung $P$ erfüllt ihre eigene Stein-Identität $\mathbb{E}_P[g'] = -\mathbb{E}_P[g\, s_P]$. Setzen wir diese in den Stein-Erwartungswert des Modells ein, erhalten wir

<!--en:191998b9-->
Der Testmittelwert ist also die mit $g$ gewichtete und gemittelte Differenz der Scores. Für die verschobene Stichprobe ist die Scoredifferenz konstant gleich $-x + (x - 0.5) = -0.5$. Sie wird von $g = 1$ vollständig erfasst. Bei der stärker streuenden Stichprobe beträgt die Scoredifferenz $-x + x/4 = -\tfrac{3}{4}x$. Das ist eine ungerade Funktion, deren Integral bezüglich jeder symmetrischen Dichte bei Gewichtung mit $g = 1$ null ist. Keine einzelne Testfunktion kann mit jeder möglichen Scoredifferenz korrelieren.

<!--en:1e159305-->
## Die Kernel-Stein-Diskrepanz

<!--en:c89a8cd4-->
Eine Testfunktion prüft nur eine Eigenschaft. Deshalb gehen wir so vor wie in :numref:`sec_mdl-ipm-mmd` bei der Maximum Mean Discrepancy (MMD). Wir prüfen eine ganze Familie von Funktionen auf einmal, nämlich alle Funktionen in der Einheitskugel eines Merkmalsraums mit reproduzierendem Kernel, und suchen die größte Verletzung der Identität. Erstaunlicherweise hat auch dieses Supremum über unendlich viele Testfunktionen eine geschlossene Form. Dafür müssen wir den Kernel nur an Paaren von Datenpunkten auswerten.

<!--en:a463be8a-->
Mit $\mathcal{A}_Q$ bezeichnen wir den Stein-Operator von $Q$, definiert durch $(\mathcal{A}_Q g)(x) = g'(x) + s_Q(x)\, g(x)$. Nach der Stein-Identität gilt $\mathbb{E}_Q[\mathcal{A}_Q g] = 0$. Wie stark $P$ die Identität des Modells verletzt, messen wir daher mit $\sup_g \mathbb{E}_P[\mathcal{A}_Q g]$.

<!--en:6511f9b5-->
Betrachten wir Testfunktionen der Form $g(x) = \langle \mathbf{w}, \boldsymbol{\varphi}(x) \rangle$ mit Merkmalsabbildung $\boldsymbol{\varphi}$ und $\|\mathbf{w}\| \leq 1$. Aufgrund seiner Linearität wirkt der Stein-Operator direkt auf die Merkmale: Es gilt $(\mathcal{A}_Q g)(x) = \langle \mathbf{w}, \boldsymbol{\psi}(x) \rangle$ mit $\boldsymbol{\psi} = \boldsymbol{\varphi}' + s_Q\, \boldsymbol{\varphi}$. Nach der Cauchy-Schwarz-Ungleichung ist das Supremum von $\langle \mathbf{w}, \mathbb{E}_P[\boldsymbol{\psi}] \rangle$ über die Einheitskugel gleich $\big\| \mathbb{E}_P[\boldsymbol{\psi}] \big\|$. Quadrieren wir diese Norm, erhalten wir $\mathbb{E}\langle \boldsymbol{\psi}(x), \boldsymbol{\psi}(x') \rangle$ für zwei unabhängige Ziehungen $x, x' \sim P$. Das Skalarprodukt lässt sich in vier Terme zerlegen. Jeder davon lässt sich durch den Kernel $k(x, x') = \langle \boldsymbol{\varphi}(x), \boldsymbol{\varphi}(x') \rangle$ und dessen partielle Ableitungen ausdrücken:

<!--en:ce6796b7-->
Fassen wir diese vier Terme mit den zugehörigen Score-Gewichten zusammen, erhalten wir den **Stein-Kernel** von $Q$:

<!--en:331c1efb-->
Die quadrierte **Kernel-Stein-Diskrepanz** (KSD) ist dann der Erwartungswert des Stein-Kernels für zwei unabhängige Ziehungen aus der Datenverteilung:

<!--en:5ff25df0-->
Da wir für diesen Erwartungswert nur unter $P$ mitteln, liefert eine Stichprobe $x_1, \ldots, x_N$ aus $P$ den erwartungstreuen Schätzer

<!--en:188f3662-->
bei dem wir wie bei der MMD über die Paare außerhalb der Diagonale mitteln. In der Statistik heißt ein solcher Mittelwert über Tupel mit verschiedenen Indizes U-Statistik. Daher kommt der Funktionsname `ksd2_ustat` im folgenden Code.

<!--en:c96d0938-->
Um zu verstehen, was die KSD misst, wenden wir :eqref:`eq_mdl-stein-score-diff` auf jede Koordinate von $\boldsymbol{\varphi}$ an. Das ergibt $\mathbb{E}_P[\boldsymbol{\psi}] = \mathbb{E}_P[(s_Q - s_P)\, \boldsymbol{\varphi}]$. Quadrieren wir die Norm, so erhalten wir

<!--en:3d84ef59-->
Die KSD ist also mit der Fisher-Divergenz $D_{\textrm{F}}(P\|Q)$ verwandt, glättet die Scoredifferenzen aber mit einem Kernel. Statt die Scoredifferenz an jedem Punkt einzeln zu quadrieren, multipliziert sie die Scoredifferenzen je zweier Datenpunkte und gewichtet das Produkt mit $k(x, x')$. Leider hat die KSD damit denselben blinden Fleck bei den Modusgewichten, und dieser ist sogar noch ausgeprägter. Bei den Mischungen mit weit auseinanderliegenden Modi aus :numref:`sec_mdl-fisher-divergence` müssen nun *beide* Punkte eines Paars im leeren Tal liegen, wo die Scores voneinander abweichen. Daher fällt $\mathrm{KSD}^2$ etwa wie $e^{-m^2}$ ab, also mit dem Quadrat der Abklingrate der Fisher-Divergenz (:exref:`ex_mdl-divergences-distances-score-blindness`e).

<!--en:34dee076-->
Kennen wir beide Scores, können wir mit :eqref:`eq_mdl-ksd-score-diff` den exakten Wert für die Grundgesamtheit vorhersagen. Die Daten seien $P = \mathcal{N}(0, 1)$. Wir testen drei mögliche Modelle. Beim wahren Modell $\mathcal{N}(0, 1)$ verschwindet die Scoredifferenz, und es gilt $\mathrm{KSD}^2 = 0$. Beim verschobenen Modell $\mathcal{N}(1, 1)$ beträgt die Scoredifferenz überall $-(x - 1) - (-x) = 1$. Somit gilt $\mathrm{KSD}^2 = \mathbb{E}[k(X, X')]$. Für den RBF-Kernel $k(x, x') = e^{-(x - x')^2/(2\ell^2)}$ mit Längenskala $\ell$ ergibt sich dasselbe Gauß-Integral wie in :numref:`sec_mdl-ipm-mmd`. Wegen $X - X' \sim \mathcal{N}(0, 2)$ beträgt der Erwartungswert $(1 + 2/\ell^2)^{-1/2}$. Für $\ell = 1$ ergibt das $\mathrm{KSD}^2 = 1/\sqrt{3} \approx 0.577$.

<!--en:f2600f95-->
Unser dritter Kandidat ist das unnormierte quartische Modell $\tilde{q}(x) = e^{-x^4/4}$ aus :numref:`sec_mdl-fisher-divergence`. Seine Normierungskonstante benötigen wir an keiner Stelle. Sein Score beträgt $-x^3$, die Scoredifferenz also $x - x^3$. Für das resultierende Doppelintegral in :eqref:`eq_mdl-ksd-score-diff` gibt es keinen elementaren geschlossenen Ausdruck. Deshalb wertet die nachstehende Funktion `ksd2_population` es durch numerische Quadratur auf einem Gitter aus. Das ist hier nur möglich, weil wir bei diesem synthetischen Vergleich die wahre Datendichte kennen.

<!--en:b5a7cbf1-->
Für den eindimensionalen RBF-Kernel lauten die Ableitungen in :eqref:`eq_mdl-stein-kernel` wie folgt: $\partial_{x'} k = \frac{x - x'}{\ell^2}\, k$, $\partial_x k = -\frac{x - x'}{\ell^2}\, k$ und $\partial_x \partial_{x'} k = \big(\frac{1}{\ell^2} - \frac{(x - x')^2}{\ell^4}\big)\, k$. Im folgenden Code bezeichnet `dk_dy` die Ableitung nach dem zweiten Argument, und `kappa` enthält die Matrix der Werte $k_Q(x_i, x_j)$. Wir prüfen eine einzige Stichprobe von $2{,}000$ Punkten aus $\mathcal{N}(0, 1)$ gegen alle drei Modelle. Von jedem Modell übergeben wir `ksd2_ustat` nur die Score-Funktion.

<!--en:a1b6bdd1-->
Beim wahren Modell weicht der Schätzwert nur um wenige Tausendstel von null ab. Er kann sogar leicht negativ sein, wie es bei einem erwartungstreuen Schätzer für null möglich ist. Beim verschobenen Modell liegt er nahe an unserer Vorhersage $1/\sqrt{3} \approx 0.58$, typischerweise mit einer Abweichung von höchstens etwa $0.1$. Beim quartischen Modell beträgt er mehrere Zehntel und liegt damit deutlich von null entfernt. Der Wert für die Grundgesamtheit beträgt hier $0.32$. Eine einzelne Stichprobe kann ihn leicht um $0.1$ oder mehr verfehlen. Der kubische Score $-x^3$ wächst in den Randbereichen rasch im Betrag, sodass wenige extreme Punkte den Mittelwert bestimmen können.

<!--en:2c6f510f-->
Bei diesen drei Tests hat sich nur die Score-Funktion geändert. Die Datenstichprobe blieb gleich, aus keinem der Modelle haben wir Samples gezogen, und die Normierungskonstante $Z \approx 2.56$ des quartischen Modells tauchte im Code nie auf.

<!--en:5ecc251f-->
Beim RBF-Kernel ist ein Wert von null für die Grundgesamtheit aussagekräftig. Für glatte Datendichten und unter milden Regularitätsbedingungen an die Scores gilt $\mathrm{KSD}^2(P, Q) = 0$ genau dann, wenn $P = Q$ gilt :cite:`Liu.Lee.Jordan.2016,Chwialkowski.Strathmann.Gretton.2016`. Unser Beispiel mit den beiden Mischungen mahnt dennoch zur Vorsicht. Selbst wenn die Verteilungen weit auseinanderliegen, kann ein theoretisch von null verschiedener Wert verschwindend klein sein.

<!--en:b6703a37-->
Bei der Erweiterung der KSD auf $\mathbb{R}^n$ ändert sich nur die Schreibweise. Die Testfunktion $g$ wird zum Vektorfeld und der Stein-Operator zu $s_Q^\top g + \nabla \cdot g$. Die Produkte von Skalaren in :eqref:`eq_mdl-stein-kernel` werden zu Skalarprodukten von Vektoren und der letzte Term zu $\sum_i \partial_{x_i} \partial_{x'_i} k$. In der Praxis dient die KSD als Anpassungstest für unnormierte Modelle und zur Diagnose von MCMC-Samplern. Wenden wir den Stein-Operator auf den Kernel selbst an, erhalten wir außerdem eine deterministische Aktualisierungsrichtung, die Partikel in Richtung der Verteilung $Q$ bewegt. Dieser Algorithmus heißt Stein Variational Gradient Descent :cite:`Liu.Wang.2016`. Keines der Trainingsziele in diesem Buch verwendet die KSD direkt.

<!--en:5598e5d6-->
## Trainingsziele und ihre Divergenzen

<!--en:0e658cf2-->
In den beiden vorigen Abschnitten haben wir drei große Familien von Verteilungsvergleichen entwickelt. In diesem Abschnitt kamen zwei Score-basierte Vergleiche hinzu. :numref:`fig_mdl-divergence-taxonomy` zeigt die drei Familien und die Divergenzen, die mehreren Familien angehören. Die Fisher-Divergenz fehlt, weil sie keiner der drei Familien angehört. Auch die KSD fehlt. Zwar hat sie als Supremum über Testfunktionen die Form einer integralen Wahrscheinlichkeitsmetrik, doch ihre Klasse von Testfunktionen hängt vom Modell-Score ab und ändert sich daher mit $Q$.

<!--en:4a9da3c7-->
![Familien von Verteilungsvergleichen anhand von Dichteverhältnissen, Testfunktionen und Transport sowie ihre gemeinsamen Mitglieder. f-Divergenzen mitteln eine konvexe Funktion des Dichteverhältnisses. Integrale Wahrscheinlichkeitsmetriken maximieren die Differenz der Erwartungswerte über eine Klasse von Testfunktionen. Abstände des optimalen Transports messen die minimalen Kosten der Verlagerung von Masse. Die Totalvariation gehört zu den ersten beiden Familien, der Wasserstein-1-Abstand zu den letzten beiden. (Die Totalvariation entspricht auch Transportkosten von 1 für jede Bewegung über einen von null verschiedenen Abstand. Der Übersicht halber fehlt diese Zuordnung im Diagramm.) Eingerahmte Namen bezeichnen Metriken. JS und die quadrierte Hellinger-Divergenz $d_{\mathrm{H}}^2$ sind Quadrate von Metriken.](../img/mdl-it-divergence-taxonomy.svg)

<!--en:e829a530-->
:numref:`tab_mdl-divergence-objectives` ordnet die Trainingsziele, die auf diesen Vergleichen beruhen. Für jedes Ziel gibt die Tabelle an, was es vom Modell und von den Daten benötigt und welche Divergenz es im Idealfall minimiert. Dieser Idealfall setzt exakte Erwartungswerte voraus und bei adversariellen Zielen einen optimalen Kritiker. Das ist das Hilfsnetz, das in :numref:`sec_mdl-f-gan-dual` lernt, echte von generierten Samples zu unterscheiden. Die vierte Spalte greift den Testfall vom Beginn von :numref:`sec_mdl-divergences-distances` auf. Die Daten liegen konzentriert in einem einzigen Punkt ($P = \delta_0$), die Modell-Samples in einem anderen ($Q = \delta_d$) im Abstand $d$. Bleibt ein Vergleich bei variablem $d$ konstant, liefert er dem Modell keinen Gradienten, der zu den Daten weist.

<!--en:bd444773-->
:Trainingsziele nach ihren Voraussetzungen und die Divergenzen, die sie bei exakten Mittelwerten und optimalem Kritiker minimieren. Die Spalte für Punktmassen gibt den Wert für $\delta_0$ gegenüber $\delta_d$ bei $d \neq 0$ an. Dabei bezeichnen $f(0)$ und $f'(\infty)$ die Grenzwerte von $f(u)$ bei $0$ beziehungsweise von $f(u)/u$ im Unendlichen. Für die MMD verwenden wir den RBF-Kernel mit Längenskala $\ell$.

<!--en:c72b02d4-->
| Voraussetzung | Trainingsziel | Divergenz | Zwei Punktmassen im Abstand $d$ | Behandelt in |

<!--en:3ea906a1-->
| Modelldichte an den Datenpunkten | Maximum-Likelihood (autoregressive Modelle, Flows) | Vorwärts-KL | $\infty$ | :numref:`sec_mdl-fwd-vs-rev-kl` |

<!--en:fd75d82f-->
| Samples aus beiden Verteilungen, mit Kritiker | ursprüngliches GAN | Jensen-Shannon-Divergenz | $\log 2$, konstant | :numref:`sec_mdl-f-gan-dual` |

<!--en:4dfeef84-->
| Samples aus beiden Verteilungen, mit Kritiker | f-GAN | gewählte Divergenz $D_f$ | $f(0) + f'(\infty)$, konstant | :numref:`sec_mdl-f-gan-dual` |

<!--en:80bec6d6-->
| Samples aus beiden Verteilungen, bei festem Kernel kein Kritiker | Momentenabgleich, Zweistichprobentests | MMD | $\textrm{MMD}^2 = 2 - 2e^{-d^2/(2\ell^2)}$, flach jenseits einiger Vielfacher von $\ell$ | :numref:`sec_mdl-ipm-mmd` |

<!--en:0133a7ce-->
| Samples aus beiden Verteilungen, 1-Lipschitz-stetiger Kritiker | WGAN | $W_1$ | $d$ | :numref:`sec_mdl-optimal-transport` |

<!--en:3fa382fe-->
| Modell-Samples, Modelldichte an diesen Punkten, unnormierte Zieldichte | variationelle Inferenz, A-posteriori-Verteilung beim VAE | Rückwärts-KL | $\infty$ | :numref:`sec_mdl-fwd-vs-rev-kl` |

<!--en:2f9d9a1f-->
| Modell-Score, Datenstichprobe | Score-Matching, Diffusionsmodelle | Fisher-Divergenz | undefiniert (eine Punktmasse hat keinen Score) | :numref:`sec_mdl-fisher-divergence` |

<!--en:dec8c23b-->
| Modell-Score, Datenstichprobe | Anpassungstests, Güte von Samplern | KSD | undefiniert (eine Punktmasse hat keinen Score) | :numref:`sec_mdl-ksd` |

<!--en:d8ebabde-->
Von oben nach unten gelesen hilft die Tabelle bei der Auswahl, denn ihre Zeilen sind danach geordnet, was das Modell liefern kann. Können Sie die normierte Modelldichte auswerten und ist sie überall dort positiv, wo Daten auftreten können, trainieren Sie mit Maximum-Likelihood. Für die Vorwärts-KL genügen dann die Daten. Sie liefert eine erwartungstreue Stichprobenschätzung des Trainingsziels und bestraft jeden unberücksichtigten Datenmodus. Kann ein Modell nur Samples erzeugen, gibt es drei Möglichkeiten. Für solche Modelle ist bei komplexen Daten wie Bildern ein lernbarer Kritiker wie beim GAN (:numref:`sec_basic_gan`) die übliche Wahl. Hier wissen wir vorab nicht, welche Merkmale echte von generierten Samples unterscheiden. Heute handelt es sich meist um Einschrittgeneratoren, die aus Diffusionsmodellen destilliert wurden, und um die Decoder latenter Autoencoder. Ein gelernter Kritiker hat Nachteile: Er unterschätzt die Divergenz im Mittel, und die geschätzte Divergenz flacht ab, wenn das Modell weit von den Daten entfernt ist (:numref:`sec_mdl-f-gan-dual`). Beim optimalen 1-Lipschitz-stetigen Kritiker gibt es keinen solchen flachen Bereich, denn $W_1$ hat bei jedem Abstand dieselbe Steigung. Trainierte Kritiker müssen sich allerdings nicht wie ihre optimalen Gegenstücke verhalten. Im Experiment in :numref:`subsec_gan_objectives_separated_start` zeigt ein trainierter logistischer Kritiker auch an weit entfernten generierten Samples noch eine Steigung. Hält der übliche zweiseitige Gradientenstrafterm die Steigung eines $W_1$-Kritikers nahe bei eins, führt dieser seinen Generator bis zu den Daten und dann darüber hinaus. Mit einem festen Kernel entfällt das Training eines zweiten Netzes ganz. Der Schätzfehler liegt in jeder Dimension bei $O(N^{-1/2})$ (:numref:`sec_mdl-ipm-mmd`). Das macht ihn ideal für Zweistichprobentests und die Bewertung von Samples. Meist wendet man ihn auf gelernte Merkmale an, denn mit wachsender Dimension der Rohdaten verliert die MMD mit festem Kernel an Aussagekraft. Jenseits einiger weniger Längenskalen verschwindet ihr Gradient.

<!--en:e753139e-->
Die letzten drei Zeilen behandeln Fälle mit unbekannter Normierungskonstante. Ist die *Zieldichte* unnormiert und liefert das *Modell* sowohl Samples als auch eine berechenbare Dichte wie bei der variationellen Inferenz, eignet sich die Rückwärts-KL. Dabei kann das Modell allerdings auf einen einzigen Modus kollabieren. Kennen wir vom Modell selbst nur den Score, liegt die Fisher-Divergenz nahe, die wir durch Score-Matching minimieren. Damit die Gewichte entfernter Modi sichtbar bleiben, müssen wir wie bei Diffusionsmodellen Rauschen auf mehreren Skalen hinzufügen (:numref:`sec_diffusion-denoising`).

<!--en:1b972753-->
Die Tabelle zeigt, *welchen* Vergleich das jeweilige Trainingsziel anstrebt. Wie zuverlässig wir ihn aus endlich vielen Daten schätzen können, bleibt offen. Alle stichprobenbasierten Schätzer der letzten drei Abschnitte sind mit einem Schätzfehler behaftet: die Schranke des Kritikers, $\textrm{MMD}^2$ und $\mathrm{KSD}^2$. Die Schranke des Kritikers ist zudem nach unten verzerrt. In den nächsten beiden Abschnitten gehen wir vom Vergleich zweier Verteilungen zur Abhängigkeit zweier Zufallsvariablen über. In :numref:`sec_mdl-mutual-information` definieren wir mithilfe der KL-Divergenz, wie viel eine Variable über eine andere verrät. In :numref:`sec_mdl-mi-estimation` fragen wir, wie viel Abhängigkeit sich mit einer Stichprobe von $n$ Paaren tatsächlich nachweisen lässt.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:87f9b38d-->
Selbst wenn sich die Normierungskonstante eines Modells praktisch nicht berechnen lässt, können wir das Modell anhand seines Scores mit Daten vergleichen. Der Score ist der Gradient der Log-Dichte nach der Eingabe. Beim Ableiten fällt die Konstante weg. Die Fisher-Divergenz mittelt die quadrierte Differenz zwischen Daten-Score und Modell-Score. Mit der Stein-Identität genügt eine partielle Integration, um den unbekannten Daten-Score zu eliminieren. So erhalten wir das Score-Matching. Da Scores rein lokal sind, erfassen sie allerdings nicht, wie sich die Wahrscheinlichkeitsmasse auf weit voneinander entfernte Modi verteilt. Bei zwei Mischungen, deren Modi zehn Standardabweichungen auseinanderliegen, fällt die Fisher-Divergenz auf $4 \times 10^{-6}$, während die KL-Divergenz bei $0.339$ Nat bleibt. Wenden wir die Stein-Identität auf das Modell an, erhalten wir die Kernel-Stein-Diskrepanz. Sie prüft Stichproben allein anhand des Modell-Scores. Welches Trainingsziel sich eignet, hängt letztlich davon ab, was das Modell liefern kann: eine normierte Dichte, Samples oder einen Score.

<!--en:2294243a-->
## Übungen

<!--en:78cbd4ce-->
1. **Fisher-Divergenz und Stein-Momente.** Der Score von $\mathcal{N}(\mu, \sigma^2)$ lautet $-(x - \mu)/\sigma^2$.
    1. Berechnen Sie die Fisher-Divergenz :eqref:`eq_mdl-fisher-div-def` zwischen $\mathcal{N}(\mu_1, \sigma_1^2)$ und $\mathcal{N}(\mu_2, \sigma_2^2)$ in geschlossener Form. Prüfen Sie, dass sie sich bei gleichen Varianzen auf $(\mu_1 - \mu_2)^2 / (2\sigma^4)$ reduziert. Ist die Fisher-Divergenz symmetrisch?
    1. Wenden Sie die Stein-Identität :eqref:`eq_mdl-stein-identity` für $\mathcal{N}(0, 1)$ auf $g(x) = x^k$ an. Leiten Sie daraus die Rekursion $\mathbb{E}[X^{k+1}] = k\, \mathbb{E}[X^{k-1}]$ ab und berechnen Sie damit $\mathbb{E}[X^4]$ und $\mathbb{E}[X^6]$. Was ergibt $g(x) = 1$ für jede Dichte, die die Voraussetzungen der Identität erfüllt?
    1. Eine Verteilung $P$ auf $\mathbb{R}$ habe endliche Momente jeder Ordnung und erfülle $\mathbb{E}_P[g'(X) - X g(X)] = 0$ für jedes Polynom $g$. Die Normalverteilung ist durch ihre Momente eindeutig bestimmt. Begründen Sie damit, warum $P = \mathcal{N}(0, 1)$ gilt. Sei nun $P(X = 1) = P(X = -1) = \tfrac{1}{2}$. Zeigen Sie, dass diese Verteilung die Identität für jedes Polynom $g$ vom Grad höchstens $2$ erfüllt, für $g(x) = x^3$ jedoch verletzt. Was bedeutet das für einen Anpassungstest, der die Stein-Identität nur an Polynomen bis zu einem festen Grad prüft?
   :exlabel:`ex_mdl-divergences-distances-fisher-stein`

<!--en:e45da74f-->
1. [code] **Der blinde Fleck des Scores bei Mischungsgewichten.** ● Sei $P_w = w\, \mathcal{N}(-m, 1) + (1 - w)\, \mathcal{N}(m, 1)$.
    1. Im Text berechnen wir $D_{\textrm{F}}(P_{0.7}\|P_{0.3})$ und $D_{\textrm{KL}}(P_{0.7}\|P_{0.3})$ für $m \in \{1, 3, 5\}$. Erweitern Sie die Berechnung auf $m \in \{1, \ldots, 6\}$. Zeigen Sie, dass die KL-Divergenz gegen $0.4 \log\tfrac{7}{3} \approx 0.339$ strebt. Bestimmen Sie die Abklingrate der Fisher-Divergenz, indem Sie eine Gerade an $\log D_{\textrm{F}}$ als Funktion von $m^2$ anpassen.
    1. Leiten Sie die Score-Formel $s_w(x) = -x + m \tanh(m x - c_w)$ aus den Zugehörigkeitswahrscheinlichkeiten der Komponenten her. Zeigen Sie, dass sich $s_{0.7}$ und $s_{0.3}$ an der Stelle $x = 0$ am stärksten unterscheiden, und zwar um $0.8\,m$. Substituieren Sie $u = m x$ im Integral für die Fisher-Divergenz. Zeigen Sie damit, dass sie wie $C\, m\, e^{-m^2/2}$ für große $m$ skaliert, wobei $C$ nur von den Gewichten abhängt. Vergleichen Sie dies mit der in (a) geschätzten Rate. Was folgt daraus für die Mischungsgewichte, die das Score-Matching lernt :cite:`Wenliang.Kanagawa.2020`?
    1. Addieren Sie unabhängiges Rauschen mit Verteilung $\mathcal{N}(0, \sigma^2)$ zu Samples beider Mischungen. Damit geht jede Komponente in $\mathcal{N}(\pm m, 1 + \sigma^2)$ über. Wiederholen Sie (a) bei $m = 5$ für $\sigma \in \{0, 0.5, 1, 2\}$. Wie wirkt sich das Rauschen auf die beiden Divergenzen aus? Starkes Rauschen macht die Mischungsgewichte sichtbar, verwischt aber die Form der einzelnen Modi. Bei schwachem Rauschen verhält es sich umgekehrt. Warum kann ein Modell beides rekonstruieren, wenn es die Scores bei mehreren Rauschstärken abgleicht :cite:`song2019generative`?
    1. Mit $P_t$ und $Q_t$ bezeichnen wir zwei Verteilungen, nachdem wir zu beiden unabhängiges Rauschen mit Verteilung $\mathcal{N}(0, t)$ addiert haben. Betrachten Sie zwei Normalverteilungen mit gleicher Varianz $\sigma^2$, die durch das Rauschen auf $\sigma^2 + t$ steigt. Prüfen Sie anhand der geschlossenen Formeln im Text, dass $\frac{d}{dt} D_{\textrm{KL}}(P_t\|Q_t) = -D_{\textrm{F}}(P_t\|Q_t)$ gilt. Diese Identität gilt für jedes Paar glatter Dichten. Bestätigen Sie sie mit einer finiten Differenz für die beiden Mischungen bei $m = 5$ und $t \in \{1, 4, 9\}$. Folgern Sie daraus $D_{\textrm{KL}}(P\|Q) = \int_0^\infty D_{\textrm{F}}(P_t\|Q_t)\, dt$. Bei welcher Rauschstärke ist die Fisher-Divergenz am größten? Was sagt das Integral über den Abgleich von Scores bei einer einzelnen Rauschstärke im Vergleich zum Abgleich über alle Rauschstärken aus :cite:`Song.Durkan.Murray.ea.2021`?
    1. Die Kernel-Stein-Diskrepanz hat denselben blinden Fleck. Werten Sie :eqref:`eq_mdl-ksd-score-diff` für beide Mischungen durch numerische Quadratur auf einem Gitter aus. Verwenden Sie den RBF-Kernel mit $\ell = 1$ und $m \in \{1, 3, 5\}$. Vergleichen Sie die Ergebnisse mit den Fisher-Divergenzen im Text (bei $m = 5$ sollten Sie etwa $10^{-11}$ erhalten). Passen Sie `ksd2_population` aus dem Text an und ersetzen Sie die Datenverteilung $\mathcal{N}(0, 1)$ durch $P_{0.7}$. Wo müssen beide Punkte eines Paars liegen, um einen Beitrag zu leisten? Wie breit ist dieses Gebiet im Verhältnis zu $\ell$? Folgern Sie, dass $\mathrm{KSD}^2$ wie $e^{-m^2}$ abfällt. Würde eine kleinere Längenskala helfen?
   :exlabel:`ex_mdl-divergences-distances-score-blindness`

<!--en:55369b69-->
1. **Score-Matching für ein gaußsches Modell.** Die Datenverteilung $P$ habe den Mittelwert $a$ und die Varianz $v$. Betrachten Sie das Modell $Q = \mathcal{N}(\mu, \sigma^2)$ und schreiben Sie $J = \mathbb{E}_P\big[\tfrac{1}{2} s_Q^2 + s_Q'\big]$ für den modellabhängigen Teil von :eqref:`eq_mdl-score-matching-line`.
    1. Zeigen Sie, dass $J = \big(v + (a - \mu)^2\big)/(2\sigma^4) - 1/\sigma^2$ gilt. Prüfen Sie, dass das Minimum bei $\mu = a$ und $\sigma^2 = v$ liegt, unabhängig von den übrigen Eigenschaften von $P$. Vergleichen Sie das Ergebnis mit der Normalverteilung, die die Vorwärts-KL-Divergenz minimiert (:numref:`sec_mdl-fwd-vs-rev-kl`).
    1. Im Minimum gilt $J = -1/(2v)$. Daher beträgt die Fisher-Divergenz von $P$ zu ihrer besten gaußschen Näherung $\tfrac{1}{2}\big(\mathbb{E}_P[s_P^2] - 1/v\big)$. Wenden Sie die Stein-Identität für $P$ auf $g(x) = x - a$ an und anschließend die Cauchy-Schwarz-Ungleichung. Beweisen Sie so, dass $\mathbb{E}_P[s_P^2] \geq 1/v$ gilt, mit Gleichheit genau dann, wenn $P$ eine Normalverteilung ist. Welche Dichte mit vorgegebener Varianz hat den kleinsten mittleren quadrierten Score?
    1. Parametrisieren Sie das Modell um als $\tilde{q}_{\boldsymbol{\theta}}(x) = \exp(\theta_1 x + \theta_2 x^2)$. Zeigen Sie, dass $J$ eine quadratische Funktion von $\boldsymbol{\theta}$ ist. Damit reduziert sich das Score-Matching auf die Lösung zweier linearer Gleichungen mit den Datenmomenten als Koeffizienten. Zeigen Sie, dass dies für jedes Modell der Exponentialfamilie $\tilde{q}_{\boldsymbol{\theta}}(x) = \exp\big(\boldsymbol{\theta}^\top \boldsymbol{\phi}(x)\big)$ gilt. Was müsste man stattdessen bei Maximum-Likelihood berechnen?
   :exlabel:`ex_mdl-scores-gaussian-score-matching`

<!--en:cee36df1-->
1. [code] **Ein unnormiertes quartisches Modell.** Sei $\tilde{q}_\theta(x) = e^{-\theta x^4/4}$ mit $\theta > 0$.
    1. Berechnen Sie den Score. Wenden Sie die Stein-Identität mit $g(x) = x$ und mit $g(x) = x^3$ an. Zeigen Sie damit, dass unter dem Modell $\mathbb{E}[X^4] = 1/\theta$ und $\mathbb{E}[X^6] = (3/\theta)\, \mathbb{E}[X^2]$ gelten, ohne die Normierungskonstante zu berechnen. Lässt sich mit der Stein-Identität für ein Polynom $g$ auch $\mathbb{E}[X^2]$ bestimmen?
    1. Im Text leiten wir den Score-Matching-Schätzer $\hat\theta = 3\, \overline{x^2} / \overline{x^6}$ her, wobei der Überstrich den Stichprobenmittelwert bezeichnet. Zeigen Sie, dass die Normierungskonstante gemäß $Z(\theta) = \theta^{-1/4} Z(1)$ skaliert und dass Maximum-Likelihood $\hat\theta = 1 / \overline{x^4}$ liefert. Begründen Sie mit Teil (a), warum beide Schätzer bei wachsender Stichprobe aus dem Modell gegen den wahren Wert von $\theta$ konvergieren, obwohl sie unterschiedliche Funktionen der Daten sind.
    1. Folgt $U$ einer Gammaverteilung mit Formparameter $\tfrac{1}{4}$ und Skalenparameter $1$, so hat $(4U/\theta)^{1/4}$ mit zufälligem Vorzeichen die Dichte $\tilde{q}_\theta / Z(\theta)$. Ziehen Sie $1{,}000$ Punkte bei $\theta = 1$, berechnen Sie beide Schätzwerte und wiederholen Sie dies $2{,}000$ Mal. Welcher Schätzer hat die kleinere Standardabweichung, und um welchen Faktor? Was gibt man beim Score-Matching dafür auf, dass man $Z$ umgeht?
    1. Nehmen Sie nun an, die Daten seien standardnormalverteilt (wie im KSD-Experiment). Dann ist kein quartisches Modell exakt. Bei wachsender Stichprobe konvergieren die beiden Schätzer aus (b) gegen $3\, \mathbb{E}[X^2]/\mathbb{E}[X^6] = \tfrac{1}{5}$ und $1/\mathbb{E}[X^4] = \tfrac{1}{3}$. Werten Sie `ksd2_population` mit den Scores beider angepasster Modelle aus und bestimmen Sie das $\theta$, bei dem die Funktion ihr Minimum erreicht. Drei Kriterien haben drei verschiedene quartische Modelle ausgewählt. Warum ist das kein Widerspruch, und unter welcher Bedingung würden alle drei übereinstimmen?
   :exlabel:`ex_mdl-scores-unnormalized-quartic`

<!--en:d2666521-->
1. [code] **Die Längenskala des Stein-Kernels.** Wie im KSD-Experiment sei die Datenverteilung $P = \mathcal{N}(0, 1)$. Verwenden Sie den RBF-Kernel mit Längenskala $\ell$.
    1. Zeigen Sie für das Modell $\mathcal{N}(\mu, 1)$ mithilfe von :eqref:`eq_mdl-ksd-score-diff`, dass $\mathrm{KSD}^2 = \mu^2\, (1 + 2/\ell^2)^{-1/2}$ gilt. Wie lautet der Grenzwert für $\ell \to \infty$? Welche einzelne Testfunktion $g$ liefert denselben Grenzwert, wenn Sie das Quadrat von $\mathbb{E}_P[g' + g\, s_Q]$ bilden?
    1. Zeigen Sie für das Modell $\mathcal{N}(0, 4)$ mit Score $-x/4$, dass $\mathrm{KSD}^2 = \tfrac{9}{16}\, \ell\, (\ell^2 + 2)^{-3/2}$ gilt. (Hinweis: Drücken Sie $X X'$ durch die unabhängigen Variablen $X + X'$ und $X - X'$ aus.) Zeigen Sie, dass dieser Wert bei $\ell = 1$, der Standardabweichung der Daten, maximal wird und für $\ell \to \infty$ gegen null geht. Erklären Sie diesen Grenzwert anhand Ihrer Antwort auf (a).
    1. Prüfen Sie beide Formeln mit `ksd2_population` für $\ell \in \{0.25, 1, 4, 16\}$. Berechnen Sie die entsprechenden vier Werte für das quartische Modell mit Score $-x^3$. Warum kann keine einzelne Längenskala für den Test gegen jedes falsche Modell optimal sein?
   :exlabel:`ex_mdl-scores-ksd-length-scale`

<!--en:4932cba9-->
[Diskussionen](https://d2l.discourse.group/t/36896)

<!--en:7f904fd4-->
Score-basierte Vergleiche<br>**Stein-Score-Funktion · Fisher-Divergenz · blinder Fleck bei Modusgewichten · Stein-Identität · Score-Matching · KSD**

<!--en:a6c5f840-->
::: {.slide title="Warum die Normierung energiebasierter Modelle so schwierig ist"}

<!--en:3fdccfb8-->
[Motivation: unnormierte Dichten]{.kicker}

<!--en:a701fe7c-->
Ein **energiebasiertes Modell (EBM)** definiert $q_{\boldsymbol{\theta}}(\mathbf{x}) = e^{-E_{\boldsymbol{\theta}}(\mathbf{x})} / Z(\boldsymbol{\theta})$ mit dem Normierungsintegral $Z(\boldsymbol{\theta}) = \int_{\mathbb{R}^n} e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}\,d\mathbf{x}$:

<!--en:f35d9090-->
- **Schwierigkeiten bei MLE und GANs:** MLE benötigt den praktisch unberechenbaren Gradienten $\nabla_{\boldsymbol{\theta}}\log Z(\boldsymbol{\theta})$. GANs, MMD und $W_1$ benötigen aufwendige Langevin-MCMC-Samples aus $Q_{\boldsymbol{\theta}}$.

<!--en:dbba3c92-->
- **Ausweg über den Score:** Beim Ableiten nach der Eingabe $\mathbf{x}$ fällt $\log Z(\boldsymbol{\theta})$ in einem einzigen Rückwärtsdurchlauf weg.

<!--en:23e30755-->
::: {.slide title="Die Score-Funktion eliminiert die Normierungskonstante Z"}

<!--en:794f76ee-->
[Gradienten der Log-Dichte]{.kicker}

<!--en:e6efab76-->
An jedem Punkt $\mathbf{x} \in \mathbb{R}^n$ zeigt der **(Stein-)Score** in die Richtung des steilsten Anstiegs der Log-Dichte:

<!--en:a8626570-->
- **Definition und $Z$-Invarianz:** Wegen $\nabla_{\mathbf{x}} \log Z = \mathbf{0}$ gilt:
  $$s_Q(\mathbf{x}) = \nabla_{\mathbf{x}} \log q(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})$$

<!--en:7c9c5adf-->
- **Rückstellkraft bei der Normalverteilung:** Für $\mathcal{N}(\mu, \sigma^2)$ zieht $s(x) = -\frac{x - \mu}{\sigma^2}$ den Punkt $x$ zum Mittelwert $\mu$. Die Federkonstante beträgt $1/\sigma^2$.

<!--en:7a8125e4-->
- **Übergang zwischen Komponenten:** Für $p(x) = \sum_k w_k p_k(x)$ gewichtet die Kettenregel den Score jeder Komponente mit ihrer Zugehörigkeitswahrscheinlichkeit $r_k(x) = \frac{w_k p_k(x)}{p(x)}$:
  $$s(x) = \sum_k r_k(x)\,s_k(x)$$

<!--en:bbcc9bcd-->
::: {.slide title="Numerisch stabile Scores für Mischungen im Code"}

<!--en:46a9f8d2-->
[Gewichtung mit Zugehörigkeitswahrscheinlichkeiten im logarithmischen Raum]{.kicker}

<!--en:1a7ce87f-->
Subtrahieren wir den größten logarithmierten Komponentenbeitrag $c_{\max}(x)$, vermeiden wir bei Log-Dichte und Score der Mischung einen Unterlauf in den Randbereichen:

<!--en:c996815c-->
::: {.slide title="Die Fisher-Divergenz vergleicht Score-Felder"}

<!--en:12d20ac4-->
[Mittlere quadratische Scoredifferenz]{.kicker}

<!--en:cf26f38e-->
Die **Fisher-Divergenz** ist der erwartete quadrierte euklidische Abstand zwischen dem Daten-Score $s_P(\mathbf{x})$ und dem Modell-Score $s_Q(\mathbf{x})$. Gemittelt wird über die Datenverteilung $P$:

<!--en:a7ebf69b-->
- **Identität des Ununterscheidbaren auf zusammenhängenden Trägern:** Offenbar gilt $D_{\textrm{F}}(P \| Q) \ge 0$. Sind $p$ und $q$ auf einem **zusammenhängenden** Gebiet strikt positiv und glatt, so gilt $s_P = s_Q$ $\implies$ $\nabla(\log p - \log q) = \mathbf{0}$ $\implies$ $p(\mathbf{x}) = c\,q(\mathbf{x})$. Aus $\int p = \int q = 1$ folgt zwingend $c = 1$ ($P = Q$).

<!--en:bcc4d779-->
- **Abhängigkeit von der Koordinatenskala (Einheit $1/x^2$ gegenüber dimensionslosen Nat):**
  Für zwei Normalverteilungen $\mathcal{N}(\mu_1, \sigma^2)$ und $\mathcal{N}(\mu_2, \sigma^2)$ mit Mittelwertabstand $\Delta = |\mu_1 - \mu_2|$ beträgt die Scoredifferenz überall $\Delta/\sigma^2$. Daraus folgt:
  $$D_{\textrm{F}}(P \| Q) = \frac{\Delta^2}{2\sigma^4} \qquad \text{whereas} \qquad D_{\textrm{KL}}(P \| Q) = \frac{\Delta^2}{2\sigma^2}$$
  Verdoppeln wir die Koordinatenskala ($x \mapsto 2x$), bleibt $D_{\textrm{KL}}$ invariant, während $D_{\textrm{F}}$ durch $4$ geteilt wird.

<!--en:5f8a096b-->
::: {.slide title="Der blinde Fleck von Score-Vergleichen bei Modusgewichten"}

<!--en:834a24eb-->
[Warum der Träger zusammenhängen muss]{.kicker}

<!--en:64837436-->
Vergleichen wir $P_{0.7}$ und $P_{0.3}$ für die Mischung $P_w = w\,\mathcal{N}(-m, 1) + (1 - w)\,\mathcal{N}(m, 1)$ mit dem Abstandsparameter $m = 5$:

<!--en:0fc406f6-->
- **Übereinstimmung an beiden Modi:** In der Nähe von $x \approx \pm m$ gilt $\tanh(\pm m^2 - c_w) \approx \pm 1$ sowohl für $w = 0.7$ als auch für $w = 0.3$. Daher gilt $s_{0.7}(x) \approx s_{0.3}(x) \approx -(x \mp m)$.

<!--en:b42f2d68-->
- **Der Unterschied liegt im leeren Tal:** Bei $x \approx 0$ beträgt die Dichte nur $p(0) \approx 1.5 \times 10^{-6}$. Daher fällt die Divergenz auf $D_{\textrm{F}}(P_{0.7} \| P_{0.3}) \approx 4 \times 10^{-6} \approx 0$.

<!--en:d18390ab-->
::: {.slide title="Wie schnell fällt die Fisher-Divergenz exponentiell ab?"}

<!--en:76c5b72d-->
[Numerische Quadratur]{.kicker}

<!--en:49dbf667-->
Wächst der Abstandsparameter $m$ von $1$ auf $5$, konvergiert $D_{\textrm{KL}}(P_{0.7} \| P_{0.3})$ gegen die Divergenz der diskreten Gewichte $0.7\ln\frac{7}{3} + 0.3\ln\frac{3}{7} = 0.339\text{ nats}$. Zugleich fällt $D_{\textrm{F}}(P_{0.7} \| P_{0.3})$ um vier Größenordnungen ($\sim e^{-m^2/2}$):

<!--en:22456f9d-->
::: {.slide title="Wie mehrskaliges Rauschen die Modusgewichte sichtbar macht"}

<!--en:49a58abf-->
[Warum Diffusionsmodelle die Daten verrauschen]{.kicker}

<!--en:f3525c67-->
- **Mischung ohne Rauschen ($\sigma_{\textrm{noise}} = 0$):** Die Modi bei $\pm 5$ liegen $10\sigma$ auseinander. Bei einer Dichte von $p(0) = 1.5 \times 10^{-6}$ im Tal erkennt $D_{\textrm{F}}$ die Mischungsgewichte $w$ nicht.

<!--en:18bb4b1b-->
- **Verrauschte Mischung ($\sigma_{\textrm{noise}} = 2$):** Die Faltung mit $\mathcal{N}(0, 2^2)$ erhöht die Varianz auf $\sigma_{\textrm{tot}}^2 = 5$ (Abstand $4.47\sigma$). Die Dichte im Tal steigt um den Faktor $10{,}000\times$ auf $1.5 \times 10^{-2}$.

<!--en:13b3a340-->
- **Mehrskalige Diffusion (Song und Ermon, 2019):** Starkes Rauschen überbrückt Täler und macht globale Modusgewichte lernbar. Schwaches Rauschen lässt lokale Formen erkennen.

<!--en:d6c0ce53-->
::: {.slide title="Ausmultiplizieren der Fisher-Divergenz: der unbekannte Daten-Score"}

<!--en:0e0821cd-->
[Schätzen aus einer Stichprobe]{.kicker}

<!--en:1a5cf760-->
Schon bei einem einzigen zusammenhängenden Modus stellt sich die Frage: Wie minimieren wir $D_{\textrm{F}}(P \| Q)$, wenn wir von der Datenverteilung $P$ nur eine endliche Stichprobe $\{x_i\}_{i=1}^N$ beobachten und keine Formel für den Daten-Score $s_P(x)$ haben?

<!--en:82924265-->
- **Quadrierte Scoredifferenz in einer Dimension ausmultiplizieren:**
  $$D_{\textrm{F}}(P \| Q) = \underbrace{\frac{1}{2}\mathbb{E}_{x \sim P}\big[s_P(x)^2\big]}_{\text{constant independent of } Q} - \underbrace{\mathbb{E}_{x \sim P}\big[s_P(x)\,s_Q(x)\big]}_{\text{cross term containing unknown } s_P} + \underbrace{\frac{1}{2}\mathbb{E}_{x \sim P}\big[s_Q(x)^2\big]}_{\text{direct data average of model score}}$$

<!--en:5ee889cd-->
- **Der schwierige Term:** Nur der mittlere Kreuzterm $-\mathbb{E}_P[s_P(x)s_Q(x)]$ enthält sowohl die unbekannte Steigung $s_P(x) = p'(x)/p(x)$ der Log-Dichte der Daten als auch das Modell $Q$.

<!--en:551a115c-->
- **Entscheidende Beobachtung:** Wegen $p(x) s_P(x) = p'(x)$ lässt sich der Kreuzterm als Integral $-\int p'(x) s_Q(x)\,dx$ schreiben. Damit bietet sich partielle Integration an.

<!--en:f9825b70-->
::: {.slide title="Die Stein-Identität: partielle Integration für Dichten"}

<!--en:2219e63c-->
[Unbekannte Ableitungen eliminieren]{.kicker}

<!--en:07b9091b-->
Für jede glatte Dichte $p(x) > 0$ und jede Testfunktion $g(x)$ mit $g(x)p(x) \to 0$ für $\pm\infty$ gilt:

<!--en:7d133490-->
- **Beweis durch partielle Integration in einer Zeile:** Wegen $p'(x) = p(x)s_p(x)$ gilt $\int g'(x)p(x)\,dx = [gp]_{-\infty}^\infty - \int g(x)p'(x)\,dx = -\mathbb{E}_p[g(x)s_p(x)]$.

<!--en:394d3a32-->
- **Steins Lemma für die Normalverteilung ($s_p(x) = -x$):** Es gilt $\mathbb{E}[g'(X)] = \mathbb{E}[Xg(X)]$ für $X \sim \mathcal{N}(0, 1)$.

<!--en:61707000-->
::: {.slide title="Score-Matching nach Hyvärinen mit der Stein-Identität"}

<!--en:1e1a7ef0-->
[Unnormierte Modelle trainieren]{.kicker}

<!--en:eb6f7edd-->
Setzen wir $g(x) = s_Q(x)$ in die Stein-Identität ein, können wir den unbekannten Kreuzterm $-\mathbb{E}_P[s_P s_Q]$ durch $+\mathbb{E}_P[s_Q'(x)]$ ersetzen:

<!--en:c559df1f-->
- **Weder $Z$ noch $s_P$ nötig:** Durch Minimieren von $J(Q) = D_{\textrm{F}}(P \| Q) - \tfrac12\mathbb{E}_P\|s_P\|^2$ trainieren wir unnormierte Modelle direkt an Datenstichproben.

<!--en:e80e6cab-->
- **Geometrie der beiden Terme:** $\tfrac12\|s_Q\|^2$ zieht die Steigung an den Datenpunkten gegen $\mathbf{0}$, während $\textrm{tr}(\nabla s_Q) < 0$ die Log-Dichte nach unten zu einem Gipfel krümmt.

<!--en:9b5709f3-->
::: {.slide title="Rechenbeispiel: Anpassung eines unnormierten quartischen Modells"}

<!--en:95382f59-->
[Score-Matching in geschlossener Form]{.kicker}

<!--en:c8c3ab2e-->
Wir passen die unnormierte Dichte mit quartischem Energieterm $\tilde{q}_\theta(x) = e^{-\theta x^4/4}$ ($\theta > 0$) an die Daten $x_1, \ldots, x_N$ an, ohne $Z(\theta) = \int_{-\infty}^\infty e^{-\theta x^4/4}\,dx$ zu berechnen:

<!--en:5e4bb706-->
- **Schritt 1 (Modell-Score und Ableitung berechnen):**
  $$s_Q(x) = \frac{d}{dx}\!\left(-\frac{\theta x^4}{4}\right) = -\theta x^3, \qquad s_Q'(x) = -3\theta x^2$$

<!--en:3e5a4f59-->
- **Schritt 2 (empirische Zielfunktion des Score-Matchings bilden):** Mit $\overline{f(x)} = \frac{1}{N}\sum_{i=1}^N f(x_i)$ gilt:
  $$\hat{J}(\theta) = \overline{\frac{1}{2}s_Q(x)^2 + s_Q'(x)} = \frac{1}{2}\theta^2\,\overline{x^6} - 3\theta\,\overline{x^2}$$

<!--en:abdc70bd-->
- **Schritt 3 (quadratische Zielfunktion in $\theta$ minimieren):** Aus $\hat{J}'(\theta) = \theta\,\overline{x^6} - 3\overline{x^2} = 0$ erhalten wir den Schätzer $\hat{\theta}_{\textrm{SM}} = \frac{3\overline{x^2}}{\overline{x^6}}$ in geschlossener Form aus zwei empirischen Momenten.
  *(Hinweis: In $\mathbb{R}^n$ erfordert die Berechnung von $\textrm{tr}(\nabla_{\mathbf{x}} s_Q(\mathbf{x})) = \sum_{i=1}^n \partial_{x_i} s_{Q,i}(\mathbf{x})$ insgesamt $n$ Rückwärtsdurchläufe. Das motiviert Sliced-Score-Matching und Denoising-Score-Matching in Kapitel §{num:chap_diffusion} und §{sec_mdl-score-matching-diffusion-flow}.)*

<!--en:9ff97ebe-->
::: {.slide title="Der Stein-Operator als Anpassungstest"}

<!--en:8c16c026-->
[Stichproben anhand eines Scores prüfen]{.kicker}

<!--en:5aa9da10-->
Was geschieht, wenn wir die Rollen vertauschen und die Stein-Identität auf das *Modell* $Q$ statt auf die Daten $P$ anwenden? Wir definieren den linearen **Stein-Operator** $\mathcal{A}_Q$ des Modells $Q$:

<!--en:a7e7c2ac-->
- **Anpassungstest ohne $Z$ oder Modell-Samples:** Gegeben sei eine Datenstichprobe $x_1, \ldots, x_N \sim P$. Weicht der empirische Mittelwert $\frac{1}{N}\sum_{i=1}^N (\mathcal{A}_Q g)(x_i)$ signifikant von $0$ ab, können wir $P = Q$ allein anhand von $s_Q(x_i)$ verwerfen.

<!--en:88cc5aac-->
- **Was misst $\mathbb{E}_P[\mathcal{A}_Q g]$?** Die Datenverteilung $P$ erfüllt ihre eigene Stein-Identität $\mathbb{E}_P[g'(x)] = -\mathbb{E}_P[s_P(x)g(x)]$. Einsetzen in $\mathbb{E}_P[g' + s_Q g]$ ergibt:
  $$\mathbb{E}_{x \sim P}\big[(\mathcal{A}_Q g)(x)\big] = \mathbb{E}_{x \sim P}\Big[\big(s_Q(x) - s_P(x)\big)\,g(x)\Big]$$
  Jede Testfunktion $g(x)$ prüft die Projektion der Scoredifferenz $s_Q - s_P$ auf $g$.

<!--en:19ba9dc6-->
::: {.slide title="Verschiebung und größere Streuung in Stichproben mit Code untersuchen"}

<!--en:8f8ff0e8-->
[Blinde Flecken einzelner Testfunktionen]{.kicker}

<!--en:7cfa90f2-->
Wir testen $Q = \mathcal{N}(0, 1)$ ($s_Q(x) = -x$, also $(\mathcal{A}_Q g)(x) = g'(x) - x g(x)$) anhand von je $200{,}000$ Ziehungen aus $\mathcal{N}(0, 1)$, $\mathcal{N}(0.5, 1)$ und $\mathcal{N}(0, 4)$. Das Beispiel zeigt, warum keine einzelne Testfunktion $g$ genügt:

<!--en:e13f39a5-->
- Für $P = \mathcal{N}(0, 4)$ ist die Scoredifferenz $s_Q(x) - s_P(x) = -x - (-x/4) = -\tfrac34 x$ ungerade. Mit der geraden Testfunktion $g(x) = 1$ ergibt das Integral $0$. Eine Abweichung der Varianz um den Faktor $4\times$ bleibt unentdeckt.

<!--en:ecb8b7ed-->
::: {.slide title="Kernel-Stein-Diskrepanz (KSD) und Stein-Kernel"}

<!--en:31727af7-->
[Supremum über eine RKHS-Kugel]{.kicker}

<!--en:a525b465-->
Maximieren wir $\mathbb{E}_P[(\mathcal{A}_Q g)(x)]$ über die Einheitskugel im RKHS ($\|\mathbf{w}\|_2 \le 1$), erhalten wir die **Kernel-Stein-Diskrepanz**:

<!--en:33c1e85d-->
- **Stein-Kernel $k_Q(x, x')$:** Er benötigt nur den Modell-Score $s_Q(x)$ und Ableitungen von $k(x, x')$. Für den Anpassungstest genügt also eine Datenstichprobe, ganz ohne $Z$ oder Samples aus $Q$.

<!--en:db4ef684-->
::: {.slide title="Erwartungstreue KSD-Schätzung mit einer U-Statistik im Code"}

<!--en:b8a28eb3-->
[Anpassungstest ohne Samples aus Q]{.kicker}

<!--en:445d24dd-->
Eine einzige Stichprobe mit $N = 2{,}000$ Datenpunkten aus $P = \mathcal{N}(0, 1)$ genügt: Die U-Statistik $\widehat{\mathrm{KSD}}^2 = \frac{1}{N(N-1)}\sum_{i \neq j} k_Q(x_i, x_j)$ über die Paare außerhalb der Diagonale unterscheidet das wahre Modell klar vom verschobenen und vom quartischen Modell. Dazu braucht sie nur deren Scores $s_Q$:

<!--en:ecd99da2-->
::: {.slide title="Familien von Verteilungsvergleichen"}

<!--en:76d0f4d3-->
[Dichteverhältnis, Testfunktion, Transport, Score]{.kicker}

<!--en:600f94af-->
In §§{sec_mdl-divergences-distances}–§{num:sec_mdl-scores} haben wir vier Familien von Verteilungsvergleichen entwickelt:

<!--en:d02e6750-->
- **$f$-Divergenzen ($\mathbb{E}_Q[f(p/q)]$):** KL, Rückwärts-KL, $\chi^2$, Hellinger, JS, TV. Skaleninvariant, bei disjunkten Trägern jedoch konstant.

<!--en:d78fcfd3-->
- **Integrale Wahrscheinlichkeitsmetriken ($\sup_{g \in \mathcal{G}}(\mathbb{E}_P g - \mathbb{E}_Q g)$):** TV ($|g| \le 1/2$), MMD (RKHS-Kugel), $W_1$ ($\textrm{Lip}_1$). Direkt aus Stichproben schätzbar.

<!--en:fa74d0b8-->
- **Optimaler Transport ($\inf_\gamma \mathbb{E}_\gamma[c(x, y)]$):** $W_1$, $W_2$ (FID), Sinkhorn. Berücksichtigt die Geometrie bei jedem Abstand.

<!--en:d214fa49-->
- **Score-basierte Vergleiche ($D_{\textrm{F}}, \mathrm{KSD}$):** Vergleichen $\nabla_{\mathbf{x}} \log q$ ohne Normierungskonstante $Z$, sind aber unempfindlich gegenüber den Gewichten weit entfernter Modi.

<!--en:6b738290-->
::: {.slide title="Entscheidungshilfe zur Wahl des Trainingsverlusts"}

<!--en:6300d94a-->
[Trainingsziele nach den Möglichkeiten des Modells auswählen]{.kicker}

<!--en:458d902f-->
- **Explizite normierte Dichte $q_{\boldsymbol{\theta}} > 0$:** Maximum-Likelihood minimiert die **Vorwärts-KL** (deckt Modi ab).

<!--en:059e8369-->
- **Berechenbare Dichte $q_{\boldsymbol{\theta}}$ und unnormierte Zieldichte $\tilde{p}$:** Variationelle Inferenz (ELBO) minimiert die **Rückwärts-KL** (bevorzugt einzelne Modi).

<!--en:2c1f2751-->
- **Impliziter Generator oder EBM:** **MMD** (Zweistichprobentests in geschlossener Form), **WGAN / Sinkhorn-$W_1$** (nicht verschwindende Gradienten in den Randbereichen) oder **Score-Matching / KSD** (ohne Normierungskonstante $Z$).

<!--en:01cfc43f-->
::: {.slide title="Verständnisfragen: Skalierung der Fisher-Divergenz, KSD und Normierung"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:02ff1e8c-->
Prüfen Sie Ihr Verständnis Score-basierter Vergleiche:

<!--en:5ce00f2a-->
1. **Skalierung der Koordinaten ($D_{\textrm{F}}$ gegenüber $D_{\textrm{KL}}$):** Berechnen Sie $D_{\textrm{F}}(P \| Q) = \frac{\Delta^2}{2\sigma^4}$ und $D_{\textrm{KL}}(P \| Q) = \frac{\Delta^2}{2\sigma^2}$ zunächst für $P = \mathcal{N}(0, 1)$ gegenüber $Q = \mathcal{N}(1, 1)$, dann bei verdoppelten Koordinaten für $P = \mathcal{N}(0, 4)$ gegenüber $Q = \mathcal{N}(2, 4)$. Warum ändert sich nur eine der beiden Divergenzen?

<!--en:b2084bd1-->
2. **KSD-Vorhersage in geschlossener Form:** Berechnen Sie mit $\mathrm{KSD}^2(P, Q) = \mathbb{E}_{x, x' \sim P}[(s_Q - s_P)(x)k(x, x')(s_Q - s_P)(x')]$ den exakten Wert von $\mathrm{KSD}^2$ für die Grundgesamtheit, wenn die Daten der Verteilung $P = \mathcal{N}(0, 1)$ folgen und das Modell $Q = \mathcal{N}(1, 1)$ verschoben ist. Verwenden Sie den RBF-Kernel $k(x, x') = e^{-(x-x')^2/2}$ ($\ell = 1$).

<!--en:3cf6ba6f-->
3. **Prüfung eines energiebasierten Modells:** Gegeben seien ein EBM $\tilde{q}(\mathbf{x}) = e^{-E(\mathbf{x})}$ mit praktisch unberechenbarem $Z$ ohne MCMC-Sampler und ein Datensatz $\{\mathbf{x}_i\}_{i=1}^N \sim P$. Welche der Größen $D_{\textrm{KL}}(P \| Q)$, $\textrm{JS}$, $\textrm{MMD}$, $W_1$, $D_{\textrm{F}}$ und $\mathrm{KSD}$ können Sie berechnen oder optimieren?

<!--en:d36b3183-->
::: {.slide title="Verständnisfragen: Skalierung der Fisher-Divergenz, KSD und Normierung (Fortsetzung)"}

<!--en:929b7671-2-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:557b98b6-->
**Antworten:**

<!--en:b44fc459-->
1. **Skalierung:** Für $\mathcal{N}(0,1)\|\mathcal{N}(1,1)$ ($\Delta=1, \sigma^2=1$) sind beide Divergenzen gleich $1/2$. Für $\mathcal{N}(0,4)\|\mathcal{N}(2,4)$ ($\Delta=2, \sigma^2=4$) bleibt $D_{\textrm{KL}} = \frac{4}{2(4)} = 1/2$ unverändert. Dagegen sinkt $D_{\textrm{F}} = \frac{4}{2(16)} = 1/8$ um den Faktor $4\times$, weil Scores die Einheit $1/x$ haben.

<!--en:260a7b6c-->
2. **KSD bei Verschiebung:** Wegen $s_P(x) = -x$ und $s_Q(x) = -(x-1)$ beträgt die Scoredifferenz überall $s_Q(x) - s_P(x) = 1$. Damit gilt $\mathrm{KSD}^2 = \mathbb{E}_{X, X' \sim \mathcal{N}(0,1)}[e^{-(X-X')^2/2}] = (1 + 2/1)^{-1/2} = 1/\sqrt{3} \approx 0.577$ (denn $X - X' \sim \mathcal{N}(0, 2)$).

<!--en:bb3065ee-->
3. **Prüfung des EBM:** $\mathrm{KSD}^2(P, Q)$ lässt sich direkt berechnen. Die Divergenz $D_{\textrm{F}}(P \| Q)$ lässt sich bis auf die Datenkonstante $\tfrac12\mathbb{E}_P[s_P^2]$ durch Score-Matching nach Hyvärinen optimieren. Die Vorwärts-KL benötigt $Z$. JS, MMD und $W_1$ benötigen Samples aus $Q$.

<!--en:00ef99f1-->
::: {.slide title="Zusammenfassung: Score-basierte Vergleiche und Trainingsziele"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:5ced5987-->
| Begriff oder Verfahren | Formel | Eigenschaft oder Voraussetzung | Bedeutung und Grenzen im Deep Learning |

<!--en:7e2e701f-->
| **Stein-Score** | $s_Q(\mathbf{x}) = \nabla_{\mathbf{x}} \log q(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})$ | Normierungskonstante $Z = \int e^{-E(\mathbf{x})}\,d\mathbf{x}$ fällt weg | Vektorfeld von Rückstellkräften zu Modi mit hoher Dichte |

<!--en:e7db5c7b-->
| **Fisher-Divergenz** | $D_{\textrm{F}}(P \| Q) = \tfrac12\mathbb{E}_P[\|s_P(\mathbf{x}) - s_Q(\mathbf{x})\|_2^2]$ | Einheit $1/x^2$. Auf zusammenhängendem Träger gilt $0 \iff P = Q$ | Fällt bei isolierten Modi mit vertauschten Gewichten wie $\sim e^{-m^2/2}$ ab |

<!--en:7962e9c8-->
| **Stein-Identität** | $\mathbb{E}_{x \sim p}[g'(x) + g(x)s_p(x)] = 0$ | Partielle Integration unter der Voraussetzung $g(x)p(x) \to 0$ | Ersetzt unbekannte Dichteableitungen durch aus Stichproben schätzbare Erwartungswerte |

<!--en:d836eaa3-->
| **Score-Matching** | $J(Q) = \mathbb{E}_P[\tfrac12\|s_Q(\mathbf{x})\|_2^2 + \textrm{tr}(\nabla_{\mathbf{x}} s_Q(\mathbf{x}))]$ | Gleich $D_{\textrm{F}}(P \| Q) - \tfrac12\mathbb{E}_P[\|s_P\|^2]$ | Trainiert EBMs und Diffusionsmodelle (mit mehrskalig verrauschten Daten) ohne $Z$ |

<!--en:37f7cd04-->
| **Kernel-Stein-Diskrepanz (KSD)** | $\mathrm{KSD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k_Q(x, x')]$ | Mit dem Kernel geglättete Scoredifferenz $\mathbb{E}_P[\Delta s(x) k(x,x') \Delta s(x')]$ | Nichtparametrischer Anpassungstest, der nur $s_Q$ und Datenstichproben benötigt |
